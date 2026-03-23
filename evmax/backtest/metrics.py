"""Compute backtest metrics: calibration, Brier score, accuracy, EV distribution."""

from __future__ import annotations

import math
from typing import Optional

from evmax.backtest.models import BacktestReport, BacktestRow, CalibrationBin


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def calibration_bins(
    predicted: list[float],
    actual: list[bool],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Bin predicted probs into deciles and compute empirical win rate per bin."""
    bins: list[CalibrationBin] = []
    step = 1.0 / n_bins

    for b in range(n_bins):
        lo = b * step
        hi = lo + step
        indices = [i for i, p in enumerate(predicted) if lo <= p < hi]
        if b == n_bins - 1:
            indices = [i for i, p in enumerate(predicted) if lo <= p <= hi]

        n = len(indices)
        if n == 0:
            bins.append(CalibrationBin(lo, hi, (lo + hi) / 2, 0.0, 0, 0.0, 0.0))
            continue

        mean_pred = sum(predicted[i] for i in indices) / n
        wins = sum(1 for i in indices if actual[i])
        win_rate = wins / n
        ci_lo, ci_hi = _wilson_ci(wins, n)

        bins.append(CalibrationBin(
            prob_low=lo, prob_high=hi,
            mean_pred=mean_pred, actual_rate=win_rate,
            n=n, ci_low=ci_lo, ci_high=ci_hi,
        ))

    return bins


def brier_score(predicted: list[float], actual: list[bool]) -> float:
    """Mean squared error between predicted probs and binary outcomes."""
    if not predicted:
        return 0.0
    return sum((p - float(a)) ** 2 for p, a in zip(predicted, actual)) / len(predicted)


def log_loss_score(predicted: list[float], actual: list[bool], eps: float = 1e-7) -> float:
    """Binary cross-entropy loss."""
    if not predicted:
        return 0.0
    total = 0.0
    for p, a in zip(predicted, actual):
        p = max(eps, min(1 - eps, p))
        total += -(float(a) * math.log(p) + (1 - float(a)) * math.log(1 - p))
    return total / len(predicted)


def accuracy_score(predicted: list[float], actual: list[bool]) -> float:
    """Fraction where predicted > 0.5 matches actual outcome."""
    if not predicted:
        return 0.0
    correct = sum(1 for p, a in zip(predicted, actual) if (p >= 0.5) == a)
    return correct / len(predicted)


def _ev_bucket(ev: float) -> str:
    if ev < 0:
        return "< 0%"
    if ev < 0.02:
        return "0–2%"
    if ev < 0.05:
        return "2–5%"
    if ev < 0.10:
        return "5–10%"
    return "> 10%"


def compute_report(
    rows: list[BacktestRow],
    sector: str,
    leagues: list[str],
    seasons: list[str],
    ev_threshold: float = 0.02,
) -> BacktestReport:
    """Compute full BacktestReport from a list of BacktestRows (with optional Kalshi joins)."""
    if not rows:
        return BacktestReport(sector=sector, leagues=leagues, seasons=seasons, n_games=0)

    # --- Primary outcome calibration ---
    # For tennis: use both players per match (winner + loser) to avoid always-true bias.
    # For soccer: use home win probability vs actual home win.
    if sector == "tennis":
        pred_home = [p for r in rows for p in [r.true_prob_home, r.true_prob_away]]
        actual_home = [a for r in rows for a in [True, False]]  # home=winner always
    else:
        pred_home = [r.true_prob_home for r in rows]
        actual_home = [bool(r.home_won) for r in rows]

    bs = brier_score(pred_home, actual_home)
    ll = log_loss_score(pred_home, actual_home)
    acc = accuracy_score(pred_home, actual_home)
    # Baseline: always pick the outcome with highest Pinnacle probability
    baseline_correct = 0
    for r in rows:
        probs = {"home": r.true_prob_home, "away": r.true_prob_away}
        if r.true_prob_draw:
            probs["draw"] = r.true_prob_draw
        best = max(probs, key=probs.get)
        if (best == "home" and r.home_won) or (best == "draw" and r.draw) or (best == "away" and not r.home_won and not r.draw):
            baseline_correct += 1
    baseline_acc = baseline_correct / len(rows)

    # Mean Pinnacle margin (approximate: 1 - sum(1/decimal))
    margins = []
    for r in rows:
        if r.pinnacle_draw_dec:
            impl = 1/r.pinnacle_home_dec + 1/r.pinnacle_away_dec + 1/r.pinnacle_draw_dec
        else:
            impl = 1/r.pinnacle_home_dec + 1/r.pinnacle_away_dec
        margins.append(impl - 1.0)
    mean_margin = sum(margins) / len(margins) if margins else 0.0

    cal_bins = calibration_bins(pred_home, actual_home)

    # By league
    by_league: dict[str, dict] = {}
    for league in set(r.league for r in rows):
        sub = [r for r in rows if r.league == league]
        sp = [r.true_prob_home for r in sub]
        sa = [bool(r.home_won) for r in sub]
        by_league[league] = {
            "n": len(sub),
            "brier": round(brier_score(sp, sa), 4),
            "accuracy": round(accuracy_score(sp, sa), 3),
        }

    # --- Kalshi EV analysis (optional) ---
    kalshi_rows = [r for r in rows if r.theoretical_ev is not None]
    n_kalshi = len(kalshi_rows)
    ev_buckets = {"< 0%": 0, "0–2%": 0, "2–5%": 0, "5–10%": 0, "> 10%": 0}
    n_positive_ev = 0
    n_above_threshold = 0
    positive_evs: list[float] = []

    for r in kalshi_rows:
        ev = r.theoretical_ev
        bucket = _ev_bucket(ev)
        ev_buckets[bucket] = ev_buckets.get(bucket, 0) + 1
        if ev > 0:
            n_positive_ev += 1
            positive_evs.append(ev)
        if ev >= ev_threshold:
            n_above_threshold += 1

    mean_ev_positive = sum(positive_evs) / len(positive_evs) if positive_evs else 0.0

    return BacktestReport(
        sector=sector,
        leagues=leagues,
        seasons=seasons,
        n_games=len(rows),
        n_kalshi_matched=n_kalshi,
        brier_score=bs,
        log_loss=ll,
        accuracy=acc,
        baseline_accuracy=baseline_acc,
        mean_pinnacle_margin=mean_margin,
        calibration_bins=cal_bins,
        n_positive_ev=n_positive_ev,
        n_above_threshold=n_above_threshold,
        ev_threshold=ev_threshold,
        mean_ev_positive=mean_ev_positive,
        ev_buckets=ev_buckets,
        by_league=by_league,
    )
