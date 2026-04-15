"""Prop-level backtest metrics: Brier, log-loss, calibration, ROI.

Reuses the pure math helpers from `evmax.backtest.metrics` but operates
on `PropBacktestRow` instances and produces a `PropBacktestReport`.
"""

from __future__ import annotations

from collections import defaultdict

from evmax.backtest.metrics import (
    accuracy_score,
    brier_score,
    calibration_bins,
    log_loss_score,
)
from evmax.backtest.models import PropBacktestReport, PropBacktestRow

EV_THRESHOLDS = [0.02, 0.03, 0.05, 0.08]

# Kalshi's `last_price_dollars` on settled markets is the last observed
# trade price — which near settlement converges toward 0.0 or 1.0. Prices
# outside this band are treated as "already knew the answer" and skipped,
# since taking a bet vs a near-0 closing price is not a real opportunity
# (it's just backtest leakage from post-outcome price movement).
PRICE_MIN = 0.05
PRICE_MAX = 0.95


def _roi_at_threshold(
    rows: list[PropBacktestRow],
    ev_threshold: float,
    min_volume: float = 0.0,
    side: str = "both",
) -> dict:
    """Simulate flat $1-unit bets where model disagrees with market by ≥ threshold.

    `side` controls which side(s) the bettor takes:
      "yes"  — bet YES only when model_prob - closing_yes_price ≥ threshold
      "no"   — bet NO  only when closing_yes_price - model_prob ≥ threshold
      "both" — take whichever side is positive (the two sets are disjoint)

    Markets with closing prices outside [PRICE_MIN, PRICE_MAX] are skipped
    because those prices are saturated by near-settlement information and
    don't represent real betting opportunities in the historical data.
    If closing_price isn't available, the row is skipped.
    """
    bets = 0
    wins = 0
    pnl = 0.0
    for r in rows:
        if r.closing_yes_price is None:
            continue
        if r.closing_yes_price < PRICE_MIN or r.closing_yes_price > PRICE_MAX:
            continue
        if r.volume < min_volume:
            continue

        edge_yes = r.model_prob - r.closing_yes_price
        edge_no = r.closing_yes_price - r.model_prob

        take_yes = (side in ("yes", "both")) and edge_yes >= ev_threshold
        take_no = (side in ("no", "both")) and edge_no >= ev_threshold

        if take_yes:
            decimal_odds = 1.0 / r.closing_yes_price
            bets += 1
            if r.actual_result == 1:
                wins += 1
                pnl += decimal_odds - 1.0
            else:
                pnl -= 1.0
        elif take_no:
            no_price = 1.0 - r.closing_yes_price
            if no_price <= 0:
                continue
            decimal_odds = 1.0 / no_price
            bets += 1
            if r.actual_result == 0:
                wins += 1
                pnl += decimal_odds - 1.0
            else:
                pnl -= 1.0

    roi = pnl / bets if bets > 0 else 0.0
    win_rate = wins / bets if bets > 0 else 0.0
    return {
        "bets": bets,
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "roi": round(roi, 4),
        "pnl": round(pnl, 2),
    }


def compute_prop_report(
    rows: list[PropBacktestRow],
    min_volume: float = 0.0,
) -> PropBacktestReport:
    """Build a PropBacktestReport from scored rows."""
    n_markets = len(rows)
    scored = [r for r in rows if 0.0 <= r.model_prob <= 1.0 and r.n_games_used > 0]
    if not scored:
        return PropBacktestReport(
            sector="nfl_props",
            stat_types=[],
            n_markets=n_markets,
            n_scored=0,
            min_volume=min_volume,
        )

    gated = [r for r in scored if r.volume >= min_volume]

    preds = [r.model_prob for r in gated]
    actual = [bool(r.actual_result) for r in gated]
    bs = brier_score(preds, actual)
    ll = log_loss_score(preds, actual)
    acc = accuracy_score(preds, actual)
    cal = calibration_bins(preds, actual)

    # Baselines
    # 1. Per-stat prior baseline: every market's prediction is the overall
    #    yes-rate for that stat in the dataset. Scored markets only.
    by_stat_yes: dict[str, list[int]] = defaultdict(list)
    for r in gated:
        by_stat_yes[r.stat_type].append(r.actual_result)
    stat_priors = {k: sum(v) / len(v) for k, v in by_stat_yes.items() if v}
    prior_preds = [stat_priors.get(r.stat_type, 0.5) for r in gated]
    baseline_prior = brier_score(prior_preds, actual)

    # 2. Closing market price baseline — ONLY on markets with closing prices
    # inside [PRICE_MIN, PRICE_MAX]. Outside that range the price is
    # effectively a post-outcome snapshot (converges to 0/1 near settlement)
    # and is not a fair comparison for a pre-game model.
    market_subset = [
        (r.closing_yes_price, bool(r.actual_result))
        for r in gated
        if r.closing_yes_price is not None
        and PRICE_MIN <= r.closing_yes_price <= PRICE_MAX
    ]
    if market_subset:
        baseline_market = brier_score(
            [p for p, _ in market_subset],
            [a for _, a in market_subset],
        )
    else:
        baseline_market = 0.0

    # Per-stat breakdown
    by_stat: dict[str, dict] = {}
    for stat, yes_vals in by_stat_yes.items():
        sub = [r for r in gated if r.stat_type == stat]
        spreds = [r.model_prob for r in sub]
        sactual = [bool(r.actual_result) for r in sub]
        by_stat[stat] = {
            "n": len(sub),
            "yes_rate": round(sum(yes_vals) / len(yes_vals), 3),
            "brier": round(brier_score(spreds, sactual), 4),
            "log_loss": round(log_loss_score(spreds, sactual), 4),
            "accuracy": round(accuracy_score(spreds, sactual), 3),
        }

    # ROI at multiple EV thresholds. Reports both-sides, YES-only, and
    # NO-only splits so we can see whether one side is systematically
    # better (Kalshi prop markets exhibit a retail-overprice-YES asymmetry
    # — see MODEL-8 diagnostic, scripts/diagnose_nfl_prop_roi.py).
    roi_by_ev: dict[str, dict] = {}
    for ev in EV_THRESHOLDS:
        label = f"ev>={ev:.0%}"
        roi_by_ev[label] = _roi_at_threshold(gated, ev, min_volume=0.0, side="both")
        roi_by_ev[f"{label}_YES"] = _roi_at_threshold(gated, ev, min_volume=0.0, side="yes")
        roi_by_ev[f"{label}_NO"] = _roi_at_threshold(gated, ev, min_volume=0.0, side="no")
        roi_by_ev[f"{label}_vol>=1000"] = _roi_at_threshold(gated, ev, min_volume=1000.0, side="both")
        roi_by_ev[f"{label}_vol>=1000_NO"] = _roi_at_threshold(gated, ev, min_volume=1000.0, side="no")

    return PropBacktestReport(
        sector="nfl_props",
        stat_types=sorted(by_stat.keys()),
        n_markets=n_markets,
        n_scored=len(scored),
        brier_score=bs,
        log_loss=ll,
        accuracy=acc,
        baseline_mean_prior=baseline_prior,
        baseline_market=baseline_market,
        calibration_bins=cal,
        by_stat=by_stat,
        roi_by_ev_threshold=roi_by_ev,
        min_volume=min_volume,
    )
