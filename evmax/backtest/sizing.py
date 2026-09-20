"""Sizing replay harness — the evidence engine for the stake-sizing layers.

Every sizing change (edge shrinkage, base fraction, hard cap, liquidity discount,
exposure guard) is judged here, out of sample, before its flag is flipped in the live
path. The harness replays resolved prediction rows under a pluggable sizing *policy*
and reports geometric growth and drawdown — the quantities Kelly sizing actually
optimizes — not Brier or hit rate.

Design decisions that matter for correctness:

* **Same-day settlement is simultaneous.** Bets whose games settle on the same date
  are staked against ONE bankroll and settled together. Compounding them serially
  (as a naive replay does) overstates both growth and drawdown, because it lets an
  early win in the day fund a later bet the bettor could not actually have placed yet.

* **The per-game exposure cap runs inside the replay.** Multiple markets on one game
  are correlated; the live path caps total stake per game at ``event_cap`` (8%). The
  harness applies the same cap so a policy cannot claim growth from stacking correlated
  legs the live guard would have trimmed.

* **Contamination rows are excluded by rule**, reusing
  :func:`evmax.agents.cleanup.contamination.is_contaminated` — the same filter the
  promotion gates use — so a superseded-code row (e.g. the mis-aligned April esports
  YES sides) can never flatter a policy.

* **Fees are charged at the effective price** via :func:`evmax.ev.calculator.effective_price`.

The block bootstrap resamples whole weeks (not individual bets) so the confidence
interval respects the serial correlation of a betting record.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from evmax.agents.cleanup.contamination import is_contaminated
from evmax.ev.calculator import effective_price

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "predictions.db"


@dataclass
class ResolvedRow:
    """One resolved market: what the model said, what it cost, what happened."""

    market_id: str
    sector: str
    market_type: str
    event_id: str
    event_date: str          # 'YYYY-MM-DD' (bucketing key for simultaneous settlement)
    blended: float           # blended_true_prob (raw model probability)
    price: float             # kalshi_yes_price (venue YES ask at scan)
    outcome: int             # 1 = YES won, 0 = NO
    venue: str = "kalshi"
    line: Optional[float] = None
    model_sources: Optional[str] = None
    ev_pct: float = 0.0

    def base_event(self) -> str:
        # Reuse the coordinator's exact grouping so the harness's per-game cap
        # matches the live exposure guard (ML + spread + total on one matchup
        # share a budget; props group by player).
        from evmax.agents.coordinator import _base_event
        return _base_event(self.event_id) if self.event_id else self.market_id


# A policy maps a row to a stake as a FRACTION of current bankroll (pre-exposure-cap).
SizingPolicy = Callable[[ResolvedRow], float]


def load_resolved_rows(
    db_path: Path | str = DEFAULT_DB,
    *,
    days: int = 180,
    modes: Iterable[str] = ("live",),
    exclude_contaminated: bool = True,
    price_lo: float = 0.02,
    price_hi: float = 0.98,
    require_sized: bool = True,
) -> list[ResolvedRow]:
    """Load resolved rows joined to their outcomes, newest ``days`` days.

    ``modes`` selects live and/or shadow rows — shadow rows carry the same blended
    probability, captured price, and outcome, so they are valid sizing evidence and
    roughly triple the sample. One row per market (latest scan) via GROUP BY.

    ``require_sized`` (default True) keeps only rows the scanner actually sized
    (``kelly_fraction > 0``) — the right sample for a *sizing* verdict on live
    rows. Shadow rows carry Kelly = 0 by construction (shadow never stakes), so a
    replay over a shadow sector (nfl / ncaaf today) must pass ``require_sized=
    False`` or it loads zero rows. Any replay that scores an ADMISSION rule
    (which rows should have been played at all) rather than a sizing rule
    needs the full logged sample and should pass False.
    """
    mode_list = list(modes)
    placeholders = ",".join("?" for _ in mode_list)
    sized_clause = "AND p.kelly_fraction > 0" if require_sized else ""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            f"""
            SELECT p.market_id, p.sector, p.market_type, p.event_id, p.event_date,
                   p.blended_true_prob AS blended, p.kalshi_yes_price AS price,
                   o.outcome AS outcome, p.venue, p.line, p.model_sources, p.ev_pct
            FROM ev_predictions p
            JOIN ev_outcomes o ON o.market_id = p.market_id
            WHERE p.mode IN ({placeholders})
              AND o.outcome IS NOT NULL
              AND p.voided = 0
              {sized_clause}
              AND p.kalshi_yes_price > ?
              AND p.kalshi_yes_price < ?
              AND p.scan_date >= date('now', ?)
            GROUP BY p.market_id
            ORDER BY p.event_date, p.market_id
            """,
            (*mode_list, price_lo, price_hi, f"-{days} day"),
        ).fetchall()
    finally:
        con.close()

    out: list[ResolvedRow] = []
    for r in rows:
        if exclude_contaminated and is_contaminated(
            r["sector"], r["market_type"], r["model_sources"], r["line"]
        ):
            continue
        out.append(
            ResolvedRow(
                market_id=r["market_id"],
                sector=r["sector"] or "",
                market_type=r["market_type"] or "",
                event_id=r["event_id"] or "",
                event_date=(r["event_date"] or "")[:10],
                blended=float(r["blended"]),
                price=float(r["price"]),
                outcome=int(r["outcome"]),
                venue=r["venue"] or "kalshi",
                line=r["line"],
                model_sources=r["model_sources"],
                ev_pct=float(r["ev_pct"] or 0.0),
            )
        )
    return out


@dataclass
class SimResult:
    n_bets: int
    log_growth: float
    final_multiple: float
    max_drawdown: float
    bet_returns: np.ndarray  # per-bet return on bankroll (in placement order)


def _iso_week(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"
    except (ValueError, TypeError):
        return date_str[:7]  # fall back to month


def simulate(
    rows: list[ResolvedRow],
    policy: SizingPolicy,
    *,
    fee_venue: Optional[str] = "kalshi",
    event_cap: float = 0.08,
) -> SimResult:
    """Replay ``rows`` under ``policy`` with day-batched simultaneous settlement.

    Returns the realized log growth, terminal wealth multiple, worst peak-to-trough
    drawdown, and the per-bet bankroll returns in placement order (for bootstrapping).
    """
    by_day: dict[str, list[ResolvedRow]] = {}
    for r in rows:
        by_day.setdefault(r.event_date, []).append(r)

    wealth = 1.0
    peak = 1.0
    max_dd = 0.0
    returns: list[float] = []

    for day in sorted(by_day):
        day_rows = sorted(by_day[day], key=lambda r: r.ev_pct, reverse=True)
        event_used: dict[str, float] = {}
        day_pnl = 0.0
        for r in day_rows:
            eff = effective_price(r.price, fee_venue)
            if not (0.0 < eff < 1.0):
                continue
            b = 1.0 / eff - 1.0
            frac = max(0.0, policy(r))
            # Per-game exposure cap, best-EV-first (mirrors _apply_exposure_guard).
            base = r.base_event()
            remaining = event_cap - event_used.get(base, 0.0)
            if remaining <= 0:
                continue
            frac = min(frac, remaining)
            if frac <= 0:
                continue
            event_used[base] = event_used.get(base, 0.0) + frac
            ret = frac * b if r.outcome == 1 else -frac
            day_pnl += ret
            returns.append(ret)
        # Simultaneous settlement: the whole day's P&L hits one bankroll.
        wealth *= (1.0 + day_pnl)
        if wealth <= 0:
            wealth = 1e-9
        peak = max(peak, wealth)
        max_dd = max(max_dd, 1.0 - wealth / peak)

    arr = np.array(returns, dtype=float)
    return SimResult(
        n_bets=len(returns),
        log_growth=float(np.log(max(wealth, 1e-9))),
        final_multiple=float(wealth),
        max_drawdown=float(max_dd),
        bet_returns=arr,
    )


def block_bootstrap_log_growth(
    rows: list[ResolvedRow],
    policy: SizingPolicy,
    *,
    fee_venue: Optional[str] = "kalshi",
    event_cap: float = 0.08,
    n_boot: int = 2000,
    seed: int = 0,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
) -> dict[float, float]:
    """Resample whole ISO-weeks with replacement; report log-growth percentiles.

    Resampling weeks (not bets) preserves within-week serial correlation, so the
    interval reflects the variance of a real betting record rather than i.i.d. bets.
    """
    weeks: dict[str, list[ResolvedRow]] = {}
    for r in rows:
        weeks.setdefault(_iso_week(r.event_date), []).append(r)
    week_keys = list(weeks)
    if not week_keys:
        return {p: 0.0 for p in percentiles}

    rng = np.random.default_rng(seed)
    growths = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(len(week_keys), size=len(week_keys), replace=True)
        sample: list[ResolvedRow] = []
        for idx in picked:
            sample.extend(weeks[week_keys[idx]])
        growths[i] = simulate(
            sample, policy, fee_venue=fee_venue, event_cap=event_cap
        ).log_growth
    return {p: float(np.percentile(growths, p)) for p in percentiles}


def edge_ratio(rows: list[ResolvedRow], *, fee_venue: Optional[str] = "kalshi") -> float:
    """Mean realized edge / mean predicted edge (net of fee). 1.0 = perfectly honest.

    Below 1.0 means the model's edges shrink out of sample — the tail-selection bias
    the shrinkage layer exists to correct.
    """
    if not rows:
        return float("nan")
    pred = []
    real = []
    for r in rows:
        eff = effective_price(r.price, fee_venue)
        pred.append(r.blended - eff)
        real.append(r.outcome - eff)
    mp = float(np.mean(pred))
    return float(np.mean(real) / mp) if mp else float("nan")


def walk_forward_months(
    rows: list[ResolvedRow],
    fit_and_policy: Callable[[list[ResolvedRow]], SizingPolicy],
    *,
    fee_venue: Optional[str] = "kalshi",
    event_cap: float = 0.08,
    min_train_months: int = 2,
) -> SimResult:
    """Score a policy out of sample: fit on months ``1..k``, place month ``k+1``.

    ``fit_and_policy`` receives the training rows and returns a policy fitted only on
    them (e.g. shrinkage coefficients estimated on the past). The returned SimResult
    concatenates the held-out months, so its growth is a genuine forward record.
    """
    by_month: dict[str, list[ResolvedRow]] = {}
    for r in rows:
        by_month.setdefault(r.event_date[:7], []).append(r)
    months = sorted(by_month)
    all_returns: list[float] = []
    wealth = 1.0
    peak = 1.0
    max_dd = 0.0
    for k in range(min_train_months, len(months)):
        train = [r for m in months[:k] for r in by_month[m]]
        test = by_month[months[k]]
        if not train or not test:
            continue
        policy = fit_and_policy(train)
        res = simulate(test, policy, fee_venue=fee_venue, event_cap=event_cap)
        wealth *= res.final_multiple
        peak = max(peak, wealth)
        max_dd = max(max_dd, 1.0 - wealth / peak)
        all_returns.extend(res.bet_returns.tolist())
    return SimResult(
        n_bets=len(all_returns),
        log_growth=float(np.log(max(wealth, 1e-9))),
        final_multiple=float(wealth),
        max_drawdown=float(max_dd),
        bet_returns=np.array(all_returns, dtype=float),
    )


# --------------------------------------------------------------------------
# Standard policies
# --------------------------------------------------------------------------

def make_kelly_policy(
    *,
    base_fraction: float = 0.5,
    max_kelly: float = 0.05,
    fee_venue: Optional[str] = "kalshi",
    shrinkage_model=None,
) -> SizingPolicy:
    """Fractional-Kelly-with-cap policy, optionally sizing on shrunk probability.

    ``shrinkage_model`` is an :class:`evmax.ev.sizing.ShrinkageModel`; when given,
    the Kelly numerator uses ``P(win | blended, price)`` instead of the raw blend.
    """
    from evmax.ev.sizing import ShrinkageModel  # noqa: F401 (type only)

    def policy(r: ResolvedRow) -> float:
        eff = effective_price(r.price, fee_venue)
        if not (0.0 < eff < 1.0):
            return 0.0
        b = 1.0 / eff - 1.0
        p = r.blended
        if shrinkage_model is not None:
            p = shrinkage_model.sizing_probability(r.blended, r.price, r.sector)
        f_star = (p * b - (1.0 - p)) / b
        return min(max(0.0, f_star) * base_fraction, max_kelly)

    return policy
