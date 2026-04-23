"""Tennis model backtest — compares the full tennis ensemble against Pinnacle.

Replays historical tennis-data.co.uk matches (2024/2025/2026) through the
current tennis ensemble (surface Elo + serve/return + form + advanced stats
+ H2H + ranking trend), with sharp_weight=0 so the output is model-only.
Then reports accuracy / Brier / log-loss / calibration delta against the
Pinnacle-devigged baseline that the existing `evmax backtest run --sectors
tennis` command measures.

USAGE
    python scripts/backtest_tennis_model.py                # all years
    python scripts/backtest_tennis_model.py --years 2026   # one year
    python scripts/backtest_tennis_model.py --limit 200    # sample

CAVEAT
    The model state was seeded from the same Sackmann/MCP dataset that
    overlaps these tennis-data.co.uk rows, so this is an IN-SAMPLE calibration
    check, not a true walk-forward. It tells you whether the current ratings
    are well-calibrated against Pinnacle today — not whether the model would
    have been predictive without hindsight.
"""

from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path
from typing import Optional

from evmax.agents.base import AgentBus, AgentRequest
from evmax.agents.models.ensemble_agent import EnsembleModelAgent
from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent
from evmax.agents.models.tennis_form_agent import TennisFormAgent
from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent
from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent
from evmax.backtest.models import BacktestRow
from evmax.backtest.sources.tennis_xlsx import load_tennis
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# Map tournament name substrings → event.product_metadata.competition form
# used by the tennis surface resolver. Anything not matched falls back to
# using the raw tournament string (still matches many clay/grass cities).
_TOURNAMENT_SURFACE_HINTS = {
    "Wimbledon": "ATP Wimbledon",
    "French Open": "ATP Roland Garros",
    "US Open": "ATP US Open",
    "Australian Open": "ATP Australian Open",
}


def _row_to_market(row: BacktestRow, idx: int) -> tuple[PredictionMarket, SharpOdds]:
    """Build (market, sharp) pair for the ensemble. Player A = Winner (by xlsx convention)."""
    event_id = f"tennis::{row.date.isoformat()}::{idx}"
    comp_hint = _TOURNAMENT_SURFACE_HINTS.get(row.league, row.league)

    market = PredictionMarket(
        id=f"kalshi:{event_id}",
        source=MarketSource.kalshi,
        sector="tennis",
        market_type=MarketType.moneyline,
        title=f"Will {row.team_home} win : {row.league}",
        yes_price=0.5,
        no_price=0.5,
        team_home=row.team_home.lower().strip(),
        team_away=row.team_away.lower().strip(),
        event_id=event_id,
        competition=comp_hint,
    )
    sharp = SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector="tennis",
        outcome_a_label=row.team_home.lower().strip(),
        outcome_b_label=row.team_away.lower().strip(),
        outcome_a_decimal=row.pinnacle_home_dec,
        outcome_b_decimal=row.pinnacle_away_dec,
        true_prob_a=row.true_prob_home,
        true_prob_b=row.true_prob_away,
        margin=1 / row.pinnacle_home_dec + 1 / row.pinnacle_away_dec - 1.0,
    )
    return market, sharp


def _brier(preds: list[float], actuals: list[bool]) -> float:
    return sum((p - float(a)) ** 2 for p, a in zip(preds, actuals)) / len(preds)


def _log_loss(preds: list[float], actuals: list[bool], eps: float = 1e-7) -> float:
    total = 0.0
    for p, a in zip(preds, actuals):
        p = max(eps, min(1 - eps, p))
        total += -(float(a) * math.log(p) + (1 - float(a)) * math.log(1 - p))
    return total / len(preds)


def _accuracy(preds: list[float], actuals: list[bool]) -> float:
    return sum(1 for p, a in zip(preds, actuals) if (p >= 0.5) == a) / len(preds)


def _calibration_table(preds: list[float], actuals: list[bool], n_bins: int = 10) -> list[dict]:
    step = 1.0 / n_bins
    rows = []
    for b in range(n_bins):
        lo, hi = b * step, (b + 1) * step
        if b == n_bins - 1:
            idxs = [i for i, p in enumerate(preds) if lo <= p <= hi]
        else:
            idxs = [i for i, p in enumerate(preds) if lo <= p < hi]
        n = len(idxs)
        if n == 0:
            rows.append({"lo": lo, "hi": hi, "n": 0, "pred": 0.0, "actual": 0.0})
            continue
        mean_pred = sum(preds[i] for i in idxs) / n
        actual = sum(1 for i in idxs if actuals[i]) / n
        rows.append({"lo": lo, "hi": hi, "n": n, "pred": mean_pred, "actual": actual})
    return rows


async def run(years: list[int], limit: Optional[int] = None) -> None:
    print(f"Loading tennis-data.co.uk rows for years={years} …")
    rows = load_tennis(years)
    print(f"  Loaded {len(rows)} rows")

    if limit:
        rows = rows[:limit]
        print(f"  Trimmed to {limit} rows for fast pass")

    # Wire up the tennis ensemble exactly as the coordinator does.
    agents = [
        TennisModelAgent(),
        TennisServeReturnAgent(),
        TennisFormAgent(),
        TennisAdvancedStatsAgent(),
        TennisH2HAgent(),
        TennisRankingTrendAgent(),
    ]
    ensemble = EnsembleModelAgent(models=agents, sharp_weight=0.0)
    bus = AgentBus()
    ensemble.attach_bus(bus)

    pairs = []
    row_by_event: dict[str, BacktestRow] = {}
    for i, row in enumerate(rows):
        market, sharp = _row_to_market(row, i)
        pairs.append({"market": market, "sharp": sharp})
        row_by_event[sharp.event_id] = row

    print(f"Running ensemble on {len(pairs)} matches (sharp_weight=0, model-only) …")
    resp = await ensemble(AgentRequest(sector="tennis", params={"pairs": pairs, "sharp_weight": 0.0}))
    blended = resp.data
    print(f"  Got {len(blended)} blended predictions")

    # Collect paired arrays: model vs pinnacle, two entries per match (A + B sides)
    model_preds: list[float] = []
    sharp_preds: list[float] = []
    actuals: list[bool] = []
    model_sources_seen: dict[str, int] = {}

    skipped_no_model = 0
    for event_id, pred in blended.items():
        row = row_by_event[event_id]
        # If no model contributed (sharp_weight=0 so falls back to sharp only),
        # skip — we can't attribute those to the model.
        if pred.model_sources == "sharp" or pred.model_sources == "":
            skipped_no_model += 1
            continue
        for src in pred.model_sources.split("+"):
            if src:
                model_sources_seen[src] = model_sources_seen.get(src, 0) + 1
        # Row convention: home/outcome_a is the Winner, so side A always wins.
        model_preds.append(pred.true_prob_a)
        sharp_preds.append(row.true_prob_home)
        actuals.append(True)
        model_preds.append(pred.true_prob_b)
        sharp_preds.append(row.true_prob_away)
        actuals.append(False)

    if not model_preds:
        print("No rows produced a model prediction — ensure model state files are seeded.")
        return

    n = len(model_preds) // 2
    print()
    print(f"=== Tennis model vs Pinnacle backtest — {n} matches ({len(model_preds)} sides) ===")
    print(f"  Skipped (no model contribution, only sharp): {skipped_no_model}")
    print(f"  Model sources (times contributed):")
    for src, c in sorted(model_sources_seen.items(), key=lambda kv: -kv[1]):
        print(f"    {src:28s}  {c}")
    print()

    m_b = _brier(model_preds, actuals)
    m_l = _log_loss(model_preds, actuals)
    m_a = _accuracy(model_preds, actuals)
    s_b = _brier(sharp_preds, actuals)
    s_l = _log_loss(sharp_preds, actuals)
    s_a = _accuracy(sharp_preds, actuals)

    print(f"{'Metric':<15} {'Model':>10} {'Pinnacle':>10} {'Delta':>10}   (lower=better for Brier/LogLoss)")
    print(f"{'-'*50}")
    print(f"{'Brier':<15} {m_b:>10.4f} {s_b:>10.4f} {m_b - s_b:>+10.4f}")
    print(f"{'Log loss':<15} {m_l:>10.4f} {s_l:>10.4f} {m_l - s_l:>+10.4f}")
    print(f"{'Accuracy':<15} {m_a:>10.4f} {s_a:>10.4f} {m_a - s_a:>+10.4f}")

    print()
    print("=== Calibration by decile (how well does predicted prob match realized win rate?) ===")
    print(f"{'Bucket':<14} {'Model N':>8} {'M pred':>8} {'M actual':>9} {'M gap':>8}    "
          f"{'P pred':>8} {'P actual':>9} {'P gap':>8}")
    m_cal = _calibration_table(model_preds, actuals)
    s_cal = _calibration_table(sharp_preds, actuals)
    for mc, sc in zip(m_cal, s_cal):
        bucket = f"{int(mc['lo']*100):>3}–{int(mc['hi']*100):>3}%"
        if mc["n"] == 0 and sc["n"] == 0:
            continue
        m_gap = mc["actual"] - mc["pred"] if mc["n"] else 0.0
        s_gap = sc["actual"] - sc["pred"] if sc["n"] else 0.0
        print(
            f"{bucket:<14} {mc['n']:>8} "
            f"{mc['pred']*100:>7.1f}% {mc['actual']*100:>8.1f}% {m_gap*100:>+7.1f}%    "
            f"{sc['pred']*100:>7.1f}% {sc['actual']*100:>8.1f}% {s_gap*100:>+7.1f}%"
        )

    # By-league / tournament breakdown
    print()
    print("=== By tournament (N, Brier: model vs sharp, Δ) ===")
    by_league: dict[str, dict] = {}
    for event_id, pred in blended.items():
        if pred.model_sources in ("sharp", ""):
            continue
        row = row_by_event[event_id]
        d = by_league.setdefault(row.league, {"n": 0, "m_b": 0.0, "s_b": 0.0})
        d["n"] += 2
        d["m_b"] += (pred.true_prob_a - 1.0) ** 2 + (pred.true_prob_b - 0.0) ** 2
        d["s_b"] += (row.true_prob_home - 1.0) ** 2 + (row.true_prob_away - 0.0) ** 2
    leagues_sorted = sorted(by_league.items(), key=lambda kv: -kv[1]["n"])
    print(f"{'League':<34} {'N':>5} {'Model Brier':>13} {'Sharp Brier':>13} {'Δ Brier':>9}")
    for lg, d in leagues_sorted[:40]:
        m = d["m_b"] / d["n"]
        s = d["s_b"] / d["n"]
        print(f"{lg[:33]:<34} {d['n']:>5} {m:>13.4f} {s:>13.4f} {m-s:>+9.4f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026],
                   help="Tennis-data years to replay (default: 2024 2025 2026)")
    p.add_argument("--limit", type=int, default=None,
                   help="Trim to the first N rows for a fast sanity check")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(args.years, args.limit))
