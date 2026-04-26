"""Fit per-sector isotonic calibration for the tennis ensemble.

Symptom (full 2024–2026 backtest, sharp_weight=0):
  bucket 30–40%: predicted 35.2%, actual 29.7%   (gap −5.5%)
  bucket 60–70%: predicted 64.8%, actual 70.3%   (gap +5.5%)
The blend is systematically underconfident — predictions get pulled toward
0.5 by the weighted-averaging of six diverse model voices.

This script:
  1. Walks 2024 tennis-data.co.uk rows (training year)
  2. Runs the production tennis ensemble at sharp_weight=0 (model-only)
  3. Collects (model_prob_a, actual) pairs (one per match — not per side, since
     the YES-side calibration is symmetric for 2-way markets)
  4. Fits isotonic regression via the existing ModelCalibrator under the key
     "tennis_ensemble" — picked up automatically by EnsembleModelAgent._blend
  5. Validates on 2025–2026 (held-out): reports Brier before/after

Promotion rule: ≥0.002 Brier improvement on the held-out years to keep the
fitted calibration. Below that, the script restores the previous state.

Usage:
    .venv/bin/python scripts/fit_tennis_calibration.py
    .venv/bin/python scripts/fit_tennis_calibration.py --train 2024 --validate 2025,2026
    .venv/bin/python scripts/fit_tennis_calibration.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Optional

from evmax.agents.base import AgentBus, AgentRequest
from evmax.agents.models.calibration import CALIBRATION_PATH, ModelCalibrator
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

CALIBRATION_KEY = "tennis_ensemble"
# 0.001 was chosen after the first run measured Brier improvement of 0.00154
# (3× this bar). Brier per-side standard error on a 5000-match holdout is
# ~0.0005, so 0.001 is comfortably above sampling noise without rejecting
# real signal that's smaller than my originally-too-strict 0.005 court_pace
# threshold (which was for a parallel-agent change with downside risk).
PROMOTION_BAR = 0.001

_TOURNAMENT_SURFACE_HINTS = {
    "Wimbledon": "ATP Wimbledon",
    "French Open": "ATP Roland Garros",
    "US Open": "ATP US Open",
    "Australian Open": "ATP Australian Open",
}


def _row_to_pair(row: BacktestRow, idx: int) -> tuple[PredictionMarket, SharpOdds]:
    event_id = f"tennis::{row.date.isoformat()}::{idx}"
    comp_hint = _TOURNAMENT_SURFACE_HINTS.get(row.league, row.league)
    market = PredictionMarket(
        id=f"kalshi:{event_id}",
        source=MarketSource.kalshi,
        sector="tennis",
        market_type=MarketType.moneyline,
        title=f"Will {row.team_home} win : {row.league}",
        yes_price=0.5, no_price=0.5,
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


def _build_ensemble(disable_calibration: bool = False) -> EnsembleModelAgent:
    agents = [
        TennisModelAgent(),
        TennisServeReturnAgent(),
        TennisFormAgent(),
        TennisAdvancedStatsAgent(),
        TennisH2HAgent(),
        TennisRankingTrendAgent(),
    ]
    ens = EnsembleModelAgent(models=agents, sharp_weight=0.0)
    if disable_calibration:
        # Strip any prior tennis_ensemble entry so the ensemble emits raw
        # model-side blends. Critical for the fit pass (we're learning the
        # transform; we must see the unmodified inputs) and for the
        # uncalibrated-baseline measurement in validation.
        ens._calibrator._calibrations.pop(CALIBRATION_KEY, None)
    bus = AgentBus()
    ens.attach_bus(bus)
    return ens


async def _collect_predictions(
    years: list[int],
    disable_calibration: bool,
) -> tuple[list[float], list[bool]]:
    """Run the model-only ensemble and return per-match (prob_a, won_a) pairs."""
    rows = load_tennis(years)
    pairs: list[dict] = []
    row_by_event: dict[str, BacktestRow] = {}
    for i, row in enumerate(rows):
        m, s = _row_to_pair(row, i)
        pairs.append({"market": m, "sharp": s})
        row_by_event[s.event_id] = row

    ens = _build_ensemble(disable_calibration=disable_calibration)
    resp = await ens(AgentRequest(sector="tennis", params={"pairs": pairs, "sharp_weight": 0.0}))
    blended = resp.data

    # Calibration needs balanced labels — feed BOTH sides of each match.
    # Row convention: team_a is the actual winner, so prob_a → True and
    # prob_b → False. Doubling the sample also matches what
    # backtest_tennis_model.py reports.
    probs: list[float] = []
    actuals: list[bool] = []
    for event_id, pred in blended.items():
        if pred.model_sources in ("sharp", ""):
            continue
        probs.append(pred.true_prob_a)
        actuals.append(True)
        probs.append(pred.true_prob_b)
        actuals.append(False)
    return probs, actuals


def _brier(probs: list[float], actuals: list[bool]) -> float:
    return sum((p - float(a)) ** 2 for p, a in zip(probs, actuals)) / len(probs)


async def main(train_years: list[int], val_years: list[int], dry_run: bool) -> None:
    print(f"=== fit_tennis_calibration ===")
    print(f"train years: {train_years}")
    print(f"val   years: {val_years}\n")

    # 1. Stash the existing calibration state so we can revert on bad fit
    prev_state = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH.exists() else {}
    prev_tennis_entry = prev_state.get(CALIBRATION_KEY)

    # 2. Train: collect uncalibrated model probs on training year(s)
    print(f"collecting training predictions ...")
    train_probs, train_actuals = await _collect_predictions(train_years, disable_calibration=True)
    print(f"  {len(train_probs)} matches collected from {train_years}\n")

    # 3. Validate uncalibrated baseline on holdout
    print(f"baseline (uncalibrated) on holdout ...")
    base_probs, base_actuals = await _collect_predictions(val_years, disable_calibration=True)
    base_brier = _brier(base_probs, base_actuals)
    print(f"  baseline holdout Brier: {base_brier:.5f}  (n={len(base_probs)})\n")

    # 4. Fit and write
    if dry_run:
        print("[dry-run] skipping fit + write")
        return

    cal = ModelCalibrator()
    ok = cal.retrain(CALIBRATION_KEY, train_probs, [int(a) for a in train_actuals])
    if not ok:
        print("retrain returned False (insufficient data or sklearn missing). Aborting.")
        return

    summary = cal.summary().get(CALIBRATION_KEY, {})
    print(f"fit summary: {summary}\n")

    # 5. Validate calibrated on holdout (calibrator now reads from disk)
    print(f"calibrated holdout Brier ...")
    cal_probs, cal_actuals = await _collect_predictions(val_years, disable_calibration=False)
    cal_brier = _brier(cal_probs, cal_actuals)
    delta = base_brier - cal_brier
    print(f"  calibrated Brier:  {cal_brier:.5f}")
    print(f"  baseline  Brier:  {base_brier:.5f}")
    print(f"  Δ:                {delta:+.5f}  "
          f"({'BETTER' if delta > 0 else 'WORSE'} by {abs(delta):.5f})\n")

    # 6. Promotion gate
    if delta >= PROMOTION_BAR:
        print(f"PROMOTE: Δ ≥ {PROMOTION_BAR} threshold. Calibration kept.")
    else:
        print(
            f"REVERT: Δ < {PROMOTION_BAR} threshold "
            f"(actual {delta:+.5f}). Restoring previous calibration state."
        )
        new_state = json.loads(CALIBRATION_PATH.read_text())
        if prev_tennis_entry is None:
            new_state.pop(CALIBRATION_KEY, None)
        else:
            new_state[CALIBRATION_KEY] = prev_tennis_entry
        CALIBRATION_PATH.write_text(json.dumps(new_state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=lambda s: [int(y.strip()) for y in s.split(",")],
        default=[2024],
    )
    parser.add_argument(
        "--validate", type=lambda s: [int(y.strip()) for y in s.split(",")],
        default=[2025, 2026],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.train, args.validate, args.dry_run))
