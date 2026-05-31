"""Backtest the WNBA ML recalibration: 2026 re-seed + EB shrinkage + smooth confidence.

Replays each resolved WNBA ML bet from `ev_predictions` through the new
agents (efficiency + possession_sim + elo) blended at the same sharp_weight
used in production, and reports new Brier vs the historical blend.

Caveat: this is a point-in-time replay using TODAY's elo and the freshly
re-seeded 2026 efficiency state. Walk-forward would require reconstructing
the state per game-date, which is overkill for a calibration sanity check.
The interesting comparison is "if we ran the same matchups TODAY, would the
overall calibration be better?" — that's what this answers.
"""

from __future__ import annotations

import asyncio
import sqlite3
import statistics
from pathlib import Path
from typing import Optional

from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.wnba_efficiency_agent import WNBAEfficiencyModelAgent
from evmax.agents.models.wnba_possession_sim_agent import WNBAPossessionSimAgent
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data" / "predictions.db"

# WNBA sector weights — mirror evmax/agents/models/ensemble_agent.py
SECTOR_WEIGHTS = {
    "wnba_efficiency":     0.40,
    "wnba_possession_sim": 0.45,
    "elo":                 0.15,
}
SHARP_WEIGHT = 0.85         # default in model_config.json
CONFIDENCE_GATE = 0.45      # ensemble per-model gate


def parse_event_id(event_id: str, yes_team: str) -> tuple[str, str, str]:
    """Return (home_canonical, away_canonical, yes_team) from event_id.

    event_id format: `wnba::YYYY-MM-DD::{team_a}_vs_{team_b}`
    team_a is conventionally home in our pipeline (KXWNBAGAME tickers carry
    away first historically, but the stored event_id is canonical home_vs_away).
    """
    _, date_str, matchup = event_id.split("::")
    team_a, team_b = matchup.split("_vs_")
    return team_a, team_b, yes_team


def build_market_and_sharp(row: dict) -> tuple[PredictionMarket, SharpOdds]:
    home, away, yes_team = parse_event_id(row["event_id"], row["yes_team"])
    sharp_prob_a = row["sharp_true_prob"]
    # The stored sharp_true_prob is the YES-team perspective; rebuild a/b symmetric.
    # When yes_team == home, sharp.true_prob_a is YES-side; else swap.
    if yes_team == home:
        true_a, true_b = sharp_prob_a, 1.0 - sharp_prob_a
    else:
        true_a, true_b = 1.0 - sharp_prob_a, sharp_prob_a

    market = PredictionMarket(
        id=row["market_id"],
        event_id=row["event_id"],
        ticker=row["market_id"].replace("kalshi:", ""),
        title=f"{home} vs {away}",
        yes_price=row["kalshi_yes_price"],
        no_price=1.0 - row["kalshi_yes_price"],
        sector="wnba",
        source=MarketSource.kalshi,
        market_type=MarketType.moneyline,
        team_home=home,
        team_away=away,
        yes_team=yes_team,
    )
    sharp = SharpOdds(
        event_id=row["event_id"],
        book=SharpBook.pinnacle,
        sector="wnba",
        team_a=home,
        team_b=away,
        true_prob_a=true_a,
        true_prob_b=true_b,
        outcome_a_decimal=1.0 / max(true_a, 0.01),
        outcome_b_decimal=1.0 / max(true_b, 0.01),
        outcome_a_label=home,
        outcome_b_label=away,
    )
    return market, sharp


async def predict_one(
    market: PredictionMarket,
    sharp: SharpOdds,
    eff: WNBAEfficiencyModelAgent,
    sim: WNBAPossessionSimAgent,
    elo: EloModelAgent,
) -> Optional[dict]:
    """Run all three agents and produce a single YES-side blended probability."""
    eff_pred = await eff.predict_pair(market, sharp)
    sim_pred = await sim.predict_pair(market, sharp)
    elo_pred = await elo.predict_pair(market, sharp)

    preds = {}
    for name, p in (("wnba_efficiency", eff_pred), ("wnba_possession_sim", sim_pred), ("elo", elo_pred)):
        if p is None:
            continue
        if p.confidence < CONFIDENCE_GATE:
            continue
        preds[name] = p

    if not preds:
        # No model contribution → fall back to sharp alone
        model_prob_a = sharp.true_prob_a
        contributing = []
    else:
        total_w = sum(SECTOR_WEIGHTS[name] * preds[name].confidence for name in preds)
        model_prob_a = sum(
            SECTOR_WEIGHTS[name] * preds[name].confidence * preds[name].true_prob_a
            for name in preds
        ) / total_w
        contributing = list(preds.keys())

    final_a = SHARP_WEIGHT * sharp.true_prob_a + (1 - SHARP_WEIGHT) * model_prob_a
    yes_prob = final_a if market.yes_team == market.team_home else 1.0 - final_a
    return {
        "yes_prob_new": yes_prob,
        "n_contributing": len(contributing),
        "contributing": contributing,
        "yes_prob_sharp_only": sharp.true_prob_a if market.yes_team == market.team_home
                                else 1.0 - sharp.true_prob_a,
    }


async def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT ep.market_id, ep.event_id, ep.event_date, ep.yes_team,
               ep.kalshi_yes_price, ep.sharp_true_prob, ep.blended_true_prob,
               eo.outcome
          FROM ev_predictions ep
          JOIN ev_outcomes eo ON ep.market_id = eo.market_id
         WHERE eo.outcome IS NOT NULL
           AND ep.sector = 'wnba'
           AND ep.market_type = 'moneyline'
         ORDER BY ep.event_date ASC
        """
    ).fetchall()
    conn.close()

    eff = WNBAEfficiencyModelAgent()
    sim = WNBAPossessionSimAgent()
    elo = EloModelAgent()

    out_rows = []
    n_skipped = 0
    for r in rows:
        row = dict(r)
        try:
            market, sharp = build_market_and_sharp(row)
        except Exception as e:
            n_skipped += 1
            continue
        result = await predict_one(market, sharp, eff, sim, elo)
        if result is None:
            n_skipped += 1
            continue
        out_rows.append({**row, **result})

    if not out_rows:
        print("No rows replayed.")
        return

    def brier(pred_key: str) -> float:
        return statistics.mean((r[pred_key] - r["outcome"]) ** 2 for r in out_rows)

    def bias_pp(pred_key: str) -> float:
        return statistics.mean(r[pred_key] - r["outcome"] for r in out_rows) * 100

    n = len(out_rows)
    print(f"Replayed {n} WNBA ML bets ({n_skipped} skipped)")
    print()
    print(f"{'predictor':<22} {'Brier':>8}  {'bias_pp':>8}")
    print("-" * 42)
    print(f"{'historical blend':<22} {brier('blended_true_prob'):>8.4f}  {bias_pp('blended_true_prob'):>+8.2f}")
    print(f"{'kalshi listing':<22} {brier('kalshi_yes_price'):>8.4f}  {bias_pp('kalshi_yes_price'):>+8.2f}")
    print(f"{'sharp alone':<22} {brier('yes_prob_sharp_only'):>8.4f}  {bias_pp('yes_prob_sharp_only'):>+8.2f}")
    print(f"{'NEW blend (shrunk)':<22} {brier('yes_prob_new'):>8.4f}  {bias_pp('yes_prob_new'):>+8.2f}")
    print()

    # Bucket breakdown — same bucketing the diagnostic used
    def bucket(p: float) -> str:
        if p < 0.30: return "1. <30c"
        if p < 0.50: return "2. 30-50c"
        if p < 0.70: return "3. 50-70c"
        return "4. >70c"

    print(f"{'bucket':<10} {'n':>3} {'old_model':>9} {'new_model':>9} {'actual':>7} {'old_bias':>9} {'new_bias':>9}")
    print("-" * 66)
    buckets = {}
    for r in out_rows:
        b = bucket(r["kalshi_yes_price"])
        buckets.setdefault(b, []).append(r)
    for b in sorted(buckets):
        bs = buckets[b]
        if not bs:
            continue
        avg = lambda k: statistics.mean(r[k] for r in bs)
        print(f"{b:<10} {len(bs):>3} {avg('blended_true_prob'):>9.3f} {avg('yes_prob_new'):>9.3f} "
              f"{avg('outcome'):>7.3f} {(avg('blended_true_prob')-avg('outcome'))*100:>+8.2f} "
              f"{(avg('yes_prob_new')-avg('outcome'))*100:>+8.2f}")

    print()
    n_silenced = sum(1 for r in out_rows if r["n_contributing"] == 0)
    print(f"Bets where new model side fell back to sharp (no contributing models): {n_silenced} / {n}")
    n_partial = sum(1 for r in out_rows if 0 < r["n_contributing"] < 3)
    n_full = sum(1 for r in out_rows if r["n_contributing"] == 3)
    print(f"  partial contribution (1-2 models): {n_partial}")
    print(f"  full contribution (3 models):      {n_full}")


if __name__ == "__main__":
    asyncio.run(main())
