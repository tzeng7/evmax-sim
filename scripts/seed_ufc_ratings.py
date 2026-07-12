"""Seed data/models/ufc_rating_state.json walk-forward from ESPN history.

Reads the committed dataset (data/backtest/ufc/fights.csv + fighters.csv —
produced by scripts/fetch_ufc_history.py) and replays it chronologically
through the exact same code the live agent uses (replay_history), then
full-replaces the agent state. Idempotent; ABORTS WITHOUT WRITING when the
dataset is missing or empty (the tennis-seed precedent).

The fitted feature layer (data/backtest/ufc/feature_weights.json, written by
scripts/backtest_ufc_model.py) is embedded into the state only when its
recorded verdict is VALIDATED_BLEND — otherwise the agent runs rating-only.

Usage:
    python scripts/seed_ufc_ratings.py             # seed from committed CSVs
    python scripts/seed_ufc_ratings.py --fetch     # refresh dataset first
    python scripts/seed_ufc_ratings.py --no-features   # force rating-only

Weekly cadence: run with --fetch after each UFC card (or ride a scheduled
task) so `last_updated` stays inside the agent's 60-day staleness guard.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.agents.models.ufc_rating_agent import UFCRatingAgent

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "backtest" / "ufc"
WEIGHTS_PATH = DATA_DIR / "feature_weights.json"


def main(fetch: bool, use_features: bool) -> None:
    if fetch:
        from scripts.fetch_ufc_history import main as fetch_main
        asyncio.run(fetch_main("2010-01", with_methods=True))

    fights_path = DATA_DIR / "fights.csv"
    fighters_path = DATA_DIR / "fighters.csv"
    if not fights_path.exists() or not fighters_path.exists():
        sys.exit(f"ABORT: dataset missing at {DATA_DIR} — run scripts/fetch_ufc_history.py")

    with fights_path.open() as fh:
        fights = list(csv.DictReader(fh))
    with fighters_path.open() as fh:
        bios = {r["athlete_id"]: r for r in csv.DictReader(fh)}
    if not fights:
        sys.exit("ABORT: empty fights.csv — refusing to clobber existing state")
    fights.sort(key=lambda f: (f["date"], f["event_id"], f["comp_id"]))

    feature_model = None
    if use_features and WEIGHTS_PATH.exists():
        fm = json.loads(WEIGHTS_PATH.read_text())
        if fm.get("verdict") == "VALIDATED_BLEND":
            feature_model = fm
            print(f"feature layer: VALIDATED_BLEND (fitted through {fm.get('fitted_through')})")
        else:
            print(f"feature layer NOT embedded (verdict={fm.get('verdict')}) — rating-only")
    else:
        print("feature layer: none — rating-only")

    agent = UFCRatingAgent()
    n = agent.seed_from_history(fights, bios, feature_model=feature_model)
    if n == 0:
        sys.exit("ABORT: seed produced no fighters — state untouched")

    ratings = {
        k: v for k, v in agent._state["fighters"].items() if v.get("n_fights", 0) >= 5
    }
    top = sorted(ratings.items(), key=lambda kv: kv[1]["rating"], reverse=True)[:12]
    print(f"seeded {n} fighters through {agent._state.get('last_updated')}")
    print("top rated (n_fights ≥ 5):")
    for name, rec in top:
        print(f"  {rec['rating']:7.1f} ±{rec['rd']:5.1f}  {name}  ({rec['wins']}-{rec['losses']}, n={rec['n_fights']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="refresh the ESPN dataset first")
    ap.add_argument("--no-features", action="store_true", help="skip the fitted feature layer")
    args = ap.parse_args()
    main(fetch=args.fetch, use_features=not args.no_features)
