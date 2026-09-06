"""Seed nfl_qb_elo_state.json by walk-forwarding through nflreadpy PBP.

For each historical NFL game:
  1. Identify the starting QB per team (most pass attempts where posteam = team)
  2. Predict using the current state
  3. Update state via apply_qb_elo_update (split: QB_UPDATE_SHARE to QB rating,
     1 - QB_UPDATE_SHARE to team_base)

The most recent starter per team is recorded in `current_starters` so live
predictions can look up the QB delta without external roster fetches.

Re-seed weekly during the season — same cadence as nfl_efficiency. Each run
takes ~30s.

Usage:
    python scripts/seed_nfl_qb_elo.py                 # last 6 NFL seasons
    python scripts/seed_nfl_qb_elo.py --seasons 2024  # current only
    python scripts/seed_nfl_qb_elo.py --dry-run       # print summary, no write
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import polars as pl
import nflreadpy as nfl

from evmax.agents.models.nfl_efficiency_agent import NFL_ABBREV_TO_NAME
from evmax.agents.models.nfl_qb_elo_agent import (
    STATE_PATH,
    apply_qb_elo_update,
)

DEFAULT_NUM_SEASONS = 6


def _starters_per_game(df: pl.DataFrame) -> pl.DataFrame:
    """For each (game_id, posteam) pair return the player_name with the most
    pass attempts. That's the starter. Ties broken alphabetically (rare)."""
    return (
        df.filter(pl.col("passer_player_name").is_not_null())
        .group_by(["game_id", "posteam", "passer_player_name"])
        .agg(pl.len().alias("att"))
        .sort(["game_id", "posteam", "att"], descending=[False, False, True])
        .group_by(["game_id", "posteam"], maintain_order=True)
        .first()
        .select(["game_id", "posteam", "passer_player_name"])
        .rename({"passer_player_name": "starter"})
    )


def _games_with_starters(df: pl.DataFrame) -> pl.DataFrame:
    """One row per game with home_team, away_team, scores, starters, date."""
    games = (
        df.group_by(["game_id", "home_team", "away_team", "season", "week", "game_date"])
        .agg([
            pl.col("home_score").max().alias("home_score"),
            pl.col("away_score").max().alias("away_score"),
        ])
    )
    starters = _starters_per_game(df)
    home_starters = (
        starters.rename({"posteam": "home_team", "starter": "home_starter"})
    )
    away_starters = (
        starters.rename({"posteam": "away_team", "starter": "away_starter"})
    )
    out = (
        games.join(home_starters, on=["game_id", "home_team"], how="left")
        .join(away_starters, on=["game_id", "away_team"], how="left")
        .sort("game_date")
    )
    return out


def _walk_forward(games: pl.DataFrame) -> dict:
    """Build the qb-elo state by walking games chronologically. Returns the
    full sector dict (team_base / qb_deltas / qb_games / current_starters)."""
    state: dict = {"team_base": {}, "qb_deltas": {}, "qb_games": {}, "current_starters": {}}
    seasons_used = set()

    for row in games.iter_rows(named=True):
        home_abbr = row["home_team"]
        away_abbr = row["away_team"]
        home_team = NFL_ABBREV_TO_NAME.get(home_abbr)
        away_team = NFL_ABBREV_TO_NAME.get(away_abbr)
        if home_team is None or away_team is None:
            continue
        if row["home_score"] is None or row["away_score"] is None:
            continue
        if row["home_score"] == row["away_score"]:
            # NFL ties are rare — skip rather than awarding 0.5 (Elo update
            # math here is binary). One missed game per ~thousand has zero
            # impact on long-run ratings.
            continue

        home_won = row["home_score"] > row["away_score"]
        home_starter = row.get("home_starter")
        away_starter = row.get("away_starter")

        apply_qb_elo_update(
            state,
            home_team=home_team,
            away_team=away_team,
            home_starter=home_starter,
            away_starter=away_starter,
            home_won=home_won,
        )

        # Track most recent starter per team (overwrites until last game)
        if home_starter:
            state["current_starters"][home_team] = home_starter
        if away_starter:
            state["current_starters"][away_team] = away_starter

        seasons_used.add(int(row["season"]))

    return state, sorted(seasons_used)


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed NFL QB-Elo state from nflreadpy PBP")
    ap.add_argument("--seasons", type=str, default="",
                    help="Comma-separated seasons. Default: last 6 NFL seasons.")
    ap.add_argument("--dry-run", action="store_true", help="Print summary, don't write state.")
    args = ap.parse_args()

    if args.seasons:
        seasons = [int(s) for s in args.seasons.split(",")]
    else:
        current = nfl.get_current_season()
        seasons = list(range(current - DEFAULT_NUM_SEASONS + 1, current + 1))

    print(f"Loading PBP for seasons: {seasons}")
    # Load season-by-season so an unpublished season (nflverse hasn't posted the
    # current year's parquet yet, early in a new season) is skipped rather than
    # aborting the whole reseed — mirrors evmax/clients/nfl_props_cache.py.
    frames: list[pl.DataFrame] = []
    for season in seasons:
        try:
            frames.append(nfl.load_pbp(seasons=[season]))
        except Exception as exc:  # noqa: BLE001 — nflreadpy re-raises 404 as ConnectionError
            print(
                f"  season {season}: PBP not available yet, skipping "
                f"({type(exc).__name__})",
                file=sys.stderr,
            )
    if not frames:
        print("ERROR: no PBP loaded for any requested season — abort", file=sys.stderr)
        return 1
    df = pl.concat(frames, how="vertical_relaxed")
    print(f"  total rows: {len(df):,}")

    games = _games_with_starters(df)
    print(f"  unique games: {len(games)}")

    state, seasons_used = _walk_forward(games)
    if not seasons_used:
        print("ERROR: no games with identifiable starters — abort", file=sys.stderr)
        return 1

    out = {
        "nfl": {
            **state,
            "fetched_at": date.today().isoformat(),
            "seasons_used": seasons_used,
            "qb_update_share": 0.60,
        }
    }

    teams = state["team_base"]
    qbs = state["qb_deltas"]
    print(f"\nTeam base ratings: {len(teams)} teams")
    print(f"QB ratings:        {len(qbs)} QBs")

    # Top 5 QBs by current delta + min 6 starts
    qb_games = state["qb_games"]
    eligible = [(name, delta) for name, delta in qbs.items() if qb_games.get(name, 0) >= 6]
    eligible.sort(key=lambda kv: kv[1], reverse=True)
    print(f"\nTop 10 QBs (≥6 starts) by Elo delta:")
    for name, d in eligible[:10]:
        gp = qb_games.get(name, 0)
        print(f"  {name:20s} {d:+7.1f}  ({gp} starts)")
    print(f"\nBottom 5:")
    for name, d in eligible[-5:]:
        gp = qb_games.get(name, 0)
        print(f"  {name:20s} {d:+7.1f}  ({gp} starts)")

    print(f"\nCurrent starters (latest seen per team):")
    for team in sorted(state["current_starters"].keys()):
        starter = state["current_starters"][team]
        d = qbs.get(starter, 0.0)
        print(f"  {team:30s} {starter:18s} delta={d:+.1f}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write to {STATE_PATH}")
        return 0

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote state to {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
