"""Pull NFL feature data for the prop backtest directly from nflverse releases.

nfl_data_py is stale and hardcodes a dead URL (it targets the old
`player_stats/player_stats_{year}.parquet` release tag that nflverse
renamed to `stats_player/stats_player_week_{year}.parquet`). Rather than
add a broken wrapper as a dep, this script pulls the public parquet files
directly from GitHub release assets — no auth, stable URLs, pandas-version
agnostic.

Writes three files under `data/backtest/nfl_props/`:
  - weekly_stats.parquet   — per-player per-week game logs (2024–2025)
  - schedules.parquet      — every NFL game with teams, date, week, game_type
  - rosters.parquet        — per-player per-week team assignment

Usage:
    python3 scripts/fetch_nfl_features.py
    python3 scripts/fetch_nfl_features.py --seasons 2023,2024,2025
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
DEFAULT_SEASONS = [2024, 2025]
DEFAULT_OUT = Path("data/backtest/nfl_props")


def _fetch_parquet(url: str) -> pd.DataFrame:
    """Download a parquet file into memory and return a DataFrame.

    We can't pass GitHub release URLs straight to pandas.read_parquet
    because older pandas + pyarrow combinations choke on the 302
    redirect to S3. Urllib handles the redirect cleanly.
    """
    print(f"  ← {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "evmax-backtest"})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read()
    return pd.read_parquet(io.BytesIO(blob))


def fetch_weekly_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for y in seasons:
        url = f"{NFLVERSE}/stats_player/stats_player_week_{y}.parquet"
        df = _fetch_parquet(url)
        if "season" not in df.columns:
            df["season"] = y
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(
        f"  weekly_stats: {out.shape[0]} rows × {out.shape[1]} cols, "
        f"seasons={sorted(out.season.unique().tolist())}, "
        f"weeks={sorted(out.week.unique().tolist())}"
    )
    return out


def fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    url = f"{NFLVERSE}/schedules/games.parquet"
    df = _fetch_parquet(url)
    df = df[df.season.isin(seasons)].reset_index(drop=True)
    print(f"  schedules:    {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def fetch_rosters(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for y in seasons:
        url = f"{NFLVERSE}/weekly_rosters/roster_weekly_{y}.parquet"
        df = _fetch_parquet(url)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"  rosters:      {out.shape[0]} rows × {out.shape[1]} cols")
    return out


def spot_check(weekly: pd.DataFrame) -> None:
    """Verify point-in-time filtering works for a known Week 13 2025 game.

    The Stage 1 Kalshi dump confirmed KXNFLPASSYDS-25DEC01NYGNE had
    Jaxson Dart passing-yards markets. His prior-week (≤ Week 12) game log
    must be non-empty for the backtest's point-in-time feature build.
    """
    target_player = "Jaxson Dart"
    target_season = 2025
    target_week = 13
    hist = weekly[
        (weekly.player_display_name == target_player)
        & (weekly.season == target_season)
        & (weekly.week < target_week)
    ].sort_values("week")
    print()
    print(f"spot check — {target_player} games < Week {target_week} of {target_season}:")
    if hist.empty:
        print("  ! NO HISTORY FOUND — point-in-time filter will return empty features")
        return
    cols = ["week", "team", "opponent_team", "attempts", "passing_yards", "passing_tds"]
    cols = [c for c in cols if c in hist.columns]
    print(hist[cols].to_string(index=False))
    print(f"  → {len(hist)} games available for prior-week feature build")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seasons",
        type=str,
        default=",".join(str(y) for y in DEFAULT_SEASONS),
        help=f"comma-separated seasons (default: {','.join(str(y) for y in DEFAULT_SEASONS)})",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    seasons = sorted({int(s) for s in args.seasons.split(",") if s.strip()})
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"seasons: {seasons}")
    print(f"out dir: {args.out}")
    print()
    print("fetching weekly stats …")
    weekly = fetch_weekly_stats(seasons)
    print("fetching schedules …")
    schedules = fetch_schedules(seasons)
    print("fetching rosters …")
    rosters = fetch_rosters(seasons)

    weekly.to_parquet(args.out / "weekly_stats.parquet", index=False)
    schedules.to_parquet(args.out / "schedules.parquet", index=False)
    rosters.to_parquet(args.out / "rosters.parquet", index=False)
    print()
    print(f"wrote {args.out / 'weekly_stats.parquet'}")
    print(f"wrote {args.out / 'schedules.parquet'}")
    print(f"wrote {args.out / 'rosters.parquet'}")

    spot_check(weekly)
    return 0


if __name__ == "__main__":
    sys.exit(main())
