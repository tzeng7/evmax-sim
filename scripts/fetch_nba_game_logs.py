"""Pull NBA player game logs from nba_api for a given season.

Used by the player-props v2 backtest (TODO #22). Hits LeagueGameLog
month-by-month and writes parquet to data/historical_nba/.

Usage:
    # Default — 2024-25 regular season
    python scripts/fetch_nba_game_logs.py

    # Custom season + season type
    python scripts/fetch_nba_game_logs.py --season 2023-24
    python scripts/fetch_nba_game_logs.py --season 2024-25 --include-playoffs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "historical_nba"

# Season month ranges (regular season runs Oct → mid-Apr; playoffs Apr → Jun).
# Pull month-by-month to keep individual requests under the 30s nba_api timeout.
SEASON_MONTHS: dict[str, list[tuple[str, str]]] = {
    "2024-25": [
        ("2024-10-22", "2024-10-31"),
        ("2024-11-01", "2024-11-30"),
        ("2024-12-01", "2024-12-31"),
        ("2025-01-01", "2025-01-31"),
        ("2025-02-01", "2025-02-28"),
        ("2025-03-01", "2025-03-31"),
        ("2025-04-01", "2025-04-13"),
    ],
    "2023-24": [
        ("2023-10-24", "2023-10-31"),
        ("2023-11-01", "2023-11-30"),
        ("2023-12-01", "2023-12-31"),
        ("2024-01-01", "2024-01-31"),
        ("2024-02-01", "2024-02-29"),
        ("2024-03-01", "2024-03-31"),
        ("2024-04-01", "2024-04-14"),
    ],
}


def fetch_month(season: str, season_type: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch one month of player-game logs."""
    from nba_api.stats.endpoints import LeagueGameLog

    log = LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
        date_from_nullable=date_from,
        date_to_nullable=date_to,
        timeout=60,
    )
    df = log.get_data_frames()[0]
    df["SEASON_TYPE"] = season_type
    return df


def fetch_season(season: str, include_playoffs: bool) -> pd.DataFrame:
    months = SEASON_MONTHS.get(season)
    if not months:
        raise ValueError(f"No month range configured for season {season}. "
                         f"Add to SEASON_MONTHS dict.")

    frames: list[pd.DataFrame] = []
    for date_from, date_to in months:
        print(f"  Regular season {date_from} → {date_to}: ", end="", flush=True)
        df = fetch_month(season, "Regular Season", date_from, date_to)
        print(f"{len(df)} player-games, {df.GAME_ID.nunique()} games")
        frames.append(df)
        time.sleep(0.6)   # polite spacing between requests

    if include_playoffs:
        # Playoffs typically run mid-April through mid-June
        po_from = months[-1][1]   # day after RS ended
        # Use a fat upper bound; nba_api truncates to actual game dates
        po_to = f"{int(season[:4]) + 1}-06-30"
        print(f"  Playoffs {po_from} → {po_to}: ", end="", flush=True)
        df = fetch_month(season, "Playoffs", po_from, po_to)
        print(f"{len(df)} player-games, {df.GAME_ID.nunique()} games")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--season", default="2024-25",
                   help="Season string (e.g. 2024-25)")
    p.add_argument("--include-playoffs", action="store_true",
                   help="Also fetch playoff player-game logs")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {args.season} player game logs"
          f"{' + playoffs' if args.include_playoffs else ''}...")
    df = fetch_season(args.season, args.include_playoffs)

    # Normalize the date column to ISO YYYY-MM-DD strings (already is, but be explicit)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    out = OUT_DIR / f"player_game_logs_{args.season.replace('-', '_')}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {out}")
    print(f"  Total player-games: {len(df):,}")
    print(f"  Distinct games: {df.GAME_ID.nunique():,}")
    print(f"  Distinct players: {df.PLAYER_ID.nunique():,}")
    print(f"  Date range: {df.GAME_DATE.min()} → {df.GAME_DATE.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
