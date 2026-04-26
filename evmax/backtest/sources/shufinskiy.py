"""Loader for shufinskiy/nba_data historical archives.

Reads the CSVs that scripts/fetch_shufinskiy.py extracts to
`data/historical_nba/`, with a parquet-backed cache so repeated reads
are sub-second.

Year convention: `_2025` means the season STARTING in 2025 (i.e. the
2025-26 NBA season). `_2024` = 2024-25, etc.

Coverage as of April 2026:
  - shotdetail_*: through 2025 (current season, ~219k shots, 1230 games)
  - pbpstats_*:   through 2024 (last completed season, ~483k possessions)

Both files are keyed by GAME_ID (shotdetail) / GAMEID (pbpstats) so a
join across the two requires either a date alignment or per-row matching
on GAME_ID prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "historical_nba"


# ---------------------------------------------------------------------------
# Single-season loaders
# ---------------------------------------------------------------------------

def load_shot_details(season_start_year: int) -> pd.DataFrame:
    """Return one season of NBA shot details.

    `season_start_year=2025` returns the 2025-26 season's shots.
    """
    return _load_with_parquet_cache(f"shotdetail_{season_start_year}")


def load_pbpstats(season_start_year: int) -> pd.DataFrame:
    """Return one season of pbpstats per-possession events.

    `season_start_year=2024` returns the 2024-25 season's possessions.
    Note: pbpstats lags one season behind shot details upstream.
    """
    return _load_with_parquet_cache(f"pbpstats_{season_start_year}")


def load_shot_details_multi(season_start_years: list[int]) -> pd.DataFrame:
    """Concat shot details across multiple seasons. Adds `season_start` col."""
    frames = []
    for y in season_start_years:
        df = load_shot_details(y)
        df = df.copy()
        df["season_start"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_pbpstats_multi(season_start_years: list[int]) -> pd.DataFrame:
    """Concat pbpstats across multiple seasons. Adds `season_start` col."""
    frames = []
    for y in season_start_years:
        df = load_pbpstats(y)
        df = df.copy()
        df["season_start"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

def _load_with_parquet_cache(key: str) -> pd.DataFrame:
    """Read the CSV → parquet on first call, parquet on subsequent calls.

    Parquet is ~10x smaller and ~30x faster to load than the raw CSV for
    these files. Cache invalidates if the CSV's mtime is newer than the
    parquet's (re-fetched archive triggers a refresh).
    """
    csv_path = DATA_DIR / f"{key}.csv"
    parquet_path = DATA_DIR / f"{key}.parquet"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python scripts/fetch_shufinskiy.py "
            f"--keys {key}` to download it."
        )

    if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_parquet(parquet_path)

    # low_memory=False silences the DtypeWarning on pbpstats' URL column
    # (sometimes blank, sometimes a string); avoiding the chunked parser
    # also speeds the one-time conversion.
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Convenience helpers used by player-prop / shot-quality models
# ---------------------------------------------------------------------------

def player_shots(
    season_start_year: int,
    player_name: Optional[str] = None,
    player_id: Optional[int] = None,
) -> pd.DataFrame:
    """Filter shot details to one player by name (substring) or ID."""
    df = load_shot_details(season_start_year)
    if player_id is not None:
        return df[df.PLAYER_ID == player_id].copy()
    if player_name is not None:
        return df[df.PLAYER_NAME.str.contains(player_name, case=False, na=False)].copy()
    raise ValueError("provide player_name or player_id")


def game_possessions(season_start_year: int, game_id: int) -> pd.DataFrame:
    """All pbpstats possessions for one game."""
    df = load_pbpstats(season_start_year)
    return df[df.GAMEID == game_id].copy()


def possession_lengths(season_start_year: int) -> pd.Series:
    """Return possession durations in seconds for one season.

    Useful for calibrating possession_sim's per-team pace variance and the
    overall N(pace, σ=3) heuristic. Excludes possessions with malformed
    STARTTIME/ENDTIME strings.
    """
    df = load_pbpstats(season_start_year)
    return _extract_possession_lengths(df)


def _extract_possession_lengths(df: pd.DataFrame) -> pd.Series:
    def _to_seconds(s: str) -> float:
        try:
            mm, ss = s.split(":")
            return int(mm) * 60 + int(ss)
        except (ValueError, AttributeError):
            return float("nan")
    start_s = df.STARTTIME.map(_to_seconds)
    end_s = df.ENDTIME.map(_to_seconds)
    # STARTTIME and ENDTIME are clock-remaining strings; possession length =
    # start - end (clock counts down within a period). Filter out negatives
    # (period boundaries) and ridiculously long possessions (data errors).
    diffs = start_s - end_s
    return diffs[(diffs > 0) & (diffs < 30)]
