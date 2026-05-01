"""Seed nfl_efficiency_state.json from nflreadpy play-by-play.

Pulls multi-season PBP, applies garbage-time + valid-play filters, computes
per-team opponent-adjusted EPA on offense and defense, plus secondary stats
(success rate, explosive-play rate, red-zone TD rate). Writes state in the
shape consumed by NflEfficiencyModelAgent.

Re-run weekly during the season — there is no incremental update path
because per-game EPA cannot be reconstructed from a final score.

Usage:
    python scripts/seed_nfl_efficiency.py                # default: last 6 seasons
    python scripts/seed_nfl_efficiency.py --seasons 2024 # current only
    python scripts/seed_nfl_efficiency.py --dry-run      # print summary only

Filters applied to PBP:
  - play_type ∈ {"pass", "run"}            (drop punts, kicks, no-plays)
  - epa is not null                        (drop malformed rows)
  - wp ∈ [0.10, 0.90]                      (drop garbage-time)
  - season >= cutoff_season                (typically 6 most recent)

Opponent adjustment:
  raw_off_epa[T]  = mean of EPA on plays where T is on offense (post-filter)
  raw_def_epa[T]  = mean of EPA on plays where T is on defense (post-filter)
  league_avg_epa  = global mean of all valid plays
  off_epa_adj[T]  = raw_off_epa[T] − mean over all opponents' raw_def_epa
  def_epa_adj[T]  = raw_def_epa[T] − mean over all opponents' raw_off_epa
  (subtracts the strength of opponents faced — proper SoS correction)

Recency weighting:
  Plays in the most recent season are kept at full weight. Older seasons are
  exponentially decayed (weight = 0.45 ** seasons_back). With 6 seasons this
  means: current=1.0, t-1=0.45, t-2=0.20, t-3=0.09, t-4=0.04, t-5=0.018.
  Weights are applied via Polars groupby on (team, season) → weighted mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Make the evmax package importable when running this script directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import polars as pl
import nflreadpy as nfl

from evmax.agents.models.nfl_efficiency_agent import NFL_ABBREV_TO_NAME, STATE_PATH

# Recency decay applied per-season. 0.45 picks ~half-life of 1 season —
# strong recency bias is appropriate for NFL because rosters and schemes
# turn over fast (free agency + draft + coaching changes).
SEASON_DECAY = 0.45
DEFAULT_NUM_SEASONS = 6
WP_LOWER, WP_UPPER = 0.10, 0.90
EXPLOSIVE_YARDS = 15  # plays gaining 15+ yards


def filter_valid_pbp(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the standard EPA-quality filter: real plays, not garbage time.

    Public so the backtest can call it directly on a once-loaded PBP frame
    rather than re-loading per cutoff.
    """
    return df.filter(
        pl.col("play_type").is_in(["pass", "run"])
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("wp").is_between(WP_LOWER, WP_UPPER, closed="both")
    )


# Backwards-compat alias for any in-tree callers
_filter_valid = filter_valid_pbp


def compute_team_stats(df: pl.DataFrame, current_season: int) -> dict[str, dict]:
    """Group by team and compute weighted, opponent-adjusted EPA stats.

    Returns: {full_lowercase_team_name: {off_epa_adj, def_epa_adj, ...}}
    """
    # Per-season decay weight per row. Using Polars expression so this
    # vectorizes cleanly across millions of rows.
    df = df.with_columns(
        (pl.lit(SEASON_DECAY) ** (current_season - pl.col("season"))).alias("w")
    )

    # Raw offensive stats (each row's `posteam` is the offense)
    off = (
        df.group_by("posteam")
        .agg([
            ((pl.col("epa") * pl.col("w")).sum() / pl.col("w").sum()).alias("raw_off_epa"),
            ((pl.col("success") * pl.col("w")).sum() / pl.col("w").sum()).alias("raw_off_success"),
            (((pl.col("yards_gained") >= EXPLOSIVE_YARDS).cast(pl.Float64) * pl.col("w")).sum()
             / pl.col("w").sum()).alias("raw_off_explosive"),
            pl.col("game_id").n_unique().alias("gp"),
        ])
        .rename({"posteam": "team"})
    )

    # Raw defensive stats (each row's `defteam` is the defense; EPA from offense's POV)
    defn = (
        df.group_by("defteam")
        .agg([
            ((pl.col("epa") * pl.col("w")).sum() / pl.col("w").sum()).alias("raw_def_epa"),
            ((pl.col("success") * pl.col("w")).sum() / pl.col("w").sum()).alias("raw_def_success"),
            (((pl.col("yards_gained") >= EXPLOSIVE_YARDS).cast(pl.Float64) * pl.col("w")).sum()
             / pl.col("w").sum()).alias("raw_def_explosive"),
        ])
        .rename({"defteam": "team"})
    )

    # Strength-of-schedule adjustment: subtract average opponent's raw stat.
    # For offense: subtract mean(raw_def_epa over opponents A actually faced).
    # For defense: subtract mean(raw_off_epa over opponents A actually faced).
    sched = (
        df.group_by(["posteam", "defteam"])
        .agg(pl.col("w").sum().alias("plays"))
    )

    off_to_avg_opp_def = (
        sched.join(defn, left_on="defteam", right_on="team", how="left")
        .group_by("posteam")
        .agg(
            ((pl.col("raw_def_epa") * pl.col("plays")).sum() / pl.col("plays").sum())
            .alias("avg_opp_def_epa")
        )
        .rename({"posteam": "team"})
    )
    def_to_avg_opp_off = (
        sched.join(off, left_on="posteam", right_on="team", how="left")
        .group_by("defteam")
        .agg(
            ((pl.col("raw_off_epa") * pl.col("plays")).sum() / pl.col("plays").sum())
            .alias("avg_opp_off_epa")
        )
        .rename({"defteam": "team"})
    )

    merged = (
        off.join(defn, on="team", how="inner")
        .join(off_to_avg_opp_def, on="team", how="left")
        .join(def_to_avg_opp_off, on="team", how="left")
    )

    # Red-zone TD rate (drives entering yardline_100 <= 20).
    rz = (
        df.filter(pl.col("yardline_100") <= 20)
        .group_by("posteam")
        .agg([
            ((pl.col("touchdown") * pl.col("w")).sum() / pl.col("w").sum())
            .alias("rz_td_rate_off"),
        ])
        .rename({"posteam": "team"})
    )
    rz_def = (
        df.filter(pl.col("yardline_100") <= 20)
        .group_by("defteam")
        .agg([
            ((pl.col("touchdown") * pl.col("w")).sum() / pl.col("w").sum())
            .alias("rz_td_rate_def"),
        ])
        .rename({"defteam": "team"})
    )
    merged = merged.join(rz, on="team", how="left").join(rz_def, on="team", how="left")

    out: dict[str, dict] = {}
    for row in merged.iter_rows(named=True):
        abbrev = row["team"]
        full_name = NFL_ABBREV_TO_NAME.get(abbrev)
        if full_name is None:
            # Unknown abbreviation (relocated team, all-star game, etc.) — skip
            continue

        off_epa_adj = (row["raw_off_epa"] or 0.0) - (row["avg_opp_def_epa"] or 0.0)
        def_epa_adj = (row["raw_def_epa"] or 0.0) - (row["avg_opp_off_epa"] or 0.0)

        out[full_name] = {
            "abbrev": abbrev,
            "off_epa_adj": round(off_epa_adj, 4),
            "def_epa_adj": round(def_epa_adj, 4),
            "off_success_rate": round(row["raw_off_success"] or 0.0, 4),
            "def_success_rate": round(row["raw_def_success"] or 0.0, 4),
            "off_explosive_rate": round(row["raw_off_explosive"] or 0.0, 4),
            "def_explosive_rate": round(row["raw_def_explosive"] or 0.0, 4),
            "rz_td_rate_off": round(row["rz_td_rate_off"] or 0.0, 4),
            "rz_td_rate_def": round(row["rz_td_rate_def"] or 0.0, 4),
            "gp": int(row["gp"] or 0),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed NFL efficiency state from nflreadpy PBP")
    ap.add_argument(
        "--seasons",
        type=str,
        default="",
        help="Comma-separated seasons (e.g. '2020,2021,2022,2023,2024,2025'). "
             f"Default: last {DEFAULT_NUM_SEASONS} seasons ending with the current one.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print summary, don't write state file")
    args = ap.parse_args()

    if args.seasons:
        seasons = [int(s) for s in args.seasons.split(",")]
    else:
        current = nfl.get_current_season()
        seasons = list(range(current - DEFAULT_NUM_SEASONS + 1, current + 1))

    print(f"Loading PBP for seasons: {seasons}")
    df = nfl.load_pbp(seasons=seasons)
    print(f"  total rows: {len(df):,}")

    df = _filter_valid(df)
    print(f"  rows after filter: {len(df):,}")

    current_season = max(seasons)
    teams = compute_team_stats(df, current_season)

    if not teams:
        print("ERROR: no team rows produced — abort", file=sys.stderr)
        return 1

    state = {
        "nfl": {
            "teams": teams,
            "fetched_at": date.today().isoformat(),
            "seasons_used": seasons,
            "season_decay": SEASON_DECAY,
            "wp_window": [WP_LOWER, WP_UPPER],
        }
    }

    # Sort sample for sanity check by net EPA
    sample = sorted(
        teams.items(),
        key=lambda kv: kv[1]["off_epa_adj"] - kv[1]["def_epa_adj"],
        reverse=True,
    )
    print(f"\nTop 5 by net opponent-adjusted EPA:")
    for name, s in sample[:5]:
        print(f"  {name:30s} net={s['off_epa_adj'] - s['def_epa_adj']:+.3f} "
              f"(off={s['off_epa_adj']:+.3f} def={s['def_epa_adj']:+.3f}) gp={s['gp']}")
    print(f"\nBottom 5:")
    for name, s in sample[-5:]:
        print(f"  {name:30s} net={s['off_epa_adj'] - s['def_epa_adj']:+.3f} "
              f"(off={s['off_epa_adj']:+.3f} def={s['def_epa_adj']:+.3f}) gp={s['gp']}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(teams)} teams to {STATE_PATH}")
        return 0

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nWrote {len(teams)} teams to {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
