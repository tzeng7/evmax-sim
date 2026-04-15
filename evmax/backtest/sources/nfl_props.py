"""Backtest loader for NFL player props.

Reads the Stage 3 join output (`data/backtest/nfl_props/joined.parquet`)
and the Stage 2 nflverse weekly stats parquet, then — for every settled
market — builds point-in-time features and calls the pure probability
model in `evmax.clients.nfl_props_cache`.

All features are computed strictly from games in the same season
occurring **before** the target week. No leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evmax.backtest.models import PropBacktestRow
from evmax.clients.nfl_props_cache import (
    LAST_N_GAMES,
    MIN_GAMES,
    compute_nfl_prop_prob,
    compute_opponent_adjustment,
)

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path("data/backtest/nfl_props")

STAT_TO_COLUMN: dict[str, str] = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "passing_tds": "passing_tds",
    # Anytime TD — compose across rush + rec + (rare) pass TDs per game
    "anytime_td": "__anytime_td__",
}

# For opponent adjustment: what does "allowed" mean for this player prop?
# We look up defensive stats from the perspective of the player's position.
OPP_STAT_PER_PROP: dict[str, str] = {
    "passing_yards": "pass_yards_allowed",
    "rushing_yards": "rush_yards_allowed",
    "receiving_yards": "pass_yards_allowed",
    "receptions": "pass_yards_allowed",
    "passing_tds": "pass_tds_allowed",
    "anytime_td": "tds_allowed",
}


def _prepare_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Coerce + add synthetic columns used by the loader."""
    # Anytime TD column — any offensive TD in the game.
    for col in ("rushing_tds", "receiving_tds", "passing_tds"):
        if col not in weekly.columns:
            weekly[col] = 0
    weekly = weekly.copy()
    weekly["anytime_td"] = (
        weekly["rushing_tds"].fillna(0)
        + weekly["receiving_tds"].fillna(0)
        + (weekly["passing_tds"].fillna(0) > 0).astype(int)
    )
    # Ensure int week/season
    weekly["week"] = weekly["week"].astype(int)
    weekly["season"] = weekly["season"].astype(int)
    return weekly


def _build_defense_table(weekly: pd.DataFrame) -> dict[tuple[int, int, str], dict[str, float]]:
    """Return {(season, week, team): {stat: allowed_per_game_to_date}}.

    For each (season, week, team) we sum up the stats the team's opponents
    generated against them across all prior weeks in the same season, then
    divide by the number of those games. This is the feature used to
    compute the multiplicative opponent adjustment in the model.
    """
    df = weekly.copy()
    df = df.dropna(subset=["opponent_team"])
    # Group by (season, week, opponent_team) — the opponent is the defense
    # giving up these stats. Sum stats by defense-week-season.
    by_def = (
        df.groupby(["season", "week", "opponent_team"])
        .agg(
            pass_yards_allowed=("passing_yards", "sum"),
            rush_yards_allowed=("rushing_yards", "sum"),
            pass_tds_allowed=("passing_tds", "sum"),
            rush_tds_allowed=("rushing_tds", "sum"),
            rec_tds_allowed=("receiving_tds", "sum"),
        )
        .reset_index()
        .rename(columns={"opponent_team": "team"})
    )
    by_def["tds_allowed"] = (
        by_def["rush_tds_allowed"] + by_def["rec_tds_allowed"]
        + (by_def["pass_tds_allowed"] > 0).astype(int)
    )

    # Cumulative-to-date (strictly before the target week)
    by_def = by_def.sort_values(["season", "team", "week"])
    stat_cols = [
        "pass_yards_allowed",
        "rush_yards_allowed",
        "pass_tds_allowed",
        "tds_allowed",
    ]
    for col in stat_cols:
        by_def[f"cum_{col}"] = by_def.groupby(["season", "team"])[col].cumsum()
    by_def["games_played"] = by_def.groupby(["season", "team"]).cumcount() + 1

    # Build the lookup: (season, week, team) → point-in-time averages.
    # IMPORTANT: for target_week=T, averages must reflect games through
    # week T-1. So we shift the cumulative by one game per team.
    out: dict[tuple[int, int, str], dict[str, float]] = {}
    for _, row in by_def.iterrows():
        prior_games = row["games_played"] - 1
        if prior_games < 1:
            continue
        key = (int(row["season"]), int(row["week"]), row["team"])
        # Subtract this week's contribution to get prior-to-week
        prior = {}
        for col in stat_cols:
            cum = row[f"cum_{col}"] - row[col]
            prior[col] = float(cum / prior_games) if prior_games > 0 else 0.0
        out[key] = prior
    return out


def _league_averages(
    defense: dict[tuple[int, int, str], dict[str, float]]
) -> dict[tuple[int, int], dict[str, float]]:
    """League-average defensive stats by (season, week). Used as the
    baseline for the opponent adjustment multiplier."""
    agg: dict[tuple[int, int], list[dict[str, float]]] = {}
    for (season, week, _team), stats in defense.items():
        agg.setdefault((season, week), []).append(stats)
    out: dict[tuple[int, int], dict[str, float]] = {}
    for key, rows in agg.items():
        if not rows:
            continue
        combined: dict[str, float] = {}
        for stat in rows[0]:
            vals = [r[stat] for r in rows]
            combined[stat] = float(np.mean(vals)) if vals else 0.0
        out[key] = combined
    return out


def _player_history(
    by_player: dict[str, pd.DataFrame],
    player_id: str,
    season: int,
    target_week: int,
    stat_col: str,
) -> np.ndarray:
    """Return most-recent-first array of the player's prior-week stats
    within the same season. Limited to LAST_N_GAMES.

    `by_player` is a pre-grouped dict `{player_id: full-season frame}` so
    we avoid re-filtering a 38k-row DataFrame on every call — this turns
    the backtest from O(markets × rows) into O(markets × player_games).
    """
    player_df = by_player.get(player_id)
    if player_df is None:
        return np.array([], dtype=float)
    sub = player_df[
        (player_df["season"] == season) & (player_df["week"] < target_week)
    ]
    if len(sub) == 0:
        return np.array([], dtype=float)
    sub = sub.sort_values("week", ascending=False).head(LAST_N_GAMES)
    return sub[stat_col].fillna(0).to_numpy(dtype=float)


def load_nfl_props(
    data_dir: Path = DEFAULT_DIR,
    stats_filter: list[str] | None = None,
) -> list[PropBacktestRow]:
    """Load every joined.parquet row, score the model, return PropBacktestRows.

    Args:
      data_dir: directory containing joined.parquet + weekly_stats.parquet.
      stats_filter: if given, only markets whose stat_type appears in this
        list are scored. Used to narrow the backtest to a subset
        (e.g. ["passing_yards","passing_tds"] for QB-only analysis).
    """
    joined_path = data_dir / "joined.parquet"
    weekly_path = data_dir / "weekly_stats.parquet"
    if not joined_path.exists():
        raise FileNotFoundError(
            f"{joined_path} missing — run scripts/join_nfl_prop_backtest.py"
        )
    if not weekly_path.exists():
        raise FileNotFoundError(
            f"{weekly_path} missing — run scripts/fetch_nfl_features.py"
        )

    joined = pd.read_parquet(joined_path)
    weekly = pd.read_parquet(weekly_path)
    weekly = _prepare_weekly(weekly)

    defense = _build_defense_table(weekly)
    league_avg = _league_averages(defense)

    # Pre-group weekly by player_id for fast per-market history lookups
    by_player: dict[str, pd.DataFrame] = {
        pid: grp for pid, grp in weekly.groupby("player_id")
    }

    # Pre-filter to rows that have everything we need
    joined = joined.dropna(subset=["gsis_id", "season", "week", "team", "opponent"])
    joined = joined[joined["week"] > MIN_GAMES]  # need priors
    if stats_filter:
        joined = joined[joined["stat_type"].isin(stats_filter)]

    rows: list[PropBacktestRow] = []
    n_skipped_nohistory = 0
    n_skipped_thinsample = 0

    for rec in joined.itertuples(index=False):
        stat = rec.stat_type
        col = STAT_TO_COLUMN.get(stat)
        if col is None:
            continue
        season = int(rec.season)
        week = int(rec.week)

        # Player history (point-in-time)
        history_col = col if col != "__anytime_td__" else "anytime_td"
        hist = _player_history(by_player, rec.gsis_id, season, week, history_col)
        if len(hist) == 0:
            n_skipped_nohistory += 1
            continue
        if len(hist) < MIN_GAMES:
            n_skipped_thinsample += 1
            continue

        # Opponent adjustment
        opp_key = (season, week, rec.opponent)
        league_key = (season, week)
        opp_stat_name = OPP_STAT_PER_PROP.get(stat, "pass_yards_allowed")
        opp_allowed = defense.get(opp_key, {}).get(opp_stat_name, 0.0)
        lg_allowed = league_avg.get(league_key, {}).get(opp_stat_name, 0.0)
        opp_adj = compute_opponent_adjustment(stat, opp_allowed, lg_allowed)

        # Run the model
        result = compute_nfl_prop_prob(
            stat_type=stat,
            threshold=float(rec.threshold),
            values=hist,
            opp_adj=opp_adj,
        )
        if result is None:
            n_skipped_thinsample += 1
            continue
        prob, n_used = result

        rows.append(
            PropBacktestRow(
                sector="nfl_props",
                stat_type=stat,
                player_name=rec.player_name,
                player_id=rec.gsis_id,
                team=rec.team,
                opponent=rec.opponent,
                is_home=bool(rec.is_home),
                season=season,
                week=week,
                game_date=str(rec.game_date),
                threshold=float(rec.threshold),
                actual_result=int(rec.result),
                volume=float(rec.volume_fp or 0.0),
                closing_yes_price=(
                    float(rec.last_price_dollars)
                    if rec.last_price_dollars is not None and not pd.isna(rec.last_price_dollars)
                    else None
                ),
                model_prob=prob,
                n_games_used=n_used,
            )
        )

    logger.info(
        "nfl_prop_backtest_loaded",
        n_scored=len(rows),
        n_skipped_nohistory=n_skipped_nohistory,
        n_skipped_thinsample=n_skipped_thinsample,
    )
    return rows
