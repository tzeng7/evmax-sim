"""One row per player per game: who PLAYED (offensive snaps) and their official stats.

Sources: nflverse official weekly player stats (what sportsbooks and Kalshi
settle on) joined with PFR snap counts. A player with offensive snaps but no
recorded stat gets explicit zeros — dropping those rows would hide every
zero-catch game (the resolver lesson from the NFL prop work).

Each row also carries its team's totals for the game (targets, carries, pass
attempts), so shares are ``player stat / team total``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from evmax.nfl_projections import data

SCHEMA_VERSION = 1
STATS = ["targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
         "carries", "rushing_yards", "rushing_tds", "attempts", "completions", "passing_yards",
         "passing_tds", "sacks_suffered"]
OFFENSE_STATS = ["targets", "receptions", "carries", "attempts"]


def build_player_games(weekly: pd.DataFrame, snaps: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    w = weekly[["game_id", "player_id", "player_display_name", "position", "team"] + STATS].copy()
    w[STATS] = w[STATS].fillna(0.0)
    w = w[w[OFFENSE_STATS].sum(axis=1) > 0]                     # offensive participants only
    sn = snaps[snaps["player_id"].notna() & (snaps["offense_snaps"] > 0)][
        ["game_id", "player_id", "player", "position", "team", "offense_snaps", "offense_pct"]]
    pg = w.merge(sn, on=["game_id", "player_id"], how="outer", suffixes=("", "_snap"))
    pg["team"] = pg["team"].fillna(pg["team_snap"])
    pg["position"] = pg["position"].fillna(pg["position_snap"])
    pg["player_display_name"] = pg["player_display_name"].fillna(pg["player"])
    pg[STATS] = pg[STATS].fillna(0.0)
    pg[["offense_snaps", "offense_pct"]] = pg[["offense_snaps", "offense_pct"]].fillna(0.0)

    tot = pg.groupby(["game_id", "team"])[["targets", "carries", "attempts"]].sum().add_prefix("team_")
    pg = pg.join(tot, on=["game_id", "team"])

    g = games[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team"]]
    pg = pg.merge(g, on="game_id", how="inner")
    pg["opp"] = np.where(pg["team"] == pg["home_team"], pg["away_team"], pg["home_team"])
    pg["season_type"] = np.where(pg["game_type"] == "REG", "REG", "POST")
    cols = (["game_id", "season", "week", "season_type", "gameday", "team", "opp", "player_id",
             "player_display_name", "position", "offense_snaps", "offense_pct"] + STATS
            + ["team_targets", "team_carries", "team_attempts"])
    return pg[cols].sort_values(["gameday", "game_id", "team", "player_id"]).reset_index(drop=True)


def player_games_file(d: Optional[Path] = None) -> Path:
    return data.data_dir(d) / f"player_games_v{SCHEMA_VERSION}.parquet"


def load_player_games(seasons: Iterable[int], d: Optional[Path] = None,
                      rebuild: bool = False) -> pd.DataFrame:
    """Cached player-game table for ``seasons`` (rebuilt when a source file is newer)."""
    seasons = sorted(set(seasons))
    out = player_games_file(d)
    sources = ([data.player_week_file(s, d) for s in seasons] + [data.snaps_file(s, d) for s in seasons]
               + [data.games_file(d), data.players_file(d)])
    newest = max((p.stat().st_mtime for p in sources if p.exists()), default=0)
    if not rebuild and out.exists() and out.stat().st_mtime >= newest:
        pg = pd.read_parquet(out)
        if set(seasons) <= set(pg["season"].unique()):
            return pg[pg["season"].isin(seasons)].reset_index(drop=True)
    pg = build_player_games(data.load_player_week(seasons, d), data.load_snaps(seasons, d),
                            data.load_games(d))
    out.parent.mkdir(parents=True, exist_ok=True)
    pg.to_parquet(out, index=False)
    return pg
