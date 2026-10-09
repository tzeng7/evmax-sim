"""One row per team per game, built from nflverse play-by-play + schedules.

Each row is the team's OFFENSIVE game: points scored, scrimmage plays, drives,
efficiency (EPA/play and success rate on competitive snaps), pace, turnovers,
red-zone finishing and its starting QB. The opponent's row of the same game is
that team's defensive game, so ratings fit ``metric = mu + off[team] +
def[opp] + home`` on these rows directly.

Competitive snaps: pre-snap win probability in [0.10, 0.90] (the same
garbage-time cut ``scripts/seed_nfl_efficiency.py`` uses). Counts (plays,
drives, points) use every snap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from evmax.nfl_projections import data

# Bump when the row definition changes so cached tables are rebuilt.
SCHEMA_VERSION = 2
WP_LO, WP_HI = 0.10, 0.90


def _mmss_to_seconds(s: pd.Series) -> pd.Series:
    parts = s.astype("string").str.split(":", n=1, expand=True)
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def build_team_games(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play to team-game rows and join schedule context.

    Returns columns: game_id, season, week, season_type, gameday, team, opp,
    home (1 home / 0 away / 0 both sides at a neutral site), neutral,
    points_for, points_against, plays, dropbacks, pass_rate, epa_pp, sr,
    pass_epa, rush_epa, proe, drives, top_s, sec_per_play, turnovers,
    rz_drives, rz_td, starter_id, starter_name, starter_dropbacks, starter_epa
    (the most-dropback passer — a POST-game fact) and first_qb_id/name (the
    passer on the first dropback — the pre-game starter).
    """
    scrim = pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1)) & pbp["posteam"].notna()
                & pbp["epa"].notna() & (pbp["play_type"] != "no_play")].copy()
    comp = scrim[(scrim["wp"] >= WP_LO) & (scrim["wp"] <= WP_HI)]

    key = ["game_id", "posteam"]
    counts = scrim.groupby(key).agg(
        plays=("epa", "size"),
        dropbacks=("qb_dropback", "sum"),
        pass_rate=("pass", "mean"),
        turnovers=("interception", "sum"),
        fum_lost=("fumble_lost", "sum"),
    )
    counts["turnovers"] = counts["turnovers"] + counts.pop("fum_lost")
    eff = comp.groupby(key).agg(epa_pp=("epa", "mean"), sr=("success", "mean"),
                                proe=("pass_oe", "mean"), comp_plays=("epa", "size"))
    pass_eff = comp[comp["pass"] == 1].groupby(key)["epa"].mean().rename("pass_epa")
    rush_eff = comp[comp["rush"] == 1].groupby(key)["epa"].mean().rename("rush_epa")

    # Drives: every distinct fixed_drive the team had the ball on.
    dr = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()].drop_duplicates(
        ["game_id", "posteam", "fixed_drive"])
    dr = dr.assign(top_s=_mmss_to_seconds(dr["drive_time_of_possession"]),
                   rz=(dr["drive_inside20"] == 1).astype(int),
                   rz_td=((dr["drive_inside20"] == 1) & (dr["fixed_drive_result"] == "Touchdown")).astype(int))
    drives = dr.groupby(key).agg(drives=("fixed_drive", "size"), top_s=("top_s", "sum"),
                                 rz_drives=("rz", "sum"), rz_td=("rz_td", "sum"))

    # Starter: the passer with the most dropbacks for the team in that game.
    qb = scrim[scrim["passer_player_id"].notna()]
    qb_agg = qb.groupby(key + ["passer_player_id", "passer_player_name"]).agg(
        n=("qb_dropback", "sum"), qb_epa=("qb_epa", "mean")).reset_index()
    qb_agg = qb_agg.sort_values("n", ascending=False).drop_duplicates(key)
    qb_agg = qb_agg.set_index(key).rename(columns={
        "passer_player_id": "starter_id", "passer_player_name": "starter_name",
        "n": "starter_dropbacks", "qb_epa": "starter_epa"})

    # The STARTER known before kickoff is the passer on the team's first
    # dropback. (``starter_id`` — most dropbacks — would leak an in-game injury.)
    first = (qb.sort_values(["game_id", "play_id"]).drop_duplicates(key)
             .set_index(key)[["passer_player_id", "passer_player_name"]]
             .rename(columns={"passer_player_id": "first_qb_id", "passer_player_name": "first_qb_name"}))

    tg = counts.join([eff, pass_eff, rush_eff, drives, qb_agg, first], how="left").reset_index()
    tg = tg.rename(columns={"posteam": "team"})
    tg["sec_per_play"] = tg["top_s"] / tg["plays"]

    g = games[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
               "home_score", "away_score", "location"]].copy()
    tg = tg.merge(g, on="game_id", how="inner")
    is_home = tg["team"] == tg["home_team"]
    tg["opp"] = np.where(is_home, tg["away_team"], tg["home_team"])
    tg["neutral"] = (tg["location"] == "Neutral").astype(int)
    tg["home"] = (is_home & (tg["neutral"] == 0)).astype(int)
    tg["points_for"] = np.where(is_home, tg["home_score"], tg["away_score"])
    tg["points_against"] = np.where(is_home, tg["away_score"], tg["home_score"])
    tg["season_type"] = np.where(tg["game_type"] == "REG", "REG", "POST")
    cols = ["game_id", "season", "week", "season_type", "gameday", "team", "opp", "home", "neutral",
            "points_for", "points_against", "plays", "dropbacks", "pass_rate", "epa_pp", "sr",
            "pass_epa", "rush_epa", "proe", "comp_plays", "drives", "top_s", "sec_per_play",
            "turnovers", "rz_drives", "rz_td", "starter_id", "starter_name", "starter_dropbacks",
            "starter_epa", "first_qb_id", "first_qb_name"]
    return tg[cols].sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)


def team_games_file(d: Optional[Path] = None) -> Path:
    return data.data_dir(d) / f"team_games_v{SCHEMA_VERSION}.parquet"


def load_team_games(seasons: Iterable[int], d: Optional[Path] = None,
                    rebuild: bool = False) -> pd.DataFrame:
    """Cached team-game table for ``seasons`` (rebuilt when a source file is newer)."""
    seasons = sorted(set(seasons))
    out = team_games_file(d)
    sources = [data.pbp_file(s, d) for s in seasons] + [data.games_file(d)]
    sources = [p for p in sources if p.exists()]
    newest = max((p.stat().st_mtime for p in sources), default=0)
    if not rebuild and out.exists() and out.stat().st_mtime >= newest:
        tg = pd.read_parquet(out)
        if set(seasons) <= set(tg["season"].unique()):
            return tg[tg["season"].isin(seasons)].reset_index(drop=True)
    tg = build_team_games(data.load_pbp(seasons, d), data.load_games(d))
    out.parent.mkdir(parents=True, exist_ok=True)
    tg.to_parquet(out, index=False)
    return tg
