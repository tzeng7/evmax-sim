"""One NFL projection run: project a week, store it, grade the past, post it.

``evmax project nfl-run`` (and the scheduled weekly / Sunday-morning tasks) call
``run_week``; ``evmax project nfl-resolve`` calls ``resolve_pending``. The
dashboard and the Discord bot only read what these store.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from evmax.nfl_projections import data, live, player_games, store


@dataclass
class WeekRun:
    season: int
    week: int
    games: pd.DataFrame
    players: pd.DataFrame
    games_logged: int = 0
    players_logged: int = 0
    espn_out: int = 0
    notes: list[str] = field(default_factory=list)


def run_week(conn: sqlite3.Connection, season: int, week: int, *, refresh: bool = True,
             espn: bool = True, d: Optional[Path] = None) -> WeekRun:
    """Project games and players for ``season`` ``week`` and upsert them into projections.db.

    Rows of games that already kicked off are left untouched (``store.log_*``).
    """
    from evmax.provenance import code_version

    version = code_version()
    games = live.project_week(season, week, d=d, refresh=refresh)
    reports = live.fetch_espn_injury_reports() if espn else {}
    players = live.project_week_players(season, week, d=d, refresh=refresh, espn_reports=reports)
    run = WeekRun(season, week, games, players)
    if espn and not reports:
        run.notes.append("ESPN injury feed unavailable; nflverse injury report only")
    run.games_logged = store.log_games(conn, games, version)
    run.players_logged = store.log_players(conn, players, version)
    return run


def resolve_pending(conn: sqlite3.Connection, *, refresh: bool = True, d: Optional[Path] = None) -> dict[str, int]:
    """Grade every stored row whose game is final (refreshing nflverse data first)."""
    seasons = [int(r[0]) for r in conn.execute(
        "SELECT DISTINCT season FROM nfl_game_projections WHERE resolved_at IS NULL "
        "UNION SELECT DISTINCT season FROM nfl_player_projections WHERE resolved_at IS NULL")]
    if not seasons:
        return {"games": 0, "players": 0}
    if refresh:
        data.ensure_games(d, max_age_hours=1.0)
        data.ensure_player_sources(seasons, d, refresh_seasons=seasons, max_age_hours=1.0)
    games = data.load_games(d)
    # Built directly (not via load_player_games, whose cache holds the full history the
    # projection runs need and would be overwritten with just these seasons).
    weekly, snaps = data.load_player_week(seasons, d), data.load_snaps(seasons, d)
    if weekly.empty or snaps.empty:
        return store.resolve(conn, games, pd.DataFrame(), ready_games=set())
    pg = player_games.build_player_games(weekly, snaps, games)
    # A game's player rows are graded only once BOTH stat lines and snap counts are
    # published (snaps identify players who played and recorded nothing).
    ready = set(weekly["game_id"]) & set(snaps["game_id"])
    return store.resolve(conn, games, pg, ready_games=ready)


def post_week(conn: sqlite3.Connection, season: int, week: int) -> bool:
    """Post the stored week to the configured Discord channel/DM. False when unconfigured or on failure."""
    from evmax.discord_bot.client import DiscordBotClient
    from evmax.discord_bot.embeds import nfl_projection_embeds

    client = DiscordBotClient.from_settings()
    if client is None:
        return False
    games, players = store.week_rows(conn, season, week)
    embeds = nfl_projection_embeds(season, week, games, players, store.accuracy(conn, season=season))
    return bool(client.post_embeds(embeds))
