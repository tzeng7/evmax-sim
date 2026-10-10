"""NFL engine for the Projections tab: ``evmax.nfl_projections``.

* Slate run: ``live.project_week`` (+ ``project_week_players``), or
  ``pipeline.run_week`` when ``store`` is on (``evmax project nfl-run
  --no-resolve``: it stores games and players and grades nothing).
* ESPN injuries apply only to the next unplayed week or later: the live feed
  lists today's injuries, which would wrongly prune a played week's roster.
* Stored view: the rows the scheduled ``nfl-projections-*`` tasks write,
  with results and tracked accuracy once graded.
* Game run: the joint box-score simulation (``live.simulate_game``) for one
  game, plus the QB + receiver stacks.

A week's player projection takes ~10 s and a one-game simulation ~0.2 s on
top of it, so the engine keeps each week's last run (``CACHE_TTL_S``) and the
per-game simulation reuses it: the simulated game is the one the page shows.
Runs hold ``_lock`` because the nflverse cache files are not safe for
concurrent writers.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Iterator, Optional

import pandas as pd

from evmax.projections.base import (
    Column, OptionSpec, ProjectionEngine, ProjectionError, game_row, jsonable, range_cell, slate_result,
)

PLAYER_COLUMNS = [
    Column("receptions", "Rec"),
    Column("receiving_yards", "Rec yds"),
    Column("rushing_yards", "Rush yds"),
    Column("passing_yards", "Pass yds"),
    Column("anytime_td", "Anytime TD", "prob",
           "Probability of a rushing or receiving touchdown (Poisson on expected TDs)"),
]
RANGE_STATS = ("receptions", "receiving_yards", "rushing_yards", "passing_yards")

FOOTNOTE = (
    "Model only (no market inputs); the market line is the nflverse consensus, shown for comparison. "
    "Walk-forward 2020–25: margin MAE 10.13 (Vegas close 9.76), total MAE 10.49 (10.28). Player medians "
    "beat a last-8-games average by 6–11% on the 2025 holdout but trail the Kalshi market by 3–6%. "
    "Teammates of players ruled out absorb 60% of their targets and carries. Ranges are the 10th–90th "
    "percentiles: about 10% of results fall below the low end and 10% above the high end."
)
RUN_NOTE = ("Outdoor wind uses the league median until a forecast feed is wired; starters come from the "
            "nflverse schedule (fallback: last game's starter).")
SIM_NOTE = ("Receivers' yards sum to the QB's passing yards in every simulation, so teammates' lines are "
            "correlated the way games are (2025 holdout: QB–WR1 simulated +0.55 vs realized +0.46).")

CACHE_TTL_S = 3600.0
CACHE_WEEKS = 4


@dataclass
class _WeekRun:
    at: float
    games: Optional[pd.DataFrame]
    players: Optional[pd.DataFrame]
    espn: bool


def full_name(abbr: str) -> str:
    """'KC' -> 'Kansas City Chiefs' (the CLI's display rule)."""
    from evmax.agents.models.nfl_efficiency_agent import NFL_ABBREV_TO_NAME

    return NFL_ABBREV_TO_NAME.get(abbr, abbr).title().replace("49Ers", "49ers")


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    from evmax.nfl_projections import store

    conn = store.connect()
    try:
        yield conn
    finally:
        conn.close()


def _as_int(v, name: str) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ProjectionError(f"{name} must be a whole number") from None


class NflProjectionEngine(ProjectionEngine):
    name = "nfl_projections"
    supports_stored = True
    supports_game_run = True
    game_run_label = "Simulate"

    def __init__(self, clock: Callable[[], float] = time.time, today: Callable[[], date] = date.today) -> None:
        self._lock = threading.Lock()
        self._runs: dict[tuple[int, int], _WeekRun] = {}
        self._clock = clock
        self._today = today

    # ── options ──────────────────────────────────────────────────────────────

    def slate_options(self, sector: str) -> list[OptionSpec]:
        return [
            OptionSpec("season", "Season", "int", None, "Blank = the season of the next unplayed week",
                       min=2000, max=2100, nullable=True, placeholder="auto"),
            OptionSpec("week", "Week", "int", None, "Blank = the next week with an unplayed game",
                       min=1, max=22, nullable=True, placeholder="auto"),
            OptionSpec("players", "Players", "bool", True, "Also project player stat lines (about 10 s more)"),
            OptionSpec("espn", "ESPN injuries", "bool", True,
                       "Drop players the live ESPN feed lists as out (the nflverse report always applies)"),
            OptionSpec("refresh", "Refresh data", "bool", True,
                       "Re-download the current season's nflverse data when it is stale"),
            OptionSpec("store", "Store", "bool", False,
                       "Save games and players to projections.db like `evmax project nfl-run --no-resolve` "
                       "(always includes players; rows freeze at kickoff)"),
        ]

    def game_options(self, sector: str) -> list[OptionSpec]:
        return [OptionSpec("sims", "Simulations", "int", 10000, "Simulated games",
                           min=1000, max=50000, step=1000)]

    # ── slate ────────────────────────────────────────────────────────────────

    @staticmethod
    def _schedule(refresh: bool) -> pd.DataFrame:
        from evmax.nfl_projections import data

        if refresh:
            data.ensure_games()
        try:
            return data.load_games()
        except FileNotFoundError:
            raise ProjectionError("No nflverse schedule cached yet; run with Refresh data on") from None

    def _espn_applies(self, schedule: pd.DataFrame, season: int, week: int) -> bool:
        """The live ESPN feed lists TODAY's injuries, so it only fits the next unplayed week or later."""
        from evmax.nfl_projections import live

        try:
            return (season, week) >= live.next_week(schedule, self._today())
        except ValueError:          # no unplayed games left: every week is history
            return False

    def _resolve_week(self, schedule: pd.DataFrame, season: Optional[int], week: Optional[int]) -> tuple[int, int]:
        from evmax.nfl_projections import live

        if season is None or week is None:
            try:
                s, w = live.next_week(schedule, self._today())
            except ValueError as e:
                raise ProjectionError(f"Pick a season and week: {e}") from None
            season, week = season or s, week or w
        if schedule[(schedule["season"] == season) & (schedule["week"] == week)].empty:
            raise ProjectionError(f"No NFL games in the schedule for {season} week {week}")
        return season, week

    def _remember(self, season: int, week: int, run: _WeekRun) -> None:
        self._runs[(season, week)] = run
        while len(self._runs) > CACHE_WEEKS:
            del self._runs[min(self._runs, key=lambda k: self._runs[k].at)]

    def run_slate(self, sector: str, options: dict) -> dict:
        from evmax.nfl_projections import live, pipeline

        refresh, espn = options["refresh"], options["espn"]
        notes: list[str] = []
        with self._lock:
            schedule = self._schedule(refresh)
            season, week = self._resolve_week(schedule, options["season"], options["week"])
            if espn and not self._espn_applies(schedule, season, week):
                espn = False
                notes.append("ESPN injuries skipped: the live feed lists today's injuries, not this played week's.")
            if options["store"]:
                with _db() as conn:
                    run = pipeline.run_week(conn, season, week, refresh=refresh, espn=espn)
                games, players = run.games, run.players
                notes += run.notes
                notes.append(f"Stored {run.games_logged} games and {run.players_logged} player rows "
                             "(games that already kicked off stay frozen).")
            else:
                games = live.project_week(season, week, refresh=refresh)
                players = None
                if options["players"]:
                    reports = live.fetch_espn_injury_reports() if espn else None
                    if espn and not reports:
                        notes.append("ESPN injury feed unavailable; nflverse injury report only")
                    players = live.project_week_players(season, week, refresh=refresh, espn_reports=reports,
                                                        game_proj=games)
            self._remember(season, week, _WeekRun(self._clock(), games, players, espn))

        games = games.sort_values(["gameday", "gametime"])
        game_rows = [self._game_row(g, store_row=False) for g in games.to_dict("records")]
        player_rows = None
        if players is not None:
            player_rows = [r for r in (self._player_row(p) for p in players.to_dict("records")) if r]
        return slate_result(
            title=f"NFL {season} · Week {week}", source="run", games=game_rows, players=player_rows,
            player_columns=PLAYER_COLUMNS, player_sort="receiving_yards", period=f"{season}-{week}",
            summary=self._accuracy_lines(season), notes=notes + [RUN_NOTE], footnote=FOOTNOTE,
        )

    def stored(self, sector: str, params: dict) -> dict:
        from evmax.nfl_projections import store

        season, week = _as_int(params.get("season"), "season"), _as_int(params.get("week"), "week")
        with _db() as conn:
            weeks = store.stored_weeks(conn)
            if not weeks:
                return slate_result(
                    title="NFL", source="stored", games=[], player_columns=PLAYER_COLUMNS,
                    notes=["No stored NFL projections yet. Run the model with Store on, or wait for the "
                           "scheduled nfl-projections task."], footnote=FOOTNOTE)
            if season is None or week is None:
                season, week = weeks[0]
            games, players = store.week_rows(conn, season, week)
        periods = [{"key": f"{s}-{w}", "label": f"{s} · Week {w}", "params": {"season": s, "week": w}}
                   for s, w in weeks]
        return slate_result(
            title=f"NFL {season} · Week {week}", source="stored",
            games=[self._game_row(g, store_row=True) for g in games],
            players=[r for r in (self._player_row(p) for p in players) if r],
            player_columns=PLAYER_COLUMNS, player_sort="receiving_yards",
            periods=periods, period=f"{season}-{week}", summary=self._accuracy_lines(season),
            notes=[] if games else [f"No stored projections for {season} week {week}."],
            footnote=FOOTNOTE,
        )

    @staticmethod
    def _game_row(g: dict, *, store_row: bool) -> dict:
        from evmax.nfl_projections import store

        if store_row:
            kickoff, market_margin, market_total = g.get("kickoff_utc"), g.get("market_home_margin"), g.get("market_total")
        else:
            kickoff = store.kickoff_utc(g["gameday"], g.get("gametime"))
            market_margin, market_total = g.get("market_spread_line"), g.get("market_total_line")
        gameday = g["gameday"]
        return game_row(
            game_id=g["game_id"], home=g["home_team"], away=g["away_team"],
            home_name=full_name(g["home_team"]), away_name=full_name(g["away_team"]),
            proj_home=g["proj_home"], proj_away=g["proj_away"], p_home_win=g["p_home_win"],
            kickoff=kickoff, game_date=gameday.isoformat() if isinstance(gameday, (date, datetime)) else str(gameday),
            neutral=bool(g.get("neutral")), market_home_margin=market_margin, market_total=market_total,
            actual_home=g.get("actual_home"), actual_away=g.get("actual_away"),
            # A missing starter name is None from SQLite but NaN from pandas.
            subtitle=f"{jsonable(g.get('away_qb_name')) or '?'} / {jsonable(g.get('home_qb_name')) or '?'}",
            context={"season": int(g["season"]), "week": int(g["week"])},
        )

    @staticmethod
    def _player_row(p: dict) -> Optional[dict]:
        targets = jsonable(p.get("proj_targets")) or 0.0
        carries = jsonable(p.get("proj_carries")) or 0.0
        qb = bool(p.get("is_starting_qb"))
        if targets < 1 and carries < 1 and not qb:
            return None
        shown = {"receptions": targets >= 1, "receiving_yards": targets >= 1,
                 "rushing_yards": carries >= 1, "passing_yards": qb}
        cells: dict = {
            st: range_cell(p.get(f"proj_{st}"), p.get(f"p10_{st}"), p.get(f"p90_{st}"), p.get(f"actual_{st}"))
            if shown[st] else None
            for st in RANGE_STATS
        }
        td = jsonable(p.get("p_anytime_td"))
        if td is not None:
            cell: dict = {"value": td}
            pass_tds = jsonable(p.get("proj_passing_tds"))
            if qb and pass_tds is not None:
                cell["sub"] = f"{pass_tds:.1f} pass TD"
            actual_tds = jsonable(p.get("actual_tds"))
            if actual_tds is not None:
                cell["result"] = f"scored {actual_tds:.0f}" if actual_tds > 0 else "no TD"
                if actual_tds > 0:
                    cell["hit"] = True
            cells["anytime_td"] = cell
        else:
            cells["anytime_td"] = None
        did_not_play = p.get("played") == 0
        team, position = p["team"], p.get("position") or "?"
        return {
            "game_id": p["game_id"], "player_id": p["player_id"],
            "name": jsonable(p.get("player_name")) or jsonable(p.get("player_display_name")) or p["player_id"],
            "detail": f"{position}, {team}", "team": team, "event": f"{team} vs {p['opp']}",
            "note": "did not play" if did_not_play else None, "dimmed": did_not_play, "cells": cells,
        }

    @staticmethod
    def _accuracy_lines(season: int) -> list[str]:
        """The season's tracked accuracy (graded stored rows), as display lines."""
        from evmax.nfl_projections import store

        try:
            with _db() as conn:
                acc = store.accuracy(conn, season=season)
        except sqlite3.Error:
            return []
        lines: list[str] = []
        g = acc["games"]
        if g["n"]:
            def close(v):
                return f"{v:.2f}" if v is not None else "—"
            lines.append(f"{season} tracked over {g['n']} games: margin MAE {g['margin_mae']:.2f} (closing line "
                         f"{close(g['close_margin_mae'])}), total MAE {g['total_mae']:.2f} (closing line "
                         f"{close(g['close_total_mae'])})")
        labels = {"receptions": "Rec", "receiving_yards": "Rec yds", "rushing_yards": "Rush yds",
                  "passing_yards": "Pass yds"}
        parts = [f"{labels[k]} {m['mae']:.1f} (n {m['n']})" for k, m in acc["players"].items() if k in labels]
        if parts:
            lines.append("Player MAE: " + " · ".join(parts))
        td = acc["players"].get("anytime_td")
        if td:
            lines.append(f"Anytime TD: predicted {td['mean_p'] * 100:.0f}% vs actual {td['rate'] * 100:.0f}% "
                         f"(Brier {td['brier']:.3f}, n {td['n']})")
        return lines

    # ── one game ─────────────────────────────────────────────────────────────

    @staticmethod
    def _game_week(game: dict) -> tuple[int, int]:
        ctx = game.get("context") or {}
        season, week = _as_int(ctx.get("season"), "season"), _as_int(ctx.get("week"), "week")
        if season is None or week is None:
            # nflverse game ids are SEASON_WEEK_AWAY_HOME.
            parts = str(game.get("game_id", "")).split("_")
            try:
                season, week = int(parts[0]), int(parts[1])
            except (IndexError, ValueError):
                raise ProjectionError("game has no season/week") from None
        return season, week

    def run_game(self, sector: str, game: dict, options: dict) -> dict:
        from evmax.nfl_projections import live, simulate

        season, week = self._game_week(game)
        home, away = game.get("home"), game.get("away")
        if not home or not away:
            raise ProjectionError("game has no teams")
        notes: list[str] = []
        with self._lock:
            run = self._runs.get((season, week))
            if run is not None and run.players is not None and self._clock() - run.at <= CACHE_TTL_S:
                proj = run.players
                stamp = datetime.fromtimestamp(run.at).strftime("%H:%M")
                notes.append(f"Simulated from this page's {stamp} run "
                             f"(ESPN injuries {'on' if run.espn else 'off'}).")
            else:
                espn = self._espn_applies(self._schedule(refresh=False), season, week)
                reports = live.fetch_espn_injury_reports() if espn else None
                proj = live.project_week_players(season, week, refresh=False, espn_reports=reports)
                self._remember(season, week, _WeekRun(self._clock(), run.games if run else None, proj, espn))
                notes.append("Simulated from a fresh player projection (cached nflverse data"
                             + (", live ESPN injuries" if espn else "; ESPN injuries skipped for a played week")
                             + "); it can differ slightly from stored rows.")
            if proj[proj["team"] == home].empty:
                raise ProjectionError(f"{home} has no projected game in {season} week {week}")
            res = live.simulate_game(season, week, home, n=options["sims"], refresh=False, proj=proj)

        sections: list[dict] = []
        stacks: list[dict] = []
        for team in (away, home):
            if team not in res:
                continue
            players, sims = res[team]
            sections.append({"kind": "players", "title": f"{full_name(team)} — simulated box score",
                             "columns": [c.to_dict() for c in PLAYER_COLUMNS],
                             "rows": self._sim_rows(team, players, simulate.summarize(players, sims))})
            stacks += simulate.qb_stacks(players, sims)
        if stacks:
            sections.append({
                "kind": "table", "title": "Stacks — both legs clear their simulated medians",
                "columns": [{"key": "outcome", "label": "Outcome"},
                            {"key": "joint", "label": "Joint", "align": "right"},
                            {"key": "independent", "label": "If independent", "align": "right"},
                            {"key": "lift", "label": "Lift", "align": "right"}],
                "rows": [{
                    "outcome": f"{k['qb']} {k['qb_passing_yards']:.0f}+ pass yds & "
                               f"{k['receiver']} {k['receiver_receiving_yards']:.0f}+ rec yds",
                    "joint": f"{k['joint'] * 100:.1f}%",
                    "independent": f"{k['independent'] * 100:.1f}%",
                    "lift": f"×{k['joint'] / k['independent']:.2f}" if k["independent"] > 0 else "—",
                } for k in stacks],
            })
        return {"title": f"{away} @ {home} — {options['sims']:,} simulated games",
                "sections": sections, "notes": notes + [SIM_NOTE]}

    @staticmethod
    def _sim_rows(team: str, players: pd.DataFrame, summ: pd.DataFrame) -> list[dict]:
        show = (players["proj_targets"] >= 2.5) | (players["proj_carries"] >= 5) | players["is_starting_qb"].astype(bool)
        order = players[show].sort_values(["is_starting_qb", "proj_targets"], ascending=[False, False]).index
        rows = []
        for i in order:
            p, m = players.loc[i], summ.loc[i]
            qb = bool(p["is_starting_qb"])
            shown = {"receptions": p["proj_targets"] >= 1, "receiving_yards": p["proj_targets"] >= 1,
                     "rushing_yards": p["proj_carries"] >= 1, "passing_yards": qb}
            cells: dict = {st: range_cell(m[f"sim_{st}"], m[f"sim_p10_{st}"], m[f"sim_p90_{st}"]) if shown[st] else None
                           for st in RANGE_STATS}
            cells["anytime_td"] = {"value": jsonable(m["sim_p_anytime_td"])}
            rows.append({"game_id": p["game_id"], "player_id": p["player_id"],
                         "name": jsonable(p["player_display_name"]) or p["player_id"],
                         "detail": f"{p['position']}, {team}", "team": team, "event": f"{team} vs {p['opp']}",
                         "note": None, "dimmed": False, "cells": cells})
        return rows
