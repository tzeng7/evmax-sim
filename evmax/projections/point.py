"""Point-projection engine for the Projections tab (NBA, NCAAB, NCAAW).

Projects every game on the sector's Pinnacle board with
``PointProjectionModel``: the NBA possession simulation, or the Poisson + Elo
blend elsewhere. It is the model behind ``evmax project slate``, which builds
its slate with ``build_slate`` too.

``build_slate`` turns Pinnacle records into one entry per game and must
respect the ``PinnacleGuestClient`` record conventions:

* The moneyline record carries home/away: ``outcome_a_label`` is the home team.
* A spread record is FAVORITE-oriented: ``outcome_a_label`` is the team laying
  points and ``spread_line`` is its negative handicap. ``build_slate`` converts
  it to the home handicap (negative = home favored).
* Alternate spread rungs (``is_alternate``) are skipped.
* Totals arrive as the whole alternate ladder with no main-line flag. The
  main line is the most balanced rung (devigged P(over) closest to 0.5).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from evmax.projections.base import OptionSpec, ProjectionEngine, ProjectionError, game_row, slate_result

ET = ZoneInfo("America/New_York")
_SUFFIXES = (("::spread", "spread"), ("::total", "total"), ("::advance", "advance"))

NOTE_NOT_EV = ("A projection difference is not an EV edge: the scanner prices plays off Pinnacle's "
               "devigged line, and this model is not an input to it.")


@dataclass(frozen=True)
class SlateGame:
    event_id: str                 # Pinnacle base event key
    home: str
    away: str
    kickoff: Optional[datetime]   # UTC
    game_date: Optional[str]      # ET calendar day (YYYY-MM-DD)
    book_spread: Optional[float]  # home handicap, negative = home favored
    book_total: Optional[float]


def build_slate(odds: Iterable) -> list[SlateGame]:
    """One ``SlateGame`` per Pinnacle event with a moneyline, sorted by kickoff.

    Events without a moneyline record are dropped: only the moneyline says
    which team is home.
    """
    events: dict[str, dict] = {}
    for o in odds:
        eid = o.event_id
        if "::prop::" in eid:
            continue
        base, kind = eid, "moneyline"
        for suffix, k in _SUFFIXES:
            if suffix in eid:
                base, kind = eid[:eid.index(suffix)], k
                break
        ev = events.setdefault(base, {"moneyline": None, "spread": None, "totals": []})
        if kind == "moneyline":
            ev["moneyline"] = o
        elif kind == "spread" and not o.is_alternate and o.spread_line is not None:
            ev["spread"] = o
        elif kind == "total" and o.total_line is not None and o.true_prob_over is not None:
            ev["totals"].append(o)

    out: list[SlateGame] = []
    for base, ev in events.items():
        ml = ev["moneyline"]
        if ml is None:
            continue
        home, away = ml.outcome_a_label, ml.outcome_b_label
        book_spread = None
        sp = ev["spread"]
        if sp is not None:
            if sp.outcome_a_label == home:
                book_spread = float(sp.spread_line)
            elif sp.outcome_a_label == away:
                book_spread = -float(sp.spread_line)
        book_total = None
        if ev["totals"]:
            book_total = float(min(ev["totals"], key=lambda t: abs(t.true_prob_over - 0.5)).total_line)
        kickoff = ml.event_date
        if kickoff is not None and kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        out.append(SlateGame(
            event_id=base, home=home, away=away, kickoff=kickoff,
            game_date=kickoff.astimezone(ET).date().isoformat() if kickoff else None,
            book_spread=book_spread, book_total=book_total,
        ))
    out.sort(key=lambda g: (g.kickoff is None, g.kickoff.timestamp() if g.kickoff else 0.0, g.event_id))
    return out


async def _injury_reports(sector: str) -> dict:
    """ESPN injury reports via InjuryReportAgent; {} on any failure."""
    try:
        from evmax.agents.base import AgentRequest
        from evmax.agents.intelligence.injury_agent import InjuryReportAgent

        resp = await InjuryReportAgent().run(AgentRequest(sector=sector, correlation_id="projections-tab"))
        return resp.data or {}
    except Exception:  # noqa: BLE001 — injuries are optional; the projection runs without them
        return {}


async def _fetch_board(sector: str, injuries: bool) -> tuple[list, Optional[dict], dict]:
    from evmax.clients.esports_pinnacle import PinnacleGuestClient

    async with PinnacleGuestClient() as client:
        odds = await client.get_odds(sector)
        error = client.last_error
    reports = await _injury_reports(sector) if injuries else {}
    return odds, error, reports


def _state_hint(sector: str, model) -> str:
    """Why the model may lack ratings for a game (it returns None without a reason)."""
    if sector == "nba":
        nba = model._efficiency_state.get("nba", {})
        teams = nba.get("teams", {})
        thin = sum(1 for t in teams.values() if t.get("gp", 0) < 20)
        return (f"The NBA efficiency ratings (fetched {nba.get('fetched_at', 'never')}) cover {len(teams)} teams, "
                f"{thin} with fewer than the 20 games the possession sim needs.")
    n = len(model._poisson_state.get(sector, {}).get("teams", {}))
    return f"A team is missing from the {sector.upper()} Poisson/Elo ratings ({n} teams rated)."


def _injury_option() -> OptionSpec:
    return OptionSpec("injuries", "ESPN injuries", "bool", True,
                      "Lower a team's offensive rating for stars and starters listed out or day-to-day")


class PointProjectionEngine(ProjectionEngine):
    name = "point_projection"
    supports_game_run = True
    game_run_label = "Run"

    def slate_options(self, sector: str) -> list[OptionSpec]:
        from evmax.models_ml.point_projection import INJURY_SECTORS

        return [_injury_option()] if sector in INJURY_SECTORS else []

    def game_options(self, sector: str) -> list[OptionSpec]:
        return self.slate_options(sector)

    def run_slate(self, sector: str, options: dict) -> dict:
        from evmax.models_ml.point_projection import PointProjectionModel

        injuries = bool(options.get("injuries"))
        odds, error, reports = asyncio.run(_fetch_board(sector, injuries))
        if not odds and error:
            raise ProjectionError(f"Pinnacle board unavailable ({error.get('reason') or error.get('status')})")
        slate = build_slate(odds)
        model = PointProjectionModel()
        rows, missing = [], []
        for g in slate:
            res = model.project_from_sharp(
                home_team=g.home, away_team=g.away, sector=sector, book_spread=g.book_spread,
                book_total=g.book_total, game_date=g.game_date, injury_reports=reports,
            )
            if res is None:
                missing.append(f"{g.away} @ {g.home}")
                continue
            rows.append(self._row(g, res["projection"]))
        notes: list[str] = []
        if not slate:
            notes.append(f"No {sector.upper()} games on Pinnacle's board right now.")
        if missing:
            shown = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
            notes.append(f"No usable ratings for {len(missing)} of {len(slate)} game(s): {shown}. "
                         + _state_hint(sector, model))
        if injuries and not reports:
            notes.append("ESPN injury feed unavailable; projected without injury adjustments.")
        return slate_result(
            title=f"{sector.upper()} · Pinnacle board", source="run", games=rows, notes=notes,
            footnote=("Market = Pinnacle's main spread and total. " + NOTE_NOT_EV),
        )

    @staticmethod
    def _row(g: SlateGame, proj) -> dict:
        flags = []
        if proj.confidence == "low":
            flags.append("low confidence")
        if proj.is_playoff:
            flags.append("playoff")
        if proj.home_ortg_adj or proj.away_ortg_adj:
            flags.append("injuries")
        parts = []
        if proj.home_elo is not None and proj.away_elo is not None:
            parts.append(f"Elo {proj.away_elo:.0f} / {proj.home_elo:.0f}")
        parts.append(f"{proj.confidence} confidence")
        return game_row(
            game_id=g.event_id, home=g.home, away=g.away, home_name=g.home, away_name=g.away,
            proj_home=proj.home_points, proj_away=proj.away_points, p_home_win=proj.win_prob_home,
            kickoff=g.kickoff.isoformat() if g.kickoff else None, game_date=g.game_date,
            market_home_margin=-g.book_spread if g.book_spread is not None else None,
            market_total=g.book_total, subtitle=" · ".join(parts), flags=flags,
            context={"book_spread": g.book_spread, "book_total": g.book_total, "game_date": g.game_date},
        )

    def run_game(self, sector: str, game: dict, options: dict) -> dict:
        from evmax.models_ml.point_projection import PointProjectionModel

        home, away = game.get("home"), game.get("away")
        if not home or not away:
            raise ProjectionError("game has no teams")
        ctx = game.get("context") or {}
        reports = asyncio.run(_injury_reports(sector)) if options.get("injuries") else {}
        res = PointProjectionModel().project_from_sharp(
            home_team=home, away_team=away, sector=sector, book_spread=ctx.get("book_spread"),
            book_total=ctx.get("book_total"), game_date=ctx.get("game_date"), injury_reports=reports,
        )
        if res is None:
            raise ProjectionError(f"No model state for {away} @ {home}")
        proj = res["projection"]
        row = self._row(SlateGame(str(game.get("game_id", "")), home, away, None, ctx.get("game_date"),
                                  ctx.get("book_spread"), ctx.get("book_total")), proj)
        wp = proj.win_prob_home
        items = [
            {"label": "Score", "value": f"{away} {proj.away_points:.1f} – {home} {proj.home_points:.1f}"},
            {"label": "Win probability", "value": f"{home} {wp * 100:.1f}% · {away} {(1 - wp) * 100:.1f}%"},
            {"label": "Spread", "value": f"{row['model_line']} (σ {proj.margin_sigma:.1f})"
                                         f" · market {row['market_line'] or '—'}"},
            {"label": "Total", "value": f"{proj.projected_total:.1f} (σ {proj.total_sigma:.1f})"
                                        f" · market {row['market_total'] if row['market_total'] is not None else '—'}"},
        ]
        deltas = []
        if row["market_home_margin"] is not None:
            deltas.append(f"home margin {row['proj_margin'] - row['market_home_margin']:+.1f}")
        if row["market_total"] is not None:
            deltas.append(f"total {row['proj_total'] - row['market_total']:+.1f}")
        if deltas:
            items.append({"label": "Model − market", "value": " · ".join(deltas)})
        if proj.home_elo is not None and proj.away_elo is not None:
            items.append({"label": "Elo", "value": f"{away} {proj.away_elo:.0f} · {home} {proj.home_elo:.0f}"})
        items.append({"label": "Engine", "value": f"{proj.engine} · {proj.confidence} confidence"
                                                  + (" · playoff tightening" if proj.is_playoff else "")})
        sections: list[dict] = [{"kind": "kv", "title": "Projection", "items": items}]
        injured = [(t, adj, notes) for t, adj, notes in (
            (away, proj.away_ortg_adj, proj.away_injury_notes), (home, proj.home_ortg_adj, proj.home_injury_notes),
        ) if adj]
        if injured:
            sections.append({
                "kind": "table", "title": "Injury adjustments",
                "columns": [{"key": "team", "label": "Team"}, {"key": "ortg", "label": "ORTG Δ", "align": "right"},
                            {"key": "notes", "label": "Players"}],
                "rows": [{"team": t, "ortg": f"{adj:+.1f}", "notes": notes or "—"} for t, adj, notes in injured],
            })
        return {"title": f"{away} @ {home}", "sections": sections, "notes": [NOTE_NOT_EV]}
