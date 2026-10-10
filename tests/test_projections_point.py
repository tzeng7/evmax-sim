"""evmax.projections.point: the Pinnacle slate builder and the point-projection engine."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evmax.models.odds import SharpBook, SharpOdds
from evmax.models_ml.point_projection import GameProjection
from evmax.projections import point
from evmax.projections.base import ProjectionError

KICK = datetime(2026, 10, 21, 23, 30, tzinfo=timezone.utc)          # 19:30 ET, 10-21
LATE = datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)            # 22:00 ET on 10-21, next UTC day
BASE = "nba::2026-10-21::nets_vs_hornets"


def _ml(base=BASE, home="Brooklyn Nets", away="Charlotte Hornets", when=KICK):
    return SharpOdds(event_id=base, book=SharpBook.pinnacle, sector="nba", outcome_a_label=home,
                     outcome_b_label=away, outcome_a_decimal=2.2, outcome_b_decimal=1.7, event_date=when)


def _spread(fav, line, base=BASE, alt=False, dog="Brooklyn Nets"):
    eid = f"{base}::spread::{line}" if alt else f"{base}::spread"
    return SharpOdds(event_id=eid, book=SharpBook.pinnacle, sector="nba", outcome_a_label=fav,
                     outcome_b_label=dog, outcome_a_decimal=1.9, outcome_b_decimal=1.9, spread_line=line,
                     is_alternate=alt, event_date=KICK)


def _total(line, p_over, base=BASE):
    return SharpOdds(event_id=f"{base}::total::{line}", book=SharpBook.pinnacle, sector="nba",
                     outcome_a_label="over", outcome_b_label="under", outcome_a_decimal=1.9,
                     outcome_b_decimal=1.9, total_line=line, true_prob_over=p_over, event_date=KICK)


# ── build_slate ──────────────────────────────────────────────────────────────

def test_away_favorite_spread_becomes_a_positive_home_handicap():
    # Pinnacle spread records are favorite-oriented: Charlotte (away) -3 => Brooklyn (home) +3.
    [g] = point.build_slate([_ml(), _spread("Charlotte Hornets", -3.0)])
    assert (g.home, g.away, g.book_spread) == ("Brooklyn Nets", "Charlotte Hornets", 3.0)


def test_home_favorite_spread_keeps_its_sign():
    [g] = point.build_slate([_ml(), _spread("Brooklyn Nets", -5.5, dog="Charlotte Hornets")])
    assert g.book_spread == -5.5


def test_alternate_spreads_are_ignored_and_order_does_not_matter():
    odds = [_spread("Charlotte Hornets", -8.5, alt=True), _spread("Charlotte Hornets", -3.0), _ml(),
            _spread("Charlotte Hornets", -1.5, alt=True)]
    [g] = point.build_slate(odds)
    assert g.home == "Brooklyn Nets" and g.book_spread == 3.0


def test_main_total_is_the_most_balanced_rung():
    # Pinnacle sends the whole ladder with no main-line flag; the last rung is NOT the main line.
    odds = [_ml(), _total(220.5, 0.51), _total(222.0, 0.497), _total(224.5, 0.42), _total(218.0, 0.60)]
    [g] = point.build_slate(odds)
    assert g.book_total == 222.0


def test_events_without_a_moneyline_are_dropped_and_props_ignored():
    odds = [_spread("Charlotte Hornets", -3.0),                       # no ML: home unknown
            SharpOdds(event_id=f"{BASE}::prop::x::points", book=SharpBook.pinnacle, sector="nba",
                      outcome_a_decimal=1.9, outcome_b_decimal=1.9)]
    assert point.build_slate(odds) == []


def test_game_date_is_the_eastern_day_and_slate_is_sorted_by_kickoff():
    other = "nba::2026-10-22::lakers_vs_warriors"
    games = point.build_slate([_ml(other, "Los Angeles Lakers", "Golden State Warriors", when=LATE), _ml()])
    assert [g.event_id for g in games] == [BASE, other]
    assert games[1].game_date == "2026-10-21"                          # 02:00 UTC = 22:00 ET the day before
    assert games[0].book_spread is None and games[0].book_total is None


def test_spread_for_neither_team_is_ignored():
    [g] = point.build_slate([_ml(), _spread("Someone Else", -2.0)])
    assert g.book_spread is None


# ── engine ───────────────────────────────────────────────────────────────────

def _proj(home="Brooklyn Nets", away="Charlotte Hornets", **kw):
    base = dict(home_team=home, away_team=away, home_points=108.0, away_points=112.0, projected_spread=4.0,
                projected_total=220.0, margin_sigma=12.1, total_sigma=17.3, win_prob_home=0.37, home_elo=1480.0,
                away_elo=1540.0, confidence="high", engine="possession_sim", sector="nba")
    base.update(kw)
    return GameProjection(**base)


class FakeModel:
    """Projects only the games listed in ``known``; records the calls."""

    calls: list = []
    known: dict = {}

    def __init__(self):
        self._efficiency_state = {"nba": {"fetched_at": "2026-05-30", "teams": {"a": {"gp": 8}, "b": {"gp": 30}}}}
        self._poisson_state = {"ncaab": {"teams": {"x": {}, "y": {}}}}

    def project_from_sharp(self, **kw):
        FakeModel.calls.append(kw)
        proj = FakeModel.known.get(kw["home_team"])
        return {"projection": proj} if proj else None


@pytest.fixture
def fake_board(monkeypatch):
    import evmax.models_ml.point_projection as pp

    FakeModel.calls, FakeModel.known = [], {"Brooklyn Nets": _proj()}
    monkeypatch.setattr(pp, "PointProjectionModel", FakeModel)
    board = {"odds": [_ml(), _spread("Charlotte Hornets", -3.0), _total(222.0, 0.5),
                      _ml("nba::2026-10-21::x_vs_y", "Utah Jazz", "Memphis Grizzlies")],
             "error": None, "reports": {"Brooklyn Nets": object()}}

    async def fetch(sector, injuries):
        board["injuries_requested"] = injuries
        return board["odds"], board["error"], board["reports"] if injuries else {}

    monkeypatch.setattr(point, "_fetch_board", fetch)
    return board


def test_run_slate_projects_the_board(fake_board):
    eng = point.PointProjectionEngine()
    out = eng.run_slate("nba", {"injuries": True})
    assert out["source"] == "run" and len(out["games"]) == 1
    g = out["games"][0]
    assert g["model_line"] == "Charlotte Hornets -4.0" and g["market_line"] == "Charlotte Hornets -3.0"
    assert g["market_home_margin"] == -3.0 and g["market_total"] == 222.0 and g["p_home_win"] == 0.37
    assert g["context"] == {"book_spread": 3.0, "book_total": 222.0, "game_date": "2026-10-21"}
    assert g["subtitle"] == "Elo 1540 / 1480 · high confidence"
    assert FakeModel.calls[0]["book_spread"] == 3.0 and FakeModel.calls[0]["injury_reports"]
    [missing] = out["notes"]
    assert "No usable ratings for 1 of 2 game(s): Memphis Grizzlies @ Utah Jazz" in missing
    assert "fetched 2026-05-30" in missing and "1 with fewer than the 20 games" in missing


def test_run_slate_without_injuries_option(fake_board):
    point.PointProjectionEngine().run_slate("ncaab", {})
    assert fake_board["injuries_requested"] is False
    assert FakeModel.calls[0]["injury_reports"] == {}


def test_run_slate_reports_a_pinnacle_outage(fake_board):
    fake_board["odds"], fake_board["error"] = [], {"status": 403, "reason": "geo_block"}
    with pytest.raises(ProjectionError, match="geo_block"):
        point.PointProjectionEngine().run_slate("nba", {"injuries": False})


def test_run_slate_empty_board_is_not_an_error(fake_board):
    fake_board["odds"] = []
    out = point.PointProjectionEngine().run_slate("ncaab", {})
    assert out["games"] == [] and out["notes"] == ["No NCAAB games on Pinnacle's board right now."]


def test_run_game_sections(fake_board, monkeypatch):
    FakeModel.known = {"Brooklyn Nets": _proj(home_ortg_adj=-2.0, home_injury_notes="Star Guy (S/out)",
                                              is_playoff=True)}

    async def reports(sector):
        return {"x": 1}

    monkeypatch.setattr(point, "_injury_reports", reports)
    game = {"game_id": BASE, "home": "Brooklyn Nets", "away": "Charlotte Hornets",
            "context": {"book_spread": 3.0, "book_total": 222.0, "game_date": "2026-10-21"}}
    out = point.PointProjectionEngine().run_game("nba", game, {"injuries": True})
    kv, inj = out["sections"]
    items = {i["label"]: i["value"] for i in kv["items"]}
    assert items["Spread"].startswith("Charlotte Hornets -4.0 (σ 12.1) · market Charlotte Hornets -3.0")
    assert items["Model − market"] == "home margin -1.0 · total -2.0"
    assert "playoff tightening" in items["Engine"]
    assert inj["rows"] == [{"team": "Brooklyn Nets", "ortg": "-2.0", "notes": "Star Guy (S/out)"}]
    assert FakeModel.calls[-1]["injury_reports"] == {"x": 1}
    with pytest.raises(ProjectionError, match="No model state"):
        point.PointProjectionEngine().run_game("nba", {**game, "home": "Utah Jazz"}, {"injuries": False})


def test_injury_option_only_where_the_model_uses_injuries():
    eng = point.PointProjectionEngine()
    assert [o.key for o in eng.slate_options("nba")] == ["injuries"]
    assert eng.slate_options("ncaab") == [] and eng.game_options("ncaaw") == []


# ── the CLI shares the builder ───────────────────────────────────────────────

def test_cli_slate_passes_the_home_handicap(monkeypatch):
    """Regression: `evmax project slate` read a favorite-oriented spread as the home handicap."""
    from typer.testing import CliRunner

    import evmax.clients.esports_pinnacle as pin
    from evmax.cli.commands import project as cli

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get_odds(self, sector):
            return [_ml(), _spread("Charlotte Hornets", -3.0), _total(222.0, 0.5), _total(224.5, 0.42)]

    seen = []

    class Model:
        def project_from_sharp(self, **kw):
            seen.append(kw)
            return None

    monkeypatch.setattr(pin, "PinnacleGuestClient", FakeClient)
    monkeypatch.setattr(cli, "PointProjectionModel", Model)
    out = CliRunner().invoke(cli.app, ["slate", "--sector", "nba", "--no-injuries"])
    assert out.exit_code == 0
    assert seen == [{"home_team": "Brooklyn Nets", "away_team": "Charlotte Hornets", "sector": "nba",
                     "book_spread": 3.0, "book_total": 222.0, "game_date": "2026-10-21", "injury_reports": {}}]


@pytest.mark.parametrize("play_is_home, margin, handicap, hit", [
    (True, 5, -3.5, 1),     # home -3.5 wins by 5: covers
    (True, 3, -3.5, 0),     # home -3.5 wins by 3: does not
    (False, 3, -3.5, 1),    # away +3.5 loses by 3: covers (graded a loss before 2026-10-10)
    (False, -1, -3.5, 1),   # away +3.5 wins outright
    (True, -1, 3.0, 1),     # home +3 loses by 1: covers (graded a loss before)
    (False, -4, 3.0, 1),    # away -3 wins by 4: covers
    (False, -2, 3.0, 0),    # away -3 wins by 2: does not
    (True, 3, -3.0, 0),     # push grades 0, as before
])
def test_project_resolve_spread_cover_rule(play_is_home, margin, handicap, hit):
    from evmax.cli.commands.project import _spread_hit

    assert _spread_hit(play_is_home=play_is_home, home_margin=margin, home_handicap=handicap) == hit
