"""evmax.nfl_projections.store: log / freeze-at-kickoff / grade / track NFL projections."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from evmax.nfl_projections import store

UTC = ZoneInfo("UTC")


def _games_proj(proj_home=24.0):
    return pd.DataFrame([
        {"game_id": "2026_06_DAL_PHI", "season": 2026, "week": 6, "gameday": date(2026, 10, 18), "gametime": "13:00",
         "home_team": "PHI", "away_team": "DAL", "neutral": False, "roof": "outdoors", "proj_home": proj_home,
         "proj_away": 20.0, "proj_margin": proj_home - 20.0, "proj_total": proj_home + 20.0, "p_home_win": 0.6,
         "home_qb_id": "q1", "away_qb_id": "q2", "home_qb_name": "A", "away_qb_name": "B",
         "market_spread_line": 3.0, "market_total_line": 45.5},
        {"game_id": "2026_06_KC_LV", "season": 2026, "week": 6, "gameday": date(2026, 10, 15), "gametime": "20:15",
         "home_team": "LV", "away_team": "KC", "neutral": False, "roof": "dome", "proj_home": 17.0,
         "proj_away": 27.0, "proj_margin": -10.0, "proj_total": 44.0, "p_home_win": 0.23,
         "home_qb_id": "q3", "away_qb_id": "q4", "home_qb_name": "C", "away_qb_name": "D",
         "market_spread_line": -7.5, "market_total_line": 46.0},
    ])


def _players_proj(ids=("p1", "p2"), rec_yds=50.0):
    rows = []
    for pid in ids:
        rows.append({"game_id": "2026_06_DAL_PHI", "player_id": pid, "season": 2026, "week": 6,
                     "gameday": pd.Timestamp("2026-10-18"), "gametime": "13:00", "team": "PHI", "opp": "DAL",
                     "player_display_name": pid.upper(), "position": "WR", "is_starting_qb": False,
                     "proj_targets": 6.0, "proj_carries": 0.0,
                     "proj_receptions": 4.0, "mean_receptions": 4.2, "p10_receptions": 2.0, "p90_receptions": 7.0,
                     "proj_receiving_yards": rec_yds, "mean_receiving_yards": rec_yds + 5,
                     "p10_receiving_yards": 15.0, "p90_receiving_yards": 100.0,
                     "proj_rushing_yards": 0.0, "mean_rushing_yards": 0.0, "p10_rushing_yards": 0.0,
                     "p90_rushing_yards": 0.0, "proj_passing_yards": 0.0, "p10_passing_yards": 0.0,
                     "p90_passing_yards": 0.0})
    return pd.DataFrame(rows)


def test_kickoff_utc_converts_eastern_time():
    assert store.kickoff_utc(date(2026, 10, 18), "13:00") == "2026-10-18T17:00+00:00"    # EDT
    assert store.kickoff_utc(date(2026, 12, 6), "20:20") == "2026-12-07T01:20+00:00"     # EST, next UTC day
    assert store.kickoff_utc(date(2026, 12, 6), None) is None


def test_log_games_upserts_until_kickoff_then_freezes(tmp_path):
    conn = store.connect(tmp_path / "p.db")
    fri = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)        # after Thursday's LV game, before Sunday's
    assert store.log_games(conn, _games_proj(), "v1", now=datetime(2026, 10, 14, tzinfo=UTC)) == 2
    assert store.log_games(conn, _games_proj(proj_home=30.0), "v2", now=fri) == 1    # LV game already kicked off
    rows = {r["game_id"]: r for r in conn.execute("SELECT * FROM nfl_game_projections")}
    assert rows["2026_06_DAL_PHI"]["proj_home"] == 30.0 and rows["2026_06_DAL_PHI"]["model_version"] == "v2"
    assert rows["2026_06_KC_LV"]["proj_home"] == 17.0 and rows["2026_06_KC_LV"]["model_version"] == "v1"
    # line convention: nflverse spread_line = expected HOME margin (home PHI favored by 3)
    assert rows["2026_06_DAL_PHI"]["market_home_margin"] == 3.0
    assert rows["2026_06_DAL_PHI"]["logged_at"] < rows["2026_06_DAL_PHI"]["updated_at"]


def test_log_players_drops_players_ruled_out_before_kickoff(tmp_path):
    conn = store.connect(tmp_path / "p.db")
    sat = datetime(2026, 10, 17, tzinfo=UTC)
    sun = datetime(2026, 10, 18, 14, 0, tzinfo=UTC)       # before the 17:00 UTC kickoff
    store.log_players(conn, _players_proj(("p1", "p2")), "v1", now=sat)
    store.log_players(conn, _players_proj(("p1",), rec_yds=60.0), "v1", now=sun)   # p2 ruled out Sunday
    rows = {r["player_id"]: r for r in conn.execute("SELECT * FROM nfl_player_projections")}
    assert set(rows) == {"p1"} and rows["p1"]["proj_receiving_yards"] == 60.0
    after = datetime(2026, 10, 18, 18, 0, tzinfo=UTC)     # game started: frozen
    assert store.log_players(conn, _players_proj(("p1", "p3"), rec_yds=99.0), "v1", now=after) == 0
    assert {r[0] for r in conn.execute("SELECT player_id FROM nfl_player_projections")} == {"p1"}


def test_resolve_and_accuracy(tmp_path):
    conn = store.connect(tmp_path / "p.db")
    early = datetime(2026, 10, 14, tzinfo=UTC)
    store.log_games(conn, _games_proj(), "v1", now=early)
    store.log_players(conn, _players_proj(("p1", "p2")), "v1", now=early)
    games = pd.DataFrame([
        {"game_id": "2026_06_DAL_PHI", "home_score": 27.0, "away_score": 17.0, "spread_line": 2.5, "total_line": 44.0},
        {"game_id": "2026_06_KC_LV", "home_score": None, "away_score": None, "spread_line": -7.0, "total_line": 46.0},
    ])
    pg = pd.DataFrame([{"game_id": "2026_06_DAL_PHI", "player_id": "p1", "receptions": 5, "receiving_yards": 80.0,
                        "rushing_yards": 0.0, "passing_yards": 0.0}])
    n = store.resolve(conn, games, pg, now=datetime(2026, 10, 20, tzinfo=UTC))
    assert n == {"games": 1, "players": 2}
    p = {r["player_id"]: r for r in conn.execute("SELECT * FROM nfl_player_projections")}
    assert p["p1"]["played"] == 1 and p["p1"]["actual_receiving_yards"] == 80.0
    assert p["p2"]["played"] == 0 and p["p2"]["actual_receiving_yards"] is None     # no row for a final game = DNP
    assert store.resolve(conn, games, pg) == {"games": 0, "players": 0}               # idempotent

    acc = store.accuracy(conn, season=2026)
    assert acc["games"]["n"] == 1
    assert acc["games"]["margin_mae"] == pytest.approx(abs(4.0 - 10.0))
    assert acc["games"]["close_margin_mae"] == pytest.approx(abs(2.5 - 10.0))
    assert acc["games"]["total_mae"] == pytest.approx(abs(44.0 - 44.0))
    assert acc["games"]["winner_pct"] == 1.0
    ry = acc["players"]["receiving_yards"]
    assert ry["n"] == 1 and ry["mae"] == pytest.approx(30.0) and ry["bias"] == pytest.approx(-30.0)
    assert ry["below_p10"] == 0.0 and ry["at_or_below_p90"] == 1.0
    assert "rushing_yards" not in acc["players"]                          # nobody in the rushing population


def test_week_rows_and_latest_week(tmp_path):
    conn = store.connect(tmp_path / "p.db")
    assert store.latest_week(conn) is None
    store.log_games(conn, _games_proj(), "v1", now=datetime(2026, 10, 14, tzinfo=UTC))
    store.log_players(conn, _players_proj(("p1", "p2")), "v1", now=datetime(2026, 10, 14, tzinfo=UTC))
    assert store.latest_week(conn) == (2026, 6)
    games, players = store.week_rows(conn, 2026, 6, team="phi")
    assert [g["game_id"] for g in games] == ["2026_06_KC_LV", "2026_06_DAL_PHI"]   # kickoff order
    assert {p["player_id"] for p in players} == {"p1", "p2"}
    assert store.week_rows(conn, 2026, 6, team="KC")[1] == []


# ── consumers: web endpoint, Discord embeds / handler / post, CLI ────────────

@pytest.fixture
def stored_db(tmp_path, monkeypatch):
    path = tmp_path / "projections.db"
    monkeypatch.setenv("EVMAX_PROJ_DB", str(path))
    conn = store.connect()
    early = datetime(2026, 10, 14, tzinfo=UTC)
    store.log_games(conn, _games_proj(), "v1", now=early)
    store.log_players(conn, _players_proj(("p1", "p2")), "v1", now=early)
    conn.close()
    return path


def test_api_nfl_projections(stored_db):
    from fastapi.testclient import TestClient

    from evmax.web.app import app

    with TestClient(app) as client:
        j = client.get("/api/nfl-projections").json()
        assert (j["season"], j["week"]) == (2026, 6) and j["weeks"] == [{"season": 2026, "week": 6}]
        g = {r["game_id"]: r for r in j["games"]}
        assert g["2026_06_DAL_PHI"]["model_line"] == "PHI -4.0" and g["2026_06_DAL_PHI"]["market_line"] == "PHI -3.0"
        assert g["2026_06_KC_LV"]["market_line"] == "KC -7.5"          # negative home margin = away favored
        assert len(j["players"]) == 2
        assert client.get("/api/nfl-projections?season=2026&week=6&team=kc").json()["players"] == []
        assert client.get("/api/nfl-projections?season=2025&week=1").json()["games"] == []


def test_api_nfl_projections_empty_db(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from evmax.web.app import app

    monkeypatch.setenv("EVMAX_PROJ_DB", str(tmp_path / "empty.db"))
    with TestClient(app) as client:
        j = client.get("/api/nfl-projections").json()
    assert j["season"] is None and j["games"] == [] and j["weeks"] == []


def test_nfl_projection_embeds(stored_db):
    from evmax.discord_bot.embeds import nfl_projection_embeds

    conn = store.connect()
    games, players = store.week_rows(conn, 2026, 6)
    embeds = nfl_projection_embeds(2026, 6, games, players)
    text = "\n".join(e["description"] for e in embeds)
    assert "DAL@PHI" in text and "PHI -4.0 / 44.0" in text and "PHI -3.0 / 45.5" in text
    assert "P1 (PHI)" in text and "15-100" in text                    # player median + range
    assert "footer" in embeds[-1]
    assert nfl_projection_embeds(2026, 9, [], [])[0]["description"].startswith("No projections")


def test_discord_nfl_handler(stored_db):
    import asyncio

    from evmax.discord_bot.handlers import CommandHandlers

    h = CommandHandlers()
    r = asyncio.run(h.nfl())
    assert r.embeds and "Week 6" in r.embeds[0]["title"]
    r = asyncio.run(h.nfl(team="kc"))
    assert "KC@LV" in r.embeds[0]["description"] and "DAL@PHI" not in r.embeds[0]["description"]
    assert asyncio.run(h.nfl(season=2024)).ephemeral                  # nothing stored for that season


def test_post_week_uses_configured_client(stored_db, monkeypatch):
    from evmax.discord_bot import client as client_mod
    from evmax.nfl_projections import pipeline

    sent = []

    class FakeClient:
        def post_embeds(self, embeds, **kw):
            sent.append(embeds)
            return True

    monkeypatch.setattr(client_mod.DiscordBotClient, "from_settings", classmethod(lambda cls, *a, **k: FakeClient()))
    conn = store.connect()
    assert pipeline.post_week(conn, 2026, 6) is True and sent and sent[0]
    monkeypatch.setattr(client_mod.DiscordBotClient, "from_settings", classmethod(lambda cls, *a, **k: None))
    assert pipeline.post_week(conn, 2026, 6) is False


def test_cli_nfl_track(stored_db):
    from typer.testing import CliRunner

    from evmax.cli.commands.project import app

    runner = CliRunner()
    out = runner.invoke(app, ["nfl-track"], env={"COLUMNS": "200"})
    assert out.exit_code == 0 and "No graded NFL projections" in out.output
    conn = store.connect()
    games = pd.DataFrame([{"game_id": "2026_06_DAL_PHI", "home_score": 27.0, "away_score": 17.0,
                           "spread_line": 2.5, "total_line": 44.0}])
    store.resolve(conn, games, pd.DataFrame())
    out = runner.invoke(app, ["nfl-track", "--season", "2026"], env={"COLUMNS": "200"})
    assert out.exit_code == 0 and "Margin (home - away)" in out.output and "6.00" in out.output
