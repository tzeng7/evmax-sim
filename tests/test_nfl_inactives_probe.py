"""evmax.nfl_projections.inactives_probe: record-only snapshots of the two live inactive sources."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from evmax.nfl_projections import inactives_probe as probe
from evmax.nfl_projections import live, store

UTC = ZoneInfo("UTC")
# 2026-10-11 1pm ET kickoff = 17:00 UTC
GAME = "2026_05_MIN_NO"


def _games():
    return pd.DataFrame([
        {"game_id": GAME, "season": 2026, "week": 5, "gameday": pd.Timestamp("2026-10-11"), "gametime": "13:00",
         "home_team": "NO", "away_team": "MIN"},
        {"game_id": "2026_05_BAL_ATL", "season": 2026, "week": 5, "gameday": pd.Timestamp("2026-10-11"),
         "gametime": "20:20", "home_team": "ATL", "away_team": "BAL"},
    ])


def _roster(week=5, ina=("Jordan Addison",)):
    rows = [{"season": 2026, "week": week, "team": "MIN", "gsis_id": f"m{i}", "full_name": n, "position": "WR",
             "status": "INA"} for i, n in enumerate(ina)]
    rows += [{"season": 2026, "week": week, "team": "MIN", "gsis_id": "m9", "full_name": "Justin Jefferson",
              "position": "WR", "status": "ACT"},
             {"season": 2026, "week": week, "team": "NO", "gsis_id": "n1", "full_name": "Chris Olave",
              "position": "WR", "status": "ACT"}]
    return pd.DataFrame(rows)


def _espn(status="Questionable"):
    return {"minnesota vikings": live.EspnTeamReport("minnesota vikings", [
        live.EspnStatus("Jordan Addison", "WR", status, "2026-10-13T17:10Z"),
        live.EspnStatus("T.J. Hockenson", "TE", "Out", None)])}


@pytest.fixture
def conn(tmp_path):
    c = probe.connect(tmp_path / "proj.db")
    yield c
    c.close()


def _now(h, m):
    return datetime(2026, 10, 11, h, m, tzinfo=UTC)


def test_games_in_window_uses_kickoff_and_skips_far_games():
    near = probe.games_in_window(_games(), _now(15, 30))        # kickoff 90 min out; 20:20 ET game is 8+ h out
    assert [g["game_id"] for g in near] == [GAME]
    assert probe.games_in_window(_games(), _now(12, 0)) == []   # 5 h before: outside the 150-minute lookahead
    assert [g["game_id"] for g in probe.games_in_window(_games(), _now(17, 20))] == [GAME]   # 20 min after kickoff
    assert probe.games_in_window(_games(), _now(18, 0)) == []   # 60 min after: past the 30-minute lookback


def test_no_game_in_window_makes_no_network_calls(conn):
    def boom(*a, **k):
        raise AssertionError("fetched with nothing to probe")

    r = probe.probe_once(conn, now=_now(9, 0), games=_games(), roster_fetch=boom, espn_fetch=boom)
    assert r == {"games": 0, "run_rows": 0, "player_rows": 0}


def test_snapshot_records_roster_ina_and_espn_statuses(conn):
    r = probe.probe_once(conn, now=_now(15, 51), games=_games(),
                         roster_fetch=lambda s: (_roster(), "Sun, 11 Oct 2026 15:40:00 GMT"),
                         espn_fetch=lambda: _espn())
    assert r["games"] == 1 and r["run_rows"] == 4                 # 2 teams x 2 sources
    runs = pd.read_sql_query("SELECT * FROM nfl_inactives_probe_runs", conn)
    mn = runs[(runs.team == "MIN") & (runs.source == probe.SOURCE_ROSTER)].iloc[0]
    assert mn.minutes_to_kickoff == pytest.approx(69.0)
    assert mn.roster_week == 5 and mn.n_rows == 2 and mn.ok == 1
    assert mn.source_updated == "Sun, 11 Oct 2026 15:40:00 GMT"
    players = pd.read_sql_query("SELECT * FROM nfl_inactives_probe", conn)
    assert set(zip(players.source, players.player_name, players.status)) == {
        (probe.SOURCE_ROSTER, "Jordan Addison", "INA"),
        (probe.SOURCE_ESPN, "Jordan Addison", "QUESTIONABLE"),
        (probe.SOURCE_ESPN, "T.J. Hockenson", "OUT")}


def test_previous_weeks_roster_lists_no_inactives(conn):
    """Week 5's roster not published yet: the team's latest roster is week 4, whose INA rows belonged to another game."""
    probe.probe_once(conn, now=_now(15, 51), games=_games(),
                     roster_fetch=lambda s: (_roster(week=4), None), espn_fetch=lambda: {})
    runs = pd.read_sql_query("SELECT * FROM nfl_inactives_probe_runs WHERE source = 'nflverse_roster'", conn)
    assert set(runs.roster_week) == {4}
    assert pd.read_sql_query("SELECT * FROM nfl_inactives_probe WHERE source = 'nflverse_roster'", conn).empty


def test_failed_fetches_are_recorded_not_raised(conn):
    probe.probe_once(conn, now=_now(15, 51), games=_games(),
                     roster_fetch=lambda s: (None, None), espn_fetch=lambda: {})
    runs = pd.read_sql_query("SELECT * FROM nfl_inactives_probe_runs", conn)
    assert len(runs) == 4 and set(runs.ok) == {0}
    assert pd.read_sql_query("SELECT * FROM nfl_inactives_probe", conn).empty


def test_probe_never_touches_projection_tables(conn):
    store_rows_before = conn.execute("SELECT COUNT(*) FROM nfl_game_projections").fetchone()[0]
    probe.probe_once(conn, now=_now(15, 51), games=_games(),
                     roster_fetch=lambda s: (_roster(), None), espn_fetch=lambda: _espn())
    assert conn.execute("SELECT COUNT(*) FROM nfl_game_projections").fetchone()[0] == store_rows_before == 0
    assert conn.execute("SELECT COUNT(*) FROM nfl_player_projections").fetchone()[0] == 0


def test_lead_time_is_first_snapshot_showing_the_player_out(conn):
    """ESPN lists Addison Questionable at T-90, Out at T-30; the roster shows INA only after kickoff."""
    for hh, mm, status, ina in ((15, 30, "Questionable", ()), (16, 30, "Out", ()), (17, 10, "Out", ("Jordan Addison",))):
        probe.probe_once(conn, now=_now(hh, mm), games=_games(),
                         roster_fetch=lambda s, ina=ina: (_roster(ina=ina), None),
                         espn_fetch=lambda status=status: _espn(status))
    lead = probe.lead_times(conn, 2026, 5)
    a = lead[lead.player_name == "Jordan Addison"].set_index("source")
    assert a.loc[probe.SOURCE_ESPN, "first_listed_min"] == pytest.approx(90.0)
    assert a.loc[probe.SOURCE_ESPN, "first_out_min"] == pytest.approx(30.0)
    assert a.loc[probe.SOURCE_ROSTER, "first_out_min"] == pytest.approx(-10.0)      # only after kickoff
    assert lead[(lead.source == probe.SOURCE_ESPN) & (lead.player_name == "T.J. Hockenson")
                ].iloc[0].first_listed_min == pytest.approx(90.0)


def test_repeated_snapshot_at_same_instant_is_idempotent(conn):
    for _ in range(2):
        probe.probe_once(conn, now=_now(15, 51), games=_games(),
                         roster_fetch=lambda s: (_roster(), None), espn_fetch=lambda: _espn())
    assert conn.execute("SELECT COUNT(*) FROM nfl_inactives_probe_runs").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM nfl_inactives_probe").fetchone()[0] == 3


def test_reuses_store_kickoff_conversion():
    """The window rests on store.kickoff_utc (ET date + HH:MM -> UTC)."""
    assert store.kickoff_utc(pd.Timestamp("2026-10-11"), "13:00") == "2026-10-11T17:00+00:00"


# ── CLI wiring: the launchd job runs `evmax project nfl-inactives-probe` ──────

def _cli(monkeypatch, tmp_path, args):
    from typer.testing import CliRunner

    from evmax.cli.app import app

    db = tmp_path / "cli.db"
    real_connect = probe.connect
    monkeypatch.setattr(probe, "connect", lambda path=None: real_connect(db))
    return CliRunner().invoke(app, ["project", "nfl-inactives-probe", *args], env={"COLUMNS": "240"}), db


def test_cli_command_is_registered_with_the_launchd_argv(monkeypatch, tmp_path):
    """Regression: the plist ran this exact argv for weeks and got 'No such command'."""
    seen = {}

    def fake_probe_once(conn, lookahead_min, lookback_min):
        seen.update(lookahead=lookahead_min, lookback=lookback_min)
        return {"games": 0, "run_rows": 0, "player_rows": 0}

    monkeypatch.setattr(probe, "probe_once", fake_probe_once)
    res, _ = _cli(monkeypatch, tmp_path, [])
    assert res.exit_code == 0, res.output
    assert "No game inside the probe window" in res.output
    assert seen == {"lookahead": probe.DEFAULT_LOOKAHEAD_MIN, "lookback": probe.DEFAULT_LOOKBACK_MIN}


def test_cli_report_shows_event_and_outcome(monkeypatch, tmp_path):
    res, db = _cli(monkeypatch, tmp_path, ["--report", "--season", "2026", "--week", "5"])
    assert res.exit_code == 0 and "No probe data yet" in res.output
    with probe.connect(db) as c:
        probe.probe_once(c, now=_now(15, 51), games=_games(),
                         roster_fetch=lambda s: (_roster(), None), espn_fetch=lambda: _espn())
    res, _ = _cli(monkeypatch, tmp_path, ["--report", "--season", "2026", "--week", "5"])
    assert res.exit_code == 0, res.output
    assert "Event" in res.output and "Outcome" in res.output
    assert "MIN @ NO" in res.output and "Jordan Addison (MIN)" in res.output
