"""NFL player-prop resolution via the ESPN football boxscore.

Regression: ``_resolve_prop_observations`` used to send every non-baseball prop
to the NBA endpoints, so nfl_props rows never resolved (0 of 50k+).
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from evmax.agents.cleanup import resolver
from evmax.agents.cleanup.resolver import (
    _nfl_player_key,
    _parse_nfl_box_player_stats,
)


def _block(name, keys, athletes):
    return {
        "name": name,
        "keys": keys,
        "athletes": [
            {"athlete": {"id": str(aid), "displayName": n}, "stats": stats}
            for aid, n, stats in athletes
        ],
    }


PASS_KEYS = ["completions/passingAttempts", "passingYards", "passingTouchdowns"]
RUSH_KEYS = ["rushingAttempts", "rushingYards", "rushingTouchdowns"]
REC_KEYS = ["receptions", "receivingYards", "receivingTouchdowns"]


def _box():
    return {
        "boxscore": {
            "players": [
                {
                    "statistics": [
                        _block("passing", PASS_KEYS, [(1, "C.J. Stroud", ["30/46", "350", "2"])]),
                        _block("rushing", RUSH_KEYS, [
                            (1, "C.J. Stroud", ["3", "11", "1"]),
                            (2, "Marvin Harrison Jr.", ["1", "4", "0"]),
                        ]),
                        _block("receiving", REC_KEYS, [
                            (3, "Amon-Ra St. Brown", ["10", "126", "1"]),
                            (2, "Marvin Harrison Jr.", ["5", "70", "0"]),
                        ]),
                    ]
                }
            ]
        }
    }


@pytest.mark.parametrize(
    "a,b",
    [
        ("c.j._stroud", "C.J. Stroud"),
        ("amon-ra_st._brown", "Amon-Ra St. Brown"),
        ("de'von_achane", "De'Von Achane"),
        ("marvin_harrison", "Marvin Harrison Jr."),
        ("jose_ramirez", "José Ramírez"),
    ],
)
def test_player_key_joins_kalshi_slug_and_espn_name(a, b):
    assert _nfl_player_key(a) == _nfl_player_key(b)


def test_player_key_keeps_same_surname_players_apart():
    assert _nfl_player_key("josh_allen") != _nfl_player_key("allen_lazard")


def test_parse_reads_blocks_and_defaults_absent_blocks_to_zero():
    stats = _parse_nfl_box_player_stats(_box())
    qb = stats[_nfl_player_key("C.J. Stroud")]
    assert qb["passing_yards"] == 350 and qb["passing_tds"] == 2
    assert qb["rushing_yards"] == 11 and qb["anytime_td"] == 1  # 1 rush TD
    wr = stats[_nfl_player_key("Amon-Ra St. Brown")]
    assert wr["receptions"] == 10 and wr["receiving_yards"] == 126
    assert wr["rushing_yards"] == 0.0  # played, no carries -> 0, not missing
    assert wr["anytime_td"] == 1


def test_parse_drops_two_athletes_sharing_a_name():
    box = _box()
    box["boxscore"]["players"][0]["statistics"].append(
        _block("receiving", REC_KEYS, [(99, "Amon-Ra St. Brown", ["1", "5", "0"])])
    )
    assert _nfl_player_key("Amon-Ra St. Brown") not in _parse_nfl_box_player_stats(box)


def test_parse_absent_player_not_returned():
    assert _nfl_player_key("Nobody Here") not in _parse_nfl_box_player_stats(_box())


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None):
        if url.endswith("/scoreboard"):
            assert "football/nfl" in url
            return _Resp({"events": [
                {"id": "g1", "competitions": [{"status": {"type": {"completed": True}}}]},
                {"id": "g2", "competitions": [{"status": {"type": {"completed": False}}}]},
            ]})
        assert params == {"event": "g1"}  # unfinished game never fetched
        return _Resp(_box())

    def close(self):
        pass


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE prop_observations (id INTEGER PRIMARY KEY, scan_date TEXT,
           sector TEXT, player_name TEXT, stat_type TEXT, line REAL, event_id TEXT,
           outcome INTEGER, actual_value REAL)"""
    )
    return conn


def _add(conn, player, stat, line, day):
    conn.execute(
        "INSERT INTO prop_observations (scan_date, sector, player_name, stat_type, line, event_id)"
        " VALUES (?, 'nfl', ?, ?, ?, ?)",
        (day, player, stat, line, f"nfl::{day}::prop::{player}::{stat}::{line}"),
    )


def test_resolve_prop_observations_routes_nfl_and_scores_over_at_threshold(monkeypatch):
    monkeypatch.setattr(resolver.httpx, "Client", _FakeClient)
    today = date.today().isoformat()
    conn = _db()
    _add(conn, "c.j._stroud", "passing_yards", 350.0, today)   # == line -> over
    _add(conn, "c.j._stroud", "passing_yards", 375.0, today)   # under
    _add(conn, "amon-ra_st._brown", "rushing_yards", 5.0, today)  # 0 -> under
    _add(conn, "amon-ra_st._brown", "anytime_td", 1.0, today)  # 1 -> over
    _add(conn, "nobody_here", "receptions", 3.0, today)        # absent -> pending
    n = resolver._resolve_prop_observations(conn, date.today())
    got = {
        (r["player_name"], r["stat_type"], r["line"]): (r["outcome"], r["actual_value"])
        for r in conn.execute("SELECT * FROM prop_observations")
    }
    assert n == 4
    assert got[("c.j._stroud", "passing_yards", 350.0)] == (1, 350.0)
    assert got[("c.j._stroud", "passing_yards", 375.0)] == (0, 350.0)
    assert got[("amon-ra_st._brown", "rushing_yards", 5.0)] == (0, 0.0)
    assert got[("amon-ra_st._brown", "anytime_td", 1.0)] == (1, 1.0)
    assert got[("nobody_here", "receptions", 3.0)] == (None, None)


def test_nfl_rows_never_hit_the_nba_endpoint(monkeypatch):
    seen = []

    class Spy(_FakeClient):
        def get(self, url, params=None):
            seen.append(url)
            return super().get(url, params)

    monkeypatch.setattr(resolver.httpx, "Client", Spy)
    conn = _db()
    _add(conn, "c.j._stroud", "passing_yards", 300.0, date.today().isoformat())
    resolver._resolve_prop_observations(conn, date.today())
    assert seen and all("basketball" not in u for u in seen)


# ── Kalshi-settlement fallback (zero-catch games) ─────────────────────────────
# ESPN lists a player only in blocks where he recorded a stat, so a player who
# played and caught nothing is absent from the boxscore. Kalshi settles those
# markets NO (it voids only players who did not play). Before the fallback the
# rows stayed pending forever and dropped zero outcomes from every NFL prop
# calibration / shadow ROI.

_PAST = "2026-10-04"


class _KResp(_Resp):
    def __init__(self, payload, status_code=200):
        super().__init__(payload)
        self.status_code = status_code


def _make_client(kalshi: dict, scoreboard_fails: bool = False, seen: list | None = None):
    """Fake httpx.Client: ESPN scoreboard/summary + Kalshi GET /markets/{ticker}.

    ``kalshi`` maps ticker -> result string, or a list of status codes/results
    consumed in order (to simulate a 429 followed by success).
    """

    class Client(_FakeClient):
        def get(self, url, params=None):
            if url.startswith("/markets/"):
                ticker = url.removeprefix("/markets/")
                if seen is not None:
                    seen.append(ticker)
                spec = kalshi[ticker]
                if isinstance(spec, list):
                    spec = spec.pop(0)
                if spec == 429:
                    return _KResp({}, status_code=429)
                return _KResp({"market": {"ticker": ticker, "result": spec}})
            if url.endswith("/scoreboard") and scoreboard_fails:
                raise RuntimeError("espn down")
            return super().get(url, params)

    return Client


def _db_full():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE prop_observations (id INTEGER PRIMARY KEY, scan_date TEXT,
           sector TEXT, player_name TEXT, stat_type TEXT, line REAL, event_id TEXT,
           outcome INTEGER, actual_value REAL, market_id TEXT, venue TEXT)"""
    )
    return conn


def _add_k(conn, player, stat, line, day, ticker, venue="kalshi"):
    prefix = "kalshi:" if venue == "kalshi" else "polymarket_us:"
    conn.execute(
        "INSERT INTO prop_observations (scan_date, sector, player_name, stat_type, line,"
        " event_id, market_id, venue) VALUES (?, 'nfl', ?, ?, ?, ?, ?, ?)",
        (day, player, stat, line, f"nfl::{day}::prop::{player}::{stat}::{line}",
         prefix + ticker, venue),
    )


def _rows(conn):
    return {r["market_id"]: (r["outcome"], r["actual_value"])
            for r in conn.execute("SELECT * FROM prop_observations")}


def test_zero_catch_player_resolves_no_from_kalshi_settlement(monkeypatch):
    monkeypatch.setattr(resolver.httpx, "Client", _make_client(
        {"KXNFLREC-26OCT04X-TE-2": "no", "KXNFLRECYDS-26OCT04X-TE-15": "no"}))
    conn = _db_full()
    _add_k(conn, "c.j._stroud", "passing_yards", 300.0, _PAST, "KXNFLPASSYDS-26OCT04X-QB-300")
    _add_k(conn, "zero_catch_te", "receptions", 2.0, _PAST, "KXNFLREC-26OCT04X-TE-2")
    _add_k(conn, "zero_catch_te", "receiving_yards", 15.0, _PAST, "KXNFLRECYDS-26OCT04X-TE-15")
    n = resolver._resolve_prop_observations(conn, date(2026, 10, 4))
    got = _rows(conn)
    assert n == 3
    assert got["kalshi:KXNFLPASSYDS-26OCT04X-QB-300"] == (1, 350.0)  # ESPN graded it
    # Kalshi NO -> outcome 0; the stat itself is unknown, so actual stays NULL
    assert got["kalshi:KXNFLREC-26OCT04X-TE-2"] == (0, None)
    assert got["kalshi:KXNFLRECYDS-26OCT04X-TE-15"] == (0, None)


def test_kalshi_void_and_open_markets_stay_pending(monkeypatch):
    monkeypatch.setattr(resolver.httpx, "Client", _make_client(
        {"KXNFLREC-26OCT04X-DNP-3": "scalar", "KXNFLREC-26OCT04X-OPEN-3": ""}))
    conn = _db_full()
    _add_k(conn, "scratched_wr", "receptions", 3.0, _PAST, "KXNFLREC-26OCT04X-DNP-3")
    _add_k(conn, "late_game_wr", "receptions", 3.0, _PAST, "KXNFLREC-26OCT04X-OPEN-3")
    assert resolver._resolve_prop_observations(conn, date(2026, 10, 4)) == 0
    assert set(_rows(conn).values()) == {(None, None)}


def test_espn_outage_falls_back_to_kalshi(monkeypatch):
    monkeypatch.setattr(resolver.httpx, "Client", _make_client(
        {"KXNFLRECYDS-26OCT04X-WR-60": "yes"}, scoreboard_fails=True))
    conn = _db_full()
    _add_k(conn, "some_wr", "receiving_yards", 60.0, _PAST, "KXNFLRECYDS-26OCT04X-WR-60")
    assert resolver._resolve_prop_observations(conn, date(2026, 10, 4)) == 1
    assert _rows(conn)["kalshi:KXNFLRECYDS-26OCT04X-WR-60"] == (1, None)


def test_todays_and_non_kalshi_rows_never_query_kalshi(monkeypatch):
    seen: list = []
    monkeypatch.setattr(resolver.httpx, "Client", _make_client({}, seen=seen))
    conn = _db_full()
    today = date.today().isoformat()
    _add_k(conn, "absent_today", "receptions", 3.0, today, "KXNFLREC-TODAY-X-3")
    _add_k(conn, "absent_poly", "receptions", 3.0, _PAST, "nfl-poly-slug", venue="polymarket_us")
    conn.execute(
        "INSERT INTO prop_observations (scan_date, sector, player_name, stat_type, line, event_id,"
        " market_id, venue) VALUES (?, 'nfl', 'no_side', 'receptions', 3.0, ?, ?, 'kalshi')",
        (_PAST, f"nfl::{_PAST}::prop::no_side::receptions::3.0", "kalshi:KXNFLREC-26OCT04X-NS-3:no"),
    )
    resolver._resolve_prop_observations(conn, date.today())
    resolver._resolve_prop_observations(conn, date(2026, 10, 4))
    assert seen == []
    assert set(_rows(conn).values()) == {(None, None)}


def test_kalshi_429_is_retried(monkeypatch):
    monkeypatch.setattr(resolver.time, "sleep", lambda s: None)
    monkeypatch.setattr(resolver.httpx, "Client", _make_client(
        {"KXNFLREC-26OCT04X-TE-2": [429, "no"]}))
    conn = _db_full()
    _add_k(conn, "zero_catch_te", "receptions", 2.0, _PAST, "KXNFLREC-26OCT04X-TE-2")
    assert resolver._resolve_prop_observations(conn, date(2026, 10, 4)) == 1
    assert _rows(conn)["kalshi:KXNFLREC-26OCT04X-TE-2"] == (0, None)
