"""Unit tests for the extracted resolve-time model updater.

`evmax.agents.cleanup.model_updater.update_models_for_date` is the library that
both `evmax update scores` and the `evmax cleanup resolve` hook call. These
tests pin the behavior the refactor must preserve from the old CLI `_run_update`:
the no-ESPN-scores edge, the (team_a, team_b) dedup, the soccer-only xG branch,
and the FUZZY_THRESHOLD=65 slug-match value.

State and DB are fully isolated — `fetch_completed_scores` / `get_connection`
are patched at module level (the established idiom), and the coordinator is a
mock so no real model state is read or written.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from evmax.agents.cleanup import model_updater


def _empty_conn(with_ledger: bool = True):
    """In-memory DB with the ev_predictions table but no rows.

    No DB match → the updater falls back to NameNormalizer slugs, which is the
    common path for sectors with no logged predictions on the date.

    `with_ledger=False` omits `applied_model_games` to exercise the tolerant
    path for databases that predate the idempotency guard.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY,
            event_id TEXT,
            sector TEXT,
            event_date TEXT
        )"""
    )
    if with_ledger:
        conn.execute(
            """CREATE TABLE applied_model_games (
                id INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT (datetime('now')),
                sector TEXT NOT NULL,
                event_date TEXT NOT NULL,
                team_a TEXT NOT NULL,
                team_b TEXT NOT NULL,
                UNIQUE(sector, event_date, team_a, team_b)
            )"""
        )
    return conn


def _score(home_name, away_name, home_score, away_score, **extra):
    base = {
        "home_name": home_name,
        "away_name": away_name,
        "home_score": home_score,
        "away_score": away_score,
    }
    base.update(extra)
    return base


def test_no_espn_scores_does_not_touch_coordinator():
    coord = MagicMock()
    with patch.object(model_updater, "fetch_completed_scores", return_value=[]) as fetch, \
         patch.object(model_updater, "get_connection", side_effect=AssertionError("DB hit on empty fetch")):
        result = asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    assert fetch.call_count == 1
    assert coord.update_models.call_count == 0
    assert result.updated == 0
    assert result.games == []


def test_happy_path_feeds_each_game_once():
    coord = MagicMock()
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        result = asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    assert coord.update_models.call_count == 1
    assert result.updated == 1
    kwargs = coord.update_models.call_args.kwargs
    assert kwargs["sector"] == "nba"
    assert kwargs["score_a"] == 110
    assert kwargs["score_b"] == 102
    assert kwargs["event_date"] == "2026-06-09"


def test_dry_run_does_not_mutate_state():
    coord = MagicMock()
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        result = asyncio.run(
            model_updater.update_models_for_date(
                ["nba"], date(2026, 6, 9), coordinator=coord, dry_run=True
            )
        )

    assert coord.update_models.call_count == 0
    assert result.updated == 0
    assert len(result.games) == 1
    assert result.games[0].applied is False


def test_duplicate_pairs_deduped():
    coord = MagicMock()
    # Two ESPN rows that normalize to the same (team_a, team_b) — must update once.
    scores = [
        _score("Boston Celtics", "Miami Heat", 110, 102),
        _score("Boston Celtics", "Miami Heat", 110, 102),
    ]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        result = asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    assert coord.update_models.call_count == 1
    assert result.updated == 1


def test_soccer_records_xg_for_both_teams():
    coord = MagicMock()
    xg = MagicMock()
    coord.soccer_xg_agent = xg
    scores = [
        _score(
            "Inter Miami", "LA Galaxy", 2, 1,
            home_sot=6, away_sot=3, home_shots=14, away_shots=9,
        )
    ]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        asyncio.run(
            model_updater.update_models_for_date(["soccer"], date(2026, 6, 9), coordinator=coord)
        )

    assert xg.record_match.call_count == 2
    assert xg.save_state.call_count == 1


def test_worldcup_records_xg_with_sector():
    """National-team WC carries shotsOnTarget/totalShots from ESPN too, so the
    xG feed must fire for `worldcup` — and pass sector='worldcup' so the agent
    writes to the WC namespace, not the club `soccer` pool."""
    coord = MagicMock()
    xg = MagicMock()
    coord.soccer_xg_agent = xg
    scores = [
        _score(
            "Spain", "Argentina", 2, 1,
            home_sot=7, away_sot=4, home_shots=15, away_shots=11,
        )
    ]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        asyncio.run(
            model_updater.update_models_for_date(["worldcup"], date(2026, 6, 25), coordinator=coord)
        )

    assert xg.record_match.call_count == 2
    assert xg.save_state.call_count == 1
    # Every record_match call must be tagged with the worldcup namespace.
    assert all(c.kwargs.get("sector") == "worldcup" for c in xg.record_match.call_args_list)


def test_non_soccer_does_not_touch_xg():
    coord = MagicMock()
    xg = MagicMock()
    coord.soccer_xg_agent = xg
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    assert xg.record_match.call_count == 0


def test_db_slug_match_uses_slug_names_at_threshold_65():
    # The updater must adopt the canonical slug names from a matched DB event
    # rather than the raw normalizer output. Pin the threshold at 65: this
    # mascot-rich ESPN name vs the bare-city slug matches at 65 but the partial
    # overlap is below resolver's 72 — so importing 72 would change behavior.
    coord = MagicMock()
    conn = _empty_conn()
    conn.execute(
        "INSERT INTO ev_predictions (event_id, sector, event_date) VALUES (?, ?, ?)",
        ("nba::2026-06-09::celtics_vs_heat", "nba", "2026-06-09"),
    )
    conn.commit()
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=conn):
        asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    kwargs = coord.update_models.call_args.kwargs
    assert kwargs["team_a"] == "celtics"
    assert kwargs["team_b"] == "heat"


def test_threshold_constant_is_65():
    # Importing resolver.FUZZY_THRESHOLD (72) instead of keeping the local 65
    # would silently change dedup/match behavior — pin it here.
    assert model_updater.FUZZY_THRESHOLD == 65


# ---------------------------------------------------------------------------
# Applied-games ledger (idempotency guard)
#
# `evmax update scores` and the `evmax cleanup resolve` hook both call
# update_models_for_date. Before the ledger, a daily run of both fed every game
# of that date into Elo/Form/Poisson twice — the in-loop `processed` set only
# dedups within a single invocation. Observed live: PR #128 recorded 72 game
# increments for a 36-game slate, PR #130 recorded 30 for 15.
# ---------------------------------------------------------------------------


def _conn_factory(tmp_path, with_ledger: bool = True):
    """Open a fresh connection to one on-disk DB per call.

    Mirrors production `get_connection`, which opens (and closes) a new
    connection per sector. An in-memory DB cannot be used here: the updater
    closes the connection when a sector finishes, so state must outlive it.
    """
    db = tmp_path / "predictions.db"
    seed = sqlite3.connect(str(db))
    seed.execute(
        """CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY, event_id TEXT, sector TEXT, event_date TEXT
        )"""
    )
    if with_ledger:
        seed.execute(
            """CREATE TABLE applied_model_games (
                id INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT (datetime('now')),
                sector TEXT NOT NULL,
                event_date TEXT NOT NULL,
                team_a TEXT NOT NULL,
                team_b TEXT NOT NULL,
                UNIQUE(sector, event_date, team_a, team_b)
            )"""
        )
    seed.commit()
    seed.close()

    def _open():
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        return conn

    return _open


def _run(scores, coord, opener, **kwargs):
    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", side_effect=opener):
        return asyncio.run(
            model_updater.update_models_for_date(
                ["nba"], date(2026, 6, 9), coordinator=coord, **kwargs
            )
        )


def test_second_pass_over_same_date_does_not_reapply(tmp_path):
    """The regression: update scores + cleanup resolve must feed a game once."""
    opener = _conn_factory(tmp_path)
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    coord = MagicMock()
    first = _run(scores, coord, opener)
    assert first.updated == 1
    assert first.skipped == 0
    assert coord.update_models.call_count == 1

    coord2 = MagicMock()
    second = _run(scores, coord2, opener)
    assert second.updated == 0
    assert second.skipped == 1
    assert coord2.update_models.call_count == 0, "game was fed into model state twice"
    assert second.games[0].already_applied is True
    assert second.games[0].applied is False


def test_force_reapplies_an_already_ledgered_game(tmp_path):
    opener = _conn_factory(tmp_path)
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    _run(scores, MagicMock(), opener)

    coord = MagicMock()
    result = _run(scores, coord, opener, force=True)
    assert result.updated == 1
    assert result.skipped == 0
    assert coord.update_models.call_count == 1


def test_ledger_is_keyed_per_date_and_sector(tmp_path):
    """A game on a later date is a different key — it must still be applied."""
    opener = _conn_factory(tmp_path)
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]
    _run(scores, MagicMock(), opener)

    coord = MagicMock()
    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", side_effect=opener):
        result = asyncio.run(
            model_updater.update_models_for_date(
                ["nba"], date(2026, 6, 10), coordinator=coord
            )
        )
    assert result.updated == 1
    assert coord.update_models.call_count == 1


def test_dry_run_does_not_write_the_ledger(tmp_path):
    """A preview must not make the subsequent real run a no-op."""
    opener = _conn_factory(tmp_path)
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    _run(scores, MagicMock(), opener, dry_run=True)

    coord = MagicMock()
    result = _run(scores, coord, opener)
    assert result.updated == 1
    assert coord.update_models.call_count == 1


def test_failed_game_is_not_ledgered_and_stays_retryable(tmp_path):
    """update_models raising must leave the game eligible for a later retry."""
    opener = _conn_factory(tmp_path)
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    failing = MagicMock()
    failing.update_models.side_effect = RuntimeError("ESPN blew up")
    first = _run(scores, failing, opener)
    assert first.updated == 0
    assert first.games[0].error == "ESPN blew up"

    coord = MagicMock()
    retry = _run(scores, coord, opener)
    assert retry.updated == 1
    assert retry.skipped == 0
    assert coord.update_models.call_count == 1


def test_missing_ledger_table_degrades_to_no_guard(tmp_path):
    """A database predating the guard must still update, not raise."""
    opener = _conn_factory(tmp_path, with_ledger=False)
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    coord = MagicMock()
    result = _run(scores, coord, opener)
    assert result.updated == 1
    assert result.skipped == 0
    assert coord.update_models.call_count == 1


def test_ledger_table_is_in_the_shipped_schema():
    """get_connection must create applied_model_games, or the guard is inert."""
    from evmax.agents.cleanup import db as cleanup_db

    assert "applied_model_games" in cleanup_db.SCHEMA
    assert "UNIQUE(sector, event_date, team_a, team_b)" in cleanup_db.SCHEMA


# ---------------------------------------------------------------------------
# Batched model-state saves (Phase 3) — per-game update calls must NOT save;
# the hook flushes each state exactly once per sector.
# ---------------------------------------------------------------------------

def test_update_models_called_with_save_false():
    """The batch path must defer persistence: every per-game update_models call
    passes save=False so it doesn't re-serialize the state files per game."""
    coord = MagicMock()
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    assert coord.update_models.call_count == 1
    assert coord.update_models.call_args.kwargs["save"] is False


def test_state_flushed_once_per_sector_over_many_games():
    """Ten games in one sector → ten update calls but ONE save_model_states."""
    coord = MagicMock()
    scores = [
        _score(f"Home {i}", f"Away {i}", 100 + i, 90 + i) for i in range(10)
    ]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", return_value=_empty_conn()):
        result = asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    assert result.updated == 10
    assert coord.update_models.call_count == 10
    assert coord.save_model_states.call_count == 1
    assert coord.save_model_states.call_args.args[0] == "nba"


def test_no_flush_when_nothing_applied(tmp_path):
    """A sector whose only game is already ledgered must not flush state."""
    opener = _conn_factory(tmp_path)  # on-disk so the ledger survives run 1
    coord = MagicMock()
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", side_effect=opener):
        asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )
        coord.reset_mock()
        second = asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    assert second.skipped == 1
    assert coord.update_models.call_count == 0
    assert coord.save_model_states.call_count == 0


def test_state_saved_before_ledger_written(tmp_path):
    """Crash-safety ordering: the state flush must happen BEFORE any ledger row
    is committed, so a crash between them re-applies rather than loses games."""
    opener = _conn_factory(tmp_path)
    scores = [_score("Boston Celtics", "Miami Heat", 110, 102)]

    order: list[str] = []
    coord = MagicMock()
    coord.save_model_states.side_effect = lambda *a, **k: order.append("save")

    real_record = model_updater._record_applied

    def _tracking_record(*args, **kwargs):
        order.append("ledger")
        return real_record(*args, **kwargs)

    with patch.object(model_updater, "fetch_completed_scores", return_value=scores), \
         patch.object(model_updater, "get_connection", side_effect=opener), \
         patch.object(model_updater, "_record_applied", side_effect=_tracking_record):
        asyncio.run(
            model_updater.update_models_for_date(["nba"], date(2026, 6, 9), coordinator=coord)
        )

    assert order == ["save", "ledger"], f"expected save before ledger, got {order}"


def test_prefetch_fetches_each_sector_once():
    """The prefetch phase must request each sector's scores exactly once."""
    coord = MagicMock()
    calls: list[str] = []

    async def _fake_fetch(sector, target_date, cache=None):
        calls.append(sector)
        return [_score("H", "A", 1, 0)]

    with patch.object(model_updater, "fetch_completed_scores", side_effect=_fake_fetch), \
         patch.object(model_updater, "get_connection", side_effect=lambda: _empty_conn()):
        asyncio.run(
            model_updater.update_models_for_date(
                ["nba", "nfl", "nhl"], date(2026, 6, 9), coordinator=coord
            )
        )

    assert sorted(calls) == ["nba", "nfl", "nhl"]


def test_espn_cache_threaded_to_fetch():
    """A shared espn_cache must be passed through to fetch_completed_scores so
    the resolve phase and the model-update hook share one scoreboard fetch."""
    coord = MagicMock()
    shared: dict = {}
    seen_cache = []

    async def _fake_fetch(sector, target_date, cache=None):
        seen_cache.append(cache)
        return []

    with patch.object(model_updater, "fetch_completed_scores", side_effect=_fake_fetch):
        asyncio.run(
            model_updater.update_models_for_date(
                ["nba"], date(2026, 6, 9), coordinator=coord, espn_cache=shared
            )
        )

    assert seen_cache == [shared]
