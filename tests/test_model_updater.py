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


def _empty_conn():
    """In-memory DB with the ev_predictions table but no rows.

    No DB match → the updater falls back to NameNormalizer slugs, which is the
    common path for sectors with no logged predictions on the date.
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
