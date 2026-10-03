"""Tests for scripts/clv_by_entry_timing.py (bucketing + game declustering)."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "clv_by_entry_timing.py"
_spec = importlib.util.spec_from_file_location("clv_by_entry_timing", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_bucket_edges():
    assert mod.bucket_of(1) == "<1h"
    assert mod.bucket_of(59) == "<1h"
    assert mod.bucket_of(60) == "1-3h"
    assert mod.bucket_of(1439) == "12-24h"
    assert mod.bucket_of(1440) == "24h+"
    assert mod.bucket_of(0) is None  # at/after tip is never bucketed


def test_game_key_collapses_rungs_and_markets():
    a = mod.game_key("nfl::2026-09-13::lions_vs_saints::spread::-3.5")
    b = mod.game_key("nfl::2026-09-13::lions_vs_saints::total")
    assert a == b == "nfl::2026-09-13::lions_vs_saints"


def _db(tmp_path, rows):
    p = tmp_path / "p.db"
    with sqlite3.connect(p) as c:
        c.execute(
            """CREATE TABLE ev_predictions (sector, market_type, venue, event_id,
               minutes_to_tipoff, kalshi_clv_pct, line, yes_team, scan_date,
               voided, void_reason)"""
        )
        c.executemany(
            "INSERT INTO ev_predictions VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
        )
    return p


def test_load_filters_voids_and_in_play(tmp_path):
    g = "nfl::2026-09-13::a_vs_b::spread"
    rows = [
        ("nfl", "spread", "kalshi", g, 2000, 1.0, -3.5, "A", "2026-09-12", 0, None),
        ("nfl", "spread", "kalshi", g, 2000, 1.0, -3.5, "A", "2026-09-12", 1, "stale_reverted"),
        ("nfl", "spread", "kalshi", g, 2000, 9.0, -3.5, "A", "2026-09-12", 1, "manual"),
        ("nfl", "spread", "kalshi", g, 0, 9.0, -3.5, "A", "2026-09-12", 0, None),
        ("nfl", "spread", "kalshi", g, 2000, None, -3.5, "A", "2026-09-12", 0, None),
    ]
    out = mod.load(_db(tmp_path, rows), ["nfl"], None)
    assert len(out) == 2  # live + stale_reverted kept; cancel/in-play/NULL dropped
    assert all(r["kalshi_clv_pct"] == 1.0 for r in out)


def test_summarize_empty_and_single():
    assert mod.summarize([]) == "n=0"
    assert "n=1" in mod.summarize([1.0])
