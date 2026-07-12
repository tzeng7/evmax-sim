"""anchored_entry: the first-anchored-sweep entry rule in the WNBA spread
walk-forward CLV audit (scripts/backtest_wnba_spread_walkforward.py).

Entry must be the first Kalshi snapshot at/after the game's first Pinnacle
anchor — never a pre-anchor MM placeholder quote (those leak and inflate CLV).
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from backtest_wnba_spread_walkforward import anchored_entry  # noqa: E402

SNAPS = [
    ("2026-07-11T17:13:59+00:00", 0.30),  # pre-anchor placeholder
    ("2026-07-11T23:37:32+00:00", 0.35),  # pre-anchor placeholder
    ("2026-07-12T01:00:31+00:00", 0.42),  # first post-anchor
    ("2026-07-12T08:00:35+00:00", 0.47),  # close
]
ANCHOR = "2026-07-12T00:30:00+00:00"


def test_entry_is_first_post_anchor_snapshot():
    assert anchored_entry(SNAPS, ANCHOR) == 0.42


def test_all_snapshots_pre_anchor_returns_none():
    assert anchored_entry(SNAPS, "2026-07-12T09:00:00+00:00") is None


def test_never_anchored_returns_none():
    assert anchored_entry(SNAPS, None) is None


def test_anchor_exactly_at_snapshot_includes_it():
    # >= semantics: a snapshot taken at the anchor instant is anchored.
    assert anchored_entry(SNAPS, "2026-07-12T01:00:31+00:00") == 0.42


def test_anchor_before_all_snapshots_returns_first():
    assert anchored_entry(SNAPS, "2026-07-10T00:00:00+00:00") == 0.30


def test_empty_stream_returns_none():
    assert anchored_entry([], ANCHOR) is None
