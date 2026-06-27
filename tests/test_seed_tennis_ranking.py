"""Tests for the matchmx-derived ranking-history aggregator.

`aggregate_ranking_history` replaced the offline Sackmann `_rankings_current.csv`
path (and the ESPN top-150 fallback): ranking_trend is now seeded purely from
Tennis Abstract's matchmx `winner_rank` / `loser_rank` columns. Covers:
  - winner + loser both contribute a dated rank snapshot
  - one snapshot per calendar day (intra-day duplicate matches collapse)
  - rank 0 / missing rank / bad date rows are dropped
  - players with fewer than _MIN_RANK_SNAPSHOTS dated points are excluded
  - output is sorted by date and shaped for TennisRankingTrendAgent.seed_history
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import seed_tennis_models as mod  # noqa: E402


def test_winner_and_loser_both_get_snapshots():
    rows = [
        {"tourney_date": "20240101", "winner_name": "Carlos Alcaraz",
         "loser_name": "Jannik Sinner", "winner_rank": "2", "loser_rank": "4"},
        {"tourney_date": "20240115", "winner_name": "Jannik Sinner",
         "loser_name": "Carlos Alcaraz", "winner_rank": "1", "loser_rank": "3"},
    ]
    h = mod.aggregate_ranking_history(rows)
    assert h["carlos alcaraz"] == [
        {"date": "2024-01-01", "rank": 2},
        {"date": "2024-01-15", "rank": 3},
    ]
    assert h["jannik sinner"] == [
        {"date": "2024-01-01", "rank": 4},
        {"date": "2024-01-15", "rank": 1},
    ]


def test_one_snapshot_per_day():
    # Two matches on the same day for the same player collapse to one snapshot.
    rows = [
        {"tourney_date": "20240115", "winner_name": "Carlos Alcaraz",
         "loser_name": "A", "winner_rank": "2", "loser_rank": "50"},
        {"tourney_date": "20240115", "winner_name": "Carlos Alcaraz",
         "loser_name": "B", "winner_rank": "2", "loser_rank": "60"},
        {"tourney_date": "20240201", "winner_name": "Carlos Alcaraz",
         "loser_name": "C", "winner_rank": "2", "loser_rank": "70"},
    ]
    h = mod.aggregate_ranking_history(rows)
    dates = [s["date"] for s in h["carlos alcaraz"]]
    assert dates == ["2024-01-15", "2024-02-01"]  # not three


def test_invalid_rank_and_date_rows_dropped():
    rows = [
        {"tourney_date": "20240101", "winner_name": "P", "loser_name": "Q",
         "winner_rank": "0", "loser_rank": "5"},        # winner rank 0 → skip winner
        {"tourney_date": "20240115", "winner_name": "P", "loser_name": "Q",
         "winner_rank": "", "loser_rank": "5"},         # winner rank missing → skip winner
        {"tourney_date": "", "winner_name": "P", "loser_name": "Q",
         "winner_rank": "3", "loser_rank": "5"},        # bad date → whole row skipped
    ]
    h = mod.aggregate_ranking_history(rows)
    # P never recorded a usable (date, rank); Q has rank 5 on two valid dates.
    assert "p" not in h
    assert h["q"] == [{"date": "2024-01-01", "rank": 5},
                      {"date": "2024-01-15", "rank": 5}]


def test_below_min_snapshots_excluded():
    # A player with a single dated rank point can never fire the trend window.
    rows = [
        {"tourney_date": "20240101", "winner_name": "Solo",
         "loser_name": "Other", "winner_rank": "10", "loser_rank": "0"},
    ]
    h = mod.aggregate_ranking_history(rows)
    assert "solo" not in h  # only 1 snapshot < _MIN_RANK_SNAPSHOTS


def test_output_feeds_seed_history_shape():
    rows = [
        {"tourney_date": "20240201", "winner_name": "Z", "loser_name": "Y",
         "winner_rank": "9", "loser_rank": "12"},
        {"tourney_date": "20240101", "winner_name": "Z", "loser_name": "Y",
         "winner_rank": "11", "loser_rank": "13"},
    ]
    h = mod.aggregate_ranking_history(rows)
    # Sorted ascending by date despite reversed input order.
    assert [s["date"] for s in h["z"]] == ["2024-01-01", "2024-02-01"]
    # Every snapshot is the {"date": str, "rank": int} contract seed_history wants.
    for snaps in h.values():
        for s in snaps:
            assert isinstance(s["date"], str) and isinstance(s["rank"], int)
