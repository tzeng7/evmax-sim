"""Pure-helper tests for the NCAAF backtest harness (no network).

The walk-forward + fetch paths are exercised by running the script; here we guard
the small pure functions whose bugs would silently corrupt the report: week
bucketing, market-spread orientation, and the Brier/accuracy scorer.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "ncaaf_bt", Path(__file__).resolve().parents[1] / "scripts" / "backtest_ncaaf_efficiency.py"
)
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def test_week_index_and_bucket():
    start = dt.date(2024, 8, 24)
    assert bt._week_index("2024-08-25", start) == 1
    assert bt._bucket(bt._week_index("2024-08-25", start)) == "0-3"
    assert bt._bucket(bt._week_index("2024-10-05", start)) == "4-8"      # ~week 7
    assert bt._bucket(bt._week_index("2024-11-30", start)) == "9+"       # ~week 15


def test_bucket_boundaries():
    assert bt._bucket(3) == "0-3"
    assert bt._bucket(4) == "4-8"
    assert bt._bucket(8) == "4-8"
    assert bt._bucket(9) == "9+"


def test_brier_scorer():
    rows = [
        {"model": 0.9, "y": 1},
        {"model": 0.1, "y": 0},
        {"model": 0.5, "y": 1},
    ]
    (brier, acc), n = bt._brier(rows, "model")
    assert n == 3
    # (0.9-1)^2 + (0.1-0)^2 + (0.5-1)^2 = 0.01 + 0.01 + 0.25 = 0.27 / 3 = 0.09
    assert brier == pytest.approx(0.09, abs=1e-4)
    # 0.9→1 correct, 0.1→0 correct, 0.5→(>=0.5)=1 correct
    assert acc == pytest.approx(1.0)


def test_brier_skips_missing():
    rows = [{"model": 0.8, "y": 1}, {"model": None, "y": 0}]
    (_, _), n = bt._brier(rows, "model")
    assert n == 1   # the None row is skipped, not counted as 0


def test_orient_spreads_matches_full_names_and_flips_away_rows():
    """cfbfastR's `abbr` column carries full team names for 2023+ ("Penn State");
    the old abbreviation-only join matched ~5% of rows. Home rows keep their
    sign, away rows flip, abbreviations still work as a fallback, unmatched
    labels and unknown games are skipped, and the median spans books."""
    games = [
        {"game_id": "1", "home": {"location": "Penn State", "abbr": "PSU"},
         "away": {"location": "Ohio State", "abbr": "OSU"}},
        {"game_id": "2", "home": {"location": "South Florida", "abbr": "USF"},
         "away": {"location": "Old Dominion", "abbr": "ODU"}},
    ]
    rows = [
        ("1", "Penn State", 3.5),        # home row, full name
        ("1", "Ohio State", -3.5),       # away row, flipped -> +3.5
        (1.0, "OSU", -4.5),              # abbreviation fallback, float game id -> +4.5
        ("1", "Nobody", 9.0),            # unmatched label skipped
        ("2", "South Florida", -4.0),
        ("2", "Old Dominion", 4.0),
        ("2", "USF", float("nan")),      # NaN skipped
        ("3", "Penn State", 1.0),        # unknown game skipped
    ]
    out = bt._orient_spreads(rows, games)
    assert out == {"1": 3.5, "2": -4.0}
