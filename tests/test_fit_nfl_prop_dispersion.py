"""Tests for scripts/fit_nfl_prop_dispersion.py — anchor recovery + parity.

The script is the reproducible evidence behind the NFL yardage Gamma in
evmax/ev/prop_pricing.py. Its two load-bearing pieces are (1) recovering the
real Pinnacle anchor from archived rows — new rows carry it, legacy rows are
inverted through the pricing family in force when they were written — and
(2) vectorized families that must equal the shipped module exactly.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evmax.archiver import DataArchiver
from evmax.ev.prop_pricing import (
    _GAMMA_STAT_SCALE,
    NegBinomialProp,
    NormalProp,
    price_kalshi_threshold,
)
from evmax.models.odds import SharpBook, SharpOdds

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_nfl_prop_dispersion.py"
_spec = importlib.util.spec_from_file_location("fit_nfl_prop_dispersion", _SCRIPT)
fitmod = importlib.util.module_from_spec(_spec)
sys.modules["fit_nfl_prop_dispersion"] = fitmod
_spec.loader.exec_module(fitmod)

KICK = datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc)


def test_nfl_week_is_tuesday_to_monday():
    weeks = fitmod.nfl_week(pd.Series(
        ["2026-09-10", "2026-09-13", "2026-09-14", "2026-09-15", "2026-09-17"]))
    assert weeks.tolist() == [1, 1, 1, 2, 2]


def test_legacy_anchor_inverts_normal_rungs():
    """Legacy rungs were NormalProp(σ=24) re-lines of one anchor."""
    dist = NormalProp.from_anchor(83.5, 0.47, sigma=24.0)
    rungs = pd.DataFrame({"K": [60.0, 70.0, 90.0, 100.0, 150.0]})
    rungs["pk"] = [dist.prob_at_or_above(k) for k in rungs.K]
    assert fitmod._legacy_anchor("receiving_yards", rungs, 0.47) == 83.5


def test_legacy_anchor_inverts_negbin_rungs():
    dist = NegBinomialProp.from_anchor(4.5, 0.55, k=5.0)
    rungs = pd.DataFrame({"K": [3.0, 4.0, 6.0, 7.0]})
    rungs["pk"] = [dist.prob_at_or_above(k) for k in rungs.K]
    assert fitmod._legacy_anchor("receptions", rungs, 0.55) == 4.5


def test_legacy_anchor_needs_a_usable_rung():
    rungs = pd.DataFrame({"K": [200.0], "pk": [0.0001]})
    assert fitmod._legacy_anchor("receiving_yards", rungs, 0.5) is None


def test_shipped_gamma_family_matches_module():
    L = np.array([12.5, 47.5, 83.5])
    p = np.array([0.50, 0.44, 0.58])
    K = np.array([25.0, 90.0, 150.0])
    for stat in ("receiving_yards", "rushing_yards"):
        vec = fitmod.gamma_power([_GAMMA_STAT_SCALE[stat]], L, p, K, power=1.0)
        mod = [price_kalshi_threshold(stat, a, b, c) for a, b, c in zip(L, p, K)]
        assert vec == pytest.approx(mod, abs=1e-7)


def _odds(player, line, prob, fetched, derived=False, anchor=None):
    o = SharpOdds(
        event_id=f"nfl::2026-10-04::prop::{player}::receiving_yards::{line}",
        book=SharpBook.pinnacle,
        sector="nfl",
        outcome_a_label="over",
        outcome_b_label="under",
        outcome_a_decimal=1.95,
        outcome_b_decimal=1.87,
        margin=0.04,
        total_line=line,
        true_prob_over=prob,
        true_prob_under=1 - prob,
        prop_player_name=player,
        prop_stat_type="receiving_yards",
        event_date=KICK,
        fetched_at=fetched,
    )
    if derived:
        o = o.model_copy(update={"derived": True, "anchor_line": anchor[0],
                                 "anchor_prob_over": anchor[1]})
    return o


def test_load_anchors_prefers_quote_then_carried_anchor(tmp_path, monkeypatch):
    """New archive rows: a real quote (derived=0) is used as-is; a derived
    rung's carried anchor is used when no quote was archived; snapshots at or
    after kickoff are ignored."""
    db = tmp_path / "archive.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db)
    arch = DataArchiver()
    pre = KICK - timedelta(hours=1)
    arch.open_session("s1", ["nfl"], "test")
    arch.archive_sharp_odds("s1", "nfl", [
        _odds("a", 60.5, 0.52, pre),                                       # quote
        _odds("a", 80.0, 0.20, pre, derived=True, anchor=(60.5, 0.52)),
        _odds("b", 70.0, 0.31, pre, derived=True, anchor=(55.5, 0.49)),    # carried only
    ])
    arch.open_session("s2", ["nfl"], "test")
    arch.archive_sharp_odds("s2", "nfl", [_odds("a", 99.5, 0.9, KICK + timedelta(minutes=5))])

    got = fitmod.load_anchors(db, ("receiving_yards",)).set_index("player")
    assert got.loc["a", ["line", "p_anchor", "anchor_source"]].tolist() == [60.5, 0.52, "quote"]
    assert got.loc["b", ["line", "p_anchor", "anchor_source"]].tolist() == [55.5, 0.49, "carried"]


def test_load_anchors_inverts_legacy_rows(tmp_path):
    """A pre-change archive (no derived columns, rungs only) recovers the
    anchor line from the legacy σ=24 family and the prob from the decimals."""
    db = tmp_path / "legacy.db"
    dist = NormalProp.from_anchor(83.5, 0.5, sigma=24.0)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE archived_sharp_odds (fetched_at TEXT, sector TEXT, event_id TEXT, "
            "book TEXT, outcome_a_decimal REAL, outcome_b_decimal REAL, total_line REAL, "
            "true_prob_over REAL, event_date TEXT, prop_player_name TEXT, prop_stat_type TEXT)"
        )
        for k in (60.0, 90.0, 130.0):
            conn.execute(
                "INSERT INTO archived_sharp_odds VALUES (?, 'nfl', ?, 'pinnacle', 1.909, 1.909, "
                "?, ?, ?, 'chase', 'receiving_yards')",
                ((KICK - timedelta(hours=1)).isoformat(),
                 f"nfl::2026-10-04::prop::chase::receiving_yards::{k}", k,
                 dist.prob_at_or_above(k), KICK.isoformat()),
            )
    got = fitmod.load_anchors(db, ("receiving_yards",))
    assert got.line.tolist() == [83.5]
    assert got.p_anchor.iloc[0] == pytest.approx(0.5, abs=1e-9)
    assert got.anchor_source.tolist() == ["legacy_inverted"]
