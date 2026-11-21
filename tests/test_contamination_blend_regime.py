"""Blend-weight regime contamination: pre-#315 / #318 / #253 0.40 sharp-weight blends.

Before PR #315 dashboard scans blended nfl/ncaaf/wnba/ncaab/ncaaw/nhl/worldcup
at sharp_weight 0.40 (CLI used 0.85), with every row stamped 0.85. Soccer MLS /
unmapped leagues sat at 0.40 until #318, and PolyUS soccer fell to the 0.40
default until #253. Those rows made the board and value audit report NCAAF and
PolyUS soccer moneylines as significantly worse than sharp long after the fix.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evmax.agents.cleanup.contamination import (
    DASHBOARD_WEIGHT_FIX_UTC,
    POLYUS_SOCCER_LEAGUE_FIX_UTC,
    SOCCER_TIER_WEIGHT_FIX_UTC,
    blend_weight_regime_contaminated,
    is_contaminated,
    is_contaminated_row,
)

_BEFORE_315 = "2026-09-23 05:18:09"  # last 0.40 dashboard row in the live DB
_AFTER_315 = "2026-09-23 08:37:38"


@pytest.mark.parametrize(
    "sector", ["nfl", "ncaaf", "wnba", "ncaab", "ncaaw", "nhl", "worldcup"]
)
def test_dashboard_weight_sectors_ml_before_fix_flagged(sector):
    assert blend_weight_regime_contaminated(sector, "moneyline", _BEFORE_315)
    assert not blend_weight_regime_contaminated(sector, "moneyline", _AFTER_315)


def test_cutoff_is_exclusive_at_merge_second():
    assert not blend_weight_regime_contaminated("nfl", "moneyline", DASHBOARD_WEIGHT_FIX_UTC)


def test_sector_and_market_type_case_insensitive():
    assert blend_weight_regime_contaminated("NFL", "Moneyline", _BEFORE_315)


def test_worldcup_advance_is_a_blended_market():
    assert blend_weight_regime_contaminated("worldcup", "advance", "2026-07-01 12:00:00")


@pytest.mark.parametrize("mt", ["spread", "total", "player_prop", None])
def test_sharp_derived_market_types_never_flagged(mt):
    # Spread/total fairs are sharp-derived: the sharp weight never touched them.
    assert not blend_weight_regime_contaminated("nfl", mt, _BEFORE_315)


@pytest.mark.parametrize("sector", ["nba", "tennis", "baseball", "ufc", "lol", "cs2"])
def test_sectors_with_their_own_weight_entry_not_flagged(sector):
    assert not blend_weight_regime_contaminated(sector, "moneyline", _BEFORE_315)


def test_soccer_mls_and_unmapped_flagged_until_318():
    between = "2026-09-23 07:30:00"  # after #315, before #318
    assert DASHBOARD_WEIGHT_FIX_UTC < between < SOCCER_TIER_WEIGHT_FIX_UTC
    assert blend_weight_regime_contaminated("soccer", "moneyline", between, "kalshi", "mls")
    assert blend_weight_regime_contaminated("soccer", "moneyline", between, "kalshi", None)
    assert not blend_weight_regime_contaminated(
        "soccer", "moneyline", "2026-09-23 08:00:00", "kalshi", "mls"
    )


def test_soccer_kalshi_top_tier_stays_clean_before_fixes():
    assert not blend_weight_regime_contaminated(
        "soccer", "moneyline", "2026-08-01 12:00:00", "kalshi", "epl"
    )


def test_soccer_polyus_flagged_only_before_league_fix():
    before = "2026-09-01 12:00:00"
    after = "2026-09-10 12:00:00"
    assert before < POLYUS_SOCCER_LEAGUE_FIX_UTC < after
    assert blend_weight_regime_contaminated(
        "soccer", "moneyline", before, "polymarket_us", "epl"
    )
    assert not blend_weight_regime_contaminated(
        "soccer", "moneyline", after, "polymarket_us", "epl"
    )


def test_iso_t_separator_and_missing_timestamp():
    assert blend_weight_regime_contaminated("nfl", "moneyline", "2026-09-20T12:00:00+00:00")
    assert not blend_weight_regime_contaminated("nfl", "moneyline", None)
    assert not blend_weight_regime_contaminated("nfl", "moneyline", "")


def test_is_contaminated_without_logged_at_keeps_legacy_behavior():
    v2 = "elo+ncaaf_efficiency_v2+sharp"
    # Positional call: the time rule cannot fire; code-state rules still do.
    assert not is_contaminated("ncaaf", "moneyline", v2, None)
    assert is_contaminated("ncaaf", "moneyline", "elo+sharp", None)
    assert is_contaminated("ncaaf", "moneyline", v2, None, logged_at=_BEFORE_315)
    assert not is_contaminated("ncaaf", "moneyline", v2, None, logged_at=_AFTER_315)


def test_is_contaminated_row_reads_sqlite_row_and_dict():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    r = conn.execute(
        "SELECT 'ncaaf' AS sector, 'moneyline' AS market_type,"
        " 'elo+ncaaf_efficiency_v2+sharp' AS model_sources, NULL AS line,"
        " ? AS logged_at, 'kalshi' AS venue, NULL AS league",
        (_BEFORE_315,),
    ).fetchone()
    assert is_contaminated_row(r)
    # A row without the time columns reads them as None (never flagged by time).
    assert not is_contaminated_row(
        {"sector": "ncaaf", "market_type": "moneyline",
         "model_sources": "elo+ncaaf_efficiency_v2+sharp", "line": None}
    )


def test_gate_callers_use_the_row_based_check():
    """Every evmax caller goes through ``is_contaminated_row``. A bare
    ``is_contaminated(...)`` call outside the contamination module would
    silently skip the time-based rule."""
    root = Path(__file__).resolve().parents[1] / "evmax"
    offenders = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if p.name != "contamination.py" and "is_contaminated(" in p.read_text()
    )
    assert offenders == []


def test_value_audit_excludes_pre_fix_rows(tmp_path, monkeypatch):
    from evmax.agents.cleanup import db as db_module
    from evmax.agents.cleanup.value_audit import compute_value_audit

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "predictions.db")
    game = "2099-01-01"  # inside any look-back window
    with db_module.get_connection() as conn:
        for i, logged in enumerate(
            [_BEFORE_315] * 3 + [_AFTER_315] * 2
        ):
            mid = f"kalshi:NCAAF-{i}"
            conn.execute(
                """INSERT INTO ev_predictions
                   (logged_at, scan_date, market_id, event_id, sector, yes_team,
                    market_type, event_date, kalshi_yes_price, sharp_true_prob,
                    blended_true_prob, ev_pct, kelly_fraction, model_sources, mode)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (logged, game, mid, f"ncaaf::{i}", "ncaaf", "a", "moneyline", game,
                 0.5, 0.55, 0.60, 0.05, 0.01, "elo+ncaaf_efficiency_v2+sharp", "live"),
            )
            conn.execute(
                """INSERT INTO ev_outcomes
                   (market_id, event_id, event_date, sector, yes_team, outcome,
                    sharp_true_prob, blended_true_prob)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (mid, f"ncaaf::{i}", game, "ncaaf", "a", 1, 0.55, 0.60),
            )
        conn.commit()

    audit = {a["sector"]: a for a in compute_value_audit(weeks=12)}
    assert audit["ncaaf"]["n"] == 2
    assert audit["ncaaf"]["n_excluded_contaminated"] == 3
