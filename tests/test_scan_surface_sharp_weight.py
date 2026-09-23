"""Every scan surface must blend at the same sharp weight, and every logged row
must record the weight its blend actually used.

Regression for the 2026-09 finding: ``AgentCoordinator`` defaulted to
``sharp_weight=0.40`` while the CLI passed the config's 0.85, so dashboard and
portfolio scans blended nfl/ncaaf/wnba/ncaab/ncaaw/nhl/worldcup markets at 0.40
(≈4x the documented model share) and Kelly-sized live NFL/WNBA plays on the
inflated edge — while ``log_gaps`` stamped a constant 0.85 on every row, hiding
it (and mis-sizing the pick-time re-blend, which shifts the fair by the stamp).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from evmax.agents.cleanup import logger as logger_module
from evmax.agents.cleanup.logger import log_gaps, stamped_sharp_weight
from evmax.agents.coordinator import AgentCoordinator

from tests.test_logger_mode_routing import _gap, patched_db  # noqa: F401

_CFG = {
    "sharp_weight": 0.85,
    "sharp_weight_by_sector": {"nba": 0.7, "tennis": 0.85, "soccer": 0.88},
}


def _coord(**kw) -> AgentCoordinator:
    return AgentCoordinator(sectors=["nba"], respect_season_window=False, **kw)


def test_coordinator_default_reads_config_not_040():
    with patch("evmax.agents.cleanup.metrics.load_config", return_value={"sharp_weight": 0.77}):
        assert _coord().sharp_weight == pytest.approx(0.77)


def test_coordinator_default_matches_shipped_config():
    from evmax.agents.cleanup.metrics import load_config

    assert _coord().sharp_weight == pytest.approx(float(load_config()["sharp_weight"]))


def test_coordinator_explicit_weight_wins():
    with patch("evmax.agents.cleanup.metrics.load_config", return_value={"sharp_weight": 0.77}):
        assert _coord(sharp_weight=0.6).sharp_weight == pytest.approx(0.6)


def test_dashboard_scan_path_builds_coordinator_without_040():
    """The dashboard's shared scan path must not pin a weight of its own."""
    import inspect

    from evmax.web import app as web_app

    src = inspect.getsource(web_app)
    assert "sharp_weight=0.4" not in src.replace(" ", "")
    assert "sharp_weight_used=getattr(coord, \"sharp_weight\", None)" in src


@pytest.mark.parametrize(
    "sector,global_w,expected",
    [
        ("nba", None, 0.7),      # per-sector entry wins
        ("nba", 0.6, 0.7),       # …even over an explicit global
        ("nfl", None, 0.85),     # no entry → config global
        ("nfl", 0.6, 0.6),       # no entry → the scan's explicit global
        ("wnba", None, 0.85),
    ],
)
def test_stamped_sharp_weight_mirrors_coordinator(sector, global_w, expected):
    assert stamped_sharp_weight(_gap("m", sector=sector), global_w, _CFG) == pytest.approx(expected)


def test_stamped_sharp_weight_soccer_uses_league_tier():
    from evmax.sectors.soccer_tiers import sharp_weight_for_league

    epl = _gap("KXEPLGAME-26SEP27ARSCHE-ARS", sector="soccer")
    epl.league = "epl"
    mls = _gap("KXMLSGAME-26SEP27SEALA-SEA", sector="soccer")
    mls.league = "mls"
    assert stamped_sharp_weight(epl, 0.85, _CFG) == pytest.approx(sharp_weight_for_league("epl"))
    assert stamped_sharp_weight(mls, 0.85, _CFG) == pytest.approx(sharp_weight_for_league("mls"))
    # the soccer sector-level 0.88 is NOT what the coordinator blends at
    assert stamped_sharp_weight(mls, 0.85, _CFG) != pytest.approx(0.88)


def test_log_gaps_stamps_per_sector_effective_weight(patched_db, monkeypatch):  # noqa: F811
    monkeypatch.setattr("evmax.agents.cleanup.metrics.load_config", lambda: dict(_CFG))
    monkeypatch.setattr(logger_module, "_default_mode_resolver", lambda cat, *a, **k: "live")
    assert log_gaps([_gap("g_nba", sector="nba"), _gap("g_nfl", sector="nfl")], bankroll_used=500.0) == 2
    rows = {
        r["market_id"]: r["sharp_weight_used"]
        for r in patched_db.execute("SELECT market_id, sharp_weight_used FROM ev_predictions")
    }
    assert rows == {"g_nba": pytest.approx(0.7), "g_nfl": pytest.approx(0.85)}


@pytest.mark.parametrize("sw", [0.85, 0.88, 0.40])
def test_flb_double_blend_effective_model_share_is_pinned(sw):
    """The FLB step re-blends the ALREADY sharp-blended prob, so a model's
    effective share at a 50/50 line is (1-sw)^2 — 0.0225 at the 0.85 default.
    Kept deliberately (2026-09-22 replay: ~30%-share rows were significantly
    worse than sharp). Changing this multiplies live model weight several-fold;
    it must be a deliberate, evidence-backed decision that updates this test."""
    from evmax.agents.models.ensemble_agent import EnsembleModelAgent
    from evmax.models.odds import SharpBook, SharpOdds

    s, m = 0.5, 0.7
    sharp = SharpOdds(
        event_id="e", book=SharpBook.pinnacle, sector="nfl",
        outcome_a_label="a", outcome_b_label="b",
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=s, true_prob_b=1 - s, margin=0.0,
    )
    first = sw * s + (1 - sw) * m
    pa, pb, _ = EnsembleModelAgent._flb_correct(first, 1 - first, None, sharp, sw)
    assert pa == pytest.approx(s + (1 - sw) ** 2 * (m - s))


@pytest.mark.parametrize("sector,sw,share", [("nfl", 0.85, 0.15 ** 2), ("nba", 0.70, 0.30)])
def test_blend_pipeline_effective_share_is_pinned(sector, sw, share):
    """Pins the FULL ``_blend`` path, not just ``_flb_correct``: dropping the FLB
    call, passing it a different weight, or running it for NBA all fail here.
    A 2pp model/sharp gap keeps the disagreement ramp inactive."""
    from evmax.agents.models.base import ModelAgentPrediction
    from evmax.agents.models.ensemble_agent import EnsembleModelAgent
    from evmax.models.odds import SharpBook, SharpOdds

    s, m = 0.50, 0.52
    sharp = SharpOdds(
        event_id="e", book=SharpBook.pinnacle, sector=sector,
        outcome_a_label="a", outcome_b_label="b",
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=s, true_prob_b=1 - s, margin=0.0,
    )
    preds = {"elo": ModelAgentPrediction(
        event_id="e", model_name="elo", true_prob_a=m, true_prob_b=1 - m,
        true_prob_draw=None, confidence=0.60, weight=0.35,
    )}
    blend = EnsembleModelAgent(models=[], sharp_weight=sw)._blend("e", preds, sharp, sw, sector=sector)
    assert blend.true_prob_a == pytest.approx(s + share * (m - s), abs=2e-5)
