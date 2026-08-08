"""NCAAF generic-Elo calibration (MODEL-2 NCAAF half, 2026-08-07).

Before this change `ncaaf` appeared nowhere in evmax/agents/models/elo_agent.py,
so generic elo silently took K_FACTORS.get("ncaaf", 20.0)=20 and
HOME_ADVANTAGE_ELO.get("ncaaf", 0.0)=0 — a ZERO college-football home edge, which
scripts/backtest_ncaaf_elo.py showed costs +0.0221 holdout Brier vs the swept
winner K=40 / home_adv=60. These tests pin the calibrated constants and prove the
home bonus is applied. NOTE: ncaaf elo is still unseeded and absent from the
resolve-time update hook, so it stays gated (confidence 0.30) in the live blend
until it is explicitly activated — see test_ncaaf_elo_inert_until_seeded.
"""

from __future__ import annotations

import asyncio

from evmax.agents.models.elo_agent import (
    EloModelAgent,
    HOME_ADVANTAGE_ELO,
    K_FACTORS,
    MED_DATA_THRESHOLD,
)
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook


def _pair(home: str, away: str, sector: str = "ncaaf") -> tuple[PredictionMarket, SharpOdds]:
    market = PredictionMarket(
        id="t", market_id="t", event_id=f"t_{home}_{away}",
        sector=sector, team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=0.5, no_price=0.5,
    )
    sharp = SharpOdds(
        event_id=f"t_{home}_{away}", book=SharpBook.pinnacle, sector=sector,
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=0.5, true_prob_b=0.5,
    )
    return market, sharp


def _warm_agent(ratings: dict[str, float], gp: int = 20) -> EloModelAgent:
    """Agent with a WARM ncaaf state (gp above the confidence gate) so predict
    isn't dropped for thin data — lets us isolate the home-advantage effect."""
    agent = EloModelAgent()
    agent._state = {
        "ncaaf": {
            "ratings": dict(ratings),
            "game_counts": {t: gp for t in ratings},
            "season_games": {t: gp for t in ratings},
            "h2h": {},
        }
    }
    return agent


def test_ncaaf_elo_constants_calibrated():
    """ncaaf must carry its OWN K/home_adv, not the silent fallbacks. The home
    edge being > 0 is the core regression guard for the fixed bug."""
    assert "ncaaf" in K_FACTORS
    assert "ncaaf" in HOME_ADVANTAGE_ELO
    # Sane range, not a re-assertion of the exact winner (so a future re-sweep
    # doesn't force a test edit for a minor retune).
    assert 16.0 <= K_FACTORS["ncaaf"] <= 50.0
    assert HOME_ADVANTAGE_ELO["ncaaf"] > 0.0, "zero CFB home edge is the bug"


def test_ncaaf_elo_constants_exact_sweep_winner():
    """Pins the 2026-08-07 sweep result (scripts/backtest_ncaaf_elo.py)."""
    assert K_FACTORS["ncaaf"] == 40.0
    assert HOME_ADVANTAGE_ELO["ncaaf"] == 60.0


def test_ncaaf_elo_applies_home_advantage():
    """Two EQUALLY-rated ncaaf teams: the home team must be favored purely from
    the home bonus. Under the old fallback (home_adv=0) this was exactly 0.500."""
    agent = _warm_agent({"alabama": 1500.0, "auburn": 1500.0})
    market, sharp = _pair("alabama", "auburn")  # outcome_a = home
    result = asyncio.run(agent.predict_pair(market, sharp))
    assert result is not None
    # ~60 Elo home bump on even ratings ⇒ P(home) ≈ 0.585, comfortably > 0.55.
    assert result.true_prob_a > 0.55
    assert result.true_prob_b < 0.45


def test_ncaaf_elo_predicts_stronger_home_favorite():
    """Sanity: a higher-rated home team is favored more than the home bump alone."""
    agent = _warm_agent({"georgia": 1650.0, "vanderbilt": 1400.0})
    market, sharp = _pair("georgia", "vanderbilt")
    result = asyncio.run(agent.predict_pair(market, sharp))
    assert result is not None
    assert result.true_prob_a > 0.70


def test_ncaaf_elo_inert_until_seeded():
    """With no seeded ncaaf ratings every team is 1500 and min_count=0, so
    predict_pair returns confidence 0.30 — at/below the ensemble's 0.45 gate.
    Documents that the calibrated constants are correct-ahead-of-activation:
    they have NO live effect until ncaaf elo is seeded + fed (a separate change).
    """
    agent = EloModelAgent()
    agent._state = {}  # no ncaaf key → empty state
    market, sharp = _pair("alabama", "auburn")
    result = asyncio.run(agent.predict_pair(market, sharp))
    assert result is not None
    assert result.confidence == 0.30
    assert result.sample_size == 0
    # Above-gate warm data is what flips it live — proves the 0.30 is the gate,
    # not a hardcode.
    warm = _warm_agent({"alabama": 1500.0, "auburn": 1500.0}, gp=MED_DATA_THRESHOLD + 1)
    warm_res = asyncio.run(warm.predict_pair(market, sharp))
    assert warm_res is not None and warm_res.confidence > 0.45
