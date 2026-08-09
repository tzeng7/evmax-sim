"""Tests for the ensemble's why-not diagnostics (WS3, 2026-07-19).

BlendedPrediction.diagnostics classifies every sector-configured model as
fired / gated / missing per event, so shadow analysis can separate
recoverable coverage gaps (player known but thin, or absent from state)
from correctly-withheld thin matches — without re-running predict_pair.
"""

from __future__ import annotations

import pytest

from evmax.agents.base import AgentRequest
from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.ensemble_agent import EnsembleModelAgent
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


def _make_market(sector: str = "soccer") -> PredictionMarket:
    return PredictionMarket(
        id="mkt-1",
        source=MarketSource.kalshi,
        sector=sector,
        market_type=MarketType.moneyline,
        title="A vs B",
        yes_price=0.50,
        no_price=0.50,
        team_home="team a",
        team_away="team b",
    )


def _make_sharp(event_id: str, sector: str = "soccer") -> SharpOdds:
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector=sector,
        market_type=MarketType.moneyline,
        outcome_a_label="team a",
        outcome_b_label="team b",
        outcome_a_decimal=1.0 / 0.55,
        outcome_b_decimal=1.0 / 0.45,
        true_prob_a=0.55,
        true_prob_b=0.45,
        margin=0.03,
    )


def _stub(name_: str, *, conf: float, w: float = 1.0, returns_none: bool = False):
    class _Stub(ModelAgent):
        name = name_
        weight = w

        def load_state(self) -> None:
            self._state = {}

        def save_state(self) -> None:
            pass

        async def update(self, *args, **kwargs):
            return None

        async def predict_pair(self, market, sharp_odds):
            if returns_none:
                return None
            return ModelAgentPrediction(
                event_id=sharp_odds.event_id,
                model_name=self.name,
                true_prob_a=0.56,
                true_prob_b=0.44,
                true_prob_draw=None,
                confidence=conf,
                weight=self.weight,
                sample_size=17,
                notes=f"{self.name} note n=17",
            )

    return _Stub()


@pytest.fixture
def ensemble(tmp_path, monkeypatch):
    monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
    # Named after real soccer-configured models so the categories.yaml
    # sector scoping keeps them in the diagnostics universe.
    fires = _stub("elo", conf=0.80)
    gated = _stub("form", conf=0.30)
    absent = _stub("poisson", conf=0.80, returns_none=True)
    return EnsembleModelAgent(models=[fires, gated, absent], sharp_weight=0.85)


@pytest.mark.asyncio
async def test_fired_gated_missing_classified(ensemble):
    market = _make_market()
    sharp = _make_sharp("ev-1")
    req = AgentRequest(
        sector="soccer",
        params={"pairs": [{"market": market, "sharp": sharp}], "sharp_weight": 0.85},
    )
    resp = await ensemble(req)
    blend = resp.data["ev-1"]

    diag = blend.diagnostics
    assert "elo" in diag["fired"]
    assert diag["fired"]["elo"]["n"] == 17
    # Gated: predicted but fell at the 0.45 confidence gate — the agent's
    # own note is preserved as the "why".
    assert "form" in diag["gated"]
    assert diag["gated"]["form"]["conf"] == 0.30
    assert "note" in diag["gated"]["form"]
    # Missing: returned no prediction at all.
    assert "poisson" in diag["missing"]
    # And gated/missing models never appear in model_sources.
    assert "form" not in blend.model_sources
    assert "poisson" not in blend.model_sources
    assert "elo" in blend.model_sources


@pytest.mark.asyncio
async def test_sharp_only_fallback_lists_all_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
    absent_a = _stub("elo", conf=0.80, returns_none=True)
    absent_b = _stub("poisson", conf=0.80, returns_none=True)
    ensemble = EnsembleModelAgent(models=[absent_a, absent_b], sharp_weight=0.85)

    market = _make_market()
    sharp = _make_sharp("ev-2")
    req = AgentRequest(
        sector="soccer",
        params={"pairs": [{"market": market, "sharp": sharp}], "sharp_weight": 0.85},
    )
    resp = await ensemble(req)
    blend = resp.data["ev-2"]

    assert blend.model_sources == "sharp"
    assert blend.diagnostics["fired"] == {}
    assert set(blend.diagnostics["missing"]) == {"elo", "poisson"}
    # Sharp-only fallback → no model share → default effective_sharp_weight 1.0,
    # so the downstream injury/rest/playoff layers are fully suppressed (they
    # would otherwise double-count news the pure-sharp blend already prices).
    assert blend.effective_sharp_weight == 1.0


@pytest.mark.asyncio
async def test_effective_sharp_weight_populated_on_mixed_blend(ensemble):
    """A model-carrying blend exposes the effective sharp weight so the
    EV-gap agent can scale injury/rest/playoff to the model's share."""
    market = _make_market()
    sharp = _make_sharp("ev-3")
    req = AgentRequest(
        sector="soccer",
        params={"pairs": [{"market": market, "sharp": sharp}], "sharp_weight": 0.85},
    )
    resp = await ensemble(req)
    blend = resp.data["ev-3"]
    # elo and sharp agree on the favorite, so no disagreement boost → base 0.85.
    assert blend.effective_sharp_weight == pytest.approx(0.85)
    assert 0.0 < blend.effective_sharp_weight <= 1.0


def test_sector_model_names_scoped_to_registry(tmp_path, monkeypatch):
    """A tennis row must never list NBA models as missing."""
    monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
    # Running set includes an NBA-only model name; tennis registry excludes it.
    elo = _stub("elo", conf=0.80)
    nba_only = _stub("efficiency", conf=0.80)
    ensemble = EnsembleModelAgent(models=[elo, nba_only], sharp_weight=0.85)

    tennis_models = ensemble._sector_model_names("tennis")
    assert "efficiency" not in tennis_models
    # And "sharp" is never a model name.
    assert "sharp" not in ensemble._sector_model_names("soccer")
