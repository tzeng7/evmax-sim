"""ModelAgent.run() must isolate per-pair failures.

Regression: a single matchup raising inside predict_pair (e.g. the pitcher
agent's ZeroDivisionError on an unseeded both-starters game) used to propagate
out of run(), so the ensemble logged the whole model as failed and dropped it
for every game in the slate. For baseball that silently removed the pitcher
signal, and the pitcher-required guard then dropped every moneyline. run()
now catches per-pair, logs, and keeps the surviving predictions.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evmax.agents.base import AgentRequest
from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


def _pair(team_a: str, team_b: str) -> dict:
    market = PredictionMarket(
        id=f"kalshi:{team_a}_vs_{team_b}",
        source=MarketSource.kalshi,
        sector="baseball",
        market_type=MarketType.moneyline,
        title=f"{team_a} vs {team_b}",
        ticker="KXMLBGAME-TEST",
        yes_price=0.5,
        no_price=0.5,
        volume_usd=1000.0,
        open_interest_usd=500.0,
        team_home=team_a,
        team_away=team_b,
        event_date=datetime(2026, 6, 13, 23, 0, tzinfo=timezone.utc),
    )
    sharp = SharpOdds(
        event_id=f"baseball::2026-06-13::{team_a}_vs_{team_b}",
        book=SharpBook.pinnacle,
        sector="baseball",
        outcome_a_label=team_a,
        outcome_b_label=team_b,
        outcome_a_decimal=2.0,
        outcome_b_decimal=2.0,
        true_prob_a=0.5,
        true_prob_b=0.5,
        event_date=datetime(2026, 6, 13, 23, 0, tzinfo=timezone.utc),
    )
    return {"market": market, "sharp": sharp}


class _PartlyBrokenAgent(ModelAgent):
    """Raises on any pair whose event_id contains 'boom', predicts otherwise."""

    name = "broken"
    weight = 0.5

    async def predict_pair(self, market, sharp_odds):
        if "boom" in sharp_odds.event_id:
            raise ZeroDivisionError("float division by zero")
        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=0.55,
            true_prob_b=0.45,
            confidence=0.6,
            weight=self.weight,
        )

    def update(self, *args, **kwargs) -> None:  # abstract no-op for the stub
        pass


@pytest.fixture
def broken_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
    a = _PartlyBrokenAgent()
    a._state_path = tmp_path / "broken_state.json"
    return a


@pytest.mark.asyncio
async def test_one_bad_pair_does_not_sink_the_batch(broken_agent):
    req = AgentRequest(
        sector="baseball",
        params={"pairs": [_pair("good", "boom"), _pair("alpha", "beta")]},
    )
    resp = await broken_agent.run(req)
    # The 'boom' pair raised and was skipped; the clean pair survived.
    assert resp.status == "ok"
    assert "baseball::2026-06-13::alpha_vs_beta" in resp.data
    assert "baseball::2026-06-13::good_vs_boom" not in resp.data
    assert len(resp.data) == 1
