"""Tests for the agent framework."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evmax.agents.base import Agent, AgentBus, AgentMessage, AgentRequest, AgentResponse, AgentStatus
from evmax.agents.odds.ev_gap_agent import EVGapAgent, EVGap
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent, _poisson_pmf, _score_matrix, _win_draw_probs
from evmax.agents.models.ensemble_agent import EnsembleModelAgent
from evmax.models.market import PredictionMarket, MarketSource, MarketType
from evmax.models.odds import SharpOdds, SharpBook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_market(
    yes_price: float = 0.45,
    team_home: str = "lakers",
    team_away: str = "celtics",
    yes_team: str = "lakers",
    sector: str = "nba",
    market_type: MarketType = MarketType.moneyline,
    volume: float = 10_000.0,
) -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:{team_home}_vs_{team_away}",
        source=MarketSource.kalshi,
        sector=sector,
        market_type=market_type,
        title=f"{team_home} vs {team_away}",
        ticker=f"KXNBAGAME-TEST-{team_home.upper()}",
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        volume_usd=volume,
        open_interest_usd=5_000.0,
        team_home=team_home,
        team_away=team_away,
        yes_team=yes_team,
        event_date=datetime(2026, 3, 15, 20, 0, tzinfo=timezone.utc),
    )


def make_sharp(
    event_id: str = "nba::2026-03-15::lakers_vs_celtics",
    prob_a: float = 0.58,
    prob_b: float = 0.42,
    team_a: str = "lakers",
    team_b: str = "celtics",
    sector: str = "nba",
) -> SharpOdds:
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_label=team_a,
        outcome_b_label=team_b,
        outcome_a_decimal=1.0 / prob_a,
        outcome_b_decimal=1.0 / prob_b,
        true_prob_a=prob_a,
        true_prob_b=prob_b,
        margin=0.04,
        event_date=datetime(2026, 3, 15, 20, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# AgentBus
# ---------------------------------------------------------------------------

class TestAgentBus:
    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        bus = AgentBus()
        received = []

        async def handler(msg: AgentMessage):
            received.append(msg)

        bus.subscribe("odds.kalshi.nba", handler)
        await bus.publish(AgentMessage(topic="odds.kalshi.nba", payload=[1, 2], sender="test"))
        assert len(received) == 1
        assert received[0].payload == [1, 2]

    @pytest.mark.asyncio
    async def test_wildcard_subscribe(self):
        bus = AgentBus()
        received = []

        async def handler(msg):
            received.append(msg.topic)

        bus.subscribe("*", handler)
        await bus.publish(AgentMessage(topic="a.b.c", payload=None, sender="x"))
        await bus.publish(AgentMessage(topic="d.e.f", payload=None, sender="y"))
        assert "a.b.c" in received
        assert "d.e.f" in received

    @pytest.mark.asyncio
    async def test_no_subscribers_no_error(self):
        bus = AgentBus()
        # should not raise
        await bus.publish(AgentMessage(topic="unknown", payload={}, sender="x"))

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_propagate(self):
        bus = AgentBus()

        async def bad_handler(msg):
            raise ValueError("oops")

        bus.subscribe("topic", bad_handler)
        # should not raise
        await bus.publish(AgentMessage(topic="topic", payload=None, sender="x"))


# ---------------------------------------------------------------------------
# Agent base
# ---------------------------------------------------------------------------

class ConcreteAgent(Agent):
    name = "test_agent"
    description = "A test agent"

    def __init__(self, raise_on_run: bool = False):
        super().__init__()
        self._raise = raise_on_run

    async def run(self, request: AgentRequest) -> AgentResponse:
        if self._raise:
            raise RuntimeError("deliberate error")
        return AgentResponse(agent_name=self.name, sector=request.sector, data={"ok": True})


class TestAgent:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        agent = ConcreteAgent()
        resp = await agent(AgentRequest(sector="nba"))
        assert resp.status == "ok"
        assert resp.data == {"ok": True}
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_error_captured(self):
        agent = ConcreteAgent(raise_on_run=True)
        resp = await agent(AgentRequest(sector="nba"))
        assert resp.status == "error"
        assert "deliberate error" in resp.error

    @pytest.mark.asyncio
    async def test_status_transitions(self):
        agent = ConcreteAgent()
        assert agent.status == AgentStatus.IDLE
        await agent(AgentRequest(sector="nba"))
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_publish_requires_bus(self):
        agent = ConcreteAgent()
        # No bus attached — should be a no-op
        await agent.publish("some.topic", {"x": 1})

    @pytest.mark.asyncio
    async def test_publish_with_bus(self):
        bus = AgentBus()
        received = []

        async def handler(msg):
            received.append(msg)

        bus.subscribe("test.topic", handler)
        agent = ConcreteAgent()
        agent.attach_bus(bus)
        await agent.publish("test.topic", 42)
        assert received[0].payload == 42
        assert received[0].sender == "test_agent"


# ---------------------------------------------------------------------------
# EVGapAgent
# ---------------------------------------------------------------------------

class TestEVGapAgent:
    @pytest.mark.asyncio
    async def test_finds_positive_ev(self):
        agent = EVGapAgent()
        market = make_market(yes_price=0.45, yes_team="lakers")
        sharp = make_sharp(prob_a=0.58, team_a="lakers")

        # Bypass matching engine — inject pre-matched data
        with patch.object(agent._matching, "match_all", return_value=[(market, sharp, 95.0)]):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = await agent(req)

        assert resp.status == "ok"
        assert len(resp.data) >= 1
        gap: EVGap = resp.data[0]
        assert gap.ev_pct > 0.02
        assert gap.kelly_fraction > 0
        assert gap.blended_true_prob == pytest.approx(0.58, abs=0.001)

    @pytest.mark.asyncio
    async def test_negative_ev_filtered(self):
        agent = EVGapAgent()
        # Kalshi price higher than true prob → negative EV
        market = make_market(yes_price=0.70, yes_team="lakers")
        sharp = make_sharp(prob_a=0.58, team_a="lakers")

        with patch.object(agent._matching, "match_all", return_value=[(market, sharp, 95.0)]):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = await agent(req)

        assert len(resp.data) == 0

    def test_prop_injury_boost_applies_via_player_team_lookup(self, monkeypatch):
        """Regression for BUG-5: prop event_ids have no game slug, so the
        boost used to fall through silently. The fix looks up the player's
        team directly from the NBA props cache and checks injuries for that
        team only.
        """
        from evmax.agents.intelligence.injury_agent import (
            InjuredPlayer,
            InjuryReport,
            InjuryReportAgent,
        )

        agent = EVGapAgent()

        prop_market = PredictionMarket(
            id="kalshi:LEBRON-PTS",
            source=MarketSource.kalshi,
            sector="nba",
            market_type=MarketType.player_prop,
            title="LeBron 24.5+ Points",
            ticker="KXNBAPTS-LEBRON",
            yes_price=0.48,
            no_price=0.52,
            volume_usd=5_000.0,
            player_name="LeBron James",
            stat_type="points",
            threshold=24.5,
        )
        prop_sharp = SharpOdds(
            event_id="nba::2026-03-20::prop::LeBron James::points::24.5",
            book=SharpBook.pinnacle,
            sector="nba",
            outcome_a_label="over",
            outcome_b_label="under",
            outcome_a_decimal=1.0,
            outcome_b_decimal=1.0,
            true_prob_a=0.0,
            true_prob_b=0.0,
            true_prob_over=0.55,
            true_prob_under=0.45,
            total_line=24.5,
            margin=0.0,
            prop_player_name="LeBron James",
            prop_stat_type="points",
            prop_l15_games=15,
        )

        # Pretend the props cache knows LeBron plays for the Lakers.
        monkeypatch.setattr(
            "evmax.clients.nba_props_cache.lookup_player_team",
            lambda name: "lakers" if "lebron" in name.lower() else None,
        )

        out_star = InjuredPlayer(
            name="Anthony Davis",
            position="PF",
            status="Out",
            tier="star",
            impact=0.045,
            injury_type="Knee",
            notes="",
        )
        injuries = {
            "los angeles lakers": InjuryReport(
                team="los angeles lakers", sector="nba", players=[out_star]
            ),
        }

        gap_no_inj = agent._evaluate_prop_pair(
            prop_market, prop_sharp, confidence=95.0, sector="nba", injuries=None
        )
        gap_with_inj = agent._evaluate_prop_pair(
            prop_market, prop_sharp, confidence=95.0, sector="nba", injuries=injuries
        )

        assert gap_no_inj is not None
        assert gap_with_inj is not None
        # The boost must raise the blended probability above the raw sharp prob.
        assert gap_with_inj.blended_true_prob > gap_no_inj.blended_true_prob
        assert "inj" in gap_with_inj.model_sources
        # Sanity: boost size matches the helper's computation.
        expected_boost = InjuryReportAgent.compute_prop_injury_boost(injuries, "lakers")
        assert expected_boost > 0
        assert gap_with_inj.blended_true_prob == pytest.approx(
            min(0.95, 0.55 * (1 + expected_boost)), abs=1e-6
        )

    @pytest.mark.asyncio
    async def test_model_prob_override(self):
        from evmax.agents.models.ensemble_agent import BlendedPrediction
        agent = EVGapAgent()
        market = make_market(yes_price=0.45, yes_team="lakers")
        sharp = make_sharp(prob_a=0.50, prob_b=0.50, team_a="lakers")  # borderline with sharp only

        # Model gives a higher true prob → pushes into +EV
        blended_preds = {
            sharp.event_id: BlendedPrediction(
                event_id=sharp.event_id,
                true_prob_a=0.62,
                true_prob_b=0.38,
                true_prob_draw=None,
                confidence=0.80,
                model_sources="elo+sharp",
                per_model={},
            )
        }

        with patch.object(agent._matching, "match_all", return_value=[(market, sharp, 95.0)]):
            req = AgentRequest(
                sector="nba",
                params={
                    "kalshi_markets": [market],
                    "sharp_odds": [sharp],
                    "blended_preds": blended_preds,
                },
            )
            resp = await agent(req)

        assert len(resp.data) >= 1
        gap = resp.data[0]
        assert gap.blended_true_prob == pytest.approx(0.62, abs=0.001)

    @pytest.mark.asyncio
    async def test_publishes_to_bus(self):
        bus = AgentBus()
        received = []

        async def handler(msg):
            received.append(msg)

        bus.subscribe("ev.gaps.nba", handler)
        agent = EVGapAgent()
        agent.attach_bus(bus)

        market = make_market(yes_price=0.45)
        sharp = make_sharp(prob_a=0.58)

        with patch.object(agent._matching, "match_all", return_value=[(market, sharp, 95.0)]):
            await agent(AgentRequest(sector="nba", params={"kalshi_markets": [market], "sharp_odds": [sharp]}))

        assert len(received) == 1
        assert received[0].topic == "ev.gaps.nba"


# ---------------------------------------------------------------------------
# EloModelAgent
# ---------------------------------------------------------------------------

class TestEloModelAgent:
    def test_default_rating(self):
        agent = EloModelAgent()
        assert agent.get_rating("nba", "unknown_team") == 1500.0

    def test_seed_ratings(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "evmax.agents.models.base.STATE_DIR", tmp_path
        )
        agent = EloModelAgent()
        agent._state_path = tmp_path / "elo_state.json"
        agent.seed_ratings("nba", {"lakers": 1600.0, "celtics": 1550.0})
        assert agent.get_rating("nba", "lakers") == 1600.0

    def test_update_winner_gains_elo(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = EloModelAgent()
        agent._state_path = tmp_path / "elo_state.json"

        # Equal teams — winner should gain points
        agent.update("lakers", "celtics", score_a=110, score_b=100, sector="nba")
        assert agent.get_rating("nba", "lakers") > 1500.0
        assert agent.get_rating("nba", "celtics") < 1500.0

    def test_update_loser_loses_elo(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = EloModelAgent()
        agent._state_path = tmp_path / "elo_state.json"

        agent.update("lakers", "celtics", score_a=90, score_b=110, sector="nba")
        assert agent.get_rating("nba", "lakers") < 1500.0
        assert agent.get_rating("nba", "celtics") > 1500.0

    def test_zero_sum_ratings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = EloModelAgent()
        agent._state_path = tmp_path / "elo_state.json"

        for _ in range(20):
            agent.update("lakers", "celtics", 110, 100, "nba")
            agent.update("celtics", "lakers", 110, 100, "nba")

        # Sum of ratings stays close to 3000 (two teams, each started at 1500)
        total = agent.get_rating("nba", "lakers") + agent.get_rating("nba", "celtics")
        assert abs(total - 3000.0) < 5.0

    @pytest.mark.asyncio
    async def test_predict_pair_returns_prediction(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = EloModelAgent()
        agent._state_path = tmp_path / "elo_state.json"

        market = make_market()
        sharp = make_sharp()

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert abs(pred.true_prob_a + pred.true_prob_b - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_predict_pair_probs_sum_to_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = EloModelAgent()
        agent._state_path = tmp_path / "elo_state.json"

        market = make_market(sector="soccer")
        market = market.model_copy(update={"sector": "soccer"})
        sharp = make_sharp(sector="soccer")
        sharp = sharp.model_copy(update={"sector": "soccer", "true_prob_draw": 0.25, "true_prob_a": 0.45, "true_prob_b": 0.30})

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        total = pred.true_prob_a + pred.true_prob_b + (pred.true_prob_draw or 0.0)
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# FormModelAgent
# ---------------------------------------------------------------------------

class TestFormModelAgent:
    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        market = make_market()
        sharp = make_sharp()
        pred = await agent.predict_pair(market, sharp)
        assert pred is None  # no game records

    @pytest.mark.asyncio
    async def test_prediction_after_seeding(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        results = [
            {"date": f"2026-01-{i:02d}", "home": "lakers", "away": "celtics", "score_home": 110, "score_away": 100}
            for i in range(1, 8)
        ]
        agent.seed_results("nba", results)

        market = make_market()
        sharp = make_sharp()
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a > 0.5   # lakers won all seeded games

    def test_update_adds_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        agent.update("lakers", "celtics", 112, 108, "nba", "2026-01-15")
        recs = agent._team_records("nba", "lakers")
        assert len(recs) == 1
        assert recs[0].won is True


# ---------------------------------------------------------------------------
# PoissonModelAgent
# ---------------------------------------------------------------------------

class TestPoissonModelAgent:
    def test_poisson_pmf(self):
        # P(X=0 | λ=1.5) ≈ 0.2231
        import math
        expected = math.exp(-1.5)
        assert abs(_poisson_pmf(1.5, 0) - expected) < 1e-6

    def test_score_matrix_sums_to_one(self):
        matrix = _score_matrix(1.5, 1.2, max_g=8, rho=0.0)
        total = sum(p for row in matrix for p in row)
        assert abs(total - 1.0) < 0.05  # DC correction changes total slightly

    def test_win_draw_probs_sum_to_one(self):
        matrix = _score_matrix(1.5, 1.2, max_g=8)
        h, d, a = _win_draw_probs(matrix)
        assert abs(h + d + a - 1.0) < 1e-6

    def test_stronger_team_has_higher_win_prob(self):
        matrix_even = _score_matrix(1.5, 1.5, max_g=8)
        h_even, _, _ = _win_draw_probs(matrix_even)

        matrix_home_strong = _score_matrix(2.5, 1.0, max_g=8)
        h_strong, _, _ = _win_draw_probs(matrix_home_strong)

        assert h_strong > h_even

    @pytest.mark.asyncio
    async def test_predict_without_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = PoissonModelAgent()
        agent._state_path = tmp_path / "poisson_state.json"

        market = make_market(sector="soccer")
        market = market.model_copy(update={"sector": "soccer"})
        sharp = make_sharp(sector="soccer")

        pred = await agent.predict_pair(market, sharp)
        # Should still return something (league-average fallback)
        assert pred is not None
        assert pred.confidence < 0.5  # low confidence without data

    @pytest.mark.asyncio
    async def test_predict_with_seeded_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = PoissonModelAgent()
        agent._state_path = tmp_path / "poisson_state.json"

        agent.seed_team_stats(
            sector="soccer",
            team_stats={
                "manchester city": {"attack": 1.8, "defense": 0.6, "games": 20},
                "bournemouth": {"attack": 0.7, "defense": 1.4, "games": 20},
            },
            league_avg={"home": 1.55, "away": 1.15},
        )

        market = make_market(sector="soccer", team_home="manchester city", team_away="bournemouth")
        market = market.model_copy(update={"sector": "soccer"})
        sharp = make_sharp(team_a="manchester city", team_b="bournemouth", sector="soccer")

        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a > 0.5   # Man City strong attack vs weak Bournemouth defense

    @pytest.mark.asyncio
    async def test_nba_score_matrix_is_bucketed(self, tmp_path, monkeypatch):
        """Regression for BUG-4: NBA lambdas (~113 pts/game) must be bucketed
        before going into the score matrix, otherwise the Poisson mass lives
        far above MAX_SCORE=25 and the distribution collapses to a degenerate
        region where small λ differences produce overconfident predictions."""
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = PoissonModelAgent()
        agent._state_path = tmp_path / "poisson_state.json"

        # Seed a modest favorite: 5% stronger offense, 5% weaker defense.
        agent.seed_team_stats(
            sector="nba",
            team_stats={
                "lakers": {"attack": 1.05, "defense": 0.95, "games": 20},
                "celtics": {"attack": 1.00, "defense": 1.00, "games": 20},
            },
            league_avg={"home": 113.5, "away": 111.0},
        )

        market = make_market(sector="nba", team_home="lakers", team_away="celtics")
        sharp = make_sharp(team_a="lakers", team_b="celtics", sector="nba")
        pred = await agent.predict_pair(market, sharp)

        assert pred is not None
        # A ~5% rating edge should produce a moderate favorite, not a lock.
        # Without bucketing the matrix was truncated and this agent returned
        # values near 0.5 or pushed into extremes depending on λ alignment;
        # with bucketing we expect a sane favorite somewhere in (0.52, 0.80).
        assert 0.52 < pred.true_prob_a < 0.80
        # (pre-existing: draw merge in non-soccer path leaves a small residual)
        assert pred.true_prob_a + pred.true_prob_b == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# EnsembleModelAgent
# ---------------------------------------------------------------------------

class TestEnsembleModelAgent:
    @pytest.mark.asyncio
    async def test_blends_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        elo = EloModelAgent()
        elo._state_path = tmp_path / "elo_state.json"
        form = FormModelAgent()
        form._state_path = tmp_path / "form_state.json"
        poisson = PoissonModelAgent()
        poisson._state_path = tmp_path / "poisson_state.json"

        ensemble = EnsembleModelAgent(models=[elo, form, poisson], sharp_weight=0.40)

        market = make_market()
        sharp = make_sharp()

        req = AgentRequest(
            sector="nba",
            params={"pairs": [{"market": market, "sharp": sharp}]},
        )
        resp = await ensemble(req)
        assert resp.status == "ok"
        # Even without model data, falls back to sharp probs
        blended = resp.data
        assert sharp.event_id in blended
        blend = blended[sharp.event_id]
        assert abs(blend.true_prob_a + blend.true_prob_b - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_empty_pairs_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        ensemble = EnsembleModelAgent(models=[], sharp_weight=0.40)
        resp = await ensemble(AgentRequest(sector="nba", params={"pairs": []}))
        assert resp.data == {}
