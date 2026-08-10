"""Tests for the agent framework."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evmax.agents.base import Agent, AgentBus, AgentMessage, AgentRequest, AgentResponse, AgentStatus
from evmax.agents.odds.ev_gap_agent import EVGapAgent, EVGap
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent, GameRecord
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
    # Simulate a real liquid book with ~2% vig so spread_pct > 0 and the
    # empty-orderbook filter doesn't drop these synthetic test markets.
    no_price = round(min(0.99, max(0.01, (1.0 - yes_price) + 0.02)), 4)
    return PredictionMarket(
        id=f"kalshi:{team_home}_vs_{team_away}",
        source=MarketSource.kalshi,
        sector=sector,
        market_type=market_type,
        title=f"{team_home} vs {team_away}",
        ticker=f"KXNBAGAME-TEST-{team_home.upper()}",
        yes_price=yes_price,
        no_price=no_price,
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
    async def test_maker_only_gap_surfaced(self):
        """A coin-flip edge the taker fee eats but the maker fee spares is
        surfaced as a maker-only gap: maker_only=True, maker_ev_pct clears the
        floor, taker ev_pct is below it, and Kelly is zero (fill-contingent —
        not crossable at the ask, promoted via `agents fill` once it fills)."""
        agent = EVGapAgent()
        # yes=0.50, true=0.525: taker nets ~1.4% (<2%), maker nets ~4% (>=2%).
        market = make_market(yes_price=0.50, yes_team="lakers")
        sharp = make_sharp(prob_a=0.525, prob_b=0.475, team_a="lakers")

        with patch.object(agent._matching, "match_all", return_value=[(market, sharp, 95.0)]):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = await agent(req)

        assert resp.status == "ok"
        assert len(resp.data) >= 1
        gap: EVGap = resp.data[0]
        assert gap.maker_only is True
        assert gap.ev_pct < 0.02
        assert gap.maker_ev_pct is not None and gap.maker_ev_pct >= 0.02
        assert gap.kelly_fraction == 0.0

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

    @pytest.mark.asyncio
    async def test_dead_orderbook_floor_filtered(self):
        """Markets where YES ask has collapsed to the 1¢ floor are dead — all
        bids pulled. Without this filter, the (sharp_prob / 0.01) − 1 EV
        calculation produces bogus 5000-7000% edges (regression for the 3
        $0.01 tennis rows surfaced in commit 3de2c26 diagnostic)."""
        agent = EVGapAgent()
        # YES at 1¢ floor; sharp says the team should be ~58% — would
        # nominally be a 5700% edge if the filter didn't fire.
        market = make_market(yes_price=0.01, yes_team="lakers")
        sharp = make_sharp(prob_a=0.58, team_a="lakers")

        with patch.object(agent._matching, "match_all", return_value=[(market, sharp, 95.0)]):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = await agent(req)

        assert len(resp.data) == 0

    @pytest.mark.asyncio
    async def test_dead_orderbook_ceiling_filtered(self):
        """Symmetric case: YES ask at 99¢ ceiling means all asks pulled."""
        agent = EVGapAgent()
        # YES at 99¢ ceiling with sharp at 50% — wouldn't actually trigger
        # positive EV but verifies the filter short-circuits regardless.
        market = make_market(yes_price=0.99, yes_team="lakers")
        sharp = make_sharp(prob_a=0.50, prob_b=0.50, team_a="lakers")

        with patch.object(agent._matching, "match_all", return_value=[(market, sharp, 95.0)]):
            req = AgentRequest(
                sector="nba",
                params={"kalshi_markets": [market], "sharp_odds": [sharp]},
            )
            resp = await agent(req)

        assert len(resp.data) == 0

    def test_prop_dead_orderbook_floor_filtered(self):
        """Same dead-orderbook symptom as the game-market tests above, but on
        the player-prop evaluator (_evaluate_prop_pair lacked the guard
        entirely). A pitcher-strikeout prop whose outcome is already decided
        in-game (pulled from the game) collapses to Kalshi's 1¢ floor; the
        anchor-pricing model has no live in-game state and kept projecting
        the pre-game probability, producing bogus 9000%+ 'edges' on markets
        that are effectively closed."""
        agent = EVGapAgent()
        prop_market = PredictionMarket(
            id="kalshi:CAMERON-2K",
            source=MarketSource.kalshi,
            sector="baseball",
            market_type=MarketType.player_prop,
            title="Noah Cameron 2+ Strikeouts",
            ticker="KXMLBPROPS-CAMERON",
            yes_price=0.01,
            no_price=0.99,
            volume_usd=800.0,
            player_name="Noah Cameron",
            stat_type="strikeouts",
            threshold=1.5,
        )
        prop_sharp = SharpOdds(
            event_id="baseball::2026-07-19::prop::Noah Cameron::strikeouts::1.5",
            book=SharpBook.pinnacle,
            sector="baseball",
            outcome_a_label="over",
            outcome_b_label="under",
            outcome_a_decimal=1.0,
            outcome_b_decimal=1.0,
            true_prob_a=0.0,
            true_prob_b=0.0,
            true_prob_over=0.93,
            true_prob_under=0.07,
            total_line=1.5,
            margin=0.0,
            prop_player_name="Noah Cameron",
            prop_stat_type="strikeouts",
        )

        gap = agent._evaluate_prop_pair(
            prop_market, prop_sharp, confidence=95.0, sector="baseball", injuries=None
        )
        assert gap is None

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

        # _evaluate_prop_pair was re-enabled 2026-05-10 once anchor pricing
        # replaced the L15 model. The Pinnacle-anchored sharp prob (0.55)
        # against a 0.48 Kalshi quote = ~14% EV, well above threshold.
        gap_no_inj = agent._evaluate_prop_pair(
            prop_market, prop_sharp, confidence=95.0, sector="nba", injuries=None
        )
        gap_with_inj = agent._evaluate_prop_pair(
            prop_market, prop_sharp, confidence=95.0, sector="nba", injuries=injuries
        )

        # Both should produce EVGaps now. Injury boost should raise the
        # blended prob since the OUT teammate (Davis) frees up usage.
        assert gap_no_inj is not None
        assert gap_with_inj is not None
        assert gap_no_inj.sharp_true_prob == pytest.approx(0.55)
        assert gap_no_inj.blended_true_prob == pytest.approx(0.55)
        assert gap_with_inj.blended_true_prob > gap_no_inj.blended_true_prob
        assert "inj" in (gap_with_inj.model_sources or "")

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

    # ------------------------------------------------------------------
    # ARCH-7: _resolve_yes_via_market_teams price-fallback
    # ------------------------------------------------------------------

    def _price_fallback(self, ask: float, prob_a: float, prob_b: float):
        """Invoke the price-based branch by passing a YES abbrev that matches neither team."""
        market = make_market(yes_price=ask, team_home="lakers", team_away="celtics", yes_team="lakers")
        sharp = make_sharp(prob_a=prob_a, prob_b=prob_b, team_a="lakers", team_b="celtics")
        return EVGapAgent._resolve_yes_via_market_teams(
            yes_team_norm="xxx",  # matches neither home nor away → forces price fallback
            market=market,
            sharp=sharp,
            sector="nba",
        )

    def test_price_fallback_aligns_when_one_side_clearly_closer(self):
        # ask 0.60 vs (a=0.58, b=0.42): dist_a=0.02, dist_b=0.18 → align to a
        result = self._price_fallback(ask=0.60, prob_a=0.58, prob_b=0.42)
        assert result == (0.58, False)

    def test_price_fallback_aligns_to_b_when_b_closer(self):
        # ask 0.40 vs (a=0.58, b=0.42): dist_a=0.18, dist_b=0.02 → align to b
        result = self._price_fallback(ask=0.40, prob_a=0.58, prob_b=0.42)
        assert result == (0.42, True)

    def test_price_fallback_rejects_near_coin_flip(self):
        """Regression for ARCH-7: near-50/50 markets must not be force-aligned.

        At prob_a=0.52, prob_b=0.48, ask=0.50 both distances are 0.02 — the
        old heuristic would silently pick outcome_a; the new logic rejects it.
        """
        result = self._price_fallback(ask=0.50, prob_a=0.52, prob_b=0.48)
        assert result is None

    def test_price_fallback_rejects_when_gap_too_small(self):
        # dist_a=0.03, dist_b=0.07: closer tight enough but gap (0.04) < 0.05 → reject
        result = self._price_fallback(ask=0.52, prob_a=0.55, prob_b=0.45)
        assert result is None

    def test_price_fallback_rejects_when_closer_side_too_far(self):
        # dist_a=0.05 exceeds 0.04 ceiling even though gap is large → reject
        result = self._price_fallback(ask=0.53, prob_a=0.58, prob_b=0.42)
        assert result is None


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

    @pytest.mark.asyncio
    async def test_soccer_draw_strength_dependent(self, tmp_path, monkeypatch):
        """Soccer draw allocation should scale with team-strength gap:
        even matches get ~26% draw, mismatched get much less."""
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = EloModelAgent()
        agent._state_path = tmp_path / "elo_state.json"

        market = make_market(
            sector="soccer", team_home="man city", team_away="burnley",
            yes_team="man city",
        )
        market = market.model_copy(update={"sector": "soccer"})
        sharp = make_sharp(
            sector="soccer", team_a="man city", team_b="burnley",
            event_id="soccer::2026-04-15::man_city_vs_burnley",
        )
        sharp = sharp.model_copy(update={
            "sector": "soccer", "true_prob_draw": 0.25,
            "true_prob_a": 0.40, "true_prob_b": 0.35,
        })

        # Even match: both teams at default 1500
        pred_even = await agent.predict_pair(market, sharp)
        assert pred_even is not None
        assert pred_even.true_prob_draw is not None
        # Even match draw should be ~26%
        assert pred_even.true_prob_draw > 0.24

        # Mismatched: seed one team much higher
        agent.seed_ratings("soccer", {"man city": 1700, "burnley": 1300})
        pred_mis = await agent.predict_pair(market, sharp)
        assert pred_mis is not None
        assert pred_mis.true_prob_draw is not None
        # Mismatched draw should be much lower
        assert pred_mis.true_prob_draw < 0.15
        # And draw should be lower than even match
        assert pred_mis.true_prob_draw < pred_even.true_prob_draw

    @pytest.mark.asyncio
    async def test_soccer_draw_never_exceeds_30pct(self, tmp_path, monkeypatch):
        """Draw probability caps at 30% even for perfectly even teams.

        Retuned 2026-04-23 after walk-forward calibration showed all three
        stat models underpredicting draws by 2.5–7.9pp. Formula:
            draw_prob = max(0.10, 0.30 - 0.40 * gap * gap)
        """
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = EloModelAgent()
        agent._state_path = tmp_path / "elo_state.json"

        market = make_market(sector="soccer")
        market = market.model_copy(update={"sector": "soccer"})
        sharp = make_sharp(sector="soccer")
        sharp = sharp.model_copy(update={
            "sector": "soccer", "true_prob_draw": 0.25,
            "true_prob_a": 0.40, "true_prob_b": 0.35,
        })
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_draw is not None
        assert pred.true_prob_draw <= 0.31  # 30% + small rounding margin


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
        from datetime import date, timedelta
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        today = date.today()
        results = [
            {"date": (today - timedelta(days=i)).isoformat(),
             "home": "lakers", "away": "celtics", "score_home": 110, "score_away": 100}
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

    @pytest.mark.asyncio
    async def test_stale_records_return_none(self, tmp_path, monkeypatch):
        """When the most recent record is > STALE_DAYS old relative to the
        market's event_date, the form model skips."""
        from datetime import date, timedelta
        from evmax.agents.models import form_agent as fa

        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        # make_market() uses event_date=2026-03-15. Seed records from >
        # STALE_DAYS before that so the guard fires regardless of wall clock.
        market_day = date(2026, 3, 15)
        old_day = market_day - timedelta(days=fa.STALE_DAYS + 30)
        results = [
            {"date": (old_day - timedelta(days=i)).isoformat(),
             "home": "lakers", "away": "celtics", "score_home": 110, "score_away": 100}
            for i in range(0, 7)
        ]
        agent.seed_results("nba", results)

        pred = await agent.predict_pair(make_market(), make_sharp())
        assert pred is None, "Form should skip when data is older than STALE_DAYS"

    def test_is_stale_uses_reference_date(self):
        """Staleness check against a specified reference date (walk-forward replay)."""
        from datetime import date, timedelta
        from evmax.agents.models import form_agent as fa

        # A record from 2025-06-01, referenced against 2025-06-15 → not stale
        rec = [GameRecord(date="2025-06-01", won=True, opp="celtics", home=True)]
        assert not FormModelAgent._is_stale(rec, reference=date(2025, 6, 15))

        # Same record, referenced 90 days later → stale
        assert FormModelAgent._is_stale(rec, reference=date(2025, 9, 1))

        # Exactly STALE_DAYS old → not stale (boundary); one day past → stale
        today = date(2026, 4, 1)
        boundary = [GameRecord(
            date=(today - timedelta(days=fa.STALE_DAYS)).isoformat(),
            won=True, opp="celtics", home=True,
        )]
        assert not FormModelAgent._is_stale(boundary, reference=today)

        past = [GameRecord(
            date=(today - timedelta(days=fa.STALE_DAYS + 1)).isoformat(),
            won=True, opp="celtics", home=True,
        )]
        assert FormModelAgent._is_stale(past, reference=today)

    @pytest.mark.asyncio
    async def test_nfl_fresh_records_predicts_with_home_adj(self, tmp_path, monkeypatch):
        """Characterization: with fresh NFL records the form model fires and the
        home side (team_a) carries the NFL home probability bump (HOME_ADJ['nfl']
        = 0.03). team_a here won every seeded game, so prob_a > 0.5."""
        from datetime import date, timedelta
        from evmax.agents.models import form_agent as fa

        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        game_day = date(2025, 11, 16)  # mid-NFL-season
        # NFL plays weekly — space records 7 days apart so they stay within
        # STALE_DAYS of the game date.
        results = [
            {"date": (game_day - timedelta(days=7 * i)).isoformat(),
             "home": "chiefs", "away": "broncos", "score_home": 27, "score_away": 17}
            for i in range(1, 6)
        ]
        agent.seed_results("nfl", results)

        market = make_market(sector="nfl", team_home="chiefs", team_away="broncos")
        market.event_date = datetime(2025, 11, 16, 20, 0, tzinfo=timezone.utc)
        sharp = make_sharp(
            event_id="nfl::2025-11-16::chiefs_vs_broncos",
            team_a="chiefs", team_b="broncos", sector="nfl",
        )
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a > 0.5  # chiefs won every seeded game

    @pytest.mark.asyncio
    async def test_nfl_prior_season_records_return_none(self, tmp_path, monkeypatch):
        """Characterization: a September game whose only records are from the
        prior February (> STALE_DAYS) must skip — cross-season contamination
        guard, the same one that protected the WNBA opener."""
        from datetime import date, timedelta
        from evmax.agents.models import form_agent as fa

        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        # Game is Sep 2025; records are from Feb 2025 (prior season playoffs),
        # ~200+ days stale relative to the game date.
        feb_day = date(2025, 2, 2)
        results = [
            {"date": (feb_day - timedelta(days=7 * i)).isoformat(),
             "home": "chiefs", "away": "broncos", "score_home": 24, "score_away": 21}
            for i in range(0, 5)
        ]
        agent.seed_results("nfl", results)

        market = make_market(sector="nfl", team_home="chiefs", team_away="broncos")
        market.event_date = datetime(2025, 9, 7, 20, 0, tzinfo=timezone.utc)
        sharp = make_sharp(
            event_id="nfl::2025-09-07::chiefs_vs_broncos",
            team_a="chiefs", team_b="broncos", sector="nfl",
        )
        pred = await agent.predict_pair(market, sharp)
        assert pred is None, "Prior-season NFL form must skip via STALE_DAYS guard"

    def test_update_records_draw(self, tmp_path, monkeypatch):
        """Soccer draws must persist drew=True on both sides, not won=False."""
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        agent.update("arsenal", "chelsea", 1, 1, "soccer", "2026-02-10")
        arsenal = agent._team_records("soccer", "arsenal")
        chelsea = agent._team_records("soccer", "chelsea")

        assert len(arsenal) == 1 and len(chelsea) == 1
        assert arsenal[0].drew is True and arsenal[0].won is False
        assert chelsea[0].drew is True and chelsea[0].won is False

    def test_form_rate_points_per_game_with_shrinkage(self):
        """Form rate is (W*3 + D*1 + L*0)/(3*N) shrunk toward 0.5 with PRIOR_N.

        At WINDOW games (10 if seeded), shrinkage = 10/(10+6) = 0.625.
        At MIN_GAMES (3), shrinkage = 3/(3+6) = 0.333 — strongly pulled to 0.5.
        """
        from evmax.agents.models import form_agent as fa

        agent = FormModelAgent()
        d = "2026-03-01"

        # 3 wins, raw = 1.0, shrinkage 3/9 = 0.333 → 1.0*0.333 + 0.5*0.667 = 0.667
        recs_wins = [GameRecord(date=d, won=True, opp="x", home=True) for _ in range(3)]
        assert abs(agent._form_rate(recs_wins) - (2.0 / 3.0)) < 1e-9

        # 3 draws, raw = 1/3, shrinkage 0.333 → 0.333*0.333 + 0.5*0.667 ≈ 0.444
        recs_draws = [
            GameRecord(date=d, won=False, opp="x", home=True, drew=True) for _ in range(3)
        ]
        assert abs(agent._form_rate(recs_draws) - ((1.0 / 3.0) * (1.0 / 3.0) + 0.5 * (2.0 / 3.0))) < 1e-9

        # 3 losses, raw = 0.0 → 0 + 0.5*0.667 = 0.333
        recs_losses = [GameRecord(date=d, won=False, opp="x", home=True) for _ in range(3)]
        assert abs(agent._form_rate(recs_losses) - (1.0 / 3.0)) < 1e-9

    def test_form_rate_shrinkage_weaker_at_larger_n(self):
        """Larger N → weaker pull toward 0.5 (more trust in the data)."""
        agent = FormModelAgent()
        d = "2026-03-01"
        short_streak = [GameRecord(date=d, won=True, opp="x", home=True) for _ in range(3)]
        long_streak = [GameRecord(date=d, won=True, opp="x", home=True) for _ in range(10)]
        # Decay is uniform when all dates equal → raw is still 1.0 for both.
        # Shrinkage differs: 3/9=0.333 vs 10/16=0.625.
        short_rate = agent._form_rate(short_streak)
        long_rate = agent._form_rate(long_streak)
        assert long_rate > short_rate  # larger N → closer to 1.0

    def test_form_rate_distinguishes_draw_from_loss(self):
        """A drawing team must rate higher than an equally-sized losing team.
        This is the specific bug the points-per-game switch fixes."""
        agent = FormModelAgent()
        d = "2026-03-01"
        drawing = [GameRecord(date=d, won=False, opp="x", home=True, drew=True) for _ in range(5)]
        losing = [GameRecord(date=d, won=False, opp="x", home=True, drew=False) for _ in range(5)]
        assert agent._form_rate(drawing) > agent._form_rate(losing)

    def test_legacy_records_backward_compatible(self, tmp_path, monkeypatch):
        """Pre-migration state dicts (no `drew` key) still deserialize and
        behave exactly as before — wins → 1.0, non-wins → 0.0."""
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = FormModelAgent()
        agent._state_path = tmp_path / "form_state.json"

        agent._state["nba"] = {
            "lakers": [
                {"date": "2026-02-01", "won": True, "opp": "celtics", "home": True},
                {"date": "2026-02-03", "won": False, "opp": "bulls", "home": False},
                {"date": "2026-02-05", "won": True, "opp": "heat", "home": True},
            ]
        }
        recs = agent._team_records("nba", "lakers")
        # drew defaulted to False — wins score 1.0, losses 0.0 (NBA still binary)
        assert all(r.drew is False for r in recs)
        rate = agent._form_rate(recs)
        assert 0.5 < rate < 0.8  # 2 wins out of 3, exponentially weighted


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

    def test_poisson_is_football_only(self):
        """Poisson serves only the two football sectors — club `soccer` and
        national-team `worldcup`, both genuine low-count goal processes.
        Membership in SUPPORTED_SECTORS — not the ensemble override weight — is
        what keeps it out of other sectors: an unlisted model falls back to its
        0.30 class weight in the blend (ensemble_agent.py:469), so
        baseball/ncaab/nba/wnba/nfl must be excluded here, not merely zeroed
        downstream."""
        from evmax.agents.models.poisson_agent import SUPPORTED_SECTORS
        assert SUPPORTED_SECTORS == {"soccer", "worldcup"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector", ["baseball", "ncaab", "nba", "wnba", "nfl"])
    async def test_unsupported_sector_predict_pair_returns_none(
        self, sector, tmp_path, monkeypatch
    ):
        """Even with teams seeded in state, predict_pair must short-circuit to
        None for any non-soccer sector so poisson never enters model_preds."""
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)
        agent = PoissonModelAgent()
        agent._state_path = tmp_path / "poisson_state.json"
        agent.seed_team_stats(
            sector=sector,
            team_stats={
                "team a": {"attack": 1.1, "defense": 0.9, "games": 30},
                "team b": {"attack": 1.0, "defense": 1.0, "games": 30},
            },
            league_avg={"home": 4.65, "away": 4.37},
        )
        market = make_market(sector=sector, team_home="team a", team_away="team b")
        market = market.model_copy(update={"sector": sector})
        sharp = make_sharp(team_a="team a", team_b="team b", sector=sector)

        pred = await agent.predict_pair(market, sharp)
        assert pred is None

    # NOTE: the former BUG-4 regression (test_nba_score_matrix_is_bucketed) was
    # removed when poisson became soccer-only (2026-06-01). It routed through
    # predict_pair("nba"), which now returns None; NBA exclusion is covered by
    # test_unsupported_sector_predict_pair_returns_none. The high-λ bucketing
    # path it guarded is only reachable for soccer, where BUCKET_SIZE == 1 (a
    # no-op), so the scenario is no longer live.

    def test_shrink_pulls_extreme_ratios_toward_one(self):
        """Bayesian shrinkage on attack/defense ratios: extreme values at low
        sample size get pulled toward 1.0; many-game teams keep their signal."""
        from evmax.agents.models.poisson_agent import _shrink, POISSON_PRIOR_GAMES

        # Cold-start: 3 games + raw 2.0 → closer to 1.0 than to 2.0.
        shrunk_cold = _shrink(2.0, games=3)
        assert abs(shrunk_cold - 1.0) < abs(shrunk_cold - 2.0)

        # Mid-season: 15 games + raw 2.0 → closer to 2.0 than to 1.0.
        shrunk_mid = _shrink(2.0, games=15)
        assert abs(shrunk_mid - 2.0) < abs(shrunk_mid - 1.0)

        # Zero games returns baseline (avoids div-by-zero, mimics "no data").
        assert _shrink(3.0, games=0) == 1.0

        # Monotonic: more games = less shrinkage toward 1.0.
        assert _shrink(2.0, games=30) > _shrink(2.0, games=10)

    @pytest.mark.asyncio
    async def test_shrinkage_compresses_predictions_at_low_sample(self, tmp_path, monkeypatch):
        """A 3-game team with raw 2.0 attack should not produce the same
        prediction as a 30-game team with the same rating — shrinkage should
        pull the low-N team's P(home) closer to 0.5."""
        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)

        def _predict(home_games: int, away_games: int) -> float:
            agent = PoissonModelAgent()
            agent._state_path = tmp_path / f"poisson_{home_games}_{away_games}.json"
            agent.seed_team_stats(
                sector="soccer",
                team_stats={
                    "strong": {"attack": 2.0, "defense": 0.5, "games": home_games},
                    "weak": {"attack": 0.5, "defense": 2.0, "games": away_games},
                },
                league_avg={"home": 1.55, "away": 1.15},
            )
            lam_h, lam_a = agent._expected_goals("soccer", "strong", "weak")
            return lam_h, lam_a

        lam_h_cold, lam_a_cold = _predict(3, 3)
        lam_h_mid, lam_a_mid = _predict(30, 30)

        # Same raw ratios but low-N should be much closer to league average.
        # League avg is 1.55/1.15, raw λ_h = 2.0*2.0*1.55 = 6.20.
        assert lam_h_mid > lam_h_cold
        assert lam_a_mid < lam_a_cold

    def test_update_is_opponent_adjusted(self):
        """update() credits goals relative to the OPPONENT's strength, not a
        flat league average.

        This is the fix for the World Cup underdog-bias bug: the old update
        divided goals only by the league average, so scoring 3 against a stingy
        defense earned the same attack credit as scoring 3 against a leaky one.
        On a balanced league schedule that washes out, but national-team
        schedules are split by confederation strength — minnows that beat up on
        weak opponents (Japan/Russia/Iran in AFC/friendly play) were ranked
        above Brazil/France, inflating underdog win probs and +EV flags.
        """
        agent = PoissonModelAgent()
        agent._state = {
            "worldcup": {
                "league_avg": {"home": 1.30, "away": 1.30},
                "teams": {
                    # Stable, high-N reference opponents with known defenses.
                    "stingy": {"attack": 1.0, "defense": 0.4, "games": 50},
                    "leaky": {"attack": 1.0, "defense": 1.8, "games": 50},
                },
            }
        }
        # Two fresh teams each score 3 at home — one vs the stingy defense,
        # one vs the leaky defense. Identical scorelines, different opponents.
        agent.update("beat_stingy", "stingy", 3, 0, "worldcup")
        agent.update("beat_leaky", "leaky", 3, 0, "worldcup")

        teams = agent._state["worldcup"]["teams"]
        # Scoring 3 on a stingy defense implies a far stronger attack than the
        # same 3 on a leaky one. The OLD (SOS-blind) update made these EQUAL.
        assert teams["beat_stingy"]["attack"] > teams["beat_leaky"]["attack"]

    def test_update_clamps_extreme_single_game_implied_rating(self):
        """A freak result against an extreme opponent can't blow past the clamp.

        Scoring 4 on a def≈0.3 elite defense implies attack ≈ 4/(1.30·0.3) ≈ 10;
        IMPLIED_RATING_CLAMP keeps one game from dominating a team's rating.
        """
        from evmax.agents.models.poisson_agent import IMPLIED_RATING_CLAMP

        agent = PoissonModelAgent()
        agent._state = {
            "worldcup": {
                "league_avg": {"home": 1.30, "away": 1.30},
                "teams": {"elite_def": {"attack": 1.0, "defense": 0.3, "games": 50}},
            }
        }
        agent.update("freak", "elite_def", 4, 0, "worldcup")
        # First game → rating == the single-game implied value, capped at hi.
        assert agent._state["worldcup"]["teams"]["freak"]["attack"] <= IMPLIED_RATING_CLAMP[1]


# ---------------------------------------------------------------------------
# EnsembleModelAgent
# ---------------------------------------------------------------------------

class TestEnsembleModelAgent:
    @pytest.mark.asyncio
    async def test_per_event_sharp_weight_override(self, tmp_path, monkeypatch):
        """`sharp_weight_by_event` per-event override wins over the sector default.

        Used for the soccer tier system: top-5 European leagues pass 0.85
        here and secondary leagues pass 0.40.
        """
        from evmax.agents.models.base import ModelAgent, ModelAgentPrediction

        monkeypatch.setattr("evmax.agents.models.base.STATE_DIR", tmp_path)

        class _StubModel(ModelAgent):
            name = "stub"
            weight = 1.0

            def load_state(self) -> None:
                self._state = {}

            def save_state(self) -> None:
                pass

            async def update(self, *args, **kwargs):
                return None

            async def predict_pair(self, market, sharp_odds):
                # Keep model/sharp disagreement at 0.05 so the base weight
                # passes through without the disagreement bump saturating.
                return ModelAgentPrediction(
                    event_id=sharp_odds.event_id,
                    model_name=self.name,
                    true_prob_a=0.50,
                    true_prob_b=0.50,
                    true_prob_draw=None,
                    confidence=0.80,
                    weight=self.weight,
                    sample_size=30,
                    notes="",
                )

        # Route the stub through the soccer override table at w=1.0 so it
        # actually contributes (override would otherwise zero it out).
        monkeypatch.setitem(
            EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["soccer"], "stub", 1.0,
        )

        stub = _StubModel()
        ensemble = EnsembleModelAgent(models=[stub], sharp_weight=0.40)

        market = make_market(sector="soccer")
        sharp_top = make_sharp(event_id="ev_topflight", sector="soccer",
                               prob_a=0.55, prob_b=0.45)
        sharp_sec = make_sharp(event_id="ev_secondary", sector="soccer",
                               prob_a=0.55, prob_b=0.45)

        pairs = [
            {"market": market, "sharp": sharp_top},
            {"market": market, "sharp": sharp_sec},
        ]

        req = AgentRequest(
            sector="soccer",
            params={
                "pairs": pairs,
                "sharp_weight": 0.40,
                "sharp_weight_by_event": {"ev_topflight": 0.85},
            },
        )
        resp = await ensemble(req)
        blended = resp.data

        # Topflight event used 0.85 → prob_a closer to sharp's 0.55.
        # Secondary event used 0.40 → prob_a closer to model's 0.50.
        topflight = blended["ev_topflight"].true_prob_a
        secondary = blended["ev_secondary"].true_prob_a
        assert topflight > secondary, (
            f"Higher sharp_weight should push blend toward sharp's 0.55 "
            f"(topflight={topflight:.3f}, secondary={secondary:.3f})"
        )

    def test_disagreement_sharp_weight_below_threshold(self):
        """Small model/sharp gaps leave sharp_weight untouched."""
        w = EnsembleModelAgent._disagreement_sharp_weight(
            model_a=0.55, model_b=0.45, model_draw=None,
            sharp_a=0.60, sharp_b=0.40, sharp_draw=None,
            base_sharp_weight=0.40,
        )
        assert w == 0.40  # max gap = 0.05 ≤ 0.10 threshold

    def test_disagreement_sharp_weight_ramps_between_threshold_and_saturation(self):
        """At 0.20 gap (halfway between 0.10 threshold and 0.30 saturate), the
        adjusted weight should be halfway between base (0.40) and cap (0.95)."""
        w = EnsembleModelAgent._disagreement_sharp_weight(
            model_a=0.30, model_b=0.70, model_draw=None,
            sharp_a=0.50, sharp_b=0.50, sharp_draw=None,
            base_sharp_weight=0.40,
        )
        assert 0.65 < w < 0.70  # ~0.40 + 0.5 * (0.95 - 0.40) = 0.675

    def test_disagreement_sharp_weight_saturates_at_cap(self):
        """A 0.40 gap saturates — use the full cap."""
        w = EnsembleModelAgent._disagreement_sharp_weight(
            model_a=0.10, model_b=0.90, model_draw=None,
            sharp_a=0.50, sharp_b=0.50, sharp_draw=None,
            base_sharp_weight=0.40,
        )
        assert w == 0.95

    def test_disagreement_considers_draw_leg(self):
        """Gap on the draw leg should trigger the bump even when H/A agree."""
        w = EnsembleModelAgent._disagreement_sharp_weight(
            model_a=0.40, model_b=0.40, model_draw=0.20,
            sharp_a=0.40, sharp_b=0.40, sharp_draw=0.20,
            base_sharp_weight=0.40,
        )
        assert w == 0.40  # no gap anywhere
        w2 = EnsembleModelAgent._disagreement_sharp_weight(
            model_a=0.40, model_b=0.40, model_draw=0.20,
            sharp_a=0.40, sharp_b=0.30, sharp_draw=0.30,
            base_sharp_weight=0.40,
        )
        assert w2 > 0.40  # draw gap 0.10, b gap 0.10 — at threshold or just above

    def test_disagreement_soccer_override_lowers_threshold(self):
        """Soccer override starts the ramp at 0.04 instead of the 0.10 default.
        A 0.05 gap is below the global threshold but above the soccer one, so
        the soccer call must boost while the default call must not."""
        kwargs = dict(
            model_a=0.55, model_b=0.45, model_draw=None,
            sharp_a=0.60, sharp_b=0.40, sharp_draw=None,
            base_sharp_weight=0.40,
        )
        # Default sector: 0.05 gap < 0.10 threshold → no boost
        w_default = EnsembleModelAgent._disagreement_sharp_weight(**kwargs)
        assert w_default == 0.40
        # Soccer sector: 0.05 gap > 0.04 threshold → boost
        w_soccer = EnsembleModelAgent._disagreement_sharp_weight(**kwargs, sector="soccer")
        assert w_soccer > 0.40

    def test_disagreement_soccer_override_saturates_at_higher_cap(self):
        """At a 0.20 gap (well past the soccer 0.10 saturate_at), the weight
        should be at the soccer cap (1.00), not the global cap (0.95)."""
        w_soccer = EnsembleModelAgent._disagreement_sharp_weight(
            model_a=0.30, model_b=0.70, model_draw=None,
            sharp_a=0.50, sharp_b=0.50, sharp_draw=None,
            base_sharp_weight=0.40,
            sector="soccer",
        )
        assert w_soccer == pytest.approx(1.00)

    def test_disagreement_tennis_falls_through_to_default_ramp(self):
        """Tennis override REMOVED 2026-05-02 (see scripts/backtest_tennis_
        walkforward.py). A 0.025 gap that previously triggered the tennis-
        specific 0.02 threshold should now use the default 0.10 threshold
        and produce no boost — same behavior as soccer at this gap and as
        any unconfigured sector."""
        kwargs = dict(
            model_a=0.215, model_b=0.785, model_draw=None,
            sharp_a=0.190, sharp_b=0.810, sharp_draw=None,
            base_sharp_weight=0.85,
        )
        # Default: 0.025 < default 0.10 → no boost
        assert EnsembleModelAgent._disagreement_sharp_weight(**kwargs) == 0.85
        # Soccer override: 0.025 < soccer 0.04 → no boost
        assert (
            EnsembleModelAgent._disagreement_sharp_weight(**kwargs, sector="soccer")
            == 0.85
        )
        # Tennis (no override): now uses defaults → 0.025 < 0.10 → no boost
        assert (
            EnsembleModelAgent._disagreement_sharp_weight(**kwargs, sector="tennis")
            == 0.85
        )

    def test_tennis_not_in_disagreement_overrides(self):
        """Regression-lock: tennis should fall through to default ramp params.
        Walk-forward (n=2576) showed flat 0.85 beats sharp_only by Brier
        +0.40/1000 — keeping the previous override aggressively pulled the
        blend back to sharp on disagreements, wiping out that edge."""
        assert "tennis" not in EnsembleModelAgent.DISAGREEMENT_OVERRIDES

    def test_disagreement_unknown_sector_falls_through_to_defaults(self):
        """Sectors not in DISAGREEMENT_OVERRIDES use the global ramp params."""
        w = EnsembleModelAgent._disagreement_sharp_weight(
            model_a=0.55, model_b=0.45, model_draw=None,
            sharp_a=0.60, sharp_b=0.40, sharp_draw=None,
            base_sharp_weight=0.40,
            sector="nba",
        )
        assert w == 0.40  # 0.05 gap below global 0.10 threshold

    def test_disagreement_explicit_kwargs_beat_sector_override(self):
        """The A/B harness sweeps params explicitly — caller's non-default
        values must win over the per-sector defaults."""
        # Pass an explicit threshold of 0.20: a 0.05 gap should NOT trigger
        # even though the soccer override would (0.04 threshold).
        w = EnsembleModelAgent._disagreement_sharp_weight(
            model_a=0.55, model_b=0.45, model_draw=None,
            sharp_a=0.60, sharp_b=0.40, sharp_draw=None,
            base_sharp_weight=0.40,
            threshold=0.20,
            sector="soccer",
        )
        assert w == 0.40

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

    def test_soccer_weight_overrides_exist(self):
        """Ensure soccer has sector-specific weight overrides that boost
        Poisson (empirical draws) and reduce Elo/Form (fixed draw formula)."""
        overrides = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES.get("soccer")
        assert overrides is not None, "Soccer must have SECTOR_WEIGHT_OVERRIDES"
        assert overrides["poisson"] > overrides["elo"], (
            "Poisson should be weighted higher than Elo for soccer"
        )
        assert overrides["poisson"] > overrides["form"], (
            "Poisson should be weighted higher than Form for soccer"
        )
        # Poisson should be the dominant model for soccer
        assert overrides["poisson"] >= 0.40

    def test_wnba_weight_overrides_post_sweep(self):
        """Lock in the post-sweep WNBA blend (2026-05-14, scripts/sweep_wnba_weights.py).

        The sweep over 321 walk-forward 2025 games found that:
          - form was the worst standalone model (Brier 0.2303); every top-20
            combo had form=0
          - generic Elo (0.2227) was weaker than the two WNBA-specific models
            (~0.202 each), so WNBA-specific weight should dominate
          - poisson stays out (basketball isn't a Poisson process)

        Regression test ensures these properties hold so a future edit can't
        silently re-introduce form/poisson or starve the WNBA-specific stack
        without the sweep showing it's safe.
        """
        overrides = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES.get("wnba")
        assert overrides is not None, "WNBA must have SECTOR_WEIGHT_OVERRIDES"
        assert overrides["form"] == 0.0, (
            "form must be zeroed for WNBA — worst standalone model in sweep"
        )
        assert overrides["poisson"] == 0.0, (
            "poisson must stay zero for WNBA (basketball ≠ Poisson)"
        )
        wnba_specific = overrides["wnba_efficiency"] + overrides["wnba_possession_sim"]
        assert wnba_specific >= 0.80, (
            f"WNBA-specific stack should carry ≥80% of model weight, got {wnba_specific:.2f}"
        )
        assert wnba_specific > overrides["elo"], (
            "WNBA-specific stack should outweigh generic Elo"
        )

    def test_flb_correction_extreme_longshot(self):
        """FLB correction should suppress model inflation at extreme probs.

        Uses sector='nfl' because:
          - FLB is skipped for NBA/WNBA (their advanced models produce
            already-well-calibrated extremes)
          - tennis has sharp_weight=1.00 which makes FLB a no-op
          - baseball was in this slot historically but now has both a
            SECTOR_WEIGHT_OVERRIDES entry AND an isotonic calibration that
            shifts extreme outputs, both of which would confound the test
        nfl runs the standard FLB path with no per-sector overrides and no
        ensemble-level calibration, so the model weight passes through cleanly.
        """
        from evmax.agents.models.base import ModelAgentPrediction

        ensemble = EnsembleModelAgent(models=[], sharp_weight=0.85)
        sharp = make_sharp()
        # Sharp says 90% favorite / 10% underdog
        sharp = sharp.model_copy(update={
            "true_prob_a": 0.90, "true_prob_b": 0.10, "true_prob_draw": None,
        })
        # Model says 80% / 20% (compressing toward 50/50)
        preds = {
            "elo": ModelAgentPrediction(
                event_id="test", model_name="elo",
                true_prob_a=0.80, true_prob_b=0.20, true_prob_draw=None,
                confidence=0.60, weight=0.35,
            ),
        }
        blend = ensemble._blend("test", preds, sharp, 0.85, sector="nfl")
        assert blend is not None
        # Without FLB: blend_b = 0.85*0.10 + 0.15*0.20 = 0.115
        # With FLB (S=1.5): effective sharp weight at 10% is ~88.6%
        # The underdog inflation should be dampened but models still contribute
        assert blend.true_prob_b < 0.115, (
            f"FLB correction should dampen underdog inflation: got {blend.true_prob_b:.3f}"
        )
        # Should be closer to sharp than uncorrected, but not pure sharp
        assert blend.true_prob_b > 0.10, "Models should still contribute at extremes"
        assert blend.true_prob_b < 0.113

    def test_flb_correction_near_50_50_minimal(self):
        """FLB correction should have minimal effect near 50/50."""
        from evmax.agents.models.base import ModelAgentPrediction

        ensemble = EnsembleModelAgent(models=[], sharp_weight=0.85)
        sharp = make_sharp()
        sharp = sharp.model_copy(update={
            "true_prob_a": 0.52, "true_prob_b": 0.48, "true_prob_draw": None,
        })
        preds = {
            "elo": ModelAgentPrediction(
                event_id="test", model_name="elo",
                true_prob_a=0.55, true_prob_b=0.45, true_prob_draw=None,
                confidence=0.60, weight=0.35,
            ),
        }
        blend = ensemble._blend("test", preds, sharp, 0.85, sector="nba")
        assert blend is not None
        # Near 50/50, FLB correction is negligible — model input preserved
        # Without FLB: blend_a = 0.85*0.52 + 0.15*0.55 = 0.5245
        # With FLB: deviation=(0.52-0.5)^2=0.0004, extra≈0.002, eff_sharp≈0.85
        # Should be almost identical to uncorrected
        assert abs(blend.true_prob_a - 0.525) < 0.01

    def test_soccer_weight_overrides_applied_in_blend(self):
        """Verify _blend uses sector overrides, not class-level weights."""
        from evmax.agents.models.base import ModelAgentPrediction

        ensemble = EnsembleModelAgent(models=[], sharp_weight=0.50)

        # Two fake model predictions: elo (overridden to 0.20) and poisson (0.45)
        preds = {
            "elo": ModelAgentPrediction(
                event_id="test", model_name="elo",
                true_prob_a=0.30, true_prob_b=0.50, true_prob_draw=0.20,
                confidence=0.60, weight=0.35,  # class-level weight
            ),
            "poisson": ModelAgentPrediction(
                event_id="test", model_name="poisson",
                true_prob_a=0.40, true_prob_b=0.35, true_prob_draw=0.25,
                confidence=0.60, weight=0.30,  # class-level weight
            ),
        }
        sharp = make_sharp(sector="soccer")
        sharp = sharp.model_copy(update={
            "sector": "soccer", "true_prob_a": 0.35, "true_prob_b": 0.40,
            "true_prob_draw": 0.25,
        })

        # Blend with soccer sector → should use overrides
        blend_soccer = ensemble._blend("test", preds, sharp, 0.50, sector="soccer")
        # Blend with NBA → should use class-level weights
        blend_nba = ensemble._blend("test", preds, sharp, 0.50, sector="nba")

        assert blend_soccer is not None
        assert blend_nba is not None
        # Soccer blend should be closer to Poisson's prob_a (0.40) because
        # Poisson gets 0.45 weight vs Elo's 0.20 in soccer overrides
        # NBA blend has Elo 0.35 vs Poisson 0.30 — more balanced
        assert blend_soccer.true_prob_a != blend_nba.true_prob_a

    def test_sector_override_zeroed_model_excluded_from_model_sources(self):
        """A model zeroed by SECTOR_WEIGHT_OVERRIDES must not appear in
        model_sources — downstream consumers (tennis REQUIRED_BLEND_MODELS
        gate, blend-composition analysis) parse that string as the set of
        models that actually contributed to the blend."""
        from evmax.agents.models.base import ModelAgentPrediction

        ensemble = EnsembleModelAgent(models=[], sharp_weight=0.50)

        # NFL overrides zero out poisson; elo stays at 0.20. Poisson's
        # class-level weight (0.30) × confidence (0.60) > 0, so the old
        # class-weight filter wrongly listed it as contributing.
        preds = {
            "elo": ModelAgentPrediction(
                event_id="test", model_name="elo",
                true_prob_a=0.55, true_prob_b=0.45, true_prob_draw=None,
                confidence=0.60, weight=0.35,
            ),
            "poisson": ModelAgentPrediction(
                event_id="test", model_name="poisson",
                true_prob_a=0.70, true_prob_b=0.30, true_prob_draw=None,
                confidence=0.60, weight=0.30,
            ),
        }
        sharp = make_sharp(sector="nfl")
        sharp = sharp.model_copy(update={"sector": "nfl"})

        assert EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["nfl"]["poisson"] == 0.0
        blend = ensemble._blend("test", preds, sharp, 0.50, sector="nfl")

        assert blend is not None
        sources = blend.model_sources.split("+")
        assert "poisson" not in sources
        assert "elo" in sources
        assert "sharp" in sources

    def test_avg_conf_correct_when_model_filtered_out(self):
        """Blend confidence must use the confidences of the models that
        actually contributed. The old code indexed list(model_preds.values())
        with the model_contribs index — misaligned whenever an earlier model
        was filtered, attaching the wrong confidence to each weight."""
        from evmax.agents.models.base import ModelAgentPrediction

        ensemble = EnsembleModelAgent(models=[], sharp_weight=0.50)

        # First model in dict order fails the confidence gate (0.30 ≤ 0.45)
        # and is filtered out; only elo (confidence 0.80) contributes. The
        # old indexing bug attached form's 0.30 to elo's weight → 0.30.
        preds = {
            "form": ModelAgentPrediction(
                event_id="test", model_name="form",
                true_prob_a=0.50, true_prob_b=0.50, true_prob_draw=None,
                confidence=0.30, weight=0.25,
            ),
            "elo": ModelAgentPrediction(
                event_id="test", model_name="elo",
                true_prob_a=0.55, true_prob_b=0.45, true_prob_draw=None,
                confidence=0.80, weight=0.35,
            ),
        }
        sharp = make_sharp(sector="nfl")
        sharp = sharp.model_copy(update={"sector": "nfl"})

        blend = ensemble._blend("test", preds, sharp, 0.50, sector="nfl")

        assert blend is not None
        # Single contributor → weighted average of confidences is exactly 0.80
        assert blend.confidence == pytest.approx(0.80, abs=1e-3)
        assert "form" not in blend.model_sources.split("+")


# ---------------------------------------------------------------------------
# MODEL-5: tennis weight trim (0.45 → 0.35)
# ---------------------------------------------------------------------------

class TestTennisWeightTrim:
    """Paired with MODEL-1: trimming TennisModelAgent.weight from 0.45 to
    0.35 reduces tennis's intra-model dominance so the ensemble blend is
    less hostage to a single model's prediction when multiple tennis-
    capable models fire on the same event."""

    def test_weight_lock_in(self):
        """Lock in MODEL-5: prevents silent drift back toward 0.45."""
        from evmax.agents.models.tennis_model_agent import TennisModelAgent
        assert TennisModelAgent.weight == 0.35, (
            f"MODEL-5: TennisModelAgent.weight should be 0.35, got "
            f"{TennisModelAgent.weight}. If you're changing this, update "
            f"TODO.md MODEL-5 + this test's docstring with the new rationale."
        )

    def test_weight_change_measurably_shifts_blend(self):
        """Direct arithmetic check on the intra-model confidence-weighted
        average: when tennis (prob_a=0.70) competes with a generic model
        (prob_a=0.40), dropping tennis weight 0.45 → 0.35 moves the blended
        model probability measurably closer to the generic model's prediction.

        This mirrors the ensemble_agent ``_blend`` Step 1 math:
            model_a = Σ(eff_w × prob_a) / Σ(eff_w)
        where ``eff_w = weight × confidence``.
        """
        # Shared confidence (0.6 > 0.45 gate) for both models
        conf = 0.6
        tennis_prob_a = 0.70
        other_prob_a = 0.40
        other_weight = 0.35

        def blended(tennis_weight: float) -> float:
            eff_tennis = tennis_weight * conf
            eff_other = other_weight * conf
            return (
                eff_tennis * tennis_prob_a + eff_other * other_prob_a
            ) / (eff_tennis + eff_other)

        blend_old = blended(0.45)   # old tennis weight
        blend_new = blended(0.35)   # MODEL-5

        # New blend should be closer to the "other" model's prediction
        # than the old blend, meaning tennis dominates less.
        assert abs(blend_new - other_prob_a) < abs(blend_old - other_prob_a), (
            f"MODEL-5 should reduce tennis dominance in the blend. "
            f"Old (w=0.45): {blend_old:.4f}, New (w=0.35): {blend_new:.4f}, "
            f"other: {other_prob_a}"
        )

        # And both blends still lean toward tennis since it's the higher
        # prediction — the trim reduces dominance but doesn't flip the sign.
        assert blend_new > other_prob_a
        assert blend_new < blend_old


# ---------------------------------------------------------------------------
# NFL ensemble weight lock-in (Phase 2, validated 2026-05-01)
# ---------------------------------------------------------------------------

class TestNflWeightOverrides:
    """Lock-in guard for SECTOR_WEIGHT_OVERRIDES['nfl'].

    The NFL blend weights (nfl_efficiency 0.25 · nfl_qb_elo 0.25 · elo 0.20 ·
    form 0.30 · poisson 0.0) were validated via scripts/backtest_nfl_efficiency.py
    (Phase 2, 2026-05-01) but had NO test asserting the numbers — any weight
    iteration could edit them with zero deterministic guard. These tests pin
    the values and the documented structural invariants so a future weight
    sweep must update both the override AND this test's rationale together
    (same pattern as TestTennisWeightTrim).
    """

    def test_nfl_override_values_locked(self):
        nfl = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["nfl"]
        assert nfl == {
            "nfl_efficiency": 0.25,
            "nfl_qb_elo":     0.25,
            "elo":            0.20,
            "form":           0.30,
            "poisson":        0.0,
        }, (
            "SECTOR_WEIGHT_OVERRIDES['nfl'] drifted from the Phase 2 weights. "
            "If you're re-tuning, re-run scripts/backtest_nfl_efficiency.py on "
            "the in-sample seasons (2324,2425), confirm the 2526 holdout, and "
            "update this test's docstring with the new ΔBrier/ΔAcc rationale."
        )

    def test_generic_elo_downweighted_vs_qb_elo(self):
        """Documented rationale (ensemble_agent.py): generic elo overlaps
        nfl_qb_elo (both Elo-shaped team strength), so generic elo is held
        BELOW nfl_qb_elo when the QB-Elo layer is present. Guards the overlap
        invariant, not just the literal value."""
        nfl = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["nfl"]
        assert nfl["elo"] < nfl["nfl_qb_elo"]

    def test_poisson_zeroed_for_nfl(self):
        """NFL scoring is drive-based, not minute-uniform — poisson must stay
        zeroed so it never contributes to or appears in the NFL blend."""
        assert EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["nfl"]["poisson"] == 0.0

    def test_nfl_effective_weights_applied_in_blend(self):
        """End-to-end: _blend must use the NFL override weights, so the
        model-side blend is the confidence-weighted average over the override
        weights — NOT the class-level weights. Construct two NFL models with
        equal confidence and verify the blended model prob matches the
        override-weighted average to the documented Step-1 formula."""
        from evmax.agents.models.base import ModelAgentPrediction

        ensemble = EnsembleModelAgent(models=[], sharp_weight=0.0)

        conf = 0.60  # > 0.45 gate; equal for both so eff_w ∝ override weight
        preds = {
            # class-level weights deliberately wrong (0.35) to prove the
            # override (0.20) is what _blend actually uses.
            "elo": ModelAgentPrediction(
                event_id="test", model_name="elo",
                true_prob_a=0.40, true_prob_b=0.60, true_prob_draw=None,
                confidence=conf, weight=0.35,
            ),
            "nfl_qb_elo": ModelAgentPrediction(
                event_id="test", model_name="nfl_qb_elo",
                true_prob_a=0.70, true_prob_b=0.30, true_prob_draw=None,
                confidence=conf, weight=0.30,
            ),
        }
        # sharp_weight=0 → blend is pure model side (no FLB since no sharp lean)
        sharp = make_sharp(sector="nfl").model_copy(update={
            "sector": "nfl", "true_prob_a": 0.50, "true_prob_b": 0.50,
        })

        blend = ensemble._blend("test", preds, sharp, 0.0, sector="nfl")
        assert blend is not None

        nfl = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["nfl"]
        w_elo = nfl["elo"] * conf
        w_qb = nfl["nfl_qb_elo"] * conf
        expected_model_a = (w_elo * 0.40 + w_qb * 0.70) / (w_elo + w_qb)
        # With sharp_weight=0 the model side passes through (FLB is skipped
        # when sharp_weight is 0 → no model-weight scaling), then normalized.
        assert blend.true_prob_a == pytest.approx(expected_model_a, abs=1e-3)
        # nfl_qb_elo (0.25) outweighs elo (0.20), so the blend leans toward
        # qb_elo's higher prob_a — confirming the override is in force.
        assert blend.true_prob_a > 0.50


class TestDedupMutuallyExclusiveSides:
    """The no-middling rule: when two +EV bets cover opposite sides of the
    same Kalshi market (same event, same market_type, same |line|), keep
    only the higher-EV side. Logging both is a vig double-pay with zero
    net edge — outcomes are mutually exclusive."""

    def _gap(self, **overrides):
        from evmax.agents.odds.ev_gap_agent import EVGap
        defaults = dict(
            market_id="m", event_id="nba::2026-05-11::lal_vs_gsw",
            sector="nba", yes_team="lakers", market_type="moneyline",
            kalshi_yes_price=0.5, sharp_true_prob=0.55,
            blended_true_prob=0.55, ev_pct=0.10,
            kelly_full=0.05, kelly_fraction=0.025, match_confidence=95.0,
            volume_usd=10_000.0, spread_pct=0.02,
            event_date=None, line=None,
        )
        defaults.update(overrides)
        return EVGap(**defaults)

    def test_keeps_higher_ev_side_of_moneyline(self):
        from evmax.agents.odds.ev_gap_agent import _dedup_mutually_exclusive_sides
        gaps = [
            self._gap(yes_team="lakers", ev_pct=0.08),
            self._gap(yes_team="warriors", ev_pct=0.12, market_id="m2"),
        ]
        result = _dedup_mutually_exclusive_sides(gaps)
        assert len(result) == 1
        assert result[0].yes_team == "warriors"

    def test_keeps_both_when_different_market_types(self):
        from evmax.agents.odds.ev_gap_agent import _dedup_mutually_exclusive_sides
        gaps = [
            self._gap(market_type="moneyline", ev_pct=0.08),
            self._gap(market_type="spread", ev_pct=0.05, line=-7.5, market_id="m2"),
        ]
        assert len(_dedup_mutually_exclusive_sides(gaps)) == 2

    def test_keeps_both_when_different_spread_lines(self):
        """Lakers -3 and Lakers -7 are different markets, not opposite sides."""
        from evmax.agents.odds.ev_gap_agent import _dedup_mutually_exclusive_sides
        gaps = [
            self._gap(market_type="spread", line=-3.0, ev_pct=0.05),
            self._gap(market_type="spread", line=-7.5, ev_pct=0.08, market_id="m2"),
        ]
        assert len(_dedup_mutually_exclusive_sides(gaps)) == 2

    def test_collapses_opposite_spread_sides(self):
        """Lakers -7.5 and Warriors +7.5 have the same |line| — opposite sides."""
        from evmax.agents.odds.ev_gap_agent import _dedup_mutually_exclusive_sides
        gaps = [
            self._gap(market_type="spread", yes_team="lakers",
                      line=-7.5, ev_pct=0.04),
            self._gap(market_type="spread", yes_team="warriors",
                      line=7.5, ev_pct=0.06, market_id="m2"),
        ]
        result = _dedup_mutually_exclusive_sides(gaps)
        assert len(result) == 1
        assert result[0].yes_team == "warriors"

    def test_collapses_over_under(self):
        """Same total line, over vs under."""
        from evmax.agents.odds.ev_gap_agent import _dedup_mutually_exclusive_sides
        gaps = [
            self._gap(market_type="total", yes_team="over",
                      line=224.5, ev_pct=0.03),
            self._gap(market_type="total", yes_team="under",
                      line=224.5, ev_pct=0.07, market_id="m2"),
        ]
        result = _dedup_mutually_exclusive_sides(gaps)
        assert len(result) == 1
        assert result[0].yes_team == "under"

    def test_props_are_pass_through(self):
        """Player props with different thresholds are independent bets — Kalshi
        posts 20+ / 25+ / 30+ separately, never as opposite sides."""
        from evmax.agents.odds.ev_gap_agent import _dedup_mutually_exclusive_sides
        gaps = [
            self._gap(market_type="player_prop", line=20.0, ev_pct=0.05),
            self._gap(market_type="player_prop", line=25.0, ev_pct=0.08, market_id="m2"),
            self._gap(market_type="player_prop", line=30.0, ev_pct=0.03, market_id="m3"),
        ]
        assert len(_dedup_mutually_exclusive_sides(gaps)) == 3
