"""Pre-start gate: the EV agent never prices a game that has already started.

Regression for 2026-11-14: Pinnacle's guest API kept serving Le Moyne vs
Manhattan's frozen pre-game line 39 minutes after the 23:00Z start, and the
scan priced the in-play venue ask against it (+135% EV, logged live).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from evmax.agents.base import AgentRequest
from evmax.agents.odds.ev_gap_agent import EVGapAgent, _event_started
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds

NOW = datetime(2026, 11, 14, 23, 39, tzinfo=timezone.utc)


def _market() -> PredictionMarket:
    return PredictionMarket(
        id="kalshi:lakers_vs_celtics",
        source=MarketSource.kalshi,
        sector="nba",
        market_type=MarketType.moneyline,
        title="lakers vs celtics",
        ticker="KXNBAGAME-TEST-LAKERS",
        yes_price=0.45,
        no_price=0.57,
        volume_usd=10_000.0,
        open_interest_usd=5_000.0,
        team_home="lakers",
        team_away="celtics",
        yes_team="lakers",
        event_date=datetime.now(timezone.utc) + timedelta(days=1),
    )


def _sharp(start: datetime | None) -> SharpOdds:
    return SharpOdds(
        event_id="nba::2026-11-14::lakers_vs_celtics",
        book=SharpBook.pinnacle,
        sector="nba",
        outcome_a_label="lakers",
        outcome_b_label="celtics",
        outcome_a_decimal=1.0 / 0.58,
        outcome_b_decimal=1.0 / 0.42,
        true_prob_a=0.58,
        true_prob_b=0.42,
        margin=0.04,
        event_date=start,
    )


class TestEventStarted:
    def test_start_in_the_past_is_started(self):
        assert _event_started(_sharp(NOW - timedelta(minutes=39)), NOW)

    def test_start_exactly_now_is_started(self):
        assert _event_started(_sharp(NOW), NOW)

    def test_start_in_the_future_is_not_started(self):
        assert not _event_started(_sharp(NOW + timedelta(minutes=1)), NOW)

    def test_missing_start_is_not_started(self):
        assert not _event_started(_sharp(None), NOW)

    def test_naive_start_is_read_as_utc(self):
        naive = (NOW - timedelta(minutes=5)).replace(tzinfo=None)
        assert _event_started(_sharp(naive), NOW)


async def _run(sharp: SharpOdds):
    agent = EVGapAgent()
    market = _market()
    with patch.object(agent._matching, "match_all", return_value=[(market, sharp, 95.0)]):
        return await agent(
            AgentRequest(sector="nba", params={"kalshi_markets": [market], "sharp_odds": [sharp]})
        )


class TestRunDropsStartedEvents:
    @pytest.mark.asyncio
    async def test_started_event_produces_no_gap(self):
        resp = await _run(_sharp(datetime.now(timezone.utc) - timedelta(minutes=39)))
        assert resp.status == "ok"
        assert resp.data == []

    @pytest.mark.asyncio
    async def test_upcoming_event_still_produces_a_gap(self):
        resp = await _run(_sharp(datetime.now(timezone.utc) + timedelta(hours=3)))
        assert len(resp.data) >= 1
        assert resp.data[0].ev_pct > 0.02
