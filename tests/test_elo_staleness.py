"""Elo staleness guard — mirror of FormModelAgent's STALE_DAYS gate.

A sector whose `update()` cron silently stopped (e.g. baseball form froze
2026-03-08) keeps feeding near-seed Elo into the ensemble at full weight. The
guard stamps a per-sector `last_updated` date in state and drops the model out
of the blend when that date is > STALE_DAYS before the game's event_date.

Legacy state with no `last_updated` field keeps contributing (guard disabled)
so seeded states behave exactly as before.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import pytest

from evmax.agents.models.elo_agent import STALE_DAYS, EloModelAgent
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook


def _agent() -> EloModelAgent:
    a = EloModelAgent()
    a._state = {}  # isolate from disk-loaded production state
    return a


def _pair(home: str, away: str, event_date: datetime, sector: str = "baseball"):
    market = PredictionMarket(
        id="t", market_id="t", event_id=f"t_{home}_{away}",
        sector=sector, team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=0.5, no_price=0.5,
        event_date=event_date,
    )
    sharp = SharpOdds(
        event_id=f"t_{home}_{away}", book=SharpBook.pinnacle, sector=sector,
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=0.5, true_prob_b=0.5,
    )
    return market, sharp


class TestIsStaleStatic:
    def test_fresh_within_window_not_stale(self) -> None:
        ref = date(2026, 6, 9)
        assert EloModelAgent._is_stale("2026-06-01", ref) is False

    def test_stale_beyond_window(self) -> None:
        ref = date(2026, 6, 9)
        # March 8 → June 9 is well over 60 days
        assert EloModelAgent._is_stale("2026-03-08", ref) is True

    def test_exactly_at_boundary_not_stale(self) -> None:
        ref = date(2026, 6, 9)
        boundary = (ref.toordinal() - STALE_DAYS)
        assert EloModelAgent._is_stale(date.fromordinal(boundary).isoformat(), ref) is False

    def test_missing_date_disables_guard(self) -> None:
        # Legacy state without last_updated → never stale.
        assert EloModelAgent._is_stale(None, date(2026, 6, 9)) is False

    def test_unparseable_date_is_stale(self) -> None:
        assert EloModelAgent._is_stale("not-a-date", date(2026, 6, 9)) is True


class TestPredictPairGate:
    def _seed(self, agent: EloModelAgent, last_updated, *, with_metadata: bool) -> None:
        state = {
            "ratings": {"yankees": 1560.0, "red sox": 1500.0},
            "game_counts": {"yankees": 30, "red sox": 30},
            "season_games": {"yankees": 30, "red sox": 30},
            "h2h": {},
        }
        if with_metadata:
            state["last_updated"] = last_updated
        agent._state = {"baseball": state}

    def test_fresh_sector_contributes(self) -> None:
        agent = _agent()
        self._seed(agent, "2026-06-01", with_metadata=True)
        market, sharp = _pair("yankees", "red sox", datetime(2026, 6, 9, tzinfo=timezone.utc))
        pred = asyncio.run(agent.predict_pair(market, sharp))
        assert pred is not None
        assert pred.confidence >= 0.45

    def test_frozen_sector_gated_out(self) -> None:
        agent = _agent()
        self._seed(agent, "2026-03-08", with_metadata=True)
        market, sharp = _pair("yankees", "red sox", datetime(2026, 6, 9, tzinfo=timezone.utc))
        pred = asyncio.run(agent.predict_pair(market, sharp))
        assert pred is None

    def test_legacy_state_without_metadata_still_contributes(self) -> None:
        # No last_updated key → guard disabled, model contributes as before.
        agent = _agent()
        self._seed(agent, None, with_metadata=False)
        market, sharp = _pair("yankees", "red sox", datetime(2026, 6, 9, tzinfo=timezone.utc))
        pred = asyncio.run(agent.predict_pair(market, sharp))
        assert pred is not None
        assert pred.confidence >= 0.45

    def test_offseason_within_window_unaffected(self) -> None:
        # Game during the offseason but within 60 days of last update → fine.
        agent = _agent()
        self._seed(agent, "2026-01-15", with_metadata=True)
        market, sharp = _pair("yankees", "red sox", datetime(2026, 2, 28, tzinfo=timezone.utc))
        pred = asyncio.run(agent.predict_pair(market, sharp))
        assert pred is not None


class TestUpdateStampsDate:
    def test_update_records_event_date(self) -> None:
        agent = _agent()
        agent.update("yankees", "red sox", score_a=5, score_b=3,
                     sector="baseball", event_date="2026-06-09")
        assert agent._state["baseball"]["last_updated"] == "2026-06-09"

    def test_update_advances_to_latest_date(self) -> None:
        agent = _agent()
        agent.update("yankees", "red sox", 5, 3, "baseball", "2026-06-09")
        # An earlier event must NOT roll the stamp backward.
        agent.update("yankees", "red sox", 4, 2, "baseball", "2026-06-01")
        assert agent._state["baseball"]["last_updated"] == "2026-06-09"
        # A later event advances it.
        agent.update("yankees", "red sox", 6, 1, "baseball", "2026-06-12")
        assert agent._state["baseball"]["last_updated"] == "2026-06-12"
