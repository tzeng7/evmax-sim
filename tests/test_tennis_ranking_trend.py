"""Tests for TennisRankingTrendAgent — rank+momentum logit, anchoring, staleness.

Covers the 2026-06-10 rework: absolute log-rank differential + log-space
momentum (replacing the raw-spots 0.5-centered nudge), event-date anchoring
(no lookahead in backtests), and the stale-history guard.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from datetime import date, datetime, timedelta
from math import exp, log

import pytest

from evmax.agents.models.tennis_ranking_trend_agent import (
    TennisRankingTrendAgent,
    _MAX_LOGIT,
    _MOMENTUM_K,
    _RANK_ALPHA,
    _STALE_WEEKS,
)
from evmax.models.market import MarketSource, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


def _weekly_history(start: date, ranks: list[int]) -> list[dict]:
    return [
        {"date": (start + timedelta(weeks=i)).isoformat(), "rank": r}
        for i, r in enumerate(ranks)
    ]


def _agent_with(history: dict[str, list[dict]]) -> TennisRankingTrendAgent:
    agent = TennisRankingTrendAgent.__new__(TennisRankingTrendAgent)
    agent._state = {"history": history}
    agent.save_state = lambda: None
    return agent


def _pair(player_a: str, player_b: str, event_date: datetime | None):
    market = PredictionMarket(
        id="t", market_id="t", event_id="e", sector="tennis",
        team_home=player_a, team_away=player_b,
        source=MarketSource.kalshi, yes_price=0.5, no_price=0.5,
        title=f"{player_a} vs {player_b}", event_date=event_date,
    )
    sharp = SharpOdds(
        event_id="e", book=SharpBook.pinnacle, sector="tennis",
        outcome_a_label=player_a, outcome_b_label=player_b,
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=0.5, true_prob_b=0.5,
    )
    return market, sharp


START = date(2026, 1, 5)
MATCH_DT = datetime(2026, 3, 16, 12)  # 10 weeks after START


class TestRankMomentumLogit:
    def test_better_ranked_player_favored(self):
        agent = _agent_with({
            "top seed": _weekly_history(START, [2] * 11),
            "mid tier": _weekly_history(START, [60] * 11),
        })
        pred = asyncio.run(agent.predict_pair(*_pair("top seed", "mid tier", MATCH_DT)))
        expected = 1.0 / (1.0 + exp(-_RANK_ALPHA * (log(60) - log(2))))
        assert pred is not None
        assert pred.true_prob_a == pytest.approx(expected, abs=1e-9)
        assert pred.true_prob_a > 0.7

    def test_equal_rank_equal_momentum_is_coin_flip(self):
        agent = _agent_with({
            "player one": _weekly_history(START, [30] * 11),
            "player two": _weekly_history(START, [30] * 11),
        })
        pred = asyncio.run(agent.predict_pair(*_pair("player one", "player two", MATCH_DT)))
        assert pred.true_prob_a == pytest.approx(0.5, abs=1e-9)

    def test_momentum_breaks_equal_rank_tie(self):
        # Both end at rank 30; A climbed from 60, B fell from 15.
        agent = _agent_with({
            "rising star": _weekly_history(START, [60, 55, 50, 45, 42, 40, 38, 35, 33, 31, 30]),
            "fading vet": _weekly_history(START, [15, 16, 18, 20, 21, 23, 25, 27, 28, 29, 30]),
        })
        pred = asyncio.run(agent.predict_pair(*_pair("rising star", "fading vet", MATCH_DT)))
        assert pred.true_prob_a > 0.5

    def test_momentum_contribution_is_capped(self):
        # Extreme momentum gap (1000 → 2 vs flat), identical final rank.
        agent = _agent_with({
            "meteor rise": _weekly_history(START, [1000, 500, 250, 120, 60, 30, 15, 8, 4, 3, 2]),
            "steady two": _weekly_history(START, [2] * 11),
        })
        pred = asyncio.run(agent.predict_pair(*_pair("meteor rise", "steady two", MATCH_DT)))
        max_prob = 1.0 / (1.0 + exp(-_MAX_LOGIT))
        assert pred.true_prob_a <= max_prob + 1e-9
        assert pred.true_prob_a > 0.5

    def test_momentum_k_scaling_below_cap(self):
        # A halved rank (60→30) vs flat B at 30: pure momentum, no rank gap.
        agent = _agent_with({
            "halved rank": _weekly_history(START, [60, 57, 59, 60, 55, 50, 45, 40, 36, 32, 30]),
            "flat thirty": _weekly_history(START, [30] * 11),
        })
        pred = asyncio.run(agent.predict_pair(*_pair("halved rank", "flat thirty", MATCH_DT)))
        expected = 1.0 / (1.0 + exp(-_MOMENTUM_K * log(2)))
        assert pred.true_prob_a == pytest.approx(expected, abs=1e-9)


class TestDateAnchoring:
    def test_no_lookahead_past_event_date(self):
        # A's rank collapses AFTER the match date — prediction must not see it.
        clean = _weekly_history(START, [10] * 11)
        with_future = clean + _weekly_history(START + timedelta(weeks=11), [500] * 5)
        agent = _agent_with({
            "future faller": with_future,
            "steady rival": _weekly_history(START, [10] * 16),
        })
        pred = asyncio.run(agent.predict_pair(*_pair("future faller", "steady rival", MATCH_DT)))
        assert pred.true_prob_a == pytest.approx(0.5, abs=1e-9)

    def test_stale_history_returns_none(self):
        # Latest snapshot ends > _STALE_WEEKS before the match.
        stale_dt = datetime.combine(
            START + timedelta(weeks=10 + _STALE_WEEKS + 2), datetime.min.time()
        )
        agent = _agent_with({
            "old entry": _weekly_history(START, [10] * 11),
            "also old": _weekly_history(START, [20] * 11),
        })
        pred = asyncio.run(agent.predict_pair(*_pair("old entry", "also old", stale_dt)))
        assert pred is None

    def test_too_few_snapshots_returns_none(self):
        agent = _agent_with({
            "newcomer kid": _weekly_history(START + timedelta(weeks=8), [50, 40]),
            "established pro": _weekly_history(START, [10] * 11),
        })
        pred = asyncio.run(agent.predict_pair(*_pair("newcomer kid", "established pro", MATCH_DT)))
        assert pred is None

    def test_non_tennis_sector_returns_none(self):
        agent = _agent_with({})
        market, sharp = _pair("a player", "b player", MATCH_DT)
        market.sector = "nba"
        assert asyncio.run(agent.predict_pair(market, sharp)) is None


class TestSeedHistoryReplaces:
    def test_seed_history_fully_replaces_store(self):
        # A stale legacy player must not survive a fresh matchmx reseed — the
        # store is the authoritative current snapshot, not an accumulator.
        agent = _agent_with({
            "legacy player": _weekly_history(START, [10] * 11),
        })
        agent.log = SimpleNamespace(info=lambda *a, **k: None)
        agent.seed_history({
            "Current Player": _weekly_history(START, [5] * 11),
        })
        store = agent._state["history"]
        assert "legacy player" not in store          # dropped, not merged
        assert "current player" in store             # lowercased key
        assert store["current player"][0]["rank"] == 5
