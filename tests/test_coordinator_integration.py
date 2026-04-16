"""Integration tests for AgentCoordinator.run_cycle().

TEST-4 from TODO.md — the single test that exercises the full scan pipeline
end-to-end against fixture data. Every individual agent has unit tests, but
nothing verified they were wired together correctly until this file.

These tests mock the network boundary (Kalshi, Pinnacle, ESPN, nflverse)
and run a real coordinator through matching → ensemble → EVGap → exposure
guard. The assertions check that the output CycleResult has the right
shape and that EV gaps appear when the fixture prices imply an edge.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evmax.agents.base import AgentRequest, AgentResponse
from evmax.agents.coordinator import AgentCoordinator, CycleResult
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_market(
    *,
    market_id: str,
    sector: str,
    team_home: str,
    team_away: str,
    yes_team: str,
    yes_price: float,
    market_type: MarketType = MarketType.moneyline,
    event_date: Optional[datetime] = None,
    player_name: Optional[str] = None,
    stat_type: Optional[str] = None,
    threshold: Optional[float] = None,
    event_id: Optional[str] = None,
) -> PredictionMarket:
    ed = event_date or datetime(2026, 4, 20, 0, 0, tzinfo=timezone.utc)
    return PredictionMarket(
        id=market_id,
        source=MarketSource.kalshi,
        sector=sector,
        market_type=market_type,
        title=f"{team_home} vs {team_away}",
        yes_team=yes_team,
        yes_price=yes_price,
        no_price=round(1.0 - yes_price, 4),
        team_home=team_home,
        team_away=team_away,
        event_date=ed,
        event_id=event_id or f"{sector}::2026-04-20::{team_home.lower()}_vs_{team_away.lower()}",
        player_name=player_name,
        stat_type=stat_type,
        threshold=threshold,
    )


def _make_sharp(
    *,
    event_id: str,
    sector: str,
    outcome_a: str,
    outcome_b: str,
    true_prob_a: float,
    event_date: Optional[datetime] = None,
    prop_player_name: Optional[str] = None,
    prop_stat_type: Optional[str] = None,
    total_line: Optional[float] = None,
    true_prob_over: Optional[float] = None,
    true_prob_under: Optional[float] = None,
    prop_l15_games: int = 0,
) -> SharpOdds:
    ed = event_date or datetime(2026, 4, 20, 0, 0, tzinfo=timezone.utc)
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_label=outcome_a,
        outcome_b_label=outcome_b,
        outcome_a_decimal=1.0 / true_prob_a if true_prob_a > 0 else 10.0,
        outcome_b_decimal=1.0 / (1 - true_prob_a) if true_prob_a < 1 else 10.0,
        true_prob_a=true_prob_a,
        true_prob_b=1.0 - true_prob_a,
        margin=0.02,
        event_date=ed,
        total_line=total_line,
        prop_player_name=prop_player_name,
        prop_stat_type=prop_stat_type,
        prop_l15_games=prop_l15_games,
        true_prob_over=true_prob_over,
        true_prob_under=true_prob_under,
    )


def _resp(agent_name: str, sector: str, data: Any) -> AgentResponse:
    return AgentResponse(agent_name=agent_name, sector=sector, data=data)


def _noop_archiver():
    """Return a no-op DataArchiver mock."""
    archiver = MagicMock()
    archiver.archive_kalshi_markets.return_value = 0
    archiver.archive_sharp_odds.return_value = 0
    return archiver


# ---------------------------------------------------------------------------
# NBA game-level integration test
# ---------------------------------------------------------------------------


class TestNbaGameCycle:
    """Run a full cycle for NBA with one game where the model edge is
    ~10% — Kalshi prices the home team at 45% but Pinnacle devigs to 55%.
    The ensemble is disabled so the blend = sharp. EVGap should fire."""

    NBA_MARKETS = [
        _make_market(
            market_id="kalshi-nba-lal-gsw-yes",
            sector="nba",
            team_home="Los Angeles Lakers",
            team_away="Golden State Warriors",
            yes_team="Los Angeles Lakers",
            yes_price=0.45,
        ),
    ]

    NBA_SHARP = [
        _make_sharp(
            event_id="nba::2026-04-20::lakers_vs_warriors",
            sector="nba",
            outcome_a="Los Angeles Lakers",
            outcome_b="Golden State Warriors",
            true_prob_a=0.55,
        ),
    ]

    def _build_coord(self) -> AgentCoordinator:
        coord = AgentCoordinator(
            sectors=["nba"],
            enable_models=False,
            enable_injuries=False,
            bankroll=500.0,
            kelly_fraction=0.5,
        )
        # Replace agent callables with async stubs
        async def _kalshi_stub(req):
            return _resp("kalshi", req.sector, self.NBA_MARKETS)

        async def _sharp_stub(req):
            return _resp("sharp", req.sector, self.NBA_SHARP)

        async def _standings_stub(req):
            return _resp("standings", req.sector, {})

        coord.kalshi_agent = _kalshi_stub
        coord.sharp_agent = _sharp_stub
        coord.standings_agent = _standings_stub
        coord._archiver = _noop_archiver()
        coord._notifier = MagicMock()
        return coord

    @patch("evmax.agents.coordinator._load_steam_cache", return_value={})
    @patch("evmax.agents.coordinator._save_steam_cache")
    def test_cycle_produces_ev_gap_for_mispriced_game(self, _save, _load):
        coord = self._build_coord()
        result = _run(coord.run_cycle())

        assert isinstance(result, CycleResult)
        assert "nba" in result.sectors_scanned
        assert result.markets_fetched >= 1
        assert len(result.errors) == 0
        assert len(result.ev_gaps) >= 1

        gap = result.ev_gaps[0]
        assert gap.sector == "nba"
        assert gap.ev_pct > 0.05  # ~22% EV from 0.45 vs 0.55
        assert gap.blended_true_prob >= 0.50
        assert gap.kelly_fraction > 0

    @patch("evmax.agents.coordinator._load_steam_cache", return_value={})
    @patch("evmax.agents.coordinator._save_steam_cache")
    def test_cycle_result_has_correct_bankroll(self, _save, _load):
        coord = self._build_coord()
        result = _run(coord.run_cycle())
        assert result.bankroll == 500.0
        assert result.kelly_fraction == 0.5


# ---------------------------------------------------------------------------
# No-edge scenario — Kalshi and Pinnacle agree
# ---------------------------------------------------------------------------


class TestNoEdgeCycle:
    """When Kalshi yes_price ≈ Pinnacle true_prob, no EVGap should fire."""

    def _build_coord(self) -> AgentCoordinator:
        coord = AgentCoordinator(
            sectors=["nba"],
            enable_models=False,
            enable_injuries=False,
        )

        async def _kalshi_stub(req):
            return _resp("kalshi", req.sector, [
                _make_market(
                    market_id="kalshi-nba-fair",
                    sector="nba",
                    team_home="Miami Heat",
                    team_away="Boston Celtics",
                    yes_team="Miami Heat",
                    yes_price=0.52,
                ),
            ])

        async def _sharp_stub(req):
            return _resp("sharp", req.sector, [
                _make_sharp(
                    event_id="nba::2026-04-20::heat_vs_celtics",
                    sector="nba",
                    outcome_a="Miami Heat",
                    outcome_b="Boston Celtics",
                    true_prob_a=0.50,
                ),
            ])

        async def _standings_stub(req):
            return _resp("standings", req.sector, {})

        coord.kalshi_agent = _kalshi_stub
        coord.sharp_agent = _sharp_stub
        coord.standings_agent = _standings_stub
        coord._archiver = _noop_archiver()
        coord._notifier = MagicMock()
        return coord

    @patch("evmax.agents.coordinator._load_steam_cache", return_value={})
    @patch("evmax.agents.coordinator._save_steam_cache")
    def test_no_gaps_when_price_matches_sharp(self, _save, _load):
        coord = self._build_coord()
        result = _run(coord.run_cycle())
        assert len(result.ev_gaps) == 0
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Multi-sector cycle
# ---------------------------------------------------------------------------


class TestMultiSectorCycle:
    """Run NBA + soccer in one cycle. Both have an edge. Verify both
    sectors produce gaps and appear in sectors_scanned."""

    def _build_coord(self) -> AgentCoordinator:
        coord = AgentCoordinator(
            sectors=["nba", "soccer"],
            enable_models=False,
            enable_injuries=False,
        )
        markets_by_sector = {
            "nba": [
                _make_market(
                    market_id="kalshi-nba-multi",
                    sector="nba",
                    team_home="Denver Nuggets",
                    team_away="Phoenix Suns",
                    yes_team="Denver Nuggets",
                    yes_price=0.42,
                ),
            ],
            "soccer": [
                _make_market(
                    market_id="kalshi-epl-multi",
                    sector="soccer",
                    team_home="Arsenal",
                    team_away="Chelsea",
                    yes_team="Arsenal",
                    yes_price=0.40,
                ),
            ],
        }
        sharp_by_sector = {
            "nba": [
                _make_sharp(
                    event_id="nba::2026-04-20::nuggets_vs_suns",
                    sector="nba",
                    outcome_a="Denver Nuggets",
                    outcome_b="Phoenix Suns",
                    true_prob_a=0.58,
                ),
            ],
            "soccer": [
                _make_sharp(
                    event_id="soccer::2026-04-20::arsenal_vs_chelsea",
                    sector="soccer",
                    outcome_a="Arsenal",
                    outcome_b="Chelsea",
                    true_prob_a=0.55,
                ),
            ],
        }

        async def _kalshi_stub(req):
            return _resp("kalshi", req.sector, markets_by_sector.get(req.sector, []))

        async def _sharp_stub(req):
            return _resp("sharp", req.sector, sharp_by_sector.get(req.sector, []))

        async def _standings_stub(req):
            return _resp("standings", req.sector, {})

        coord.kalshi_agent = _kalshi_stub
        coord.sharp_agent = _sharp_stub
        coord.standings_agent = _standings_stub
        coord._archiver = _noop_archiver()
        coord._notifier = MagicMock()
        return coord

    @patch("evmax.agents.coordinator._load_steam_cache", return_value={})
    @patch("evmax.agents.coordinator._save_steam_cache")
    def test_both_sectors_produce_gaps(self, _save, _load):
        coord = self._build_coord()
        result = _run(coord.run_cycle())

        assert set(result.sectors_scanned) == {"nba", "soccer"}
        assert len(result.errors) == 0
        sectors_in_gaps = {g.sector for g in result.ev_gaps}
        assert "nba" in sectors_in_gaps
        assert "soccer" in sectors_in_gaps


# ---------------------------------------------------------------------------
# Sector failure resilience
# ---------------------------------------------------------------------------


class TestSectorFailureResilience:
    """If one sector's _run_sector hard-fails, the other sectors still
    produce gaps and the error is captured in result.errors."""

    def _build_coord(self) -> AgentCoordinator:
        coord = AgentCoordinator(
            sectors=["nba", "nhl"],
            enable_models=False,
            enable_injuries=False,
        )
        # NBA stubs — returns a game with no edge (we just want it to survive)
        async def _kalshi_stub(req):
            return _resp("kalshi", req.sector, [
                _make_market(
                    market_id="kalshi-nba-resil",
                    sector="nba",
                    team_home="Chicago Bulls",
                    team_away="Milwaukee Bucks",
                    yes_team="Milwaukee Bucks",
                    yes_price=0.50,
                ),
            ])

        async def _sharp_stub(req):
            return _resp("sharp", req.sector, [
                _make_sharp(
                    event_id="nba::2026-04-20::bulls_vs_bucks",
                    sector="nba",
                    outcome_a="Chicago Bulls",
                    outcome_b="Milwaukee Bucks",
                    true_prob_a=0.50,
                ),
            ])

        async def _standings_stub(req):
            return _resp("standings", req.sector, {})

        coord.kalshi_agent = _kalshi_stub
        coord.sharp_agent = _sharp_stub
        coord.standings_agent = _standings_stub
        coord._archiver = _noop_archiver()
        coord._notifier = MagicMock()

        # Make NHL's _run_sector hard-crash. This simulates a real sector
        # failure (unhandled exception in matching/model/gap code) — the
        # kind of crash that asyncio.gather with return_exceptions=True
        # captures and routes to result.errors.
        original = coord._run_sector

        async def _crashing_run_sector(sector, cid):
            if sector == "nhl":
                raise RuntimeError("NHL sector crashed unexpectedly")
            return await original(sector, cid)

        coord._run_sector = _crashing_run_sector
        return coord

    @patch("evmax.agents.coordinator._load_steam_cache", return_value={})
    @patch("evmax.agents.coordinator._save_steam_cache")
    def test_healthy_sector_survives_sibling_failure(self, _save, _load):
        coord = self._build_coord()
        result = _run(coord.run_cycle())

        assert "nba" in result.sectors_scanned
        assert "nhl" not in result.sectors_scanned
        assert len(result.errors) >= 1
        assert any("nhl" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# NBA prop flow — verify _fetch_props produces prop_sharp_pairs
# ---------------------------------------------------------------------------


class TestNbaPropFlow:
    """Verify that NBA props flow through _fetch_props into the CycleResult
    prop_sharp_pairs, independent of whether they produce EVGaps."""

    def _build_coord(self, nfl_parquets_fixture=None) -> AgentCoordinator:
        coord = AgentCoordinator(
            sectors=["nba"],
            enable_models=False,
            enable_injuries=False,
        )
        # Game markets + sharp (no edge, so no game-level EVGap noise)
        async def _kalshi_stub(req):
            return _resp("kalshi", req.sector, [])

        async def _sharp_stub(req):
            return _resp("sharp", req.sector, [])

        async def _standings_stub(req):
            return _resp("standings", req.sector, {})

        coord.kalshi_agent = _kalshi_stub
        coord.sharp_agent = _sharp_stub
        coord.standings_agent = _standings_stub
        coord._archiver = _noop_archiver()
        coord._notifier = MagicMock()

        # Mock _fetch_props to return canned prop data
        prop_market = _make_market(
            market_id="kalshi-nba-pts-lebron",
            sector="nba",
            team_home="LAL",
            team_away="GSW",
            yes_team="LeBron James",
            yes_price=0.55,
            market_type=MarketType.player_prop,
            player_name="LeBron James",
            stat_type="points",
            threshold=25.5,
            event_id="nba::2026-04-20::prop::lebron_james::points::25.5",
        )
        prop_sharp = _make_sharp(
            event_id="nba::2026-04-20::prop::lebron_james::points::25.5",
            sector="nba",
            outcome_a="over",
            outcome_b="under",
            true_prob_a=0.0,
            prop_player_name="LeBron James",
            prop_stat_type="points",
            total_line=25.5,
            true_prob_over=0.60,
            true_prob_under=0.40,
            prop_l15_games=15,
        )

        async def _mock_fetch_props(sector):
            return [prop_market], [prop_sharp]

        coord._fetch_props = _mock_fetch_props
        return coord

    @patch("evmax.agents.coordinator._load_steam_cache", return_value={})
    @patch("evmax.agents.coordinator._save_steam_cache")
    def test_prop_sharp_pairs_appear_in_result(self, _save, _load):
        coord = self._build_coord()
        result = _run(coord.run_cycle())

        assert len(result.prop_sharp_pairs) == 1
        sharp, market = result.prop_sharp_pairs[0]
        assert sharp.prop_player_name == "LeBron James"
        assert market.player_name == "LeBron James"
        assert sharp.true_prob_over == 0.60
