"""Bet resolver — settles open paper bets by querying market outcomes.

Resolution modes:
  1. Kalshi price check (existing): checks if yes_price settled to 0.0 or 1.0.
     Use when: Kalshi markets are fully closed (usually 1-2 days after game).

  2. ESPN game scores (new): fetches ESPN final scores for a target date,
     fuzzy-matches event_id teams to determine the winner, resolves bets.
     Use when: the day after a game to get fast automated resolution.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evmax.db import AsyncSessionLocal
from evmax.models.simulated_bet import BetStatus, SimulatedBet, SimulatedBetORM

logger = structlog.get_logger(__name__)


class BetResolver:
    """
    Resolves open simulated bets.

    Resolution strategy:
      - Query current market data for each open bet's market_id
      - If yes_price >= 0.99 → outcome "yes" wins → yes bets WIN, no bets LOSE
      - If yes_price <= 0.01 → outcome "no" wins → no bets WIN, yes bets LOSE
      - If market still open → leave as open
    """

    async def resolve_bet(
        self,
        bet_orm: SimulatedBetORM,
        won: bool,
        session: AsyncSession,
    ) -> SimulatedBet:
        """
        Settle a single bet.

        Args:
            bet_orm: ORM record of the open bet.
            won: Whether the bet outcome resolved in our favor.
            session: DB session.

        Returns:
            Updated SimulatedBet.
        """
        stake = bet_orm.stake_usd
        odds = bet_orm.odds_decimal

        if won:
            pnl = stake * (odds - 1.0)
            status = BetStatus.won
        else:
            pnl = -stake
            status = BetStatus.lost

        bet_orm.status = status
        bet_orm.pnl_usd = pnl
        bet_orm.resolved_at = datetime.now(timezone.utc)
        # bankroll_after tracks the restoration/loss
        bet_orm.bankroll_after = (bet_orm.bankroll_before or 0.0) + (stake * odds if won else 0.0)

        await session.commit()

        logger.info(
            "bet_resolved",
            bet_id=bet_orm.id,
            market_id=bet_orm.market_id,
            outcome=bet_orm.outcome,
            status=status,
            pnl_usd=pnl,
        )

        return SimulatedBet.model_validate(bet_orm.__dict__)

    async def resolve_from_market_price(
        self,
        market_id: str,
        current_yes_price: float,
        session: AsyncSession,
    ) -> list[SimulatedBet]:
        """
        Auto-resolve bets for a market based on current yes price.

        Args:
            market_id: Market identifier.
            current_yes_price: Current YES price (0.0–1.0).
            session: DB session.

        Returns:
            List of resolved bets.
        """
        result = await session.execute(
            select(SimulatedBetORM).where(
                SimulatedBetORM.market_id == market_id,
                SimulatedBetORM.status == BetStatus.open,
            )
        )
        open_bets = result.scalars().all()

        if not open_bets:
            return []

        # Determine resolution
        if current_yes_price >= 0.99:
            # YES resolved
            resolved = []
            for bet in open_bets:
                won = bet.outcome.lower() == "yes"
                resolved.append(await self.resolve_bet(bet, won, session))
            return resolved
        elif current_yes_price <= 0.01:
            # NO resolved
            resolved = []
            for bet in open_bets:
                won = bet.outcome.lower() == "no"
                resolved.append(await self.resolve_bet(bet, won, session))
            return resolved

        return []  # Market not yet resolved

    async def resolve_all_settled(
        self,
        settled_markets: dict[str, float],
    ) -> list[SimulatedBet]:
        """
        Resolve all open bets for a batch of settled markets.

        Args:
            settled_markets: {market_id: final_yes_price}

        Returns:
            All resolved bets.
        """
        all_resolved: list[SimulatedBet] = []

        async with AsyncSessionLocal() as session:
            for market_id, yes_price in settled_markets.items():
                resolved = await self.resolve_from_market_price(
                    market_id, yes_price, session
                )
                all_resolved.extend(resolved)

        return all_resolved

    async def resolve_from_espn_date(self, target_date: date) -> dict:
        """
        Resolve open bets using ESPN final scores for a target date.

        Fetches completed game scores, fuzzy-matches event_id teams to
        determine winners, and settles any matching open bets.

        Args:
            target_date: Date to fetch ESPN results for (usually yesterday).

        Returns:
            {"resolved": int, "unmatched": int, "skipped": int,
             "unmatched_ids": list[str]}
        """
        from evmax.agents.cleanup.resolver import (
            fetch_completed_scores,
            _match_espn,
            _slug_teams,
            _fuzzy_team_match,
        )

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(SimulatedBetORM).where(SimulatedBetORM.status == BetStatus.open)
            )
            open_bets = result.scalars().all()

        if not open_bets:
            return {"resolved": 0, "unmatched": 0, "skipped": 0, "unmatched_ids": []}

        # Group by sector for batched ESPN fetches
        by_sector: dict[str, list[SimulatedBetORM]] = {}
        for bet in open_bets:
            by_sector.setdefault(bet.sector, []).append(bet)

        resolved = 0
        unmatched = 0
        skipped = 0
        unmatched_ids: list[str] = []

        for sector, bets in by_sector.items():
            scores = await fetch_completed_scores(sector, target_date)
            if not scores:
                # ESPN has no coverage for this sector (e.g. tennis uses Kalshi settlement)
                skipped += len(bets)
                continue

            async with AsyncSessionLocal() as session:
                for bet in bets:
                    yes_team = _derive_yes_team(bet.market_id, bet.event_id)
                    pred = {
                        "event_id": bet.event_id,
                        "yes_team": yes_team,
                        "sector": sector,
                    }
                    outcome = _match_espn(pred, scores)

                    if outcome is not None:
                        # Re-fetch within this session to get an attached ORM object
                        result = await session.execute(
                            select(SimulatedBetORM).where(SimulatedBetORM.id == bet.id)
                        )
                        bet_orm = result.scalar_one_or_none()
                        if bet_orm is None or bet_orm.status != BetStatus.open:
                            continue
                        won = bool(outcome)
                        await self.resolve_bet(bet_orm, won, session)
                        resolved += 1
                    else:
                        unmatched += 1
                        unmatched_ids.append(bet.event_id)

        logger.info(
            "espn_resolve_done",
            date=target_date.isoformat(),
            resolved=resolved,
            unmatched=unmatched,
            skipped=skipped,
        )
        return {
            "resolved": resolved,
            "unmatched": unmatched,
            "skipped": skipped,
            "unmatched_ids": unmatched_ids,
        }


def _derive_yes_team(market_id: str, event_id: str) -> str:
    """
    Derive the YES-side team name from a Kalshi market_id by fuzzy-matching
    the distinctive ticker segments against the teams in the event_id slug.

    e.g. market_id="kalshi:KXNBA-25MAR22-WARRIORS-WIN",
         event_id="nba::2026-03-22::jazz_vs_warriors"
         → returns "warriors"

    Falls back to team_a (first team in slug) when no clear match exists.
    """
    from evmax.agents.cleanup.resolver import _slug_teams, _fuzzy_team_match

    slug_a, slug_b = _slug_teams(event_id)
    if not slug_a:
        return ""

    ticker = market_id.split(":")[-1].upper()
    # Strip: KX prefix + series code (all caps), 6-8 digit dates, common suffixes
    ticker_clean = re.sub(r"^KX[A-Z]+", "", ticker)
    ticker_clean = re.sub(r"\d{6,8}", "", ticker_clean)
    ticker_clean = re.sub(r"[-_](WIN|ML|VS|V|OPP|NO)[-_]?", " ", ticker_clean)
    ticker_clean = ticker_clean.replace("-", " ").replace("_", " ").strip().lower()

    if not ticker_clean:
        return slug_a

    score_a = _fuzzy_team_match(ticker_clean, slug_a)
    score_b = _fuzzy_team_match(ticker_clean, slug_b)

    # Need a meaningful signal to pick team_b; default to team_a otherwise
    if score_b > score_a and score_b >= 30:
        return slug_b
    return slug_a
