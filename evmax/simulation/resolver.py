"""Bet resolver — settles open paper bets by querying market outcomes.

For Phase 1 (simulation only), resolution is done by checking if a
Kalshi/Polymarket market has been resolved (yes_price → 1.0 or 0.0).
"""

from __future__ import annotations

from datetime import datetime
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
        bet_orm.resolved_at = datetime.utcnow()
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
