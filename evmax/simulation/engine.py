"""Simulation engine — paper trading.

Places and tracks simulated bets based on flagged EVBets.
Manages bankroll state and records snapshots.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evmax.db import AsyncSessionLocal
from evmax.ev.kelly import compute_kelly
from evmax.models.bankroll import BankrollSnapshot, BankrollSnapshotORM
from evmax.models.ev_bet import EVBet, EVBetORM
from evmax.models.simulated_bet import BetStatus, SimulatedBet, SimulatedBetORM
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)


class SimulationEngine:
    """
    Paper trading engine.

    Maintains a simulated bankroll, places bets from EVBet signals,
    and records all activity to the database.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._bankroll: Optional[float] = None

    async def _get_bankroll(self, session: AsyncSession) -> float:
        """Load current bankroll from latest snapshot or use initial value."""
        if self._bankroll is not None:
            return self._bankroll

        result = await session.execute(
            select(BankrollSnapshotORM).order_by(BankrollSnapshotORM.id.desc()).limit(1)
        )
        latest = result.scalar_one_or_none()

        if latest:
            self._bankroll = latest.balance_usd
        else:
            self._bankroll = self._settings.initial_bankroll
        return self._bankroll

    async def place_bet(
        self,
        ev_bet: EVBet,
        ev_bet_db_id: int,
        session: Optional[AsyncSession] = None,
    ) -> Optional[SimulatedBet]:
        """
        Place a paper bet for a flagged EV opportunity.

        Args:
            ev_bet: The EV opportunity to bet on.
            ev_bet_db_id: DB ID of the EVBet record.
            session: Optional DB session (creates one if not provided).

        Returns:
            SimulatedBet record, or None if bet couldn't be placed.
        """
        async def _place(sess: AsyncSession) -> Optional[SimulatedBet]:
            # Skip if we already have an open bet on this market
            existing = await sess.execute(
                select(SimulatedBetORM).where(
                    SimulatedBetORM.market_id == ev_bet.market_id,
                    SimulatedBetORM.status == BetStatus.open,
                )
            )
            if existing.scalar_one_or_none():
                logger.debug("sim_skip_duplicate", market_id=ev_bet.market_id)
                return None

            bankroll = await self._get_bankroll(sess)

            # Compute stake
            stake_fraction = ev_bet.suggested_units
            stake_usd = bankroll * stake_fraction

            # Floor stake to minimum bet size; skip if Kelly says zero
            min_bet = self._settings.min_bet_usd
            if stake_usd <= 0:
                return None
            stake_usd = max(stake_usd, min_bet)

            # Cap stake at available bankroll
            stake_usd = min(stake_usd, bankroll)

            profit_if_win = stake_usd * (ev_bet.payout_decimal - 1.0)

            sim_bet = SimulatedBet(
                ev_bet_id=ev_bet_db_id,
                market_id=ev_bet.market_id,
                event_id=ev_bet.event_id,
                sector=ev_bet.sector,
                outcome=ev_bet.outcome,
                stake_usd=stake_usd,
                odds_decimal=ev_bet.payout_decimal,
                potential_profit_usd=profit_if_win,
                bankroll_before=bankroll,
                event_date=ev_bet.event_date,
            )

            orm = sim_bet.to_orm()
            sess.add(orm)
            await sess.flush()
            sim_bet = sim_bet.model_copy(update={"id": orm.id})

            # Deduct stake from bankroll immediately (reserved)
            self._bankroll = bankroll - stake_usd
            await self._save_snapshot(sess)

            await sess.commit()
            logger.info(
                "sim_bet_placed",
                market_id=ev_bet.market_id,
                stake_usd=stake_usd,
                ev=ev_bet.ev,
                bankroll_after=self._bankroll,
            )
            return sim_bet

        if session:
            return await _place(session)
        else:
            async with AsyncSessionLocal() as sess:
                return await _place(sess)

    async def _save_snapshot(self, session: AsyncSession) -> None:
        """Record a bankroll snapshot."""
        # Gather stats
        result = await session.execute(select(SimulatedBetORM))
        all_bets = result.scalars().all()

        open_count = sum(1 for b in all_bets if b.status == BetStatus.open)
        won_count = sum(1 for b in all_bets if b.status == BetStatus.won)
        lost_count = sum(1 for b in all_bets if b.status == BetStatus.lost)
        total_wagered = sum(b.stake_usd for b in all_bets if b.status != BetStatus.void)
        total_pnl = sum(b.pnl_usd for b in all_bets if b.status in (BetStatus.won, BetStatus.lost))

        roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0.0

        # Balance = initial bankroll + all resolved P&L - all still-open stakes
        # This is correct even when called from a fresh engine instance (self._bankroll=None).
        open_stakes = sum(b.stake_usd for b in all_bets if b.status == BetStatus.open)
        balance = self._settings.initial_bankroll + total_pnl - open_stakes

        snapshot = BankrollSnapshot(
            balance_usd=balance,
            total_wagered_usd=total_wagered,
            total_pnl_usd=total_pnl,
            open_bets_count=open_count,
            won_bets_count=won_count,
            lost_bets_count=lost_count,
            roi_pct=roi,
        )
        session.add(snapshot.to_orm())

    async def get_open_bets(self) -> list[SimulatedBet]:
        """Retrieve all open simulated bets."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(SimulatedBetORM).where(SimulatedBetORM.status == BetStatus.open)
            )
            rows = result.scalars().all()
            return [SimulatedBet.model_validate(r.__dict__) for r in rows]
