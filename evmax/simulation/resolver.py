"""Bet resolver — settles open paper bets by querying market outcomes.

For Phase 1 (simulation only), resolution is done by checking if a
Kalshi/Polymarket market has been resolved (yes_price → 1.0 or 0.0).

When a market settles, the resolver also records a GameResult so that
model calibration and future training data accumulates automatically.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from evmax.db import AsyncSessionLocal
from evmax.models.game_result import GameResult, GameResultORM
from evmax.models.simulated_bet import BetStatus, SimulatedBet, SimulatedBetORM

logger = structlog.get_logger(__name__)

# Match Kalshi ticker outcome code: last segment after '-', strip trailing digits
_TICKER_OUTCOME_RE = re.compile(r"-([A-Z0-9]+)$", re.IGNORECASE)
_TRAILING_DIGITS_RE = re.compile(r"\d+$")


def _extract_yes_team_from_market_id(market_id: str, sector: str) -> Optional[str]:
    """
    Extract and normalize the YES-side team name from a Kalshi market_id.

    e.g. "kalshi:KXNCAAMBGAME-26MAR13KANDU-DUK" → normalizes "DUK" → "duke"
    """
    ticker = market_id.removeprefix("kalshi:").removeprefix("polymarket:")
    m = _TICKER_OUTCOME_RE.search(ticker.upper())
    if not m:
        return None
    outcome_code = _TRAILING_DIGITS_RE.sub("", m.group(1)).lower()
    if not outcome_code:
        return None
    from evmax.matching.normalizer import NameNormalizer
    return NameNormalizer(sector).normalize(outcome_code)


def _parse_teams_from_event_id(event_id: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse team_home and team_away from a canonical event_id.

    Format: "{sector}::{date}::{team_home}_vs_{team_away}"
    Returns (team_home, team_away) or (None, None) if not parseable.
    """
    parts = event_id.split("::")
    if len(parts) < 3:
        return None, None
    matchup = parts[2]
    vs_parts = matchup.split("_vs_", 1)
    if len(vs_parts) != 2:
        return None, None
    return vs_parts[0].replace("_", " "), vs_parts[1].replace("_", " ")


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

        Also records a GameResult when the market has definitively settled
        (yes_price >= 0.99 or <= 0.01).

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
            yes_won = True
        elif current_yes_price <= 0.01:
            yes_won = False
        else:
            return []  # Market not yet resolved

        resolved = []
        for bet in open_bets:
            won = (bet.outcome.lower() == "yes") == yes_won
            resolved.append(await self.resolve_bet(bet, won, session))

        # Record game result using the first resolved bet's event data
        if resolved:
            first_bet = open_bets[0]
            await self._try_record_game_result(
                market_id=market_id,
                event_id=first_bet.event_id,
                sector=first_bet.sector,
                yes_won=yes_won,
                event_date=first_bet.event_date,
                session=session,
            )

        return resolved

    async def _try_record_game_result(
        self,
        market_id: str,
        event_id: str,
        sector: str,
        yes_won: bool,
        event_date: Optional[datetime],
        session: AsyncSession,
    ) -> None:
        """
        Record a GameResult row when a market settles.

        Silently skips if:
        - Teams cannot be parsed from event_id
        - YES team cannot be extracted from market_id
        - A result for this event_id already exists (idempotent)
        """
        team_home, team_away = _parse_teams_from_event_id(event_id)
        if not team_home or not team_away:
            logger.debug("game_result_skip_no_teams", event_id=event_id)
            return

        yes_team = _extract_yes_team_from_market_id(market_id, sector)
        if not yes_team:
            logger.debug("game_result_skip_no_yes_team", market_id=market_id)
            return

        # Determine winner
        if yes_won:
            winner = "home" if yes_team == team_home else "away"
        else:
            # YES side lost → the other team won
            winner = "away" if yes_team == team_home else "home"

        game_result = GameResult(
            event_id=event_id,
            sector=sector,
            team_home=team_home,
            team_away=team_away,
            winner=winner,
            game_date=event_date,
            source="kalshi_resolved",
        )

        try:
            # INSERT OR IGNORE so duplicate settlements don't error
            stmt = (
                sqlite_insert(GameResultORM)
                .values(**game_result.model_dump(exclude={"id"}))
                .on_conflict_do_nothing(index_elements=["event_id"])
            )
            await session.execute(stmt)
            await session.commit()
            logger.info(
                "game_result_recorded",
                event_id=event_id,
                sector=sector,
                winner=winner,
                home=team_home,
                away=team_away,
            )
        except Exception as e:
            logger.warning("game_result_record_failed", event_id=event_id, error=str(e))

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
