"""SimulatedBet ORM model and Pydantic schema - a paper trade."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Enum, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from evmax.db import Base


class BetStatus(str, enum.Enum):
    open = "open"
    won = "won"
    lost = "lost"
    void = "void"


class SimulatedBetORM(Base):
    __tablename__ = "simulated_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ev_bet_id: Mapped[int] = mapped_column(Integer, index=True)
    market_id: Mapped[str] = mapped_column(String, index=True)
    event_id: Mapped[str] = mapped_column(String, index=True)
    sector: Mapped[str] = mapped_column(String)
    outcome: Mapped[str] = mapped_column(String)

    stake_usd: Mapped[float] = mapped_column(Float)
    odds_decimal: Mapped[float] = mapped_column(Float)
    potential_profit_usd: Mapped[float] = mapped_column(Float)

    status: Mapped[str] = mapped_column(Enum(BetStatus), default=BetStatus.open)
    pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    bankroll_before: Mapped[float] = mapped_column(Float)
    bankroll_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    placed_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    event_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SimulatedBet(BaseModel):
    """A paper trade placed by the simulation engine."""

    id: Optional[int] = None
    ev_bet_id: int
    market_id: str
    event_id: str
    sector: str
    outcome: str

    stake_usd: float
    odds_decimal: float
    potential_profit_usd: float

    status: BetStatus = BetStatus.open
    pnl_usd: float = 0.0
    bankroll_before: float = 0.0
    bankroll_after: Optional[float] = None

    placed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    event_date: Optional[datetime] = None

    def to_orm(self) -> SimulatedBetORM:
        data = self.model_dump(exclude={"id"})
        return SimulatedBetORM(**data)
