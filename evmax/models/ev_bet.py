"""EVBet ORM model and Pydantic schema - a flagged +EV opportunity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from evmax.db import Base


class EVBetORM(Base):
    __tablename__ = "ev_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String, index=True)
    event_id: Mapped[str] = mapped_column(String, index=True)
    sector: Mapped[str] = mapped_column(String)
    outcome: Mapped[str] = mapped_column(String)  # "yes" | "no" | label

    market_implied_prob: Mapped[float] = mapped_column(Float)
    true_prob: Mapped[float] = mapped_column(Float)
    payout_decimal: Mapped[float] = mapped_column(Float)
    ev: Mapped[float] = mapped_column(Float)
    edge_pct: Mapped[float] = mapped_column(Float)

    kelly_full: Mapped[float] = mapped_column(Float)
    kelly_fraction: Mapped[float] = mapped_column(Float)
    suggested_units: Mapped[float] = mapped_column(Float)  # fraction of bankroll

    volume_usd: Mapped[float] = mapped_column(Float, default=0.0)
    open_interest_usd: Mapped[float] = mapped_column(Float, default=0.0)
    spread_pct: Mapped[float] = mapped_column(Float, default=0.0)

    scanned_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    event_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class EVBet(BaseModel):
    """A flagged +EV betting opportunity."""

    market_id: str
    event_id: str
    sector: str
    outcome: str  # "yes" | "no" | specific label

    market_implied_prob: float
    true_prob: float
    payout_decimal: float
    ev: float
    edge_pct: float

    kelly_full: float
    kelly_fraction: float
    suggested_units: float

    volume_usd: float = 0.0
    open_interest_usd: float = 0.0
    spread_pct: float = 0.0

    scanned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_date: Optional[datetime] = None

    @property
    def suggested_stake(self) -> float:
        """Suggested stake as fraction of bankroll (capped at 5%)."""
        return self.suggested_units

    def to_orm(self) -> EVBetORM:
        return EVBetORM(**self.model_dump())
