"""PredictionMarket ORM model and Pydantic schema."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import DateTime, Enum, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from evmax.db import Base


class MarketSource(str, enum.Enum):
    kalshi = "kalshi"
    polymarket = "polymarket"


class MarketType(str, enum.Enum):
    moneyline = "moneyline"
    spread = "spread"
    total = "total"
    map_handicap = "map_handicap"
    series_winner = "series_winner"
    other = "other"


class PredictionMarketORM(Base):
    __tablename__ = "prediction_markets"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "{source}:{ticker}"
    source: Mapped[str] = mapped_column(Enum(MarketSource))
    sector: Mapped[str] = mapped_column(String)
    market_type: Mapped[str] = mapped_column(Enum(MarketType), default=MarketType.moneyline)
    title: Mapped[str] = mapped_column(String, default="")
    ticker: Mapped[str] = mapped_column(String, default="")

    yes_price: Mapped[float] = mapped_column(Float)
    no_price: Mapped[float] = mapped_column(Float)
    volume_usd: Mapped[float] = mapped_column(Float, default=0.0)
    open_interest_usd: Mapped[float] = mapped_column(Float, default=0.0)

    team_home: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    team_away: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    event_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # matched event key
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PredictionMarket(BaseModel):
    """Pydantic schema for a prediction market."""

    id: str
    source: MarketSource
    sector: str
    market_type: MarketType = MarketType.moneyline
    title: str = ""
    ticker: str = ""

    yes_price: float  # 0.0–1.0
    no_price: float  # 0.0–1.0
    volume_usd: float = 0.0
    open_interest_usd: float = 0.0

    team_home: Optional[str] = None
    team_away: Optional[str] = None
    line: Optional[float] = None
    event_date: Optional[datetime] = None
    event_id: Optional[str] = None
    # Which team the YES side represents (normalized alias name).
    # None means default: YES = home/outcome_a side.
    yes_team: Optional[str] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("yes_price", "no_price")
    @classmethod
    def price_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Price must be 0.0–1.0, got {v}")
        return v

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as fraction of mid-price (proxy for liquidity cost)."""
        mid = (self.yes_price + (1 - self.no_price)) / 2
        spread = abs(self.yes_price - (1 - self.no_price))
        return spread / mid if mid > 0 else 1.0

    def to_orm(self) -> PredictionMarketORM:
        return PredictionMarketORM(**self.model_dump())
