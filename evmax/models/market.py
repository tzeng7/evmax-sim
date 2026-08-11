"""PredictionMarket ORM model and Pydantic schema."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import DateTime, Enum, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from evmax.db import Base


# Player-prop event_ids embed this marker:
# "{sector}::{date}::prop::{player}::{stat}::{threshold}". It's how prop gaps
# are routed to prop_observations (vs game markets to ev_predictions). Defined
# here (a dependency leaf imported by agents/web/cli alike) so the check has a
# single source instead of a copy-pasted literal at every call site.
PROP_MARKER = "::prop::"


def is_prop_event(event_id: Optional[str]) -> bool:
    """True if an event_id is a player prop (vs a game market)."""
    return bool(event_id) and PROP_MARKER in event_id


class MarketSource(str, enum.Enum):
    kalshi = "kalshi"
    polymarket = "polymarket"          # legacy international CLOB (never wired)
    polymarket_us = "polymarket_us"    # Polymarket US (CFTC-regulated exchange)


# Short human-readable venue labels for text surfaces (CLI tables, logs).
# The dashboard renders logos instead — see evmax/web/app.py.
VENUE_LABELS: dict[str, str] = {
    MarketSource.kalshi.value: "Kalshi",
    MarketSource.polymarket.value: "Poly",
    MarketSource.polymarket_us.value: "PolyUS",
}


def venue_label(venue: Optional[str]) -> str:
    """Display label for a venue string; unknown/missing venues fall back
    to 'Kalshi' (every row predating the venue column is a Kalshi row)."""
    return VENUE_LABELS.get(venue or MarketSource.kalshi.value, venue or "Kalshi")


class MarketType(str, enum.Enum):
    moneyline = "moneyline"
    spread = "spread"
    total = "total"
    # Knockout "to advance" (winner incl. extra time / penalties) — 2-way even
    # in soccer-like sectors. Distinct from moneyline, which for soccer/worldcup
    # is the 3-way REGULATION result (knockout games can still draw after 90').
    advance = "advance"
    map_handicap = "map_handicap"
    series_winner = "series_winner"
    player_prop = "player_prop"
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
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class PredictionMarket(BaseModel):
    """Pydantic schema for a prediction market."""

    id: str
    source: MarketSource
    sector: str
    market_type: MarketType = MarketType.moneyline
    title: str = ""
    ticker: str = ""

    yes_price: float  # 0.0–1.0 — best YES ask (cost to buy YES as a taker)
    no_price: float  # 0.0–1.0 — best NO ask (cost to buy NO as a taker)
    # Best resting bids (highest price a buyer is offering) — what a MAKER would
    # join or improve to rest a limit order without crossing the spread. None
    # when the venue doesn't expose a bid ladder (e.g. Polymarket US quotes a
    # single price per side; callers derive the bid as 1 − opposite-side ask).
    yes_bid: Optional[float] = None  # 0.0–1.0
    no_bid: Optional[float] = None   # 0.0–1.0
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
    # Player prop fields (only set when market_type == player_prop)
    player_name: Optional[str] = None   # normalized player name
    stat_type: Optional[str] = None     # e.g. "points", "rebounds", "assists"
    threshold: Optional[float] = None   # e.g. 24.5
    # Upstream structured tournament label from Kalshi
    # event.product_metadata.competition (e.g. "ATP Munich", "WTA Rouen").
    # Used by TennisModelAgent surface resolver. None for non-tennis sectors.
    competition: Optional[str] = None
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
        return PredictionMarketORM(
            **self.model_dump(exclude={"competition"})
        )
