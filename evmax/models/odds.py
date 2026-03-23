"""SharpOdds ORM model and Pydantic schema."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import DateTime, Enum, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from evmax.db import Base


class SharpBook(str, enum.Enum):
    pinnacle = "pinnacle"


class SharpOddsORM(Base):
    __tablename__ = "sharp_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, index=True)
    book: Mapped[str] = mapped_column(Enum(SharpBook))
    sector: Mapped[str] = mapped_column(String)

    outcome_a_label: Mapped[str] = mapped_column(String, default="")
    outcome_b_label: Mapped[str] = mapped_column(String, default="")
    outcome_a_decimal: Mapped[float] = mapped_column(Float)
    outcome_b_decimal: Mapped[float] = mapped_column(Float)
    outcome_draw_decimal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    true_prob_a: Mapped[float] = mapped_column(Float)
    true_prob_b: Mapped[float] = mapped_column(Float)
    true_prob_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    margin: Mapped[float] = mapped_column(Float)

    event_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SharpOdds(BaseModel):
    """Pydantic schema for sharp sportsbook odds."""

    event_id: str
    book: SharpBook
    sector: str

    outcome_a_label: str = ""
    outcome_b_label: str = ""
    outcome_a_decimal: float
    outcome_b_decimal: float
    outcome_draw_decimal: Optional[float] = None

    true_prob_a: float = 0.0
    true_prob_b: float = 0.0
    true_prob_draw: Optional[float] = None
    margin: float = 0.0

    # Spread-specific: line from outcome_a's perspective (e.g. -7.5 means team_a -7.5)
    spread_line: Optional[float] = None

    # Totals-specific: devigged over/under probabilities and the posted line
    total_line: Optional[float] = None
    true_prob_over: Optional[float] = None
    true_prob_under: Optional[float] = None

    # Player prop fields (reuses true_prob_over/true_prob_under for over/under probs)
    prop_player_name: Optional[str] = None  # normalized player name
    prop_stat_type: Optional[str] = None    # "points", "rebounds", "assists", etc.

    event_date: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def probs_sum_to_one(self) -> "SharpOdds":
        # Moneyline / spread validation
        total = self.true_prob_a + self.true_prob_b + (self.true_prob_draw or 0.0)
        if self.true_prob_a > 0 and abs(total - 1.0) > 0.01:
            raise ValueError(f"True probs must sum to ~1.0, got {total}")
        # Totals validation
        if self.true_prob_over is not None and self.true_prob_under is not None:
            ot = self.true_prob_over + self.true_prob_under
            if abs(ot - 1.0) > 0.01:
                raise ValueError(f"true_prob_over + true_prob_under must sum to ~1.0, got {ot}")
        return self

    def to_orm(self) -> SharpOddsORM:
        return SharpOddsORM(**self.model_dump())
