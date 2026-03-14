"""GameResult ORM + Pydantic schema — records actual game outcomes.

Used for:
  - Model calibration (did our predicted probabilities match actual win rates?)
  - Training future regression/Elo models with historical outcomes
  - Performance tracking per sector and model

Populated automatically by BetResolver when Kalshi markets settle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from evmax.db import Base


class GameResultORM(Base):
    __tablename__ = "game_results"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_game_result_event_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, index=True)
    sector: Mapped[str] = mapped_column(String, index=True)

    team_home: Mapped[str] = mapped_column(String)
    team_away: Mapped[str] = mapped_column(String)

    # "home" | "away" | "draw"
    winner: Mapped[str] = mapped_column(String)

    # Optional score — populated when available (e.g. future Sports-Reference backfill)
    score_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    game_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Where this result came from: "kalshi_resolved", "manual", "sports_reference"
    source: Mapped[str] = mapped_column(String, default="kalshi_resolved")

    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GameResult(BaseModel):
    """Actual game outcome for model calibration and tracking."""

    id: Optional[int] = None
    event_id: str
    sector: str

    team_home: str
    team_away: str
    winner: str  # "home" | "away" | "draw"

    score_home: Optional[float] = None
    score_away: Optional[float] = None
    game_date: Optional[datetime] = None
    source: str = "kalshi_resolved"
    recorded_at: datetime = datetime.utcnow()

    def to_orm(self) -> GameResultORM:
        data = self.model_dump(exclude={"id"})
        return GameResultORM(**data)
