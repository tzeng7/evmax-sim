"""BankrollSnapshot ORM model and Pydantic schema."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from evmax.db import Base


class BankrollSnapshotORM(Base):
    __tablename__ = "bankroll_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    balance_usd: Mapped[float] = mapped_column(Float)
    total_wagered_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    open_bets_count: Mapped[int] = mapped_column(Integer, default=0)
    won_bets_count: Mapped[int] = mapped_column(Integer, default=0)
    lost_bets_count: Mapped[int] = mapped_column(Integer, default=0)
    roi_pct: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str] = mapped_column(String, default="")
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class BankrollSnapshot(BaseModel):
    """Point-in-time snapshot of the simulation bankroll."""

    balance_usd: float
    total_wagered_usd: float = 0.0
    total_pnl_usd: float = 0.0
    open_bets_count: int = 0
    won_bets_count: int = 0
    lost_bets_count: int = 0
    roi_pct: float = 0.0
    notes: str = ""
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_orm(self) -> BankrollSnapshotORM:
        return BankrollSnapshotORM(**self.model_dump())
