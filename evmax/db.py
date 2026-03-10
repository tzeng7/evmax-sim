"""SQLAlchemy async engine + session factory."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from evmax.settings import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=False,
        future=True,
    )


engine = _make_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables."""
    from evmax.models import market, odds, ev_bet, simulated_bet, bankroll  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
