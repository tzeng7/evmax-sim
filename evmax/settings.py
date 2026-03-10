"""Application settings loaded from environment / .env file."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Kalshi
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"

    # TheOddsAPI (Pinnacle lines)
    the_odds_api_key: str = ""
    the_odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # Database
    database_url: str = "sqlite+aiosqlite:///./evmax.db"

    # Simulation
    initial_bankroll: float = Field(default=10_000.0, ge=0)
    max_kelly_fraction: float = Field(default=0.05, ge=0.001, le=1.0)
    min_kelly_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    min_bet_usd: float = Field(default=1.0, ge=0)
    max_odds_age_s: int = Field(default=120, ge=1)
    spread_line_tolerance: float = Field(default=1.5, ge=0)

    # Logging
    log_level: str = "INFO"

    # EV threshold
    ev_threshold: float = 0.02  # 2%

    # Minimum market volume to place a simulated bet (filters stale/illiquid markets)
    min_volume_usd: float = 500.0

    # Matching
    fuzzy_threshold: int = 88  # rapidfuzz score threshold


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
