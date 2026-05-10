"""Application settings loaded from environment / .env file."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
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
    ev_threshold: float = Field(default=0.02, ge=0.0, le=1.0)  # 2%

    # Minimum market volume to place a simulated bet (filters stale/illiquid markets)
    min_volume_usd: float = Field(default=500.0, ge=0.0)

    # Heavy-chalk filter for game-level markets (moneyline / spread / total).
    # Skip any side whose ask is above this. Default 0.90 → never stake $9 to
    # win <$1 even when the model says it's +EV. Reason: at this price level,
    # tiny calibration drift swamps the edge (a 1pp miss on a 95% market is a
    # 20% miss on the 5% NO side that holds your money), one loss undoes
    # 10+ wins, and Kalshi liquidity is usually too thin to fill at the quote.
    # Player props are unaffected — they go through a separate evaluator.
    chalk_price_ceiling: float = Field(default=0.90, ge=0.5, le=1.0)

    # Kelly multiplier applied to a same-side correlated bet. When two +EV
    # gaps share the same `yes_team` on the same base event (ML + same-team
    # spread, or alt-spread stacks), the second-best one's Kelly is scaled by
    # this factor before consuming the per-game exposure budget. Default 0.5
    # roughly cancels the over-allocation from naive independent Kelly on
    # ρ ≈ 0.8 correlated positions — at full Kelly each you're effectively
    # ~1.7x Kelly on a single position, which sits past the geometric-growth
    # peak. Set to 1.0 to disable the discount.
    same_side_kelly_discount: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("ev_threshold")
    @classmethod
    def ev_threshold_sane(cls, v: float) -> float:
        if v > 0.5:
            raise ValueError(
                f"ev_threshold={v} is > 50% which would filter almost all bets. "
                "Typical values are 0.02–0.10."
            )
        return v

    def warn_missing_keys(self) -> list[str]:
        """Return list of missing required API keys (non-fatal — allows read-only use)."""
        missing = []
        if not self.kalshi_api_key_id:
            missing.append("KALSHI_API_KEY_ID (required for Kalshi WebSocket + trading)")
        if not self.kalshi_private_key_path:
            # PEM key only needed for trading and WebSocket auth.
            # Market reads are unauthenticated — not a blocker for scanning.
            missing.append("KALSHI_PRIVATE_KEY_PATH (optional — needed for WS price refresh + trading, not scanning)")
        if not self.the_odds_api_key:
            missing.append("THE_ODDS_API_KEY (required to fetch Pinnacle sharp lines)")
        return missing

    # Matching
    fuzzy_threshold: int = 88  # rapidfuzz score threshold

    # Push notifications (Slack and/or Discord webhooks)
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    notification_min_ev_pct: float = 5.0  # only notify when EV >= this %

    # Kalshi WebSocket (real-time orderbook prices)
    kalshi_ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    kalshi_ws_snapshot_timeout: float = 8.0  # seconds to collect all orderbook snapshots
    kalshi_ws_enabled: bool = True  # kill-switch; set False to force REST-only

    # Response cache — reuses recent API responses to avoid redundant fetches.
    # Default 120s covers back-to-back scans. Set EVMAX_CACHE_TTL_SECS=3600 for dev.
    # Set EVMAX_OFFLINE=true to never hit live APIs (fails if no cache exists)
    cache_ttl_secs: int = Field(default=120, ge=0)  # 0 = disabled
    offline_mode: bool = False


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
