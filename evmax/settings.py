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

    # Polymarket US (gateway API — separate exchange from international
    # Polymarket; no CLOB/Gamma). Public market data needs NO auth; the
    # Ed25519 key pair (keyId=POLYMARKET_API_KEY + secretKey) is only used
    # for trading/WebSocket paths.
    polymarket_api_key: str = ""          # keyId from polymarket.us/developer
    polymarket_us_secret_key: str = ""    # Ed25519 secretKey (shown once)
    polymarket_us_base_url: str = "https://gateway.polymarket.us"
    # Portfolio/account API lives on a DIFFERENT host than the gateway
    # (market-data) API. Balance + positions are served from api.polymarket.us,
    # not gateway.polymarket.us. Ed25519-signed, same key pair.
    polymarket_us_api_base_url: str = "https://api.polymarket.us"
    polymarket_us_enabled: bool = True    # kill-switch for the venue fetch
    # Venue-level shadow firewall (MODEL-9 pattern): until True, every
    # Polymarket US gap is demoted to mode='shadow' at persistence and its
    # Kelly is zeroed — logged for calibration/CLV validation, never sized
    # against the bankroll. Flip after the venue clears the usual shadow
    # gates (n>=30 resolved, CLV >= 0, no matching/parse errors).
    #
    # ``polymarket_us_live`` is the MASTER switch: True clears the firewall for
    # ALL sectors at once. Leave it False and use ``polymarket_us_live_sectors``
    # to clear the firewall for SPECIFIC sectors only — mirroring the per-
    # category granularity of the Kalshi mode registry, so a single sector
    # (e.g. wnba moneyline) can go live on Polymarket while every other Poly
    # sector stays shadow. A sector still only goes live if its category mode
    # resolves to ``live`` upstream (get_mode) — the allowlist refines the
    # venue firewall, it does not override a shadow/disabled category.
    polymarket_us_live: bool = False
    # Comma-separated sector allowlist, e.g. "wnba,tennis". Whitespace and case
    # are normalized; empty means "no per-sector exceptions" (firewall fully up
    # unless the master switch is True). Env: POLYMARKET_US_LIVE_SECTORS.
    #
    # ``wnba`` is cleared as the first Poly-live sector (2026-08-22): WNBA is
    # the only Poly market with genuine model divergence from the sharp line
    # (~3.9pp on moneyline, matching Kalshi's own WNBA divergence) rather than
    # sharp-passthrough, and after the near-tip CLV capture fix its Poly ML
    # sample reached n=82 with mean CLV +0.37pp. Only WNBA MONEYLINE actually
    # goes live: wnba spread/total resolve to shadow upstream via the category's
    # shadow_market_types, so the sector-level clear can't promote them.
    polymarket_us_live_sectors: str = "wnba"

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

    # Price bets net of the venue's trading fee. When True (default), the EV
    # gate and Kelly sizing treat a contract as costing `ask + venue_fee` (the
    # quadratic θ·p·(1-p) fee from evmax/fees.py) instead of the raw ask, so a
    # gross edge the fee eats no longer surfaces as a live play. The fee peaks
    # near p=0.50 (~1.75¢ on Kalshi ⇒ ~3.5pp EV drag) and shrinks toward the
    # extremes, so this is not a flat threshold bump — it re-prices coin-flip
    # and longshot markets the most. Set False to revert to gross-of-fee
    # pricing (offline analysis / backtest A/B). Displayed asks are unchanged;
    # only EV and stake go net.
    fees_in_pricing: bool = True

    # Minimum cumulative Kalshi volume to count a market as bettable.
    # Default 0 = disabled. Set per-sector or globally if you want to filter
    # out low-volume markets, but be aware that thin-volume sectors (WNBA,
    # mid-week MLB, early-week NFL) are where soft-vs-sharp gaps tend to be
    # widest — raising this floor sacrifices edge opportunities in exchange
    # for variance reduction. spread_pct < 0.5% already drops empty books.
    min_volume_usd: float = Field(default=0.0, ge=0.0)

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

    # Correlation-aware joint Kelly sizing (per-event). When enabled, legs that
    # share a game outcome (ML/spread on the same margin, over/under on the same
    # total) are sized jointly via a Gaussian-copula log-growth optimization
    # instead of independently + the proportional exposure guard. The per-event
    # gross cap is variance-scaled: it expands from exposure_guard (0.08) toward
    # joint_kelly_max_gross_pct only as hedging cuts portfolio variance below the
    # naive independent sum; redundant (positively correlated) legs stay pinned
    # at the base cap. Single-leg events reduce exactly to fractional Kelly.
    # See evmax/ev/joint_kelly.py and AgentCoordinator._apply_joint_kelly.
    joint_kelly_enabled: bool = False
    joint_kelly_max_gross_pct: float = Field(default=0.15, ge=0.08, le=0.5)
    joint_kelly_rho_margin_total: float = Field(default=0.0, ge=-0.9, le=0.9)
    joint_kelly_samples: int = Field(default=20000, ge=2000, le=200000)

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
        return missing

    def polymarket_us_live_sector_set(self) -> set[str]:
        """Parsed, normalized allowlist from ``polymarket_us_live_sectors``.

        Lowercased, whitespace-trimmed, empties dropped. Cheap to recompute —
        the string is tiny and get_settings() is cached.
        """
        raw = self.polymarket_us_live_sectors or ""
        return {s.strip().lower() for s in raw.split(",") if s.strip()}

    @property
    def discord_bot_configured(self) -> bool:
        """True when the bot token and at least one target (channel or DM user) are set."""
        return bool(self.discord_bot_token and (self.discord_channel_id or self.discord_dm_user_id))

    def discord_allowed_users(self) -> frozenset[int]:
        """Parsed ``DISCORD_ALLOWED_USER_IDS`` (empty = no restriction)."""
        out: set[int] = set()
        for tok in (self.discord_allowed_user_ids or "").split(","):
            tok = tok.strip()
            if tok.isdigit():
                out.add(int(tok))
        return frozenset(out)

    def polymarket_us_sector_live(self, sector: Optional[str]) -> bool:
        """Whether the Polymarket US venue firewall is CLEARED for ``sector``.

        True when the master switch ``polymarket_us_live`` is on (all sectors),
        or when ``sector`` is in the per-sector allowlist. A gap only reaches a
        live persistence if its category mode also resolves to ``live``
        upstream — this method only governs the venue firewall, not the mode.
        """
        if self.polymarket_us_live:
            return True
        if not sector:
            return False
        return sector.lower() in self.polymarket_us_live_sector_set()

    # Matching
    fuzzy_threshold: int = 88  # rapidfuzz score threshold

    # Push notifications (Slack and/or Discord webhooks)
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    notification_min_ev_pct: float = 5.0  # only notify when EV >= this %

    # Discord BOT (evmax/discord_bot/) — distinct from the plain-text webhook
    # above. With a bot token + channel id, every scan cycle posts its play
    # list to the channel as the dashboard's Scan Results table (same rows,
    # order, columns — evmax.web.playlist), operational alerts post as colored
    # embeds, and `evmax discord run` serves the /scan /plays /settled /status
    # slash commands. Ids are Discord snowflakes; kept as strings.
    discord_bot_token: str = ""
    discord_channel_id: str = ""        # channel receiving the scan feed + alerts (optional if DM set)
    discord_dm_user_id: str = ""        # your user id: feed + alerts go to your DMs (bot must share a server with you)
    discord_guild_id: str = ""          # optional: guild-scoped slash-command sync (instant)
    discord_allowed_user_ids: str = ""  # comma-separated user ids allowed to run commands; empty = any member
    discord_scan_feed: bool = True      # post each scan cycle's play table to the channel
    discord_post_empty_scans: bool = False  # also post cycles with 0 plays (a scan heartbeat)

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
