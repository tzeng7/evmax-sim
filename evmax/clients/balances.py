"""Live account-balance fetch, venue-agnostic.

Reads the actual deployable cash from each prediction-market venue so the
scanner can size Kelly against real capital instead of a hand-typed
``--bankroll`` number:

  * Kalshi     — ``GET /portfolio/balance``  → available cash (cents → USD)
  * Polymarket — ``GET /v1/account/balances`` → USD ``buyingPower``

Both calls are authenticated (RSA-PSS / Ed25519). Every function here is
FAIL-LOUD-then-SOFT: a missing key, an unscoped key, or a network error
returns ``None`` (never a fabricated number), and the caller falls back to the
manual bankroll. This keeps a balance-fetch outage from silently sizing real
money against a guess, while never blocking an offline / no-credential scan.

The venue string matches the ``venue`` column / ``EVGap.venue`` values:
``"kalshi"`` and ``"polymarket_us"``.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

KALSHI = "kalshi"
POLYMARKET_US = "polymarket_us"
SUPPORTED_VENUES = (KALSHI, POLYMARKET_US)


async def fetch_balance(venue: str) -> Optional[float]:
    """Deployable balance in USD for one venue, or None if unavailable.

    Opens the venue's client in its own async context (the balance endpoints
    need a live ``httpx.AsyncClient``), calls ``get_balance``, and returns the
    dollar figure. None on any failure — the caller decides the fallback.
    """
    v = (venue or "").lower()
    try:
        if v == KALSHI:
            from evmax.clients.kalshi import KalshiClient

            async with KalshiClient() as client:
                return await client.get_balance()
        if v == POLYMARKET_US:
            from evmax.clients.polymarket_us import PolymarketUSClient

            async with PolymarketUSClient() as client:
                return await client.get_balance()
    except Exception as e:  # never let a balance probe crash a scan
        logger.warning("balance_fetch_failed", venue=v, error=str(e))
        return None
    logger.warning("balance_fetch_unknown_venue", venue=venue)
    return None


async def resolve_bankroll(
    bankroll: float, venue: Optional[str]
) -> tuple[float, str]:
    """Resolve the bankroll to size Kelly against, optionally from a live balance.

    When ``venue`` is set, the venue's live deployable balance REPLACES the
    passed ``bankroll`` (the "size against my actual balance" flow the venue
    filter drives). If that balance is unavailable (no credentials / fetch
    error), fall back to the passed ``bankroll`` and flag it — real money is
    never sized against a fabricated figure, and the caller can surface that the
    live balance did not apply.

    Returns ``(bankroll, source)`` where source is one of:
      * ``"manual"``          — no venue requested; the passed value is used
      * ``"live:{venue}"``    — the venue's live balance is used
      * ``"manual_fallback"`` — venue requested but balance unavailable
    """
    if not venue:
        return bankroll, "manual"
    bal = await fetch_balance(venue)
    if bal is None:
        logger.warning("bankroll_live_unavailable", venue=venue, fallback=bankroll)
        return bankroll, "manual_fallback"
    return bal, f"live:{venue.lower()}"


async def fetch_all_balances(
    venues: Optional[list[str]] = None,
) -> dict[str, Optional[float]]:
    """Fetch balances for every requested venue concurrently.

    Returns ``{venue: dollars_or_None}``. Defaults to all supported venues.
    """
    targets = list(venues) if venues else list(SUPPORTED_VENUES)
    results = await asyncio.gather(*(fetch_balance(v) for v in targets))
    return dict(zip(targets, results))
