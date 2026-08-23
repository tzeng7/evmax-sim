"""Live account-balance fetch, venue-agnostic.

Reads live account balances from each prediction-market venue so the scanner
can size Kelly against real capital instead of a hand-typed ``--bankroll``
number. Two figures per venue, for two different jobs:

  * TOTAL WEALTH (cash + open-position value) — the Kelly base, via
    :func:`fetch_balance` / :func:`fetch_all_balances`.
  * DEPLOYABLE CASH — the per-venue fundability-cap base, via
    :func:`fetch_cash_balance` / :func:`fetch_cash_balances`. A new bet is
    funded from cash, not from the value locked in open positions, so the
    cash cap must size against this, NOT total wealth.

  * Kalshi     — ``GET /portfolio/balance``   (RSA-PSS signed)
  * Polymarket — ``GET /v1/account/balances`` (Ed25519 signed)

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
from dataclasses import dataclass, field
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


async def fetch_cash_balance(venue: str) -> Optional[float]:
    """DEPLOYABLE CASH in USD for one venue, or None if unavailable.

    Distinct from :func:`fetch_balance` (total wealth): a new bet is funded from
    cash, so the per-venue fundability cap sizes against this. Same fail-soft
    contract — None on any failure, so the caller can skip that venue's cap
    rather than block on a missing figure.
    """
    v = (venue or "").lower()
    try:
        if v == KALSHI:
            from evmax.clients.kalshi import KalshiClient

            async with KalshiClient() as client:
                return await client.get_cash_balance()
        if v == POLYMARKET_US:
            from evmax.clients.polymarket_us import PolymarketUSClient

            async with PolymarketUSClient() as client:
                return await client.get_cash_balance()
    except Exception as e:  # never let a balance probe crash a scan
        logger.warning("cash_balance_fetch_failed", venue=v, error=str(e))
        return None
    logger.warning("cash_balance_fetch_unknown_venue", venue=venue)
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


async def fetch_cash_balances(
    venues: Optional[list[str]] = None,
) -> dict[str, Optional[float]]:
    """Fetch DEPLOYABLE CASH for every requested venue concurrently.

    Returns ``{venue: dollars_or_None}``. Defaults to all supported venues.
    The per-venue fundability cap consumes this; the Kelly base uses
    :func:`fetch_all_balances` (total wealth) instead.
    """
    targets = list(venues) if venues else list(SUPPORTED_VENUES)
    results = await asyncio.gather(*(fetch_cash_balance(v) for v in targets))
    return dict(zip(targets, results))


@dataclass
class BankrollPlan:
    """Everything a scan needs from a venue selection.

    Fields:
      * ``bankroll``        — the Kelly base (TOTAL WEALTH): the passed value
        (manual), one venue's wealth (single), or the sum of the selected
        venues' wealth (combined). Never a fabricated figure — falls back to
        the passed bankroll if a live balance is unavailable.
      * ``source``          — ``manual`` | ``live:{venue}`` | ``live:both`` |
        ``manual_fallback`` (echoed to the UI so the user sees what was used).
      * ``selected_venues`` — the venue restriction to pass to the coordinator.
        ``None`` for manual (no restriction — all firewall-cleared venues show).
      * ``cash_by_venue``   — DEPLOYABLE CASH per selected venue for the
        fundability cap. Empty for manual (cash unknown → no cap) and omits any
        venue whose cash was unavailable (that venue is simply not capped).
    """

    bankroll: float
    source: str
    selected_venues: Optional[list[str]] = None
    cash_by_venue: dict[str, float] = field(default_factory=dict)


async def resolve_bankroll_plan(
    bankroll: float, selection: Optional[str]
) -> BankrollPlan:
    """Resolve bankroll base + venue scoping + per-venue cash from a selection.

    ``selection`` mirrors the dashboard's venue dropdown:
      * ``""`` / ``None`` / ``"manual"`` — manual bankroll, no scoping, no cash cap.
      * ``"kalshi"`` / ``"polymarket_us"`` — that venue only: bankroll = its total
        wealth, plays scoped to it, cash cap against its deployable cash.
      * ``"both"`` / ``"all"`` — combined: bankroll = sum of both venues' total
        wealth, both venues in scope, cash cap per venue.

    Fail-soft throughout: any unavailable balance degrades to the manual
    bankroll (``manual_fallback``) and simply omits that venue's cash cap —
    real money is never sized against a fabricated figure.
    """
    sel = (selection or "").strip().lower()
    if not sel or sel == "manual":
        return BankrollPlan(bankroll, "manual", None, {})

    if sel in ("both", "all"):
        venues = list(SUPPORTED_VENUES)
        totals = await fetch_all_balances(venues)
        cash = await fetch_cash_balances(venues)
        cash_by = {v: c for v, c in cash.items() if c is not None}
        live_totals = {v: t for v, t in totals.items() if t is not None}
        if not live_totals:
            logger.warning("bankroll_plan_combined_unavailable", fallback=bankroll)
            return BankrollPlan(bankroll, "manual_fallback", venues, cash_by)
        return BankrollPlan(round(sum(live_totals.values()), 2), "live:both", venues, cash_by)

    if sel not in SUPPORTED_VENUES:
        logger.warning("bankroll_plan_unknown_selection", selection=selection)
        return BankrollPlan(bankroll, "manual", None, {})

    total = await fetch_balance(sel)
    c = await fetch_cash_balance(sel)
    cash_by = {sel: c} if c is not None else {}
    if total is None:
        logger.warning("bankroll_plan_single_unavailable", venue=sel, fallback=bankroll)
        return BankrollPlan(bankroll, "manual_fallback", [sel], cash_by)
    return BankrollPlan(total, f"live:{sel}", [sel], cash_by)
