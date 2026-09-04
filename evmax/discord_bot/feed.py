"""The Discord scan feed: post a coordinator cycle's play list to the channel.

Built from ``evmax.web.playlist`` so the rows are, in the same order, what
the dashboard's Scan Results panel would show for the same cycle:
``cycle.plays()`` (full-blend, EV-descending) → best-execution collapse →
per-row dict → scan-view filters (today+tomorrow window, no map handicaps,
no props, nothing already placed).

:func:`suppress_scan_feed` lets a caller that will render the cycle itself
(the ``/scan`` slash command replies with the table) stop the coordinator's
notifier from posting the same table a second time. It is a ``ContextVar``
so it follows the task across ``await`` and into ``asyncio.to_thread``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterator, Optional

import structlog

from evmax.discord_bot.embeds import scan_result_embeds
from evmax.web.playlist import dashboard_play_dicts, filter_scan_view

if TYPE_CHECKING:
    from evmax.agents.coordinator import CycleResult
    from evmax.discord_bot.client import DiscordBotClient

logger = structlog.get_logger(__name__)

_SUPPRESSED: ContextVar[bool] = ContextVar("evmax_discord_scan_feed_suppressed", default=False)


@contextmanager
def suppress_scan_feed() -> Iterator[None]:
    """Within this block the coordinator's notifier does not post the scan feed."""
    token = _SUPPRESSED.set(True)
    try:
        yield
    finally:
        _SUPPRESSED.reset(token)


def scan_feed_suppressed() -> bool:
    return _SUPPRESSED.get()


def build_scan_feed(
    result: "CycleResult",
    *,
    date_from: str = "",
    date_to: str = "",
    placed_mids: Optional[set[str]] = None,
    source: Optional[str] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(plays, embeds)`` for one cycle — the dashboard's rows and their
    Discord rendering. Pure apart from the placed-market lookup (pass
    ``placed_mids`` to avoid the DB)."""
    plays = filter_scan_view(
        dashboard_play_dicts(result, result.bankroll),
        date_from,
        date_to,
        placed_mids,
    )
    embeds = scan_result_embeds(
        plays,
        markets_fetched=result.markets_fetched,
        markets_matched=result.markets_matched,
        bankroll=result.bankroll,
        kelly=result.kelly_fraction,
        sectors=list(result.sectors_scanned),
        date_from=date_from,
        date_to=date_to,
        duration_s=result.cycle_duration_s,
        errors=list(result.errors),
        source=source,
    )
    return plays, embeds


def post_scan_feed(
    client: "DiscordBotClient",
    result: "CycleResult",
    *,
    date_from: str = "",
    date_to: str = "",
    post_empty: bool = False,
    source: Optional[str] = None,
) -> bool:
    """Post the cycle's play table to the channel. A cycle with no plays is
    skipped unless ``post_empty`` (then a one-line "No +EV plays found."
    embed goes out, which doubles as a heartbeat that the scan ran).
    Returns True when every message was accepted; False when skipped or on
    delivery failure (logged, never raised)."""
    try:
        plays, embeds = build_scan_feed(
            result, date_from=date_from, date_to=date_to, source=source,
        )
    except Exception as exc:  # noqa: BLE001 — never let the feed break a scan
        logger.warning("discord_scan_feed_build_failed", error=str(exc))
        return False
    if not plays and not post_empty:
        logger.info("discord_scan_feed_skipped_empty", sectors=result.sectors_scanned)
        return False
    ok = client.post_embeds(embeds)
    logger.info(
        "discord_scan_feed_posted", plays=len(plays), messages=len(embeds), ok=ok,
    )
    return ok
