"""Slash-command bodies — framework-free so they are testable without discord.py.

Each handler takes plain arguments and returns a :class:`Reply` (text and/or
embed dicts). :mod:`evmax.discord_bot.bot` adapts them to discord.py
interactions; the CLI could equally print them. Every handler is a thin
adapter over the dashboard's own data functions so Discord shows what the
dashboard shows:

* ``/scan``    → ``evmax.web.app.run_dashboard_scan`` (the ``/api/scan`` body)
* ``/plays``   → ``evmax.web.app._open_bets``      (Open Positions panel)
* ``/settled`` → ``evmax.web.app._settled_bets`` + ``_summary_stats`` (Recent Settled + KPIs)
* ``/status``  → ``evmax.agents.cleanup.heartbeat.run_heartbeat``
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import structlog

from evmax.discord_bot.embeds import (
    open_positions_embeds,
    recent_settled_embeds,
    scan_result_embeds,
    status_embed,
)
from evmax.discord_bot.feed import suppress_scan_feed

logger = structlog.get_logger(__name__)

# Dashboard defaults (frontend/src/App.tsx: bankroll '250', kelly 0.5).
DEFAULT_BANKROLL = 250.0
DEFAULT_KELLY = 0.5

HELP_TEXT = (
    "**evmax commands**\n"
    "• `/scan [sectors] [bankroll] [kelly] [date_from] [date_to] [bankroll_venue]` — run a scan "
    "and post the dashboard's Scan Results table (also persists rows, like the dashboard).\n"
    "• `/plays [sector] [bankroll] [kelly]` — Open Positions: scanned, unplaced live rows awaiting a pick.\n"
    "• `/settled [placed_only]` — Recent Settled Bets with the KPI summary.\n"
    "• `/status [probe_pinnacle]` — pipeline health (cadence, seed states, Pinnacle).\n"
    "• `/help` — this message.\n"
    "Bankroll: with `DISCORD_BANKROLL_VENUE` set (kalshi / polymarket_us / both) `/scan` and "
    "`/plays` size against that venue's LIVE balance unless you pass `bankroll`.\n"
    "Every scan cycle run anywhere (CLI, scheduled, dashboard) also posts its play table "
    "to the feed channel automatically."
)


@dataclass
class Reply:
    """What a handler wants sent back: optional text + zero or more embeds.
    ``ephemeral`` marks an error/permission reply only the invoker should see."""

    content: Optional[str] = None
    embeds: list[dict[str, Any]] = field(default_factory=list)
    ephemeral: bool = False


def _valid_iso_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


class CommandHandlers:
    """The command bodies. One instance per bot; holds the scan lock so two
    ``/scan`` invocations cannot hammer the venues concurrently."""

    def __init__(
        self,
        *,
        allowed_user_ids: frozenset[int] = frozenset(),
        default_bankroll: float = DEFAULT_BANKROLL,
        default_kelly: float = DEFAULT_KELLY,
        default_bankroll_venue: str = "",
    ) -> None:
        self._allowed = allowed_user_ids
        self._default_bankroll = default_bankroll
        self._default_kelly = default_kelly
        # ``DISCORD_BANKROLL_VENUE``: when set, a command run WITHOUT an explicit
        # bankroll sizes against this venue's live balance (see ``/scan``,
        # ``/plays``). An explicit ``bankroll`` always means "manual".
        self._default_bankroll_venue = (default_bankroll_venue or "").strip().lower()
        self._scan_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def is_allowed(self, user_id: int) -> bool:
        """True when the invoker may run commands: no allow-list configured,
        or the id is on it."""
        return not self._allowed or int(user_id) in self._allowed

    def denied(self) -> Reply:
        return Reply(content="You are not on this bot's allow-list.", ephemeral=True)

    @property
    def scan_running(self) -> bool:
        return self._scan_lock.locked()

    # ------------------------------------------------------------------
    # /help
    # ------------------------------------------------------------------

    async def help(self) -> Reply:
        return Reply(content=HELP_TEXT, ephemeral=True)

    # ------------------------------------------------------------------
    # /scan
    # ------------------------------------------------------------------

    async def scan(
        self,
        *,
        sectors: str = "",
        bankroll: Optional[float] = None,
        kelly: Optional[float] = None,
        date_from: str = "",
        date_to: str = "",
        bankroll_venue: str = "",
        fan_out_portfolios: bool = True,
    ) -> Reply:
        """Run the dashboard scan and reply with its Scan Results table.

        Persists rows exactly like the dashboard's Scan button (``log_gaps`` +
        portfolio fan-out). The coordinator's own feed post is suppressed for
        this cycle so the table appears once — as this reply.

        Bankroll precedence: explicit ``bankroll_venue`` > explicit ``bankroll``
        (manual) > ``DISCORD_BANKROLL_VENUE`` (live balance) > the $250 default."""
        if not bankroll_venue and bankroll is None:
            bankroll_venue = self._default_bankroll_venue
        bankroll = self._default_bankroll if bankroll is None else float(bankroll)
        kelly = self._default_kelly if kelly is None else float(kelly)
        if bankroll <= 0:
            return Reply(content="bankroll must be > 0.", ephemeral=True)
        if not (0 < kelly <= 1):
            return Reply(content="kelly must be in (0, 1].", ephemeral=True)
        for label, val in (("date_from", date_from), ("date_to", date_to)):
            if val and not _valid_iso_date(val):
                return Reply(content=f"{label} must be YYYY-MM-DD (got {val!r}).", ephemeral=True)
        if date_from and date_to and date_from > date_to:
            return Reply(content="date_from is after date_to.", ephemeral=True)
        if sectors:
            from evmax.sectors.registry import ALL_SECTORS
            wanted = [s.strip().lower() for s in sectors.split(",") if s.strip()]
            bad = [s for s in wanted if s not in ALL_SECTORS]
            if bad:
                return Reply(
                    content=f"Unknown sector(s): {', '.join(bad)}. Valid: {', '.join(ALL_SECTORS)}",
                    ephemeral=True,
                )
            sectors = ",".join(wanted)
        if self._scan_lock.locked():
            return Reply(content="A scan is already running — try again in a minute.", ephemeral=True)

        from evmax.web.app import run_dashboard_scan

        async with self._scan_lock:
            with suppress_scan_feed():
                payload = await run_dashboard_scan(
                    sectors_str=sectors,
                    bankroll=bankroll,
                    kelly=kelly,
                    date_from=date_from,
                    date_to=date_to,
                    bankroll_venue=bankroll_venue or None,
                    fan_out_portfolios=fan_out_portfolios,
                )
        source = None
        if payload.get("bankroll_source") and str(payload["bankroll_source"]).startswith("live:"):
            source = f"bankroll {payload['bankroll_source']}"
        embeds = scan_result_embeds(
            payload["gaps"],
            markets_fetched=payload["markets_fetched"],
            markets_matched=payload["markets_matched"],
            bankroll=float(payload["bankroll"]),
            kelly=kelly,
            sectors=payload.get("sectors"),
            date_from=date_from,
            date_to=date_to,
            source=source,
        )
        return Reply(embeds=embeds)

    # ------------------------------------------------------------------
    # /plays  (Open Positions)
    # ------------------------------------------------------------------

    async def plays(
        self,
        *,
        sector: str = "",
        bankroll: Optional[float] = None,
        kelly: Optional[float] = None,
    ) -> Reply:
        from evmax.web.app import _open_bets

        kelly = self._default_kelly if kelly is None else float(kelly)
        source: Optional[str] = None
        if bankroll is None and self._default_bankroll_venue:
            # Size the Stake column against the live venue balance, exactly as
            # a `--bankroll-venue` scan would. Fail-soft to the manual default.
            from evmax.clients.balances import resolve_bankroll_plan

            plan = await resolve_bankroll_plan(self._default_bankroll, self._default_bankroll_venue)
            bankroll = plan.bankroll
            source = f"bankroll {plan.source}"
        else:
            bankroll = self._default_bankroll if bankroll is None else float(bankroll)
        bets = await asyncio.to_thread(_open_bets)
        return Reply(embeds=open_positions_embeds(
            bets, bankroll=bankroll, kelly=kelly, sector=(sector or "").strip().lower() or None,
            source=source,
        ))

    # ------------------------------------------------------------------
    # /settled  (Recent Settled Bets + KPI summary)
    # ------------------------------------------------------------------

    async def settled(self, *, placed_only: bool = False) -> Reply:
        from evmax.web.app import _settled_bets, _summary_stats

        settled = await asyncio.to_thread(_settled_bets)
        pool = [b for b in settled if b.get("placed")] if placed_only else settled
        summary = _summary_stats(pool)
        recent = list(reversed(settled))  # newest first, like /api/dashboard
        return Reply(embeds=recent_settled_embeds(
            recent, summary=summary, placed_only=placed_only,
        ))

    # ------------------------------------------------------------------
    # /status  (heartbeat)
    # ------------------------------------------------------------------

    async def status(self, *, probe_pinnacle: bool = True) -> Reply:
        from evmax.agents.cleanup.heartbeat import run_heartbeat

        report = await asyncio.to_thread(
            run_heartbeat, check_pinnacle_reachability=probe_pinnacle, notify=False,
        )
        return Reply(embeds=[status_embed(report)])
