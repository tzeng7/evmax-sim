"""The gateway (slash-command) bot — discord.py wiring around CommandHandlers.

``discord.py`` is imported lazily so the rest of the package (REST feed,
embed builders, handlers) works without it. Start with ``evmax discord run``.

Slash commands: ``/scan``, ``/plays``, ``/settled``, ``/status``, ``/help``.
When ``DISCORD_GUILD_ID`` is set the commands are synced to that guild only
(instant, and the bot refuses interactions from anywhere else); otherwise
they are registered globally (Discord can take up to an hour to propagate).
``DISCORD_ALLOWED_USER_IDS`` restricts who may invoke them.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import structlog

from evmax.discord_bot.client import batch_embeds
from evmax.discord_bot.handlers import CommandHandlers, Reply

logger = structlog.get_logger(__name__)

COMMAND_NAMES = ("scan", "plays", "settled", "status", "help")


class DiscordNotInstalled(RuntimeError):
    """discord.py is not installed (optional extra ``evmax[discord]``)."""


def _import_discord() -> tuple[Any, Any]:
    try:
        import discord
        from discord import app_commands
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise DiscordNotInstalled(
            "discord.py is required for the slash-command bot: "
            "`uv sync --extra discord` (or `pip install 'evmax[discord]'`)."
        ) from exc
    return discord, app_commands


async def send_reply(interaction: Any, reply: Reply, discord_mod: Any = None) -> None:
    """Deliver a :class:`Reply` through an interaction that has already been
    deferred: text on the first followup, embeds batched under Discord's
    per-message caps (so a long scan table arrives as several messages)."""
    discord = discord_mod or _import_discord()[0]
    batches = batch_embeds(reply.embeds) if reply.embeds else []
    if not batches:
        await interaction.followup.send(
            content=reply.content or "(no output)", ephemeral=reply.ephemeral,
        )
        return
    for i, batch in enumerate(batches):
        kwargs: dict[str, Any] = {
            "embeds": [discord.Embed.from_dict(e) for e in batch],
            "ephemeral": reply.ephemeral,
        }
        if i == 0 and reply.content:
            kwargs["content"] = reply.content
        await interaction.followup.send(**kwargs)


def register_commands(
    tree: Any,
    handlers: CommandHandlers,
    *,
    discord_mod: Any = None,
    app_commands_mod: Any = None,
    guild_id: Optional[int] = None,
) -> None:
    """Attach the slash commands to an ``app_commands.CommandTree``."""
    discord, app_commands = (
        (discord_mod, app_commands_mod)
        if discord_mod is not None and app_commands_mod is not None
        else _import_discord()
    )

    async def _guard(interaction: Any) -> bool:
        if guild_id is not None and interaction.guild_id != guild_id:
            await interaction.response.send_message(
                "This bot only serves its home server.", ephemeral=True,
            )
            return False
        if not handlers.is_allowed(interaction.user.id):
            denied = handlers.denied()
            await interaction.response.send_message(denied.content, ephemeral=True)
            return False
        return True

    async def _run(interaction: Any, make_reply: Any) -> None:
        if not await _guard(interaction):
            return
        # Every command may exceed Discord's 3-second initial-response window
        # (a scan takes seconds to a minute), so acknowledge first and reply
        # via followups (allowed for 15 minutes).
        await interaction.response.defer(thinking=True)
        try:
            reply = await make_reply()
        except Exception as exc:  # noqa: BLE001 — surface, never crash the bot
            logger.exception("discord_command_failed", error=str(exc))
            reply = Reply(content=f"Command failed: {exc}")
        await send_reply(interaction, reply, discord)

    @tree.command(name="scan", description="Run an EV scan and post the dashboard's Scan Results table")
    @app_commands.describe(
        sectors="Comma-separated sectors (default: every in-season game sector)",
        bankroll="Bankroll in USD (default 250)",
        kelly="Kelly multiplier 0–1 (default 0.5)",
        date_from="Window start YYYY-MM-DD (default today)",
        date_to="Window end YYYY-MM-DD (default tomorrow)",
        bankroll_venue="Size against a venue's LIVE balance instead of --bankroll",
    )
    async def scan_cmd(
        interaction: Any,
        sectors: str = "",
        bankroll: Optional[float] = None,
        kelly: Optional[float] = None,
        date_from: str = "",
        date_to: str = "",
        bankroll_venue: Optional[Literal["kalshi", "polymarket_us", "both"]] = None,
    ) -> None:
        await _run(interaction, lambda: handlers.scan(
            sectors=sectors, bankroll=bankroll, kelly=kelly,
            date_from=date_from, date_to=date_to, bankroll_venue=bankroll_venue or "",
        ))

    @tree.command(name="plays", description="Open Positions — scanned, unplaced live rows awaiting a pick")
    @app_commands.describe(
        sector="Only this sector",
        bankroll="Bankroll in USD for the Stake column (default 250)",
        kelly="Kelly multiplier for the Stake column (default 0.5)",
    )
    async def plays_cmd(
        interaction: Any,
        sector: str = "",
        bankroll: Optional[float] = None,
        kelly: Optional[float] = None,
    ) -> None:
        await _run(interaction, lambda: handlers.plays(sector=sector, bankroll=bankroll, kelly=kelly))

    @tree.command(name="settled", description="Recent Settled Bets with the P&L / ROI summary")
    @app_commands.describe(placed_only="Only bets you actually placed via pick")
    async def settled_cmd(interaction: Any, placed_only: bool = False) -> None:
        await _run(interaction, lambda: handlers.settled(placed_only=placed_only))

    @tree.command(name="status", description="Pipeline health: scan/resolve cadence, seed states, Pinnacle")
    @app_commands.describe(probe_pinnacle="Also probe Pinnacle reachability (one live request)")
    async def status_cmd(interaction: Any, probe_pinnacle: bool = True) -> None:
        await _run(interaction, lambda: handlers.status(probe_pinnacle=probe_pinnacle))

    @tree.command(name="help", description="List evmax commands")
    async def help_cmd(interaction: Any) -> None:
        await _run(interaction, handlers.help)


def build_bot(settings: Any = None, handlers: Optional[CommandHandlers] = None) -> Any:
    """Construct (but do not start) the discord.py client with the commands
    attached. Raises :class:`DiscordNotInstalled` without the extra and
    ``RuntimeError`` without a token."""
    discord, app_commands = _import_discord()
    if settings is None:
        from evmax.settings import get_settings
        settings = get_settings()
    if not settings.discord_bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is not set — see docs/DISCORD_BOT.md")

    guild_id = int(settings.discord_guild_id) if settings.discord_guild_id else None
    handlers = handlers or CommandHandlers(allowed_user_ids=settings.discord_allowed_users())
    intents = discord.Intents.default()  # slash commands need no privileged intents

    class EvmaxBot(discord.Client):
        def __init__(self) -> None:
            super().__init__(intents=intents)
            self.tree = app_commands.CommandTree(self)
            self.handlers = handlers
            self.guild_id = guild_id

        async def setup_hook(self) -> None:
            if guild_id is not None:
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
            else:
                await self.tree.sync()

        async def on_ready(self) -> None:
            logger.info(
                "discord_bot_ready",
                user=str(self.user),
                guild=guild_id or "global",
                channel=settings.discord_channel_id or None,
            )

    bot = EvmaxBot()
    register_commands(
        bot.tree, handlers, discord_mod=discord, app_commands_mod=app_commands, guild_id=guild_id,
    )
    return bot


def run_bot(settings: Any = None) -> None:
    """Build the bot and block on the gateway connection until interrupted."""
    if settings is None:
        from evmax.settings import get_settings
        settings = get_settings()
    bot = build_bot(settings)
    bot.run(settings.discord_bot_token)
