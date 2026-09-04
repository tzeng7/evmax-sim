"""evmax discord — run the slash-command bot / test the channel transport.

The scan FEED needs no command here: once ``DISCORD_BOT_TOKEN`` and
``DISCORD_CHANNEL_ID`` are in ``.env``, every scan cycle (CLI, scheduled,
dashboard) posts its Scan Results table to the channel through the notifier.
"""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Discord bot: scan feed + slash commands (see docs/DISCORD_BOT.md).")


@app.command("run")
def run() -> None:
    """Start the gateway bot (blocking). Serves /scan /plays /settled /status /help.

    Requires the optional extra: `uv sync --extra discord`.
    """
    from evmax.settings import get_settings

    s = get_settings()
    if not s.discord_bot_token:
        console.print("[red]DISCORD_BOT_TOKEN is not set.[/red] See docs/DISCORD_BOT.md.")
        raise typer.Exit(1)
    if not (s.discord_channel_id or s.discord_dm_user_id):
        console.print(
            "[yellow]Neither DISCORD_CHANNEL_ID nor DISCORD_DM_USER_ID is set — slash commands "
            "will work but no scan feed / alerts will be posted.[/yellow]"
        )
    if s.discord_dm_user_id and s.discord_guild_id:
        console.print(
            "[dim]DISCORD_GUILD_ID is set, so slash commands live in that server only. Leave it "
            "empty (global sync) to use /scan etc. inside the DM as well.[/dim]"
        )
    try:
        from evmax.discord_bot.bot import DiscordNotInstalled, run_bot
    except ImportError as exc:  # pragma: no cover
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold cyan]evmax discord bot[/bold cyan] — guild: "
        f"{s.discord_guild_id or 'global'} · feed → {s.discord_channel_id and f'channel {s.discord_channel_id}' or ''}"
        f"{' + ' if s.discord_channel_id and s.discord_dm_user_id else ''}"
        f"{s.discord_dm_user_id and f'DM to {s.discord_dm_user_id}' or ''}{'' if (s.discord_channel_id or s.discord_dm_user_id) else '—'} · "
        f"allowed users: {', '.join(map(str, sorted(s.discord_allowed_users()))) or 'any member'}"
    )
    try:
        run_bot(s)
    except DiscordNotInstalled as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command("test")
def test(
    message: str = typer.Option(
        "Discord bot transport is working.", "--message", "-m",
        help="Body of the test alert.",
    ),
) -> None:
    """Post a test alert embed to the channel and/or your DMs and report delivery."""
    from evmax.discord_bot.client import DiscordBotClient
    from evmax.discord_bot.embeds import alert_embed

    client = DiscordBotClient.from_settings()
    if client is None:
        console.print(
            "[red]DISCORD_BOT_TOKEN plus DISCORD_CHANNEL_ID or DISCORD_DM_USER_ID must be set.[/red] "
            "See docs/DISCORD_BOT.md."
        )
        raise typer.Exit(1)
    ok = client.post_embeds([alert_embed("test", message, severity="info")])
    if ok:
        console.print(f"[green]✓ Posted to {client.describe_targets()}.[/green]")
    else:
        console.print(
            f"[red]✗ Discord rejected the post to {client.describe_targets()}.[/red] "
            "Check the token; for a channel, that the bot is in the server with View Channel + "
            "Send Messages + Embed Links; for a DM, that you share a server with the bot and "
            "allow DMs from server members (Server → Privacy Settings)."
        )
        raise typer.Exit(1)
