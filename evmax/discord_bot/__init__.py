"""Discord bot integration for evmax.

Two halves, one message format:

* **Scan feed (push).** Every coordinator cycle — CLI ``evmax agents scan``,
  the scheduled light scans, the dashboard's Scan button — posts its play
  list to the configured channel through :class:`DiscordBotClient`, a
  bot-token REST transport that needs no gateway connection and no extra
  dependency. The table is built by ``evmax.web.playlist`` — the SAME rows,
  order and columns as the dashboard's Scan Results panel.
* **Slash commands (pull).** ``evmax discord run`` starts a gateway bot
  (``discord.py``, optional extra ``evmax[discord]``) with ``/scan``,
  ``/plays``, ``/settled``, ``/status`` and ``/help``. The command bodies live
  in :mod:`evmax.discord_bot.handlers` and are framework-free, so they are
  unit-tested without discord.py installed.

Configuration (``.env``): ``DISCORD_BOT_TOKEN``, ``DISCORD_CHANNEL_ID``,
optional ``DISCORD_GUILD_ID`` / ``DISCORD_ALLOWED_USER_IDS`` /
``DISCORD_SCAN_FEED`` / ``DISCORD_POST_EMPTY_SCANS``. See docs/DISCORD_BOT.md.
"""

from evmax.discord_bot.client import DiscordBotClient

__all__ = ["DiscordBotClient"]
