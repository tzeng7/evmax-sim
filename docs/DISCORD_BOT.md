# Discord bot

evmax can run as a Discord bot application. It does three things:

| Feature | Mechanism | Needs the gateway bot running? |
|---|---|---|
| **Scan feed** — every scan cycle (CLI `evmax agents scan`, the scheduled `ev-scan-light-*` tasks, the dashboard's Scan button) posts its play list to a channel | `Notifier.notify_cycle` → `evmax.discord_bot.feed.post_scan_feed` over the Discord REST API (stdlib `urllib`, bot token) | No |
| **Alerts** — heartbeat, CLV monitor, arb `--notify`, and any `Notifier.send_text` reach the same channel as colored embeds / text | `Notifier.notify_alert` / `send_text` → `DiscordBotClient` | No |
| **Slash commands** — `/scan`, `/plays`, `/settled`, `/status`, `/help` | `evmax discord run` (discord.py gateway client, optional extra `evmax[discord]`) | Yes |

The feed and alerts work as soon as `DISCORD_BOT_TOKEN` plus a target are in `.env`: either
`DISCORD_DM_USER_ID` (the bot **direct-messages you**; no channel needed) or
`DISCORD_CHANNEL_ID` (a server channel), or both. A DM needs no server at all: add the app to
your **account** (a Discord *user install*, step 3 below) and the bot can DM you. Adding it to
a server you're in works too, as long as your privacy settings allow DMs from server members.
Nothing else in the pipeline changes: the feed is a third notifier transport next to the
existing Slack / Discord webhooks (which keep sending the compact ≥5%-EV text alert).

## The feed is the dashboard's Scan Results panel

The Discord table is built from the same code as the React dashboard's Scan Results panel
(`frontend/src/components/ScanResults.tsx`), so the two cannot drift:

* **Rows** — `evmax.web.playlist.dashboard_play_dicts` + `filter_scan_view`: `cycle.plays()`
  (full-blend gaps, EV-descending), best-execution collapse of the same bet on two venues,
  per-venue cash cap, then the scan-view filters — today + tomorrow window, no esports map
  handicaps, no player props, nothing already placed. `/api/scan` uses the identical
  functions (`evmax/web/app.py`).
* **Columns, in the panel's order** — Date · Sector · Venue · Event · Outcome · Ask · Fair
  Value · Model · EV · Maker EV · Limit ¢ · Bid ¢ · Fill ¢ · Stake ($) · Models.
* **Cell formatting** — the panel's own JS helpers, ported: `probToCents` (`45¢`, `12.7¢`),
  `toFixed(1)%`, the maker columns show `—` when absent, Fill ¢ / Stake seed to the ask +
  taker stake (or the resting bid + maker stake for MAKER rows), exactly like `defaultFill`.
* **Badges become tags in the same column** — a non-live row reads `nfl shadow`, a maker-only
  row `nfl MAKER`; a bet quoted on two venues lists the venue dropdown's options in the Venue
  cell (`Kalshi · 4.1% | Poly · 3.2% mkr`); a single-venue row with a cheaper alternative gets
  the panel's `· also Poly 47¢` note on the Outcome.
* **Title** — the panel header verbatim: `Scan Results — 12 plays (600 markets, 71 matched)`.
  The footer carries bankroll, Kelly, sectors, the date window, duration and any sector errors.

Discord has no table markup, and an embed is only ~55 monospace characters wide on desktop
(fewer on mobile), so a 15-column table would wrap every row into four lines. Each row is
therefore rendered as a compact **card** — the same cells, in the panel's order, over a few
short markdown lines:

```
🟢 **Brentford ML** — Brentford vs Sunderland
soccer · Kalshi · 34.9% | Poly · 9.7% · 2026-09-05
Ask **41¢** · fair 57.6¢ (57.6%) · EV **+34.9%** · stake **$83.24**
maker EV 39.0% · limit 56¢ · bid 40¢
`elo+form+poisson+xg+sharp`
```

🟢 = live row, 🟡 = shadow (with a `shadow` tag), 🟣 = maker-only (`MAKER` tag; its fill
line shows the resting bid it seeds to). Maker cells the panel shows as `—` are omitted. Long
lists split across embeds (a card never splits) and across messages under Discord's caps (4096
chars per description, 6000 per message). `/plays` mirrors the Open Positions panel (3-line
cards with the panel's `NEW` / `LIVE` tags) and `/settled` the Recent Settled Bets panel (✅ /
❌ 2-line cards, P&L bold) + KPI summary the same way. The `*_row` cell builders in
`evmax/discord_bot/embeds.py` remain the single source of every figure.

## Setup (once, ~5 minutes)

You create the application and paste the token yourself — evmax never asks for it
interactively.

1. **Create the application.** <https://discord.com/developers/applications> → *New
   Application* → name it (e.g. `evmax`).
2. **Bot token.** Left menu *Bot* → *Reset Token* → copy it into `.env`:
   `DISCORD_BOT_TOKEN=...`. Leave every *Privileged Gateway Intent* OFF — slash commands and
   channel posts need none.
3. **Add the bot.** Two options — pick one (or both):
   * **To your account only (DM feed, no server).** Left menu *Installation* → under
     *Installation Contexts* tick **User Install** → *Save Changes*. Open
     `https://discord.com/oauth2/authorize?client_id=<APPLICATION_ID>&integration_type=1&scope=applications.commands`
     (the application id is on *General Information*) and authorize — the dialog says *Add to
     My Apps*. Verified 2026-09-04: a user-installed bot can open the DM and post the feed to it
     with no shared server. The commands are registered for user installs too, so `/scan` etc.
     are available in that DM once `evmax discord run` is up (leave `DISCORD_GUILD_ID` empty).
   * **To a server (channel feed).** Left menu *OAuth2* → *URL Generator* → scopes **`bot`**
     and **`applications.commands`**; bot permissions **View Channels**, **Send Messages**,
     **Embed Links** (integer `19456`). Open the generated URL, pick your server, authorize.
4. **Ids.** In Discord: *User Settings → Advanced → Developer Mode* ON. To be **DM'd**:
   right-click your own name (any message of yours, or your avatar) → *Copy User ID* →
   `DISCORD_DM_USER_ID`. For a **channel** feed instead/as well: right-click the channel →
   *Copy Channel ID* → `DISCORD_CHANNEL_ID`. For a server install, right-click the server icon →
   *Copy Server ID* → `DISCORD_GUILD_ID` (commands sync instantly to that server and the bot
   refuses interactions from any other). Leave `DISCORD_GUILD_ID` EMPTY for a user install —
   the commands must be global to appear in your DM. Optionally right-click your
   name → *Copy User ID* → `DISCORD_ALLOWED_USER_IDS` (comma-separated; empty = any member of
   the server may run commands — note `/scan` persists rows like the dashboard's Scan button).
5. **Test the channel transport.**

   ```bash
   evmax discord test
   ```

   A blue *evmax info — test* embed should appear in your DMs / the channel. From now on every
   scan cycle posts its table there.
6. **Slash commands** need the gateway extra and a long-running process:

   ```bash
   uv sync --extra dev --extra discord
   ```

   ```bash
   evmax discord run
   ```

   With `DISCORD_GUILD_ID` set the commands appear in that server within seconds; global
   registration (no guild id) can take up to an hour to propagate.

### Configuration reference (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `DISCORD_BOT_TOKEN` | — | Bot token from the Developer Portal. Enables the bot transport. |
| `DISCORD_CHANNEL_ID` | — | Channel that receives the scan feed + alerts. Optional when a DM target is set. |
| `DISCORD_DM_USER_ID` | — | Your user id: the bot DMs you the feed + alerts. Resolved to a DM channel once per process via `POST /users/@me/channels`. Needs the app added to your account (user install) or a shared server. |
| `DISCORD_GUILD_ID` | — | Optional. Guild-scoped command sync (instant) + interaction lockdown. |
| `DISCORD_ALLOWED_USER_IDS` | — | Optional comma-separated user ids allowed to run commands. Empty = any member. |
| `DISCORD_SCAN_FEED` | `true` | Post each scan cycle's play table to the channel. |
| `DISCORD_POST_EMPTY_SCANS` | `false` | Also post cycles with zero plays (`No +EV plays found.` — a heartbeat that the scan ran). |
| `DISCORD_BANKROLL_VENUE` | — | `kalshi` / `polymarket_us` / `both`: `/scan` and `/plays` size against that venue's **live balance** (cash + open positions, the CLI's `--bankroll-venue` plan) whenever you omit `bankroll`. An explicit `bankroll` is always manual; an explicit `bankroll_venue` wins. Unavailable balance → falls back to $250 and the footer says `manual_fallback`. The automatic feed reflects whatever bankroll the scan itself ran with (the scheduled scans already pass `--bankroll-venue kalshi`). |
| `DISCORD_WEBHOOK_URL` | — | Unchanged: the pre-existing plain-text ≥5%-EV webhook alert. Independent of the bot. |

Runtime layering: `Notifier.from_settings()` builds the bot client only when the token and at
least one target (channel or DM user) are set; `is_configured()` is true with the bot alone.
When both targets are set every post goes to both.

**Slash commands in the DM:** commands registered with `DISCORD_GUILD_ID` exist only in that
server. To type `/scan` inside your DM with the bot, leave `DISCORD_GUILD_ID` empty (global
registration, up to an hour to appear) — the bot still enforces `DISCORD_ALLOWED_USER_IDS`.

## Commands

| Command | Mirrors | Notes |
|---|---|---|
| `/scan [sectors] [bankroll] [kelly] [date_from] [date_to] [bankroll_venue]` | Dashboard Scan button (`/api/scan`) | Runs `evmax.web.app.run_dashboard_scan`: persists rows (`log_gaps`) and fans out to portfolios exactly like the dashboard, then replies with the Scan Results table. Defaults: all in-season game sectors, bankroll 250, Kelly 0.5, today + tomorrow. One scan at a time (a second `/scan` is refused while one runs). The coordinator's own feed post is suppressed for this cycle so the table appears once, as the reply. |
| `/plays [sector] [bankroll] [kelly]` | Open Positions panel | Unresolved, unplaced live rows from `predictions.db`; `LIVE` tag = started, awaiting resolution. 40-row cap like the panel. |
| `/settled [placed_only]` | Recent Settled Bets panel + KPI cards | Newest first, 50-row cap; footer = bets / W-L / win rate / P&L / ROI / avg EV from `_summary_stats`. |
| `/status [probe_pinnacle]` | `evmax cleanup heartbeat` | Scan/resolve cadence, seed-state staleness, and (default on) a live Pinnacle probe. Never sends the heartbeat alert itself. |
| `/help` | — | Command list. |

Every command acknowledges within Discord's 3-second window and replies via follow-ups, so a
60-second scan is fine. Errors come back as `Command failed: …` instead of crashing the bot.

## Running the gateway bot as a service (launchd)

The feed needs no service. For the slash commands, keep `evmax discord run` up with launchd —
unlike the `--once` agents in `docs/SCHEDULED_RUNS.md` this one is long-lived, so the template
uses `KeepAlive` (auto-restart on crash / after sleep) and no `caffeinate`:

```bash
cp docs/launchd/com.evmax.discord-bot.plist ~/Library/LaunchAgents/
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.evmax.discord-bot.plist
```

Logs land in `logs/launchd.discord-bot.{out,err}`. Stop with
`launchctl bootout gui/$(id -u)/com.evmax.discord-bot`. Edit the paths in the plist if the
checkout is not at `/Users/ktzeng/Projects/evmax`. Because the process is long-lived, restart
it (`launchctl kickstart -k gui/$(id -u)/com.evmax.discord-bot`) after pulling code changes.

## Troubleshooting

* **`evmax discord test` fails with 403 / "Missing Access".** The bot is not in the server or
  lacks View Channel / Send Messages / Embed Links on that channel. Re-run the invite URL or
  fix the channel permission overrides.
* **DM never arrives / 403 `Cannot send messages to this user due to having no mutual guilds`.**
  The app is neither added to your account (user-install link, step 3) nor in a server you're
  in, or your *Privacy Settings → Direct Messages* block server members. Also
  double-check `DISCORD_DM_USER_ID` is your USER id, not the server or channel id.
* **404 "Unknown Channel".** `DISCORD_CHANNEL_ID` is not a channel id (server and user ids look
  identical — copy from the channel's context menu).
* **401.** Token was reset or pasted with whitespace.
* **Commands don't appear.** Without `DISCORD_GUILD_ID` global sync takes up to an hour; with it,
  the bot must have been invited with the `applications.commands` scope.
* **No feed posts after a scan.** `DISCORD_SCAN_FEED=false`, the cycle had zero plays in the
  today + tomorrow window (set `DISCORD_POST_EMPTY_SCANS=true` to see those), or the scan was
  started from `/scan` (which replies with the table instead). Delivery failures are logged as
  `discord_bot_*` events; the scan itself is never affected.
* **429s.** The client honours Discord's `retry_after`; a burst of scheduled scans is well
  within limits.

## Code map

```
evmax/discord_bot/
├── client.py    DiscordBotClient — REST post/post_embeds/post_text, retry, batching under caps
├── embeds.py    Scan Results / Open Positions / Recent Settled / alert / status embed builders (JS-parity formatting)
├── feed.py      build_scan_feed / post_scan_feed + suppress_scan_feed ContextVar
├── handlers.py  CommandHandlers — framework-free command bodies (tested without discord.py)
└── bot.py       discord.py wiring: build_bot / run_bot / register_commands / send_reply
evmax/web/playlist.py   gap_to_dict / dashboard_play_dicts / filter_scan_view (shared with /api/scan)
evmax/cli/commands/discord.py   evmax discord run | test
tests/test_discord_bot.py, tests/test_web_playlist.py
```
