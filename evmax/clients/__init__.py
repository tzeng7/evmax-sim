"""API clients for Kalshi and Pinnacle.

Active clients (used in live pipeline):
- kalshi.py          — KalshiClient: RSA-auth REST + WebSocket orderbook, AsyncLimiter(10/s)
- esports_pinnacle.py — PinnacleGuestClient: Pinnacle guest API for ALL sectors (name is
                       historical — it covers every sector, not just esports). The single
                       sharp-odds source for both the scan pipeline and `evmax project`.
- mlb_statsapi.py    — MLBStatsClient: official MLB Stats API (free, no key); primary source for
                       baseball probable starters, keyed by stable team id (ESPN scoreboard fallback)

Seed-time clients (NOT in the live scan pipeline):
- tennisabstract.py  — Tennis Abstract scrapers, the replacement for Sackmann's offline (2026)
                       GitHub data repos. Provides: Elo leaderboards (surface Elo seed),
                       `matchmx` per-match serve/return data (serve_return/advanced/form/h2h seed),
                       and winners/errors leaderboards (advanced UE feature).
                       Used by scripts/seed_tennis_abstract_elo.py + seed_tennis_models.py.
- ufc_espn.py        — UFCESPNClient: ESPN MMA public API (UFC results, methods, fighter bios,
                       2010→present). Source of truth for the ufc_rating seed/backtest —
                       ufcstats.com is behind a JS anti-bot gate we don't bypass. Disk-cached
                       under data/cache/ufc_espn/. Used by scripts/fetch_ufc_history.py.
"""
