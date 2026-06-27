"""API clients for Kalshi and Pinnacle.

Active clients (used in live pipeline):
- kalshi.py          — KalshiClient: RSA-auth REST + WebSocket orderbook, AsyncLimiter(10/s)
- esports_pinnacle.py — PinnacleGuestClient: Pinnacle guest API for ALL sectors (name is
                       historical — it covers every sector, not just esports). The single
                       sharp-odds source for both the scan pipeline and `evmax project`.
- mlb_statsapi.py    — MLBStatsClient: official MLB Stats API (free, no key); primary source for
                       baseball probable starters, keyed by stable team id (ESPN scoreboard fallback)

Seed-time clients (NOT in the live scan pipeline):
- tennisabstract.py  — fetch/parse Tennis Abstract's weekly Elo leaderboards (tennisabstract.com);
                       seeds tennis surface Elo after Sackmann's GitHub data repos went offline (2026).
                       Used by scripts/seed_tennis_abstract_elo.py, not by any live agent.
"""
