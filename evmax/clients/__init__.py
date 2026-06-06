"""API clients for Kalshi and Pinnacle.

Active clients (used in live pipeline):
- kalshi.py          — KalshiClient: RSA-auth REST + WebSocket orderbook, AsyncLimiter(10/s)
- esports_pinnacle.py — PinnacleGuestClient: Pinnacle guest API for ALL sectors (name is misleading)
- mlb_statsapi.py    — MLBStatsClient: official MLB Stats API (free, no key); primary source for
                       baseball probable starters, keyed by stable team id (ESPN scoreboard fallback)

Legacy (not used in live pipeline):
- pinnacle.py        — PinnacleClient: TheOddsAPI wrapper; superseded by PinnacleGuestClient
"""
