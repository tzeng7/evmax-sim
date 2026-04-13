"""API clients for Kalshi and Pinnacle.

Active clients (used in live pipeline):
- kalshi.py          — KalshiClient: RSA-auth REST + WebSocket orderbook, AsyncLimiter(10/s)
- esports_pinnacle.py — PinnacleGuestClient: Pinnacle guest API for ALL sectors (name is misleading)

Legacy (not used in live pipeline):
- pinnacle.py        — PinnacleClient: TheOddsAPI wrapper; superseded by PinnacleGuestClient
"""
