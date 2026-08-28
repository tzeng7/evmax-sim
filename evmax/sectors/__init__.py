"""Sector handlers: sport-specific configuration for Kalshi series, Pinnacle league IDs, and aliases.

Each handler is a SectorHandler subclass registered in registry.py.
Registered sectors: nfl, nba, ncaab, ncaaf, ncaaw, soccer, worldcup, lol, cs2,
tennis, ufc, baseball, nhl, wnba — plus valorant as a latent handler (no Kalshi
product, so not bettable). f1.py exists but is not registered. The bettable
catalog and per-sector mode live in data/categories.yaml, not here.

Key responsibilities per handler:
- kalshi_series()      — list of Kalshi series tickers to poll (e.g. ["KXNBAGAME"])
- pinnacle_league_ids() — Pinnacle league IDs for sharp odds (via PinnacleGuestClient)
- canonical_team_name() — normalise team names for matching (delegates to aliases/ YAML)
"""
