"""Sector handlers: sport-specific configuration for Kalshi series, Pinnacle league IDs, and aliases.

Each handler is a SectorHandler subclass registered in registry.py.
Active sectors: nba, nfl, ncaab, ncaaw, soccer, tennis, lol, cs2, baseball, nhl, valorant.

Key responsibilities per handler:
- kalshi_series()      — list of Kalshi series tickers to poll (e.g. ["KXNBAGAME"])
- pinnacle_league_ids() — Pinnacle league IDs for sharp odds (via PinnacleGuestClient)
- canonical_team_name() — normalise team names for matching (delegates to aliases/ YAML)
"""
