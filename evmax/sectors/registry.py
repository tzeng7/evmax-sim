"""Sector handler registry."""

from evmax.sectors.base import SectorHandler
from evmax.sectors.nfl import NFLHandler
from evmax.sectors.nba import NBAHandler
from evmax.sectors.ncaab import NCAABHandler
from evmax.sectors.soccer import SoccerHandler
from evmax.sectors.lol import LoLHandler
from evmax.sectors.cs2 import CS2Handler
from evmax.sectors.valorant import ValorantHandler
from evmax.sectors.tennis import TennisHandler
from evmax.sectors.baseball import BaseballHandler
from evmax.sectors.ncaaw import NCAAWHandler
from evmax.sectors.nhl import NHLHandler

_REGISTRY: dict[str, SectorHandler] = {
    "nfl": NFLHandler(),
    "nba": NBAHandler(),
    "ncaab": NCAABHandler(),
    "ncaaw": NCAAWHandler(),
    "soccer": SoccerHandler(),
    "lol": LoLHandler(),
    "cs2": CS2Handler(),
    "valorant": ValorantHandler(),
    "tennis": TennisHandler(),
    "baseball": BaseballHandler(),
    "nhl": NHLHandler(),
}

ALL_SECTORS = list(_REGISTRY.keys())


def get_handler(sector: str) -> SectorHandler:
    """Get the handler for a sector. Raises KeyError if not found."""
    sector = sector.lower()
    if sector not in _REGISTRY:
        raise KeyError(f"Unknown sector: {sector!r}. Available: {ALL_SECTORS}")
    return _REGISTRY[sector]
