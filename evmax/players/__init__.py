"""Player name normalization for prop markets."""

from evmax.players.nba_codes import NBA_PLAYER_ALIASES
from evmax.players.nfl_codes import NFL_PLAYER_ALIASES

__all__ = ["NBA_PLAYER_ALIASES", "NFL_PLAYER_ALIASES", "normalize_player_name"]


def normalize_player_name(name: str, sector: str) -> str:
    """Normalize a player name to a canonical form for matching.

    Looks up known aliases first (handles initials like LBJ, nicknames).
    Falls back to lowercase last name.
    """
    if not name:
        return ""
    lower = name.strip().lower()
    aliases = NBA_PLAYER_ALIASES if sector == "nba" else NFL_PLAYER_ALIASES
    if lower in aliases:
        return aliases[lower]
    # Last name fallback
    parts = lower.split()
    return parts[-1] if parts else lower
