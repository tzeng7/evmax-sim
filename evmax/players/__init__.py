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
        # Normalize alias output: "mikal bridges" → "mikal_bridges"
        return aliases[lower].replace(" ", "_")
    # Strip generational suffixes (jr, sr, ii, iii, iv)
    _SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
    parts = lower.split()
    parts = [p.rstrip(".") for p in parts]  # strip trailing dots (e.g. "Jr.")
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    # Return full underscore-joined name for disambiguation
    # "Mitchell Robinson" → "mitchell_robinson", "LeBron James" → "lebron_james"
    return "_".join(parts) if parts else lower
