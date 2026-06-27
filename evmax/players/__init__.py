"""Player name normalization for prop markets."""

import unicodedata

from evmax.players.baseball_codes import BASEBALL_PLAYER_ALIASES
from evmax.players.nba_codes import NBA_PLAYER_ALIASES
from evmax.players.nfl_codes import NFL_PLAYER_ALIASES

__all__ = [
    "BASEBALL_PLAYER_ALIASES",
    "NBA_PLAYER_ALIASES",
    "NFL_PLAYER_ALIASES",
    "normalize_player_name",
]

_SECTOR_ALIASES: dict[str, dict[str, str]] = {
    "nba": NBA_PLAYER_ALIASES,
    "nfl": NFL_PLAYER_ALIASES,
    "baseball": BASEBALL_PLAYER_ALIASES,
}

# Sectors whose two data sources (Kalshi vs Pinnacle) disagree on accents —
# fold to ASCII before matching. MLB rosters are accent-heavy (José Ramírez,
# Luis Robert) and Kalshi strips accents while Pinnacle keeps them.
_ACCENT_FOLD_SECTORS = {"baseball"}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _ascii_fold(text: str) -> str:
    """Lowercase + strip combining accents. 'José Ramírez' → 'jose ramirez'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_player_name(name: str, sector: str) -> str:
    """Normalize a player name to a canonical form for matching.

    Looks up known aliases first (handles initials like LBJ, nicknames).
    Folds accents for sectors whose sources disagree on them (MLB). Falls back
    to the suffix-stripped, underscore-joined full name.
    """
    if not name:
        return ""
    lower = name.strip().lower()
    if sector in _ACCENT_FOLD_SECTORS:
        lower = _ascii_fold(lower)
    aliases = _SECTOR_ALIASES.get(sector, {})
    if lower in aliases:
        # Normalize alias output: "mikal bridges" → "mikal_bridges"
        return aliases[lower].replace(" ", "_")
    # Strip generational suffixes (jr, sr, ii, iii, iv)
    parts = lower.split()
    parts = [p.rstrip(".") for p in parts]  # strip trailing dots (e.g. "Jr.")
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    # Return full underscore-joined name for disambiguation
    # "Mitchell Robinson" → "mitchell_robinson", "LeBron James" → "lebron_james"
    return "_".join(parts) if parts else lower
