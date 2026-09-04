"""Name normalizer with alias resolution."""

from __future__ import annotations

import re
from typing import Optional

from evmax.sectors.registry import get_handler


class NameNormalizer:
    """
    Normalizes team/player names for cross-source matching.

    Steps:
      1. Lowercase and strip whitespace
      2. Remove common noise words (FC, CF, SC, AFC, etc.)
      3. Apply sector-specific alias map
      4. Normalize unicode and punctuation
    """

    # Noise prefixes/suffixes to remove from club names
    _NOISE_WORDS = {
        "fc", "cf", "sc", "ac", "af", "bk", "sk", "fk",
        "afc", "fcu", "rfc", "utd", "united",
        "esports", "gaming", "team", "club",
    }

    # Game-name suffixes Kalshi appends to away team names (longest first)
    _GAME_SUFFIXES = [
        "league of legends",
        "valorant",
        "cs2",
    ]

    def __init__(self, sector: str) -> None:
        self._sector = sector
        try:
            self._handler = get_handler(sector)
        except KeyError:
            self._handler = None

    def normalize(self, name: str) -> str:
        """Return a normalized team name."""
        if not name:
            return ""

        # Basic cleanup
        result = name.lower().strip()
        result = re.sub(r"['\u2019\u2018]", "", result)  # Remove apostrophes
        result = re.sub(r"[^\w\s\-\.]", " ", result)
        result = re.sub(r"\s+", " ", result).strip()

        # Strip game-name suffixes Kalshi appends (e.g. "G2 NORD League of Legends")
        for suffix in self._GAME_SUFFIXES:
            if result.endswith(suffix):
                result = result[: -len(suffix)].strip()
                break

        # Try alias lookup on full cleaned name BEFORE noise stripping.
        # This ensures "Los Angeles FC" -> "lafc" fires before "fc" is stripped.
        if self._handler:
            pre_strip = self._handler.normalize_team(result)
            if pre_strip != result:
                return pre_strip
            # Already a canonical target → idempotent. Without this a
            # canonical containing a noise word ("man united", "dc united")
            # degrades on every re-normalization ("man", "dc"), so the
            # Kalshi-side event key never equals the Pinnacle-side key.
            if self._handler.is_canonical(result):
                return result

        # Strip noise words from start/end
        parts = result.split()
        parts = [p for p in parts if p not in self._NOISE_WORDS or len(parts) == 1]
        result = " ".join(parts)

        # Apply sector alias after noise stripping
        if self._handler:
            result = self._handler.normalize_team(result)

        return result

    def normalize_event_key(
        self,
        team_a: str,
        team_b: str,
        date_str: str,
        sector: str,
    ) -> str:
        """Build canonical event key from two team names and date."""
        norm_a = self.normalize(team_a).replace(" ", "_")
        norm_b = self.normalize(team_b).replace(" ", "_")
        return f"{sector}::{date_str}::{norm_a}_vs_{norm_b}"
