"""Base sector handler ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import yaml

from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

ALIASES_DIR = Path(__file__).parent / "aliases"


class SectorHandler(ABC):
    """
    Abstract base for sector-specific market parsing and event normalization.

    Each sector handler:
      - Defines which prediction market sources it uses
      - Provides team name aliases for cross-source matching
      - Can parse sector-specific market structures
    """

    name: str  # e.g. "nfl"
    sharp_source: str  # "pinnacle" (all sectors use Pinnacle via TheOddsAPI)

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}
        self._load_aliases()

    def _load_aliases(self) -> None:
        """Load team name aliases from YAML file."""
        alias_file = ALIASES_DIR / f"{self.name}.yaml"
        if alias_file.exists():
            with open(alias_file) as f:
                data = yaml.safe_load(f) or {}
                self._aliases = data.get("aliases", {})

    def normalize_team(self, name: str) -> str:
        """
        Normalize a team name using the alias map.
        Returns canonical name (lowercase, stripped).
        """
        if not name:
            return ""
        cleaned = name.strip().lower()
        return self._aliases.get(cleaned, cleaned)

    def make_event_key(
        self,
        team_a: str,
        team_b: str,
        date_str: str,  # YYYY-MM-DD
    ) -> str:
        """
        Build a canonical event key for cross-source matching.
        Format: "{sector}::{date}::{team_a_norm}_vs_{team_b_norm}"
        """
        norm_a = self.normalize_team(team_a).replace(" ", "_")
        norm_b = self.normalize_team(team_b).replace(" ", "_")
        return f"{self.name}::{date_str}::{norm_a}_vs_{norm_b}"

    @abstractmethod
    def enrich_market(self, market: PredictionMarket) -> PredictionMarket:
        """
        Apply sector-specific enrichment to a market.
        May fix team names, set market type, add derived fields.
        """
        ...

    @abstractmethod
    def market_types_supported(self) -> list[str]:
        """Return list of market type strings this sector handles."""
        ...
