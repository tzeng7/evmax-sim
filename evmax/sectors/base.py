"""Base sector handler ABC."""

from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import structlog
import yaml

from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

ALIASES_DIR = Path(__file__).parent / "aliases"

logger = structlog.get_logger(__name__)

# Letters that NFKD does NOT decompose into base + combining mark, so a plain
# "drop combining marks" pass would keep them (Bodø, København, Łódź, Straße).
_NON_DECOMPOSING = str.maketrans({
    "ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "ł": "l", "đ": "d",
    "ð": "d", "þ": "th", "ı": "i",
    "Ø": "O", "Æ": "AE", "Œ": "OE", "ẞ": "SS", "Ł": "L", "Đ": "D",
    "Ð": "D", "Þ": "TH",
})


def fold_accents(text: str) -> str:
    """Strip diacritics: "Montréal" → "Montreal", "Bodø" → "Bodo".

    Used only by sectors whose handler sets ``fold_accents = True`` (the
    soccer-like sectors): ESPN keeps accents ("CF Montréal", "Alavés") while
    Pinnacle and Kalshi drop them, so without folding the seed-side and
    live-side canonical names disagree and the club silently gets no rating.
    """
    if not text or text.isascii():
        return text
    decomposed = unicodedata.normalize("NFKD", text.translate(_NON_DECOMPOSING))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class SectorHandler(ABC):
    """
    Abstract base for sector-specific market parsing and event normalization.

    Each sector handler:
      - Defines which prediction market sources it uses
      - Provides team name aliases for cross-source matching
      - Can parse sector-specific market structures
    """

    name: str  # e.g. "nfl"
    sharp_source: str  # "pinnacle" (all sectors use the Pinnacle guest API)
    # When True, every name (and every alias key/target) is accent-folded
    # before lookup, so "CF Montréal" (ESPN) and "CF Montreal" (Pinnacle)
    # reach the same canonical. Off by default: person-name sectors (UFC) and
    # NCAAF ("san josé state" is an accented canonical) key state on the
    # accented form, so folding them would orphan existing ratings.
    fold_accents: bool = False

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}
        self._load_aliases()

    def _fold(self, name: str) -> str:
        return fold_accents(name) if self.fold_accents else name

    def _load_aliases(self) -> None:
        """Load team name aliases from YAML file."""
        alias_file = ALIASES_DIR / f"{self.name}.yaml"
        if alias_file.exists():
            with open(alias_file) as f:
                data = yaml.safe_load(f) or {}
                self._aliases = data.get("aliases", {}) or {}
        if self.fold_accents and self._aliases:
            folded: dict[str, str] = {}
            for key, target in self._aliases.items():
                fk, ft = fold_accents(str(key)), fold_accents(str(target))
                prev = folded.get(fk)
                if prev is not None and prev != ft:
                    # Two alias keys that differ only by accents point at
                    # different clubs — keep the first, never silently merge.
                    logger.warning(
                        "alias_accent_fold_collision",
                        sector=self.name, key=fk, kept=prev, dropped=ft,
                    )
                    continue
                folded[fk] = ft
            self._aliases = folded

    def is_canonical(self, name: str) -> bool:
        """True when `name` is already a canonical alias TARGET.

        Lets NameNormalizer short-circuit before noise-word stripping so
        normalization is idempotent: "man united" must stay "man united"
        on a second pass instead of collapsing to "man" (2026-09-04 — the
        Kalshi KXUCLGAME Man United vs Sabah key became sabah_vs_man and
        fuzzy-scored 77 against Pinnacle's man_united_vs_sabah).
        """
        if not self._aliases:
            return False
        canon = getattr(self, "_canonical_set", None)
        if canon is None:
            canon = set(self._aliases.values())
            self._canonical_set = canon
        return self._fold(name) in canon

    def normalize_team(self, name: str) -> str:
        """
        Normalize a team name using the alias map.
        Returns canonical name (lowercase, stripped; accent-folded when the
        handler sets ``fold_accents``).
        """
        if not name:
            return ""
        cleaned = self._fold(name.strip().lower())
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
