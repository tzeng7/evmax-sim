"""NCAAF conference-tier lookup (Power-4 vs Group-of-Five).

Segmentation key for the CFB soft-market edge thesis: less-watched G5 games are
where Kalshi-vs-sharp mispricing is most likely to persist, while marquee P4
games are priced efficiently. This module maps a team (any source spelling) to
its tier and a matchup to a combined tier, so resolved CLV can be compared across
tiers — see `evmax cleanup shadow clv-tiers`.

The tier map (``evmax/sectors/aliases/ncaaf_tiers.yaml``) is GENERATED from ESPN
by ``scripts/build_ncaaf_tiers.py``. Its keys are the same canonical names the
ncaaf alias map produces, so a lookup on any spelling resolves through the ncaaf
sector handler's alias normalization first and lands on the matching key.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

_TIERS_YAML = Path(__file__).parent / "aliases" / "ncaaf_tiers.yaml"

P4 = "P4"          # ACC / Big 12 / Big Ten / SEC + Notre Dame
G5 = "G5"          # American / Mtn West / MAC / Sun Belt / C-USA / Pac-12 + G5 indeps
CROSS = "cross"    # one P4 side, one G5 side
FCS = "FCS"        # ≥1 side unmapped: a true FCS buy game or an unknown team

# Deterministic display order for the segmentation table (soft → hard market).
TIER_ORDER = (G5, CROSS, P4, FCS)


@lru_cache(maxsize=1)
def _tier_map() -> dict[str, str]:
    """canonical team name → 'P4' / 'G5'. Empty dict if the map isn't built."""
    if not _TIERS_YAML.exists():
        return {}
    data = yaml.safe_load(_TIERS_YAML.read_text(encoding="utf-8")) or {}
    return {
        k: v["tier"]
        for k, v in (data.get("tiers") or {}).items()
        if isinstance(v, dict) and "tier" in v
    }


@lru_cache(maxsize=1)
def _handler():
    # Lazy import: keep sectors import order free of a load-time cycle.
    from evmax.sectors.ncaaf import NCAAFHandler

    return NCAAFHandler()


def team_tier(name: str) -> Optional[str]:
    """Return 'P4' / 'G5' for an FBS team (any source spelling), else None.

    Resolves ``name`` through the ncaaf alias map to its canonical form — the
    same key the tier map uses — so 'TCU Horned Frogs', 'TCU' and 'tcu' all hit.
    An FCS or otherwise-unmapped team returns None.
    """
    if not name:
        return None
    canonical = _handler().normalize_team(name)
    return _tier_map().get(canonical)


def matchup_tier(team_a: Optional[str], team_b: Optional[str]) -> str:
    """Combined tier for a game.

    'P4' when both sides are P4, 'G5' when both are G5, 'cross' for one of each,
    and 'FCS' when either side is unmapped (an FCS buy game or an unknown team).
    """
    ta, tb = team_tier(team_a or ""), team_tier(team_b or "")
    if ta is None or tb is None:
        return FCS
    if ta == P4 and tb == P4:
        return P4
    if ta == G5 and tb == G5:
        return G5
    return CROSS


def split_event_title(event_title: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Best-effort split of a stored '<A> vs <B>' event_title into (A, B).

    Returns (None, None) when no known separator is present. Handled separators
    cover the log's 'A vs B' convention plus 'at'/'@' forms defensively.
    """
    if not event_title:
        return None, None
    for sep in (" vs. ", " vs ", " at ", " @ "):
        if sep in event_title:
            a, b = event_title.split(sep, 1)
            return a.strip(), b.strip()
    return None, None
