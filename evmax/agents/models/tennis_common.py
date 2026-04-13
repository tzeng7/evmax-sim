"""Shared helpers for tennis model agents.

Player names arrive in many formats: "Jannik Sinner", "sinner j.", "j. sinner",
"de minaur a.". The resolver normalizes by surname so all tennis agents share
the same matching rules, and so duplicate state entries (a known issue when
seeding from multiple sources) resolve to the entry with the most signal.
"""

from __future__ import annotations

import re
from typing import Optional


def normalize_player(name: str) -> str:
    """Lowercase, strip apostrophes/hyphens/spaces. 'O''Connell' → 'oconnell'."""
    return re.sub(r"[\s''\-]+", "", name.lower().strip().rstrip("."))


def surname(name: str) -> str:
    """Extract surname token. 'sinner j.' → 'sinner', 'jannik sinner' → 'sinner'."""
    name = name.strip().rstrip(".")
    parts = name.split()
    if not parts:
        return name
    if len(parts[-1]) <= 2:   # last token is initial
        return " ".join(parts[:-1])
    return parts[-1]


def resolve_player(
    player: str,
    store: dict,
    weight_fn: Optional[callable] = None,
) -> Optional[str]:
    """Resolve a Pinnacle/Kalshi player label to a key in `store`.

    Tries: exact → surname match → normalized surname → multi-word prefix.
    When multiple candidates share a surname (e.g. duplicate seed entries
    'adrian mannarino' AND 'mannarino a.'), the optional weight_fn picks the
    entry with the highest signal (e.g. game count). Without weight_fn, the
    first candidate is returned.
    """
    if player in store:
        return player

    target = surname(player)
    candidates = [k for k in store if surname(k) == target]

    if not candidates:
        norm_player = normalize_player(player)
        candidates = [
            k for k in store
            if normalize_player(surname(k)) == norm_player
        ]

    if not candidates:
        candidates = [
            k for k in store
            if k.startswith(player + " ") or k.startswith(player + ".")
        ]

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if weight_fn is None:
        return candidates[0]
    return max(candidates, key=weight_fn)
