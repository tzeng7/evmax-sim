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


def dehyphenate(name: str) -> str:
    """Treat hyphens as spaces so 'auger-aliassime' tokenizes like the
    space-form state key 'felix auger aliassime' (Tennis Abstract's spelling)."""
    return re.sub(r"-+", " ", name).strip()


def surname(name: str) -> str:
    """Extract surname token. 'sinner j.' → 'sinner', 'jannik sinner' → 'sinner'."""
    name = name.strip().rstrip(".")
    parts = name.split()
    if not parts:
        return name
    if len(parts[-1]) <= 2:   # last token is initial
        return " ".join(parts[:-1])
    return parts[-1]


def given_name(name: str) -> str:
    """The non-surname portion of a name, used to tell same-surname players
    apart. 'jannik sinner' → 'jannik', 'xin yu wang' → 'xin yu',
    'mannarino a.' → 'a' (trailing-initial form), 'sinner' → ''.
    """
    name = name.strip().rstrip(".")
    parts = name.split()
    if len(parts) < 2:
        return ""
    if len(parts[-1]) <= 2:   # trailing initial: 'mannarino a.'
        return parts[-1]
    return " ".join(parts[:-1])


def _given_compatible(a: str, b: str) -> bool:
    """Whether two given-name fragments could belong to the same person.

    Normalized equality ('xin yu' == 'xinyu'), or one is a prefix of the
    other (initial 'j' vs 'jannik', 'alex' vs 'alexander'). An empty fragment
    can't disqualify anything.
    """
    if not a or not b:
        return True
    na, nb = normalize_player(a), normalize_player(b)
    return na.startswith(nb) or nb.startswith(na)


def resolve_player(
    player: str,
    store: dict,
    weight_fn: Optional[callable] = None,
) -> Optional[str]:
    """Resolve a Pinnacle/Kalshi player label to a key in `store`.

    Tries: exact → dehyphenated exact → surname match (hyphen-insensitive)
    → normalized surname → multi-word prefix.

    Hyphenated surnames are matched against space-form state keys — Tennis
    Abstract stores 'felix auger aliassime', while Kalshi/Pinnacle send
    'Felix Auger-Aliassime' — by treating hyphens as spaces on both sides
    before tokenizing.

    When multiple candidates share a surname, the query's given name (if it
    has one) narrows them first — 'xinyu wang' resolves to 'xin yu wang',
    never 'xiyu wang'. If the survivors are still genuinely different people
    (different first initials, e.g. the Cerundolo brothers queried by bare
    surname), returns None: wrong-player data is worse than no data. Only
    same-person duplicates (e.g. seed variants 'adrian mannarino' AND
    'mannarino a.') fall through to weight_fn, which picks the entry with
    the highest signal (e.g. game count). Without weight_fn, the first
    candidate is returned.
    """
    if player in store:
        return player

    dehyph = dehyphenate(player)
    if dehyph != player and dehyph in store:
        return dehyph

    target = surname(dehyph)
    candidates = [k for k in store if surname(dehyphenate(k)) == target]

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
            or k.startswith(dehyph + " ") or k.startswith(dehyph + ".")
        ]

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Same-surname collision: narrow by the query's given name when present.
    q_given = given_name(dehyph)
    if q_given:
        narrowed = [
            k for k in candidates
            if _given_compatible(q_given, given_name(dehyphenate(k)))
        ]
        if narrowed:
            candidates = narrowed
        if len(candidates) == 1:
            return candidates[0]

    # Distinct players (different first initials) are unresolvable from this
    # query — bail out rather than return an arbitrary wrong player.
    initials = {
        normalize_player(given_name(dehyphenate(k)))[:1]
        for k in candidates
        if given_name(dehyphenate(k))
    }
    if len(initials) > 1:
        return None

    if weight_fn is None:
        return candidates[0]
    return max(candidates, key=weight_fn)
