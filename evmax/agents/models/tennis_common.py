"""Shared helpers for tennis model agents.

Player names arrive in many formats: "Jannik Sinner", "sinner j.", "j. sinner",
"de minaur a.". The resolver normalizes by surname so all tennis agents share
the same matching rules, and so duplicate state entries (a known issue when
seeding from multiple sources) resolve to the entry with the most signal.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional

# Letters NFKD can't decompose (stroked/ligature forms) — ascii-ignore would
# delete them outright ('đorđe' → 'ore'), so map them explicitly first.
_ACCENT_OVERRIDES = str.maketrans({
    "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss",
    "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D",
})


# The string helpers below are pure str → str functions called millions of
# times per scan (resolve_player historically re-derived them for every store
# key on every call — 23M unicodedata iterations per tennis cycle, ~15s of
# event-loop CPU). lru_cache turns the repeated derivations into dict hits.
# Rosters are ~1-2k names and queries a few thousand distinct strings, so the
# caches stay small; maxsize bounds them defensively for pathological input.
@lru_cache(maxsize=65536)
def fold_accents(name: str) -> str:
    """Strip diacritics: 'duško todorović' → 'dusko todorovic'.

    ESPN state keys carry accented spellings while Kalshi/Pinnacle send
    ASCII, so every comparison in the resolution cascade folds both sides.
    """
    name = name.translate(_ACCENT_OVERRIDES)
    return "".join(
        c for c in unicodedata.normalize("NFKD", name)
        if not unicodedata.combining(c)
    )


@lru_cache(maxsize=65536)
def normalize_player(name: str) -> str:
    """Lowercase, fold accents, strip apostrophes/hyphens/spaces.
    'O''Connell' → 'oconnell', 'Peña' → 'pena'."""
    return re.sub(r"[\s''\-]+", "", fold_accents(name).lower().strip().rstrip("."))


@lru_cache(maxsize=65536)
def dehyphenate(name: str) -> str:
    """Treat hyphens as spaces so 'auger-aliassime' tokenizes like the
    space-form state key 'felix auger aliassime' (Tennis Abstract's spelling)."""
    return re.sub(r"-+", " ", name).strip()


@lru_cache(maxsize=65536)
def surname(name: str) -> str:
    """Extract surname token. 'sinner j.' → 'sinner', 'jannik sinner' → 'sinner'."""
    name = name.strip().rstrip(".")
    parts = name.split()
    if not parts:
        return name
    if len(parts[-1]) <= 2:   # last token is initial
        return " ".join(parts[:-1])
    return parts[-1]


@lru_cache(maxsize=65536)
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


class _StoreIndex:
    """Precomputed lookup structures for one store's key set.

    ``resolve_player`` historically recomputed surname(fold_accents(
    dehyphenate(k))) for EVERY store key on EVERY call — an O(roster) scan
    per resolution that dominated the tennis scan's CPU. The index derives
    those forms once per store and answers the same three candidate queries:

      by_surname       surname(fold_accents(dehyphenate(k)))  → [keys]
      by_norm_surname  normalize_player(surname(k))           → [keys]
      folded_keys      [(fold_accents(dehyphenate(k)), k)]  — prefix fallback

    Candidate lists preserve store insertion order, so the "first candidate
    wins when no weight_fn" tie-break behaves exactly as the linear scan did.
    """

    __slots__ = ("by_surname", "by_norm_surname", "folded_keys")

    def __init__(self, store: dict) -> None:
        self.by_surname: dict[str, list[str]] = {}
        self.by_norm_surname: dict[str, list[str]] = {}
        self.folded_keys: list[tuple[str, str]] = []
        for k in store:
            fk = fold_accents(dehyphenate(k))
            self.by_surname.setdefault(surname(fk), []).append(k)
            self.by_norm_surname.setdefault(normalize_player(surname(k)), []).append(k)
            self.folded_keys.append((fk, k))


# Index cache keyed by store identity, validated by a (len, first-key)
# fingerprint. The index depends only on the store's KEY SET: value updates
# (rating changes) never invalidate it; key additions/removals change len (or
# the dict object itself — seeds full-replace state dicts, giving a new id).
# The residual stale case — an equal number of key deletions and insertions
# on the same live dict that also preserves the first key — does not occur in
# this codebase's state lifecycle (update() adds keys; reseeds replace dicts).
# Bounded to keep pathological callers from growing it without limit.
_STORE_INDEX_CACHE: dict[int, tuple[tuple, "_StoreIndex"]] = {}
_STORE_INDEX_CACHE_MAX = 64


def _get_store_index(store: dict) -> _StoreIndex:
    key = id(store)
    fp = (len(store), next(iter(store), None))
    hit = _STORE_INDEX_CACHE.get(key)
    if hit is not None and hit[0] == fp:
        return hit[1]
    idx = _StoreIndex(store)
    if len(_STORE_INDEX_CACHE) >= _STORE_INDEX_CACHE_MAX:
        _STORE_INDEX_CACHE.clear()
    _STORE_INDEX_CACHE[key] = (fp, idx)
    return idx


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

    All comparisons are accent-insensitive: ESPN-seeded stores (UFC) carry
    diacritic spellings ('duško todorović') while Kalshi/Pinnacle send ASCII
    ('Dusko Todorovic'), so both sides fold through `fold_accents` before
    comparing. The returned key is always the store's own spelling.

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

    # Same three-stage candidate cascade as the historical linear scans,
    # answered from the per-store index (O(1) for the two surname stages;
    # the prefix fallback scans precomputed folded forms on the rare path).
    index = _get_store_index(store)
    folded = fold_accents(dehyph)
    target = surname(folded)
    candidates = index.by_surname.get(target, [])

    if not candidates:
        norm_player = normalize_player(player)
        candidates = index.by_norm_surname.get(norm_player, [])

    if not candidates:
        prefix_space = folded + " "
        prefix_dot = folded + "."
        candidates = [
            k for fk, k in index.folded_keys
            if fk.startswith(prefix_space) or fk.startswith(prefix_dot)
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
