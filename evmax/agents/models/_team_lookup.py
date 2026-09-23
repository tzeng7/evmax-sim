"""Shared model-side team lookup: map a source label to a model-state key.

Every generic model agent (elo, form, poisson, soccer xG, ncaaf_efficiency_v2)
stores per-team state under a canonical key and receives a raw Pinnacle label at
predict time ("Paris Saint-Germain", "Alabama State") or a slug / ESPN name at
update time. Before this module each agent carried its own copy of a lookup with
"last word" and "startswith / endswith" fallbacks that took the FIRST key that
matched. Those fallbacks priced the wrong team:

  * "paris saint-germain" → "paris" (Paris FC) in poisson — PSG ran at Paris
    FC's attack/defense in every Ligue 1 / UCL game.
  * "alabama state" → "alabama", "north dakota" → "north dakota state",
    "west georgia" → "georgia", "carritospain" → "pain" (no word boundary),
    "ence academy" → "ence" (an academy roster priced as the main team).
  * ``EloModelAgent.update`` READ through the fallback but WROTE to the raw
    key, so "alabama state hornets" was born with Alabama's 1807 rating.

The rule implemented here ("unique match"), for team sectors:

  1. the sector NameNormalizer canonical (alias map, noise words, and accent
     folding for soccer-like sectors) when it is a stored key — it outranks a
     raw exact key, because a stored raw spelling beside its canonical is a
     split left by an old alias gap ("boston celtics" gc=1 beside "celtics"
     gc=87; "middle tennessee state" gc=1 beside "middle tennessee" gc=64);
  2. the exact lowercased label;
  3. canonical equality — the unique stored key whose OWN canonical equals the
     label's canonical. This is what lets an alias fix take effect on state
     that was seeded under the old spelling ("montréal", "brighton hove
     albion") without hand-editing state JSON;
  4. word-boundary prefix / suffix fallbacks, ONLY when
       - the sector allows them (never for ncaaf, soccer, worldcup — see
         ``IDENTITY_ONLY_SECTORS``),
       - the words that differ carry no school/club-distinguishing token
         ("state", "a&m", "tech", "christian", "martin", "academy", "female",
         directional words, US state names …),
       - a label whose canonical is a registered team only matches a LONGER
         key that decorates it ("unlv" → "unlv rebels"), never a shorter one
         ("inter miami" → "inter"),
       - for college sectors, the differing words are never in front
         ("george washington" is not "washington"),
       - and exactly one key qualifies (nested "kansas" / "kansas state"
         candidates collapse to the more specific one).

Person-name sectors (ufc, tennis) normalize a full name to a bare surname, so
they keep exact-first order, skip tier 3, and fuzzy-match on the raw label only.

Write paths (``update`` / ``record_match``) resolve through
:func:`write_team_key` / :func:`identity_team_key`, which stop after tier 3: a
fuzzy match may at most READ a rating, never write into another team's state,
and the key an update reads is the key it writes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Optional

from evmax.matching.normalizer import NameNormalizer

# Sectors where only identity tiers (1–3) run. NCAAF: FCS opponents
# ("Alabama State", "Houston Christian") share a prefix with an FBS school and
# must stay unresolved (→ None / FCS pool). Soccer + worldcup: one shared,
# multi-league namespace where a place-name prefix is a different club
# ("Inter Turku" vs Inter, "Equatorial Guinea" vs Guinea); after the alias /
# accent fixes every archived Pinnacle label resolves through tiers 1–3, so the
# fuzzy tier there could only ever add wrong answers.
IDENTITY_ONLY_SECTORS: frozenset[str] = frozenset({"ncaaf", "soccer", "worldcup"})

# College names are "<location> [<mascot>]": a store key that is the TAIL of a
# label is a different, shorter-named school ("Central Florida" ≠ "Florida",
# "George Washington" ≠ "Washington", "West Georgia" ≠ "Georgia"). Only the
# mascot direction ("duke blue devils" ↔ "duke") is allowed there.
COLLEGE_SECTORS: frozenset[str] = frozenset({"ncaab", "ncaaw", "ncaaf"})

# Sectors whose normalizer maps a full name to a SURNAME ("jon jones" →
# "jones"). Canonical equality would make every same-surname fighter/player
# interchangeable, so tier 3 is skipped for them.
_SURNAME_NORMALIZER_SECTORS: frozenset[str] = frozenset({"ufc", "tennis"})

_US_STATES = (
    "alabama alaska arizona arkansas california colorado connecticut delaware "
    "florida georgia hawaii idaho illinois indiana iowa kansas kentucky "
    "louisiana maine maryland massachusetts michigan minnesota mississippi "
    "missouri montana nebraska nevada ohio oklahoma oregon pennsylvania "
    "tennessee texas utah vermont virginia washington wisconsin wyoming "
    "carolina dakota hampshire jersey mexico york"
)

# Words in the differing part of two names that make them different
# institutions / rosters rather than decorations (city, mascot, sponsor).
# School-name modifiers AFTER the shared part: "Alabama State", "Texas A&M"
# (normalized "texas a m"), "Tennessee Martin", "Miami Ohio", "Florida Atlantic".
_SCHOOL_MODIFIERS: frozenset[str] = frozenset(
    "state st st. tech a&m a&t a m t christian baptist martin poly polytechnic "
    "international lutheran methodist wesleyan commerce pacific atlantic gulf "
    "coastal upstate valley pine bluff oh fl fla".split()
) | frozenset(_US_STATES.split())
# Directional / regional words, on either side: "West Georgia", "Central Florida",
# "North Carolina Central", "Middle Tennessee".
_DIRECTIONAL: frozenset[str] = frozenset(
    "north south east west northern southern eastern western central middle "
    "northeast northwest southeast southwest northeastern northwestern "
    "southeastern southwestern mid upper lower se sw ne nw".split()
)
# Second teams / other squads of the same organisation: "ENCE Academy",
# "FURIA Female", "Gen.G GC", "MOUZ NXT", "Hanwha Life Challengers".
_SQUAD_MARKERS: frozenset[str] = frozenset(
    "academy academia akademie female fe fem women womens woman ladies girls "
    "femenino feminino feminine femminile frauen gc prospects rising youth "
    "young youngsters junior juniors next nxt challengers challenger global "
    "reserve reserves b ii iii 2 u17 u18 u19 u20 u21 u23".split()
)
# Differing words AFTER the shared part (the key is a prefix of the label, or
# the label a prefix of the key).
TAIL_DISTINGUISHING: frozenset[str] = _SCHOOL_MODIFIERS | _DIRECTIONAL | _SQUAD_MARKERS
# Differing words BEFORE the shared part. City / state names are decoration
# here ("G1 Colorado Rockies" → "rockies"; "St. Louis" is a city), so only
# directional words and squad markers disqualify.
HEAD_DISTINGUISHING: frozenset[str] = _DIRECTIONAL | _SQUAD_MARKERS

_NORMALIZERS: dict[str, NameNormalizer] = {}


def _normalizer(sector: str) -> NameNormalizer:
    norm = _NORMALIZERS.get(sector)
    if norm is None:
        norm = NameNormalizer(sector)
        _NORMALIZERS[sector] = norm
    return norm


def _canonical(sector: str, name: str) -> str:
    try:
        return _normalizer(sector).normalize(name) or ""
    except Exception:  # noqa: BLE001 — a normalizer failure must never break a lookup
        return ""


# Canonical-equality index per store: {canonical(key): [keys]}. Rebuilt when
# the store object or its size changes (keys are only ever added in-process).
_INDEX_CACHE: "OrderedDict[tuple[str, int], tuple[Mapping, int, dict[str, list[str]]]]" = OrderedDict()
_INDEX_CACHE_MAX = 64


def _canonical_index(sector: str, store: Mapping[str, Any]) -> dict[str, list[str]]:
    cache_key = (sector, id(store))
    hit = _INDEX_CACHE.get(cache_key)
    if hit is not None and hit[0] is store and hit[1] == len(store):
        _INDEX_CACHE.move_to_end(cache_key)
        return hit[2]
    index: dict[str, list[str]] = {}
    for key in store:
        canon = _canonical(sector, key)
        if canon:
            index.setdefault(canon, []).append(key)
    _INDEX_CACHE[cache_key] = (store, len(store), index)
    _INDEX_CACHE.move_to_end(cache_key)
    while len(_INDEX_CACHE) > _INDEX_CACHE_MAX:
        _INDEX_CACHE.popitem(last=False)
    return index


def _has_distinguishing_token(words: str, *, head: bool) -> bool:
    vocab = HEAD_DISTINGUISHING if head else TAIL_DISTINGUISHING
    return any(tok in vocab for tok in words.split())


def _fuzzy_key(
    sector: str, queries: tuple[str, ...], store: Mapping[str, Any], *, label_known: bool
) -> Optional[str]:
    """Word-boundary prefix/suffix fallback over the label forms in
    ``queries``; a key only when exactly one qualifies.

    ``label_known`` — the label's canonical is a registered team. It may then
    still match a LONGER key that decorates it ("unlv" → "unlv rebels"), but
    never a SHORTER one: a registered team missing from state has no rating and
    must not borrow the rating of a club whose name it contains ("inter miami"
    → "inter", "paris saint-germain" → "paris").
    """
    college = sector in COLLEGE_SECTORS
    qualifying: set[str] = set()
    for q in queries:
        for key in store:
            if not key or key == q:
                continue
            if q.startswith(key + " "):        # label = key + tail: "duke blue devils" → "duke"
                extra, at_head, key_shorter = q[len(key) + 1:], False, True
            elif key.startswith(q + " "):      # key = label + tail: "vcu" → "vcu rams"
                extra, at_head, key_shorter = key[len(q) + 1:], False, False
            elif q.endswith(" " + key):        # label = head + key: "g1 boston red sox" → "red sox"
                extra, at_head, key_shorter = q[: -len(key) - 1], True, True
            elif key.endswith(" " + q):        # key = head + label: "drx" → "kiwoom drx"
                extra, at_head, key_shorter = key[: -len(q) - 1], True, False
            else:
                continue
            if at_head and college:
                continue
            if key_shorter and label_known:
                continue
            if _has_distinguishing_token(extra, head=at_head):
                continue
            qualifying.add(key)
    # Nested candidates ("kansas" and "kansas state" both matching) collapse to
    # the most specific one; anything still ambiguous is refused.
    specific = [
        k for k in qualifying
        if not any(o != k and (o.startswith(k + " ") or o.endswith(" " + k)) for o in qualifying)
    ]
    return specific[0] if len(specific) == 1 else None


def resolve_team_key(
    sector: str,
    name: Optional[str],
    store: Mapping[str, Any],
    *,
    allow_fuzzy: bool = True,
) -> Optional[str]:
    """Return the key in ``store`` that names the same team as ``name``, or None.

    ``allow_fuzzy=False`` (see :func:`identity_team_key`) stops after the
    identity tiers — use it on every write path.
    """
    sector = (sector or "").lower()
    n = (name or "").lower().strip()
    if not n or not store:
        return None
    canon = _canonical(sector, n)
    if sector in _SURNAME_NORMALIZER_SECTORS:
        # Exact first: the canonical is a bare surname shared by many people,
        # and there is no canonical-equality tier for the same reason.
        if n in store:
            return n
        if canon and canon in store:
            return canon
    else:
        # Team sectors: the alias canonical outranks a raw exact key. When both
        # are stored the raw one is a split left by an old spelling ("middle
        # tennessee state" gc=1 beside "middle tennessee" gc=64; "alabama state
        # hornets" carrying Alabama's inherited rating) and the canonical is
        # the real record.
        if canon and canon in store:
            return canon
        if n in store:
            return n
        if canon:
            hits = _canonical_index(sector, store).get(canon)
            if hits and len(hits) == 1:
                return hits[0]
    if not allow_fuzzy or sector in IDENTITY_ONLY_SECTORS:
        return None
    label_known = bool(canon) and _normalizer(sector).is_known_team(canon)
    # The canonical is a second query form ("texas a&m" → "texas a m" matches
    # "texas a m aggies") — except in surname sectors, where it is a bare
    # surname and "apollo gomes" → "gomes" would match "denise gomes".
    if sector in _SURNAME_NORMALIZER_SECTORS or not canon or canon == n:
        queries: tuple[str, ...] = (n,)
    else:
        queries = (n, canon)
    return _fuzzy_key(sector, queries, store, label_known=label_known)


def identity_team_key(sector: str, name: Optional[str], store: Mapping[str, Any]) -> Optional[str]:
    """Identity-only resolution (tiers 1–3) for WRITE paths.

    Returns the existing key for this team, or None when the team is new.
    Never resolves through a fuzzy fallback, so an update can't write into (or
    inherit from) another team.
    """
    return resolve_team_key(sector, name, store, allow_fuzzy=False)


def write_team_key(sector: str, name: Optional[str], store: Mapping[str, Any]) -> str:
    """The key an UPDATE must read and write for ``name``.

    The team's existing key when an identity tier finds one; otherwise (a new
    team) its alias-registered canonical ("alabama state hornets" → "alabama
    state"), so the next lookup under any alias lands on the same record;
    otherwise the lowercased name itself (no registered canonical — e.g. a
    noise-word strip like "team a" → "a" is not trusted to name a team).
    """
    sector = (sector or "").lower()
    n = (name or "").lower().strip()
    found = identity_team_key(sector, n, store)
    if found:
        return found
    if sector not in _SURNAME_NORMALIZER_SECTORS:
        canon = _canonical(sector, n)
        if canon and _normalizer(sector).is_known_team(canon):
            return canon
    return n
