"""Golfer name canonicalisation — the join key across data + venues.

ESPN ("Tommy Fleetwood"), the PGA Tour feed ("Fleetwood, Tommy" or "Tommy
Fleetwood"), Kalshi (``yes_sub_title`` = "Tommy Fleetwood"), and PolyUS
(``title`` = "Tommy Fleetwood") all spell names slightly differently —
accents, suffixes (Jr/III), "Last, First" order, punctuation. ``canonical``
folds them to one comparable key so a market row lines up with its skill row.

Unlike team sports this is a light touch: golf uses FULL names (surnames
collide constantly — there are multiple "Kim"s, "Lee"s, "Smith"s on tour), so
we canonicalise to a normalised *full* name, not a surname.
"""

from __future__ import annotations

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT_RE = re.compile(r"[.\-']")
_WS_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def canonical(name: str) -> str:
    """Fold a golfer name to a comparable canonical key.

    - "Last, First" → "First Last"
    - strips accents, lowercases, removes . - ' punctuation
    - drops generational suffixes (Jr, III, …)
    - collapses whitespace

    >>> canonical("Matt Fitzpatrick")
    'matt fitzpatrick'
    >>> canonical("Fitzpatrick, Matt")
    'matt fitzpatrick'
    >>> canonical("Nicolás Echavarría")
    'nicolas echavarria'
    >>> canonical("Davis Thompson Jr.")
    'davis thompson'
    """
    if not name:
        return ""
    s = name.strip()
    # "Last, First" → "First Last"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    s = _strip_accents(s).lower()
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    tokens = [t for t in s.split(" ") if t and t not in _SUFFIXES]
    return " ".join(tokens)


def build_alias_index(canonical_names: list[str]) -> dict[str, str]:
    """Map every canonical name to itself (identity index).

    Kept as a seam: if a venue ever ships a nickname or an alternate spelling
    that ``canonical`` can't fold (e.g. "Cam" vs "Cameron"), a per-tournament
    alias map can be layered on top of this without touching callers.
    """
    return {n: n for n in canonical_names}
