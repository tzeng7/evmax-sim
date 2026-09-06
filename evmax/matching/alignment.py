"""YES-side alignment: which sharp outcome does a venue's YES market price?

Event matching (``MatchingEngine.match``) answers "which Pinnacle record is this
Kalshi/PolyUS market about?". This module answers the SECOND, separate
question: "which side of that record is the YES contract?" — and it is the
ONLY place that question is answered. Every downstream consumer (EV math,
spread/total distributions, injury/standings/playoff nudges, the NO-side
derivation) reads the resulting :class:`Alignment` instead of comparing
strings again.

Design rules (2026-09-05, after the NCAAF "2000% EV" incident):

* Once the event is matched there are exactly two (or three) candidate sides,
  so alignment is a closed-world choice. Canonical-name EQUALITY decides it.
  Fuzzy scoring is never used here — ``rapidfuzz.token_set_ratio`` scores
  "western michigan" vs "michigan" at 100, and the old outcome-A-first check
  priced the Western Michigan YES contract at Michigan's 97%.
* Every weaker rule (token subset, venue-code prefix/acronym) must hit
  EXACTLY ONE side. A rule that matches both sides is ambiguous and yields
  nothing — it never falls through to "pick A". Same-surname tennis draws
  resolve to None for the same reason.
* Unresolvable ⇒ ``None`` ⇒ the market is NOT priced (fail-clear, the same
  stance the scan takes on a Pinnacle outage). A wrong side is a booked loss;
  a dropped market is a missed play.
* The price-distance fallback survives only for the esports sectors whose
  Kalshi YES labels are bare ticker codes without a series code map
  (``PRICE_FALLBACK_SECTORS``); it is logged with ``method="price"`` so its
  usage can be audited and retired once those maps exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType, PredictionMarket
from evmax.models.odds import SharpOdds

DRAW_TOKENS = frozenset({"tie", "draw", "x", "draw/tie"})

# Sectors whose Kalshi YES label is a bare ticker code ("dsg", "7d", "keyd")
# with no series-scoped code map yet. Only here may the ask price arbitrate a
# side the names could not. Everything else fails clear.
PRICE_FALLBACK_SECTORS = frozenset({"lol", "cs2"})

# Price fallback thresholds (ARCH-7): the closer side must be tight AND
# meaningfully closer than the farther side, so near-coin-flip markets can
# never be force-aligned to an arbitrary outcome.
PRICE_MAX_CLOSE_DIST = 0.04
PRICE_MIN_GAP = 0.05

# Sanity flag: assigned side far from the ask while the OTHER side sits on it.
SUSPECT_OTHER_MAX_DIST = 0.04
SUSPECT_YES_MIN_DIST = 0.25


class YesOutcome(str, Enum):
    A = "a"
    B = "b"
    DRAW = "draw"
    OVER = "over"
    UNDER = "under"


@dataclass(frozen=True)
class Alignment:
    """Resolved YES side of one venue market against its matched sharp record."""

    outcome: YesOutcome
    method: str            # canonical | tokens | venue_team | price | draw | total
    yes_canonical: str     # the canonical name the YES side resolved to ("" for totals)

    @property
    def is_outcome_b(self) -> bool:
        return self.outcome is YesOutcome.B

    @property
    def is_under(self) -> bool:
        return self.outcome is YesOutcome.UNDER

    def sharp_prob(self, sharp: SharpOdds) -> Optional[float]:
        """The devigged sharp probability of the YES side (None if the record lacks it)."""
        if self.outcome is YesOutcome.A:
            return sharp.true_prob_a
        if self.outcome is YesOutcome.B:
            return sharp.true_prob_b
        if self.outcome is YesOutcome.DRAW:
            return sharp.true_prob_draw
        if self.outcome is YesOutcome.OVER:
            return sharp.true_prob_over
        return sharp.true_prob_under


# ---------------------------------------------------------------------------
# String helpers — deliberately dumb
# ---------------------------------------------------------------------------

_APOSTROPHES = re.compile(r"['’‘]")
_NON_WORD = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def clean(name: Optional[str]) -> str:
    """Lowercase, drop apostrophes, turn punctuation into spaces, collapse whitespace."""
    s = (name or "").lower().strip()
    s = _APOSTROPHES.sub("", s)
    s = _NON_WORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _tokens(name: str) -> set[str]:
    return set(clean(name).split())


def _acronym(name: str) -> str:
    return "".join(w[0] for w in clean(name).split() if w)


def _exactly_one(hit_a: bool, hit_b: bool) -> Optional[YesOutcome]:
    if hit_a and not hit_b:
        return YesOutcome.A
    if hit_b and not hit_a:
        return YesOutcome.B
    return None


def _equal(yes: str, side: str) -> bool:
    return bool(yes) and yes == side


def _token_subset(yes: str, side: str) -> bool:
    """Every YES token appears in the side ("sinner" ⊆ "jannik sinner").

    Requires ≥3 chars so a stray single letter can't match. Because the caller
    demands exactly one hit, "michigan" vs {"michigan", "western michigan"}
    is caught by the exact rule first and never reaches here as an ambiguous
    subset — and "florida" vs {"south florida", "florida international"} is
    correctly ambiguous.
    """
    if len(yes) < 3:
        return False
    y = _tokens(yes)
    return bool(y) and y <= _tokens(side)


def _code_match(code: str, team: str) -> bool:
    """Venue ticker code against one of the market's OWN two teams.

    Closed world: the two candidates are the teams the venue itself listed on
    the market, so prefix/acronym rules are safe here (they are NOT safe against
    sharp labels, where a third team's name could collide).
    """
    c = clean(code)
    t = clean(team)
    if not c or not t:
        return False
    if t.startswith(c):
        return True
    if any(w.startswith(c) for w in t.split()):
        return True
    return c == _acronym(t)


def resolve_side(yes: str, side_a: str, side_b: str) -> Optional[tuple[YesOutcome, str]]:
    """Pick which of two already-cleaned/canonical sides ``yes`` names.

    Returns ``(outcome, method)`` or None. Rules are tried strongest-first and
    each must hit exactly one side.
    """
    rules: list[tuple[str, Callable[[str, str], bool]]] = [
        ("canonical", _equal),
        ("tokens", _token_subset),
    ]
    for method, pred in rules:
        out = _exactly_one(pred(yes, side_a), pred(yes, side_b))
        if out is not None:
            return out, method
    return None


def price_align(ask: float, prob_a: Optional[float], prob_b: Optional[float]) -> Optional[YesOutcome]:
    """Last-resort side pick from the ask price (esports only — see module doc)."""
    if prob_a is None or prob_b is None:
        return None
    dist_a = abs(ask - prob_a)
    dist_b = abs(ask - prob_b)
    closer = min(dist_a, dist_b)
    gap = abs(dist_a - dist_b)
    if closer > PRICE_MAX_CLOSE_DIST or gap < PRICE_MIN_GAP:
        return None
    return YesOutcome.A if dist_a < dist_b else YesOutcome.B


# ---------------------------------------------------------------------------
# The alignment
# ---------------------------------------------------------------------------

def align_yes_side(
    market: PredictionMarket,
    sharp: SharpOdds,
    normalizer: NameNormalizer,
) -> Optional[Alignment]:
    """Resolve which sharp outcome the market's YES contract prices.

    Order:
      1. totals → OVER/UNDER from the YES label; draws → DRAW.
      2. canonical equality, then unique token-subset, of the YES label against
         the sharp labels (both sides normalized through the sector handler,
         and also compared raw-cleaned in case normalization diverges).
      3. venue-team step: resolve the YES label against the market's OWN two
         teams (equality / token subset / ticker-code prefix+acronym, exactly
         one hit), then map that team onto the sharp side by rule 2.
      4. price fallback — ``PRICE_FALLBACK_SECTORS`` only.
      5. None: not priced.
    """
    yes_raw = market.yes_team or ""
    yes_clean = clean(yes_raw)

    if market.market_type == MarketType.total:
        if sharp.true_prob_over is None or sharp.true_prob_under is None:
            return None
        out = YesOutcome.UNDER if yes_clean == "under" else YesOutcome.OVER
        return Alignment(out, "total", "")

    if yes_clean in DRAW_TOKENS or (yes_raw or "").lower().strip() in DRAW_TOKENS:
        if sharp.true_prob_draw is None:
            return None
        return Alignment(YesOutcome.DRAW, "draw", "draw")

    if not yes_clean:
        return None

    a_raw = clean(sharp.outcome_a_label)
    b_raw = clean(sharp.outcome_b_label)
    a_c = clean(normalizer.normalize(sharp.outcome_a_label or ""))
    b_c = clean(normalizer.normalize(sharp.outcome_b_label or ""))
    yes_c = clean(normalizer.normalize(yes_raw))

    # 2. YES label vs sharp labels (canonical, then raw-cleaned).
    for yes_s, sa, sb in ((yes_c, a_c, b_c), (yes_clean, a_raw, b_raw), (yes_c, a_raw, b_raw)):
        r = resolve_side(yes_s, sa, sb)
        if r is not None:
            out, method = r
            return Alignment(out, method, sa if out is YesOutcome.A else sb)

    # 3. Venue-team step: which of the market's own teams is YES, then map it.
    home_raw = market.team_home or ""
    away_raw = market.team_away or ""
    if home_raw and away_raw:
        home_c = clean(normalizer.normalize(home_raw))
        away_c = clean(normalizer.normalize(away_raw))
        which: Optional[YesOutcome] = None
        r = resolve_side(yes_c, home_c, away_c) or resolve_side(yes_clean, clean(home_raw), clean(away_raw))
        if r is not None:
            which = r[0]
        else:
            which = _exactly_one(_code_match(yes_clean, home_raw), _code_match(yes_clean, away_raw))
        if which is not None:
            team_c = home_c if which is YesOutcome.A else away_c
            team_raw = clean(home_raw if which is YesOutcome.A else away_raw)
            for t, sa, sb in ((team_c, a_c, b_c), (team_raw, a_raw, b_raw), (team_c, a_raw, b_raw)):
                r2 = resolve_side(t, sa, sb)
                if r2 is not None:
                    out = r2[0]
                    return Alignment(out, "venue_team", sa if out is YesOutcome.A else sb)

    # 4. Price fallback — esports ticker codes only.
    if (market.sector or "").lower() in PRICE_FALLBACK_SECTORS:
        out = price_align(market.yes_price, sharp.true_prob_a, sharp.true_prob_b)
        if out is not None:
            return Alignment(out, "price", a_c if out is YesOutcome.A else b_c)

    return None


def alignment_looks_suspect(alignment: Alignment, ask: float, sharp: SharpOdds) -> bool:
    """True when the OTHER two-way side sits on the ask while the assigned side is far.

    A flag, not a resolver: it is logged so a mis-mapped alias surfaces in the
    scan log instead of as a four-digit EV row.
    """
    if alignment.outcome not in (YesOutcome.A, YesOutcome.B):
        return False
    p_yes = alignment.sharp_prob(sharp)
    p_other = sharp.true_prob_b if alignment.outcome is YesOutcome.A else sharp.true_prob_a
    if p_yes is None or p_other is None:
        return False
    return abs(ask - p_other) <= SUSPECT_OTHER_MAX_DIST and abs(ask - p_yes) >= SUSPECT_YES_MIN_DIST
