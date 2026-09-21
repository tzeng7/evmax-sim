"""Pure helpers for the alt-spread ladder replay.

These functions carry the join logic that
``scripts/backtest_spread_ladder_replay.py`` uses to score two spread pricings
against realized outcomes:

  * CDF     — the logged ``SpreadDistributionModel`` extrapolation off the main
              line (already stored as ``blended_true_prob`` on each row).
  * ladder  — the book's OWN devigged cover probability for that exact line,
              read from the archived Pinnacle alt-spread ladder
              (``archived_sharp_odds`` rows with ``::spread::<line>`` ids).

Kept out of the script so they can be unit-tested. The script owns the SQL; this
module owns the alignment math.

Ladder-record convention (from ``PinnacleGuestClient._parse_spread``): every
archived spread rung stores ``outcome_a`` = the FAVORITE and ``true_prob_a`` =
P(favorite covers ``spread_line``), where ``spread_line`` is the favorite's
handicap and is therefore NEGATIVE. A rung keyed by ``-7.5`` gives P(favorite
wins by more than 7.5).

Row-line convention (matches ``SpreadDistributionModel.predict``): a resolved
prediction row's ``line`` is signed from the YES side's perspective — NEGATIVE
when the YES side is the favorite laying points, POSITIVE when the YES side is
the underdog getting points. The YES-side cover probability at a rung is
therefore the favorite's cover prob when YES is the favorite, and its complement
when YES is the underdog.
"""

from __future__ import annotations

from typing import Optional

# Rung-distance buckets, in display order. Distance is |abs(row_line) -
# abs(main_line)| in points, i.e. how far the priced rung sits from the game's
# main line — the axis along which the normal-CDF gap-filler is expected to
# degrade relative to the book's own rung.
BUCKET_ORDER: tuple[str, ...] = (
    "at-line (<=1)",
    "near (1-4)",
    "mid (4-8)",
    "deep tail (>8)",
)


def rung_distance_bucket(distance: float) -> str:
    """Bucket a rung by its point-distance from the game's main line."""
    d = abs(distance)
    if d <= 1.0:
        return "at-line (<=1)"
    if d <= 4.0:
        return "near (1-4)"
    if d <= 8.0:
        return "mid (4-8)"
    return "deep tail (>8)"


def ladder_yes_prob(
    row_line: float,
    ladder: dict[float, float],
    tolerance: float = 0.5,
) -> Optional[float]:
    """YES-side cover probability at ``row_line`` from the archived ladder.

    ``ladder`` maps a favorite handicap (negative ``spread_line``) to
    ``true_prob_a`` = P(favorite covers that handicap).

    Returns the YES-side cover probability for the rung whose magnitude is
    closest to ``|row_line|`` within ``tolerance`` points, or ``None`` when no
    archived rung sits close enough (the caller then reports ladder = n/a for
    that row and the CDF price is the only one scored).
    """
    if not ladder or row_line == 0:
        return None

    target_mag = abs(row_line)
    best_key: Optional[float] = None
    best_gap = tolerance
    for cover_point in ladder:
        gap = abs(abs(cover_point) - target_mag)
        # ``<=`` so an exact/first hit at the tolerance edge still qualifies;
        # ties keep the earlier (equally-close) rung — either is the same line.
        if gap <= best_gap:
            best_gap = gap
            best_key = cover_point

    if best_key is None:
        return None

    fav_cover = ladder[best_key]
    # row_line < 0  → YES is the favorite laying points → favorite cover prob.
    # row_line > 0  → YES is the underdog getting points → complement.
    return fav_cover if row_line < 0 else 1.0 - fav_cover


def brier(pairs: list[tuple[float, int]]) -> float:
    """Mean Brier score for (predicted_prob, outcome) pairs.

    Returns NaN for an empty list so callers can render a blank cell without a
    ZeroDivisionError.
    """
    if not pairs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)
