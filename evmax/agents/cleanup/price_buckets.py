"""Entry-price buckets for the favorite–longshot readout.

Segmentation key for the favorite–longshot question: does the blend over-state
cheap (underdog) contracts, and does the market move against us after entry on
them? The EV% gate mechanically rewards cheap contracts (a fixed +2pp edge is
+40% EV at 5c but +2.2% EV at 90c), and both the Kalshi literature (Bürgi/Deng/
Whelan 2026) and this system's own resolved rows show the blend sits above the
realized win rate in the 10–20c and 35–50c buckets while favorites sit at or
below it. This module buckets a resolved row by the price of OUR side at entry
so ``evmax cleanup shadow clv-prices`` and the promotion board can report
calibration (blend vs realized) and CLV per bucket — see ``clv-tiers`` for the
segmentation pattern this mirrors.

Bucket key = :func:`evmax.agents.cleanup.resolver.clv_entry_price` — the exact
price the row's CLV was measured against (placed fill for placed bets, else the
scan ask). NO-side rows (``:no`` market ids) already store their own side's ask
in ``kalshi_yes_price`` (``no_ask``) and their own side's probabilities, so no
YES/NO flip is applied here.
"""
from __future__ import annotations

from typing import Any, Optional

# Deterministic display order (cheap → expensive, then unbucketable).
BUCKET_ORDER = (
    "0-10", "10-20", "20-35", "35-50", "50-65", "65-80", "80-90", "90+", "unknown",
)

# Upper edge (exclusive) → label. Anything ≥ 0.90 lands in "90+".
_EDGES: tuple[tuple[float, str], ...] = (
    (0.10, "0-10"),
    (0.20, "10-20"),
    (0.35, "20-35"),
    (0.50, "35-50"),
    (0.65, "50-65"),
    (0.80, "65-80"),
    (0.90, "80-90"),
)

BUCKET_DESC = {
    "0-10": "0-10c · deep longshot",
    "10-20": "10-20c · longshot",
    "20-35": "20-35c · underdog",
    "35-50": "35-50c · slight dog",
    "50-65": "50-65c · slight fav",
    "65-80": "65-80c · favorite",
    "80-90": "80-90c · heavy fav",
    "90+": "90c+ · chalk",
    "unknown": "unknown · no usable entry price",
}


def price_bucket(price: Optional[float]) -> str:
    """Bucket label for an entry price in (0, 1); 'unknown' otherwise."""
    if price is None:
        return "unknown"
    try:
        p = float(price)
    except (TypeError, ValueError):
        return "unknown"
    if not (0.0 < p < 1.0):
        return "unknown"
    for edge, label in _EDGES:
        if p < edge:
            return label
    return "90+"


def _get(row: Any, key: str) -> Any:
    """Read ``key`` from a dict or sqlite3.Row, returning None when absent."""
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


def entry_price_for_row(row: Any) -> Optional[float]:
    """The price a row's CLV was anchored to (placed fill, else scan ask).

    Delegates to :func:`resolver.clv_entry_price` so the bucket can never drift
    from the CLV anchor. Works on ``sqlite3.Row`` and plain dicts; a row missing
    the price columns buckets as 'unknown'.
    """
    from evmax.agents.cleanup.resolver import clv_entry_price

    return clv_entry_price(
        _get(row, "placed"), _get(row, "placed_price"), _get(row, "kalshi_yes_price")
    )


def bucket_for_row(row: Any) -> str:
    """Bucket label for a resolved ev_predictions row."""
    return price_bucket(entry_price_for_row(row))


def validate_bucket(label: Optional[str]) -> Optional[str]:
    """Normalize/validate a user-supplied bucket label (None passes through)."""
    if label is None:
        return None
    norm = label.strip().lower()
    if norm not in BUCKET_ORDER:
        raise ValueError(
            f"price bucket must be one of {', '.join(BUCKET_ORDER)}; got {label!r}"
        )
    return norm


def calibration_summary(rows: list) -> dict:
    """Blend-vs-realized calibration for a bucket's resolved rows.

    Returns ``{n, win_rate, mean_blended, mean_sharp, blend_minus_realized_pp,
    sharp_minus_realized_pp}`` over rows that carry an outcome and a blended
    probability. A positive ``blend_minus_realized_pp`` means the blend
    over-states this bucket (the favorite–longshot signature on cheap
    contracts). ``mean_sharp`` is None when no row carries a sharp prob.
    """
    scored = [
        r for r in rows
        if _get(r, "outcome") is not None and _get(r, "blended_true_prob") is not None
    ]
    n = len(scored)
    if n == 0:
        return {
            "n": 0, "win_rate": None, "mean_blended": None, "mean_sharp": None,
            "blend_minus_realized_pp": None, "sharp_minus_realized_pp": None,
        }
    win_rate = sum(int(_get(r, "outcome")) for r in scored) / n
    mean_blended = sum(float(_get(r, "blended_true_prob")) for r in scored) / n
    sharp_vals = [
        float(_get(r, "sharp_true_prob")) for r in scored
        if _get(r, "sharp_true_prob") is not None
    ]
    mean_sharp = (sum(sharp_vals) / len(sharp_vals)) if sharp_vals else None
    return {
        "n": n,
        "win_rate": win_rate,
        "mean_blended": mean_blended,
        "mean_sharp": mean_sharp,
        "blend_minus_realized_pp": (mean_blended - win_rate) * 100,
        "sharp_minus_realized_pp": (
            (mean_sharp - win_rate) * 100 if mean_sharp is not None else None
        ),
    }
