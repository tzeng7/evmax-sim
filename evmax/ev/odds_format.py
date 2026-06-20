"""American-odds formatting — single source for the CLI table and notifications.

`evmax.ev` is a dependency leaf, so both the CLI and the notifier can import
this without a cycle.
"""

from __future__ import annotations


def american_odds(prob: float) -> str:
    """Convert an implied probability (0–1) to an American-odds string.

    Returns "N/A" outside the open interval (0, 1). Favorites (prob ≥ 0.5)
    render negative (e.g. "-108"); underdogs render with a leading "+".
    """
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        return f"{-round(prob / (1 - prob) * 100)}"
    return f"+{round((1 - prob) / prob * 100)}"


def cents(prob: float) -> str:
    """Format an implied probability (0–1) as a Kalshi cent price.

    Kalshi is denominated in cents (1–99¢) — that's the unit you actually
    trade and place limit orders in, so it's what the CLI and dashboard show.
    Plain binary markets quote whole cents, but combo/spread markets price
    continuously and the true ask can be sub-cent (e.g. a displayed 12¢ that
    really fills at 12.7¢). We keep one decimal and trim a trailing ``.0`` so
    ``0.12 → "12¢"`` while a fixed-point ask reads ``"12.7¢"``. Returns "N/A"
    outside the open interval (0, 1).
    """
    if prob <= 0 or prob >= 1:
        return "N/A"
    s = f"{prob * 100:.1f}".rstrip("0").rstrip(".")
    return f"{s}¢"


def cents_american(prob: float) -> str:
    """Both units: Kalshi cents (primary) with the American-odds equivalent.

    e.g. ``0.127 → "12.7¢ (+685)"``. Cents is the unit you place limit orders
    in; the American figure is the familiar reference. Returns "N/A" outside
    the open interval (0, 1).
    """
    c = cents(prob)
    if c == "N/A":
        return "N/A"
    return f"{c} ({american_odds(prob)})"
