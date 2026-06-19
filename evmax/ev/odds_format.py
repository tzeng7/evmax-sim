"""American-odds formatting — single source for the CLI table and notifications.

`evmax.ev` is a dependency leaf, so both the CLI and the notifier can import
this without a cycle. (Note: `cli/commands/opportunities.py` has a separate,
intentionally different decimal-odds formatter with truncation semantics — it
is NOT replaced by this and should not be, see the drift audit.)
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
