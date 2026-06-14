"""Exclude prediction rows produced by a superseded code state.

When the scanner is run from a stale branch — or before a model fix lands —
``ev_predictions`` rows get persisted that don't reflect the current pipeline.
Scoring a promotion gate against those rows gives a false read: the "baseball
June code contamination" (rows logged 2026-06-01 → 06-10 came from three
different code states) is exactly this failure. Of 49 resolved baseball-ML
shadow rows, only 3 were produced by the current code — the rest still ran
Poisson or predate the pitcher-required guard, so the headline Brier/ROI was
measuring old code, not what we'd ship live.

This module encodes the *documented* contamination signatures so the shadow
metrics / promotion path only ever scores current-code rows. It is the single
source of truth — both the SQL builder and the Python predicate derive from
``CONTAMINATION_RULES`` so they can't drift.

Each rule is a ``RowRule`` predicate over a single prediction row. A row is
contaminated if it matches ANY rule for its sector.

Baseball (see data/categories.yaml notes + CLAUDE.md):
  - ``model_sources`` containing ``poisson``: Poisson was removed from
    baseball on 2026-06-01 (``SUPPORTED_SECTORS`` in ``poisson_agent.py``).
    Any baseball row whose blend still lists poisson was scanned from pre-06-01
    code, where poisson contributed at its 0.30 class-weight fallback.
  - baseball ``moneyline`` WITHOUT ``pitcher`` in ``model_sources``: the
    pitcher-required guard shipped 2026-06-06 (``ev_gap_agent.py``, 80d491b).
    A pitcher-less baseball-ML row predates the guard and is a generic
    elo(+poisson)+sharp blend with no independent edge (−23% live ROI, n=13).
  - baseball ``spread`` with ``|line| > 1.5``: the alt-run-line gate
    (``_LOW_SCORING_MAX_ABS_LINE`` in ``models_ml/spread_distribution.py``,
    PR #13) rejects every baseball spread beyond the standard ±1.5 run line —
    the normal-CDF cover prob is unreliable that deep into a skewed margin
    distribution (−4.5 alt lines went 2-of-15 while the model predicted
    27–46%). Current code would never place these, so any resolved −2.5/−4.5
    row in the DB is pre-gate.

To add a sector rule later, append to ``CONTAMINATION_RULES`` — the metrics
command and any SQL caller pick it up automatically.
"""

from __future__ import annotations

from typing import Callable, Optional

# A rule inspects (market_type, model_sources, line) for a single row in its
# sector and returns True when the row is contaminated. ``market_type`` is the
# stored lowercase enum value ("moneyline" / "spread" / "total" /
# "player_prop"); ``model_sources`` is the "+"-joined contributor string (may
# be None/empty); ``line`` is the spread/total line (may be None).
RowRule = Callable[[Optional[str], Optional[str], Optional[float]], bool]

# Standard baseball run line. Beyond this the alt-run-line gate rejects the
# market (mirrors _LOW_SCORING_MAX_ABS_LINE["baseball"] in
# models_ml/spread_distribution.py — kept as a local literal to avoid importing
# a private from the models layer into cleanup).
_BASEBALL_MAX_ABS_SPREAD = 1.5


def _has(model_sources: Optional[str], token: str) -> bool:
    return token in (model_sources or "")


CONTAMINATION_RULES: dict[str, list[RowRule]] = {
    "baseball": [
        # Poisson removed from baseball 2026-06-01 — its presence dates the row.
        lambda mt, src, line: _has(src, "poisson"),
        # Pitcher-required guard shipped 2026-06-06; a pitcher-less ML row is
        # pre-guard generic arb.
        lambda mt, src, line: mt == "moneyline" and not _has(src, "pitcher"),
        # Alt-run-line gate (PR #13): current code rejects baseball spreads
        # beyond ±1.5, so a resolved −2.5/−4.5 row is pre-gate.
        lambda mt, src, line: (
            mt == "spread" and line is not None and abs(line) > _BASEBALL_MAX_ABS_SPREAD
        ),
    ],
}


def is_contaminated(
    sector: Optional[str],
    market_type: Optional[str],
    model_sources: Optional[str],
    line: Optional[float] = None,
) -> bool:
    """True if this row was produced by a superseded code state.

    Rows for sectors without rules are never flagged (returns False).
    """
    rules = CONTAMINATION_RULES.get((sector or "").lower())
    if not rules:
        return False
    return any(rule(market_type, model_sources, line) for rule in rules)
