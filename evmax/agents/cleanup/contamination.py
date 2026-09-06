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

WNBA (see data/categories.yaml notes + project_wnba_spread_no_edge memory):
  - ``spread`` with ``possession_sim`` in ``model_sources``:
    ``SPREAD_SIM_WEIGHT_OVERRIDES{wnba:0}`` (2026-06-19, ``ev_gap_agent.py``)
    retired the possession-sim blend from WNBA spread pricing. Current code
    prices spread as the pure Pinnacle-CDF ``SpreadDistributionModel``
    (``model_sources == "sharp+spread_dist"``, no ``possession_sim``). A wnba
    spread row whose blend still lists ``possession_sim`` was priced by the old
    ``sim_weight=0.35`` regime, which the leak-free walk-forward
    (``scripts/backtest_wnba_spread_walkforward.py``) showed was behind the
    market and degraded both Brier and bet selection. Excluding these is what
    lets ``evmax cleanup shadow clv wnba -m spread`` score only current pricing
    without a manual ``--since``.

Soccer (see MIN_NONSHARP_MODELS in ``ev_gap_agent.py``):
  - soccer ``moneyline`` with zero non-sharp model tokens: the sharp-only
    guard shipped 2026-07-19. Before it, MLS games (scanned from 2026-07-09
    but never seeded) produced ``model_sources='sharp'`` rows that logged as
    LIVE plays — sharp-passthrough arb with no model edge. Current code
    demotes those to shadow, so any live-mode sharp-only soccer ML row is
    pre-guard.

NCAAF (see the NCAAF Efficiency row in CLAUDE.md):
  - ``moneyline`` with the v1 ``ncaaf_efficiency`` token but NOT
    ``ncaaf_efficiency_v2``: v2 (2026-09-03) replaced the EPA-only margin
    with EPA + opponent-adjusted success rate on an FPI-mixed preseason
    prior (held-out Brier −5 to −7/1000 on 2023/2024/2025) and renamed the
    model so the promotion sample only scores v2-priced rows — the
    pitcher → pitcher_v2 precedent. Rows without either token (elo+sharp
    early-season blends) are legitimate and stay clean.

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


# Tokens that are not independent model signal (sharp anchor, adjustment
# layers, pricing derivations, side flags). Local literal mirroring
# _NON_MODEL_TOKENS in evmax/agents/odds/ev_gap_agent.py — kept local per
# this file's convention of not importing model-layer internals (see
# _BASEBALL_MAX_ABS_SPREAD above).
_NON_MODEL_TOKENS = frozenset({
    "", "sharp", "sharp(capped)", "injury", "late_news", "rest", "playoff",
    "advance_derived", "spread_dist", "total_dist", "no_side",
})


def _nonsharp_count(model_sources: Optional[str]) -> int:
    tokens = {tok.strip() for tok in (model_sources or "").split("+")}
    return len(tokens - _NON_MODEL_TOKENS)


CONTAMINATION_RULES: dict[str, list[RowRule]] = {
    "baseball": [
        # Poisson removed from baseball 2026-06-01 — its presence dates the row.
        lambda mt, src, line: _has(src, "poisson"),
        # pitcher_v2 shipped 2026-07-19 (bullpen + park + offense + xERA;
        # model renamed pitcher → pitcher_v2). Any ML row without the v2
        # token was priced by a superseded model state: v1 rows carry
        # "pitcher", pre-guard rows carry neither — both are dated out of
        # the clean promotion sample. Subsumes the old "without pitcher"
        # rule (guard shipped 2026-06-06, −23% live ROI on pitcher-less ML).
        lambda mt, src, line: mt == "moneyline" and not _has(src, "pitcher_v2"),
        # Alt-run-line gate (PR #13): current code rejects baseball spreads
        # beyond ±1.5, so a resolved −2.5/−4.5 row is pre-gate.
        lambda mt, src, line: (
            mt == "spread" and line is not None and abs(line) > _BASEBALL_MAX_ABS_SPREAD
        ),
    ],
    "wnba": [
        # WNBA spread sim weight zeroed 2026-06-19 (SPREAD_SIM_WEIGHT_OVERRIDES).
        # Current code prices spread as pure sharp+spread_dist; a row that still
        # lists possession_sim in its blend was priced by the retired
        # sim_weight=0.35 regime.
        lambda mt, src, line: mt == "spread" and _has(src, "possession_sim"),
    ],
    "soccer": [
        # Sharp-only guard shipped 2026-07-19 (MIN_NONSHARP_MODELS in
        # ev_gap_agent.py). A soccer ML row with zero non-sharp model tokens
        # (the unseeded-MLS sharp-passthrough failure) predates the guard —
        # current code demotes those to shadow instead of logging them live.
        lambda mt, src, line: mt == "moneyline" and _nonsharp_count(src) == 0,
    ],
    "ncaaf": [
        # ncaaf_efficiency_v2 shipped 2026-09-03 (EPA + success rate, FPI-mixed
        # prior; model renamed). Any ML row WITHOUT the v2 token is outside the
        # v2 promotion sample: v1-token rows were priced by the superseded
        # EPA-only margin, and token-less (elo+sharp) rows were priced with v2
        # silent — including 2026-09-03→09-05 when the renamed agent read a
        # non-existent state file and withheld on every game. Mirrors the
        # baseball "ML without pitcher_v2" rule.
        lambda mt, src, line: mt == "moneyline" and not _has(src, "ncaaf_efficiency_v2"),
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
