"""Automatic per-sector devig-method selection.

The weekly integrity sweep runs :func:`recommend_devig_methods`, which re-devigs
every recently-resolved archived Pinnacle line three ways (power / shin /
multiplicative) and scores each against the actual outcome. When a non-power
method beats power by a SIGNIFICANT and material margin on an adequate sample,
the sweep surfaces a one-line recommendation with the exact one-tap command:

    evmax cleanup devig promote <sector>

which writes the choice to ``devig_method_state.json`` (read by
``resolve_devig_method``) — no code edit, no commit. So the operator never
investigates on their own bandwidth; the system flags a winner and applying it
is a single confirmation.

Why Brier-vs-outcome, not CLV: devigging EXTRACTS the sharp book's own implied
probability from its vigged odds. Its quality is how well that extracted
probability predicts the outcome — a calibration question whose ground truth is
the result. That is distinct from MODEL promotion, which asks whether a forecast
beats the market and is judged on CLV (the tennis lesson). The tennis lesson
still applies as the SIGNIFICANCE guard here: a marginal Brier edge below the
noise floor never triggers a flip — the gate requires a paired z as well as a
material delta and a large n. Power stays sticky; ties go to power.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

from evmax.ev.devig import (
    DEVIG_METHOD_STATE_PATH,
    DEVIG_METHODS,
    devig,
    invalidate_selected_methods_cache,
    resolve_devig_method,
)

# Gate constants — a challenger must clear ALL THREE to be recommended.
DEVIG_SELECT_MIN_N = 200          # devigged lines for the sector in the window
DEVIG_SELECT_MIN_BRIER_DELTA = 0.002   # >= 2/1000 mean paired Brier improvement
DEVIG_SELECT_MIN_Z = 1.64         # one-sided 95% on the paired per-line diff


@dataclass
class DevigRecommendation:
    sector: str
    current_method: str      # what resolve_devig_method uses today
    best_method: str         # best-Brier method over the window
    power_brier: float
    best_brier: float
    delta: float             # power_brier - best_brier (>0 = challenger better)
    z: float                 # paired z of the per-line squared-error diff
    n: int
    clears_gate: bool        # best is a non-power method AND clears all gates

    @property
    def is_actionable(self) -> bool:
        """A real, un-applied recommendation the sweep should surface."""
        return self.clears_gate and self.best_method != self.current_method


def evaluate_sector(
    sector: str,
    pairs_by_method: dict[str, list[tuple[float, int]]],
    current_method: Optional[str] = None,
) -> Optional[DevigRecommendation]:
    """Gate one sector from per-method (prob, outcome) pairs (side-A aligned).

    ``pairs_by_method`` must carry the SAME lines under every method, in the
    same order — the paired significance test differences them per line. Returns
    None when power isn't present or samples are empty; otherwise a
    DevigRecommendation whose ``clears_gate`` reflects the three-part gate.
    """
    power = pairs_by_method.get("power")
    if not power:
        return None
    n = len(power)
    if current_method is None:
        current_method = resolve_devig_method(sector)

    def _brier(pairs: list[tuple[float, int]]) -> float:
        return sum((p - o) ** 2 for p, o in pairs) / len(pairs) if pairs else float("nan")

    power_brier = _brier(power)

    # Best non-power challenger by Brier.
    best_method, best_brier, best_pairs = "power", power_brier, power
    for m in DEVIG_METHODS:
        if m == "power":
            continue
        pairs = pairs_by_method.get(m)
        if not pairs or len(pairs) != n:
            continue
        b = _brier(pairs)
        if b < best_brier:
            best_method, best_brier, best_pairs = m, b, pairs

    delta = power_brier - best_brier

    # Paired significance of the per-line squared-error improvement.
    z = 0.0
    if best_method != "power" and n >= 2:
        diffs = [
            (pp - po) ** 2 - (cp - co) ** 2
            for (pp, po), (cp, co) in zip(power, best_pairs)
        ]
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
        z = mean / se if se > 0 else 0.0

    clears = (
        best_method != "power"
        and n >= DEVIG_SELECT_MIN_N
        and delta >= DEVIG_SELECT_MIN_BRIER_DELTA
        and z >= DEVIG_SELECT_MIN_Z
    )
    return DevigRecommendation(
        sector=sector,
        current_method=current_method,
        best_method=best_method,
        power_brier=power_brier,
        best_brier=best_brier,
        delta=delta,
        z=z,
        n=n,
        clears_gate=clears,
    )


def save_selected_method(sector: str, method: str) -> None:
    """Persist the one-tap selection and refresh the runtime cache."""
    m = method.lower()
    if m not in DEVIG_METHODS:
        raise ValueError(f"unknown devig method: {method!r} (choices: {DEVIG_METHODS})")
    DEVIG_METHOD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(DEVIG_METHOD_STATE_PATH.read_text())
        methods = state.get("methods") if isinstance(state, dict) else None
        methods = dict(methods) if isinstance(methods, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        methods = {}
    if m == "power":
        methods.pop(sector.lower(), None)  # power is the default — store nothing
    else:
        methods[sector.lower()] = m
    DEVIG_METHOD_STATE_PATH.write_text(json.dumps({"methods": methods}, indent=2, sort_keys=True))
    invalidate_selected_methods_cache()


def clear_selected_method(sector: str) -> None:
    """Revert a sector to power (removes its persisted entry)."""
    save_selected_method(sector, "power")


def recommend_devig_methods(days: int = 120) -> list[DevigRecommendation]:
    """Prod-only: gather archived Pinnacle lines + outcomes and gate each sector.

    Re-devigs each resolved line three ways and returns a recommendation per
    sector. Degrades to ``[]`` off the prod box (no archive.db / no rows). The
    join mirrors ``scripts/backtest_devig_ab.py``.
    """
    from collections import defaultdict

    try:
        from evmax.agents.cleanup.db import get_connection
    except Exception:  # noqa: BLE001
        return []

    try:
        with get_connection() as conn:
            outcomes = {
                r["event_id"]: (r["yes_team"], int(r["outcome"]), (r["sector"] or "").lower())
                for r in conn.execute(
                    """
                    SELECT event_id, yes_team, outcome, sector
                    FROM ev_outcomes
                    WHERE outcome IS NOT NULL AND event_id IS NOT NULL
                      AND resolved_at >= datetime('now', ?)
                    """,
                    (f"-{days} days",),
                )
            }
    except Exception:  # noqa: BLE001
        return []
    if not outcomes:
        return []

    import sqlite3
    from pathlib import Path as _P

    archive_path = _P(DEVIG_METHOD_STATE_PATH).resolve().parents[2] / "data" / "archive.db"
    if not archive_path.exists():
        return []
    arc = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
    arc.row_factory = sqlite3.Row

    pairs: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    try:
        rows = arc.execute(
            """
            SELECT event_id, sector, outcome_a_label, outcome_b_label,
                   outcome_a_decimal, outcome_b_decimal, outcome_draw_decimal
            FROM archived_sharp_odds
            WHERE outcome_a_decimal IS NOT NULL AND outcome_b_decimal IS NOT NULL
            """
        )
        for r in rows:
            ev = r["event_id"]
            if ev not in outcomes:
                continue
            yes_team, won, _sec = outcomes[ev]
            sec = (r["sector"] or "").lower()
            a_label = (r["outcome_a_label"] or "").lower()
            yes = (yes_team or "").lower()
            if yes and yes == a_label:
                side_a_won = won
            elif yes and yes == (r["outcome_b_label"] or "").lower():
                side_a_won = 1 - won
            else:
                continue
            decimals = [r["outcome_a_decimal"], r["outcome_b_decimal"]]
            if r["outcome_draw_decimal"]:
                decimals.append(r["outcome_draw_decimal"])
            for method in DEVIG_METHODS:
                try:
                    res = devig(decimals, method=method)
                except Exception:  # noqa: BLE001
                    break
                pairs[sec][method].append((res.true_probs[0], side_a_won))
    finally:
        arc.close()

    recs: list[DevigRecommendation] = []
    for sec, by_method in pairs.items():
        rec = evaluate_sector(sec, by_method)
        if rec is not None:
            recs.append(rec)
    return recs
