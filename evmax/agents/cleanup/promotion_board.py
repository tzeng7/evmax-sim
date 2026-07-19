"""Promotion scoreboard — one health row per (sector, market_type, venue).

The single surface that answers "which sectors can I actually rely on, and
what is blocking the rest?" (2026-07-19, WS4 of the diversification plan).
Combines, per group over a trailing window of GAME dates:

  - sample:      n_logged / n_resolved / n_clean (contamination-filtered),
                 live/shadow mode split
  - accuracy:    Brier of the blend vs the sharp anchor on clean resolved
                 rows, paired Δ/1000 with z (reuses value_audit internals)
  - CLV:         evmax.cli.commands.shadow.clv_stats — the promotion lens
                 (contamination-filtered, venue-aware, staleness-filtered)
  - divergence:  mean |blended − sharp| in pp over ALL logged rows — the
                 first-class "is this blend sharp-passthrough?" indicator.
                 Measured 2026-07-18: wnba 4.82pp (real signal) vs tennis
                 0.17 / baseball 0.10 / soccer 0.00 (passthrough). A blend
                 that never disagrees with Pinnacle cannot out-predict it.
  - gates:       the shadow-promotion checklist (clean-n ≥ 30; CLV n ≥ 30,
                 mean ≥ 0, %pos ≥ 55) — constants imported from shadow.py,
                 the single source of truth
  - blockers:    aggregated why-not diagnostics (model_diagnostics JSON) —
                 which model is most often missing/gated for the sector

Scope: game markets (ev_predictions). Player props live in
prop_observations and have their own evaluator; they are not board rows.

Verdict ladder (first match wins):
  SHARP-PASSTHROUGH  moneyline blend divergence < 0.5pp — fix seeding/models
                     before reading anything else
  COLLECTING n/30    not enough clean resolved rows to judge
  PROMOTE-READY      shadow-mode group whose gates all clear
  FAILING-CLV        enough sample, CLV gate fails
  LIVE-HEALTHY       live-mode group with non-negative mean CLV
  LIVE-DEGRADING     live-mode group with n ≥ 30 and negative mean CLV
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# A moneyline blend whose mean |blended − sharp| is below this is treated as
# sharp passthrough: any apparent EV is Kalshi-vs-Pinnacle divergence, not
# model edge.
SHARP_PASSTHROUGH_PP = 0.5


def _effective_mode(sector: str, market_type: str) -> Optional[str]:
    """Effective mode for a (sector, market_type), or None if unknown.

    Delegates to evmax.modes.get_mode, which already layers runtime/env/YAML
    resolution with the per-market-type shadow/disabled refinements.
    """
    try:
        from evmax.modes import get_mode

        return get_mode(sector, market_type)
    except Exception:
        return None


def _verdict(
    market_type: str,
    divergence_pp: Optional[float],
    n_clean: int,
    clv: dict,
    mode: Optional[str],
    min_clean: int,
) -> str:
    if (
        market_type == "moneyline"
        and divergence_pp is not None
        and divergence_pp < SHARP_PASSTHROUGH_PP
    ):
        return "SHARP-PASSTHROUGH"
    if n_clean < min_clean:
        return f"COLLECTING {n_clean}/{min_clean}"
    if mode == "shadow" and clv.get("clears"):
        return "PROMOTE-READY"
    if not clv.get("clears"):
        if mode == "live":
            if clv.get("n", 0) >= min_clean and clv.get("mean_clv_pp", 0.0) < 0:
                return "LIVE-DEGRADING"
            return "LIVE-HEALTHY"
        return "FAILING-CLV"
    if mode == "live":
        return "LIVE-HEALTHY"
    return "PROMOTE-READY"


def _top_blockers(diag_rows: list[str], top_n: int = 2) -> list[str]:
    """Most frequent missing/gated models across a group's diagnostics."""
    counts: Counter[str] = Counter()
    for raw in diag_rows:
        if not raw:
            continue
        try:
            diag = json.loads(raw)
        except (ValueError, TypeError):
            continue
        for name in diag.get("missing") or []:
            counts[f"{name}:missing"] += 1
        for name in (diag.get("gated") or {}):
            counts[f"{name}:gated"] += 1
    return [f"{k}×{v}" for k, v in counts.most_common(top_n)]


def compute_promotion_board(
    days: int = 30,
    staleness_h: Optional[float] = 3.0,
    sector: Optional[str] = None,
) -> list[dict]:
    """Build the board: one dict per (sector, market_type, venue).

    ``days`` windows on event_date. ``staleness_h`` feeds clv_stats'
    stale-close filter (None disables it). ``sector`` restricts to one
    sector. Rows are sorted worst-first within sector (passthrough and
    degrading groups float up).
    """
    from evmax.agents.cleanup.contamination import is_contaminated
    from evmax.agents.cleanup.db import get_connection
    from evmax.agents.cleanup.value_audit import _brier, _paired_diff_stats
    from evmax.cli.commands.shadow import (
        MIN_CLEAN_RESOLVED,
        clv_stats,
    )

    since = (date.today() - timedelta(days=days)).isoformat()
    where = [
        "p.voided = 0",
        "(p.event_date >= ? OR p.event_date IS NULL)",
        "p.event_id NOT LIKE '%::prop::%'",
    ]
    params: list = [since]
    if sector:
        where.append("p.sector = ?")
        params.append(sector.lower())

    sql = f"""
        SELECT p.sector, p.market_type, p.venue, p.mode, p.line,
               p.model_sources, p.model_diagnostics,
               p.blended_true_prob, p.sharp_true_prob,
               o.outcome
        FROM ev_predictions p
        LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE {' AND '.join(where)}
    """
    with get_connection() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r["sector"] or "?",
            r["market_type"] or "?",
            r["venue"] or "kalshi",
        )
        groups[key].append(r)

    board: list[dict] = []
    for (sec, mt, ven), grp in sorted(groups.items()):
        resolved = [r for r in grp if r["outcome"] is not None]
        clean = [
            r for r in resolved
            if not is_contaminated(sec, mt, r["model_sources"], r["line"])
        ]
        mode_split = Counter(r["mode"] or "live" for r in grp)

        div_all = [
            abs(r["blended_true_prob"] - r["sharp_true_prob"]) * 100
            for r in grp
            if r["blended_true_prob"] is not None and r["sharp_true_prob"] is not None
        ]
        div_resolved = [
            abs(r["blended_true_prob"] - r["sharp_true_prob"]) * 100
            for r in clean
            if r["blended_true_prob"] is not None and r["sharp_true_prob"] is not None
        ]
        divergence_pp = round(sum(div_all) / len(div_all), 2) if div_all else None
        divergence_resolved_pp = (
            round(sum(div_resolved) / len(div_resolved), 2) if div_resolved else None
        )

        brier_blend = _brier(clean, "blended_true_prob")
        brier_sharp = _brier(clean, "sharp_true_prob")
        paired = _paired_diff_stats(clean, "sharp_true_prob", "blended_true_prob")

        clv = clv_stats(
            sec,
            market_type=mt,
            mode=None,
            venue=ven if ven in ("kalshi", "polymarket_us") else None,
            max_staleness_h=staleness_h if ven == "kalshi" else None,
        )

        mode = _effective_mode(sec, mt)
        gates = {
            "clean_n": {
                "value": len(clean),
                "required": MIN_CLEAN_RESOLVED,
                "ok": len(clean) >= MIN_CLEAN_RESOLVED,
            },
            "clv_n": {
                "value": clv["n"],
                "required": MIN_CLEAN_RESOLVED,
                "ok": clv["n"] >= MIN_CLEAN_RESOLVED,
            },
            "clv_mean": {
                "value": clv["mean_clv_pp"],
                "required": 0.0,
                "ok": clv["mean_clv_pp"] >= 0.0,
            },
            "clv_frac_pos": {
                "value": clv["frac_positive"],
                "required": 0.55,
                "ok": clv["frac_positive"] >= 0.55,
            },
        }

        passthrough = (
            mt == "moneyline"
            and divergence_pp is not None
            and divergence_pp < SHARP_PASSTHROUGH_PP
        )

        board.append({
            "sector": sec,
            "market_type": mt,
            "venue": ven,
            "mode": mode,
            "mode_split": dict(mode_split),
            "n_logged": len(grp),
            "n_resolved": len(resolved),
            "n_clean_resolved": len(clean),
            "brier_blend": round(brier_blend, 4) if brier_blend is not None else None,
            "brier_sharp": round(brier_sharp, 4) if brier_sharp is not None else None,
            "brier_delta_per_1000": (
                round(paired["mean"] * 1000, 2) if paired else None
            ),
            "brier_delta_z": round(paired["z"], 2) if paired else None,
            "blend_divergence_pp": divergence_pp,
            "blend_divergence_resolved_pp": divergence_resolved_pp,
            "sharp_passthrough": passthrough,
            "clv": {
                "n": clv["n"],
                "mean_clv_pp": clv["mean_clv_pp"],
                "frac_positive": clv["frac_positive"],
                "clears": clv["clears"],
                "excluded_stale": clv.get("excluded_stale", 0),
            },
            "gates": gates,
            "verdict": _verdict(
                mt, divergence_pp, len(clean), clv, mode, MIN_CLEAN_RESOLVED
            ),
            "top_blockers": _top_blockers(
                [r.get("model_diagnostics") for r in grp]
            ),
        })

    _SEVERITY = {
        "SHARP-PASSTHROUGH": 0,
        "LIVE-DEGRADING": 1,
        "FAILING-CLV": 2,
        "PROMOTE-READY": 3,
        "LIVE-HEALTHY": 5,
    }

    def _rank(row: dict) -> tuple:
        v = row["verdict"]
        sev = _SEVERITY.get(v, 4 if v.startswith("COLLECTING") else 6)
        return (row["sector"], sev, row["market_type"], row["venue"])

    board.sort(key=_rank)
    return board
