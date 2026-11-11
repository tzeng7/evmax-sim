"""Broader triggers for the model-improve graph (read-only measurements).

The model-improve graph (``.claude/workflows/model-improve.js``) used to act only
on the value-audit rule: a sector's blend significantly WORSE than the entry
sharp line (z <= -1.64) or a consistent calibration bias. With ``sharp_weight``
0.85 plus the FLB re-blend, the model share at a 50/50 line is about
``(1 - sw)^2`` (roughly 2%). A blend that close to sharp can almost never be
significantly worse than sharp, so that trigger almost never fires: every run
from 2026-10-23 stopped at ``no_actionable_gap``.

This module computes four more triggers from measurements that already exist.
Each trigger names the metric that judges a fix for it. A coverage or data fix
is judged on its own metric (rows priced, CLV coverage), never on Brier.

| trigger              | source                                   | judge metric          |
|----------------------|------------------------------------------|-----------------------|
| ``passthrough``      | promotion board: LIVE lane, divergence   | ``rows_priced_frac``  |
|                      | < 0.5pp (``SHARP-PASSTHROUGH``)          |                       |
| ``model_coverage``   | ``ev_predictions.model_diagnostics``:    | ``rows_priced_frac``  |
|                      | configured model missing/gated on most   |                       |
|                      | recent rows                              |                       |
| ``close_capture``    | resolved Kalshi rows with NULL CLV       | ``clv_coverage_frac`` |
| ``calibration_bucket``| entry-price bucket where the blend is   | ``bucket_calib_err_pp``|
|                      | miscalibrated AND worse than sharp       |                       |

The functions here only measure. They never change state, and they never decide
what the graph does: the graph re-checks every threshold in JavaScript
(``TRIGGER_RULES``) and keeps new triggers shadow-only unless a run enables them.
``tests/test_model_improve_workflow.py`` pins the JS thresholds to the constants
below.

Each finding is a dict:

    {trigger, sector, market_type, venue, model, bucket,
     n, value, threshold, z, fires, judge_metric, priority, evidence, details}

``fires`` is this module's view of the threshold. ``priority`` orders findings
inside one trigger (bigger = more urgent).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Iterable, Optional

import structlog

logger = structlog.get_logger(__name__)

TRIGGERS = ("passthrough", "model_coverage", "close_capture", "calibration_bucket")

JUDGE_METRIC = {
    "passthrough": "rows_priced_frac",
    "model_coverage": "rows_priced_frac",
    "close_capture": "clv_coverage_frac",
    "calibration_bucket": "bucket_calib_err_pp",
}

# --- passthrough --------------------------------------------------------------
# Same 0.5pp line as promotion_board.SHARP_PASSTHROUGH_PP; only LIVE lanes fire,
# because a shadow passthrough lane stakes nothing.
PASSTHROUGH_MAX_DIVERGENCE_PP = 0.5
PASSTHROUGH_MIN_N = 10

# --- model_coverage -----------------------------------------------------------
# A configured model missing or gated on at least this share of a sector's
# recent moneyline-family rows. Unlike integrity's `model_missing` check (which
# needs a regime CHANGE), this fires on a persistent gap too: a model that never
# fires is the same lost signal whether it broke yesterday or never worked.
COVERAGE_MIN_N = 20
COVERAGE_MIN_UNPRICED_FRAC = 0.50
COVERAGE_DAYS = 14
# Models that are absent on most rows BY DESIGN. Firing on them would queue a
# "fix" for correct behavior. Tennis h2h needs >=3 prior meetings and ranking
# trend needs >=4 rank snapshots (CLAUDE.md: "structurally can't fire on most
# matches").
STRUCTURALLY_SPARSE: dict[str, frozenset[str]] = {
    "tennis": frozenset({"tennis_h2h", "tennis_ranking_trend"}),
}

# --- close_capture ------------------------------------------------------------
# Same thresholds as integrity._close_capture_issues.
CLOSE_CAPTURE_MIN_N = 10
CLOSE_CAPTURE_MAX_NULL_FRAC = 0.30
CLOSE_CAPTURE_DAYS = 14

# --- calibration_bucket -------------------------------------------------------
# Per entry-price bucket (price_buckets.BUCKET_ORDER). Every logged row is
# EV-selected (blend > ask), so blend - realized > 0 is partly selection bias
# that sharp shares. The trigger therefore needs the BLEND to be worse than the
# SHARP line in the same bucket, not only miscalibrated.
CALIB_BUCKET_MIN_N = 50
CALIB_BUCKET_MIN_ERR_PP = 3.0
CALIB_BUCKET_MIN_EXCESS_PP = 1.0
# 8 buckets x ~10 sectors are tested at once, so the z bar is above the
# single-test 1.96 (roughly a Bonferroni correction for ~40 tests at 2.5%).
CALIB_BUCKET_MIN_Z = 2.5
CALIB_BUCKET_DAYS = 60

_NON_MODEL_TOKENS = frozenset({
    "sharp", "sharp(capped)", "injury", "late_news", "rest", "playoff",
    "advance_derived", "spread_dist", "total_dist", "no_side", "anchored_entry",
    "sharp_ladder", "spread_pmf", "maker_pilot", "puck_line", "contested", "near_tip",
})


def _finding(trigger: str, sector: str, **kw) -> dict:
    out = {
        "trigger": trigger,
        "sector": sector,
        "market_type": kw.pop("market_type", None),
        "venue": kw.pop("venue", None),
        "model": kw.pop("model", None),
        "bucket": kw.pop("bucket", None),
        "n": int(kw.pop("n")),
        "value": kw.pop("value"),
        "threshold": kw.pop("threshold"),
        "z": kw.pop("z", None),
        "fires": bool(kw.pop("fires")),
        "judge_metric": JUDGE_METRIC[trigger],
        "priority": kw.pop("priority"),
        "evidence": kw.pop("evidence"),
        "details": kw.pop("details", {}),
    }
    if kw:
        raise TypeError(f"unexpected finding fields: {sorted(kw)}")
    return out


# ---------------------------------------------------------------------------
# passthrough — LIVE lanes whose blend is the sharp line
# ---------------------------------------------------------------------------

def passthrough_findings(board: Iterable[dict]) -> list[dict]:
    """``board``: rows from ``promotion_board.compute_promotion_board``.

    A LIVE moneyline lane with mean |blend - sharp| below 0.5pp stakes bankroll
    with no independent model signal. The only model-side remedy the graph may
    propose is a coverage fix (a configured model that should fire but does
    not). ``sharp_weight`` itself is owned by ``cleanup adjust`` and judged on
    CLV, and demotion is a manual promotion decision, so neither is in scope.
    """
    out: list[dict] = []
    for r in board:
        div = r.get("blend_divergence_pp")
        if r.get("market_type") != "moneyline" or div is None:
            continue
        if r.get("mode") != "live":
            continue
        n = int(r.get("n_logged") or 0)
        fires = div < PASSTHROUGH_MAX_DIVERGENCE_PP and n >= PASSTHROUGH_MIN_N
        if not fires:
            continue
        out.append(_finding(
            "passthrough", r.get("sector") or "?",
            market_type="moneyline", venue=r.get("venue"),
            n=n, value=round(float(div), 3), threshold=PASSTHROUGH_MAX_DIVERGENCE_PP,
            fires=True,
            # Lower divergence = more completely passthrough = more urgent.
            priority=round(PASSTHROUGH_MAX_DIVERGENCE_PP - float(div), 3),
            evidence=(
                f"{r.get('sector')} moneyline [{r.get('venue')}] LIVE: blend divergence "
                f"{div:.2f}pp < {PASSTHROUGH_MAX_DIVERGENCE_PP}pp over n={n} logged rows; "
                f"top blockers {r.get('top_blockers') or '—'}"
            ),
            details={
                "top_blockers": list(r.get("top_blockers") or []),
                "n_clean_resolved": r.get("n_clean_resolved"),
                "clv": r.get("clv"),
            },
        ))
    return out


# ---------------------------------------------------------------------------
# model_coverage — configured model missing / gated on most rows
# ---------------------------------------------------------------------------

def _parse_diag(raw) -> dict:
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def coverage_findings(
    rows: Iterable[dict],
    configured: Optional[dict[str, set[str]]] = None,
) -> list[dict]:
    """``rows``: dicts with ``sector`` and ``model_diagnostics`` (raw JSON).

    Rows without diagnostics are skipped: they cannot say which models were
    configured. For each (sector, model) the unpriced share is
    (missing + gated) / rows-with-diagnostics.

    ``configured`` maps sector → the models that carry blend weight there. The
    diagnostics ``gated`` map also lists agents that run for every sector but
    are not configured for this one (generic ``elo`` on a tennis row), and a
    model the sector override weights 0 is absent by design. When a sector has
    an entry, only its models can fire. ``None`` (tests) accepts every model.
    """
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        diag = _parse_diag(r.get("model_diagnostics"))
        if not diag:
            continue
        by_sector[(r.get("sector") or "?").lower()].append(diag)

    out: list[dict] = []
    for sector, diags in sorted(by_sector.items()):
        n = len(diags)
        if n < COVERAGE_MIN_N:
            continue
        missing: dict[str, int] = defaultdict(int)
        gated: dict[str, int] = defaultdict(int)
        notes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for d in diags:
            for m in d.get("missing") or []:
                missing[m] += 1
            for m, info in (d.get("gated") or {}).items():
                gated[m] += 1
                note = (info or {}).get("note") if isinstance(info, dict) else None
                if note:
                    notes[m][str(note)[:80]] += 1
        sparse = STRUCTURALLY_SPARSE.get(sector, frozenset())
        allowed = (configured or {}).get(sector)
        for model in sorted(set(missing) | set(gated)):
            if model in sparse or (allowed is not None and model not in allowed):
                continue
            unpriced = missing[model] + gated[model]
            frac = unpriced / n
            if frac < COVERAGE_MIN_UNPRICED_FRAC:
                continue
            top_notes = sorted(notes[model].items(), key=lambda kv: -kv[1])[:3]
            out.append(_finding(
                "model_coverage", sector,
                model=model, n=n, value=round(frac, 3),
                threshold=COVERAGE_MIN_UNPRICED_FRAC, fires=True,
                priority=round(frac, 3),
                evidence=(
                    f"{sector}: `{model}` unpriced on {unpriced}/{n} recent rows "
                    f"({frac * 100:.0f}%: missing {missing[model]}, gated {gated[model]})"
                ),
                details={
                    "missing": missing[model],
                    "gated": gated[model],
                    "top_gate_notes": [f"{k} ×{v}" for k, v in top_notes],
                },
            ))
    return out


# ---------------------------------------------------------------------------
# close_capture — resolved Kalshi rows with no Kalshi close
# ---------------------------------------------------------------------------

def close_capture_findings(rows: Iterable[dict]) -> list[dict]:
    """``rows``: resolved Kalshi rows with ``sector``, ``market_type``,
    ``market_id`` and ``kalshi_clv_pct`` (None = no close captured)."""
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sector[(r.get("sector") or "?").lower()].append(r)

    out: list[dict] = []
    for sector, srows in sorted(by_sector.items()):
        n = len(srows)
        nulls = [r for r in srows if r.get("kalshi_clv_pct") is None]
        frac = len(nulls) / n if n else 0.0
        if n < CLOSE_CAPTURE_MIN_N or frac <= CLOSE_CAPTURE_MAX_NULL_FRAC:
            continue
        by_mt: dict[str, int] = defaultdict(int)
        for r in nulls:
            by_mt[r.get("market_type") or "?"] += 1
        out.append(_finding(
            "close_capture", sector, venue="kalshi",
            n=n, value=round(frac, 3), threshold=CLOSE_CAPTURE_MAX_NULL_FRAC,
            fires=True, priority=round(frac, 3),
            evidence=(
                f"{sector}: {len(nulls)}/{n} resolved Kalshi rows have NULL CLV "
                f"({frac * 100:.0f}%; by market type {dict(by_mt)})"
            ),
            details={
                "null_n": len(nulls),
                "null_by_market_type": dict(by_mt),
                "example_market_ids": [r.get("market_id") for r in nulls[:5]],
            },
        ))
    return out


# ---------------------------------------------------------------------------
# calibration_bucket — blend miscalibrated in a price bucket, beyond sharp
# ---------------------------------------------------------------------------

def calibration_bucket_findings(rows: Iterable[dict]) -> list[dict]:
    """``rows``: resolved rows with ``sector``, ``market_type``, ``outcome``,
    ``blended_true_prob``, ``sharp_true_prob`` and the entry-price columns
    (``placed``, ``placed_price``, ``kalshi_yes_price``).

    Per (sector, market_type, bucket): ``err = mean(blend) - win rate`` in pp,
    ``z = err / se`` with ``se`` from the per-row residual ``blend - outcome``,
    and ``excess = |err| - |sharp err|``. Fires when n, |err|, |z| and excess
    all clear their bars.
    """
    from evmax.agents.cleanup.price_buckets import bucket_for_row

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("outcome") is None or r.get("blended_true_prob") is None:
            continue
        if r.get("sharp_true_prob") is None:
            continue
        b = bucket_for_row(r)
        if b == "unknown":
            continue
        groups[((r.get("sector") or "?").lower(), r.get("market_type") or "?", b)].append(r)

    out: list[dict] = []
    for (sector, mt, bucket), g in sorted(groups.items()):
        n = len(g)
        if n < CALIB_BUCKET_MIN_N:
            continue
        resid = [float(r["blended_true_prob"]) - int(r["outcome"]) for r in g]
        err = sum(resid) / n
        var = sum((x - err) ** 2 for x in resid) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
        z = err / se if se > 0 else 0.0
        sharp_err = sum(float(r["sharp_true_prob"]) - int(r["outcome"]) for r in g) / n
        err_pp, sharp_pp = err * 100, sharp_err * 100
        excess = abs(err_pp) - abs(sharp_pp)
        if (
            abs(err_pp) < CALIB_BUCKET_MIN_ERR_PP
            or abs(z) < CALIB_BUCKET_MIN_Z
            or excess < CALIB_BUCKET_MIN_EXCESS_PP
        ):
            continue
        out.append(_finding(
            "calibration_bucket", sector, market_type=mt, bucket=bucket,
            n=n, value=round(err_pp, 2), threshold=CALIB_BUCKET_MIN_ERR_PP,
            z=round(z, 2), fires=True, priority=round(abs(z), 2),
            evidence=(
                f"{sector} {mt} {bucket}c: blend − realized {err_pp:+.1f}pp (z {z:+.2f}, n={n}) "
                f"vs sharp − realized {sharp_pp:+.1f}pp → blend adds {excess:.1f}pp of miscalibration"
            ),
            details={"sharp_err_pp": round(sharp_pp, 2), "excess_pp": round(excess, 2)},
        ))
    return out


# ---------------------------------------------------------------------------
# DB-backed orchestration (read-only)
# ---------------------------------------------------------------------------

def _since(days: int, today: Optional[date] = None) -> str:
    return ((today or date.today()) - timedelta(days=days)).isoformat()


def _fetch_coverage_rows(days: int) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT sector, model_diagnostics FROM ev_predictions
               WHERE scan_date >= ? AND COALESCE(voided, 0) = 0
                 AND event_id NOT LIKE '%::prop::%'
                 AND model_diagnostics IS NOT NULL""",
            (_since(days),),
        )]


def configured_models() -> dict[str, set[str]]:
    """Sector → models that carry blend weight (categories.yaml ``models:``
    minus ``sharp`` minus any model the sector override weights 0)."""
    from evmax.agents.models.ensemble_agent import EnsembleModelAgent
    from evmax.categories import all_categories

    overrides = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES

    out: dict[str, set[str]] = {}
    for cat in all_categories():
        models = set(getattr(cat, "models", None) or []) - {"sharp"}
        zero = {m for m, w in (overrides.get(cat.key) or {}).items() if not w}
        out[cat.key] = models - zero
    return out


def _fetch_close_rows(days: int) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT p.sector, p.market_type, p.market_id, p.kalshi_clv_pct
               FROM ev_predictions p
               JOIN ev_outcomes o ON o.market_id = p.market_id
               WHERE o.outcome IS NOT NULL AND COALESCE(p.venue, 'kalshi') = 'kalshi'
                 AND COALESCE(p.voided, 0) = 0
                 AND (p.minutes_to_tipoff IS NULL OR p.minutes_to_tipoff > 0)
                 AND p.event_id NOT LIKE '%::prop::%'
                 AND p.event_date >= ?""",
            (_since(days),),
        )]


def _fetch_calibration_rows(days: int) -> list[dict]:
    from evmax.agents.cleanup.contamination import is_contaminated
    from evmax.agents.cleanup.db import get_connection

    with get_connection() as conn:
        rows = [dict(r) for r in conn.execute(
            """SELECT p.sector, p.market_type, p.model_sources, p.line,
                      p.placed, p.placed_price, p.kalshi_yes_price,
                      p.blended_true_prob, p.sharp_true_prob, o.outcome
               FROM ev_predictions p
               JOIN ev_outcomes o ON o.market_id = p.market_id
               WHERE o.outcome IS NOT NULL AND COALESCE(p.voided, 0) = 0
                 AND p.event_id NOT LIKE '%::prop::%'
                 AND p.event_date >= ?""",
            (_since(days),),
        )]
    return [
        r for r in rows
        if not is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"])
    ]


def compute_improve_triggers(
    triggers: Optional[Iterable[str]] = None,
    board_days: int = 30,
) -> dict:
    """Run every requested trigger against the configured DBs (read-only).

    Returns ``{"findings": [...], "ran": [...], "failed": {trigger: error}}``.
    One failing trigger never masks the others.
    """
    wanted = list(triggers) if triggers else list(TRIGGERS)
    unknown = [t for t in wanted if t not in TRIGGERS]
    if unknown:
        raise ValueError(f"unknown trigger(s) {unknown}; choose from {TRIGGERS}")

    def _passthrough():
        from evmax.agents.cleanup.promotion_board import compute_promotion_board

        return passthrough_findings(compute_promotion_board(days=board_days))

    runners = {
        "passthrough": _passthrough,
        "model_coverage": lambda: coverage_findings(
            _fetch_coverage_rows(COVERAGE_DAYS), configured_models()
        ),
        "close_capture": lambda: close_capture_findings(_fetch_close_rows(CLOSE_CAPTURE_DAYS)),
        "calibration_bucket": lambda: calibration_bucket_findings(
            _fetch_calibration_rows(CALIB_BUCKET_DAYS)
        ),
    }
    findings: list[dict] = []
    ran: list[str] = []
    failed: dict[str, str] = {}
    for t in wanted:
        try:
            findings.extend(runners[t]())
            ran.append(t)
        except Exception as e:  # noqa: BLE001 — one broken trigger must not mask the rest
            logger.error("improve_trigger_failed", trigger=t, error=str(e))
            failed[t] = str(e)
    findings.sort(key=lambda f: (TRIGGERS.index(f["trigger"]), -f["priority"]))
    return {"findings": findings, "ran": ran, "failed": failed}
