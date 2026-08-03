"""Per-sector model-blend VALUE audit: Brier + CLV with significance.

This is the read-only "Brier/CLV observer" — it measures, per sector, whether our
*model blend* is actually adding value over the sharp benchmark, and whether a gap
is real signal or just noise. It does NOT change anything; `evmax cleanup value-audit`
prints it and the weekly drift/value loop consumes it.

Three benchmarks, in increasing order of how hard they are to beat:
  - brier_sharp  — devigged Pinnacle at ENTRY time (`sharp_true_prob`). Beating this
                   means the blend improved on the entry line we already had.
  - brier_close  — devigged Pinnacle at CLOSE (`pinnacle_close_prob`). Beating THIS is
                   genuine forecasting skill — the closing line is the sharpest public
                   estimate. Most sectors will NOT beat it (that's expected, not a bug).
  - CLV          — realized `kalshi_clv_pct` (entry→Kalshi-close). Positive CLV with no
                   Brier-vs-close edge = stale-line capture, real but timing-bound.

Why significance matters here: the project has repeatedly found that close-Brier
differences between the blend and the sharp line are *below the noise floor*
(see the tennis weight notes in CLAUDE.md). So every Brier comparison carries a
paired z-score / 95% CI on the per-row squared-error difference. A "gap" that the
CI straddles zero is NOT actionable — chasing it would be overfitting.

Actionability rule (what counts as a fixable MODEL gap, not a gating decision):
  - `model_subtracts` — the blend's Brier is significantly WORSE than the ENTRY sharp
    line (z <= -1.64). The blend is actively mis-weighted/mis-calibrated; this is
    fixable in the models (reweight / recalibrate), never by gating plays.
  - `calibration_bias` — a systematic, monotone over/under-confidence across buckets
    (e.g. chalk overprediction). Fixable via the sector's isotonic calibration.
  - otherwise `neutral` (within noise) or `adds_value`.
CLV is reported for context but a negative-CLV / fine-Brier sector is a timing/selection
problem, not a model-blend problem, so it is NOT tagged model-actionable here.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import structlog

from evmax.agents.cleanup.db import get_connection

logger = structlog.get_logger(__name__)

# Minimum resolved sample before we trust a sector's Brier comparison at all.
MIN_N = 30
# z thresholds (one-sided ~95% / ~97.5%). A gap must clear this to be "real".
Z_ACTIONABLE = 1.64
# Decile calibration: flag systematic bias only if the mean signed error across the
# populated buckets exceeds this and is consistently one-signed.
CALIB_BIAS_PP = 4.0


def _brier(rows: list, prob_key: str) -> Optional[float]:
    vals = [r for r in rows if r[prob_key] is not None]
    if not vals:
        return None
    return sum((r[prob_key] - r["outcome"]) ** 2 for r in vals) / len(vals)


def _paired_diff_stats(rows: list, a_key: str, b_key: str) -> Optional[dict]:
    """Paired comparison of two probability columns' squared errors.

    d_i = (a-out)^2 - (b-out)^2, so a POSITIVE mean means column `b` (the model)
    is better than column `a` (the benchmark). Returns mean, standard error, z,
    and a 95% CI. Only rows where BOTH columns are present are used.
    """
    paired = [r for r in rows if r[a_key] is not None and r[b_key] is not None]
    n = len(paired)
    if n < 2:
        return None
    diffs = [
        (r[a_key] - r["outcome"]) ** 2 - (r[b_key] - r["outcome"]) ** 2
        for r in paired
    ]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    z = (mean / se) if se > 0 else 0.0
    return {
        "n": n,
        "mean": mean,          # >0 → model (b) beats benchmark (a)
        "se": se,
        "z": z,
        "ci_lo": mean - 1.96 * se,
        "ci_hi": mean + 1.96 * se,
    }


def _clv_stats(rows: list) -> Optional[dict]:
    vals = [r["kalshi_clv_pct"] for r in rows if r["kalshi_clv_pct"] is not None]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    pos = sum(1 for v in vals if v > 0)
    if n >= 2:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
    else:
        se = 0.0
    return {
        "n": n,
        "mean_pp": mean,
        "frac_positive": pos / n,
        "se": se,
        "z": (mean / se) if se > 0 else 0.0,
    }


def _calibration(rows: list, n_buckets: int = 10) -> dict:
    """Signed calibration error of the model probs, bucketed by predicted value.

    Returns the per-bucket (mean_pred, mean_out, n) plus two summary metrics:

      - `signed_bias_pp` — mean over populated buckets of (mean_pred - mean_out)
        * 100. Over- and under-confident buckets CANCEL, so a model that is high
        on favorites and low on dogs can read ~0. A large positive value means
        systematic OVER-confidence (e.g. chalk bias); negative under-confidence.
      - `ece_pp` — population-weighted classwise Expected Calibration Error:
        Σ_b (n_b/N)·|mean_pred_b - mean_out_b| · 100. Absolute per bucket, so it
        NEVER cancels. This is the calibration metric the ML-for-sports-betting
        literature (arXiv:2303.06021) selects models on — it exposes miscalibration
        that `signed_bias_pp` hides. Lower is better; 0 = perfectly calibrated.
    """
    have = [r for r in rows if r["blended_true_prob"] is not None]
    buckets: list[dict] = []
    signed = []
    for i in range(n_buckets):
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        hi_cmp = hi + (0.001 if i == n_buckets - 1 else 0.0)
        b = [r for r in have if lo <= r["blended_true_prob"] < hi_cmp]
        if not b:
            continue
        mp = sum(r["blended_true_prob"] for r in b) / len(b)
        mo = sum(r["outcome"] for r in b) / len(b)
        buckets.append({"lo": lo, "hi": hi, "n": len(b), "mean_pred": mp, "mean_out": mo})
        signed.append((mp - mo) * 100.0)
    signed_bias_pp = (sum(signed) / len(signed)) if signed else 0.0
    # Classwise-ECE: population-weighted mean ABSOLUTE bucket error (never cancels).
    total_n = sum(b["n"] for b in buckets)
    ece_pp = (
        sum(b["n"] * abs(b["mean_pred"] - b["mean_out"]) for b in buckets) / total_n * 100.0
        if total_n else 0.0
    )
    # "monotone" if every populated bucket errs the same direction (consistent bias)
    consistent = bool(signed) and (all(s >= 0 for s in signed) or all(s <= 0 for s in signed))
    return {
        "buckets": buckets,
        "signed_bias_pp": signed_bias_pp,
        "ece_pp": ece_pp,
        "consistent_direction": consistent,
    }


def _verdict(edge_sharp: Optional[dict], calib: dict, n: int) -> dict:
    """Classify the sector. See module docstring for the rule."""
    if n < MIN_N:
        return {"tag": "insufficient", "actionable": False,
                "reason": f"only {n} resolved (need {MIN_N}+)"}

    # Model significantly WORSE than the entry sharp line → mis-weighted blend.
    if edge_sharp and edge_sharp["z"] <= -Z_ACTIONABLE:
        return {
            "tag": "model_subtracts",
            "actionable": True,
            "reason": (f"blend Brier worse than entry-sharp by "
                       f"{-edge_sharp['mean']*1000:.2f}/1000 (z={edge_sharp['z']:.2f})"),
        }

    # Systematic, consistent calibration bias → recalibrate.
    if calib["consistent_direction"] and abs(calib["signed_bias_pp"]) >= CALIB_BIAS_PP:
        d = "over" if calib["signed_bias_pp"] > 0 else "under"
        return {
            "tag": "calibration_bias",
            "actionable": True,
            "reason": (f"systematic {d}-confidence: mean signed error "
                       f"{calib['signed_bias_pp']:+.2f}pp, consistent across buckets"),
        }

    if edge_sharp and edge_sharp["z"] >= Z_ACTIONABLE:
        return {"tag": "adds_value", "actionable": False,
                "reason": (f"blend beats entry-sharp by {edge_sharp['mean']*1000:.2f}/1000 "
                           f"(z={edge_sharp['z']:.2f})")}

    return {"tag": "neutral", "actionable": False,
            "reason": "blend within noise of the sharp line — not model-actionable"}


def _fetch_rows(weeks: int) -> list:
    since = (date.today() - timedelta(weeks=weeks)).isoformat()
    conn = get_connection()
    rows = conn.execute(
        """SELECT p.sector, p.market_type,
                  o.outcome, o.sharp_true_prob, o.blended_true_prob,
                  o.pinnacle_close_prob, p.kalshi_clv_pct
           FROM ev_outcomes o
           JOIN ev_predictions p ON o.market_id = p.market_id
           INNER JOIN (
               SELECT market_id, MAX(scan_date) AS latest_scan
               FROM ev_predictions WHERE voided = 0 GROUP BY market_id
           ) latest ON p.market_id = latest.market_id
                   AND p.scan_date = latest.latest_scan
           WHERE o.outcome IS NOT NULL
             AND p.event_date >= ?
             AND p.mode = 'live'""",
        (since,),
    ).fetchall()
    conn.close()
    return rows


def compute_value_audit(weeks: int = 12) -> list[dict]:
    """Per-sector model-blend value audit over the last `weeks` weeks of resolved data.

    Returns one dict per sector (sorted by resolved sample count desc):
      {
        sector, n,
        brier_model, brier_sharp, brier_close,        # close may be None if no coverage
        edge_vs_sharp: {mean, se, z, ci_lo, ci_hi, n} | None,
        edge_vs_close: {...} | None,
        clv: {n, mean_pp, frac_positive, z} | None,
        calibration: {signed_bias_pp, ece_pp, consistent_direction, buckets},
        by_market_type: {moneyline: {...}, spread: {...}, total: {...}},
        verdict: {tag, actionable, reason},
      }
    """
    rows = _fetch_rows(weeks)
    by_sector: dict[str, list] = {}
    for r in rows:
        by_sector.setdefault((r["sector"] or "").lower(), []).append(r)

    out: list[dict] = []
    for sector, srows in by_sector.items():
        n = len(srows)
        edge_sharp = _paired_diff_stats(srows, "sharp_true_prob", "blended_true_prob")
        edge_close = _paired_diff_stats(srows, "pinnacle_close_prob", "blended_true_prob")
        calib = _calibration(srows)

        # per-market-type Brier model-vs-sharp (no significance, just direction)
        by_mt: dict[str, dict] = {}
        mts: dict[str, list] = {}
        for r in srows:
            mts.setdefault((r["market_type"] or "?").lower(), []).append(r)
        for mt, mrows in mts.items():
            by_mt[mt] = {
                "n": len(mrows),
                "brier_model": _brier(mrows, "blended_true_prob"),
                "brier_sharp": _brier(mrows, "sharp_true_prob"),
            }

        out.append({
            "sector": sector or "(unknown)",
            "n": n,
            "brier_model": _brier(srows, "blended_true_prob"),
            "brier_sharp": _brier(srows, "sharp_true_prob"),
            "brier_close": _brier(srows, "pinnacle_close_prob"),
            "edge_vs_sharp": edge_sharp,
            "edge_vs_close": edge_close,
            "clv": _clv_stats(srows),
            "calibration": calib,
            "by_market_type": by_mt,
            "verdict": _verdict(edge_sharp, calib, n),
        })

    out.sort(key=lambda x: x["n"], reverse=True)
    return out


def format_calibration_alert(flagged: list[dict]) -> Optional[str]:
    """Build the tripwire alert text for sectors flagged `calibration_bias`.

    Returns None when nothing is flagged (the quiet, no-alert case). Each line
    names the sector, its bias reason, the ECE, the sample size, and the exact
    fix command. Pure/formatting only — the selection was already made by
    `_verdict` (which requires n>=MIN_N, consistent direction, and >=CALIB_BIAS_PP).
    """
    if not flagged:
        return None
    lines = ["⚠️ evmax calibration tripwire — sector(s) drifting into a real bias:"]
    for a in flagged:
        c = a.get("calibration", {})
        lines.append(
            f"• {a['sector']}: {a['verdict']['reason']} "
            f"(ECE {c.get('ece_pp', 0.0):.1f}pp, n={a['n']}) "
            f"→ evmax cleanup recalibrate --sector {a['sector']}"
        )
    return "\n".join(lines)
