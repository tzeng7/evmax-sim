"""Brier score computation and sharp_weight auto-adjustment.

Brier score = mean((predicted_prob - outcome)^2)
Lower is better (0 = perfect calibration).

Adjustment rules:
  - Require >= 30 resolved predictions over the last 4 weeks
  - If model Brier < sharp Brier by > 5%: lower sharp_weight by 0.05 (min 0.40)
  - If model Brier > sharp Brier by > 5%: raise sharp_weight by 0.05 (max 0.95)
  - Won't adjust more than once per 7 days (use --force to override)
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import structlog

from evmax.agents.cleanup.db import get_connection

logger = structlog.get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[3] / "data" / "model_config.json"

# Per-sector sharp_weight baseline. Tennis and baseball were already hardcoded
# in the coordinator before this was factored out — see evmax/agents/coordinator.py.
# Rationale: our statistical stack is thin for those sports, so we trust Pinnacle
# more heavily to avoid phantom edges from vig removal alone.
#
# Tennis 1.00 (2026-05-01): production resolved data (n=112) showed the tennis
# stat ensemble (surface_elo + serve_return + form + advanced + h2h + ranking)
# net-subtracts ~3.77pp ROI on top of Pinnacle — +1.89% sharp-only counterfactual
# vs −1.88% production. The 28 bets that triggered ONLY because tennis models
# inflated EV averaged −13.2% ROI. Tennis flipped to shadow mode in
# data/categories.yaml for 30d validation; promote back to live once shadow
# metrics confirm positive ROI on 50+ new resolved bets.
# NOTE: only `_DEFAULT_SHARP_WEIGHT_BY_SECTOR` is the seed default; the live
# value lives in data/model_config.json under `sharp_weight_by_sector`.
_DEFAULT_SHARP_WEIGHT_BY_SECTOR: dict[str, float] = {
    "tennis":   1.00,
    "baseball": 0.88,
    # All other sectors fall back to the top-level `sharp_weight`.
}

_DEFAULT_CONFIG: dict = {
    "sharp_weight": 0.85,                                    # default / fallback
    "sharp_weight_by_sector": dict(_DEFAULT_SHARP_WEIGHT_BY_SECTOR),
    "last_brier_model": None,
    "last_brier_sharp": None,
    "last_adjusted": None,
    "brier_history": [],
}


def load_config() -> dict:
    """Load model_config.json, filling in any missing keys from defaults.

    Older configs only stored a scalar `sharp_weight`; we preserve it as the
    cross-sector default while seeding `sharp_weight_by_sector` with the
    historical hardcoded overrides.
    """
    cfg: dict = dict(_DEFAULT_CONFIG)
    # dict(_DEFAULT_CONFIG) is a SHALLOW copy — the nested mutable containers
    # would otherwise be shared with the module-level default. Callers mutate
    # them in place (the per-sector auto-tuner writes sharp_weight_by_sector;
    # every run appends to brier_history), so give each load its own copies or
    # those writes leak into _DEFAULT_CONFIG for the process lifetime.
    cfg["sharp_weight_by_sector"] = dict(_DEFAULT_CONFIG["sharp_weight_by_sector"])
    cfg["brier_history"] = list(_DEFAULT_CONFIG["brier_history"])
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except Exception:
            pass
    cfg.setdefault("sharp_weight_by_sector", dict(_DEFAULT_SHARP_WEIGHT_BY_SECTOR))
    return cfg


def get_sharp_weight(sector: str, cfg: Optional[dict] = None) -> float:
    """Return the sharp_weight to use for a given sector.

    Per-sector overrides take precedence; otherwise the top-level scalar
    `sharp_weight` is used. Sector name is case-insensitive.
    """
    if cfg is None:
        cfg = load_config()
    by_sector = cfg.get("sharp_weight_by_sector") or {}
    key = (sector or "").lower()
    if key in by_sector:
        return float(by_sector[key])
    return float(cfg.get("sharp_weight", 0.85))


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


PROB_TIERS = [
    ("< 20%",  0.00, 0.20),
    ("20–40%", 0.20, 0.40),
    ("40–60%", 0.40, 0.60),
    ("> 60%",  0.60, 1.01),
]


def compute_brier_scores(weeks: int = 1) -> Optional[dict]:
    """
    Compute Brier scores for resolved predictions in the last `weeks` weeks.

    Returns:
      {
        "brier_model": float,   # using blended_true_prob
        "brier_sharp": float,   # using sharp_true_prob
        "n": int,
        "period_start": str,
        "period_end": str,
      }
    or None if no data.
    """
    since = (date.today() - timedelta(weeks=weeks)).isoformat()

    conn = get_connection()
    rows = conn.execute(
        """SELECT o.outcome, o.sharp_true_prob, o.blended_true_prob,
                  p.sector, p.market_type, p.model_sources, p.line
           FROM ev_outcomes o
           JOIN ev_predictions p ON o.market_id = p.market_id
           INNER JOIN (
               SELECT market_id, MAX(scan_date) AS latest_scan
               FROM ev_predictions WHERE voided = 0 GROUP BY market_id
           ) latest ON p.market_id = latest.market_id
                   AND p.scan_date = latest.latest_scan
           WHERE o.outcome IS NOT NULL
             AND p.scan_date >= ?
             AND p.mode = 'live'""",
        (since,),
    ).fetchall()
    conn.close()

    # Drop rows produced by a superseded code state — the SAME contamination
    # guard the shadow-promotion path applies. Without this, the LIVE
    # sharp_weight auto-tuner (which sizes real bets) could be nudged by
    # predictions a later code change already invalidated. is_contaminated
    # returns False for sectors without rules, so this only removes
    # documented-contamination rows.
    from evmax.agents.cleanup.contamination import is_contaminated
    rows = [
        r for r in rows
        if not is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"])
    ]

    if not rows:
        return None

    n = len(rows)
    brier_model = sum((r["blended_true_prob"] - r["outcome"]) ** 2 for r in rows) / n
    brier_sharp = sum((r["sharp_true_prob"] - r["outcome"]) ** 2 for r in rows) / n

    # Per-tier breakdown
    tiers = []
    for label, lo, hi in PROB_TIERS:
        tier_rows = [r for r in rows if lo <= r["blended_true_prob"] < hi]
        if not tier_rows:
            tiers.append({"label": label, "n": 0, "brier_model": None, "brier_sharp": None})
            continue
        tn = len(tier_rows)
        tiers.append({
            "label":       label,
            "n":           tn,
            "brier_model": round(sum((r["blended_true_prob"] - r["outcome"]) ** 2 for r in tier_rows) / tn, 6),
            "brier_sharp": round(sum((r["sharp_true_prob"]   - r["outcome"]) ** 2 for r in tier_rows) / tn, 6),
        })

    return {
        "brier_model":   round(brier_model, 6),
        "brier_sharp":   round(brier_sharp, 6),
        "n":             n,
        "period_start":  since,
        "period_end":    date.today().isoformat(),
        "tiers":         tiers,
    }


def compute_brier_scores_by_sector(weeks: int = 4) -> list[dict]:
    """Per-sector Brier breakdown over the last `weeks` weeks.

    Returns one row per sector sorted by sample count descending, e.g.
      [
        {"sector": "nba", "n": 87, "brier_model": 0.210, "brier_sharp": 0.216,
         "edge_pct": 2.8},
        ...
      ]
    where `edge_pct` is (brier_sharp - brier_model) / brier_sharp × 100.
    Positive → our model is beating Pinnacle, negative → sharp is better and
    we should raise that sector's sharp_weight.
    """
    since = (date.today() - timedelta(weeks=weeks)).isoformat()

    conn = get_connection()
    rows = conn.execute(
        """SELECT p.sector, o.outcome, o.sharp_true_prob, o.blended_true_prob,
                  p.market_type, p.model_sources, p.line
           FROM ev_outcomes o
           JOIN ev_predictions p ON o.market_id = p.market_id
           INNER JOIN (
               SELECT market_id, MAX(scan_date) AS latest_scan
               FROM ev_predictions WHERE voided = 0 GROUP BY market_id
           ) latest ON p.market_id = latest.market_id
                   AND p.scan_date = latest.latest_scan
           WHERE o.outcome IS NOT NULL
             AND p.scan_date >= ?
             AND p.mode = 'live'""",
        (since,),
    ).fetchall()
    conn.close()

    # Same contamination guard the global compute_brier_scores + the shadow path
    # use — this feeds the PER-SECTOR auto-tuner, so superseded-code rows must
    # not drive a sector's live sharp_weight.
    from evmax.agents.cleanup.contamination import is_contaminated
    by_sector: dict[str, list] = {}
    for r in rows:
        if is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"]):
            continue
        by_sector.setdefault((r["sector"] or "").lower(), []).append(r)

    out: list[dict] = []
    for sector, sector_rows in by_sector.items():
        n = len(sector_rows)
        if n == 0:
            continue
        bm = sum((r["blended_true_prob"] - r["outcome"]) ** 2 for r in sector_rows) / n
        bs = sum((r["sharp_true_prob"]   - r["outcome"]) ** 2 for r in sector_rows) / n
        edge = ((bs - bm) / bs * 100.0) if bs > 0 else 0.0
        out.append({
            "sector":      sector or "(unknown)",
            "n":           n,
            "brier_model": round(bm, 6),
            "brier_sharp": round(bs, 6),
            "edge_pct":    round(edge, 2),
        })

    out.sort(key=lambda x: x["n"], reverse=True)
    return out


def adjust_sharp_weight(force: bool = False) -> dict:
    """Auto-adjust sharp_weight PER SECTOR from 4-week per-sector Brier.

    DANGER-B fix. Previously ONE global ``sharp_weight`` was tuned on a POOLED
    all-sector Brier, so a high-volume weak sector dragged the fallback weight
    that unrelated sectors rely on. Now each sector with ≥30 clean resolved
    LIVE bets is tuned on ITS OWN Brier, writing ``sharp_weight_by_sector[sector]``:

      * improvement (bs−bm)/bs > +5%  → models beating sharp → lower that
        sector's weight by 0.05 (floor 0.40);
      * improvement < −5%             → models worse → raise by 0.05 (cap 0.95);
      * otherwise                     → hold.

    LOCKED sectors — those in ``_DEFAULT_SHARP_WEIGHT_BY_SECTOR`` (tennis,
    baseball) — are pinned high for documented thin-stack reasons and are NEVER
    auto-moved. Sectors with n<30 keep the static global fallback (never tuned
    on thin data). The 7-day cooldown and [0.40, 0.95] bounds are unchanged.

    Returns ``{adjusted: bool, reason?: str, sectors: [ {sector, n, old_weight,
    new_weight, brier_model, brier_sharp, improvement_pct, direction} ]}``.
    """
    cfg = load_config()
    today = date.today().isoformat()

    # Cooldown check
    if not force and cfg.get("last_adjusted"):
        last = date.fromisoformat(cfg["last_adjusted"])
        days_since = (date.today() - last).days
        if days_since < 7:
            return {
                "adjusted": False,
                "reason": f"Adjusted {days_since}d ago (min 7d). Use --force to override.",
                "sectors": [],
            }

    by_sector = compute_brier_scores_by_sector(weeks=4)
    eligible = [s for s in by_sector if s["n"] >= 30]
    if not eligible:
        total = sum(s["n"] for s in by_sector)
        return {
            "adjusted": False,
            "reason": f"Insufficient data: no sector with 30+ resolved predictions ({total} total).",
            "sectors": [],
        }

    locked = {s.lower() for s in _DEFAULT_SHARP_WEIGHT_BY_SECTOR}
    by_map = cfg.setdefault("sharp_weight_by_sector", {})
    fallback = float(cfg.get("sharp_weight", 0.85))

    results: list[dict] = []
    any_changed = False
    for s in eligible:
        sector = (s["sector"] or "").lower()
        if sector in locked:
            continue  # deliberately-pinned sector — never auto-tune
        bm, bs, n = s["brier_model"], s["brier_sharp"], s["n"]
        improvement = (bs - bm) / bs if bs > 0 else 0.0
        old_w = float(by_map.get(sector, fallback))
        new_w = old_w
        if improvement > 0.05:
            new_w = max(0.40, round(old_w - 0.05, 2))
            direction = "down (models improving vs sharp)"
        elif improvement < -0.05:
            new_w = min(0.95, round(old_w + 0.05, 2))
            direction = "up (models underperforming vs sharp)"
        else:
            direction = "flat (no significant difference)"
        if new_w != old_w:
            by_map[sector] = new_w
            any_changed = True
        results.append({
            "sector": sector, "n": n, "old_weight": old_w, "new_weight": new_w,
            "brier_model": bm, "brier_sharp": bs,
            "improvement_pct": round(improvement * 100, 2), "direction": direction,
        })

    cfg["last_adjusted"] = today
    cfg.setdefault("brier_history", []).append({
        "date": today,
        "per_sector": [
            {"sector": r["sector"], "n": r["n"], "old_weight": r["old_weight"],
             "new_weight": r["new_weight"], "brier_model": r["brier_model"],
             "brier_sharp": r["brier_sharp"]}
            for r in results
        ],
    })
    save_config(cfg)

    logger.info(
        "sharp_weight_adjusted_by_sector",
        changed=any_changed,
        weights={r["sector"]: r["new_weight"] for r in results},
    )

    return {"adjusted": any_changed, "sectors": results,
            "sharp_weight_by_sector": dict(by_map)}
