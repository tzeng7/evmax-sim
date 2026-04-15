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
_DEFAULT_SHARP_WEIGHT_BY_SECTOR: dict[str, float] = {
    "tennis":   0.92,
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
        """SELECT o.outcome, o.sharp_true_prob, o.blended_true_prob
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
        """SELECT p.sector, o.outcome, o.sharp_true_prob, o.blended_true_prob
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

    by_sector: dict[str, list] = {}
    for r in rows:
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
    """
    Auto-adjust sharp_weight based on 4-week Brier score comparison.

    Returns a result dict with keys: adjusted, reason, sharp_weight, direction,
    brier_model, brier_sharp, n, improvement_pct.
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
                "sharp_weight": cfg["sharp_weight"],
            }

    scores = compute_brier_scores(weeks=4)
    n = scores["n"] if scores else 0

    if scores is None or n < 30:
        return {
            "adjusted": False,
            "reason": f"Insufficient data: {n} resolved predictions (need 30+).",
            "sharp_weight": cfg["sharp_weight"],
        }

    bm = scores["brier_model"]
    bs = scores["brier_sharp"]
    # Positive improvement = model is better than sharp-only
    improvement = (bs - bm) / bs if bs > 0 else 0.0

    old_weight = cfg["sharp_weight"]
    new_weight = old_weight

    if improvement > 0.05:
        new_weight = max(0.40, round(old_weight - 0.05, 2))
        direction = "↓ (models improving vs sharp)"
    elif improvement < -0.05:
        new_weight = min(0.95, round(old_weight + 0.05, 2))
        direction = "↑ (models underperforming vs sharp)"
    else:
        direction = "= (no significant difference)"

    changed = new_weight != old_weight

    cfg["sharp_weight"] = new_weight
    cfg["last_brier_model"] = bm
    cfg["last_brier_sharp"] = bs
    cfg["last_adjusted"] = today
    cfg.setdefault("brier_history", []).append({
        "date":       today,
        "brier_model": bm,
        "brier_sharp": bs,
        "n":           n,
        "old_weight":  old_weight,
        "new_weight":  new_weight,
    })
    save_config(cfg)

    logger.info(
        "sharp_weight_adjusted",
        old=old_weight,
        new=new_weight,
        brier_model=bm,
        brier_sharp=bs,
        n=n,
        improvement_pct=round(improvement * 100, 2),
    )

    return {
        "adjusted":       changed,
        "direction":      direction,
        "sharp_weight":   new_weight,
        "brier_model":    bm,
        "brier_sharp":    bs,
        "n":              n,
        "improvement_pct": round(improvement * 100, 2),
    }
