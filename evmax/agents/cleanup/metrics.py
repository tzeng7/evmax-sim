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

_DEFAULT_CONFIG: dict = {
    "sharp_weight": 0.85,
    "last_brier_model": None,
    "last_brier_sharp": None,
    "last_adjusted": None,
    "brier_history": [],
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return dict(_DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


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
           WHERE o.outcome IS NOT NULL
             AND p.scan_date >= ?""",
        (since,),
    ).fetchall()
    conn.close()

    if not rows:
        return None

    n = len(rows)
    brier_model = sum((r["blended_true_prob"] - r["outcome"]) ** 2 for r in rows) / n
    brier_sharp = sum((r["sharp_true_prob"] - r["outcome"]) ** 2 for r in rows) / n

    return {
        "brier_model":   round(brier_model, 6),
        "brier_sharp":   round(brier_sharp, 6),
        "n":             n,
        "period_start":  since,
        "period_end":    date.today().isoformat(),
    }


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
