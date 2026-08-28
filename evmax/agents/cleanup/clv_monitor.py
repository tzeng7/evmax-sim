"""CLV tripwire — push an alert when a LIVE sector's realized CLV degrades.

Why a tripwire and not a model knob: per the value-audit doctrine (CLAUDE.md,
docs/value-audits/README.md), **CLV is an entry-timing / selection signal, not a
model target**. A live sector bleeding CLV is an entry-timing problem (stale
Kalshi price, wrong entry window), fixable by *when* we enter — never by
reweighting the blend. So this module deliberately does NOT touch sharp_weight
or any model parameter. It reuses the promotion board's per-(sector, market,
venue) CLV computation (same staleness filter, same LIVE-DEGRADING verdict) and
turns the board's already-computed verdict into a *push* so a degrading live
book is caught without anyone having to open the dashboard.

``LIVE-DEGRADING`` = a live-mode group with clv n ≥ 30 and mean CLV < 0.
"""
from __future__ import annotations

from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

_DEGRADING = "LIVE-DEGRADING"


def find_degrading_groups(
    days: int = 30,
    staleness_h: Optional[float] = 3.0,
    sector: Optional[str] = None,
) -> list[dict]:
    """Return the board rows whose verdict is LIVE-DEGRADING, worst CLV first.

    Thin wrapper over ``compute_promotion_board`` so the tripwire and the
    dashboard/CLI board can never disagree about what "degrading" means.
    """
    from evmax.agents.cleanup.promotion_board import compute_promotion_board

    board = compute_promotion_board(days=days, staleness_h=staleness_h, sector=sector)
    degrading = [row for row in board if row.get("verdict") == _DEGRADING]
    degrading.sort(key=lambda r: (r.get("clv") or {}).get("mean_clv_pp", 0.0))
    return degrading


def format_clv_alert(rows: list[dict], days: int) -> tuple[str, str]:
    """Build the (title, message) for a CLV-degradation alert."""
    n = len(rows)
    title = f"{n} live book{'s' if n != 1 else ''} bleeding CLV ({days}d)"
    lines = []
    for r in rows:
        clv = r.get("clv") or {}
        lines.append(
            f"• {r['sector']} {r['market_type']} [{r['venue']}] — "
            f"mean CLV {clv.get('mean_clv_pp', 0.0):+.2f}pp "
            f"over n={clv.get('n', 0)} ({clv.get('frac_positive', 0.0) * 100:.0f}% pos)"
        )
    lines.append(
        "Entry-timing/selection issue (stale line) — inspect the entry window, "
        "not the model blend. `evmax cleanup shadow clv <sector>` for the split."
    )
    return title, "\n".join(lines)


def run_clv_tripwire(
    days: int = 30,
    staleness_h: Optional[float] = 3.0,
    sector: Optional[str] = None,
    notify: bool = False,
) -> dict:
    """Find degrading live books and, if ``notify``, push a single warning alert.

    Returns ``{"degrading": [...rows...], "notified": bool}``. ``notified`` is
    True only when there was something to report AND a webhook accepted it.
    """
    degrading = find_degrading_groups(days=days, staleness_h=staleness_h, sector=sector)
    notified = False
    if degrading and notify:
        from evmax.notifications import Notifier

        title, message = format_clv_alert(degrading, days)
        notified = Notifier.from_settings().notify_alert(
            title, message, severity="warning"
        )
        logger.info("clv_tripwire_alert", n=len(degrading), notified=notified)
    elif degrading:
        logger.info("clv_tripwire_degrading", n=len(degrading))
    return {"degrading": degrading, "notified": notified}
