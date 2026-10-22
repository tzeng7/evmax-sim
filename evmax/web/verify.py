"""On-demand live verify for the dashboard — backs ``POST /api/verify``.

The CLI's ``agents verify`` / ``pick --live`` re-fetch the live venue ask and a
fresh Pinnacle line, then re-gate and re-size at the CURRENT price. The dashboard
used to place at the stale scan-time ask. This module reuses the same engine
(:func:`evmax.agents.cleanup.reprice.reprice_rows`) for a list of market_ids and
classifies each row keep / drop. It is READ-ONLY: nothing is written to the DB.

Fail-safes inherited from the engine: an empty / >=0.99 / missing book is never a
valid price, and a PolyUS fetch failure demotes only PolyUS rows. Added here:
a game whose Pinnacle start time has passed is dropped as in-play.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from evmax.agents.cleanup.reprice import reprice_rows
from evmax.ev.calculator import tiered_min_ev

DEFAULT_MIN_EV = 0.02
DEFAULT_MIN_PROB = 0.15

_ROW_COLUMNS = (
    "p.market_id, p.event_id, p.event_title, p.yes_team, p.sector, p.market_type, "
    "p.kalshi_yes_price, p.sharp_true_prob, p.blended_true_prob, p.ev_pct, "
    "p.kelly_fraction, p.line, p.venue, p.bankroll_used, p.sharp_weight_used, "
    "p.voided, p.placed, p.mode"
)


def load_rows(conn, market_ids: list[str]) -> tuple[list[dict], dict[str, str]]:
    """Latest ``ev_predictions`` row per market_id.

    Returns ``(verifiable_rows, rejected)`` where ``rejected`` maps market_id to
    why it can't be verified (not found / placed / voided / not live).
    """
    rows: list[dict] = []
    rejected: dict[str, str] = {}
    for mid in dict.fromkeys(market_ids):
        r = conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM ev_predictions p "
            "WHERE p.market_id = ? ORDER BY p.id DESC LIMIT 1",
            (mid,),
        ).fetchone()
        if r is None:
            rejected[mid] = "not in ev_predictions"
            continue
        d = dict(r)
        if d.get("placed"):
            rejected[mid] = "already placed"
        elif d.get("voided"):
            rejected[mid] = "voided"
        elif (d.get("mode") or "live") != "live":
            rejected[mid] = f"mode={d['mode']}"
        else:
            rows.append(d)
    return rows, rejected


def classify(bets: list[dict], *, min_ev: float, min_prob: float,
             now: Optional[datetime] = None) -> list[dict]:
    """Turn :func:`reprice_rows` output into per-row keep/drop verdicts."""
    now = now or datetime.now(timezone.utc)
    out: list[dict] = []
    for b in bets:
        blended = b["blended_true_prob"]
        threshold = tiered_min_ev(blended, min_ev=min_ev, min_prob=min_prob)
        live_ev = b.get("live_ev_real")
        es = b.get("event_start")
        if es is not None and es.tzinfo is None:
            es = es.replace(tzinfo=timezone.utc)
        scan_ask, live_ask = b.get("scan_ask"), b.get("live_ask")

        quote_ok = bool(b.get("price_is_live")) and live_ev is not None
        if not quote_ok:
            keep, reason = False, "no live quote (empty book, settled or fetch failed)"
        elif es is not None and es <= now:
            keep, reason = False, "in-play: game already started"
        elif not b.get("is_live"):
            keep, reason = False, (
                f"below EV floor: live EV {live_ev * 100:+.1f}% < {threshold * 100:.1f}%"
            )
        else:
            keep, reason = True, "ok"

        out.append({
            "market_id": b["market_id"],
            "scan_ask": scan_ask,
            # No usable quote => no live ask; never hand the UI a scan price
            # dressed up as a live one.
            "live_ask": live_ask if quote_ok else None,
            "delta": (live_ask - scan_ask) if quote_ok and scan_ask is not None else None,
            "scan_ev": b.get("ev_pct"),
            "live_ev": live_ev,
            "delta_ev": (live_ev - b["ev_pct"]) if live_ev is not None and b.get("ev_pct") is not None else None,
            "live_fair": blended,
            "threshold": threshold,
            "stake": round(b.get("stake") or 0.0, 2) if keep else 0.0,
            "keep": keep,
            "reason": reason,
            "price_is_live": bool(b.get("price_is_live")),
            "pin_is_live": bool(b.get("pin_is_live")),
            "in_play": bool(es is not None and es <= now),
            "start_unknown": es is None,
            "venue": b.get("venue"),
        })
    return out


async def verify_market_ids(
    conn,
    market_ids: list[str],
    *,
    bankroll: Optional[float],
    kelly: float,
    min_ev: float = DEFAULT_MIN_EV,
    min_prob: float = DEFAULT_MIN_PROB,
) -> dict[str, Any]:
    from evmax.settings import get_settings

    rows, rejected = load_rows(conn, market_ids)
    warnings: list[str] = []
    results: list[dict] = []
    live_effective = True
    if rows:
        if bankroll is None:
            bankroll = next((r["bankroll_used"] for r in rows if r.get("bankroll_used")), None) or 250.0
        settings = get_settings()
        bets = await reprice_rows(
            rows,
            live=True,
            kelly=kelly,
            bankroll=bankroll,
            min_ev=min_ev,
            min_prob=min_prob,
            fees_in_pricing=settings.fees_in_pricing,
            max_kelly=settings.max_kelly_fraction,
            on_warning=lambda m: warnings.append(m.strip()),
        )
        live_effective = bool(bets and bets[0]["live_effective"])
        results = classify(bets, min_ev=min_ev, min_prob=min_prob)

    for mid, why in rejected.items():
        results.append({
            "market_id": mid, "keep": False, "reason": why, "scan_ask": None,
            "live_ask": None, "delta": None, "scan_ev": None, "live_ev": None,
            "delta_ev": None, "live_fair": None, "threshold": None, "stake": 0.0,
            "price_is_live": False, "pin_is_live": False, "in_play": False,
            "start_unknown": False, "venue": None,
        })
    return {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "live": live_effective,
        "kept": sum(1 for r in results if r["keep"]),
        "results": results,
        "warnings": warnings,
    }
