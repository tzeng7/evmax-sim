"""Dynamic pruner for stale open bet-candidates.

The scanner freezes every +EV gap into ``ev_predictions`` at its scan-time price
(``UNIQUE(market_id)``, freeze-on-first-insert). Un-picked candidates
(``placed=0``) linger in the dashboard "Open Positions" panel and
``cleanup show`` showing that frozen edge — but as tip-off approaches the live
venue ask reverts toward the sharp close and the edge becomes phantom.

This module re-prices those candidates against the CURRENT market (via the shared
:func:`evmax.agents.cleanup.reprice.reprice_rows` — the identical engine
``agents pick --live`` uses) and voids the ones whose edge has reverted below the
flag threshold, so the open list reflects only edges that still exist. It is the
core of ``evmax cleanup prune-stale``.

Design invariants (all fail-safe — the pruner never removes value it can't prove
has evaporated):

- **Scope: un-picked live candidates only** (``placed=0, mode='live'``). Placed
  bets, shadow rows, and resolved rows are never touched.
- **Never void on a missing quote.** A void requires a real live ask
  (``price_is_live`` and a non-``None`` ``live_ev_real``). An empty orderbook
  returns a ≥0.99 ask which reads as "not live" but is a liquidity gap, NOT a
  reverted edge — those are skipped.
- **Never void an in-play / started game.** The upcoming check uses the tz-aware
  Pinnacle start time; a pulled/missing line ⇒ can't confirm upcoming ⇒ skip.
- **Reversible via un-void with hysteresis.** ``UNIQUE(market_id)`` means a
  re-scan can't resurrect a voided row, so the pruner un-voids ONLY its own rows
  (``void_reason='stale_reverted'``) when the edge recovers past
  ``threshold + hysteresis`` — a dead-band that stops a candidate hovering at the
  threshold from flapping void/un-void across sweeps.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Callable, Optional

from evmax.agents.cleanup.reprice import reprice_rows
from evmax.ev.calculator import tiered_min_ev

# Marker written to ``ev_predictions.void_reason`` for a pruner auto-void. It is
# what scopes un-void to the pruner's own rows — a manual/game-cancel void leaves
# void_reason NULL and is never un-voided.
STALE_VOID_REASON = "stale_reverted"


def prune_candidates(
    conn: sqlite3.Connection,
    *,
    today: str,
    date_hi: Optional[str] = None,
) -> list[dict]:
    """Load un-picked live candidates eligible for stale-pruning.

    Returns un-picked (``placed=0``), live (``mode='live'``), unresolved rows
    whose event is today or later — INCLUDING rows the pruner previously voided
    (``void_reason='stale_reverted'``), so a recovered edge can be un-voided.
    Manual/game-cancel voids (``voided=1`` with a NULL/other reason) are excluded
    and never reopened.

    The flat query (no latest-scan self-join) is correct because
    ``ev_predictions`` is ``UNIQUE(market_id)`` — one row per market — and a
    ``voided=0`` self-join would hide exactly the rows we need to un-void.

    ``date_hi`` (inclusive, ``YYYY-MM-DD``) coarsely bounds how far ahead to
    consider; the precise kickoff-window gate is applied later in
    :func:`decide_actions` against the tz-aware Pinnacle start time.
    """
    sql = """
        SELECT p.id, p.market_id, p.event_id, p.event_title, p.event_date,
               p.yes_team, p.sector, p.market_type, p.line, p.venue,
               p.kalshi_yes_price, p.sharp_true_prob, p.blended_true_prob,
               p.ev_pct, p.sharp_weight_used, p.voided, p.void_reason
        FROM ev_predictions p
        LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE p.placed = 0
          AND p.mode = 'live'
          AND (p.voided = 0 OR p.void_reason = ?)
          AND (o.outcome IS NULL OR o.id IS NULL)
          AND COALESCE(p.event_date, p.scan_date) >= ?
    """
    params: list = [STALE_VOID_REASON, today]
    if date_hi is not None:
        sql += " AND COALESCE(p.event_date, p.scan_date) <= ?"
        params.append(date_hi)
    sql += " ORDER BY p.ev_pct DESC"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def decide_actions(
    bets: list[dict],
    *,
    min_ev: float,
    min_prob: float,
    hysteresis: float,
    now: datetime,
    lookahead_hours: Optional[float] = None,
) -> list[dict]:
    """Classify each re-priced candidate into an action.

    ``bets`` is the output of :func:`reprice_rows`. Per bet, exactly one of:

    - ``void`` — an open row whose edge reverted below the tiered flag threshold
      at a real live price, for an upcoming game. This is the SAME gate
      ``pick``/``verify``/scan use (``is_live=False``), so a voided candidate is
      one the scanner would no longer flag at the current price.
    - ``unvoid`` — a row the pruner previously voided whose live EV has recovered
      to ``threshold + hysteresis`` (the anti-flap dead-band), for an upcoming
      game.
    - ``skip_no_quote`` — no live ask / no-book (≥0.99) / settled price. NEVER
      voided (a liquidity gap is not a reverted edge).
    - ``skip_no_start`` — a live ask but no fresh Pinnacle start time, so we
      can't confirm the game is upcoming. Fail-safe skip.
    - ``skip_inplay`` — the game has started (start ≤ now).
    - ``skip_future`` — the game is beyond the lookahead window.
    - ``noop`` — an open row still live, or a voided row not yet recovered.

    ``now`` and each bet's ``event_start`` must both be tz-aware.
    """
    horizon = now + timedelta(hours=lookahead_hours) if lookahead_hours else None
    actions: list[dict] = []
    for b in bets:
        blended = b["blended_true_prob"]
        threshold = tiered_min_ev(blended, min_ev=min_ev, min_prob=min_prob)
        live_ev = b.get("live_ev_real")
        quote_ok = bool(b.get("price_is_live")) and live_ev is not None
        es = b.get("event_start")
        open_row = b.get("voided", 0) == 0
        pruner_voided = b.get("voided", 0) == 1 and b.get("void_reason") == STALE_VOID_REASON

        if not quote_ok:
            action, reason = "skip_no_quote", "no live orderbook quote"
        elif es is None:
            action, reason = "skip_no_start", "no fresh Pinnacle start time"
        elif es <= now:
            action, reason = "skip_inplay", "event already started"
        elif horizon is not None and es > horizon:
            action, reason = "skip_future", "beyond lookahead window"
        elif open_row and not b.get("is_live"):
            action = "void"
            reason = (
                f"edge reverted: live EV {live_ev * 100:+.1f}% "
                f"< {threshold * 100:.1f}% threshold"
            )
        elif pruner_voided and b.get("is_live") and live_ev >= threshold + hysteresis:
            action = "unvoid"
            reason = (
                f"edge recovered: live EV {live_ev * 100:+.1f}% "
                f"≥ {threshold * 100:.1f}%+{hysteresis * 100:.1f}pp"
            )
        else:
            action, reason = "noop", "unchanged"

        actions.append({
            "action": action,
            "reason": reason,
            "market_id": b["market_id"],
            "event_title": b.get("event_title"),
            "sector": b.get("sector"),
            "yes_team": b.get("yes_team"),
            "market_type": b.get("market_type"),
            "line": b.get("line"),
            "venue": b.get("venue"),
            "scan_ask": b.get("scan_ask"),
            "live_ask": b.get("live_ask"),
            "scan_ev": b.get("ev_pct"),
            "live_ev": live_ev,
            "threshold": threshold,
            "pin_is_live": b.get("pin_is_live"),
            "event_start": es,
        })
    return actions


def apply_actions(
    conn: sqlite3.Connection,
    actions: list[dict],
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Persist the ``void`` / ``unvoid`` actions; return a per-action count.

    Each write is a single parametrized ``UPDATE`` with a defensive guard in the
    ``WHERE`` (void only an open row; un-void only a stale-void row) so a
    concurrent resolve/void/pick between :func:`prune_candidates` and here can't
    be clobbered. ``dry_run=True`` counts without writing.
    """
    counts: dict[str, int] = {}
    for a in actions:
        counts[a["action"]] = counts.get(a["action"], 0) + 1

    if dry_run:
        return counts

    wrote = False
    for a in actions:
        if a["action"] == "void":
            conn.execute(
                "UPDATE ev_predictions SET voided = 1, void_reason = ? "
                "WHERE market_id = ? AND voided = 0",
                (STALE_VOID_REASON, a["market_id"]),
            )
            wrote = True
        elif a["action"] == "unvoid":
            conn.execute(
                "UPDATE ev_predictions SET voided = 0, void_reason = NULL "
                "WHERE market_id = ? AND void_reason = ?",
                (a["market_id"], STALE_VOID_REASON),
            )
            wrote = True
    if wrote:
        conn.commit()
    return counts


async def evaluate_candidates(
    conn: sqlite3.Connection,
    *,
    min_ev: float,
    min_prob: float,
    hysteresis: float,
    kelly: float,
    bankroll: float,
    fees_in_pricing: bool,
    max_kelly: float,
    lookahead_hours: Optional[float],
    now: datetime,
    today: str,
    date_hi: Optional[str] = None,
    sectors: Optional[list[str]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> tuple[list[dict], list[dict]]:
    """Load candidates, re-price them live, and classify — no writes.

    Returns ``(bets, actions)``: the re-priced rows and the per-row action
    decisions. The caller persists them with :func:`apply_actions`. Split from
    the write step so the whole decision pipeline is unit-testable with a
    monkeypatched :func:`reprice_rows`.
    """
    rows = prune_candidates(conn, today=today, date_hi=date_hi)
    if sectors:
        wanted = {s.lower() for s in sectors}
        rows = [r for r in rows if (r.get("sector") or "").lower() in wanted]
    if not rows:
        return [], []
    bets = await reprice_rows(
        rows,
        live=True,
        kelly=kelly,
        bankroll=bankroll,
        min_ev=min_ev,
        min_prob=min_prob,
        fees_in_pricing=fees_in_pricing,
        max_kelly=max_kelly,
        on_warning=on_warning,
    )
    actions = decide_actions(
        bets,
        min_ev=min_ev,
        min_prob=min_prob,
        hysteresis=hysteresis,
        now=now,
        lookahead_hours=lookahead_hours,
    )
    return bets, actions
