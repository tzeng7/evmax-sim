"""Live re-pricing engine shared by ``evmax agents pick`` and
``evmax cleanup prune-stale``.

Re-fetches the CURRENT venue ask (Kalshi WebSocket / Polymarket US gateway) and
re-derives ``blended_true_prob`` from a freshly re-fetched Pinnacle moneyline
line, then re-gates and re-sizes each candidate at that price. ``pick`` uses
this to stop you placing scan-time edges that have already reverted; the
stale-position pruner uses the *identical* rule to void reverted un-picked
candidates — so "mispriced" means exactly the same thing in both places.

The four pure helpers (``recompute_at_price`` / ``reblend_with_fresh_sharp`` /
``polymarket_live_ask`` / ``fetch_polymarket_live_asks``) live here and are
re-exported from ``evmax.cli.commands.agents`` for backward compatibility.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from evmax.ev.calculator import tiered_min_ev

# Market types whose fresh Pinnacle two-way devig can be applied to the blend.
# The empty string covers rows scanned before market_type was populated.
_MONEYLINE_TYPES = ("moneyline", "ml", "")


def polymarket_live_ask(market_id: str, markets_by_id: dict) -> Optional[float]:
    """Resolve the live ask for a Polymarket US row from a freshly-fetched
    ``{market.id: PredictionMarket}`` map.

    A stored PolyUS row id is ``polymarket_us:{slug}[:{side_key}]`` for the YES
    side, with a ``:no`` suffix appended for the derived NO side of a spread or
    total. Stripping ``:no`` recovers the market's own id, which is the map key.

    Unlike Kalshi — where the WebSocket only returns the YES ask, so a NO ask is
    approximated as ``1 - yes`` and understates the true ask by the spread —
    Polymarket quotes BOTH sides in the same payload. So a NO row returns the
    market's ``no_price`` DIRECTLY, an exact live ask with no spread fudge.

    Returns ``None`` when the market is no longer listed (settled / closed → not
    in the fresh map) or unquoted, so the caller falls back to the scan price.
    """
    is_no = market_id.endswith(":no")
    base = market_id.removesuffix(":no")
    market = markets_by_id.get(base)
    if market is None:
        return None
    price = market.no_price if is_no else market.yes_price
    if price is None:
        return None
    return max(0.0, min(1.0, float(price)))


async def fetch_polymarket_live_asks(poly_rows: list[dict]) -> dict[str, Optional[float]]:
    """Fetch current Polymarket US asks for ``poly_rows``, keyed by stored id.

    Groups the rows by sector, re-fetches each sector's live league events with
    the cache bypassed (``fresh=True``), and resolves every row's ask through
    :func:`polymarket_live_ask`. A sector whose fetch raises is skipped — its
    rows resolve to ``None`` and fall back to the scan price at the call site,
    exactly like a Kalshi row whose ticker the WebSocket missed.
    """
    from evmax.clients.polymarket_us import PolymarketUSClient

    sectors = sorted({(r.get("sector") or "").lower() for r in poly_rows if r.get("sector")})
    markets_by_id: dict[str, object] = {}
    if sectors:
        async with PolymarketUSClient() as client:
            results = await asyncio.gather(
                *(client.get_markets(s, fresh=True) for s in sectors),
                return_exceptions=True,
            )
        for res in results:
            if isinstance(res, Exception):
                continue
            for market in res:
                markets_by_id[market.id] = market
    return {
        r["market_id"]: polymarket_live_ask(r["market_id"], markets_by_id)
        for r in poly_rows
    }


def recompute_at_price(
    *,
    blended_prob: float,
    price: Optional[float],
    base_kelly: float,
    max_kelly: float,
    bankroll: float,
    min_ev: float,
    min_prob: float,
    venue: Optional[str] = None,
) -> dict:
    """Recompute EV / live-gate / Kelly stake at a given Kalshi ask ``price``.

    Pure helper so ``pick`` can size and gate a bet against the CURRENT price
    instead of the stale scan-time ask — the fix for the selection curse where
    the fattest scan-time edges are the most reverting. Returns
    ``{ev, is_live, kelly_fraction, stake}``; a missing/degenerate price (None,
    settled 0/1, or empty-book ≥0.99) yields a not-live, zero-stake result.

    ``venue`` selects the trading-fee model (``"kalshi"`` / ``"polymarket_us"``);
    the contract is priced net of that fee for both the gate and the stake.
    ``venue=None`` prices gross of fees (the caller passes None when the
    ``fees_in_pricing`` setting is off).
    """
    from evmax.ev.calculator import calculate_ev, effective_price
    from evmax.ev.kelly import compute_kelly

    if price is None or not (0 < price < 0.99):
        return {"ev": None, "is_live": False, "kelly_fraction": 0.0, "stake": 0.0}

    eff_price = effective_price(price, venue)
    ev, _ = calculate_ev(eff_price, blended_prob)
    threshold = tiered_min_ev(blended_prob, min_ev=min_ev, min_prob=min_prob)
    is_live = ev >= threshold and blended_prob >= min_prob
    kelly_fraction = 0.0
    if is_live:
        k = compute_kelly(
            true_prob=blended_prob,
            payout_decimal=1.0 / eff_price,
            edge_pct=ev,
            spread_pct=0.0,
            base_fraction=base_kelly,
            max_kelly=max_kelly,
        )
        kelly_fraction = k.kelly_fraction
    return {
        "ev": ev,
        "is_live": is_live,
        "kelly_fraction": kelly_fraction,
        "stake": bankroll * kelly_fraction,
    }


def reblend_with_fresh_sharp(
    *,
    blended_prob: float,
    scan_sharp_prob: Optional[float],
    fresh_sharp_prob: Optional[float],
    sharp_weight: Optional[float],
) -> float:
    """Re-derive ``blended_true_prob`` from a freshly re-fetched Pinnacle line.

    Pure helper so ``pick`` can re-blend against the CURRENT sharp line instead
    of the stale scan-time one — the other half of the entry-timing fix
    (``recompute_at_price`` above only refreshes the Kalshi side). Measured on
    resolved live bets: Pinnacle's own devigged prob drifts a median ~0.8pp /
    p90 ~4pp between scan time and a T-60 pre-tip snapshot, and leaving
    blended_true_prob frozen flips the bet/no-bet decision on ~19% of rows
    (15% of those are false positives — pick() would size and place a bet that
    no longer has an edge once the sharp line is fresh).

    Mirrors ``EnsembleModelAgent._blend``'s linear formula exactly:
    ``blended = sharp_weight * sharp_prob + (1 - sharp_weight) * model_prob``.
    Shifting the frozen blend by ``sharp_weight * (fresh - scan)`` is
    algebraically identical to backing out the model component and re-blending
    it against the fresh sharp prob — provided the model component itself
    hasn't moved, which holds over a scan-to-pick window since models only
    update at resolve time / weekly seeds, never intraday. Falls back to the
    frozen ``blended_prob`` unchanged when any input needed to re-derive is
    missing (no fresh quote for this event, no recorded sharp_weight, spread/
    total row, or ``--no-live``).
    """
    if scan_sharp_prob is None or fresh_sharp_prob is None or sharp_weight is None:
        return blended_prob
    updated = blended_prob + sharp_weight * (fresh_sharp_prob - scan_sharp_prob)
    return max(0.001, min(0.999, updated))


def apply_cash_cap_to_bets(
    bets: list[dict], cash_by_venue: Optional[dict[str, float]]
) -> None:
    """In-place per-venue cash cap on re-priced bets (GAP 2 at placement).

    Scale each venue's summed LIVE stakes down to fit its deployable cash. The
    Kelly base is total wealth, but a new bet is funded from cash, so a venue
    whose wealth is mostly locked in open positions can be told to stake more
    than it can fund. Cash is a LIVE overlay (it changes as positions settle),
    so it is applied here at re-price time rather than persisted at scan.

    Fail-soft: a venue absent from ``cash_by_venue`` (cash unknown, or not
    selected) is left uncapped; an empty / None map is a no-op. Mutates the
    ``kelly_fraction_used`` and ``stake`` of scaled bets in place.
    """
    if not cash_by_venue:
        return
    requested: dict[str, float] = {}
    for b in bets:
        if not b.get("is_live") or (b.get("kelly_fraction_used") or 0.0) <= 0:
            continue
        v = b.get("venue") or "kalshi"
        requested[v] = requested.get(v, 0.0) + (b.get("stake") or 0.0)
    scale: dict[str, float] = {}
    for v, req in requested.items():
        cash = cash_by_venue.get(v)
        if cash is None or req <= 0 or req <= cash:
            continue
        scale[v] = cash / req
    if not scale:
        return
    for b in bets:
        f = scale.get(b.get("venue") or "kalshi")
        if f is not None and (b.get("kelly_fraction_used") or 0.0) > 0:
            b["kelly_fraction_used"] = round(b["kelly_fraction_used"] * f, 4)
            b["stake"] = round((b.get("stake") or 0.0) * f, 2)


async def reprice_rows(
    rows: list[dict],
    *,
    live: bool,
    kelly: float,
    bankroll: float,
    min_ev: float,
    min_prob: float,
    fees_in_pricing: bool,
    max_kelly: float,
    on_warning: Optional[Callable[[str], None]] = None,
    cash_by_venue: Optional[dict[str, float]] = None,
) -> list[dict]:
    """Re-price each candidate row against the CURRENT market.

    For every row: refresh the venue ask (Kalshi WS / PolyUS gateway), re-blend
    ``blended_true_prob`` off a freshly re-fetched Pinnacle *moneyline* line,
    then re-gate/re-size at that price via :func:`recompute_at_price`. This is
    the enrichment ``pick`` runs under ``--live``, extracted so
    ``cleanup prune-stale`` gates stale candidates by the identical rule.

    Each returned dict preserves the input row's keys (``{**r}``) and adds:

    - ``scan_ask`` — the frozen scan-time ask (``kalshi_yes_price``).
    - ``live_ask`` — the price actually gated against (live ask, else scan ask).
    - ``price_is_live`` — True only when a live ask was fetched for this row.
    - ``pin_is_live`` — True when the fair was re-blended off a fresh Pinnacle
      quote (moneyline rows only).
    - ``blended_true_prob`` — possibly re-blended fair.
    - ``live_ev`` — display EV (raw recompute EV, or the frozen ``ev_pct`` when
      the price was degenerate) — kept identical to pick's column.
    - ``live_ev_real`` — the RAW recompute EV, ``None`` for a degenerate /
      no-book (≥0.99) / settled price. Distinguishes "real ask, edge reverted"
      from "no orderbook" — the pruner must never void on the latter.
    - ``event_start`` — tz-aware Pinnacle start datetime for the event, or None
      (lets the pruner skip in-play / already-started games).
    - ``is_live`` / ``kelly_fraction_used`` / ``stake`` — the re-gate result.
    - ``live_effective`` — the possibly-downgraded ``live`` flag (a Kalshi-fetch
      failure collapses the whole call to scan mode, mirroring pick).

    ``on_warning(msg)`` receives the same human warnings pick prints on a fetch
    failure; pass ``console.print`` to reproduce them, or ``None`` to silence.

    ``cash_by_venue`` (``{venue: deployable_cash}``) caps each venue's summed
    live stakes at its cash (see :func:`apply_cash_cap_to_bets`). Pass it from a
    venue-selection plan at placement time; ``None`` (the default, and what the
    background pruner passes) skips the cap.
    """
    from evmax.agents.cleanup.resolver import yes_aligned_close_prob
    from evmax.clients.kalshi import KalshiClient

    def _warn(msg: str) -> None:
        if on_warning is not None:
            on_warning(msg)

    # --- 1. Refresh the venue ask (Kalshi WS first, PolyUS gateway second) ---
    live_prices: dict[str, Optional[float]] = {}
    if live:
        raw_ids = [r["market_id"] for r in rows if (r.get("venue") or "kalshi") == "kalshi"]
        fetch_tickers = list({mid.removesuffix(":no") for mid in raw_ids})
        try:
            async with KalshiClient() as client:
                yes_asks = await client.get_market_asks_batch(fetch_tickers)
            for mid in raw_ids:
                if mid.endswith(":no"):
                    ya = yes_asks.get(mid.removesuffix(":no"))
                    live_prices[mid] = (max(0.0, min(1.0, 1.0 - ya)) if ya is not None else None)
                else:
                    live_prices[mid] = yes_asks.get(mid)
        except Exception as fetch_err:  # noqa: BLE001 — fall back to scan prices
            _warn(f"[yellow]Live fetch failed ({fetch_err}); using scan prices.[/yellow]\n")
            live = False

        # A PolyUS failure only affects PolyUS rows (they fall back to scan
        # price below); it must NOT clear ``live``, which still governs the
        # Kalshi refresh + Pinnacle re-blend for every other row.
        poly_rows = [r for r in rows if r.get("venue") == "polymarket_us"]
        if live and poly_rows:
            try:
                live_prices.update(await fetch_polymarket_live_asks(poly_rows))
            except Exception as poly_err:  # noqa: BLE001 — fall back to scan prices
                _warn(
                    f"[yellow]Polymarket US live fetch failed ({poly_err}); "
                    f"using scan prices for PolyUS rows.[/yellow]\n"
                )

    # --- 2. Refresh the Pinnacle line for the moneyline sectors in play ---
    fresh_sharp_odds: dict[str, object] = {}
    if live:
        from evmax.clients.esports_pinnacle import PinnacleGuestClient

        ml_sectors = sorted({
            r["sector"] for r in rows
            if (r.get("market_type") or "").lower() in _MONEYLINE_TYPES
        })
        if ml_sectors:
            try:
                async with PinnacleGuestClient() as client:
                    odds_lists = await asyncio.gather(
                        *(client.get_odds(s) for s in ml_sectors),
                        return_exceptions=True,
                    )
                for odds in odds_lists:
                    if isinstance(odds, Exception):
                        continue
                    for o in odds:
                        fresh_sharp_odds[o.event_id] = o
            except Exception as sharp_err:  # noqa: BLE001 — keep frozen sharp line
                _warn(
                    f"[yellow]Pinnacle refresh failed ({sharp_err}); "
                    f"using scan-time sharp line.[/yellow]\n"
                )

    # --- 3. Per-row re-blend + re-gate ---
    bets: list[dict] = []
    for r in rows:
        blended_prob = r["blended_true_prob"]
        scan_ask = r["kalshi_yes_price"]
        live_ask = live_prices.get(r["market_id"]) if live else scan_ask
        # On a settled/errored live fetch, fall back to the scan ask so the row
        # still re-gates (flagged price_is_live=False), rather than vanishing.
        price = live_ask if live_ask is not None else scan_ask

        fresh_so = fresh_sharp_odds.get(r["event_id"])
        # event_start comes from ANY fresh quote for this event, regardless of
        # market type — the pruner needs it to gate on kickoff time even for
        # spread/total rows that don't re-blend.
        event_start = getattr(fresh_so, "event_date", None) if fresh_so is not None else None

        # Re-derive blended_true_prob from the fresh Pinnacle line — MONEYLINE
        # ONLY. The fresh prob is a two-way moneyline devig; applying it to a
        # spread/total blend would mix market types. So spread/total fairs stay
        # frozen and are re-gated purely on ask movement (matches pick's
        # "moneyline only" note; get_odds returns spread SharpOdds too, keyed by
        # the same event_id, so the guard is what keeps them out of the blend).
        pin_is_live = False
        sharp_weight_used = r.get("sharp_weight_used")
        mt = (r.get("market_type") or "").lower()
        if (
            fresh_so is not None
            and mt in _MONEYLINE_TYPES
            and sharp_weight_used is not None
            and r.get("sharp_true_prob") is not None
        ):
            fresh_sharp_prob = yes_aligned_close_prob(
                yes_team=r["yes_team"],
                outcome_a_label=fresh_so.outcome_a_label,
                outcome_b_label=fresh_so.outcome_b_label,
                true_prob_a=fresh_so.true_prob_a,
                true_prob_b=fresh_so.true_prob_b,
            )
            if fresh_sharp_prob is not None:
                blended_prob = reblend_with_fresh_sharp(
                    blended_prob=blended_prob,
                    scan_sharp_prob=r["sharp_true_prob"],
                    fresh_sharp_prob=fresh_sharp_prob,
                    sharp_weight=sharp_weight_used,
                )
                pin_is_live = True

        rc = recompute_at_price(
            blended_prob=blended_prob,
            price=price,
            base_kelly=kelly,
            max_kelly=max_kelly,
            bankroll=bankroll,
            min_ev=min_ev,
            min_prob=min_prob,
            venue=(r.get("venue") or "kalshi") if fees_in_pricing else None,
        )

        bets.append({
            **r,
            "blended_true_prob": blended_prob,
            "pin_is_live": pin_is_live,
            "scan_ask": scan_ask,
            "live_ask": price,
            "price_is_live": live and live_ask is not None,
            "live_ev": rc["ev"] if rc["ev"] is not None else r["ev_pct"],
            "live_ev_real": rc["ev"],
            "event_start": event_start,
            "is_live": rc["is_live"],
            "kelly_fraction_used": rc["kelly_fraction"],
            "stake": rc["stake"],
            "live_effective": live,
        })

    # GAP 2 at placement: scale each selected venue's summed live stakes to fit
    # its deployable cash (no-op when cash_by_venue is None/empty — manual
    # bankroll, or the background pruner which never places).
    apply_cash_cap_to_bets(bets, cash_by_venue)
    return bets
