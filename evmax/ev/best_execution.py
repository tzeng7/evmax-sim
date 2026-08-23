"""Best-execution collapse for the display / actionable play list (GAP 3).

The same game outcome quoted on both Kalshi and Polymarket US produces two
EVGaps by design — two independent books, kept apart for CLV / shadow tracking.
A bettor, though, buys the ONE better book, not both, so showing two rows for a
single bet invites an accidental double (the exact failure the venue-siloed
workflow risks). This module collapses same-bet dual-venue legs to one
actionable row on the best-execution venue, recording the alternative venue's
price and EV on the survivor so the user can still line-shop from one row.

SIDE-AWARE, unlike the portfolio-ledger collapse
(:func:`evmax.portfolios.select_best_execution_plays`): a portfolio takes at
most one position per game outcome and is side-agnostic, but the DISPLAY must
keep genuinely different bets visible. Team-A-YES on one venue and Team-B-YES on
the other is a price DISLOCATION, not a duplicate — both stay. Over and under of
one total stay. Only the SAME outcome AND the SAME side on two venues collapses.

VIEW-LAYER ONLY: both venue rows are still persisted independently — ``log_gaps``
writes them before this runs — so CLV and shadow tracking always see both books.
This function never touches the database and never mutates its inputs.
"""

from __future__ import annotations

import dataclasses

from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.portfolios import portfolio_outcome_key


def _rank(g: EVGap) -> tuple[bool, float]:
    """Best-execution ordering key.

    A live-sized leg (kelly_fraction > 0) beats a shadow-bound zero-Kelly leg —
    the venue firewall zeroes an un-promoted venue's Kelly, and a bettor
    executes the book they can actually stake. Among equally-sized legs, higher
    EV (the cheaper net-of-fee price) wins.
    """
    return ((g.kelly_fraction or 0.0) > 0.0, g.ev_pct or 0.0)


def _display_key(g: EVGap) -> tuple:
    """Identity of one specific actionable bet.

    The venue-agnostic market key (game + market_type + |line|) PLUS the side
    (``yes_team``). Over/under and each moneyline team therefore stay distinct;
    only the identical bet quoted on two venues shares this key.
    """
    return portfolio_outcome_key(g.event_id, g.market_type, g.line) + (
        (g.yes_team or "").lower().strip(),
    )


def collapse_best_execution(gaps: list[EVGap]) -> list[EVGap]:
    """Collapse same-bet dual-venue legs to one best-execution row, annotated.

    For each ``(outcome, side)`` group, keep the leg a bettor would actually
    execute (see :func:`_rank`) and, when a genuine cross-venue alternative
    exists for that same bet, record the alternative's venue / YES price / EV on
    the survivor (``alt_venue`` / ``alt_venue_price`` / ``alt_venue_ev_pct``).
    Groups with a single leg pass through untouched. Survivor order follows
    first-appearance order of the input. Inputs are not mutated — annotated
    survivors are produced via :func:`dataclasses.replace`.
    """
    groups: dict[tuple, list[EVGap]] = {}
    order: list[tuple] = []
    for g in gaps:
        k = _display_key(g)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(g)

    out: list[EVGap] = []
    for k in order:
        legs = groups[k]
        if len(legs) == 1:
            out.append(legs[0])
            continue
        ranked = sorted(legs, key=_rank, reverse=True)
        winner = ranked[0]
        # Annotate the best leg on a DIFFERENT venue (a real line-shop
        # alternative for the same bet), if any. Same-venue duplicates — which
        # should not occur, one venue quotes a market once — collapse silently.
        alt = next((g for g in ranked[1:] if g.venue != winner.venue), None)
        if alt is not None:
            winner = dataclasses.replace(
                winner,
                alt_venue=alt.venue,
                alt_venue_price=alt.kalshi_yes_price,
                alt_venue_ev_pct=alt.ev_pct,
            )
        out.append(winner)
    return out


def apply_venue_cash_cap(
    gaps: list[EVGap],
    bankroll: float,
    cash_by_venue: dict[str, float],
) -> list[EVGap]:
    """Scale each venue's summed stakes down to fit its deployable cash (GAP 2).

    The Kelly base is TOTAL WEALTH, but a new bet is funded from CASH — so a
    venue whose wealth is mostly locked in open positions can be told to stake
    more than it can actually fund. This caps that: for each venue, if the sum
    of its legs' stakes (``kelly_fraction * bankroll``) exceeds its deployable
    cash, every leg on that venue is scaled by ``cash / requested`` so the total
    fits.

    MUST run on the COLLAPSED set (one venue per bet, see
    :func:`collapse_best_execution`) — otherwise a bet quoted on both venues
    would count its stake against both venues' cash, double-counting a position
    you'd only take once.

    Fail-soft: a venue absent from ``cash_by_venue`` (cash unknown, or not
    selected) is left uncapped; an empty map or ``bankroll <= 0`` is a no-op.
    Does not mutate inputs.
    """
    if not cash_by_venue or not bankroll or bankroll <= 0:
        return list(gaps)

    requested: dict[str, float] = {}
    for g in gaps:
        kf = g.kelly_fraction or 0.0
        if kf <= 0:
            continue
        v = getattr(g, "venue", "kalshi") or "kalshi"
        requested[v] = requested.get(v, 0.0) + kf * bankroll

    scale: dict[str, float] = {}
    for v, req in requested.items():
        cash = cash_by_venue.get(v)
        if cash is None or req <= 0 or req <= cash:
            continue
        scale[v] = cash / req

    if not scale:
        return list(gaps)

    out: list[EVGap] = []
    for g in gaps:
        v = getattr(g, "venue", "kalshi") or "kalshi"
        f = scale.get(v)
        if f is not None and (g.kelly_fraction or 0.0) > 0:
            out.append(dataclasses.replace(
                g, kelly_fraction=round(g.kelly_fraction * f, 4)))
        else:
            out.append(g)
    return out
