"""Cross-venue arbitrage detection between Kalshi and Polymarket US.

An arbitrage exists when the cheapest way to own EVERY outcome of one event
market costs less than the $1.00 it pays out — after venue taker fees
(:mod:`evmax.fees`). The 2026-07-09 baseline sweep showed the venues run
~1pp of combined vig with no steady-state net arbs, so the realistic quarry
is transient dislocations (e.g. Kalshi's T-24h market-maker placeholder
quotes before Polymarket US tightens).

Pairing rules (each one guards a real failure mode):

- Outcomes are grouped by ``(sector, frozenset(canonical teams), ET date)``.
  The date MUST be the exact US/Eastern calendar date: MLB series play
  consecutive nights, and a looser tolerance pairs tonight's in-play Kalshi
  market against tomorrow's pre-game Polymarket US market for the same team
  pair, manufacturing fake double-digit "edges". Kalshi's noon-UTC ticker
  anchor converts to the correct ET day; Polymarket US carries the true
  start time.
- Team names are canonicalized through the sector alias maps
  (``NameNormalizer``) so both venues speak the same names.
- Baskets are built per market instrument: 2-way moneyline (YES on one
  side / NO on the other, either venue), 3-way soccer (cheapest YES per
  outcome incl. draw), spreads keyed by (team, line) with the mirrored
  ``(opponent, -line)`` instrument accepted, totals keyed by line.
- Prices are the venue ASK for that side (`yes_price`/`no_price` on
  ``PredictionMarket``), i.e. what a crossing (taker) order pays now.
  Taker fees are charged on every leg; resting maker orders would pay less
  (Kalshi) or earn a rebate (Polymarket US) but lose the fill guarantee.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from evmax.fees import venue_fee_prob
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType, PredictionMarket

_ET = ZoneInfo("America/New_York")

# Outcome labels that mean "the draw" across venues (Kalshi TIE / PolyUS draw)
_DRAW_LABELS = {"draw", "tie"}

# Sectors whose regulation moneyline is ALWAYS 3-way. The draw outcome must
# be a required basket leg even when no draw market parsed in this fetch —
# otherwise a missing draw quote degrades the event to a fake 2-way "arb"
# (YES both teams, draw uncovered).
_THREE_WAY_SECTORS = {"soccer", "worldcup"}


class ArbLeg(BaseModel):
    """One order in an arb basket: buy `side` of `market_id` at `ask`."""

    venue: str                 # "kalshi" | "polymarket_us"
    market_id: str
    side: str                  # "yes" | "no"
    outcome: str               # canonical outcome this leg pays on
    ask: float                 # price paid per contract
    fee: float                 # taker fee per contract (probability units)
    volume_usd: float = 0.0


class ArbOpportunity(BaseModel):
    """A complete-outcome basket and its cost."""

    sector: str
    event_title: str
    event_date: Optional[datetime]  # true start when known (PolyUS side)
    market_desc: str                # e.g. "moneyline", "spread astros -1.5"
    legs: list[ArbLeg]
    gross_cost: float               # sum of asks
    net_cost: float                 # gross + taker fees
    in_play: bool = False

    @property
    def net_edge(self) -> float:
        """Guaranteed profit per $1 basket after fees (negative = no arb)."""
        return 1.0 - self.net_cost

    @property
    def cross_venue(self) -> bool:
        return len({leg.venue for leg in self.legs}) > 1


def _venue_of(market: PredictionMarket) -> str:
    return market.source.value


def _et_date(market: PredictionMarket) -> Optional[date]:
    if market.event_date is None:
        return None
    return market.event_date.astimezone(_ET).date()


def _canon(normalizers: dict[str, NameNormalizer], sector: str, name: str) -> str:
    nn = normalizers.setdefault(sector, NameNormalizer(sector))
    out = nn.normalize(name)
    return (out or name).strip().lower()


def _best_leg(candidates: list[ArbLeg]) -> Optional[ArbLeg]:
    """Cheapest leg by fee-inclusive cost."""
    if not candidates:
        return None
    return min(candidates, key=lambda leg: leg.ask + leg.fee)


def _leg(market: PredictionMarket, side: str, outcome: str) -> ArbLeg:
    ask = market.yes_price if side == "yes" else market.no_price
    venue = _venue_of(market)
    return ArbLeg(
        venue=venue,
        market_id=market.id,
        side=side,
        outcome=outcome,
        ask=ask,
        fee=venue_fee_prob(venue, ask),
        volume_usd=market.volume_usd,
    )


def _basket(
    sector: str,
    title: str,
    start: Optional[datetime],
    market_desc: str,
    legs_by_outcome: dict[str, list[ArbLeg]],
    now: datetime,
) -> Optional[ArbOpportunity]:
    """Cheapest complete basket over the outcome space, or None if any
    outcome has no purchasable leg."""
    if len(legs_by_outcome) < 2:
        return None
    chosen: list[ArbLeg] = []
    for outcome, candidates in legs_by_outcome.items():
        best = _best_leg(candidates)
        if best is None:
            return None
        chosen.append(best)
    gross = sum(leg.ask for leg in chosen)
    net = sum(leg.ask + leg.fee for leg in chosen)
    return ArbOpportunity(
        sector=sector,
        event_title=title,
        event_date=start,
        market_desc=market_desc,
        legs=chosen,
        gross_cost=gross,
        net_cost=net,
        in_play=bool(start and start <= now),
    )


def find_arbs(
    markets: Iterable[PredictionMarket],
    max_net_cost: float = 1.02,
    include_in_play: bool = False,
    now: Optional[datetime] = None,
) -> list[ArbOpportunity]:
    """Find complete-outcome baskets costing at most ``max_net_cost`` after
    fees, across an already-fetched market pool (any mix of venues).

    Returns opportunities sorted by net edge, best first. ``max_net_cost``
    above 1.0 keeps near-misses visible (default surfaces anything within
    2pp of breakeven); pass 1.0 for true arbs only.
    """
    now = now or datetime.now(timezone.utc)
    normalizers: dict[str, NameNormalizer] = {}

    # ---- group by (sector, teamset, ET date) --------------------------------
    groups: dict[tuple, list[PredictionMarket]] = {}
    for m in markets:
        if m.market_type not in (MarketType.moneyline, MarketType.spread, MarketType.total):
            continue
        d = _et_date(m)
        if d is None or not m.team_home or not m.team_away:
            continue
        home = _canon(normalizers, m.sector, m.team_home)
        away = _canon(normalizers, m.sector, m.team_away)
        if not home or not away or home == away:
            continue
        groups.setdefault((m.sector, frozenset((home, away)), d), []).append(m)

    out: list[ArbOpportunity] = []
    for (sector, teams, _d), group in groups.items():
        title = max((m.title for m in group), key=len, default="")
        # True start time only exists on the PolyUS side (Kalshi's ticker
        # anchor is a date, not a time).
        starts = [
            m.event_date
            for m in group
            if m.event_date and _venue_of(m) == "polymarket_us"
        ]
        start = min(starts) if starts else None

        def yes_label(m: PredictionMarket) -> str:
            raw = (m.yes_team or "").lower()
            if raw in _DRAW_LABELS:
                return "draw"
            if raw in ("over", "under"):
                return raw
            return _canon(normalizers, sector, raw)

        # ---- moneyline ------------------------------------------------------
        mls = [m for m in group if m.market_type == MarketType.moneyline]
        if mls:
            has_draw = sector in _THREE_WAY_SECTORS or any(
                yes_label(m) == "draw" for m in mls
            )
            outcomes = set(teams) | ({"draw"} if has_draw else set())
            legs: dict[str, list[ArbLeg]] = {o: [] for o in outcomes}
            for m in mls:
                yl = yes_label(m)
                if yl not in outcomes:
                    continue
                legs[yl].append(_leg(m, "yes", yl))
                if not has_draw:
                    # 2-way: the NO side of this market pays on the opponent
                    (other,) = outcomes - {yl}
                    legs[other].append(_leg(m, "no", other))
            basket = _basket(sector, title, start, "moneyline", legs, now)
            if basket:
                out.append(basket)

        # ---- spreads, keyed by (team, line) ---------------------------------
        spreads: dict[tuple[str, float], list[PredictionMarket]] = {}
        for m in group:
            if m.market_type == MarketType.spread and m.line is not None:
                spreads.setdefault((yes_label(m), round(m.line, 1)), []).append(m)
        for (team, line), inst in spreads.items():
            covers = f"{team} {line:+.1f}"
            fails = f"not {covers}"
            legs = {covers: [], fails: []}
            for m in inst:
                legs[covers].append(_leg(m, "yes", covers))
                legs[fails].append(_leg(m, "no", fails))
            # mirrored instrument: opponent at -line covers ⇔ this side fails
            mirror = spreads.get((next(iter(teams - {team}), ""), round(-line, 1)))
            for m in mirror or []:
                legs[fails].append(_leg(m, "yes", fails))
            basket = _basket(sector, title, start, f"spread {covers}", legs, now)
            if basket:
                out.append(basket)

        # ---- totals, keyed by line ------------------------------------------
        totals: dict[float, list[PredictionMarket]] = {}
        for m in group:
            if m.market_type == MarketType.total and m.line is not None:
                totals.setdefault(round(m.line, 1), []).append(m)
        for line, inst in totals.items():
            over, under = f"over {line:g}", f"under {line:g}"
            legs = {over: [], under: []}
            for m in inst:
                legs[over].append(_leg(m, "yes", over))
                legs[under].append(_leg(m, "no", under))
            basket = _basket(sector, title, start, f"total {line:g}", legs, now)
            if basket:
                out.append(basket)

    out = [
        a for a in out
        if a.net_cost <= max_net_cost and (include_in_play or not a.in_play)
    ]
    out.sort(key=lambda a: a.net_cost)
    return out


async def fetch_arb_markets(sectors: list[str]) -> dict[str, list[PredictionMarket]]:
    """Fetch the merged Kalshi + Polymarket US market pool per sector."""
    import asyncio

    from evmax.clients.kalshi import KalshiClient
    from evmax.clients.polymarket_us import PolymarketUSClient

    async def _one(sector: str) -> tuple[str, list[PredictionMarket]]:
        async with KalshiClient() as k, PolymarketUSClient() as p:
            kal, pus = await asyncio.gather(
                k.get_markets(sector), p.get_markets(sector),
                return_exceptions=True,
            )
        pool: list[PredictionMarket] = []
        for batch in (kal, pus):
            if isinstance(batch, BaseException):
                continue
            pool.extend(batch)
        return sector, pool

    results = await asyncio.gather(*(_one(s) for s in sectors))
    return dict(results)
