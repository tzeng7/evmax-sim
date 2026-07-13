"""Edge layer — join model simulation probabilities to devigged market prices.

For every golfer with a market, compare the simulator's model probability for
that market family against the venue's devigged (fair) probability. Two edge
numbers are reported:

  * ``edge``        = model − devigged_market  (disagreement with the fair line)
  * ``edge_vs_raw`` = model − YES ask           (the honest +EV number: what you
                       would actually pay). Bet selection keys on THIS.

Round-leader markets are parsed/devigged (capability) but the base simulator
does not emit a round-leader probability, so they yield no ``GolfEdge`` here —
they are surfaced as market coverage, not model plays. This is the honest state
of the first slice: the module OBSERVES model-vs-market divergence; it does not
assert that divergence is exploitable edge (that needs walk-forward calibration
against historical results, which the ``result_to_sg_proxy`` path is built to
feed later).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from evmax.golf.devig import devig_field
from evmax.golf.models import FieldPrices, GolfEdge, GolfMarketType, SimResult

_log = logging.getLogger(__name__)


@dataclass
class EdgeConfig:
    # Minimum |edge_vs_raw| to flag a play, by market family. Placement markets
    # carry lower variance so a smaller threshold applies (design doc: ~3%
    # placement, ~5% outright).
    outright_threshold: float = 0.05
    placement_threshold: float = 0.03
    # Liquidity guardrails for a flaggable play.
    max_spread: float = 0.06
    min_volume: float = 0.0
    # Ignore sub-penny longshots (the nba_props cheap-longshot lesson).
    min_market_prob: float = 0.01


def _threshold(mt: GolfMarketType, cfg: EdgeConfig) -> float:
    return cfg.outright_threshold if mt is GolfMarketType.win else cfg.placement_threshold


def price_field(
    sims: dict[str, SimResult],
    field: FieldPrices,
    devig: bool = True,
    cut_target: Optional[float] = None,
) -> list[GolfEdge]:
    """Produce a ``GolfEdge`` for every priced golfer the model can price."""
    if devig and not field.devigged:
        devig_field(field, cut_target=cut_target)
    edges: list[GolfEdge] = []
    for m in field.priced_markets():
        sim = sims.get(m.golfer)
        if sim is None:
            continue
        model_prob = sim.prob_for(field.market_type)
        if model_prob is None:  # market family the base sim doesn't emit
            continue
        market_prob = field.devigged.get(m.golfer)
        if market_prob is None:
            continue
        edges.append(
            GolfEdge(
                venue=field.venue,
                tournament=field.tournament,
                market_type=field.market_type,
                golfer=m.golfer,
                model_prob=model_prob,
                market_prob=market_prob,
                market_raw=m.implied_raw,
                ticker=m.ticker,
            )
        )
    return edges


def flag_plays(
    edges: list[GolfEdge],
    fields: list[FieldPrices],
    cfg: Optional[EdgeConfig] = None,
) -> list[GolfEdge]:
    """Filter edges to flaggable plays: positive edge vs the ASK past the
    market-type threshold, liquid enough to actually fill.

    ``fields`` supplies per-golfer spread/volume for the liquidity gate.
    """
    cfg = cfg or EdgeConfig()
    # Index market metadata by (venue, market_type, golfer, ticker).
    meta: dict[tuple, object] = {}
    for f in fields:
        for m in f.markets:
            meta[(m.venue, m.market_type, m.golfer, m.ticker)] = m

    plays: list[GolfEdge] = []
    for e in edges:
        ev = e.edge_vs_raw
        if ev is None or ev < _threshold(e.market_type, cfg):
            continue
        if e.market_prob < cfg.min_market_prob:
            continue
        m = meta.get((e.venue, e.market_type, e.golfer, e.ticker))
        if m is not None:
            if m.spread is not None and m.spread > cfg.max_spread:
                continue
            if cfg.min_volume and (m.volume or 0) < cfg.min_volume:
                continue
        plays.append(e)
    plays.sort(key=lambda e: (e.edge_vs_raw or -1.0), reverse=True)
    return plays
