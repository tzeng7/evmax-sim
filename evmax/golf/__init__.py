"""Standalone golf modelling + market-pricing module.

Golf does not fit evmax's core scan pipeline, which is hard-wired to 2-way /
3-way head-to-head events end to end (pairwise matching keys, a/b/draw odds
and blend shapes, the ``predict_pair`` model primitive). Golf's flagship
markets are *fields*: a 100-150 player tournament outright (one YES contract
per golfer), plus top-5/10/20, make-cut, and round-leader markets — all priced
off the SAME per-player score distribution. So golf lives here as a
self-contained package that is deliberately NOT registered in
``data/categories.yaml`` or ``SECTOR_SERIES_MAP`` (doing so would crash the
import-time ``validate_registry()`` and destabilise the pairwise pipeline).

Pipeline (mirrors the design doc, adapted to the free-data ceiling):

    data (ESPN field/scores + PGA Tour SG)   evmax.golf.data
      -> skill model (decayed per-EVENT SG,   evmax.golf.skill
         player-specific variance)
      -> Monte-Carlo field simulator          evmax.golf.simulator
         (win / top-N / make-cut)
      -> venue field prices (Kalshi + PolyUS)  evmax.golf.venues
      -> field devig                           evmax.golf.devig
      -> edge = model_prob - market_prob       evmax.golf.pricing

The simulator is the data-agnostic heart: given each player's mean and
standard deviation it prices every market at once. The realistic FREE data
ceiling is per-*tournament* strokes-gained (PGA Tour public GraphQL +
historical ASA set), NOT the per-round shot-level SG that paid DataGolf
carries — the skill model is built around per-event granularity accordingly,
with an ESPN score-derived SG-total proxy as the no-SG fallback so the whole
chain runs offline / in tests without a network or paid key.
"""

from __future__ import annotations

from evmax.golf.models import (
    GolfEdge,
    GolfMarketType,
    GolferMarket,
    FieldPrices,
    PlayerSkill,
    SGRecord,
    SimResult,
)

__all__ = [
    "GolfEdge",
    "GolfMarketType",
    "GolferMarket",
    "FieldPrices",
    "PlayerSkill",
    "SGRecord",
    "SimResult",
]
