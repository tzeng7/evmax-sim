"""Core dataclasses for the golf module.

Everything here is plain ``@dataclass`` (matching ``evmax.ev.devig`` style) so
the module has no Pydantic / ORM coupling to the pairwise scan pipeline. Names
are the join key across layers: a *canonical* golfer name (see
``evmax.golf.names.canonical``) links an SG record, a skill estimate, a
simulator row, and a venue market for the same player.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GolfMarketType(str, Enum):
    """The golf market families we price off one simulated score distribution.

    Values are lowercase snake so they read cleanly in CLI tables and can key
    dicts. ``win`` is the deep/liquid outright; ``top_*`` and ``make_cut`` are
    per-golfer YES contracts over the field; ``rN_leader`` are the PolyUS /
    Kalshi end-of-round-leader markets.
    """

    win = "win"
    top_5 = "top_5"
    top_10 = "top_10"
    top_20 = "top_20"
    make_cut = "make_cut"
    r1_leader = "r1_leader"
    r2_leader = "r2_leader"
    r3_leader = "r3_leader"

    @property
    def top_n(self) -> Optional[int]:
        """The N for a top-N market, else None."""
        return {"top_5": 5, "top_10": 10, "top_20": 20}.get(self.value)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


@dataclass
class SGRecord:
    """One player's strokes-gained result at one tournament (per-EVENT grain).

    Per-round shot-level SG is not available for free, so the finest free
    granularity is one row per player per event. Any SG component may be
    ``None`` (e.g. an ESPN score-proxy record carries only ``sg_total``).
    ``sg_total`` is expressed as strokes gained on the field PER ROUND
    (positive = better than the field average), which is what the skill model
    and simulator consume.
    """

    player: str  # canonical name
    event: str
    date: str  # ISO YYYY-MM-DD (tournament end / scoring date)
    sg_total: Optional[float] = None
    sg_ott: Optional[float] = None
    sg_app: Optional[float] = None
    sg_arg: Optional[float] = None
    sg_putt: Optional[float] = None
    made_cut: Optional[bool] = None
    finish_pos: Optional[int] = None
    n_rounds: Optional[int] = None
    source: str = "unknown"  # "pgatour" | "espn_proxy" | "asa" | "fixture"

    @property
    def has_categories(self) -> bool:
        return all(
            v is not None
            for v in (self.sg_ott, self.sg_app, self.sg_arg, self.sg_putt)
        )


# ---------------------------------------------------------------------------
# Skill layer
# ---------------------------------------------------------------------------


@dataclass
class PlayerSkill:
    """A player's projected form entering a tournament.

    ``skill_sg`` is the projected strokes-gained-per-round vs the field
    (positive = better). ``round_sd`` is the player-specific per-round score
    standard deviation (shrunk toward the field average, ~2.8). ``mean_score``
    is the simulator input: expected strokes-to-par per round, lower is better
    — derived as ``-skill_sg`` so the field mean sits at 0.
    """

    player: str
    skill_sg: float
    round_sd: float
    n_events: int
    confidence: float
    components: dict[str, float] = field(default_factory=dict)  # ott/app/arg/putt

    @property
    def mean_score(self) -> float:
        return -self.skill_sg


# ---------------------------------------------------------------------------
# Simulation layer
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    """Model probabilities for one golfer, produced by the field simulator."""

    player: str
    win: float
    top_5: float
    top_10: float
    top_20: float
    make_cut: float
    mean_finish: float  # expected finishing position (1 = best)

    def prob_for(self, mt: GolfMarketType) -> Optional[float]:
        """Model probability for a given market type (round-leader → None)."""
        return {
            GolfMarketType.win: self.win,
            GolfMarketType.top_5: self.top_5,
            GolfMarketType.top_10: self.top_10,
            GolfMarketType.top_20: self.top_20,
            GolfMarketType.make_cut: self.make_cut,
        }.get(mt)


# ---------------------------------------------------------------------------
# Market layer
# ---------------------------------------------------------------------------


@dataclass
class GolferMarket:
    """A single per-golfer YES contract at a venue, before devig.

    ``implied_raw`` is the venue's YES ask converted to a probability (0-1),
    i.e. the vigged market price. ``None`` when the contract has no resting
    ask. ``golfer`` is canonicalised to match the model field.
    """

    venue: str  # "kalshi" | "polymarket_us"
    tournament: str
    market_type: GolfMarketType
    golfer: str  # canonical name
    implied_raw: Optional[float]  # YES-ask implied prob (0-1, vigged) — pay price
    ticker: str  # venue ticker / market slug (for tracing / capture)
    yes_ask: Optional[float] = None  # native ask (cents for Kalshi, 0-1 for PolyUS)
    yes_bid: Optional[float] = None  # native YES bid (0-1)
    implied_no_raw: Optional[float] = None  # NO-ask implied prob (0-1, vigged)
    volume: Optional[float] = None

    @property
    def has_both_sides(self) -> bool:
        return self.implied_raw is not None and self.implied_no_raw is not None

    @property
    def spread(self) -> Optional[float]:
        if self.yes_ask is None or self.yes_bid is None:
            return None
        return self.yes_ask - self.yes_bid

    @property
    def mid(self) -> Optional[float]:
        if self.yes_ask is None or self.yes_bid is None:
            return self.implied_raw
        return (self.yes_ask + self.yes_bid) / 2.0

    def robust_implied(self, max_spread: float = 0.05) -> Optional[float]:
        """A liquidity-robust implied YES probability for field normalisation.

        Tight two-sided quote → the midpoint. Wide quote (illiquid longshot with
        a placeholder ask) → the BID, which reflects real demand; the ask is
        non-informative there and would otherwise poison a sum-to-1 field devig.
        """
        if self.yes_bid is not None and self.spread is not None and self.spread > max_spread:
            return self.yes_bid
        m = self.mid
        return m if m is not None else self.implied_raw


@dataclass
class FieldPrices:
    """The full field's market prices for one (venue, tournament, market_type).

    ``devigged`` is populated by ``evmax.golf.devig`` — the raw per-golfer
    implied probs (which sum to >1 across the field due to the per-contract
    overround) are power-devigged to a normalised distribution.
    """

    venue: str
    tournament: str
    market_type: GolfMarketType
    markets: list[GolferMarket]
    devigged: dict[str, float] = field(default_factory=dict)  # canonical name -> prob
    overround: Optional[float] = None

    def priced_markets(self) -> list[GolferMarket]:
        return [m for m in self.markets if m.implied_raw is not None]


# ---------------------------------------------------------------------------
# Edge layer
# ---------------------------------------------------------------------------


@dataclass
class GolfEdge:
    """One model-vs-market comparison — the observation the whole module exists
    to surface. ``edge`` is model_prob - market_prob (devigged)."""

    venue: str
    tournament: str
    market_type: GolfMarketType
    golfer: str
    model_prob: float
    market_prob: float  # devigged
    market_raw: Optional[float]  # vigged ask-implied (what you'd actually pay)
    ticker: str

    @property
    def edge(self) -> float:
        return self.model_prob - self.market_prob

    @property
    def edge_vs_raw(self) -> Optional[float]:
        """Edge against the price you actually pay (after vig). The honest
        number for bet selection — positive here is a real +EV signal."""
        if self.market_raw is None:
            return None
        return self.model_prob - self.market_raw
