"""MLB player-prop projection model.

Produces a per-stat expected value from point-in-time player rates + the
day's matchup, then wraps it in the same distribution family the sharp-anchor
pricer uses (:mod:`evmax.ev.prop_pricing`) so the model probability and the
Pinnacle-anchor probability are read off comparable surfaces and can be blended.

Why a model at all, given props are anchor-priced today:
  - Pinnacle posts MLB anchors for **strikeouts / pitching_outs / total_bases /
    home_runs only**. Hits / hits_runs_rbis / rbis have NO sharp line, so the
    projection is the *only* price for those (optionally bridged off the
    Pinnacle total-bases anchor — see project_prob's ``tb_anchor_mean``).
  - For the anchored stats the projection is blended with the sharp prob so the
    final number can express a matchup view the stale Kalshi line misses.

Projections are deliberately transparent (rate × opponent-adjustment ×
park × expected-PA), not a black box — every term is inspectable and every
league constant is one place. Dispersion (the variance of the per-game
outcome) lives in prop_pricing's per-stat tables, shared with anchor pricing.

All functions are pure and network-free; the data layer
(evmax/clients/baseball_props_cache.py) fills the profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evmax.ev.prop_pricing import model_prob_at_or_above


# League baselines (2026 MLB, approximate — used only as denominators for
# opponent-adjustment ratios, so small errors wash out). Refresh annually.
LEAGUE_K_RATE_PER_PA = 0.22       # team strikeouts / plate appearances
LEAGUE_HITS_PER_BF = 0.232        # pitcher hits allowed / batters faced
LEAGUE_HR_PER_BF = 0.029          # pitcher HR allowed / batters faced
DEFAULT_HITTER_PA = 4.1           # plate appearances per game (lineup-avg)

# Opponent multipliers are clamped to this band so one extreme matchup (or a
# tiny-sample opposing starter) can't blow up a projection.
_MATCHUP_CLAMP = (0.70, 1.40)

# Static park factors (run/HR environment), keyed by MLB team id of the HOME
# club. 1.0 = neutral; >1 inflates. Only the notable extremes are listed; the
# rest default to neutral. Coors (115) is the dominant outlier.
_PARK_HR_FACTOR: dict[int, float] = {
    115: 1.35,  # Coors Field (Rockies)
    143: 1.12,  # Citizens Bank (Phillies)
    147: 1.10,  # Yankee Stadium
    112: 1.08,  # Wrigley (Cubs) — wind-dependent, season-avg
    137: 0.90,  # Oracle Park (Giants)
    145: 0.92,  # Guaranteed Rate (White Sox) — pitcher-leaning recent
    119: 0.95,  # Dodger Stadium
    136: 0.92,  # T-Mobile (Mariners)
}
_PARK_HIT_FACTOR: dict[int, float] = {
    115: 1.15,  # Coors
    137: 0.95,  # Oracle
    136: 0.95,  # T-Mobile
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class HitterProfile:
    """Point-in-time season-to-date (or rolling) batting totals."""
    plate_appearances: float
    at_bats: float
    hits: float
    doubles: float
    triples: float
    home_runs: float
    rbi: float
    runs: float
    strike_outs: float
    total_bases: float
    games: float


@dataclass(frozen=True)
class PitcherProfile:
    """Point-in-time season-to-date (or rolling) pitching totals (as a starter)."""
    batters_faced: float
    outs: float
    strike_outs: float
    home_runs: float          # HR allowed
    hits: float               # hits allowed
    games_started: float
    innings_pitched: float


@dataclass(frozen=True)
class MatchupContext:
    """The day's matchup adjustments for one prop."""
    opp_team_k_rate: Optional[float] = None      # opposing lineup SO/PA (pitcher K)
    opp_starter: Optional[PitcherProfile] = None  # opposing starter (hitter props)
    home_team_id: Optional[int] = None            # for park factors
    expected_pa: float = DEFAULT_HITTER_PA


def _opp_hit_suppression(opp: Optional[PitcherProfile]) -> float:
    if opp is None or opp.batters_faced <= 0:
        return 1.0
    rate = (opp.hits / opp.batters_faced) / LEAGUE_HITS_PER_BF
    return _clamp(rate, *_MATCHUP_CLAMP)


def _opp_hr_suppression(opp: Optional[PitcherProfile]) -> float:
    if opp is None or opp.batters_faced <= 0:
        return 1.0
    rate = (opp.home_runs / opp.batters_faced) / LEAGUE_HR_PER_BF
    return _clamp(rate, *_MATCHUP_CLAMP)


# --- Pitcher props -------------------------------------------------------

def project_pitcher_strikeouts(p: PitcherProfile, ctx: MatchupContext) -> Optional[float]:
    """Expected strikeouts in the start: K/BF × BF-per-start × opponent K-mult."""
    if p.batters_faced <= 0 or p.games_started <= 0:
        return None
    k_per_bf = p.strike_outs / p.batters_faced
    bf_per_start = p.batters_faced / p.games_started
    mult = 1.0
    if ctx.opp_team_k_rate:
        mult = _clamp(ctx.opp_team_k_rate / LEAGUE_K_RATE_PER_PA, *_MATCHUP_CLAMP)
    return k_per_bf * bf_per_start * mult


def project_pitcher_outs(p: PitcherProfile, ctx: MatchupContext) -> Optional[float]:
    """Expected outs recorded = season outs per start (workload, manager-driven)."""
    if p.games_started <= 0:
        return None
    return p.outs / p.games_started


# --- Hitter props --------------------------------------------------------

def project_hitter_total_bases(h: HitterProfile, ctx: MatchupContext) -> Optional[float]:
    if h.plate_appearances <= 0:
        return None
    tb_per_pa = h.total_bases / h.plate_appearances
    park = _PARK_HIT_FACTOR.get(ctx.home_team_id or -1, 1.0)
    return tb_per_pa * _opp_hit_suppression(ctx.opp_starter) * park * ctx.expected_pa


def project_hitter_hits(h: HitterProfile, ctx: MatchupContext) -> Optional[float]:
    if h.plate_appearances <= 0:
        return None
    h_per_pa = h.hits / h.plate_appearances
    return h_per_pa * _opp_hit_suppression(ctx.opp_starter) * ctx.expected_pa


def project_hitter_home_runs(h: HitterProfile, ctx: MatchupContext) -> Optional[float]:
    if h.plate_appearances <= 0:
        return None
    hr_per_pa = h.home_runs / h.plate_appearances
    park = _PARK_HR_FACTOR.get(ctx.home_team_id or -1, 1.0)
    return hr_per_pa * _opp_hr_suppression(ctx.opp_starter) * park * ctx.expected_pa


def project_hitter_rbi(h: HitterProfile, ctx: MatchupContext) -> Optional[float]:
    if h.games <= 0:
        return None
    rbi_per_game = h.rbi / h.games
    return rbi_per_game * _opp_hit_suppression(ctx.opp_starter)


def project_hitter_hits_runs_rbis(h: HitterProfile, ctx: MatchupContext) -> Optional[float]:
    if h.games <= 0:
        return None
    per_game = (h.hits + h.runs + h.rbi) / h.games
    return per_game * _opp_hit_suppression(ctx.opp_starter)


_HITTER_PROJECTORS = {
    "total_bases": project_hitter_total_bases,
    "hits": project_hitter_hits,
    "home_runs": project_hitter_home_runs,
    "rbis": project_hitter_rbi,
    "hits_runs_rbis": project_hitter_hits_runs_rbis,
}
_PITCHER_PROJECTORS = {
    "strikeouts": project_pitcher_strikeouts,
    "pitching_outs": project_pitcher_outs,
}

PITCHER_STATS = frozenset(_PITCHER_PROJECTORS)
HITTER_STATS = frozenset(_HITTER_PROJECTORS)


def project_mean(
    stat_type: str,
    ctx: MatchupContext,
    hitter: Optional[HitterProfile] = None,
    pitcher: Optional[PitcherProfile] = None,
) -> Optional[float]:
    """Projected per-game mean for ``stat_type``. ``None`` if not projectable.

    A ``tb_anchor_mean`` rescale for the unanchored hitter stats is applied by
    :func:`project_prob`, not here — this returns the raw rate projection.
    """
    if stat_type in _PITCHER_PROJECTORS:
        if pitcher is None:
            return None
        return _PITCHER_PROJECTORS[stat_type](pitcher, ctx)
    if stat_type in _HITTER_PROJECTORS:
        if hitter is None:
            return None
        return _HITTER_PROJECTORS[stat_type](hitter, ctx)
    return None


def project_prob(
    stat_type: str,
    threshold: float,
    ctx: MatchupContext,
    hitter: Optional[HitterProfile] = None,
    pitcher: Optional[PitcherProfile] = None,
) -> Optional[float]:
    """Model ``P(stat >= threshold)`` for one Kalshi prop. ``None`` if unpriceable."""
    mean = project_mean(stat_type, ctx, hitter=hitter, pitcher=pitcher)
    if mean is None:
        return None
    return model_prob_at_or_above(stat_type, mean, threshold)
