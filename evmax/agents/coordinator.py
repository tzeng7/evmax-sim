"""AgentCoordinator — orchestrates the full agent pipeline for one sector.

Execution order per cycle:
  1.  KalshiOddsAgent    — fetch live Kalshi markets
  2.  SharpOddsAgent     — fetch devigged Pinnacle odds
  3.  InjuryReportAgent  — fetch ESPN injury data
      (steps 1-3 run concurrently)
  4.  MatchingEngine     — match Kalshi markets → sharp events
  5.  EnsembleModelAgent — run Elo + Form + Poisson in parallel, blend predictions
  6.  Injury adjustment  — shift blended probs based on injury impact
  7.  EVGapAgent         — compute EV gaps with injury-adjusted blended probs
  8.  Publish summary    — "coordinator.cycle.done" with all gaps

All agents share one AgentBus so any subscriber (dashboard, Slack notifier,
simulation engine) can react to results without modifying this class.

Usage:
    coordinator = AgentCoordinator(
        sectors=["nba", "soccer"],
        bankroll=250.0,
        kelly_fraction=0.5,   # half Kelly
    )
    cycle = await coordinator.run_cycle()
    for gap in cycle.ev_gaps:
        print(gap.summary())
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from evmax.agents.base import Agent, AgentBus, AgentMessage, AgentRequest, AgentResponse
from evmax.archiver import DataArchiver
from evmax.agents.odds.kalshi_agent import KalshiOddsAgent
from evmax.agents.odds.polymarket_us_agent import PolymarketUSOddsAgent
from evmax.agents.odds.sharp_agent import SharpOddsAgent
from evmax.agents.odds.ev_gap_agent import EVGapAgent, EVGap
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.agents.models.ensemble_agent import EnsembleModelAgent, BlendedPrediction
from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent
from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent
from evmax.agents.models.tennis_form_agent import TennisFormAgent
from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent
from evmax.agents.models.ufc_rating_agent import UFCRatingAgent
from evmax.agents.models.pitcher_agent import PitcherModelAgent
from evmax.agents.models.soccer_xg_agent import SoccerXgAgent
from evmax.agents.models.efficiency_agent import EfficiencyModelAgent
from evmax.agents.models.nfl_efficiency_agent import NflEfficiencyModelAgent
from evmax.agents.models.nfl_qb_elo_agent import NflQbEloModelAgent
from evmax.agents.models.nhl_xg_agent import NhlXgModelAgent
from evmax.agents.models.ncaab_efficiency_agent import NcaabEfficiencyModelAgent
from evmax.agents.models.ncaaf_efficiency_agent import NcaafEfficiencyModelAgent
from evmax.agents.models.ncaab_possession_sim_agent import NcaabPossessionSimAgent
from evmax.agents.models.ncaaw_efficiency_agent import NcaawEfficiencyModelAgent
from evmax.agents.models.ncaaw_possession_sim_agent import NcaawPossessionSimAgent
from evmax.agents.models.wnba_efficiency_agent import WNBAEfficiencyModelAgent
from evmax.agents.models.shot_quality_agent import ShotQualityAgent
from evmax.agents.models.matchup_agent import MatchupAgent
from evmax.agents.models.possession_sim_agent import PossessionSimAgent
from evmax.agents.models.wnba_possession_sim_agent import WNBAPossessionSimAgent
from evmax.agents.intelligence.injury_agent import InjuryReportAgent, InjuryReport
from evmax.agents.intelligence.playoff_agent import PlayoffAgent, PlayoffSeries
from evmax.agents.intelligence.standings_agent import StandingsAgent, TeamStanding
from evmax.models.market import MarketSource, PredictionMarket, is_prop_event
from evmax.models.odds import SharpOdds, SharpBook
from evmax.matching.engine import MatchingEngine
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

_STEAM_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "steam_cache.json"
_STEAM_THRESHOLD = 0.02  # 2 percentage points

# Weight of the MLB prop projection model when blended with the Pinnacle
# anchor (final = w·model + (1−w)·sharp). Conservative: the model runs on a
# neutral matchup context in live scan today, so it nudges rather than
# overrides the sharp price. Tune up once the walk-forward backtest
# (scripts/backtest_mlb_props.py) shows the model beats the anchor.
BASEBALL_PROP_MODEL_WEIGHT = 0.35

# Hitter stats Pinnacle does NOT post a line for. Priced by the projection
# model, bridged off the player's Total-Bases anchor (same game) so they still
# "play with sharp" — see _baseball_bridged_prop_sharp.
_BASEBALL_BRIDGE_STATS = {"hits", "hits_runs_rbis", "rbis"}


@dataclass
class CycleResult:
    """Results from a single coordinator cycle."""

    sectors_scanned: list[str] = field(default_factory=list)
    markets_fetched: int = 0
    markets_matched: int = 0
    ev_gaps: list[EVGap] = field(default_factory=list)
    blended_predictions: dict[str, BlendedPrediction] = field(default_factory=dict)
    injury_reports: dict[str, dict[str, InjuryReport]] = field(default_factory=dict)  # sector → team → report
    cycle_duration_s: float = 0.0
    exposure_guard_dropped: int = 0
    errors: list[str] = field(default_factory=list)
    bankroll: float = 250.0
    kelly_fraction: float = 0.5
    # Raw prop (SharpOdds, PredictionMarket) pairs for calibration logging
    # when _evaluate_prop is disabled and no EVGaps are created.
    prop_sharp_pairs: list[tuple[SharpOdds, PredictionMarket]] = field(default_factory=list)

    @property
    def top_gaps(self) -> list[EVGap]:
        return sorted(self.ev_gaps, key=lambda g: g.ev_pct, reverse=True)

    def loggable_gaps(self) -> list[EVGap]:
        """Game-market gaps to persist to ``ev_predictions``.

        Player props are routed to ``prop_observations`` separately, so they're
        excluded here. Partial-blend gaps ARE included — ``log_gaps`` demotes
        them to ``mode='shadow'``.
        """
        return [g for g in self.ev_gaps if not is_prop_event(g.event_id)]

    def plays(
        self,
        *,
        require_full_blend: bool = True,
        drop_props: bool = False,
        drop_map_handicap: bool = False,
    ) -> list[EVGap]:
        """The canonical "actionable plays" list, sorted by EV descending.

        Every surface that DISPLAYS or SIZES bets (CLI scan, dashboard,
        portfolio fan-out) must start here rather than from raw
        ``ev_gaps``/``top_gaps`` — that's what keeps each from re-deriving (and
        forgetting) the play-selection rules, the drift this accessor exists to
        prevent.

        - ``require_full_blend``: drop partial-blend gaps (``full_blend=False``).
          These are shadow-bound with Kelly zeroed (see ``run_cycle``) and are
          never plays.
        - ``drop_props``: drop player-prop gaps (``event_id`` carries the prop
          marker).
        - ``drop_map_handicap``: drop esports map-handicap markets not on Kalshi.

        Surface-specific policy (min_prob floors, tiered-EV ramps, date windows,
        placed-exclusion, per-type caps) stays at the call site.
        """
        gaps = self.top_gaps
        if require_full_blend:
            gaps = [g for g in gaps if g.full_blend]
        if drop_props:
            gaps = [g for g in gaps if not is_prop_event(g.event_id)]
        if drop_map_handicap:
            gaps = [g for g in gaps if g.market_type != "map_handicap"]
        return gaps

    def stake_for(self, gap: EVGap) -> float:
        """Dollar stake for a gap given the bankroll and Kelly fraction."""
        return round(self.bankroll * gap.kelly_fraction, 2)

    def print_plays(self, min_ev: float = 0.02, max_plays: int = 30) -> None:
        """Pretty-print +EV plays. Partial-blend gaps are shadow-bound — not plays."""
        gaps = [g for g in self.plays() if g.ev_pct >= min_ev][:max_plays]
        if not gaps:
            print("No +EV plays found.")
            return
        print(f"\n{'='*80}")
        print(f"  +EV PLAYS  |  Bankroll: ${self.bankroll:.0f}  |  Kelly: {self.kelly_fraction:.0%}")
        print(f"{'='*80}")
        for i, g in enumerate(gaps, 1):
            stake = self.stake_for(g)
            print(
                f"{i:>2}. [{g.edge_label:<8}] {g.sector.upper():<6} "
                f"{g.display_label:<26} "
                f"Kalshi={g.kalshi_yes_price:.2f}  TrueP={g.blended_true_prob:.3f}  "
                f"EV={g.ev_pct*100:+.1f}%  Kelly={g.kelly_fraction*100:.2f}%  "
                f"Stake=${stake:.2f}  [{g.model_sources}]"
            )
        print(f"{'='*80}")
        total_stake = sum(self.stake_for(g) for g in gaps)
        print(f"  Total at risk: ${total_stake:.2f} / ${self.bankroll:.0f} ({total_stake/self.bankroll*100:.1f}%)")
        print()


def _load_steam_cache() -> dict[str, float]:
    """Load previous scan's sharp probabilities from disk."""
    try:
        if _STEAM_CACHE_PATH.exists():
            return json.loads(_STEAM_CACHE_PATH.read_text())
    except Exception as exc:
        logger.warning("steam_cache_load_failed", path=str(_STEAM_CACHE_PATH), error=str(exc))
    return {}


def _save_steam_cache(probs: dict[str, float]) -> None:
    try:
        _STEAM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STEAM_CACHE_PATH.write_text(json.dumps(probs))
    except Exception as exc:
        logger.error("steam_cache_save_failed", path=str(_STEAM_CACHE_PATH), error=str(exc))


def _detect_steam(
    sharp_odds: list,
    prev_probs: dict[str, float],
    threshold: float = _STEAM_THRESHOLD,
) -> set[str]:
    """Return event_ids where Pinnacle true_prob_a moved >= threshold since last scan."""
    steam: set[str] = set()
    for so in sharp_odds:
        prev = prev_probs.get(so.event_id)
        if prev is not None and abs(so.true_prob_a - prev) >= threshold:
            steam.add(so.event_id)
            logger.info(
                "steam_move_detected",
                event_id=so.event_id,
                prev=round(prev, 3),
                current=round(so.true_prob_a, 3),
                delta=round(so.true_prob_a - prev, 3),
            )
    return steam


def _base_event(event_id: str) -> str:
    """Strip ::spread / ::total / etc. suffixes so ML + spread + total share a budget.

    Prop events: "nba::2026-03-24::prop::player::stat::line" → group by player
    Game events: "nba::2026-03-24::team_vs_team[::spread|total|...]" → group by matchup
    """
    parts = event_id.split("::")
    if len(parts) > 3 and parts[2] == "prop":
        return "::".join(parts[:4])
    return "::".join(parts[:3])


def _venue_is_live(
    venue: str,
    sector: Optional[str],
    selected_venues: Optional[set[str]],
    settings,
) -> bool:
    """Whether a gap on ``venue`` is a live play.

    A gap is live only when it is BOTH within the per-scan venue selection (if
    any) AND cleared by the PolyUS shadow firewall. The selection only ever
    RESTRICTS: ``selected_venues=None`` means no restriction (all venues,
    firewall still applies), and selecting a venue can never force an
    un-validated PolyUS sector live — the firewall (``polymarket_us_sector_live``)
    still gates within the selection. Kalshi is always firewall-clear.
    """
    if selected_venues is not None and venue not in selected_venues:
        return False
    if venue == "kalshi":
        return True
    if venue == "polymarket_us":
        return settings.polymarket_us_sector_live(sector)
    return False


def _load_placed_exposure() -> dict[str, float]:
    """Sum DOLLARS at risk from already-placed un-resolved live bets, per base event.

    Returns DOLLARS, not Kelly fractions. Fractions from bets sized against
    different bankroll bases are NOT additive: a 5% bet on a $200 venue and a
    5% bet on an $800 venue are $10 and $40, not "10%". Summing the raw
    fractions (the old behaviour) silently breaks the per-game cap the moment
    two bets used different bases — different venues, or a bankroll that changed
    between scans. Dollars are always additive. The caller divides this by the
    current scan bankroll to recover the fraction of THIS bankroll already
    committed to each game.

    Per row, dollars = ``placed_stake`` when recorded (the real amount staked),
    else ``kelly_fraction * bankroll_used`` (the stake sized at scan time). A
    row with neither is skipped (can't be valued) — normal rows always carry
    ``bankroll_used``.

    Includes only:
      - placed = 1     (user confirmed bet placement, not just flagged)
      - voided = 0
      - mode = 'live'  (shadow bets don't touch bankroll)
      - outcome IS NULL (un-resolved — settled bets aren't active exposure)
    """
    try:
        from evmax.agents.cleanup.db import get_connection

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT p.event_id, p.kelly_fraction, p.placed_stake, p.bankroll_used
                FROM ev_predictions p
                LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
                WHERE p.placed = 1
                  AND p.voided = 0
                  AND p.mode = 'live'
                  AND (o.outcome IS NULL OR o.id IS NULL)
                """
            ).fetchall()
    except Exception as e:
        logger.warning("placed_exposure_load_failed", error=str(e))
        return {}

    exposure: dict[str, float] = {}
    for r in rows:
        eid = r["event_id"] if isinstance(r, dict) else r[0]
        kf = r["kelly_fraction"] if isinstance(r, dict) else r[1]
        stake = r["placed_stake"] if isinstance(r, dict) else r[2]
        bankroll_used = r["bankroll_used"] if isinstance(r, dict) else r[3]
        if not eid:
            continue
        if stake is not None and float(stake) > 0:
            dollars = float(stake)
        elif kf is not None and bankroll_used:
            dollars = float(kf) * float(bankroll_used)
        else:
            continue
        base = _base_event(eid)
        exposure[base] = exposure.get(base, 0.0) + dollars
    return exposure


def _apply_joint_kelly(
    gaps: list[EVGap],
    kelly_multiplier: float = 0.25,
    max_kelly_fraction: float = 0.05,
    base_gross_cap: float = 0.08,
    max_gross_cap: float = 0.15,
    rho_margin_total: float = 0.0,
    n_samples: int = 20000,
    prior_exposure: dict[str, float] | None = None,
) -> list[EVGap]:
    """Correlation-aware per-event sizing (replaces _compute-then-guard when
    settings.joint_kelly_enabled).

    Legs sharing a game outcome are sized jointly so contradictory legs (which
    hedge) and same-direction legs (which overlap) are sized on the portfolio,
    not leg-by-leg. The per-event gross cap is variance-scaled inside the
    optimizer (expands toward max_gross_cap only as hedging cuts variance), so
    no separate exposure guard is needed afterward. Single-leg events reduce
    exactly to fractional Kelly. prior_exposure (Kelly already committed on a
    game in earlier scans) shrinks that game's remaining gross budget.
    """
    from evmax.ev.joint_kelly import (
        JointKellyConfig,
        JointLeg,
        infer_axis_and_sign,
        joint_kelly_fractions,
    )

    # Group by the SAME base-event key the exposure guard and prior_exposure
    # use (_base_event: strips ::spread / ::total so ML + spread + total of one
    # game co-group; groups props by player). The old split(PROP_MARKER)[0] key
    # left the market-type suffix on, so ML and spread of one game were sized as
    # two independent events — defeating the whole correlation-aware point (and
    # mis-aligning with prior_exposure, which is _base_event-keyed).
    prior = dict(prior_exposure or {})

    by_event: dict[str, list[EVGap]] = {}
    for g in gaps:
        by_event.setdefault(_base_event(g.event_id or ""), []).append(g)

    guarded: list[EVGap] = []
    for base, group in by_event.items():
        # The first margin leg's team orients the event; the opposing team's
        # margin legs get the opposite sign so they anti-correlate.
        reference_team = None
        for g in group:
            if (g.market_type or "").lower() in ("moneyline", "spread"):
                reference_team = g.yes_team
                break

        legs: list[JointLeg] = []
        for g in group:
            axis, sign = infer_axis_and_sign(g.market_type, g.yes_team, reference_team)
            decimal_odds = (
                1.0 / g.kalshi_yes_price
                if getattr(g, "kalshi_yes_price", 0) and g.kalshi_yes_price > 0
                else 0.0
            )
            # Match compute_kelly's liquidity discount so single legs reduce
            # exactly to the independent fractional-Kelly stake.
            liquidity = max(0.25, 1.0 - (getattr(g, "spread_pct", 0.0) or 0.0) * 5.0)
            legs.append(
                JointLeg(
                    win_prob=g.blended_true_prob,
                    decimal_odds=decimal_odds,
                    axis=axis,
                    sign=sign,
                    confidence=1.0,
                    liquidity=liquidity,
                    label=g.yes_team,
                )
            )

        used = prior.get(base, 0.0)
        config = JointKellyConfig(
            kelly_multiplier=kelly_multiplier,
            max_fraction=max_kelly_fraction,
            base_gross_cap=max(0.0, base_gross_cap - used),
            max_gross_cap=max(0.0, max_gross_cap - used),
            rho_margin_total=rho_margin_total,
            n_samples=n_samples,
        )
        result = joint_kelly_fractions(legs, config)

        for g, frac in zip(group, result.fractions):
            if frac <= 0:
                logger.info("joint_kelly_dropped", base_event=base, yes_team=g.yes_team)
                continue
            guarded.append(dataclasses.replace(g, kelly_fraction=round(frac, 4)))

    return guarded


def _apply_exposure_guard(
    gaps: list[EVGap],
    max_event_exposure: float = 0.08,
    prior_exposure: dict[str, float] | None = None,
    same_side_kelly_discount: float = 0.5,
) -> list[EVGap]:
    """Cap total Kelly allocation per game to max_event_exposure (default 8%).

    Multiple markets on the same underlying game (ML + spread + total) are
    correlated — betting all at full Kelly compounds risk beyond the intended
    exposure. Best plays (by EV) consume budget first; lower-EV plays are
    scaled down or dropped when the cap is hit.

    Same-side discount: when a later gap covers the same yes_team as one
    already in the budget for the event (ρ ≈ 0.8 territory — ML + same-team
    spread, alt-spread stacks), its Kelly is multiplied by
    same_side_kelly_discount before consuming budget. Naive independent
    Kelly across correlated bets puts the bettor past the geometric-growth
    peak; the discount pulls the joint position back toward 1x effective
    Kelly. NO-side spread derivations naturally flip yes_team to the
    opponent label, so an ML-on-favorite + spread-on-underdog pair (like
    Sixers ML + Knicks +8.5) is recognized as opposite-side and gets no
    discount — that pairing already hedges, no correction needed.

    prior_exposure: per-base-event Kelly fractions already committed in
      earlier scans today (placed un-resolved bets). When supplied, the
      remaining budget for each game starts at (cap - prior), so a 4%
      bet placed this morning leaves only 4% for new bets this afternoon.
    """
    event_budget: dict[str, float] = dict(prior_exposure or {})
    event_sides: dict[str, set[str]] = {}
    guarded: list[EVGap] = []

    for base, used in event_budget.items():
        if used > 0:
            logger.debug(
                "exposure_carried_over",
                base_event=base,
                used=round(used, 4),
                cap=max_event_exposure,
                remaining=round(max(0.0, max_event_exposure - used), 4),
            )

    for gap in sorted(gaps, key=lambda g: g.ev_pct, reverse=True):
        base = _base_event(gap.event_id)
        used = event_budget.get(base, 0.0)
        remaining = max_event_exposure - used

        if remaining <= 0.005:  # < 0.5% left — skip
            logger.debug(
                "exposure_guard_dropped",
                event_id=gap.event_id,
                used=round(used, 4),
                cap=max_event_exposure,
            )
            continue

        sides = event_sides.setdefault(base, set())
        side_key = (gap.yes_team or "").lower().strip()
        kelly = gap.kelly_fraction
        if side_key and side_key in sides and same_side_kelly_discount < 1.0:
            discounted = round(kelly * same_side_kelly_discount, 4)
            logger.debug(
                "same_side_kelly_discount",
                event_id=gap.event_id,
                yes_team=side_key,
                original=round(kelly, 4),
                discounted=discounted,
            )
            gap = dataclasses.replace(gap, kelly_fraction=discounted)
            kelly = discounted

        if kelly <= remaining:
            event_budget[base] = used + kelly
            guarded.append(gap)
        else:
            # Scale down to fit remaining budget
            logger.debug(
                "exposure_guard_capped",
                event_id=gap.event_id,
                original=round(kelly, 4),
                capped=round(remaining, 4),
            )
            capped = dataclasses.replace(gap, kelly_fraction=round(remaining, 4))
            event_budget[base] = max_event_exposure
            guarded.append(capped)

        if side_key:
            sides.add(side_key)

    return guarded


class AgentCoordinator:
    """
    Top-level orchestrator for the evmax agent pipeline.

    Args:
        sectors:        List of sector keys to scan (default: all).
        enable_models:  If False, skip model agents and use sharp probs only.
        sharp_weight:   Weight given to Pinnacle in ensemble blend (0–1).
        enable_injuries: If True, fetch ESPN injury reports and adjust probs.
        bankroll:       Current bankroll in USD (default $250).
        kelly_fraction: Kelly multiplier (0.5 = half Kelly, 0.25 = quarter Kelly).
        selected_venues: Restrict live plays to these venues (the dashboard's
                        venue selection). None = no restriction (all venues,
                        subject to the firewall). The selection only ever
                        RESTRICTS — it never overrides the PolyUS shadow
                        firewall, so picking a venue can't force an un-validated
                        sector live.
    """

    def __init__(
        self,
        sectors: Optional[list[str]] = None,
        enable_models: bool = True,
        sharp_weight: float = 0.40,
        enable_injuries: bool = True,
        bankroll: float = 250.0,
        kelly_fraction: float = 0.5,
        respect_season_window: bool = True,
        selected_venues: Optional[list[str]] = None,
    ) -> None:
        from evmax.sectors.registry import ALL_SECTORS
        from evmax.categories import get_category

        requested = [s.lower() for s in (sectors or ALL_SECTORS)]

        # Drop out-of-season sectors so we don't burn Kalshi rate-limit
        # tokens (one call per series prefix) and Pinnacle calls (one
        # /matchups call per sector) on dead sectors. Skipped when
        # `respect_season_window=False`, which the CLI sets whenever the
        # user passes --sectors explicitly so they can still hit dormant
        # sectors for testing.
        self._skipped_off_season: list[str] = []
        if respect_season_window:
            kept: list[str] = []
            for s in requested:
                try:
                    spec = get_category(s)
                except KeyError:
                    # Unknown sectors (e.g. latent registry entries) — pass
                    # through and let downstream raise the usual error.
                    kept.append(s)
                    continue
                if spec.is_in_season():
                    kept.append(s)
                else:
                    self._skipped_off_season.append(s)
            self._sectors = kept
        else:
            self._sectors = requested

        self._enable_models = enable_models
        self._sharp_weight = sharp_weight
        self._enable_injuries = enable_injuries
        self._bankroll = bankroll
        self._kelly_fraction = kelly_fraction
        # Per-scan venue restriction (dashboard venue selection). None = all
        # venues. Normalized to a lowercase set; empty selection is treated as
        # None (no restriction) so a stray empty list can't silently drop
        # every play.
        self._selected_venues: Optional[set[str]] = (
            {v.lower() for v in selected_venues} if selected_venues else None
        )

        self.bus = AgentBus()

        # Odds checker agents
        self.kalshi_agent = KalshiOddsAgent()
        self.polymarket_us_agent = PolymarketUSOddsAgent()
        self.sharp_agent = SharpOddsAgent()
        self.ev_gap_agent = EVGapAgent()

        # Intelligence agents
        self.injury_agent = InjuryReportAgent()
        self.standings_agent = StandingsAgent()
        self.playoff_agent = PlayoffAgent()

        # Statistical model agents
        self.elo_agent = EloModelAgent()
        self.form_agent = FormModelAgent()
        self.poisson_agent = PoissonModelAgent()
        self.tennis_agent = TennisModelAgent()
        self.tennis_serve_agent = TennisServeReturnAgent()
        self.tennis_h2h_agent = TennisH2HAgent()
        self.tennis_trend_agent = TennisRankingTrendAgent()
        self.tennis_form_agent = TennisFormAgent()
        self.tennis_advanced_agent = TennisAdvancedStatsAgent()
        self.ufc_rating_agent = UFCRatingAgent()
        self.pitcher_agent = PitcherModelAgent()
        self.soccer_xg_agent = SoccerXgAgent()
        self.efficiency_agent = EfficiencyModelAgent()
        self.nfl_efficiency_agent = NflEfficiencyModelAgent()
        self.nfl_qb_elo_agent = NflQbEloModelAgent()
        self.nhl_xg_agent = NhlXgModelAgent()
        self.wnba_efficiency_agent = WNBAEfficiencyModelAgent()
        self.ncaab_efficiency_agent = NcaabEfficiencyModelAgent()
        self.ncaaf_efficiency_agent = NcaafEfficiencyModelAgent()
        self.ncaab_possession_sim_agent = NcaabPossessionSimAgent()
        self.ncaaw_efficiency_agent = NcaawEfficiencyModelAgent()
        self.ncaaw_possession_sim_agent = NcaawPossessionSimAgent()
        self.shot_quality_agent = ShotQualityAgent()
        self.matchup_agent = MatchupAgent()
        self.possession_sim_agent = PossessionSimAgent()
        self.wnba_possession_sim_agent = WNBAPossessionSimAgent()
        self.ensemble_agent = EnsembleModelAgent(
            models=[
                self.elo_agent, self.form_agent, self.poisson_agent,
                self.tennis_agent, self.tennis_serve_agent,
                self.tennis_h2h_agent, self.tennis_trend_agent,
                self.tennis_form_agent, self.tennis_advanced_agent,
                self.ufc_rating_agent,
                self.pitcher_agent,
                self.soccer_xg_agent,
                self.efficiency_agent,
                self.nfl_efficiency_agent,
                self.nfl_qb_elo_agent,
                self.nhl_xg_agent,
                self.wnba_efficiency_agent,
                self.ncaab_efficiency_agent,
                self.ncaab_possession_sim_agent,
                self.ncaaf_efficiency_agent,
                self.ncaaw_efficiency_agent,
                self.ncaaw_possession_sim_agent,
                self.shot_quality_agent,
                self.matchup_agent,
                self.possession_sim_agent,
                self.wnba_possession_sim_agent,
            ],
            sharp_weight=sharp_weight,
        )

        self._matching = MatchingEngine()
        self._archiver = DataArchiver()

        from evmax.notifications import Notifier
        self._notifier = Notifier.from_settings()

        for agent in self._all_agents():
            agent.attach_bus(self.bus)

        self.log = structlog.get_logger(__name__)

    def _all_agents(self) -> list[Agent]:
        return [
            self.kalshi_agent, self.polymarket_us_agent,
            self.sharp_agent, self.ev_gap_agent,
            self.injury_agent, self.standings_agent, self.playoff_agent,
            self.elo_agent, self.form_agent, self.poisson_agent,
            self.tennis_agent, self.tennis_serve_agent,
            self.tennis_h2h_agent, self.tennis_trend_agent,
            self.ufc_rating_agent,
            self.pitcher_agent, self.soccer_xg_agent,
            self.efficiency_agent, self.nfl_efficiency_agent, self.nfl_qb_elo_agent,
            self.nhl_xg_agent,
            self.wnba_efficiency_agent,
            self.ncaab_efficiency_agent, self.ncaab_possession_sim_agent,
            self.ncaaf_efficiency_agent,
            self.ncaaw_efficiency_agent, self.ncaaw_possession_sim_agent,
            self.shot_quality_agent, self.matchup_agent,
            self.possession_sim_agent, self.wnba_possession_sim_agent,
            self.ensemble_agent,
        ]

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> CycleResult:
        t0 = time.perf_counter()
        correlation_id = str(uuid.uuid4())[:8]
        result = CycleResult(bankroll=self._bankroll, kelly_fraction=self._kelly_fraction)

        self.log.info("cycle_start", correlation_id=correlation_id, sectors=self._sectors)
        if self._skipped_off_season:
            self.log.info(
                "sectors_skipped_off_season",
                correlation_id=correlation_id,
                skipped=self._skipped_off_season,
            )
        self._archiver.open_session(correlation_id, self._sectors, "agents")

        # All sectors run in parallel — Kalshi rate limiting is handled by the
        # module-level AsyncLimiter token bucket in kalshi.py (8 req/s).
        # Hard 45s timeout per sector: prevents a single hanging API call from
        # stalling the entire cycle indefinitely.
        async def _run_sector_with_timeout(sector: str) -> dict:
            try:
                return await asyncio.wait_for(
                    self._run_sector(sector, correlation_id),
                    timeout=45.0,
                )
            except asyncio.TimeoutError:
                self.log.warning("sector_timeout", sector=sector, timeout_s=45)
                raise asyncio.TimeoutError(f"{sector} timed out after 45s")

        sector_results = await asyncio.gather(
            *(_run_sector_with_timeout(sector) for sector in self._sectors),
            return_exceptions=True,
        )

        for sector, sr in zip(self._sectors, sector_results):
            if isinstance(sr, Exception):
                self.log.error("sector_failed", sector=sector, error=str(sr))
                result.errors.append(f"{sector}: {sr}")
                continue
            result.sectors_scanned.append(sector)
            result.markets_fetched += sr.get("markets_fetched", 0)
            result.markets_matched += sr.get("markets_matched", 0)
            result.ev_gaps.extend(sr.get("ev_gaps", []))
            result.blended_predictions.update(sr.get("blended_predictions", {}))
            if sr.get("injuries"):
                result.injury_reports[sector] = sr["injuries"]
            result.prop_sharp_pairs.extend(sr.get("prop_sharp_pairs", []))

        # Partial-blend gaps (full_blend=False, see REQUIRED_BLEND_MODELS in
        # ev_gap_agent.py) are shadow-bound: log_gaps demotes them to
        # mode='shadow'. They never consume Kelly or the per-game exposure
        # budget, so pull them out before sizing and re-attach after with
        # kelly zeroed.
        partial_blend_gaps = [
            dataclasses.replace(g, kelly_fraction=0.0)
            for g in result.ev_gaps if not g.full_blend
        ]
        if partial_blend_gaps:
            result.ev_gaps = [g for g in result.ev_gaps if g.full_blend]
            self.log.info(
                "partial_blend_gaps_shadowed",
                count=len(partial_blend_gaps),
                sectors=sorted({g.sector for g in partial_blend_gaps}),
            )

        # Venue shadow firewall: a Polymarket US gap is shadow-bound — kelly
        # zeroed, excluded from the exposure budget, demoted to mode='shadow'
        # by log_gaps — UNLESS its sector clears the per-sector allowlist
        # (settings.polymarket_us_sector_live). Same treatment as partial-blend
        # gaps above. Any other non-Kalshi venue has no live path and is always
        # shadow-bound.
        _settings = get_settings()

        def _venue_gap_live(g) -> bool:
            return _venue_is_live(
                getattr(g, "venue", "kalshi"),
                getattr(g, "sector", None),
                self._selected_venues,
                _settings,
            )

        shadow_venue_gaps = [
            dataclasses.replace(g, kelly_fraction=0.0)
            for g in result.ev_gaps
            if not _venue_gap_live(g)
        ]
        if shadow_venue_gaps:
            result.ev_gaps = [g for g in result.ev_gaps if _venue_gap_live(g)]
            self.log.info(
                "venue_gaps_shadowed",
                count=len(shadow_venue_gaps),
                venue="polymarket_us",
                sectors=sorted({g.sector for g in shadow_venue_gaps}),
            )

        pre_guard = len(result.ev_gaps)
        # Bets the user already placed in earlier scans today make the per-game
        # cap cumulative across scans, not per-scan. _load_placed_exposure
        # returns DOLLARS at risk per game (additive across venues / a changed
        # bankroll); convert to a fraction of THIS scan's bankroll W so the
        # fraction-based guard and joint-Kelly caps stay consistent — a game
        # already holding $X of exposure has committed X/W of the current
        # bankroll, whatever base each prior bet was originally sized against.
        prior_exposure_dollars = _load_placed_exposure()
        W = self._bankroll
        prior_exposure = (
            {b: d / W for b, d in prior_exposure_dollars.items()}
            if W and W > 0 else {}
        )
        if prior_exposure:
            self.log.info(
                "exposure_prior_loaded",
                games_with_prior=len(prior_exposure),
                total_prior_dollars=round(sum(prior_exposure_dollars.values()), 2),
                bankroll=round(float(W), 2),
            )
        if get_settings().joint_kelly_enabled:
            _jk = get_settings()
            result.ev_gaps = _apply_joint_kelly(
                result.ev_gaps,
                kelly_multiplier=self._kelly_fraction,
                max_kelly_fraction=_jk.max_kelly_fraction,
                max_gross_cap=_jk.joint_kelly_max_gross_pct,
                rho_margin_total=_jk.joint_kelly_rho_margin_total,
                n_samples=_jk.joint_kelly_samples,
                prior_exposure=prior_exposure,
            )
        else:
            result.ev_gaps = _apply_exposure_guard(
                result.ev_gaps,
                prior_exposure=prior_exposure,
                same_side_kelly_discount=get_settings().same_side_kelly_discount,
            )
        dropped = pre_guard - len(result.ev_gaps)
        if dropped > 0:
            self.log.info("exposure_guard_applied", dropped=dropped, remaining=len(result.ev_gaps))
        result.exposure_guard_dropped = dropped
        # Re-attach shadow-bound partial-blend + shadow-venue gaps so they
        # reach persistence.
        result.ev_gaps.extend(partial_blend_gaps)
        result.ev_gaps.extend(shadow_venue_gaps)
        result.cycle_duration_s = time.perf_counter() - t0

        # Archive all raw fetched data for historical analysis
        k_total = s_total = 0
        for sector, sr in zip(self._sectors, sector_results):
            if isinstance(sr, Exception):
                continue
            # Archive only Kalshi rows: the archive powers Kalshi close-price
            # CLV lookups (watch-closes / backfill_clv). Polymarket US close
            # capture is a follow-up — see the PolyUS PR 3 notes.
            k_total += self._archiver.archive_kalshi_markets(
                correlation_id,
                sector,
                [
                    m for m in sr.get("markets", [])
                    if m.source == MarketSource.kalshi
                ],
            )
            s_total += self._archiver.archive_sharp_odds(
                correlation_id, sector, sr.get("sharp_odds", [])
            )
        self._archiver.close_session(
            correlation_id,
            int(result.cycle_duration_s * 1000),
            k_total,
            s_total,
        )

        self.log.info(
            "cycle_done",
            correlation_id=correlation_id,
            sectors=len(result.sectors_scanned),
            markets=result.markets_fetched,
            matched=result.markets_matched,
            ev_gaps=len(result.ev_gaps),
            duration_s=round(result.cycle_duration_s, 2),
        )

        await self.bus.publish(AgentMessage(
            topic="coordinator.cycle.done",
            payload=result,
            sender="coordinator",
            correlation_id=correlation_id,
        ))

        # Fire notifications (sync, non-blocking — uses urllib internally)
        try:
            self._notifier.notify_cycle(result)
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # Single-sector cycle
    # ------------------------------------------------------------------

    # Sectors that have Kalshi player prop series + a sharp prop anchor.
    # baseball added 2026-07-13: the KXMLBKS/KXMLBOUTS/KXMLBTB/KXMLBHR/
    # KXMLBHIT/KXMLBHRR/KXMLBRBI props + Pinnacle-anchored pricing were
    # built and validated (see baseball_props_cache.py, ev/prop_pricing.py)
    # but this set was never updated, so _fetch_props was never invoked for
    # baseball scans — the whole pipeline was dead in production.
    _PROP_SECTORS = {"nba", "nfl", "baseball"}

    async def _run_sector(self, sector: str, correlation_id: str) -> dict:
        req = AgentRequest(sector=sector, correlation_id=correlation_id)

        # Steps 1-3: Fetch Kalshi + sharp + injuries concurrently
        # Also fetch player props for supported sectors
        fetch_tasks = [self.kalshi_agent(req), self.sharp_agent(req)]
        if self._enable_injuries:
            fetch_tasks.append(self.injury_agent(req))
        fetch_tasks.append(self.standings_agent(req))
        fetch_tasks.append(self.playoff_agent(req))
        # Second venue: Polymarket US. Appended LAST so the positional
        # unpacking below stays index-stable. Sectors without a Polymarket
        # US product return [] (see POLYMARKET_US_LEAGUE_MAP).
        fetch_polymarket = get_settings().polymarket_us_enabled
        if fetch_polymarket:
            fetch_tasks.append(self.polymarket_us_agent(req))

        prop_task = None
        if sector.lower() in self._PROP_SECTORS:
            prop_task = asyncio.create_task(self._fetch_props(sector))

        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        kalshi_resp = fetch_results[0]
        sharp_resp = fetch_results[1]
        _idx = 2
        injury_resp = None
        if self._enable_injuries:
            injury_resp = fetch_results[_idx]
            _idx += 1
        standings_resp = fetch_results[_idx]
        _idx += 1
        playoff_resp = fetch_results[_idx]
        _idx += 1
        polymarket_resp = fetch_results[_idx] if fetch_polymarket else None

        markets: list[PredictionMarket] = (
            kalshi_resp.data if not isinstance(kalshi_resp, Exception) else []
        ) or []
        polymarket_markets: list[PredictionMarket] = (
            polymarket_resp.data
            if polymarket_resp is not None and not isinstance(polymarket_resp, Exception)
            else []
        ) or []
        if polymarket_markets:
            # Merge into the same pool — matching, ensemble, and EV gap
            # analysis are venue-agnostic (PredictionMarket.source / EVGap.venue
            # carry the venue through to persistence and display).
            markets = markets + polymarket_markets
            self.log.info(
                "polymarket_us_merged",
                sector=sector,
                count=len(polymarket_markets),
            )
        sharp_odds: list[SharpOdds] = (
            sharp_resp.data if not isinstance(sharp_resp, Exception) else []
        ) or []

        # Merge prop markets and prop odds if available. Props run concurrently
        # with the main fetch (create_task above), so this timeout only gates
        # how long we wait *after* Kalshi/Pinnacle/injuries finish. 20s is
        # enough for stats.nba.com without blocking other sectors.
        #
        # Dead-sector guard: when the game-market fetch SUCCEEDED and returned
        # zero markets (offseason nba is the canonical case — a no-season-window
        # live sector that scans year-round), there are no games, hence nothing
        # to price props against — cancel the in-flight prop fetch instead of
        # waiting up to 20s of stats-API I/O for an unusable result. A FAILED
        # Kalshi fetch (exception) keeps the old await-the-props behavior:
        # an empty pool from a fetch error is not evidence the sector is dead.
        prop_sharp_pairs: list[tuple[SharpOdds, PredictionMarket]] = []
        if prop_task is not None and not markets and not isinstance(kalshi_resp, Exception):
            prop_task.cancel()
            self.log.info("prop_fetch_cancelled_no_markets", sector=sector)
            prop_task = None
        if prop_task is not None:
            try:
                prop_markets, prop_sharp = await asyncio.wait_for(prop_task, timeout=20.0)
                # Build (SharpOdds, PredictionMarket) pairs for calibration logging.
                # Match by player+stat+threshold since _fetch_props deduplicates markets.
                _pm_by_key = {
                    (m.player_name, m.stat_type, m.threshold): m
                    for m in prop_markets if m.player_name and m.stat_type
                }
                for so in prop_sharp:
                    key = (so.prop_player_name, so.prop_stat_type, so.total_line)
                    pm = _pm_by_key.get(key)
                    if pm is not None:
                        prop_sharp_pairs.append((so, pm))
                markets = markets + prop_markets
                sharp_odds = sharp_odds + prop_sharp
            except (asyncio.TimeoutError, Exception) as e:
                self.log.debug("prop_fetch_skipped", sector=sector, reason=str(e))
        injuries: dict[str, InjuryReport] = (
            injury_resp.data if injury_resp and not isinstance(injury_resp, Exception) else {}
        ) or {}
        standings: dict[str, TeamStanding] = (
            standings_resp.data if not isinstance(standings_resp, Exception) else {}
        ) or {}
        playoff_data: dict[str, PlayoffSeries] = (
            playoff_resp.data if not isinstance(playoff_resp, Exception) else {}
        ) or {}

        if isinstance(kalshi_resp, Exception):
            self.log.error("kalshi_failed", sector=sector, error=str(kalshi_resp))
        if isinstance(polymarket_resp, Exception):
            self.log.error(
                "polymarket_us_failed", sector=sector, error=str(polymarket_resp)
            )
        if isinstance(sharp_resp, Exception):
            self.log.error("sharp_failed", sector=sector, error=str(sharp_resp))

        # Steam move detection: compare current sharp probs to last scan's probs
        prev_probs = _load_steam_cache()
        steam_events = _detect_steam(sharp_odds, prev_probs)
        new_probs = {so.event_id: so.true_prob_a for so in sharp_odds}
        _save_steam_cache({**prev_probs, **new_probs})

        if not markets or not sharp_odds:
            return {
                "markets_fetched": len(markets),
                "markets_matched": 0,
                "ev_gaps": [],
                "blended_predictions": {},
                "injuries": injuries,
                "markets": markets,
                "sharp_odds": sharp_odds,
                "prop_sharp_pairs": prop_sharp_pairs,
            }

        # Step 4: Match non-prop markets → sharp events (for model ensemble)
        from evmax.models.market import MarketType as _MT
        regular_markets = [m for m in markets if m.market_type != _MT.player_prop]
        regular_sharp = [s for s in sharp_odds if s.prop_player_name is None]
        matched_pairs = self._matching.match_all(regular_markets, regular_sharp)
        # Advance markets don't get their own ensemble run — EVGapAgent derives
        # their model prob from the SAME game's regulation 3-way blend (the ML
        # pair below), keyed without the ::advance suffix. Feeding advance
        # pairs to the ensemble would just produce a nonsense regulation-blend
        # entry under the ::advance key.
        pairs = [
            {"market": m, "sharp": s}
            for m, s, _ in matched_pairs
            if m.market_type != _MT.advance
        ]

        # Step 5: Ensemble model predictions
        blended: dict[str, BlendedPrediction] = {}
        model_probs: dict[str, float] = {}
        model_sources: dict[str, str] = {}

        blended_preds: dict = {}  # event_id → BlendedPrediction
        if self._enable_models and pairs:
            # Per-sector sharp_weight override comes from data/model_config.json's
            # `sharp_weight_by_sector` map (see evmax/agents/cleanup/metrics.py).
            # Tennis / baseball live there as defaults because our models are thin
            # and we want to trust Pinnacle more heavily on those sports.
            from evmax.agents.cleanup.metrics import load_config as _load_cfg
            cfg = _load_cfg()
            by_sector = cfg.get("sharp_weight_by_sector") or {}
            sector_sharp_weight = float(by_sector.get(sector.lower(), self._sharp_weight))

            # For soccer, override sharp_weight per event based on league tier.
            # Top-5 European leagues get pushed toward pure sharp (Pinnacle is
            # the ceiling there); secondary leagues keep the sector default.
            # See data/soccer_league_tiers.yaml + evmax/sectors/soccer_tiers.py.
            sharp_weight_by_event: dict[str, float] = {}
            if sector.lower() == "soccer":
                from evmax.sectors.soccer_tiers import sharp_weight_for_ticker
                for pair in pairs:
                    market = pair["market"]
                    event_id = pair["sharp"].event_id
                    sharp_weight_by_event[event_id] = sharp_weight_for_ticker(market.ticker)

            ensemble_req = AgentRequest(
                sector=sector,
                params={
                    "pairs": pairs,
                    "sharp_weight": sector_sharp_weight,
                    "sharp_weight_by_event": sharp_weight_by_event,
                },
                correlation_id=correlation_id,
            )
            ensemble_resp = await self.ensemble_agent(ensemble_req)
            blended = ensemble_resp.data or {}
            blended_preds = blended

            for event_id, blend in blended.items():
                # Store full objects; EVGapAgent will pick correct side per YES alignment
                model_sources[event_id] = blend.model_sources

        # Step 7: EV gap analysis — pass full blended predictions for correct side alignment
        ev_req = AgentRequest(
            sector=sector,
            params={
                "kalshi_markets": markets,
                "sharp_odds": sharp_odds,
                "blended_preds": blended_preds,
                "injuries": injuries,
                "standings": standings,
                "playoff_data": playoff_data,
                "model_sources": model_sources,
                "kelly_base_fraction": self._kelly_fraction,
                "steam_events": steam_events,
                "possession_sim_agent": self.possession_sim_agent,
                "wnba_possession_sim_agent": self.wnba_possession_sim_agent,
            },
            correlation_id=correlation_id,
        )
        ev_resp = await self.ev_gap_agent(ev_req)
        ev_gaps: list[EVGap] = ev_resp.data or []

        return {
            "markets_fetched": len(markets),
            "markets_matched": len(matched_pairs),
            "ev_gaps": ev_gaps,
            "blended_predictions": blended,
            "injuries": injuries,
            "markets": markets,
            "sharp_odds": sharp_odds,
            "prop_sharp_pairs": prop_sharp_pairs,
        }

    async def _fetch_props(
        self,
        sector: str,
    ) -> tuple[list[PredictionMarket], list[SharpOdds]]:
        """Fetch Kalshi player prop markets paired with Pinnacle devigged lines.

        For each Kalshi prop, we look up the matching Pinnacle player+stat+line
        and use Pinnacle's devigged over-probability as the sharp anchor.
        Kalshi props with no matching Pinnacle line are dropped — we don't
        bet props without a sharp reference.

        The local L15 game-log cache is still consulted, but only to attach
        diagnostic metadata (sample size, minutes volatility) for display
        and downstream analysis. It does NOT feed the EV calculation.
        """
        from evmax.clients.kalshi import KalshiClient
        from evmax.clients.esports_pinnacle import PinnacleGuestClient
        # Production reads compute_prop_diagnostics — sample size + minutes
        # volatility only. The legacy compute_prop_prob_cached still exists
        # for backtest scripts and `evmax cleanup replay-props` but is no
        # longer in the live scan path. See nba_props_cache.PropDiagnostics
        # docstring for the design rationale.
        from evmax.clients.nba_props_cache import (
            compute_prop_diagnostics,
            is_cache_fresh,
            refresh_props_cache,
        )
        from evmax.clients.nfl_props_cache import (
            compute_nfl_prop_diagnostics,
            is_nfl_cache_fresh,
            refresh_nfl_props_cache,
        )
        from evmax.clients.baseball_props_cache import (
            is_cache_fresh as is_bb_props_cache_fresh,
            refresh_baseball_props_cache,
        )

        prop_sector = f"{sector}_props"

        # Fetch Kalshi prop markets and Pinnacle prop lines in parallel
        async def _kalshi_fetch() -> list[PredictionMarket]:
            async with KalshiClient() as kalshi:
                raw = await kalshi.get_markets(prop_sector)
            return raw if isinstance(raw, list) else []

        async def _pinn_fetch() -> list[SharpOdds]:
            async with PinnacleGuestClient() as pinn:
                return await pinn.get_prop_odds(sector)

        kalshi_res, pinn_res = await asyncio.gather(
            _kalshi_fetch(), _pinn_fetch(), return_exceptions=True,
        )
        prop_markets: list[PredictionMarket] = (
            kalshi_res if isinstance(kalshi_res, list) else []
        )
        pinn_lines: list[SharpOdds] = (
            pinn_res if isinstance(pinn_res, list) else []
        )

        if not prop_markets:
            self.log.debug("props_fetched", sector=sector, prop_markets=0, prop_sharp=0)
            return [], []

        # Index Pinnacle props by (player_norm, stat_type) — Pinnacle posts ONE
        # half-point line per (player, stat); Kalshi posts MANY integer 'X+'
        # thresholds. Distribution-based pricing (evmax/ev/prop_pricing.py) reads
        # off P(stat >= K) for any K from the single anchor.
        from evmax.ev.prop_pricing import price_kalshi_threshold

        pinn_by_anchor: dict[tuple[str, str], SharpOdds] = {}
        # Per-stat fallback index: { stat_type: [(player_norm, anchor), ...] }
        # for fuzzy lookup when the exact (player, stat) key misses (accent /
        # suffix / spelling variants between Kalshi and Pinnacle normalization).
        pinn_by_stat: dict[str, list[tuple[str, SharpOdds]]] = {}
        for p in pinn_lines:
            if (
                p.prop_player_name
                and p.prop_stat_type
                and p.total_line is not None
                and p.true_prob_over is not None
            ):
                pinn_by_anchor[(p.prop_player_name, p.prop_stat_type)] = p
                pinn_by_stat.setdefault(p.prop_stat_type, []).append(
                    (p.prop_player_name, p)
                )

        # Deduplicate (player, stat, threshold) so we only emit each prob once
        seen: set[tuple] = set()
        unique: list[PredictionMarket] = []
        for m in prop_markets:
            if m.player_name and m.stat_type and m.threshold is not None:
                key = (m.player_name, m.stat_type, m.threshold)
                if key not in seen:
                    seen.add(key)
                    unique.append(m)

        # Auto-refresh L15 cache (used now only for diagnostic metadata)
        if sector == "nba" and not is_cache_fresh():
            player_names = list({m.player_name for m in unique if m.player_name})
            self.log.info("props_cache_refreshing", players=len(player_names))
            await refresh_props_cache(force=True, player_names=player_names)
        elif sector == "nfl" and not is_nfl_cache_fresh():
            # NFL cache loads from pre-downloaded nflverse parquets, not from
            # the network — the refresh just materializes the in-memory
            # tables. player_names is ignored (the parquet already covers
            # every player) but passed for API symmetry with NBA.
            player_names = list({m.player_name for m in unique if m.player_name})
            self.log.info("nfl_props_cache_refreshing", players=len(player_names))
            await refresh_nfl_props_cache(force=True, player_names=player_names)
        elif sector == "baseball" and not is_bb_props_cache_fresh():
            # Season-to-date hitting/pitching rates from the MLB Stats API,
            # one league-wide pull. Feeds the projection model that blends
            # with the Pinnacle anchor below.
            self.log.info("baseball_props_cache_refreshing")
            await refresh_baseball_props_cache(force=True)

        # Build SharpOdds list — for each Kalshi threshold, price off the
        # Pinnacle anchor for the same (player, stat) via prop_pricing.
        # Player-name fuzzy fallback: PropMatcher uses token_sort_ratio ≥85
        # for the same reason — Kalshi and Pinnacle normalize accents /
        # suffixes / spelling variants slightly differently. Try exact lookup
        # first, then fuzzy within the same stat_type. Threshold mirrors
        # evmax/matching/prop_matcher.py::PLAYER_MATCH_THRESHOLD.
        from rapidfuzz import fuzz
        FUZZY_PLAYER_THRESHOLD = 85

        prop_sharp: list[SharpOdds] = []
        unmatched = 0
        unpriced = 0
        fuzzy_hits = 0
        bridged = 0
        for market in unique:
            anchor = pinn_by_anchor.get((market.player_name, market.stat_type))
            if anchor is None:
                # Fuzzy fallback within the same stat type
                candidates = pinn_by_stat.get(market.stat_type, [])
                best_score = 0.0
                for pinn_name, pinn_anchor in candidates:
                    score = fuzz.token_sort_ratio(
                        market.player_name.lower(),
                        pinn_name.lower(),
                    )
                    if score >= FUZZY_PLAYER_THRESHOLD and score > best_score:
                        best_score = score
                        anchor = pinn_anchor
                if anchor is not None:
                    fuzzy_hits += 1
            if anchor is None:
                # Baseball hitter stats with no Pinnacle line (hits / H+R+RBI /
                # RBI) are priced by the projection model, bridged off the
                # player's Total-Bases anchor so they still play with sharp.
                if sector == "baseball" and market.stat_type in _BASEBALL_BRIDGE_STATS:
                    tb_anchor = pinn_by_anchor.get((market.player_name, "total_bases"))
                    if tb_anchor is not None:
                        synth = self._baseball_bridged_prop_sharp(market, tb_anchor, sector)
                        if synth is not None:
                            prop_sharp.append(synth)
                            bridged += 1
                            continue
                unmatched += 1
                continue

            prob_over = price_kalshi_threshold(
                stat_type=market.stat_type,
                pinn_line=anchor.total_line,
                pinn_prob_over=anchor.true_prob_over,
                kalshi_threshold=market.threshold,
            )
            if prob_over is None:
                # Anchor present but stat unsupported by pricing module, or
                # anchor probability degenerate. Counted separately so we can
                # tell "no Pinnacle line" from "could not price" in logs.
                unpriced += 1
                continue

            # Baseball: blend the projection model with the sharp anchor so the
            # price can express a player-rate view the stale Kalshi line misses.
            # Conservative weight (BASEBALL_PROP_MODEL_WEIGHT) — the model uses a
            # neutral matchup context for now (opponent/park enrichment is the
            # next increment; the backtest evaluates the model's full-context
            # ceiling). Anchored stats only this increment; the unanchored
            # Hits/H+R+RBI/RBI stats (no Pinnacle line) are dropped above and
            # await model-only synthetic odds.
            if sector == "baseball":
                model_over = self._baseball_prop_model_prob(market)
                if model_over is not None:
                    prob_over = (
                        BASEBALL_PROP_MODEL_WEIGHT * model_over
                        + (1.0 - BASEBALL_PROP_MODEL_WEIGHT) * prob_over
                    )

            # Augment Pinnacle SharpOdds with L15 sample-size + minutes
            # volatility diagnostics. No probability math — anchor pricing
            # owns the prob estimate.
            if sector == "nba":
                l15 = compute_prop_diagnostics(market.player_name)
            elif sector == "nfl":
                l15 = compute_nfl_prop_diagnostics(market.player_name)
            else:
                l15 = None

            # Synthesize SharpOdds at the Kalshi threshold using the priced prob.
            # outcome_a/b decimal odds carry over from the anchor (they were the
            # raw Pinnacle quote at the anchor line, no longer meaningful at a
            # different threshold) — downstream EV uses true_prob_over only.
            #
            # Rewrite event_id to encode the Kalshi threshold, not the anchor
            # line. The cleanup resolver parses event_id parts[5] to extract the
            # threshold for prop resolution; if we left the anchor's event_id
            # in place, a 10+ market would be resolved against the anchor's
            # 4.5 threshold and incorrectly marked WON.
            new_event_id = (
                f"{sector}::{anchor.event_id.split('::')[1]}"
                f"::prop::{market.player_name}::{market.stat_type}"
                f"::{market.threshold}"
            )
            prop_sharp.append(anchor.model_copy(update={
                "event_id": new_event_id,
                "total_line": market.threshold,
                "true_prob_over": prob_over,
                "true_prob_under": 1.0 - prob_over,
                "prop_l15_games": l15.n_games if l15 else 0,
                "prop_minutes_volatile": l15.minutes_volatile if l15 else False,
                "prop_minutes_cv": l15.minutes_cv if l15 else 0.0,
            }))

        self.log.info(
            "props_fetched",
            sector=sector,
            prop_markets=len(prop_markets),
            prop_unique=len(unique),
            pinn_lines=len(pinn_lines),
            pinn_anchors=len(pinn_by_anchor),
            matched=len(prop_sharp),
            fuzzy_hits=fuzzy_hits,
            bridged=bridged,
            unmatched_dropped=unmatched,
            unpriced=unpriced,
        )
        return prop_markets, prop_sharp

    def _baseball_prop_model_prob(self, market: PredictionMarket) -> Optional[float]:
        """Model P(stat >= threshold) for one MLB prop, or None if not projectable.

        Pitcher stats project off the seeded pitcher profile, hitter stats off
        the hitter profile (both season-to-date from the MLB Stats API cache).
        Uses a neutral matchup context for now — opponent-K / park / opposing-
        starter enrichment is the next increment.
        """
        from evmax.clients.baseball_props_cache import (
            get_hitter_profile,
            get_pitcher_profile,
        )
        from evmax.models_ml.baseball_props import (
            HITTER_STATS,
            MatchupContext,
            PITCHER_STATS,
            project_prob,
        )

        if market.player_name is None or market.threshold is None:
            return None
        ctx = MatchupContext()
        if market.stat_type in PITCHER_STATS:
            prof = get_pitcher_profile(market.player_name)
            if prof is None:
                return None
            return project_prob(market.stat_type, market.threshold, ctx, pitcher=prof)
        if market.stat_type in HITTER_STATS:
            prof = get_hitter_profile(market.player_name)
            if prof is None:
                return None
            return project_prob(market.stat_type, market.threshold, ctx, hitter=prof)
        return None

    def _baseball_bridged_prop_sharp(
        self,
        market: PredictionMarket,
        tb_anchor: SharpOdds,
        sector: str,
    ) -> Optional[SharpOdds]:
        """Price an unanchored hitter stat off the player's Total-Bases anchor.

        Pinnacle posts no line for hits / H+R+RBI / RBI, so the projection
        model carries them — but we *bridge* it to the sharp Total-Bases line:
        scale the target-stat projection by the ratio of the anchor-implied TB
        mean to the model's own TB mean, so the unanchored stat inherits the
        sharp's read on the hitter's output instead of trusting the raw model
        alone. Returns a synthetic SharpOdds (reusing the TB anchor's game key
        + player) or None when it can't be priced.
        """
        from evmax.clients.baseball_props_cache import get_hitter_profile
        from evmax.ev.prop_pricing import fit_distribution, model_prob_at_or_above
        from evmax.models_ml.baseball_props import (
            MatchupContext,
            project_hitter_total_bases,
            project_mean,
        )

        if market.threshold is None:
            return None
        prof = get_hitter_profile(market.player_name)
        if prof is None:
            return None
        ctx = MatchupContext()
        model_tb = project_hitter_total_bases(prof, ctx)
        anchor_dist = fit_distribution(
            "total_bases", tb_anchor.total_line, tb_anchor.true_prob_over
        )
        anchor_tb = getattr(anchor_dist, "mu", None)
        if not model_tb or model_tb <= 0 or not anchor_tb or anchor_tb <= 0:
            return None
        ratio = max(0.5, min(2.0, anchor_tb / model_tb))
        base_mean = project_mean(market.stat_type, ctx, hitter=prof)
        if base_mean is None:
            return None
        prob_over = model_prob_at_or_above(
            market.stat_type, base_mean * ratio, market.threshold
        )
        if prob_over is None:
            return None
        new_event_id = (
            f"{sector}::{tb_anchor.event_id.split('::')[1]}"
            f"::prop::{market.player_name}::{market.stat_type}::{market.threshold}"
        )
        return tb_anchor.model_copy(update={
            "event_id": new_event_id,
            "total_line": market.threshold,
            "true_prob_over": prob_over,
            "true_prob_under": 1.0 - prob_over,
            "prop_stat_type": market.stat_type,
            "prop_l15_games": 0,
            "prop_minutes_volatile": False,
            "prop_minutes_cv": 0.0,
        })

    # ------------------------------------------------------------------
    # Model training helpers
    # ------------------------------------------------------------------

    def update_models(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
        surface: str = "overall",
        save: bool = True,
    ) -> None:
        """Feed a completed game result into all model agents.

        ``save`` defaults to True so existing callers persist per game. Batch
        callers that apply many games in a row (the resolve-time model-update
        hook) pass ``save=False`` and call ``save_model_states()`` once per
        sector instead, avoiding a full re-serialization of every state file on
        each game (~90 writes on a 30-game slate).
        """
        for model in [self.elo_agent, self.form_agent, self.poisson_agent]:
            model.update(team_a, team_b, score_a, score_b, sector, event_date)
        if sector == "tennis":
            self.tennis_agent.update(team_a, team_b, score_a, score_b, sector, event_date, surface=surface)
        if sector == "ufc":
            # Method of victory isn't available through this interface —
            # KO/finish counters refresh at the weekly reseed
            # (scripts/seed_ufc_ratings.py --fetch). Ratings update here.
            self.ufc_rating_agent.update(team_a, team_b, score_a, score_b, sector, event_date)
        if save:
            self.save_model_states(sector)

    def save_model_states(self, sector: Optional[str] = None) -> None:
        """Persist the model states mutated by ``update_models``.

        Saves the elo/form/poisson states always. When ``sector`` is given,
        also saves the sector-specific state (UFC ratings) only for that sector,
        matching which agents ``update_models`` actually touched. When ``sector``
        is None, saves every stateful agent (safe for a batch flush).
        """
        self.elo_agent.save_state()
        self.form_agent.save_state()
        self.poisson_agent.save_state()
        if sector in (None, "ufc"):
            self.ufc_rating_agent.save_state()

    def next_scan_interval_seconds(self, result: "CycleResult") -> int:
        """Compute adaptive scan interval based on time until soonest game.

        Uses event_date from EV gaps found in the most recent cycle result.

        Tier logic (time to soonest kickoff):
          Live / already started: 90 sec
          < 60 min:               3 min
          60 min – 4 hours:       10 min
          > 4 hours / no games:   30 min
        """
        now_utc = datetime.now(timezone.utc)
        soonest_delta: float | None = None

        for gap in result.ev_gaps:
            ed = gap.event_date
            if ed is None:
                continue
            # Normalise to offset-aware UTC
            if not hasattr(ed, "tzinfo") or ed.tzinfo is None:
                ed = ed.replace(tzinfo=timezone.utc)
            delta = (ed - now_utc).total_seconds()
            if soonest_delta is None or delta < soonest_delta:
                soonest_delta = delta

        if soonest_delta is None:
            return 1800  # 30 min — no games found

        if soonest_delta <= 0:
            return 90   # game is live
        if soonest_delta < 60 * 60:
            return 180  # < 1 hour to kickoff
        if soonest_delta < 4 * 60 * 60:
            return 600  # 1–4 hours
        return 1800  # > 4 hours

    def subscribe(self, topic: str, handler) -> None:
        self.bus.subscribe(topic, handler)
