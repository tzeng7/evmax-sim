"""EVGapAgent — compares Kalshi odds vs sharp odds to identify +EV gaps.

Inputs (via request.params):
  kalshi_markets  : list[PredictionMarket]
  sharp_odds      : list[SharpOdds]
  blended_preds   : dict[event_id, BlendedPrediction]  — full model blend (prob_a+prob_b)
  injuries        : dict[team_name, InjuryReport]       — from InjuryReportAgent
  model_sources   : dict[event_id, str]
  kelly_base_fraction : float   — Kelly multiplier (default 0.25)

Published topic: "ev.gaps.{sector}"
Output data: list[EVGap] sorted descending by ev_pct.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from evmax.agents.base import Agent, AgentRequest, AgentResponse
from evmax.ev.calculator import calculate_ev
from evmax.ev.kelly import compute_kelly
from evmax.matching.engine import MatchingEngine
from evmax.matching.prop_matcher import PropMatcher
from evmax.models.market import PredictionMarket, MarketType
from evmax.models.odds import SharpOdds
from evmax.models_ml.spread_distribution import SpreadDistributionModel
from evmax.models_ml.total_distribution import TotalDistributionModel, is_game_total
from evmax.settings import get_settings


@dataclass
class EVGap:
    """A single +EV opportunity found by comparing Kalshi vs sharp odds."""

    market_id: str
    event_id: str
    sector: str
    yes_team: str
    market_type: str
    kalshi_yes_price: float
    sharp_true_prob: float      # devigged Pinnacle, YES-aligned
    blended_true_prob: float    # after model blend + injury adj
    ev_pct: float
    kelly_full: float
    kelly_fraction: float
    match_confidence: float
    volume_usd: float
    spread_pct: float
    event_date: Optional[datetime] = None
    model_sources: str = "sharp"
    line: Optional[float] = None          # spread/total line (e.g. -8.5, 220.5)
    event_title: str = ""                 # e.g. "Celtics vs Wizards"
    steam_move: bool = False              # True if Pinnacle line moved ≥ 2pp since last scan
    # Player prop fields (only set when market_type == player_prop)
    prop_player_name: Optional[str] = None
    prop_stat_type: Optional[str] = None
    prop_threshold: Optional[float] = None

    @property
    def edge_label(self) -> str:
        if self.ev_pct >= 0.10:
            return "STRONG"
        if self.ev_pct >= 0.05:
            return "GOOD"
        return "MARGINAL"

    @property
    def implied_prob(self) -> float:
        return self.kalshi_yes_price

    @property
    def display_label(self) -> str:
        """Human-readable label: team + market type + line where applicable."""
        # Capitalize first letter only to avoid "76Ers" -> "76ers" style issues
        team = self.yes_team.capitalize()
        if self.market_type == "player_prop":
            stat = (self.prop_stat_type or "prop").replace("_", " ").title()
            thr = f"{self.prop_threshold:.1f}" if self.prop_threshold is not None else "?"
            return f"{self.prop_player_name or team} {stat} O {thr}"
        if self.market_type == "moneyline":
            return f"{team} ML"
        if self.market_type == "spread" and self.line is not None:
            # line is stored as negative (win by X); format without trailing zeros
            line_str = f"{self.line:.1f}".rstrip("0").rstrip(".")
            return f"{team} {line_str}"
        if self.market_type in ("over_under", "total") and self.line is not None:
            return f"O/U {self.line:.1f}"
        return team

    @property
    def confidence_stars(self) -> int:
        """0–3 confidence stars based on match quality, volume, and model signal."""
        stars = 0
        if self.match_confidence >= 0.90:
            stars += 1
        if self.volume_usd >= 5000:
            stars += 1
        if self.model_sources not in ("sharp", "sharp(capped)"):
            stars += 1
        return stars

    @property
    def stars_display(self) -> str:
        n = self.confidence_stars
        return "★" * n + "☆" * (3 - n)

    def summary(self) -> str:
        return (
            f"[{self.edge_label}] {self.display_label} | {self.sector.upper()} "
            f"| Kalshi={self.kalshi_yes_price:.2f} TrueP={self.blended_true_prob:.3f} "
            f"EV={self.ev_pct*100:.1f}% Kelly={self.kelly_fraction*100:.2f}% {self.stars_display}"
        )


class EVGapAgent(Agent):
    """
    Matches Kalshi markets to sharp events and computes EV gaps.

    YES-team alignment:
      Kalshi has one YES market per team/outcome.  outcome_a in SharpOdds is always
      the home/favored team (Pinnacle convention).  When the YES side of a Kalshi
      market is the away team, we use true_prob_b (and blended.true_prob_b) instead
      of the default true_prob_a.

    Model blend integration:
      BlendedPrediction objects (from EnsembleModelAgent) carry both true_prob_a and
      true_prob_b.  EVGapAgent picks the correct side based on YES alignment and
      applies injury adjustments on top.
    """

    name = "ev_gap"
    description = (
        "Matches Kalshi markets to Pinnacle odds, aligns YES probabilities, "
        "applies model blend and injury adjustments, computes EV gaps ≥ threshold."
    )

    def __init__(self) -> None:
        super().__init__()
        self._settings = get_settings()
        self._matching = MatchingEngine()
        self._prop_matcher = PropMatcher()
        self._spread_model = SpreadDistributionModel()
        self._total_model = TotalDistributionModel()

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        markets: list[PredictionMarket] = request.params.get("kalshi_markets", [])
        sharp_list: list[SharpOdds] = request.params.get("sharp_odds", [])
        blended_preds: dict = request.params.get("blended_preds", {})
        injuries: dict = request.params.get("injuries", {})
        model_sources: dict[str, str] = request.params.get("model_sources", {})
        kelly_base: float = request.params.get("kelly_base_fraction", 0.25)
        steam_events: set[str] = request.params.get("steam_events", set())

        self.log.info(
            "ev_gap_start",
            sector=sector,
            markets=len(markets),
            sharp_events=len(sharp_list),
            model_overrides=len(blended_preds),
        )

        # Split prop markets from regular markets
        prop_markets = [m for m in markets if m.market_type == MarketType.player_prop]
        regular_markets = [m for m in markets if m.market_type != MarketType.player_prop]
        sharp_props = [s for s in sharp_list if s.prop_player_name is not None]
        sharp_regular = [s for s in sharp_list if s.prop_player_name is None]

        matched = self._matching.match_all(regular_markets, sharp_regular)
        prop_matched = self._prop_matcher.match_all(prop_markets, sharp_props)
        gaps: list[EVGap] = []

        for market, sharp, confidence in matched:
            gap = self._evaluate_pair(
                market=market,
                sharp=sharp,
                confidence=confidence,
                sector=sector,
                blended_preds=blended_preds,
                injuries=injuries,
                model_sources=model_sources,
                kelly_base=kelly_base,
                steam_events=steam_events,
            )
            if gap is not None:
                gaps.append(gap)

        for market, sharp, confidence in prop_matched:
            gap = self._evaluate_prop_pair(
                market=market,
                sharp=sharp,
                confidence=confidence,
                sector=sector,
                kelly_base=kelly_base,
            )
            if gap is not None:
                gaps.append(gap)

        gaps.sort(key=lambda g: g.ev_pct, reverse=True)

        self.log.info("ev_gaps_found", sector=sector, count=len(gaps))

        await self.publish(f"ev.gaps.{sector}", gaps, request.correlation_id)

        return AgentResponse(
            agent_name=self.name,
            sector=sector,
            data=gaps,
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate_pair(
        self,
        market: PredictionMarket,
        sharp: SharpOdds,
        confidence: float,
        sector: str,
        blended_preds: dict,
        injuries: dict,
        model_sources: dict[str, str],
        kelly_base: float = 0.25,
        steam_events: Optional[set] = None,
    ) -> Optional[EVGap]:
        if market.yes_price <= 0 or market.yes_price >= 1.0:
            return None

        # ------------------------------------------------------------------
        # Type-mismatch guards: incompatible market/sharp combinations
        # ------------------------------------------------------------------
        is_total = market.market_type == MarketType.total
        is_sharp_total = sharp.total_line is not None

        # Totals market must match totals sharp record and vice versa
        if is_total and not is_sharp_total:
            return None
        if not is_total and is_sharp_total:
            return None

        # Moneyline must not match spread record
        if market.market_type == MarketType.moneyline and sharp.spread_line is not None:
            return None

        # Reject team scoring props mistakenly typed as totals (e.g. "Lakers O109.5")
        if is_total and market.line is not None and not is_game_total(market.line, sector):
            return None

        # ------------------------------------------------------------------
        # Step 1: YES-team alignment (which sharp prob belongs to the YES side?)
        # ------------------------------------------------------------------
        yes_team_norm = (market.yes_team or "").lower().strip()
        outcome_a_norm = (sharp.outcome_a_label or "").lower().strip()

        is_draw = yes_team_norm in ("tie", "draw", "x", "draw/tie")

        # Totals: YES side is "over" or "under" — not a team name
        if is_total:
            yes_is_under = yes_team_norm == "under"
            if sharp.true_prob_over is None or sharp.true_prob_under is None:
                return None
            sharp_true_prob = sharp.true_prob_under if yes_is_under else sharp.true_prob_over
            yes_is_outcome_b = False
        elif is_draw:
            if sharp.true_prob_draw is None:
                return None
            sharp_true_prob = sharp.true_prob_draw
            yes_is_outcome_b = False
            yes_is_under = False
        elif yes_team_norm and outcome_a_norm and yes_team_norm not in outcome_a_norm:
            # YES team is outcome_b (away/underdog)
            sharp_true_prob = sharp.true_prob_b
            yes_is_outcome_b = True
            yes_is_under = False
        else:
            sharp_true_prob = sharp.true_prob_a
            yes_is_outcome_b = False
            yes_is_under = False

        # ------------------------------------------------------------------
        # Step 2a: Spread markets — use SpreadDistributionModel (line-adjusted)
        # ------------------------------------------------------------------
        used_spread_model = False
        if market.market_type == MarketType.spread and market.line is not None:
            spread_result = self._spread_model.predict(
                sharp_odds=sharp,
                target_line=market.line,
                sector=sector,
                yes_is_underdog=yes_is_outcome_b,
            )
            if spread_result is not None:
                sharp_true_prob = spread_result.true_prob
                yes_is_outcome_b = False
                used_spread_model = True
            else:
                return None  # line too far — extrapolation unreliable

        # ------------------------------------------------------------------
        # Step 2b: Total markets — use TotalDistributionModel (line-adjusted)
        # ------------------------------------------------------------------
        used_total_model = False
        if is_total and market.line is not None:
            total_result = self._total_model.predict(
                sharp_odds=sharp,
                target_line=market.line,
                sector=sector,
                yes_is_under=yes_is_under,
            )
            if total_result is not None:
                sharp_true_prob = total_result.true_prob
                used_total_model = True
            # If None (line too far), fall through with raw Pinnacle over/under prob

        skip_blend = used_spread_model or used_total_model or is_total

        # ------------------------------------------------------------------
        # Step 3: Model blend override — skip for spreads/totals
        # EnsembleModelAgent only carries team-side probs (true_prob_a/b),
        # not over/under probs, so totals always use sharp directly.
        # ------------------------------------------------------------------
        blend = blended_preds.get(sharp.event_id)
        if blend is not None and not skip_blend:
            if is_draw:
                blended_prob = blend.true_prob_draw if blend.true_prob_draw is not None else sharp_true_prob
            elif yes_is_outcome_b:
                blended_prob = blend.true_prob_b
            else:
                blended_prob = blend.true_prob_a
            src = model_sources.get(sharp.event_id, blend.model_sources)
        else:
            blended_prob = sharp_true_prob
            if used_spread_model:
                src = "sharp+spread_dist"
            elif used_total_model:
                src = "sharp+total_dist"
            else:
                src = model_sources.get(sharp.event_id, "sharp")

        # ------------------------------------------------------------------
        # Step 3b: Sanity-cap model drift vs sharp
        # ------------------------------------------------------------------
        if blend is not None and not skip_blend and sharp_true_prob > 0:
            ratio = blended_prob / sharp_true_prob
            if ratio >= 2.0 or ratio <= 0.5:
                self.log.debug(
                    "model_drift_capped",
                    event_id=sharp.event_id,
                    sharp=round(sharp_true_prob, 3),
                    blended=round(blended_prob, 3),
                    ratio=round(ratio, 2),
                )
                blended_prob = sharp_true_prob
                src = "sharp(capped)"

        # ------------------------------------------------------------------
        # Step 4: Injury adjustment — skip for totals (affects both teams equally)
        # ------------------------------------------------------------------
        if injuries and not is_total:
            from evmax.agents.intelligence.injury_agent import InjuryReportAgent
            team_a = (sharp.outcome_a_label or "").lower()
            team_b = (sharp.outcome_b_label or "").lower()
            new_a, new_b, inj_notes = InjuryReportAgent.apply_adjustments(
                reports=injuries,
                true_prob_a=blended_prob if not yes_is_outcome_b else 1.0 - blended_prob,
                true_prob_b=blended_prob if yes_is_outcome_b else 1.0 - blended_prob,
                team_a=team_a,
                team_b=team_b,
                spread_multiplier=2.0 if used_spread_model else 1.0,
            )
            if inj_notes:
                blended_prob = new_b if yes_is_outcome_b else new_a
                src = f"{src}+injury"

        # ------------------------------------------------------------------
        # Step 5: EV and Kelly sizing
        # ------------------------------------------------------------------
        ev, edge_pct = calculate_ev(market.yes_price, blended_prob)
        if ev < self._settings.ev_threshold:
            return None

        payout = 1.0 / market.yes_price
        kelly = compute_kelly(
            true_prob=blended_prob,
            payout_decimal=payout,
            edge_pct=edge_pct,
            spread_pct=market.spread_pct,
            base_fraction=kelly_base,
            max_kelly=self._settings.max_kelly_fraction,
        )

        is_steam = bool(steam_events and sharp.event_id in steam_events)

        return EVGap(
            market_id=market.id,
            event_id=sharp.event_id,
            sector=sector,
            yes_team=market.yes_team or market.team_home or "?",
            market_type=market.market_type.value if market.market_type else "moneyline",
            kalshi_yes_price=market.yes_price,
            sharp_true_prob=sharp_true_prob,
            blended_true_prob=blended_prob,
            ev_pct=ev,
            kelly_full=kelly.kelly_full,
            kelly_fraction=kelly.kelly_fraction,
            match_confidence=confidence,
            volume_usd=market.volume_usd,
            spread_pct=market.spread_pct,
            event_date=sharp.event_date or market.event_date,
            line=market.line,
            model_sources=src,
            event_title=(
                # For totals, outcome_a/b_label is "over"/"under" — extract team names
                # from the event_id slug instead so the title shows "Lakers vs Celtics"
                " vs ".join(
                    p.replace("_", " ").title()
                    for p in sharp.event_id.split("::")[2].split("::")[0].split("_vs_")
                ) if is_total and "::" in sharp.event_id
                else f"{sharp.outcome_a_label or '?'} vs {sharp.outcome_b_label or '?'}"
            ),
            steam_move=is_steam,
        )

    def _evaluate_prop_pair(
        self,
        market: PredictionMarket,
        sharp: SharpOdds,
        confidence: float,
        sector: str,
        kelly_base: float = 0.25,
    ) -> Optional[EVGap]:
        """Evaluate a matched player prop pair for EV.

        YES side on Kalshi = player goes OVER the threshold.
        Uses sharp.true_prob_over as the true probability.
        """
        if market.yes_price <= 0 or market.yes_price >= 1.0:
            return None

        if sharp.true_prob_over is None:
            return None

        sharp_true_prob = sharp.true_prob_over
        ev, edge_pct = calculate_ev(market.yes_price, sharp_true_prob)
        if ev < self._settings.ev_threshold:
            return None

        payout = 1.0 / market.yes_price
        kelly = compute_kelly(
            true_prob=sharp_true_prob,
            payout_decimal=payout,
            edge_pct=edge_pct,
            spread_pct=market.spread_pct,
            base_fraction=kelly_base,
            max_kelly=self._settings.max_kelly_fraction,
        )

        player_display = (market.player_name or "?").replace("_", " ").title()
        stat_display = (market.stat_type or "prop").replace("_", " ").title()

        return EVGap(
            market_id=market.id,
            event_id=sharp.event_id,
            sector=sector,
            yes_team=market.player_name or "?",
            market_type=MarketType.player_prop.value,
            kalshi_yes_price=market.yes_price,
            sharp_true_prob=sharp_true_prob,
            blended_true_prob=sharp_true_prob,
            ev_pct=ev,
            kelly_full=kelly.kelly_full,
            kelly_fraction=kelly.kelly_fraction,
            match_confidence=confidence,
            volume_usd=market.volume_usd,
            spread_pct=market.spread_pct,
            event_date=sharp.event_date or market.event_date,
            line=market.threshold,
            model_sources="sharp",
            event_title=f"{player_display} {stat_display} O{market.threshold}",
            prop_player_name=market.player_name,
            prop_stat_type=market.stat_type,
            prop_threshold=market.threshold,
        )
