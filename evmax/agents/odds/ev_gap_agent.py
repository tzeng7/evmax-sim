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
    line_velocity: float | None = None   # prob change per hour (pp/hr), from archived line history
    velocity_flag: str | None = None     # "STEAM" if |velocity| > 1.5 pp/hr, "STALE" if no move 4+hrs
    book_count: int = 1                  # number of sharp books in consensus
    # Player prop fields (only set when market_type == player_prop)
    prop_player_name: Optional[str] = None
    prop_stat_type: Optional[str] = None
    prop_threshold: Optional[float] = None
    prop_l15_games: int = 0   # number of games in the L15 sample (0 = unknown)
    prop_minutes_volatile: bool = False  # True if player had abnormal recent minutes
    prop_minutes_cv: float = 0.0         # minutes coefficient of variation

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
            return f"{stat} O{thr}"
        if self.market_type == "moneyline":
            return f"{team} ML"
        if self.market_type == "spread" and self.line is not None:
            # Negative line = team wins by |line| (favorite-style cover);
            # positive line = team loses by less than |line| (NO-side / +spread cover).
            sign = "+" if self.line > 0 else ""
            line_str = f"{sign}{self.line:.1f}".rstrip("0").rstrip(".")
            return f"{team} {line_str}"
        if self.market_type == "total" and self.line is not None:
            return f"O/U {self.line:.1f}"
        return team

    @property
    def confidence_stars(self) -> int:
        """0–3 confidence stars based on match quality, volume, and signal quality."""
        stars = 0
        if self.match_confidence >= 0.90:
            stars += 1
        if self.volume_usd >= 5000:
            stars += 1
        # Extra star for multi-book consensus OR model blend
        if self.book_count >= 2 or self.model_sources not in ("sharp", "sharp(capped)"):
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

    # Line velocity thresholds (pp/hr)
    _STEAM_THRESHOLD = 1.5   # |velocity| > this → "STEAM" flag
    _STALE_HOURS = 4         # no line history in this many hours → "STALE"

    def __init__(self) -> None:
        super().__init__()
        self._settings = get_settings()
        self._matching = MatchingEngine()
        self._prop_matcher = PropMatcher()
        self._spread_model = SpreadDistributionModel()
        self._total_model = TotalDistributionModel()
        try:
            from evmax.archiver import DataArchiver
            self._archiver = DataArchiver()
        except Exception:
            self._archiver = None

    async def run(self, request: AgentRequest) -> AgentResponse:
        sector = request.sector
        markets: list[PredictionMarket] = request.params.get("kalshi_markets", [])
        sharp_list: list[SharpOdds] = request.params.get("sharp_odds", [])
        blended_preds: dict = request.params.get("blended_preds", {})
        injuries: dict = request.params.get("injuries", {})
        standings: dict = request.params.get("standings", {})
        playoff_data: dict = request.params.get("playoff_data", {})
        model_sources: dict[str, str] = request.params.get("model_sources", {})
        kelly_base: float = request.params.get("kelly_base_fraction", 0.25)
        steam_events: set[str] = request.params.get("steam_events", set())
        self._possession_sim = request.params.get("possession_sim_agent")

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
            yes_gap, blend_payload = self._evaluate_pair(
                market=market,
                sharp=sharp,
                confidence=confidence,
                sector=sector,
                blended_preds=blended_preds,
                injuries=injuries,
                standings=standings,
                playoff_data=playoff_data,
                model_sources=model_sources,
                kelly_base=kelly_base,
                steam_events=steam_events,
                return_blend=True,
            )
            if yes_gap is not None:
                gaps.append(yes_gap)

            # For spreads, also evaluate the NO side (i.e. the opposing
            # team's "+spread"). NO on a "Team X wins by N+" Kalshi ticker
            # is the only way to get +spread exposure — there is no Kalshi
            # ticker for "Team Y +N.5". Vig guarantees only one of YES / NO
            # can be +EV at a given price/prob, so we never double-count.
            if (
                market.market_type == MarketType.spread
                and market.line is not None
                and blend_payload is not None
            ):
                no_gap = self._build_no_side_spread_gap(
                    market=market,
                    sharp=sharp,
                    blend_payload=blend_payload,
                    confidence=confidence,
                    sector=sector,
                    kelly_base=kelly_base,
                    steam_events=steam_events,
                )
                if no_gap is not None:
                    gaps.append(no_gap)

        for market, sharp, confidence in prop_matched:
            gap = self._evaluate_prop_pair(
                market=market,
                sharp=sharp,
                confidence=confidence,
                sector=sector,
                kelly_base=kelly_base,
                injuries=injuries,
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
    # Line velocity
    # ------------------------------------------------------------------

    def _compute_velocity(self, event_id: str) -> tuple[float | None, str | None]:
        """Compute line velocity (pp/hr) and flag from archived line history.

        Returns (velocity, flag) where flag is "STEAM", "STALE", or None.
        """
        if self._archiver is None:
            return None, None

        try:
            history = self._archiver.get_line_history(event_id, hours=6)
        except Exception:
            return None, None

        if len(history) < 2:
            if not history:
                return None, "STALE"
            return None, None

        # Compute velocity from first to last snapshot
        first_time, first_prob = history[0]
        last_time, last_prob = history[-1]

        try:
            from datetime import datetime as _dt
            t0 = _dt.fromisoformat(first_time)
            t1 = _dt.fromisoformat(last_time)
            hours_elapsed = (t1 - t0).total_seconds() / 3600.0
        except Exception:
            return None, None

        if hours_elapsed < 0.01:
            return 0.0, None

        # Velocity in probability points per hour (multiply by 100 for pp/hr)
        velocity_pp_hr = ((last_prob - first_prob) * 100) / hours_elapsed

        if abs(velocity_pp_hr) > self._STEAM_THRESHOLD:
            return round(velocity_pp_hr, 2), "STEAM"

        # Check staleness: if all snapshots are old (> _STALE_HOURS ago)
        try:
            from datetime import datetime as _dt, timezone as _tz
            now = _dt.now(_tz.utc)
            last_dt = _dt.fromisoformat(last_time)
            if not last_dt.tzinfo:
                from datetime import timezone as _tz2
                last_dt = last_dt.replace(tzinfo=_tz2.utc)
            hours_since_last = (now - last_dt).total_seconds() / 3600.0
            if hours_since_last > self._STALE_HOURS:
                return round(velocity_pp_hr, 2), "STALE"
        except Exception:
            pass

        return round(velocity_pp_hr, 2), None

    # ------------------------------------------------------------------
    # YES-team resolution via market fields
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_yes_via_market_teams(
        yes_team_norm: str,
        market: PredictionMarket,
        sharp: SharpOdds,
        sector: str,
    ) -> Optional[tuple[float, bool]]:
        """Resolve YES alignment using the market's team_home/team_away fields.

        Returns (sharp_true_prob, yes_is_outcome_b) or None if unresolvable.
        """
        from evmax.matching.normalizer import NameNormalizer

        home_raw = (market.team_home or "").lower()
        away_raw = (market.team_away or "").lower()
        if not home_raw or not away_raw:
            return None

        # Strip game-name suffixes Kalshi appends to away team names
        _SUFFIXES = ["league of legends", "valorant", "cs2"]
        for suffix in _SUFFIXES:
            if home_raw.endswith(suffix):
                home_raw = home_raw[: -len(suffix)].strip()
            if away_raw.endswith(suffix):
                away_raw = away_raw[: -len(suffix)].strip()

        def _acronym(name: str) -> str:
            return "".join(w[0] for w in name.split() if w)

        def _matches_team(abbrev: str, team: str) -> bool:
            if abbrev in team or team.startswith(abbrev):
                return True
            if any(w.startswith(abbrev) for w in team.split()):
                return True
            if abbrev == _acronym(team):
                return True
            return False

        # Determine which market team the YES abbreviation belongs to
        yes_is_home: Optional[bool] = None
        home_hit = _matches_team(yes_team_norm, home_raw)
        away_hit = _matches_team(yes_team_norm, away_raw)
        if home_hit and not away_hit:
            yes_is_home = True
        elif away_hit and not home_hit:
            yes_is_home = False

        if yes_is_home is None:
            # Name matching failed — try price-based alignment.
            # Require the closer side to be tight AND meaningfully closer than the
            # farther side, so near-coin-flip markets (where both distances are
            # similar) can never be force-aligned to an arbitrary outcome.
            ask = market.yes_price
            dist_a = abs(ask - sharp.true_prob_a)
            dist_b = abs(ask - sharp.true_prob_b)
            closer = min(dist_a, dist_b)
            gap = abs(dist_a - dist_b)
            if closer > 0.04 or gap < 0.05:
                return None
            if dist_a < dist_b:
                return sharp.true_prob_a, False
            return sharp.true_prob_b, True

        # Normalize the YES team's full name and match to sharp outcomes
        normalizer = NameNormalizer(sector)
        yes_full = home_raw if yes_is_home else away_raw
        yes_norm = normalizer.normalize(yes_full)

        outcome_a_norm = normalizer.normalize(sharp.outcome_a_label or "")
        outcome_b_norm = normalizer.normalize(sharp.outcome_b_label or "")

        from rapidfuzz import fuzz
        score_a = fuzz.token_set_ratio(yes_norm, outcome_a_norm)
        score_b = fuzz.token_set_ratio(yes_norm, outcome_b_norm)

        if score_a > score_b and score_a >= 60:
            return sharp.true_prob_a, False
        elif score_b > score_a and score_b >= 60:
            return sharp.true_prob_b, True

        return None

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
        standings: dict | None = None,
        playoff_data: dict | None = None,
        model_sources: dict[str, str] | None = None,
        kelly_base: float = 0.25,
        steam_events: Optional[set] = None,
        return_blend: bool = False,
    ):
        """Evaluate a Kalshi/Sharp pair for the YES side.

        Default: returns the EVGap or None (legacy).

        When return_blend=True: returns a 2-tuple (gap_or_none, blend_payload).
        blend_payload is None when the YES blend itself could not be computed
        (alignment failed, spread line too far, type mismatch) — caller MUST
        skip NO-side derivation in that case. It is non-None as soon as the
        full blend pipeline (alignment → spread/total dist → model blend →
        drift cap → injuries / standings / playoff) has produced a sound
        blended_prob, even if YES later fails the EV threshold or dead-book
        guard. NO-side derivation is therefore safe to attempt whenever
        blend_payload is not None.
        """
        def _ret(gap, payload):
            return (gap, payload) if return_blend else gap

        if market.yes_price <= 0 or market.yes_price >= 1.0:
            return _ret(None, None)

        # ------------------------------------------------------------------
        # Type-mismatch guards: incompatible market/sharp combinations
        # ------------------------------------------------------------------
        is_total = market.market_type == MarketType.total
        is_sharp_total = sharp.total_line is not None

        # Totals market must match totals sharp record and vice versa
        if is_total and not is_sharp_total:
            return _ret(None, None)
        if not is_total and is_sharp_total:
            return _ret(None, None)

        # Moneyline must not match spread record
        if market.market_type == MarketType.moneyline and sharp.spread_line is not None:
            return _ret(None, None)

        # Reject team scoring props mistakenly typed as totals (e.g. "Lakers O109.5")
        if is_total and market.line is not None and not is_game_total(market.line, sector):
            return _ret(None, None)

        # ------------------------------------------------------------------
        # Step 1: YES-team alignment (which sharp prob belongs to the YES side?)
        # ------------------------------------------------------------------
        yes_team_norm = (market.yes_team or "").lower().strip()
        outcome_a_norm = (sharp.outcome_a_label or "").lower().strip()
        outcome_b_norm = (sharp.outcome_b_label or "").lower().strip()

        is_draw = yes_team_norm in ("tie", "draw", "x", "draw/tie")

        def _yes_matches(yes: str, candidate: str) -> bool:
            """Check if YES team name matches a Pinnacle outcome label.

            Uses fuzzy matching (rapidfuzz token_set_ratio) instead of raw substring
            to avoid false positives like 'm' in 'hoffenheim' or 'oconnell' missing
            "o'connell". Short names (<3 chars) require the candidate to START with
            them as a word boundary (e.g. 'm' only matches 'mainz', not 'hoffenheim').
            """
            if not yes or not candidate:
                return False
            # Strip apostrophes/hyphens for comparison
            y = yes.replace("'", "").replace("\u2019", "").replace("-", "")
            c = candidate.replace("'", "").replace("\u2019", "").replace("-", "")
            # Exact match (post-strip)
            if y == c:
                return True
            # Short names (<3 chars): require word-boundary prefix match
            if len(y) < 3:
                words = c.split()
                return any(w.startswith(y) for w in words) and not any(
                    w.startswith(y) for w in (outcome_b_norm if candidate == outcome_a_norm else outcome_a_norm).split()
                )
            # Substring match (only if yes_team is 3+ chars to avoid spurious hits)
            if y in c:
                return True
            # Fuzzy fallback for names with punctuation differences
            from rapidfuzz import fuzz
            return fuzz.token_set_ratio(y, c) >= 80

        # Totals: YES side is "over" or "under" — not a team name
        if is_total:
            yes_is_under = yes_team_norm == "under"
            if sharp.true_prob_over is None or sharp.true_prob_under is None:
                return _ret(None, None)
            sharp_true_prob = sharp.true_prob_under if yes_is_under else sharp.true_prob_over
            yes_is_outcome_b = False
        elif is_draw:
            if sharp.true_prob_draw is None:
                return _ret(None, None)
            sharp_true_prob = sharp.true_prob_draw
            yes_is_outcome_b = False
            yes_is_under = False
        elif _yes_matches(yes_team_norm, outcome_a_norm):
            # YES team matches outcome_a (home/favorite)
            sharp_true_prob = sharp.true_prob_a
            yes_is_outcome_b = False
            yes_is_under = False
        elif _yes_matches(yes_team_norm, outcome_b_norm):
            # YES team matches outcome_b (away/underdog)
            sharp_true_prob = sharp.true_prob_b
            yes_is_outcome_b = True
            yes_is_under = False
        else:
            # Fallback: match YES team via market's own home/away fields.
            # Kalshi esports use short tickers ("th", "dsg") that can't match
            # Pinnacle labels directly. Determine which market team is YES,
            # normalize it, then match against sharp outcomes.
            resolved = self._resolve_yes_via_market_teams(
                yes_team_norm, market, sharp, sector,
            )
            if resolved is not None:
                sharp_true_prob, yes_is_outcome_b = resolved
            else:
                sharp_true_prob = sharp.true_prob_a
                yes_is_outcome_b = False
            yes_is_under = False

        # ------------------------------------------------------------------
        # Step 2a: Spread markets — blend Pinnacle CDF with sim distribution
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

                # Blend with PossessionSim margin distribution for NBA
                sim = getattr(self, "_possession_sim", None)
                if sim is not None and sector == "nba":
                    sim_prob = sim.cover_probability(
                        sharp.event_id, market.line, yes_is_underdog=yes_is_outcome_b,
                    )
                    if sim_prob is not None:
                        sim_weight = 0.35
                        sharp_true_prob = (1.0 - sim_weight) * sharp_true_prob + sim_weight * sim_prob
                        used_spread_model = True
            else:
                # Spread distribution model bailed (line too far) — blend is
                # unreliable for both YES and NO. Skip the NO-side too.
                return _ret(None, None)

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

                sim = getattr(self, "_possession_sim", None)
                if sim is not None and sector == "nba":
                    sim_prob = sim.total_probability(
                        sharp.event_id, market.line, is_over=not yes_is_under,
                    )
                    if sim_prob is not None:
                        sim_weight = 0.35
                        sharp_true_prob = (1.0 - sim_weight) * sharp_true_prob + sim_weight * sim_prob
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
                sim = getattr(self, "_possession_sim", None)
                has_sim = sim is not None and sharp.event_id in getattr(sim, "_margin_cache", {})
                src = "sharp+spread_dist+possession_sim" if has_sim and sector == "nba" else "sharp+spread_dist"
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
        # Step 4b: Standings / rest-risk adjustment
        # ------------------------------------------------------------------
        if standings and not is_total:
            from evmax.agents.intelligence.standings_agent import StandingsAgent
            team_a = (sharp.outcome_a_label or "").lower()
            team_b = (sharp.outcome_b_label or "").lower()
            new_a, new_b, rest_notes = StandingsAgent.apply_adjustments(
                standings=standings,
                true_prob_a=blended_prob if not yes_is_outcome_b else 1.0 - blended_prob,
                true_prob_b=blended_prob if yes_is_outcome_b else 1.0 - blended_prob,
                team_a=team_a,
                team_b=team_b,
            )
            if rest_notes:
                blended_prob = new_b if yes_is_outcome_b else new_a
                src = f"{src}+rest"

        # ------------------------------------------------------------------
        # Step 4c: Playoff adjustment — series score, elimination, home court
        # ------------------------------------------------------------------
        if playoff_data:
            from evmax.agents.intelligence.playoff_agent import PlayoffAgent
            team_a = (sharp.outcome_a_label or "").lower()
            team_b = (sharp.outcome_b_label or "").lower()
            new_a, new_b, playoff_notes, is_playoff = PlayoffAgent.apply_adjustments(
                playoff_data=playoff_data,
                true_prob_a=blended_prob if not yes_is_outcome_b else 1.0 - blended_prob,
                true_prob_b=blended_prob if yes_is_outcome_b else 1.0 - blended_prob,
                team_a=team_a,
                team_b=team_b,
                is_total=is_total,
            )
            if playoff_notes:
                blended_prob = new_b if yes_is_outcome_b else new_a
                src = f"{src}+playoff"

        # ------------------------------------------------------------------
        # Blend pipeline complete — capture the payload that the NO-side
        # spread derivation needs. From this point on, blend_payload is
        # available even if the YES side fails the dead-book or ev_floor
        # filters below.
        # ------------------------------------------------------------------
        blend_payload = {
            "blended_prob_yes": blended_prob,
            "sharp_true_prob_yes": sharp_true_prob,
            "src": src,
            "yes_is_outcome_b": yes_is_outcome_b,
            "used_spread_model": used_spread_model,
        }

        # ------------------------------------------------------------------
        # Step 5: EV and Kelly sizing
        # ------------------------------------------------------------------
        # Filter out dead orderbooks: 99¢ asks are Kalshi's default when all
        # orders have been pulled (common for live/in-game markets).
        if market.yes_price >= 0.99:
            return _ret(None, blend_payload)

        # Phantom-EV guard for esports: opaque Kalshi ticker codes ("th",
        # "dcg", "shg") can defeat the YES-alignment resolver and historically
        # assigned the opposite team's devigged prob to the YES side — the
        # 442%/107%/53% LoL bugs on 2026-04-12. Signature: sharp-only blend
        # where blended_prob mirrors (1 − market.yes_price) with a large face
        # edge. Scoped to esports because team-sport markets have reliable
        # name matching and symmetric mispricings there are real edges.
        is_sharp_only = src in ("sharp", "sharp(capped)") or src.startswith("sharp+")
        if is_sharp_only and (market.sector or "").lower() in ("lol", "cs2", "valorant"):
            mirror_distance = abs(blended_prob - (1.0 - market.yes_price))
            face_edge = abs(blended_prob - market.yes_price)
            if mirror_distance < 0.05 and face_edge > 0.15:
                # Alignment is suspect — skip both YES and NO.
                return _ret(None, None)

        ev, edge_pct = calculate_ev(market.yes_price, blended_prob)

        # Sharp-only signals need a higher EV bar — vig removal alone can
        # produce phantom 2-3% edges that don't hold empirically.
        ev_floor = self._settings.ev_threshold
        sector_lower = (market.sector or "").lower()
        if sector_lower == "tennis" and src == "sharp":
            ev_floor = 0.05  # Tennis sharp-only needs bigger edge
        elif sector_lower == "baseball" and src == "sharp":
            ev_floor = 0.05  # Baseball sharp-only needs bigger edge (stale model data)

        if ev < ev_floor:
            return _ret(None, blend_payload)

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

        # Line velocity from archived sharp odds history
        velocity, vel_flag = self._compute_velocity(sharp.event_id)
        # Legacy steam detection also sets steam flag
        if vel_flag == "STEAM":
            is_steam = True

        gap = EVGap(
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
            line_velocity=velocity,
            velocity_flag=vel_flag,
            book_count=getattr(sharp, "book_count", 1),
        )
        return _ret(gap, blend_payload)

    # ------------------------------------------------------------------
    # NO-side spread derivation
    # ------------------------------------------------------------------

    def _build_no_side_spread_gap(
        self,
        market: PredictionMarket,
        sharp: SharpOdds,
        blend_payload: dict,
        confidence: float,
        sector: str,
        kelly_base: float = 0.25,
        steam_events: Optional[set] = None,
    ) -> Optional[EVGap]:
        """Build the NO-side EVGap for a Kalshi spread market.

        Kalshi spread tickers are phrased as "Team X wins by over N.5". The
        NO side of such a ticker is the opposing team's "+spread" cover —
        the only way to get +spread exposure on Kalshi. We synthesize an
        EVGap as if it were a YES bet on the opponent at a positive line:

          - market_id : original_id + ":no"  (avoids UNIQUE collision)
          - yes_team  : opponent label
          - line      : -market.line  (positive value, e.g. +9.5)
          - kalshi_yes_price : market.no_price (actual NO ask, includes vig)
          - true_prob : 1 - blended_yes_prob

        Returns None if NO is below EV threshold or NO orderbook is dead.
        """
        # Determine the actual NO ask. Kalshi reports yes_price + no_price
        # separately (sum > 1 due to vig). market.no_price is set by the
        # Kalshi parser; only fall back if missing.
        no_ask = market.no_price
        if no_ask is None or no_ask <= 0 or no_ask >= 1.0:
            no_ask = 1.0 - market.yes_price
        if no_ask <= 0.01 or no_ask >= 0.99:
            return None  # dead orderbook on the NO side

        blended_no = max(0.01, min(0.99, 1.0 - blend_payload["blended_prob_yes"]))
        sharp_no = max(0.01, min(0.99, 1.0 - blend_payload["sharp_true_prob_yes"]))

        ev, edge_pct = calculate_ev(no_ask, blended_no)

        # Use the same EV floor logic as the YES side. NO bets carry the
        # same sharp-only risk profile so the elevated floor for tennis /
        # baseball sharp-only signals applies symmetrically.
        src_yes = blend_payload["src"]
        ev_floor = self._settings.ev_threshold
        sector_lower = (market.sector or "").lower()
        if sector_lower == "tennis" and src_yes == "sharp":
            ev_floor = 0.05
        elif sector_lower == "baseball" and src_yes == "sharp":
            ev_floor = 0.05
        if ev < ev_floor:
            return None

        # Opponent name = whichever sharp outcome the YES side did NOT cover.
        opp_label = (
            sharp.outcome_a_label if blend_payload["yes_is_outcome_b"]
            else sharp.outcome_b_label
        )
        opp_team = (opp_label or "?").lower().strip() or "?"

        payout = 1.0 / no_ask
        kelly = compute_kelly(
            true_prob=blended_no,
            payout_decimal=payout,
            edge_pct=edge_pct,
            spread_pct=market.spread_pct,
            base_fraction=kelly_base,
            max_kelly=self._settings.max_kelly_fraction,
        )

        is_steam = bool(steam_events and sharp.event_id in steam_events)
        velocity, vel_flag = self._compute_velocity(sharp.event_id)
        if vel_flag == "STEAM":
            is_steam = True

        # Flip sign so the line stored is positive (e.g. -9.5 → +9.5)
        no_line = -market.line if market.line is not None else None

        return EVGap(
            market_id=f"{market.id}:no",
            event_id=sharp.event_id,
            sector=sector,
            yes_team=opp_team,
            market_type=MarketType.spread.value,
            kalshi_yes_price=no_ask,
            sharp_true_prob=sharp_no,
            blended_true_prob=blended_no,
            ev_pct=ev,
            kelly_full=kelly.kelly_full,
            kelly_fraction=kelly.kelly_fraction,
            match_confidence=confidence,
            volume_usd=market.volume_usd,
            spread_pct=market.spread_pct,
            event_date=sharp.event_date or market.event_date,
            line=no_line,
            model_sources=f"{src_yes}+no_side",
            event_title=f"{sharp.outcome_a_label or '?'} vs {sharp.outcome_b_label or '?'}",
            steam_move=is_steam,
            line_velocity=velocity,
            velocity_flag=vel_flag,
            book_count=getattr(sharp, "book_count", 1),
        )

    def _evaluate_prop_pair(
        self,
        market: PredictionMarket,
        sharp: SharpOdds,
        confidence: float,
        sector: str,
        kelly_base: float = 0.25,
        injuries: dict | None = None,
    ) -> Optional[EVGap]:
        """Evaluate a matched player prop pair for EV.

        YES side on Kalshi = player goes OVER the threshold.

        DISABLED (Apr 2026): Backtest on 143 resolved prop observations showed
        the L15-based model (Brier 0.253) loses to predicting 37% for everything
        (Brier 0.233). Kalshi only offers overs, which are systematically
        overpriced (62c implied, 37% actual). The model computes over-probability
        from the same L15 stats the line-setters already know — zero information
        edge. Non-volatile qualifying bets went 3/20 (-71% ROI).

        Props are still logged to prop_observations for calibration data
        via log_prop_from_sharp() in the CLI layer (uses raw SharpOdds pairs).
        Re-enable when we have a model with genuine edge (real-time lineup
        data, closing-line-value arbitrage, or cross-book sharp lines).
        """
        return None

        sharp_true_prob = sharp.true_prob_over
        src = "nba_stats"

        # Injury boost: when TEAMMATES are OUT, this player absorbs usage.
        #
        # We look up the player's team directly from the NBA props cache rather
        # than trying to parse a game slug out of the event_id — prop event_ids
        # are `{sector}::{date}::prop::{player}::{stat}::{threshold}` and carry
        # no game slug, so the old `parts[2]` approach always produced "prop"
        # and silently skipped the boost.
        injury_boost = 0.0
        if injuries and market.player_name:
            try:
                from evmax.agents.intelligence.injury_agent import InjuryReportAgent
                from evmax.clients.nba_props_cache import lookup_player_team

                player_name_norm = market.player_name.lower().replace("_", " ")
                team_str = lookup_player_team(market.player_name)

                if team_str:
                    out_players = InjuryReportAgent.get_out_players(injuries, team_str)
                    if out_players:
                        # Don't boost if our prop player is themselves OUT.
                        player_is_out = any(
                            player_name_norm in op["name"].lower()
                            or op["name"].lower() in player_name_norm
                            for op in out_players
                        )
                        if not player_is_out:
                            injury_boost = InjuryReportAgent.compute_prop_injury_boost(
                                injuries, team_str
                            )

                if injury_boost > 0:
                    sharp_true_prob = min(0.95, sharp_true_prob * (1 + injury_boost))
                    src = f"nba_stats+inj({injury_boost:.0%})"
            except Exception:
                pass

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

        return EVGap(
            market_id=market.id,
            event_id=sharp.event_id,
            sector=sector,
            yes_team=market.player_name or "?",
            market_type=MarketType.player_prop.value,
            kalshi_yes_price=market.yes_price,
            sharp_true_prob=sharp.true_prob_over,  # original sharp prob (pre-boost)
            blended_true_prob=sharp_true_prob,      # after injury boost
            ev_pct=ev,
            kelly_full=kelly.kelly_full,
            kelly_fraction=kelly.kelly_fraction,
            match_confidence=confidence,
            volume_usd=market.volume_usd,
            spread_pct=market.spread_pct,
            event_date=sharp.event_date or market.event_date,
            line=market.threshold,
            model_sources=src,
            event_title=player_display,
            prop_player_name=market.player_name,
            prop_stat_type=market.stat_type,
            prop_threshold=market.threshold,
            prop_l15_games=sharp.prop_l15_games,
            prop_minutes_volatile=sharp.prop_minutes_volatile,
            prop_minutes_cv=sharp.prop_minutes_cv,
        )
