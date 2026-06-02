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
    # Scan timing — minutes between scan time and tipoff. Drives the CLV
    # stratification in scripts/clv_report.py: late scans (<30 min pre-tipoff)
    # are when late-news edge can manifest; early scans (>6h) measure mostly
    # post-scan market drift and are inherently noisy regardless of news flow.
    minutes_to_tipoff: Optional[int] = None

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


def _minutes_to_tipoff(event_date: Optional[datetime]) -> Optional[int]:
    """Minutes between scan time and event start, or None if unknown.

    Negative values are clamped to 0 — happens occasionally when scans
    fire after a game already started but Kalshi still has the market
    listed (pre-tipoff exclusives that resolved late, sport rescheduled,
    etc.). For CLV stratification we only care about pre-tipoff windows.
    """
    if event_date is None:
        return None
    now = datetime.now(timezone.utc)
    et = event_date if event_date.tzinfo else event_date.replace(tzinfo=timezone.utc)
    delta = (et - now).total_seconds() / 60.0
    return max(0, int(delta))


def _dedup_mutually_exclusive_sides(gaps: "list[EVGap]") -> "list[EVGap]":
    """Drop the lower-EV gap when two bets cover opposite sides of a market.

    A Kalshi market quotes YES on each side independently — same event_id +
    market_type + line, different yes_team. The outcomes are mutually
    exclusive (only one can win), so logging both is a vig double-pay with
    no net edge. Keep the higher-EV side per ``(event_id, market_type,
    |line|)`` group. Player props are exempt: Kalshi posts each threshold
    (e.g. 20+ / 25+ / 30+ points for the same player) as a separate market,
    and those are independent bets, not opposite sides of the same market.
    """
    best: dict[tuple, "EVGap"] = {}
    pass_through: list["EVGap"] = []
    for g in gaps:
        if g.market_type == "player_prop":
            pass_through.append(g)
            continue
        line_key = abs(g.line) if g.line is not None else 0.0
        key = (g.event_id, g.market_type, line_key)
        prev = best.get(key)
        if prev is None or (g.ev_pct or 0) > (prev.ev_pct or 0):
            best[key] = g
    return list(best.values()) + pass_through


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
        self._wnba_possession_sim = request.params.get("wnba_possession_sim_agent")

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

            # Totals are framed YES = OVER on Kalshi, so the UNDER is only
            # reachable as the NO side — mirror the spread NO-side derivation.
            if (
                market.market_type == MarketType.total
                and market.line is not None
                and blend_payload is not None
            ):
                no_gap = self._build_no_side_total_gap(
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

        # No-middling dedup: when Kalshi's vig is wide enough that BOTH
        # sides of a market (Lakers ML vs Warriors ML; Lakers -7.5 vs
        # Warriors +7.5; Over vs Under same line) pass our EV threshold,
        # we used to log both. The outcomes are mutually exclusive so
        # we're effectively arbing against ourselves and paying Kalshi
        # vig twice. Industry-standard sharp-bettor convention (OddsJam,
        # Unabated): keep only the higher-EV side per (event, market_type,
        # |line|) group. Player props are exempt — different thresholds on
        # the same player+stat are genuinely independent bets (Kalshi
        # posts 20+ / 25+ / 30+ as separate markets, none are "opposites").
        gaps = _dedup_mutually_exclusive_sides(gaps)
        gaps.sort(key=lambda g: g.ev_pct, reverse=True)

        self.log.info("ev_gaps_found", sector=sector, count=len(gaps))

        await self.publish(f"ev.gaps.{sector}", gaps, request.correlation_id)

        return AgentResponse(
            agent_name=self.name,
            sector=sector,
            data=gaps,
        )

    # ------------------------------------------------------------------
    # Possession-sim dispatch
    # ------------------------------------------------------------------

    def _sim_for_sector(self, sector: str):
        """Return the possession-sim agent for a given sector, or None.

        NBA and WNBA each have their own possession-sim agent with
        sector-tuned constants (HOME_EDGE_PTS, SCORE_STDEV, pace clip).
        Other sectors don't have a sim integration and return None.
        """
        s = (sector or "").lower()
        if s == "nba":
            return getattr(self, "_possession_sim", None)
        if s == "wnba":
            return getattr(self, "_wnba_possession_sim", None)
        return None

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
                used_spread_model = True

                # Blend with PossessionSim margin distribution for NBA/WNBA.
                # The sim's `cover_probability` is computed from the raw
                # outcome_a-perspective margin distribution; we MUST pass
                # the original `yes_is_outcome_b` so it flips correctly when
                # YES is the underdog. (Earlier a post-reset value of False
                # silently fed the favorite's cover prob into underdog-cover
                # spreads — inflating EVs by 30+pp.)
                sim = self._sim_for_sector(sector)
                if sim is not None:
                    sim_prob = sim.cover_probability(
                        sharp.event_id, market.line, yes_is_underdog=yes_is_outcome_b,
                    )
                    if sim_prob is not None:
                        sim_weight = 0.35
                        sharp_true_prob = (1.0 - sim_weight) * sharp_true_prob + sim_weight * sim_prob
                        used_spread_model = True

                # Do NOT reset yes_is_outcome_b here. sharp_true_prob is now
                # the YES-side cover prob (already aligned). Step 3 (model
                # blend) is guarded by skip_blend=True for spreads, so no
                # double-swap risk. Steps 4–4c (injury / standings / playoff)
                # do their own orientation flip via:
                #   true_prob_a = blended_prob if not yes_is_outcome_b else 1 - blended_prob
                # which only puts the YES cover prob on the right team if
                # yes_is_outcome_b reflects the actual alignment. Forcing it
                # to False here previously caused underdog-cover injury
                # adjustments to flow in the WRONG direction (e.g., PHI
                # injured → P(PHI covers) went UP).
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

                sim = self._sim_for_sector(sector)
                if sim is not None:
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
                sim = self._sim_for_sector(sector)
                has_sim = sim is not None and sharp.event_id in getattr(sim, "_margin_cache", {})
                src = "sharp+spread_dist+possession_sim" if has_sim else "sharp+spread_dist"
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
        # Step 4: Injury adjustment.
        # Skipped for:
        #   - totals: injury impact on team scoring is symmetric in O/U math
        #   - spreads: Pinnacle's spread line already prices in known injuries
        #     (sharp book, fast reaction, insider networks). Adding ESPN-derived
        #     adjustments on top double-counts the signal AND uses moneyline-
        #     shaped impact (zero-sum, cap ±10pp) on a tail probability where
        #     the relationship is non-linear. See commit `__PATH_B__` for the
        #     Cavaliers +9.5 case study where this layer drove EV from +1.3%
        #     (real model edge) to phantom +12.9%.
        # ------------------------------------------------------------------
        if injuries and not is_total and not used_spread_model:
            from evmax.agents.intelligence.injury_agent import InjuryReportAgent
            team_a = (sharp.outcome_a_label or "").lower()
            team_b = (sharp.outcome_b_label or "").lower()
            new_a, new_b, inj_notes = InjuryReportAgent.apply_adjustments(
                reports=injuries,
                true_prob_a=blended_prob if not yes_is_outcome_b else 1.0 - blended_prob,
                true_prob_b=blended_prob if yes_is_outcome_b else 1.0 - blended_prob,
                team_a=team_a,
                team_b=team_b,
                # spread_multiplier=2.0 was redundant under the old guard
                # (the line was always reached with used_spread_model True).
                # With Path B it can never fire, so default 1.0 is fine.
            )
            if inj_notes:
                blended_prob = new_b if yes_is_outcome_b else new_a
                src = f"{src}+injury"

            # Late-news edge detector. Pinnacle's algorithm typically adjusts
            # on injury news within minutes, but Kalshi's retail-driven market
            # can lag by hours. If a starter/star on either team had a status
            # change reported in the last 6 hours AND the change is OUT or
            # DOUBTFUL (the moves that actually shift lines), tag the gap so
            # post-hoc CLV analysis can isolate whether we're capturing the
            # edge there. Threshold from Pikkit/Unabated guidance: 6 hours
            # is the window where soft books historically haven't caught up.
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
            late_news = False
            for tname in (team_a, team_b):
                report = injuries.get(tname)
                if not report or not report.has_significant_out:
                    continue
                stamp = report.latest_report_at
                if stamp and stamp >= cutoff:
                    late_news = True
                    break
            if late_news:
                src = f"{src}+late_news"

        # ------------------------------------------------------------------
        # Step 4b: Standings / rest-risk adjustment. Skipped for totals and
        # spreads (Pinnacle's line prices rest/standings already).
        # ------------------------------------------------------------------
        if standings and not is_total and not used_spread_model:
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
        # Step 4c: Playoff adjustment — series score, elimination, home court.
        # Skipped for spreads (Pinnacle's spread line already prices in
        # playoff context — series-trailing teams' urgency, home court, etc.).
        # Still applies to totals (different shape — affects pace/scoring,
        # not relative team strength).
        # ------------------------------------------------------------------
        if playoff_data and not used_spread_model:
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
        # orders have been pulled (common for live/in-game markets). The 1¢
        # floor is the symmetric case — when all YES bids are pulled, ask
        # collapses to the minimum tick. We saw this on 3 settled tennis
        # markets where blended prob (~0.6-0.8) vs 1¢ price produced bogus
        # 5000-7000% EVs (commit 3de2c26 diagnostic).
        if market.yes_price >= 0.99 or market.yes_price <= 0.01:
            return _ret(None, blend_payload)

        # Heavy-chalk guard: skip $22-to-win-$2 bets even when +EV. blend_payload
        # is preserved so the NO side (which would NOT be chalk in this case)
        # can still derive normally.
        if market.yes_price > self._settings.chalk_price_ceiling:
            return _ret(None, blend_payload)

        # Filter out empty/stale orderbooks. spread_pct = |yes_ask - (1 -
        # no_ask)| / mid. A real two-sided book always sums to >1.0 due to
        # vig, so spread_pct < 0.5% means we're seeing a one-sided ladder,
        # last-trade fallback, or computed complement — not fillable
        # liquidity. volume_usd is CUMULATIVE LIFETIME (not current depth)
        # and can show $thousands on a market with no live quotes, which
        # is why we use spread_pct, not volume, as the floor.
        if market.spread_pct < 0.005:
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
            event_date=sharp.event_date or market.event_date,\
            minutes_to_tipoff=_minutes_to_tipoff(sharp.event_date or market.event_date),
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

        # Same empty-book filter as the YES side: spread_pct < 0.5% means
        # the book isn't really two-sided, even if cumulative volume looks
        # healthy. Don't surface a NO bet against a stale ladder.
        if market.spread_pct < 0.005:
            return None

        blended_no = max(0.01, min(0.99, 1.0 - blend_payload["blended_prob_yes"]))
        sharp_no = max(0.01, min(0.99, 1.0 - blend_payload["sharp_true_prob_yes"]))

        ev, edge_pct = calculate_ev(no_ask, blended_no)

        # NO-side spreads run a 5% EV floor regardless of sector / model source.
        # Normal-CDF cover-prob conversion under-estimates fat tails on alt-line
        # covers, so the scanner's standard 2% threshold mostly surfaces tail
        # mispricing rather than real edge. Backtest 2026-05-07 (sigma=14
        # sharp-only, archive sample, scripts/backtest_no_side_spreads.py):
        # EV>=2% NO-side ROI -11c/$1; EV>=5% cleared breakeven.
        ev_floor = max(self._settings.ev_threshold, 0.05)
        if ev < ev_floor:
            return None
        src_yes = blend_payload["src"]

        # Opponent name = whichever sharp outcome the YES side did NOT cover.
        # Normalize to the canonical short form (e.g. "Cleveland Cavaliers"
        # → "cavaliers") so it matches how YES-side rows store yes_team
        # (parsed from the Kalshi ticker via the same NameNormalizer).
        opp_label = (
            sharp.outcome_a_label if blend_payload["yes_is_outcome_b"]
            else sharp.outcome_b_label
        )
        from evmax.matching.normalizer import NameNormalizer
        opp_team = NameNormalizer(sector).normalize(opp_label or "") or (
            (opp_label or "?").lower().strip() or "?"
        )

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
            event_date=sharp.event_date or market.event_date,\
            minutes_to_tipoff=_minutes_to_tipoff(sharp.event_date or market.event_date),
            line=no_line,
            model_sources=f"{src_yes}+no_side",
            event_title=f"{sharp.outcome_a_label or '?'} vs {sharp.outcome_b_label or '?'}",
            steam_move=is_steam,
            line_velocity=velocity,
            velocity_flag=vel_flag,
            book_count=getattr(sharp, "book_count", 1),
        )

    # ------------------------------------------------------------------
    # NO-side total derivation
    # ------------------------------------------------------------------

    def _build_no_side_total_gap(
        self,
        market: PredictionMarket,
        sharp: SharpOdds,
        blend_payload: dict,
        confidence: float,
        sector: str,
        kelly_base: float = 0.25,
        steam_events: Optional[set] = None,
    ) -> Optional[EVGap]:
        """Build the NO-side EVGap for a Kalshi total market.

        Kalshi totals tickers are framed YES = OVER (``kalshi.py`` sets
        ``yes_team = "over"`` for every total series), so the NO side is the
        UNDER — the only way to get under exposure on Kalshi. We synthesize an
        EVGap as if it were a YES bet on the under at the same line:

          - market_id : original_id + ":no"  (avoids UNIQUE collision)
          - yes_team  : "under"
          - line      : market.line  (totals line is side-agnostic, NOT flipped)
          - kalshi_yes_price : market.no_price (actual UNDER ask, includes vig)
          - true_prob : 1 - blended_over_prob

        Returns None if UNDER is below the EV threshold or the NO book is dead.
        Mirrors :meth:`_build_no_side_spread_gap`; vig guarantees only one of
        OVER (YES) / UNDER (NO) can be +EV at a given price, so no double-count.
        """
        no_ask = market.no_price
        if no_ask is None or no_ask <= 0 or no_ask >= 1.0:
            no_ask = 1.0 - market.yes_price
        if no_ask <= 0.01 or no_ask >= 0.99:
            return None  # dead orderbook on the UNDER side

        if market.spread_pct < 0.005:
            return None  # one-sided / stale ladder

        blended_under = max(0.01, min(0.99, 1.0 - blend_payload["blended_prob_yes"]))
        sharp_under = max(0.01, min(0.99, 1.0 - blend_payload["sharp_true_prob_yes"]))

        ev, edge_pct = calculate_ev(no_ask, blended_under)
        if ev < self._settings.ev_threshold:
            return None
        src_yes = blend_payload["src"]

        payout = 1.0 / no_ask
        kelly = compute_kelly(
            true_prob=blended_under,
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

        return EVGap(
            market_id=f"{market.id}:no",
            event_id=sharp.event_id,
            sector=sector,
            yes_team="under",
            market_type=MarketType.total.value,
            kalshi_yes_price=no_ask,
            sharp_true_prob=sharp_under,
            blended_true_prob=blended_under,
            ev_pct=ev,
            kelly_full=kelly.kelly_full,
            kelly_fraction=kelly.kelly_fraction,
            match_confidence=confidence,
            volume_usd=market.volume_usd,
            spread_pct=market.spread_pct,
            event_date=sharp.event_date or market.event_date,
            minutes_to_tipoff=_minutes_to_tipoff(sharp.event_date or market.event_date),
            line=market.line,
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

        YES side on Kalshi = player goes OVER the threshold. The probability
        on ``sharp.true_prob_over`` comes from anchor pricing in
        :mod:`evmax.ev.prop_pricing` — Pinnacle posts a single half-point line
        per (player, stat) and the pricing module fits a Normal (high-mean
        stats) or Negative Binomial (low-mean count stats) to that anchor,
        then reads off P(stat >= K) for the Kalshi threshold ``K``.

        Bankroll safety: production prop categories sit in ``mode='shadow'``
        per ``data/categories.yaml``; shadow rows skip Kelly bankroll
        accounting in the logger. Flipping a prop category from shadow → live
        is the single point where real money becomes at risk.

        Pre-2026-05-10 this returned ``None`` unconditionally — the old L15
        model that fed this evaluator was structurally bad (Brier 0.253 vs
        0.233 baseline). Anchor pricing replaced L15; this evaluator is now
        the standard School-1 (sharp-anchor + devig + threshold compare)
        path, uniform with game-market evaluation.
        """
        sharp_true_prob = sharp.true_prob_over
        if sharp_true_prob is None:
            return None
        src = "pinnacle_anchor"

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
                    src = f"pinnacle_anchor+inj({injury_boost:.0%})"
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
            event_date=sharp.event_date or market.event_date,\
            minutes_to_tipoff=_minutes_to_tipoff(sharp.event_date or market.event_date),
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
