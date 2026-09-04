"""FormModelAgent — recent-form probability model.

Uses each team's last N games to compute a form-adjusted win probability.

Algorithm:
  1. Retrieve last N results for team_a and team_b.
  2. Compute exponentially-weighted win rates:
       w_i = decay^(N-1-i)   (most-recent game weighted highest)
       form_rate = Σ(result_i × w_i) / Σ w_i
  3. Convert to head-to-head probability via log5 formula:
       P(A beats B) = (form_a - form_a×form_b) / (form_a + form_b - 2×form_a×form_b)
  4. Apply home-court/field adjustment (additive on probability).

State file: data/models/form_state.json
  {
    "nba": {
      "lakers": [
        {"date": "2026-02-10", "won": true, "opp": "bulls", "home": true},
        ...
      ],
      ...
    }
  }

Seeding:
  Import recent box-score data as a list of game dicts.
  Use seed_results() to bulk load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

# Exponential decay factor (per game going back in time)
DECAY: float = 0.85

# Window of recent games to consider
WINDOW: int = 10

# Skip the form model when the most recent record for either team is
# older than this many days. Protects against carrying stale prior-season
# form into a new season's opening games (see WNBA offseason case where
# October 2025 records were being read as May 2026 signal).
STALE_DAYS: int = 60

# Home-side probability bonus (additive)
HOME_ADJ: dict[str, float] = {
    "nfl": 0.03,
    "nba": 0.04,
    "ncaab": 0.05,
    "soccer": 0.04,
    "baseball": 0.04,
    "wnba": 0.025,    # ~54% home win rate (weaker than NBA)
    "ufc": 0.0,   # neutral venue
    "f1": 0.0,    # different circuit each race
    "lol": 0.0,
    "cs2": 0.0,
}

# Minimum games to produce any prediction
MIN_GAMES: int = 3

# Bayesian shrinkage prior — virtual game count pulling form_rate toward 0.5.
# Addresses the small-sample overconfidence visible in walk-forward (Form's
# 70-80% bucket predicted 75% but actual was 55%, and 80-100% bucket predicted
# 85% but actual was 66% — extreme streaks of 3-5 games spuriously pushed
# form to the rails). With PRIOR_N=6 and WINDOW=10, a max-window team sees
# shrinkage = 10/(10+6) = 0.625 toward the 0.5 baseline.
SHRINKAGE_PRIOR_N: float = 6.0
SHRINKAGE_BASELINE: float = 0.5

# Margin-based form. Win/loss form throws the score away: over a 17-game NFL
# season a 3-point loss and a 30-point loss are the same record. For sectors
# listed here, when EVERY in-window record carries a `margin`, form is the
# exponentially-weighted average point margin (home-edge adjusted), the
# difference shrunk toward 0, converted to P(home) through the sector's
# score stdev. Any in-window record without a margin (legacy rows, or a
# sector not listed) falls back to W/L form unchanged.
#
# NFL: backtest-REJECTED 2026-09-03 (scripts/backtest_nfl_efficiency.py
# --form-shrink, walk-forward 2324/2425/2526, W/L form vs margin form inside
# the shipped P2 v1 blend). Margin form helped 2023-24 at every shrink
# (0.2377 → 0.234x) but HURT 2024-25 and the 2025-26 holdout at every
# shrink swept: shrink 0.3 / 0.5 / 0.7 / 1.0 → holdout blend Brier
# 0.2252 / 0.2226 / 0.2209 / 0.2197 vs 0.2194 with W/L form; 2024-25
# 0.2164 / 0.2142 / 0.2127 / 0.2113 vs 0.2110. W/L form's Bayesian
# sample-size shrinkage toward 0.5 is doing real work that a scale-only
# shrink on the margin does not replicate; an n-dependent shrink was NOT
# tried (one change per iteration) and is the follow-up if this is revisited.
# So the set ships EMPTY: records still carry `margin` (cheap, and it is
# what any future margin model needs), and the harness toggles the sector
# in to score the `form_margin` column.
MARGIN_FORM_SECTORS: set[str] = set()
MARGIN_FORM_PARAMS: dict[str, dict[str, float]] = {
    "nfl": {"home_edge": 2.0, "score_stdev": 13.5, "shrink": 0.5},
}


@dataclass
class GameRecord:
    date: str    # ISO date string
    won: bool    # did this team win? (ignored when drew=True)
    opp: str     # opponent name (normalized)
    home: bool   # was this team the home team?
    drew: bool = False  # soccer only; legacy records default False (= L if won=False)
    margin: Optional[float] = None  # this team's score minus opponent's (None on legacy rows)


class FormModelAgent(ModelAgent):
    """
    Recent-form probability model.

    Weights the last WINDOW games with exponential decay so the most
    recent result counts more than a game from 10 matches ago.
    """

    name = "form"
    weight = 0.25   # blending weight in ensemble

    def _team_records(self, sector: str, team: str) -> list[GameRecord]:
        sector_data = self._state.get(sector, {})
        team_data = sector_data.get(team)
        # Normalizer-driven resolution: apply sector aliases so Pinnacle labels
        # ("Karmine Corp", "LGD Gaming") hit stored keys ("kc", "lgd").
        if not team_data:
            try:
                from evmax.matching.normalizer import NameNormalizer
                normed = NameNormalizer(sector).normalize(team)
                if normed and normed != team:
                    team_data = sector_data.get(normed)
            except Exception:
                pass
        # Fallback: try last word (e.g. "new york knicks" → "knicks")
        if not team_data and " " in team:
            last = team.rsplit(" ", 1)[-1]
            team_data = sector_data.get(last)
        # Fallback: prefix/suffix/substring match
        # Handles "duke blue devils" → "duke" and "st. johns red storm" → "st. johns red storm"
        if not team_data:
            for key, val in sector_data.items():
                if (team.startswith(key + " ") or key.startswith(team + " ")
                        or team.endswith(key) or key.endswith(team)):
                    team_data = val
                    break
        return [GameRecord(**r) for r in (team_data or [])]

    def _form_rate(self, records: list[GameRecord]) -> float:
        """Exponentially-weighted form over the most recent WINDOW games.

        Per-game score: points-per-game normalized to [0, 1] — 3 for win,
        1 for draw, 0 for loss, divided by 3. This matches standard football
        table math and avoids the collapse where a drawing team looks
        identical to a losing team (the pre-2026-04 behavior that pushed
        Form's soccer Brier above the always-home baseline).

        Non-soccer sectors (no draws): records have drew=False by default,
        so the formula reduces to the old 0/1 win indicator — wins score
        3/3=1.0 and losses score 0/3=0.0.

        Opponent-quality weighting was tried (multiplying each game's value
        by (1 + elo_gap/400)) and backtested net-negative on WNBA — it
        double-counts opponent strength that Elo already captures.
        """
        recent = sorted(records, key=lambda r: r.date, reverse=True)[:WINDOW]
        if not recent:
            return 0.5  # unknown → assume 50/50

        total_weight = 0.0
        weighted_points = 0.0
        for i, rec in enumerate(recent):
            w = DECAY ** i  # most recent = decay^0 = 1.0
            if rec.drew:
                pts = 1.0
            elif rec.won:
                pts = 3.0
            else:
                pts = 0.0
            weighted_points += w * (pts / 3.0)
            total_weight += w

        raw = weighted_points / total_weight if total_weight > 0 else 0.5
        n = len(recent)
        shrinkage = n / (n + SHRINKAGE_PRIOR_N)
        return raw * shrinkage + SHRINKAGE_BASELINE * (1.0 - shrinkage)

    @staticmethod
    def _margin_rate(records: list[GameRecord], params: dict[str, float]) -> Optional[float]:
        """Exponentially-weighted average margin over the WINDOW, home-edge adjusted.

        Returns None when any in-window record lacks a margin so the caller
        falls back to W/L form (mixed legacy/new state must never blend two
        different definitions of form).
        """
        recent = sorted(records, key=lambda r: r.date, reverse=True)[:WINDOW]
        if not recent or any(r.margin is None for r in recent):
            return None
        home_edge = float(params.get("home_edge", 0.0))
        total_w = 0.0
        acc = 0.0
        for i, rec in enumerate(recent):
            w = DECAY ** i
            adj = (rec.margin or 0.0) - (home_edge if rec.home else -home_edge)
            acc += w * adj
            total_w += w
        return acc / total_w if total_w > 0 else None

    @staticmethod
    def _norm_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def pair_home_probability(
        self, sector: str, recs_a: list[GameRecord], recs_b: list[GameRecord]
    ) -> tuple[float, str]:
        """P(team_a wins) for team_a at home, plus a notes fragment.

        Single source of truth for the form → probability step, shared by
        `predict_pair` and the walk-forward harness so a replay exercises the
        exact live path (margin form for MARGIN_FORM_SECTORS when every
        in-window record carries a margin; log5 on W/L form otherwise).
        """
        if sector in MARGIN_FORM_SECTORS:
            params = MARGIN_FORM_PARAMS.get(sector, {})
            m_a = self._margin_rate(recs_a, params)
            m_b = self._margin_rate(recs_b, params)
            if m_a is not None and m_b is not None:
                pred_margin = (float(params.get("shrink", 0.5)) * (m_a - m_b)
                               + float(params.get("home_edge", 0.0)))
                prob_a = self._norm_cdf(pred_margin / float(params.get("score_stdev", 13.5)))
                prob_a = min(0.95, max(0.05, prob_a))
                return prob_a, f"margin_a={m_a:+.1f} margin_b={m_b:+.1f} pred={pred_margin:+.1f} "
        form_a = self._form_rate(recs_a)
        form_b = self._form_rate(recs_b)
        prob_a = self._log5(form_a, form_b)
        home_adj = HOME_ADJ.get(sector, 0.03)
        prob_a = min(0.95, max(0.05, prob_a + home_adj))  # team_a is home
        return prob_a, f"form_a={form_a:.3f} form_b={form_b:.3f} "

    @staticmethod
    def _is_stale(records: list[GameRecord], reference: Optional[date] = None) -> bool:
        """True when the most recent record is older than STALE_DAYS.

        `reference` defaults to today (UTC). Accepting an override makes this
        trivial to test without monkey-patching datetime.
        """
        if not records:
            return True
        ref = reference or datetime.now(timezone.utc).date()
        most_recent = max(r.date for r in records)
        try:
            rec_date = date.fromisoformat(most_recent)
        except ValueError:
            return True  # unparseable date → treat as stale
        return (ref - rec_date).days > STALE_DAYS

    @staticmethod
    def _log5(p_a: float, p_b: float) -> float:
        """
        Bill James Log5 formula: head-to-head win probability for team A.

        P(A beats B) = (p_a - p_a*p_b) / (p_a + p_b - 2*p_a*p_b)
        """
        denom = p_a + p_b - 2.0 * p_a * p_b
        if denom < 1e-9:
            return 0.5
        return (p_a - p_a * p_b) / denom

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()

        if not team_a or not team_b:
            return None

        recs_a = self._team_records(sector, team_a)
        recs_b = self._team_records(sector, team_b)

        if len(recs_a) < MIN_GAMES or len(recs_b) < MIN_GAMES:
            return None   # insufficient data — let ensemble skip this model

        # Staleness guard: if either team's most recent record is older than
        # STALE_DAYS relative to the game being predicted, the form signal
        # is carrying prior-season data into a new season and is actively
        # harmful (WNBA offseason was the smoking gun). Skip and let Elo +
        # Efficiency drive the blend until fresh in-season games accumulate.
        # Reference against the market's event_date when available so the
        # check is meaningful during historical walk-forward replays.
        reference = market.event_date.date() if market.event_date else None
        if self._is_stale(recs_a, reference) or self._is_stale(recs_b, reference):
            return None

        prob_a, form_note = self.pair_home_probability(sector, recs_a, recs_b)
        prob_b = 1.0 - prob_a
        prob_draw: Optional[float] = None

        if sector == "soccer":
            # Strength-dependent draw allocation (matches Elo model).
            # Quadratic decay: ~26% for even matches, ~10% for mismatches.
            gap = abs(prob_a - 0.5) * 2.0
            # Matches Elo draw allocator (retuned 2026-04-23 walk-forward).
            prob_draw = max(0.10, 0.30 - 0.40 * gap * gap)
            scale = (1.0 - prob_draw) / (prob_a + prob_b)
            prob_a = prob_a * scale
            prob_b = prob_b * scale

        sample = min(len(recs_a), len(recs_b))
        confidence = 0.5 + 0.3 * min(1.0, (sample - MIN_GAMES) / (WINDOW - MIN_GAMES))

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_a,
            true_prob_b=prob_b,
            true_prob_draw=prob_draw,
            confidence=confidence,
            weight=self.weight,
            sample_size=sample,
            notes=form_note + f"p={prob_a:.3f} n={sample}",
        )

    # ------------------------------------------------------------------
    # Update from result
    # ------------------------------------------------------------------

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        date_str = event_date or datetime.now(timezone.utc).date().isoformat()
        sector = sector.lower()
        team_a = team_a.lower().strip()
        team_b = team_b.lower().strip()

        drew = score_a == score_b
        a_won = score_a > score_b
        b_won = score_b > score_a

        if sector not in self._state:
            self._state[sector] = {}

        def _add_record(
            team: str, won: bool, opp: str, home: bool, drew: bool, margin: float
        ) -> None:
            if team not in self._state[sector]:
                self._state[sector][team] = []
            existing = self._state[sector][team]
            # Dedup: skip if same (date, opp, home) already recorded
            key = (date_str, opp, home)
            if any((r["date"], r["opp"], r["home"]) == key for r in existing):
                return
            existing.append(
                {"date": date_str, "won": won, "opp": opp, "home": home, "drew": drew,
                 "margin": float(margin)}
            )
            # Keep only the most recent 2×WINDOW entries to control file size
            self._state[sector][team] = sorted(
                existing, key=lambda r: r["date"], reverse=True
            )[: WINDOW * 2]

        _add_record(team_a, a_won, team_b, home=True, drew=drew, margin=score_a - score_b)
        _add_record(team_b, b_won, team_a, home=False, drew=drew, margin=score_b - score_a)

        self.log.debug("form_updated", team_a=team_a, team_b=team_b, a_won=a_won)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_results(self, sector: str, results: list[dict]) -> None:
        """
        Bulk-load historical results.

        Each result dict:
          {
            "date": "2026-01-15",
            "home": "lakers",
            "away": "celtics",
            "score_home": 112,
            "score_away": 108,
          }
        """
        for r in results:
            self.update(
                team_a=r["home"],
                team_b=r["away"],
                score_a=r["score_home"],
                score_b=r["score_away"],
                sector=sector,
                event_date=r.get("date"),
            )
        self.save_state()
        self.log.info("form_seeded", sector=sector, records=len(results))
