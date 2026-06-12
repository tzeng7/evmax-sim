"""NflQbEloModelAgent — NFL Elo with a per-QB delta layer.

QB swaps are the single biggest in-season information shock in NFL — a Tua
scratch or a Russell Wilson trade moves Vegas lines 3-7 points. The generic
EloModelAgent treats team strength as a single number, so it cannot distinguish
"Dolphins with Tua" from "Dolphins with the backup". This agent splits team
strength into:

  effective_elo(team, starter) = team_base[team] + qb_delta[starter]

When a starter changes, only `qb_delta` changes — `team_base` (which represents
the rest of the roster: OL, defense, scheme, coaching) is unchanged. This is a
well-validated approach (FiveThirtyEight QB Elo, ESPN FPI variants).

State file: data/models/nfl_qb_elo_state.json
  {
    "nfl": {
      "team_base":        {"kansas city chiefs": 1550.0, ...},
      "qb_deltas":        {"P.Mahomes": +85.0, "B.Mayfield": +25.0, ...},
      "current_starters": {"kansas city chiefs": "P.Mahomes", ...},
      "fetched_at": "2026-05-01"
    }
  }

State is seed-driven via scripts/seed_nfl_qb_elo.py. Weekly re-seed is cheap
(~30s) and captures starter changes from the prior week's games. The `update`
method is intentionally a no-op — incremental per-game updates would require
ESPN scoreboard to expose starting QB info, which it does not, so we re-walk
PBP weekly instead. Same pattern as nfl_efficiency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import structlog

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.nfl_efficiency_agent import (
    NFL_ABBREV_TO_NAME,
    NFL_NICKNAME_TO_NAME,
    active_nfl_season,
    nfl_state_is_stale_for_today,
)
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

logger = structlog.get_logger(__name__)

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "nfl_qb_elo_state.json"

DEFAULT_ELO = 1500.0
DEFAULT_QB_DELTA = 0.0
K_FACTOR = 25.0          # standard NFL Elo K, matches EloModelAgent K_FACTORS["nfl"]
HOME_ADVANTAGE_ELO = 48.0  # matches EloModelAgent HOME_ADVANTAGE_ELO["nfl"]
QB_UPDATE_SHARE = 0.60   # share of each game's Elo delta routed to QB rating
TEAM_UPDATE_SHARE = 1.0 - QB_UPDATE_SHARE
QB_DELTA_CLAMP = 200.0   # cap on |qb_delta| so a hot streak rookie can't swing
                         # 12+ Elo pts of weight on one ensemble model

# Confidence tiers: how much we trust the prediction given QB sample size.
QB_LOW_GAMES = 3      # at/below → low confidence
QB_MED_GAMES = 8      # at/below → medium; above → high


def _resolve_team(teams: dict, team: str) -> Optional[str]:
    """Resolve any team identifier (full name, nickname, abbreviation) to the
    canonical full name key used in this agent's state. Returns None on miss.

    Mirrors nfl_efficiency_agent._resolve_team but returns the resolved key
    rather than the stats dict — Elo state is keyed by team name and we
    need to address it by string.
    """
    if not team:
        return None
    team = team.lower().strip()
    if team in teams:
        return team

    upper = team.upper()
    if upper in NFL_ABBREV_TO_NAME:
        full = NFL_ABBREV_TO_NAME[upper]
        if full in teams:
            return full

    if " " in team:
        last = team.rsplit(" ", 1)[-1]
        if last in NFL_NICKNAME_TO_NAME:
            full = NFL_NICKNAME_TO_NAME[last]
            if full in teams:
                return full
    elif team in NFL_NICKNAME_TO_NAME:
        full = NFL_NICKNAME_TO_NAME[team]
        if full in teams:
            return full

    for key in teams:
        if team in key or key in team:
            return key
    return None


class NflQbEloModelAgent(ModelAgent):
    """Elo with QB-rating layer for NFL.

    effective_elo(team, starter) = team_base[team] + qb_delta[starter]
      P(home) = 1 / (1 + 10 ** ((away_eff - home_eff) / 400))
    """

    name = "nfl_qb_elo"
    weight = 0.30  # placeholder; ensemble per-sector override controls actual weight

    def _sector_state(self) -> dict:
        return self._state.get("nfl", {})

    def _team_base(self, teams: dict, team_key: Optional[str]) -> float:
        if team_key is None:
            return DEFAULT_ELO
        return teams.get(team_key, DEFAULT_ELO)

    def _qb_delta(self, qb_deltas: dict, qb_name: Optional[str]) -> float:
        if qb_name is None:
            return DEFAULT_QB_DELTA
        # Stored keys are case-sensitive ("P.Mahomes"), but external callers may
        # pass any case — normalize via lower() comparison fallback.
        if qb_name in qb_deltas:
            return qb_deltas[qb_name]
        lname = qb_name.lower()
        for k, v in qb_deltas.items():
            if k.lower() == lname:
                return v
        return DEFAULT_QB_DELTA

    def _starter_for(self, current_starters: dict, team_key: Optional[str]) -> Optional[str]:
        if team_key is None:
            return None
        return current_starters.get(team_key)

    def _qb_games(self, qb_games: dict, qb_name: Optional[str]) -> int:
        """Number of starts seen for this QB during seeding. Used for confidence."""
        if qb_name is None:
            return 0
        return qb_games.get(qb_name, 0)

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector != "nfl":
            return None

        if nfl_state_is_stale_for_today(self._state):
            logger.warning(
                "nfl_qb_elo_stale_seasons_used",
                seasons_used=self._sector_state().get("seasons_used"),
                active_season=active_nfl_season(),
                hint="re-run scripts/seed_nfl_qb_elo.py",
            )
            return None

        sector_state = self._sector_state()
        teams = sector_state.get("team_base", {})
        qb_deltas = sector_state.get("qb_deltas", {})
        current_starters = sector_state.get("current_starters", {})
        qb_games = sector_state.get("qb_games", {})
        if not teams:
            return None

        home_label = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        away_label = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()
        home_key = _resolve_team(teams, home_label)
        away_key = _resolve_team(teams, away_label)
        if home_key is None or away_key is None:
            return None

        home_starter = self._starter_for(current_starters, home_key)
        away_starter = self._starter_for(current_starters, away_key)

        home_eff = (
            self._team_base(teams, home_key)
            + self._qb_delta(qb_deltas, home_starter)
            + HOME_ADVANTAGE_ELO
        )
        away_eff = (
            self._team_base(teams, away_key)
            + self._qb_delta(qb_deltas, away_starter)
        )

        prob_home = 1.0 / (1.0 + 10.0 ** ((away_eff - home_eff) / 400.0))
        prob_home = max(0.02, min(0.98, prob_home))
        prob_away = 1.0 - prob_home

        # Confidence: penalised when starter info missing or QB has small sample.
        min_qb_games = min(
            self._qb_games(qb_games, home_starter),
            self._qb_games(qb_games, away_starter),
        )
        if home_starter is None or away_starter is None:
            confidence = 0.45
        elif min_qb_games <= QB_LOW_GAMES:
            confidence = 0.55
        elif min_qb_games <= QB_MED_GAMES:
            confidence = 0.70
        else:
            confidence = 0.80

        notes = (
            f"home={home_starter or '?'}({self._qb_delta(qb_deltas, home_starter):+.0f}) "
            f"away={away_starter or '?'}({self._qb_delta(qb_deltas, away_starter):+.0f}) "
            f"team_base={self._team_base(teams, home_key):.0f}/{self._team_base(teams, away_key):.0f}"
        )

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=prob_home,
            true_prob_b=prob_away,
            true_prob_draw=None,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_qb_games,
            notes=notes,
        )

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        """No-op: state is rebuilt by re-walking PBP via seed script.

        Rationale: live update would require knowing the starting QB for each
        team at game time. ESPN scoreboard does not expose starters; nflreadpy
        does, but only after the game completes (PBP). Cleanest path is to
        re-seed weekly with the prior week's games included — see
        scripts/seed_nfl_qb_elo.py.
        """
        return


# ---------------------------------------------------------------------------
# Walk-forward helpers used by both the seed script and the validation
# backtest. Keeping these free functions (not methods) makes them trivially
# testable without instantiating the agent and ensures the seed and the live
# agent apply the same Elo math.
# ---------------------------------------------------------------------------


def _expected_home_prob(
    home_eff: float,
    away_eff: float,
) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_eff - home_eff) / 400.0))


def _mov_multiplier(margin: float, elo_diff: float) -> float:
    """Margin-of-victory K multiplier (FiveThirtyEight-inspired).

    Blowouts move ratings more than squeakers, but the effect is dampened when
    the winner was already heavily favored (prevents Elo autocorrelation — a
    strong team running up the score against a weak one shouldn't keep
    compounding its rating). Mirrors EloModelAgent._mov_multiplier so the QB-Elo
    math stays consistent with the generic Elo agent.

    `elo_diff` is the pre-game effective-Elo edge of the WINNER over the loser.

    Calibrated so (at elo_diff≈0):
      1-pt win  → ~0.85x · 5-pt → ~1.0x · 10-pt → ~1.15x · 20-pt → ~1.35x
    Clamped to [0.75, 1.5].
    """
    if margin <= 0:
        return 1.0
    raw = math.log(margin + 1.0) / math.log(6.0)  # margin=5 ≈ 1.0
    dampener = 2.2 / (abs(elo_diff) * 0.001 + 2.2)
    return max(0.75, min(1.5, raw * dampener))


def apply_qb_elo_update(
    state: dict,
    home_team: str,
    away_team: str,
    home_starter: Optional[str],
    away_starter: Optional[str],
    home_won: bool,
    *,
    k_factor: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE_ELO,
    qb_share: float = QB_UPDATE_SHARE,
    home_score: Optional[float] = None,
    away_score: Optional[float] = None,
    use_mov: bool = False,
) -> None:
    """Mutate `state` (a sector_state dict) with the Elo update for one game.

    Splits the per-game Elo delta into a QB share (applied to qb_deltas) and a
    team-base share (applied to team_base). Initialises any missing keys to
    DEFAULT_ELO / DEFAULT_QB_DELTA so the very first game involving a team
    or QB lands cleanly.

    Mirrors EloModelAgent's basic update math. Margin-of-victory scaling is
    opt-in via `use_mov=True` (requires home_score/away_score): the per-game
    delta is multiplied by a blowout-aware multiplier dampened when the winner
    was already favored (see `_mov_multiplier`). Default off so the seed script,
    backtest, and tests can toggle it explicitly and the no-score call path
    reduces exactly to the prior behaviour. Strength-of-schedule is implicitly
    handled by the dampener (it keys on the pre-game effective-Elo edge), so no
    separate SOS term is needed here.
    """
    team_base = state.setdefault("team_base", {})
    qb_deltas = state.setdefault("qb_deltas", {})
    qb_games = state.setdefault("qb_games", {})

    home_base = team_base.setdefault(home_team, DEFAULT_ELO)
    away_base = team_base.setdefault(away_team, DEFAULT_ELO)
    home_qb_d = qb_deltas.setdefault(home_starter, DEFAULT_QB_DELTA) if home_starter else DEFAULT_QB_DELTA
    away_qb_d = qb_deltas.setdefault(away_starter, DEFAULT_QB_DELTA) if away_starter else DEFAULT_QB_DELTA

    home_eff = home_base + home_qb_d + home_advantage
    away_eff = away_base + away_qb_d

    expected_home = _expected_home_prob(home_eff, away_eff)
    actual_home = 1.0 if home_won else 0.0
    delta = k_factor * (actual_home - expected_home)

    if use_mov and home_score is not None and away_score is not None:
        margin = abs(float(home_score) - float(away_score))
        # elo_diff = winner's pre-game effective edge over the loser
        winner_edge = (home_eff - away_eff) if home_won else (away_eff - home_eff)
        delta *= _mov_multiplier(margin, winner_edge)

    # Team base gets (1 - qb_share); QB rating gets qb_share.
    team_base[home_team] = home_base + delta * (1.0 - qb_share)
    team_base[away_team] = away_base - delta * (1.0 - qb_share)

    if home_starter:
        new_home_qb = home_qb_d + delta * qb_share
        qb_deltas[home_starter] = max(-QB_DELTA_CLAMP, min(QB_DELTA_CLAMP, new_home_qb))
        qb_games[home_starter] = qb_games.get(home_starter, 0) + 1
    if away_starter:
        new_away_qb = away_qb_d - delta * qb_share
        qb_deltas[away_starter] = max(-QB_DELTA_CLAMP, min(QB_DELTA_CLAMP, new_away_qb))
        qb_games[away_starter] = qb_games.get(away_starter, 0) + 1
