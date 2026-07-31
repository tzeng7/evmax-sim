"""SoccerXgAgent — xG regression model for soccer and national-team World Cup.

Tracks a rolling window of per-team shot data (shots on target, total shots,
goals scored/conceded) from ESPN box scores.  At prediction time it computes a
crude xG proxy:

    xG = shots_on_target × 0.30 + (total_shots - shots_on_target) × 0.03

Teams whose actual goals significantly exceed their xG are due for regression.
The agent adjusts attack/defense lambdas accordingly, then runs a Poisson score
matrix (with Dixon-Coles correction) to produce win/draw/loss probabilities.

State file: data/models/soccer_xg_state.json

Per-sector namespaces share one file but NEVER share teams — club `soccer` and
national-team `worldcup` are separate pools (a club's xG says nothing about a
national side and vice-versa). For backward compatibility the `soccer` pool
stays at the top-level `teams` key (the legacy flat format); every other sector
lives under `state[sector]["teams"]`. See `_teams_for()`.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import PredictionMarket
from evmax.models.odds import SharpOdds

STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "models" / "soccer_xg_state.json"

WINDOW = 10
SOT_CONVERSION = 0.30
OFF_TARGET_CONVERSION = 0.03
LEAGUE_AVG_HOME = 1.55
LEAGUE_AVG_AWAY = 1.15

# Sectors this agent serves. `worldcup` reuses the identical xG-regression model
# on the national-team namespace (seeded from international results), mirroring
# how the soccer ensemble leans on xG. Membership here gates predict_pair.
SUPPORTED_SECTORS = {"soccer", "worldcup"}

# Per-sector mean goals used as the opponent-defense baseline. Club soccer skews
# higher than international football; the World Cup baseline is the symmetric
# neutral-venue average that the Poisson agent uses for `worldcup`.
SECTOR_LEAGUE_AVG: dict[str, float] = {
    "soccer":   LEAGUE_AVG_HOME,
    "worldcup": 1.30,
}

DC_RHO = 0.1
MAX_GOALS = 8
MIN_MATCHES = 4


def xg_from_shots(shots_on_target: int, total_shots: int) -> float:
    off_target = max(0, total_shots - shots_on_target)
    return shots_on_target * SOT_CONVERSION + off_target * OFF_TARGET_CONVERSION


def _poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dc_correction(lam_h: float, lam_a: float, h: int, a: int, rho: float) -> float:
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _score_matrix(lam_h: float, lam_a: float) -> list[list[float]]:
    matrix = []
    for h in range(MAX_GOALS + 1):
        row = []
        for a in range(MAX_GOALS + 1):
            p = (
                _poisson_pmf(lam_h, h)
                * _poisson_pmf(lam_a, a)
                * _dc_correction(lam_h, lam_a, h, a, DC_RHO)
            )
            row.append(p)
        matrix.append(row)
    return matrix


def _win_draw_probs(matrix: list[list[float]]) -> tuple[float, float, float]:
    p_home = p_draw = p_away = 0.0
    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    if total < 1e-9:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / total, p_draw / total, p_away / total


class SoccerXgAgent(ModelAgent):
    """xG-regressed Poisson model for soccer."""

    name = "xg"
    weight = 0.25

    def __init__(self) -> None:
        super().__init__()
        self._state: dict = {}
        self._state_path = STATE_PATH
        self._normalizers: dict[str, NameNormalizer] = {}
        self.load_state()

    def _normalizer_for(self, sector: str) -> NameNormalizer:
        sector = (sector or "soccer").lower()
        if sector not in self._normalizers:
            self._normalizers[sector] = NameNormalizer(sector)
        return self._normalizers[sector]

    def _resolve_team_key(self, team: str, sector: str) -> str:
        """Map a raw source label to this sector's state-store key.

        Exact lowercase match wins: state keys are already canonical (the seed
        scripts write through NameNormalizer) and the normalizer is NOT
        idempotent — normalize("man united") == "man" — so re-normalizing an
        exact key would orphan it. Raw labels that miss ("Philadelphia Union",
        "Manchester City") fall back to the sector normalizer, matching the
        seed-time keying. Before this, predict_pair looked up Pinnacle labels
        with a bare .lower(), so any club whose label differs from its
        canonical key silently dropped xG from the blend (all of MLS).

        For teams absent from the store the canonical form (or the raw
        lowercase when normalization yields nothing) is returned, so writes
        for new teams land on seed-compatible keys.
        """
        teams = self._teams_for(sector)
        key = team.lower().strip()
        if key in teams:
            return key
        norm = self._normalizer_for(sector).normalize(team)
        if norm and norm in teams:
            return norm
        return norm or key

    def update(
        self,
        team_a: str,
        team_b: str,
        score_a: float,
        score_b: float,
        sector: str,
        event_date: Optional[str] = None,
    ) -> None:
        """No-op — xG updates require shot data via record_match()."""
        pass

    def load_state(self) -> None:
        if self._state_path.exists():
            try:
                self._state = json.loads(self._state_path.read_text())
            except Exception:
                self._state = {}
        # `soccer` keeps the legacy flat layout at the top-level "teams" key.
        if "teams" not in self._state:
            self._state["teams"] = {}

    def _teams_for(self, sector: str) -> dict:
        """Return the team store for a sector, creating it if absent.

        `soccer` maps to the legacy flat top-level "teams" dict (so existing
        state files and callers keep working unchanged); every other sector is
        namespaced under state[sector]["teams"]. Pools never overlap, so club
        and national-team xG can't contaminate each other.
        """
        sector = (sector or "soccer").lower()
        if sector == "soccer":
            return self._state.setdefault("teams", {})
        return self._state.setdefault(sector, {}).setdefault("teams", {})

    def save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state, indent=2))

    def record_match(
        self,
        team: str,
        goals_for: int,
        goals_against: int,
        shots_on_target: int,
        total_shots: int,
        opponent_sot: int,
        opponent_shots: int,
        match_date: str,
        is_home: bool,
        sector: str = "soccer",
    ) -> None:
        """Record one side of a match result with shot data."""
        teams = self._teams_for(sector)
        key = self._resolve_team_key(team, sector)
        if key not in teams:
            teams[key] = {"matches": []}

        xg = xg_from_shots(shots_on_target, total_shots)
        xga = xg_from_shots(opponent_sot, opponent_shots)

        teams[key]["matches"].insert(0, {
            "date": match_date,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "sot": shots_on_target,
            "shots": total_shots,
            "xg": round(xg, 3),
            "xga": round(xga, 3),
            "is_home": is_home,
        })
        # Keep only last WINDOW * 2 matches (generous buffer)
        teams[key]["matches"] = teams[key]["matches"][:WINDOW * 2]

    def _team_xg_stats(self, team: str, sector: str = "soccer") -> Optional[dict]:
        """Compute rolling xG averages for a team over the last WINDOW matches."""
        data = self._teams_for(sector).get(self._resolve_team_key(team, sector))
        if not data:
            return None
        matches = data.get("matches", [])[:WINDOW]
        if len(matches) < MIN_MATCHES:
            return None

        n = len(matches)
        total_gf = sum(m["goals_for"] for m in matches)
        total_ga = sum(m["goals_against"] for m in matches)
        total_xg = sum(m["xg"] for m in matches)
        total_xga = sum(m["xga"] for m in matches)

        return {
            "n": n,
            "goals_per_game": total_gf / n,
            "goals_against_per_game": total_ga / n,
            "xg_per_game": total_xg / n,
            "xga_per_game": total_xga / n,
            "attack_ratio": (total_gf / total_xg) if total_xg > 0 else 1.0,
            "defense_ratio": (total_ga / total_xga) if total_xga > 0 else 1.0,
        }

    def _regressed_lambda(
        self, team: str, is_home: bool, opponent: str, sector: str = "soccer",
    ) -> Optional[float]:
        """Compute xG-regressed expected goals for a team.

        The key insight: if a team's actual goals >> their xG, they've been
        finishing above expectation and will regress. We blend actual and xG
        rates with a 40/60 regression weight (lean toward xG).
        """
        stats = self._team_xg_stats(team, sector)
        if stats is None:
            return None

        opp_stats = self._team_xg_stats(opponent, sector)

        # Blend actual goals with xG — 40% actual, 60% xG (regression toward xG)
        regressed_attack = 0.40 * stats["goals_per_game"] + 0.60 * stats["xg_per_game"]

        # Opponent defensive quality relative to the sector's baseline goal rate.
        baseline = SECTOR_LEAGUE_AVG.get(sector, LEAGUE_AVG_HOME)
        if opp_stats:
            opp_defensive_factor = opp_stats["xga_per_game"] / baseline
        else:
            opp_defensive_factor = 1.0

        lam = regressed_attack * opp_defensive_factor

        # Clamp to reasonable range
        return max(0.3, min(4.0, lam))

    async def predict_pair(
        self,
        market: PredictionMarket,
        sharp_odds: SharpOdds,
    ) -> Optional[ModelAgentPrediction]:
        sector = (market.sector or "").lower()
        if sector not in SUPPORTED_SECTORS:
            return None

        team_a = (sharp_odds.outcome_a_label or market.team_home or "").lower().strip()
        team_b = (sharp_odds.outcome_b_label or market.team_away or "").lower().strip()
        if not team_a or not team_b:
            return None

        lam_a = self._regressed_lambda(team_a, is_home=True, opponent=team_b, sector=sector)
        lam_b = self._regressed_lambda(team_b, is_home=False, opponent=team_a, sector=sector)

        if lam_a is None or lam_b is None:
            return None

        matrix = _score_matrix(lam_a, lam_b)
        p_home, p_draw, p_away = _win_draw_probs(matrix)

        stats_a = self._team_xg_stats(team_a, sector)
        stats_b = self._team_xg_stats(team_b, sector)
        min_n = min(
            (stats_a["n"] if stats_a else 0),
            (stats_b["n"] if stats_b else 0),
        )

        if min_n < MIN_MATCHES:
            confidence = 0.3
        elif min_n < 8:
            confidence = 0.50
        else:
            confidence = 0.65

        notes = f"xG λ_h={lam_a:.2f} λ_a={lam_b:.2f} n={min_n}"
        if stats_a:
            ratio_a = stats_a["attack_ratio"]
            if ratio_a > 1.2:
                notes += f" {team_a[:10]}↓regress({ratio_a:.2f})"
            elif ratio_a < 0.8:
                notes += f" {team_a[:10]}↑unlucky({ratio_a:.2f})"
        if stats_b:
            ratio_b = stats_b["attack_ratio"]
            if ratio_b > 1.2:
                notes += f" {team_b[:10]}↓regress({ratio_b:.2f})"
            elif ratio_b < 0.8:
                notes += f" {team_b[:10]}↑unlucky({ratio_b:.2f})"

        return ModelAgentPrediction(
            event_id=sharp_odds.event_id,
            model_name=self.name,
            true_prob_a=p_home,
            true_prob_b=p_away,
            true_prob_draw=p_draw,
            confidence=confidence,
            weight=self.weight,
            sample_size=min_n,
            notes=notes,
        )
