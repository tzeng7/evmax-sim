"""Walk-forward backtest using ESPN scoreboard data.

Supports any ESPN-based sector (NBA, WNBA, NCAAB, NFL, MLB, etc.).
Fetches all completed games chronologically, then runs a walk-forward
evaluation: predict each game with current model state, record the
prediction, then update the model with the actual result.

No external odds needed — this tests the MODEL calibration, not the edge.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import httpx
import structlog

from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.backtest.models import BacktestRow
from evmax.matching.normalizer import NameNormalizer

logger = structlog.get_logger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# ESPN sport/league paths by sector
ESPN_SECTORS: dict[str, tuple[str, str]] = {
    "wnba": ("basketball", "wnba"),
    "nba": ("basketball", "nba"),
    "ncaab": ("basketball", "mens-college-basketball"),
    "ncaaw": ("basketball", "womens-college-basketball"),
    "nfl": ("football", "nfl"),
    "baseball": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
}


@dataclass
class WalkForwardResult:
    """One game's walk-forward prediction vs actual outcome."""
    date: date
    home: str
    away: str
    home_score: int
    away_score: int
    home_won: bool
    # Model predictions (before observing this game)
    elo_prob_home: Optional[float] = None
    form_prob_home: Optional[float] = None
    poisson_prob_home: Optional[float] = None
    ensemble_prob_home: Optional[float] = None


@dataclass
class WalkForwardReport:
    """Summary of walk-forward backtest."""
    sector: str
    n_games: int
    n_predicted: int  # games where at least one model had enough data
    # Per-model metrics
    elo_brier: float = 0.0
    elo_accuracy: float = 0.0
    elo_n: int = 0
    form_brier: float = 0.0
    form_accuracy: float = 0.0
    form_n: int = 0
    poisson_brier: float = 0.0
    poisson_accuracy: float = 0.0
    poisson_n: int = 0
    # Ensemble
    ensemble_brier: float = 0.0
    ensemble_accuracy: float = 0.0
    ensemble_n: int = 0
    # Baselines
    home_win_rate: float = 0.0
    baseline_always_home_brier: float = 0.0
    # Calibration (ensemble)
    calibration_bins: list = field(default_factory=list)
    # Game-by-game results
    results: list[WalkForwardResult] = field(default_factory=list)


def _fetch_espn_month(
    client: httpx.Client,
    sport: str,
    league: str,
    month: str,
) -> list[dict]:
    """Fetch completed ESPN games for a YYYYMM month. Returns parsed game dicts."""
    url = f"{ESPN_BASE}/{sport}/{league}/scoreboard"
    r = client.get(url, params={"dates": month, "limit": 200}, timeout=15)
    r.raise_for_status()
    events = r.json().get("events", [])

    games = []
    for e in events:
        comp = e["competitions"][0]
        status = comp["status"]["type"]["name"]
        if status != "STATUS_FINAL":
            continue

        teams = comp["competitors"]
        home_t = next((t for t in teams if t["homeAway"] == "home"), None)
        away_t = next((t for t in teams if t["homeAway"] == "away"), None)
        if not home_t or not away_t:
            continue

        try:
            game_date = date.fromisoformat(e["date"][:10])
        except (ValueError, KeyError):
            continue

        games.append({
            "date": game_date,
            "home": home_t["team"]["displayName"],
            "away": away_t["team"]["displayName"],
            "home_score": int(home_t.get("score", 0)),
            "away_score": int(away_t.get("score", 0)),
        })

    return games


def fetch_espn_games(
    sector: str,
    months: list[str],
) -> list[dict]:
    """Fetch all completed games for a sector across the given months.

    Args:
        sector: e.g. "wnba"
        months: list of YYYYMM strings e.g. ["202505", "202506", ...]

    Returns sorted by date.
    """
    if sector not in ESPN_SECTORS:
        raise ValueError(f"Unknown sector {sector!r}. Available: {list(ESPN_SECTORS)}")

    sport, league = ESPN_SECTORS[sector]
    all_games: list[dict] = []

    with httpx.Client() as client:
        for month in months:
            try:
                games = _fetch_espn_month(client, sport, league, month)
                all_games.extend(games)
                logger.info("espn_walkforward_month", sector=sector, month=month, games=len(games))
            except Exception as e:
                logger.warning("espn_walkforward_month_failed", month=month, error=str(e))

    # Sort chronologically and deduplicate (overlapping month queries)
    seen = set()
    unique = []
    for g in sorted(all_games, key=lambda x: (x["date"], x["home"])):
        key = (g["date"], g["home"], g["away"])
        if key not in seen:
            seen.add(key)
            unique.append(g)

    return unique


def run_walkforward(
    sector: str,
    months: list[str],
    elo_weight: float = 0.35,
    form_weight: float = 0.25,
    poisson_weight: float = 0.30,
) -> WalkForwardReport:
    """Run walk-forward backtest: predict → observe → update for each game.

    Returns a WalkForwardReport with per-model and ensemble metrics.
    """
    games = fetch_espn_games(sector, months)
    if not games:
        return WalkForwardReport(sector=sector, n_games=0, n_predicted=0)

    logger.info("walkforward_start", sector=sector, n_games=len(games))

    # Initialize fresh model agents
    elo = EloModelAgent()
    elo._state = {}  # start empty — no prior knowledge
    form = FormModelAgent()
    form._state = {}
    poisson = PoissonModelAgent()
    poisson._state = {}

    norm = NameNormalizer(sector)
    results: list[WalkForwardResult] = []

    for g in games:
        home_raw = g["home"]
        away_raw = g["away"]
        home = norm.normalize(home_raw)
        away = norm.normalize(away_raw)
        if not home or not away:
            continue

        home_won = g["home_score"] > g["away_score"]
        game_date = g["date"]

        # --- Predict with current state (before observing outcome) ---
        elo_prob = _elo_predict(elo, sector, home, away)
        form_prob = _form_predict(form, sector, home, away)
        poisson_prob = _poisson_predict(poisson, sector, home, away)

        # Ensemble: weighted average of available models
        ensemble_prob = _ensemble(
            [(elo_prob, elo_weight), (form_prob, form_weight), (poisson_prob, poisson_weight)]
        )

        result = WalkForwardResult(
            date=game_date,
            home=home,
            away=away,
            home_score=g["home_score"],
            away_score=g["away_score"],
            home_won=home_won,
            elo_prob_home=elo_prob,
            form_prob_home=form_prob,
            poisson_prob_home=poisson_prob,
            ensemble_prob_home=ensemble_prob,
        )
        results.append(result)

        # --- Update models with actual result ---
        _elo_update(elo, sector, home, away, home_won, game_date)
        _form_update(form, sector, home, away, home_won, game_date)
        _poisson_update(poisson, sector, home, away, g["home_score"], g["away_score"])

    # --- Compute metrics ---
    report = _compute_metrics(sector, results, elo_weight, form_weight, poisson_weight)
    return report


def _elo_predict(elo: EloModelAgent, sector: str, home: str, away: str) -> Optional[float]:
    """Get Elo prediction for home win probability. None if no data."""
    state = elo._state.get(sector, {})
    ratings = state.get("ratings", {})
    if home not in ratings and away not in ratings:
        return None
    from evmax.agents.models.elo_agent import HOME_ADVANTAGE_ELO, DEFAULT_ELO
    ha = HOME_ADVANTAGE_ELO.get(sector, 0.0)
    elo_h = ratings.get(home, DEFAULT_ELO) + ha
    elo_a = ratings.get(away, DEFAULT_ELO)
    prob = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
    return prob


def _form_predict(form: FormModelAgent, sector: str, home: str, away: str) -> Optional[float]:
    """Get Form prediction. None if insufficient data."""
    state = form._state.get(sector, {})
    home_records = state.get(home, [])
    away_records = state.get(away, [])
    if len(home_records) < 3 or len(away_records) < 3:
        return None

    from evmax.agents.models.form_agent import DECAY, HOME_ADJ

    def _weighted_rate(records: list[dict]) -> float:
        recent = records[-10:]  # last 10
        total_w = 0.0
        win_w = 0.0
        for i, rec in enumerate(recent):
            w = DECAY ** (len(recent) - 1 - i)
            total_w += w
            if rec["won"]:
                win_w += w
        return win_w / total_w if total_w > 0 else 0.5

    fa = _weighted_rate(home_records)
    fb = _weighted_rate(away_records)

    # Log5 formula
    if fa + fb == 0 or fa + fb == 2 * fa * fb:
        prob = 0.5
    else:
        prob = (fa - fa * fb) / (fa + fb - 2 * fa * fb)

    # Home adjustment
    ha = HOME_ADJ.get(sector, 0.0)
    prob = max(0.01, min(0.99, prob + ha))
    return prob


def _poisson_predict(poisson: PoissonModelAgent, sector: str, home: str, away: str) -> Optional[float]:
    """Get Poisson prediction. None if insufficient data."""
    state = poisson._state.get(sector, {})
    teams = state.get("teams", {})
    if home not in teams or away not in teams:
        return None

    from evmax.agents.models.poisson_agent import (
        LEAGUE_AVG_DEFAULTS, MAX_SCORE, BUCKET_SIZE, _poisson_pmf,
    )

    avg = state.get("league_avg", LEAGUE_AVG_DEFAULTS.get(sector, {"home": 1.0, "away": 1.0}))
    h_data = teams[home]
    a_data = teams[away]

    bucket = BUCKET_SIZE.get(sector, 1)
    max_s = MAX_SCORE.get(sector, 10)

    lam_h = h_data["attack"] * a_data["defense"] * avg["home"] / bucket
    lam_a = a_data["attack"] * h_data["defense"] * avg["away"] / bucket

    # Score matrix
    p_home_win = 0.0
    p_away_win = 0.0
    for i in range(max_s + 1):
        for j in range(max_s + 1):
            p = _poisson_pmf(lam_h, i) * _poisson_pmf(lam_a, j)
            if i > j:
                p_home_win += p
            elif j > i:
                p_away_win += p

    total = p_home_win + p_away_win
    if total == 0:
        return 0.5
    return p_home_win / total


def _ensemble(
    model_probs: list[tuple[Optional[float], float]],
) -> Optional[float]:
    """Weighted ensemble of available model predictions."""
    total_w = 0.0
    total_p = 0.0
    for prob, weight in model_probs:
        if prob is not None:
            total_w += weight
            total_p += prob * weight
    if total_w == 0:
        return None
    return total_p / total_w


def _elo_update(
    elo: EloModelAgent,
    sector: str,
    home: str,
    away: str,
    home_won: bool,
    game_date: date,
) -> None:
    """Update Elo ratings after observing a game result."""
    from evmax.agents.models.elo_agent import K_FACTORS, HOME_ADVANTAGE_ELO, DEFAULT_ELO

    if sector not in elo._state:
        elo._state[sector] = {"ratings": {}, "game_counts": {}, "h2h": {}}

    state = elo._state[sector]
    ratings = state["ratings"]
    counts = state["game_counts"]

    elo_h = ratings.get(home, DEFAULT_ELO)
    elo_a = ratings.get(away, DEFAULT_ELO)
    ha = HOME_ADVANTAGE_ELO.get(sector, 0.0)

    expected_h = 1.0 / (1.0 + 10 ** ((elo_a - (elo_h + ha)) / 400.0))
    k = K_FACTORS.get(sector, 20.0)

    actual_h = 1.0 if home_won else 0.0
    delta = k * (actual_h - expected_h)
    ratings[home] = elo_h + delta
    ratings[away] = elo_a - delta
    counts[home] = counts.get(home, 0) + 1
    counts[away] = counts.get(away, 0) + 1


def _form_update(
    form: FormModelAgent,
    sector: str,
    home: str,
    away: str,
    home_won: bool,
    game_date: date,
) -> None:
    """Add game record to form state."""
    if sector not in form._state:
        form._state[sector] = {}

    state = form._state[sector]
    date_str = game_date.isoformat()

    if home not in state:
        state[home] = []
    state[home].append({"date": date_str, "won": home_won, "opp": away, "home": True})

    if away not in state:
        state[away] = []
    state[away].append({"date": date_str, "won": not home_won, "opp": home, "home": False})


def _poisson_update(
    poisson: PoissonModelAgent,
    sector: str,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> None:
    """Update Poisson attack/defense strengths after observing a game."""
    from evmax.agents.models.poisson_agent import LEAGUE_AVG_DEFAULTS, BUCKET_SIZE

    if sector not in poisson._state:
        avg = LEAGUE_AVG_DEFAULTS.get(sector, {"home": 1.0, "away": 1.0})
        poisson._state[sector] = {"league_avg": dict(avg), "teams": {}}

    state = poisson._state[sector]
    teams = state["teams"]
    avg = state["league_avg"]

    # Initialize teams if new
    for team in [home, away]:
        if team not in teams:
            teams[team] = {"attack": 1.0, "defense": 1.0, "games": 0, "total_scored": 0, "total_conceded": 0}

    h = teams[home]
    a = teams[away]

    # Update running totals
    h["total_scored"] += home_score
    h["total_conceded"] += away_score
    h["games"] += 1
    a["total_scored"] += away_score
    a["total_conceded"] += home_score
    a["games"] += 1

    # Recalculate attack/defense relative to league average
    # Attack = team's avg scored / league avg scored
    # Defense = team's avg conceded / league avg conceded
    league_avg_scored = (avg["home"] + avg["away"]) / 2
    if league_avg_scored > 0:
        for t_data in [h, a]:
            if t_data["games"] > 0:
                avg_scored = t_data["total_scored"] / t_data["games"]
                avg_conceded = t_data["total_conceded"] / t_data["games"]
                t_data["attack"] = max(0.3, min(2.5, avg_scored / league_avg_scored))
                t_data["defense"] = max(0.3, min(2.5, avg_conceded / league_avg_scored))


def _compute_metrics(
    sector: str,
    results: list[WalkForwardResult],
    elo_weight: float,
    form_weight: float,
    poisson_weight: float,
) -> WalkForwardReport:
    """Compute Brier, accuracy, calibration from walk-forward results."""
    from evmax.backtest.metrics import brier_score, accuracy_score, calibration_bins

    # Per-model
    elo_pred, elo_actual = [], []
    form_pred, form_actual = [], []
    pois_pred, pois_actual = [], []
    ens_pred, ens_actual = [], []

    for r in results:
        actual = r.home_won
        if r.elo_prob_home is not None:
            elo_pred.append(r.elo_prob_home)
            elo_actual.append(actual)
        if r.form_prob_home is not None:
            form_pred.append(r.form_prob_home)
            form_actual.append(actual)
        if r.poisson_prob_home is not None:
            pois_pred.append(r.poisson_prob_home)
            pois_actual.append(actual)
        if r.ensemble_prob_home is not None:
            ens_pred.append(r.ensemble_prob_home)
            ens_actual.append(actual)

    n_total = len(results)
    home_wins = sum(1 for r in results if r.home_won)
    home_rate = home_wins / n_total if n_total else 0.0
    # Baseline Brier: always predict home_rate
    baseline_brier = sum((home_rate - float(r.home_won)) ** 2 for r in results) / n_total if n_total else 0.0

    report = WalkForwardReport(
        sector=sector,
        n_games=n_total,
        n_predicted=len(ens_pred),
        elo_brier=brier_score(elo_pred, elo_actual),
        elo_accuracy=accuracy_score(elo_pred, elo_actual),
        elo_n=len(elo_pred),
        form_brier=brier_score(form_pred, form_actual),
        form_accuracy=accuracy_score(form_pred, form_actual),
        form_n=len(form_pred),
        poisson_brier=brier_score(pois_pred, pois_actual),
        poisson_accuracy=accuracy_score(pois_pred, pois_actual),
        poisson_n=len(pois_pred),
        ensemble_brier=brier_score(ens_pred, ens_actual),
        ensemble_accuracy=accuracy_score(ens_pred, ens_actual),
        ensemble_n=len(ens_pred),
        home_win_rate=home_rate,
        baseline_always_home_brier=baseline_brier,
        calibration_bins=calibration_bins(ens_pred, ens_actual),
        results=results,
    )
    return report
