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
from evmax.agents.models.efficiency_agent import EfficiencyModelAgent
from evmax.agents.models.wnba_efficiency_agent import WNBAEfficiencyModelAgent
from evmax.agents.models.wnba_possession_sim_agent import WNBAPossessionSimAgent
from evmax.agents.models.possession_sim_agent import PossessionSimAgent
from evmax.agents.models.shot_quality_agent import ShotQualityAgent
from evmax.agents.models.matchup_agent import MatchupAgent
from evmax.backtest.models import BacktestRow
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook

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
    efficiency_prob_home: Optional[float] = None
    possession_sim_prob_home: Optional[float] = None
    shot_quality_prob_home: Optional[float] = None
    matchup_prob_home: Optional[float] = None
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
    efficiency_brier: float = 0.0
    efficiency_accuracy: float = 0.0
    efficiency_n: int = 0
    possession_sim_brier: float = 0.0
    possession_sim_accuracy: float = 0.0
    possession_sim_n: int = 0
    shot_quality_brier: float = 0.0
    shot_quality_accuracy: float = 0.0
    shot_quality_n: int = 0
    matchup_brier: float = 0.0
    matchup_accuracy: float = 0.0
    matchup_n: int = 0
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


NBA_WEIGHT_OVERRIDES: dict[str, float] = {
    "efficiency": 0.30, "possession_sim": 0.30,
    "elo": 0.10, "form": 0.10,
    "shot_quality": 0.10, "matchup": 0.10,
    "poisson": 0.0,
}

# Mirrors SECTOR_WEIGHT_OVERRIDES["wnba"] in ensemble_agent.py. Kept in sync
# manually so the walk-forward evaluates the same blend the live scanner uses.
WNBA_WEIGHT_OVERRIDES: dict[str, float] = {
    "wnba_efficiency":     0.25,
    "wnba_possession_sim": 0.25,
    "elo":                 0.30,
    "form":                0.15,
    "poisson":             0.0,
}


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

    # NBA advanced models — use current season stats (not walk-forward)
    is_nba = sector.lower() == "nba"
    # WNBA advanced models — use prior-season stats loaded from the WNBA
    # efficiency agent's seeded state file. Intentionally not walk-forward:
    # efficiency stats stabilise slowly and opening-day predictions benefit
    # from Y−1 priors, matching live behaviour.
    is_wnba = sector.lower() == "wnba"
    efficiency_agent = None
    wnba_efficiency_agent = None
    wnba_possession_sim_agent = None
    possession_sim_agent = None
    shot_quality_agent = None
    matchup_agent = None
    if is_nba:
        efficiency_agent = EfficiencyModelAgent()
        possession_sim_agent = PossessionSimAgent()
        shot_quality_agent = ShotQualityAgent()
        matchup_agent = MatchupAgent()
    elif is_wnba:
        wnba_efficiency_agent = WNBAEfficiencyModelAgent()
        wnba_possession_sim_agent = WNBAPossessionSimAgent()

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
        form_prob = _form_predict(form, sector, home, away, game_date)
        poisson_prob = _poisson_predict(poisson, sector, home, away)

        eff_prob = None
        possim_prob = None
        sq_prob = None
        mu_prob = None
        if is_nba:
            eff_prob, possim_prob, sq_prob, mu_prob = _nba_model_predict(
                home, away, efficiency_agent, possession_sim_agent,
                shot_quality_agent, matchup_agent,
            )
        elif is_wnba:
            eff_prob = _wnba_efficiency_predict(home, away, wnba_efficiency_agent)
            possim_prob = _wnba_possession_sim_predict(
                home, away, wnba_possession_sim_agent,
            )

        # Ensemble: weighted average of available models
        if is_nba:
            w = NBA_WEIGHT_OVERRIDES
            ensemble_prob = _ensemble([
                (elo_prob, w["elo"]), (form_prob, w["form"]),
                (poisson_prob, w["poisson"]),
                (eff_prob, w["efficiency"]), (possim_prob, w["possession_sim"]),
                (sq_prob, w["shot_quality"]), (mu_prob, w["matchup"]),
            ])
        elif is_wnba:
            w = WNBA_WEIGHT_OVERRIDES
            ensemble_prob = _ensemble([
                (elo_prob, w["elo"]), (form_prob, w["form"]),
                (poisson_prob, w["poisson"]),
                (eff_prob, w["wnba_efficiency"]),
                (possim_prob, w["wnba_possession_sim"]),
            ])
        else:
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
            efficiency_prob_home=eff_prob,
            possession_sim_prob_home=possim_prob,
            shot_quality_prob_home=sq_prob,
            matchup_prob_home=mu_prob,
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


def _nba_model_predict(
    home: str,
    away: str,
    efficiency_agent: Optional[EfficiencyModelAgent],
    possession_sim_agent: Optional[PossessionSimAgent],
    shot_quality_agent: Optional[ShotQualityAgent],
    matchup_agent: Optional[MatchupAgent],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Get predictions from NBA advanced models. Returns (eff, possim, sq, matchup)."""
    prob_a_default = 0.55
    prob_b_default = 0.45
    market = PredictionMarket(
        id="bt", market_id="bt", event_id=f"bt_{home}_{away}",
        sector="nba", team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=prob_a_default, no_price=prob_b_default,
    )
    sharp = SharpOdds(
        event_id=f"bt_{home}_{away}", book=SharpBook.pinnacle, sector="nba",
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=1 / prob_a_default, outcome_b_decimal=1 / prob_b_default,
        true_prob_a=prob_a_default, true_prob_b=prob_b_default,
    )

    results = []
    for agent in [efficiency_agent, possession_sim_agent, shot_quality_agent, matchup_agent]:
        if agent is None:
            results.append(None)
            continue
        try:
            pred = asyncio.run(agent.predict_pair(market, sharp))
            results.append(pred.true_prob_a if pred else None)
        except Exception:
            results.append(None)

    return tuple(results)  # type: ignore[return-value]


def _wnba_efficiency_predict(
    home: str,
    away: str,
    agent: Optional[WNBAEfficiencyModelAgent],
) -> Optional[float]:
    """Get WNBA efficiency-model probability for home win. None if no data."""
    if agent is None:
        return None
    prob_a_default = 0.55
    prob_b_default = 0.45
    market = PredictionMarket(
        id="bt", market_id="bt", event_id=f"bt_{home}_{away}",
        sector="wnba", team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=prob_a_default, no_price=prob_b_default,
    )
    sharp = SharpOdds(
        event_id=f"bt_{home}_{away}", book=SharpBook.pinnacle, sector="wnba",
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=1 / prob_a_default, outcome_b_decimal=1 / prob_b_default,
        true_prob_a=prob_a_default, true_prob_b=prob_b_default,
    )
    try:
        pred = asyncio.run(agent.predict_pair(market, sharp))
        return pred.true_prob_a if pred else None
    except Exception:
        return None


def _wnba_possession_sim_predict(
    home: str,
    away: str,
    agent: Optional[WNBAPossessionSimAgent],
) -> Optional[float]:
    """Get WNBA possession-sim probability for home win. None if no data."""
    if agent is None:
        return None
    prob_a_default = 0.55
    prob_b_default = 0.45
    market = PredictionMarket(
        id="bt", market_id="bt", event_id=f"bt_{home}_{away}",
        sector="wnba", team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=prob_a_default, no_price=prob_b_default,
    )
    sharp = SharpOdds(
        event_id=f"bt_{home}_{away}", book=SharpBook.pinnacle, sector="wnba",
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=1 / prob_a_default, outcome_b_decimal=1 / prob_b_default,
        true_prob_a=prob_a_default, true_prob_b=prob_b_default,
    )
    try:
        pred = asyncio.run(agent.predict_pair(market, sharp))
        return pred.true_prob_a if pred else None
    except Exception:
        return None


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


def _form_predict(
    form: FormModelAgent,
    sector: str,
    home: str,
    away: str,
    game_date: Optional[date] = None,
) -> Optional[float]:
    """Get Form prediction. None if insufficient data or records are stale.

    Delegates to the agent's own _form_rate / _is_stale so the walk-forward
    exercises the same staleness guard and opponent-quality weighting the
    live scanner uses. `game_date` is the date of the game being predicted;
    staleness is checked relative to it (not wall-clock today), which is
    essential for historical replays.
    """
    from evmax.agents.models.form_agent import HOME_ADJ, GameRecord

    state = form._state.get(sector, {})
    home_records_raw = state.get(home, [])
    away_records_raw = state.get(away, [])
    if len(home_records_raw) < 3 or len(away_records_raw) < 3:
        return None

    home_records = [GameRecord(**r) for r in home_records_raw]
    away_records = [GameRecord(**r) for r in away_records_raw]

    if form._is_stale(home_records, game_date) or form._is_stale(away_records, game_date):
        return None

    fa = form._form_rate(home_records)
    fb = form._form_rate(away_records)

    if fa + fb == 0 or fa + fb == 2 * fa * fb:
        prob = 0.5
    else:
        prob = (fa - fa * fb) / (fa + fb - 2 * fa * fb)

    ha = HOME_ADJ.get(sector, 0.0)
    return max(0.01, min(0.99, prob + ha))


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


@dataclass
class SpreadBacktestResult:
    """One spread prediction from a single game × line combination."""
    date: date
    home: str
    away: str
    line: float
    sim_cover_prob: float
    cdf_cover_prob: float
    actual_margin: float
    covered: bool


@dataclass
class SpreadBacktestReport:
    """Summary of spread backtest across all games and lines."""
    n_games: int
    n_predictions: int
    # Sim-based
    sim_brier: float = 0.0
    sim_accuracy: float = 0.0
    # Normal CDF baseline
    cdf_brier: float = 0.0
    cdf_accuracy: float = 0.0
    # Calibration (sim)
    calibration_bins: list = field(default_factory=list)
    # Calibration (cdf)
    cdf_calibration_bins: list = field(default_factory=list)
    # Per-line breakdown
    per_line: dict = field(default_factory=dict)
    results: list[SpreadBacktestResult] = field(default_factory=list)


SPREAD_LINES = [-1.5, -3.5, -5.5, -7.5, -9.5, -11.5, -13.5]


def run_spread_backtest(months: list[str]) -> SpreadBacktestReport:
    """Backtest PossessionSim spread predictions against actual NBA margins.

    For each historical game, runs PossessionSim to get the margin distribution,
    then tests cover probability at standard spread lines. Compares sim-based
    predictions against normal CDF with σ=11.5.
    """
    from scipy.stats import norm as norm_dist

    games = fetch_espn_games("nba", months)
    if not games:
        return SpreadBacktestReport(n_games=0, n_predictions=0)

    logger.info("spread_backtest_start", n_games=len(games))

    sim_agent = PossessionSimAgent()
    norm_obj = NameNormalizer("nba")
    sigma_cdf = 11.5

    all_results: list[SpreadBacktestResult] = []
    per_line_data: dict[float, list[tuple[float, float, bool]]] = {
        line: [] for line in SPREAD_LINES
    }

    for g in games:
        home = norm_obj.normalize(g["home"])
        away = norm_obj.normalize(g["away"])
        if not home or not away:
            continue

        actual_margin = g["home_score"] - g["away_score"]

        market = PredictionMarket(
            id="bt", market_id="bt", event_id=f"spread_{home}_{away}",
            sector="nba", team_home=home, team_away=away,
            source=MarketSource.kalshi, yes_price=0.55, no_price=0.45,
        )
        sharp = SharpOdds(
            event_id=f"spread_{home}_{away}", book=SharpBook.pinnacle, sector="nba",
            outcome_a_label=home, outcome_b_label=away,
            outcome_a_decimal=1.82, outcome_b_decimal=2.22,
            true_prob_a=0.55, true_prob_b=0.45,
        )

        try:
            pred = asyncio.run(sim_agent.predict_pair(market, sharp))
        except Exception:
            continue

        if pred is None:
            continue

        margins = sim_agent._margin_cache.get(sharp.event_id)
        if margins is None:
            continue

        sim_mean = float(margins.mean())

        for line in SPREAD_LINES:
            sim_prob = sim_agent.cover_probability(sharp.event_id, line)
            if sim_prob is None:
                continue

            z = (abs(line) - sim_mean) / sigma_cdf
            cdf_prob = float(1.0 - norm_dist.cdf(z))
            cdf_prob = max(0.01, min(0.99, cdf_prob))

            covered = actual_margin > abs(line)

            result = SpreadBacktestResult(
                date=g["date"], home=home, away=away,
                line=line, sim_cover_prob=sim_prob,
                cdf_cover_prob=cdf_prob, actual_margin=actual_margin,
                covered=covered,
            )
            all_results.append(result)
            per_line_data[line].append((sim_prob, cdf_prob, covered))

    if not all_results:
        return SpreadBacktestReport(n_games=len(games), n_predictions=0)

    sim_preds = [r.sim_cover_prob for r in all_results]
    cdf_preds = [r.cdf_cover_prob for r in all_results]
    actuals = [r.covered for r in all_results]

    from evmax.backtest.metrics import brier_score, accuracy_score, calibration_bins

    per_line_summary = {}
    for line in SPREAD_LINES:
        data = per_line_data[line]
        if not data:
            continue
        s_preds = [d[0] for d in data]
        c_preds = [d[1] for d in data]
        acts = [d[2] for d in data]
        per_line_summary[line] = {
            "n": len(data),
            "sim_brier": brier_score(s_preds, acts),
            "sim_accuracy": accuracy_score(s_preds, acts),
            "cdf_brier": brier_score(c_preds, acts),
            "cdf_accuracy": accuracy_score(c_preds, acts),
            "actual_cover_rate": sum(acts) / len(acts),
            "sim_mean_prob": sum(s_preds) / len(s_preds),
            "cdf_mean_prob": sum(c_preds) / len(c_preds),
        }

    return SpreadBacktestReport(
        n_games=len(games),
        n_predictions=len(all_results),
        sim_brier=brier_score(sim_preds, actuals),
        sim_accuracy=accuracy_score(sim_preds, actuals),
        cdf_brier=brier_score(cdf_preds, actuals),
        cdf_accuracy=accuracy_score(cdf_preds, actuals),
        calibration_bins=calibration_bins(sim_preds, actuals),
        cdf_calibration_bins=calibration_bins(cdf_preds, actuals),
        per_line=per_line_summary,
        results=all_results,
    )


@dataclass
class TotalsBacktestResult:
    """One totals prediction from a single game × line combination."""
    date: date
    home: str
    away: str
    line: float
    sim_over_prob: float
    cdf_over_prob: float
    actual_total: float
    went_over: bool


@dataclass
class TotalsBacktestReport:
    """Summary of totals backtest across all games and lines."""
    n_games: int
    n_predictions: int
    sim_brier: float = 0.0
    sim_accuracy: float = 0.0
    cdf_brier: float = 0.0
    cdf_accuracy: float = 0.0
    calibration_bins: list = field(default_factory=list)
    cdf_calibration_bins: list = field(default_factory=list)
    per_line: dict = field(default_factory=dict)
    results: list[TotalsBacktestResult] = field(default_factory=list)


# Typical NBA totals range (2025-26 median ~225)
TOTALS_LINES = [210.5, 215.5, 220.5, 225.5, 230.5, 235.5, 240.5]
TOTALS_SIGMA_CDF = 20.0  # empirical std of NBA game totals


def run_totals_backtest(months: list[str]) -> TotalsBacktestReport:
    """Backtest PossessionSim totals predictions against actual NBA game totals.

    For each historical game, runs PossessionSim to get the total distribution,
    then tests P(total > line) at standard lines. Compares sim-based predictions
    against a normal CDF baseline anchored on the sim mean with σ=20.
    """
    from scipy.stats import norm as norm_dist

    games = fetch_espn_games("nba", months)
    if not games:
        return TotalsBacktestReport(n_games=0, n_predictions=0)

    logger.info("totals_backtest_start", n_games=len(games))

    sim_agent = PossessionSimAgent()
    norm_obj = NameNormalizer("nba")

    all_results: list[TotalsBacktestResult] = []
    per_line_data: dict[float, list[tuple[float, float, bool]]] = {
        line: [] for line in TOTALS_LINES
    }

    for g in games:
        home = norm_obj.normalize(g["home"])
        away = norm_obj.normalize(g["away"])
        if not home or not away:
            continue

        actual_total = g["home_score"] + g["away_score"]

        event_id = f"totals_{home}_{away}_{g['date'].isoformat()}"
        try:
            sim = sim_agent.simulate_matchup(home, away, event_id=event_id)
        except Exception:
            continue
        if sim is None:
            continue

        sim_total_mean = float(sim["total"].mean())

        for line in TOTALS_LINES:
            sim_prob = sim_agent.total_probability(event_id, line, is_over=True)
            if sim_prob is None:
                continue

            z = (line - sim_total_mean) / TOTALS_SIGMA_CDF
            cdf_prob = float(1.0 - norm_dist.cdf(z))
            cdf_prob = max(0.01, min(0.99, cdf_prob))

            went_over = actual_total > line

            result = TotalsBacktestResult(
                date=g["date"], home=home, away=away,
                line=line, sim_over_prob=sim_prob,
                cdf_over_prob=cdf_prob, actual_total=actual_total,
                went_over=went_over,
            )
            all_results.append(result)
            per_line_data[line].append((sim_prob, cdf_prob, went_over))

    if not all_results:
        return TotalsBacktestReport(n_games=len(games), n_predictions=0)

    sim_preds = [r.sim_over_prob for r in all_results]
    cdf_preds = [r.cdf_over_prob for r in all_results]
    actuals = [r.went_over for r in all_results]

    from evmax.backtest.metrics import brier_score, accuracy_score, calibration_bins

    per_line_summary = {}
    for line in TOTALS_LINES:
        data = per_line_data[line]
        if not data:
            continue
        s_preds = [d[0] for d in data]
        c_preds = [d[1] for d in data]
        acts = [d[2] for d in data]
        per_line_summary[line] = {
            "n": len(data),
            "sim_brier": brier_score(s_preds, acts),
            "sim_accuracy": accuracy_score(s_preds, acts),
            "cdf_brier": brier_score(c_preds, acts),
            "cdf_accuracy": accuracy_score(c_preds, acts),
            "actual_over_rate": sum(acts) / len(acts),
            "sim_mean_prob": sum(s_preds) / len(s_preds),
            "cdf_mean_prob": sum(c_preds) / len(c_preds),
        }

    return TotalsBacktestReport(
        n_games=len(games),
        n_predictions=len(all_results),
        sim_brier=brier_score(sim_preds, actuals),
        sim_accuracy=accuracy_score(sim_preds, actuals),
        cdf_brier=brier_score(cdf_preds, actuals),
        cdf_accuracy=accuracy_score(cdf_preds, actuals),
        calibration_bins=calibration_bins(sim_preds, actuals),
        cdf_calibration_bins=calibration_bins(cdf_preds, actuals),
        per_line=per_line_summary,
        results=all_results,
    )


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
    eff_pred, eff_actual = [], []
    possim_pred, possim_actual = [], []
    sq_pred, sq_actual = [], []
    mu_pred, mu_actual = [], []
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
        if r.efficiency_prob_home is not None:
            eff_pred.append(r.efficiency_prob_home)
            eff_actual.append(actual)
        if r.possession_sim_prob_home is not None:
            possim_pred.append(r.possession_sim_prob_home)
            possim_actual.append(actual)
        if r.shot_quality_prob_home is not None:
            sq_pred.append(r.shot_quality_prob_home)
            sq_actual.append(actual)
        if r.matchup_prob_home is not None:
            mu_pred.append(r.matchup_prob_home)
            mu_actual.append(actual)
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
        efficiency_brier=brier_score(eff_pred, eff_actual),
        efficiency_accuracy=accuracy_score(eff_pred, eff_actual),
        efficiency_n=len(eff_pred),
        possession_sim_brier=brier_score(possim_pred, possim_actual),
        possession_sim_accuracy=accuracy_score(possim_pred, possim_actual),
        possession_sim_n=len(possim_pred),
        shot_quality_brier=brier_score(sq_pred, sq_actual),
        shot_quality_accuracy=accuracy_score(sq_pred, sq_actual),
        shot_quality_n=len(sq_pred),
        matchup_brier=brier_score(mu_pred, mu_actual),
        matchup_accuracy=accuracy_score(mu_pred, mu_actual),
        matchup_n=len(mu_pred),
        ensemble_brier=brier_score(ens_pred, ens_actual),
        ensemble_accuracy=accuracy_score(ens_pred, ens_actual),
        ensemble_n=len(ens_pred),
        home_win_rate=home_rate,
        baseline_always_home_brier=baseline_brier,
        calibration_bins=calibration_bins(ens_pred, ens_actual),
        results=results,
    )
    return report
