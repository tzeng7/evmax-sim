"""Walk-forward backtest for soccer — per-model Brier on ALL games.

Unlike the Pinnacle-only soccer backtest (sources/soccer_csv.py), this one
replays historical matches chronologically through the live Elo/Form/Poisson
agents: predict-with-current-state, then update-with-actual-result. This
avoids the selection bias in live predictions.db (+EV-filtered subset) and
gives a clean per-model calibration read on the unfiltered universe.

Key differences from espn_walkforward:
  - Soccer is 3-way (home/draw/away). We report both binary Brier (on
    home_won using P_h from the 3-way distribution) and 3-way Brier
    (sum over one-hot legs) to see if draw handling is the problem.
  - Data source is football-data.co.uk CSVs already cached for the
    Pinnacle-only backtest (parse_soccer_csv).
  - Pinnacle reference Brier is computed on the same row set so the
    model columns and the market column are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import structlog

from evmax.agents.models.elo_agent import (
    DEFAULT_ELO,
    HOME_ADVANTAGE_ELO,
    EloModelAgent,
)
from evmax.agents.models.form_agent import FormModelAgent, GameRecord, HOME_ADJ
from evmax.agents.models.poisson_agent import (
    BUCKET_SIZE,
    LEAGUE_AVG_DEFAULTS,
    MAX_SCORE,
    PoissonModelAgent,
    _poisson_pmf,
    _shrink as _poisson_shrink,
)
from evmax.backtest.models import BacktestRow

logger = structlog.get_logger(__name__)


@dataclass
class SoccerWalkForwardRow:
    """One game's walk-forward prediction."""

    date: date
    league: str
    home: str
    away: str
    result: str  # "H"/"D"/"A"
    # Each model's 3-way prediction at the time the game was played
    # (None if the model didn't have enough data yet)
    elo_ph: Optional[float] = None
    elo_pd: Optional[float] = None
    elo_pa: Optional[float] = None
    form_ph: Optional[float] = None
    form_pd: Optional[float] = None
    form_pa: Optional[float] = None
    poisson_ph: Optional[float] = None
    poisson_pd: Optional[float] = None
    poisson_pa: Optional[float] = None
    xg_ph: Optional[float] = None
    xg_pd: Optional[float] = None
    xg_pa: Optional[float] = None
    # Pinnacle reference (already devigged)
    sharp_ph: Optional[float] = None
    sharp_pd: Optional[float] = None
    sharp_pa: Optional[float] = None


@dataclass
class CalibrationBucket:
    """One row of the calibration curve."""
    prob_low: float
    prob_high: float
    n: int
    mean_pred: float
    actual_rate: float


@dataclass
class SoccerWalkForwardMetrics:
    """Per-model Brier / accuracy over the walk-forward corpus."""

    n: int = 0
    brier_binary: float = 0.0  # P(home) vs home_won {0,1}
    brier_3way: float = 0.0    # sum over legs vs one-hot
    accuracy: float = 0.0       # correct when argmax matches actual
    # Miscalibration breakdown on the binary P(home)
    mean_pred_home: float = 0.0
    mean_actual_home: float = 0.0
    # Mean predicted draw vs actual draw rate — exposes draw-allocator bias
    mean_pred_draw: float = 0.0
    mean_actual_draw: float = 0.0
    calibration: list[CalibrationBucket] = field(default_factory=list)


@dataclass
class SoccerWalkForwardReport:
    n_games: int
    by_model: dict[str, SoccerWalkForwardMetrics] = field(default_factory=dict)
    # Per-league breakdown of the ensemble
    by_league: dict[str, dict] = field(default_factory=dict)
    # Home base rate for reference
    home_win_rate: float = 0.0
    draw_rate: float = 0.0
    # Where the stat ensemble disagrees with sharp on the argmax outcome:
    #   n_disagree = games where argmax(stat) != argmax(sharp)
    #   stat_correct = of those, how many stat called correctly
    #   sharp_correct = of those, how many sharp called correctly
    # Tells us whether stat models add value when they differ from sharp.
    n_disagree: int = 0
    stat_correct_on_disagree: int = 0
    sharp_correct_on_disagree: int = 0


# --------------------------------------------------------------------------- #
# Prediction helpers — mirror the live agents, but operate on a bare state
# dict we own so we can reset between runs.
# --------------------------------------------------------------------------- #


def _elo_3way(
    state: dict,
    sector: str,
    home: str,
    away: str,
) -> Optional[tuple[float, float, float]]:
    ratings = state.get("ratings", {})
    if home not in ratings and away not in ratings:
        return None
    ha = HOME_ADVANTAGE_ELO.get(sector, 0.0)
    elo_h = ratings.get(home, DEFAULT_ELO) + ha
    elo_a = ratings.get(away, DEFAULT_ELO)
    exp_h = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
    exp_a = 1.0 - exp_h
    # Same draw allocation as the live EloModelAgent._win_probs for soccer
    gap = abs(exp_h - exp_a)
    draw_prob = max(0.10, 0.30 - 0.40 * gap * gap)
    scale = (1.0 - draw_prob) / (exp_h + exp_a)
    return exp_h * scale, draw_prob, exp_a * scale


def _form_3way(
    form: FormModelAgent,
    sector: str,
    home: str,
    away: str,
    game_date: date,
) -> Optional[tuple[float, float, float]]:
    state = form._state.get(sector, {})
    home_raw = state.get(home, [])
    away_raw = state.get(away, [])
    if len(home_raw) < 3 or len(away_raw) < 3:
        return None

    home_records = [GameRecord(**r) for r in home_raw]
    away_records = [GameRecord(**r) for r in away_raw]
    if form._is_stale(home_records, game_date) or form._is_stale(away_records, game_date):
        return None

    fa = form._form_rate(home_records)
    fb = form._form_rate(away_records)
    if fa + fb == 0 or fa + fb == 2 * fa * fb:
        prob_h = 0.5
    else:
        prob_h = (fa - fa * fb) / (fa + fb - 2 * fa * fb)

    ha = HOME_ADJ.get(sector, 0.0)
    prob_h = max(0.01, min(0.99, prob_h + ha))
    prob_a = 1.0 - prob_h
    gap = abs(prob_h - prob_a)
    draw_prob = max(0.10, 0.30 - 0.40 * gap * gap)
    scale = (1.0 - draw_prob) / (prob_h + prob_a)
    return prob_h * scale, draw_prob, prob_a * scale


def _poisson_3way(
    state: dict,
    sector: str,
    home: str,
    away: str,
) -> Optional[tuple[float, float, float]]:
    teams = state.get("teams", {})
    if home not in teams or away not in teams:
        return None

    avg = state.get("league_avg", LEAGUE_AVG_DEFAULTS.get(sector, {"home": 1.0, "away": 1.0}))
    h_data = teams[home]
    a_data = teams[away]
    bucket = BUCKET_SIZE.get(sector, 1)
    max_s = MAX_SCORE.get(sector, 10)

    # Shrink ratios toward 1.0 based on games seen — matches live PoissonModelAgent.
    h_games = h_data.get("games", 0)
    a_games = a_data.get("games", 0)
    h_att = _poisson_shrink(h_data["attack"], h_games)
    h_def = _poisson_shrink(h_data["defense"], h_games)
    a_att = _poisson_shrink(a_data["attack"], a_games)
    a_def = _poisson_shrink(a_data["defense"], a_games)

    lam_h = h_att * a_def * avg["home"] / bucket
    lam_a = a_att * h_def * avg["away"] / bucket

    p_h = p_d = p_a = 0.0
    for i in range(max_s + 1):
        pi = _poisson_pmf(lam_h, i)
        for j in range(max_s + 1):
            p = pi * _poisson_pmf(lam_a, j)
            if i > j:
                p_h += p
            elif i == j:
                p_d += p
            else:
                p_a += p
    total = p_h + p_d + p_a
    if total <= 0:
        return None
    return p_h / total, p_d / total, p_a / total


# --------------------------------------------------------------------------- #
# Update helpers — apply the actual result to each model's state.
# --------------------------------------------------------------------------- #


def _elo_update(state: dict, sector: str, home: str, away: str, row: BacktestRow) -> None:
    from evmax.agents.models.elo_agent import K_FACTORS

    ratings = state.setdefault("ratings", {})
    counts = state.setdefault("game_counts", {})
    elo_h = ratings.get(home, DEFAULT_ELO)
    elo_a = ratings.get(away, DEFAULT_ELO)
    ha = HOME_ADVANTAGE_ELO.get(sector, 0.0)
    expected_h = 1.0 / (1.0 + 10 ** ((elo_a - (elo_h + ha)) / 400.0))
    k = K_FACTORS.get(sector, 20.0)

    # Draws score 0.5; wins 1.0; losses 0.0 — standard 3-way Elo.
    if row.draw:
        actual_h = 0.5
    elif row.home_won:
        actual_h = 1.0
    else:
        actual_h = 0.0

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
    row: BacktestRow,
) -> None:
    state = form._state.setdefault(sector, {})
    date_str = row.date.isoformat()
    drew = bool(row.draw)
    home_won_bool = bool(row.home_won) and not drew
    away_won_bool = (not home_won_bool) and (not drew)
    state.setdefault(home, []).append(
        {"date": date_str, "won": home_won_bool, "opp": away, "home": True, "drew": drew}
    )
    state.setdefault(away, []).append(
        {"date": date_str, "won": away_won_bool, "opp": home, "home": False, "drew": drew}
    )


def _poisson_update(state: dict, sector: str, home: str, away: str, row: BacktestRow) -> None:
    if row.home_goals is None or row.away_goals is None:
        return
    avg = state.setdefault("league_avg", dict(LEAGUE_AVG_DEFAULTS.get(sector, {"home": 1.0, "away": 1.0})))
    teams = state.setdefault("teams", {})

    for team in (home, away):
        if team not in teams:
            teams[team] = {"attack": 1.0, "defense": 1.0, "games": 0, "total_scored": 0, "total_conceded": 0}

    h = teams[home]
    a = teams[away]
    h["total_scored"] += row.home_goals
    h["total_conceded"] += row.away_goals
    h["games"] += 1
    a["total_scored"] += row.away_goals
    a["total_conceded"] += row.home_goals
    a["games"] += 1

    league_avg_scored = (avg["home"] + avg["away"]) / 2
    if league_avg_scored > 0:
        for t_data in (h, a):
            if t_data["games"] > 0:
                avg_scored = t_data["total_scored"] / t_data["games"]
                avg_conceded = t_data["total_conceded"] / t_data["games"]
                t_data["attack"] = max(0.3, min(2.5, avg_scored / league_avg_scored))
                t_data["defense"] = max(0.3, min(2.5, avg_conceded / league_avg_scored))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _onehot(result: str) -> tuple[float, float, float]:
    return (1.0 if result == "H" else 0.0, 1.0 if result == "D" else 0.0, 1.0 if result == "A" else 0.0)


CALIBRATION_EDGES: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01)


def _compute_metrics(
    rows: list[SoccerWalkForwardRow],
    getter,
) -> SoccerWalkForwardMetrics:
    """Compute Brier/accuracy + calibration curve for one model.

    `getter` returns (ph, pd, pa) per row, or (None, _, _) to skip."""
    total = 0
    brier_bin = 0.0
    brier_3 = 0.0
    correct = 0
    sum_pred_h = 0.0
    sum_actual_h = 0
    sum_pred_d = 0.0
    sum_actual_d = 0

    # Per-bucket counters (by P(home))
    edges = CALIBRATION_EDGES
    bucket_n = [0] * (len(edges) - 1)
    bucket_pred_sum = [0.0] * (len(edges) - 1)
    bucket_actual = [0] * (len(edges) - 1)

    for r in rows:
        ph, pd, pa = getter(r)
        if ph is None:
            continue
        yh, yd, ya = _onehot(r.result)
        total += 1
        brier_bin += (ph - yh) ** 2
        brier_3 += (ph - yh) ** 2 + (pd - yd) ** 2 + (pa - ya) ** 2
        probs = {"H": ph, "D": pd, "A": pa}
        if max(probs, key=probs.get) == r.result:
            correct += 1
        sum_pred_h += ph
        sum_actual_h += int(yh)
        if pd is not None:
            sum_pred_d += pd
            sum_actual_d += int(yd)
        # bucket by P(home)
        for i in range(len(edges) - 1):
            if edges[i] <= ph < edges[i + 1]:
                bucket_n[i] += 1
                bucket_pred_sum[i] += ph
                bucket_actual[i] += int(yh)
                break

    if total == 0:
        return SoccerWalkForwardMetrics()

    buckets: list[CalibrationBucket] = []
    for i in range(len(edges) - 1):
        if bucket_n[i] == 0:
            continue
        buckets.append(
            CalibrationBucket(
                prob_low=edges[i],
                prob_high=min(edges[i + 1], 1.0),
                n=bucket_n[i],
                mean_pred=bucket_pred_sum[i] / bucket_n[i],
                actual_rate=bucket_actual[i] / bucket_n[i],
            )
        )

    return SoccerWalkForwardMetrics(
        n=total,
        brier_binary=brier_bin / total,
        brier_3way=brier_3 / total,
        accuracy=correct / total,
        mean_pred_home=sum_pred_h / total,
        mean_actual_home=sum_actual_h / total,
        mean_pred_draw=sum_pred_d / total,
        mean_actual_draw=sum_actual_d / total,
        calibration=buckets,
    )


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #


SOCCER_W: dict[str, float] = {"elo": 0.15, "form": 0.10, "poisson": 0.40, "xg": 0.25}
BASE_SHARP_WEIGHT: float = 0.40

# Mirror the per-league tier from data/soccer_league_tiers.yaml. The walk-
# forward operates on football-data.co.uk league labels, not Kalshi series,
# so the mapping is inlined here. Top-5 European leagues are the ones where
# Pinnacle's closing Brier is the informational ceiling.
TOP_TIER_LEAGUES: set[str] = {"EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
TOP_TIER_WEIGHT: float = 0.85


def base_sharp_weight_for(r: SoccerWalkForwardRow) -> float:
    return TOP_TIER_WEIGHT if r.league in TOP_TIER_LEAGUES else BASE_SHARP_WEIGHT


def stat_ensemble(
    r: SoccerWalkForwardRow,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Equal-weighted Elo+Form+Poisson+xG, ignoring sharp. Used for disagreement
    measurement and as a diagnostic baseline."""
    probs = [
        (r.elo_ph, r.elo_pd, r.elo_pa),
        (r.form_ph, r.form_pd, r.form_pa),
        (r.poisson_ph, r.poisson_pd, r.poisson_pa),
        (r.xg_ph, r.xg_pd, r.xg_pa),
    ]
    probs = [p for p in probs if p[0] is not None]
    if not probs:
        return (None, None, None)
    n = len(probs)
    return (sum(p[0] for p in probs) / n, sum(p[1] for p in probs) / n, sum(p[2] for p in probs) / n)


def _model_side_blend(
    r: SoccerWalkForwardRow,
) -> Optional[tuple[float, float, float]]:
    """SOCCER_W-weighted average of Elo/Form/Poisson/xG (renormalized across
    available models). Returns None when no model produced a prediction."""
    contribs: list[tuple[float, float, float, float]] = []
    for name, (ph, pd, pa) in [
        ("elo", (r.elo_ph, r.elo_pd, r.elo_pa)),
        ("form", (r.form_ph, r.form_pd, r.form_pa)),
        ("poisson", (r.poisson_ph, r.poisson_pd, r.poisson_pa)),
        ("xg", (r.xg_ph, r.xg_pd, r.xg_pa)),
    ]:
        if ph is None:
            continue
        contribs.append((SOCCER_W[name], ph, pd, pa))
    if not contribs:
        return None
    total_w = sum(c[0] for c in contribs)
    model_h = sum(c[0] * c[1] for c in contribs) / total_w
    model_d = sum(c[0] * c[2] for c in contribs) / total_w
    model_a = sum(c[0] * c[3] for c in contribs) / total_w
    return model_h, model_d, model_a


def blended_ensemble_3way(
    r: SoccerWalkForwardRow,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Production blend: SECTOR_WEIGHT_OVERRIDES + tier-aware sharp_weight +
    disagreement-adjusted sharp lean. Mirrors EnsembleModelAgent._blend
    minus xG (not available in walk-forward) and FLB (handled outside).

    Used by `run_soccer_walkforward` for the `blended_ensemble` row of the
    report and by `scripts/fit_soccer_calibration.py` to fit a post-hoc
    isotonic calibrator on this exact output."""
    from evmax.agents.models.ensemble_agent import EnsembleModelAgent

    side = _model_side_blend(r)
    if side is None or r.sharp_ph is None:
        return (None, None, None)
    model_h, model_d, model_a = side
    eff_sw = EnsembleModelAgent._disagreement_sharp_weight(
        model_h, model_a, model_d,
        r.sharp_ph, r.sharp_pa, r.sharp_pd,
        base_sharp_weight=base_sharp_weight_for(r),
        sector="soccer",
    )
    mw = 1.0 - eff_sw
    ph = eff_sw * r.sharp_ph + mw * model_h
    pd = eff_sw * r.sharp_pd + mw * model_d
    pa = eff_sw * r.sharp_pa + mw * model_a
    s = ph + pd + pa
    return (ph / s, pd / s, pa / s) if s > 1e-9 else (None, None, None)


def blended_ensemble_no_disagree(
    r: SoccerWalkForwardRow,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Same blend as `blended_ensemble_3way` but without the disagreement
    sharp_weight bump — kept as the diagnostic baseline."""
    side = _model_side_blend(r)
    if side is None or r.sharp_ph is None:
        return (None, None, None)
    model_h, model_d, model_a = side
    bsw = base_sharp_weight_for(r)
    mw = 1.0 - bsw
    ph = bsw * r.sharp_ph + mw * model_h
    pd = bsw * r.sharp_pd + mw * model_d
    pa = bsw * r.sharp_pa + mw * model_a
    s = ph + pd + pa
    return (ph / s, pd / s, pa / s) if s > 1e-9 else (None, None, None)


def _xg_3way(
    xg_agent,
    home: str,
    away: str,
) -> Optional[tuple[float, float, float]]:
    """Predict via the live SoccerXgAgent's internal helpers without the
    market/sharp_odds plumbing. Returns None if either team has < MIN_MATCHES."""
    from evmax.agents.models.soccer_xg_agent import (
        _score_matrix, _win_draw_probs,
    )
    lam_h = xg_agent._regressed_lambda(home, is_home=True, opponent=away)
    lam_a = xg_agent._regressed_lambda(away, is_home=False, opponent=home)
    if lam_h is None or lam_a is None:
        return None
    matrix = _score_matrix(lam_h, lam_a)
    p_h, p_d, p_a = _win_draw_probs(matrix)
    return p_h, p_d, p_a


def _xg_update(xg_agent, row: BacktestRow) -> None:
    """Feed both legs of a played match into the xG agent's rolling window."""
    if (row.home_shots is None or row.away_shots is None
            or row.home_sot is None or row.away_sot is None
            or row.home_goals is None or row.away_goals is None):
        return
    xg_agent.record_match(
        team=row.team_home, goals_for=row.home_goals, goals_against=row.away_goals,
        shots_on_target=row.home_sot, total_shots=row.home_shots,
        opponent_sot=row.away_sot, opponent_shots=row.away_shots,
        match_date=row.date.isoformat(), is_home=True,
    )
    xg_agent.record_match(
        team=row.team_away, goals_for=row.away_goals, goals_against=row.home_goals,
        shots_on_target=row.away_sot, total_shots=row.away_shots,
        opponent_sot=row.home_sot, opponent_shots=row.home_shots,
        match_date=row.date.isoformat(), is_home=False,
    )


def walk_forward_predictions(rows: list[BacktestRow]) -> list[SoccerWalkForwardRow]:
    """Replay rows chronologically through fresh Elo/Form/Poisson/xG agents,
    returning per-game predictions WITHOUT the metrics aggregation.

    Useful for downstream consumers (calibration fitting) that need the raw
    blend output rather than Brier summaries."""
    from evmax.agents.models.soccer_xg_agent import SoccerXgAgent

    rows_sorted = sorted(rows, key=lambda r: (r.date, r.team_home))

    elo_state: dict = {"ratings": {}, "game_counts": {}, "h2h": {}}
    form = FormModelAgent()
    form._state = {}
    poisson_state: dict = {}
    # Walk-forward needs a fresh xG agent — instantiate, then wipe the state
    # that load_state() pulled from disk so we don't leak production data.
    xg_agent = SoccerXgAgent()
    xg_agent._state = {"teams": {}}

    predictions: list[SoccerWalkForwardRow] = []
    for row in rows_sorted:
        home = row.team_home
        away = row.team_away

        elo_probs = _elo_3way(elo_state, "soccer", home, away)
        form_probs = _form_3way(form, "soccer", home, away, row.date)
        poisson_probs = _poisson_3way(poisson_state, "soccer", home, away)
        # xG agent normalizes team keys to lowercase internally; the helper
        # passes through whatever we hand it. Keep raw casing for consistency
        # with the other agents which use the row's team_home directly.
        xg_probs = _xg_3way(xg_agent, home.lower().strip(), away.lower().strip())

        predictions.append(
            SoccerWalkForwardRow(
                date=row.date,
                league=row.league,
                home=home,
                away=away,
                result=row.result,
                elo_ph=elo_probs[0] if elo_probs else None,
                elo_pd=elo_probs[1] if elo_probs else None,
                elo_pa=elo_probs[2] if elo_probs else None,
                form_ph=form_probs[0] if form_probs else None,
                form_pd=form_probs[1] if form_probs else None,
                form_pa=form_probs[2] if form_probs else None,
                poisson_ph=poisson_probs[0] if poisson_probs else None,
                poisson_pd=poisson_probs[1] if poisson_probs else None,
                poisson_pa=poisson_probs[2] if poisson_probs else None,
                xg_ph=xg_probs[0] if xg_probs else None,
                xg_pd=xg_probs[1] if xg_probs else None,
                xg_pa=xg_probs[2] if xg_probs else None,
                sharp_ph=row.true_prob_home,
                sharp_pd=row.true_prob_draw,
                sharp_pa=row.true_prob_away,
            )
        )

        _elo_update(elo_state, "soccer", home, away, row)
        _form_update(form, "soccer", home, away, row)
        _poisson_update(poisson_state, "soccer", home, away, row)
        _xg_update(xg_agent, row)

    return predictions


def run_soccer_walkforward(rows: list[BacktestRow]) -> SoccerWalkForwardReport:
    """Replay rows chronologically through fresh Elo/Form/Poisson agents.

    rows: BacktestRow objects with populated Pinnacle prob fields AND home_goals/away_goals.
    Returns a SoccerWalkForwardReport with per-model Brier + 3-way Brier + accuracy.
    """
    predictions = walk_forward_predictions(rows)

    # --- Metrics per model ------------------------------------------------ #
    by_model: dict[str, SoccerWalkForwardMetrics] = {
        "elo": _compute_metrics(predictions, lambda r: (r.elo_ph, r.elo_pd, r.elo_pa)),
        "form": _compute_metrics(predictions, lambda r: (r.form_ph, r.form_pd, r.form_pa)),
        "poisson": _compute_metrics(predictions, lambda r: (r.poisson_ph, r.poisson_pd, r.poisson_pa)),
        "xg": _compute_metrics(predictions, lambda r: (r.xg_ph, r.xg_pd, r.xg_pa)),
        "sharp": _compute_metrics(predictions, lambda r: (r.sharp_ph, r.sharp_pd, r.sharp_pa)),
        "stat_ensemble": _compute_metrics(predictions, stat_ensemble),
        "blended_ensemble": _compute_metrics(predictions, blended_ensemble_3way),
        "blended_baseline": _compute_metrics(predictions, blended_ensemble_no_disagree),
    }

    # Per-league breakdown of the sharp baseline for context (league-level
    # calibration isn't very illuminating for individual models because
    # they share a global state, but sharp is per-league).
    by_league: dict[str, dict] = {}
    for league in sorted({p.league for p in predictions}):
        league_rows = [p for p in predictions if p.league == league]
        for name, getter in [
            ("elo", lambda r: (r.elo_ph, r.elo_pd, r.elo_pa)),
            ("form", lambda r: (r.form_ph, r.form_pd, r.form_pa)),
            ("poisson", lambda r: (r.poisson_ph, r.poisson_pd, r.poisson_pa)),
            ("sharp", lambda r: (r.sharp_ph, r.sharp_pd, r.sharp_pa)),
            ("blended_ensemble", blended_ensemble_3way),
            ("blended_baseline", blended_ensemble_no_disagree),
        ]:
            m = _compute_metrics(league_rows, getter)
            by_league.setdefault(league, {})[name] = {
                "n": m.n,
                "brier_binary": m.brier_binary,
                "brier_3way": m.brier_3way,
                "accuracy": m.accuracy,
            }

    home_win_rate = sum(1 for p in predictions if p.result == "H") / max(len(predictions), 1)
    draw_rate = sum(1 for p in predictions if p.result == "D") / max(len(predictions), 1)

    # Sharp vs stat-ensemble disagreement — how often does blending help?
    n_disagree = 0
    stat_correct = 0
    sharp_correct = 0
    for r in predictions:
        stat = stat_ensemble(r)
        if stat[0] is None or r.sharp_ph is None:
            continue
        stat_pick = max({"H": stat[0], "D": stat[1], "A": stat[2]}.items(), key=lambda x: x[1])[0]
        sharp_pick = max({"H": r.sharp_ph, "D": r.sharp_pd, "A": r.sharp_pa}.items(), key=lambda x: x[1])[0]
        if stat_pick == sharp_pick:
            continue
        n_disagree += 1
        if stat_pick == r.result:
            stat_correct += 1
        if sharp_pick == r.result:
            sharp_correct += 1

    return SoccerWalkForwardReport(
        n_games=len(predictions),
        by_model=by_model,
        by_league=by_league,
        home_win_rate=home_win_rate,
        draw_rate=draw_rate,
        n_disagree=n_disagree,
        stat_correct_on_disagree=stat_correct,
        sharp_correct_on_disagree=sharp_correct,
    )
