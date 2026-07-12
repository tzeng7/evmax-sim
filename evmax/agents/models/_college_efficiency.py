"""Shared math core for the NCAAB / NCAAW opponent-adjusted efficiency stack.

College basketball has ~360 D1 teams playing wildly unequal schedules, so raw
Dean Oliver ORTG/DRTG (the WNBA approach) is meaningless across conferences: a
mid-major can post a 118 raw ORTG against bottom-100 defenses while a Big Ten
team grinds out 108 against top-25 defenses. The KenPom-style fix is an
ITERATIVE OPPONENT ADJUSTMENT: rate each team's per-game efficiency relative
to the (adjusted) strength of the opponent it was produced against, and
iterate to convergence. That solver lives here, shared by the `ncaab` and
`ncaaw` stacks (same data source, same formulas — only the constants and
state files diverge per league; see the per-league agent modules).

This module deliberately imports NOTHING from the NBA or WNBA stacks — the
college stack is its own parallel stack per the repo's parallel-stack rule.

Consumers:
  - scripts/seed_ncaab_efficiency.py       (builds the state files)
  - scripts/backtest_ncaab_efficiency.py   (walk-forward validation)
  - evmax/agents/models/ncaab_efficiency_agent.py / ncaaw_efficiency_agent.py
  - evmax/agents/models/ncaab_possession_sim_agent.py / ncaaw variants
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Season calendar
# ---------------------------------------------------------------------------

# Months during which a college basketball season is live. The season wraps
# the year boundary (Nov -> early Apr, incl. the NCAA tournament).
COLLEGE_SEASON_MONTHS = {11, 12, 1, 2, 3, 4}

# Non-D1 opponents (exhibitions, buy games vs D2/D3/NAIA) appear on the D1
# scoreboard a handful of times per season. Real D1 teams play 28-35 games.
# Requiring this many appearances in the season's scoreboard universe filters
# the non-D1 rows out — the college analogue of the WNBA seed's
# REAL_WNBA_TEAMS allow-list (which can't work here: 360+ teams, yearly
# reclassifications).
MIN_D1_APPEARANCES = 8


def college_season_start_year(d: date) -> int:
    """Season label = calendar year the season STARTS in.

    Nov 2025 – Apr 2026 is season 2025. Aug+ counts toward the upcoming
    season so a fresh autumn seed labels itself correctly.
    """
    return d.year if d.month >= 8 else d.year - 1


def state_is_stale_for_today(state: dict, today: Optional[date] = None) -> bool:
    """True when the state's source_season is older than the current college season.

    Mirrors the WNBA staleness guard (the +24pp chalk-bias lesson): during the
    Nov–Apr season window, prior-season ratings must NOT fire — college roster
    turnover (transfers + draft + graduation) is even more extreme than the
    WNBA offseason. In November this silences the agent until the current
    season has been re-seeded with >= MIN_GAMES per team, which is the correct
    behavior: elo/form/sharp carry the blend until then.

    Outside the season window there are no games to predict, so prior-season
    state is treated as fresh (guard relaxed). Missing / unparseable
    source_season disables the guard for back-compat.
    """
    today = today or date.today()
    if today.month not in COLLEGE_SEASON_MONTHS:
        return False
    raw = state.get("source_season")
    if raw is None:
        return False
    try:
        source_year = int(raw)
    except (TypeError, ValueError):
        return False
    return source_year < college_season_start_year(today)


# ---------------------------------------------------------------------------
# Dean Oliver possessions
# ---------------------------------------------------------------------------


def dean_oliver_possessions(fga: float, fta: float, tov: float, oreb: float) -> float:
    """POSS = FGA + 0.44·FTA + TO − OREB (Dean Oliver)."""
    return fga + 0.44 * fta + tov - oreb


# ---------------------------------------------------------------------------
# D1 universe filter
# ---------------------------------------------------------------------------


def filter_d1_games(
    games: list[dict],
    min_appearances: int = MIN_D1_APPEARANCES,
) -> tuple[list[dict], set[str]]:
    """Keep only games where BOTH teams clear the appearance threshold.

    `games` rows must carry `home_id` / `away_id`. Returns (filtered_games,
    d1_team_ids). Appearances are counted over the given slice, so a
    partial-season slice early in a walk-forward uses a proportionally lower
    effective bar — pass a smaller `min_appearances` for early-season slices
    (the backtest scales it by elapsed fraction).
    """
    counts: dict[str, int] = {}
    for g in games:
        counts[g["home_id"]] = counts.get(g["home_id"], 0) + 1
        counts[g["away_id"]] = counts.get(g["away_id"], 0) + 1
    d1_ids = {tid for tid, n in counts.items() if n >= min_appearances}
    kept = [g for g in games if g["home_id"] in d1_ids and g["away_id"] in d1_ids]
    return kept, d1_ids


# ---------------------------------------------------------------------------
# Iterative opponent adjustment (KenPom-style core)
# ---------------------------------------------------------------------------


def opponent_adjust(
    games: list[dict],
    max_iters: int = 1000,
    tol: float = 1e-4,
    estimate_hca: bool = True,
    ridge: float = 0.25,
) -> dict:
    """Solve opponent-adjusted offensive/defensive efficiencies.

    Args:
        games: rows with keys home_id, away_id, neutral (bool), home_pts,
            away_pts, home_poss, away_poss. Rows with non-positive possessions
            are skipped.
        max_iters: iteration cap.
        tol: convergence threshold on the max absolute rating change
            (efficiency points).
        estimate_hca: jointly estimate the home-court advantage (in efficiency
            points per 100 possessions) from non-neutral games. When False,
            hca_eff stays 0 (useful for synthetic tests).
        ridge: virtual league-average games added per team-side. Ridge toward
            the league mean makes the least-squares optimum UNIQUE even when
            the offense-defense incidence graph is disconnected (pathological
            schedules), and mildly stabilizes 2-3-game early-season ratings.
            Kept SMALL (0.25) so extreme-schedule teams aren't biased — the
            heavy small-sample regularization lives in shrink_team_stats at
            predict time, not here.

    Returns:
        {
          "teams": {team_id: {"adj_o", "adj_d", "raw_o", "raw_d", "pace", "gp"}},
          "league_avg_eff": float,     # mean per-game offensive efficiency
          "league_avg_pace": float,    # mean per-game possessions (one side)
          "hca_eff": float,            # home edge, efficiency pts / 100 poss
          "iterations": int,
          "converged": bool,
        }

    Model (multiplicative, solved as ridge least squares in LOG space):
        log(OE_g,t) = log(L) + α_t + β_opp + h·side/2
    where α_t = log(adj_o_t / L), β_u = log(adj_d_u / L), L = league mean
    efficiency, and side = +1 for home offense, −1 for away offense, 0 on
    neutral courts. Solved by exact block-coordinate descent (α given β,h is
    a per-team ridge mean; likewise β; likewise h), which converges
    monotonically to the UNIQUE ridge-least-squares optimum.

    The naive multiplicative-averaging iteration (adj_o = mean of
    opponent-rescaled OEs, KenPom folklore) has NON-unique fixed points on
    structured schedules — a team playing only strong opponents could land on
    an inflated self-consistent solution. The log-LS formulation is the same
    model without that failure mode.

    Efficiency is score-based, not win-based, so unbeaten and winless teams
    keep finite, sane ratings.
    """
    # Build per-side observations: one row per (game, offense side).
    team_ids: list[str] = []
    team_index: dict[str, int] = {}

    def _idx(tid: str) -> int:
        if tid not in team_index:
            team_index[tid] = len(team_ids)
            team_ids.append(tid)
        return team_index[tid]

    obs_team: list[int] = []
    obs_opp: list[int] = []
    obs_oe: list[float] = []
    obs_side: list[float] = []   # +1 home offense, -1 away offense, 0 neutral
    obs_poss: list[float] = []

    for g in games:
        h_poss = float(g["home_poss"])
        a_poss = float(g["away_poss"])
        if h_poss <= 0 or a_poss <= 0:
            continue
        h, a = _idx(g["home_id"]), _idx(g["away_id"])
        neutral = bool(g.get("neutral", False))
        oe_h = 100.0 * float(g["home_pts"]) / h_poss
        oe_a = 100.0 * float(g["away_pts"]) / a_poss
        obs_team.extend([h, a])
        obs_opp.extend([a, h])
        obs_oe.extend([oe_h, oe_a])
        obs_side.extend([0.0, 0.0] if neutral else [1.0, -1.0])
        obs_poss.extend([h_poss, a_poss])

    n_teams = len(team_ids)
    if n_teams == 0 or not obs_team:
        return {
            "teams": {}, "league_avg_eff": 0.0, "league_avg_pace": 0.0,
            "hca_eff": 0.0, "iterations": 0, "converged": True,
        }

    t_arr = np.asarray(obs_team, dtype=np.int64)
    o_arr = np.asarray(obs_opp, dtype=np.int64)
    oe_arr = np.asarray(obs_oe, dtype=np.float64)
    side_arr = np.asarray(obs_side, dtype=np.float64)
    poss_arr = np.asarray(obs_poss, dtype=np.float64)

    league_avg = float(oe_arr.mean())

    counts = np.bincount(t_arr, minlength=n_teams).astype(np.float64)
    counts[counts == 0] = 1.0

    def _group_mean(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
        sums = np.zeros(n_teams)
        np.add.at(sums, idx, values)
        return sums / counts

    # Raw (unadjusted) reference stats.
    raw_o = _group_mean(oe_arr, t_arr)
    raw_d = _group_mean(oe_arr, o_arr)   # opponent's OE = own DE
    pace = _group_mean(poss_arr, t_arr)

    # Log-space response. A sub-5 efficiency observation is data corruption
    # (a real college team scores >30 per 100); clamp so log() stays finite.
    y = np.log(np.maximum(oe_arr, 5.0) / league_avg)

    alpha = np.zeros(n_teams)   # log(adj_o / L)
    beta = np.zeros(n_teams)    # log(adj_d / L)
    h = 0.0                     # log-scale home edge (full home-away split)
    home_mask = side_arr > 0
    away_mask = side_arr < 0
    can_hca = estimate_hca and bool(home_mask.any()) and bool(away_mask.any())

    def _ridge_group_mean(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Per-team ridge mean: `ridge` virtual observations at the prior (0)."""
        sums = np.zeros(n_teams)
        np.add.at(sums, idx, values)
        return sums / (counts + ridge)

    converged = False
    iterations = 0
    for iterations in range(1, max_iters + 1):
        # α block: per-team exact ridge minimizer given (β, h).
        alpha_new = _ridge_group_mean(y - beta[o_arr] - h * side_arr / 2.0, t_arr)
        # β block: per-team exact ridge minimizer given (α, h).
        beta_new = _ridge_group_mean(y - alpha_new[t_arr] - h * side_arr / 2.0, o_arr)

        h_new = h
        if can_hca:
            resid = y - alpha_new[t_arr] - beta_new[o_arr]
            h_new = float(resid[home_mask].mean() - resid[away_mask].mean())

        delta = league_avg * max(
            float(np.max(np.abs(alpha_new - alpha))),
            float(np.max(np.abs(beta_new - beta))),
            abs(h_new - h) / 2.0,
        )
        alpha, beta, h = alpha_new, beta_new, h_new
        if delta < tol:
            converged = True
            break

    adj_o = league_avg * np.exp(alpha)
    adj_d = league_avg * np.exp(beta)
    hca_eff = league_avg * (math.exp(h / 2.0) - math.exp(-h / 2.0))

    teams = {
        team_ids[i]: {
            "adj_o": round(float(adj_o[i]), 3),
            "adj_d": round(float(adj_d[i]), 3),
            "raw_o": round(float(raw_o[i]), 3),
            "raw_d": round(float(raw_d[i]), 3),
            "pace": round(float(pace[i]), 3),
            "gp": int(round(counts[i])),
        }
        for i in range(n_teams)
    }
    return {
        "teams": teams,
        "league_avg_eff": round(league_avg, 3),
        "league_avg_pace": round(float(poss_arr.mean()), 3),
        "hca_eff": round(float(hca_eff), 3),
        "iterations": iterations,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage + confidence (college-tuned)
# ---------------------------------------------------------------------------


def shrink_team_stats(stats: dict, league: dict, k: float) -> dict:
    """Empirical-Bayes shrinkage of team ORTG/DRTG/pace toward league means.

    Same formula as the WNBA stack — shrunk = (gp·raw + k·avg) / (gp + k) —
    but `k` is a required argument because college needs HEAVIER early-season
    shrinkage (roster turnover is near-total year over year and November
    schedules are cupcake-loaded). Returns a new dict; keys not in `league`
    pass through unchanged.
    """
    gp = stats.get("gp", 0)
    if gp <= 0:
        return dict(stats)
    weight = gp / (gp + k)
    prior = 1.0 - weight

    def _shrunk(val: float, prior_val: float) -> float:
        return weight * val + prior * prior_val

    out = dict(stats)
    out["ortg"] = _shrunk(stats.get("ortg", league["league_avg_ortg"]), league["league_avg_ortg"])
    out["drtg"] = _shrunk(stats.get("drtg", league["league_avg_drtg"]), league["league_avg_drtg"])
    out["pace"] = _shrunk(stats.get("pace", league["league_avg_pace"]), league["league_avg_pace"])
    if "tov_pct" in stats and "league_avg_tov_pct" in league:
        out["tov_pct"] = _shrunk(stats["tov_pct"], league["league_avg_tov_pct"])
    return out


def smooth_confidence(min_gp: int, full_gp: int = 30) -> float:
    """Ramp confidence 0.40 (gp=0) → 0.80 (gp >= full_gp).

    College regular seasons run ~30 games, so the ramp completes within one
    season. Clears the ensemble's 0.45 gate at gp >= ceil(full_gp/8) ≈ 4.
    """
    return 0.40 + 0.40 * min(min_gp / float(full_gp), 1.0)


# ---------------------------------------------------------------------------
# Team-name resolution (college-safe)
# ---------------------------------------------------------------------------


def resolve_team(teams: dict, name: str, normalize=None) -> Optional[dict]:
    """Resolve a Pinnacle/Kalshi team label to a stats dict.

    College-safe: NO bare last-word fallback (the WNBA trick) — 360+ teams
    share mascots ("Wildcats", "Tigers", "Bulldogs"). Resolution order:
      1. exact key (state keys are normalized ESPN `location` strings,
         e.g. "duke", "michigan state")
      2. `normalize(name)` exact key (caller passes NameNormalizer.normalize)
      3. full-name / prefix match, longest key wins; ambiguity → None
         (skipping a game beats mis-rating one — "north carolina" must not
         swallow "north carolina central").
    """
    n = (name or "").lower().strip()
    if not n:
        return None
    if n in teams:
        return teams[n]
    if normalize is not None:
        try:
            normed = normalize(n)
        except Exception:
            normed = None
        if normed and normed in teams:
            return teams[normed]
        if normed:
            n = normed
            if n in teams:
                return teams[n]

    candidates: list[tuple[int, str]] = []
    for key, val in teams.items():
        full = val.get("full_name", "")
        if n == full:
            return val
        if full.startswith(n + " ") or n.startswith(key + " "):
            candidates.append((len(key), key))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None  # ambiguous — refuse to guess
    return teams[candidates[0][1]]


# ---------------------------------------------------------------------------
# Analytic win probability (shared by both college efficiency agents)
# ---------------------------------------------------------------------------

from evmax.models_ml._math import normal_cdf as _normal_cdf


def project_matchup(
    stats_home: dict,
    stats_away: dict,
    league: dict,
    hca_eff: float,
    score_stdev: float,
    shrink_k: float,
) -> dict:
    """Project margin + win prob for home vs away from (shrunk) adjusted stats.

    stats dicts carry ortg/drtg/pace/gp (ortg = adj_o etc. mapped by the
    agent). Returns {"prob_home", "margin", "pace", "stats_home", "stats_away"}
    with the SHRUNK stats echoed back for notes/possession-sim reuse.
    """
    sh = shrink_team_stats(stats_home, league, k=shrink_k)
    sa = shrink_team_stats(stats_away, league, k=shrink_k)

    league_ortg = league["league_avg_ortg"]
    league_drtg = league["league_avg_drtg"]

    off_h = sh["ortg"] / league_ortg
    off_a = sa["ortg"] / league_ortg
    def_h = sh["drtg"] / league_drtg
    def_a = sa["drtg"] / league_drtg

    pace = (sh["pace"] + sa["pace"]) / 2.0
    proj_h = off_h * def_a * league_ortg * pace / 100.0
    proj_a = off_a * def_h * league_ortg * pace / 100.0

    margin = proj_h - proj_a + hca_eff * pace / 100.0
    prob_home = _normal_cdf(margin / score_stdev)
    prob_home = max(0.02, min(0.98, prob_home))
    return {
        "prob_home": prob_home,
        "margin": margin,
        "pace": pace,
        "stats_home": sh,
        "stats_away": sa,
    }


# ---------------------------------------------------------------------------
# Vectorized possession-level Monte Carlo (shared by both college sim agents)
# ---------------------------------------------------------------------------


def simulate_matchup_vectorized(
    stats_home: dict,
    stats_away: dict,
    league: dict,
    hca_eff: float,
    n_sims: int,
    rng: np.random.Generator,
    poss_clip: tuple[float, float],
    pts_std: float = 1.10,
    avg_tov_rate: float = 0.17,
) -> dict:
    """Monte Carlo possession sim, fully vectorized (no per-sim Python loop).

    Distributionally equivalent to the WNBA sim's per-possession loop: with
    n_tov ~ Binomial(n_poss, tov) turnovers, the sum of per-possession
    Normal(ppp/(1−tov), pts_std) draws over the scoring possessions is
    Normal(scoring_poss · ppp/(1−tov), pts_std·sqrt(scoring_poss)) — sampled
    directly here so 10k sims cost microseconds instead of ~0.3s.

    Caller passes ALREADY-SHRUNK stats (share the same shrinkage as the
    analytic agent so both consumers see identical regularized inputs).
    """
    league_ortg = league["league_avg_ortg"]

    pace = (stats_home["pace"] + stats_away["pace"]) / 2.0
    poss = rng.normal(pace, 3.0, size=n_sims)
    poss = np.clip(np.round(poss), poss_clip[0], poss_clip[1])

    tov_h = stats_home.get("tov_pct", avg_tov_rate)
    tov_a = stats_away.get("tov_pct", avg_tov_rate)

    off_h = (stats_home["ortg"] + hca_eff) / league_ortg
    off_a = stats_away["ortg"] / league_ortg
    def_h = stats_home["drtg"] / league_ortg
    def_a = stats_away["drtg"] / league_ortg

    ppp_h = off_h * def_a * league_ortg / 100.0
    ppp_a = off_a * def_h * league_ortg / 100.0

    def _side_scores(ppp: float, tov: float) -> np.ndarray:
        tov = min(max(tov, 0.0), 0.5)
        n_tov = rng.binomial(poss.astype(np.int64), tov)
        scoring = np.maximum(poss - n_tov, 0.0)
        mean = scoring * (ppp / (1.0 - tov))
        std = pts_std * np.sqrt(np.maximum(scoring, 1.0))
        return rng.normal(mean, std)

    scores_h = _side_scores(ppp_h, tov_h)
    scores_a = _side_scores(ppp_a, tov_a)

    wins_h = (scores_h > scores_a).sum()
    ties = (scores_h == scores_a).sum()
    prob_home = float((wins_h + 0.5 * ties) / n_sims)
    prob_home = max(0.02, min(0.98, prob_home))

    return {
        "prob_home": prob_home,
        "margin": scores_h - scores_a,
        "total": scores_h + scores_a,
        "scores_home": scores_h,
        "scores_away": scores_a,
        "pace": pace,
    }
