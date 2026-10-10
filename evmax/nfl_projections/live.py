"""Project an upcoming NFL week with the walk-forward-validated game model.

Live inputs differ from the backtest in three ways, handled here:

* **Starting QB.** The backtest reads the first-dropback passer of the game
  itself. Live, the starter comes from (in order) an explicit override, the
  nflverse schedule's projected starter (``home_qb_id`` / ``away_qb_id``), or
  the team's most recent starter.
* **Wind.** Unplayed games have no recorded wind; outdoor games use the league
  median (a forecast feed is a follow-up — wind is where the totals gain lives).
* **Roof.** Retractable stadiums are listed with an empty roof until game day;
  they take the stadium's most common historical setting.

The combiner is trained on completed seasons before ``season`` and the ratings
on every game before the week's first kickoff — exactly the walk-forward setup.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from evmax.nfl_projections import data, player_games, td_model, team_games
from evmax.nfl_projections.game_model import (
    MARGIN_SD, TOTAL_SD, GameModelConfig, context_features, drive_points, feature_table, fit_combiner,
    fit_ratings, side_features,
)
from evmax.nfl_projections.player_model import (
    VOLUME_STATS, PlayerModelConfig, first_script_season, fit_player_state, game_script,
    injury_share_multipliers, out_players, project_players, recent_team_players,
    team_volume_rows, volume_combiners, volume_feature_table,
)
from evmax.nfl_projections.ratings import fit_qb_ratings


def next_week(games: pd.DataFrame, today: date) -> tuple[int, int]:
    """(season, week) of the earliest regular/post-season week with an unplayed game on or after today."""
    upcoming = games[games["home_score"].isna() & (games["gameday"].dt.date >= today)]
    if upcoming.empty:
        raise ValueError("no unplayed games on or after today in the schedule")
    first = upcoming.sort_values("gameday").iloc[0]
    return int(first["season"]), int(first["week"])


def infer_roof(games: pd.DataFrame, stadium_id: str, roof: str) -> str:
    """The game's roof, or the stadium's most common played roof when unknown ('')."""
    if roof:
        return roof
    hist = games[(games["stadium_id"] == stadium_id) & games["home_score"].notna() & (games["roof"] != "")]
    return hist["roof"].mode().iloc[0] if not hist.empty else "outdoors"


def latest_starters(tg: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, str]:
    """Each team's most recent pre-game starter (first-dropback passer) before ``cutoff``."""
    past = tg[(tg["gameday"] < cutoff) & tg["first_qb_id"].notna()].sort_values("gameday")
    return past.groupby("team")["first_qb_id"].last().to_dict()


def project_week(season: int, week: int, cfg: GameModelConfig = GameModelConfig(),
                 d: Optional[Path] = None, refresh: bool = True,
                 starters: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Projections for every game of ``season`` week ``week``.

    ``starters`` maps team abbreviation -> gsis QB id and overrides the
    schedule's projected starter. Returns one row per game, home perspective:
    proj_home / proj_away points, proj_margin (home - away), proj_total,
    p_home_win, the QB ids used, and the schedule's consensus lines (for
    comparison only — never a model input).
    """
    first_season = cfg.first_feature_season - 2  # ratings need two prior seasons
    if refresh:
        data.ensure_games(d)
        data.ensure_pbp(range(first_season, season + 1), d, refresh_seasons=[season])
    tg = team_games.load_team_games(range(first_season, season + 1), d)
    games = data.load_games(d)
    wk = games[(games["season"] == season) & (games["week"] == week)].copy()
    if wk.empty:
        raise ValueError(f"no games for season {season} week {week}")
    cutoff = wk["gameday"].min()

    comb = fit_combiner(feature_table(tg, games, list(range(cfg.first_feature_season, season)), cfg),
                        cfg.features)
    fits = fit_ratings(tg, cutoff, cfg)
    qbr = fit_qb_ratings(tg, cutoff)
    dpts = drive_points(tg, games, cutoff, wk) if "drive" in cfg.features else {}
    recent = latest_starters(tg, cutoff)
    overrides = starters or {}

    rows = []
    for g in wk.itertuples():
        g = g._replace(roof=infer_roof(games, g.stadium_id, g.roof or ""))
        ctx = context_features(g)
        neutral = g.location == "Neutral"
        sides = {}
        for team, opp, home, qb_col in ((g.home_team, g.away_team, 0 if neutral else 1, "home_qb_id"),
                                         (g.away_team, g.home_team, 0, "away_qb_id")):
            sched_qb = getattr(g, qb_col)
            qb = overrides.get(team) or (sched_qb if isinstance(sched_qb, str) and sched_qb else None) \
                or recent.get(team)
            feats = {**side_features(fits, team, opp, home), **ctx, "qb": qbr.delta(team, qb)}
            if dpts:
                feats["drive"] = dpts.get((g.game_id, team))
            sides[team] = (comb.predict(feats), qb)
        (ph, hq), (pa, aq) = sides[g.home_team], sides[g.away_team]
        rows.append({
            "game_id": g.game_id, "season": season, "week": week, "gameday": g.gameday.date(), "gametime": g.gametime,
            "home_team": g.home_team, "away_team": g.away_team, "neutral": bool(neutral),
            "roof": g.roof, "proj_home": ph, "proj_away": pa, "proj_margin": ph - pa,
            "proj_total": ph + pa, "p_home_win": float(stats.norm.cdf((ph - pa) / MARGIN_SD)),
            "home_qb_id": hq, "away_qb_id": aq,
            "home_qb_name": g.home_qb_name, "away_qb_name": g.away_qb_name,
            "market_spread_line": g.spread_line, "market_total_line": g.total_line,
        })
    return pd.DataFrame(rows)


def total_over_probability(proj_total: float, line: float) -> float:
    """P(total > line) under the model's total distribution (Normal, SD from walk-forward residuals)."""
    return float(stats.norm.sf(line, loc=proj_total, scale=TOTAL_SD))


def home_cover_probability(proj_margin: float, home_handicap: float) -> Optional[float]:
    """P(home margin + handicap > 0), e.g. handicap -3.5 = home laying 3.5.

    Uses the key-number margin PMF (evmax.models_ml.spread_pmf) located at the
    projected margin; None if the artifact is unavailable. Pushes (integer
    handicaps) are excluded from both sides: P(cover | no push).
    """
    from evmax.models_ml.spread_pmf import load_margin_pmf

    pmf = load_margin_pmf("nfl")
    if pmf is None:
        return None
    fav_home = proj_margin >= 0
    p = pmf.pmf(abs(proj_margin))  # favorite-margin distribution over pmf.ks
    home_margin = pmf.ks if fav_home else -pmf.ks
    net = home_margin + home_handicap
    win, lose = float(p[net > 0].sum()), float(p[net < 0].sum())
    return win / (win + lose) if (win + lose) > 0 else None


# ── players ──────────────────────────────────────────────────────────────────

def active_roster(pg: pd.DataFrame, injuries: pd.DataFrame, week_games: pd.DataFrame,
                  cutoff: pd.Timestamp, week: int,
                  extra_out: Optional[set[tuple[str, str]]] = None) -> pd.DataFrame:
    """Expected active skill players for each team of ``week_games``.

    A player is expected active if he played for the team in any of its last
    ``player_model.RECENT_GAMES`` games and is not listed Out/Doubtful on that
    week's injury report or in ``extra_out`` (e.g. the live ESPN feed). (A player
    returning after missing those games is not included — a known limitation; the
    starting QB is always added from the schedule.)
    """
    teams = set(week_games["home_team"]) | set(week_games["away_team"])
    recent = recent_team_players(pg, teams, cutoff)
    out = out_players(injuries, week) | (extra_out or set())
    recent = recent[[(t, pid) not in out for t, pid in zip(recent["team"], recent["player_id"])]]
    names = pg.drop_duplicates("player_id", keep="last").set_index("player_id")["player_display_name"]
    rows = []
    for g in week_games.itertuples():
        for team, opp, home, qb_col in ((g.home_team, g.away_team, 0 if g.location == "Neutral" else 1, "home_qb_id"),
                                        (g.away_team, g.home_team, 0, "away_qb_id")):
            players = recent[recent["team"] == team]
            qb = getattr(g, qb_col)
            qb = qb if isinstance(qb, str) and qb else None
            for p in players.itertuples():
                rows.append({"game_id": g.game_id, "team": team, "opp": opp, "home": home,
                             "player_id": p.player_id, "player_display_name": p.player_display_name,
                             "position": p.position, "is_starting_qb": p.player_id == qb})
            if qb and qb not in set(players["player_id"]):
                rows.append({"game_id": g.game_id, "team": team, "opp": opp, "home": home, "player_id": qb,
                             "player_display_name": names.get(qb, qb), "position": "QB", "is_starting_qb": True})
    return pd.DataFrame(rows)


# Live ESPN injury feed (the same InjuryReportAgent the scanner uses). nflverse
# injury reports are the backtested source but publish with a lag; on game day
# the ESPN feed carries late downgrades. Same out rule as the coordinator's
# depth-chart QB starters (AgentCoordinator._QB_OUT_STATUSES).
ESPN_OUT_STATUSES = frozenset({
    "OUT", "INJURED RESERVE", "IR", "SUSPENSION", "SUSPENDED", "DOUBTFUL",
    "PHYSICALLY UNABLE TO PERFORM", "PUP", "NON-FOOTBALL INJURY",
})


def fetch_espn_injury_reports() -> dict:
    """ESPN NFL injury reports (team full name -> InjuryReport); {} on any failure."""
    import asyncio

    try:
        from evmax.agents.base import AgentRequest
        from evmax.agents.intelligence.injury_agent import InjuryReportAgent

        resp = asyncio.run(InjuryReportAgent().run(AgentRequest(sector="nfl", correlation_id="nfl-projections")))
        return resp.data or {}
    except Exception:  # noqa: BLE001 — the live feed is optional; nflverse reports remain
        return {}


def espn_out_players(reports: dict, recent: pd.DataFrame) -> set[tuple[str, str]]:
    """(team, gsis id) of ``recent`` players the ESPN feed lists with an out status.

    Players are matched by normalized name within their team (ESPN carries no
    gsis id); an ambiguous name (two teammates) matches nobody.
    """
    from evmax.agents.models.nfl_efficiency_agent import NFL_ABBREV_TO_NAME
    from evmax.clients.nfl_depth_charts import normalize_person

    abbr_of = {v: k for k, v in NFL_ABBREV_TO_NAME.items()}
    by_team: dict[str, dict[str, list[str]]] = {}
    for r in recent.itertuples(index=False):
        by_team.setdefault(r.team, {}).setdefault(normalize_person(r.player_display_name), []).append(r.player_id)
    out: set[tuple[str, str]] = set()
    for team, report in (reports or {}).items():
        abbr = abbr_of.get(str(team).lower().strip())
        names = by_team.get(abbr or "", {})
        for p in getattr(report, "players", None) or []:
            if (getattr(p, "status", "") or "").upper().strip() not in ESPN_OUT_STATUSES:
                continue
            ids = names.get(normalize_person(p.name), [])
            if len(ids) == 1:
                out.add((abbr, ids[0]))
    return out


def project_week_players(season: int, week: int, cfg: PlayerModelConfig = PlayerModelConfig(),
                         d: Optional[Path] = None, refresh: bool = True,
                         espn_reports: Optional[dict] = None) -> pd.DataFrame:
    """Player stat-line projections (median, mean, 10th/90th percentile) for a week.

    Team volume is conditioned on this week's GAME projections (``project_week``)
    through the script model trained on completed seasons; usage and efficiency
    use every game before the week's first kickoff. Ruled-out players come from
    the nflverse injury report plus ``espn_reports`` (``fetch_espn_injury_reports()``),
    when given.
    """
    gcfg = GameModelConfig()
    first_season = gcfg.first_feature_season - 2
    if refresh:
        data.ensure_player_sources(range(first_season, season + 1), d, refresh_seasons=[season])
        data.ensure_injuries(season, d)
    game_proj = project_week(season, week, gcfg, d, refresh=refresh)
    tg = team_games.load_team_games(range(first_season, season + 1), d)
    pg = player_games.load_player_games(range(first_season, season + 1), d)
    games = data.load_games(d)
    wk = games[(games["season"] == season) & (games["week"] == week)]
    cutoff = wk["gameday"].min()

    vol = team_volume_rows(pg, tg)
    hist_seasons = list(range(first_script_season(tg), season))
    vft = volume_feature_table(vol, game_script(tg, games, hist_seasons), games, hist_seasons, cfg)
    coefs = volume_combiners(vft)
    rz = td_model.load_rz_usage(range(first_season, season + 1), d)
    state = fit_player_state(pg, vol, cutoff, cfg, rz=rz)

    injuries = data.load_injuries(season, d)
    recent = recent_team_players(pg, set(wk["home_team"]) | set(wk["away_team"]), cutoff)
    espn_out = espn_out_players(espn_reports, recent) if espn_reports else set()
    roster = active_roster(pg, injuries, wk, cutoff, week, extra_out=espn_out)
    # Teammates of players ruled Out/Doubtful take part of their usage (same rule as the backtest).
    mult = injury_share_multipliers(state.usage, recent, out_players(injuries, week) | espn_out,
                                    cfg.injury_redistribution)
    script = {}
    for r in game_proj.itertuples():
        script[(r.game_id, r.home_team)] = (r.proj_margin, r.proj_total)
        script[(r.game_id, r.away_team)] = (-r.proj_margin, r.proj_total)
    tv = []
    for (gid, team), grp in roster.groupby(["game_id", "team"]):
        opp, home = grp["opp"].iloc[0], int(grp["home"].iloc[0])
        margin, total = script[(gid, team)]
        row = {"game_id": gid, "team": team}
        for m in VOLUME_STATS:
            if m not in coefs:
                continue
            c = coefs[m]
            row[m] = c[0] + c[1] * state.fits[m].expect(team, opp, home) + c[2] * margin + c[3] * total
        tv.append(row)
    team_volume = pd.DataFrame(tv).set_index(["game_id", "team"])
    proj = project_players(state, roster, team_volume, mult)
    return proj.merge(wk[["game_id", "gameday", "gametime", "home_team", "away_team"]], on="game_id").assign(
        season=season, week=week)


def simulate_game(season: int, week: int, team: str, n: int = 10000, d: Optional[Path] = None,
                  refresh: bool = True, espn_reports: Optional[dict] = None, seed: int = 0):
    """Joint simulation of ``team``'s game in ``season`` ``week``.

    Returns {team: (players DataFrame, sims dict)} for both teams; the shape
    parameters are fitted on the six completed seasons before ``season``.
    """
    from evmax.nfl_projections import simulate

    proj = project_week_players(season, week, d=d, refresh=refresh, espn_reports=espn_reports)
    game = proj[proj["team"] == team.upper()]
    if game.empty:
        raise ValueError(f"{team.upper()} has no projected game in {season} week {week}")
    gid = game["game_id"].iloc[0]
    pg = player_games.load_player_games(range(season - 6, season), d)
    params = simulate.fit_sim_params(pg[pg["season_type"] == "REG"])
    rng = np.random.default_rng(seed)
    out = {}
    for t, grp in proj[proj["game_id"] == gid].groupby("team"):
        grp = grp.reset_index(drop=True)
        f = grp.iloc[0]
        out[t] = (grp, simulate.simulate_team(grp, f["exp_team_targets"], f["exp_team_carries"],
                                              f["exp_team_rush_tds"], f["exp_team_rec_tds"], params, n=n, rng=rng))
    return out
