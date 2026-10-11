"""Project an upcoming NFL week with the walk-forward-validated game model.

Live inputs differ from the backtest in three ways, handled here:

* **Starting QB.** The backtest reads the first-dropback passer of the game
  itself. Live, the starter comes from (in order) an explicit override, the
  nflverse depth chart's highest QB not ruled out (injury report, weekly roster,
  ESPN feed), the team's most recent starter if not ruled out, or the nflverse
  schedule's projected starter (``home_qb_id`` / ``away_qb_id``). The schedule
  comes last because it goes stale: in 2026 it kept Drew Lock as Seattle's
  starter for Weeks 3-5 while Sam Darnold started (see ``pick_starter``).
* **Wind.** Unplayed games have no recorded wind; outdoor games use the league
  median (a forecast feed is a follow-up — wind is where the totals gain lives).
* **Roof.** Retractable stadiums are listed with an empty roof until game day;
  they take the stadium's most common historical setting.

The combiner is trained on completed seasons before ``season`` and the ratings
on every game before the week's first kickoff — exactly the walk-forward setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from evmax.nfl_projections import data, player_games, td_model, team_games
from evmax.nfl_projections.game_model import (
    MARGIN_SD, TOTAL_SD, GameModelConfig, context_features, drive_points, feature_table, fit_combiner,
    fit_ratings, side_features,
)
from evmax.nfl_projections.player_model import (
    VOLUME_STATS, PlayerModelConfig, first_script_season, fit_player_state, game_script,
    injury_share_multipliers, out_players, project_players, recent_team_players, roster_unavailable,
    team_volume_rows, volume_combiners, volume_feature_table,
)
from evmax.nfl_projections.ratings import fit_qb_ratings

logger = structlog.get_logger(__name__)


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


def ensure_week_status(season: int, d: Optional[Path] = None) -> list[str]:
    """Refresh the season's injury reports and weekly rosters; notes for whatever could not be fetched."""
    notes = []
    for name, fn in (("injury report", data.ensure_injuries), ("weekly roster", data.ensure_rosters)):
        try:
            ok = fn(season, d)
        except Exception as e:  # noqa: BLE001 — a stale cache or none at all is handled downstream
            logger.warning("nfl_proj_status_fetch_failed", source=name, error=str(e))
            ok = False
        if not ok:
            notes.append(f"nflverse {name} for {season} unavailable")
    return notes


def _str_id(v) -> Optional[str]:
    return v if isinstance(v, str) and v else None


def qb_name_index(rosters: pd.DataFrame, tg: pd.DataFrame, season: int, week: int,
                  cutoff: pd.Timestamp) -> tuple[dict[str, dict[str, Optional[str]]], dict[str, str]]:
    """QB name lookups: (team -> {normalized name: gsis id, None when ambiguous}, gsis id -> display name).

    Names come from the team's weekly roster (latest week <= ``week``) and from the
    pbp passer names (``S.Darnold``) of its past starters, so a depth-chart full
    name resolves by either spelling.
    """
    from evmax.clients.nfl_depth_charts import normalize_person, pbp_passer_name

    by_team: dict[str, dict[str, Optional[str]]] = {}
    names: dict[str, str] = {}

    def add(team: str, key: str, pid: str) -> None:
        if not key:
            return
        t = by_team.setdefault(team, {})
        t[key] = pid if t.get(key, pid) == pid else None

    r = rosters[(rosters["season"] == season) & (rosters["week"] <= week) & (rosters["position"] == "QB")] \
        if rosters is not None and not rosters.empty else pd.DataFrame(columns=["team", "week", "gsis_id", "full_name"])
    r = r.dropna(subset=["gsis_id", "full_name"])
    if not r.empty:
        r = r[r["week"].to_numpy() == r.groupby("team")["week"].transform("max").to_numpy()]
    for x in r.itertuples(index=False):
        add(x.team, normalize_person(x.full_name), x.gsis_id)
        add(x.team, normalize_person(pbp_passer_name(x.full_name)), x.gsis_id)
        names[x.gsis_id] = x.full_name
    past = tg[(tg["gameday"] < cutoff) & tg["first_qb_id"].notna()]
    if "first_qb_name" in past:
        for x in past[["team", "first_qb_id", "first_qb_name"]].drop_duplicates().itertuples(index=False):
            if isinstance(x.first_qb_name, str):
                add(x.team, normalize_person(x.first_qb_name), x.first_qb_id)
                names.setdefault(x.first_qb_id, x.first_qb_name)
    return by_team, names


@dataclass
class StarterInputs:
    """Pre-game QB evidence for one team (``starter_inputs``); ``pick_starter`` decides."""
    depth: list[Optional[str]] = field(default_factory=list)  # depth-chart QBs as gsis ids, rank order (None = unknown name)
    ruled_out: set[str] = field(default_factory=set)          # Out/Doubtful, reserve list / released, ESPN out
    questionable: set[str] = field(default_factory=set)       # Questionable on the injury report or the ESPN feed
    last_start: Optional[str] = None                          # first-dropback passer of the team's last game
    schedule: Optional[str] = None                            # nflverse schedule's projected starter
    override: Optional[str] = None


def pick_starter(inp: StarterInputs) -> tuple[Optional[str], str]:
    """(gsis id, source) of one team's starting QB.

    Order: ``override`` > the depth chart's highest QB not ruled out > the last
    starter > the schedule's projected starter (each skipped when ruled out; if
    every candidate is, the schedule's or the last starter is kept). Exception:
    a depth-chart QB1 listed Questionable who did not start the team's last game
    yields to that game's starter when he is available (a QB coming back from
    injury stays first on the chart before he is cleared — CHI 2026 Week 5,
    Caleb Williams vs Tyson Bagent). An unknown depth-chart name stops the walk
    rather than promoting the QB below him.

    ``scripts/eval_nfl_starter_sources.py`` (team-games of Weeks 2+, pre-game
    information only — the depth chart before game day, no game-day inactive
    list): 98.4% (2025) and 99.0% (2026) vs 97.1% / 99.0% without the
    Questionable exception and 98.4% / 94.9% for the schedule-first order used
    before; on games where the starter changed, 86% / 90% vs the schedule's
    95% / 70%. The schedule's 2025 figure is likely inflated: completed seasons'
    schedule QB columns look backfilled with the actual starter. Rejected:
    yielding to the last starter whenever the schedule agrees with him (98.2% /
    96.9%).
    """
    if inp.override:
        return inp.override, "override"
    last_ok = bool(inp.last_start) and inp.last_start not in inp.ruled_out
    for pid in inp.depth:
        if pid is None:
            break
        if pid in inp.ruled_out:
            continue
        if pid in inp.questionable and pid != inp.last_start and last_ok:
            return inp.last_start, "last start (QB1 questionable)"
        return pid, "depth chart"
    for pid, src in ((inp.last_start, "last start"), (inp.schedule, "schedule")):
        if pid and pid not in inp.ruled_out:
            return pid, src
    if inp.schedule:
        return inp.schedule, "schedule"
    return (inp.last_start, "last start") if inp.last_start else (None, "none")


def starter_inputs(season: int, week: int, wk: pd.DataFrame, tg: pd.DataFrame, cutoff: pd.Timestamp,
                   injuries: Optional[pd.DataFrame], rosters: Optional[pd.DataFrame],
                   espn_reports: Optional[dict] = None, qb_depth: Optional[list] = None,
                   overrides: Optional[dict[str, str]] = None,
                   as_of: Optional[date] = None,
                   inactive: bool = True) -> tuple[dict[str, StarterInputs], dict[str, str]]:
    """({team: StarterInputs} for every team of ``wk``, {gsis id: display name}).

    ``qb_depth`` holds ``nfl_depth_charts`` QB rows (fetched when None; ``[]`` = no
    depth chart). The chart read is the latest snapshot before ``as_of`` (None =
    the latest); weekly-schema rows (seasons <= 2024) are ignored. ``inactive``:
    whether the week's game-day inactive list (``roster_unavailable``) rules a QB
    out — it is published ~90 minutes before kickoff, so a replay of a played
    week must pass False to see only what a pre-game run saw.
    """
    from evmax.agents.models.nfl_efficiency_agent import NFL_ABBREV_TO_NAME
    from evmax.clients import nfl_depth_charts as dc

    if qb_depth is None:
        qb_depth = dc.load_qb_chart_rows(season)
    # Only dated snapshots (2025+): the one weekly chart of earlier seasons names the
    # starter less often than the last start does (92% vs 95-96%, 2023-24).
    qb_depth = [r for r in qb_depth or [] if r.as_of is not None]
    depth = dc.qb_depth_as_of(qb_depth, as_of=as_of, week=week)
    by_team, names = qb_name_index(rosters, tg, season, week, cutoff)
    last = latest_starters(tg, cutoff)
    teams = list(wk["home_team"]) + list(wk["away_team"])
    cands = pd.DataFrame([(t, pid) for t in teams for pid in {p for p in by_team.get(t, {}).values() if p}
                          | {last.get(t)} - {None}], columns=["team", "player_id"])
    ruled = out_players(injuries, week, season) if injuries is not None else set()
    ruled |= roster_unavailable(rosters, cands, season, week, inactive=inactive)
    quest = questionable_players(injuries[injuries["season"] == season] if injuries is not None and "season" in injuries
                                 else injuries, week)
    espn = espn_statuses(espn_reports)
    sched = {}
    for g in wk.itertuples():
        for side in ("home", "away"):
            team, pid, name = (getattr(g, f"{side}_team"), _str_id(getattr(g, f"{side}_qb_id", None)),
                               getattr(g, f"{side}_qb_name", None))
            sched[team] = pid
            if pid and isinstance(name, str):
                names.setdefault(pid, name)
    out = {}
    for team in teams:
        lookup = by_team.get(team, {})
        ids = [lookup.get(dc.normalize_person(n)) or lookup.get(dc.normalize_person(dc.pbp_passer_name(n)))
               for n in depth.get(NFL_ABBREV_TO_NAME.get(team, ""), [])]
        espn_out = {lookup[n] for n, st in espn.get(team, {}).items() if st in ESPN_OUT_STATUSES and lookup.get(n)}
        espn_q = {lookup[n] for n, st in espn.get(team, {}).items() if st == "QUESTIONABLE" and lookup.get(n)}
        out[team] = StarterInputs(
            depth=ids, ruled_out={pid for t, pid in ruled if t == team} | espn_out,
            questionable={pid for t, pid in quest if t == team} | espn_q,
            last_start=last.get(team), schedule=sched.get(team), override=(overrides or {}).get(team))
    return out, names


def week_starters(season: int, week: int, wk: pd.DataFrame, tg: pd.DataFrame, cutoff: pd.Timestamp,
                  injuries: Optional[pd.DataFrame], rosters: Optional[pd.DataFrame],
                  espn_reports: Optional[dict] = None, qb_depth: Optional[list] = None,
                  overrides: Optional[dict[str, str]] = None) -> dict[str, tuple[Optional[str], str, Optional[str]]]:
    """team -> (QB gsis id, source, display name) for every team of ``wk`` (``pick_starter`` per team).

    An unplayed week reads the latest depth chart and any game-day inactive
    list; a played week the chart before its first kickoff and no inactives.
    """
    played = bool(wk["home_score"].notna().all()) if "home_score" in wk else False
    inputs, names = starter_inputs(season, week, wk, tg, cutoff, injuries, rosters, espn_reports, qb_depth,
                                   overrides, as_of=cutoff.date() if played else None, inactive=not played)
    out = {}
    for team, inp in inputs.items():
        qb, src = pick_starter(inp)
        out[team] = (qb, src, names.get(qb) if qb else None)
    return out


def project_week(season: int, week: int, cfg: GameModelConfig = GameModelConfig(),
                 d: Optional[Path] = None, refresh: bool = True,
                 starters: Optional[dict[str, str]] = None, espn_reports: Optional[dict] = None,
                 qb_depth: Optional[list] = None) -> pd.DataFrame:
    """Projections for every game of ``season`` week ``week``.

    ``starters`` maps team abbreviation -> gsis QB id and overrides the
    starter resolution (``week_starters``; ``espn_reports`` and ``qb_depth``
    feed it). Returns one row per game, home perspective:
    proj_home / proj_away points, proj_margin (home - away), proj_total,
    p_home_win, the QB ids used (with names and ``*_qb_source``) and each
    starter's delta vs the team's QB level (EPA/dropback; ``picks`` flags
    |delta| > 0.05 as a QB change), and the schedule's consensus lines (for
    comparison and model picks only — never a model input).
    """
    first_season = cfg.first_feature_season - 2  # ratings need two prior seasons
    if refresh:
        data.ensure_games(d)
        data.ensure_pbp(range(first_season, season + 1), d, refresh_seasons=[season])
        ensure_week_status(season, d)
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
    qbs = week_starters(season, week, wk, tg, cutoff, data.load_injuries(season, d), data.load_rosters([season], d),
                        espn_reports=espn_reports, qb_depth=qb_depth, overrides=starters)

    rows = []
    for g in wk.itertuples():
        g = g._replace(roof=infer_roof(games, g.stadium_id, g.roof or ""))
        ctx = context_features(g)
        neutral = g.location == "Neutral"
        sides = {}
        for team, opp, home in ((g.home_team, g.away_team, 0 if neutral else 1), (g.away_team, g.home_team, 0)):
            qb, src, name = qbs[team]
            feats = {**side_features(fits, team, opp, home), **ctx, "qb": qbr.delta(team, qb)}
            if dpts:
                feats["drive"] = dpts.get((g.game_id, team))
            sides[team] = (comb.predict(feats), qb, feats["qb"], src, name)
        (ph, hq, hd, hs, hn), (pa, aq, ad, as_, an) = sides[g.home_team], sides[g.away_team]
        rows.append({
            "game_id": g.game_id, "season": season, "week": week, "gameday": g.gameday.date(), "gametime": g.gametime,
            "home_team": g.home_team, "away_team": g.away_team, "neutral": bool(neutral),
            "roof": g.roof, "proj_home": ph, "proj_away": pa, "proj_margin": ph - pa,
            "proj_total": ph + pa, "p_home_win": float(stats.norm.cdf((ph - pa) / MARGIN_SD)),
            "home_qb_id": hq, "away_qb_id": aq, "home_qb_delta": hd, "away_qb_delta": ad,
            "home_qb_name": hn, "away_qb_name": an, "home_qb_source": hs, "away_qb_source": as_,
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
                  extra_out: Optional[set[tuple[str, str]]] = None,
                  starters: Optional[dict[str, Optional[str]]] = None,
                  extra_questionable: Optional[set[tuple[str, str]]] = None) -> pd.DataFrame:
    """Expected active skill players for each team of ``week_games``.

    A player is expected active if he played for the team in any of its last
    ``player_model.RECENT_GAMES`` games and is not listed Out/Doubtful on that
    week's injury report or in ``extra_out`` (the weekly roster's reserve lists,
    the live ESPN feed). The starting QB comes from ``starters`` (team -> gsis id;
    the schedule's ``*_qb_id`` for a team not in it) and is always added; when he
    is known, the team's other QBs are dropped (a backup's share of carries goes
    to the unprojected slot instead of diluting the backs). Each row carries the
    participation evidence ``player_model.participation_probability`` reads:
    ``recent_games``, ``played_last`` and ``questionable`` (the week's report or
    ``extra_questionable``). (A player returning after missing those games is
    not included — a known limitation.)
    """
    teams = set(week_games["home_team"]) | set(week_games["away_team"])
    recent = recent_team_players(pg, teams, cutoff)
    out = out_players(injuries, week) | (extra_out or set())
    recent = recent[[(t, pid) not in out for t, pid in zip(recent["team"], recent["player_id"])]]
    quest = questionable_players(injuries, week) | (extra_questionable or set())
    names = pg.drop_duplicates("player_id", keep="last").set_index("player_id")["player_display_name"]
    rows = []
    for g in week_games.itertuples():
        for team, opp, home, qb_col in ((g.home_team, g.away_team, 0 if g.location == "Neutral" else 1, "home_qb_id"),
                                        (g.away_team, g.home_team, 0, "away_qb_id")):
            players = recent[recent["team"] == team]
            qb = (starters or {}).get(team) if team in (starters or {}) else getattr(g, qb_col)
            qb = qb if isinstance(qb, str) and qb else None
            if qb:
                players = players[(players["position"] != "QB") | (players["player_id"] == qb)]
            for p in players.itertuples():
                rows.append({"game_id": g.game_id, "team": team, "opp": opp, "home": home,
                             "player_id": p.player_id, "player_display_name": p.player_display_name,
                             "position": p.position, "is_starting_qb": p.player_id == qb,
                             "recent_games": int(p.recent_games), "played_last": bool(p.played_last),
                             "questionable": (team, p.player_id) in quest})
            if qb and qb not in set(players["player_id"]):
                rows.append({"game_id": g.game_id, "team": team, "opp": opp, "home": home, "player_id": qb,
                             "player_display_name": names.get(qb, qb), "position": "QB", "is_starting_qb": True,
                             "recent_games": 0, "played_last": False, "questionable": (team, qb) in quest})
    return pd.DataFrame(rows)


def questionable_players(injuries: Optional[pd.DataFrame], week: int) -> set[tuple[str, str]]:
    """(team, gsis id) listed Questionable on the ``week`` injury report (one season's report)."""
    if injuries is None or injuries.empty:
        return set()
    i = injuries[(injuries["week"] == week) & (injuries["report_status"] == "Questionable")].dropna(subset=["gsis_id"])
    return set(zip(i["team"], i["gsis_id"]))


# Live ESPN injury feed (the scanner's source). nflverse injury reports are the
# backtested source but publish with a lag; on game day the ESPN feed carries
# late downgrades. Same out rule as the coordinator's depth-chart QB starters
# (AgentCoordinator._QB_OUT_STATUSES). The feed keeps only recently updated
# entries (a two-week-old IR placement is gone from it), so the weekly roster
# (player_model.roster_unavailable) is the primary source for reserve lists.
ESPN_OUT_STATUSES = frozenset({
    "OUT", "INJURED RESERVE", "IR", "SUSPENSION", "SUSPENDED", "DOUBTFUL",
    "PHYSICALLY UNABLE TO PERFORM", "PUP", "NON-FOOTBALL INJURY",
})


@dataclass(frozen=True)
class EspnStatus:
    name: str
    position: str
    status: str
    reported_at: Optional[str] = None


@dataclass
class EspnTeamReport:
    team: str
    players: list[EspnStatus] = field(default_factory=list)


def parse_espn_injuries(doc: dict) -> dict[str, EspnTeamReport]:
    """Team full name (lowercase) -> every listed player whose status is not Active."""
    out: dict[str, EspnTeamReport] = {}
    for team in (doc or {}).get("injuries", []) or []:
        if not isinstance(team, dict) or not team.get("displayName"):
            continue
        key = str(team["displayName"]).lower().strip()
        for inj in team.get("injuries", []) or []:
            if not isinstance(inj, dict):
                continue
            raw = inj.get("status", "")
            status = (raw if isinstance(raw, str) else (raw or {}).get("name", "")).strip()
            athlete = inj.get("athlete") or {}
            if not status or status.lower() == "active" or not athlete.get("displayName"):
                continue
            out.setdefault(key, EspnTeamReport(key)).players.append(EspnStatus(
                athlete["displayName"], ((athlete.get("position") or {}).get("abbreviation") or "").upper(),
                status, inj.get("date")))
    return out


def fetch_espn_injury_reports() -> dict[str, EspnTeamReport]:
    """ESPN NFL injury feed, every non-Active status (``parse_espn_injuries``); {} on any failure.

    Read raw, not through ``InjuryReportAgent.run``: the agent keeps a player
    only when he moves a team's win probability, so it drops "Injured Reserve"
    (no impact weight) and decays an "Out" older than 14 days to nothing —
    right for pricing a game, wrong for deciding who plays.
    """
    import httpx

    from evmax.agents.intelligence.injury_agent import SECTOR_INJURY_URLS

    try:
        resp = httpx.get(SECTOR_INJURY_URLS["nfl"][0], timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
        return parse_espn_injuries(resp.json())
    except Exception as e:  # noqa: BLE001 — the live feed is optional; nflverse reports remain
        logger.warning("nfl_proj_espn_injuries_failed", error=str(e))
        return {}


def _espn_abbr(team) -> Optional[str]:
    from evmax.agents.models.nfl_efficiency_agent import NFL_ABBREV_TO_NAME

    return {v: k for k, v in NFL_ABBREV_TO_NAME.items()}.get(str(team).lower().strip())


def espn_statuses(reports: Optional[dict]) -> dict[str, dict[str, str]]:
    """Team abbreviation -> {normalized player name: upper-case ESPN status}."""
    from evmax.clients.nfl_depth_charts import normalize_person

    out: dict[str, dict[str, str]] = {}
    for team, report in (reports or {}).items():
        abbr = _espn_abbr(team)
        if not abbr:
            continue
        for p in getattr(report, "players", None) or []:
            status = (getattr(p, "status", "") or "").upper().strip()
            if status:
                out.setdefault(abbr, {})[normalize_person(p.name)] = status
    return out


# ESPN statuses whose usage teammates absorb — the injury report's Out/Doubtful
# (player_model.OUT_STATUSES). Reserve-list statuses only drop the player (the
# weekly roster's rule: PlayerModelConfig.roster_out_redistribution).
ESPN_REDISTRIBUTED_STATUSES = frozenset({"OUT", "DOUBTFUL"})


def espn_out_players(reports: dict, recent: pd.DataFrame,
                     statuses: frozenset[str] = ESPN_OUT_STATUSES) -> set[tuple[str, str]]:
    """(team, gsis id) of ``recent`` players the ESPN feed lists with one of ``statuses``.

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
            if (getattr(p, "status", "") or "").upper().strip() not in statuses:
                continue
            ids = names.get(normalize_person(p.name), [])
            if len(ids) == 1:
                out.add((abbr, ids[0]))
    return out


def project_week_players(season: int, week: int, cfg: PlayerModelConfig = PlayerModelConfig(),
                         d: Optional[Path] = None, refresh: bool = True,
                         espn_reports: Optional[dict] = None,
                         game_proj: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Player stat-line projections (median, mean, 10th/90th percentile) for a week.

    Team volume is conditioned on this week's GAME projections (``project_week``)
    through the script model trained on completed seasons; usage and efficiency
    use every game before the week's first kickoff. Ruled-out players come from
    the nflverse injury report (Out/Doubtful), the weekly roster (reserve lists,
    practice squad, released; ``roster_unavailable``) and ``espn_reports``
    (``fetch_espn_injury_reports()``), when given; their teammates absorb part of
    their usage. The starting QBs are ``game_proj``'s (``week_starters``).
    ``game_proj`` is this week's ``project_week`` output when the caller already
    has it (it is recomputed otherwise). Run notes (e.g. a missing weekly
    roster) are in ``.attrs["notes"]``.
    """
    gcfg = GameModelConfig()
    first_season = gcfg.first_feature_season - 2
    notes: list[str] = []
    if refresh:
        data.ensure_player_sources(range(first_season, season + 1), d, refresh_seasons=[season])
        notes += ensure_week_status(season, d)
    if game_proj is None:
        game_proj = project_week(season, week, gcfg, d, refresh=refresh, espn_reports=espn_reports)
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
    rosters = data.load_rosters([season], d)
    if rosters.empty:
        notes.append("nflverse weekly roster not cached: injured-reserve players are not filtered")
    recent = recent_team_players(pg, set(wk["home_team"]) | set(wk["away_team"]), cutoff)
    espn_out = espn_out_players(espn_reports, recent) if espn_reports else set()
    espn_freed = espn_out_players(espn_reports, recent, ESPN_REDISTRIBUTED_STATUSES) if espn_reports else set()
    espn_q = espn_out_players(espn_reports, recent, frozenset({"QUESTIONABLE"})) if espn_reports else set()
    unavailable = roster_unavailable(rosters, recent, season, week)
    roster = active_roster(pg, injuries, wk, cutoff, week, extra_out=espn_out | unavailable,
                           starters=game_starters(game_proj), extra_questionable=espn_q)
    # Teammates of players ruled Out/Doubtful take part of their usage (same rule as the backtest).
    freed = out_players(injuries, week) | espn_freed | (unavailable if cfg.roster_out_redistribution else set())
    mult = injury_share_multipliers(state.usage, recent, freed, cfg.injury_redistribution)
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
    proj = proj.merge(wk[["game_id", "gameday", "gametime", "home_team", "away_team"]], on="game_id").assign(
        season=season, week=week)
    proj.attrs["notes"] = notes
    return proj


def game_starters(game_proj: Optional[pd.DataFrame]) -> dict[str, Optional[str]]:
    """team -> starting QB gsis id from a ``project_week`` frame ({} without the QB columns)."""
    if game_proj is None or game_proj.empty or "home_qb_id" not in game_proj:
        return {}
    out: dict[str, Optional[str]] = {}
    for r in game_proj.itertuples(index=False):
        out[r.home_team] = _str_id(r.home_qb_id)
        out[r.away_team] = _str_id(r.away_qb_id)
    return out


def simulate_game(season: int, week: int, team: str, n: int = 10000, d: Optional[Path] = None,
                  refresh: bool = True, espn_reports: Optional[dict] = None, seed: int = 0,
                  proj: Optional[pd.DataFrame] = None):
    """Joint simulation of ``team``'s game in ``season`` ``week``.

    Returns {team: (players DataFrame, sims dict)} for both teams; the shape
    parameters are fitted on the six completed seasons before ``season``.
    ``proj`` is the week's ``project_week_players`` output when the caller
    already has it (the dashboard reuses its last run); it is recomputed otherwise.
    """
    from evmax.nfl_projections import simulate

    if proj is None:
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
