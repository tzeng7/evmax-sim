"""Point Projection Model — ensemble-backed team score predictions.

For NBA, uses PossessionSimAgent's Monte Carlo engine directly (the same one
the EV pipeline uses for spreads/totals). For other sectors, uses a
Poisson attack/defense × Elo-implied-margin blend from state files.

Returns None (not silent defaults) when either team is missing from state —
this is the conviction tool used in playoffs, and fake projections would be
worse than no projection at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

DATA_DIR = Path("data/models")

ELO_HOME_ADV: dict[str, float] = {
    "nba": 100.0,
    "nfl": 48.0,
    "ncaab": 70.0,
    "ncaaw": 70.0,
    "soccer": 60.0,
}

LEAGUE_AVG_DEFAULTS: dict[str, dict[str, float]] = {
    "nba": {"home": 113.5, "away": 111.0},
    "nfl": {"home": 23.5, "away": 21.5},
    "ncaab": {"home": 72.0, "away": 70.0},
    "ncaaw": {"home": 68.0, "away": 65.0},
    "soccer": {"home": 1.55, "away": 1.15},
}

SECTOR_SIGMA: dict[str, float] = {
    "nba": 11.5,
    "nfl": 14.0,
    "ncaab": 12.5,
    "ncaaw": 12.5,
    "soccer": 1.9,
}

DEFAULT_ELO = 1500.0
ELO_BLEND_WEIGHT = 0.35

# NBA playoff window (months). Play-in starts mid-April, Finals end mid-June.
NBA_PLAYOFF_MONTHS = {4, 5, 6}

# Injury-based ORTG deltas. Derived from the intuition that a star scorer
# contributes ~4 ORTG above a replacement-level starter. Stars out = big
# offensive hit; starters out = moderate; day-to-day = half impact; rotation
# players left alone (they weren't meaningfully on the floor anyway).
ORTG_DELTA_STAR_OUT = -4.0
ORTG_DELTA_STARTER_OUT = -1.5
ORTG_DELTA_STAR_DTD = -2.0
ORTG_DELTA_STARTER_DTD = -0.75
ORTG_DELTA_CAP = -8.0  # Per team; prevents runaway when ≥3 starters out


def is_nba_playoff_date(game_date: Optional[str]) -> bool:
    """Return True if ``game_date`` (YYYY-MM-DD) falls in the NBA playoff window."""
    if not game_date:
        return False
    try:
        parsed = date.fromisoformat(game_date[:10])
    except (ValueError, TypeError):
        return False
    return parsed.month in NBA_PLAYOFF_MONTHS


def compute_injury_ortg_delta(injury_report) -> tuple[float, list[str]]:
    """Compute ORTG delta (negative number) + human-readable notes for one team.

    ``injury_report`` is an ``InjuryReport`` instance from ``InjuryReportAgent``
    (or any object exposing ``.players`` with ``.name``, ``.status``, ``.tier``).
    Returns (delta, notes). Delta is floored at ORTG_DELTA_CAP.
    """
    if injury_report is None or not getattr(injury_report, "players", None):
        return 0.0, []

    delta = 0.0
    notes: list[str] = []
    for p in injury_report.players:
        status = (p.status or "").lower().strip()
        tier = (p.tier or "").lower().strip()

        if status == "out":
            if tier == "star":
                d = ORTG_DELTA_STAR_OUT
            elif tier == "starter":
                d = ORTG_DELTA_STARTER_OUT
            else:
                d = 0.0  # rotation players: ignore
        elif status == "day-to-day":
            if tier == "star":
                d = ORTG_DELTA_STAR_DTD
            elif tier == "starter":
                d = ORTG_DELTA_STARTER_DTD
            else:
                d = 0.0
        else:
            d = 0.0

        if d != 0.0:
            delta += d
            notes.append(f"{p.name} ({tier}/{status})")

    # Cap at ORTG_DELTA_CAP (more negative than the cap → clamp).
    delta = max(delta, ORTG_DELTA_CAP)
    return delta, notes


def _match_injury_team(reports: dict, team_name: str):
    """Find the injury report matching ``team_name`` via substring match.

    Mirrors the matching style of InjuryReportAgent.apply_adjustments.
    """
    if not reports or not team_name:
        return None
    key = team_name.lower().strip()
    for team_key, report in reports.items():
        tk = team_key.lower()
        if key in tk or tk in key:
            return report
    return None

# City → team name mapping for common lookups (lowercase)
CITY_TO_TEAM: dict[str, str] = {
    # NBA
    "atlanta": "hawks", "boston": "celtics", "brooklyn": "nets",
    "charlotte": "hornets", "chicago": "bulls", "cleveland": "cavaliers",
    "dallas": "mavericks", "denver": "nuggets", "detroit": "pistons",
    "golden state": "warriors", "houston": "rockets", "indiana": "pacers",
    "memphis": "grizzlies", "miami": "heat", "milwaukee": "bucks",
    "minnesota": "timberwolves", "new orleans": "pelicans",
    "new york": "knicks", "oklahoma city": "thunder", "okc": "thunder",
    "orlando": "magic", "philadelphia": "76ers", "philly": "76ers",
    "phoenix": "suns", "portland": "trail blazers", "sacramento": "kings",
    "san antonio": "spurs", "toronto": "raptors", "utah": "jazz",
    "washington": "wizards", "la clippers": "la clippers",
    "los angeles lakers": "lakers", "los angeles clippers": "la clippers",
    # NFL
    "green bay": "packers", "kansas city": "chiefs",
    "las vegas": "raiders", "jacksonville": "jaguars",
    "pittsburgh": "steelers", "baltimore": "ravens",
    "cincinnati": "bengals", "buffalo": "bills",
    "seattle": "seahawks", "tampa bay": "buccaneers",
    "tennessee": "titans", "carolina": "panthers",
    "new england": "patriots", "arizona": "cardinals",
}


@dataclass
class GameProjection:
    """Projected scores and derived lines for a single game."""

    home_team: str
    away_team: str
    home_points: float
    away_points: float
    projected_spread: float  # negative = home favored
    projected_total: float
    margin_sigma: float       # std of modelled margin distribution
    total_sigma: float        # std of modelled total distribution
    win_prob_home: float      # P(home wins) from the engine
    home_elo: Optional[float]
    away_elo: Optional[float]
    confidence: str           # "high", "medium", "low"
    engine: str               # "possession_sim" | "poisson_elo"
    sector: str
    is_playoff: bool = False  # whether playoff tightening was applied
    home_ortg_adj: float = 0.0          # ORTG delta applied from injuries
    away_ortg_adj: float = 0.0
    home_injury_notes: Optional[str] = None  # "Luka Doncic (S/out); Austin Reaves (S/out)"
    away_injury_notes: Optional[str] = None

    # Backwards-compat shim — legacy CLI/tests read ``elo_win_prob_home``.
    @property
    def elo_win_prob_home(self) -> float:
        return self.win_prob_home

    @property
    def spread_pick(self) -> str:
        if abs(self.projected_spread) < 1.0:
            return "Pick"
        return self.home_team if self.projected_spread < 0 else self.away_team


class PointProjectionModel:
    """Projects scores using the PossessionSim engine (NBA) or Poisson+Elo blend (other sectors)."""

    def __init__(self) -> None:
        self._poisson_state: dict = {}
        self._elo_state: dict = {}
        self._efficiency_state: dict = {}
        self._possession_agent = None
        self._load_state()

    def _load_state(self) -> None:
        for attr, filename in [
            ("_poisson_state", "poisson_state.json"),
            ("_elo_state", "elo_state.json"),
            ("_efficiency_state", "efficiency_state.json"),
        ]:
            path = DATA_DIR / filename
            if path.exists():
                try:
                    setattr(self, attr, json.loads(path.read_text()))
                except Exception:
                    logger.warning("point_projection_state_load_failed", file=filename)

    def _get_possession_agent(self):
        if self._possession_agent is None:
            from evmax.agents.models.possession_sim_agent import PossessionSimAgent
            self._possession_agent = PossessionSimAgent()
        return self._possession_agent

    def _get_league_avg(self, sector: str) -> Optional[dict[str, float]]:
        return (
            self._poisson_state.get(sector, {}).get("league_avg")
            or LEAGUE_AVG_DEFAULTS.get(sector)
        )

    def _resolve_key(self, store: dict, team: str) -> Optional[str]:
        """Resolve a team name against a dict using exact, last-word, suffix fallbacks."""
        if team in store:
            return team
        if " " in team:
            last = team.rsplit(" ", 1)[-1]
            if last in store:
                return last
        for key in store:
            if team.endswith(key) or key.endswith(team):
                return key
        return None

    def _normalize_team(self, sector: str, team: str) -> str:
        team_lower = team.lower().strip()
        if team_lower in CITY_TO_TEAM:
            return CITY_TO_TEAM[team_lower]
        try:
            from evmax.matching.normalizer import NameNormalizer
            norm = NameNormalizer(sector)
            result = norm.normalize(team_lower)
            return result if result else team_lower
        except Exception:
            return team_lower

    def _get_poisson_stats(self, sector: str, team: str) -> Optional[dict]:
        teams = self._poisson_state.get(sector, {}).get("teams", {})
        key = self._resolve_key(teams, team)
        return teams.get(key) if key else None

    def _get_elo(self, sector: str, team: str) -> Optional[float]:
        ratings = self._elo_state.get(sector, {}).get("ratings", {})
        key = self._resolve_key(ratings, team)
        if key is None:
            return None
        return ratings.get(key)

    def project(
        self,
        home_team: str,
        away_team: str,
        sector: str,
        game_date: Optional[str] = None,
        injury_reports: Optional[dict] = None,
    ) -> Optional[GameProjection]:
        """Generate a point projection. Returns None if required state is missing.

        ``game_date`` (YYYY-MM-DD) enables playoff detection for NBA. When the
        date falls inside the NBA playoff window the sim applies defensive
        tightening (see PossessionSimAgent.PLAYOFF_ORTG_FACTOR).

        ``injury_reports`` is the dict produced by ``InjuryReportAgent`` (team
        name → InjuryReport). NBA only; other sectors ignore it for now.
        """
        sector = sector.lower()
        home_key = self._normalize_team(sector, home_team)
        away_key = self._normalize_team(sector, away_team)

        if sector == "nba":
            return self._project_nba(
                home_team, away_team, home_key, away_key,
                game_date=game_date, injury_reports=injury_reports,
            )
        return self._project_poisson_elo(home_team, away_team, home_key, away_key, sector)

    def _project_nba(
        self,
        home_display: str,
        away_display: str,
        home_key: str,
        away_key: str,
        game_date: Optional[str] = None,
        injury_reports: Optional[dict] = None,
    ) -> Optional[GameProjection]:
        agent = self._get_possession_agent()
        is_playoff = is_nba_playoff_date(game_date)

        home_ortg_adj = 0.0
        away_ortg_adj = 0.0
        home_notes: list[str] = []
        away_notes: list[str] = []
        if injury_reports:
            home_report = _match_injury_team(injury_reports, home_display)
            away_report = _match_injury_team(injury_reports, away_display)
            home_ortg_adj, home_notes = compute_injury_ortg_delta(home_report)
            away_ortg_adj, away_notes = compute_injury_ortg_delta(away_report)

        sim = agent.simulate_matchup(
            home_key, away_key,
            is_playoff=is_playoff,
            ortg_adj_a=home_ortg_adj,
            ortg_adj_b=away_ortg_adj,
        )
        if sim is None:
            logger.debug(
                "point_projection_nba_sim_unavailable",
                home=home_key, away=away_key,
            )
            return None

        scores_a = sim["scores_a"]
        scores_b = sim["scores_b"]
        margins = sim["margin"]
        totals = sim["total"]

        home_pts = float(scores_a.mean())
        away_pts = float(scores_b.mean())
        margin_mean = float(margins.mean())
        total_mean = float(totals.mean())
        margin_sigma = float(margins.std())
        total_sigma = float(totals.std())

        stats_a = sim["stats_a"]
        stats_b = sim["stats_b"]
        min_gp = min(stats_a.get("gp", 0), stats_b.get("gp", 0))
        confidence = "high" if min_gp >= 60 else ("medium" if min_gp >= 30 else "low")

        elo_h = self._get_elo("nba", home_key)
        elo_a = self._get_elo("nba", away_key)

        return GameProjection(
            home_team=home_display,
            away_team=away_display,
            home_points=round(home_pts, 2),
            away_points=round(away_pts, 2),
            projected_spread=round(-margin_mean, 2),
            projected_total=round(total_mean, 2),
            margin_sigma=round(margin_sigma, 2),
            total_sigma=round(total_sigma, 2),
            win_prob_home=round(float(sim["prob_a"]), 4),
            home_elo=round(elo_h, 1) if elo_h is not None else None,
            away_elo=round(elo_a, 1) if elo_a is not None else None,
            confidence=confidence,
            engine="possession_sim",
            sector="nba",
            is_playoff=is_playoff,
            home_ortg_adj=round(home_ortg_adj, 2),
            away_ortg_adj=round(away_ortg_adj, 2),
            home_injury_notes="; ".join(home_notes) if home_notes else None,
            away_injury_notes="; ".join(away_notes) if away_notes else None,
        )

    def _project_poisson_elo(
        self,
        home_display: str,
        away_display: str,
        home_key: str,
        away_key: str,
        sector: str,
    ) -> Optional[GameProjection]:
        from scipy.stats import norm

        home_stats = self._get_poisson_stats(sector, home_key)
        away_stats = self._get_poisson_stats(sector, away_key)
        if not home_stats or not away_stats:
            logger.debug(
                "point_projection_poisson_state_missing",
                sector=sector, home=home_key, away=away_key,
                home_found=bool(home_stats), away_found=bool(away_stats),
            )
            return None

        elo_h = self._get_elo(sector, home_key)
        elo_a = self._get_elo(sector, away_key)
        if elo_h is None or elo_a is None:
            logger.debug(
                "point_projection_elo_missing",
                sector=sector, home=home_key, away=away_key,
            )
            return None

        avg = self._get_league_avg(sector)
        if avg is None:
            return None
        avg_h = avg["home"]
        avg_a = avg["away"]

        home_attack = home_stats["attack"]
        home_defense = home_stats["defense"]
        away_attack = away_stats["attack"]
        away_defense = away_stats["defense"]

        poisson_home = home_attack * away_defense * avg_h
        poisson_away = away_attack * home_defense * avg_a
        poisson_margin = poisson_home - poisson_away

        home_adv_elo = ELO_HOME_ADV.get(sector, 50.0)
        sigma = SECTOR_SIGMA.get(sector, 12.0)
        elo_wp = 1.0 / (1.0 + 10 ** ((elo_a - elo_h - home_adv_elo) / 400))
        elo_wp_clamped = max(0.01, min(0.99, elo_wp))
        elo_margin = norm.ppf(elo_wp_clamped) * sigma

        blended_margin = (1 - ELO_BLEND_WEIGHT) * poisson_margin + ELO_BLEND_WEIGHT * elo_margin
        margin_shift = (blended_margin - poisson_margin) / 2
        home_pts = poisson_home + margin_shift
        away_pts = poisson_away - margin_shift

        min_games = min(home_stats.get("games", 0), away_stats.get("games", 0))
        confidence = "high" if min_games >= 15 else ("medium" if min_games >= 5 else "low")

        # Total σ heuristic — without a sim we scale margin σ up since totals vary more
        total_sigma = sigma * 1.6

        return GameProjection(
            home_team=home_display,
            away_team=away_display,
            home_points=round(home_pts, 2),
            away_points=round(away_pts, 2),
            projected_spread=round(-(home_pts - away_pts), 2),
            projected_total=round(home_pts + away_pts, 2),
            margin_sigma=round(sigma, 2),
            total_sigma=round(total_sigma, 2),
            win_prob_home=round(float(elo_wp), 4),
            home_elo=round(elo_h, 1),
            away_elo=round(elo_a, 1),
            confidence=confidence,
            engine="poisson_elo",
            sector=sector,
        )

    def project_from_sharp(
        self,
        home_team: str,
        away_team: str,
        sector: str,
        book_spread: Optional[float] = None,
        book_total: Optional[float] = None,
        spread_threshold: float = 1.0,
        total_threshold: float = 1.5,
        game_date: Optional[str] = None,
        injury_reports: Optional[dict] = None,
    ) -> Optional[dict]:
        """Project scores and compare against sportsbook lines.

        ``book_spread`` is expressed as the home team's handicap (negative = home fav).
        ``game_date`` (YYYY-MM-DD) enables playoff detection for NBA.
        ``injury_reports`` (NBA only) applies ORTG deltas for OUT/DAY-TO-DAY players.
        Returns None if the projection itself is unavailable.
        """
        proj = self.project(
            home_team, away_team, sector,
            game_date=game_date, injury_reports=injury_reports,
        )
        if proj is None:
            return None

        result = {
            "projection": proj,
            "book_spread": book_spread,
            "book_total": book_total,
            "spread_edge": None,
            "total_edge": None,
            "spread_play": None,
            "total_play": None,
        }

        if book_spread is not None:
            spread_diff = proj.projected_spread - book_spread
            result["spread_edge"] = round(spread_diff, 2)
            if abs(spread_diff) >= spread_threshold:
                # Model projects a more-negative home spread → home is the value side
                result["spread_play"] = proj.home_team if spread_diff < 0 else proj.away_team

        if book_total is not None:
            total_diff = proj.projected_total - book_total
            result["total_edge"] = round(total_diff, 2)
            if abs(total_diff) >= total_threshold:
                result["total_play"] = "Over" if total_diff > 0 else "Under"

        return result

    def available_teams(self, sector: str) -> list[str]:
        """List teams with projection data for a sector."""
        sector = sector.lower()
        if sector == "nba":
            nba = self._efficiency_state.get("nba", {})
            teams = nba.get("teams", {})
            if teams:
                return sorted(teams.keys())
        return sorted(self._poisson_state.get(sector, {}).get("teams", {}).keys())

    def detect_sector(self, team: str) -> Optional[str]:
        """Auto-detect which sector a team belongs to by checking Poisson + Elo state."""
        team_lower = team.lower().strip()

        # NBA gets priority since it has the richest state (efficiency)
        nba_teams = self._efficiency_state.get("nba", {}).get("teams", {})
        if nba_teams:
            if self._resolve_key(nba_teams, team_lower):
                return "nba"
            normalized = self._normalize_team("nba", team_lower)
            if normalized != team_lower and self._resolve_key(nba_teams, normalized):
                return "nba"

        for sector in self._poisson_state:
            teams = self._poisson_state[sector].get("teams", {})
            if self._resolve_key(teams, team_lower):
                return sector
            normalized = self._normalize_team(sector, team_lower)
            if normalized != team_lower and self._resolve_key(teams, normalized):
                return sector

        for sector in self._elo_state:
            ratings = self._elo_state[sector].get("ratings", {})
            if self._resolve_key(ratings, team_lower):
                return sector
            normalized = self._normalize_team(sector, team_lower)
            if normalized != team_lower and self._resolve_key(ratings, normalized):
                return sector
        return None
