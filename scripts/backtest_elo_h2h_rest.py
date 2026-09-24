"""Walk-forward A/B of the Elo agent's H2H nudge and rest-day layer.

Why: ``EloModelAgent`` reads two features under the RAW lowercased Pinnacle
label instead of the resolved state key:

  * H2H — ``update()`` records the pair under the names it is fed (canonical
    slugs such as "lakers"), but ``_h2h_adjustment`` looks the pair up under
    the Pinnacle label ("los angeles lakers").
  * Rest — ``_days_of_rest`` / ``_congestion_penalty`` read form_state.json
    with ``form[sector].get(label)``, while form_state is keyed by the Form
    agent's resolved keys.

For sectors whose Pinnacle labels differ from their keys (NBA, NFL, WNBA,
baseball, much of soccer) both features are silently dead in production.
Routing the lookups through ``_team_lookup.resolve_team_key`` would switch
them ON — a model change. This harness measures whether switching them on
helps, out of sample, before anything ships.

Protocol (leak-free, mirrors scripts/backtest_nfl_elo_regression.py):
  - Replay the PRODUCTION ``EloModelAgent.update()`` / ``_win_probs()`` from a
    cold start over the sector's game history in date order, with the agent's
    clock patched to each game date (recency-K fidelity). Team names are the
    sector canonicals, so H2H records and reads use the SAME keys — the "fixed"
    behaviour.
  - Rest days and 7-day game counts come from the replay schedule itself (each
    team's previous replay game), measured to the game date — exactly what the
    live layer reads from form_state.json, point-in-time.
  - Variants, predicted from the SAME rating trajectory (H2H and rest only
    touch prediction, never the update):
        off   — H2H nudge 0, rest bonus 0 (today's production for label≠key)
        h2h   — H2H on, rest off
        rest  — rest on (incl. soccer congestion), H2H off
        on    — both on (what routing the lookups through resolve_team_key does)
  - Offseason handling mirrors live: WNBA keep 0.65 and NFL keep 0.667 with the
    season_games reset; every other sector carries ratings across seasons.
    The H2H store is never reset (live accumulates it for the life of the
    state file).
  - The 60-day Elo staleness guard is applied (live returns None on a stale
    sector, so those games carry no Elo in either variant and are skipped),
    and only games where both sides have >= LOW_DATA_THRESHOLD Elo games are
    scored (below it Elo's confidence is <= 0.45 and the ensemble drops it).

Scoring:
  - Standalone Elo Brier (2-way P(home); soccer = 3-way H/D/A sum-of-squares),
    paired per game against ``off``: mean Δ×1000, SE, z. Negative = better.
  - Blend Brier through the production ``EnsembleModelAgent._blend`` (weight
    overrides, 0.45 confidence gate, disagreement ramp, FLB re-blend) with the
    Elo + Form generic stack and the real Pinnacle close as the sharp anchor,
    at the live sharp weight (model_config.json; soccer = league tier). Only
    Elo + Form fire here, so Elo's share of the model side is 2–5× its live
    share — this is an UPPER bound on the live blend effect. Per-sector
    isotonic calibration is disabled (it was fit on the full live stack).
  - CLV lens: OLS slope of the sharp line's move (close − open, home side) on
    the variant's Elo shift (p_variant − p_off). A positive, significant slope
    means the feature carries information the market prices in later. Soccer:
    football-data PSH → PSCH; NBA/WNBA/baseball: archive.db first → last
    pre-tip Pinnacle snapshot (2026 only).

Sharp anchors: soccer = football-data.co.uk Pinnacle close (data/backtest/
soccer/<season>/*.csv, 5 domestic leagues); NFL = nflreadpy closing
moneylines (power devig); NBA/WNBA/baseball = archive.db Pinnacle (2026
live era only, opened read-only). archive.db / predictions.db default to the
MAIN checkout's data/ (resolved via git) so the script works from a worktree.

--kalshi-clv adds the Kalshi CLV lens: logged moneyline rows from the same
row fetch as ``evmax cleanup shadow clv`` (on a snapshot of predictions.db),
with each row's Kalshi CLV regressed on the feature's shift of the backed
team's Elo probability.

Gate (fixed before running): a feature ships for a sector only when
  (a) pooled evaluation-season standalone ΔBrier < 0 with z ≤ −1.64, AND
  (b) the holdout season's standalone ΔBrier ≤ 0, AND
  (c) the holdout blend ΔBrier ≤ 0 where a sharp anchor exists.
No constant is tuned here (H2H_MAX_ADJ / H2H_MIN_GAMES / REST_ELO_ADJ are
the shipped values), so the evaluation/holdout split is a robustness check,
not a fit/test split.

Usage:
    uv run python scripts/backtest_elo_h2h_rest.py --coverage       # label-miss table only
    uv run python scripts/backtest_elo_h2h_rest.py                  # all sectors
    uv run python scripts/backtest_elo_h2h_rest.py -s nba,soccer
    uv run python scripts/backtest_elo_h2h_rest.py --kalshi-clv --json out.json
ESPN scoreboards are cached under data/cache/elo_feature_backtest/ (gitignored).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sqlite3
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from zoneinfo import ZoneInfo

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import evmax.agents.models.elo_agent as elo_mod  # noqa: E402
from evmax.agents.models.base import ModelAgentPrediction  # noqa: E402
from evmax.agents.models.elo_agent import (  # noqa: E402
    DEFAULT_ELO,
    LOW_DATA_THRESHOLD,
    MED_DATA_THRESHOLD,
    EloModelAgent,
)
from evmax.agents.models.ensemble_agent import EnsembleModelAgent  # noqa: E402
from evmax.agents.models.form_agent import FormModelAgent  # noqa: E402
from evmax.ev.devig import devig_three_way, devig_two_way  # noqa: E402
from evmax.matching.normalizer import NameNormalizer  # noqa: E402

VARIANTS = ("off", "h2h", "rest", "on")
_ET = ZoneInfo("America/New_York")
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
CACHE_DIR = _REPO_ROOT / "data" / "cache" / "elo_feature_backtest"
SOCCER_CSV_DIR = _REPO_ROOT / "data" / "backtest" / "soccer"
MODEL_CONFIG = _REPO_ROOT / "data" / "model_config.json"


def _live_data_dir() -> Path:
    """The MAIN checkout's data/ (archive.db / predictions.db are gitignored,
    so a worktree only has stubs). Falls back to this checkout's data/."""
    import subprocess

    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(common).parent / "data"
    except (OSError, subprocess.CalledProcessError):
        return _REPO_ROOT / "data"


ARCHIVE_DB = _live_data_dir() / "archive.db"
PREDICTIONS_DB = _live_data_dir() / "predictions.db"

SOCCER_ESPN_LEAGUES = {
    "epl": "eng.1", "laliga": "esp.1", "bundesliga": "ger.1", "seriea": "ita.1",
    "ligue1": "fra.1", "ucl": "UEFA.CHAMPIONS", "uel": "uefa.europa",
}
SOCCER_CSV_LEAGUES = {"E0": "epl", "SP1": "laliga", "D1": "bundesliga", "I1": "seriea", "F1": "ligue1"}


def _months(start: str, end: str) -> list[str]:
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _days_in(months: list[str]) -> list[str]:
    out = []
    for mo in months:
        d = date(int(mo[:4]), int(mo[4:]), 1)
        while d.month == int(mo[4:]):
            out.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
    return out


# sector → replay config. `eval`/`holdout` are season labels (start year for
# split-year leagues). `regress` = offseason keep applied before each new season.
SECTORS: dict[str, dict] = {
    "nba": {"sport": "basketball", "league": "nba", "months": _months("2021-10", "2026-06"),
            "eval": [2022, 2023, 2024], "holdout": 2025, "archive": True},
    "wnba": {"sport": "basketball", "league": "wnba", "months": _months("2021-05", "2026-09"),
             "eval": [2022, 2023, 2024, 2025], "holdout": 2026, "archive": True, "regress": 0.65},
    "baseball": {"sport": "baseball", "league": "mlb", "months": _months("2022-04", "2026-09"),
                 "eval": [2023, 2024, 2025], "holdout": 2026, "archive": True},
    "ncaab": {"sport": "basketball", "league": "mens-college-basketball",
              "months": _months("2022-11", "2023-04") + _months("2023-11", "2024-04")
              + _months("2024-11", "2025-04") + _months("2025-11", "2026-04"),
              "daily": True, "params": {"groups": "50"},
              "eval": [2023, 2024], "holdout": 2025, "archive": True},
    "nhl": {"sport": "hockey", "league": "nhl", "months": _months("2021-10", "2026-06"),
            "eval": [2022, 2023, 2024], "holdout": 2025},
    "soccer": {"months": _months("2022-08", "2026-06"),
               "eval": [2023, 2024], "holdout": 2025},
    "nfl": {"eval": list(range(2017, 2025)), "holdout": 2025, "regress": 0.667},
}


def season_of(sector: str, d: date) -> int:
    """Season label: calendar year for summer leagues, start year otherwise."""
    if sector in ("wnba", "baseball"):
        return d.year
    return d.year if d.month >= 7 else d.year - 1


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_backtest_elo_h2h_rest.py)
# ---------------------------------------------------------------------------


class ScheduleTracker:
    """Point-in-time schedule memory for the rest layer.

    Mirrors what ``EloModelAgent._days_of_rest`` / ``_games_in_last_n_days``
    read from form_state.json live: a team's PRIOR games only (the game being
    predicted is recorded after it is scored).
    """

    def __init__(self) -> None:
        self._dates: dict[str, list[date]] = defaultdict(list)

    def record(self, team: str, d: date) -> None:
        ds = self._dates[team]
        if ds and d < ds[-1]:
            ds.insert(bisect_left(ds, d), d)
        else:
            ds.append(d)

    def days_of_rest(self, team: str, reference: date) -> Optional[int]:
        ds = self._dates.get(team)
        if not ds:
            return None
        return (reference - ds[-1]).days

    def games_in_window(self, team: str, reference: date, days: int = 7) -> int:
        """Games on or after ``reference - days`` and before ``reference``
        (the live ``gd >= cutoff`` loop over prior records)."""
        ds = self._dates.get(team, [])
        lo = bisect_left(ds, reference - timedelta(days=days))
        hi = bisect_left(ds, reference)
        return max(0, hi - lo)


def paired_delta(a: list[float], b: list[float]) -> tuple[float, float, float, int]:
    """Mean of (a − b) ×1000, its SE ×1000, z, n. Negative mean = a better."""
    n = len(a)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    z = m / se if se > 0 else 0.0
    return m * 1000, se * 1000, z, n


def ols_slope(x: list[float], y: list[float]) -> tuple[float, float, int]:
    """OLS slope of y on x (with intercept) and its t-statistic."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return float("nan"), float("nan"), n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    slope = sxy / sxx
    resid = [b - my - slope * (a - mx) for a, b in zip(x, y)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    if se == 0:  # perfect fit
        return slope, (math.copysign(math.inf, slope) if slope else 0.0), n
    return slope, slope / se, n


def brier(probs: tuple[float, ...], outcome: tuple[float, ...]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcome))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class Game:
    d: date
    home: str
    away: str
    hs: float
    as_: float
    league: Optional[str] = None
    sharp: Optional[tuple[float, float, Optional[float]]] = None      # close (a, b, draw)
    sharp_open: Optional[tuple[float, float, Optional[float]]] = None
    season: int = 0


def _espn_get(client, url: str, params: dict) -> dict:
    key = url.split("/sports/")[1].replace("/", "_") + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = client.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    period = str(params.get("dates", ""))
    today = date.today().strftime("%Y%m%d")
    if period and period[:6] < today[:6]:  # only cache closed months
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return data


def _parse_espn(data: dict, league_slug: str) -> list[dict]:
    from evmax.sectors.soccer_leagues import espn_display_name

    out = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        if (ev.get("season") or {}).get("type") == 1:   # preseason never trains live state
            continue
        cs = comp.get("competitors", [])
        home = next((c for c in cs if c.get("homeAway") == "home"), None)
        away = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            hs, as_ = float(home.get("score", 0)), float(away.get("score", 0))
        except (TypeError, ValueError):
            continue
        # ET calendar day, not ESPN's UTC stamp: live rest is measured between
        # ET days (form records carry the ET resolve date; Kalshi event_date
        # is anchored to the ET game day). A 7:30 pm ET tip is the NEXT UTC
        # day, so UTC days turn a Sat-night → Sun-matinee pair into "0 days".
        try:
            et_day = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(_ET).date()
        except (KeyError, ValueError):
            continue
        out.append({
            "date": et_day.isoformat(),
            "home": espn_display_name(league_slug, home["team"].get("displayName", "")),
            "away": espn_display_name(league_slug, away["team"].get("displayName", "")),
            "hs": hs, "as": as_,
        })
    return out


def load_espn_games(sector: str) -> list[Game]:
    import httpx

    from evmax.agents.cleanup.resolver import _ESPN_HTTP_UA

    cfg = SECTORS[sector]
    norm = NameNormalizer(sector)
    raw: list[tuple[dict, Optional[str]]] = []
    with httpx.Client(headers={"User-Agent": _ESPN_HTTP_UA}) as client:
        if sector == "soccer":
            jobs = [(lg, "soccer", slug, cfg["months"]) for lg, slug in SOCCER_ESPN_LEAGUES.items()]
        else:
            periods = _days_in(cfg["months"]) if cfg.get("daily") else cfg["months"]
            jobs = [(None, cfg["sport"], cfg["league"], periods)]
        for lg, sport, slug, periods in jobs:
            url = f"{ESPN_BASE}/{sport}/{slug}/scoreboard"
            for p in periods:
                params = {"dates": p, "limit": 1000, **cfg.get("params", {})}
                try:
                    data = _espn_get(client, url, params)
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARN {sector} {slug} {p}: {exc}")
                    continue
                raw.extend((g, lg) for g in _parse_espn(data, slug))
    # Closed leagues: drop All-Star / exhibition sides (live prunes them from
    # Elo via seed_espn.prune_low_game_teams). Soccer and college keep every
    # side — their alias maps don't enumerate every club/school.
    closed = sector in ("nba", "wnba", "baseball", "nhl")
    games: list[Game] = []
    seen = set()
    for g, lg in raw:
        h, a = norm.normalize(g["home"]), norm.normalize(g["away"])
        if not h or not a or h == a:
            continue
        if closed and not (norm.is_known_team(h) and norm.is_known_team(a)):
            continue
        d = date.fromisoformat(g["date"])
        key = (d, h, a, g["hs"], g["as"])
        if key in seen:
            continue
        seen.add(key)
        games.append(Game(d, h, a, g["hs"], g["as"], league=lg, season=season_of(sector, d)))
    games.sort(key=lambda x: (x.d, x.home, x.away))
    return games


def load_nfl_games() -> list[Game]:
    import nflreadpy as nfl
    import yaml

    y = yaml.safe_load((_REPO_ROOT / "evmax" / "sectors" / "aliases" / "nfl.yaml").read_text())
    aliases = y.get("aliases", y)
    codes = {str(k).lower(): v for k, v in aliases.items() if isinstance(v, str)}
    codes.update({"la": "rams", "stl": "rams", "oak": "raiders", "sd": "chargers"})
    df = nfl.load_schedules(seasons=list(range(2015, 2026)))
    df = df.filter(df["home_score"].is_not_null() & df["away_score"].is_not_null())
    df = df.sort(["gameday", "gametime", "game_id"])
    out: list[Game] = []
    for r in df.iter_rows(named=True):
        h, a = codes[str(r["home_team"]).lower()], codes[str(r["away_team"]).lower()]
        sharp = None
        hm, am = r.get("home_moneyline"), r.get("away_moneyline")
        if hm and am:
            dec = [1 + (m / 100 if m > 0 else 100 / -m) for m in (hm, am)]
            pa, pb, _ = devig_two_way(dec[0], dec[1])
            sharp = (pa, pb, None)
        out.append(Game(date.fromisoformat(str(r["gameday"])), h, a, float(r["home_score"]),
                        float(r["away_score"]), sharp=sharp, season=int(r["season"])))
    return out


def attach_soccer_csv(games: list[Game]) -> int:
    """Join football-data Pinnacle PSH (early) / PSCH (close) onto ESPN games."""
    norm = NameNormalizer("soccer")
    by_day: dict[tuple[date, str], list[Game]] = defaultdict(list)
    for g in games:
        if g.league in SOCCER_CSV_LEAGUES.values():
            by_day[(g.d, g.league)].append(g)
    attached = 0
    for season_dir in sorted(SOCCER_CSV_DIR.glob("[0-9][0-9][0-9][0-9]")):
        for code, lg in SOCCER_CSV_LEAGUES.items():
            path = season_dir / f"{code}.csv"
            if not path.exists():
                continue
            with open(path, encoding="utf-8-sig", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    try:
                        d = datetime.strptime(row["Date"].strip(), "%d/%m/%Y").date()
                        close = [float(row[c]) for c in ("PSCH", "PSCA", "PSCD")]
                    except (KeyError, ValueError):
                        continue
                    try:
                        early = [float(row[c]) for c in ("PSH", "PSA", "PSD")]
                    except (KeyError, ValueError):
                        early = None
                    h, a = norm.normalize(row["HomeTeam"]), norm.normalize(row["AwayTeam"])
                    cands = [g for dd in (d, d - timedelta(days=1), d + timedelta(days=1))
                             for g in by_day.get((dd, lg), [])]
                    exact = [g for g in cands if g.home == h and g.away == a]
                    if not exact:  # one unmapped football-data spelling: match on the other side
                        exact = [g for g in cands if (g.home == h) != (g.away == a)
                                 and (g.home == h or g.away == a)]
                    if len(exact) != 1:
                        continue
                    g = exact[0]
                    ph, pa, pd_, _ = devig_three_way(*close)
                    g.sharp = (ph, pa, pd_)
                    if early:
                        eh, ea, ed, _ = devig_three_way(*early)
                        g.sharp_open = (eh, ea, ed)
                    attached += 1
    return attached


def attach_archive(sector: str, games: list[Game], archive_db: Path = ARCHIVE_DB) -> int:
    """Attach the 2026 Pinnacle open (first snapshot) / close (last pre-tip
    snapshot) from archive.db — opened read-only."""
    con = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True)
    rows = con.execute(
        """SELECT event_id, outcome_a_label, outcome_b_label, true_prob_a, true_prob_b,
                  fetched_at, event_date
           FROM archived_sharp_odds
           WHERE sector = ? AND spread_line IS NULL AND total_line IS NULL
             AND prop_player_name IS NULL
             AND (length(event_id) - length(replace(event_id, '::', ''))) = 4
             AND event_date IS NOT NULL AND fetched_at < event_date
           ORDER BY fetched_at""",
        (sector,),
    ).fetchall()
    con.close()
    norm = NameNormalizer(sector)
    first: dict[str, tuple] = {}
    last: dict[str, tuple] = {}
    for eid, la, lb, pa, pb, _f, ed in rows:
        first.setdefault(eid, (la, lb, pa, pb, ed))
        last[eid] = (la, lb, pa, pb, ed)
    idx: dict[tuple[date, str, str], Game] = {}
    for g in games:
        idx[(g.d, g.home, g.away)] = g
    attached = 0
    for eid, (la, lb, pa, pb, ed) in last.items():
        ca, cb = norm.normalize(la), norm.normalize(lb)
        d0 = date.fromisoformat(ed[:10])
        for dd in (d0, d0 - timedelta(days=1), d0 + timedelta(days=1)):
            g = idx.get((dd, ca, cb))
            flip = False
            if g is None:
                g = idx.get((dd, cb, ca))
                flip = g is not None
            if g is None:
                continue
            o = first[eid]
            g.sharp = (pb, pa, None) if flip else (pa, pb, None)
            g.sharp_open = (o[3], o[2], None) if flip else (o[2], o[3], None)
            attached += 1
            break
    return attached


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class _ReplayClock(date):
    _now: date = date(2000, 1, 1)

    @classmethod
    def today(cls) -> date:  # type: ignore[override]
        return cls._now


def make_replay_agent(tracker: ScheduleTracker) -> EloModelAgent:
    """Cold-start production Elo agent whose rest layer reads the replay
    schedule and whose H2H / rest layers are switchable per prediction."""
    agent = EloModelAgent()
    agent._state = {}
    agent.save_state = lambda: None  # type: ignore[method-assign]
    agent.use_h2h = True  # type: ignore[attr-defined]
    agent.use_rest = True  # type: ignore[attr-defined]
    orig_h2h = agent._h2h_adjustment
    orig_rest = agent._rest_elo_bonus
    agent._h2h_adjustment = (  # type: ignore[method-assign]
        lambda s, a, b: orig_h2h(s, a, b) if agent.use_h2h else 0.0
    )
    agent._rest_elo_bonus = (  # type: ignore[method-assign]
        lambda s, t, reference=None: orig_rest(s, t, reference) if agent.use_rest else 0.0
    )
    agent._days_of_rest = (  # type: ignore[method-assign]
        lambda s, t, reference=None: tracker.days_of_rest(t, reference or _ReplayClock._now)
    )
    agent._games_in_last_n_days = (  # type: ignore[method-assign]
        lambda s, t, days=7, reference=None: tracker.games_in_window(t, reference or _ReplayClock._now, days)
    )
    return agent


def _elo_confidence(n: int) -> float:
    if n == 0:
        return 0.3
    if n < LOW_DATA_THRESHOLD:
        return 0.45
    if n < MED_DATA_THRESHOLD:
        return 0.60
    return 0.80


@lru_cache(maxsize=None)
def _sharp_weight(sector: str, league: Optional[str]) -> tuple[float, Optional[tuple]]:
    if sector == "soccer":
        from evmax.sectors.soccer_tiers import disagreement_ramp_for_league, sharp_weight_for_league

        return sharp_weight_for_league(league), disagreement_ramp_for_league(league)
    cfg = json.loads(MODEL_CONFIG.read_text())
    return float(cfg.get("sharp_weight_by_sector", {}).get(sector, cfg.get("sharp_weight", 0.85))), None


@dataclass
class Scored:
    season: int
    outcome: tuple[float, ...]
    d: Optional[date] = None
    home: str = ""
    away: str = ""
    elo: dict[str, tuple[float, ...]] = field(default_factory=dict)
    blend: dict[str, tuple[float, ...]] = field(default_factory=dict)
    sharp: Optional[tuple[float, ...]] = None
    move: Optional[float] = None          # sharp close − open, home side
    fired: dict[str, bool] = field(default_factory=dict)


def replay(sector: str, games: list[Game]) -> list[Scored]:
    cfg = SECTORS[sector]
    three_way = sector == "soccer"
    tracker = ScheduleTracker()
    agent = make_replay_agent(tracker)
    form = FormModelAgent()
    form._state = {}
    form.save_state = lambda: None  # type: ignore[method-assign]
    ens = EnsembleModelAgent(models=[])
    ens._apply_sector_calibration = lambda s, a, b, d: (a, b, d)  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()
    saved_date = elo_mod.date
    elo_mod.date = _ReplayClock  # type: ignore[assignment]
    out: list[Scored] = []
    cur_season: Optional[int] = None
    scored_seasons = set(cfg["eval"]) | {cfg["holdout"]}
    try:
        for g in games:
            if g.season != cur_season:
                if cur_season is not None and cfg.get("regress"):
                    st = agent._sector_state(sector)
                    for t, r in list(st["ratings"].items()):
                        st["ratings"][t] = round(DEFAULT_ELO + cfg["regress"] * (r - DEFAULT_ELO), 2)
                        st["season_games"][t] = 0
                cur_season = g.season
            _ReplayClock._now = g.d
            tie = g.hs == g.as_
            n_min = min(agent._get_count(sector, g.home), agent._get_count(sector, g.away))
            # Score only what live can blend: an Elo prediction with < LOW_DATA_THRESHOLD
            # games on either side has confidence <= 0.45 and is dropped at the
            # ensemble gate. (Without this, NCAAB's one-off non-D1 opponents —
            # no rest history, bonus 0 — let the rest layer act as a
            # "has-a-rating" proxy worth z≈−15.)
            if g.season in scored_seasons and (three_way or not tie) and n_min >= LOW_DATA_THRESHOLD \
                    and agent._last_updated(sector) is not None \
                    and not agent._is_stale(agent._last_updated(sector), g.d):
                outcome = ((1.0 if g.hs > g.as_ else 0.0, 1.0 if g.hs < g.as_ else 0.0, 1.0 if tie else 0.0)
                           if three_way else (1.0 if g.hs > g.as_ else 0.0,))
                rec = Scored(season=g.season, outcome=outcome, d=g.d, home=g.home, away=g.away)
                form_pred = None
                if g.sharp is not None:
                    mkt = SimpleNamespace(sector=sector, team_home=g.home, team_away=g.away,
                                          event_date=datetime(g.d.year, g.d.month, g.d.day))
                    sh = SimpleNamespace(outcome_a_label=g.home, outcome_b_label=g.away, event_id="e",
                                         true_prob_a=g.sharp[0], true_prob_b=g.sharp[1],
                                         true_prob_draw=g.sharp[2], margin=0.03)
                    form_pred = loop.run_until_complete(form.predict_pair(mkt, sh))
                base = None
                for v in VARIANTS:
                    agent.use_h2h = v in ("h2h", "on")
                    agent.use_rest = v in ("rest", "on")
                    pa, pb, pd_ = agent._win_probs(sector, g.home, g.away, g.d)
                    probs = (pa, pb, pd_) if three_way else (pa,)
                    rec.elo[v] = probs
                    if v == "off":
                        base = pa
                    rec.fired[v] = abs(pa - base) > 1e-9
                    if g.sharp is not None:
                        preds = {"elo": ModelAgentPrediction(
                            event_id="e", model_name="elo", true_prob_a=pa, true_prob_b=pb,
                            true_prob_draw=pd_, confidence=_elo_confidence(n_min), weight=agent.weight)}
                        if form_pred is not None:
                            preds["form"] = form_pred
                        sw, ramp = _sharp_weight(sector, g.league)
                        bp = ens._blend("e", preds, sh, sw, sector=sector, disagreement_params=ramp)
                        rec.blend[v] = ((bp.true_prob_a, bp.true_prob_b, bp.true_prob_draw or 0.0)
                                        if three_way else (bp.true_prob_a,))
                if g.sharp is not None:
                    rec.sharp = ((g.sharp[0], g.sharp[1], g.sharp[2] or 0.0) if three_way else (g.sharp[0],))
                    if g.sharp_open is not None:
                        rec.move = g.sharp[0] - g.sharp_open[0]
                out.append(rec)
            agent.update(g.home, g.away, g.hs, g.as_, sector, event_date=g.d.isoformat())
            form.update(g.home, g.away, g.hs, g.as_, sector, g.d.isoformat())
            tracker.record(g.home, g.d)
            tracker.record(g.away, g.d)
    finally:
        elo_mod.date = saved_date  # type: ignore[assignment]
        loop.close()
    return out


def kalshi_clv_lens(sector: str, recs: list[Scored], predictions_db: Path = PREDICTIONS_DB) -> dict:
    """Does the feature's Elo shift predict logged rows' Kalshi CLV?

    Rows come from the same ``_fetch_clv_rows`` behind ``evmax cleanup shadow
    clv`` (resolved, non-contaminated, pre-tip, Kalshi moneyline), read from a
    snapshot of the live DB (the source is opened read-only; the copy absorbs
    ``get_connection``'s migrations). For each row the backed team is the YES
    team (the other side on ``:no`` rows); x = the variant's shift of that
    team's Elo win probability (pp), y = the row's ``kalshi_clv_pct``. Every
    logged row was priced with these features dead, so a positive slope means
    the feature points where the Kalshi price later moved.
    """
    import tempfile

    import evmax.agents.cleanup.db as cleanup_db
    from evmax.cli.commands.shadow import _fetch_clv_rows

    if not predictions_db.exists():
        return {}
    tmp = Path(tempfile.mkdtemp()) / "predictions.db"
    src = sqlite3.connect(f"file:{predictions_db}?mode=ro", uri=True)
    dst = sqlite3.connect(tmp)
    src.backup(dst)
    dst.close()
    src.close()
    saved = cleanup_db.DB_PATH
    cleanup_db.DB_PATH = tmp
    try:
        rows, _ = _fetch_clv_rows(sector, market_type="moneyline", venue="kalshi")
        with cleanup_db.get_connection() as conn:
            yes_team = dict(conn.execute(
                "SELECT market_id, yes_team FROM ev_predictions WHERE sector = ?", (sector,)
            ).fetchall())
    finally:
        cleanup_db.DB_PATH = saved
    norm = NameNormalizer(sector)
    by_game: dict[tuple[date, str, str], Scored] = {}
    for r in recs:
        by_game[(r.d, r.home, r.away)] = r
    out: dict = {"n_rows": len(rows)}
    joined: list[tuple[Scored, bool, float]] = []
    for row in rows:
        try:
            d0 = date.fromisoformat(row["event_id"].split("::")[1])
        except (IndexError, ValueError, AttributeError):
            continue
        team = norm.normalize(yes_team.get(row["market_id"]) or "")
        if not team:
            continue
        is_no = row["market_id"].endswith(":no")
        rec = None
        for dd in (d0, d0 - timedelta(days=1), d0 + timedelta(days=1)):
            rec = next((by_game[k] for k in by_game if k[0] == dd and team in (k[1], k[2])), None)
            if rec is not None:
                break
        if rec is None:
            continue
        backs_home = (team == rec.home) != is_no
        joined.append((rec, backs_home, float(row["kalshi_clv_pct"])))
    out["n_joined"] = len(joined)
    for v in VARIANTS[1:]:
        x = [100 * (r.elo[v][0] - r.elo["off"][0]) * (1 if home else -1) for r, home, _ in joined]
        y = [clv for _, _, clv in joined]
        slope, t, n = ols_slope(x, y)
        pos = [c for xi, c in zip(x, y) if xi > 1e-9]
        neg = [c for xi, c in zip(x, y) if xi < -1e-9]
        out[v] = {"slope": slope, "t": t, "n_shifted": len(pos) + len(neg),
                  "clv_when_up": sum(pos) / len(pos) if pos else float("nan"),
                  "clv_when_down": sum(neg) / len(neg) if neg else float("nan")}
    return out


COVERAGE_SECTORS = ("nba", "nfl", "ncaab", "ncaaw", "ncaaf", "soccer", "worldcup", "baseball", "wnba", "nhl")


def coverage_table(archive_db: Path = ARCHIVE_DB) -> list[dict]:
    """How many archived Pinnacle moneyline labels miss the H2H / rest reads.

    Read-only on archive.db and the committed model state. For every distinct
    (outcome_a_label, outcome_b_label) moneyline pair per sector:
      rest   — labels whose raw lowercased form is a form_state key, vs labels
               resolve_team_key finds in form_state.
      h2h    — pairs with a >= H2H_MIN_GAMES record under the raw labels, vs
               under the keys the labels resolve to in the Elo ratings store.
    """
    from evmax.agents.models._team_lookup import resolve_team_key
    from evmax.agents.models.elo_agent import FORM_STATE_PATH, H2H_MIN_GAMES, REST_ELO_ADJ

    elo = json.loads((_REPO_ROOT / "data" / "models" / "elo_state.json").read_text())
    form = json.loads(FORM_STATE_PATH.read_text())
    con = sqlite3.connect(f"file:{archive_db}?mode=ro", uri=True)
    out = []
    for sector in COVERAGE_SECTORS:
        pairs = con.execute(
            """SELECT DISTINCT lower(trim(outcome_a_label)), lower(trim(outcome_b_label))
               FROM archived_sharp_odds
               WHERE sector = ? AND spread_line IS NULL AND total_line IS NULL
                 AND prop_player_name IS NULL
                 AND (length(event_id) - length(replace(event_id, '::', ''))) = 4
                 AND outcome_a_label IS NOT NULL AND outcome_b_label IS NOT NULL""",
            (sector,),
        ).fetchall()
        labels = sorted({x for pr in pairs for x in pr if x})
        st = elo.get(sector, {})
        h2h = st.get("h2h", {})
        fs = form.get(sector, {})

        def key(lab: str) -> Optional[str]:
            return (resolve_team_key(sector, lab, st.get("ratings", {}))
                    or resolve_team_key(sector, lab, st.get("game_counts", {})))

        def fires(a: Optional[str], b: Optional[str]) -> bool:
            if not a or not b:
                return False
            rec = h2h.get(f"{a}::{b}" if a <= b else f"{b}::{a}")
            return rec is not None and rec["games"] >= H2H_MIN_GAMES

        out.append({
            "sector": sector, "labels": len(labels), "rest_table": sector in REST_ELO_ADJ,
            "rest_raw": sum(lab in fs for lab in labels),
            "rest_resolved": sum(bool(resolve_team_key(sector, lab, fs)) for lab in labels),
            "pairs": len(pairs),
            "h2h_raw": sum(fires(a, b) for a, b in pairs),
            "h2h_resolved": sum(fires(key(a), key(b)) for a, b in pairs),
        })
    con.close()
    return out


def print_coverage(rows: list[dict]) -> None:
    print(f"  {'sector':9} {'labels':>6} {'rest tbl':>8} {'rest raw':>8} {'rest res':>8} | "
          f"{'pairs':>6} {'h2h raw':>8} {'h2h res':>8}")
    for r in rows:
        print(f"  {r['sector']:9} {r['labels']:>6} {str(r['rest_table']):>8} {r['rest_raw']:>8} "
              f"{r['rest_resolved']:>8} | {r['pairs']:>6} {r['h2h_raw']:>8} {r['h2h_resolved']:>8}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summarize(sector: str, recs: list[Scored]) -> dict:
    cfg = SECTORS[sector]
    groups = {"eval": [r for r in recs if r.season in cfg["eval"]],
              "holdout": [r for r in recs if r.season == cfg["holdout"]]}
    for s in sorted({r.season for r in recs}):
        groups[str(s)] = [r for r in recs if r.season == s]
    res: dict = {"sector": sector, "groups": {}}
    for name, rs in groups.items():
        g: dict = {"n": len(rs)}
        if rs:
            b_off = [brier(r.elo["off"], r.outcome) for r in rs]
            g["elo_off_brier"] = sum(b_off) / len(b_off)
            for v in VARIANTS[1:]:
                b_v = [brier(r.elo[v], r.outcome) for r in rs]
                m, se, z, n = paired_delta(b_v, b_off)
                fired = sum(r.fired[v] for r in rs)
                shift = [abs(r.elo[v][0] - r.elo["off"][0]) for r in rs if r.fired[v]]
                g[v] = {"d_elo": m, "se": se, "z": z, "fire_rate": fired / len(rs),
                        "mean_abs_shift_pp": 100 * sum(shift) / len(shift) if shift else 0.0}
            rb = [r for r in rs if r.blend]
            g["n_blend"] = len(rb)
            if rb:
                bb_off = [brier(r.blend["off"], r.outcome) for r in rb]
                g["blend_off_brier"] = sum(bb_off) / len(bb_off)
                g["sharp_brier"] = sum(brier(r.sharp, r.outcome) for r in rb) / len(rb)
                for v in VARIANTS[1:]:
                    m, se, z, n = paired_delta([brier(r.blend[v], r.outcome) for r in rb], bb_off)
                    g[v].update({"d_blend": m, "se_blend": se, "z_blend": z})
            rm = [r for r in rs if r.move is not None]
            g["n_clv"] = len(rm)
            for v in VARIANTS[1:]:
                x = [r.elo[v][0] - r.elo["off"][0] for r in rm]
                y = [r.move for r in rm]
                slope, t, n = ols_slope(x, y)
                g[v].update({"clv_slope": slope, "clv_t": t})
        res["groups"][name] = g
    ev, ho = res["groups"]["eval"], res["groups"]["holdout"]
    verdict = {}
    for v in VARIANTS[1:]:
        if not ev.get("n") or not ho.get("n"):
            verdict[v] = "NO-DATA"
            continue
        if ev[v]["fire_rate"] == 0 and ho[v]["fire_rate"] == 0:
            verdict[v] = "NEVER-FIRES"
            continue
        a = ev[v]["d_elo"] < 0 and ev[v]["z"] <= -1.64
        b = ho[v]["d_elo"] <= 0
        c = ho.get("n_blend", 0) == 0 or ho[v].get("d_blend", 0.0) <= 0
        verdict[v] = "SHIP" if (a and b and c) else "REJECT"
    res["verdict"] = verdict
    return res


def _fmt(x: float, nd: int = 2) -> str:
    return "   nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def print_report(res: dict) -> None:
    s = res["sector"]
    print(f"\n{'=' * 104}\n  {s.upper()}   verdict: " + ", ".join(f"{k}={v}" for k, v in res["verdict"].items()))
    print(f"{'=' * 104}")
    print(f"  {'window':8} {'n':>6} {'Elo off':>8} | {'var':4} {'fire%':>6} {'|Δp|pp':>7} "
          f"{'ΔElo/1k':>8} {'z':>6} | {'nBl':>5} {'ΔBlend/1k':>9} {'z':>6} | {'nCLV':>5} {'slope':>7} {'t':>6}")
    for name, g in res["groups"].items():
        if not g.get("n"):
            continue
        for i, v in enumerate(VARIANTS[1:]):
            d = g[v]
            head = f"  {name:8} {g['n']:>6} {g['elo_off_brier']:>8.4f}" if i == 0 else f"  {'':8} {'':>6} {'':>8}"
            print(f"{head} | {v:4} {100 * d['fire_rate']:>6.1f} {d['mean_abs_shift_pp']:>7.2f} "
                  f"{_fmt(d['d_elo'])[:8]:>8} {_fmt(d['z'])[:6]:>6} | {g.get('n_blend', 0):>5} "
                  f"{_fmt(d.get('d_blend', float('nan')), 3):>9} {_fmt(d.get('z_blend', float('nan'))):>6} | "
                  f"{g.get('n_clv', 0):>5} {_fmt(d.get('clv_slope', float('nan')), 3):>7} "
                  f"{_fmt(d.get('clv_t', float('nan'))):>6}")
    k = res.get("kalshi_clv")
    if k:
        print(f"  Kalshi CLV lens (evmax cleanup shadow clv rows): {k['n_rows']} rows, {k['n_joined']} joined")
        for v in VARIANTS[1:]:
            d = k[v]
            print(f"    {v:4} shifted={d['n_shifted']:>4}  slope={_fmt(d['slope'], 3)} t={_fmt(d['t'])}  "
                  f"CLV when feature backs the side: {_fmt(d['clv_when_up'])}pp  "
                  f"when it fades it: {_fmt(d['clv_when_down'])}pp")


def main(argv: Optional[list[str]] = None) -> int:
    import logging

    import structlog

    # The replay calls update() tens of thousands of times; the agents'
    # per-game debug events would dominate the runtime and bury the report.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-s", "--sectors", default=",".join(SECTORS))
    ap.add_argument("--json", default=None, help="write the summary JSON here")
    ap.add_argument("--archive-db", type=Path, default=ARCHIVE_DB, help="archive.db (opened read-only)")
    ap.add_argument("--predictions-db", type=Path, default=PREDICTIONS_DB,
                    help="predictions.db for --kalshi-clv (snapshotted; the source is opened read-only)")
    ap.add_argument("--kalshi-clv", action="store_true",
                    help="also run the Kalshi CLV lens on logged moneyline rows (reads a snapshot of predictions.db)")
    ap.add_argument("--coverage", action="store_true",
                    help="print how many archived Pinnacle labels miss the H2H / rest reads, then exit")
    args = ap.parse_args(argv)
    if args.coverage:
        rows = coverage_table(args.archive_db)
        print_coverage(rows)
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=2))
        return 0
    results = []
    for sector in [x.strip() for x in args.sectors.split(",") if x.strip()]:
        print(f"\n[{sector}] loading games…", flush=True)
        games = load_nfl_games() if sector == "nfl" else load_espn_games(sector)
        if sector == "soccer":
            print(f"  football-data sharp attached: {attach_soccer_csv(games)}")
        elif SECTORS[sector].get("archive"):
            print(f"  archive.db sharp attached: {attach_archive(sector, games, args.archive_db)}")
        print(f"  games: {len(games)}  seasons: {sorted({g.season for g in games})}", flush=True)
        recs = replay(sector, games)
        res = summarize(sector, recs)
        if args.kalshi_clv:
            res["kalshi_clv"] = kalshi_clv_lens(sector, recs, args.predictions_db)
        print_report(res)
        results.append(res)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
