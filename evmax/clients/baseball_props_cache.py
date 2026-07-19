"""MLB player-prop data layer — season rates, team rates, box scores, game logs.

Feeds the projection model (evmax/models_ml/baseball_props.py) and the
resolver/backtest. Three concerns:

  1. **Live profiles (cached)** — one daily refresh pulls league-wide
     season-to-date hitting + pitching + team-hitting from the MLB Stats API
     and writes ``data/baseball_props_cache.json``. ``get_hitter_profile`` /
     ``get_pitcher_profile`` / ``get_team_k_rate`` return typed profiles keyed
     by the same normalized name the props pipeline matches on.

  2. **Resolution** — ``fetch_player_box_stats(date)`` returns each player's
     actual batting + pitching line for a date (computed composites: total
     bases, hits+runs+RBIs), so prop outcomes can be settled.

  3. **Backtest** — ``fetch_game_logs`` pulls a player's per-game series so the
     walk-forward can reconstruct point-in-time rates AND read the outcome from
     the same source.

All network access is async + best-effort: failures degrade to empty so the
coordinator falls back to sharp-anchor-only pricing.
"""
from __future__ import annotations

import json
import time
from datetime import date as _date
from pathlib import Path
from typing import Any, Optional

import structlog

from evmax.clients.mlb_statsapi import MLBStatsClient
from evmax.models_ml.baseball_props import HitterProfile, PitcherProfile
from evmax.players import normalize_player_name

logger = structlog.get_logger(__name__)

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "baseball_props_cache.json"
_CACHE_TTL_SECS = 12 * 60 * 60  # half a day — season totals move slowly

# In-process memo of the parsed cache.
_mem: Optional[dict[str, Any]] = None
_mem_ts: float = 0.0


def _ip_to_outs(innings_pitched: Any) -> float:
    """MLB 'inningsPitched' is a string like '99.2' = 99 innings + 2 outs."""
    try:
        s = str(innings_pitched)
        whole, _, frac = s.partition(".")
        return int(whole or 0) * 3 + int(frac or 0)
    except (ValueError, TypeError):
        return 0.0


def _f(v: Any) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Season-rate fetch + cache
# ---------------------------------------------------------------------------

async def _fetch_league_stats(client: MLBStatsClient, group: str, season: int) -> list[dict]:
    """All players' season totals for a stat group (hitting|pitching)."""
    data = await client._get(
        "/stats",
        params={
            "stats": "season", "group": group, "season": season,
            "sportId": 1, "playerPool": "all", "limit": 3000,
        },
    )
    stats = data.get("stats", [])
    return stats[0].get("splits", []) if stats else []


async def _fetch_team_hitting(
    client: MLBStatsClient, season: int
) -> tuple[dict[int, float], dict[int, dict]]:
    """One /teams/stats hitting call → (k_rates, offense).

    k_rates:  team_id → season strikeout rate (SO / PA), the opponent-K adj.
    offense:  team_id → {"runs_per_game", "games"} — the moneyline model's
              team-offense input (v1 assumed every lineup scores at the
              league-average rate; the opposing starter's rate is now scaled
              by this, clamped at the agent).
    """
    data = await client._get(
        "/teams/stats",
        params={"stats": "season", "group": "hitting", "season": season, "sportIds": 1},
    )
    stats = data.get("stats", [])
    splits = stats[0].get("splits", []) if stats else []
    k_rates: dict[int, float] = {}
    offense: dict[int, dict] = {}
    for sp in splits:
        tid = sp.get("team", {}).get("id")
        st = sp.get("stat", {})
        if not tid:
            continue
        pa = _f(st.get("plateAppearances"))
        so = _f(st.get("strikeOuts"))
        if pa > 0:
            k_rates[int(tid)] = so / pa
        games = _f(st.get("gamesPlayed"))
        runs = _f(st.get("runs"))
        if games > 0:
            offense[int(tid)] = {
                "runs_per_game": runs / games,
                "games": int(games),
            }
    return k_rates, offense


def _build_cache(hit_splits: list[dict], pit_splits: list[dict],
                 team_k: dict[int, float], season: int,
                 team_offense: Optional[dict[int, dict]] = None) -> dict[str, Any]:
    hitters: dict[str, dict] = {}
    for sp in hit_splits:
        player = sp.get("player", {})
        name = player.get("fullName")
        if not name:
            continue
        st = sp.get("stat", {})
        key = normalize_player_name(name, "baseball")
        hitters[key] = {
            "mlb_id": player.get("id"),
            "team_id": (sp.get("team") or {}).get("id"),
            "plate_appearances": _f(st.get("plateAppearances")),
            "at_bats": _f(st.get("atBats")),
            "hits": _f(st.get("hits")),
            "doubles": _f(st.get("doubles")),
            "triples": _f(st.get("triples")),
            "home_runs": _f(st.get("homeRuns")),
            "rbi": _f(st.get("rbi")),
            "runs": _f(st.get("runs")),
            "strike_outs": _f(st.get("strikeOuts")),
            "total_bases": _f(st.get("totalBases")),
            "games": _f(st.get("gamesPlayed")),
        }
    pitchers: dict[str, dict] = {}
    for sp in pit_splits:
        player = sp.get("player", {})
        name = player.get("fullName")
        if not name:
            continue
        st = sp.get("stat", {})
        key = normalize_player_name(name, "baseball")
        pitchers[key] = {
            "mlb_id": player.get("id"),
            "team_id": (sp.get("team") or {}).get("id"),
            "batters_faced": _f(st.get("battersFaced")),
            "outs": _f(st.get("outs")) or _ip_to_outs(st.get("inningsPitched")),
            "strike_outs": _f(st.get("strikeOuts")),
            "home_runs": _f(st.get("homeRuns")),
            "hits": _f(st.get("hits")),
            "games_started": _f(st.get("gamesStarted")),
            "innings_pitched": _f(st.get("inningsPitched")),
        }
    return {
        "season": season,
        "fetched_at": time.time(),
        "hitters": hitters,
        "pitchers": pitchers,
        "team_k_rate": {str(k): v for k, v in team_k.items()},
        "team_offense": {str(k): v for k, v in (team_offense or {}).items()},
    }


async def refresh_baseball_props_cache(force: bool = False, season: Optional[int] = None) -> int:
    """Refresh the on-disk season-rate cache. Returns players cached (hitters+pitchers)."""
    global _mem, _mem_ts
    season = season or _date.today().year
    if not force and _CACHE_PATH.exists():
        age = time.time() - _CACHE_PATH.stat().st_mtime
        if age < _CACHE_TTL_SECS:
            return 0
    try:
        async with MLBStatsClient() as client:
            hit = await _fetch_league_stats(client, "hitting", season)
            pit = await _fetch_league_stats(client, "pitching", season)
            team_k, team_offense = await _fetch_team_hitting(client, season)
        cache = _build_cache(hit, pit, team_k, season, team_offense)
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache))
        _mem = cache
        _mem_ts = time.time()
        n = len(cache["hitters"]) + len(cache["pitchers"])
        logger.info("baseball_props_cache_refreshed", hitters=len(cache["hitters"]),
                    pitchers=len(cache["pitchers"]), season=season)
        return n
    except Exception as e:  # noqa: BLE001 — degrade to sharp-only pricing
        logger.warning("baseball_props_cache_refresh_failed", error=str(e))
        return 0


def is_cache_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    return (time.time() - _CACHE_PATH.stat().st_mtime) < _CACHE_TTL_SECS


def _load() -> dict[str, Any]:
    global _mem, _mem_ts
    if _mem is not None and (time.time() - _mem_ts) < _CACHE_TTL_SECS:
        return _mem
    if _CACHE_PATH.exists():
        try:
            _mem = json.loads(_CACHE_PATH.read_text())
            _mem_ts = time.time()
            return _mem
        except (json.JSONDecodeError, OSError):
            pass
    return {"hitters": {}, "pitchers": {}, "team_k_rate": {}}


def get_hitter_profile(player_norm: str) -> Optional[HitterProfile]:
    d = _load()["hitters"].get(player_norm)
    if not d or d.get("plate_appearances", 0) <= 0:
        return None
    return HitterProfile(
        plate_appearances=d["plate_appearances"], at_bats=d["at_bats"], hits=d["hits"],
        doubles=d["doubles"], triples=d["triples"], home_runs=d["home_runs"],
        rbi=d["rbi"], runs=d["runs"], strike_outs=d["strike_outs"],
        total_bases=d["total_bases"], games=d["games"],
    )


def get_pitcher_profile(player_norm: str) -> Optional[PitcherProfile]:
    d = _load()["pitchers"].get(player_norm)
    if not d or d.get("batters_faced", 0) <= 0:
        return None
    return PitcherProfile(
        batters_faced=d["batters_faced"], outs=d["outs"], strike_outs=d["strike_outs"],
        home_runs=d["home_runs"], hits=d["hits"], games_started=d["games_started"],
        innings_pitched=d["innings_pitched"],
    )


def get_team_id_for_player(player_norm: str) -> Optional[int]:
    c = _load()
    d = c["hitters"].get(player_norm) or c["pitchers"].get(player_norm)
    tid = d.get("team_id") if d else None
    return int(tid) if tid else None


def get_team_k_rate(team_id: int) -> Optional[float]:
    return _load()["team_k_rate"].get(str(team_id))


def get_team_offense(team_id: int) -> Optional[dict]:
    """Season-to-date {"runs_per_game", "games"} for a team, or None.

    Legacy caches predate the team_offense block — .get keeps them safe
    until the next 12h refresh.
    """
    return _load().get("team_offense", {}).get(str(team_id))


def league_runs_per_game() -> Optional[float]:
    """League-average runs/game across all cached teams (offense baseline)."""
    offense = _load().get("team_offense", {})
    vals = [v.get("runs_per_game") for v in offense.values() if v.get("runs_per_game")]
    if not vals:
        return None
    return sum(vals) / len(vals)


def hitter_games(player_norm: str) -> int:
    d = _load()["hitters"].get(player_norm)
    return int(d.get("games", 0)) if d else 0


def pitcher_starts(player_norm: str) -> int:
    d = _load()["pitchers"].get(player_norm)
    return int(d.get("games_started", 0)) if d else 0


# ---------------------------------------------------------------------------
# Box scores (resolution + backtest outcomes)
# ---------------------------------------------------------------------------

def _player_box_line(batting: dict, pitching: dict) -> dict[str, float]:
    """Flatten an MLB boxscore player into the canonical prop stat outcomes."""
    bats = batting or {}
    pit = pitching or {}
    hits = _f(bats.get("hits"))
    doubles = _f(bats.get("doubles"))
    triples = _f(bats.get("triples"))
    hr = _f(bats.get("homeRuns"))
    rbi = _f(bats.get("rbi"))
    runs = _f(bats.get("runs"))
    total_bases = (hits - doubles - triples - hr) + 2 * doubles + 3 * triples + 4 * hr
    return {
        # hitter
        "hits": hits,
        "home_runs": hr,
        "total_bases": total_bases,
        "rbis": rbi,
        "hits_runs_rbis": hits + runs + rbi,
        # pitcher
        "strikeouts": _f(pit.get("strikeOuts")),
        "pitching_outs": _ip_to_outs(pit.get("inningsPitched")) if pit else 0.0,
    }


async def fetch_player_box_stats(game_date: str) -> dict[str, dict[str, float]]:
    """``{normalized_player_name: {stat_type: actual_value}}`` for all games on a date.

    Pitcher ``strikeouts``/``pitching_outs`` come from the pitching line; hitter
    stats from the batting line. A two-way player (Ohtani) merges both. Returns
    {} on any error so the resolver can skip and retry later.
    """
    out: dict[str, dict[str, float]] = {}
    try:
        async with MLBStatsClient() as client:
            sched = await client.schedule(date=game_date, hydrate="")
            game_pks = [
                g.get("gamePk")
                for db in sched.get("dates", [])
                for g in db.get("games", [])
                if g.get("status", {}).get("abstractGameState") == "Final"
            ]
            for pk in game_pks:
                if not pk:
                    continue
                box = await client._get(f"/game/{pk}/boxscore")
                for side in ("home", "away"):
                    players = (box.get("teams", {}).get(side, {}) or {}).get("players", {})
                    for p in players.values():
                        name = (p.get("person") or {}).get("fullName")
                        if not name:
                            continue
                        stats = p.get("stats", {})
                        line = _player_box_line(stats.get("batting", {}), stats.get("pitching", {}))
                        key = normalize_player_name(name, "baseball")
                        # merge (two-way players appear once with both lines)
                        cur = out.setdefault(key, {})
                        for k, v in line.items():
                            cur[k] = v
    except Exception as e:  # noqa: BLE001
        logger.warning("mlb_box_stats_failed", date=game_date, error=str(e))
        return {}
    return out


async def fetch_game_logs(player_id: int, season: int, group: str) -> list[dict]:
    """Per-game stat splits for a player (group='hitting'|'pitching'), chronological.

    Used by the walk-forward backtest to reconstruct point-in-time rates and to
    read the target-game outcome from the same source.
    """
    try:
        async with MLBStatsClient() as client:
            data = await client._get(
                f"/people/{player_id}/stats",
                params={"stats": "gameLog", "season": season, "group": group, "sportId": 1},
            )
        stats = data.get("stats", [])
        return stats[0].get("splits", []) if stats else []
    except Exception as e:  # noqa: BLE001
        logger.warning("mlb_game_logs_failed", player_id=player_id, error=str(e))
        return []
