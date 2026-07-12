"""Seed NCAAB / NCAAW opponent-adjusted efficiency state from ESPN box scores.

Walks a full college basketball season (Nov -> mid-Apr) day by day on ESPN's
public scoreboard (groups=50 = Division I), pulls per-game box scores from the
summary endpoint, derives Dean Oliver possession-based per-game efficiencies,
runs the ITERATIVE OPPONENT ADJUSTMENT solver (KenPom-style core — the whole
point in a 360-team league with wildly unequal schedules), and writes the
per-league state file:

    data/models/ncaab_efficiency_state.json    (--league mens, default)
    data/models/ncaaw_efficiency_state.json    (--league womens)

Raw per-game extracted stats are cached to disk one JSON per day under
data/backtest/college_efficiency/{league}/{season}/ so re-runs (and the
walk-forward backtest, which imports `load_season_games` from here) are free.

USAGE
    python scripts/seed_ncaab_efficiency.py                          # mens, current/most recent season
    python scripts/seed_ncaab_efficiency.py --league womens --season 2025
    python scripts/seed_ncaab_efficiency.py --season 2024 --fetch-only   # populate cache only
    python scripts/seed_ncaab_efficiency.py --dry-run                # compute, don't write

Seasons are labeled by START year: --season 2025 = the 2025-26 season.

Re-run cadence in production: weekly during Nov-Apr (same as the WNBA seed).
The agent's update() is a no-op — box-score detail can't be reconstructed
from a score pair — so this script is the only path data flows in. The
staleness guard in the agents silences prior-season state once a new season
starts, so an autumn re-seed is MANDATORY (the WNBA +24pp chalk-bias lesson).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evmax.agents.models._college_efficiency import (
    MIN_D1_APPEARANCES,
    dean_oliver_possessions,
    filter_d1_games,
    opponent_adjust,
)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball"

LEAGUE_CONFIG: dict[str, dict] = {
    "mens": {
        "espn_league": "mens-college-basketball",
        "sector": "ncaab",
        "state_file": "ncaab_efficiency_state.json",
    },
    "womens": {
        "espn_league": "womens-college-basketball",
        "sector": "ncaaw",
        "state_file": "ncaaw_efficiency_state.json",
    },
}

CACHE_ROOT = REPO_ROOT / "data" / "backtest" / "college_efficiency"
STATE_DIR = REPO_ROOT / "data" / "models"

# D1 group id on ESPN's college basketball APIs (same for men + women).
ESPN_D1_GROUP = "50"

MAX_CONCURRENT_SUMMARIES = 8
RETRIES = 3


def season_days(season: int, today: Optional[date] = None) -> list[date]:
    """All calendar days of a season: Nov 1 (start year) -> Apr 15 (next year).

    Clipped at `today` so an in-season re-seed only walks played days.
    """
    today = today or date.today()
    start = date(season, 11, 1)
    end = min(date(season + 1, 4, 15), today)
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def _parse_made_attempted(s: str) -> tuple[int, int]:
    """'27-63' -> (27, 63); bad input -> (0, 0)."""
    try:
        made, att = s.split("-")
        return int(made), int(att)
    except (ValueError, AttributeError):
        return 0, 0


def _extract_team_stats(team_entry: dict) -> Optional[dict]:
    """Pull FGM/FGA/3PM/FTM/FTA/OREB/DREB/TOV from one boxscore team entry."""
    stats_list = team_entry.get("statistics", [])
    s = {item.get("name", ""): item.get("displayValue", "") for item in stats_list}

    fgm, fga = _parse_made_attempted(s.get("fieldGoalsMade-fieldGoalsAttempted", ""))
    tpm, _ = _parse_made_attempted(s.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", ""))
    ftm, fta = _parse_made_attempted(s.get("freeThrowsMade-freeThrowsAttempted", ""))

    def _int(key: str) -> int:
        try:
            return int(s.get(key, "0") or 0)
        except ValueError:
            return 0

    oreb = _int("offensiveRebounds")
    dreb = _int("defensiveRebounds")
    tov = _int("totalTurnovers") or _int("turnovers")

    if fga == 0:
        return None
    return {
        "fgm": fgm, "fga": fga, "tpm": tpm,
        "ftm": ftm, "fta": fta,
        "oreb": oreb, "dreb": dreb, "tov": tov,
    }


async def _get_json(client: httpx.AsyncClient, url: str, params: dict) -> Optional[dict]:
    for attempt in range(RETRIES):
        try:
            r = await client.get(url, params=params, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == RETRIES - 1:
                return None
            await asyncio.sleep(1.0 + attempt)
    return None


async def _fetch_day(
    client: httpx.AsyncClient,
    espn_league: str,
    day: date,
    sem: asyncio.Semaphore,
) -> list[dict]:
    """Fetch one day's completed D1 games with extracted box stats."""
    sb = await _get_json(
        client,
        f"{ESPN_BASE}/{espn_league}/scoreboard",
        {"dates": day.strftime("%Y%m%d"), "groups": ESPN_D1_GROUP, "limit": 500},
    )
    if sb is None:
        print(f"    ! scoreboard fetch failed for {day}", file=sys.stderr)
        return []

    shells: list[dict] = []
    for e in sb.get("events", []):
        try:
            comp = e["competitions"][0]
        except (KeyError, IndexError):
            continue
        if comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
            continue
        home = next((t for t in comp.get("competitors", []) if t.get("homeAway") == "home"), None)
        away = next((t for t in comp.get("competitors", []) if t.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        def _side(t: dict) -> dict:
            team = t.get("team", {})
            return {
                "id": str(team.get("id", "")),
                "location": team.get("location", ""),
                "display": team.get("displayName", ""),
                "abbr": team.get("abbreviation", ""),
                "score": int(t.get("score", 0) or 0),
            }

        shells.append({
            "event_id": str(e.get("id", "")),
            "date": day.isoformat(),
            "neutral": bool(comp.get("neutralSite", False)),
            "home": _side(home),
            "away": _side(away),
        })

    async def _fill(shell: dict) -> Optional[dict]:
        async with sem:
            summary = await _get_json(
                client,
                f"{ESPN_BASE}/{espn_league}/summary",
                {"event": shell["event_id"]},
            )
        if summary is None:
            return None
        bs_teams = summary.get("boxscore", {}).get("teams", [])
        if len(bs_teams) != 2:
            return None
        bs_home = next((t for t in bs_teams if t.get("homeAway") == "home"), None)
        bs_away = next((t for t in bs_teams if t.get("homeAway") == "away"), None)
        if not bs_home or not bs_away:
            return None
        hs = _extract_team_stats(bs_home)
        as_ = _extract_team_stats(bs_away)
        if not hs or not as_:
            return None
        shell["home"].update(hs)
        shell["away"].update(as_)
        return shell

    filled = await asyncio.gather(*(_fill(s) for s in shells))
    return [g for g in filled if g is not None]


def _cache_dir(league: str, season: int) -> Path:
    return CACHE_ROOT / league / str(season)


async def _fetch_season(league: str, season: int) -> None:
    """Populate the day-file cache for one league-season (skips cached days)."""
    cfg = LEAGUE_CONFIG[league]
    cdir = _cache_dir(league, season)
    cdir.mkdir(parents=True, exist_ok=True)

    days = season_days(season)
    todo = [d for d in days if not (cdir / f"{d.strftime('%Y%m%d')}.json").exists()]
    print(f"[{league} {season}] {len(days)} days, {len(todo)} to fetch")
    if not todo:
        return

    sem = asyncio.Semaphore(MAX_CONCURRENT_SUMMARIES)
    async with httpx.AsyncClient() as client:
        for i, day in enumerate(todo):
            games = await _fetch_day(client, cfg["espn_league"], day, sem)
            (cdir / f"{day.strftime('%Y%m%d')}.json").write_text(
                json.dumps({"date": day.isoformat(), "games": games})
            )
            if (i + 1) % 20 == 0 or i == len(todo) - 1:
                print(f"  [{league} {season}] {i + 1}/{len(todo)} days ({day}, {len(games)} games)")


def load_season_games(league: str, season: int, fetch_missing: bool = True) -> list[dict]:
    """Return all cached game rows for a league-season, sorted by date.

    When `fetch_missing` and the cache is incomplete, fetches the gaps first.
    This is the entry point the walk-forward backtest uses.
    """
    if fetch_missing:
        asyncio.run(_fetch_season(league, season))
    cdir = _cache_dir(league, season)
    games: list[dict] = []
    if cdir.exists():
        for f in sorted(cdir.glob("*.json")):
            try:
                games.extend(json.loads(f.read_text()).get("games", []))
            except json.JSONDecodeError:
                print(f"    ! corrupt cache file {f}, skipping", file=sys.stderr)
    games.sort(key=lambda g: (g["date"], g["event_id"]))
    return games


# ---------------------------------------------------------------------------
# State building
# ---------------------------------------------------------------------------


def to_solver_rows(games: list[dict]) -> list[dict]:
    """Convert cached game rows -> opponent_adjust() input rows (keyed by ESPN id)."""
    rows = []
    for g in games:
        h, a = g["home"], g["away"]
        h_poss = dean_oliver_possessions(h["fga"], h["fta"], h["tov"], h["oreb"])
        a_poss = dean_oliver_possessions(a["fga"], a["fta"], a["tov"], a["oreb"])
        if h_poss <= 0 or a_poss <= 0:
            continue
        rows.append({
            "home_id": h["id"], "away_id": a["id"],
            "neutral": g["neutral"],
            "home_pts": h["score"], "away_pts": a["score"],
            "home_poss": h_poss, "away_poss": a_poss,
        })
    return rows


def build_state(
    games: list[dict],
    season: int,
    sector: str,
    min_appearances: int = MIN_D1_APPEARANCES,
) -> dict:
    """Aggregate + opponent-adjust one season of cached games into agent state.

    Team keys are NameNormalizer-normalized ESPN `location` strings ("duke",
    "michigan state") — the shape Pinnacle labels arrive in — with the full
    display name stored per team for the resolver's fallback matching.
    """
    from evmax.matching.normalizer import NameNormalizer

    solver_rows = to_solver_rows(games)
    d1_rows, d1_ids = filter_d1_games(solver_rows, min_appearances)
    if not d1_rows:
        raise RuntimeError("no D1 games after filtering — aborting")

    solved = opponent_adjust(d1_rows)
    if not solved["converged"]:
        print(f"    ! solver did not converge in {solved['iterations']} iterations",
              file=sys.stderr)

    # Four-factors totals per team over D1 games only.
    d1_game_ok = {(r["home_id"], r["away_id"]) for r in d1_rows}
    totals: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for g in games:
        h, a = g["home"], g["away"]
        if (h["id"], a["id"]) not in d1_game_ok:
            continue
        for own, opp in ((h, a), (a, h)):
            t = totals.setdefault(own["id"], {
                "fgm": 0, "fga": 0, "tpm": 0, "fta": 0,
                "oreb": 0, "tov": 0, "opp_dreb": 0, "poss": 0.0,
            })
            t["fgm"] += own["fgm"]; t["fga"] += own["fga"]; t["tpm"] += own["tpm"]
            t["fta"] += own["fta"]; t["oreb"] += own["oreb"]; t["tov"] += own["tov"]
            t["opp_dreb"] += opp["dreb"]
            t["poss"] += dean_oliver_possessions(own["fga"], own["fta"], own["tov"], own["oreb"])
            meta[own["id"]] = {
                "location": own["location"], "display": own["display"], "abbr": own["abbr"],
            }

    norm = NameNormalizer(sector)
    teams: dict[str, dict] = {}
    for tid, solved_stats in solved["teams"].items():
        m = meta.get(tid)
        t = totals.get(tid)
        if not m or not t:
            continue
        key = norm.normalize(m["location"]) or m["location"].lower()
        if key in teams:
            # Location collision (rare) — disambiguate with the abbreviation.
            key = f"{key} {m['abbr'].lower()}".strip()
        entry = {
            "ortg": solved_stats["adj_o"],
            "drtg": solved_stats["adj_d"],
            "raw_ortg": solved_stats["raw_o"],
            "raw_drtg": solved_stats["raw_d"],
            "pace": solved_stats["pace"],
            "net": round(solved_stats["adj_o"] - solved_stats["adj_d"], 3),
            "gp": solved_stats["gp"],
            "efg": round((t["fgm"] + 0.5 * t["tpm"]) / t["fga"], 4) if t["fga"] else 0.0,
            "tov_pct": round(t["tov"] / t["poss"], 4) if t["poss"] > 0 else 0.0,
            "oreb_pct": round(t["oreb"] / max(t["oreb"] + t["opp_dreb"], 1), 4),
            "ftr": round(t["fta"] / t["fga"], 4) if t["fga"] else 0.0,
            "full_name": m["display"].lower(),
            "team_id": tid,
        }
        teams[key] = entry

    tovs = [v["tov_pct"] for v in teams.values() if v["tov_pct"] > 0]
    return {
        "league_avg_ortg": solved["league_avg_eff"],
        "league_avg_drtg": solved["league_avg_eff"],
        "league_avg_pace": solved["league_avg_pace"],
        "league_avg_tov_pct": round(sum(tovs) / len(tovs), 4) if tovs else 0.17,
        "hca_eff": solved["hca_eff"],
        "teams": teams,
        "fetched_at": date.today().isoformat(),
        "source_season": str(season),
        "n_games_aggregated": len(d1_rows),
        "solver_iterations": solved["iterations"],
        "solver_converged": solved["converged"],
    }


def print_summary(state: dict, top: int = 15) -> None:
    print()
    print(f"League avg eff={state['league_avg_ortg']}  pace={state['league_avg_pace']}  "
          f"HCA={state['hca_eff']} pts/100  ({state['n_games_aggregated']} D1 games, "
          f"{len(state['teams'])} teams, solver {state['solver_iterations']} iters)")
    rows = sorted(state["teams"].items(), key=lambda kv: -kv[1]["net"])
    header = f"{'team':<24} {'gp':>4} {'adjO':>7} {'adjD':>7} {'net':>7} {'pace':>6}"
    print(header)
    print("-" * len(header))
    for key, t in rows[:top]:
        print(f"{key:<24} {t['gp']:>4} {t['ortg']:>7.1f} {t['drtg']:>7.1f} "
              f"{t['net']:>+7.1f} {t['pace']:>6.1f}")


def default_season(today: Optional[date] = None) -> int:
    """Most recent season with data: start year if Nov+, else previous year."""
    today = today or date.today()
    return today.year if today.month >= 11 else today.year - 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--league", choices=list(LEAGUE_CONFIG), default="mens")
    p.add_argument("--season", type=int, default=None,
                   help="Season START year (2025 = the 2025-26 season). "
                        "Default: most recent season.")
    p.add_argument("--fetch-only", action="store_true",
                   help="Populate the day-file cache and exit (no state write)")
    p.add_argument("--dry-run", action="store_true", help="Compute but do not write state")
    p.add_argument("--state", type=Path, default=None, help="Override output path")
    args = p.parse_args()

    season = args.season if args.season is not None else default_season()
    cfg = LEAGUE_CONFIG[args.league]

    print(f"Seeding {cfg['sector']} efficiency from ESPN {season}-{(season + 1) % 100:02d} box scores")
    games = load_season_games(args.league, season, fetch_missing=True)
    print(f"Loaded {len(games)} completed games from cache")
    if args.fetch_only:
        return 0

    state = build_state(games, season, cfg["sector"])
    print_summary(state)

    if args.dry_run:
        print("\ndry-run — no files written")
        return 0

    out = args.state or (STATE_DIR / cfg["state_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = out.with_suffix(f".backup.{stamp}.json")
        shutil.copy2(out, backup)
        print(f"\nBackup of existing state → {backup}")
    out.write_text(json.dumps(state, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
