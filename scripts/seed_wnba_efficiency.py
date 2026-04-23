"""Seed WNBA efficiency state from ESPN box scores.

Walks the 2025 WNBA season (or any --year you pass), pulls per-game box
scores from ESPN's public summary endpoint, derives advanced stats using
Dean Oliver formulas, aggregates per team, and writes to
data/models/wnba_efficiency_state.json.

This is the only path by which the WNBA efficiency agent gets data —
the agent's `update()` is a no-op (game-level box totals can't be
reconstructed from just a score pair). Re-run this script weekly once
the 2026 season is live to refresh.

USAGE
    python scripts/seed_wnba_efficiency.py                 # 2025 full season
    python scripts/seed_wnba_efficiency.py --year 2024     # replay 2024
    python scripts/seed_wnba_efficiency.py --dry-run       # don't write state

Derived stats per team:
    ORtg   = 100 × PTS / POSS
    DRtg   = 100 × OPP_PTS / OPP_POSS
    Pace   = POSS per 40 min (WNBA game length)
    eFG%   = (FGM + 0.5·3PM) / FGA
    TOV%   = TO / POSS
    OREB%  = own_OREB / (own_OREB + opp_DREB)
    FTr    = FTA / FGA

Possessions (Dean Oliver):
    POSS = FGA + 0.44·FTA + TO − OREB
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "data" / "models" / "wnba_efficiency_state.json"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
REQUEST_DELAY = 0.08   # seconds between summary fetches — polite pacing

# Only the 15 real WNBA franchises go into state. ESPN's WNBA scoreboard
# also carries the annual All-Star Game (teams named "Team Clark" / "Team
# Collier") and occasional international exhibitions ("Team Brazil" /
# "Team Antelopes"). Those rows have tiny game counts and extreme ORtg
# values that would skew league averages — filter them out by canonical
# team key rather than a games-played threshold (expansion teams have
# zero games but must still survive for future updates).
REAL_WNBA_TEAMS = {
    "dream", "sky", "sun", "wings", "fever", "aces", "sparks",
    "lynx", "liberty", "mercury", "storm", "mystics",
    "valkyries", "fire", "tempo",
}

# WNBA regular season + playoffs typically run May-October.
# We pull May-October 2025 (season ran mid-May through mid-October 2025).
DEFAULT_MONTHS: dict[int, list[str]] = {
    2024: [f"2024{m:02d}" for m in range(5, 11)],
    2025: [f"2025{m:02d}" for m in range(5, 11)],
    2026: [f"2026{m:02d}" for m in range(5, 11)],
}


@dataclass
class TeamSeasonTotals:
    """Running totals for one team across a season."""
    full_name: str = ""
    gp: int = 0
    # Offensive totals
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    three_pm: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    tov: int = 0
    # Opponent totals (accumulated when team is on either side)
    opp_pts: int = 0
    opp_fga: int = 0
    opp_fta: int = 0
    opp_oreb: int = 0
    opp_dreb: int = 0
    opp_tov: int = 0


def _parse_made_attempted(s: str) -> tuple[int, int]:
    """'27-63' → (27, 63); handles bad inputs by returning (0, 0)."""
    try:
        made, att = s.split("-")
        return int(made), int(att)
    except (ValueError, AttributeError):
        return 0, 0


def _canonical_team_key(display_name: str) -> str:
    """ESPN display name → short canonical key (e.g. 'Atlanta Dream' → 'dream').

    Every WNBA team has a unique last word across the 15 franchises
    (Dream/Sky/Sun/Wings/Fever/Aces/Sparks/Lynx/Liberty/Mercury/Storm/
    Mystics/Valkyries/Fire/Tempo) so last-word reduction is unambiguous.
    """
    if not display_name:
        return ""
    return display_name.strip().split()[-1].lower()


def _extract_team_stats(team_entry: dict) -> Optional[dict]:
    """Pull FGA/FTA/TO/OREB/DREB/3PM/FGM etc. from one boxscore team entry."""
    stats_list = team_entry.get("statistics", [])
    s: dict[str, str] = {item.get("name", ""): item.get("displayValue", "") for item in stats_list}

    fgm, fga = _parse_made_attempted(s.get("fieldGoalsMade-fieldGoalsAttempted", ""))
    tpm, _tpa = _parse_made_attempted(s.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", ""))
    ftm, fta = _parse_made_attempted(s.get("freeThrowsMade-freeThrowsAttempted", ""))

    def _int(key: str) -> int:
        try:
            return int(s.get(key, "0") or 0)
        except ValueError:
            return 0

    oreb = _int("offensiveRebounds")
    dreb = _int("defensiveRebounds")
    # 'totalTurnovers' is team-level canonical; 'turnovers' is player-coded.
    # Use totalTurnovers when present, else fall back to turnovers.
    tov = _int("totalTurnovers") or _int("turnovers")

    if fga == 0:
        # Empty stat block — skip this team
        return None

    # Points derived from shot made totals: 2·FGM + 3PM + FTM
    # (FGM includes 3PM, so 2-pointers contribute 2·(FGM−3PM), threes contribute 3·3PM)
    pts = 2 * fgm + tpm + ftm

    return {
        "fgm": fgm, "fga": fga,
        "three_pm": tpm,
        "ftm": ftm, "fta": fta,
        "oreb": oreb, "dreb": dreb,
        "tov": tov,
        "pts": pts,
    }


def _accumulate_game(
    totals: dict[str, TeamSeasonTotals],
    home_name: str, home_stats: dict, home_score: int,
    away_name: str, away_stats: dict, away_score: int,
) -> None:
    """Update both teams' running totals from one completed game."""
    home_key = _canonical_team_key(home_name)
    away_key = _canonical_team_key(away_name)

    for (key, full, own, opp, own_score, opp_score) in [
        (home_key, home_name, home_stats, away_stats, home_score, away_score),
        (away_key, away_name, away_stats, home_stats, away_score, home_score),
    ]:
        if not key or not own or not opp:
            continue
        t = totals.setdefault(key, TeamSeasonTotals(full_name=full.lower()))
        if not t.full_name:
            t.full_name = full.lower()
        t.gp += 1
        # Prefer scoreboard score over derived points (final truth)
        t.pts += int(own_score) if own_score else own["pts"]
        t.fgm += own["fgm"]
        t.fga += own["fga"]
        t.three_pm += own["three_pm"]
        t.ftm += own["ftm"]
        t.fta += own["fta"]
        t.oreb += own["oreb"]
        t.dreb += own["dreb"]
        t.tov += own["tov"]
        t.opp_pts += int(opp_score) if opp_score else opp["pts"]
        t.opp_fga += opp["fga"]
        t.opp_fta += opp["fta"]
        t.opp_oreb += opp["oreb"]
        t.opp_dreb += opp["dreb"]
        t.opp_tov += opp["tov"]


def _derive_advanced(t: TeamSeasonTotals) -> dict:
    """Per-team season-average advanced stats from accumulated totals."""
    # Possessions — Dean Oliver (FGA + 0.44·FTA + TO − OREB), averaged per game
    own_poss = t.fga + 0.44 * t.fta + t.tov - t.oreb
    opp_poss = t.opp_fga + 0.44 * t.opp_fta + t.opp_tov - t.opp_oreb

    if own_poss <= 0 or opp_poss <= 0 or t.gp == 0:
        return {}

    ortg = 100.0 * t.pts / own_poss
    drtg = 100.0 * t.opp_pts / opp_poss
    # WNBA regulation game = 40 min
    pace = own_poss / t.gp
    efg = (t.fgm + 0.5 * t.three_pm) / t.fga if t.fga > 0 else 0.0
    tov_pct = t.tov / own_poss if own_poss > 0 else 0.0
    # OREB% = own OREB / (own OREB + opp DREB) — standard Four Factors definition
    oreb_pct = t.oreb / max(t.oreb + t.opp_dreb, 1)
    ftr = t.fta / t.fga if t.fga > 0 else 0.0

    return {
        "ortg": round(ortg, 2),
        "drtg": round(drtg, 2),
        "pace": round(pace, 2),
        "net": round(ortg - drtg, 2),
        "efg": round(efg, 4),
        "tov_pct": round(tov_pct, 4),
        "oreb_pct": round(oreb_pct, 4),
        "ftr": round(ftr, 4),
        "gp": t.gp,
        "full_name": t.full_name,
    }


def _fetch_month_events(client: httpx.Client, month: str) -> list[dict]:
    """Fetch ESPN scoreboard for a YYYYMM month; return completed events."""
    url = f"{ESPN_BASE}/scoreboard"
    r = client.get(url, params={"dates": month, "limit": 200}, timeout=20)
    r.raise_for_status()
    out = []
    for e in r.json().get("events", []):
        comp = e["competitions"][0]
        if comp["status"]["type"]["name"] != "STATUS_FINAL":
            continue
        out.append(e)
    return out


def _fetch_summary(client: httpx.Client, event_id: str) -> Optional[dict]:
    """Fetch ESPN summary (box score) for one game."""
    try:
        r = client.get(f"{ESPN_BASE}/summary", params={"event": event_id}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ! summary fetch failed for {event_id}: {e}", file=sys.stderr)
        return None


def seed(year: int, months: Optional[list[str]] = None) -> dict:
    months = months or DEFAULT_MONTHS.get(year, [])
    if not months:
        raise ValueError(f"No default months configured for year {year}")

    totals: dict[str, TeamSeasonTotals] = {}
    n_games = 0
    n_skipped = 0

    with httpx.Client() as client:
        for month in months:
            print(f"Fetching scoreboard for {month} …")
            events = _fetch_month_events(client, month)
            print(f"  {len(events)} completed games")

            for i, e in enumerate(events):
                event_id = e.get("id", "")
                comp = e["competitions"][0]
                teams = comp.get("competitors", [])
                home_t = next((t for t in teams if t.get("homeAway") == "home"), None)
                away_t = next((t for t in teams if t.get("homeAway") == "away"), None)
                if not home_t or not away_t:
                    n_skipped += 1
                    continue

                home_name = home_t["team"].get("displayName", "")
                away_name = away_t["team"].get("displayName", "")
                home_score = int(home_t.get("score", 0) or 0)
                away_score = int(away_t.get("score", 0) or 0)

                summary = _fetch_summary(client, event_id)
                time.sleep(REQUEST_DELAY)
                if not summary:
                    n_skipped += 1
                    continue

                bs = summary.get("boxscore", {})
                bs_teams = bs.get("teams", [])
                if len(bs_teams) != 2:
                    n_skipped += 1
                    continue

                bs_home = next((t for t in bs_teams if t.get("homeAway") == "home"), None)
                bs_away = next((t for t in bs_teams if t.get("homeAway") == "away"), None)
                if not bs_home or not bs_away:
                    n_skipped += 1
                    continue

                home_stats = _extract_team_stats(bs_home)
                away_stats = _extract_team_stats(bs_away)
                if not home_stats or not away_stats:
                    n_skipped += 1
                    continue

                _accumulate_game(
                    totals,
                    home_name, home_stats, home_score,
                    away_name, away_stats, away_score,
                )
                n_games += 1

                if (i + 1) % 25 == 0:
                    print(f"    processed {i+1}/{len(events)}")

    print(f"\nAggregated {n_games} games across {len(totals)} teams (skipped {n_skipped})")

    team_stats = {key: _derive_advanced(t) for key, t in totals.items()}
    team_stats = {k: v for k, v in team_stats.items() if v}

    # Drop All-Star / exhibition contamination before computing league averages
    filtered_out = {k for k in team_stats if k not in REAL_WNBA_TEAMS}
    if filtered_out:
        print(f"Filtering non-WNBA rows: {sorted(filtered_out)}")
    team_stats = {k: v for k, v in team_stats.items() if k in REAL_WNBA_TEAMS}

    if not team_stats:
        raise RuntimeError("No teams produced valid stats — aborting")

    league_ortg = sum(v["ortg"] for v in team_stats.values()) / len(team_stats)
    league_drtg = sum(v["drtg"] for v in team_stats.values()) / len(team_stats)
    league_pace = sum(v["pace"] for v in team_stats.values()) / len(team_stats)

    state = {
        "league_avg_ortg": round(league_ortg, 2),
        "league_avg_drtg": round(league_drtg, 2),
        "league_avg_pace": round(league_pace, 2),
        "teams": team_stats,
        "fetched_at": date.today().isoformat(),
        "source_season": str(year),
        "n_games_aggregated": n_games,
    }
    return state


def print_summary(state: dict) -> None:
    print()
    print(f"League avg  ORtg={state['league_avg_ortg']}  DRtg={state['league_avg_drtg']}  "
          f"Pace={state['league_avg_pace']}")
    print()
    teams = state["teams"]
    rows = sorted(teams.items(), key=lambda kv: -kv[1]["net"])
    header = f"{'team':<12} {'gp':>4} {'ortg':>7} {'drtg':>7} {'net':>6} {'pace':>6} {'efg':>6} {'tov%':>6} {'orb%':>6} {'ftr':>6}"
    print(header)
    print("-" * len(header))
    for key, t in rows:
        print(
            f"{key:<12} {t['gp']:>4} {t['ortg']:>7.1f} {t['drtg']:>7.1f} "
            f"{t['net']:>+6.1f} {t['pace']:>6.1f} "
            f"{t['efg']:>6.3f} {t['tov_pct']:>6.3f} {t['oreb_pct']:>6.3f} {t['ftr']:>6.3f}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
    p.add_argument("--dry-run", action="store_true", help="Compute but do not write state file")
    p.add_argument("--state", type=Path, default=STATE_PATH,
                   help=f"Output path (default: {STATE_PATH})")
    args = p.parse_args()

    print(f"Seeding WNBA efficiency state from {args.year} ESPN box scores")
    state = seed(args.year)
    print_summary(state)

    if args.dry_run:
        print("\ndry-run — no files written")
        return 0

    args.state.parent.mkdir(parents=True, exist_ok=True)
    if args.state.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = args.state.with_suffix(f".backup.{stamp}.json")
        shutil.copy2(args.state, backup)
        print(f"\nBackup of existing state → {backup}")

    args.state.write_text(json.dumps(state, indent=2))
    print(f"Wrote {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
