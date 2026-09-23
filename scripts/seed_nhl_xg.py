"""Seed nhl_xg_state.json from MoneyPuck team CSVs.

Pulls per-team 5v5 stats from MoneyPuck's public season-summary CSV
(https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv),
filters to situation=5on5, and writes per-team xGF/60 and xGA/60 (the agent's
primary signal) plus actual goal rates (gf_per_60, ga_per_60) as a secondary
fallback.

MoneyPuck uses dotted abbreviations for two-word cities (L.A, N.J, S.J, T.B);
this script normalizes them to standard 3-letter codes (LAK, NJD, SJS, TBL)
before keying state, matching NHL_ABBREV_TO_NAME in the agent.

There is no incremental update path because per-game xG cannot be reconstructed
from just a final score — re-run weekly to refresh.

Prior block (2026-09-22): every seed ALSO fetches the PREVIOUS season's CSV
and writes it as a `prior` block of regressed rates
(lg + PRIOR_REGRESS_RHO·(rate − lg), lg = that season's league average). The
preseason-prior ramp in NhlXgModelAgent blends the prior with the in-season
rates by games played. Before a season's first game MoneyPuck 404s the new
season's CSV. The seed then writes a PRIOR-ONLY state (season_start_year = the
new season, empty `teams`, full `prior`), so the model fires on opening night
instead of going dark. Any other fetch failure (including the prior season)
aborts WITHOUT writing — a failed reseed must never regress the state file.

Usage:
    python scripts/seed_nhl_xg.py                  # current season
    python scripts/seed_nhl_xg.py --season 2025    # specific season (+ its 2024 prior)
    python scripts/seed_nhl_xg.py --dry-run        # print summary, don't write

Season identifier convention: MoneyPuck uses the year the season STARTED.
So the 2026-27 NHL season is `--season 2026`, the 2025-26 season is
`--season 2025`. (Confirmed by reading the season selector on
moneypuck.com/teams.htm: <option value='2025'>2025-2026</option>.)
Default is the season of the next/current game. It rolls over on
1 September, because the 2026-27 opener is 2026-09-29.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import httpx

from evmax.agents.models.nhl_xg_agent import (
    NHL_ABBREV_TO_NAME,
    NHL_MONEYPUCK_ABBREV_ALIASES,
    PRIOR_REGRESS_RHO,
    SEASON_ROLLOVER_MONTH,
    regress_prior_rate,
)

STATE_PATH = _REPO_ROOT / "data" / "models" / "nhl_xg_state.json"

MONEYPUCK_TEAMS_CSV = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary/"
    "{season}/regular/teams.csv"
)
SITUATION = "5on5"
SCHEMA_VERSION = 2
# A prior with fewer teams means a broken fetch or a schema change, not a league.
MIN_PRIOR_TEAMS = 30


def _current_nhl_season_start_year(today: Optional[date] = None) -> int:
    """Return the START year of the season the next/current NHL game belongs to.

    Rolls over on 1 September (SEASON_ROLLOVER_MONTH), not October: the
    2026-27 regular season opens 2026-09-29, and the week before the opener is
    exactly when the prior-only state has to be written. Before a season's
    first game its MoneyPuck CSV 404s and the seed falls back to prior-only.
    """
    today = today or date.today()
    return today.year if today.month >= SEASON_ROLLOVER_MONTH else today.year - 1


def fetch_moneypuck_teams_csv(season: int) -> list[dict]:
    """Fetch and parse MoneyPuck's per-team season-summary CSV."""
    url = MONEYPUCK_TEAMS_CSV.format(season=season)
    headers = {"User-Agent": "evmax-seeder/1.0"}
    r = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    return list(reader)


def normalize_abbrev(raw: str) -> str:
    """Normalize MoneyPuck's dotted abbrevs to standard 3-letter codes."""
    raw = raw.strip().upper()
    return NHL_MONEYPUCK_ABBREV_ALIASES.get(raw, raw)


def compute_team_stats(rows: list[dict]) -> tuple[dict[str, dict], float]:
    """Filter to 5v5 and project to ({full_name: {xgf_per_60, ...}}, league_avg).

    MoneyPuck columns used:
      team                  (3-letter abbrev, possibly dotted)
      situation             (filter to 5on5)
      games_played          (gp)
      iceTime               (TOI in seconds at this situation)
      xGoalsFor             (cumulative xG for at this situation)
      xGoalsAgainst         (cumulative xG against at this situation)
      goalsFor              (cumulative goals for)
      goalsAgainst          (cumulative goals against)

    We compute /60 rates by dividing the cumulative totals by (iceTime / 3600).
    """
    out: dict[str, dict] = {}
    league_xg_for = 0.0
    league_xg_against = 0.0
    league_minutes = 0.0

    for row in rows:
        if (row.get("situation") or "").strip() != SITUATION:
            continue
        abbrev = normalize_abbrev(row.get("team", ""))
        full = NHL_ABBREV_TO_NAME.get(abbrev)
        if full is None:
            # Skip All-Star teams, defunct franchises, etc.
            continue

        try:
            gp = int(float(row.get("games_played", 0) or 0))
            ice_seconds = float(row.get("iceTime", 0) or 0)
            xgf = float(row.get("xGoalsFor", 0) or 0)
            xga = float(row.get("xGoalsAgainst", 0) or 0)
            gf = float(row.get("goalsFor", 0) or 0)
            ga = float(row.get("goalsAgainst", 0) or 0)
        except (ValueError, TypeError):
            continue

        if ice_seconds <= 0:
            continue

        # Convert cumulative stats to per-60-min rates
        ice_hours = ice_seconds / 3600.0
        xgf_per_60 = xgf / ice_hours
        xga_per_60 = xga / ice_hours
        gf_per_60 = gf / ice_hours
        ga_per_60 = ga / ice_hours

        out[full] = {
            "abbrev": abbrev,
            "xgf_per_60": round(xgf_per_60, 4),
            "xga_per_60": round(xga_per_60, 4),
            "gf_per_60": round(gf_per_60, 4),
            "ga_per_60": round(ga_per_60, 4),
            "gp": gp,
        }

        league_xg_for += xgf
        league_xg_against += xga
        league_minutes += ice_hours

    league_avg = (
        round((league_xg_for + league_xg_against) / (2 * league_minutes), 4)
        if league_minutes > 0
        else 2.50
    )
    return out, league_avg


class SeasonNotPublished(Exception):
    """MoneyPuck has no CSV for this season yet (HTTP 404 / empty) — pre-opener."""


def fetch_season_teams(season: int) -> tuple[dict[str, dict], float]:
    """Fetch + parse one season. Raises SeasonNotPublished on a 404 or an empty CSV."""
    try:
        rows = fetch_moneypuck_teams_csv(season)
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 404:
            raise SeasonNotPublished(season) from e
        raise
    if not rows:
        raise SeasonNotPublished(season)
    return compute_team_stats(rows)


def build_prior_block(prior_teams: dict[str, dict], prior_league_avg: float, season: int) -> dict:
    """Regress each prior-season rate toward that season's league average.

    `xgf_per_60` / `xga_per_60` hold the REGRESSED rates the agent reads; the
    raw rates are kept beside them for diagnostics.
    """
    teams = {}
    for name, st in prior_teams.items():
        teams[name] = {
            "abbrev": st["abbrev"],
            "xgf_per_60": round(regress_prior_rate(st["xgf_per_60"], prior_league_avg), 4),
            "xga_per_60": round(regress_prior_rate(st["xga_per_60"], prior_league_avg), 4),
            "raw_xgf_per_60": st["xgf_per_60"],
            "raw_xga_per_60": st["xga_per_60"],
            "gp": st["gp"],
        }
    return {
        "season_start_year": season,
        "league_avg_xg_per_60": prior_league_avg,
        "regress_rho": PRIOR_REGRESS_RHO,
        "teams": teams,
    }


def build_state(season: int, today: Optional[date] = None) -> dict:
    """Build the v2 state for `season`: in-season block + regressed prior block.

    Raises on any fetch/parse failure EXCEPT the current season's 404, which
    yields a prior-only state (empty `teams`, league_avg None).
    """
    try:
        prior_teams, prior_lg = fetch_season_teams(season - 1)
    except SeasonNotPublished as e:
        raise RuntimeError(f"prior season {season - 1} is not published on MoneyPuck") from e
    if len(prior_teams) < MIN_PRIOR_TEAMS:
        raise RuntimeError(
            f"prior season {season - 1} parsed only {len(prior_teams)} teams "
            f"(need >= {MIN_PRIOR_TEAMS}) — MoneyPuck schema change?"
        )

    teams: dict[str, dict]
    league_avg: Optional[float]
    try:
        teams, league_avg = fetch_season_teams(season)
        mode = "in_season"
    except SeasonNotPublished:
        teams, league_avg, mode = {}, None, "prior_only"
    else:
        if not teams:
            raise RuntimeError(
                f"season {season} CSV had rows but no team matched — MoneyPuck schema change?"
            )

    return {
        "nhl": {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "league_avg_xg_per_60": league_avg,
            "teams": teams,
            "prior": build_prior_block(prior_teams, prior_lg, season - 1),
            "fetched_at": (today or date.today()).isoformat(),
            "season_start_year": season,
            "situation": SITUATION,
            "source": "moneypuck",
        }
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed NHL xG state from MoneyPuck")
    ap.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season STARTING year (e.g. 2026 for 2026-27). Default: current.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print summary, don't write")
    args = ap.parse_args()

    season = args.season if args.season is not None else _current_nhl_season_start_year()
    print(f"Fetching MoneyPuck teams.csv for {season}-{str(season + 1)[-2:]} "
          f"(+ prior {season - 1}-{str(season)[-2:]}), situation={SITUATION}...")

    try:
        state = build_state(season)
    except (httpx.HTTPError, RuntimeError) as e:
        print(f"ERROR: seed aborted, state NOT written: {e!r}", file=sys.stderr)
        return 1

    blk = state["nhl"]
    teams = blk["teams"]
    prior = blk["prior"]
    print(f"\nmode = {blk['mode']}  in-season teams = {len(teams)}  "
          f"prior teams = {len(prior['teams'])} (season {prior['season_start_year']}, "
          f"lg {prior['league_avg_xg_per_60']:.4f}, rho {prior['regress_rho']})")
    if blk["mode"] == "prior_only":
        print(f"  MoneyPuck has no {season} CSV yet — writing a PRIOR-ONLY state; "
              f"the agent prices every team off its regressed {season - 1} rates.")

    show = teams or prior["teams"]
    label = "in-season" if teams else "regressed prior"
    ranked = sorted(
        show.items(),
        key=lambda kv: kv[1]["xgf_per_60"] - kv[1]["xga_per_60"],
        reverse=True,
    )
    print(f"\nTop 5 by net xG/60 ({label}):")
    for name, s in ranked[:5]:
        net = s["xgf_per_60"] - s["xga_per_60"]
        print(f"  {name:25s} net={net:+.2f}  "
              f"(xGF={s['xgf_per_60']:.2f} xGA={s['xga_per_60']:.2f}) gp={s['gp']}")
    print("\nBottom 5:")
    for name, s in ranked[-5:]:
        net = s["xgf_per_60"] - s["xga_per_60"]
        print(f"  {name:25s} net={net:+.2f}  "
              f"(xGF={s['xgf_per_60']:.2f} xGA={s['xga_per_60']:.2f}) gp={s['gp']}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {STATE_PATH}")
        return 0

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nWrote {STATE_PATH} ({blk['mode']}: {len(teams)} in-season, "
          f"{len(prior['teams'])} prior teams)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
