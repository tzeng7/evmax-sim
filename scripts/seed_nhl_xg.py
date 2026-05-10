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

Usage:
    python scripts/seed_nhl_xg.py                  # current season
    python scripts/seed_nhl_xg.py --season 2024    # specific season
    python scripts/seed_nhl_xg.py --dry-run        # print summary, don't write

Season identifier convention: MoneyPuck uses the year the season STARTED.
So the 2025-26 NHL season is `--season 2025`, the 2024-25 season is
`--season 2024`. (Confirmed by reading the season selector on
moneypuck.com/teams.htm: <option value='2025'>2025-2026</option>.)
Default is the most recently-started season inferred from today's date.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import httpx

from evmax.agents.models.nhl_xg_agent import (
    NHL_ABBREV_TO_NAME,
    NHL_MONEYPUCK_ABBREV_ALIASES,
)

STATE_PATH = _REPO_ROOT / "data" / "models" / "nhl_xg_state.json"

MONEYPUCK_TEAMS_CSV = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary/"
    "{season}/regular/teams.csv"
)
SITUATION = "5on5"


def _current_nhl_season_start_year() -> int:
    """Return the START year of the most recent NHL season.

    NHL regular season runs ~October → April. From October onward we're in
    a season that started this year. From January through September we're
    either still in (Jan-Apr) or just past (May-Sep) a season that started
    last year. Either way, the most recently-started season's CSV is at
    `seasonSummary/{year-1}` for months 1-9 and `seasonSummary/{year}`
    once October rolls around.
    """
    today = date.today()
    return today.year if today.month >= 10 else today.year - 1


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


def compute_team_stats(rows: list[dict]) -> dict[str, dict]:
    """Filter to 5v5 and project to {full_name: {xgf_per_60, xga_per_60, ...}}.

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


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed NHL xG state from MoneyPuck")
    ap.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season STARTING year (e.g. 2025 for 2025-26). Default: current.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print summary, don't write")
    args = ap.parse_args()

    season = args.season if args.season is not None else _current_nhl_season_start_year()
    print(f"Fetching MoneyPuck teams.csv for {season}-{str(season + 1)[-2:]} season "
          f"(situation={SITUATION})...")

    try:
        rows = fetch_moneypuck_teams_csv(season)
    except httpx.HTTPError as e:
        print(f"ERROR: MoneyPuck fetch failed: {e}", file=sys.stderr)
        return 1

    print(f"  {len(rows)} CSV rows total")
    teams, league_avg = compute_team_stats(rows)
    if not teams:
        print(
            "ERROR: no teams matched — MoneyPuck schema may have changed or "
            "season may not have started yet",
            file=sys.stderr,
        )
        return 1

    state = {
        "nhl": {
            "league_avg_xg_per_60": league_avg,
            "teams": teams,
            "fetched_at": date.today().isoformat(),
            "season_start_year": season,
            "situation": SITUATION,
            "source": "moneypuck",
        }
    }

    # Sanity-check sample: top + bottom 5 by net xG/60
    ranked = sorted(
        teams.items(),
        key=lambda kv: kv[1]["xgf_per_60"] - kv[1]["xga_per_60"],
        reverse=True,
    )
    print(f"\nLeague avg xG/60 at 5v5: {league_avg:.2f}")
    print(f"\nTop 5 by net xG/60:")
    for name, s in ranked[:5]:
        net = s["xgf_per_60"] - s["xga_per_60"]
        print(f"  {name:25s} net={net:+.2f}  "
              f"(xGF={s['xgf_per_60']:.2f} xGA={s['xga_per_60']:.2f}) gp={s['gp']}")
    print(f"\nBottom 5:")
    for name, s in ranked[-5:]:
        net = s["xgf_per_60"] - s["xga_per_60"]
        print(f"  {name:25s} net={net:+.2f}  "
              f"(xGF={s['xgf_per_60']:.2f} xGA={s['xga_per_60']:.2f}) gp={s['gp']}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(teams)} teams to {STATE_PATH}")
        return 0

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nWrote {len(teams)} teams to {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
