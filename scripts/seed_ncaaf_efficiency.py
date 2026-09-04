#!/usr/bin/env python3
"""Seed data/models/ncaaf_efficiency_state.json — opponent-adjusted EPA + success
rate ratings, a regressed preseason prior, and a FROZEN preseason FPI prior,
keyed by canonical (normalized ESPN location) team name.

State layout (schema_version 2)::

    {"ncaaf": {
       "schema_version": 2,
       "teams": {
         "ohio state": {
           "off_epa_adj": 0.18, "def_epa_adj": -0.20,     # in-season EPA/play, league-relative
           "off_sr_adj": 0.05, "def_sr_adj": -0.04,       # in-season success rate, league-relative
           "off_epa_prior": 0.09, "def_epa_prior": -0.06,  # regressed prior-season EPA
           "off_sr_prior": 0.02, "def_sr_prior": -0.02,    # regressed prior-season success rate
           "fpi_prior": 21.3,                             # preseason FPI, points vs FBS mean (or absent)
           "gp": 9, "off_success_rate": 0.49, "def_success_rate": 0.40,
           "abbrev": "OSU"
         }, ...
       },
       "league_mean_epa": 0.01, "hfa_epa": 0.05,
       "source_season": 2026, "seasons_used": [2026],
       "prior_season": 2025, "prior_regress": 0.5,
       "fpi_season": 2026, "fpi_source": "live", "fpi_frozen_at": "2026-08-20", "fpi_teams": 136,
       "fetched_at": "2026-08-20"
    }}

The agent (evmax/agents/models/ncaaf_efficiency_agent.py, ``ncaaf_efficiency_v2``)
applies the in-season → prior RAMP w(gp)=gp/(gp+k) at predict time, so week 0–3
games lean on the prior and week 9+ games lean on the opponent-adjusted in-season
ratings. The EPA prior is mixed 50/50 with ``fpi_prior`` (converted to EPA/play)
when present — the SP+/FPI shrinkage pattern the CFB market relies on.

FPI FREEZE: the first seed of a season fetches ESPN's live FPI
(evmax/clients/cfb_fpi.py) and every later weekly seed REUSES those values from
the existing state (``fpi_season == season``). A mid-season refetch would leak
ESPN's in-season updates into what the backtest validated as a PRESEASON prior.
Pass --refresh-fpi to refetch deliberately, --no-fpi to skip the network.

Run weekly during the season (like the NFL/college-basketball reseeds); the
ESPN summaries are disk-cached by game id so a weekly refresh only fetches the
new week's games.

    python scripts/seed_ncaaf_efficiency.py                        # current season
    python scripts/seed_ncaaf_efficiency.py --season 2024          # a past season
    python scripts/seed_ncaaf_efficiency.py --regress 0.5 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

# Make the package importable when run as a script from a worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.agents.models import _cfb_efficiency as E  # noqa: E402
from evmax.clients import cfb_espn as C  # noqa: E402
from evmax.clients import cfb_fpi as F  # noqa: E402

STATE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "models" / "ncaaf_efficiency_state.json"
)

# A team is treated as FBS if it appears in at least this many FBS-slate
# (groups=80) games in the season. FCS buy-game opponents appear 1–3 times;
# real FBS teams play 12–13. Full-season separation is clean.
FBS_MIN_APPEARANCES = 6


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"['’‘]", "", s)
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fbs_universe(games: list[dict]) -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Return (fbs_ids, id→canonical_name, id→abbrev) from a season's games."""
    appear: dict[str, int] = defaultdict(int)
    name: dict[str, str] = {}
    abbr: dict[str, str] = {}
    for g in games:
        for s in ("home", "away"):
            tid = g[s]["id"]
            appear[tid] += 1
            name[tid] = _norm(g[s].get("location") or "")
            abbr[tid] = g[s].get("abbr") or ""
    fbs = {tid for tid, n in appear.items() if n >= FBS_MIN_APPEARANCES and name.get(tid)}
    return fbs, name, abbr


def _season_ratings(season: int, ep_table: dict | None, ridge: float,
                    in_season: bool = False):
    """Fetch a season, build FBS universe + opponent-adjusted ratings.

    If ep_table is None, build it from this season's own plays (structural,
    leaguewide — not a team-level leak). Returns (ratings, id→name, id→abbr,
    ep_table, n_games).

    ``in_season=True`` derives the FBS universe from the FULL SCHEDULE
    (completed + upcoming games) instead of completed games only. Early in a
    season no team has reached FBS_MIN_APPEARANCES completed games, so a
    completed-only universe is EMPTY and every team is pooled as FCS — the
    in-season ratings then never populate (gp stays 0 until ~week 6) and the
    prior→in-season ramp silently never engages. The walk-forward backtests
    see whole seasons and never hit this. (Found on the 2026-09-03 reseed.)"""
    if in_season:
        schedule = C.fetch_season_games(season, only_completed=False)
        plays, games = C.fetch_season_plays(season, games=schedule)
        fbs, name, abbr = _fbs_universe(schedule)
    else:
        plays, games = C.fetch_season_plays(season)
        fbs, name, abbr = _fbs_universe(games)
    games_by_id = {g["game_id"]: g for g in games}
    table = ep_table or E.build_ep_table(plays)
    ratings = E.build_team_ratings(plays, fbs, games_by_id, table, ridge=ridge)
    return ratings, name, abbr, table, len(games)


def load_existing_state(path: Path = STATE_PATH) -> dict:
    """The current sector state ({} when missing/unreadable) — for the FPI freeze."""
    try:
        return json.loads(path.read_text()).get("ncaaf") or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def resolve_fpi(
    existing: dict,
    season: int,
    refresh: bool = False,
    fetch_fn: Callable[[int], dict[str, dict]] = F.fetch_fpi,
) -> tuple[dict[str, dict], str]:
    """Preseason FPI ratings by ESPN team id + the source used.

    Returns ``({id: {"fpi": pts, "name": …}}, source)`` with source one of
    ``frozen`` (reused from the existing same-season state), ``live`` (fetched
    now), ``none`` (nothing available). The frozen path stores the centred
    ``fpi_prior`` per team name; it is re-keyed by name via ``fpi_prior_by_name``
    downstream, so here it returns ``{}`` with source ``frozen`` and the caller
    reads the per-team values from ``existing``.
    """
    frozen_ok = (
        not refresh
        and existing.get("fpi_season") == season
        and any(t.get("fpi_prior") is not None for t in (existing.get("teams") or {}).values())
    )
    if frozen_ok:
        return {}, "frozen"
    fetched = fetch_fn(season) or {}
    if fetched:
        return fetched, "live"
    return {}, "none"


def assemble_state(
    season: int,
    prior_season: int,
    regress: float,
    in_ratings: dict,
    in_name: dict[str, str],
    in_abbr: dict[str, str],
    prior_ratings: dict,
    prior_name: dict[str, str],
    prior_abbr: dict[str, str],
    in_games: int,
    fpi_by_id: dict[str, dict],
    fpi_source: str,
    existing: Optional[dict] = None,
    fetched_at: Optional[str] = None,
) -> dict:
    """Pure assembly of the sector state from the two seasons' ratings + FPI.

    Team universe = union of prior-FBS and in-season-FBS so week-0 teams (no
    in-season games yet) still get a row driven purely by the prior. FPI is
    centred on the FBS teams that have a rating and attached by ESPN id, then
    (on the ``frozen`` path) copied by name from the existing state.
    """
    existing = existing or {}
    fetched_at = fetched_at or dt.date.today().isoformat()

    # Canonical-name-keyed priors, regressed toward 0 (league average).
    prior_by_name: dict[str, dict] = {}
    for tid, r in prior_ratings["teams"].items():
        nm = prior_name.get(tid)
        if not nm:
            continue
        prior_by_name[nm] = {
            "off_epa_prior": round(regress * r.get("off_epa_adj", 0.0), 5),
            "def_epa_prior": round(regress * r.get("def_epa_adj", 0.0), 5),
            "off_sr_prior": round(regress * r.get("off_sr_adj", 0.0), 5),
            "def_sr_prior": round(regress * r.get("def_sr_adj", 0.0), 5),
        }

    all_ids = set(in_ratings["teams"]) | set(prior_ratings["teams"])
    fpi_centred = F.centre_fpi(fpi_by_id, all_ids) if fpi_by_id else {}
    frozen_by_name = {
        nm: t.get("fpi_prior")
        for nm, t in (existing.get("teams") or {}).items()
        if t.get("fpi_prior") is not None
    } if fpi_source == "frozen" else {}

    zero_prior = {"off_epa_prior": 0.0, "def_epa_prior": 0.0, "off_sr_prior": 0.0, "def_sr_prior": 0.0}
    teams_out: dict[str, dict] = {}
    for tid in all_ids:
        nm = in_name.get(tid) or prior_name.get(tid)
        if not nm:
            continue
        r_in = in_ratings["teams"].get(tid, {})
        prior = prior_by_name.get(nm, zero_prior)
        row = {
            "off_epa_adj": round(r_in.get("off_epa_adj", 0.0), 5),
            "def_epa_adj": round(r_in.get("def_epa_adj", 0.0), 5),
            "off_sr_adj": round(r_in.get("off_sr_adj", 0.0), 5),
            "def_sr_adj": round(r_in.get("def_sr_adj", 0.0), 5),
            **prior,
            "gp": int(r_in.get("gp", 0)),
            "off_success_rate": r_in.get("off_success_rate", 0.0),
            "def_success_rate": r_in.get("def_success_rate", 0.0),
            "abbrev": in_abbr.get(tid) or prior_abbr.get(tid) or "",
        }
        fpi = fpi_centred.get(tid) if fpi_centred else frozen_by_name.get(nm)
        if fpi is not None:
            row["fpi_prior"] = round(float(fpi), 3)
        # Keep the higher-information row if a name collision maps two ids.
        if nm not in teams_out or row["gp"] >= teams_out[nm]["gp"]:
            teams_out[nm] = row

    n_fpi = sum(1 for t in teams_out.values() if t.get("fpi_prior") is not None)
    if fpi_source == "frozen":
        fpi_meta = {
            "fpi_season": existing.get("fpi_season"),
            "fpi_source": "frozen",
            "fpi_frozen_at": existing.get("fpi_frozen_at"),
            "fpi_teams": n_fpi,
        }
    elif fpi_source == "live":
        fpi_meta = {"fpi_season": season, "fpi_source": "live",
                    "fpi_frozen_at": fetched_at, "fpi_teams": n_fpi}
    else:
        fpi_meta = {"fpi_season": None, "fpi_source": "none",
                    "fpi_frozen_at": None, "fpi_teams": 0}

    return {
        "ncaaf": {
            "schema_version": E.STATE_SCHEMA_V2,
            "teams": teams_out,
            "league_mean_epa": in_ratings["league_mean_epa"] if in_games else prior_ratings["league_mean_epa"],
            "hfa_epa": in_ratings["hfa_epa"] if in_games else prior_ratings["hfa_epa"],
            "source_season": season,
            "seasons_used": [season] if in_games else [prior_season],
            "prior_season": prior_season,
            "prior_regress": regress,
            **fpi_meta,
            "fetched_at": fetched_at,
        }
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_season = E.cfb_season_start_year(dt.date.today())
    ap.add_argument("--season", type=int, default=default_season,
                    help="in-season year to rate (default: active CFB season)")
    ap.add_argument("--prior-season", type=int, default=None,
                    help="season used for the preseason prior (default: season-1)")
    ap.add_argument("--regress", type=float, default=0.5,
                    help="prior-season rating is multiplied by this toward 0 (mean)")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--refresh-fpi", action="store_true",
                    help="refetch live FPI even if this season's prior is already frozen")
    ap.add_argument("--no-fpi", action="store_true",
                    help="skip the FPI fetch (keeps a frozen prior if present)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    season = args.season
    prior_season = args.prior_season if args.prior_season is not None else season - 1

    # Prior season first — its plays also seed a stable EP table reused for the
    # in-season ratings (EP structure is leaguewide and year-stable; using the
    # prior season's table keeps the current-season ratings from depending on
    # their own EP fit when the season is only a few weeks old).
    print(f"Building prior-season {prior_season} ratings…", file=sys.stderr)
    prior_ratings, prior_name, prior_abbr, ep_table, prior_games = _season_ratings(
        prior_season, None, args.ridge
    )
    if prior_games == 0:
        print(f"  WARN: no games for prior season {prior_season}", file=sys.stderr)

    print(f"Building in-season {season} ratings…", file=sys.stderr)
    in_ratings, in_name, in_abbr, _t, in_games = _season_ratings(
        season, ep_table, args.ridge, in_season=True
    )

    existing = load_existing_state()
    if args.no_fpi:
        fpi_by_id, fpi_source = ({}, "frozen") if (
            existing.get("fpi_season") == season
            and any(t.get("fpi_prior") is not None for t in (existing.get("teams") or {}).values())
        ) else ({}, "none")
    else:
        fpi_by_id, fpi_source = resolve_fpi(existing, season, refresh=args.refresh_fpi)
    print(f"FPI prior source: {fpi_source}"
          + (f" ({len(fpi_by_id)} teams fetched)" if fpi_by_id else ""), file=sys.stderr)
    if fpi_source == "none":
        print("  WARN: no FPI prior — v2 falls back to the EPA-only prior for every team",
              file=sys.stderr)

    state = assemble_state(
        season, prior_season, args.regress,
        in_ratings, in_name, in_abbr, prior_ratings, prior_name, prior_abbr,
        in_games, fpi_by_id, fpi_source, existing=existing,
    )
    teams_out = state["ncaaf"]["teams"]

    n_with_gp = sum(1 for t in teams_out.values() if t["gp"] > 0)
    print(
        f"Assembled {len(teams_out)} teams "
        f"({n_with_gp} with {season} games, {in_games} games this season; "
        f"prior from {prior_season}, regress={args.regress}; "
        f"fpi_prior on {state['ncaaf']['fpi_teams']} teams [{fpi_source}])",
        file=sys.stderr,
    )
    # sanity: top 5 by blended net EPA (prior-only if no in-season games)
    def net(t):
        return E.blended_component(t, k=3.0, comp="epa", fpi_share=0.5, plays_per_team=70.0)
    top = sorted(teams_out.items(), key=lambda kv: net(kv[1]), reverse=True)[:5]
    print("  top-5 blended net:", [(n, round(net(t), 3)) for n, t in top], file=sys.stderr)

    if args.dry_run:
        print(f"(dry-run — not written to {STATE_PATH})", file=sys.stderr)
        return 0
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"Wrote {STATE_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
