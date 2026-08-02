#!/usr/bin/env python3
"""Seed data/models/ncaaf_efficiency_state.json — opponent-adjusted EPA ratings
plus a regressed preseason prior, keyed by canonical (normalized ESPN location)
team name.

State layout::

    {"ncaaf": {
       "teams": {
         "ohio state": {
           "off_epa_adj": 0.18, "def_epa_adj": -0.20,     # in-season, league-relative
           "off_epa_prior": 0.09, "def_epa_prior": -0.06,  # regressed prior-season net
           "gp": 9, "off_success_rate": 0.49, "def_success_rate": 0.40,
           "abbrev": "OSU"
         }, ...
       },
       "league_mean_epa": 0.01, "hfa_epa": 0.05,
       "source_season": 2026, "seasons_used": [2026],
       "prior_season": 2025, "prior_regress": 0.5,
       "fetched_at": "2026-08-15"
    }}

The agent (evmax/agents/models/ncaaf_efficiency_agent.py) reads this and applies
the in-season → prior RAMP w(gp)=gp/(gp+k) at predict time, so week 0–3 games
lean on the prior and week 9+ games lean on the opponent-adjusted in-season EPA
(the SP+/FPI shrinkage pattern the CFB market relies on).

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

# Make the package importable when run as a script from a worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.agents.models import _cfb_efficiency as E  # noqa: E402
from evmax.clients import cfb_espn as C  # noqa: E402

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


def _season_ratings(season: int, ep_table: dict | None, ridge: float):
    """Fetch a season, build FBS universe + opponent-adjusted ratings.

    If ep_table is None, build it from this season's own plays (structural,
    leaguewide — not a team-level leak). Returns (ratings, id→name, id→abbr,
    ep_table, n_games)."""
    plays, games = C.fetch_season_plays(season)
    fbs, name, abbr = _fbs_universe(games)
    games_by_id = {g["game_id"]: g for g in games}
    table = ep_table or E.build_ep_table(plays)
    ratings = E.build_team_ratings(plays, fbs, games_by_id, table, ridge=ridge)
    return ratings, name, abbr, table, len(games)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_season = E.cfb_season_start_year(dt.date.today())
    ap.add_argument("--season", type=int, default=default_season,
                    help="in-season year to rate (default: active CFB season)")
    ap.add_argument("--prior-season", type=int, default=None,
                    help="season used for the preseason prior (default: season-1)")
    ap.add_argument("--regress", type=float, default=0.5,
                    help="prior-season net rating is multiplied by this toward 0 (mean)")
    ap.add_argument("--ridge", type=float, default=1.0)
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
        season, ep_table, args.ridge
    )

    # Canonical-name-keyed prior net, regressed toward 0.
    prior_net_by_name: dict[str, dict] = {}
    for tid, r in prior_ratings["teams"].items():
        nm = prior_name.get(tid)
        if not nm:
            continue
        prior_net_by_name[nm] = {
            "off_epa_prior": round(args.regress * r["off_epa_adj"], 5),
            "def_epa_prior": round(args.regress * r["def_epa_adj"], 5),
        }

    # Team universe = union of prior-FBS and in-season-FBS so week-0 teams (no
    # in-season games yet) still get a row driven purely by the prior.
    teams_out: dict[str, dict] = {}
    all_ids = set(in_ratings["teams"]) | set(prior_ratings["teams"])
    # index prior ratings by name for lookup by in-season id (ids are stable
    # across seasons on ESPN, but join by name too as a guard against re-id)
    for tid in all_ids:
        nm = in_name.get(tid) or prior_name.get(tid)
        if not nm:
            continue
        r_in = in_ratings["teams"].get(tid, {})
        prior = prior_net_by_name.get(nm, {"off_epa_prior": 0.0, "def_epa_prior": 0.0})
        row = {
            "off_epa_adj": round(r_in.get("off_epa_adj", 0.0), 5),
            "def_epa_adj": round(r_in.get("def_epa_adj", 0.0), 5),
            "off_epa_prior": prior["off_epa_prior"],
            "def_epa_prior": prior["def_epa_prior"],
            "gp": int(r_in.get("gp", 0)),
            "off_success_rate": r_in.get("off_success_rate", 0.0),
            "def_success_rate": r_in.get("def_success_rate", 0.0),
            "abbrev": in_abbr.get(tid) or prior_abbr.get(tid) or "",
        }
        # Keep the higher-information row if a name collision maps two ids.
        if nm not in teams_out or row["gp"] >= teams_out[nm]["gp"]:
            teams_out[nm] = row

    state = {
        "ncaaf": {
            "teams": teams_out,
            "league_mean_epa": in_ratings["league_mean_epa"] if in_games else prior_ratings["league_mean_epa"],
            "hfa_epa": in_ratings["hfa_epa"] if in_games else prior_ratings["hfa_epa"],
            "source_season": season,
            "seasons_used": [season] if in_games else [prior_season],
            "prior_season": prior_season,
            "prior_regress": args.regress,
            "fetched_at": dt.date.today().isoformat(),
        }
    }

    n_with_gp = sum(1 for t in teams_out.values() if t["gp"] > 0)
    print(
        f"Assembled {len(teams_out)} teams "
        f"({n_with_gp} with {season} games, {in_games} games this season; "
        f"prior from {prior_season}, regress={args.regress})",
        file=sys.stderr,
    )
    # sanity: top 5 by blended net (prior-only if no in-season games)
    def net(t):
        return E.blended_net(t, k=3.0)
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
