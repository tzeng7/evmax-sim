#!/usr/bin/env python3
"""Walk-forward backtest for the NCAAF opponent-adjusted EPA model.

Leak-free protocol (mirrors backtest_ncaab_efficiency.py):
  * The expected-points table is fit from seasons STRICTLY EARLIER than the eval
    season (falls back to the eval season only if no earlier season is cached,
    which it flags — a small structural, non-team-level leak).
  * The preseason prior is the regressed FINAL ratings of season-1.
  * In-season ratings for week W are recomputed from games in weeks < W only.
    A team's game count so far drives the prior→in-season ramp, so early weeks
    lean on the prior exactly as they will live.

What it reports (per season and pooled), OVERALL and by WEEK BUCKET (0-3 / 4-8 /
9+) — the bucketing is the whole point: it validates that the prior ramp helps
in the weeks the CFB literature says it must (early season):
  * model    — full blend (ramp)                → the shipping model
  * prior    — prior-only ablation (w=0)         → is in-season data adding value?
  * inseason — in-season-only ablation (no prior)→ is the prior adding value early?
  * market   — closing-spread-implied prob       → the external calibration yardstick
    (Beating the close is NOT expected — see the value-audit philosophy. The
    market column is a calibration reference, not a target.)

Usage:
  python scripts/backtest_ncaaf_efficiency.py --holdout 2024 [--detail]
  python scripts/backtest_ncaaf_efficiency.py --sweep --seasons 2022,2023 --holdout 2024
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.agents.models import _cfb_efficiency as E  # noqa: E402
from evmax.clients import cfb_espn as C  # noqa: E402
from evmax.models_ml._math import normal_cdf  # noqa: E402

BUCKETS = [("0-3", 0, 3), ("4-8", 4, 8), ("9+", 9, 99)]


def _week_index(game_date: str, season_start: dt.date) -> int:
    try:
        d = dt.date.fromisoformat(game_date[:10])
    except Exception:  # noqa: BLE001
        return 1
    return max(1, (d - season_start).days // 7 + 1)


def _bucket(week: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= week <= hi:
            return name
    return "9+"


def _orient_spreads(rows, games: list[dict]) -> dict[str, float]:
    """Median closing HOME spread per ESPN game_id from (game_id, team_label, line)
    rows, correctly oriented (negative = home favored).

    The cfbfastR betting parquet has one row per (game, team, book); the row's
    ``abbr`` column is the team the ``lines`` value belongs to. For 2023+ that
    column carries the ESPN team NAME ("Penn State", "South Florida"), older
    seasons carried the abbreviation — so match the normalized location first
    and the ESPN abbreviation as a fallback (the abbreviation-only join matched
    ~5% of 2023-25 rows and left the market column near-blank). Away-team rows
    are flipped to the home perspective. Unmatched rows are skipped.
    """
    key: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
    for g in games:
        key[g["game_id"]] = (
            (_norm(g["home"].get("location") or ""), (g["home"].get("abbr") or "").upper()),
            (_norm(g["away"].get("location") or ""), (g["away"].get("abbr") or "").upper()),
        )
    per_game: dict[str, list[float]] = defaultdict(list)
    for gid, label, line in rows:
        try:
            g = str(int(gid))
            line = float(line)
        except (TypeError, ValueError):
            continue
        if g not in key or line != line:  # NaN guard
            continue
        (h_loc, h_abbr), (a_loc, a_abbr) = key[g]
        lab_n = _norm(label)
        lab_u = (label or "").upper()
        if lab_n and (lab_n == h_loc or lab_u == h_abbr):
            per_game[g].append(line)
        elif lab_n and (lab_n == a_loc or lab_u == a_abbr):
            per_game[g].append(-line)
    return {g: float(np.median(v)) for g, v in per_game.items() if v}


def _load_home_spreads(season: int, games: list[dict], col: str = "lines") -> dict[str, float]:
    """Closing (``col="lines"``) or opening (``col="opening_lines"``) HOME spread
    per ESPN game_id from the cfbfastR betting parquet — see :func:`_orient_spreads`.
    Games with no matched book row are omitted (their market cell is blank)."""
    import httpx
    import pandas as pd

    url = ("https://github.com/sportsdataverse/cfbfastR-data/raw/main/"
           "betting/parquet/cfb_line_odds.parquet")
    try:
        df = pd.read_parquet(io.BytesIO(
            httpx.get(url, timeout=180, follow_redirects=True).content))
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: market parquet unavailable ({e})", file=sys.stderr)
        return {}
    df = df[(df.season == season) & (df.market_type == "spread") & df[col].notna()]
    return _orient_spreads(zip(df.game_id, df.abbr, df[col]), games)


def build_prior(prior_season: int, regress: float, ridge: float, ep_table):
    plays, games = C.fetch_season_plays(prior_season)
    fbs, name = _fbs(games)
    gbi = {g["game_id"]: g for g in games}
    table = ep_table or E.build_ep_table(plays)
    ratings = E.build_team_ratings(plays, fbs, gbi, table, ridge=ridge)
    prior_by_name = {}
    for tid, r in ratings["teams"].items():
        nm = name.get(tid)
        if nm:
            prior_by_name[nm] = {
                "off_epa_prior": regress * r["off_epa_adj"],
                "def_epa_prior": regress * r["def_epa_adj"],
            }
    return prior_by_name, table


def _fbs(games):
    appear = defaultdict(int)
    name = {}
    for g in games:
        for s in ("home", "away"):
            appear[g[s]["id"]] += 1
            name[g[s]["id"]] = _norm(g[s].get("location") or "")
    fbs = {t for t, n in appear.items() if n >= 6 and name.get(t)}
    return fbs, name


def _norm(s):
    import re
    s = (s or "").lower().strip()
    s = re.sub(r"['’‘]", "", s)
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def run_season(season, ep_seasons, params):
    """Walk-forward one season. Returns list of per-game result dicts."""
    ramp_k = params["ramp_k"]
    regress = params["regress"]
    ridge = params["ridge"]
    plays_per = params["plays_per"]
    stdev = params["stdev"]

    plays, games = C.fetch_season_plays(season)
    fbs, name = _fbs(games)
    gbi = {g["game_id"]: g for g in games}
    season_start = dt.date.fromisoformat(C.SEASON_WINDOWS[season][0])

    # EP table from earlier seasons (leak-free); else this season (flagged).
    ep_plays = []
    for es in ep_seasons:
        ep_plays.extend(C.fetch_season_plays(es)[0])
    ep_table = E.build_ep_table(ep_plays) if ep_plays else E.build_ep_table(plays)

    prior_by_name, _ = build_prior(season - 1, regress, ridge, ep_table)
    home_spreads = _load_home_spreads(season, games)

    # order games by (week, date); attach week
    for g in games:
        g["week"] = _week_index(g["date"], season_start)
    games_sorted = sorted(games, key=lambda g: (g["week"], g["date"]))
    weeks = sorted({g["week"] for g in games_sorted})

    results = []
    for w in weeks:
        # in-season ratings from all games in weeks < w (leak-free)
        gids_before = {g["game_id"] for g in games_sorted if g["week"] < w}
        sub_plays = [p for p in plays if p["game_id"] in gids_before]
        ratings = E.build_team_ratings(sub_plays, fbs, gbi, ep_table, ridge=ridge)["teams"]

        for g in (x for x in games_sorted if x["week"] == w):
            h, a = g["home"]["id"], g["away"]["id"]
            hn, an = name.get(h), name.get(a)
            if not hn or not an or hn not in fbs and h not in fbs:
                pass
            # need both FBS-rated (or prior) — skip FCS-involved
            if h not in fbs or a not in fbs:
                continue
            rh = dict(ratings.get(h, {"off_epa_adj": 0, "def_epa_adj": 0, "gp": 0}))
            ra = dict(ratings.get(a, {"off_epa_adj": 0, "def_epa_adj": 0, "gp": 0}))
            rh.update(prior_by_name.get(hn, {"off_epa_prior": 0.0, "def_epa_prior": 0.0}))
            ra.update(prior_by_name.get(an, {"off_epa_prior": 0.0, "def_epa_prior": 0.0}))
            # actual outcome
            hs, as_ = g["home"].get("score"), g["away"].get("score")
            if hs is None or as_ is None or hs == as_:
                continue
            y = 1 if hs > as_ else 0
            neutral = g.get("neutral", False)

            net_h = E.blended_net(rh, k=ramp_k)
            net_a = E.blended_net(ra, k=ramp_k)
            p_model, _ = E.project_win_prob(net_h, net_a, plays_per,
                                            params["home_edge"], stdev, neutral)
            # ablations
            rh0, ra0 = dict(rh, gp=0), dict(ra, gp=0)           # prior-only
            p_prior, _ = E.project_win_prob(E.blended_net(rh0, ramp_k),
                                            E.blended_net(ra0, ramp_k), plays_per,
                                            params["home_edge"], stdev, neutral)
            rhI = dict(rh, off_epa_prior=0.0, def_epa_prior=0.0)  # in-season only
            raI = dict(ra, off_epa_prior=0.0, def_epa_prior=0.0)
            p_ins, _ = E.project_win_prob(E.blended_net(rhI, ramp_k),
                                          E.blended_net(raI, ramp_k), plays_per,
                                          params["home_edge"], stdev, neutral)
            # market
            sp = home_spreads.get(g["game_id"])
            p_mkt = normal_cdf(-sp / params["spread_sigma"]) if sp is not None else None

            results.append({
                "week": w, "bucket": _bucket(w), "y": y,
                "model": p_model, "prior": p_prior, "inseason": p_ins,
                "market": p_mkt, "gp_min": min(rh.get("gp", 0), ra.get("gp", 0)),
                # Game identity — lets a downstream blend sweep join elo/form
                # probs onto the same game. Extra keys; report()/sweep ignore them.
                "home": hn, "away": an, "date": g["date"], "neutral": neutral,
            })
    return results


def _brier(rows, key):
    vals = [(r[key], r["y"]) for r in rows if r.get(key) is not None]
    if not vals:
        return None, 0
    b = np.mean([(p - y) ** 2 for p, y in vals])
    acc = np.mean([int((p >= 0.5) == bool(y)) for p, y in vals])
    return (round(float(b), 4), round(float(acc), 3)), len(vals)


def report(rows, label):
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"{'bucket':8s} {'n':>5s}  {'model':>14s} {'prior':>14s} {'inseason':>14s} {'market':>14s}")
    for name, _, _ in BUCKETS + [("ALL", 0, 99)]:
        sub = rows if name == "ALL" else [r for r in rows if r["bucket"] == name]
        if not sub:
            continue
        cells = []
        for k in ("model", "prior", "inseason", "market"):
            res, n = _brier(sub, k)
            cells.append(f"{res[0]:.4f}/{res[1]:.2f}" if res else "   -    ")
        print(f"{name:8s} {len(sub):>5d}  " + " ".join(f"{c:>14s}" for c in cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", type=int, default=2024)
    ap.add_argument("--seasons", default="2022,2023", help="train seasons (sweep)")
    ap.add_argument("--ramp-k", type=float, default=3.0)
    ap.add_argument("--regress", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--plays-per", type=float, default=70.0)
    ap.add_argument("--stdev", type=float, default=16.5)
    ap.add_argument("--home-edge", type=float, default=2.5)
    ap.add_argument("--spread-sigma", type=float, default=16.3)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    base = dict(ramp_k=args.ramp_k, regress=args.regress, ridge=args.ridge,
                plays_per=args.plays_per, stdev=args.stdev, home_edge=args.home_edge,
                spread_sigma=args.spread_sigma)

    if not args.sweep:
        ep_seasons = [args.holdout - 1]
        rows = run_season(args.holdout, ep_seasons, base)
        report(rows, f"holdout {args.holdout}  params={base}")
        return 0

    # Sweep ramp_k × regress on train seasons; hold out.
    train_seasons = [int(s) for s in args.seasons.split(",")]
    grid_k = [2.0, 3.0, 4.0, 5.0]
    grid_r = [0.35, 0.5, 0.65]
    print("Sweeping ramp_k × regress on train seasons", train_seasons, file=sys.stderr)
    best, best_b = None, 1e9
    for k in grid_k:
        for r in grid_r:
            p = dict(base, ramp_k=k, regress=r)
            allrows = []
            for s in train_seasons:
                allrows += run_season(s, [s - 1], p)
            (b, _), _ = _brier(allrows, "model")
            print(f"  k={k} regress={r}: train Brier {b}", file=sys.stderr)
            if b < best_b:
                best_b, best = b, (k, r)
    print(f"BEST train: ramp_k={best[0]} regress={best[1]} (Brier {best_b})", file=sys.stderr)
    hp = dict(base, ramp_k=best[0], regress=best[1])
    rows = run_season(args.holdout, [args.holdout - 1], hp)
    report(rows, f"HOLDOUT {args.holdout} @ best train params {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
