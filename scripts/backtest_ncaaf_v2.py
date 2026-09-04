#!/usr/bin/env python3
"""Walk-forward validation of ``ncaaf_efficiency_v2`` against the v1 model.

Leak-free protocol (mirrors backtest_ncaaf_efficiency.py):
  * EP table and the regressed prior come from season-1 only.
  * In-season EPA / success-rate ratings for week W are rebuilt from games in
    weeks < W (E.build_team_ratings, the exact seed-time code path).
  * The preseason FPI prior is the PRE-WEEK-0 Wayback snapshot for that season
    (data/backtest/ncaaf_fpi/fpi_{season}.json, built by
    scripts/build_ncaaf_fpi_history.py) — never an in-season value.
  * The v2 margin constants are the FROZEN agent constants (no per-season
    refit); ``--fit-season`` prints what a no-intercept OLS would choose so a
    future re-tune is reproducible (2024 was the fit season; 2023 + 2025 are
    the untouched holdouts).

Columns per season and per week bucket (0-3 / 4-8 / 9+):
  market   — closing-spread-implied prob (calibration reference, NOT a target)
  v1       — shipped v1 formula (EPA only, EPA prior, 70·Δ + 2.5, σ 16.5)
  v2       — agent constants (EPA + SR, FPI-mixed prior)
  v2_nofpi — v2 with fpi_share = 0 (ablation: what success rate alone adds)
  v2_fpi1  — v2 with fpi_share = 1 (ablation: pure-FPI prior)

Plus, per season: paired ΔBrier v2−v1 with a z-score, the 0.15/0.85 blend with
the closing market (production proxy — expected ≈ 0, the sharp anchor absorbs
model gains), and the open→close lens: the slope of (close − open) on
(model − open), i.e. does the model anticipate line movement. That lens is the
CLV-shaped value the shadow promotion gate scores.

    python scripts/backtest_ncaaf_v2.py                         # 2023,2024,2025
    python scripts/backtest_ncaaf_v2.py --seasons 2025 --fit-season 2024
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from evmax.agents.models import _cfb_efficiency as E  # noqa: E402
from evmax.agents.models import ncaaf_efficiency_agent as A  # noqa: E402
from evmax.clients import cfb_espn as C  # noqa: E402
from evmax.clients.cfb_fpi import centre_fpi  # noqa: E402
from evmax.models_ml._math import normal_cdf  # noqa: E402
from scripts.backtest_ncaaf_efficiency import (  # noqa: E402
    BUCKETS, _bucket, _fbs, _load_home_spreads, _week_index,
)

FPI_DIR = _REPO / "data" / "backtest" / "ncaaf_fpi"
SPREAD_SIGMA = 16.3
BLEND_SHARE = 0.15  # model share vs the market in the production-proxy blend


def load_fpi(season: int) -> dict[str, dict]:
    p = FPI_DIR / f"fpi_{season}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("teams") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _prior_by_name(prior_season: int, regress: float, ridge: float, ep_table):
    plays, games = C.fetch_season_plays(prior_season)
    fbs, name = _fbs(games)
    gbi = {g["game_id"]: g for g in games}
    ratings = E.build_team_ratings(plays, fbs, gbi, ep_table, ridge=ridge)
    out = {}
    for tid, r in ratings["teams"].items():
        nm = name.get(tid)
        if nm:
            out[nm] = {
                "off_epa_prior": regress * r["off_epa_adj"],
                "def_epa_prior": regress * r["def_epa_adj"],
                "off_sr_prior": regress * r["off_sr_adj"],
                "def_sr_prior": regress * r["def_sr_adj"],
            }
    return out


def _v2_prob(rh: dict, ra: dict, neutral: bool, fpi_share: float) -> float:
    d_epa = (E.blended_component(rh, A.RAMP_K, "epa", fpi_share, A.PLAYS_PER_TEAM_GAME)
             - E.blended_component(ra, A.RAMP_K, "epa", fpi_share, A.PLAYS_PER_TEAM_GAME))
    d_sr = E.blended_component(rh, A.RAMP_K, "sr") - E.blended_component(ra, A.RAMP_K, "sr")
    p, _ = E.project_win_prob_v2(d_epa, d_sr, A.V2_EPA_PTS, A.V2_SR_PTS,
                                 A.V2_HOME_EDGE_PTS, A.SCORE_STDEV, neutral)
    return p


def run_season(season: int, regress: float = 0.5, ridge: float = 1.0) -> list[dict]:
    """Walk-forward one season → per-game rows with v1/v2 probs + market + deltas."""
    plays, games = C.fetch_season_plays(season)
    fbs, name = _fbs(games)
    gbi = {g["game_id"]: g for g in games}
    season_start = dt.date.fromisoformat(C.SEASON_WINDOWS[season][0])

    ep_plays, _ = C.fetch_season_plays(season - 1)
    ep_table = E.build_ep_table(ep_plays)
    prior = _prior_by_name(season - 1, regress, ridge, ep_table)
    fpi = centre_fpi(load_fpi(season), fbs)
    if not fpi:
        print(f"  WARN: no preseason FPI file for {season} — v2 == v2_nofpi", file=sys.stderr)
    close = _load_home_spreads(season, games)
    open_ = _load_home_spreads(season, games, col="opening_lines")

    for g in games:
        g["week"] = _week_index(g["date"], season_start)
    games_sorted = sorted(games, key=lambda g: (g["week"], g["date"]))
    zero = {"off_epa_adj": 0.0, "def_epa_adj": 0.0, "off_sr_adj": 0.0, "def_sr_adj": 0.0, "gp": 0}
    zero_prior = {"off_epa_prior": 0.0, "def_epa_prior": 0.0, "off_sr_prior": 0.0, "def_sr_prior": 0.0}

    rows = []
    for w in sorted({g["week"] for g in games_sorted}):
        before = {g["game_id"] for g in games_sorted if g["week"] < w}
        sub = [p for p in plays if p["game_id"] in before]
        ratings = E.build_team_ratings(sub, fbs, gbi, ep_table, ridge=ridge)["teams"]
        for g in (x for x in games_sorted if x["week"] == w):
            h, a = g["home"]["id"], g["away"]["id"]
            if h not in fbs or a not in fbs:
                continue
            hs, as_ = g["home"].get("score"), g["away"].get("score")
            if hs is None or as_ is None or hs == as_:
                continue
            rh = dict(zero, **ratings.get(h, {}))
            ra = dict(zero, **ratings.get(a, {}))
            rh.update(prior.get(name.get(h), zero_prior))
            ra.update(prior.get(name.get(a), zero_prior))
            if h in fpi:
                rh["fpi_prior"] = fpi[h]
            if a in fpi:
                ra["fpi_prior"] = fpi[a]
            neutral = bool(g.get("neutral", False))

            p_v1, _ = E.project_win_prob(E.blended_net(rh, A.RAMP_K), E.blended_net(ra, A.RAMP_K),
                                         A.PLAYS_PER_TEAM_GAME, A.HOME_EDGE_PTS, A.SCORE_STDEV, neutral)
            d_epa = (E.blended_component(rh, A.RAMP_K, "epa", A.FPI_PRIOR_SHARE, A.PLAYS_PER_TEAM_GAME)
                     - E.blended_component(ra, A.RAMP_K, "epa", A.FPI_PRIOR_SHARE, A.PLAYS_PER_TEAM_GAME))
            d_sr = E.blended_component(rh, A.RAMP_K, "sr") - E.blended_component(ra, A.RAMP_K, "sr")
            sp_c, sp_o = close.get(g["game_id"]), open_.get(g["game_id"])
            rows.append({
                "season": season, "week": w, "bucket": _bucket(w), "y": int(hs > as_),
                "margin": hs - as_, "home": 0.0 if neutral else 1.0,
                "gp_min": min(rh["gp"], ra["gp"]),
                "v1": p_v1,
                "v2": _v2_prob(rh, ra, neutral, A.FPI_PRIOR_SHARE),
                "v2_nofpi": _v2_prob(rh, ra, neutral, 0.0),
                "v2_fpi1": _v2_prob(rh, ra, neutral, 1.0),
                "market": normal_cdf(-sp_c / SPREAD_SIGMA) if sp_c is not None else None,
                "open": normal_cdf(-sp_o / SPREAD_SIGMA) if sp_o is not None else None,
                "d_epa": d_epa, "d_sr": d_sr,
            })
    return rows


# ---------------------------------------------------------------- report ----

def _brier(rows, key):
    v = [(r[key], r["y"]) for r in rows if r.get(key) is not None]
    return (float(np.mean([(p - y) ** 2 for p, y in v])) if v else None), len(v)


def paired(rows, key, ref):
    d = np.array([(r[key] - r["y"]) ** 2 - (r[ref] - r["y"]) ** 2
                  for r in rows if r.get(key) is not None and r.get(ref) is not None])
    if len(d) < 2:
        return 0.0, 0.0
    se = d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean()), float(d.mean() / se) if se else 0.0


def report(rows: list[dict], label: str) -> None:
    cols = ("market", "v1", "v2", "v2_nofpi", "v2_fpi1")
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"{'bucket':8s} {'n':>5s}  " + " ".join(f"{c:>9s}" for c in cols))
    for nm, _, _ in BUCKETS + [("ALL", 0, 99)]:
        sub = rows if nm == "ALL" else [r for r in rows if r["bucket"] == nm]
        if not sub:
            continue
        cells = []
        for c in cols:
            b, _n = _brier(sub, c)
            cells.append(f"{b:.4f}" if b is not None else "-")
        print(f"{nm:8s} {len(sub):>5d}  " + " ".join(f"{c:>9s}" for c in cells))
    for key in ("v2", "v2_nofpi", "v2_fpi1"):
        dm, z = paired(rows, key, "v1")
        print(f"  Δ{key}−v1: {1000*dm:+.2f}/1000  z={z:+.2f}")

    mk = [r for r in rows if r.get("market") is not None]
    if mk:
        y = np.array([r["y"] for r in mk], float)
        pm = np.array([r["market"] for r in mk])
        for key in ("v1", "v2"):
            pb = (1 - BLEND_SHARE) * pm + BLEND_SHARE * np.array([r[key] for r in mk])
            print(f"  blend {BLEND_SHARE:.2f}·{key}+{1-BLEND_SHARE:.2f}·close: Brier {np.mean((pb-y)**2):.4f}"
                  f"  (close alone {np.mean((pm-y)**2):.4f})")
    oc = [r for r in rows if r.get("market") is not None and r.get("open") is not None]
    if oc:
        po = np.array([r["open"] for r in oc])
        pc = np.array([r["market"] for r in oc])
        print(f"  open→close lens (n={len(oc)}): MSE(open,close)={np.mean((po-pc)**2)*1000:.2f}/1000")
        for key in ("v1", "v2"):
            x = np.array([r[key] for r in oc]) - po
            t = pc - po
            slope = float(np.sum(x * t) / np.sum(x * x))
            res = t - slope * x
            se = math.sqrt(np.sum(res ** 2) / (len(x) - 1) / np.sum(x * x))
            print(f"    slope(close−open ~ {key}−open) = {slope:+.3f}  t={slope/se:+.2f}")


def fit_constants(rows: list[dict]) -> None:
    """No-intercept OLS margin ~ home + Δepa + Δsr (what the frozen constants came from)."""
    X = np.array([[r["home"], r["d_epa"], r["d_sr"]] for r in rows])
    y = np.array([r["margin"] for r in rows], float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    sigma = float(np.std(y - X @ b))
    print(f"\n--fit: home={b[0]:+.2f} epa_pts={b[1]:+.2f} sr_pts={b[2]:+.2f} sigma={sigma:.1f}"
          f"  (shipped: {A.V2_HOME_EDGE_PTS}/{A.V2_EPA_PTS}/{A.V2_SR_PTS}, σ {A.SCORE_STDEV})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--fit-season", type=int, default=None,
                    help="print the no-intercept OLS constants for this season's rows")
    ap.add_argument("--regress", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args()

    pooled = []
    for s in (int(x) for x in args.seasons.split(",")):
        rows = run_season(s, args.regress, args.ridge)
        report(rows, f"holdout {s}")
        pooled += rows
        if args.fit_season == s:
            fit_constants(rows)
    if "," in args.seasons:
        report(pooled, "POOLED " + args.seasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
