#!/usr/bin/env python3
"""Walk-forward backtest of the NFL model PICKS (evmax.nfl_projections.picks).

Projects every regular-season game with ratings and combiner fit only on
earlier data (the same ``walk_forward`` the game-model harness uses), makes
the spread and O/U pick exactly as ``evmax project nfl`` does, and grades it:

  * vs the CLOSE — nflverse ``spread_line`` / ``total_line`` (every season);
  * vs the OPENING line — ESPN's opener, which exists only for 2014-16 and
    late 2023 onward (``--espn-open``; one ESPN call per game, cached);
  * line movement — how far the opener moved toward the pick by the close.

Result (2026-10-10, ``--first-feature-season 2010 --seasons 2011-2025 --espn-open``):
spread picks 50.2% vs the close (n 3,818) and 50.1% vs the opener (n 1,462);
the opener moved toward the pick 699 times vs 492 (+0.40 pts) — the model
anticipates line moves, but no season, week window or edge size beats 52.4%
at the close. O/U: the median total removes the over lean (64% -> 48% overs).

Usage:
    python scripts/backtest_nfl_model_picks.py                       # 2016-2025, shipped config
    python scripts/backtest_nfl_model_picks.py --espn-open           # + opening-line grading
    python scripts/backtest_nfl_model_picks.py --first-feature-season 2010 --seasons 2011-2025 --download
Data: EVMAX_NFL_PROJ_DATA (see evmax/nfl_projections/data.py). Ratings need
play-by-play from two seasons before --first-feature-season (--download fetches it).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.nfl_projections import data, picks, team_games  # noqa: E402
from evmax.nfl_projections.game_model import GameModelConfig, walk_forward  # noqa: E402

BREAKEVEN = 0.5238  # -110 on both sides


def seasons_arg(s: str) -> list[int]:
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def make_all(proj: pd.DataFrame, line_m: str, line_t: str, mean_total: bool = False) -> pd.DataFrame:
    """Pick every game against the given line columns (home-margin spread, total)."""
    rows = []
    for r in proj.itertuples():
        p = picks.make_pick(r.home_team, r.away_team, r.proj_margin, r.proj_total,
                            getattr(r, line_m), getattr(r, line_t), int(r.week),
                            r.home_qb_delta, r.away_qb_delta)
        over = p.total_pick_over
        if mean_total and pd.notna(getattr(r, line_t)) and r.proj_total != getattr(r, line_t):
            over = bool(r.proj_total > getattr(r, line_t))
        rows.append({"game_id": r.Index, "pick_home": p.spread_pick_home, "edge": p.spread_edge,
                     "over": over, "flagged": bool(p.flags)})
    return pd.DataFrame(rows).set_index("game_id")


def record(results: pd.Series) -> str:
    r = results.dropna()
    w, l = int((r == "W").sum()), int((r == "L").sum())
    if not w + l:
        return "—"
    n = w + l
    z = (w - n * BREAKEVEN) / math.sqrt(n * BREAKEVEN * (1 - BREAKEVEN))
    return f"{w}-{l} {w / n:5.1%} z{z:+.1f}"


def grade(proj: pd.DataFrame, pk: pd.DataFrame, line_m: str, line_t: str) -> pd.DataFrame:
    margin = proj["home_score"] - proj["away_score"]
    total = proj["home_score"] + proj["away_score"]
    g = pd.DataFrame(index=proj.index)
    g["ats"] = [picks.grade_spread(ph, lm, m) for ph, lm, m in zip(pk["pick_home"], proj[line_m], margin)]
    g["ou"] = [picks.grade_total(o, lt, t) for o, lt, t in zip(pk["over"], proj[line_t], total)]
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=seasons_arg, default=seasons_arg("2016-2025"))
    ap.add_argument("--first-feature-season", type=int, default=GameModelConfig().first_feature_season)
    ap.add_argument("--espn-open", action="store_true", help="also grade vs ESPN opening lines (fetches + caches)")
    ap.add_argument("--download", action="store_true", help="download missing nflverse play-by-play first")
    args = ap.parse_args()

    cfg = GameModelConfig(first_feature_season=args.first_feature_season)
    pbp_seasons = range(cfg.first_feature_season - 2, max(args.seasons) + 1)
    if args.download:
        data.ensure_pbp(pbp_seasons)
    missing = [s for s in pbp_seasons if not data.pbp_file(s).exists()]
    if missing:
        print(f"missing play-by-play for {missing}; rerun with --download", file=sys.stderr)
        return 1
    games = data.load_games()
    tg = team_games.load_team_games(pbp_seasons)
    proj = walk_forward(tg, games, args.seasons, cfg)
    proj = proj[(proj["game_type"] == "REG") & proj["spread_line"].notna()].set_index("game_id")
    if args.espn_open:
        op = data.load_espn_open_lines(games[games["season"].isin(args.seasons)]).set_index("game_id")
        proj["open_margin"] = op["open_line_margin"].reindex(proj.index).astype(float)
        proj["open_total"] = op["open_total"].reindex(proj.index).astype(float)
    else:
        proj["open_margin"] = np.nan
        proj["open_total"] = np.nan

    pc = make_all(proj, "spread_line", "total_line")
    gc = grade(proj, pc, "spread_line", "total_line")
    po = make_all(proj, "open_margin", "open_total")
    go = grade(proj, po, "open_margin", "open_total")
    move = pd.Series([picks.line_move_toward(ph, a, b) for ph, a, b in
                      zip(po["pick_home"], proj["open_margin"], proj["spread_line"])], index=proj.index, dtype=float)
    margin = proj["home_score"] - proj["away_score"]

    print(f"{'season':6s} {'n':>4} {'MAE':>5} {'Vegas':>5} | {'ATS vs close':>20} {'ATS vs open':>20} | "
          f"{'O/U vs close':>20} {'O/U vs open':>20} | {'open moved to pick':>18}")
    for s, idx in list(proj.groupby("season").groups.items()) + [("ALL", proj.index)]:
        d = proj.loc[idx]
        mv = move.loc[idx].dropna()
        mvtxt = f"{(mv > 0).sum()}-{(mv < 0).sum()} {mv.mean():+.2f}" if len(mv) else "—"
        print(f"{str(s):6s} {len(d):4d} {(d.proj_margin - margin.loc[idx]).abs().mean():5.2f} "
              f"{(d.spread_line - margin.loc[idx]).abs().mean():5.2f} | {record(gc.loc[idx, 'ats']):>20} "
              f"{record(go.loc[idx, 'ats']):>20} | {record(gc.loc[idx, 'ou']):>20} {record(go.loc[idx, 'ou']):>20} | {mvtxt:>18}")

    print("\nSpread picks by slice (vs close | vs open):")
    slices = {"weeks 1-4": proj["week"] <= 4, "weeks 5-13": proj["week"].between(5, 13), "weeks 14-18": proj["week"] >= 14,
              "no check flag": ~pc["flagged"], "check flag": pc["flagged"]}
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]:
        slices[f"edge {lo}-{hi if hi < 99 else '+'}"] = (pc["edge"] >= lo) & (pc["edge"] < hi)
    for name, m in slices.items():
        print(f"  {name:14s} n={int(m.sum()):4d}  {record(gc.loc[m, 'ats']):>20} | {record(go.loc[m, 'ats']):>20}")

    mean_pc = make_all(proj, "spread_line", "total_line", mean_total=True)
    mean_g = grade(proj, mean_pc, "spread_line", "total_line")
    def over_share(pk: pd.DataFrame) -> float:
        return pk["over"].dropna().astype(float).mean()
    print(f"\nO/U vs close: median total {record(gc['ou'])} (over share {over_share(pc):.0%}) | "
          f"mean total {record(mean_g['ou'])} (over share {over_share(mean_pc):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
