"""Per-league soccer model parameters — walk-forward test of two candidates:

  A. Poisson `league_avg` PER LEAGUE (running mean of that league's home/away
     goals) instead of the one soccer-wide pair. Bundesliga and Serie A differ
     by ~0.5 goals/game, so one λ baseline mis-scales every team ratio.
  B. Elo home-field advantage PER LEAGUE (swept per league on the train
     window) instead of the soccer-wide HOME_ADVANTAGE_ELO=60.

Both are judged on the STANDALONE model's 3-way Brier (that is what the
candidate changes) and on the blended ensemble (tier sharp_weight + the
sector disagreement ramp — what a bet would see). Walk-forward: every league
replayed chronologically through fresh agents exactly as
soccer_walkforward.walk_forward_predictions does, with the candidate swapped
in. Train = 2425 (Euro) / calendar 2025 (MLS); holdout = 2526 / 2026.
Paired deltas vs the baseline carry a z-score.

Usage:
    .venv/bin/python scripts/backtest_soccer_league_params.py [--skip-hfa]
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from evmax.agents.models.ensemble_agent import EnsembleModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.backtest.loader import SOCCER_LEAGUES
from evmax.backtest.models import BacktestRow
from evmax.backtest.sources import soccer_walkforward as W
from evmax.backtest.sources.soccer_csv import parse_soccer_csv, parse_soccer_extra_csv

SECTOR_RAMP = EnsembleModelAgent.DISAGREEMENT_OVERRIDES["soccer"]
HFA_GRID = (30.0, 45.0, 60.0, 75.0, 90.0, 110.0)
LEAGUE_AVG_MIN_GAMES = 60  # per league, before the running mean replaces the default


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_rows() -> list[BacktestRow]:
    rows: list[BacktestRow] = []
    for season in ("2324", "2425", "2526"):
        for code in SOCCER_LEAGUES:
            path = Path(f"data/backtest/soccer/{season}/{code}.csv")
            if path.exists():
                rows.extend(parse_soccer_csv(path, code, season))
    usa = Path("data/backtest/soccer/extra/USA.csv")
    if usa.exists():
        rows.extend(parse_soccer_extra_csv(usa, "USA", ["2324", "2425", "2526"]))
    return rows


def window(r, which: str) -> bool:
    """train = Euro 2425 season / MLS 2025; holdout = Euro 2526 / MLS 2026."""
    d: date = r.date
    if r.league == "MLS":
        return d.year == (2025 if which == "train" else 2026)
    if which == "train":
        return (d.year == 2024 and d.month >= 7) or (d.year == 2025 and d.month <= 6)
    return (d.year == 2025 and d.month >= 7) or (d.year == 2026 and d.month <= 6)


# --------------------------------------------------------------------------- #
# Walk-forward with swappable per-league behaviour
# --------------------------------------------------------------------------- #


def walk_forward(
    rows: list[BacktestRow],
    poisson_per_league: bool = False,
    hfa_by_league: Optional[dict[str, float]] = None,
) -> list[W.SoccerWalkForwardRow]:
    from evmax.agents.models.soccer_xg_agent import SoccerXgAgent

    rows_sorted = sorted(rows, key=lambda r: (r.date, r.team_home))
    elo_state: dict = {"ratings": {}, "game_counts": {}, "h2h": {}}
    form = FormModelAgent()
    form._state = {}
    poisson_states: dict[str, dict] = defaultdict(dict)   # league → state (or "_all")
    goal_tally: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])  # league → [home, away, n]
    xg_agent = SoccerXgAgent()
    xg_agent._state = {"teams": {}}
    default_hfa = W.HOME_ADVANTAGE_ELO.get("soccer", 0.0)

    preds: list[W.SoccerWalkForwardRow] = []
    for row in rows_sorted:
        home, away, lg = row.team_home, row.team_away, row.league
        pkey = lg if poisson_per_league else "_all"
        pstate = poisson_states[pkey]
        if poisson_per_league:
            h, a, n = goal_tally[lg]
            if n >= LEAGUE_AVG_MIN_GAMES:
                pstate["league_avg"] = {"home": h / n, "away": a / n}
        if hfa_by_league is not None:
            W.HOME_ADVANTAGE_ELO["soccer"] = hfa_by_league.get(lg, default_hfa)
        try:
            elo_p = W._elo_3way(elo_state, "soccer", home, away)
            form_p = W._form_3way(form, "soccer", home, away, row.date)
            pois_p = W._poisson_3way(pstate, "soccer", home, away)
            xg_p = W._xg_3way(xg_agent, home.lower().strip(), away.lower().strip())
            preds.append(W.SoccerWalkForwardRow(
                date=row.date, league=lg, home=home, away=away, result=row.result,
                elo_ph=elo_p[0] if elo_p else None, elo_pd=elo_p[1] if elo_p else None,
                elo_pa=elo_p[2] if elo_p else None,
                form_ph=form_p[0] if form_p else None, form_pd=form_p[1] if form_p else None,
                form_pa=form_p[2] if form_p else None,
                poisson_ph=pois_p[0] if pois_p else None, poisson_pd=pois_p[1] if pois_p else None,
                poisson_pa=pois_p[2] if pois_p else None,
                xg_ph=xg_p[0] if xg_p else None, xg_pd=xg_p[1] if xg_p else None,
                xg_pa=xg_p[2] if xg_p else None,
                sharp_ph=row.true_prob_home, sharp_pd=row.true_prob_draw, sharp_pa=row.true_prob_away,
            ))
            W._elo_update(elo_state, "soccer", home, away, row)
            W._form_update(form, "soccer", home, away, row)
            W._poisson_update(pstate, "soccer", home, away, row)
            W._xg_update(xg_agent, row)
        finally:
            W.HOME_ADVANTAGE_ELO["soccer"] = default_hfa
        if row.home_goals is not None and row.away_goals is not None:
            t = goal_tally[lg]
            t[0] += row.home_goals; t[1] += row.away_goals; t[2] += 1
    return preds


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def brier3(p, result: str) -> float:
    ph, pd, pa = p
    return (ph - (result == "H")) ** 2 + (pd - (result == "D")) ** 2 + (pa - (result == "A")) ** 2


def blended(r: W.SoccerWalkForwardRow) -> Optional[tuple[float, float, float]]:
    side = W._model_side_blend(r)
    if side is None or r.sharp_ph is None:
        return None
    mh, md, ma = side
    sw = EnsembleModelAgent._disagreement_sharp_weight(
        mh, ma, md, r.sharp_ph, r.sharp_pa, r.sharp_pd,
        base_sharp_weight=W.base_sharp_weight_for(r), params=SECTOR_RAMP,
    )
    mw = 1 - sw
    ph, pd, pa = sw * r.sharp_ph + mw * mh, sw * r.sharp_pd + mw * md, sw * r.sharp_pa + mw * ma
    s = ph + pd + pa
    return (ph / s, pd / s, pa / s) if s > 1e-9 else None


def poisson_of(r):
    return (r.poisson_ph, r.poisson_pd, r.poisson_pa) if r.poisson_ph is not None else None


def elo_of(r):
    return (r.elo_ph, r.elo_pd, r.elo_pa) if r.elo_ph is not None else None


def paired(base: list, cand: list, getter: Callable, which: str, league: Optional[str] = None):
    """Mean Brier of base & cand plus paired Δ (cand − base) with z. Rows are
    aligned by (date, home, away); only rows where BOTH fire are scored."""
    idx = {(r.date, r.home, r.away): r for r in cand}
    diffs, bsum, csum = [], 0.0, 0.0
    for rb in base:
        if not window(rb, which) or (league and rb.league != league):
            continue
        rc = idx.get((rb.date, rb.home, rb.away))
        pb, pc = getter(rb), (getter(rc) if rc else None)
        if pb is None or pc is None:
            continue
        b, c = brier3(pb, rb.result), brier3(pc, rb.result)
        bsum += b; csum += c; diffs.append(c - b)
    n = len(diffs)
    if n == 0:
        return None
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / max(n - 1, 1)
    z = m / math.sqrt(var / n) if var > 0 else 0.0
    return {"n": n, "base": bsum / n, "cand": csum / n, "delta": m, "z": z}


def fmt(s) -> str:
    if s is None:
        return f"{'—':>44}"
    return (f"n={s['n']:>4}  base {s['base']:.5f}  cand {s['cand']:.5f}  "
            f"Δ {s['delta']*1000:+6.2f}/1000  z {s['z']:+5.2f}")


LEAGUE_ORDER = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "MLS"]


def report(title: str, base, cand, getter, leagues) -> None:
    print(f"\n=== {title} ===")
    for which in ("train", "holdout"):
        print(f"  [{which}]")
        print(f"    {'ALL':<11}{fmt(paired(base, cand, getter, which))}")
        for lg in leagues:
            print(f"    {lg:<11}{fmt(paired(base, cand, getter, which, lg))}")


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hfa", action="store_true")
    args = ap.parse_args()

    rows = load_rows()
    leagues = [lg for lg in LEAGUE_ORDER if any(r.league == lg for r in rows)]
    print(f"{len(rows)} rows across {leagues}")
    base = walk_forward(rows)

    # ---- A. per-league Poisson league_avg ---------------------------------
    cand = walk_forward(rows, poisson_per_league=True)
    # what the running averages converged to, for the record
    tally: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        if r.home_goals is not None and window(r, "train"):
            t = tally[r.league]; t[0] += r.home_goals; t[1] += r.away_goals; t[2] += 1
    print("\nper-league goals/game on the train window (home / away):")
    for lg in leagues:
        h, a, n = tally[lg]
        if n:
            print(f"  {lg:<11}{h/n:.3f} / {a/n:.3f}   (n={n})")
    d = W.LEAGUE_AVG_DEFAULTS.get("soccer")
    print(f"  soccer-wide default used by the baseline: {d}")
    report("A. Poisson per-league league_avg — POISSON standalone 3-way Brier", base, cand, poisson_of, leagues)
    report("A. Poisson per-league league_avg — BLENDED 3-way Brier", base, cand, blended, leagues)

    if args.skip_hfa:
        return

    # ---- B. per-league Elo HFA -----------------------------------------------
    print("\n=== B. Elo HFA sweep — ELO standalone train Brier per league ===")
    runs = {h: walk_forward(rows, hfa_by_league={lg: h for lg in leagues}) for h in HFA_GRID}
    best: dict[str, float] = {}
    print(f"  {'league':<11}" + "".join(f"{h:>9.0f}" for h in HFA_GRID) + "   best")
    for lg in leagues:
        cells = []
        for h in HFA_GRID:
            s = paired(base, runs[h], elo_of, "train", lg)
            cells.append(s["cand"] if s else float("nan"))
        b = HFA_GRID[min(range(len(HFA_GRID)), key=lambda i: cells[i])]
        best[lg] = b
        print(f"  {lg:<11}" + "".join(f"{c:>9.5f}" for c in cells) + f"   {b:.0f}")
    cand_hfa = walk_forward(rows, hfa_by_league=best)
    print(f"\n  train-best HFA per league: {best}  (baseline: soccer-wide {W.HOME_ADVANTAGE_ELO.get('soccer')})")
    report("B. Per-league HFA — ELO standalone 3-way Brier (train is in-sample by construction)", base, cand_hfa, elo_of, leagues)
    report("B. Per-league HFA — BLENDED 3-way Brier", base, cand_hfa, blended, leagues)


if __name__ == "__main__":
    main()
