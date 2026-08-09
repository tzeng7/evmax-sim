#!/usr/bin/env python3
"""Joint blend-weight sweep for the NCAAF model ensemble.

Grounds the SECTOR_WEIGHT_OVERRIDES["ncaaf"] = ncaaf_efficiency 0.55 · elo 0.25 ·
form 0.20 weights, which shipped as PHASE-2-INITIAL placeholders never jointly
swept. It only became meaningful once generic elo was WARM-SEEDED
(scripts/seed_ncaaf_elo.py): cold elo is gated (dark) for each team's first 5
games, so a pre-warm-seed blend sweep would have measured a blend where elo
barely fires early. This harness rebuilds that warm seed leak-free per holdout.

WHAT IT DOES (all three models are faithful DIRECT REPLAYS of the live agents,
the same methodology as backtest_nfl_efficiency.py / backtest_ncaab_efficiency.py):

  * ncaaf_efficiency — reuses run_season() from backtest_ncaaf_efficiency.py
    (opponent-adjusted EPA + preseason-prior ramp, leak-free per-week rebuild).
    Confidence from the agent's gp_min ramp (>=9→0.85, >=6→0.72, >=3→0.60, else
    0.52). Fires on every FBS-vs-FBS game (prior seeded).
  * elo — warm seed built from seasons < holdout via seed_ncaaf_elo.build_seed
    (bare K=40/home_adv=60, regressed carry-forward), then walk-forward over the
    holdout. Gate replicates EloModelAgent: fires only at min lifetime count >=5
    (confidence 0.60/0.80); below that it is DARK, exactly as in production.
  * form — faithful replay of FormModelAgent (WINDOW=10, DECAY=0.85, shrinkage
    n/(n+6), log5, home_adj 0.03, MIN_GAMES=3, 60-day staleness → dark early).

Each game's blend replicates EnsembleModelAgent: effective weight = base_weight ×
confidence, over the models that FIRE, normalized. Gated/dark models drop out.
Scores are BLENDED-Brier (models only, no sharp) on the held-out season — the
sharp line is a separate ~0.85 layer and beating it is not the goal (the market
column is a calibration reference, per the value-audit philosophy).

PROTOCOL: sweep the weight simplex on --sweep-season, then report the ranked
winner + the CURRENT shipped weights + individual models on the held-out
--holdout season. Two seasons of play-by-play fetch (disk-cached after run 1).

    python scripts/backtest_ncaaf_blend.py --sweep-season 2024 --holdout 2025
    python scripts/backtest_ncaaf_blend.py --sweep-season 2024 --holdout 2025 --detail
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from evmax.matching.normalizer import NameNormalizer  # noqa: E402
from scripts.backtest_ncaaf_efficiency import run_season, _brier, BUCKETS  # noqa: E402
from scripts.backtest_ncaaf_elo import RawElo, fetch_season  # noqa: E402
from scripts.seed_ncaaf_elo import PROD_HOME, PROD_K, build_seed  # noqa: E402

# Shipped weights under test.
CURRENT = {"eff": 0.55, "elo": 0.25, "form": 0.20}

# Efficiency default params (mirror backtest_ncaaf_efficiency.main defaults).
EFF_PARAMS = dict(ramp_k=3.0, regress=0.5, ridge=1.0, plays_per=70.0,
                  stdev=16.5, home_edge=2.5, spread_sigma=16.3)

# Form constants (mirror evmax/agents/models/form_agent.py).
F_WINDOW, F_DECAY, F_PRIOR_N, F_MIN, F_STALE_D, F_HOME_ADJ = 10, 0.85, 6.0, 3, 60, 0.03

_NZ = NameNormalizer("ncaaf")


def _norm(x: str) -> str:
    return _NZ.normalize(x) or (x or "").lower().strip()


def _key(date: str, home: str, away: str) -> tuple[str, str, str]:
    return (date[:10], _norm(home), _norm(away))


# ---------------------------------------------------------------------------
# Model replays over the holdout season → {game_key: (prob_home, confidence)}
# A missing key means the model was DARK (gated) for that game.
# ---------------------------------------------------------------------------
def _elo_conf(min_count: int) -> Optional[float]:
    if min_count < 5:      # 0.30 (0 games) or 0.45 (1-4) → both <= 0.45 gate → dark
        return None
    return 0.60 if min_count < 15 else 0.80


def elo_probs(train_seasons: list[int], holdout: int, regress: float) -> dict:
    ratings, counts0 = build_seed(train_seasons, regress=regress,
                                  between_season_regress=True, verbose=False)
    elo = RawElo(k=PROD_K, home_adv=PROD_HOME)
    elo.ratings = dict(ratings)
    counts: dict[str, int] = defaultdict(int, counts0)
    out: dict = {}
    for g in fetch_season(holdout):          # chronological, FBS games (groups=80)
        h, a = _norm(g["home"]), _norm(g["away"])
        conf = _elo_conf(min(counts[h], counts[a]))
        if conf is not None:
            p = elo.expected_home(h, a, g["neutral"])
            out[(g["date"][:10], h, a)] = (p, conf)
        elo.update(h, a, g["home_won"], g["neutral"])
        counts[h] += 1
        counts[a] += 1
    return out


def _form_rate(recs: list[dict]) -> float:
    recent = sorted(recs, key=lambda r: r["date"], reverse=True)[:F_WINDOW]
    if not recent:
        return 0.5
    wp = tw = 0.0
    for i, rec in enumerate(recent):
        w = F_DECAY ** i
        wp += w * (1.0 if rec["won"] else 0.0)
        tw += w
    raw = wp / tw if tw else 0.5
    n = len(recent)
    shr = n / (n + F_PRIOR_N)
    return raw * shr + 0.5 * (1.0 - shr)


def _log5(pa: float, pb: float) -> float:
    d = pa + pb - 2.0 * pa * pb
    return 0.5 if d < 1e-9 else (pa - pa * pb) / d


def form_probs(holdout: int) -> dict:
    """Walk-forward form replay over the holdout season (prior-season tail is
    always >60d stale by opening week, so holdout games alone drive it)."""
    recs: dict[str, list[dict]] = defaultdict(list)
    out: dict = {}
    for g in fetch_season(holdout):
        h, a = _norm(g["home"]), _norm(g["away"])
        gd = g["date"][:10]
        rh, ra = recs[h], recs[a]
        fires = len(rh) >= F_MIN and len(ra) >= F_MIN
        if fires:
            ref = dt.date.fromisoformat(gd)
            stale = (any((ref - dt.date.fromisoformat(max(r["date"] for r in rr))).days > F_STALE_D
                         for rr in (rh, ra)))
            if not stale:
                pa = _log5(_form_rate(rh), _form_rate(ra))
                pa = min(0.95, max(0.05, pa + F_HOME_ADJ))
                sample = min(len(rh), len(ra))
                conf = 0.5 + 0.3 * min(1.0, (sample - F_MIN) / (F_WINDOW - F_MIN))
                out[(gd, h, a)] = (pa, conf)
        recs[h].append({"date": gd, "won": bool(g["home_won"])})
        recs[a].append({"date": gd, "won": not bool(g["home_won"])})
    return out


def _eff_conf(gp_min: int) -> float:
    if gp_min >= 9:
        return 0.85
    if gp_min >= 6:
        return 0.72
    if gp_min >= 3:
        return 0.60
    return 0.52


# ---------------------------------------------------------------------------
# Assemble per-game model table, then sweep blend weights over it
# ---------------------------------------------------------------------------
def build_table(holdout: int, train_seasons: list[int], regress: float) -> list[dict]:
    eff_rows = run_season(holdout, [holdout - 1], EFF_PARAMS)
    elo = elo_probs(train_seasons, holdout, regress)
    form = form_probs(holdout)
    table, joined_elo, joined_form = [], 0, 0
    for r in eff_rows:
        k = _key(r["date"], r["home"], r["away"])
        e = elo.get(k)
        f = form.get(k)
        if e:
            joined_elo += 1
        if f:
            joined_form += 1
        table.append({
            "y": r["y"], "bucket": r["bucket"], "week": r["week"],
            "market": r["market"],
            "eff": (r["model"], _eff_conf(r["gp_min"])),
            "elo": e, "form": f,
        })
    print(f"  [{holdout}] games={len(table)} · elo fired {joined_elo} "
          f"({100*joined_elo/max(1,len(table)):.0f}%) · form fired {joined_form} "
          f"({100*joined_form/max(1,len(table)):.0f}%)", file=sys.stderr)
    return table


def blend_brier(table: list[dict], w: dict, bucket: Optional[str] = None) -> tuple[float, float, int]:
    sb = corr = n = 0
    for row in table:
        if bucket and row["bucket"] != bucket:
            continue
        num = den = 0.0
        for m in ("eff", "elo", "form"):
            mp = row[m]
            if mp is None or w[m] <= 0:
                continue
            prob, conf = mp
            ew = w[m] * conf
            num += ew * prob
            den += ew
        if den <= 0:
            continue
        p = num / den
        sb += (p - row["y"]) ** 2
        corr += int((p >= 0.5) == bool(row["y"]))
        n += 1
    if n == 0:
        return float("nan"), float("nan"), 0
    return sb / n, corr / n, n


def _single_brier(table: list[dict], m: str) -> tuple[float, float, int]:
    """Standalone Brier for one model over the games where it fires."""
    sb = corr = n = 0
    for row in table:
        mp = row[m] if m != "market" else (row["market"], 1.0) if row["market"] is not None else None
        if mp is None:
            continue
        p = mp[0]
        sb += (p - row["y"]) ** 2
        corr += int((p >= 0.5) == bool(row["y"]))
        n += 1
    return (sb / n, corr / n, n) if n else (float("nan"), float("nan"), 0)


def sweep_grid() -> list[dict]:
    effs = [0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    elos = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    forms = [0.0, 0.10, 0.20, 0.30]
    out = []
    seen = set()
    for e, l, f in product(effs, elos, forms):
        tot = e + l + f
        if tot <= 0:
            continue
        w = (round(e / tot, 3), round(l / tot, 3), round(f / tot, 3))
        if w in seen:
            continue
        seen.add(w)
        out.append({"eff": e, "elo": l, "form": f})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-season", type=int, default=2024)
    ap.add_argument("--holdout", type=int, default=2025)
    ap.add_argument("--regress", type=float, default=0.90, help="elo warm-seed regression")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    sweep_train = [s for s in range(2021, args.sweep_season)]
    hold_train = [s for s in range(2021, args.holdout)]

    print(f"Building SWEEP table (season {args.sweep_season}) …", file=sys.stderr)
    sweep_tbl = build_table(args.sweep_season, sweep_train, args.regress)
    print(f"Building HOLDOUT table (season {args.holdout}) …", file=sys.stderr)
    hold_tbl = build_table(args.holdout, hold_train, args.regress)

    # --- sweep on sweep-season ---
    grid = sweep_grid()
    ranked = []
    for w in grid:
        b, a, n = blend_brier(sweep_tbl, w)
        ranked.append((b, a, n, w))
    ranked.sort(key=lambda x: x[0])

    cur_b, cur_a, cur_n = blend_brier(sweep_tbl, CURRENT)
    print(f"\n=== SWEEP season {args.sweep_season} (n={cur_n}) — ranked by blended Brier ===")
    print(f"{'rank':>4} {'eff':>5}{'elo':>5}{'form':>5}  {'Brier':>8}{'Acc':>7}")
    for i, (b, a, n, w) in enumerate(ranked[:12], 1):
        star = "  <= CURRENT" if (w["eff"], w["elo"], w["form"]) == (0.55, 0.25, 0.20) else ""
        print(f"{i:>4} {w['eff']:>5}{w['elo']:>5}{w['form']:>5}  {b:>8.4f}{100*a:>6.1f}%{star}")
    # current's rank
    cur_rank = 1 + sum(1 for b, _, _, _ in ranked if b < cur_b)
    print(f"\nCURRENT 0.55/0.25/0.20 → Brier {cur_b:.4f} / Acc {100*cur_a:.1f}%  "
          f"(rank {cur_rank}/{len(ranked)})")

    best = ranked[0][3]

    # --- report best + current + individuals on HOLDOUT ---
    print(f"\n=== HELD-OUT season {args.holdout} ===")
    print(f"{'config':<34}{'Brier':>9}{'Acc':>7}{'n':>6}")
    for label, w in [
        (f"BEST-on-sweep {best['eff']}/{best['elo']}/{best['form']}", best),
        ("CURRENT 0.55/0.25/0.20", CURRENT),
    ]:
        b, a, n = blend_brier(hold_tbl, w)
        print(f"{label:<34}{b:>9.4f}{100*a:>6.1f}%{n:>6}")
    print(f"{'-'*56}")
    for m, lbl in [("eff", "ncaaf_efficiency (standalone)"),
                   ("elo", "warm elo (standalone)"),
                   ("form", "form (standalone)"),
                   ("market", "market close (reference)")]:
        b, a, n = _single_brier(hold_tbl, m)
        bs = "   -   " if b != b else f"{b:.4f}"
        as_ = "" if a != a else f"{100*a:.1f}%"
        print(f"{lbl:<34}{bs:>9}{as_:>7}{n:>6}")

    if args.detail:
        print(f"\n=== HELD-OUT {args.holdout} by week bucket (CURRENT vs BEST) ===")
        print(f"{'bucket':<8}{'CUR Brier':>11}{'BEST Brier':>12}{'n':>6}")
        for name, _, _ in BUCKETS:
            cb, _, cn = blend_brier(hold_tbl, CURRENT, bucket=name)
            bb, _, _ = blend_brier(hold_tbl, best, bucket=name)
            if cn:
                print(f"{name:<8}{cb:>11.4f}{bb:>12.4f}{cn:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
