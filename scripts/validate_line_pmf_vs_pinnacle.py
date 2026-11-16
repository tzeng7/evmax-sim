"""Compare the alternate-line models with Pinnacle's OWN alternate ladder.

Outcomes say whether a distribution is right on average. A sharp book's own
ladder says what the shape is believed to be game by game, and needs no
waiting for results. For every archived game this takes the last pre-tip
snapshot in which Pinnacle posted a main line and alternate rungs, anchors
each curve on the main line's devigged price, and measures how far each
curve's rung probabilities sit from Pinnacle's own.

    python scripts/validate_line_pmf_vs_pinnacle.py --archive-db data/archive.db
    python scripts/validate_line_pmf_vs_pinnacle.py --sectors nba --markets total

Curves compared: the fitted ``LinePMF`` artifact, and the production normal
(the sigma the scanner uses today). Read-only.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evmax.models_ml.line_pmf import load_line_pmf  # noqa: E402

PROD_SIGMA = {("nfl", "spread"): 14.0, ("nba", "spread"): 12.5, ("ncaab", "spread"): 12.5,
              ("ncaaw", "spread"): 11.5, ("nfl", "total"): 14.0, ("nba", "total"): 22.0,
              ("ncaab", "total"): 14.0, ("ncaaw", "total"): 18.0}
BANDS = ((0, 3, "0-3"), (3, 7, "3-7"), (7, 12, "7-12"), (12, 99, "12+"))


def load_total_ladders(db, sector, since):
    """{game: (main_line, p_over_main, [(line, p_over)])} from the last pre-tip sweep."""
    sweeps = defaultdict(dict)
    q = """SELECT event_id, substr(fetched_at, 1, 16), fetched_at, event_date, total_line,
                  true_prob_over FROM archived_sharp_odds
           WHERE sector = ? AND fetched_at >= ? AND book = 'pinnacle'
             AND event_id LIKE '%::total::%' AND total_line IS NOT NULL
             AND true_prob_over IS NOT NULL AND fetched_at < event_date"""
    for eid, minute, fa, ed, line, p in db.execute(q, (sector, since)):
        game = eid.split("::total")[0]
        sweeps[game].setdefault(minute, {})[float(line)] = float(p)
    out = {}
    for game, by_minute in sweeps.items():
        ladder = by_minute[max(by_minute)]
        if len(ladder) < 3:
            continue
        main = min(ladder, key=lambda k: abs(ladder[k] - 0.5))
        out[game] = (main, ladder[main], [(k, v) for k, v in ladder.items() if k != main])
    return out


def load_spread_ladders(db, sector, since):
    """{game: (|main|, p_fav_cover, [(threshold on the favorite margin, P(F > t))])}."""
    mains, rungs = {}, defaultdict(lambda: defaultdict(list))
    q = """SELECT event_id, substr(fetched_at, 1, 16), outcome_a_label, spread_line, true_prob_a
           FROM archived_sharp_odds
           WHERE sector = ? AND fetched_at >= ? AND book = 'pinnacle'
             AND spread_line IS NOT NULL AND true_prob_a IS NOT NULL
             AND fetched_at < event_date AND event_id LIKE '%::spread%'"""
    for eid, minute, la, line, pa in db.execute(q, (sector, since)):
        game = eid.split("::spread")[0]
        if eid.endswith("::spread"):
            if game not in mains or minute > mains[game][0]:
                mains[game] = (minute, la, abs(float(line)), float(pa))
        else:
            rungs[game][minute].append((la, abs(float(line)), float(pa)))
    out = {}
    for game, (minute, fav, a, p) in mains.items():
        ladder = rungs.get(game, {}).get(minute)
        if not ladder or len(ladder) < 2:
            continue
        pts = []
        for la, mag, pa in ladder:
            # outcome_a is the team LAYING the points at that rung.
            if la == fav:
                pts.append((mag, pa))            # P(F > mag)
            else:
                pts.append((-mag, 1.0 - pa))     # underdog lays mag: P(F > −mag) = 1 − P(F < −mag)
        out[game] = (a, p, pts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-db", default="data/archive.db")
    ap.add_argument("--sectors", default="nfl,nba,ncaab")
    ap.add_argument("--markets", default="spread,total")
    ap.add_argument("--since", default="2026-09-01")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.archive_db}?mode=ro", uri=True)
    for sector in args.sectors.split(","):
        for market in args.markets.split(","):
            model = load_line_pmf(sector, market)
            if model is None:
                print(f"\n{sector} {market}: no artifact, skipped")
                continue
            sigma = PROD_SIGMA[(sector, market)]
            ladders = (load_total_ladders if market == "total" else load_spread_ladders)(
                db, sector, args.since)
            per_game = {"model": [], "normal": []}
            bands = defaultdict(lambda: {"model": [], "normal": []})
            for game, (a, p, pts) in ladders.items():
                mu = model.anchor(a, p)
                if mu is None:
                    continue
                mu_n = a + sigma * float(ndtri(p))     # normal with P(X > a) = p
                em, en = [], []
                for t, p_book in pts:
                    pm = model.conditional_survival(mu, t, a)
                    pn = 1.0 - float(ndtr((t - mu_n) / sigma))
                    dm, dn = abs(pm - p_book) * 100.0, abs(pn - p_book) * 100.0
                    em.append(dm)
                    en.append(dn)
                    d = abs(t - a)
                    for lo, hi, name in BANDS:
                        if lo <= d < hi:
                            bands[name]["model"].append(dm)
                            bands[name]["normal"].append(dn)
                if em:
                    per_game["model"].append(float(np.mean(em)))
                    per_game["normal"].append(float(np.mean(en)))
            n = len(per_game["model"])
            if n < 5:
                print(f"\n{sector} {market}: {n} games with a Pinnacle ladder, too few")
                continue
            m, b = np.array(per_game["model"]), np.array(per_game["normal"])
            d = m - b
            z = d.mean() / (d.std(ddof=1) / math.sqrt(n)) if d.std() > 0 else float("nan")
            print(f"\n{sector} {market}: {n} games with a Pinnacle ladder | mean abs error vs "
                  f"Pinnacle's own rung price: model {m.mean():.2f}pp | production normal "
                  f"(sigma {sigma}) {b.mean():.2f}pp | model closer in "
                  f"{(d < 0).mean() * 100:.0f}% of games (z {z:+.1f})")
            for _, _, name in BANDS:
                v = bands.get(name)
                if v and v["model"]:
                    print(f"     rungs {name:>5} pts from the main line (n={len(v['model'])}): "
                          f"model {np.mean(v['model']):.2f}pp | normal {np.mean(v['normal']):.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
