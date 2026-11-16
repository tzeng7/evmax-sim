"""Slate size vs Kalshi-vs-Pinnacle gap (pre-registered: docs/attention-softness-prereg.md).

Read-only against archive.db. Usage: python scripts/exp_slate_size_gap.py [sectors...]
"""
from __future__ import annotations

import random
import re
import sqlite3
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DB = "file:data/archive.db?mode=ro"
SEED, REPS = 20261027, 2000
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
NBA_REG_END, NBA_REG_START = "2026-04-12", "2026-10-21"


def parse_iso(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def ticker_date(t: str):
    m = re.search(r"-(\d\d)([A-Z]{3})(\d\d)", t)
    if not m:
        return None
    return f"20{m.group(1)}-{MONTHS[m.group(2)]:02d}-{int(m.group(3)):02d}"


def norm(s: str) -> str:
    return s.replace("_", " ").strip().lower()


def load(sector: str):
    cx = sqlite3.connect(DB, uri=True)
    pin = cx.execute(
        """SELECT session_id,event_id,event_date,true_prob_a,true_prob_b,fetched_at FROM archived_sharp_odds
           WHERE sector=? AND book='pinnacle' AND spread_line IS NULL AND total_line IS NULL
             AND prop_player_name IS NULL AND true_prob_draw IS NULL""", (sector,)).fetchall()
    kal = cx.execute(
        """SELECT session_id,ticker,yes_team,yes_price,fetched_at FROM archived_kalshi_markets
           WHERE sector=? AND market_type='moneyline'""", (sector,)).fetchall()
    return pin, kal


WINDOW = {"lo": 1.5, "hi": 6.0, "target": 3.0}


def run(sector: str):
    pin, kal = load(sector)
    # pinnacle: (session, team, et_date) -> (prob, event_key, tip)
    pmap, events = {}, defaultdict(set)
    for sess, eid, ed, pa, pb, _ in pin:
        parts = eid.split("::")
        if len(parts) != 3 or "_vs_" not in parts[2]:
            continue
        a, b = (norm(x) for x in parts[2].split("_vs_"))
        tip = parse_iso(ed)
        etd = tip.astimezone(ET).strftime("%Y-%m-%d")
        events[etd].add(eid)
        pmap[(sess, a, etd)] = (pa, eid, tip)
        pmap[(sess, b, etd)] = (pb, eid, tip)
    # kalshi markets: best snapshot closest to 3h before tip within [1.5, 6]
    best = {}
    for sess, tk, yt, yp, fa in kal:
        d = ticker_date(tk)
        if not d or not yt:
            continue
        hit = pmap.get((sess, norm(yt), d))
        if not hit:
            continue
        prob, eid, tip = hit
        h = (tip - parse_iso(fa)).total_seconds() / 3600
        if not WINDOW["lo"] <= h <= WINDOW["hi"] or not 0.15 <= yp <= 0.85:
            continue
        key = (tk, eid)
        dist = abs(h - WINDOW["target"])
        if key not in best or dist < best[key][0]:
            best[key] = (dist, prob, yp, d, eid)
    rows = [{"gap": 100 * (p - y), "date": d, "eid": e} for _, p, y, d, e in best.values()]
    n_kal = len({tk for tk, *_ in ((k[0],) for k in best)})
    for r in rows:
        n = len(events[r["date"]])
        r["games"] = n
        r["bucket"] = "light" if n <= 4 else "mid" if n <= 9 else "heavy"
        r["phase"] = ("reg" if sector != "nba" or r["date"] <= NBA_REG_END or r["date"] >= NBA_REG_START else "playoff")
    return rows, len(kal)


def boot_diff(rows, a, b, seed=SEED):
    """Clustered (by game-day) bootstrap of mean|gap|(b) - mean|gap|(a)."""
    by = defaultdict(list)
    for r in rows:
        by[r["date"]].append(r)
    days = list(by)
    rnd = random.Random(seed)

    def stat(sample):
        ga = [abs(r["gap"]) for d in sample for r in by[d] if r["bucket"] == a]
        gb = [abs(r["gap"]) for d in sample for r in by[d] if r["bucket"] == b]
        return (st.fmean(gb) - st.fmean(ga)) if ga and gb else None

    pt = stat(days)
    vals = sorted(v for v in (stat([rnd.choice(days) for _ in days]) for _ in range(REPS)) if v is not None)
    if not vals or pt is None:
        return pt, None
    return pt, (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def summarize(sector, rows, label=""):
    print(f"\n== {sector} {label} ==")
    print(f"{'bucket':7}{'mkts':>6}{'games':>6}{'days':>5}{'mean|gap|':>10}{'mean edge':>10}{'edge>=2pp':>10}")
    stats = {}
    for b in ("light", "mid", "heavy"):
        g = [r for r in rows if r["bucket"] == b]
        if not g:
            print(f"{b:7}{0:>6}")
            continue
        games = len({r["eid"] for r in g})
        stats[b] = games
        print(f"{b:7}{len(g):>6}{games:>6}{len({r['date'] for r in g}):>5}"
              f"{st.fmean(abs(r['gap']) for r in g):>10.2f}{st.fmean(r['gap'] for r in g):>10.2f}"
              f"{100*sum(r['gap']>=2 for r in g)/len(g):>9.0f}%")
    if "light" in stats and "heavy" in stats:
        pt, ci = boot_diff(rows, "light", "heavy")
        ok = min(stats["light"], stats["heavy"]) >= 30
        print(f"heavy - light mean|gap| = {pt:+.2f}pp  CI95 {None if ci is None else (round(ci[0],2), round(ci[1],2))}"
              f"  {'' if ok else '[UNDERPOWERED: a bucket <30 games]'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    for flag in ("lo", "hi", "target"):
        if f"--{flag}" in args:
            i = args.index(f"--{flag}")
            WINDOW[flag] = float(args[i + 1])
            del args[i:i + 2]
    print("window (h before tip):", WINDOW)
    for sec in (args or ["nba", "baseball", "nhl", "wnba", "nfl"]):
        rows, nk = run(sec)
        print(f"\n#### {sector if (sector:=sec) else ''}: {len(rows)} paired markets from {nk} kalshi snapshot rows")
        if not rows:
            continue
        if sec == "nba":
            summarize(sec, [r for r in rows if r["phase"] == "reg"], "(regular season only)")
            po = [r for r in rows if r["phase"] == "playoff"]
            if po:
                print(f"playoffs: {len(po)} mkts, {len({r['eid'] for r in po})} games, mean|gap| {st.fmean(abs(r['gap']) for r in po):.2f}pp, mean edge {st.fmean(r['gap'] for r in po):+.2f}pp")
        else:
            summarize(sec, rows)
