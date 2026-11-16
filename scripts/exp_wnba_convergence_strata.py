"""Iteration 2 (§3): does Kalshi->sharp convergence CLV have an exploitable STRATUM?

Reuses the validated first-anchored-sweep replay (listings_eval) so anchor/close/
sign handling is correct. Opens the EV/depth gates (ev_min=0) to get ALL anchored
strikes, then stratifies Kalshi entry->close CLV by:
  side (lay/take, over/under) x contested(|fair-0.5|<=.15) vs tail x entry lead.
Verdict: a stratum is a CAPABILITY only if mean CLV>0, t>1.65, n>=30, and the
by-date holdout keeps the sign. Pools wnba spread+total; also prints a pooled
cross-sector (league-FE-style) convergence read.
"""
from __future__ import annotations
import math, statistics as st
from evmax.agents.cleanup.listings_eval import evaluate_sector
from evmax.archiver import _get_connection as get_conn

def tstat(xs):
    if len(xs) < 2: return 0.0
    m, sd = st.mean(xs), st.pstdev(xs)
    return m / (sd/math.sqrt(len(xs))) if sd else 0.0

def report(tag, entries):
    xs = [e.clv_pp for e in entries if e.clv_pp is not None]
    if not xs:
        print(f"  {tag:42} n=0"); return
    m = st.mean(xs); pos = sum(1 for v in xs if v > 0)/len(xs)
    flag = "  <== +CLV" if (m > 0 and tstat(xs) > 1.65 and len(xs) >= 30) else ""
    print(f"  {tag:42} n={len(xs):5}  mean={m:+.2f}pp  %pos={pos:4.0%}  t={tstat(xs):+.2f}{flag}")

def collect(sector, mts=("spread","total")):
    with get_conn() as c:
        stats = evaluate_sector(c, sector, ev_min_pp=0.0, depth_min_usd=0.0,
                                market_types=mts)
    return [e for e in stats.entries if e.clv_pp is not None]

def contested(e):  # near pick'em = highest-density strike
    return abs(e.fair_prob - 0.5) <= 0.15

wnba = collect("wnba")
print(f"=== WNBA anchored strikes with CLV: n={len(wnba)} ===")

print("\n-- by side --")
for mt in ("spread","total"):
    for side in (("lay","take") if mt=="spread" else ("over","under")):
        report(f"{mt}/{side}", [e for e in wnba if e.market_type==mt and e.side==side])

print("\n-- by side x contested/tail --")
for mt in ("spread","total"):
    for side in (("lay","take") if mt=="spread" else ("over","under")):
        sub = [e for e in wnba if e.market_type==mt and e.side==side]
        report(f"{mt}/{side} CONTESTED", [e for e in sub if contested(e)])
        report(f"{mt}/{side} tail",      [e for e in sub if not contested(e)])

print("\n-- best-looking stratum: spread/lay contested, by entry lead --")
sl = [e for e in wnba if e.market_type=="spread" and e.side=="lay" and contested(e)]
report("spread/lay contested  lead>12h", [e for e in sl if e.entry_lead_h > 12])
report("spread/lay contested  lead<=12h", [e for e in sl if e.entry_lead_h <= 12])

print("\n-- date holdout on spread/lay contested (train first 70% by entry_at) --")
sl.sort(key=lambda e: e.entry_at)
cut = int(len(sl)*0.7)
report("spread/lay contested TRAIN", sl[:cut])
report("spread/lay contested HOLDOUT", sl[cut:])

print("\n=== POOLED cross-sector convergence (structural? league-FE-style) ===")
for sec in ("wnba","nba","baseball"):
    try:
        es = collect(sec)
        report(f"{sec} all anchored strikes", es)
        report(f"{sec} spread/lay contested",
               [e for e in es if e.market_type=="spread" and e.side=="lay" and contested(e)])
    except Exception as ex:  # noqa: BLE001
        print(f"  {sec}: ERR {ex}")

# ================= DECLUSTER to game level (honest significance) =================
import re
def gamekey(e):
    m = re.match(r'KX[A-Z]+-([0-9A-Z]+?[A-Z]{2,})-', e.ticker)  # date+teams block
    return (e.sector, m.group(1) if m else e.ticker)
def declustered(entries):
    by_g = {}
    for e in entries:
        if e.clv_pp is None: continue
        by_g.setdefault(gamekey(e), []).append(e.clv_pp)
    gvals = [st.mean(v) for v in by_g.values()]  # one CLV per game
    return gvals
print("\n=== DECLUSTERED (one mean-CLV per GAME) ===")
for tag, es in [
    ("wnba spread/lay contested", [e for e in wnba if e.market_type=="spread" and e.side=="lay" and contested(e)]),
    ("wnba spread/lay ALL",       [e for e in wnba if e.market_type=="spread" and e.side=="lay"]),
]:
    g = declustered(es)
    print(f"  {tag:34} games={len(g):3}  mean={st.mean(g):+.2f}pp  t={tstat(g):+.2f}"
          + ("  <== +CLV" if st.mean(g)>0 and tstat(g)>1.65 and len(g)>=30 else ""))
try:
    bb = collect("baseball")
    for tag, es in [
        ("baseball spread/lay contested", [e for e in bb if e.market_type=="spread" and e.side=="lay" and contested(e)]),
        ("baseball spread/lay ALL",       [e for e in bb if e.market_type=="spread" and e.side=="lay"]),
    ]:
        g = declustered(es)
        print(f"  {tag:34} games={len(g):3}  mean={st.mean(g):+.2f}pp  t={tstat(g):+.2f}"
              + ("  <== +CLV" if st.mean(g)>0 and tstat(g)>1.65 and len(g)>=30 else ""))
except Exception as ex:
    print("  baseball decluster ERR", ex)
