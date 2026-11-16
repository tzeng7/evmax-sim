"""Iteration 6: ENTRY ensemble on the confirmed §3 spread-lay convergence edge.

Uses listings_eval AnchoredEntry (correct orientation, validated in iter2). Tests
whether stacking entry signals — side=lay, contested(|fair-0.5|<=.15), near-tip
(lead<=12h), depth>=$ — concentrates Kalshi CLV monotonically vs any single signal.
Declustered to game level (honest t). Cross-checked on baseball (the high-power sector).
"""
from __future__ import annotations
import math, re, statistics as st
from evmax.agents.cleanup.listings_eval import evaluate_sector
from evmax.archiver import _get_connection as get_conn

def collect(sector):
    with get_conn() as c:
        s = evaluate_sector(c, sector, ev_min_pp=0.0, depth_min_usd=0.0,
                            market_types=("spread","total"))
    return [e for e in s.entries if e.clv_pp is not None]

def gamekey(e):
    m = re.match(r'KX[A-Z]+-([0-9A-Z]+?[A-Z]{2,})-', e.ticker)
    return m.group(1) if m else e.ticker
def declust(es):
    g = {}
    for e in es: g.setdefault(gamekey(e), []).append(e.clv_pp)
    return [st.mean(v) for v in g.values()]
def t(xs):
    return st.mean(xs)/(st.pstdev(xs)/math.sqrt(len(xs))) if len(xs)>1 and st.pstdev(xs) else 0.0
def line(tag, es):
    g = declust(es)
    if not g: print(f"  {tag:52} games=0"); return
    star = "  <==" if st.mean(g)>0 and t(g)>1.65 and len(g)>=30 else ""
    print(f"  {tag:52} games={len(g):4}  meanCLV={st.mean(g):+.2f}pp  t={t(g):+.2f}{star}")

for sec in ("wnba","baseball"):
    print(f"\n=== {sec}: nested entry-signal stack (spread) ===")
    es = collect(sec)
    lay = [e for e in es if e.market_type=="spread" and e.side=="lay"]
    cont = [e for e in lay if abs(e.fair_prob-0.5)<=0.15]
    ct   = [e for e in cont if e.entry_lead_h<=12]
    ctd  = [e for e in ct if e.entry_depth_usd>=100]
    line("spread/lay (base)", lay)
    line("  + contested", cont)
    line("  + contested + near-tip(<=12h)", ct)
    line("  + contested + near-tip + depth>=$100", ctd)
    # falsification: the anti-stack should be weak/negative
    line("[anti] lay + LOPSIDED + far(>24h)", [e for e in lay if abs(e.fair_prob-0.5)>0.15 and e.entry_lead_h>24])
    # is the stack better than each single signal alone?
    line("[single] lay + near-tip only", [e for e in lay if e.entry_lead_h<=12])
    line("[single] lay + depth>=$100 only", [e for e in lay if e.entry_depth_usd>=100])
