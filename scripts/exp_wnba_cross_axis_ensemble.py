"""Iteration 7: CROSS-AXIS ensemble — does §1 delta (fair-value) improve §3 entry CLV?

§1 delta was too fragile to price spreads standalone. But it may still work as an
ENTRY FILTER: lay the favorite's spread only when the ML blend AGREES the favorite is
stronger than sharp (delta_fav > 0). Tests whether delta-agreement concentrates the
confirmed spread-lay convergence CLV beyond contested/near-tip alone.

delta_fav = sigma*(Phi^-1(p_blend_fav) - Phi^-1(p_sharp_fav)); fav = lay-side team
(the -X.5 line, = outcome_label team). Declustered to game level.
"""
from __future__ import annotations
import math, re, statistics as st, sqlite3
from pathlib import Path
from evmax.agents.cleanup.listings_eval import evaluate_sector
from evmax.archiver import _get_connection as get_conn

REPO = Path("/Users/ktzeng/Projects/evmax"); SIGMA=12.5
def phinv(p):
    p=min(max(p,1e-4),1-1e-4); lo,hi=-8.0,8.0
    for _ in range(80):
        m=(lo+hi)/2
        if 0.5*(1+math.erf(m/math.sqrt(2)))<p: lo=m
        else: hi=m
    return (lo+hi)/2
def lt(s): return (s or "").strip().lower().split()[-1] if s else ""
def t(xs): return st.mean(xs)/(st.pstdev(xs)/math.sqrt(len(xs))) if len(xs)>1 and st.pstdev(xs) else 0.0

# delta per (date, team_token) from predictions.db ML rows
con=sqlite3.connect(str(REPO/"data/predictions.db")); con.row_factory=sqlite3.Row
rows=con.execute("""SELECT p.event_date,p.yes_team,p.blended_true_prob pb,p.sharp_true_prob ps
  FROM ev_predictions p JOIN (SELECT market_id,MAX(scan_date) md FROM ev_predictions
    WHERE sector='wnba' AND market_type='moneyline' GROUP BY market_id) x
    ON x.market_id=p.market_id AND x.md=p.scan_date
  WHERE p.sector='wnba' AND p.market_type='moneyline'
    AND p.blended_true_prob IS NOT NULL AND p.sharp_true_prob IS NOT NULL""").fetchall(); con.close()
delta={}  # (date, token) -> delta_team
for r in rows:
    d=(r["event_date"] or "")[:10]; tok=lt(r["yes_team"])
    delta[(d,tok)]=SIGMA*(phinv(float(r["pb"]))-phinv(float(r["ps"])))

with get_conn() as c:
    stats=evaluate_sector(c,"wnba",ev_min_pp=0.0,depth_min_usd=0.0,market_types=("spread",))
lay=[e for e in stats.entries if e.clv_pp is not None and e.side=="lay"]

def fav_delta(e):
    # outcome_label like "storm -6.5" -> favorite token
    tok=lt(e.outcome_label.split(" -")[0]) if " -" in e.outcome_label else lt(e.outcome_label.split()[0])
    d=e.entry_at.date().isoformat()
    # match by date + token (favorite is the lay-side team)
    if (d,tok) in delta: return delta[(d,tok)]
    # entry date can be tip day; try both team tokens present
    return None

matched=[(e,fav_delta(e)) for e in lay]
have=[(e,dv) for e,dv in matched if dv is not None]
print(f"spread-lay entries={len(lay)}  delta-matched={len(have)}")

def gk(e):
    m=re.match(r'KX[A-Z]+-([0-9A-Z]+?[A-Z]{2,})-',e.ticker); return m.group(1) if m else e.ticker
def declust_clv(pairs):
    g={}
    for e,dv in pairs: g.setdefault(gk(e),[]).append(e.clv_pp)
    return [st.mean(v) for v in g.values()]
def rep(tag,pairs):
    g=declust_clv(pairs)
    if not g: print(f"  {tag:44} games=0"); return
    print(f"  {tag:44} games={len(g):4} CLV={st.mean(g):+.2f}pp t={t(g):+.2f}"+("  <==" if st.mean(g)>0 and t(g)>1.65 and len(g)>=30 else ""))

cont=[(e,dv) for e,dv in have if abs(e.fair_prob-0.5)<=0.15]
print("\n-- lay, delta-agreement split (delta_fav>0 = blend MORE bullish on favorite) --")
rep("lay ALL matched", have)
rep("lay + delta_fav>0 (agree)", [(e,dv) for e,dv in have if dv>0])
rep("lay + delta_fav<=0 (disagree)", [(e,dv) for e,dv in have if dv<=0])
print("\n-- contested lay, delta split --")
rep("lay+contested ALL", cont)
rep("lay+contested + delta_fav>0", [(e,dv) for e,dv in cont if dv>0])
rep("lay+contested + delta_fav<=0", [(e,dv) for e,dv in cont if dv<=0])
print("\n-- triple ensemble: contested + near-tip + delta-agree --")
rep("contested+near-tip", [(e,dv) for e,dv in cont if e.entry_lead_h<=12])
rep("contested+near-tip+delta>0", [(e,dv) for e,dv in cont if e.entry_lead_h<=12 and dv>0])
