"""Iteration 5: fair-value ENSEMBLE for the spread ladder — mean-shift (§1) + sigma (§2).

Sidesteps the Kalshi ticker-orientation trap: evaluates against actual outcomes at
synthetic ladder offsets Delta from the sharp mean mu_sharp (reliable game-log margins,
production's own YES orientation). At each rung L = mu_sharp + Delta, cover outcome is
1[yes_margin > L]. Compare Brier of:
  C0 sharp ladder     Phi(-Delta / 12.5)
  C1 +mean-shift      Phi((delta - Delta) / 12.5)          # §1 delta from ML blend
  C2 +sigma recal     Phi(-Delta / sigma_glob)             # §2 global sigma
  C3 ENSEMBLE         Phi((delta - Delta) / sigma_feat)    # §1 + §2 feature sigma
sigma_glob & the sigma_feat regression are fit on TRAIN only; scored on HOLDOUT.
delta near the line, sigma at the wings -> ensemble should win where a single channel can't.
"""
from __future__ import annotations
import json, sqlite3, math, statistics as st
from pathlib import Path

REPO = Path("/Users/ktzeng/Projects/evmax")
SIGMA0 = 12.5
def Phi(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def phinv(p):
    p=min(max(p,1e-4),1-1e-4); lo,hi=-8.0,8.0
    for _ in range(80):
        m=(lo+hi)/2
        if Phi(m)<p: lo=m
        else: hi=m
    return (lo+hi)/2
def lt(s): return (s or "").strip().lower().split()[-1] if s else ""
def poss(b): return b["fga"]+0.44*b["fta"]-b["oreb"]+b["tov"]

log=sorted(json.load(open(REPO/"data/backtest/wnba_totals_walkforward_games.json")),key=lambda g:g["date"])
score={}; feat={}; hist={}
for g in log:
    ht,at=lt(g["home_name"]),lt(g["away_name"]); key=(g["date"],frozenset((ht,at)))
    score[key]={ht:g["home_score"],at:g["away_score"]}
    def tr(t):
        h=hist.get(t,[]); return (st.mean(x[0] for x in h),st.mean(x[1] for x in h)) if len(h)>=3 else None
    th,ta=tr(ht),tr(at)
    if th and ta: feat[key]={"pm":abs(th[0]-ta[0]),"tpm":th[1]+ta[1]}
    hist.setdefault(ht,[]).append((poss(g["home_stats"]),g["home_stats"]["three_pm"]))
    hist.setdefault(at,[]).append((poss(g["away_stats"]),g["away_stats"]["three_pm"]))

con=sqlite3.connect(str(REPO/"data/predictions.db")); con.row_factory=sqlite3.Row
rows=con.execute("""SELECT p.event_date,p.yes_team,p.blended_true_prob pb,p.sharp_true_prob ps
  FROM ev_predictions p JOIN (SELECT market_id,MAX(scan_date) md FROM ev_predictions
    WHERE sector='wnba' AND market_type='moneyline' GROUP BY market_id) x
    ON x.market_id=p.market_id AND x.md=p.scan_date
  WHERE p.sector='wnba' AND p.market_type='moneyline'
    AND p.blended_true_prob IS NOT NULL AND p.sharp_true_prob IS NOT NULL""").fetchall(); con.close()

recs=[]
for r in rows:
    yt=lt(r["yes_team"]); d=(r["event_date"] or "")[:10]
    gk=next((k for k in score if k[0]==d and yt in k[1]),None)
    if not gk or gk not in feat: continue
    sc=score[gk]; opp=next(iter(set(sc)-{yt}),None)
    if opp is None: continue
    mu=SIGMA0*phinv(float(r["ps"])); dl=SIGMA0*phinv(float(r["pb"]))-mu
    recs.append({"d":d,"m":sc[yt]-sc[opp],"mu":mu,"dl":dl,**feat[gk]})
recs.sort(key=lambda z:z["d"]); n=len(recs); cut=int(n*0.7); tr,ho=recs[:cut],recs[cut:]
print(f"n={n} games  train={len(tr)} holdout={len(ho)}")

# --- fit sigma models on TRAIN ---
absR_tr=[abs(z["m"]-z["mu"]) for z in tr]
sigma_glob=st.mean(absR_tr)/0.7979   # half-normal E|R|=sigma*sqrt(2/pi)
# sigma_feat: OLS |R| ~ pm + tpm  (2-feature normal equations, centered)
pm=[z["pm"] for z in tr]; tp=[z["tpm"] for z in tr]; y=absR_tr
mpm,mtp,my=st.mean(pm),st.mean(tp),st.mean(y)
Spp=sum((a-mpm)**2 for a in pm); Stt=sum((a-mtp)**2 for a in tp)
Spt=sum((a-mpm)*(b-mtp) for a,b in zip(pm,tp))
Spy=sum((a-mpm)*(c-my) for a,c in zip(pm,y)); Sty=sum((a-mtp)*(c-my) for a,c in zip(tp,y))
det=Spp*Stt-Spt*Spt
b_pm=(Stt*Spy-Spt*Sty)/det; b_tp=(Spp*Sty-Spt*Spy)/det
def sig_feat(z):
    pred=my+b_pm*(z["pm"]-mpm)+b_tp*(z["tpm"]-mtp)
    return max(pred,4.0)/0.7979
print(f"  sigma_glob={sigma_glob:.2f} (vs 12.5)   sigma_feat coeffs pm={b_pm:+.3f} tpm={b_tp:+.3f}")

OFFS=[-9,-6,-3,0,3,6,9]
def brier(sub,which,band=None):
    tot=cnt=0.0
    for z in sub:
        sf=sig_feat(z)
        for D in OFFS:
            if band=="near" and abs(D)>3: continue
            if band=="wing" and abs(D)<6: continue
            y=1.0 if z["m"]>z["mu"]+D else 0.0
            p={"C0":Phi(-D/SIGMA0),"C1":Phi((z["dl"]-D)/SIGMA0),
               "C2":Phi(-D/sigma_glob),"C3":Phi((z["dl"]-D)/sf)}[which]
            tot+=(p-y)**2; cnt+=1
    return tot/cnt

for band in (None,"near","wing"):
    tag=band or "ALL"
    b={w:brier(ho,w,band) for w in ("C0","C1","C2","C3")}
    best=min(b,key=b.get)
    print(f"HOLDOUT {tag:5}  C0(sharp)={b['C0']:.4f}  C1(+shift)={b['C1']:.4f}  "
          f"C2(+sig)={b['C2']:.4f}  C3(ENSEMBLE)={b['C3']:.4f}  best={best}  "
          f"dC3vsC0={b['C0']-b['C3']:+.4f}")
