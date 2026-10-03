"""Offline maker-fill replay: Kalshi NFL/NCAAF candidates vs archived order-book snapshots.

For each resolved Kalshi candidate, simulate a resting bid (best bid + 1c, never crossing the
ask) at log time and test whether later hourly snapshots show the ask reaching it
("touch") or going through it ("thru"). Compares entry->close CLV and net-of-fee edge for the
maker path (0.0175·P·(1-P) on every NFL/NCAAF series per Kalshi's /series API) against crossing the ask as a taker.

Limits: hourly snapshots understate touches; touch-fill ignores queue position; only tickers
covered by archived_orderbook_depth (watch-listings: spread/total ladders) are testable.
Usage: python scripts/eval_maker_fill_replay.py [--pred-db PATH] [--archive-db PATH]
"""
import argparse as _ap
_a=_ap.ArgumentParser(); _a.add_argument("--pred-db",default="data/predictions.db"); _a.add_argument("--archive-db",default="data/archive.db")
_args=_a.parse_args()
import sys, sqlite3, collections, math, bisect
from datetime import datetime, timedelta, timezone
sys.path.insert(0,".")
from evmax.agents.cleanup.contamination import is_contaminated
from evmax.agents.cleanup.resolver import close_lookup_ticker
from evmax.cli.commands.shadow import game_key

P=sqlite3.connect(f"file:{_args.pred_db}?mode=ro",uri=True); P.row_factory=sqlite3.Row
A=sqlite3.connect(f"file:{_args.archive_db}?mode=ro",uri=True)
rows=P.execute("""SELECT p.*, o.outcome FROM ev_predictions p JOIN ev_outcomes o USING(market_id)
 WHERE p.venue='kalshi' AND p.sector IN('nfl','ncaaf') AND (COALESCE(p.voided,0)=0 OR p.void_reason='stale_reverted')
 AND p.minutes_to_tipoff>0 AND p.market_id NOT LIKE '%-tt-%'""").fetchall()
rows=[r for r in rows if not is_contaminated(r["sector"],r["market_type"],r["model_sources"],r["line"])]
def ts(s): 
    s=s.replace("T"," ").split("+")[0].split(".")[0]; return datetime.strptime(s,"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
cache={}
def snaps(tk):
    if tk not in cache:
        q=A.execute("select fetched_at,yes_ask,yes_bid from archived_orderbook_depth where ticker=? and yes_ask is not null and yes_bid is not null order by fetched_at",(tk,)).fetchall()
        cache[tk]=([ts(x[0]) for x in q],[(x[1],x[2]) for x in q])
    return cache[tk]
def taker_fee(p): return 0.07*p*(1-p)
def maker_fee(p,series): return 0.0175*p*(1-p)   # Kalshi series API: quadratic_with_maker_fees, multiplier 1 for NFL/NCAAF game, spread, total
out=[]; skipped=collections.Counter()
for r in rows:
    tk,isno=close_lookup_ticker(r["market_id"]); 
    if not tk: skipped["noticker"]+=1; continue
    t0=ts(r["logged_at"]); tip=t0+timedelta(minutes=r["minutes_to_tipoff"])
    T,V=snaps(tk)
    if not T: skipped["nosnaps"]+=1; continue
    i=bisect.bisect_right(T,t0)-1
    if i<0 or (t0-T[i])>timedelta(minutes=90): skipped["no_entry_snap"]+=1; continue
    ya,yb=V[i]
    ask,bid=(1-yb,1-ya) if isno else (ya,yb)           # our side's ask / best bid
    if not(0<bid<ask<1): skipped["bad_book"]+=1; continue
    limit=min(round(bid+0.01,2),round(ask-0.01,2)); 
    if limit<bid: limit=bid
    series="SPREAD" if "SPREAD" in tk else "TOTAL" if "TOTAL" in tk else "GAME"
    j0=bisect.bisect_right(T,t0); j1=bisect.bisect_left(T,tip-timedelta(minutes=5))   # later snaps until ~tip
    fut=V[j0:j1]
    if not fut: skipped["no_future"]+=1; continue
    fa=[(1-b if isno else a) for a,b in fut]            # our-side ask path
    touch=any(x<=limit+1e-9 for x in fa); thru=any(x<limit-1e-9 for x in fa)
    closeask=fa[-1]
    bl=r["blended_true_prob"]
    out.append(dict(sector=r["sector"],mt=r["market_type"],series=series,g=game_key(r["event_id"]),isno=isno,
        ask=ask,bid=bid,limit=limit,touch=touch,thru=thru,close=closeask,bl=bl,outc=r["outcome"],ev_taker=r["ev_pct"],
        ev_lim=(bl-limit-maker_fee(limit,series))/limit, ev_ask=(bl-ask-taker_fee(ask))/ask, spread=ask-bid))
print("candidates with usable book:",len(out)," skipped:",dict(skipped))
def summ(lab,rs):
    if len(rs)<8: print(f"{lab:36} n={len(rs)} (too few)"); return
    n=len(rs); g=len({x['g'] for x in rs}); tf=[x for x in rs if x['touch']]; th=[x for x in rs if x['thru']]
    imp=sum(x['ask']-x['limit'] for x in rs)/n*100
    # price improvement + CLV: maker CLV = close - limit (filled); taker CLV = close - ask (all)
    mclv=lambda xs:(sum(x['close']-x['limit'] for x in xs)/len(xs)*100) if xs else float('nan')
    tclv=sum(x['close']-x['ask'] for x in rs)/n*100
    un=[x for x in rs if not x['touch']]
    uclv=(sum(x['close']-x['ask'] for x in un)/len(un)*100) if un else float('nan')
    # per-candidate expected net edge: taker=(clv-fee)/ask ; maker = fill*(clv_m - fee)/limit
    def netroi(xs,pricekey,fee):
        return sum(((x['close']-x[pricekey])-fee(x))/x[pricekey] for x in xs)/len(xs)*100 if xs else float('nan')
    tk_roi=netroi(rs,'ask',lambda x:taker_fee(x['ask']))
    mk_roi_f=netroi(tf,'limit',lambda x:maker_fee(x['limit'],x['series']))
    print(f"{lab:36} n={n:4} g={g:3} spread={sum(x['spread'] for x in rs)/n*100:4.1f}c bid-improve={imp:4.2f}c | fill touch={len(tf)/n*100:4.0f}% thru={len(th)/n*100:4.0f}% "
          f"| CLV taker(ask→close)={tclv:+5.2f}pp  maker filled(limit→close)={mclv(tf):+5.2f}pp  UNfilled-would-be-taker={uclv:+5.2f}pp "
          f"| netEV/ct taker={tk_roi:+5.1f}% maker|filled={mk_roi_f:+5.1f}% maker per-candidate={mk_roi_f*len(tf)/n:+5.1f}%")
for sec,mt in (("nfl","spread"),("nfl","total"),("nfl","moneyline"),("ncaaf","moneyline")):
    base=[x for x in out if x['sector']==sec and x['mt']==mt]
    print(f"\n== {sec} {mt}")
    summ("all candidates",base)
    summ("taker EV>=2% (scanner flagged)",[x for x in base if x['ev_taker']>=0.02])
    summ("maker-gate: net EV at limit>=2%",[x for x in base if x['ev_lim']>=0.02])
    summ("maker-gate & spread<=2c",[x for x in base if x['ev_lim']>=0.02 and x['spread']<=0.021])
    summ("maker-gate & lay/YES",[x for x in base if x['ev_lim']>=0.02 and not x['isno']])
    summ("maker-gate & NO side",[x for x in base if x['ev_lim']>=0.02 and x['isno']])
# outcome check on filled vs unfilled (adverse selection in realized terms), maker-gate set
print("\nRealized (noisy): win rate of filled vs unfilled maker-gate candidates")
for sec,mt in (("nfl","spread"),("nfl","total"),("ncaaf","moneyline")):
    b=[x for x in out if x['sector']==sec and x['mt']==mt and x['ev_lim']>=0.02]
    for lab,xs in(("filled(touch)",[x for x in b if x['touch']]),("unfilled",[x for x in b if not x['touch']])):
        if xs: print(f"  {sec} {mt} {lab:14} n={len(xs):3} win={sum(x['outc'] for x in xs)/len(xs)*100:4.1f}% blend={sum(x['bl'] for x in xs)/len(xs)*100:4.1f}% avg entry={sum(x['limit'] for x in xs)/len(xs)*100:4.1f}c")
