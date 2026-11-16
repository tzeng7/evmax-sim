"""Iteration 1 (§1): does the WNBA moneyline BLEND's location shift transfer to the spread ladder?

The sim-weight sweep only falsified the possession sim's cover probability. It never
tested whether the *moneyline blend* (efficiency+elo+sim, the estimator with a
validated +3.6 ΔBrier / 4.82pp divergence edge vs sharp on WNBA ML) carries a better
estimate of the margin-distribution LOCATION than the sharp line — which, if real, is
maximized at the posted spread line, not at zero.

Method (leak-free: blended/sharp ML probs in predictions.db were genuine OOS forecasts
made pre-game by production; no re-fit of the blend):
  mu = sigma * Phi^-1(p_win)          # invert YES-team win prob -> implied margin
  delta = mu_blend - mu_sharp         # blend's location shift vs sharp, in points
  R = actual_yes_margin - mu_sharp    # sharp's location residual
Tests:
  (a) regress R ~ delta  -> slope beta. beta>0 & sig => location edge transfers.
  (b) parameter-free Brier at sharp pick'em (L=mu_sharp): constant-0.5 (sharp) vs
      Phi(delta/sigma) (delta-shifted) against 1[yes_margin > mu_sharp].
      This is the "highest-density point" test.
  (c) holdout = latest 30% of games by date (only (b) needs OOS; (a) is descriptive).
"""
from __future__ import annotations
import json, sqlite3, math
from pathlib import Path

REPO = Path("/Users/ktzeng/Projects/evmax")
PRED_DB = REPO / "data" / "predictions.db"
GAMELOG = REPO / "data" / "backtest" / "wnba_totals_walkforward_games.json"
SIGMA = 12.5  # WNBA possession-sim margin sd (code comment)

def phi(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def phinv(p):  # inverse normal via Acklam-ish; use bisection for robustness
    p = min(max(p, 1e-4), 1 - 1e-4)
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if phi(mid) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def last_tok(name: str) -> str:
    return (name or "").strip().lower().split()[-1] if name else ""

# ---- game log: (date, frozenset{home_tok,away_tok}) -> per-team score ----
games = {}
for g in json.load(open(GAMELOG)):
    ht, at = last_tok(g["home_name"]), last_tok(g["away_name"])
    key = (g["date"], frozenset((ht, at)))
    games[key] = {ht: g["home_score"], at: g["away_score"]}

# ---- predictions.db: latest wnba ML row per market ----
con = sqlite3.connect(str(PRED_DB))
con.row_factory = sqlite3.Row
rows = con.execute("""
  SELECT p.market_id, p.event_date, p.yes_team, p.event_title,
         p.blended_true_prob AS pb, p.sharp_true_prob AS ps
  FROM ev_predictions p
  JOIN (SELECT market_id, MAX(scan_date) md FROM ev_predictions
        WHERE sector='wnba' AND market_type='moneyline' GROUP BY market_id) x
    ON x.market_id=p.market_id AND x.md=p.scan_date
  WHERE p.sector='wnba' AND p.market_type='moneyline'
    AND p.blended_true_prob IS NOT NULL AND p.sharp_true_prob IS NOT NULL
""").fetchall()
con.close()

recs = []
misses = 0
for r in rows:
    yt = last_tok(r["yes_team"])
    date = (r["event_date"] or "")[:10]
    key = (date, None)
    # find the game whose team set contains yes_team on that date
    match = None
    for (d, ts), sc in games.items():
        if d == date and yt in ts:
            match = sc; break
    if match is None or yt not in match:
        misses += 1; continue
    opp_tok = next(iter(set(match) - {yt}), None)
    if opp_tok is None: continue
    yes_margin = match[yt] - match[opp_tok]
    pb, ps = float(r["pb"]), float(r["ps"])
    mu_b, mu_s = SIGMA * phinv(pb), SIGMA * phinv(ps)
    delta = mu_b - mu_s
    R = yes_margin - mu_s
    recs.append({"date": date, "yt": yt, "pb": pb, "ps": ps,
                 "mu_s": mu_s, "delta": delta, "yes_margin": yes_margin, "R": R})

recs.sort(key=lambda z: z["date"])
n = len(recs)
print(f"joined n={n}  (misses={misses})   sigma={SIGMA}")
if n < 30:
    print("insufficient sample"); raise SystemExit

# ---- (a) regress R ~ delta  (descriptive; delta,mu_s already OOS) ----
import statistics as st
xs = [z["delta"] for z in recs]; ys = [z["R"] for z in recs]
mx, my = st.mean(xs), st.mean(ys)
sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
beta = sxy / sxx if sxx else 0.0
alpha = my - beta*mx
resid = [y-(alpha+beta*x) for x,y in zip(xs,ys)]
s2 = sum(e*e for e in resid)/(n-2)
se_beta = math.sqrt(s2/sxx) if sxx else float('inf')
t_beta = beta/se_beta if se_beta else 0.0
corr = sxy / math.sqrt(sxx * sum((y-my)**2 for y in ys)) if sxx else 0.0
print("\n(a) R ~ delta  (does the blend's location shift predict the margin beyond sharp?)")
print(f"    beta={beta:+.3f}  t={t_beta:+.2f}  corr={corr:+.3f}  (transfer => beta≈+1, t>1.65)")
print(f"    mean|delta|={st.mean(abs(x) for x in xs):.2f}pt  sd(delta)={st.pstdev(xs):.2f}pt")

# directional: does sign(delta) match sign(R)?
dir_hits = sum(1 for z in recs if z["delta"]*z["R"] > 0)
nz = sum(1 for z in recs if abs(z["delta"])>1e-9)
print(f"    directional: sign(delta)==sign(R) on {dir_hits}/{nz} = {dir_hits/nz:.1%}")

# ---- (b) parameter-free Brier at sharp pick'em, full + holdout ----
def brier_at_pickem(sub):
    b_sharp = b_shift = 0.0
    for z in sub:
        y = 1.0 if z["yes_margin"] > z["mu_s"] else 0.0  # cover at L=mu_s
        p_shift = phi(z["delta"]/SIGMA)
        b_sharp += (0.5 - y)**2
        b_shift += (p_shift - y)**2
    k=len(sub)
    return b_sharp/k, b_shift/k, k

cut = int(n*0.7)
train, hold = recs[:cut], recs[cut:]
for tag, sub in (("FULL", recs), ("train", train), ("HOLDOUT", hold)):
    bs, bsh, k = brier_at_pickem(sub)
    flag = "  <- shift better" if bsh < bs else "  <- no improvement"
    print(f"(b) {tag:8} n={k:3}  Brier sharp(0.5)={bs:.4f}  delta-shift={bsh:.4f}  d={bs-bsh:+.4f}{flag}")

# contested subset: |mu_s| small (near pick'em games — the highest-density transfer point)
cont = [z for z in recs if abs(z["mu_s"]) <= 6.0]
if len(cont) >= 20:
    bs,bsh,k = brier_at_pickem(cont)
    print(f"(b) CONTESTED(|mu_s|<=6) n={k}  sharp={bs:.4f}  shift={bsh:.4f}  d={bs-bsh:+.4f}")

# ================= ROBUSTNESS (iteration-1 verification) =================
print("\n--- ROBUSTNESS ---")
# 1) directional accuracy by |delta| quartile: real signal => rises with |delta|
srt = sorted(recs, key=lambda z: abs(z["delta"]))
q = len(srt)//4
for i,(lo,hi) in enumerate([(0,q),(q,2*q),(2*q,3*q),(3*q,len(srt))]):
    sub = srt[lo:hi]
    hits = sum(1 for z in sub if z["delta"]*z["R"]>0)
    md = st.mean(abs(z["delta"]) for z in sub)
    print(f"  |delta| Q{i+1}: mean|delta|={md:.2f}pt  dir-acc={hits}/{len(sub)}={hits/len(sub):.1%}")

# 2) sigma sensitivity on FULL Brier delta (directional is sign-invariant so unchanged)
for sg in (11.0, 12.5, 14.0):
    b_sharp=b_shift=0.0
    for z in recs:
        y = 1.0 if z["yes_margin"]>z["mu_s"] else 0.0
        p = phi(z["delta"]/sg)
        b_sharp+=(0.5-y)**2; b_shift+=(p-y)**2
    k=len(recs)
    print(f"  sigma={sg}: FULL Brier d(sharp-shift)={ (b_sharp-b_shift)/k:+.4f}")

# 3) bootstrap 95% CI on HOLDOUT Brier delta (resample holdout with replacement)
import random
random.seed(7)
def bdelta(sub):
    bs=bsh=0.0
    for z in sub:
        y=1.0 if z["yes_margin"]>z["mu_s"] else 0.0
        bs+=(0.5-y)**2; bsh+=(phi(z["delta"]/SIGMA)-y)**2
    return (bs-bsh)/len(sub)
hold = recs[int(n*0.7):]
boots = sorted(bdelta([random.choice(hold) for _ in hold]) for _ in range(2000))
print(f"  HOLDOUT Brier d: point={bdelta(hold):+.4f}  95%CI=[{boots[50]:+.4f},{boots[1949]:+.4f}]  (n={len(hold)})")

# 4) binomial p for full-season directional
import math as _m
k_hit=sum(1 for z in recs if z["delta"]*z["R"]>0); N=len(recs)
z_=(k_hit-0.5*N)/_m.sqrt(0.25*N)
print(f"  FULL directional {k_hit}/{N}={k_hit/N:.1%}  z={z_:+.2f}  (1-sided p={0.5*_m.erfc(z_/_m.sqrt(2)):.3f})")
