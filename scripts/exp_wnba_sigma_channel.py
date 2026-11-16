"""Iteration 4 (§2): is the margin DISPERSION predictable beyond the market?

The sharp line prices the mean; the sim-weight sweep tested the mean. Untested:
does a feature-based sigma estimate predict realized dispersion |margin - mu_sharp|?
If yes, you can trade the ladder WINGS (sell when your sigma < implied, buy when >)
with no view on the mean. Clean: absolute residual needs no spread sign convention.

Walk-forward, box-score log only (no lookahead): each team's TRAILING (>=3 prior
games) pace and 3P-make rate BEFORE the game. Features:
  pace_sum        = trailing poss/g, both teams   (scoring environment -> variance)
  pace_mismatch   = |paceA - paceB|               (divergent tempo -> higher var)
  tpm_sum         = trailing 3PM/g, both teams     (3-point volume -> fat tails)
Regress |margin - mu_sharp| on features; incremental R^2 vs const-only, train/holdout.
"""
from __future__ import annotations
import json, sqlite3, math, statistics as st
from pathlib import Path

REPO = Path("/Users/ktzeng/Projects/evmax")
GAMELOG = REPO / "data" / "backtest" / "wnba_totals_walkforward_games.json"
PRED = REPO / "data" / "predictions.db"
SIGMA = 12.5

def phinv(p):
    p = min(max(p, 1e-4), 1 - 1e-4); lo, hi = -8.0, 8.0
    for _ in range(80):
        m = (lo+hi)/2
        if 0.5*(1+math.erf(m/math.sqrt(2))) < p: lo = m
        else: hi = m
    return (lo+hi)/2
def last_tok(s): return (s or "").strip().lower().split()[-1] if s else ""
def poss(bs):  # standard possession estimate
    return bs["fga"] + 0.44*bs["fta"] - bs["oreb"] + bs["tov"]

log = sorted(json.load(open(GAMELOG)), key=lambda g: g["date"])
# walk-forward trailing per-team pace & 3PM
hist = {}  # tok -> list of (poss, three_pm)
feat = {}  # (date, frozenset{toks}) -> features dict
for g in log:
    ht, at = last_tok(g["home_name"]), last_tok(g["away_name"])
    def trail(tok):
        h = hist.get(tok, [])
        if len(h) < 3: return None
        return (st.mean(x[0] for x in h), st.mean(x[1] for x in h))
    th, ta = trail(ht), trail(at)
    if th and ta:
        feat[(g["date"], frozenset((ht, at)))] = {
            "pace_sum": th[0]+ta[0],
            "pace_mismatch": abs(th[0]-ta[0]),
            "tpm_sum": th[1]+ta[1],
        }
    hist.setdefault(ht, []).append((poss(g["home_stats"]), g["home_stats"]["three_pm"]))
    hist.setdefault(at, []).append((poss(g["away_stats"]), g["away_stats"]["three_pm"]))
    feat.setdefault(("_score", g["date"], frozenset((ht,at))), None)
score = {(g["date"], frozenset((last_tok(g["home_name"]), last_tok(g["away_name"])))):
         {last_tok(g["home_name"]): g["home_score"], last_tok(g["away_name"]): g["away_score"]}
         for g in log}

# join sharp mu + margin from predictions.db ML rows
con = sqlite3.connect(str(PRED)); con.row_factory = sqlite3.Row
rows = con.execute("""
  SELECT p.event_date, p.yes_team, p.sharp_true_prob ps FROM ev_predictions p
  JOIN (SELECT market_id, MAX(scan_date) md FROM ev_predictions
        WHERE sector='wnba' AND market_type='moneyline' GROUP BY market_id) x
    ON x.market_id=p.market_id AND x.md=p.scan_date
  WHERE p.sector='wnba' AND p.market_type='moneyline' AND p.sharp_true_prob IS NOT NULL
""").fetchall(); con.close()

data = []
for r in rows:
    yt = last_tok(r["yes_team"]); d = (r["event_date"] or "")[:10]
    gk = next((k for k in score if k[0]==d and yt in k[1]), None)
    if gk is None or gk not in feat: continue
    sc = score[gk]; opp = next(iter(set(sc)-{yt}), None)
    if opp is None: continue
    margin = sc[yt]-sc[opp]; mu = SIGMA*phinv(float(r["ps"]))
    absR = abs(margin - mu)
    f = feat[gk]
    data.append({"date": d, "absR": absR, **f})

data.sort(key=lambda z: z["date"])
n = len(data)
print(f"n={n} games with walk-forward features + sharp mu")
if n < 40: print("insufficient"); raise SystemExit

# center features; multiple OLS |absR| ~ pace_sum + pace_mismatch + tpm_sum
FEATS = ["pace_sum","pace_mismatch","tpm_sum"]
def ols(sub):
    y = [z["absR"] for z in sub]; ybar = st.mean(y)
    cols = {f: [z[f] for z in sub] for f in FEATS}
    means = {f: st.mean(cols[f]) for f in FEATS}
    # simple per-feature univariate slope + t (report each; multicollinearity mild)
    out = {}
    for f in FEATS:
        x = cols[f]; mx = means[f]
        sxx = sum((xi-mx)**2 for xi in x); sxy = sum((xi-mx)*(yi-ybar) for xi,yi in zip(x,y))
        b = sxy/sxx if sxx else 0
        resid = [yi-(ybar + b*(xi-mx)) for xi,yi in zip(x,y)]
        s2 = sum(e*e for e in resid)/(len(sub)-2); se = math.sqrt(s2/sxx) if sxx else 9e9
        r = sxy/math.sqrt(sxx*sum((yi-ybar)**2 for yi in y)) if sxx else 0
        out[f] = (b, b/se if se else 0, r)
    return out

print("\n-- univariate: does each feature predict |margin - mu_sharp| (dispersion)? --")
full = ols(data)
for f in FEATS:
    b,t,r = full[f]
    print(f"  {f:14} slope={b:+.4f}  t={t:+.2f}  corr={r:+.3f}"
          + ("  <== predicts dispersion" if abs(t)>1.65 else ""))

# holdout consistency (does sign hold out-of-sample?)
cut=int(n*0.7); tr,ho=data[:cut],data[cut:]
print(f"\n-- holdout (train {len(tr)} / hold {len(ho)}): corr sign consistency --")
ot, oh = ols(tr), ols(ho)
for f in FEATS:
    print(f"  {f:14} train corr={ot[f][2]:+.3f}  holdout corr={oh[f][2]:+.3f}"
          + ("  <== consistent" if ot[f][2]*oh[f][2]>0 and abs(oh[f][2])>0.1 else ""))

# baseline: is realized dispersion sd even != the assumed SIGMA=12.5?
sd_real = st.pstdev([z["absR"] for z in data])
print(f"\n  mean|R|={st.mean(z['absR'] for z in data):.2f}  (half-normal mean = sigma*0.798 = {SIGMA*0.798:.2f} if sigma={SIGMA})")
