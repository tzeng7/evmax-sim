"""Iteration 3 (§4): model-free laddered-market arbitrage on Kalshi (no forecast).

Two locked violations, both from executable prices (yes_price=YES ask,
no_price=NO ask, so YES bid = 1 - no_price), net of the Kalshi fee 0.07*p*(1-p):

  A) within-ticker box:  yes_ask + no_ask < 1  -> buy both sides, guaranteed $1.
  B) cross-strike totals inversion (nested events, clean threshold semantics):
     for over-lines k1<k2 in the SAME game+sweep,
       long {total>k1} + short {total>k2}  pays in {0,+1},
       cost = ask_over(k1) - bid_over(k2);  arb if < 0.
     (lower line cheaper to BUY than higher line is to SELL = pure inversion)

Grouped by session_id (one near-simultaneous sweep) so the two legs are
co-observed. Reports frequency vs all snapshots + net-of-fee magnitude.
"""
from __future__ import annotations
import sqlite3
from collections import defaultdict

ARC = "/Users/ktzeng/Projects/evmax/data/archive.db"

def fee(p):  # Kalshi taker, per contract, per leg
    p = min(max(p, 0.0), 1.0)
    return 0.07 * p * (1 - p)

con = sqlite3.connect(ARC)
con.row_factory = sqlite3.Row

# ---- A) within-ticker box arbs (yes_ask + no_ask < 1 net of both fees) ----
print("=== A) within-ticker box arb (yes_ask + no_ask + fees < 1) ===")
for sec in ("wnba", "baseball", "nba"):
    rows = con.execute("""
      SELECT yes_price, no_price FROM archived_kalshi_markets
      WHERE sector=? AND market_type IN ('spread','total')
        AND yes_price>0 AND yes_price<1 AND no_price>0 AND no_price<1
    """, (sec,)).fetchall()
    n = len(rows); hits = 0; best = 0.0
    for r in rows:
        cost = r["yes_price"] + r["no_price"] + fee(r["yes_price"]) + fee(r["no_price"])
        if cost < 1.0:
            hits += 1; best = max(best, 1.0 - cost)
    print(f"  {sec:9} snapshots={n:7}  box-arbs={hits:5}  ({hits/max(n,1):.3%})  best_net=${best:.3f}")

# ---- B) cross-strike totals inversion ----
print("\n=== B) cross-strike totals inversion (ask_over(k1) < bid_over(k2), k1<k2), net of fee ===")
for sec in ("wnba", "baseball"):
    rows = con.execute("""
      SELECT session_id, ticker, line, yes_price, no_price, event_id, event_date
      FROM archived_kalshi_markets
      WHERE sector=? AND market_type='total'
        AND line IS NOT NULL AND yes_price>0 AND yes_price<1 AND no_price>0 AND no_price<1
    """, (sec,)).fetchall()
    # group by (session, game) — game key from event_id or date+ticker prefix
    grp = defaultdict(list)
    for r in rows:
        gk = (r["session_id"], r["event_id"] or r["ticker"].rsplit("-", 1)[0])
        grp[gk].append(r)
    total_ladders = sum(1 for v in grp.values() if len(v) >= 2)
    hits = 0; best = 0.0; examples = []
    for (sess, gk), items in grp.items():
        # over-YES ask/bid per line
        by_line = {}
        for r in items:
            by_line[r["line"]] = (r["yes_price"], 1.0 - r["no_price"])  # (ask_over, bid_over)
        lines = sorted(by_line)
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                k1, k2 = lines[i], lines[j]          # k1 < k2
                ask1 = by_line[k1][0]                # buy over k1
                bid2 = by_line[k2][1]                # sell over k2
                net = (bid2 - ask1) - fee(ask1) - fee(bid2)
                if net > 0:
                    hits += 1; best = max(best, net)
                    if len(examples) < 5:
                        examples.append(f"{gk} k1={k1} ask={ask1:.2f} < k2={k2} bid={bid2:.2f} net=+${net:.3f}")
    print(f"  {sec:9} total-ladders(session×game)={total_ladders:6}  inversions={hits:4}  best_net=${best:.3f}")
    for ex in examples: print(f"      {ex}")

con.close()
