# Near-close re-scan workflow — Phase 1 validation

**Date:** 2026-07-11 · **Verdict: REJECTED (underpowered — no gate cleared on current capture data)**

## Question

The daily `evmax agents scan` runs night-before/morning; its EV≥2% filter selects
upward Pinnacle-snapshot noise that reverts by close, and placed-bet CLV analysis
showed only the <1h-pre-tip entry window carries positive CLV (~+1.55pp). Baseball
totals are disabled and WNBA totals/spread sit in shadow explicitly "until a
near-close re-scan workflow exists."

Phase 1 gate (before any pipeline wiring): **if we had re-scanned at T-75..T-10
before tip and entered at the live near-tip ask with a fresh Pinnacle anchor, does
entry CLV improve by ≥1pp on n≥50 overall, or does at least one currently-gated
market type (baseball total, wnba total/spread) show a clear positive-CLV path?**

## Method

Read-only replay over the real DBs (`sqlite3` URI `mode=ro`, never written):
`scripts/eval_near_close_rescan.py`, core logic in
`evmax/agents/cleanup/rescan_eval.py` (+ an `entry window` extension to the
existing `listings_eval` harness). Two lenses:

1. **Paired ev_predictions lens** — every logged bet (since 2026-06-20, when
   watch-closes launched) that has an archived venue snapshot inside
   [T-75m, T-10m] AND a later pre-tip snapshot is scored under both entry rules
   against the **same close** (last pre-tip snapshot after the rescan entry):
   * actual entry = scan-time price (fill price for placed bets, per
     `clv_entry_price`),
   * simulated rescan entry = first in-window snapshot, **gated** on a fresh
     (≤2h) as-of Pinnacle anchor still showing edge ≥2pp at that crossable price.
   * NO-side `:no` rows follow the `backfill_clv` convention (entry stored
     NO-side; archived YES snapshots flipped `1-yes`); PolyUS venue-prefixed ids
     resolved via `close_lookup_ticker`.
2. **Archive-wide listings lens** — the gated laddered types (baseball
   spread/total, wnba spread/total) have no/few ev_predictions rows, so the
   watch-listings capture is replayed twice through the first-anchored-sweep
   entry rule (EV≥2pp + depth≥$50): unrestricted baseline vs entries restricted
   to the near-tip window, both scored to the same last-pre-tip close.

A scoring fix shipped with this harness: a listings-eval entry whose snapshot is
itself the last pre-tip capture now reports CLV **None** (measurement missing)
instead of a fabricated 0.0 — without this the near-tip window numbers would be
zero-diluted.

## Data coverage (honesty first)

522 unique logged markets since 06-20 → **175 pairable** (34%). Drops: 269 no
in-window snapshot (June predates hourly watch-listings, launched 07-01; the
watch-closes launchd job only sweeps while the Mac is awake), 78 in-window
snapshot but no later pre-tip capture to close against. Of the 175 pairs, 54
had no fresh (≤2h) Pinnacle anchor at the rescan snapshot. In-window snapshot
sources: 149 regular scan sessions, 77 watch-closes, 28 watch-listings.

## Results — paired lens (window T-75..T-10, EV gate ≥2pp)

| Sector | MT | n | Actual CLV | Act %+ | Rescan CLV | Res %+ | Δ(res−act) | n gated | Gated CLV | Gated Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| tennis | moneyline | 49 | +1.90 | 59% | +0.96 | 33% | −0.94 | 4 | +9.50 | −0.00 |
| wnba | total | 39 | +0.49 | 44% | −0.08 | 31% | −0.56 | 5 | −1.20 | +1.20 |
| wnba | moneyline | 38 | +0.47 | 55% | −0.00 | 18% | −0.47 | 0 | — | — |
| baseball | moneyline | 17 | +0.94 | 35% | +0.06 | 18% | −0.88 | 6 | +0.00 | +0.67 |
| wnba | spread | 12 | −0.67 | 42% | +0.67 | 33% | +1.33 | 4 | +0.00 | +4.50 |
| worldcup | moneyline | 10 | −0.10 | 20% | −0.10 | 10% | −0.00 | 0 | — | — |
| worldcup | advance | 5 | −0.40 | 20% | +0.00 | 20% | +0.40 | 0 | — | — |
| cs2 | moneyline | 3 | +0.00 | 33% | +0.00 | 0% | +0.00 | 0 | — | — |
| lol | moneyline | 2 | +27.00 | 100% | +16.00 | 100% | −11.00 | 1 | +2.00 | +0.00 |
| **ALL** | all | **175** | **+1.08** | 48% | **+0.48** | 26% | **−0.60** | **20** | **+1.70** | **+1.40** |

Reading it right:

* The **ungated Δ is negative by construction** — it is mechanically
  (scan price − near-tip price), and +EV bets converge toward sharp fair
  pre-tip, so blindly re-buying the same side later is worse. This is NOT the
  workflow; it confirms the entries we already select do converge (+0.60pp
  drift between scan and the near-tip window on average).
* The **workflow's actual entry rule is the gated subset** — enter only when a
  fresh anchor still shows ≥2pp at the near-tip crossable price. That subset:
  **n=20, CLV +1.70pp, Δ +1.40pp vs the same bets' scan entries**. Positive and
  in the direction the placed-bet CLV analysis predicted, but n=20 ≪ 50.

## Results — archive-wide listings lens (gated market types)

Baseline (first anchored sweep any time) vs near-tip window entries, same
EV≥2pp + depth≥$50 gates, same close:

| Sector | MT | Side | Base n(clv) | Base CLV | Base %+ | Near n(clv) | Near CLV | Near %+ |
|---|---|---|---|---|---|---|---|---|
| baseball | spread | lay | 12(11) | +1.00 | 55% | 3(1) | +2.00 | 100% |
| baseball | spread | take | 23(16) | +0.62 | 38% | 11(2) | +0.50 | 50% |
| baseball | total | over | 86(73) | +1.12 | 49% | 34(10) | −0.10 | 0% |
| baseball | total | under | 86(72) | −1.26 | 17% | 38(10) | +0.00 | 10% |
| wnba | spread | lay | 69(64) | +0.39 | 52% | 19(2) | +1.50 | 100% |
| wnba | spread | take | 66(66) | +0.41 | 48% | 15(2) | +2.00 | 100% |
| wnba | total | over | 9(9) | −2.67 | 22% | 0 | — | — |
| wnba | total | under | 8(8) | +1.00 | 75% | 0 | — | — |

The near-window `n(clv)` collapse is structural: with hourly watch-listings
cadence, a near-tip entry is usually the **last** pre-tip capture, so no later
close exists to score against. Only watch-closes (5-min cadence, logged bets
only) fills that hole — and it doesn't cover unlogged ladder tickers.

## Window sensitivity

| Window | Gated n | Gated CLV | Gated Δ | Baseball-total near n(clv) / CLV |
|---|---|---|---|---|
| T-60..T-15 | 17 | +1.88 | +1.41 | 10 / +0.00 |
| T-75..T-10 (default) | 20 | +1.70 | +1.40 | 20 / −0.05 |
| T-90..T-10 | 17 | −0.29 | +1.65 | 30 / +0.03 |
| T-120..T-10 | 22 | −0.18 | +1.50 | 57 / −0.02 |

Gated Δ (near-tip gated entry vs the same bets' scan entry) is consistently
+1.4..+1.7pp, but the gated CLV level flips sign as the window widens — at
n≈20 the estimate is dominated by which single sweep lands first in the
window. Classic underpowered-fragile signature.

## Gate check → REJECTED

* **Overall:** gated entry rule shows Δ=+1.40pp but n=20 < 50. Fail.
* **baseball total:** near-tip entries at EV≥2pp are FLAT — CLV −0.10/+0.00pp
  at T-75 (n(clv)=20) and −0.02pp even at T-120 (n(clv)=57). No positive path.
  Consistent with the stale-line finding: by T-2h the line has converged and the
  residual "edge" doesn't move further. Totals stay disabled.
* **baseball spread:** n(clv)=3. No verdict possible.
* **wnba total:** paired gated n=5, CLV −1.20pp. No path.
* **wnba spread:** the only mildly encouraging gated type — paired Δ +1.33pp
  (n=12), listings near-window +0.74..+1.75pp — but every cell is single-digit
  n(clv). Matches the existing `clv --side lay` promotion-lens story; adds no
  new evidence at this power.

## What was NOT done (per the phase gate)

No `evmax agents rescan` command, no schema change, no categories.yaml change,
no scheduling. The harness is committed as research infrastructure only.

## What would change the verdict

The binding constraint is **capture density, not methodology**. watch-closes
(5-min, both venues, live+shadow) and watch-listings (hourly, all game sectors)
now run unattended; the paired sample grows ~10-15/week at current bet volume
and faster if watch-closes' lookahead window is widened. Re-run after ~4-6 more
weeks of capture:

```bash
python scripts/eval_near_close_rescan.py                    # default gate check
python scripts/eval_near_close_rescan.py --detail           # per-bet rows
python scripts/eval_near_close_rescan.py --window-lo 60 --window-hi 15
```

Reconsider Phase 2 (the `rescan` command + shadow persistence) when the gated
subset reaches n≥50 with Δ≥+1pp, or a gated market type shows gated CLV ≥+1pp
on n≥30.
