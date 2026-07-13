# Golf module (`evmax/golf/`) — design, data, and edge findings

A **standalone** golf modelling + market-pricing module. Deliberately NOT wired
into `data/categories.yaml` / `SECTOR_SERIES_MAP` / the scan pipeline: golf's
flagship markets are **fields** (a 150-player tournament outright, one YES
contract per golfer, plus top-N / make-cut / round-leader), and the entire
evmax scan pipeline is hard-wired to 2-way / 3-way head-to-head events
(pairwise matching keys, `a/b/draw` odds & blend shapes, the `predict_pair`
model primitive, and no N-outcome `MarketType`). Forcing golf in would crash
the import-time `validate_registry()` and destabilise the pairwise pipeline, so
golf lives on its own and only *reuses* the one primitive that already
generalises to N outcomes: `evmax.ev.devig.devig_power_method`.

## Pipeline

```
data (ESPN field/scores + PGA Tour SG)   evmax/golf/data/{espn,pga}.py
  → skill model (per-category SG, shrinkage, variance)   evmax/golf/skill.py
  → Monte-Carlo field simulator (win/top-N/make-cut)     evmax/golf/simulator.py
  → venue field prices (Kalshi + PolyUS)                 evmax/golf/venues/*
  → field devig (spread-aware)                           evmax/golf/devig.py
  → edge = model_prob − market_prob                      evmax/golf/pricing.py
  → run / render                                         evmax/golf/run.py
```

Run it: `python -m evmax.golf.run` (offline, from fixtures) or
`python -m evmax.golf.run --live` (live PolyUS + PGA SG, + Kalshi if RSA creds).

## Markets (verified live 2026-07)

| Venue | How | Coverage |
|---|---|---|
| **Kalshi** | series `KXPGATOUR-{EVENT}-{CODE}`, golfer in `yes_sub_title`; prices need the **authenticated** trade API (public elections endpoint returns null bid/ask) | Winner wired. Also live but not wired: `KXPGACUTLINE` (cut-line value), `KXPGAR3TOP10`, `KXPGAMAJORTOP10`, `KXPGAPLAYERCAT`, H2H/3-ball |
| **PolyUS** | `GET /v2/sports/golf/events?type=futures`; one event per (tournament, family); golfer in `market.titleShort`, `outcomePrices=[yes_bid, yes_ask]` | **Winner + R1/R2/R3 leader**, per-tournament league slugs (`theopen`, `masters`, `pgacham`, …). Public prices. No top-N / make-cut currently |

**Field-devig subtlety.** A winner field's true probs sum to 1, but top-N sums
to N and make-cut to ~cut. So a single "normalise to 1" is only right for the
winner/leader markets. Both venues quote YES *and* NO per contract, so the
default is **per-contract 2-way devig** (market-type-agnostic). For the
exactly-one-winner markets we instead build a **liquidity-robust** per-golfer
implied (mid when the quote is tight, **bid** when it's wide) and normalise to 1
— illiquid longshots post placeholder asks (a no-hoper quoted 0.45 ask) that
would otherwise steal mass from the favourites.

## Data — the free-data ceiling

- **True per-round SG split (OTT/APP/ARG/PUTT) is not free** — only DataGolf
  (paid, Scratch Plus) and ShotLink (gated, bars betting use).
- **What IS free:** per-*season* SG per category per player from the PGA Tour's
  own public GraphQL (`orchestrator.pgatour.com/graphql`, fixed public
  `x-api-key`, `statDetails` op; the `Avg` value is SG **per round**). Confirmed
  live: Scheffler 2.154 SG/round, 157 players. Historical per-tournament SG
  2015-2022 is in the Kaggle ASA set. So the grain is **per-event/season, not
  per-round** — the skill model updates accordingly.
- **ESPN** (free, unauth) gives the field, per-round scores, finishing state —
  no SG. It is the FIELD source and a **score-derived SG-total proxy** source
  (`field_mean_to_par_per_round − player_to_par_per_round`), the no-SG fallback
  so the whole chain runs offline / without any paid key.

## Skill model

`skill_from_pga` (real SG): predictive SG = Σ retention_c·category_sg with
category-specific shrinkage (OTT/APP retain most; ARG/PUTT regress hard), then
sample-size-shrunk toward replacement. `skill_from_records` (score-proxy /
historical): per-event SG with exponential recency decay. Both emit
`{name: PlayerSkill}` → `to_sim_inputs` re-centres skill to the *field* (a weak
Corales field and a major are different baselines) and hands mean/SD arrays to
the simulator.

**Known blind spot:** PGA-only SG can't see LIV players (Rahm, DeChambeau) or
DP-World-only players — they get bucketed at replacement level. The score-proxy
path (ESPN, all tours) covers them and is the intended fix.

## Simulator

Vectorised numpy Monte-Carlo (20k sims × 150 players ≈ 0.2s): 4 rounds ~
N(mean, sd), within-player round correlation ρ≈0.08 (a shared "form" offset), a
shared per-round course-difficulty shock (fattens tails), a 36-hole cut, and
missed-cut players ranked behind made-cut players. One set of sims prices every
market. Invariants (tested): win probs sum to 1, make-cut sums to cut_size,
top-N sums to N, `win ≤ top5 ≤ top10 ≤ top20 ≤ make_cut`, seed-deterministic.

## Edge findings (honest — first slice)

On the live **2026 Open Championship** winner market (PolyUS, ~$240k volume):

- **The model reproduces the market ordering** — Scheffler > McIlroy >
  Fitzpatrick, favourites are favourites. The free SG signal is real.
- **But the model is systematically UNDER-confident vs the efficient major
  market.** Model has Scheffler ~4-5% win; the devigged market has him ~15%.
  This is **not** a shrinkage artifact (holds at near-zero shrinkage) and not
  purely the weak-tail blind spot — a season-SG + Gaussian sim simply produces
  a flatter winner distribution than a top-heavy, sharp, high-volume major
  market. **No outright favourite clears the edge-vs-ask threshold**, so the
  module flags **zero** outright plays there. This is locked in as a regression
  guard (`TestHonestUnderconfidenceFinding`); the harness is separately proven
  to fire when edge genuinely exists (`TestHarnessCanDetectEdge`).

The module is therefore an **observation harness**, not (yet) a bet source. It
does not manufacture edge.

## Walk-forward backtest (`evmax/golf/backtest.py`, `scripts/backtest_golf.py`)

A leak-free point-in-time backtest that explains the live finding. For every
tournament it builds skill from ONLY strictly-earlier events (the current
event's result is folded in AFTER prediction — no lookahead, the property most
likely to fake edge; guarded by `test_no_future_records_leak`). It backtests the
ESPN **score-proxy** skill path because that is the only skill signal
reconstructable leak-free at each point in time (the PGA season-SG snapshot
already contains post-event rounds and would leak).

Run: `python scripts/backtest_golf.py` (2025 + 2026 PGA, free ESPN, cached).

Result over **45 scored events / 8,105 predictions** (2025-26):

| market | model Brier | baseline | skill vs baseline |
|---|---|---|---|
| win | 0.0082 | 0.0084 | **+2.4%** |
| top-10 | 0.0836 | 0.0889 | **+6.0%** |
| make-cut | 0.2410 | 0.2473 | **+2.6%** |

(make-cut excludes no-cut events — signature events / Tour Championship where the
whole field plays 4 rounds — which otherwise falsely inflate under-confidence.)

The model **beats the naive base-rate baseline on all three markets** — the
ranking signal from free score-proxy SG is real (its top win pick wins 10.6% of
the time vs ~1% random; it assigns eventual winners ~3× a flat model's prob).

**But it is systematically UNDER-confident** — the calibration curves all bend
the same way (predicted < actual in the upper bins):

```
top-10   pred 13.5% → actual 15.1% ;  pred 55% → actual 74%
make-cut pred 52%   → actual 61%
win      pred 13%   → actual 22%   (small n, but consistent)
```

This **explains the live-market read**: at the Open the model rated Scheffler
~4% while the market had him ~15% — not because the market was soft, but because
the model is globally under-confident and the market was *right*. So there is no
edge fading favourites; the market beats a naive free-data model on the big
liquid outright. The model's **rankings are sound**, but its probability
*levels* are compressed.

## Recalibration + edge reality-check (`evmax/golf/calibration.py`, `scripts/golf_calibration.py`)

Built an isotonic (pool-adjacent-violators, numpy-only) recalibration layer and
tested it **out-of-sample the honest way**: fit the map on 2025 walk-forward
predictions, evaluate on 2026 (a temporal holdout the map never sees). This
corrected the earlier read.

**The under-confidence was mostly small-sample noise, not a robust signal.**
Out-of-sample, recalibration only helps **make-cut** (holdout Brier 0.2407 →
0.2340, +2.8%) — the one market with thousands of dense mid-range predictions and
a real, generalisable gap (pred 52% → actual 62%). For **win** (−5.2%) and
**top-10** (−1.3%) recalibration made the holdout Brier *worse*: those markets'
apparent under-confidence lived almost entirely in sparse high-probability tail
bins (n=3, n=14, n=16), which the isotonic fit overfits (it pushed a 55% top-10
bucket to 100%). Raw top-10 was already close to calibrated out-of-sample
(pred 24% → actual 27%). So the dramatic "pred 55 → actual 74" from the pooled
in-sample table was a small-n artifact — exactly what the out-of-sample test is
for.

**No edge survives on the winner market (the null).** Applying the calibrated
model to the live Open field: mean |model − market| over 26 contenders was 1.29%
raw / 1.40% calibrated — recalibration did **not** close the gap, but the
residual disagreement is dominated by the *unreliable* win calibrator (which we
just showed doesn't generalise): it overshoots Scheffler to 18% (market 15%) and
undershoots everyone else. That is calibration noise, not disagreement-and-right.
There is **no evidence of edge** on the liquid outright market; the market is
simply more precise than a free-data model.

## Non-outright markets (make-cut / top-N / matchups)

Checked what's actually live (2026-07, the Open week):

- **Matchups (H2H / 3-ball)** — the avenue that best fits evmax (a Pinnacle
  sharp anchor, and a fundamentally easier prediction: the relative ordering of
  two well-covered players, which sidesteps the field-coverage weakness) — are
  **not a live product on either venue.** Can't test what isn't offered.
- **Make-cut, top-5, top-10** — **live on Kalshi** (`KXPGAMAKECUT`, `KXPGATOP5`,
  `KXPGATOP10`; full 163-golfer fields per major), **not on PolyUS** (winner +
  round-leaders only). Wired into the Kalshi adapter (`SERIES_MARKET_TYPE`); the
  simulator already emits these probabilities, so the module prices them
  directly. But: no Pinnacle sharp anchor for them, and pulling live Kalshi
  prices needs RSA creds.

**The structural reason non-outrights likely inherit the outright result:** win,
top-N, and make-cut are ALL reductions of the same field score distribution —
for us AND for the market maker. We showed our distribution is less precise than
the market's on the winner market; that same distribution feeds our make-cut /
top-N prices, so the strong prior is that the market maker's field model beats us
there too. Make-cut is our best-*calibrated* market (the only one that survives
out-of-sample recalibration), so it's the least-bad candidate — but
"well-calibrated in absolute terms" is not "beats the market." The only market
that would genuinely sidestep our weakness is matchups, and those aren't offered.

**The one clean forward test available** — BUILT (`evmax/golf/capture.py`,
`scripts/golf_capture.py`). There is no historical golf-odds series for a CLV
backtest and no sharp anchor for make-cut / top-N, so the only honest test is to
freeze (model prob, market price) pairs before a tournament and score them after
it resolves.

```bash
# CAPTURE (before/at tournament start) — needs Kalshi RSA creds, OR a manually
# pulled markets JSON so it works without wiring creds:
python scripts/golf_capture.py capture --event THOC26 --tournament "The Open" --calibrate
python scripts/golf_capture.py capture --event THOC26 --tournament "The Open" \
    --markets-file open_kalshi.json --calibrate      # creds-free path

# SCORE (after the final round — this weekend for the Open):
python scripts/golf_capture.py score --snapshot "data/golf_captures/THOC26_*.json"
```

The scorer reports, per market: model Brier vs market Brier vs calibrated Brier;
who was closer on the biggest model-vs-market disagreements; and — the decisive
number — the realised ROI / hit-rate of betting the model's flagged +EV contracts
at the ask (buy YES for ask `a`, settle at outcome ∈ {0,1}, profit `outcome − a`).
Positive ROI on flagged bets = the edge was real; otherwise it was noise. Snapshots
persist to `data/golf_captures/` (git-ignored). The harness is verified end-to-end
offline; it needs only live Kalshi prices (creds or `--markets-file`) to run for
real.

**Verdict:** on outrights, fold — the market beats the model and the model can't
be made to beat it with free data + recalibration. On non-outrights: matchups
(the promising avenue) aren't a product; make-cut / top-N are priceable but are
reductions of the same beaten distribution with no sharp anchor and no completed
forward test yet. The only robust, fixable
model property is make-cut calibration, and we cannot currently price make-cut
against a market (neither venue offers a broad per-golfer make-cut market, and we
have no historical golf odds for a CLV backtest). Any future attempt should:
(1) target markets we can actually price and that are less efficient than a major
outright; (2) keep CLV as the health metric; (3) treat richer data (DataGolf
per-round SG, LIV/DP coverage) as the *next* lever only after a calibrated model
shows disagreement-that's-right somewhere — the interfaces are built for that
drop-in, but the evidence doesn't justify the spend yet.

**On DataGolf:** the coverage gap is real (on a thin opposite-field event only
~37% of the field had PGA SG; the rest ran on replacement-level guesses), and
DataGolf's global-tour per-round SG would fix exactly that. But calibration —
not data richness — is the current binding constraint, and calibration is free
to fix. Sequence: recalibrate on free data → prove edge somewhere → only then
pay for DataGolf if per-round SG / LIV+DP coverage is the next bottleneck. The
`SGRecord` / `PlayerSkill` interfaces are built so a DataGolf adapter drops in
without touching the simulator or pricing.

## Tests

`tests/golf/` (69 tests, all green; fixtures captured live):
capability (ESPN/PGA/Kalshi/PolyUS parsing, simulator invariants + determinism,
field devig shapes) + **edge observation** (full-pipeline coherence, ordering,
the under-confidence regression guard, and the harness-detects-edge proof).
