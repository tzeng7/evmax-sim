# NFL moneyline venue-lag lens — pre-registration

Written 2026-10-24, BEFORE the evaluator was run on any data. Do not edit the
decision rules below after seeing results; add a dated addendum instead.

## Question

The `nfl` ML promotion gate (n>=30 clean rows) only counts rows the scanner
flagged at EV>=2%. Kalshi/PolyUS NFL ML sit ~0.5pp from Pinnacle, so flagged
rows accrue ~2/week. This lens asks the underlying mechanism question on EVERY
matched game: **does Kalshi's NFL moneyline ask converge toward the devigged
Pinnacle price between the first anchored sweep and the last pre-tip snapshot,
and is the convergence large enough to pay the taker fee?**

## Universe and entry rule

- Source: `archive.db`, sector `nfl`, `KXNFLGAME` tickers, sessions
  `watchlist*` and `watchclos*` (Kalshi side); `archived_sharp_odds` game-level
  moneyline records `nfl::<ET date>::<a>_vs_<b>` (Pinnacle side, tip =
  `event_date`).
- Unit = one (game, team YES contract). Both teams of every matched game enter.
- Entry = the first pre-tip snapshot with a Pinnacle anchor fetched at most 2h
  before (0.25h after, for sweep skew) and lead time >= 0.5h. Entry price = the
  snapshot `yes_price` (ask). Edge_pp = (fair - ask) * 100 with fair = devigged
  Pinnacle prob of that team.
- Close = the last pre-tip snapshot at least 1h after entry and within 3h of
  tip. Same `yes_price` column as entry. Rows without a valid close get NO CLV
  (never 0). Asks >=0.99 or <=0.01 are placeholders and are dropped.
- Gross CLV_pp = (close_ask - entry_ask) * 100. Net CLV_pp = gross minus the
  Kalshi taker fee at the entry price (`fees.kalshi_fee_prob`, pp). Entry fee
  only: CLV is mark-to-close, not a sale.

## Statistics (all clustered by GAME; the two sides of a game are mirror images)

1. Convergence slope: OLS of CLV_pp on edge_pp over all units, cluster-robust
   (CR0) by game. Mechanism test: b > 0.
2. Flagged analogue: units with edge_pp >= 2.0. Reported with and without the
   depth>=$50 gate (depth exists only from 2026-10-03). Mean per-game CLV,
   %positive games, t over games.
3. Edge buckets (<-2, -2..0, 0..1, 1..2, 2+) for shape, descriptive only.

## Decision rule (fixed now)

- **SUPPORTED** (venue lag is real and fee-positive): slope b > 0 with
  cluster z >= 1.64 AND flagged-analogue NET mean CLV >= 0 over >= 30 games with
  cluster t >= 1.64.
- **LAG-BUT-FEE-NEGATIVE**: slope z >= 1.64 but flagged-analogue net mean < 0.
- **REFUTED**: slope z < 1.64 with >= 60 games, or slope <= 0.
- **UNDERPOWERED**: anything else (fewer games than needed).

## What this does and does not do

- Read-only on the DBs. Changes no mode, gate, weight or sizing.
- A SUPPORTED verdict is evidence for keeping/raising the NFL ML lane on
  venue-timing grounds; it does not replace the live-eligible shadow gate and
  does not by itself authorize sizing changes.
- Known biases: entry is a snapshot ask, not a fill; no depth before 2026-10-03;
  the sample is the 2026 season only; the tip comes from Pinnacle `event_date`.
