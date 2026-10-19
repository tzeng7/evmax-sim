# CLV promotion gate v2 — scope

Status: SCOPE ONLY (2026-10-18). No gate code has changed. Numbers come from a
fee-net audit of `predictions.db` (first-logged row per `market_id`, rows with
`ev_pct >= 2%`, game-clustered; scripts in the session scratchpad, not
committed). Re-run before trusting any figure after more data accrues.

## 1. Why the current gate cannot be passed

The gate is `clv_clears(n, mean_clv_pp, frac_positive)` in
`evmax/cli/commands/shadow.py:513-528`: games >= 30, mean >= 0.0pp,
fraction positive >= 0.55. `_aggregate_clv` (`shadow.py:698-723`) computes
`frac_positive` over ROWS (`c > 0`), not games.

| # | Defect | Evidence |
|---|---|---|
| 1 | Exact zeros count as failures. | 17-47% of rows have CLV exactly 0 (price never moved): tennis Kalshi 47%, baseball ML 41%, cs2 42%. With 47% zeros the ceiling on `%pos` is 53% — below the 55% bar before any edge is considered. |
| 2 | The mean is outlier-driven. | Soccer Kalshi mean +1.07pp vs median 0; one row at +70pp. Tennis max +52pp. Medians are 0 in most cells. |
| 3 | CLV is gross of fees. | Taker fee ~1.3-1.7pp at mid prices. Typical gross mean +0.4 to +1.1pp, so almost every cell is negative net (tennis Kalshi -1.06pp, WNBA ML -1.0pp, baseball ML -0.9pp). |
| 4 | Row-level `%pos` vs game-level sample size. | Rungs and duplicate rows of one game dominate `%pos` while the sample-size half counts games. |
| 5 | No power concept. | UCL has 11 games after 3 matchdays; the gate reads COLLECTING forever. No lane can be told "this cannot reach the gate this season". |
| 6 | Sharp-passthrough has no promotion path. | `promotion_board._verdict` (`promotion_board.py:130-142`) returns SHARP-PASSTHROUGH for any moneyline with blend divergence < 0.5pp, before the CLV gate is consulted. Every live soccer and tennis ML sits there. |
| 7 | CLV and realized ROI are never compared. | NFL total PolyUS (would-bet rows, 66 games): median CLV +2.0pp, 87% of games positive, +0.73pp net — but realized ROI -17.4% (CI -31..-3). The gate clears it on CLV alone. |
| 8 | The scored population is not the bet population. | `clv_stats` scores every logged row; a lane only bets rows passing the scan gate. |

Open question (must be resolved in Phase 0): logged `ev_pct` fits the GROSS
formula `blend/price - 1` slightly better than the fee-net one (mean abs diff
0.014 vs 0.026 on 1,593 tennis ML rows). Confirm whether `ev_pct` is net of fee
at log time; if it is gross, the "would-bet" population contains rows that are
negative after fees and the population definition must use a fee-net EV.

## 2. Validation set (known verdicts the new gate must reproduce)

Pre-register these expectations BEFORE computing v2 (would-bet rows):

| Cell | Expected v2 verdict | Why |
|---|---|---|
| baseball total Kalshi | FAIL | median -0.5pp, 14% of games positive, net -2.1pp |
| wnba spread Kalshi | FAIL | mean -0.28pp, 37% positive, net -1.6pp |
| nba ML / spread Kalshi | FAIL | median -1.0pp, 27% positive |
| soccer ML Kalshi | FAIL | net -0.5pp, median 0 |
| tennis ML Kalshi, PolyUS | FAIL | net -1.06 / -0.81pp |
| nfl spread Kalshi / PolyUS | UNDECIDED (net -0.23 / +0.06pp) | the maker-pilot lane decides |
| nfl total PolyUS | CONTESTED | CLV passes, ROI veto must fire or not — this cell tests the veto rule |

If v2 clears a FAIL row, v2 is wrong, not the data.

## 3. Proposed gate v2

Unit and population
- One observation per GAME-SIDE: the mean of that game's rows' CLV (kills rung
  and duplicate dependence).
- Population = rows the lane would bet: `ev_pct >= ev_threshold` on the logged
  price AND `has_full_blend` (already in `live_eligible_only`).

Primary statistic (all must hold)
1. `games >= 30` (unchanged).
2. Net CLV: game-level CLV minus the entry fee in pp
   (`evmax/fees.py`, venue taker rate by default; maker rate when the lane is
   maker-only, e.g. the NFL maker pilot).
3. 5%-trimmed mean of net CLV, with the game-bootstrap 80% lower bound >= 0.
4. Share positive among MOVERS (games with CLV != 0) >= 55%; `%zero` reported
   next to it so a flat market is visible, not hidden in the denominator.

Veto
5. Realized ROI, game-clustered 80% interval, must not lie entirely below 0
   (upper bound > 0). ROI is a veto only, never a pass condition: at 40-80
   games its standard error is ~25-30%, so it cannot confirm an edge but can
   contradict one.

Power verdict (new)
6. Estimate the robust SD of game-level net CLV. Required games for the
   observed SD to detect a +0.5pp net edge at 80% power: `n = (2.8 * SD / 0.5)^2`.
   If that exceeds what the lane can log before the season ends, the verdict is
   UNDERPOWERED (not COLLECTING). Indicative game-level SDs: NFL total PolyUS
   1.05pp, tennis Kalshi 2.31pp, baseball ML 2.73pp, soccer Kalshi 6.41pp
   (outlier-inflated; use a robust SD).

Verdict ladder replacing the current one
- UNDERPOWERED, COLLECTING n/30, FAILING-NET-CLV, CONTESTED (CLV passes, ROI
  veto fires), PROMOTE-READY, LIVE-HEALTHY, LIVE-DEGRADING.
- SHARP-PASSTHROUGH becomes a label on a lane (no model signal), NOT a
  verdict that skips the gate: a passthrough lane can promote on net CLV alone.

Unchanged: per-venue and per-side judging; contamination filters;
`live_eligible_only`; `max_staleness_h` (a stale close creates fake zeros).

Thresholds above are proposals. Calibrate them on the validation set in section
2 before shipping; do not tune them on the cells you are trying to promote.

## 4. Code touch points

| File | Change |
|---|---|
| `evmax/cli/commands/shadow.py:513-528` | Replace constants and `clv_clears` with a `GateResult` (verdict, net trimmed mean, LB, pos-among-movers, %zero, ROI interval, required games). Keep `clv_clears` as the v1 path behind `CLV_GATE_VERSION`. |
| `shadow.py:698-723` `_aggregate_clv` | Return game-level observations plus fee pp; compute v2 beside v1. |
| `shadow.py:726` `clv_stats` | Add `population='would_bet'|'all'`; thread fee model and ROI. |
| `shadow.py` report lines at ~901, 1007, 1138, 1245, 1288, 1441, 1535-1545 | New gate text and columns (median, %zero, net, LB). |
| `evmax/agents/cleanup/promotion_board.py:130-170` `_verdict` | New ladder; remove the SHARP-PASSTHROUGH short-circuit (keep as a flag column). |
| `evmax/agents/cleanup/integrity.py:89, 660-667` `GATE_WATCHES` | Consume `GateResult`; alert wording. |
| `evmax/ev/maker_pilot.py:177` | Replace the local copy of the constants with the shared gate. |
| `evmax/agents/cleanup/value_audit.py:91`, `rescan_eval.py:454` | Re-point their `_clv_stats` / `_pct_pos` to the shared helpers (display only). |
| `evmax/fees.py` | Expose a pure `entry_fee_pp(venue, price, maker=False)` if not already public. |
| Tests | `tests/test_cli_shadow.py`, `tests/test_promotion_board.py`, `tests/test_integrity.py`, plus a new `tests/test_clv_gate_v2.py` pinning the validation-set expectations on synthetic fixtures (zero-heavy, outlier-heavy, fee-negative, ROI-contradiction). |
| Docs | CLAUDE.md "CLV measurement rules" and Promotion board paragraphs. |

## 5. Rollout

0. Compute v2 next to v1 (extra columns in `shadow board` and `clv`). No
   verdict or YAML changes. Resolve the `ev_pct` fee question.
1. Run the section-2 validation set. Any FAIL row that v2 clears blocks the
   rollout.
2. Switch the board and integrity watches to v2 behind `CLV_GATE_VERSION=2`
   (default off for one week, then on). `shadow promote` / `promote-league`
   keep requiring a v2 PROMOTE-READY.
3. Delete v1 after two clean weekly integrity sweeps.

## 6. Risks and open questions

- CLV coverage: only Kalshi closes are stored natively; PolyUS tennis edge rows
  had CLV on 17 of 42 rows. Low coverage biases the PolyUS lens; report
  coverage next to every gate.
- Multiple looks: the board shows ~28 cells weekly. Expect one or two to look
  significant by chance. Pre-register the cells you intend to promote.
- CLV positive but ROI negative (NFL totals PolyUS) may mean the close is not
  efficient for extrapolated totals; the veto rule exists for exactly this
  case, and this cell is the test.
- A maker-only lane needs its own fee and a fill-rate adjustment (the NFL
  pilot's replay showed ~35% touch-fill); net CLV must be scored on FILLED rows
  only or it overstates the lane.
- Throughput: some lanes will read UNDERPOWERED for a whole season. That is an
  honest answer; the action then is to cut or to widen the population (pool
  leagues), not to wait.
