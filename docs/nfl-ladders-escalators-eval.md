# Kalshi NFL Ladders & Escalators — EV evaluation (2026-10-09)

**Verdict: do not wire these products into evmax yet.** Ladders are priced at
fair value. Escalators look rich against Kalshi's own binary props, but the gap
disappears once the binary mids are corrected for their own calibration. The
remaining edge is between 0 and +0.6c per contract, on about 95c of capital,
and it is not statistically established. Re-run the lens late in the season
(see "Revisit gate").

Tool: `scripts/eval_nfl_scalar_props.py` (read-only; `--fetch` refreshes a
public-API cache under `data/backtest/nfl_scalar_props/`). In a worktree, pass
`--archive /path/to/main/data/archive.db` — a worktree's own `data/archive.db`
is empty.

## Products

Kalshi pays YES a fraction of $1 that depends on the stat line. All series are
fee type `quadratic` (taker 0.07·p·(1−p); maker fee presumably zero — unverified).

| Series | YES payout | Cap |
|---|---|---|
| `KXNFLLADDERRECYDS`, `KXNFLLADDERRSHYDS` | 0.25c per yard (linear) | 400 yd |
| `KXNFLLADDERREC` | 5c per reception | 20 |
| `KXNFLFFPTSLADDER` | linear in Sleeper fantasy points | 100 pts |
| `KXNFLESCALATORRECYDS`, `KXNFLESCALATORRSHYDS` | floor(1e4·(y/200)³)/1e4, y = 10-yd floor | 200 yd |
| `KXNFLESCALATORREC` | floor(1e4·(s/14)³)/1e4 | 14 |

First settlements: 2026-09-20 (fantasy ladder), 2026-09-24 (yards/receptions).
Pre-kickoff traded YES premium, 09-24 → 10-08: ladders $154k (rec yds), $161k
(receptions), $24k (rush yds); escalators $102k / $46k / $69k; fantasy ladder
$150k. The market maker quotes ~4–5k contracts per side, about 7c wide two days
out and usually one tick wide by T-24h.

## Method

- **Sample.** 929 settled yards/receptions markets (~33 games, 10–12 kickoff
  slates) plus 227 fantasy ladders. The payout schedule above reproduces
  Kalshi's `settlement_value_dollars` on all 929.
- **Quotes.** Hourly bid/ask candles (public `candlesticks` endpoint) at T-24h
  and T-1h. Kickoff = `occurrence_datetime` − 3h (equal to Pinnacle's start
  time on 1160/1160 markets).
- **Fair value (KB).** Replicate the payoff from Kalshi's binary props for the
  same player (`KXNFLRECYDS` etc., 10-yd thresholds, archived hourly) at the
  bid/ask mid ≤6h before the entry time. P(Y ≥ y) is linear in logit between
  thresholds; tails are extrapolated.
- **Scoring.** Kalshi's settlement; taker P&L after fee; SE clustered by
  kickoff slate (games inside one slate share the market maker's regime — the
  2026-10-04 1pm slate had 50–80c-wide quotes at T-1h and dominated a
  game-clustered first pass).

## Results (quotes with spread ≤ 2c; cents per contract)

| Series | T- | n | Mid | KB | Settled | Mid/KB | Rich slates | NO@bid expected | NO@bid realized |
|---|---|---|---|---|---|---|---|---|---|
| Escalator rec yds | 24h | 157 | 4.73 | 3.83 | 3.80 | +27% | 10/10 | +0.59 | +0.62 ± 0.76 |
| Escalator rec yds | 1h | 147 | 4.70 | 3.76 | 3.69 | +33% | 11/11 | +0.48 | +0.56 ± 0.93 |
| Escalator rush yds | 24h | 84 | 5.21 | 4.41 | 4.32 | +24% | 8/10 | +0.46 | +0.55 ± 1.10 |
| Escalator receptions | 24h | 156 | 6.68 | 5.86 | 6.85 | +15% | 10/10 | +0.37 | −0.62 ± 1.17 |
| Ladder rec yds | 24h | 158 | 11.37 | 10.99 | 11.00 | +3% | 9/10 | −0.78 | −0.79 ± 0.55 |
| Ladder rush yds | 24h | 85 | 12.82 | 12.43 | 12.24 | +3% | 8/10 | −0.85 | −0.65 ± 0.82 |
| Ladder receptions | 24h | 155 | 19.90 | 19.88 | 20.57 | 0% | 6/10 | −1.54 | −2.23 ± 0.88 |

- **Ladders are at fair value.** Taking YES at the ask loses ~1.6c, taking NO
  at the bid 0.6–2.2c — the half-spread plus the fee. Fantasy ladder: T-24h
  mid matched settlement (+0.02c); takers lost ~1.7c on both sides.
- **Escalators sit above KB in nearly every slate.** The gap survives a 2×
  fatter tail assumption.
- **The binary mids are not exactly fair.** Against Kalshi's official
  settlement (15,434 binaries, 60–65 games), YES resolves 1.5–2.0pp above the
  mid in the 5–35% band (T-1h: +2.0 ± 0.9, +1.7 ± 1.2, +1.5 ± 1.3 pp).
  `--bias-correct` shifts the binary mids by their measured calibration before
  replicating. Escalator KB rises 0.3–0.5c, and the expected NO@bid edge
  becomes **−0.16c to +0.08c**: zero.
- **Realized results are noisy and tail-driven.** One 165-yd rushing game paid
  $0.51 per YES contract. Realized NO@bid on yards escalators is +0.55 to
  +0.75c with SE 0.7–1.1c.
- **No hedged arbitrage.** The binary props never span the escalator payoff (0
  of ~170 markets cover >95% of payout weight), and binary spreads (1.4–3c)
  exceed the gap.
- **Trade tape.** On yards escalators, YES-takers paid +0.6–0.7c over KB, but
  resting bids were hit by NO-takers at −1.2 to −1.6c, so the resting side
  netted −0.2 to −0.4c per contract overall. On ladders the resting side lost
  1.2–1.6c against KB.

## Related findings on NFL props

- **Pinnacle is not sharper than Kalshi on props.** Pinnacle's devigged main
  line, recovered exactly from the archived rungs, scored against Kalshi's
  binary mid on the same event (T-1h, Brier/1000, Pinnacle − Kalshi): rec yds
  −0.20 ± 0.70 (n=436), receptions +0.97 ± 1.13 (n=547), rush yds +1.68 ± 1.04
  (n=208). Each venue moves toward the other by the same amount (regression
  slopes +0.22 and +0.23); the mean gap is 1.5pp. A Pinnacle-anchored prop
  model therefore has no information edge at the main line; any edge has to
  come from distribution shape, projections or news timing.
- **Main-line overs look rich on both venues (watch item, not significant).**
  Realized minus priced at the main line, T-1h: receptions −3.7pp (±2.0 by
  game, ±2.2 by week), rushing yards −5.4pp (±3.0 / ±2.5), receiving yards
  +0.5pp. This matches the historical sportsbook under-lean on rushing yards.
  NO at Kalshi's bid on the main-line receptions threshold made only +0.8 ±
  2.0c after the fee.
- **Buying NO blindly loses.** NO at the bid lost 0.3–3.5c in every price
  bucket at T-1h. This is direct evidence against the MODEL-9 backtest's NO-side
  ROI, which used closing prices.
- **Resolver bias (fixed with this doc).** ESPN omits a player who played but
  recorded no stat, so `_resolve_nfl_prop_observations` left his rows pending
  while Kalshi settled them NO (340 markets, 61 player-games, Weeks 1–6).
  Every NFL prop calibration and shadow ROI before the fix is biased toward
  YES. Kalshi-venue rows ESPN cannot grade now fall back to Kalshi's
  settlement.
- **Archived "pinnacle" NFL prop rungs are model output.** Before the prop
  dispersion fix, the coordinator re-lined one Pinnacle main line through a
  fixed-σ Normal and archived every rung as `book='pinnacle'` (identical
  decimals on every rung of a player). That model's tail was too thin for
  high-median receivers (P(≥ median+40 yd) 5.0% vs 12.2% realized) and too wide
  for low-median ones: shadow YES rows flagged at EV > 15% returned −17% ROI
  (n=3,806), and rows priced ≤ 10c returned −32% (n=2,174) — on the YES-biased
  outcomes above, so the true figures are worse.

## Cost of wiring scalar products

Scalar payoffs break binary assumptions in `MarketType`, `EV = p × 1/price`
(`evmax/ev/calculator.py`), binary Kelly (`evmax/ev/kelly.py`), the INTEGER
`outcome` columns, and the resolver (Kalshi settles with `result="scalar"` plus
a dollar value). The Kalshi client would also need to parse `custom_strike`
and 0.0001 escalator ticks. That is a multi-module shadow pipeline for an edge
that may be zero.

## Revisit gate

Re-run with `--fetch --bias-correct` after about eight more weeks of games.
Build a shadow pipeline only if, by kickoff slate, the escalator NO@bid
realized P&L is ≥ +0.5c with z ≥ 2 **and** the bias-corrected expected edge is
positive. Kalshi keeps candle and settlement history on its public API, so
nothing has to be archived in the meantime; the binary-prop archive already
accrues from the nfl_props shadow scans.
