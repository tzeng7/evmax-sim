# Elo H2H + rest read key — walk-forward evaluation, 2026-09-23

**Verdict: rest SHIPPED for NBA and NCAAB only (`REST_RESOLVED_SECTORS`). Rest stays as-is for
WNBA (dead) and soccer (partial). H2H is REJECTED in every sector tested and keeps its raw-label
read.**

Harness: `scripts/backtest_elo_h2h_rest.py` (`--coverage` for §1, `--kalshi-clv` for the Kalshi
lens). Tests: `tests/test_elo_rest_days.py`, `tests/test_backtest_elo_h2h_rest.py`.

## 1. The defect

`EloModelAgent` reads two prediction-time layers under the raw lowercased Pinnacle label:

- **H2H** — `update()` records the pair under the names it is fed (canonical slugs such as
  `lakers`). `_h2h_adjustment` looks the pair up under the label (`los angeles lakers`).
- **Rest** — `_days_of_rest` and `_congestion_penalty` read `form_state.json` with
  `form[sector].get(label)`. The Form agent keys that file by its resolved key.

When the label differs from the key, both layers return 0. Archived Pinnacle moneyline labels
(`--coverage`, run 2026-09-23 against `archive.db` and the committed state):

| sector | labels | rest table | rest hits raw → resolved | H2H pairs | H2H fires raw → resolved |
|---|---|---|---|---|---|
| nba | 30 | yes | 6 → 30 | 211 | 0 → 211 |
| wnba | 17 | yes | 0 → 15 | 211 | 0 → 210 |
| soccer | 275 | yes | 203 → 274 | 1301 | 299 → 777 |
| ncaab | 72 | yes | 47 → 64 | 69 | 2 → 2 |
| ncaaw | 11 | yes | 10 → 10 | 10 | 3 → 3 |
| baseball | 95 | no | 1 → 65 | 683 | 0 → 377 |
| nfl | 32 | no | 0 → 32 | 48 | 0 → 0 |
| nhl | 38 | no | 29 → 30 | 57 | 51 → 53 |
| ncaaf | 173 | no | 159 → 173 | 219 | 0 → 0 |
| worldcup | 48 | no | 47 → 48 | 96 | 2 → 2 |

Rest only acts in sectors with a `REST_ELO_ADJ` table. NFL's live H2H store holds only the 2026
week 1–2 games, so no NFL pair reaches `H2H_MIN_GAMES` yet. Its labels would miss once rematches
start.

## 2. Protocol

- Replay the production `EloModelAgent.update()` / `_win_probs()` from a cold start, in date
  order. The agent clock is patched to each game date. Team names are the sector canonicals, so
  H2H records and reads share keys (the "fixed" behaviour).
- Variants are predicted from one rating trajectory, because H2H and rest never touch the
  update: `off` (both 0 — today's production where label ≠ key), `h2h`, `rest`, `on`.
- Rest days and 7-day counts come from the replay schedule on the **ET calendar day**. Live
  rest is measured between ET days: form records carry the ET resolve date, and Kalshi
  `event_date` is anchored to the ET game day. The first run used ESPN's UTC stamp, which turns
  a Saturday-night → Sunday-matinee pair into a phantom 0-day rest.
- Only **blend-eligible** games are scored. Both sides must have ≥ `LOW_DATA_THRESHOLD` (5) Elo
  games; below that, Elo confidence is ≤ 0.45 and the ensemble drops it. Without this filter,
  NCAAB's one-off non-D1 opponents (no rest history, bonus 0) let rest pose as a "has a rating"
  signal worth z ≈ −15.
- Offseason handling mirrors live: WNBA keep 0.65 and NFL keep 0.667, each with a
  `season_games` reset. The H2H store is never reset, and the 60-day staleness guard applies.
- **Blend**: the production `EnsembleModelAgent._blend` (weight overrides, confidence gate,
  disagreement ramp, FLB) with Elo + Form and the real Pinnacle close, at the live sharp
  weight. Only Elo and Form fire, so Elo's model-side share is 2–5× its live share. The blend Δ
  is therefore an upper bound. Isotonic calibration is off.
- **CLV lenses**:
  - *Pinnacle*: the slope of the line's open→close move on the feature's Elo shift.
    Soccer uses football-data PSH→PSCH. NBA/WNBA/baseball/NCAAB use the `archive.db` first →
    last pre-tip snapshot (2026 only).
  - *Kalshi*: the logged moneyline rows from `_fetch_clv_rows`, the same fetch behind
    `evmax cleanup shadow clv`, on a snapshot of `predictions.db`. Each row's `kalshi_clv_pct`
    is regressed on the feature's shift of the backed team.
- **Gate** (fixed before running): the pooled evaluation-season ΔBrier must be < 0 with
  z ≤ −1.64, AND the holdout ΔBrier must be ≤ 0, AND the holdout blend Δ must be ≤ 0 where an
  anchor exists. No constant is tuned: `H2H_MAX_ADJ`, `H2H_MIN_GAMES` and `REST_ELO_ADJ` are the
  shipped values.

Seasons: NBA eval 2022–24 / holdout 2025-26; WNBA 2022–25 / 2026; baseball 2023–25 / 2026;
NCAAB 2023-24–2024-25 / 2025-26 (ESPN `groups=50`); NHL 2022–24 / 2025-26; soccer (top-5 + UCL +
UEL) 2023-24–2024-25 / 2025-26; NFL 2017–24 / 2025. The first season is burn-in.

## 3. Results

Standalone Elo ΔBrier per 1000 games (variant − off; negative is better). Holdout blend ΔBrier
and CLV slopes are shown where an anchor exists.

### Rest

| sector | fires | eval Δ (z) | holdout Δ (z) | holdout blend Δ (z) | Pinnacle CLV slope (t) | verdict |
|---|---|---|---|---|---|---|
| **nba** | 41% | **−0.64 (−3.75)** | −0.09 (−0.32) | −0.077 (−1.42), n=277 | **+0.45 (+2.19)** | **SHIP** |
| **ncaab** | 16% | **−0.22 (−4.92)** | **−0.18 (−2.90)** | −0.013 (−1.72), n=38 | +0.67 (+1.23) | **SHIP** |
| wnba | 38–50% | −0.07 (−0.47) | −0.55 (−1.59) | −0.010 (−1.78), n=315 | +0.36 (+1.11) | reject (a) |
| soccer | 35% | −0.21 (−1.02) | +0.25 (+0.81) | −0.003 (−0.69), n=856 | −0.11 (−0.95) | reject |

- NBA rest is better in all 4 seasons: −0.25, −0.88, −0.79, −0.09.
- NCAAB rest is better in all 3 seasons: −0.20, −0.24, −0.18.
- WNBA rest is better in 4 of 5 seasons, but 2025 is worse (+0.40).
- Kalshi lens, NBA rest: slope +0.86 (t +2.46). Only 10 of 82 rows shift, because the logged
  NBA rows are mostly playoffs, where both teams are rested. CLV is +0.13pp where rest backs the
  side and −2.96pp where it fades it.
- Kalshi lens, WNBA rest: slope −0.02 (t −0.06), n=100.

### H2H

| sector | fires | eval Δ (z) | holdout Δ (z) | holdout blend Δ (z) |
|---|---|---|---|---|
| nba | 86–93% | +1.46 (+4.84) | +1.22 (+2.68) | −0.035 (−0.35) |
| baseball | 87–95% | +1.00 (+5.81) | +0.69 (+2.68) | −0.000 (−0.11) |
| ncaab | 44–55% | +0.55 (+5.03) | +1.29 (+6.14) | +0.010 (+1.02) |
| nhl | 85–91% | +0.78 (+2.68) | +1.25 (+2.52) | — |
| wnba | 78–90% | +1.28 (+2.11) | +0.75 (+0.77) | −0.001 (−0.08) |
| nfl | 60–81% | +0.09 (+0.27) | +3.02 (+3.02) | +0.015 (+1.88) |
| soccer | 45–60% | −0.34 (−1.01) | −0.32 (−0.46) | +0.008 (+0.91) |

- Kalshi lens slopes: NBA −0.07 (t −0.4), WNBA −0.08 (t −0.8), baseball −0.59 (t −1.3), soccer
  +0.20 (t +1.3).
- `on` (both layers) is worse than `rest` alone wherever H2H fires. The single exception is
  soccer: eval −0.58 (z −1.54), holdout −0.12, holdout blend +0.004 — it fails gates (a) and (c).

Why H2H hurts: Elo already prices team strength. A lifetime win-loss record between two teams
adds a ±5pp nudge that is mostly stale (rosters change) and small-sample noise on top of the
same information.

## 4. What shipped

`REST_ELO_ADJ` is unchanged. `EloModelAgent._form_games` resolves the label through
`resolve_team_key` against the sector's `form_state` store (the same rule `FormModelAgent`
reads with), but only for `REST_RESOLVED_SECTORS = {"nba", "ncaab"}`. Every other sector keeps
the raw-label read, byte for byte. `_h2h_adjustment` is untouched. Its docstring records why it
must not be resolved.

Effect on the bettable probability is small. The NBA blend uses Elo at 0.10 weight with a sharp
weight of 0.70 (no FLB), so Elo's share of the final probability is about 3%. A typical rest
shift of 1.5pp in Elo moves the NBA blend by about 0.05pp. NCAAB is similar. The value is a
correct, CLV-aligned nudge, not a new edge source.

## 5. Caveats

- **Live rest needs a daily feed.** Rest is only right when yesterday's games are in
  `form_state.json` before today's scan. On 2026-09-23 the main checkout held games through
  09-22, a one-day lag. A missed resolve degrades rest toward "off" (a back-to-back reads as
  rested); it does not invert it. The integrity sweep already alerts on resolve cadence.
- **NCAAB live history was incomplete in 2025-26.** Live `form_state` holds about 20% of D1
  team-games per month, because it was seeded from ESPN's featured-games feed. The resolve hook
  now fetches `groups=50` (every D1 game) daily, so 2026-27 is the first season whose live rest
  matches the replay. Recheck NCAAB rest with `cleanup shadow clv ncaab` once it has rows.
- The blend Δ overstates Elo's live share, and the Pinnacle / Kalshi CLV samples are 2026-only.
  NBA has 277 anchored games and 10 shifted Kalshi rows.
- NCAAW is untested. Its archived labels already equal their keys, so it was left off the list.

## 6. Follow-ups (not in this change)

- **H2H still fires today where labels equal keys**: NHL (51 of 57 pairs), NCAAB, most of soccer,
  NCAAW, worldcup, and NCAAF once pairs repeat. The same replay shows it hurting in NHL and NCAAB.
  Removing it there is a separate model change with its own gate.
- Near misses to re-run as data accrues: WNBA rest (every lens favourable, eval z only −0.47)
  and soccer `on` (Kalshi lens t +1.64).
