# Pre-registration: does venue-vs-Pinnacle gap grow with slate size? (2026-10-27)

Written BEFORE any result was computed. Script: `scripts/exp_slate_size_gap.py`.

Question: is the Kalshi price further from the Pinnacle devigged price on heavy
slates (many games the same ET day) than on light ones — the "too many games
to pay attention to" claim, tested on OUR venues (not Pinnacle)?

Data: archive.db, 2026-03-20 → 2026-10-27. Moneyline only. A Kalshi YES market
is paired with the Pinnacle ML record of the SAME game from the SAME
`session_id` (same fetch), matched on canonical team + ET game date.

Unit / snapshot: per Kalshi market, the one session whose time-to-tip is closest
to 3h inside [1.5h, 6h]. Price band: Kalshi YES ask in [0.15, 0.85] (keeps the
favorite-longshot bias out of the comparison).

Metrics per slate-size bucket (light <=4 games / mid 5-9 / heavy >=10, games =
distinct Pinnacle ML events with that ET start date in that sector):
  1. mean |sharp_prob - kalshi_ask| in pp
  2. mean (sharp_prob - kalshi_ask) in pp (positive = ask cheap vs sharp)
  3. share of markets with sharp_prob - ask >= 2pp (the scanner's flag level)

Decision rule (fixed now):
  SUPPORTED   = within a sector, heavy minus light on metric 1 >= +0.5pp AND the
                game-day-clustered 95% bootstrap CI (2000 reps, seed 20261027)
                excludes 0, AND the sign agrees on metric 3.
  REFUTED     = CI excludes 0 in the OTHER direction, or CI includes 0 with both
                bucket n_games >= 30 (adequately powered null).
  UNDERPOWERED= any bucket < 30 game-days' worth (< 30 games) — no verdict.
Confound check (secondary, not part of the rule): NBA playoffs (Apr 18+) are
light AND high-attention; so the NBA test is run on regular-season dates only,
and playoffs-vs-regular is reported separately.
Never edit this rule after seeing numbers; add a dated addendum instead.

## Addendum 2026-10-27 — result (rule unchanged)
Verdict: NOT SUPPORTED, and UNDERPOWERED by the rule (the light bucket had <30 games
in every sector). Paired markets: nba 185, baseball 2343, nhl 254, wnba 358, nfl 178.
mean|gap| at ~T-3h: nba 0.77 (mid) / 0.81 (heavy); baseball 0.65 / 0.71 / 0.74;
nhl 0.75 / 0.68 / 0.59; nfl 0.71 / 0.76; wnba 1.01 (light) / 0.77 (mid).
mean (sharp - ask) is -0.4 to -0.7pp everywhere (the ask sits above sharp: spread);
share of markets with edge >= 2pp: 0-1% in every bucket. NBA heavy - light = -0.20pp
(CI -0.29..-0.11) is on a 2-game light bucket -> no verdict. Playoffs (light/high
attention) 0.76pp vs regular season 0.77-0.81pp: no difference.
Addendum 2 (2026-10-24, written before running): second test, SAME rule, SAME buckets,
SAME price band, SAME seed, only the snapshot window changes: closest to 18h before
tip inside [12h, 24h] (the watch-listings capture window). Run via
`python scripts/exp_slate_size_gap.py --lo 12 --hi 24 --target 18`.
Addendum 2 result (2026-10-24): NOT SUPPORTED (UNDERPOWERED by the rule: light bucket <30
games in every sector). Paired: nba 424, baseball 2742, nhl 308, wnba 527, nfl 197.
mean|gap| at ~T-18h: nba 1.08 light / 1.07 mid / 1.03 heavy; baseball 0.83 / 0.76 / 0.71;
nhl 0.75 / 0.89 / 0.79; nfl 0.91 / 0.88; wnba 1.25 (light) / 1.35 (mid). heavy - light:
nba -0.05 (CI -0.19..0.11), baseball -0.12 (CI -0.26..-0.00), nhl +0.04, nfl -0.04: none
positive and significant. mean (sharp - ask) -0.4..-0.75pp; share with edge >= 2pp 0-1%
(WNBA 3-4%, light slates only). NBA playoffs 1.02pp vs regular season 1.03-1.08pp.
Limit: the first test was the near-tip window only. A "too many games" effect, if real, would show
at listing / T-12h+, which this test did not sample.
