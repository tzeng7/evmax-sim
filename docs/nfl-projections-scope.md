# NFL projection system — scope (2026-10-09)

**Goal.** One NFL projection engine that predicts, for every game, (a) the
final-score distribution (margin, total, win probability, exact-score PMF) and
(b) player stat-line distributions (pass/rush/receiving yards, receptions, TDs),
with the two internally consistent. It is a platform feature judged on
accuracy, not on betting EV. EV uses are a later, optional layer.

**Recommendation.** Build it in four gated phases (§7). Phase 1 is the game
model; it reuses most of the existing NFL stack and has a clear accuracy bar.
The player model follows and shares one per-game Monte Carlo with the game
model, so receivers' yards sum to the QB's passing yards and touchdowns agree
with points in every simulation.

## 1. What "good" looks like

Benchmarks computed from nflverse `games.parquet` (regular season) and our own
Kalshi archive, plus published model results.

| Target | Naive baseline | Our quick prototype | Best non-market public models | Vegas close | Phase gate |
|---|---|---|---|---|---|
| Margin MAE (2020–25) | 10.98 (home +1.5) | 10.30 (points-only ridge ratings, walk-forward) | ESPN FPI 10.04–10.07, Sagarin 10.07–10.18, Massey 10.22–10.25 (2024–25) | 9.76 | ≤ 10.15 |
| Total MAE (2020–25) | 10.98 (league mean) | 10.68 | best systems ≈ 10.4–10.5 (2025, labels ambiguous) | 10.30 | ≤ 10.55 |
| Straight-up winners | — | — | FPI/Elo ≈ 64–66% | 66.4% | within 2pp of close |
| Receiving-yards MAE (2026 Wks 1–5) | — | 22.4 (EWMA recency) | — | Kalshi ladder mean 21.9 | within 3% of Kalshi |
| Rushing-yards MAE | — | 18.8 | — | 17.4 | within 3% of Kalshi |
| Receptions MAE | — | 1.68 | — | 1.64 | within 3% of Kalshi |

- Non-market models trail the closing line by 0.3–0.5 points of margin MAE.
  Only market-blended models (nfelo blends ~65% market) tie the close. A pure
  model at ~10.0 is a strong product.
- Single games are mostly noise. The market's own receiving-yards projection
  explains only ~31% of variance (RMSE 29.2 vs SD 35.0). Report quantiles, not
  just point projections.
- Our prototype adds no information beyond the close (incremental slope ≈ 0).
  The accuracy target is independent of beating the market.

## 2. What evmax already has

| Piece | Location | Reuse |
|---|---|---|
| Opponent-adjusted team EPA/SR ratings (one-pass) | `evmax/agents/models/nfl_efficiency_agent.py`, `scripts/seed_nfl_efficiency.py` | Inputs. Margin scale is ~1.7× too wide (`EPA_MARGIN_PTS` 64 vs OLS 37–43); P(win) only. |
| Ridge EPA solver, EP table, preseason-prior ramp | `evmax/agents/models/_cfb_efficiency.py` | Template for the NFL rating solve (better than the NFL one-pass mean subtraction). |
| QB Elo + depth-chart starters | `nfl_qb_elo_agent.py`, `evmax/clients/nfl_depth_charts.py` | QB layer; starter resolution (QB only). |
| Key-number margin PMF | `evmax/models_ml/spread_pmf.py`, `data/models/nfl_margin_pmf.json` | Converts any projected margin into exact P(margin = k). |
| Total distribution | `evmax/models_ml/total_distribution.py` | Normal σ=14 around Pinnacle — replace with the model. |
| Prop distributions | `evmax/ev/prop_pricing.py` (`distribution_from_mean`) | Gamma yards (θ 25.3/15.3), NegBin receptions (k=5) — calibration starting point. |
| Prop dispersion fit + Kalshi-settlement grading | `scripts/fit_nfl_prop_dispersion.py` | Player-model calibration harness. |
| nflverse weekly stats / rosters / schedules cache | `evmax/clients/nfl_props_cache.py` | Data layer seed; the old usage model was deleted in `edb3d7b` (recoverable). |
| Schedules: lines, rest, roof, temp, wind, referee | nflverse `games.parquet` | Present but unused by any model. |
| Injury feed with NFL position weights | `evmax/agents/intelligence/injury_agent.py` | Inactive/questionable status input. |
| Walk-forward harnesses | `scripts/backtest_nfl_efficiency.py` (leak fixes), `backtest_nfl_sr_margin.py`, `evmax/backtest/metrics.py` | Patterns; none measures margin/total MAE vs the close yet. |
| Projection storage + CLI | `evmax/cli/commands/project.py`, `data/projections.db` | Game table shape fits; needs a player table and fixes (below). |

**NBA precedent** (`evmax project`, Mar 29 – May 1 2026, dormant): game-level
team scores only. Lessons:
- **Backtest before launch.** It shipped with no walk-forward evaluation, and its total model lost to the book (MAE 15.1 vs 13.9, +3.5 bias).
- **Sign conventions.** Stored book spreads were the favorite's line, not the home handicap, so grading is wrong whenever the away team is favored.
- **Team lookup.** One date's rows all fell back to default ratings (lookup failure).
- **Alternate rungs.** `"::spread" in eid` would now pick up alternate rungs.

Fix all four in the NFL build: shared team lookup, an explicit home-handicap convention with tests, and a main-line-only filter.

## 3. Elements of a good game model

**Team strength.**
- Opponent-adjusted offense and defense EPA/play plus success rate, from a ridge solve with home-field advantage (HFA). Garbage time is removed.
- Offense is weighted more than defense, because it is more predictive (nfelo weights offense 1.6×).
- Recency decay within the season. About ⅓ regression toward the mean between seasons (538, Glickman & Stern state-space).

**Preseason prior.**
- FPI, 538 and nfelo all seed preseason ratings from market win totals and fade them as games accrue. 538 gave the win totals 2× the weight of last season's ratings.
- Source: Kalshi lists season win markets (`KXNFLWINS-*`). We don't archive them, but Kalshi's public candle history gives point-in-time preseason prices (backfill as with the ladder study).
- Template: the NCAAF FPI-prior ramp.

**QB layer.**
- Each QB gets his own rating; the team is rated as base + QB.
- Price the actual starter, from the depth chart plus the injury report.
- Starter-to-backup gap: about 6–10 points for elite QBs, 0–2 for average ones.
- 538's per-game QB VALUE formula is a documented starting point. Our `nfl_qb_elo` already works this way.

**Situational terms.**

| Factor | Size | Status |
|---|---|---|
| Home-field advantage | ≈1.5–1.7 pts (prototype fit 1.68; recent ≈1.5) | Halved in divisional games |
| West-to-east travel, early kickoff | ≈2 pts (nfelo) | Use |
| Unfamiliar surface | ≈1 pt | Use |
| Bye | ≈0 since 2011 | Drop |
| Rest | Fully priced by the close (rest-advantage residuals −0.24 to −0.03 ± 0.55) | Feature only, no edge |
| Divisional | Fully priced (−0.21 ± 0.32) | Feature only, no edge |

**Totals.** Model each team's points as drives × points per drive, not as a scaled margin.
- **Drives** come from pace (neutral seconds per play ranges 29–34.5) and the run/pass mix (pass rate over expected).
- **Points per drive** comes from offense vs. defense efficiency, red-zone TD rate and turnover rate.
- **Weather is the one factor the market under-prices.** Since 2010, closed-roof games beat the closing total by **+1.34 ± 0.39 points**; 0–5 mph wind +1.26 ± 0.56; 10–20 mph −0.8 to −1.3. Caveat: nflverse wind is recorded at the game, so live use needs a forecast.
- **Skip referees.** Crew effects are indistinguishable from noise.

**Score distribution.**
- NFL scores are discrete and lumpy: a margin of 3 occurs in 14.7% of games, 7 in 8.7%. Totals have no strong key numbers.
- A Poisson points model misses game dynamics. A state-dependent scoring process fits real scores better (Moyer et al., WSC 2024; Baker & McHale, IJF 2013).
- v1: project team points from the rating model, then map the margin through the existing key-number PMF. v2: a drive simulator (§5).
- Spread and total errors are uncorrelated (corr 0.02), so they can be modeled separately at first.

## 4. Elements of a good player model

**Volume.** Team plays × pass/run split, conditioned on the game model's script (spread, total).
- Errors here dominate: one practitioner found team volume "way off for roughly half the league," and share errors compound it.
- Run/pass ratio correlates only 0.15 game to game, so condition on the spread rather than chasing script.

**Shares.** These are stable and should drive the projection.

| Metric | Stability (year-over-year correlation) |
|---|---|
| Target share | 0.55–0.71 |
| Targets per game | 0.66 |
| Air-yards share | 0.53 |
| RB carries per game | 0.65 |

- Use shrunk rolling shares, blended with last season and filtered by depth-chart role and snap share.
- Renormalize when a player is inactive. Vacated-target studies are weak, so validate the reallocation rather than assuming it.
- Free in-season route data does not exist (FTN participation is released after the postseason). Snaps on dropbacks are the proxy.

**Efficiency.** This is mostly noise, so shrink it hard.

| Metric | Stability (year-over-year correlation) |
|---|---|
| Yards per target | 0.22 |
| Yards per carry | 0.16 |
| TD per target | 0.12 |

- Stabilization points: about 350 routes for yards per route run, 188 for receptions per route.
- Prior: `ffopportunity` expected yards (XGBoost on nflverse play-by-play, CC-BY-SA).
- Never use raw yards per target or raw TD rate.

**Touchdowns.** Team TD expectation (from points) × red-zone and goal-line share. Yards-to-TD conversion barely persists (r ≈ 0.04).

**Distribution.**
- Negative binomial targets → binomial catches → gamma per-catch yards, with a zero mass.
- Calibrate the coefficient of variation by projection bucket on held-out weeks. The Gamma family shipped in #373 is the starting point.
- Report the median as well as the mean: yardage is right-skewed, and books price at the median.

**Data (free, nflverse).**
- Play-by-play from 1999 (cp/cpoe/xyac from 2006), parsed about 15 minutes after each game.
- Weekly player stats, PFR snaps (2012+), FTN charting (2022+, about 48 hours after each game).
- Depth charts (daily from 2025), injuries (2009+), Next Gen Stats (2016+), and schedules with closing lines and weather.
- Paid only: PFF routes and yards per route run.

## 5. Unified simulation

**Recommended v1: top-down hybrid Monte Carlo.** About 10k simulations per game, vectorized numpy, under a second per game. Per simulation:
1. Draw team plays and pass rate from the game model's script distribution.
2. Draw each team's scoring efficiency, with one shared game-level latent factor.
3. Allocate targets and carries with a Dirichlet-multinomial over shrunk shares.
4. Draw per-touch yards.
5. Derive TDs, then points (TD×6 + XP + FG×3).

The sum identities then hold by construction. Correlation emerges from the shared volume and script. Check it against empirical stacks (QB–WR1 strongest; QB vs. the opposing defense negative).

**v2 (research): drive simulator.**
- State: down/distance/field position/clock/score.
- Drive outcomes from an expected-points (EP) style model, play choice from an `xpass`-style model.
- This captures game script for totals and correlations. Public precedents are partial: Goldner 2012 Markov drives, NFLSimulatoR (resamples nflfastR plays, no clock or players), nflWAR EP/WP components.
- No maintained Python simulator with player attribution exists.

**Not recommended:** a full play-by-play player simulator first. The commercial ones (Stokastic, SportsLine, Huddle) publish no validation. Huddle's whitepaper confirms only the shape: player projections feed a per-sport Monte Carlo, which also prices same-game parlays.

## 6. Evaluation

- **Walk-forward:** 2019–2025 with weekly refits on point-in-time data; 2025 is the holdout.
- **Leak lessons to reuse:**
  - ESPN dates are UTC while PBP dates are ET (`_state_cutoff`).
  - Depth charts are daily only from 2025 and weekly before.
- **Game metrics:**
  - Margin and total MAE/RMSE vs the nflverse close.
  - Brier and log loss for winners.
  - Continuous ranked probability score (CRPS) of the score distribution.
  - Key-number calibration.
  - Incremental slope vs the close (for any later EV use).
- **Player metrics:**
  - MAE/RMSE/CRPS vs actual.
  - Quantile coverage (PIT, the probability integral transform).
  - Comparison to the Kalshi ladder mean / Pinnacle median (2026 only).
  - Exact consistency assertions in every simulation.
  - Grade on Kalshi settlement where `actual_value` is NULL (zero-stat games).
- **Joint distribution:** energy and variogram scores. Check against empirical teammate correlations.

## 7. Phased plan

| Phase | Scope | Gate | Rough effort (estimate) |
|---|---|---|---|
| 1. Game model | Data layer with local nflverse cache; ridge EPA/SR team ratings; QB layer; HFA/travel/surface/weather; team points via drives × points per drive; margin/total/win prob/score PMF; walk-forward harness | 2020–25 walk-forward margin MAE ≤ 10.15 and total MAE ≤ 10.55; win Brier ≤ the current NFL blend | ~1 week |
| 2. Player model | Volume/share/efficiency, TDs, injury reallocation, Monte Carlo consistency; calibration | Player MAE within 3% of the Kalshi mean (2026); quantile coverage within ±3pp; identities exact | 1–2 weeks |
| 3. Product | `projections.db` player table; `evmax project nfl --week`; dashboard tab; Discord weekly post; scheduled weekly run plus Sunday-inactives refresh; resolve/track | Clean live weeks with tracked accuracy | 3–5 days |
| 4. Optional | Drive simulator; market-blend display mode (≈ the close); EV hooks: feed player means into `prop_pricing` as a model input (the `baseball_props` pattern), shadow-only, judged by the incremental-information test | Per-feature | open |

## Phase 1 results (2026-10-09) — game model shipped

Package `evmax/nfl_projections/` (data, team_games, ratings, game_model, live),
harness `scripts/backtest_nfl_game_projections.py`, CLI `evmax project nfl`,
tests `tests/test_nfl_projections.py`. Built with the self-improve loop: one
change per iteration, signal `dev_score = margin MAE + total MAE` (walk-forward
2019-24), 2025 held out, each kept change reviewed for leakage/gaming.

| Iteration | Dev score | Holdout 2025 margin / total | Verdict |
|---|---|---|---|
| B0 points-only ridge ratings | 21.134 | 10.327 / 10.785 | baseline |
| H1 EPA + SR ratings, walk-forward OLS combiner | 21.041 | 10.280 / 10.638 | kept (~2/3 of the gain is recalibrating the points rating) |
| H3 dome + wind | 20.983 | 10.280 / 10.536 | kept (the gain is wind; dome alone hurts) |
| H4 starting-QB layer | 20.899 | 10.122 / 10.582 | kept (gain in starter-changed games, t −2.05) |
| H6 separate home-field feature | 20.891 | 10.169 / 10.575 | reverted (below noise) |
| H7 offseason not counted as rating decay | **20.741** | **10.002 / 10.464** | kept (Weeks 1-4 margin 10.44 → 10.03, t −3.1) |

**Gate (2020-25): margin MAE 10.132 (≤ 10.15 ✓; Vegas 9.764), total MAE
10.488 (≤ 10.55 ✓; Vegas 10.283)**, Brier 0.222, straight-up 64.0%. On par with
ESPN FPI (10.04-10.07) without any market input.

Known residuals / next levers:
- Total bias +0.4-0.5 pts (2025 holdout).
- Week 14-18 margin gap to Vegas (0.52), probably rested starters and motivation.
- Live wind uses the league median until a forecast feed exists; the backtest used recorded wind (sd-3-mph noise keeps 94% of the gain).
- Live starters come from the nflverse schedule's projected QB.
- Untested levers: pace/drives totals, travel, surface.

## Phase 2 results (2026-10-09) — player model v1

Modules:
- `evmax/nfl_projections/player_games.py`: official nflverse weekly stats joined with PFR snap counts, so players who played and recorded nothing are explicit zeros.
- `player_model.py`: team volume × usage share × shrunk efficiency, with team volume conditioned on the game model's script. It reports the median plus 10th/90th percentiles.
- `live.py::project_week_players`: active roster = played in the team's last 3 games, minus Out/Doubtful on the injury report, plus the schedule's starting QB.

CLI: `evmax project nfl --players [--team KC]`. Harness: `scripts/backtest_nfl_player_projections.py`. Tests: `tests/test_nfl_player_projections.py`.

Signal: `dev_score` = mean over receiving yards, receptions, rushing yards and passing yards of MAE_model / MAE_naive (last-8-games mean). Walk-forward 2019-24, 2025 held out, every kept change reviewed.

| Iteration | Dev score | Verdict |
|---|---|---|
| Baseline: volume × share × efficiency (means) | 0.96937 | — |
| P2 starter's share of team attempts × starts-only yards/attempt | 0.96488 | kept (passing better in all 7 seasons, t −5.8) |
| P1 renormalize shares over the realized "played" roster | 0.95801 | **rejected** — half the gain was a constant shrink; the rest leaked realized participation (garbage-time backups); absent-starter games got worse |
| P3 report medians (Gamma yardage / NegBin receptions, dispersion fit per cutoff) | 0.94872 | kept (beats a dev-fitted constant shrink in all 7 seasons) |
| P4 usage-dependent empirical median ratio | 0.94754 | reverted (below noise) |
| P5 usage-share prior 3 → 0.5 pseudo-games | 0.92988 | kept (fixed tier bias: stars −8.4 → +1.2 yds; monotone in the prior) |
| P6 team volume conditioned on the game model's margin/total | **0.92669** | kept (teams projected to score more throw more; better in 6/6 seasons) |

**Final, holdout 2025 (MAE ratio vs naive):**

| Stat | Ratio | MAE |
|---|---|---|
| Receiving yards | 0.917 | 19.0 |
| Receptions | 0.930 | 1.44 |
| Rushing yards | 0.930 | 19.3 |
| Passing yards | 0.888 | 58.0 |

**Ranges.** Empirical usage-dependent quantile curves are used for yardage, because the Gamma lower tail was too thin (23% below p10). Coverage, dev/holdout:

| Stat | Below p10 | At or below p90 |
|---|---|---|
| Receiving yards | 5.7 / 8.2% | 90.3 / 90.7% |
| Rushing yards | 8.1 / 8.7% | 90.3 / 89.5% |
| Passing yards | 11.7 / 10.5% | 89.8 / 92.2% |

The low receiving-yards share below p10 comes from exact-zero ties.

**Kalshi gate (2026 Weeks 1-5, the same 536 / 242 market-covered player-games, best of median/mean on each side): NOT met.**

| Stat | Kalshi MAE | Model MAE | Gap (gate ≤ 3%) |
|---|---|---|---|
| Receiving yards | 21.66 | 22.46 | +3.7% |
| Receptions | 1.586 | 1.646 | +3.8% |
| Rushing yards | 16.90 | 17.98 | +6.4% |

Early-season, high-usage players are where the market's information (depth charts, injuries, role changes) shows.

**Next levers**, ranked:
1. Retry the P1 idea on the real pre-game injury report rather than the realized played roster. nflverse injuries go back to 2009, so it can be backtested.
2. Depth-chart role changes (daily snapshots from 2025).
3. Rushing: split QB scrambles from designed runs, and the goal-line role.
4. TD projections.
5. Opponent-adjusted efficiency.
6. The joint Monte Carlo, so receivers' yards sum to the QB's passing yards. Not built yet: v1 projects each player independently.

## 8. Risks and limits

- **The market stays more accurate.** Display the model's tracked accuracy honestly.
- **Data limits.** No free in-season routes; early-season nflverse 404s (the cache already tolerates them); weekly reseed and staleness guards are required at season boundaries.
- **Player single-game noise caps any model.** Receiving-yards MAE about 20+ even for the market.
- **Effort estimates are rough** and assume reuse of the components in §2.

## Sources

- Benchmarks: [ThePredictionTracker NFL results](https://www.thepredictiontracker.com/nflresults.php?year=25); [nfelo model performance](https://nfeloapp.com/games/nfl-model-performance/); [nfelo market regression](https://nfeloapp.com/analysis/using-market-regression-to-improve-prediction-accuracy-in-the-nfl).
- Methodology: [ESPN FPI guide](https://africa.espn.com/blog/statsinfo/post/_/id/123048/a-guide-to-nfl-fpi); [nfelo](https://github.com/greerreNFL/nfelo); [nfelo HFA tracker](https://www.nfeloapp.com/tools/nfl-home-field-advantage-hfa-tracker/); [Inpredictable](https://inpredictable.com/p/methodology.html); [Lopez & Bliss 2024 (bye)](https://arxiv.org/abs/2408.10867); [Benz, Bliss & Lopez 2024 (HFA)](https://arxiv.org/abs/2401.16392); [Smith et al. 2013 (travel)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3825451); [Baker & McHale 2013](https://salford-repository.worktribe.com/output/1432547/forecasting-exact-scores-in-national-football-league-games); [Moyer et al. WSC 2024](https://informs-sim.org/wsc24papers/con253.pdf).
- Player modeling: [4for4 WR stability](https://www.4for4.com/2023/preseason/most-predictable-wide-receiver-stats); [Football Perspective per-route rates](https://www.footballperspective.com/yards-per-route-run-yards-per-target-and-targets-per-route-run/); [ffopportunity](https://github.com/ffverse/ffopportunity); [Ben Gretch, projection lessons](https://bengretch.substack.com/p/biggest-lessons-learned-about-all); [The Analyst projections](https://theanalyst.com/articles/explaining-the-analysts-weekly-nfl-fantasy-football-projections); [Sharp Football, game flow](https://www.sharpfootballanalysis.com/analysis/we-might-not-be-as-good-at-predicting-game-flow-as-we-think/amp/).
- Simulation: [NFLSimulatoR](https://ar5iv.labs.arxiv.org/html/2102.01846); [Goldner 2012](https://ideas.repec.org/a/bpj/jqsprt/v8y2012i1n9.html); [nflWAR](https://arxiv.org/abs/1802.00998); [fastrmodels](https://cran.case.edu/web/packages/fastrmodels/refman/fastrmodels.html); [Huddle player props overview](https://huddle.tech/wp-content/uploads/2024/02/Technical-Overview-of-Huddles-Player-Props-1.pdf); [scoringRules multivariate scores](https://rdrr.io/cran/scoringRules/man/scores_sample_multiv.html).
- Data: [nflreadr data schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html); [participation](https://nflreadr.nflverse.com/reference/load_participation.html); [injuries](https://nflreadr.nflverse.com/reference/load_injuries.html).
