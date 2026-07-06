# evmax — Agent-Based +EV Prediction Market System

evmax uses a multi-agent pipeline to find positive expected value (+EV) opportunities on Kalshi by comparing market prices against sharp Pinnacle lines, statistical models, and real-time injury data. The system recommends Kelly-fractioned bet sizes for each opportunity it surfaces.

Sharp odds come from the **Pinnacle guest API** (`guest.api.arcadia.pinnacle.com`), which is keyless — the only API credential you need is a Kalshi key for live price refresh and trading. Every bettable category, its models, mode, and resolver are declared in one registry: [`data/categories.yaml`](data/categories.yaml).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture: The Agent Pipeline](#architecture-the-agent-pipeline)
3. [Daily Workflow: Finding +EV Plays](#daily-workflow-finding-ev-plays)
4. [Real-Time Price Feed (WebSocket)](#real-time-price-feed-websocket)
5. [Statistical Models](#statistical-models)
6. [Seeding and Updating Models](#seeding-and-updating-models)
7. [Cleanup Agent: Logging, Resolution, and Calibration](#cleanup-agent-logging-resolution-and-calibration)
8. [Closing Line Value (CLV)](#closing-line-value-clv)
9. [Player Prop Pipeline](#player-prop-pipeline)
10. [Web Dashboard](#web-dashboard)
11. [Data Archive and Backtest](#data-archive-and-backtest)
12. [Core EV and Kelly Math](#core-ev-and-kelly-math)
13. [CLI Reference](#cli-reference)
14. [Configuration](#configuration)
15. [Sectors and Market Types](#sectors-and-market-types)

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/tzeng7/evmax
cd evmax
./setup.sh       # installs deps + registers git hooks (run once after cloning)
```

> `setup.sh` installs `pre-commit` hooks that remind you to keep CLAUDE.md, `__init__.py` files,
> and folder READMEs in sync whenever you edit tracked source files. The hook is advisory — it
> never blocks a commit.

### 2. Set API Keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
# Kalshi (RSA key auth) — needed for WebSocket price refresh + trading.
# Market reads and the Pinnacle guest API are unauthenticated, so scanning
# works read-only even without these.
KALSHI_API_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi.pem

# Optional
EV_THRESHOLD=0.02          # minimum EV to report (default 2%)
MAX_KELLY_FRACTION=0.05    # hard cap per bet (default 5%)

# Optional — disable WebSocket and force REST-only price fetching
# KALSHI_WS_ENABLED=false
```

> Sharp lines come from the keyless Pinnacle guest API — there is no sharp-odds API key to set.

### 3. Initialize the Database

```bash
python scripts/init_db.py
```

### 4. Run Your First Scan

```bash
# Full agent pipeline — all sectors, half Kelly, $250 bankroll
evmax agents scan --bankroll 250 --kelly 0.5

# Target specific sectors
evmax agents scan --sectors nba,soccer --bankroll 250 --kelly 0.5

# Lower noise threshold
evmax agents scan --sectors nba --min-ev 0.03 --top 20
```

---

## Architecture: The Agent Pipeline

Every scan cycle runs through a coordinated set of agents:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Concurrent fetch (all three run in parallel per sector)        │
│                                                                          │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────────┐  │
│  │  KalshiOddsAgent  │  │  SharpOddsAgent   │  │  InjuryReportAgent  │  │
│  │  Fetches live     │  │  Fetches devigged  │  │  Fetches ESPN       │  │
│  │  Kalshi markets   │  │  Pinnacle lines    │  │  injury reports     │  │
│  │  (RSA auth)       │  │  (guest API,       │  │  (public, no auth)  │  │
│  │                   │  │   no auth)         │  │                     │  │
│  └────────┬──────────┘  └────────┬───────────┘  └──────────┬──────────┘  │
│           │ PredictionMarket[]   │ SharpOdds[]             │ InjuryReport│
└───────────┼──────────────────────┼─────────────────────────┼─────────────┘
            ▼                      ▼                         │
┌───────────────────────────────────────────────────────────┐               │
│  STEP 2 — MatchingEngine                                  │               │
│  Fuzzy-matches Kalshi markets ↔ Pinnacle events           │               │
│  using canonical keys + rapidfuzz                         │               │
└───────────────────────┬───────────────────────────────────┘               │
                        │ matched pairs                                     │
                        ▼                                                   │
┌───────────────────────────────────────────────────────────────────────────┤
│  STEP 3 — EnsembleModelAgent (models run in parallel)                     │
│                                                                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ Efficiency   │ │ Possession   │ │  EloModel    │ │  FormModel   │    │
│  │ Agent (0.30) │ │ Sim (0.30)   │ │  (0.10)      │ │  (0.10)      │    │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘    │
│  ┌──────┴───────┐ ┌──────┴───────┐        │                │            │
│  │ ShotQuality  │ │  Matchup     │        │                │            │
│  │ Agent (0.10) │ │  Agent (0.10)│        │                │            │
│  └──────┬───────┘ └──────┬───────┘        │                │            │
│         └────────────────┴────────────────┴────────────────┘             │
│                    Confidence-weighted blend (NBA weights above)          │
│                    + Sharp odds (weight 0.85)                            │
│                    = BlendedPrediction per event                          │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │ blended_preds + injuries
                            ▼
┌───────────────────────────────────────────────────────────┐
│  STEP 4 — EVGapAgent                                      │
│  • YES-team alignment (home vs away side)                 │
│  • Spread markets → SpreadDistributionModel               │
│  • Apply injury probability adjustment                    │
│  • EV = (true_prob × payout) − 1                         │
│  • Kelly sizing (fractional, capped)                      │
│  • Filter: EV ≥ 2% threshold                              │
└───────────────────────────────────────────────────────────┘
                            │ list[EVGap]
                            ▼
┌───────────────────────────────────────────────────────────┐
│  STEP 5 — CleanupAgent (auto-runs after every scan)       │
│  • Logs every +EV gap to predictions.db (SQLite)          │
│  • End-of-day: resolve actual outcomes via ESPN/bo3.gg    │
│  • Weekly: compute Brier scores, auto-adjust sharp_weight │
└───────────────────────────────────────────────────────────┘
```

### Agent Communication

Agents communicate via an `AgentBus` pub/sub system. Each agent publishes its results to a topic string:

| Agent | Publishes to |
|-------|-------------|
| KalshiOddsAgent | `odds.kalshi.{sector}` |
| SharpOddsAgent | `odds.sharp.{sector}` |
| InjuryReportAgent | `intelligence.injuries.{sector}` |
| EnsembleModelAgent | `model.ensemble.{sector}` |
| EVGapAgent | `ev.gaps.{sector}` |
| AgentCoordinator | `coordinator.cycle.done` |

You can subscribe to any topic to hook in custom consumers (Slack alerts, dashboards, loggers):

```python
coordinator = AgentCoordinator(bankroll=250, kelly_fraction=0.5)
coordinator.subscribe("ev.gaps.nba", my_slack_handler)
result = await coordinator.run_cycle()
```

---

## Daily Workflow: Finding +EV Plays

### Find plays for today

```bash
evmax agents scan --bankroll 250 --kelly 0.5
```

Output:
```
+EV Plays — 3 found | 2026-03-24 | Bankroll $250 | 50% Kelly | base EV 2%
╭───┬────────┬──────────────────────────┬─────────────┬─────────┬───────────┬────────┬────────┬────────┬─────────╮
│ # │ Sector │ Event                    │ Outcome     │ K Odds  │ True Odds │ True P │ EV %   │ Kelly% │ Stake $ │
├───┼────────┼──────────────────────────┼─────────────┼─────────┼───────────┼────────┼────────┼────────┼─────────┤
│ 1 │ NBA    │ Celtics vs 76ers         │ Celtics ML  │  -162   │  -250     │ 0.714  │ +15.2% │ 4.82%  │ $12.05  │
│ 2 │ SOCCER │ Arsenal vs Spurs         │ Arsenal ML  │  +108   │  +86      │ 0.537  │ +11.9% │ 3.21%  │  $8.03  │
│ 3 │ NBA    │ Lakers vs Nuggets        │ Lakers -4.5 │  +127   │  +113     │ 0.471  │  +7.1% │ 1.90%  │  $4.75  │
╰───┴────────┴──────────────────────────┴─────────────┴─────────┴───────────┴────────┴────────┴────────┴─────────╯
  Total at risk: $24.83 / $250 (9.9%)  |  Matched 147/1084 markets
```

**Column guide:**

| Column | Description |
|--------|-------------|
| **K Odds** | Kalshi implied probability expressed as American odds — what the market is offering |
| **True Odds** | Model's estimated fair-value probability expressed as American odds |
| **True P** | Fair-value probability as a decimal (0.714 = 71.4%) |
| **EV %** | Edge: how much you gain per dollar wagered on average |
| **Kelly%** | Fraction of bankroll to wager (fractional Kelly, capped at 5%) |
| **Stake $** | Dollar amount to bet |

### Verify plays before betting (real-time price check)

```bash
evmax agents verify --date 2026-03-23 --bankroll 250 --kelly 0.5
```

Uses a WebSocket connection to Kalshi for real-time orderbook prices — sub-second accuracy vs the ~60s REST API lag. Only shows bets that are still +EV at the live ask. `verify` is read-only.

### Place bets at the live price (entry-timing gate)

```bash
evmax agents pick --date 2026-03-23 --bankroll 250 --kelly 0.5   # --live is the default
```

`pick` records the bets you're placing. By default it **re-fetches the live Kalshi ask** and recomputes EV / the live gate / Kelly stake at the *current* price — so stale scan-time edges that have already reverted toward the sharp close drop out instead of getting placed (the night-before scan is a watchlist, not a bet list). The table shows **Scan / Live / Δ** columns so you can see edge erosion since the scan, and the fill-price prompt defaults to the live ask. Use `--no-live` to fall back to scan prices offline.

Run `evmax cleanup watch-closes --once` on a 5-minute schedule (the `com.evmax.watch-closes` launchd agent does this via `StartInterval` 300) so a near-tip Kalshi snapshot is captured for every game — that's what gives placed-bet CLV a genuine post-entry close to measure against. See `docs/SCHEDULED_RUNS.md`.

### Show only high-confidence plays

```bash
evmax agents scan --bankroll 250 --kelly 0.5 --min-ev 0.05
```

### Skip models (use sharp odds only, fastest)

```bash
evmax agents scan --no-models --bankroll 250 --kelly 0.5
```

### Skip injury adjustments

```bash
evmax agents scan --no-injuries --bankroll 250 --kelly 0.5
```

---

## Real-Time Price Feed (WebSocket)

evmax connects to Kalshi's WebSocket API (`wss://api.elections.kalshi.com/trade-api/ws/v2`) for sub-second orderbook prices. This replaces the REST API polling that has a ~60s cache lag.

### How it works

1. A single WebSocket connection is opened for all tickers at once
2. Authenticates with your existing RSA key pair (same credentials as the REST API)
3. Subscribes to `orderbook_delta` channel — Kalshi pushes an `orderbook_snapshot` per ticker
4. Derives YES ask = `1.0 − best_NO_bid` from the live order book
5. Connection closes after all snapshots are received (or 5s timeout)
6. Any ticker that misses the snapshot window falls back to REST automatically

### Where it applies

| Command | Price source |
|---------|-------------|
| `evmax agents verify` | WebSocket (all tickers in one session) |
| `get_market_ask` fallback | REST orderbook → market snapshot |

### Kill-switch

If you need to force REST-only (e.g., firewall blocks WebSocket, or debugging):

```env
KALSHI_WS_ENABLED=false
```

Set in `.env` — no code changes required.

### Timeout tuning

Default snapshot timeout is 5 seconds. Adjust if your network is slow:

```env
KALSHI_WS_SNAPSHOT_TIMEOUT=8.0
```

---

## Statistical Models

The system runs sector-specific model ensembles blended with Pinnacle sharp lines. NBA uses four dedicated advanced agents plus Elo/Form; WNBA uses its own parallel efficiency + possession-sim stack (separate files, separate state, separate tuning); NFL has efficiency + QB-Elo agents; soccer and the national-team World Cup share a Poisson + xG blend; tennis has six specialized agents; MLB has a pitcher agent; NHL has a MoneyPuck xG agent. All outputs are blended by the EnsembleModelAgent. Models below the 0.45 confidence gate are silently dropped from the blend so missing coverage degrades gracefully.

The per-sector weight overrides live in `SECTOR_WEIGHT_OVERRIDES` in [`ensemble_agent.py`](evmax/agents/models/ensemble_agent.py); the `models:` list per category in [`data/categories.yaml`](data/categories.yaml) decides which agents are even instantiated.

### Per-Sector Ensemble Weights

Most sectors replace some or all of the generic Elo/Form/Poisson core with dedicated models. Sectors not listed (NCAAB, NCAAW, LoL, CS2) fall back to the default class weights (Elo 0.35 · Form 0.25 · Poisson 0.40-where-supported) — or to sharp-only where no model clears the confidence gate.

| Model | NBA | WNBA | NFL | Soccer / World Cup | Baseball | NHL |
|-------|-----|------|-----|--------------------|----------|-----|
| Efficiency | **0.30** | — | — | — | — | — |
| PossessionSim | **0.30** | — | — | — | — | — |
| WNBA Efficiency | — | **0.40** | — | — | — | — |
| WNBA PossessionSim | — | **0.45** | — | — | — | — |
| NFL Efficiency | — | — | **0.25** | — | — | — |
| NFL QB Elo | — | — | **0.25** | — | — | — |
| Pitcher | — | — | — | — | **0.50** | — |
| NHL xG (MoneyPuck) | — | — | — | — | — | **0.30** |
| Elo | 0.10 | 0.15 | 0.20 | 0.15 | 0.25 | 0.0 |
| Form | 0.10 | 0.0 | 0.30 | 0.10 | 0.25 | 0.15 |
| ShotQuality | 0.10 | — | — | — | — | — |
| Matchup | 0.10 | — | — | — | — | — |
| Poisson | 0.0 | 0.0 | 0.0 | **0.40** | — | 0.0 |
| Soccer xG | — | — | — | 0.25 | — | — |

> **World Cup** mirrors soccer's weights exactly but reads its own national-team namespaces (`elo_state['worldcup']`, `poisson_state['worldcup']`, `soccer_xg_state['worldcup']`, `form_state['worldcup']`) — never the club soccer pool.

> **Poisson is football-only** (`SUPPORTED_SECTORS = {"soccer", "worldcup"}` in `poisson_agent.py`): `predict_pair` returns `None` for every other sector, so it never enters the blend or `model_sources`. Tennis weights: surface 0.30 · serve/return 0.10 · form 0.35 · advanced 0.15 · h2h 0.05 · ranking trend 0.05.

**WNBA weights re-tuned 2026-05-14** via walk-forward sweep over 321 games (`scripts/sweep_wnba_weights.py`) — dropped blend Brier 0.2061 → 0.2019. Form was worst standalone (0.2303) and every top-20 combo zeroed it; generic Elo also weaker than the WNBA-specific stack.

**NFL Phase 2 weights** (validated 2026-05-01 via walk-forward backtest): nfl_efficiency 0.25 · nfl_qb_elo 0.25 · elo 0.20 · form 0.30 · poisson 0.0. nfl_qb_elo carries a per-QB delta layer on top of team Elo so starter swaps shift effective rating without retraining team strength.

**NBA backtest (2025-26, 1,229 games):**

| Model | Brier Score | Accuracy | vs Baseline |
|-------|------------|----------|-------------|
| Ensemble | **0.2032** | **69.2%** | -17.8% |
| PossessionSim | 0.2022 | 68.7% | -18.2% |
| Efficiency | 0.2030 | 69.4% | -17.8% |
| Elo | 0.2208 | 63.3% | -10.6% |
| Poisson | 0.2288 | 67.2% | -7.4% |
| ShotQuality | 0.2319 | 60.1% | -6.2% |
| Matchup | 0.2355 | 60.7% | -4.7% |
| Form | 0.2428 | 62.2% | -1.7% |
| Baseline (always home) | 0.2471 | 55.4% | — |

**WNBA backtest (2025, 323 games):**

| Model | Brier Score | Accuracy | vs Baseline |
|-------|------------|----------|-------------|
| Ensemble | **0.2056** | **68.8%** | -16.3% |
| WNBA Efficiency | 0.2020 | 69.5% | -17.8% |
| WNBA PossessionSim | 0.2025 | 68.6% | -17.6% |
| Elo | 0.2227 | 63.7% | -9.3% |
| Poisson | 0.2286 | 61.9% | -6.9% |
| Form | 0.2472 | 62.2% | +0.7% |
| Baseline (always home) | 0.2456 | 56.7% | — |

WNBA ensemble is within 0.0024 Brier of NBA — effectively at parity. WNBA ships in `shadow` mode for the 2026 season; live Kelly is gated on MODEL-11 shadow validation.

---

### NBA Models

#### Efficiency Model (`EfficiencyModelAgent`, NBA weight=0.30)

The strongest single NBA model. Projects point differential from team Offensive/Defensive Ratings (per-100-possession efficiency) and pace, then converts to win probability via normal CDF.

**How it works:**

Fetches ORTG/DRTG/Pace daily from stats.nba.com `LeagueDashTeamStats` (Advanced). For each matchup:

```
off_factor_a = team_a_ortg / league_avg_ortg
def_factor_b = team_b_drtg / league_avg_drtg
possessions = (pace_a + pace_b) / 2

proj_pts_a = off_factor_a × def_factor_b × league_ortg × possessions / 100
margin = proj_pts_a − proj_pts_b + HOME_EDGE (3.2 pts)
prob_a = normal_cdf(margin / σ)    where σ = 12.0
```

**Confidence:** 0.85 (82 GP), 0.70 (40+ GP), 0.50 (<40 GP, below gate at <20 GP).

**State file:** `data/models/efficiency_state.json`

---

#### Possession Sim (`PossessionSimAgent`, NBA weight=0.30)

Monte Carlo possession-level game simulator. Simulates 10,000 games per matchup at the possession level to produce a full score distribution.

**How it works:**

For each simulated game:
1. Draw total possessions from `N(avg_pace, 3.0)`, clipped to [80, 120]
2. For each possession: sample turnover (team TOV%), then draw points from `N(ppp, 0.45)` where `ppp` is pace-adjusted points-per-possession
3. Add offensive rebound extra possessions (27% of misses)
4. Home team gets +1.5 ORTG boost

Win probability = fraction of sims where team A outscores team B. Deterministic seeding per matchup+date ensures reproducible results.

**Captures what Poisson can't:** pace interaction (fast vs slow), 3PT variance (fat tails from hot/cold shooting), and score correlation (pace drives both teams' totals).

**State file:** Reuses `data/models/efficiency_state.json` (no extra API calls).

---

#### Shot Quality (`ShotQualityAgent`, NBA weight=0.10)

Evaluates offensive output from shot location data — where teams shoot from and how efficiently they convert at each zone.

**How it works:**

Fetches zone-level FGA and FG% from stats.nba.com `LeagueDashTeamShotLocations`. Computes expected points per shot from three zones (3PT, rim, mid-range), with each zone's FG% regressed 30% toward league average to filter variance from skill:

```
regressed_fg3 = team_fg3_pct × 0.70 + league_fg3_pct × 0.30
pts_per_shot = (fg3a × regressed_fg3 × 3 + rim_fga × regressed_rim × 2 + mid_fga × regressed_mid × 2) / total_fga
margin = (pps_a − pps_b) × 85 FGA/game + 2.5 home edge
prob_a = normal_cdf(margin / 12.0)
```

**Why 30% regression:** Season-long 3PT% contains real skill signal but also variance. Full regression to mean (original implementation) was anti-predictive (Brier 0.2745). Partial regression preserves the skill component.

**State file:** `data/models/shot_quality_state.json`

---

#### Matchup Agent (`MatchupAgent`, NBA weight=0.10)

Analyzes how each team's offensive style interacts with the opponent's defensive profile to produce a probability nudge.

**Three dimensions:**

1. **Paint scoring vs rim protection** — does the opponent allow more/fewer points in the paint than league average? Coefficient: 0.15 per point above/below average.
2. **Transition defense** — does the opponent allow excessive fastbreak points? Coefficient: 0.12 per point above/below average.
3. **Turnover battle** — team ball security vs opponent steal rate, with an interaction term that amplifies pain when a careless team faces an active-hands defense.

Each dimension capped at ±1.5 points, total capped at ±4 points → converted to probability via normal CDF.

**State file:** `data/models/matchup_state.json`

---

### WNBA Models

WNBA runs a parallel advanced stack to NBA — separate files, separate state, separate tuning. No code is shared between the two leagues' agents. Change one without risk to the other. ML and spread are **live** as of 2026-05-26; totals stay in shadow until ~30 resolved bets validate the totals output.

#### WNBA Efficiency Model (`WNBAEfficiencyModelAgent`, weight=0.40)

Normal-CDF margin model driven by team ORTG / DRTG / Pace. Mirrors NBA's efficiency agent architecturally but uses WNBA-tuned constants, reads its own state file, and **regularizes inputs with empirical-Bayes shrinkage at predict time**.

**How it works:**

```
# 1. Pull raw team stats from wnba_efficiency_state.json, then SHRINK toward league mean
shrunk = (gp · raw + k · league_avg) / (gp + k)    # k = 8

# 2. Compute projected margin using shrunk stats
off_factor_a = shrunk_ortg_a / league_avg_ortg     # league avg ≈ 100.6 (refreshed each season)
def_factor_b = shrunk_drtg_b / league_avg_drtg
possessions = (shrunk_pace_a + shrunk_pace_b) / 2

proj_pts_a = off_factor_a × def_factor_b × league_ortg × possessions / 100
margin = proj_pts_a − proj_pts_b + HOME_EDGE_PTS (2.6 pts)
prob_a = normal_cdf(margin / SCORE_STDEV)           # σ = 12.5
```

**Why shrinkage:** without it, an early-season team with 8 games gets full credit for noisy ORTG/DRTG samples. At gp=8 the EB formula splits 50/50 between team-specific and league prior; at gp=24 the team dominates 75/25. This regularization replaced the old hard `MIN_GAMES=12` gate (now `MIN_GAMES=4`) — the model now contributes lightly from week 1 instead of going dark for 3+ weeks at each season start. See `shrink_team_stats` in `wnba_efficiency_agent.py`.

**Staleness guard:** the agent calls `state_is_stale_for_today()` and returns `None` whenever `source_season < today.year` during May-Oct, so prior-season ratings can't silently leak into a new season. Background: the 2026 opener saw +24pp ML chalk bias (Aces predicted 74%, went 0/3) because `wnba_offseason_regress.py` regresses Elo but **not** efficiency state — that gap is now caught by the guard and resolved by re-running the seed each May.

**Confidence ramp:** smooth instead of stepped — `smooth_confidence(min_gp)` returns 0.40 (gp=0) → 0.80 (gp≥40). Clears the ensemble's 0.45 confidence gate at gp ≥ 6.

**Tunable constants vs NBA:** `HOME_EDGE_PTS 2.6` (NBA 3.2), `SCORE_STDEV 12.5` (NBA 12.5), `MIN_GAMES 4` (NBA 20), `SHRINK_K 8`. WNBA's 40-game season warrants a lower games-played gate and shrinkage; the weaker home-court edge reflects the 2025 56.7% home win rate (NBA ~59%).

**Seeding:** `scripts/seed_wnba_efficiency.py` walks ESPN's WNBA scoreboard for a season (e.g. 2025 or 2026 May-October), pulls per-game box scores, and computes ORtg/DRtg/Pace/eFG%/TOV%/OREB%/FTr via Dean Oliver formulas:

```
POSS  = FGA + 0.44·FTA + TO − OREB
ORtg  = 100 · PTS / POSS
DRtg  = 100 · OPP_PTS / OPP_POSS
Pace  = POSS per game (WNBA games are 40 min, not 48)
```

Exhibitions (All-Star, international) are filtered via a `REAL_WNBA_TEAMS` allow-list. **Re-run at the start of each season** (`--year 2026`, `--year 2027`, etc.) to overwrite the `source_season` marker and reset rating gaps. Mid-season re-runs are optional — the agent's `update()` is a no-op (score pairs alone don't carry box-score detail), so weekly re-seeds keep stats fresh.

**Backtest validation (post-shrinkage):** replay of 44 resolved ML bets from May 2026 gives Brier 0.2856 — beats the historical broken blend (0.2980), Pinnacle sharp alone (0.2956), and Kalshi listing (0.2926). See `scripts/backtest_wnba_recalibration.py`.

**State file:** `data/models/wnba_efficiency_state.json`

---

#### WNBA Possession Sim (`WNBAPossessionSimAgent`, weight=0.45)

Monte Carlo possession-level WNBA game simulator. Same architecture as NBA's sim but reads the WNBA efficiency state, applies the same EB shrinkage as the efficiency agent, and uses WNBA-tuned possession clips.

**How it works:**

1. Apply `shrink_team_stats()` to ORTG / DRTG / Pace / TOV% before simulating (shared helper from the efficiency agent — keeps both consumers of the state file consistent)
2. For each of 10,000 simulated games:
   - Draw possession count from `N(avg_pace, 3.0)`, clipped to **[65, 100]** (NBA uses [80, 120] — WNBA pace is ~82, NBA ~100)
   - For each possession: sample turnover from team TOV%, then draw points from `N(ppp, 1.10)` where `ppp` is pace-adjusted points-per-possession
   - Home team gets +1.5 ORTG boost (same as NBA)

Win probability = fraction of sims where team A outscores team B. Deterministic seeding per matchup + date for reproducibility.

**Same gates as efficiency:** `MIN_GAMES=4`, staleness guard, smooth confidence ramp. Returns `None` if either team falls below the gate or the state is stale.

**Spread / totals probabilities** (`cover_probability` / `total_probability`) use calibrated σs: margin σ = 12.5 (matches the efficiency agent), total σ = 18.0 (WNBA games are shorter → lower total variance than NBA's σ=20.0). Spread is live; totals stay in shadow until ~30 resolved bets land.

**Playoff tightening is NOT enabled** for WNBA. NBA's `PLAYOFF_ORTG_FACTOR=0.9623` was derived from a specific NBA playoff sample; porting it blindly to WNBA would add unmeasured bias. Leave off until WNBA has a comparable playoff measurement.

**State file:** Reuses `data/models/wnba_efficiency_state.json` (no extra fetches).

---

### Shared Models (all sectors)

#### Elo Model (`EloModelAgent`, default weight=0.35, NBA=0.10)

Dynamic rating system that updates after every game result.

**How it works:**

Each team starts at 1500 Elo. After each game:
```
K_factor × (actual_result − expected_result)
```
is added to the winner's rating and subtracted from the loser's.

| Sector | K-factor | Home Advantage (Elo pts) |
|--------|----------|--------------------------|
| NFL | 25 | +48 |
| NBA | 20 | +100 |
| NCAAB | 20 | +80 |
| Soccer | 30 | +60 |
| LoL / CS2 | 20 | 0 (online) |

**Win probability from Elo:**
```
P(A wins) = 1 / (1 + 10^((Elo_B − Elo_A) / 400))
```

**Strength of Schedule:** Elo gains are multiplied by an SoS factor based on opponent rating. Beating a team +200 above average yields 1.30x, beating one -200 below yields 0.70x. This prevents teams inflating ratings against weak schedules.

**Recency weighting:** Games within the last 14 days get a 1.4x K-factor boost, tapering linearly to 1.0x at 60 days. Kept modest (was 2.0x, reduced to prevent late-season noise from tanking opponents).

**Confidence levels:**

| Games played | Confidence |
|-------------|-----------|
| 0 games | 0.30 (below blend gate — excluded) |
| 1–4 games | 0.45 |
| 5–14 games | 0.60 |
| 15+ games | 0.80 |

**Soccer draws:** Draw probability is estimated from how even the matchup is: `draw_base × (0.5 + 0.5 × closeness)`, then the remaining probability is allocated to home/away proportionally.

**State file:** `data/models/elo_state.json`

---

#### Form Model (`FormModelAgent`, default weight=0.25, NBA=0.10)

Tracks each team's recent performance with exponential decay — recent results matter more than old ones.

**How it works:**

Over the last 10 games, each game gets a decayed weight:
```
weight_i = DECAY^(games_ago)   where DECAY = 0.85
```

Form strength = weighted win rate. Head-to-head probability is computed via the **Bill James Log5 formula**:
```
P(A beats B) = (p_a − p_a × p_b) / (p_a + p_b − 2 × p_a × p_b)
```

Home advantage is applied additively:

| Sector | Home Bonus |
|--------|-----------|
| NCAAB | +5% |
| NBA | +4% |
| Soccer | +4% |
| NFL | +3% |
| LoL / CS2 | 0% |

**Minimum data requirement:** Both teams need at least 3 recorded results; otherwise the model returns `None` and is excluded from the blend.

**Staleness guard:** The model also returns `None` when the most recent record for either team is more than `STALE_DAYS=60` old relative to the game's `event_date`. This protects against cross-season contamination — without the guard, May 2026 WNBA games would be priced against October 2025 form records that no longer reflect current rosters. The reference date is the game being predicted (not wall-clock today), so historical walk-forwards still work correctly.

**State file:** `data/models/form_state.json`

---

#### Poisson Model (`PoissonModelAgent`, football-only, weight=0.40)

Models scoring as a Poisson process — each team has attack and defense strength parameters that combine to predict expected goals/points. **Football-only** (`SUPPORTED_SECTORS = {"soccer", "worldcup"}`): `predict_pair` returns `None` for every other sector, so it never enters the blend elsewhere. Basketball is not a Poisson process (PossessionSim provides better score distributions for NBA/WNBA), and it was net-negative on MLB runs — see the Poisson note in the per-sector weights table above. The `worldcup` namespace uses symmetric neutral-venue league averages (1.30/1.30, no home edge); both football sectors keep the Dixon-Coles correction and explicit draw mass for the 3-way market.

**How it works:**

Each team has:
- `attack_strength` — how many goals/points above league average they score
- `defense_strength` — how many goals/points above average they concede

Expected goals for the home team:
```
λ_home = league_avg_home × attack_home × defense_away
λ_away = league_avg_away × attack_away × defense_home
```

A score matrix is computed up to `max_g` goals/points (8 for soccer, 20–25 for points sports). `P(home score = i, away score = j)` is calculated for each cell. Win/draw/loss probabilities sum over the appropriate cells.

**Dixon-Coles correction:** For low-scoring games, the joint distribution is adjusted at scores (0-0), (1-0), (0-1), and (1-1) using a correction factor `rho=0.1` to correct the Poisson independence assumption at the tails.

**Confidence levels:**

| Data state | Confidence |
|-----------|-----------|
| No data | 0.30 (excluded from blend) |
| Partial (some teams missing) | 0.45 |
| Full data for both teams | 0.65–0.80 |

**State file:** `data/models/poisson_state.json`

---

#### Soccer xG Model (`SoccerXgModelAgent`, soccer + worldcup, weight=0.25)

Expected-goals model driven by ESPN shot data (`shotsOnTarget` / `totalShots`). Each team carries an attacking and defensive xG profile that combines into a match win/draw/loss probability, complementing the goals-based Poisson model with a shot-quality view. The agent is **sector-namespaced** inside one state file: club `soccer` lives at the legacy flat `teams` key; every other sector (currently `worldcup`) lives under `state[sector]['teams']`, so national-team xG never mixes with club xG.

**State file:** `data/models/soccer_xg_state.json`

---

#### NHL xG Model (`NhlXgModelAgent`, NHL-only, weight=0.30)

Team 5-on-5 expected goals for/against per 60 (xGF/60, xGA/60), score-and-venue adjusted, sourced from MoneyPuck's public team CSVs. This is NHL's dominant non-sharp signal; generic Elo is held at 0 for NHL because its K-factor / home-advantage have never been calibrated for hockey, and Form contributes a small recency voice (0.15). Goalie GSAx and special-teams agents are planned for v2/v3.

**Seeding:** `scripts/seed_nhl_xg.py` (MoneyPuck team CSVs). NHL ships in `shadow` mode.

**State file:** `data/models/nhl_xg_state.json`

---

### Calibration and Meta-Model

#### Calibration Layer (`ModelCalibrator`)

Isotonic regression per model — learns a monotonic mapping from raw model output to calibrated probability. Trained from resolved predictions in `predictions.db` via `evmax cleanup adjust`.

**State file:** `data/models/calibration.json`

#### Meta-Model (`MetaModel`)

Logistic regression combiner that takes `sharp_logit`, `blended_logit`, `kalshi_logit`, and `ev_pct` as features. Trained from resolved outcomes. Currently informational — the ensemble uses fixed weights, and the meta-model coefficients are monitored to identify when models add/subtract value.

**State file:** `data/models/meta_model.json`

#### Player Impact Agent (`PlayerImpactAgent`)

Fetches per-player advanced stats (NET_RATING, MIN) from stats.nba.com. Computes the fraction of a team's total impact-minutes lost to injury, converting to a probability adjustment capped at ±15%.

**State file:** `data/models/player_impact_state.json`

---

### Ensemble Model (`EnsembleModelAgent`)

Blends all model agents with the Pinnacle sharp line into one final probability estimate per event. Per-sector weight overrides (see table above) replace default model weights.

**Blending:**

1. Each model produces `(true_prob_a, true_prob_b, confidence, weight)`.
2. Models with `confidence < 0.45` are excluded (prevents data-starved models from polluting the blend).
3. Effective weight = `model_weight × model_confidence`.
4. Sharp odds are blended at `sharp_weight` (default 0.85, auto-tuned by Cleanup Agent).
5. Final blend: `prob = sharp_weight × pinnacle_prob + (1 − sharp_weight) × model_avg`.
6. **Favorite-longshot bias correction:** At extreme probabilities (>80% or <20%), the blend automatically increases the effective sharp weight quadratically. This prevents models from compressing heavy favorites/underdogs toward 50/50.
7. Normalized to sum to 1.0.

When no model has enough data (all below confidence gate), the sharp probability is used directly.

The `model_sources` field on each `EVGap` shows which models contributed: `"efficiency+elo+form+matchup+possession_sim+shot_quality+sharp"` or `"sharp"` (model-only run).

---

### Sharp Books Model (Baseline)

When no statistical models have enough data (new teams, early season), the devigged Pinnacle line is used as the true probability. This is Phase 1 of the pipeline and is always available.

**Devigging method:** Power Method (industry standard).

Find exponent `k` where `Σ(raw_prob_i ^ k) = 1.0` (solved via Brent's method). Then `true_prob_i = raw_prob_i ^ k`. Correctly asymmetric — removes more vig from underdogs.

---

### Tennis Models

Tennis runs six dedicated agents (the four primary signals plus form and H2H). Each returns `None` when its store has no coverage for the players in question, so unseeded matchups fall back to the remaining models.

| Agent | Weight | Signal | State file |
|---|---|---|---|
| `TennisModelAgent` (surface Elo) | 0.30 | Surface-specific Elo (hard / clay / grass) seeded from [Tennis Abstract's Elo leaderboards](https://www.tennisabstract.com/reports/atp_elo_ratings.html) | `data/models/tennis_surface_state.json` |
| `TennisServeReturnAgent` | 0.10 | Logistic on serve-points-won differential, calibrated for bo3 (`k=14`) and bo5 (`k=18`); slam detection from market title | `data/models/tennis_serve_return_state.json` |
| `TennisFormAgent` | 0.35 | Recency-weighted match form (the momentum voice) | `data/models/tennis_form_state.json` |
| `TennisAdvancedStatsAgent` | 0.15 | Logistic on BP conv, RPW, UE rate, W/UE ratio; full 4-feature model when MCP data available, RPW-only reduced model otherwise | `data/models/tennis_advanced_state.json` |
| `TennisH2HAgent` | 0.05 | Head-to-head record nudge with Laplace smoothing + sample-size shrinkage; capped at ±18pp from 0.5; requires ≥3 meetings | `data/models/tennis_h2h_state.json` |
| `TennisRankingTrendAgent` | 0.05 | 12-week ranking-momentum log-odds nudge; positive = climbing the rankings; capped at ±0.40 logit | `data/models/tennis_ranking_trend_state.json` |

A tennis gap is only treated as a **live play** when `model_sources` contains all four primary models (`tennis_surface`, `tennis_serve_return`, `tennis_form`, `tennis_advanced`); h2h / ranking-trend are optional since they can't fire on most matches. Partial-blend gaps are demoted to `shadow` (Kelly zeroed, hidden from the play table) — see `REQUIRED_BLEND_MODELS` in `ev_gap_agent.py`. All tennis agents are seeded from **[Tennis Abstract](https://www.tennisabstract.com/)** (Jeff Sackmann's site), which replaced his `tennis_atp` / `tennis_wta` GitHub CSVs after they went offline in 2026: surface Elo from the [Elo leaderboards](https://www.tennisabstract.com/reports/atp_elo_ratings.html), and serve/return + advanced + form + H2H from the per-match `matchmx` data behind [`leaders.cgi`](https://www.tennisabstract.com/cgi-bin/leaders.cgi) (advanced's winners/UE feature comes from the [winners/errors leaderboards](https://www.tennisabstract.com/reports/winners_errors_leaders_men_last52.html)). See [Seed Tennis Models](#seed-tennis-models) below.

---

### MLB Pitcher Agent (`PitcherAgent`)

Turns the matchup of probable starters into a moneyline prob for MLB game markets via Pythagenpat expectation (exp=1.83) on each starter's run-allowed rate — a **60% FIP / 40% ERA blend** when FIP is seeded, ERA-only otherwise — plus an adaptive home-field bonus. Returns `None` for non-baseball markets and when no probable starter can be resolved.

**Live probable starters: official MLB Stats API** (`evmax/clients/mlb_statsapi.py`, `statsapi.mlb.com`, free/no key). Teams are keyed by **stable integer ID**, so resolution no longer depends on fuzzy nickname matching — the multi-word nicknames (Red Sox / White Sox / Blue Jays) that the old ESPN-scoreboard name match dropped now resolve cleanly. ESPN scoreboard remains a fallback if the Stats API is unreachable. Pitcher ERA/FIP come from the seeded DB (`scripts/seed_pitcher_fip.py`, Baseball-Reference via pybaseball), keyed by accent-folded name.

**Pitcher is required for a baseball moneyline bet.** When the starter still can't be resolved (e.g. a night-before scan before probables post), the ML bet is **skipped** rather than logged on a generic Elo+Form blend — those pitcher-less bets backtested at −23% flat ROI vs +18% when the pitcher contributed. Spread/total are unaffected (they don't consume the pitcher model).

> **Why MLB ML has thin value (measured 2026-06):** the model moves only ~+0.3pp off the sharp devig, so ~90% of the apparent edge is a Kalshi-vs-Pinnacle price gap, not model insight. Two avenues to add orthogonal signal were tested and rejected: a bullpen quality/fatigue component (Δ −0.0012 Brier — correlated with Elo+Form, not orthogonal) and ensemble isotonic calibration (overfits a single season; −0.00068 on holdout). Baseball ML is treated as a thin arb, not a model-edge play.

**State file:** `data/models/pitcher_state.json`

---

### NBA Player Prop Model (`nba_stats`)

Per-player line distribution model used by the EVGap agent's prop pipeline. For a given (player, stat, line) it pulls the last 15 games via `nba_api`, computes a weighted mean / stdev, applies a continuity correction (`+0.5` to the threshold so OVER hits the strict `actual > line` semantics Kalshi resolves on), and rescales to per-36 minutes when the player's average minutes are ≥ 5. The output is a normal-CDF tail probability for `P(X > line)`.

**Cache file:** `data/nba_props_cache.json` (game logs)

See [Player Prop Pipeline](#player-prop-pipeline) for how observations are logged and calibrated.

---

### Spread Distribution Model (Spread markets only)

For spread markets (e.g., "Lakers win by more than 5.5"), Pinnacle posts one line per game. Kalshi lists many alternative lines. This model translates Pinnacle's single line into a cover probability at any Kalshi line.

**Assumption:** Final point margin is normally distributed `N(μ, σ²)`.

| Sector | σ (standard deviation) | Notes |
|--------|----------------------|-------|
| NBA | 12.5 pts | Bumped 11.5→12.5 on 2026-05-07 (Brier improvement on 82 resolved spread bets) |
| WNBA | 12.5 pts | Matches WNBA possession sim's spread σ; 40-min games |
| NFL | 14.0 pts | |
| NCAAB | 12.5 pts | |
| Baseball | 4.0 runs | **Gated** — direct line matches only |
| NHL | 2.0 goals | **Gated** — direct line matches only |
| Soccer | 1.9 goals | **Gated** — direct line matches only |
| Default | 11.5 pts | Fallback for unconfigured sectors |

1. Infer implied mean `μ` from Pinnacle's posted line and its cover probability.
2. Estimate `P(margin > target_line)` using the normal CDF.
3. Only evaluate Kalshi lines within 1 sigma of Pinnacle's line (accuracy degrades and liquidity thins further out).

**Low-scoring sport gating** (`_LOW_SCORING_SECTORS = {baseball, nhl, soccer}`): for sports where the margin distribution is Skellam-shaped (mass concentrated near 0, fat thin tails) rather than Gaussian, the normal-CDF extrapolation is unreliable. A 3-run jump from MLB's posted run line to a Kalshi alt-spread previously produced 165%+ false EVs at default σ. Two gates apply:
- **Distance gate** — the model only fires when the Kalshi line matches Pinnacle's within `LOW_SCORING_LINE_TOLERANCE=0.5`.
- **Absolute-magnitude cap** (`_LOW_SCORING_MAX_ABS_LINE = {baseball: 1.5, nhl: 1.5, soccer: 1.5}`) — never price an alt line past the standard run line / puck line / Asian-handicap ceiling, even when a sharp book posts its own −4.5 ladder (distance == 0 in that case, so the distance gate alone let it through). MLB −4.5 alt run lines went 2-for-15 live+shadow while the model predicted 27–46% cover; this cap rejects them outright.

**Spread is also blended with PossessionSim** for NBA/WNBA: the cover prob is `0.65 × spread_dist + 0.35 × possession_sim` margin distribution. PossessionSim contributes a real empirical margin CDF (not a normal approximation), so it picks up pace interaction and fat-tail variance the Gaussian misses.

---

## Seeding and Updating Models

Models start with no data and output low-confidence 50/50 predictions (which are automatically excluded from the blend). To get meaningful model contributions, seed them with historical data.

### Seed Elo Ratings

Provide a JSON file mapping team names to Elo ratings:

```bash
evmax agents seed elo --sector nba --file data/seeds/nba_elo.json
```

`data/seeds/nba_elo.json`:
```json
{
  "celtics": 1623,
  "nuggets": 1598,
  "thunder": 1591,
  "cavaliers": 1587,
  "knicks": 1572,
  "timberwolves": 1565,
  "bucks": 1541,
  ...
}
```

### Seed Form Records

Provide a list of recent game results:

```bash
evmax agents seed form --sector nba --file data/seeds/nba_results.json
```

`data/seeds/nba_results.json`:
```json
[
  {"date": "2025-10-22", "home": "celtics", "away": "knicks", "score_home": 108, "score_away": 97},
  {"date": "2025-10-23", "home": "nuggets", "away": "lakers", "score_home": 114, "score_away": 103},
  ...
]
```

### Seed Poisson Attack/Defense Strengths

```bash
evmax agents seed poisson --sector soccer --file data/seeds/epl_poisson.json
```

`data/seeds/epl_poisson.json`:
```json
{
  "league_avg": {"home": 1.52, "away": 1.18},
  "teams": {
    "manchester city": {"attack": 1.45, "defense": 0.68, "games": 38},
    "arsenal": {"attack": 1.32, "defense": 0.71, "games": 38},
    ...
  }
}
```

### Seed / Refresh WNBA Efficiency

WNBA's advanced agents read their state from `data/models/wnba_efficiency_state.json`, seeded by walking ESPN's scoreboard + box-score endpoints and deriving ORtg / DRtg / Pace / Four Factors via Dean Oliver formulas. Re-run weekly during the 2026 season to keep inputs fresh — the agent's `update()` is a no-op because per-game score pairs alone can't reconstruct possession counts.

```bash
# 2025 full season (the default at project start)
python scripts/seed_wnba_efficiency.py

# 2026 regular-season refresh once the season is underway
python scripts/seed_wnba_efficiency.py --year 2026

# Preview without writing
python scripts/seed_wnba_efficiency.py --dry-run
```

Exhibitions (All-Star, international friendlies) are filtered out via the `REAL_WNBA_TEAMS` allow-list so league averages stay clean.

### Run WNBA Offseason Elo Regression

Once per offseason, shrink every WNBA team's Elo toward 1500 and apply the year's roster-move deltas from a YAML file. The deltas are tier-based (Superstar ±50, All-Star ±35, Quality Starter ±20, rookie-slot-specific).

```bash
# Preview
python scripts/wnba_offseason_regress.py --dry-run

# Apply + auto-backup of the existing state
python scripts/wnba_offseason_regress.py
```

Config lives at `data/models/wnba_2026_offseason.yaml` — edit the `moves:` list and `expansion_priors` for each subsequent season. The script backs up `elo_state.json` as `elo_state.backup.wnba_offseason_{timestamp}.json` (gitignored) before writing.

### Seed Tennis Models

The tennis agents are seeded from **[Tennis Abstract](https://www.tennisabstract.com/)** (after Jeff Sackmann's `tennis_atp` / `tennis_wta` GitHub CSVs went offline in 2026), via two seeders:

```bash
# Surface Elo — from Tennis Abstract's Elo leaderboards (idempotent)
uv run python scripts/seed_tennis_abstract_elo.py

# serve/return + advanced + form + H2H + ranking_trend — from the per-match `matchmx` data
uv run python scripts/seed_tennis_models.py --years 2024,2025,2026

# Skip the winners/errors (UE) augmentation → advanced runs RPW-reduced
uv run python scripts/seed_tennis_models.py --no-mcp
```

Both run weekly via the `weekly-tennis-surface-elo-refresh` scheduled task. All six tennis models are now sourced purely from Tennis Abstract — `ranking_trend` is seeded from `matchmx`'s per-match `winner_rank` / `loser_rank` columns (one dated snapshot per match, full-replace), so the old Sackmann reseed + ESPN top-150 fallback (and their `weekly-tennis-rankings-refresh` task) have been removed.

What gets written:

| Output | Source | Aggregation |
|---|---|---|
| `tennis_surface_state.json` | [Elo leaderboards](https://www.tennisabstract.com/reports/atp_elo_ratings.html) (overall + hElo/cElo/gElo) | Pre-computed surface Elo → hard / clay / grass (+ indoor←hard) |
| `tennis_serve_return_state.json` | `matchmx` serve columns (`pts`/`fwon`/`swon`) | Per-player SPW = `(1stWon + 2ndWon) / svpt`, recency-weighted at predict time |
| `tennis_advanced_state.json` | `matchmx` (BP/RPW) + [winners/errors leaderboards](https://www.tennisabstract.com/reports/winners_errors_leaders_men_last52.html) (UE) | BP conv, RPW, UE rate, W/UE ratio (RPW-reduced where UE absent) |
| `tennis_h2h_state.json` | `matchmx` (winner / loser) | Win counts per alphabetically-sorted player pair |
| `tennis_form_state.json` | `matchmx` (opp rank, surface, minutes) | Recency-weighted match history |
| `tennis_ranking_trend_state.json` | `matchmx` (`winner_rank` / `loser_rank` per match) | Dated rank series per player (one snapshot/match date); the agent reads its 12-week momentum off it |

`matchmx` covers the trailing ~2.5 seasons (2024→present) for the full bettable field across ranking segments — a typical run covers roughly **900 ATP / 390 WTA players** and ~**10,400 H2H pairs**.

---

### Feed Game Results Back

After a game completes, update all models:

```bash
evmax agents update \
  --sector nba \
  --home celtics \
  --away knicks \
  --score-home 112 \
  --score-away 104
```

This updates Elo ratings, Form records, and Poisson attack/defense strengths, then saves state. NBA advanced models (Efficiency, PossessionSim, ShotQuality, Matchup) are fetched in bulk from stats.nba.com daily and do not require per-game updates.

### View Current Elo Leaderboard

```bash
evmax agents ratings nba
evmax agents ratings soccer
```

---

## Cleanup Agent: Logging, Resolution, and Calibration

The Cleanup Agent closes the feedback loop. Every scan automatically logs its +EV plays to a local SQLite database (`data/predictions.db`). After games complete, the resolver fetches actual outcomes and computes calibration metrics that drive automatic model tuning.

### How It Works

```
Daily scan  →  auto-log gaps to predictions.db
End of day  →  evmax cleanup resolve  →  fetch real scores from ESPN/bo3.gg
Weekly      →  evmax cleanup adjust   →  compare Brier scores, tune sharp_weight
```

### Brier Score Calibration

The **Brier score** measures probability calibration: `mean((predicted_prob − outcome)²)`. Lower is better.

Two scores are tracked per resolved prediction:
- `brier_model` — using the full blended probability (Pinnacle + statistical models)
- `brier_sharp` — using the raw devigged Pinnacle probability alone

The difference tells you whether the statistical models are adding value:

| Condition | Action |
|-----------|--------|
| `brier_model` < `brier_sharp` by >5% | Models improving — lower `sharp_weight` by 0.05 |
| `brier_model` > `brier_sharp` by >5% | Models underperforming — raise `sharp_weight` by 0.05 |
| Within 5% | No change |

Bounds: `sharp_weight` stays in `[0.40, 0.95]`. Adjustments happen at most once per 7 days and require 30+ resolved predictions.

**Per-sector overrides** in `data/model_config.json::sharp_weight_by_sector` let sectors with different model maturity diverge from the global. Current values:

| Sector | sharp_weight | Why |
|--------|--------------|-----|
| NBA | 0.70 | Mature dedicated stack (efficiency + possession_sim) — lean more on models |
| Tennis | 0.85 | |
| Soccer | 0.88 | |
| Baseball | 0.88 | |
| LoL / CS2 | 1.00 | No competitive statistical model yet — pure sharp |
| Global default | 0.85 | Applied to every sector not listed above (incl. WNBA, NFL, NHL, NCAAB) |

These values live in `data/model_config.json::sharp_weight_by_sector`.

### Exposure Guard

To prevent over-concentration on a single game, the pipeline enforces a hard cap: total Kelly exposure across all bets on the same game cannot exceed **8% of bankroll**. If multiple +EV plays reference the same event (e.g., a moneyline + spread on the same game), bets are scaled proportionally until the combined stake stays within the cap. Bets that cannot fit are dropped. The scan output shows a warning when plays are dropped or capped.

### Props in Scans

Player prop markets (NBA player stats, NFL yardage, etc.) are shown in scan output but **not logged to `predictions.db`**. This keeps the predictions log focused on game-level markets while still surfacing prop opportunities for manual review. Prop event IDs contain `::prop::` and are excluded from the DB write path.

The updated `sharp_weight` is written to `data/model_config.json` and automatically picked up by the next `evmax agents scan` run — no manual flag needed.

### Sharp Weight Progression

As models mature and accumulate more resolved predictions, the expected path is:

| Phase | sharp_weight | Trigger |
|-------|-------------|---------|
| Phase 1 (now) | 0.85 | Default — trust Pinnacle heavily while models calibrate |
| Phase 2 | ~0.75 | After 2–3 weeks of resolved predictions showing model value |
| Phase 3 | ~0.65 | 50+ resolved games, Brier consistently improving |
| Phase 4 | ~0.50 | 200+ resolved games, CLV/xG integration |
| Phase 5 | ~0.40 | Models consistently outperforming Pinnacle |

### Data Sources for Outcome Resolution

Resolution is configured per-category via the `resolver` field in [`data/categories.yaml`](data/categories.yaml) — that field is authoritative, not this table.

| Resolver | Source | Categories |
|----------|--------|-----------|
| `espn_scoreboard` | ESPN scoreboard API (World Cup reads `fifa.world`) | NBA, NFL, NCAAB, NCAAW, Soccer, World Cup, Baseball, WNBA, NHL |
| `espn_boxscore` | ESPN boxscore API | NBA props, NFL props |
| `bo3gg` | bo3.gg matches API | LoL, CS2 |
| `kalshi_settlement` | Kalshi settlement (`result` field) | Tennis |

### Recommended Daily Workflow

```bash
# Morning — find plays
evmax agents scan --bankroll 250 --kelly 0.5
# → gaps auto-logged to predictions.db
# → ALL Kalshi + Pinnacle data archived to archive.db

# Before placing bets — verify prices are still live (WebSocket real-time check)
evmax agents verify --date 2026-03-23 --bankroll 250 --kelly 0.5

# Next morning — resolve yesterday's outcomes (both systems)
evmax cleanup resolve --date YYYY-MM-DD      # resolves flagged bets via ESPN/bo3.gg
evmax archive resolve --date YYYY-MM-DD      # resolves ALL archived markets via Kalshi API

# Check your bet log
evmax cleanup show --date YYYY-MM-DD

# Weekly — backtest against archived history
evmax archive backtest --since YYYY-MM-DD

# Weekly — check calibration of flagged bets
evmax cleanup metrics --weeks 4

# Weekly — auto-tune sharp_weight
evmax cleanup adjust

# Weekly — re-seed models with fresh data
evmax cleanup train --sectors lol,cs2
```

### Storage

All predictions and outcomes are stored in `data/predictions.db` (SQLite).

| Table | Contents |
|-------|---------|
| `ev_predictions` | One row per +EV gap per scan — includes kalshi price, blended prob, EV%, kelly, sharp_weight used, `bankroll_used` (scan-time bankroll for consistent verify/pick sizing) |
| `ev_outcomes` | One row per resolved market — outcome (1/0), result source, timestamps |

`data/model_config.json` persists `sharp_weight`, Brier score history, and adjustment timestamps.

---

## Closing Line Value (CLV)

CLV is the primary leading indicator for whether a betting strategy is genuinely +EV. Win-rate noise is high over weeks-to-months samples, but **closing-line CLV stabilizes much faster**: if you consistently take prices the market then moves toward, the long-run profit follows even before realized win rate confirms it. evmax captures CLV automatically alongside every resolved bet.

### What's measured

Each row in `ev_predictions` carries three CLV-related columns, populated when outcomes resolve:

| Column | Formula | Interpretation |
|--------|---------|---------------|
| `kalshi_clv_pct` | `kalshi_close − kalshi_entry` | Did Kalshi's own price drift toward your side after you logged the bet? Positive = good entry vs the market's eventual close. |
| `pinnacle_drift_pct` | `pinnacle_close_prob − pinnacle_entry_prob` | How much Pinnacle's sharp line moved between scan time and pre-tipoff close. Independent signal of *market* movement (not our model). |
| `clv_pct` | `pinnacle_close_prob − kalshi_entry_price` | The conventional CLV: how Pinnacle's closing fair-value compares to what we paid on Kalshi. Positive = beat the sharp close. |

All three use the **conventional sign convention** (positive = good for our side). For NO-side bets the calculator flips signs automatically before storing.

### Closing-line snapshot capture

The closing line is the last Pinnacle snapshot strictly **before tipoff** — not the post-game settlement price. This is enforced in `evmax/archiver.py`:

- Closing snapshots are written to `archive.db::archived_sharp_odds` continuously by every scan
- At resolution time the resolver queries the latest snapshot with `fetched_at < event_start_utc` for each `event_id`
- Snapshots fetched *after* tipoff are excluded (they would leak in-game line movement and corrupt CLV)

### Minutes-to-tipoff stratification

Each bet also captures `minutes_to_tipoff` — how far in advance of game start we logged it. This lets us slice CLV by scan timing to validate the "late beats early" hypothesis (sharper prices closer to tip, but also tighter liquidity windows).

```bash
python scripts/clv_report.py                  # all sectors, full window
python scripts/clv_report.py --sector wnba    # WNBA only
python scripts/clv_report.py --since 2026-05-01 --until 2026-05-31
```

The report strips out **both-sides cancellation** automatically — when two markets on opposite sides of the same event both resolve, their CLV averages to zero and adds noise to the headline number. The script also pairs CLV with a Brier-score check so you can spot the case where CLV is positive but Brier is bad (model finding mispriced markets that still lose — possible but suspicious; usually means an inflated probability cap somewhere).

### Late-news tagging

Bets logged within 6 hours of a starter-status change get a `late_news` tag on the `model_sources` field. These rows often have outsized CLV because they catch the market mid-adjustment — but they're also higher-variance, so the report breaks them out separately. See `b78df1b` for the implementation.

### What "good" CLV looks like

| CLV (`clv_pct`) over N bets | Interpretation |
|---|---|
| > +1.5% sustained over 100+ bets | Strong indicator of genuine edge; expect ROI to follow |
| 0 to +1.5% | Marginal; could be noise, but Brier-corroborated CLV here usually still profitable |
| < 0% over 100+ bets | Strategy is taking prices that drift away from you — reassess model or selection |

CLV is more informative than win-rate for the first 50-200 bets because it's a measurement of *every* bet's price quality, not just a binary win/loss sample.

---

## Player Prop Pipeline

NBA player props (points / rebounds / assists / threes / steals / blocks / PRA) run through a separate logging path from game-level bets. Props are deliberately excluded from `ev_predictions` to keep the bet log clean, and instead land in `prop_observations` for model calibration and offline analysis.

### How it works

1. **Scan emits prop EVGaps** alongside game gaps. A per-type cap (`--max-props`, default 10) prevents prop spam at the top of the table.
2. **All prop lines are logged** — including negative-EV ones — to `prop_observations`. This is the training set for the next round of NBA-prop calibration.
3. **The `nba_stats` model** (per-player log distribution from `nba_api`) competes side-by-side with Pinnacle's prop devig in the EVGap agent. When neither side has coverage the prop is dropped.
4. **Resolution** pulls ESPN boxscores the next morning and fills `actual_value` + `outcome` (`1` = OVER hit, `0` = UNDER). The resolver looks up each stat by name in the boxscore stat-group `keys` array — never by hardcoded index — because ESPN's column layout drifts between seasons.

### `prop_observations` schema

| Column | Notes |
|---|---|
| `scan_date` | When the scan ran |
| `event_date` | Actual game date (anchored at noon UTC in the Kalshi ticker parser to prevent local-tz date drift) |
| `player_name`, `stat_type`, `line` | The line being observed |
| `kalshi_price`, `sharp_prob`, `ev_pct` | Snapshot at scan time |
| `l15_games` | Sample size used by `nba_stats` |
| `actual_value`, `outcome`, `resolved_at` | Filled by `evmax cleanup resolve-props` |

### Daily prop workflow

```bash
# Morning — props are auto-logged inside the normal scan
evmax agents scan --bankroll 500

# Cap the number of prop rows shown in the +EV table (default 10)
evmax agents scan --max-props 5

# Next morning — fetch ESPN boxscores and fill actual_value/outcome
evmax cleanup resolve-props --sector nba --date YYYY-MM-DD

# Browse the observation log
evmax cleanup props --days 7
evmax cleanup props --stat points --resolved-only

# Calibration: predicted vs actual hit rate, bucketed by probability
evmax cleanup prop-calibration --weeks 4
evmax cleanup prop-calibration --weeks 4 --stat rebounds
```

The calibration table groups resolved props by `(stat_type, sharp_prob bucket)` and reports model hit rate vs realized hit rate. Use it to spot stat-specific biases (e.g. the model under-pricing 3-point props because per-36 normalization smooths over hot streaks).

---

## Web Dashboard

`evmax dashboard serve` launches a FastAPI + vanilla-JS dashboard at `http://127.0.0.1:8000/` that mirrors the CLI scan / verify / pick flow without leaving the browser. Useful for visual triage when you have a lot of plays open at once.

### Launch

```bash
# Default — localhost:8000, no auto-reload
evmax dashboard serve

# Custom host/port
evmax dashboard serve --host 0.0.0.0 --port 8080

# Auto-reload during development
evmax dashboard serve --reload
```

### What it shows

- **Pending plays** — every +EV gap from the most recent scan, grouped by sector with date filtering, EV%, Kelly stake, model sources, and one-click pick / unplace buttons
- **Placed bets** — current open positions with live CLV (Pinnacle close vs entry) and pending-resolution status
- **Resolved bets** — historical results with sim vs real P&L split and per-sector ROI
- **Profit chart** — running P&L curve (real bets in solid, simulated in dashed)
- **Action buttons** — kick off a fresh scan, mark bets placed, resolve a date, or run `cleanup metrics` directly from the UI without dropping back to the shell

### API surface

The dashboard is backed by a small JSON API that's also useful for ad-hoc tooling:

| Endpoint | Purpose |
|---|---|
| `GET /api/dashboard` | Combined snapshot for the landing page |
| `GET /api/profit` | P&L series for the chart |
| `GET /api/summary` | Real vs simulated stake / P&L / ROI rollups |
| `POST /api/scan` | Run a coordinator cycle and return the gap list |
| `POST /api/pick` | Mark a list of `market_id`s as placed |
| `POST /api/update-placed` | Update the price/stake of a placed bet |
| `POST /api/unplace` | Demote a placed bet back to simulated |
| `POST /api/resolve` | Trigger outcome resolution for a date |
| `POST /api/metrics` | Run the Brier-score calibration report |

### Date-handling note

Scan rows display `event_date` (the actual game date), not `scan_date`. Kalshi ticker dates are parsed at noon UTC inside `kalshi.py:_parse_ticker_date` so subsequent `.astimezone()` conversions in the dashboard payload can't roll the date back to the previous day in negative-offset US time zones.

---

## Data Archive and Backtest

Every scan automatically snapshots **all** fetched Pinnacle odds and Kalshi market prices to `data/archive.db` — not just the 2%+ EV plays. This builds a free historical dataset over time so you never need to pay for a commercial historical-odds data tier.

### Why archive everything?

The cleanup system (`ev_predictions` + `ev_outcomes`) only tracks markets that were flagged as +EV. The archive captures everything, which lets you:

- Retroactively test different EV thresholds against real results
- Measure calibration across the full probability range (not just your bets)
- Detect when the model was finding value that fell just below your threshold
- Build a growing dataset that improves backtest power over time

### Archive Pipeline (run daily)

```
Scan runs       →  archive.db auto-updated (scan + Kalshi + Pinnacle snapshots)
Day after games →  evmax archive resolve --date YYYY-MM-DD
Weekly          →  evmax archive backtest --since YYYY-MM-DD
```

### Step 1 — Build the archive (automatic)

The archive is populated automatically every time you run a scan:

```bash
evmax agents scan --bankroll 250 --kelly 0.5
# → data/archive.db updated with all Kalshi prices + Pinnacle odds
```

Check what's been collected:

```bash
evmax archive stats
```

### Step 2 — Resolve outcomes (day after games)

Kalshi settles markets within ~24 hours of game end. Run this the next morning:

```bash
# Resolve yesterday's markets
evmax archive resolve --date 2026-03-20

# Preview what would be fetched (no writes)
evmax archive resolve --date 2026-03-20 --dry-run
```

This re-fetches each archived Kalshi ticker via the API and checks the `result` field (`yes`/`no`). Settled markets are stored in `archived_outcomes`. Markets still open are skipped — re-run the next day.

### Step 3 — Run the backtest

Once you have a few weeks of data with resolved outcomes:

```bash
# Full backtest over all sectors
evmax archive backtest --since 2026-03-01

# Filter to a single sector
evmax archive backtest --since 2026-03-01 --sector soccer

# Test a higher EV threshold
evmax archive backtest --since 2026-03-01 --ev-threshold 0.05

# Custom date range
evmax archive backtest --since 2026-03-01 --until 2026-03-20 --bankroll 500
```

Output includes:
- **Summary**: bets flagged, resolved, win rate, ROI, Brier score
- **Threshold comparison**: P&L at 2%, 3%, 5%, 8%, 10% EV cutoffs side by side
- **P&L by sector**: win/loss/ROI broken out per sector
- **Calibration table**: 10-bin chart comparing predicted probabilities to actual win rates

### Recommended weekly backtest workflow

```bash
# Every morning after games finish — resolve the previous day
evmax archive resolve --date $(date -v-1d +%Y-%m-%d)   # macOS
evmax archive resolve --date $(date -d yesterday +%Y-%m-%d)  # Linux

# Weekly — evaluate strategy performance
evmax archive backtest --since 2026-03-01 --sector soccer
evmax archive backtest --since 2026-03-01 --sector tennis
evmax archive backtest --since 2026-03-01            # all sectors

# Export raw data for offline analysis
evmax archive export --sector soccer --since 2026-03-01 --format jsonl --out /tmp/soccer.jsonl
```

### Storage

| DB | Table | Contents |
|----|-------|---------|
| `data/archive.db` | `scan_sessions` | One row per scan cycle — timestamp, sectors, counts |
| `data/archive.db` | `archived_sharp_odds` | ALL Pinnacle odds fetched (every event, every scan) |
| `data/archive.db` | `archived_kalshi_markets` | ALL Kalshi market prices fetched (every market, every scan) |
| `data/archive.db` | `archived_outcomes` | Kalshi settlement results — 1=YES won, 0=NO won |
| `data/predictions.db` | `ev_predictions` | +EV flagged bets only (2%+ threshold) |
| `data/predictions.db` | `ev_outcomes` | Resolved outcomes for flagged bets (via ESPN/bo3.gg) |

The archive and cleanup DBs are independent — the archive stores raw everything, the cleanup DB stores your actual bet log with fuller resolution metadata.

---

## Core EV and Kelly Math

### American Odds Format

All odds in evmax output are displayed as **American odds** (+/−), not Kalshi's raw cent prices (0.00–1.00).

**Converting between formats:**

| Kalshi price | Implied prob | American odds |
|-------------|-------------|--------------|
| $0.65 | 65% | −186 |
| $0.50 | 50% | +100 (even) |
| $0.40 | 40% | +150 |
| $0.30 | 30% | +233 |

Conversion rules:
```
# Favorite (prob ≥ 50%):
american = -(prob / (1 - prob)) * 100   →  e.g. 0.65 → -186

# Underdog (prob < 50%):
american = ((1 - prob) / prob) * 100    →  e.g. 0.40 → +150
```

**K Odds** = Kalshi market price as American odds (what you're being offered).
**True Odds** = Model's fair-value probability as American odds (what it should be worth).
The gap between them is the edge.

### Expected Value

```
EV = (true_probability × payout) − 1

payout = 1 / yes_price  (Kalshi YES contracts are priced 0–1)
```

**Example:**
- Kalshi: YES at $0.40 (shown as **+150**) → payout = 2.5×
- Pinnacle devigged true prob = 48% (shown as **+108**)
- EV = 0.48 × 2.50 − 1 = **+20%**

Any EV ≥ 2% is flagged as a play.

### YES-Team Alignment

Kalshi markets are directional — each market represents a specific team winning (YES side). Pinnacle's `outcome_a` is always the home/favored team. The system automatically aligns:

- YES team = home team → use `true_prob_a`
- YES team = away team → use `true_prob_b` (swap)
- YES team = draw/tie → use `true_prob_draw` (soccer)

### Kelly Criterion

Full Kelly:
```
K_full = (p × b − q) / b

p = true probability of winning
q = 1 − p
b = payout − 1  (net odds)
```

Applied discounts:
```
1. Base fraction:          × kelly_fraction (0.25 = quarter Kelly, 0.5 = half Kelly)
2. Confidence discount:    × min(1.0, edge_pct / 0.20)
3. Liquidity discount:     × max(0.25, 1.0 − spread_pct × 5)

K_final = clamp(K_adjusted, 1%, 5%)
```

The 5% hard cap prevents any single bet from exceeding 5% of bankroll regardless of model output.

### Joint Kelly (Optional, Correlation-Aware)

Set `EVMAX_JOINT_KELLY_ENABLED=true` in `.env` to replace fractional-Kelly + exposure-guard with a single correlation-aware optimization (`evmax/ev/joint_kelly.py`).

Legs sharing a game outcome are sized **jointly** using a two-factor Gaussian copula:
- Moneyline + spread → share a *margin* axis (high correlation)
- Over/under totals → share a *total* axis
- Cross-market (e.g. moneyline vs total) → private axes

The optimizer expands the per-event gross cap from 8% toward `joint_kelly_max_gross_pct` (default 15%) **only when portfolio variance drops below the naive independent sum** — that is, when contradictory legs genuinely hedge. Same-direction (redundant) legs stay pinned at 8%. Single-leg events reduce exactly to fractional Kelly, so this is safe to flip on/off without regressing existing behavior.

This is the right way to size the kind of multi-leg setup you see when a model finds value on both an alt-spread blowout (longshot) and the underdog ML (hedge) — fractional Kelly per leg overcaps because it ignores the negative covariance between them.

---

## CLI Reference

### Agent Pipeline

```bash
# Full scan — all sectors
evmax agents scan

# Custom bankroll + Kelly fraction
evmax agents scan --bankroll 500 --kelly 0.25

# Target sectors
evmax agents scan --sectors nba,soccer,nfl

# Filter results
evmax agents scan --min-ev 0.05 --top 15

# Skip models (sharp-only, fastest)
evmax agents scan --no-models

# Skip injury adjustments
evmax agents scan --no-injuries

# Continuous scan with adaptive intervals (90s live, 3min <1h, 10min 1-4h, 30min >4h)
evmax agents scan --loop --sectors nba,soccer
```

### Pre-Bet Price Verification

```bash
# Re-check all logged plays for today via WebSocket (real-time prices)
evmax agents verify --date 2026-03-23 --bankroll 250 --kelly 0.5

# Shows: Scan Odds | Live Odds (American +/-) | Δ Price | Scan EV% | Live EV% | Status (LIVE/STALE)
```

### Model Management

```bash
# Seed model state from JSON
evmax agents seed elo --sector nba --file seeds/nba_elo.json
evmax agents seed form --sector nba --file seeds/nba_results.json
evmax agents seed poisson --sector soccer --file seeds/epl_poisson.json

# View Elo leaderboard
evmax agents ratings nba
evmax agents ratings soccer

# Record a completed game result
evmax agents update --sector nba --home celtics --away knicks \
  --score-home 112 --score-away 104
```

### Cleanup and Calibration

```bash
# Show logged bets + outcomes
evmax cleanup show --days 7
evmax cleanup show --sector nba --resolved

# Resolve outcomes for a date (defaults to yesterday)
evmax cleanup resolve
evmax cleanup resolve --date 2026-03-13

# Brier score report
evmax cleanup metrics --weeks 4

# Auto-adjust sharp_weight based on Brier scores
evmax cleanup adjust
evmax cleanup adjust --force   # override 7-day cooldown

# Re-seed models from live data
evmax cleanup train --sectors lol,cs2
evmax cleanup train --sectors nba,soccer
```

### Data Archive

```bash
# Check what's been archived
evmax archive stats

# Resolve outcomes for a date (run day after games)
evmax archive resolve --date 2026-03-20
evmax archive resolve --date 2026-03-20 --dry-run   # preview only

# Backtest over archived history
evmax archive backtest --since 2026-03-01
evmax archive backtest --since 2026-03-01 --sector soccer
evmax archive backtest --since 2026-03-01 --ev-threshold 0.05
evmax archive backtest --since 2026-03-01 --until 2026-03-20 --bankroll 500

# Export raw data for offline analysis
evmax archive export --sector soccer --since 2026-03-01 --format jsonl
evmax archive export --sector tennis --source pinnacle --format csv --out /tmp/tennis.csv
```

### Categories and Modes

```bash
# Inspect the betting-category registry (data/categories.yaml)
evmax categories list                       # all categories + base mode
evmax categories list --mode shadow         # filter by mode
evmax categories list --effective           # show effective mode after overrides
evmax categories show wnba                   # detail for one category
evmax categories modes                       # effective mode per category
evmax categories validate                    # fail non-zero on registry inconsistency

# Shadow-mode validation (promote to live once validated)
evmax cleanup shadow show --days 7
evmax cleanup shadow metrics --days 30 --category wnba
evmax cleanup shadow promote wnba            # flip shadow → live in the YAML

# Runtime mode overrides on a scan (flag > env var > YAML base)
evmax agents scan --shadow nfl_props --live wnba --disabled nhl
```

### Simulation, Reporting, and Projections

```bash
# Monte Carlo simulation + paper-bet tracking
evmax sim list --status open
evmax sim resolve
evmax report
evmax report bankroll

# Multi-portfolio management and comparison
evmax portfolio list

# Standalone point projections (not the EV pipeline) — backs the /nba-proj skill
evmax project slate --sector nba --log
evmax project resolve --date 2026-03-20 --sector nba
```

---

## Configuration

All settings live in `.env` (or environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `KALSHI_API_KEY_ID` | optional | Kalshi API key ID — needed for WebSocket price refresh + trading, not for scanning |
| `KALSHI_PRIVATE_KEY_PATH` | optional | Path to RSA .pem key file (same scope as above) |
| `EV_THRESHOLD` | `0.02` | Minimum EV to report (2%) |
| `MAX_KELLY_FRACTION` | `0.05` | Hard cap per bet (5% of bankroll) |
| `KALSHI_WS_ENABLED` | `true` | WebSocket real-time prices; set `false` for REST-only |
| `KALSHI_WS_SNAPSHOT_TIMEOUT` | `5.0` | Seconds to wait per ticker snapshot before REST fallback |
| `SLACK_WEBHOOK_URL` | — | Post EV alerts to Slack |
| `DISCORD_WEBHOOK_URL` | — | Post EV alerts to Discord |
| `NOTIFICATION_MIN_EV_PCT` | `5.0` | Min EV% to trigger a notification |

**Agent coordinator parameters** (passed as CLI flags or programmatically):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bankroll` | `250.0` | Current bankroll in USD |
| `--kelly` | `0.5` | Kelly multiplier (0.5 = half Kelly) |
| `--sectors` | all | Comma-separated sector list |
| `--min-ev` | `0.02` | Minimum EV filter |
| `--top` | `30` | Max plays to display |
| `--no-models` | off | Skip statistical models |
| `--no-injuries` | off | Skip injury adjustments |
| `--loop` | off | Run continuously with adaptive scan intervals |

---

## Sectors and Market Types

The full catalog — every bettable category, its models, mode, resolver, and status — lives in one registry: [`data/categories.yaml`](data/categories.yaml). Read it at runtime via `evmax.categories.all_categories()` or `evmax categories list`. The table below is a snapshot for orientation; the YAML is authoritative.

All sectors draw sharp lines from the keyless **Pinnacle guest API** (`guest.api.arcadia.pinnacle.com/0.1`). 15 categories ship today:

| Category | Models | Market Types | Resolver | Mode |
|----------|--------|--------------|----------|------|
| `nba` | Efficiency + PossessionSim + ShotQuality + Matchup + Elo + Form | moneyline, spread, total | espn_scoreboard | `live` |
| `nfl` | NFL Efficiency + NFL QB Elo + Elo + Form | moneyline, spread, total | espn_scoreboard | `live` |
| `ncaab` | Elo + Form + Poisson | moneyline, spread, total | espn_scoreboard | `live` |
| `ncaaw` | Form | moneyline, spread, total | espn_scoreboard | `live` |
| `soccer` | Poisson + xG + Elo + Form | moneyline, total | espn_scoreboard | `live` |
| `worldcup` | Poisson + xG + Elo + Form (national-team namespaces) | moneyline | espn_scoreboard (`fifa.world`) | `shadow` |
| `tennis` | Surface Elo + Serve/Return + Form + Advanced + H2H + Ranking Trend | moneyline | kalshi_settlement | `live` |
| `baseball` | Pitcher + Elo + Form (probables via MLB Stats API) | moneyline, spread (`total` disabled) | espn_scoreboard | `shadow` |
| `wnba` | WNBA Efficiency + WNBA PossessionSim + Elo | moneyline, spread (`total` → shadow) | espn_scoreboard | `live` |
| `nhl` | NHL xG (MoneyPuck) + Form | moneyline, spread, total | espn_scoreboard | `shadow` |
| `lol` | sharp-only | moneyline, map_handicap | bo3gg | `shadow` |
| `cs2` | sharp-only | moneyline, map_handicap | bo3gg | `shadow` |
| `nba_props` | NBA Props Cache | player_prop | espn_boxscore | `shadow` |
| `nfl_props` | NFL Props Cache (QB only v1) | player_prop | espn_boxscore | `shadow` (blocked) |
| `baseball_props` | Baseball Props Model (K/Outs/TB/HR anchored; Hits/H+R+RBI/RBI model-priced) | player_prop | espn_boxscore | `shadow` (wip) |

> Injury data (ESPN) is applied to NBA / NFL / NCAAB / NCAAW / soccer / worldcup / baseball / WNBA / NHL. `valorant`, `ufc`, and `f1` sector handlers exist in the registry as **latent** sectors but have no Kalshi product, so they're absent from `SECTOR_SERIES_MAP` and cannot be bet today.

### Modes

Every category runs in one of three modes (`evmax.modes.get_mode`):

| Mode | What happens |
|------|--------------|
| `live` | EVGaps persisted with `mode='live'`, Kelly sized against bankroll. |
| `shadow` | EVGaps + pre-game YES ask persisted with `mode='shadow'`; **bankroll untouched** (Kelly skipped). Used for live-wiring validation before promotion. |
| `disabled` | Scanner skips persistence entirely (gap still shows in the session's in-memory CLI output). |

**Override precedence (highest wins):** runtime CLI flag (`--shadow`/`--live`/`--disabled`) > env var `EVMAX_CATEGORY_MODES` > YAML base. Per-market-type refinements `shadow_market_types` and `disabled_market_types` narrow a category to specific market types (e.g. baseball `total` is disabled; WNBA `total` stays shadow). Promote a category with `evmax cleanup shadow promote <category>` once validation passes.

**Kalshi series tickers per sector** (from `SECTOR_SERIES_MAP` in `kalshi.py`):

| Sector | Kalshi Series |
|--------|--------------|
| NBA | `KXNBAGAME`, `KXNBASPREAD`, `KXNBATOTAL` |
| NFL | `KXNFLGAME`, `KXNFLTOTAL` |
| NCAAB | `KXNCAABGAME`, `KXNCAAMBGAME`, `KXNCAAMBSPREAD`, `KXNCAAMBTOTAL` |
| NCAAW | `KXNCAAWBGAME`, `KXNCAAWBSPREAD`, `KXNCAAWBTOTAL` |
| Baseball | `KXMLBGAME`, `KXMLBSPREAD`, `KXMLBTOTAL` |
| NHL | `KXNHLGAME`, `KXNHLSPREAD`, `KXNHLTOTAL` |
| WNBA | `KXWNBAGAME`, `KXWNBASPREAD`, `KXWNBATOTAL` |
| Soccer | `KXEPLGAME`, `KXUCLGAME`, `KXMLSGAME`, `KXLALIGAGAME`, `KXBUNDESLIGAGAME`, `KXSERIEAGAME`, `KXLIGUE1GAME`, `KXUELGAME` |
| World Cup | `KXWCGAME` |
| Tennis | `KXATPMATCH`, `KXWTAMATCH` |
| LoL | `KXLOLGAME` |
| CS2 | `KXCS2GAME`, `KXCS2GAMES` |
| NBA Props | `KXNBAPTS`, `KXNBAREB`, `KXNBAAST`, `KXNBA3PT`, `KXNBASTL`, `KXNBABLK`, `KXNBAPRA` |
| NFL Props | `KXNFLPASSYDS`, `KXNFLRSHYDS`, `KXNFLRECYDS`, `KXNFLANYTD`, `KXNFLPASSTDS`, `KXNFLREC` |

**Market types supported:**

- **Moneyline** — team A or team B wins
- **Spread** — team A wins by more/less than X points (SpreadDistributionModel)
- **Three-way (soccer / World Cup)** — home / draw / away with three-way devig
- **Totals** — over/under point total
- **Player props** — over/under a player stat line (NBA / NFL props)

---

## Performance Notes

### Parallel API Fetching

Data sources run concurrently per sector using `asyncio.gather`:
- **Kalshi**: All series-prefix requests fired in parallel, throttled by a single token-bucket `AsyncLimiter(10, 1.0)` (10 req/s) to respect Kalshi's rate limits and prevent 429s.
- **Pinnacle**: All `(sport_key × market_type)` combinations fetched simultaneously.
- **ESPN injury agent**: Shares a single `httpx.AsyncClient` across all parallel endpoint fetches to reuse TCP/TLS connections.

### Scan Bankroll Consistency

The bankroll used during `evmax agents scan` is stored in each prediction row (`bankroll_used`). When you run `evmax agents verify` or `evmax agents pick` without specifying `--bankroll`, the stored value is used automatically — ensuring Kelly stakes are identical between scan and verification.

---

## Running Tests

```bash
uv run pytest tests/ -v
```
