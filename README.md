# evmax — Agent-Based +EV Prediction Market System

evmax uses a multi-agent pipeline to find positive expected value (+EV) opportunities on Kalshi and Polymarket by comparing market prices against sharp sportsbook lines, statistical models, and real-time injury data. The system recommends Kelly-fractioned bet sizes for each opportunity it surfaces.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture: The Agent Pipeline](#architecture-the-agent-pipeline)
3. [Daily Workflow: Finding +EV Plays](#daily-workflow-finding-ev-plays)
4. [Real-Time Price Feed (WebSocket)](#real-time-price-feed-websocket)
5. [Statistical Models](#statistical-models)
6. [Seeding and Updating Models](#seeding-and-updating-models)
7. [Cleanup Agent: Logging, Resolution, and Calibration](#cleanup-agent-logging-resolution-and-calibration)
8. [Player Prop Pipeline](#player-prop-pipeline)
9. [Web Dashboard](#web-dashboard)
10. [Data Archive and Backtest](#data-archive-and-backtest)
11. [Core EV and Kelly Math](#core-ev-and-kelly-math)
12. [CLI Reference](#cli-reference)
13. [Configuration](#configuration)
14. [Sectors and Market Types](#sectors-and-market-types)

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourorg/evmax
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
# Kalshi (RSA key auth)
KALSHI_API_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi.pem

# TheOddsAPI — covers Pinnacle lines for all sectors
THE_ODDS_API_KEY=your-odds-api-key

# Optional
EV_THRESHOLD=0.02          # minimum EV to report (default 2%)
MAX_KELLY_FRACTION=0.05    # hard cap per bet (default 5%)

# Optional — disable WebSocket and force REST-only price fetching
# KALSHI_WS_ENABLED=false
```

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
│  ┌───────────────┐ ┌───────────────┐ ┌─────────────────┐                 │
│  │  EloModel     │ │  FormModel    │ │  PoissonModel   │                 │
│  │  Agent        │ │  Agent        │ │  Agent          │                 │
│  │  (weight 0.35)│ │  (weight 0.25)│ │  (weight 0.30)  │                 │
│  └───────┬───────┘ └───────┬───────┘ └────────┬────────┘                 │
│          └─────────────────┴──────────────────┘                          │
│                    Confidence-weighted blend                               │
│                    + Sharp odds (weight 0.40)                             │
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

Uses a WebSocket connection to Kalshi for real-time orderbook prices — sub-second accuracy vs the ~60s REST API lag. Only shows bets that are still +EV at the live ask.

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
| `evmax opps` live scanner | WebSocket (all matched markets per sector) |
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

The system uses three core statistical model agents (Elo / Form / Poisson), four tennis-specific agents, an MLB pitcher agent, an NBA player-prop agent, and two baseline models. All model outputs are blended by the EnsembleModelAgent. Models below the 0.45 confidence gate are silently dropped from the blend so missing coverage degrades gracefully.

### Elo Model (`EloModelAgent`, weight=0.35)

Elo is a dynamic rating system that updates after every game result.

**How it works:**

Each team starts at 1500 Elo. After each game:
```
K_factor × (actual_result − expected_result)
```
is added to the winner's rating and subtracted from the loser's. A larger K means faster adaptation; a smaller K means more stable, history-weighted ratings.

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

**Confidence levels:**

| Games played | Confidence |
|-------------|-----------|
| 0 games | 0.30 (below blend gate — excluded) |
| 1–4 games | 0.45 |
| 5–14 games | 0.60 |
| 15+ games | 0.80 |

**Soccer draws:** Elo produces a home/away split. Draw probability is estimated from how even the matchup is: `draw_base × (0.5 + 0.5 × closeness)`, then the remaining probability is allocated to home/away proportionally.

**State file:** `data/models/elo_state.json`

---

### Form Model (`FormModelAgent`, weight=0.25)

Tracks each team's recent performance with exponential decay — recent results matter more than old ones.

**How it works:**

Over the last 10 games (configurable window), each game gets a decayed weight:
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

**State file:** `data/models/form_state.json`

---

### Poisson Model (`PoissonModelAgent`, weight=0.30)

Models scoring as a Poisson process — each team has attack and defense strength parameters that combine to predict expected goals/points.

**How it works:**

For soccer/NBA/NFL/NCAAB, each team has:
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

### Ensemble Model (`EnsembleModelAgent`)

Blends all three model agents with the Pinnacle sharp line into one final probability estimate per event.

**Blending:**

1. Each model produces `(true_prob_a, true_prob_b, confidence, weight)`.
2. Models with `confidence < 0.45` are excluded (prevents data-starved models from polluting the blend).
3. Effective weight = `model_weight × model_confidence`.
4. Sharp odds are blended at `sharp_weight` (default 0.85, auto-tuned by Cleanup Agent).
5. Final blend: `prob = sharp_weight × pinnacle_prob + (1 − sharp_weight) × model_avg`.
6. Normalized to sum to 1.0.

```
BlendedPrediction.true_prob_a = Σ(eff_weight_i × prob_a_i) / Σ(eff_weight_i)
```

When no model has enough data (all below confidence gate), the sharp probability is used directly.

The `model_sources` field on each `EVGap` shows which models contributed: `"elo+form+poisson+sharp"` or `"sharp"` (model-only run) or `"spread_model"`.

---

### Sharp Books Model (Baseline)

When no statistical models have enough data (new teams, early season), the devigged Pinnacle line is used as the true probability. This is Phase 1 of the pipeline and is always available.

**Devigging method:** Power Method (industry standard).

Find exponent `k` where `Σ(raw_prob_i ^ k) = 1.0` (solved via Brent's method). Then `true_prob_i = raw_prob_i ^ k`. Correctly asymmetric — removes more vig from underdogs.

---

### Tennis Models

Tennis runs four independent agents alongside the Elo/Form/Poisson core. Each is wired into the ensemble in `agents/coordinator.py` and returns `None` when its store has no coverage for the players in question, so unseeded matchups fall back to the remaining models.

| Agent | Weight | Signal | State file |
|---|---|---|---|
| `TennisModelAgent` (surface Elo) | 0.45 | Surface-specific Elo (hard / clay / grass) seeded from ATP rankings | `data/models/tennis_surface_state.json` |
| `TennisServeReturnAgent` | 0.40 | Logistic on serve-points-won differential, calibrated for bo3 (`k=14`) and bo5 (`k=18`); slam detection from market title | `data/models/tennis_serve_return_state.json` |
| `TennisH2HAgent` | 0.10 | Head-to-head record nudge with Laplace smoothing + sample-size shrinkage; capped at ±18pp from 0.5; requires ≥3 meetings | `data/models/tennis_h2h_state.json` |
| `TennisRankingTrendAgent` | 0.10 | 12-week ranking-momentum log-odds nudge; positive = climbing the rankings; capped at ±0.40 logit | `data/models/tennis_ranking_trend_state.json` |

The serve/return, H2H, and ranking-trend agents are seeded from Jeff Sackmann's public CSVs (`tennis_atp`, `tennis_wta`) plus the Match Charting Project for 2025+ SPW augmentation. See [Seed Tennis Models](#seed-tennis-models) below.

---

### MLB Pitcher Agent (`PitcherAgent`)

Adapts the starting pitcher's recent ERA / WHIP / K-rate into a moneyline edge for MLB game markets. Falls back to `None` for non-baseball markets and when no probable pitcher is set.

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

| Sector | σ (standard deviation) |
|--------|----------------------|
| NBA | 11.5 pts |
| NFL | 14.0 pts |
| NCAAB | 12.5 pts |
| Soccer | 1.9 goals |

1. Infer implied mean `μ` from Pinnacle's posted line and its cover probability.
2. Estimate `P(margin > target_line)` using the normal CDF.
3. Only evaluate Kalshi lines within 1.5 points of Pinnacle's line (accuracy degrades and liquidity thins further out).

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

### Seed Tennis Models

The three new tennis agents (serve/return, H2H, ranking trend) are seeded directly from Jeff Sackmann's public CSV mirrors of the ATP and WTA tours, plus the Match Charting Project for 2025+ coverage where Sackmann's standard match files lag.

```bash
# Default — both tours, 2023→2026, MCP augmentation enabled
uv run python scripts/seed_tennis_models.py

# Skip MCP (Sackmann-only) if you want to keep state minimal
uv run python scripts/seed_tennis_models.py --no-mcp

# Custom slice
uv run python scripts/seed_tennis_models.py --years 2023,2024 --tours atp
```

What gets written:

| Output | Source | Aggregation |
|---|---|---|
| `tennis_serve_return_state.json` | `{tour}_matches_{year}.csv` (winner/loser SPW columns) + `charting-{m,w}-stats-Overview.csv` | Per-player SPW = `(1stWon + 2ndWon) / svpt`; merge rule keeps the entry with more service points |
| `tennis_h2h_state.json` | `{tour}_matches_{year}.csv` (winner_name / loser_name) | Win counts per alphabetically-sorted player pair |
| `tennis_ranking_trend_state.json` | `{tour}_rankings_current.csv` joined to `{tour}_players.csv` | Weekly snapshots `[{date, rank}, ...]` per player |

Years that haven't been published yet (e.g. `atp_matches_2026.csv`) are silently skipped, and the H2H pipeline is left untouched by MCP since the charting overview file has no winner column.

A typical full run produces roughly **1,150 players with serve stats**, **9,100 H2H pairs**, and **4,400 players with ranking history** across both tours.

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

This updates Elo ratings, Form records, and Poisson attack/defense strengths, then saves state.

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

| Sector | Source |
|--------|--------|
| NBA | ESPN scoreboard API |
| NFL | ESPN scoreboard API |
| NCAAB | ESPN scoreboard API |
| Soccer (EPL, La Liga, etc.) | ESPN soccer scoreboards |
| CS2 | bo3.gg matches API |
| Valorant | bo3.gg matches API |
| LoL | bo3.gg matches API |

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
evmax cleanup train --sectors lol,cs2,valorant
```

### Storage

All predictions and outcomes are stored in `data/predictions.db` (SQLite).

| Table | Contents |
|-------|---------|
| `ev_predictions` | One row per +EV gap per scan — includes kalshi price, blended prob, EV%, kelly, sharp_weight used, `bankroll_used` (scan-time bankroll for consistent verify/pick sizing) |
| `ev_outcomes` | One row per resolved market — outcome (1/0), result source, timestamps |

`data/model_config.json` persists `sharp_weight`, Brier score history, and adjustment timestamps.

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

Every scan automatically snapshots **all** fetched Pinnacle odds and Kalshi market prices to `data/archive.db` — not just the 2%+ EV plays. This builds a free historical dataset over time so you never need to pay TheOddsAPI's historical data tier.

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
evmax cleanup train --sectors lol,cs2,valorant
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

### Legacy Pipeline (Original Scanner)

```bash
# Scan once
evmax scan scan --sectors soccer,nba --once

# Continuous scan every 5 minutes
evmax scan scan --sectors nfl --interval 300

# Simulation tracking
evmax sim list --status open
evmax sim resolve
evmax report
evmax report bankroll
```

---

## Configuration

All settings live in `.env` (or environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `KALSHI_API_KEY_ID` | required | Kalshi API key ID |
| `KALSHI_PRIVATE_KEY_PATH` | required | Path to RSA .pem key file |
| `THE_ODDS_API_KEY` | required | TheOddsAPI key (Pinnacle lines) |
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

| Sector | Sharp Source | Models | Injury Data |
|--------|-------------|--------|-------------|
| NBA | Pinnacle guest API (league 487) | Elo + Form + Poisson | ESPN |
| NFL | Pinnacle guest API (league 258) | Elo + Form + Poisson | ESPN |
| NCAAB | Pinnacle guest API (league 493) | Elo + Form + Poisson | ESPN |
| NCAAW | TheOddsAPI (`basketball_wncaab`) | Elo + Form + Poisson | ESPN |
| Soccer | Pinnacle guest API (EPL/UCL/La Liga/Bundesliga/Serie A/Ligue 1) | Elo + Form + Poisson | ESPN |
| Tennis | Pinnacle guest API (ATP/WTA) | Surface Elo + ATP rankings | None |
| LoL | Pinnacle guest API (esports, sport 12) | Elo + Form | None |
| CS2 | Pinnacle guest API (esports, sport 12) | Elo + Form | None |
| Valorant | Pinnacle guest API (esports, sport 12) | Elo + Form | None |

The Pinnacle guest API (`guest.api.arcadia.pinnacle.com/0.1`) requires no credentials and covers all sectors except NCAAW (which uses TheOddsAPI).

**Kalshi series tickers per sector:**

| Sector | Kalshi Series |
|--------|--------------|
| NBA | `KXNBAGAME`, `KXNBASPREAD` |
| NFL | `KXNFLGAME` |
| NCAAB | `KXNCAAMBGAME`, `KXNCAABGAME` |
| NCAAW | `KXNCAAWBGAME`, `KXNCAAWBSPREAD`, `KXNCAAWBTOTAL` |
| Soccer | `KXEPLGAME`, `KXUCLGAME`, `KXMLSGAME`, `KXLALIGAGAME`, `KXBUNDESLIGAGAME`, `KXSERIEAGAME`, `KXLIGUE1GAME`, `KXUELGAME` |
| Tennis | `KXATPMATCH`, `KXWTAMATCH` |
| LoL | `KXLOLGAME`, `KXLOLGAMES` |
| CS2 | `KXCS2GAME`, `KXCS2GAMES`, `KXCSGOGAME` |

**Market types supported:**

- **Moneyline** — team A or team B wins
- **Spread** — team A wins by more/less than X points (SpreadDistributionModel)
- **Three-way (soccer)** — home / draw / away with three-way devig
- **Totals (NCAAW)** — over/under point total

---

## Performance Notes

### Parallel API Fetching

All three data sources run concurrently per sector using `asyncio.gather`:
- **Kalshi**: All series prefix requests fired in parallel, throttled to **3 concurrent requests** via a semaphore to respect Kalshi's rate limits (prevents 429 errors).
- **Pinnacle**: All `(sport_key × market_type)` combinations fetched simultaneously.
- **ESPN injury agent**: Shares a single `httpx.AsyncClient` across all parallel endpoint fetches to reuse TCP/TLS connections.

### Scan Bankroll Consistency

The bankroll used during `evmax agents scan` is stored in each prediction row (`bankroll_used`). When you run `evmax agents verify` or `evmax agents pick` without specifying `--bankroll`, the stored value is used automatically — ensuring Kelly stakes are identical between scan and verification.

---

## Running Tests

```bash
uv run pytest tests/ -v
```
