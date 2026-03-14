# evmax — Agent-Based +EV Prediction Market System

evmax uses a multi-agent pipeline to find positive expected value (+EV) opportunities on Kalshi and Polymarket by comparing market prices against sharp sportsbook lines, statistical models, and real-time injury data. The system recommends Kelly-fractioned bet sizes for each opportunity it surfaces.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture: The Agent Pipeline](#architecture-the-agent-pipeline)
3. [Daily Workflow: Finding +EV Plays](#daily-workflow-finding-ev-plays)
4. [Statistical Models](#statistical-models)
5. [Seeding and Updating Models](#seeding-and-updating-models)
6. [Cleanup Agent: Logging, Resolution, and Calibration](#cleanup-agent-logging-resolution-and-calibration)
7. [Core EV and Kelly Math](#core-ev-and-kelly-math)
8. [CLI Reference](#cli-reference)
9. [Configuration](#configuration)
10. [Sectors and Market Types](#sectors-and-market-types)

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourorg/evmax
cd evmax
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

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
┌───────────────────────────────────────────┐               │
│  STEP 2 — MatchingEngine                  │               │
│  Fuzzy-matches Kalshi markets ↔ Pinnacle  │               │
│  events using canonical keys + rapidfuzz  │               │
└───────────────────────┬───────────────────┘               │
                        │ matched pairs                     │
                        ▼                                   │
┌───────────────────────────────────────────────────────────┤
│  STEP 3 — EnsembleModelAgent (models run in parallel)     │
│                                                           │
│  ┌───────────────┐ ┌───────────────┐ ┌─────────────────┐ │
│  │  EloModel     │ │  FormModel    │ │  PoissonModel   │ │
│  │  Agent        │ │  Agent        │ │  Agent          │ │
│  │  (weight 0.35)│ │  (weight 0.25)│ │  (weight 0.30)  │ │
│  └───────┬───────┘ └───────┬───────┘ └────────┬────────┘ │
│          └─────────────────┴──────────────────┘          │
│                    Confidence-weighted blend               │
│                    + Sharp odds (weight 0.40)             │
│                    = BlendedPrediction per event          │
└───────────────────────────┬───────────────────────────────┘
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
================================================================================
  +EV PLAYS  |  Bankroll: $250  |  Kelly: 50%
================================================================================
 1. [STRONG  ] NBA    Celtics              Kalshi=0.62  TrueP=0.714  EV=+15.2%  Kelly=4.82%  Stake=$12.05  [sharp+elo+form]
 2. [GOOD    ] SOCCER Arsenal             Kalshi=0.48  TrueP=0.537  EV=+11.9%  Kelly=3.21%  Stake=$8.03   [sharp]
 3. [MARGINAL] NBA    Lakers -4.5         Kalshi=0.44  TrueP=0.471  EV=+7.1%   Kelly=1.90%  Stake=$4.75   [spread_model]
================================================================================
  Total at risk: $52.43 / $250 (21.0%)
```

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

## Statistical Models

The system uses three statistical model agents plus two baseline models. All model outputs are blended by the EnsembleModelAgent.

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

# End of day — resolve yesterday's outcomes
evmax cleanup resolve --date YYYY-MM-DD

# Check your log
evmax cleanup show --days 7

# Weekly — check calibration
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
| `ev_predictions` | One row per +EV gap per scan — includes kalshi price, blended prob, EV%, kelly, sharp_weight used |
| `ev_outcomes` | One row per resolved market — outcome (1/0), result source, timestamps |

`data/model_config.json` persists `sharp_weight`, Brier score history, and adjustment timestamps.

---

## Core EV and Kelly Math

### Expected Value

```
EV = (true_probability × payout) − 1

payout = 1 / yes_price  (for Kalshi YES contracts priced 0–1)
```

**Example:**
- Kalshi: YES at $0.40 → payout = 2.5×
- Pinnacle devigged true prob = 48%
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
evmax cleanup train --sectors nba,soccer --since 2025-10-01
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
| `KALSHI_ENV` | `production` | `production` or `demo` |

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

---

## Sectors and Market Types

| Sector | Sharp Source | Models | Injury Data |
|--------|-------------|--------|-------------|
| NBA | Pinnacle guest API (league 487) | Elo + Form + Poisson | ESPN |
| NFL | Pinnacle guest API (league 258) | Elo + Form + Poisson | ESPN |
| NCAAB | Pinnacle guest API (league 493) | Elo + Form + Poisson | ESPN |
| Soccer | Pinnacle guest API (EPL/UCL/La Liga/Bundesliga/Serie A/Ligue 1) | Elo + Form + Poisson | ESPN |
| LoL | Pinnacle guest API (esports, sport 12) | Elo + Form | None |
| CS2 | Pinnacle guest API (esports, sport 12) | Elo + Form | None |
| Valorant | Pinnacle guest API (esports, sport 12) | Elo + Form | None |

The Pinnacle guest API (`guest.api.arcadia.pinnacle.com/0.1`) requires no credentials and covers all sectors.

**Market types supported:**

- **Moneyline** — team A or team B wins
- **Spread** — team A wins by more/less than X points (SpreadDistributionModel)
- **Three-way (soccer)** — home / draw / away with three-way devig

---

## Running Tests

```bash
pytest tests/ -v
# or with uv:
uv run pytest tests/ -v
```

90 tests across agents, models, matching, EV calculation, devigging, Kelly sizing, and the agent pipeline.
