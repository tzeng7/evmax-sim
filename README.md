# evmax — Expected Value Maximizer for Prediction Markets

evmax finds positive expected value (+EV) betting opportunities on Kalshi and Polymarket by comparing their market prices against sharp sportsbook lines from Pinnacle. It simulates Kelly-sized paper bets to track long-run performance.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [How Expected Value Is Calculated](#how-expected-value-is-calculated)
3. [Devigging: Removing the Bookmaker's Margin](#devigging-removing-the-bookmakers-margin)
4. [Spread Markets: Normal Distribution Model](#spread-markets-normal-distribution-model)
5. [Kelly Criterion: Bet Sizing](#kelly-criterion-bet-sizing)
6. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
7. [Pre-Game vs. Live Scanning](#pre-game-vs-live-scanning)
8. [Sectors and Market Types](#sectors-and-market-types)
9. [Commands](#commands)
10. [Configuration](#configuration)

---

## Core Concepts

### Expected Value (EV)

Expected value is the long-run average return per dollar wagered. For a binary prediction market:

```
EV = (true_probability × payout) − 1
```

Where:
- **true_probability** — our best estimate of the real probability of the outcome occurring
- **payout** — how many dollars you receive per dollar bet if you win (e.g. 2.5x means you win $1.50 profit on a $1 bet)
- **payout = 1 / market_price** for a binary YES/NO market (e.g. Kalshi price of 0.40 → payout of 2.5x)

**Example:**
- Kalshi prices a team's moneyline YES at $0.40 (40¢ per share = 2.5x payout)
- Pinnacle's devigged true probability for that team is 48%
- EV = 0.48 × 2.50 − 1 = **+0.20 = +20%**

This means for every $1 wagered, you expect to earn $0.20 in profit over the long run. Any EV ≥ 2% is flagged as a betting opportunity.

### Why Kalshi Can Be Mispriced

Kalshi is a prediction market where retail traders set prices through supply and demand. Unlike sportsbooks, Kalshi has:
- **No professional market makers** adjusting lines in real time
- **Retail flow bias** — participants tend to overbet popular teams and underprice underdogs
- **Slower adjustment** to injury news, line movement, and sharp consensus

Pinnacle, by contrast, employs professional traders and accepts sharp action. Their lines are widely considered the most accurate probability estimates in sports betting. By comparing Kalshi prices to Pinnacle's devigged lines, we systematically identify where the retail market is wrong.

---

## How Expected Value Is Calculated

### Step 1: Fetch Kalshi Market Price

For each open Kalshi market, we read the mid-price between the best bid and ask:

```
yes_price = (yes_bid + yes_ask) / 2
```

This is the cost to buy one YES contract, normalized to [0, 1]. The payout if the market resolves YES is:

```
payout = 1 / yes_price
```

### Step 2: Fetch Pinnacle's Sharp Line

Pinnacle's odds are fetched via TheOddsAPI in decimal format. For example, a game line might be:

```
Team A: 1.87 decimal  (implied: 1/1.87 = 53.5%)
Team B: 2.10 decimal  (implied: 1/2.10 = 47.6%)
Sum: 101.1%  ← the bookmaker's overround (1.1% margin)
```

The sum exceeds 100% because Pinnacle builds in a margin. We must remove this margin to get the true probability.

### Step 3: Devig to Get True Probability

See [Devigging](#devigging-removing-the-bookmakers-margin) below.

### Step 4: Align Probability to the YES Side

Kalshi markets are directional — each market has a specific YES team or outcome. We check which outcome the YES side represents and map the devigged probability accordingly:

- `market.yes_team == outcome_a` → use `true_prob_a`
- `market.yes_team == outcome_b` → use `true_prob_b` (swap)
- `market.yes_team in ("tie", "draw", "x")` → use `true_prob_draw` (soccer)

### Step 5: Compute EV and Filter

```
EV = true_prob × payout − 1
```

Only bets with `EV ≥ 2%` are surfaced. This threshold filters noise and accounts for execution friction (bid-ask spread, timing).

---

## Devigging: Removing the Bookmaker's Margin

Raw implied probabilities from decimal odds sum to more than 100% — the excess is the bookmaker's margin (vig). We use the **Power Method** to remove this asymmetrically.

### Why Not Simple Proportional Devigging?

Proportional devigging divides each raw probability by the total overround:

```
true_prob_i = raw_prob_i / sum(raw_probs)
```

This is simple but incorrect — it assumes the vig is distributed equally across all outcomes. In practice, bookmakers charge more vig on underdogs. The Power Method accounts for this asymmetry.

### Power Method (Industry Standard)

The Power Method finds an exponent `k` such that the sum of devigged probabilities equals exactly 1.0:

```
Find k where:  Σ (raw_prob_i ^ k)  =  1.0

Then:  true_prob_i = raw_prob_i ^ k
```

This is solved numerically via Brent's method (`scipy.optimize.brentq`). The exponent `k` is typically close to 1 (e.g., 0.97–0.99 for low-margin books like Pinnacle). Lower `k` means more vig is being removed from underdogs.

**Example with 4% Pinnacle margin:**
```
Raw odds:    Team A: 1.87 (53.5% implied)   Team B: 2.10 (47.6% implied)
Overround:   53.5% + 47.6% = 101.1%  →  1.1% margin

Power Method: find k where 0.535^k + 0.476^k = 1.0
Solved:       k ≈ 0.974

Devigged:     true_prob_A = 0.535^0.974 ≈ 53.9%
              true_prob_B = 0.476^0.974 ≈ 46.1%
              Sum = 100.0% ✓
```

### Three-Way Markets (Soccer)

Soccer markets have three outcomes: Home / Draw / Away. The same Power Method applies with three terms:

```
Find k where:  p_home^k + p_draw^k + p_away^k  =  1.0
```

Each devigged probability sums to 1.0, giving us three true probabilities.

---

## Spread Markets: Normal Distribution Model

For spread markets (e.g., "OKC wins by more than 8.5 points"), we use a **normal distribution of game margins** to estimate the probability of covering any specific line.

### Why a Separate Model?

Pinnacle posts one spread per game (e.g., OKC −7.5). Kalshi lists multiple spread thresholds for the same game (e.g., OKC −2.5, −5.5, −8.5, −11.5...). The normal distribution model lets us translate Pinnacle's single posted line into a probability estimate for any Kalshi line.

### The Model

**Assumption:** The game's final point margin follows a normal distribution:
```
Margin ~ N(μ, σ²)
```

Where `μ` is the implied mean margin and `σ` is the empirical standard deviation of game margins by sport:

| Sport | σ (points) | Rationale |
|-------|-----------|-----------|
| NBA   | 11.5      | High-scoring, tight games common |
| NFL   | 14.0      | Higher variance, blowouts more frequent |
| NCAAB | 12.5     | More variable than NBA |
| Soccer | 1.9     | Goals, not points — much tighter |

**Step 1: Infer the implied mean from Pinnacle's line**

Pinnacle posts OKC −7.5 with a devigged cover probability of, say, 54.2%:
```
P(margin > 7.5) = 54.2%

Using the normal CDF inverse:
  z = Φ⁻¹(1 − 0.542) = Φ⁻¹(0.458) ≈ −0.107

  μ = |pinnacle_line| − z × σ
  μ = 7.5 − (−0.107 × 11.5) = 7.5 + 1.23 = 8.73
```

So Pinnacle's line implies OKC is expected to win by ~8.73 points on average.

**Step 2: Estimate P(covers Kalshi line)**

For Kalshi's OKC −8.5 market (YES = OKC wins by more than 8.5):
```
P(margin > 8.5) = 1 − Φ((8.5 − 8.73) / 11.5)
                = 1 − Φ(−0.020)
                = 1 − 0.492
                = 50.8%
```

If Kalshi is pricing this at 42%, there's a +8.8pp edge → significant +EV.

**Step 3: Underdog direction**

If the YES side is the underdog (e.g., DEN +8.5 wins by more than 8.5), the probability is the lower tail:
```
P(margin < −8.5) = Φ((−8.5 − μ) / σ)
```

### Line Tolerance Filter

Kalshi lists many alternative lines. We only evaluate Kalshi lines within **1.5 points** of Pinnacle's posted line, rejecting deep alternatives where model accuracy degrades and liquidity is thin.

---

## Kelly Criterion: Bet Sizing

The Kelly Criterion determines the optimal fraction of bankroll to wager to maximize long-run growth.

### Full Kelly Formula

```
K_full = (p × b − q) / b

Where:
  p = true probability of winning
  q = 1 − p (probability of losing)
  b = payout − 1  (net odds, e.g. 2.5x payout → b = 1.5)
```

**Example:**
```
true_prob = 0.48,  payout = 2.50x
b = 1.50
K_full = (0.48 × 1.50 − 0.52) / 1.50 = (0.72 − 0.52) / 1.50 = 0.133 = 13.3%
```

Betting 13.3% of bankroll per bet would be optimal in theory — but only if our probability estimate is perfectly accurate.

### Fractional Kelly with Discounts

Because our model is imperfect, we apply three discounts to the full Kelly fraction:

**1. Base fraction (Quarter Kelly = 0.25)**

Reduces the bet to account for model error and estimation uncertainty. Quarter Kelly is a widely-used conservative baseline:
```
k_base = K_full × 0.25
```

**2. Confidence discount**

Scales with the size of the edge. Full confidence is reached at a 20% edge:
```
confidence_discount = min(1.0, edge_pct / 0.20)
```

A 5% EV edge applies 25% confidence discount; a 20%+ edge applies none.

**3. Liquidity discount**

Penalizes wide bid-ask spreads (thin markets where execution is harder):
```
liquidity_discount = max(0.25, 1.0 − spread_pct × 5)
```

A 10% bid-ask spread applies a 50% discount; spread ≥ 15% caps at 75% discount.

**Final Kelly:**
```
K_adjusted = K_full × 0.25 × confidence_discount × liquidity_discount
K_final    = clamp(K_adjusted, min_kelly, max_kelly)
             = clamp(K_adjusted, 1%, 5%)
```

The hard cap of 5% prevents any single bet from risking more than 5% of bankroll regardless of model output.

---

## Full Pipeline Walkthrough

Each scan cycle executes the following steps concurrently:

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Concurrent Data Fetch                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │ Kalshi REST  │  │ Polymarket    │  │ Pinnacle via        │  │
│  │ API v2       │  │ CLOB API      │  │ TheOddsAPI          │  │
│  │ (RSA auth)   │  │ (no auth)     │  │ (h2h + spreads)     │  │
│  └──────┬───────┘  └──────┬────────┘  └──────────┬──────────┘  │
│         └────────────┬────┘                      │             │
│                      ▼                           ▼             │
│              PredictionMarket[]           SharpOdds[]          │
└────────────────────────────┬─────────────────────┬─────────────┘
                             │                     │
┌────────────────────────────▼─────────────────────▼─────────────┐
│  STEP 2: Sector Handler Enrichment                              │
│  Each sector handler normalizes team names, infers market type  │
│  (moneyline / spread / total / series winner), and extracts     │
│  the event date from the ticker.                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 3: Event Matching                                         │
│                                                                 │
│  Canonical key: "{sector}::{YYYY-MM-DD}::{team_a}_vs_{team_b}" │
│  Spread key:    "{canonical}::spread"                           │
│                                                                 │
│  Priority:  1. Exact key match                                  │
│             2. Date-windowed fuzzy match (rapidfuzz ≥ 88)       │
│             3. Manual override JSON                             │
│             (Spread markets: exact-only, no fuzzy)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 4: Staleness Guards                                       │
│                                                                 │
│  • Skip if event ended > 24h ago (Kalshi date is midnight UTC)  │
│  • Skip if game already started (Pinnacle commence_time < now)  │
│    → Pre-game scan only; live scan handled separately           │
│  • Skip if sharp odds fetched > 120s ago (max_odds_age_s)       │
│  • Skip if Kalshi price fetched > 120s ago                      │
│  • Skip if Kalshi price at extreme (< 4% or > 96%)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 5: True Probability Estimation                            │
│                                                                 │
│  Moneyline / ML:                                                │
│    SharpBooksModel → Pinnacle odds → Power Method devig         │
│    → true_prob_a / true_prob_b / true_prob_draw                 │
│    → Align to YES side (swap if YES = away team)                │
│                                                                 │
│  Spread:                                                        │
│    SpreadDistributionModel → infer μ from Pinnacle line         │
│    → P(YES side covers Kalshi line) via normal CDF              │
│    → Skip if line differs from Pinnacle by > 1.5 pts            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 6: EV Calculation                                         │
│                                                                 │
│  EV = true_prob × (1 / kalshi_price) − 1                       │
│  Filter: EV ≥ 2% (ev_threshold)                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 7: Kelly Sizing                                           │
│                                                                 │
│  K = K_full × 0.25 × confidence_discount × liquidity_discount  │
│  Clamp: [1%, 5%] of bankroll                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 8: Live Price Confirmation                                 │
│                                                                 │
│  Re-fetch the Kalshi market's current bid/ask mid-price.        │
│  Recompute EV with the live price. Skip if EV < 2%.            │
│  This catches price drift between scan time and placement.      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 9: Simulation                                             │
│                                                                 │
│  • Save EVBet record to SQLite                                  │
│  • Volume gate: skip if volume < $500 (stale/illiquid)          │
│  • Place SimulatedBet: deduct stake from bankroll               │
│  • Record BankrollSnapshot                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pre-Game vs. Live Scanning

### Pre-Game (`evmax scan scan`)

Uses Pinnacle's **pre-game closing line** as the true probability. This is only valid before the game starts. Once the game tips off, Kalshi prices shift with live score/time-remaining while Pinnacle's line in our data reflects the pre-game consensus — comparing the two would produce meaningless EV signals.

The pipeline automatically filters out any market where `sharp_odds.event_date < now` (i.e., the game has started).

### Live (`python scripts/live_ev.py`)

During in-progress games, we invert the filter and fetch Pinnacle's **current live line** from TheOddsAPI. Pinnacle updates live moneylines continuously based on score and game state. Kalshi can lag behind due to slower retail price discovery.

**Important caveat:** TheOddsAPI may introduce a small delay (5–30 seconds) in propagating Pinnacle's live line. Always verify the current Pinnacle line independently before acting on a live EV signal.

### Combined View (`evmax opps opps`)

The `opps` command runs both scanners concurrently and presents all opportunities in a unified table with per-bet statistical breakdowns. You select which bets to place interactively. A final live price re-confirmation runs at placement time.

---

## Sectors and Market Types

| Sector | Sharp Source | Market Types |
|--------|-------------|--------------|
| NBA    | Pinnacle h2h + spreads | Moneyline, Spread |
| NFL    | Pinnacle h2h + spreads | Moneyline, Spread |
| NCAAB  | Pinnacle h2h | Moneyline |
| Soccer | Pinnacle h2h (3-way) | Moneyline (Home/Draw/Away) |
| LoL    | Pinnacle h2h | Moneyline, Map handicap |
| CS2    | Pinnacle h2h | Moneyline, Map handicap |

Soccer markets are inherently three-way (home / draw / away). Kalshi lists separate YES markets for each outcome. The devig_three_way Power Method is applied, and draw markets use `true_prob_draw` for EV calculation.

---

## Commands

### `evmax opps opps` — Recommended Entry Point

View all pre-game and live +EV opportunities with detailed statistical breakdowns. Interactively select which bets to place.

```bash
evmax opps opps                          # All sectors
evmax opps opps --sectors nba,nfl        # Specific sectors
evmax opps opps --threshold 0.05         # Tighter 5% EV filter
evmax opps opps --no-sim                 # Display only, no placement prompt
```

**Sample output:**
```
  +EV Opportunities  [14:32 UTC]  (threshold: 2%)
╭─────┬──────┬────────┬──────────────────────┬───────┬───────┬───────┬────────╮
│ #   │ Type │ Sector │ Market               │ Price │ True  │  EV%  │ Kelly% │
├─────┼──────┼────────┼──────────────────────┼───────┼───────┼───────┼────────┤
│ 1   │ LIVE │ NBA    │ KXNBAGAME-...-MEM    │ 0.395 │ 0.469 │ 18.6% │  2.10% │
│ 2   │ PRE  │ NBA    │ KXNBASPREAD-...-OKC8 │ 0.430 │ 0.508 │ 18.1% │  2.00% │
╰─────┴──────┴────────┴──────────────────────┴───────┴───────┴───────┴────────╯

╭─ #1 LIVE  NBA  KXNBAGAME-...-MEM ────────────────────────────────────────────╮
│  Sharp source:      Pinnacle via TheOddsAPI (Power Method devigged)          │
│  Sharp probability: 46.9%  (fair line: +113)                                 │
│  Kalshi price:      39.5% implied  (2.53x payout / +153)                     │
│  Probability edge:  +7.4pp  (46.9% − 39.5%)                                  │
│  EV formula:        46.9% × 2.53x − 1  =  +18.6%                            │
│  Kelly sizing:      2.10% of bankroll                                        │
╰───────────────────────────────────────────────────────────────────────────────╯
```

### `evmax scan scan` — Pre-Game Autopilot

Scans and auto-places simulated bets without user interaction. Can run continuously.

```bash
evmax scan scan --sectors nba --once          # Single cycle
evmax scan scan --sectors soccer,nba          # Continuous (every 5 min)
evmax scan scan --sectors nfl --interval 120  # Custom interval
evmax scan scan --no-sim                      # Scan only, no bets
```

### `python scripts/live_ev.py` — Live Standalone Scanner

Lightweight live-only scan. Display-only, no bet placement.

```bash
python scripts/live_ev.py --sectors nba
python scripts/live_ev.py --sectors nba,nfl --threshold 0.05
```

### `evmax sim list` — View Simulated Bets

```bash
evmax sim list                   # All bets
evmax sim list --status open     # Open bets only
evmax sim list --status won      # Won bets
```

### `evmax sim resolve` — Resolve a Bet

```bash
evmax sim resolve --market kalshi:KXNBAGAME-26MAR09PHICLE-PHI --yes-price 1
evmax sim resolve --market kalshi:KXNBAGAME-26MAR09PHICLE-PHI --yes-price 0
```

### `evmax report report` — Performance Summary

```bash
evmax report report              # Overall performance
evmax report report --sector nba # NBA only
evmax report bankroll            # Bankroll history
```

---

## Configuration

All settings are loaded from `.env`:

```env
# API Keys
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY_PATH=./kalshi_private.pem
THE_ODDS_API_KEY=your_odds_api_key

# Bankroll
INITIAL_BANKROLL=1000.0
MIN_BET_USD=10.0

# Kelly
MAX_KELLY_FRACTION=0.05     # Hard cap: 5% of bankroll per bet
MIN_KELLY_FRACTION=0.01     # Floor: 1% of bankroll minimum

# EV
EV_THRESHOLD=0.02           # Minimum EV to flag (2%)

# Odds freshness
MAX_ODDS_AGE_S=120          # Reject odds older than 2 minutes

# Matching
FUZZY_THRESHOLD=88          # rapidfuzz score threshold (0-100)
SPREAD_LINE_TOLERANCE=1.5   # Max pts difference from Pinnacle line

# Liquidity
MIN_VOLUME_USD=500           # Minimum market volume to place a simulated bet
```

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
python scripts/init_db.py
cp .env.example .env
# Add API keys to .env
evmax opps opps --sectors nba
```

---

## Statistical Limitations

- **Phase 1 model:** The sole probability source is Pinnacle's devigged line. No additional statistical models (Elo, regression, neural nets) are incorporated yet. This means the "edge" we find is entirely attributable to Kalshi's retail mispricing relative to Pinnacle — not proprietary model alpha.

- **Spread distribution assumption:** The normal distribution of game margins is an approximation. Real game margins are slightly leptokurtic (fatter tails) and not perfectly symmetric. The σ values used are empirical averages; individual games can vary significantly.

- **Live line lag:** TheOddsAPI introduces a propagation delay for Pinnacle's live lines. During fast-moving game situations (late-game comebacks, key injuries), the displayed live probability may be stale by seconds to minutes.

- **Sample size:** Kelly sizing is theoretically optimal only in the long run (hundreds of bets). Small sample performance will have high variance regardless of true edge.
