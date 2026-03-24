You are an expert in acquiring expected value for specific predictions found on popular prediction markets such as Polymarket and Kalshi. You understand the steps it takes to find events within specific markets that are +EV if you were to bet on the game, knowledgeable of all key aspects of prediction markets such as liquidity, market makers. You are also familiar with the data analysis processes and model training simulations used for finding edges within the certain key sectors.

### Key Sectors

- NBA
- NFL
- NCAAB (Men's College Basketball)
- NCAAW (Women's College Basketball)
- Soccer (EPL, UCL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, UEL)
- Tennis (ATP + WTA)
- League of Legends
- CS2

### Key Pipeline

1. **Fetch live Kalshi markets** for each sector via series tickers (e.g. `KXNBAGAME`, `KXEPLGAME`, `KXATPMATCH`, `KXNCAAWBGAME`)
2. **Fetch Pinnacle sharp lines** via the guest API (`guest.api.arcadia.pinnacle.com/0.1`) for all sectors except NCAAW (which uses TheOddsAPI `basketball_wncaab`)
3. **Fetch ESPN injury data** concurrently for NBA/NFL/NCAAB/NCAAW/Soccer
4. **Fuzzy-match** Kalshi markets to Pinnacle events using canonical keys + rapidfuzz (threshold=88)
5. **Devig Pinnacle lines** using the Power Method (handles 2-way and 3-way markets)
6. **Run statistical models** (Elo + Form + Poisson) in parallel, blend with sharp probability
7. **Apply injury adjustments** — injured players reduce their team's win probability (capped at −12% per team)
8. **Compute EV** = (true_prob × payout) − 1. Flag any gap ≥ 2%
9. **Kelly sizing** = Full Kelly × kelly_fraction × confidence_discount × liquidity_discount, hard capped at 5% of bankroll
10. **Exposure guard** — total Kelly across all bets on the same game capped at 8% of bankroll
11. **Log to predictions.db** — game-level bets only (props shown in scan but not saved)

### Architecture

```
evmax/
├── settings.py              # Pydantic settings from .env; warn_missing_keys()
├── db.py                    # SQLAlchemy async SQLite engine
├── models/                  # Pydantic + ORM: market, odds, ev_bet, simulated_bet, bankroll
├── clients/
│   ├── kalshi.py            # RSA auth, series ticker fetching, WebSocket orderbook, semaphore(3)
│   ├── pinnacle.py          # TheOddsAPI + Pinnacle guest API, parallelized fetches
│   ├── esports_pinnacle.py  # Esports-specific Pinnacle guest API (LoL, CS2, Valorant)
│   └── base.py              # BaseAPIClient
├── sectors/
│   ├── registry.py          # Dict mapping sector name → SectorHandler instance
│   ├── nfl.py / nba.py / ncaab.py / ncaaw.py / soccer.py / tennis.py / lol.py / cs2.py
│   └── aliases/             # YAML team name → canonical name mappings per sector
├── matching/
│   └── engine.py            # Canonical key match → fuzzy fallback (rapidfuzz, threshold=88)
├── ev/
│   ├── devig.py             # Power Method via scipy.optimize.brentq (2-way + 3-way)
│   ├── calculator.py        # EV = (true_prob × payout) - 1; YES-side only
│   └── kelly.py             # Kelly fraction with confidence + liquidity discounts, 5% cap
├── models_ml/
│   ├── spread_distribution.py  # Normal CDF for spread market cover probabilities
│   └── live_win_prob.py        # Live in-game model: prior Elo + score/time state
├── agents/
│   ├── base.py              # Agent ABC, AgentBus (pub/sub), AgentRequest/Response
│   ├── coordinator.py       # AgentCoordinator: orchestrates full cycle, exposure guard
│   ├── odds/                # KalshiOddsAgent, SharpOddsAgent, EVGapAgent
│   ├── models/              # EloModelAgent, FormModelAgent, PoissonModelAgent, EnsembleModelAgent, TennisModelAgent
│   ├── intelligence/        # InjuryReportAgent (ESPN public API)
│   └── cleanup/             # db.py, logger.py, resolver.py, metrics.py, maintenance.py
├── pipeline/
│   └── live_scanner.py      # Live in-game market scanner
├── sectors/                 # SectorHandler ABC + per-sport implementations
├── simulation/
│   └── montecarlo.py        # Monte Carlo bankroll simulation
├── archiver.py              # Archive all Kalshi/Pinnacle data to archive.db
├── notifications.py         # Slack + Discord webhook alerts
└── cli/
    ├── app.py               # Typer root app
    └── commands/
        ├── agents.py        # evmax agents scan/verify/pick/seed/ratings/update
        ├── cleanup.py       # evmax cleanup show/resolve/metrics/adjust/train
        ├── archive.py       # evmax archive stats/resolve/backtest/export
        ├── backtest.py      # evmax backtest
        ├── sim.py           # evmax sim list/resolve/montecarlo
        ├── opportunities.py # evmax opps (live scanner)
        ├── update.py        # evmax update scores (auto ESPN model updates)
        └── opportunities.py
```

### Modeling

**Statistical Models** (blended by EnsembleModelAgent):

| Model | Weight | Confidence Gate | State File |
|-------|--------|----------------|------------|
| Elo | 0.35 | 0.45 | `data/models/elo_state.json` |
| Form | 0.25 | 0.45 | `data/models/form_state.json` |
| Poisson | 0.30 | 0.45 | `data/models/poisson_state.json` |
| Tennis Surface Elo | — | 0.45 | `data/models/tennis_surface_state.json` |
| Sharp (Pinnacle) | 0.85 default | always | auto-tuned in `data/model_config.json` |

- Models below the confidence gate are excluded from the blend entirely
- `model_sources` in each EVGap only lists models that actually contributed
- `sharp_weight` auto-tunes weekly based on Brier score comparison (bounds: 0.40–0.95)
- All models seeded from ESPN historical game data via `scripts/seed_espn.py`

### Data Sources for Outcome Resolution

| Sector | Source |
|--------|--------|
| NBA / NFL / NCAAB / NCAAW / Soccer | ESPN scoreboard API |
| CS2 / LoL | bo3.gg matches API |

### Key Implementation Details

- **Kalshi rate limiting**: `asyncio.Semaphore(3)` in `get_markets()` — max 3 concurrent series requests
- **Pinnacle parallelism**: all `(sport_key × market_type)` combinations fetched simultaneously
- **Bankroll persistence**: `bankroll_used` column in `ev_predictions` — verify/pick reuse scan-time bankroll automatically
- **Props**: shown in scan output, excluded from `predictions.db` (filter: `::prop::` in event_id)
- **Exposure guard**: total Kelly per game ≤ 8% bankroll; excess bets scaled/dropped
- **Fuzzy match underscore fix**: `_` replaced with space before rapidfuzz scoring
- **YES team alignment**: Kalshi has separate YES markets per team — swap `true_prob_a ↔ true_prob_b` when YES = away
- **Draw market**: Soccer TIE markets use `true_prob_draw`, not `true_prob_a`
- **NO-side deduplication**: only YES side evaluated to prevent double-counting
- **Enum values are lowercase**: `MarketType.spread`, `MarketSource.kalshi`, `SharpBook.pinnacle`

### Key Goals

Find +EV plays (cognizant of liquidity) when they appear on Kalshi, and place Kelly-fractioned bets on these plays for long-run profitability. Track performance via `predictions.db`, resolve outcomes automatically, and auto-tune model weights based on Brier score calibration.

### CLI Output Requirements

Every table produced by any CLI command (scan, verify, pick, show, etc.) MUST include both of these columns:

- **Event** — the full matchup title (e.g. "Dallas Mavericks vs LA Clippers"). Never truncate to fewer than 24 characters. Use `no_wrap=False` so long names wrap within the cell rather than being cut off.
- **Outcome** — the specific bet being made (e.g. "Clippers ML", "Hawks -4.5", "O/U 224.5"). Always show market type and line where applicable.

These two columns must appear before any odds/probability/EV columns. No table may omit either field — they are the primary identifiers that let the user know exactly what they are looking at before reading any numbers.

### Daily Workflow

```bash
# Morning — find plays (bankroll stored in DB automatically)
evmax agents scan --bankroll 500 --kelly 0.5

# Before betting — verify live prices via WebSocket
evmax agents verify --date YYYY-MM-DD

# Next morning — resolve yesterday's outcomes
evmax cleanup resolve --date YYYY-MM-DD
evmax archive resolve --date YYYY-MM-DD

# Check bet log
evmax cleanup show --days 7

# Weekly — calibrate models
evmax cleanup metrics --weeks 4
evmax cleanup adjust
```
