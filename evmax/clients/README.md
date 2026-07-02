# evmax/clients

API clients for external data sources.

## Active clients (used in the live scan pipeline)

### `kalshi.py` — `KalshiClient`
- RSA-signed authentication against the Kalshi REST API
- `get_markets(series_tickers)` — fetches open markets for a list of series tickers
- `get_orderbook(market_ticker)` — WebSocket orderbook snapshot for live price verification
- Rate limiting: `AsyncLimiter(10, 1.0)` from `aiolimiter` (token bucket, 10 req/s)
- Used by: `KalshiOddsAgent`

### `esports_pinnacle.py` — `PinnacleGuestClient`
**Despite the filename, this handles ALL sectors — not just esports.**
The file was originally written for esports but was extended to cover every sport.
- No credentials required — uses Pinnacle's public guest API at `guest.api.arcadia.pinnacle.com/0.1`
- `get_odds(sector)` — fetches moneyline, spread, and total odds for any sector
- Sport/league IDs are hardcoded; see the module docstring for the full mapping
- 3-way soccer markets are devigged with `devig_three_way()`
- Used by: `SharpOddsAgent` (scan pipeline) and `evmax project slate` (standalone projections)

### `mlb_statsapi.py` — `MLBStatsClient`
- Official MLB Stats API (`statsapi.mlb.com`) — free, no credentials
- `probable_starters(date)` — today's probable starters keyed by **stable team id**
  (`MLB_TEAM_NICKNAME`), so multi-word nicknames (Red Sox / White Sox / Blue Jays)
  can't drop out of name matching the way they did against the ESPN scoreboard
- Primary live source for `PitcherModelAgent`; ESPN scoreboard is the fallback
- Module-level `fetch_probable_starters()` adds a 30-min cache and degrades to `{}`
  on error so the agent falls back cleanly
- Structured to add lineups / bullpen (the same schedule endpoint hydrates both)

## Seed-time clients (not in the live scan pipeline)

### `tennisabstract.py` — Tennis Abstract scrapers
**Why it exists:** Jeff Sackmann's `tennis_atp` / `tennis_wta` GitHub data repos went offline in 2026,
killing the match-CSV path for the entire tennis model stack. Tennis Abstract (his own site) is still up
and is the replacement source. Three feeds:
- **Elo leaderboards** — `fetch_elo_ratings` / `fetch_elo_page` → `PlayerElo` (overall + hElo/cElo/gElo),
  parsed from `tennisabstract.com/reports/{tour}_elo_ratings.html`. Seeds surface Elo.
- **`matchmx`** — `fetch_matchmx(tour)` → per-match dicts (svpt/1stWon/2ndWon/bpSaved/bpFaced + opponent's,
  surface, ranks, minutes), extracted from the `leadersource*.js` files behind `leaders.cgi`. A drop-in
  for the dead Sackmann match CSVs; unions the top-50/51-100/challenger ranking segments. Seeds
  serve/return, advanced, form, h2h, and (since 2026-06-27) ranking_trend — each match's
  `winner_rank`/`loser_rank` becomes a dated rank snapshot via `aggregate_ranking_history`.
- **Winners/errors** — `fetch_winners_errors(tour)` → `{player: {winners, unforced}}` from
  `winners_errors_leaders_{men,women}_last52.html`. The advanced model's UE feature (sparse coverage).

Used by `scripts/seed_tennis_abstract_elo.py` + `scripts/seed_tennis_models.py`; **not** called by any
live agent. Pure-stdlib HTML/JS-array parsing — no extra deps.

## Prop data / diagnostics layers

### `nba_props_cache.py`
Daily NBA player game-log cache (`data/nba_props_cache.json`) from stats.nba.com with ESPN
fallback. **Diagnostics only since 2026-05-10 (`edb3d7b`)** — the scan reads sample-size +
minutes-volatility metadata via `compute_prop_diagnostics`; prop P(over) comes from Pinnacle
anchor pricing in `evmax/ev/prop_pricing.py`.

### `nfl_props_cache.py`
NFL twin of the above: loads/refreshes the nflverse parquet history
(`data/backtest/nfl_props/*.parquet`), resolves next opponents from the schedule, exposes
`compute_nfl_prop_diagnostics`. The L15-style prob model was removed in the same commit.

### `baseball_props_cache.py`
MLB player-prop data layer (season rates, team rates, box scores, game logs) feeding the
projection model (`evmax/models_ml/baseball_props.py`) and prop resolution/backtest.

### `nba_stats.py`
stats.nba.com (`nba_api`) helpers: `_find_player_id`, game-log fetch, `STAT_COL` — used by the
NBA props diagnostics cache and by prop resolution in `cleanup/resolver.py`. (The legacy L15
prop probability model that lived here was caller-less after the 2026-05-10 anchor-pricing
switch and was removed in the 2026-07-01 drift-audit follow-up.)

## Shared helpers

### `time_util.py`
Date/time helpers shared across market clients — normalizes Pinnacle's UTC `commence_time`
so late-evening US games don't land on the next calendar day relative to Kalshi tickers.

### `base.py` — `BaseAPIClient`
Thin async HTTP base class using `httpx.AsyncClient`. Handles retries and shared headers.
All active clients inherit from this.
