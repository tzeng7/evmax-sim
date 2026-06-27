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

### `tennisabstract.py` — Tennis Abstract Elo leaderboards
- Fetches + parses the public ATP/WTA Elo reports at `tennisabstract.com/reports/{tour}_elo_ratings.html`
  (`fetch_elo_ratings` / `fetch_elo_page` → `PlayerElo` rows: overall + hElo/cElo/gElo)
- **Why it exists:** Jeff Sackmann's `tennis_atp` / `tennis_wta` GitHub data repos went offline in 2026,
  so the match-CSV path for surface Elo died. Tennis Abstract (his own site) still publishes weekly
  pre-computed surface Elo on the standard 400-pt scale.
- Used by `scripts/seed_tennis_abstract_elo.py` (seeds `TennisModelAgent` surface Elo); **not** called by
  any live agent. Pure-stdlib HTML parsing — no extra deps.

## `base.py` — `BaseAPIClient`
Thin async HTTP base class using `httpx.AsyncClient`. Handles retries and shared headers.
All active clients inherit from this.
