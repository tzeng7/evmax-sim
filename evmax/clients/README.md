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
- Used by: `SharpOddsAgent`

## Legacy (NOT used in the live scan pipeline)

### `pinnacle.py` — `PinnacleClient`
- TheOddsAPI wrapper (`api.the-odds-api.com`)
- Superseded by `PinnacleGuestClient` which gives direct Pinnacle lines without an API key
- `SECTOR_SPORT_KEYS` dict and `get_odds()` method are dead code in the live pipeline
- Still imported by `pipeline/runner.py` (also legacy)

## `base.py` — `BaseAPIClient`
Thin async HTTP base class using `httpx.AsyncClient`. Handles retries and shared headers.
All active clients inherit from this.
