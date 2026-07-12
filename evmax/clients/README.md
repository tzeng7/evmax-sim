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
  serve/return, advanced, form, h2h.
- **Winners/errors** — `fetch_winners_errors(tour)` → `{player: {winners, unforced}}` from
  `winners_errors_leaders_{men,women}_last52.html`. The advanced model's UE feature (sparse coverage).

Used by `scripts/seed_tennis_abstract_elo.py` + `scripts/seed_tennis_models.py`; **not** called by any
live agent. Pure-stdlib HTML/JS-array parsing — no extra deps.

## `base.py` — `BaseAPIClient`
Thin async HTTP base class using `httpx.AsyncClient`. Handles retries and shared headers.
All active clients inherit from this.

### `ufc_espn.py` — `UFCESPNClient`
**Why ESPN and not ufcstats.com:** ufcstats.com fronts a JavaScript proof-of-work anti-bot
interstitial (verified 2026-07-11); an automated client can't fetch it without defeating bot
detection, which we don't do. ESPN's public MMA JSON API is the repo-standard fallback family
(same source as `scripts/seed_espn.py`) and carries full UFC cards 2010→present. Three feeds:
- **Monthly scoreboards** — `fetch_month("YYYYMM")` → `FightResult` rows (winner flags,
  finish round/clock, weight class, athlete ids).
- **Method of victory** — `fetch_method(event_id, comp_id)` → the core status object's
  `result` ("submission" / "ko/tko" / "decision" / draw / NC), normalized by `normalize_method`.
- **Fighter bios** — `fetch_athlete(id)` → `FighterBio` (DOB / height / reach / stance).

Known gap: NO per-fight strike/takedown counts, so per-minute volume stats (SLpM/SApM/TD avg)
can't be built from this source. All fetches disk-cache under `data/cache/ufc_espn/` (gitignored).
Used by `scripts/fetch_ufc_history.py` → committed dataset `data/backtest/ufc/*.csv`; **not**
called by any live agent.
