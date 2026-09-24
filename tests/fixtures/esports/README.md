# Esports (lol / cs2) fixtures

Captured live **2026-09-24 ~04:30 UTC** (read-only) from the production APIs.
Values are real; the rows are only trimmed to the fields the parsers read.

| File | Source | Cut |
|---|---|---|
| `kxlolgame_short_title_markets.json` | Kalshi `GET /markets?series_ticker=KXLOLGAME&status=open` | 7 matches (14 markets) of 23 |
| `kxlolgame_short_title_events.json` | Kalshi `GET /events?series_ticker=KXLOLGAME&status=open` | the same 7 events |
| `kxcs2game_short_title_markets.json` | Kalshi `GET /markets?series_ticker=KXCS2GAME&status=open` | 10 matches (20 markets) of 90 |
| `kxcs2game_short_title_events.json` | Kalshi `GET /events?series_ticker=KXCS2GAME&status=open` | the same 10 events |
| `pinnacle_esports_odds.json` | `PinnacleGuestClient.get_odds("lol" / "cs2")` (`SharpOdds` dumps) | all 17 records Pinnacle listed (9 lol + 8 cs2) |

Kalshi base URL: `https://api.elections.kalshi.com/trade-api/v2` (no API key
needed for these GETs). Market rows keep the live listing order, so the two
siblings of an event are NOT always listed away-first.

These are the SHORT title format Kalshi switched esports to in 2026-08
("FURIA Esports wins", with no opponent). The event title
("FURIA Esports vs. RED Canids") and each market's `yes_sub_title` carry the
full team names. The events were picked for these edge cases:

- Accents: "Movistar KOI Fénix" (Pinnacle "Movistar KOI Fenix").
- Noise words: "FURIA Esports", "EDward Gaming Youth Team",
  "Bilibili Gaming Junior", "Team Nemesis", "Bounty Hunters Esports".
- Kalshi-only spellings that need aliases: "Just_Players" and "Keyd"
  (Pinnacle "Just Players" and "Keyd Stars"), and "Overtake".
- Names that fooled title-keyword market-type inference:
  - "Overtake" contains "over" (read as a total).
  - "THUNDER dOWNUNDER" contains "under" (total).
  - "Ground Zero" contains "round" (map handicap).
  - "ex-Zero Tenacity" has a hyphen (spread).
- Ticker codes with digits: C9, K27, 9G, 3DMAX and 4M, including
  `...26SEP2407154MINF` (time 0715, then 4M + INF).
- A late-ET start that is the next UTC day: `...26SEP252100OTLAG`
  (9:00 PM EDT = 01:00Z).

Used by `tests/test_esports_sector.py`.

Refresh: re-run the four Kalshi GETs above plus `PinnacleGuestClient.get_odds`.
Then re-trim to the same fields. Pinnacle lists only the next ~1–2 days of
matches, so pick Kalshi events that Pinnacle also lists.
