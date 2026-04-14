# Kalshi Tennis Fixtures

Captured live Kalshi `/markets` and `/events` responses used by:

- `tests/test_tennis_kalshi_fixtures.py` — resolver replay + end-to-end `get_markets()` join via monkey-patched `_get`
- Reference for `evmax/agents/models/tennis_model_agent.py::_resolve_surface` keyword coverage

## Files

| File | Source endpoint | Params |
|---|---|---|
| `atp_markets.json` | `/markets` | `status=open, series_ticker=KXATPMATCH, limit=5` |
| `atp_events.json` | `/events` | `status=open, series_ticker=KXATPMATCH, limit=5` |
| `wta_markets.json` | `/markets` | `status=open, series_ticker=KXWTAMATCH, limit=5` |
| `wta_events.json` | `/events` | `status=open, series_ticker=KXWTAMATCH, limit=5` |

## Provenance

Captured **2026-04-13** from `https://api.elections.kalshi.com/trade-api/v2` (production, unauthenticated GET — the endpoints do not require an API key for read access). Contents reflect the ATP Munich / Barcelona and WTA Rouen / Stuttgart clay-court draws live at capture time.

Key field observed on every event: `product_metadata.competition` is a structured string of the form `"{ATP|WTA} {City}"` (e.g. `"ATP Munich"`, `"WTA Rouen"`). This is the primary signal for `_resolve_surface`.

## Refresh procedure

Fixtures rarely need refreshing — they're used for offline replay of parser and resolver behavior, not to validate live data currency. Refresh only if:

- Kalshi changes their response schema (new fields needed)
- A new edge case tournament must be captured (e.g. the first time Wimbledon appears on Kalshi)

To refresh, from the repo root with `secrets/KALSHI_API_KEY_ID` and `secrets/KALSHI_PRIVATE_KEY.pem` populated:

```bash
KALSHI_API_KEY_ID=$(cat secrets/KALSHI_API_KEY_ID) \
KALSHI_PRIVATE_KEY_PATH=$(pwd)/secrets/KALSHI_PRIVATE_KEY.pem \
python3 -c "
import asyncio, json
from evmax.clients.kalshi import KalshiClient

async def main():
    async with KalshiClient() as c:
        for tour in ['ATP', 'WTA']:
            series = f'KX{tour}MATCH'
            markets = await c._get('/markets', params={'status':'open','series_ticker':series,'limit':5})
            events = await c._get('/events', params={'status':'open','series_ticker':series,'limit':5})
            with open(f'tests/fixtures/kalshi/{tour.lower()}_markets.json','w') as f: json.dump(markets, f, indent=2)
            with open(f'tests/fixtures/kalshi/{tour.lower()}_events.json','w') as f: json.dump(events, f, indent=2)
            print(f'{tour}: {len(markets.get(\"markets\",[]))} markets, {len(events.get(\"events\",[]))} events')

asyncio.run(main())
"
```

No scrubbing needed — Kalshi responses contain no PII, no account data, and no credentials.
