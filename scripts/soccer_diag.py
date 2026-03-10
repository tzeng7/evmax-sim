"""Deep diagnostic: show Kalshi keys vs Pinnacle keys for soccer."""
import asyncio
import structlog

# Show INFO so we can see fetch details
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))

from evmax.clients.kalshi import KalshiClient
from evmax.clients.pinnacle import PinnacleClient
from evmax.matching.normalizer import NameNormalizer


async def main() -> None:
    norm = NameNormalizer("soccer")

    # Fetch Kalshi soccer markets
    async with KalshiClient() as kalshi:
        markets = await kalshi.get_markets("soccer")

    print(f"\n=== Kalshi soccer markets: {len(markets)} ===")
    # Show canonical keys
    from evmax.matching.normalizer import NameNormalizer as N
    n = N("soccer")
    keys = set()
    for m in markets:
        if m.event_date and m.team_home and m.team_away:
            key = n.normalize_event_key(m.team_home, m.team_away, str(m.event_date)[:10], "soccer")
            keys.add(key)
    # Print a sample of keys (first 20)
    for k in sorted(keys)[:30]:
        print(f"  KALSHI: {k}")

    # Fetch Pinnacle soccer odds
    async with PinnacleClient() as pinnacle:
        sharp_list = await pinnacle.get_odds("soccer")

    print(f"\n=== Pinnacle soccer events: {len(sharp_list)} ===")
    for s in sharp_list:
        print(f"  PINNACLE: {s.event_id}  (raw_a={s.outcome_a_label!r}, raw_b={s.outcome_b_label!r})")


if __name__ == "__main__":
    asyncio.run(main())
