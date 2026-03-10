"""Debug: trace exactly which Pinnacle event matches each Kalshi market."""
import asyncio
import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

from evmax.clients.kalshi import KalshiClient
from evmax.clients.pinnacle import PinnacleClient
from evmax.matching.engine import MatchingEngine
from evmax.matching.normalizer import NameNormalizer


SECTORS = ["soccer"]


async def main() -> None:
    engine = MatchingEngine()

    for sector in SECTORS:
        async with KalshiClient() as kalshi:
            markets = await kalshi.get_markets(sector)
        async with PinnacleClient() as pinnacle:
            sharp_list = await pinnacle.get_odds(sector)

        print(f"\n=== {sector.upper()}: {len(markets)} markets, {len(sharp_list)} pinnacle events ===")

        norm = NameNormalizer(sector)
        matched = engine.match_all(markets, sharp_list)

        # Print only non-obvious matches (score < 100 or fuzzy)
        for mkt, sharp, conf in matched:
            mkey = engine.build_market_key(mkt)
            print(
                f"  [{conf:5.1f}] kalshi={mkey}"
                f"\n           pinnacle={sharp.event_id}"
                f"\n           raw_a={sharp.outcome_a_label!r} raw_b={sharp.outcome_b_label!r}"
                f"\n           probs: a={sharp.true_prob_a:.3f} b={sharp.true_prob_b:.3f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
