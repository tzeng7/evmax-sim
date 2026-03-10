"""Deep dive: all UCL 2/24-25 markets with their EV regardless of threshold."""
import asyncio
import structlog
from datetime import date

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

from evmax.clients.kalshi import KalshiClient
from evmax.clients.pinnacle import PinnacleClient
from evmax.matching.engine import MatchingEngine
from evmax.ev.calculator import calculate_ev, evaluate_market
from evmax.models_ml.sharp_only import SharpBooksModel
from evmax.matching.normalizer import NameNormalizer
from evmax.sectors.registry import get_handler

TARGET_DATES = {date(2026, 2, 24), date(2026, 2, 25)}


async def main() -> None:
    norm = NameNormalizer("soccer")
    engine = MatchingEngine()
    model = SharpBooksModel()
    handler = get_handler("soccer")

    async with KalshiClient() as kalshi:
        markets = await kalshi.get_markets("soccer")

    async with PinnacleClient() as pinnacle:
        sharp_list = await pinnacle.get_odds("soccer")

    # Filter Kalshi to UCL 2/24-25
    ucl_markets = [
        m for m in markets
        if "UCL" in m.ticker.upper() and m.event_date and m.event_date.date() in TARGET_DATES
    ]
    print(f"UCL markets on 2/24-25: {len(ucl_markets)}")

    # Match and compute EV for ALL sides (no threshold)
    matched = engine.match_all(ucl_markets, sharp_list)
    print(f"Matched: {len(matched)}")

    results = []
    for market, sharp_odds, conf in matched:
        prediction = await model.predict(market, sharp_odds)
        if not prediction:
            continue

        blended = sharp_odds.model_copy(update={
            "true_prob_a": prediction.true_prob_a,
            "true_prob_b": prediction.true_prob_b,
        })

        # Apply YES team alignment
        if market.yes_team is not None:
            if market.yes_team in ("tie", "draw", "x"):
                if blended.true_prob_draw is None:
                    continue
                blended = blended.model_copy(update={"true_prob_a": blended.true_prob_draw})
            else:
                norm2 = NameNormalizer("soccer")
                norm_b = norm2.normalize(sharp_odds.outcome_b_label)
                if market.yes_team == norm_b:
                    blended = blended.model_copy(update={
                        "true_prob_a": blended.true_prob_b,
                        "true_prob_b": blended.true_prob_a,
                    })

        ev, _ = calculate_ev(market.yes_price, blended.true_prob_a)
        results.append((ev, market, blended, sharp_odds, conf))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'EV%':>6}  {'PM':>5}  {'True':>5}  {'Payout':>6}  {'Vol':>8}  {'Yes Team':>12}  Market")
    for ev, mkt, blended, sharp, conf in results:
        d = mkt.event_date.strftime("%m/%d") if mkt.event_date else "?"
        print(
            f"  {ev:+.1%}  {mkt.yes_price:.3f}  {blended.true_prob_a:.3f}  "
            f"{1/mkt.yes_price:.2f}x  ${mkt.volume_usd:>7,.0f}  "
            f"{str(mkt.yes_team or '?'):>12}  "
            f"[{d}/{conf:.0f}] {mkt.ticker}"
        )
        print(f"         sharp_event={sharp.event_id}")


if __name__ == "__main__":
    asyncio.run(main())
