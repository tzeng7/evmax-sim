"""Scan all sectors and show only games on today (2/23) and tomorrow (2/24)."""
import asyncio
import structlog
from datetime import date, timezone

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

from evmax.pipeline.runner import PipelineRunner
from evmax.models.ev_bet import EVBet

TARGET_DATES = {date(2026, 2, 23), date(2026, 2, 24)}


async def main() -> None:
    runner = PipelineRunner(sectors=["soccer", "nba", "ncaab", "nfl", "lol", "cs2"], simulate=False)
    result = await runner.run_cycle()

    print(f"\nTotal: {result.markets_fetched} fetched, {result.markets_matched} matched, {result.ev_bets_found} EV bets")

    # Filter to today/tomorrow
    today_bets: list[EVBet] = []
    for bet in result.ev_bets:
        if bet.event_date and bet.event_date.date() in TARGET_DATES:
            today_bets.append(bet)

    print(f"\n=== Games on 2/23-2/24 ({len(today_bets)} +EV bets) ===")
    # Sort by EV descending
    today_bets.sort(key=lambda b: b.ev, reverse=True)
    for bet in today_bets:
        d = bet.event_date.strftime("%m/%d") if bet.event_date else "?"
        print(
            f"  [{d}] {bet.sector.upper():6s} {bet.market_id.split(':',1)[-1]:35s} "
            f"| EV={bet.ev:.1%} | true_prob={bet.true_prob:.3f} "
            f"| kelly={bet.suggested_units*100:.3f}% | vol=${bet.volume_usd:,.0f}"
        )

    print(f"\n=== All other +EV bets (upcoming) ===")
    other = sorted([b for b in result.ev_bets if b not in today_bets], key=lambda b: (b.event_date or date.min, -b.ev))
    for bet in other[:10]:
        d = bet.event_date.strftime("%m/%d") if bet.event_date else "?"
        print(
            f"  [{d}] {bet.sector.upper():6s} {bet.market_id.split(':',1)[-1]:35s} "
            f"| EV={bet.ev:.1%} | vol=${bet.volume_usd:,.0f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
