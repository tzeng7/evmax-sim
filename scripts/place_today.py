"""Place the 4 best bets for 2/23-25 games.

Target bets:
  1. LEVOLY-TIE  (2/24 UCL)  - Leverkusen vs Olympiacos draw, 2.3% EV
  2. CHACHI-CHA  (2/24 NBA)  - Charlotte Hornets, 2.2% EV
  3. PSGASM-ASM  (2/25 UCL)  - Monaco upset PSG, 2.1% EV
  4. RMABEN-RMA  (2/25 UCL)  - Real Madrid vs Benfica, 1.9% EV ($76k volume)
"""
import asyncio
import structlog
from datetime import date

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))

from evmax.pipeline.runner import PipelineRunner

TARGET_MARKETS = {
    "kalshi:KXUCLGAME-26FEB24LEVOLY-TIE",
    "kalshi:KXNBAGAME-26FEB24CHACHI-CHA",
    "kalshi:KXUCLGAME-26FEB25PSGASM-ASM",
    "kalshi:KXUCLGAME-26FEB25RMABEN-RMA",
}


async def main() -> None:
    # Use lower EV threshold to capture the 1.9% Real Madrid play
    from evmax.settings import get_settings
    settings = get_settings()
    original_threshold = settings.ev_threshold
    settings.ev_threshold = 0.018  # type: ignore[assignment]

    runner = PipelineRunner(sectors=["soccer", "nba"], simulate=True)
    runner._settings.ev_threshold = 0.018
    result = await runner.run_cycle()

    print(f"\nCycle: {result.markets_fetched} fetched, {result.markets_matched} matched, {result.ev_bets_found} EV bets")

    placed = [b for b in result.ev_bets if b.market_id in TARGET_MARKETS]
    print(f"\n=== Target bets placed: {len(placed)} ===")
    for bet in placed:
        d = bet.event_date.strftime("%m/%d") if bet.event_date else "?"
        print(
            f"  [{d}] {bet.market_id.split(':',1)[-1]:35s} "
            f"EV={bet.ev:+.1%}  prob={bet.true_prob:.3f}  "
            f"kelly={bet.suggested_units*100:.3f}%"
        )

    missed = TARGET_MARKETS - {b.market_id for b in result.ev_bets}
    if missed:
        print(f"\n  (not found in this cycle: {', '.join(m.split(':',1)[-1] for m in missed)})")


if __name__ == "__main__":
    asyncio.run(main())
