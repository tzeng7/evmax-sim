"""Quick soccer scan diagnostic."""
import asyncio
import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))  # WARNING+

from evmax.pipeline.runner import PipelineRunner


async def main() -> None:
    runner = PipelineRunner(sectors=["soccer"], simulate=False)
    result = await runner.run_cycle()
    print(f"Markets fetched : {result.markets_fetched}")
    print(f"Markets matched : {result.markets_matched}")
    print(f"EV bets found   : {result.ev_bets_found}")
    for bet in result.ev_bets:
        print(
            f"  {bet.market_id} | {bet.outcome} "
            f"| true_prob={bet.true_prob:.3f} | EV={bet.ev:.1%} | kelly={bet.suggested_units:.4f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
