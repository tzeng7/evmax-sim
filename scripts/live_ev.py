"""Live EV scanner — finds +EV on Kalshi during in-progress games.

Compares live Kalshi prices against Pinnacle's current in-game lines
(via TheOddsAPI, which updates Pinnacle odds during live games).

Key difference from the main pipeline:
  - Only evaluates games that have ALREADY STARTED (commence_time < now)
  - Never auto-places bets — display only
  - No Kelly sizing (live stakes require manual judgment on game state)

Usage:
    python scripts/live_ev.py
    python scripts/live_ev.py --sectors nba,nfl
    python scripts/live_ev.py --threshold 0.03
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

import structlog

from evmax.clients.kalshi import KalshiClient
from evmax.clients.pinnacle import PinnacleClient
from evmax.ev.calculator import calculate_ev
from evmax.matching.engine import MatchingEngine
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType
from evmax.models_ml.spread_distribution import SpreadDistributionModel
from evmax.sectors.registry import ALL_SECTORS, get_handler
from evmax.settings import get_settings

logger = structlog.get_logger(__name__)

_SETTINGS = get_settings()
_MATCHING_ENGINE = MatchingEngine()
_SPREAD_MODEL = SpreadDistributionModel()


async def scan_live(sectors: list[str], threshold: float) -> None:
    now_utc = datetime.now(timezone.utc)
    results: list[dict] = []

    for sector in sectors:
        # Fetch Pinnacle odds and Kalshi markets concurrently
        async with KalshiClient() as kalshi, PinnacleClient() as pinnacle:
            pm_markets, sharp_odds_list = await asyncio.gather(
                kalshi.get_markets(sector),
                pinnacle.get_odds(sector),
            )

        if not pm_markets or not sharp_odds_list:
            continue

        # Only keep sharp odds for games that have already started
        live_sharp = [
            so for so in sharp_odds_list
            if so.event_date is not None and so.event_date < now_utc
        ]
        if not live_sharp:
            logger.info("no_live_games", sector=sector)
            continue

        # Enrich markets via sector handler
        handler = get_handler(sector)
        pm_markets = [handler.enrich_market(m) for m in pm_markets]

        # Match Kalshi markets to live sharp events
        matched = _MATCHING_ENGINE.match_all(pm_markets, live_sharp)
        if not matched:
            continue

        for market, sharp_odds, confidence in matched:
            # Skip extreme prices (near-resolved)
            if market.yes_price < 0.04 or market.yes_price > 0.96:
                continue

            # Determine true probability
            if market.market_type == MarketType.spread and market.line is not None:
                if sharp_odds.spread_line is None:
                    continue
                line_diff = abs(abs(market.line) - abs(sharp_odds.spread_line))
                if line_diff > _SETTINGS.spread_line_tolerance:
                    continue
                normalizer = NameNormalizer(sector)
                norm_b = normalizer.normalize(sharp_odds.outcome_b_label)
                yes_is_underdog = (market.yes_team == norm_b)
                pred = _SPREAD_MODEL.predict(sharp_odds, market.line, sector, yes_is_underdog)
                if not pred:
                    continue
                true_prob = pred.true_prob
            else:
                if sharp_odds.true_prob_a is None:
                    continue
                true_prob = sharp_odds.true_prob_a

                # Align to YES team
                if market.yes_team is not None:
                    if market.yes_team in ("tie", "draw", "x"):
                        if sharp_odds.true_prob_draw is None:
                            continue
                        true_prob = sharp_odds.true_prob_draw
                    else:
                        normalizer = NameNormalizer(sector)
                        norm_b = normalizer.normalize(sharp_odds.outcome_b_label)
                        if market.yes_team == norm_b:
                            true_prob = sharp_odds.true_prob_b

            # Calculate EV using live Kalshi price
            live_price = market.yes_price
            ev, edge_pct = calculate_ev(live_price, true_prob)

            if ev < threshold:
                continue

            # Fetch real-time Kalshi price to confirm (live markets move fast)
            async with KalshiClient() as kalshi:
                ticker = market.ticker or market.id.removeprefix("kalshi:")
                rt_price = await kalshi.get_market_price(ticker)

            if rt_price is None:
                confirmed_ev = ev
                confirmed_price = live_price
            else:
                confirmed_ev, _ = calculate_ev(rt_price, true_prob)
                confirmed_price = rt_price

            if confirmed_ev < threshold:
                logger.info(
                    "live_ev_drifted",
                    market_id=market.id,
                    scan_ev=round(ev, 4),
                    rt_ev=round(confirmed_ev, 4),
                )
                continue

            payout = 1.0 / confirmed_price
            results.append({
                "sector": sector.upper(),
                "market_id": market.id,
                "title": market.title,
                "live_price": confirmed_price,
                "true_prob": true_prob,
                "payout": payout,
                "ev_pct": confirmed_ev * 100,
                "volume_usd": market.volume_usd,
                "commence_time": sharp_odds.event_date.strftime("%H:%M UTC") if sharp_odds.event_date else "?",
            })

    # Display results
    if not results:
        print("\nNo live +EV opportunities found.\n")
        return

    results.sort(key=lambda r: r["ev_pct"], reverse=True)
    print(f"\n{'='*72}")
    print(f"  LIVE +EV OPPORTUNITIES  (threshold: {threshold*100:.0f}%)  —  {now_utc.strftime('%H:%M UTC')}")
    print(f"{'='*72}")
    header = f"{'Market':<38} {'Price':>6} {'TrueP':>6} {'EV%':>7} {'Payout':>7} {'Vol':>9}  Started"
    print(header)
    print("-" * 72)
    for r in results:
        title = r["title"][:36] if len(r["title"]) > 36 else r["title"]
        print(
            f"{title:<38} "
            f"{r['live_price']:>6.3f} "
            f"{r['true_prob']:>6.3f} "
            f"{r['ev_pct']:>6.1f}% "
            f"{r['payout']:>6.2f}x "
            f"${r['volume_usd']:>8,.0f}  "
            f"{r['commence_time']}"
        )
    print(f"{'='*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live EV scanner for in-progress games")
    parser.add_argument(
        "--sectors",
        default=",".join(ALL_SECTORS),
        help=f"Comma-separated sectors (default: all). Available: {', '.join(ALL_SECTORS)}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Minimum EV threshold (default: 0.02 = 2%%)",
    )
    args = parser.parse_args()

    sectors = [s.strip().lower() for s in args.sectors.split(",")]
    asyncio.run(scan_live(sectors, args.threshold))


if __name__ == "__main__":
    main()
