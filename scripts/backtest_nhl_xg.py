"""Prior-season-holdout backtest for the NHL xG agent.

This is a deliberately simple validation: pull MoneyPuck's end-of-season
team xG state for season N, then predict every game of season N+1 from that
frozen prior-season state and compare against actual outcomes.

It's leak-free (the state predates every game it predicts) but it's NOT a
within-season walk-forward — team strength shifts mid-season are not tracked.
Treat the output as a directional sanity check before promoting the agent
from shadow to live, not as a precise weight tuning. A within-season walk-
forward (re-seeding from MoneyPuck snapshots monthly) is the natural follow-up
once we have month-stamped CSV archives to read from.

Output:
  - Brier score (lower = better; sharp-only NHL is typically ~0.225)
  - Accuracy on the home side
  - Calibration buckets (5 buckets of 0.10 width)
  - Coverage (how many games the agent had enough state to predict)

Season identifier: starting year (matches MoneyPuck's URL convention).
  --train 2024  → 2024-25 season (used as the frozen training state)
  --test-season 2025 → 2025-26 season (games used for evaluation)

Usage:
    python scripts/backtest_nhl_xg.py                    # default: 2024 → 2025
    python scripts/backtest_nhl_xg.py --train 2023 --test-season 2024
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import httpx

from evmax.agents.models.nhl_xg_agent import NhlXgModelAgent
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook
from scripts.seed_nhl_xg import compute_team_stats, fetch_moneypuck_teams_csv

ESPN_NHL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"


async def fetch_espn_games(test_season_start: int) -> list[dict]:
    """Fetch completed regular-season NHL games for the given season.

    test_season_start is the year the season STARTED (e.g. 2025 for 2025-26).
    The NHL regular season runs ~Oct (start year) → April (start year + 1);
    we walk month-by-month and let ESPN tell us which games are completed.
    """
    games: list[dict] = []
    current = date(test_season_start, 10, 1)
    end = date(test_season_start + 1, 4, 30)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        headers={"User-Agent": "evmax-backtest/1.0"},
        follow_redirects=True,
    ) as client:
        while current <= end:
            month_str = current.strftime("%Y%m")
            try:
                r = await client.get(ESPN_NHL_SCOREBOARD,
                                     params={"dates": month_str, "limit": 200})
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  WARN {month_str}: {e}")
                current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
                continue

            for ev in data.get("events", []):
                comp = ev.get("competitions", [{}])[0]
                status = comp.get("status", {}).get("type", {})
                if not status.get("completed"):
                    continue
                competitors = comp.get("competitors", [])
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue
                try:
                    hs = int(home.get("score", 0))
                    as_ = int(away.get("score", 0))
                except (ValueError, TypeError):
                    continue
                games.append({
                    "date": ev.get("date", "")[:10],
                    "home": home.get("team", {}).get("displayName", "").lower(),
                    "away": away.get("team", {}).get("displayName", "").lower(),
                    "home_score": hs,
                    "away_score": as_,
                    "home_won": int(hs > as_),
                })

            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
            await asyncio.sleep(0.2)

    return games


def make_pair(home: str, away: str) -> tuple[PredictionMarket, SharpOdds]:
    eid = f"bt_{home}_{away}"
    market = PredictionMarket(
        id=eid, market_id=eid, event_id=eid,
        sector="nhl", team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=0.5, no_price=0.5,
    )
    sharp = SharpOdds(
        event_id=eid, book=SharpBook.pinnacle, sector="nhl",
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=0.5, true_prob_b=0.5,
    )
    return market, sharp


def brier(pred: float, actual: int) -> float:
    return (pred - actual) ** 2


async def run_backtest(train_season: int, test_season: int) -> int:
    print(f"NHL xG backtest")
    print(f"  Train season: {train_season}-{str(train_season + 1)[-2:]} "
          f"(MoneyPuck snapshot)")
    print(f"  Test season:  {test_season}-{str(test_season + 1)[-2:]} "
          f"(ESPN completed games)\n")

    # Build agent state from prior season's MoneyPuck snapshot
    print("Loading MoneyPuck training state...")
    rows = fetch_moneypuck_teams_csv(train_season)
    teams, league_avg = compute_team_stats(rows)
    if not teams:
        print("ERROR: no training teams produced", file=sys.stderr)
        return 1
    print(f"  {len(teams)} teams loaded, league avg xG/60 = {league_avg:.2f}\n")

    agent = NhlXgModelAgent()
    agent._state = {
        "nhl": {
            "league_avg_xg_per_60": league_avg,
            "teams": teams,
            "fetched_at": "backtest",
        }
    }

    # Fetch test-season games
    print("Fetching ESPN test-season games...")
    games = await fetch_espn_games(test_season)
    print(f"  {len(games)} completed games\n")

    # Predict
    total_brier = 0.0
    correct = 0
    predicted = 0
    skipped_no_state = 0
    bucket_pred: dict[int, list[int]] = defaultdict(list)

    for g in games:
        market, sharp = make_pair(g["home"], g["away"])
        pred = await agent.predict_pair(market, sharp)
        if pred is None:
            skipped_no_state += 1
            continue
        predicted += 1
        p = pred.true_prob_a
        total_brier += brier(p, g["home_won"])
        if (p > 0.5) == bool(g["home_won"]):
            correct += 1
        # Calibration bucket on home prob
        bucket = min(9, int(p * 10))
        bucket_pred[bucket].append(g["home_won"])

    if predicted == 0:
        print("No games predicted — check team-name resolution", file=sys.stderr)
        return 1

    print(f"Coverage: {predicted}/{len(games)} games "
          f"({100 * predicted / len(games):.1f}%)")
    print(f"Skipped (no state):       {skipped_no_state}")
    print(f"Accuracy (home pick):     {correct}/{predicted} "
          f"({100 * correct / predicted:.1f}%)")
    print(f"Brier score:              {total_brier / predicted:.4f}\n")

    print("Calibration (home P bucket → actual home WP):")
    for b in sorted(bucket_pred.keys()):
        actuals = bucket_pred[b]
        if not actuals:
            continue
        lo = b / 10
        hi = lo + 0.10
        mid = lo + 0.05
        actual_rate = sum(actuals) / len(actuals)
        gap = actual_rate - mid
        print(f"  [{lo:.2f}, {hi:.2f})  n={len(actuals):4d}  "
              f"actual={actual_rate:.3f}  gap={gap:+.3f}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train",
        type=int,
        default=2024,
        help="Training-season STARTING year (e.g. 2024 = 2024-25). Default 2024.",
    )
    ap.add_argument(
        "--test-season",
        type=int,
        default=None,
        help="Test-season STARTING year. Default: train + 1.",
    )
    args = ap.parse_args()
    test_season = args.test_season or args.train + 1
    return asyncio.run(run_backtest(args.train, test_season))


if __name__ == "__main__":
    raise SystemExit(main())
