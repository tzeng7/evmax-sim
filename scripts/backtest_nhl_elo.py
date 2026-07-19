"""Cold-start walk-forward calibration sweep for NHL generic Elo (MODEL-2 NHL half).

NHL has never had an `elo_state.json["nhl"]` entry — `K_FACTORS["nhl"]` and
`HOME_ADVANTAGE_ELO["nhl"]` don't exist in `evmax/agents/models/elo_agent.py`,
so `elo` is deliberately left out of `data/categories.yaml`'s nhl `models:`
list. This mirrors the NCAAW protocol used for MODEL-2 (see
`scripts/backtest_ncaab_efficiency.py --league womens` and the K=35/
HOME_ADVANTAGE_ELO=80 comment in elo_agent.py): sweep a small (K, home_adv)
grid via genuine walk-forward Brier on one season, confirm on a second, then
report the FINAL number on a fully held-out most-recent season.

Leak-free protocol (NHL's public ESPN history only goes back a few seasons,
so the split is adapted from NCAAW's "sweep 2024-25, hold out 2025-26"):
  - RANK window:  2023-24 season, walk-forward running Brier (state starts
    empty at 1500 and is fed games chronologically; each game is scored on
    its PRE-game rating before the update is applied — no leakage within
    the season itself). This ranks the grid.
  - CONFIRM window: 2024-25 season, same walk-forward protocol continuing
    from a state seeded through 2023-24 for that grid cell (this is closer
    to the live agent's behavior — ratings persist across seasons, they
    aren't reset each October). Cells that don't hold up here are dropped.
  - HOLDOUT window: 2025-26 season (now complete — today is 2026-07-18),
    same running-Brier protocol continuing from the RANK+CONFIRM state. The
    (K, home_adv) pair is *fixed* before this pass runs — its result is
    reported once, not used to pick among cells.

Only the raw Elo win-probability formula is evaluated here (no MOV/SOS/H2H/
recency-K/early-season-boost layers, no injuries) — those extra layers are
part of `EloModelAgent.update()`'s production behavior but would make a
clean K/home-adv grid search harder to interpret. This mirrors how NCAAW's
sweep isolated K and HOME_ADVANTAGE_ELO before those layers apply.

Usage:
    python scripts/backtest_nhl_elo.py
    python scripts/backtest_nhl_elo.py --k-grid 6,10,14,20,25 --home-grid 0,20,32,48,60
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

ESPN_NHL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"

DEFAULT_K_GRID = [6.0, 10.0, 14.0, 20.0, 25.0]
DEFAULT_HOME_GRID = [0.0, 20.0, 32.0, 48.0, 60.0]

# The "naive default" comparison point cited in the report: generic sport
# defaults an engineer might reach for without any NHL-specific tuning
# (K=20 mirrors NBA/WNBA/generic team sports, home_adv=60 mirrors soccer's
# level of home-ice advantage).
NAIVE_K = 20.0
NAIVE_HOME_ADV = 60.0


async def fetch_espn_season(season_start: int) -> list[dict]:
    """Fetch completed regular-season NHL games for one season (starting year)."""
    games: list[dict] = []
    current = date(season_start, 10, 1)
    end = date(season_start + 1, 4, 30)

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
                # Regular season only (ESPN season.type: 1=pre, 2=regular,
                # 3=post, 4=offseason).
                season_type = ev.get("season", {}).get("type", 2)
                if season_type != 2:
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
                if hs == as_:
                    continue  # NHL games can't end in a tie (OT/SO decides)
                games.append({
                    "date": ev.get("date", "")[:10],
                    "home": home.get("team", {}).get("displayName", "").lower(),
                    "away": away.get("team", {}).get("displayName", "").lower(),
                    "home_score": hs,
                    "away_score": as_,
                    "home_won": int(hs > as_),
                })

            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
            await asyncio.sleep(0.15)

    return sorted(games, key=lambda g: g["date"])


def brier(pred: float, actual: int) -> float:
    return (pred - actual) ** 2


class RawElo:
    """Bare-bones Elo state (ratings only) used to isolate K / home_adv."""

    def __init__(self, k: float, home_adv: float) -> None:
        self.k = k
        self.home_adv = home_adv
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, 1500.0)

    def expected_home(self, home: str, away: str) -> float:
        elo_h = self.get(home) + self.home_adv
        elo_a = self.get(away)
        return 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h) / 400.0))

    def update(self, home: str, away: str, home_won: int) -> None:
        exp_home = self.expected_home(home, away)
        elo_h, elo_a = self.get(home), self.get(away)
        delta = self.k * (home_won - exp_home)
        self.ratings[home] = elo_h + delta
        self.ratings[away] = elo_a - delta


def walk_forward(elo: RawElo, games: list[dict]) -> tuple[float, float, int]:
    """Score every game's PRE-game prediction, then apply the update. Returns
    (mean_brier, accuracy, n_scored)."""
    total_brier = 0.0
    correct = 0
    n = 0
    for g in games:
        p_home = elo.expected_home(g["home"], g["away"])
        total_brier += brier(p_home, g["home_won"])
        if (p_home > 0.5) == bool(g["home_won"]):
            correct += 1
        n += 1
        elo.update(g["home"], g["away"], g["home_won"])
    if n == 0:
        return float("nan"), float("nan"), 0
    return total_brier / n, correct / n, n


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-grid", default=",".join(str(k) for k in DEFAULT_K_GRID))
    ap.add_argument("--home-grid", default=",".join(str(h) for h in DEFAULT_HOME_GRID))
    ap.add_argument("--rank-season", type=int, default=2023, help="Rank window season start year")
    ap.add_argument("--confirm-season", type=int, default=2024, help="Confirm window season start year")
    ap.add_argument("--holdout-season", type=int, default=2025, help="Final holdout season start year")
    args = ap.parse_args()

    k_grid = [float(x) for x in args.k_grid.split(",")]
    home_grid = [float(x) for x in args.home_grid.split(",")]

    print("NHL Elo cold-start calibration sweep (MODEL-2 NHL half)")
    print(f"  Rank:    {args.rank_season}-{str(args.rank_season+1)[-2:]}")
    print(f"  Confirm: {args.confirm_season}-{str(args.confirm_season+1)[-2:]}")
    print(f"  Holdout: {args.holdout_season}-{str(args.holdout_season+1)[-2:]}\n")

    print("Fetching ESPN NHL seasons...")
    rank_games = await fetch_espn_season(args.rank_season)
    print(f"  {args.rank_season}-{str(args.rank_season+1)[-2:]}: {len(rank_games)} games")
    confirm_games = await fetch_espn_season(args.confirm_season)
    print(f"  {args.confirm_season}-{str(args.confirm_season+1)[-2:]}: {len(confirm_games)} games")
    holdout_games = await fetch_espn_season(args.holdout_season)
    print(f"  {args.holdout_season}-{str(args.holdout_season+1)[-2:]}: {len(holdout_games)} games\n")

    if not rank_games or not confirm_games or not holdout_games:
        print("ERROR: one or more seasons returned no games — aborting", file=sys.stderr)
        return 1

    # ---- Grid sweep: rank on 2023-24, confirm on 2024-25 (state carries over) ----
    print(f"{'K':>6} {'HomeAdv':>8} {'RankBrier':>10} {'RankAcc':>8}  "
          f"{'ConfBrier':>10} {'ConfAcc':>8}")
    results = []
    for k in k_grid:
        for home_adv in home_grid:
            elo = RawElo(k=k, home_adv=home_adv)
            rank_brier, rank_acc, rank_n = walk_forward(elo, rank_games)
            # Continue the SAME state (ratings carry across seasons, matching
            # how the live agent behaves — no reset between seasons).
            conf_brier, conf_acc, conf_n = walk_forward(elo, confirm_games)
            results.append({
                "k": k, "home_adv": home_adv,
                "rank_brier": rank_brier, "rank_acc": rank_acc,
                "conf_brier": conf_brier, "conf_acc": conf_acc,
                "elo": elo,  # state after RANK+CONFIRM, ready for HOLDOUT
            })
            print(f"{k:6.1f} {home_adv:8.1f} {rank_brier:10.4f} {rank_acc:8.3f}  "
                  f"{conf_brier:10.4f} {conf_acc:8.3f}")

    # Pick the winner using ONLY rank+confirm (never touches holdout).
    # Rank by confirm-window Brier (2024-25) — the later, more representative
    # of the two pre-holdout windows — with rank-window Brier as a tiebreak.
    results.sort(key=lambda r: (r["conf_brier"], r["rank_brier"]))
    winner = results[0]
    print(f"\nWinner (by confirm-window Brier): K={winner['k']}, "
          f"HOME_ADVANTAGE_ELO={winner['home_adv']}")
    print(f"  Rank    Brier={winner['rank_brier']:.4f}  Acc={winner['rank_acc']:.3f}")
    print(f"  Confirm Brier={winner['conf_brier']:.4f}  Acc={winner['conf_acc']:.3f}")

    # ---- Final holdout: winner (state frozen at end of confirm) vs naive default ----
    print(f"\n{'='*60}")
    print(f"FINAL HOLDOUT — {args.holdout_season}-{str(args.holdout_season+1)[-2:]} "
          f"(fixed before this pass runs)")
    print(f"{'='*60}")

    winner_brier, winner_acc, winner_n = walk_forward(winner["elo"], holdout_games)
    print(f"Winner  K={winner['k']:.1f} home_adv={winner['home_adv']:.1f}  "
          f"n={winner_n}  Brier={winner_brier:.4f}  Acc={winner_acc:.3f}")

    naive_elo = RawElo(k=NAIVE_K, home_adv=NAIVE_HOME_ADV)
    walk_forward(naive_elo, rank_games)
    walk_forward(naive_elo, confirm_games)
    naive_brier, naive_acc, naive_n = walk_forward(naive_elo, holdout_games)
    print(f"Naive   K={NAIVE_K:.1f} home_adv={NAIVE_HOME_ADV:.1f}  "
          f"n={naive_n}  Brier={naive_brier:.4f}  Acc={naive_acc:.3f}")

    delta = naive_brier - winner_brier
    print(f"\nDelta Brier (naive - winner): {delta:+.4f} "
          f"({'winner better' if delta > 0 else 'naive better or tied'})")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
