"""Cold-start walk-forward calibration sweep for NCAAF generic Elo (MODEL-2 NCAAF half).

`ncaaf` has never had an `elo_state.json["ncaaf"]` entry and appears NOWHERE in
`evmax/agents/models/elo_agent.py`, so generic `elo` silently takes the module
fallbacks — `K_FACTORS.get("ncaaf", 20.0) = 20.0` (identical to NBA's) and
`HOME_ADVANTAGE_ELO.get("ncaaf", 0.0) = 0.0`. A **zero** college-football home
edge is near-certainly wrong (FBS home teams win ~57-60% and the ncaaf_efficiency
model itself carries HOME_EDGE_PTS=2.5), yet `data/categories.yaml` lists generic
`elo` at ensemble weight 0.25 for ncaaf. This is the "CFB-calibrated generic Elo"
that categories.yaml flags as PHASE 2 work.

This script mirrors the NCAAW / NHL MODEL-2 protocol exactly
(`scripts/backtest_nhl_elo.py`, and the K=35/HOME_ADVANTAGE_ELO=80 comment in
elo_agent.py): sweep a small (K, home_adv) grid via genuine walk-forward Brier,
rank on one season, confirm on the next, then report the FINAL number once on a
fully held-out most-recent complete season.

Leak-free protocol (CFB public ESPN history covers 2021→present):
  - WARMUP window(s): seasons strictly earlier than RANK. State starts empty at
    1500 and is fed these seasons chronologically so ratings are realistically
    warm (not cold 1500s) by the time the ranking window begins — matching the
    live agent, whose persistent state is always warm. Default: 2021, 2022.
  - RANK window:  one season, walk-forward running Brier. Each game is scored on
    its PRE-game rating BEFORE its update is applied — no within-season leakage.
    Ranks the grid. Default: 2023.
  - CONFIRM window: the next season, same protocol, state carried over (ratings
    persist across seasons, they are not reset each August). Default: 2024.
  - HOLDOUT window: the most-recent complete season, same running-Brier protocol
    continuing from the WARMUP+RANK+CONFIRM state. The (K, home_adv) pair is
    *fixed* before this pass runs — its result is reported once, never used to
    pick among cells. Default: 2025.

Naive baseline = the CURRENT production fallback K=20 / home_adv=0 (what generic
Elo does for ncaaf today). The winner's holdout delta against this baseline is
the direct cost of the zero-home-advantage bug.

Only the raw Elo win-probability formula is evaluated (no MOV/SOS/H2H/recency-K
layers, no injuries) — same isolation the NCAAW/NHL sweeps used to keep the K /
home_adv grid interpretable. Home advantage is applied to the NOMINAL home team
on EVERY game, including neutral-site bowls/kickoff games, because the live
`EloModelAgent` does exactly that (it does not thread a neutral-site flag). A
neutral-aware variant is printed for information only; it is NOT how production
behaves and is NOT used to pick the winner. Making the live agent neutral-aware
would be a separate change.

Usage:
    python scripts/backtest_ncaaf_elo.py
    python scripts/backtest_ncaaf_elo.py --k-grid 16,20,25,32,40 --home-grid 0,40,60,80,100
    python scripts/backtest_ncaaf_elo.py --warmup 2021,2022 --rank 2023 --confirm 2024 --holdout 2025
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import httpx

from evmax.agents.cleanup.resolver import _ESPN_HTTP_UA
from evmax.clients import cfb_espn as C

DEFAULT_K_GRID = [16.0, 20.0, 25.0, 32.0, 40.0]
DEFAULT_HOME_GRID = [0.0, 40.0, 60.0, 80.0, 100.0]

# The naive comparison point: the CURRENT production fallback for ncaaf. Generic
# Elo takes K_FACTORS.get("ncaaf", 20.0)=20 and HOME_ADVANTAGE_ELO.get("ncaaf",
# 0.0)=0 today. The winner's holdout delta against this pair quantifies the
# zero-home-advantage bug.
NAIVE_K = 20.0
NAIVE_HOME_ADV = 0.0


def fetch_season(season: int) -> list[dict]:
    """Completed FBS games for one CFB season (label = fall year).

    Walks the season window via cfb_espn.fetch_scoreboard_day (which already
    filters to FBS groups=80 and parses the neutral-site flag), using the
    resolver's neutral ESPN User-Agent (_ESPN_HTTP_UA, a plain curl/* string).
    ESPN's WAF 403s identifying "evmax-*" and browser-impersonator UAs alike
    (PR #164 / resolver.py:52), so reuse the one recognized tool UA. Days are
    fetched CONCURRENTLY over a shared thread-safe httpx.Client (a CFB season is
    ~150 days and most weekdays are empty, so a serial walk is dominated by
    round-trip latency); pool.map preserves chronological order. Drops
    non-completed games, games with a missing score, and the rare tie (CFB games
    are decided in OT, so a tie means bad/partial data).
    """
    start_s, end_s = C.SEASON_WINDOWS[season]
    start = C.dt.date.fromisoformat(start_s)
    end = C.dt.date.fromisoformat(end_s)
    days: list = []
    d = start
    while d <= end:
        days.append(d)
        d += C.dt.timedelta(days=1)

    out: list[dict] = []
    seen: set[str] = set()
    with httpx.Client(
        timeout=httpx.Timeout(30.0),
        headers={"User-Agent": _ESPN_HTTP_UA},
        follow_redirects=True,
    ) as client:
        with ThreadPoolExecutor(max_workers=10) as pool:
            per_day = list(pool.map(lambda day: C.fetch_scoreboard_day(client, day), days))
    for games in per_day:  # chronological (pool.map preserves input order)
        for g in games:
            gid = g["game_id"]
            if gid in seen or not g["completed"]:
                continue
            hs, as_ = g["home"]["score"], g["away"]["score"]
            if hs is None or as_ is None or hs == as_:
                continue
            home_loc = (g["home"]["location"] or "").lower().strip()
            away_loc = (g["away"]["location"] or "").lower().strip()
            if not home_loc or not away_loc:
                continue
            seen.add(gid)
            out.append({
                "date": g["date"],
                "home": home_loc,
                "away": away_loc,
                "neutral": g["neutral"],
                "home_won": int(hs > as_),
            })
    return sorted(out, key=lambda r: r["date"])


def brier(pred: float, actual: int) -> float:
    return (pred - actual) ** 2


class RawElo:
    """Bare-bones Elo state (ratings only) used to isolate K / home_adv."""

    def __init__(self, k: float, home_adv: float, neutral_aware: bool = False) -> None:
        self.k = k
        self.home_adv = home_adv
        self.neutral_aware = neutral_aware
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, 1500.0)

    def _home_bonus(self, neutral: bool) -> float:
        # Production-faithful default: the live EloModelAgent applies the sector
        # home bonus on every game and never inspects a neutral-site flag, so we
        # match that here. Only the informational neutral-aware variant zeroes it
        # at neutral venues.
        if self.neutral_aware and neutral:
            return 0.0
        return self.home_adv

    def expected_home(self, home: str, away: str, neutral: bool) -> float:
        elo_h = self.get(home) + self._home_bonus(neutral)
        elo_a = self.get(away)
        return 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h) / 400.0))

    def update(self, home: str, away: str, home_won: int, neutral: bool) -> None:
        exp_home = self.expected_home(home, away, neutral)
        elo_h, elo_a = self.get(home), self.get(away)
        delta = self.k * (home_won - exp_home)
        self.ratings[home] = elo_h + delta
        self.ratings[away] = elo_a - delta


def walk_forward(elo: RawElo, games: list[dict]) -> tuple[float, float, int]:
    """Score each game's PRE-game prediction, then apply the update.

    Returns (mean_brier, accuracy, n_scored).
    """
    total_brier = 0.0
    correct = 0
    n = 0
    for g in games:
        p_home = elo.expected_home(g["home"], g["away"], g["neutral"])
        total_brier += brier(p_home, g["home_won"])
        if (p_home > 0.5) == bool(g["home_won"]):
            correct += 1
        n += 1
        elo.update(g["home"], g["away"], g["home_won"], g["neutral"])
    if n == 0:
        return float("nan"), float("nan"), 0
    return total_brier / n, correct / n, n


def _seed_state(k: float, home_adv: float, warmup: list[list[dict]],
                neutral_aware: bool = False) -> RawElo:
    """Fresh Elo walked through every warmup season (no scoring, just updates)."""
    elo = RawElo(k=k, home_adv=home_adv, neutral_aware=neutral_aware)
    for season_games in warmup:
        for g in season_games:
            elo.update(g["home"], g["away"], g["home_won"], g["neutral"])
    return elo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-grid", default=",".join(str(k) for k in DEFAULT_K_GRID))
    ap.add_argument("--home-grid", default=",".join(str(h) for h in DEFAULT_HOME_GRID))
    ap.add_argument("--warmup", default="2021,2022",
                    help="Comma-separated warmup season start years (strictly earlier than --rank)")
    ap.add_argument("--rank", type=int, default=2023, help="Rank window season")
    ap.add_argument("--confirm", type=int, default=2024, help="Confirm window season")
    ap.add_argument("--holdout", type=int, default=2025, help="Final holdout season")
    args = ap.parse_args()

    k_grid = [float(x) for x in args.k_grid.split(",")]
    home_grid = [float(x) for x in args.home_grid.split(",")]
    warmup_seasons = [int(x) for x in args.warmup.split(",") if x.strip()]

    print("NCAAF Elo cold-start calibration sweep (MODEL-2 NCAAF half)")
    print(f"  Warmup:  {warmup_seasons}")
    print(f"  Rank:    {args.rank}")
    print(f"  Confirm: {args.confirm}")
    print(f"  Holdout: {args.holdout}\n")

    print("Fetching ESPN CFB seasons (FBS, groups=80)...")
    warmup_games = []
    for s in warmup_seasons:
        g = fetch_season(s)
        print(f"  warmup {s}: {len(g)} games")
        warmup_games.append(g)
    rank_games = fetch_season(args.rank)
    print(f"  rank   {args.rank}: {len(rank_games)} games")
    confirm_games = fetch_season(args.confirm)
    print(f"  confirm {args.confirm}: {len(confirm_games)} games")
    holdout_games = fetch_season(args.holdout)
    print(f"  holdout {args.holdout}: {len(holdout_games)} games\n")

    if not rank_games or not confirm_games or not holdout_games:
        print("ERROR: one or more scored seasons returned no games — aborting",
              file=sys.stderr)
        return 1

    # ---- Grid sweep: warmup → rank → confirm (state carries across all) ----
    print(f"{'K':>6} {'HomeAdv':>8} {'RankBrier':>10} {'RankAcc':>8}  "
          f"{'ConfBrier':>10} {'ConfAcc':>8}")
    results = []
    for k in k_grid:
        for home_adv in home_grid:
            elo = _seed_state(k, home_adv, warmup_games)
            rank_brier, rank_acc, _ = walk_forward(elo, rank_games)
            conf_brier, conf_acc, _ = walk_forward(elo, confirm_games)
            results.append({
                "k": k, "home_adv": home_adv,
                "rank_brier": rank_brier, "rank_acc": rank_acc,
                "conf_brier": conf_brier, "conf_acc": conf_acc,
                "elo": elo,  # state after warmup+rank+confirm, ready for HOLDOUT
            })
            print(f"{k:6.1f} {home_adv:8.1f} {rank_brier:10.4f} {rank_acc:8.3f}  "
                  f"{conf_brier:10.4f} {conf_acc:8.3f}")

    # Winner chosen using ONLY rank+confirm (never touches holdout). Rank by
    # confirm-window Brier (later, more representative) with rank-window Brier as
    # the tiebreak — identical to backtest_nhl_elo.py.
    results.sort(key=lambda r: (r["conf_brier"], r["rank_brier"]))
    winner = results[0]
    print(f"\nWinner (by confirm-window Brier): K={winner['k']}, "
          f"HOME_ADVANTAGE_ELO={winner['home_adv']}")
    print(f"  Rank    Brier={winner['rank_brier']:.4f}  Acc={winner['rank_acc']:.3f}")
    print(f"  Confirm Brier={winner['conf_brier']:.4f}  Acc={winner['conf_acc']:.3f}")

    # ---- Final holdout: winner (state frozen at end of confirm) vs naive fallback ----
    print(f"\n{'='*64}")
    print(f"FINAL HOLDOUT — {args.holdout} (fixed before this pass runs)")
    print(f"{'='*64}")

    winner_brier, winner_acc, winner_n = walk_forward(winner["elo"], holdout_games)
    print(f"Winner  K={winner['k']:.1f} home_adv={winner['home_adv']:.1f}  "
          f"n={winner_n}  Brier={winner_brier:.4f}  Acc={winner_acc:.3f}")

    naive = _seed_state(NAIVE_K, NAIVE_HOME_ADV, warmup_games)
    walk_forward(naive, rank_games)
    walk_forward(naive, confirm_games)
    naive_brier, naive_acc, naive_n = walk_forward(naive, holdout_games)
    print(f"Naive   K={NAIVE_K:.1f} home_adv={NAIVE_HOME_ADV:.1f} (current fallback)  "
          f"n={naive_n}  Brier={naive_brier:.4f}  Acc={naive_acc:.3f}")

    delta = naive_brier - winner_brier
    print(f"\nDelta Brier (naive fallback - winner): {delta:+.4f} "
          f"({'winner better' if delta > 0 else 'naive better or tied'})")

    # ---- Informational: neutral-aware variant of the winner (NOT production) ----
    na_winner = _seed_state(winner["k"], winner["home_adv"], warmup_games, neutral_aware=True)
    walk_forward(na_winner, rank_games)
    walk_forward(na_winner, confirm_games)
    na_brier, na_acc, _ = walk_forward(na_winner, holdout_games)
    print(f"\n[info] neutral-aware winner (home_adv zeroed at neutral sites; NOT "
          f"how production behaves): Brier={na_brier:.4f} Acc={na_acc:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
