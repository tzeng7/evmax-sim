"""Walk-forward sweep of the NCAAB offseason Elo regression coefficient.

Why: docs/SEASON_START.md §5 names NCAAB the #1 target for offseason Elo
regression (transfer-portal churn), but no sweep existed and `offseason_regress.py`
has no `ncaab` entry in SECTOR_DEFAULT_KEEP. Sibling of
`backtest_nba_elo_regression.py` (same protocol, same replay helpers): the
production `EloModelAgent` replayed over the ESPN D1 scoreboard cache loaded by
`backtest_elo_h2h_rest.py` (seasons 2022-23 .. 2025-26, start-year labels).

Differences from the NBA script: sector ncaab; season = start year with the
season rolling over Jul 1; OPEN_GAMES = 6 (college teams play ~30 games, so
"opening" is the first ~6 of a team's season, the window before efficiency
ratings reach MIN_GAMES and Form un-gates); rest layer ON (NCAAB is in
REST_RESOLVED_SECTORS), H2H OFF. RANK = 2023, CONFIRM = 2024, HOLDOUT = 2025;
the winner is fixed on RANK opening Brier, must not lose on CONFIRM, and
HOLDOUT is reported once. Only four seasons exist in the loader, so the
power is limited: read the z-scores, not just the point winner.

Usage:
    uv run python scripts/backtest_ncaab_elo_regression.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import evmax.agents.models.elo_agent as elo_mod  # noqa: E402
from evmax.agents.models.elo_agent import DEFAULT_ELO, LOW_DATA_THRESHOLD  # noqa: E402

import backtest_elo_h2h_rest as h2h  # noqa: E402

SECTOR = "ncaab"
OPEN_GAMES = 6
DEFAULT_KEEP_GRID = [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2]


def nba_season_of(d: date) -> int:
    """Start-year season label (season rolls over Jul 1)."""
    return d.year if d.month >= 7 else d.year - 1


def load_games(first_month: str) -> list:  # first_month unused: the loader carries the in-season months only
    games = h2h.load_espn_games(SECTOR)
    for g in games:
        g.season = nba_season_of(g.d)
    games.sort(key=lambda g: (g.d, g.home, g.away))
    return games


def walk_forward(games: list, keep: float, boost: float, decay: int,
                 score_seasons: set[int]) -> dict[int, dict[str, list[float]]]:
    """Replay every game; return {season: {'open': [...], 'full': [...]}} Brier
    lists in game order (identical game sets across variants → paired)."""
    if boost > 1.0:
        elo_mod.EARLY_K_BOOST[SECTOR] = boost
        elo_mod.EARLY_K_DECAY_GAMES[SECTOR] = decay
    else:
        elo_mod.EARLY_K_BOOST.pop(SECTOR, None)
        elo_mod.EARLY_K_DECAY_GAMES.pop(SECTOR, None)
    tracker = h2h.ScheduleTracker()
    agent = h2h.make_replay_agent(tracker)
    agent.use_h2h = False   # H2H rejected everywhere (docs/elo-h2h-rest-eval.md)
    agent.use_rest = True   # NCAAB is in REST_RESOLVED_SECTORS
    out: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"open": [], "full": []})
    season_games: dict[str, int] = defaultdict(int)
    cur: Optional[int] = None
    for g in games:
        if g.season != cur:
            if cur is not None:
                st = agent._sector_state(SECTOR)
                for t, r in list(st["ratings"].items()):
                    st["ratings"][t] = round(DEFAULT_ELO + keep * (r - DEFAULT_ELO), 2)
                    st["season_games"][t] = 0
            season_games.clear()
            cur = g.season
        h2h._ReplayClock._now = g.d
        n_min = min(agent._get_count(SECTOR, g.home), agent._get_count(SECTOR, g.away))
        if g.season in score_seasons and g.hs != g.as_ and n_min >= LOW_DATA_THRESHOLD:
            p_home, _, _ = agent._win_probs(SECTOR, g.home, g.away, g.d)
            b = (p_home - (1.0 if g.hs > g.as_ else 0.0)) ** 2
            out[g.season]["full"].append(b)
            if min(season_games[g.home], season_games[g.away]) < OPEN_GAMES:
                out[g.season]["open"].append(b)
        agent.update(g.home, g.away, g.hs, g.as_, SECTOR, event_date=g.d.isoformat())
        tracker.record(g.home, g.d)
        tracker.record(g.away, g.d)
        season_games[g.home] += 1
        season_games[g.away] += 1
    return out


def _pool(res: dict, seasons: list[int], key: str) -> list[float]:
    return [b for s in seasons for b in res.get(s, {}).get(key, [])]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _delta(res_a: dict, res_b: dict, seasons: list[int], key: str) -> str:
    """Paired Δ (a − b) ×1000 with z; negative = a better."""
    m, se, z, n = h2h.paired_delta(_pool(res_a, seasons, key), _pool(res_b, seasons, key))
    return f"{m:+.2f}/1000 (z {z:+.2f}, n {n})"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first-month", default="2022-11", help="Burn-in starts here (first season not scored)")
    ap.add_argument("--rank-seasons", default="2023")
    ap.add_argument("--confirm-season", type=int, default=2024)
    ap.add_argument("--holdout-season", type=int, default=2025)
    ap.add_argument("--keep-grid", default=",".join(str(k) for k in DEFAULT_KEEP_GRID))
    ap.add_argument("--boost-grid", default="1.0")
    ap.add_argument("--decay-grid", default="10,20,30")
    args = ap.parse_args(argv)

    rank = [int(s) for s in args.rank_seasons.split(",") if s]
    confirm, holdout = args.confirm_season, args.holdout_season
    keep_grid = [float(k) for k in args.keep_grid.split(",") if k]
    boost_grid = [float(b) for b in args.boost_grid.split(",") if b]
    decay_grid = [int(d) for d in args.decay_grid.split(",") if d]
    if 1.0 not in keep_grid:
        keep_grid.insert(0, 1.0)
    score = set(rank) | {confirm, holdout}

    games = load_games(args.first_month)
    by_season = defaultdict(int)
    for g in games:
        by_season[g.season] += 1
    print(f"Loaded {len(games)} games: " + ", ".join(f"{s}:{n}" for s, n in sorted(by_season.items())))
    print(f"RANK {rank}  CONFIRM {confirm}  HOLDOUT {holdout}  "
          f"opening window = either team < {OPEN_GAMES} season games")

    saved_date = elo_mod.date
    saved_boost = dict(elo_mod.EARLY_K_BOOST)
    saved_decay = dict(elo_mod.EARLY_K_DECAY_GAMES)
    elo_mod.date = h2h._ReplayClock  # type: ignore[assignment]
    try:
        results = {k: walk_forward(games, k, 1.0, 0, score) for k in keep_grid}
        base = results[1.0]
        print(f"\n{'keep':>5} {'rank open':>10} {'n':>5} {'rank full':>10}  {'conf open':>10} {'n':>4} "
              f"{'conf full':>10}   rank open Δ vs keep=1.0")
        for k in keep_grid:
            r = results[k]
            print(f"{k:5.2f} {_mean(_pool(r, rank, 'open')):10.4f} {len(_pool(r, rank, 'open')):5d} "
                  f"{_mean(_pool(r, rank, 'full')):10.4f}  {_mean(_pool(r, [confirm], 'open')):10.4f} "
                  f"{len(_pool(r, [confirm], 'open')):4d} {_mean(_pool(r, [confirm], 'full')):10.4f}   "
                  + ("—" if k == 1.0 else _delta(r, base, rank, "open")))

        winner = min(keep_grid, key=lambda k: _mean(_pool(results[k], rank, "open")))
        w = results[winner]
        conf_ok = _mean(_pool(w, [confirm], "open")) <= _mean(_pool(base, [confirm], "open")) + 1e-12
        print(f"\nStage 1 winner on RANK opening Brier: keep={winner:.2f}")
        if winner != 1.0:
            print(f"  RANK    open {_delta(w, base, rank, 'open')}   full {_delta(w, base, rank, 'full')}")
            print(f"  CONFIRM open {_delta(w, base, [confirm], 'open')}   full {_delta(w, base, [confirm], 'full')}"
                  f"  → {'holds' if conf_ok else 'FAILS'}")
            print("  per-season opening Δ: " + ", ".join(
                f"{s}: {1000 * (_mean(w[s]['open']) - _mean(base[s]['open'])):+.2f}" for s in rank + [confirm]))
        chosen = winner if (winner != 1.0 and conf_ok) else 1.0

        chosen_boost, chosen_decay = 1.0, 0
        if chosen < 1.0 and any(b > 1.0 for b in boost_grid):
            print(f"\nStage 2 — early-K boost on keep={chosen:.2f}")
            nob = results[chosen]
            best = None
            for boost in [b for b in boost_grid if b > 1.0]:
                for decay in decay_grid:
                    r = walk_forward(games, chosen, boost, decay, score)
                    print(f"  boost={boost:.2f} decay={decay:3d}  rank open {_delta(r, nob, rank, 'open')}  "
                          f"confirm open {_delta(r, nob, [confirm], 'open')}")
                    ro = _mean(_pool(r, rank, "open"))
                    if best is None or ro < best[0]:
                        best = (ro, boost, decay, r)
            if best and best[0] < _mean(_pool(nob, rank, "open")) and \
                    _mean(_pool(best[3], [confirm], "open")) <= _mean(_pool(nob, [confirm], "open")):
                chosen_boost, chosen_decay = best[1], best[2]
                print(f"  Stage 2 winner: boost={chosen_boost} decay={chosen_decay}")
            else:
                print("  Stage 2: no boost clears both gates — boost OFF")

        print(f"\nHOLDOUT {holdout} (choice fixed before this line; reported once)")
        final = walk_forward(games, chosen, chosen_boost, chosen_decay, score) if chosen_boost > 1.0 else results[chosen]
        for label, r in (("keep=1.00 (today)", base), (f"chosen keep={chosen:.2f}", final)):
            print(f"  {label:22} open Brier {_mean(_pool(r, [holdout], 'open')):.4f} "
                  f"(n {len(_pool(r, [holdout], 'open'))})  full {_mean(_pool(r, [holdout], 'full')):.4f} "
                  f"(n {len(_pool(r, [holdout], 'full'))})")
        if chosen != 1.0:
            print(f"  Δ open {_delta(final, base, [holdout], 'open')}   Δ full {_delta(final, base, [holdout], 'full')}")
        print(f"\nVERDICT: keep={chosen:.2f}, boost={chosen_boost}, decay={chosen_decay}")
    finally:
        elo_mod.date = saved_date  # type: ignore[assignment]
        elo_mod.EARLY_K_BOOST.clear(); elo_mod.EARLY_K_BOOST.update(saved_boost)
        elo_mod.EARLY_K_DECAY_GAMES.clear(); elo_mod.EARLY_K_DECAY_GAMES.update(saved_decay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
