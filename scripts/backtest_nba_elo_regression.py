"""Walk-forward sweep of the NBA offseason Elo regression coefficient.

Why: `elo_state.json['nba']` carries raw end-of-Finals ratings into October
and has no `last_updated` stamp, so the 60-day Elo staleness guard never gates
it — NBA Elo fires from opening night. For the first ~5 weeks of a season the
four NBA stat models abstain (efficiency / possession_sim / shot_quality /
matchup need 20 games per team, and their fetch follows the current season
since 2026-10-13), and Form is stale-gated until fresh results arrive. In that
window generic Elo is most of the model side of the NBA blend (sharp_weight
0.70, no FLB re-blend for NBA ⇒ 30% model share). docs/SEASON_START.md §5 rated
NBA regression "marginal" assuming Elo stays at its 0.10 override weight behind
the stat stack; the stack is absent exactly when the regression matters.

Protocol (leak-free; mirrors scripts/backtest_nfl_elo_regression.py):
  - Data: ESPN NBA scoreboards via scripts/backtest_elo_h2h_rest.py's loader
    (preseason / All-Star dropped, ET game days, cached under
    data/cache/elo_feature_backtest/), --first-month .. 2026-06.
  - Replay uses the PRODUCTION `EloModelAgent.update()` / `_win_probs()` with
    the agent's clock patched to each game's date. Layers mirror how NBA Elo
    runs live: the rest layer ON (NBA is in REST_RESOLVED_SECTORS; days of
    rest come from the replay schedule), H2H OFF (live reads H2H under the raw
    Pinnacle label, "los angeles lakers", which never equals the state key, so
    it never fires for NBA).
  - `keep` = fraction of the deviation from 1500 RETAINED across the boundary
    (1.0 = today's behaviour). Applied before each season's first game;
    season_games reset to 0.
  - OPENING window = games where either team has played < OPEN_GAMES (20)
    games of the season — the window where the NBA stat stack abstains.
    Full-season Brier is the secondary column. Burn-in season fed, not scored.
  - RANK = --rank-seasons, CONFIRM = --confirm-season, HOLDOUT =
    --holdout-season (start-year labels). Winner fixed on RANK opening Brier
    and must not lose on CONFIRM; HOLDOUT is reported once, never used to
    choose. The paired z is game-level (each game scored once per variant).
  - Optional stage 2: EARLY_K_BOOST swept ONLY on top of a winning keep < 1.0
    (§5 rule 2), same gates. Off by default (--boost-grid 1.0).

Season labels: start year (Oct–Jun). The 2020 bubble restart (Jul 30 –
Oct 11 2020) belongs to 2019-20, so dates in Jul–Nov 2020 map to 2019; the
2020-21 season opened Dec 22 2020.

Usage:
    uv run python scripts/backtest_nba_elo_regression.py
    uv run python scripts/backtest_nba_elo_regression.py --keep-grid 1.0,0.75,0.6 --boost-grid 1.0,1.5,2.0
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

SECTOR = "nba"
OPEN_GAMES = 20
DEFAULT_KEEP_GRID = [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5]


def nba_season_of(d: date) -> int:
    """Start-year season label, with the 2020 bubble restart kept in 2019-20."""
    if date(2020, 7, 1) <= d <= date(2020, 11, 30):
        return 2019
    return d.year if d.month >= 7 else d.year - 1


def load_games(first_month: str) -> list:
    h2h.SECTORS[SECTOR] = {**h2h.SECTORS[SECTOR], "months": h2h._months(first_month, "2026-06")}
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
    agent.use_h2h = False   # dead live for NBA (raw-label read)
    agent.use_rest = True   # NBA is in REST_RESOLVED_SECTORS
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
    ap.add_argument("--first-month", default="2017-10", help="Burn-in starts here (first season not scored)")
    ap.add_argument("--rank-seasons", default="2018,2019,2020,2021,2022,2023")
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
