"""Walk-forward sweep of the NFL offseason Elo regression coefficient.

Why: only WNBA regresses generic Elo across the offseason today. NFL's
`elo_state.json['nfl']` carries raw February ratings into September at 0.20
blend weight, and docs/SEASON_START.md §5 names NFL the #2 target — with the
house rule that the coefficient is SWEPT per sector on a walk-forward replay,
never copied from WNBA's 35% (538's NFL prior is ⅓ toward the mean).

Protocol (leak-free, mirrors scripts/backtest_nhl_elo.py):
  - Data: nflreadpy schedules (REG + playoffs, final scores) for
    --first-season..--holdout-season. Team codes map to evmax canonical
    nicknames via evmax/sectors/aliases/nfl.yaml (+ legacy OAK/SD/STL/LA).
  - Replay uses the PRODUCTION `EloModelAgent.update()` / `_win_probs()`:
    K=25, HOME_ADVANTAGE_ELO=48, MOV + SOS multipliers, H2H nudge, and the
    recency-K layer with the agent's clock patched to each game's date (so
    "last 14 days" means the replay's last 14 days, not 2026's). The only
    production layer excluded is the rest-day bonus (it reads the live
    form_state.json from disk per call — a within-week effect with no bearing
    on the offseason boundary).
  - `keep` = fraction of the deviation from 1500 RETAINED across the
    boundary (1.0 = today's no-regression behaviour; 0.667 = 538's ⅓-toward-
    mean). Applied before each season's first game; season_games reset to 0.
  - Scored on P(home win) for weeks 1–6 of each season (the window the
    boundary affects) with full-season Brier as a secondary column. Burn-in
    seasons are fed but not scored.
  - RANK window = --rank-seasons, CONFIRM = --confirm-season, HOLDOUT =
    --holdout-season. The winner is fixed on RANK (must not lose on CONFIRM)
    and the HOLDOUT number is reported once, never used to choose.
  - Optional second stage: an early-season K boost (EARLY_K_BOOST /
    EARLY_K_DECAY_GAMES) swept ONLY on top of the winning `keep`, same gates.
    Rule 2 of §5: a boost without the season_games reset just amplifies
    late-season noise, so it is never evaluated without regression.

Usage:
    uv run python scripts/backtest_nfl_elo_regression.py
    uv run python scripts/backtest_nfl_elo_regression.py --keep-grid 1.0,0.8,0.667,0.5 --boost-grid 1.0,2.0
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

import yaml  # noqa: E402

import evmax.agents.models.elo_agent as elo_mod  # noqa: E402
from evmax.agents.models.elo_agent import DEFAULT_ELO, EloModelAgent  # noqa: E402

SECTOR = "nfl"
OPENING_WEEKS = 6
LEGACY_CODES = {"la": "rams", "stl": "rams", "oak": "raiders", "sd": "chargers"}
DEFAULT_KEEP_GRID = [1.0, 0.9, 0.8, 0.75, 0.667, 0.6, 0.5]
DEFAULT_BOOST_GRID = [1.0, 1.5, 2.0, 3.0]
DEFAULT_DECAY_GRID = [4, 6, 8]


def load_code_map() -> dict[str, str]:
    y = yaml.safe_load((_REPO_ROOT / "evmax" / "sectors" / "aliases" / "nfl.yaml").read_text())
    aliases = y.get("aliases", y)
    m = {str(k).lower(): v for k, v in aliases.items() if isinstance(v, str)}
    m.update(LEGACY_CODES)
    return m


def load_games(first_season: int, last_season: int) -> list[dict]:
    import nflreadpy as nfl

    codes = load_code_map()
    df = nfl.load_schedules(seasons=list(range(first_season, last_season + 1)))
    df = df.filter(df["home_score"].is_not_null() & df["away_score"].is_not_null())
    df = df.sort(["gameday", "gametime", "game_id"])
    games: list[dict] = []
    for r in df.iter_rows(named=True):
        home = codes.get(str(r["home_team"]).lower())
        away = codes.get(str(r["away_team"]).lower())
        if not home or not away:
            raise SystemExit(f"unmapped team code in {r['game_id']}: {r['home_team']} / {r['away_team']}")
        games.append({
            "season": int(r["season"]), "week": int(r["week"]), "game_type": r["game_type"],
            "gameday": date.fromisoformat(str(r["gameday"])), "home": home, "away": away,
            "hs": int(r["home_score"]), "as": int(r["away_score"]),
        })
    return games


class _ReplayClock(date):
    """`date` stand-in whose today() follows the replay cursor (recency-K fidelity)."""

    _now: date = date(2000, 1, 1)

    @classmethod
    def today(cls) -> date:  # type: ignore[override]
        return cls._now


def make_agent() -> EloModelAgent:
    agent = EloModelAgent()
    agent._state = {}                      # cold start; production state untouched
    agent._rest_elo_bonus = lambda sector, team: 0.0  # type: ignore[method-assign]
    return agent


def regress(agent: EloModelAgent, keep: float) -> None:
    st = agent._sector_state(SECTOR)
    for team, rating in list(st["ratings"].items()):
        st["ratings"][team] = round(DEFAULT_ELO + keep * (rating - DEFAULT_ELO), 2)
        st["season_games"][team] = 0


def walk_forward(games: list[dict], keep: float, boost: float, decay: int,
                 score_seasons: set[int]) -> dict[int, dict[str, list[float]]]:
    """Replay all games; return per-season Brier lists {season: {'open': [...], 'full': [...]}}."""
    if boost > 1.0:
        elo_mod.EARLY_K_BOOST[SECTOR] = boost
        elo_mod.EARLY_K_DECAY_GAMES[SECTOR] = decay
    else:
        elo_mod.EARLY_K_BOOST.pop(SECTOR, None)
        elo_mod.EARLY_K_DECAY_GAMES.pop(SECTOR, None)
    agent = make_agent()
    out: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"open": [], "full": []})
    cur: Optional[int] = None
    for g in games:
        if g["season"] != cur:
            if cur is not None:
                regress(agent, keep)
            cur = g["season"]
        _ReplayClock._now = g["gameday"]
        if g["season"] in score_seasons and g["hs"] != g["as"]:
            p_home, _, _ = agent._win_probs(SECTOR, g["home"], g["away"])
            b = (p_home - (1.0 if g["hs"] > g["as"] else 0.0)) ** 2
            out[g["season"]]["full"].append(b)
            if g["game_type"] == "REG" and g["week"] <= OPENING_WEEKS:
                out[g["season"]]["open"].append(b)
        agent.update(g["home"], g["away"], g["hs"], g["as"], SECTOR, event_date=g["gameday"].isoformat())
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pooled(res: dict[int, dict[str, list[float]]], seasons: list[int], key: str) -> tuple[float, int]:
    vals = [b for s in seasons for b in res.get(s, {}).get(key, [])]
    return _mean(vals), len(vals)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first-season", type=int, default=2015, help="Burn-in starts here (not scored)")
    ap.add_argument("--rank-seasons", default="2019,2020,2021,2022,2023")
    ap.add_argument("--confirm-season", type=int, default=2024)
    ap.add_argument("--holdout-season", type=int, default=2025)
    ap.add_argument("--keep-grid", default=",".join(str(k) for k in DEFAULT_KEEP_GRID))
    ap.add_argument("--boost-grid", default=",".join(str(b) for b in DEFAULT_BOOST_GRID))
    ap.add_argument("--decay-grid", default=",".join(str(d) for d in DEFAULT_DECAY_GRID))
    args = ap.parse_args(argv)

    rank_seasons = [int(s) for s in args.rank_seasons.split(",") if s]
    confirm = args.confirm_season
    holdout = args.holdout_season
    keep_grid = [float(k) for k in args.keep_grid.split(",") if k]
    boost_grid = [float(b) for b in args.boost_grid.split(",") if b]
    decay_grid = [int(d) for d in args.decay_grid.split(",") if d]
    score_seasons = set(rank_seasons) | {confirm, holdout}

    games = load_games(args.first_season, holdout)
    print(f"Loaded {len(games)} games {args.first_season}–{holdout} "
          f"(burn-in {args.first_season}–{min(rank_seasons) - 1}, rank {rank_seasons}, "
          f"confirm {confirm}, holdout {holdout}); opening window = REG weeks 1–{OPENING_WEEKS}")

    saved_date = elo_mod.date
    saved_boost = dict(elo_mod.EARLY_K_BOOST)
    saved_decay = dict(elo_mod.EARLY_K_DECAY_GAMES)
    elo_mod.date = _ReplayClock  # type: ignore[assignment]
    try:
        # ---- Stage 1: regression coefficient, no boost ----
        print(f"\n{'keep':>6} {'rank open':>10} {'n':>5} {'rank full':>10}  {'conf open':>10} {'n':>4} {'conf full':>10}")
        rows = []
        results: dict[float, dict] = {}
        for keep in keep_grid:
            res = walk_forward(games, keep, 1.0, 0, score_seasons)
            results[keep] = res
            r_open, r_n = _pooled(res, rank_seasons, "open")
            r_full, _ = _pooled(res, rank_seasons, "full")
            c_open, c_n = _pooled(res, [confirm], "open")
            c_full, _ = _pooled(res, [confirm], "full")
            rows.append({"keep": keep, "r_open": r_open, "r_full": r_full, "c_open": c_open, "c_full": c_full})
            print(f"{keep:6.3f} {r_open:10.4f} {r_n:5d} {r_full:10.4f}  {c_open:10.4f} {c_n:4d} {c_full:10.4f}")
        base = next(r for r in rows if r["keep"] == 1.0) if any(r["keep"] == 1.0 for r in rows) else rows[0]
        ranked = sorted(rows, key=lambda r: r["r_open"])
        winner = ranked[0]
        confirmed = winner["c_open"] <= base["c_open"] + 1e-12
        print(f"\nStage 1 winner by RANK opening Brier: keep={winner['keep']:.3f} "
              f"(Δ vs no-regression {1000 * (base['r_open'] - winner['r_open']):+.2f}/1000); "
              f"CONFIRM {'holds' if confirmed else 'FAILS'} "
              f"({1000 * (base['c_open'] - winner['c_open']):+.2f}/1000 vs baseline)")
        per_season = {s: (_mean(results[winner['keep']][s]['open']), _mean(results[base['keep']][s]['open']))
                      for s in rank_seasons + [confirm]}
        print("  per-season opening Brier (winner vs baseline): " +
              ", ".join(f"{s}: {w:.4f} vs {b:.4f}" for s, (w, b) in per_season.items()))

        chosen_keep = winner["keep"] if confirmed else base["keep"]

        # ---- Stage 2: early-season K boost on top of the chosen keep ----
        chosen_boost, chosen_decay = 1.0, 0
        if chosen_keep < 1.0 and any(b > 1.0 for b in boost_grid):
            print(f"\nStage 2 — early-K boost on keep={chosen_keep:.3f}")
            print(f"{'boost':>6} {'decay':>6} {'rank open':>10} {'rank full':>10}  {'conf open':>10} {'conf full':>10}")
            brows = []
            for boost in boost_grid:
                for decay in (decay_grid if boost > 1.0 else [0]):
                    res = walk_forward(games, chosen_keep, boost, decay, score_seasons)
                    r_open, _ = _pooled(res, rank_seasons, "open")
                    r_full, _ = _pooled(res, rank_seasons, "full")
                    c_open, _ = _pooled(res, [confirm], "open")
                    c_full, _ = _pooled(res, [confirm], "full")
                    brows.append({"boost": boost, "decay": decay, "r_open": r_open, "r_full": r_full,
                                  "c_open": c_open, "c_full": c_full})
                    print(f"{boost:6.2f} {decay:6d} {r_open:10.4f} {r_full:10.4f}  {c_open:10.4f} {c_full:10.4f}")
            nob = next(r for r in brows if r["boost"] == 1.0)
            bwin = sorted(brows, key=lambda r: r["r_open"])[0]
            if bwin["boost"] > 1.0 and bwin["c_open"] <= nob["c_open"] + 1e-12 and bwin["r_open"] < nob["r_open"]:
                chosen_boost, chosen_decay = bwin["boost"], bwin["decay"]
                print(f"Stage 2 winner: boost={chosen_boost} decay={chosen_decay} "
                      f"(rank Δ {1000 * (nob['r_open'] - bwin['r_open']):+.2f}/1000, "
                      f"confirm Δ {1000 * (nob['c_open'] - bwin['c_open']):+.2f}/1000) — SHIP")
            else:
                print(f"Stage 2: no boost clears both gates (best boost={bwin['boost']} decay={bwin['decay']}, "
                      f"rank Δ {1000 * (nob['r_open'] - bwin['r_open']):+.2f}/1000, "
                      f"confirm Δ {1000 * (nob['c_open'] - bwin['c_open']):+.2f}/1000) — keep boost OFF")

        # ---- Holdout: reported once for the fixed choice + the baseline ----
        print(f"\nHOLDOUT {holdout} (fixed before this pass; reported once)")
        for label, keep, boost, decay in (
            ("no-regression baseline", 1.0, 1.0, 0),
            (f"chosen keep={chosen_keep:.3f}" + (f" + boost {chosen_boost}/{chosen_decay}" if chosen_boost > 1.0 else ""),
             chosen_keep, chosen_boost, chosen_decay),
        ):
            res = walk_forward(games, keep, boost, decay, score_seasons)
            h_open, h_n = _pooled(res, [holdout], "open")
            h_full, h_fn = _pooled(res, [holdout], "full")
            print(f"  {label:44} opening Brier={h_open:.4f} (n={h_n})  full-season Brier={h_full:.4f} (n={h_fn})")
        print(f"\nVERDICT: keep={chosen_keep:.3f}, boost={chosen_boost}, decay={chosen_decay}")
    finally:
        elo_mod.date = saved_date  # type: ignore[assignment]
        elo_mod.EARLY_K_BOOST.clear(); elo_mod.EARLY_K_BOOST.update(saved_boost)
        elo_mod.EARLY_K_DECAY_GAMES.clear(); elo_mod.EARLY_K_DECAY_GAMES.update(saved_decay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
