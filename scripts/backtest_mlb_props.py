"""Walk-forward backtest for the MLB player-prop projection model.

For each player's game log, at every game i ≥ MIN_GAMES we build a *point-in-
time* profile from games[:i] (season-to-date BEFORE that game — no leakage),
project the per-stat mean with the SAME neutral matchup context the live scan
uses, and read off model P(stat ≥ k) for a realistic threshold ladder. We
compare each prediction to the actual outcome (stat in game i ≥ k).

What this measures: the projection model's **calibration and signal** —
Brier vs an empirical base-rate baseline (the player's own prior frequency)
and a global base-rate baseline, plus per-stat calibration deciles and the
P(over) buckets the dashboard reads. It does NOT measure edge vs Pinnacle —
we have no historical prop lines offline; that validation happens live in
shadow (blended_true_prob vs the captured anchor). A model that can't beat the
base-rate baselines here has no business blending into the live price.

Usage:
    python scripts/backtest_mlb_props.py --season 2025 --max-players 150
    python scripts/backtest_mlb_props.py --season 2025 --json-out data/backtest/mlb_props/calibration_2025.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from evmax.clients.baseball_props_cache import _f, _ip_to_outs, fetch_game_logs
from evmax.clients.mlb_statsapi import MLBStatsClient
from evmax.models_ml.baseball_props import (
    HitterProfile,
    MatchupContext,
    PitcherProfile,
    project_prob,
)

# Threshold ladders — the integer "X+" lines Kalshi actually posts, over the
# meaningful range for each stat. Predictions are evaluated at each.
THRESHOLDS: dict[str, list[int]] = {
    "strikeouts":     [4, 5, 6, 7, 8],
    "pitching_outs":  [12, 15, 18, 21],
    "total_bases":    [1, 2, 3],
    "hits":           [1, 2],
    "home_runs":      [1],
    "rbis":           [1, 2],
    "hits_runs_rbis": [1, 2, 3],
}
PITCHER_STATS = {"strikeouts", "pitching_outs"}
HITTER_STATS = {"total_bases", "hits", "home_runs", "rbis", "hits_runs_rbis"}


# --- point-in-time profile builders from prior game-log splits -------------

def _pitcher_profile(prior: list[dict]) -> Optional[PitcherProfile]:
    bf = outs = k = hr = hits = starts = 0.0
    for g in prior:
        st = g["stat"]
        if _f(st.get("gamesStarted")) < 1:
            continue  # rate the starter off prior STARTS only
        starts += 1
        bf += _f(st.get("battersFaced"))
        outs += _f(st.get("outs")) or _ip_to_outs(st.get("inningsPitched"))
        k += _f(st.get("strikeOuts"))
        hr += _f(st.get("homeRuns"))
        hits += _f(st.get("hits"))
    if starts < 1 or bf <= 0:
        return None
    return PitcherProfile(
        batters_faced=bf, outs=outs, strike_outs=k, home_runs=hr,
        hits=hits, games_started=starts, innings_pitched=outs / 3.0,
    )


def _hitter_profile(prior: list[dict]) -> Optional[HitterProfile]:
    pa = h = dbl = tpl = hr = rbi = runs = so = tb = games = 0.0
    for g in prior:
        st = g["stat"]
        games += 1
        pa += _f(st.get("plateAppearances"))
        h += _f(st.get("hits"))
        dbl += _f(st.get("doubles"))
        tpl += _f(st.get("triples"))
        hr += _f(st.get("homeRuns"))
        rbi += _f(st.get("rbi"))
        runs += _f(st.get("runs"))
        so += _f(st.get("strikeOuts"))
        tb += _f(st.get("totalBases"))
    if pa <= 0 or games < 1:
        return None
    return HitterProfile(
        plate_appearances=pa, at_bats=0.0, hits=h, doubles=dbl, triples=tpl,
        home_runs=hr, rbi=rbi, runs=runs, strike_outs=so, total_bases=tb, games=games,
    )


def _actual(stat_type: str, g: dict) -> Optional[float]:
    st = g["stat"]
    if stat_type == "strikeouts":
        return _f(st.get("strikeOuts"))
    if stat_type == "pitching_outs":
        return _f(st.get("outs")) or _ip_to_outs(st.get("inningsPitched"))
    if stat_type == "total_bases":
        return _f(st.get("totalBases"))
    if stat_type == "hits":
        return _f(st.get("hits"))
    if stat_type == "home_runs":
        return _f(st.get("homeRuns"))
    if stat_type == "rbis":
        return _f(st.get("rbi"))
    if stat_type == "hits_runs_rbis":
        return _f(st.get("hits")) + _f(st.get("runs")) + _f(st.get("rbi"))
    return None


# --- player universe -------------------------------------------------------

async def _top_players(client: MLBStatsClient, group: str, season: int,
                       sort_key: str, n: int) -> list[int]:
    data = await client._get("/stats", params={
        "stats": "season", "group": group, "season": season,
        "sportId": 1, "playerPool": "all", "limit": 3000,
    })
    splits = (data.get("stats") or [{}])[0].get("splits", [])
    ranked = sorted(
        splits, key=lambda s: _f(s.get("stat", {}).get(sort_key)), reverse=True,
    )
    out = []
    for s in ranked:
        pid = s.get("player", {}).get("id")
        if pid:
            out.append(pid)
        if len(out) >= n:
            break
    return out


# --- accumulation ----------------------------------------------------------

class StatAccumulator:
    def __init__(self) -> None:
        self.model: list[tuple[float, int]] = []      # (pred, outcome)
        self.emp: list[tuple[float, int]] = []        # player prior-rate baseline
        self.outcomes: list[int] = []                 # for global base rate


def _walk_player(logs: list[dict], stats: set[str], min_games: int,
                 acc: dict[str, StatAccumulator],
                 profile_fn, starts_only: bool) -> None:
    for i in range(min_games, len(logs)):
        target = logs[i]
        if starts_only and _f(target["stat"].get("gamesStarted")) < 1:
            continue
        prior = logs[:i]
        prof = profile_fn(prior)
        if prof is None:
            continue
        for stat_type in stats:
            actual = _actual(stat_type, target)
            if actual is None:
                continue
            for k in THRESHOLDS[stat_type]:
                kw = {"pitcher": prof} if stat_type in PITCHER_STATS else {"hitter": prof}
                pred = project_prob(stat_type, k, MatchupContext(), **kw)
                if pred is None:
                    continue
                outcome = 1 if actual >= k else 0
                a = acc[stat_type]
                a.model.append((pred, outcome))
                a.outcomes.append(outcome)
                # empirical prior-rate baseline for this threshold
                prior_hits = sum(1 for g in prior
                                 if (_actual(stat_type, g) or -1) >= k)
                prior_n = sum(1 for g in prior if _actual(stat_type, g) is not None)
                if prior_n > 0:
                    a.emp.append((prior_hits / prior_n, outcome))


# --- metrics ---------------------------------------------------------------

def _brier(pairs: list[tuple[float, int]]) -> Optional[float]:
    if not pairs:
        return None
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def _accuracy(pairs: list[tuple[float, int]]) -> Optional[float]:
    if not pairs:
        return None
    return sum(1 for p, o in pairs if (p >= 0.5) == bool(o)) / len(pairs)


def _calibration_deciles(pairs: list[tuple[float, int]]) -> list[dict]:
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, o in pairs:
        b = min(9, int(p * 10))
        bins[b].append((p, o))
    rows = []
    for b in range(10):
        bp = bins.get(b, [])
        if not bp:
            continue
        rows.append({
            "bucket": f"{b*10}-{b*10+10}%",
            "n": len(bp),
            "pred": round(sum(p for p, _ in bp) / len(bp), 4),
            "actual": round(sum(o for _, o in bp) / len(bp), 4),
        })
    return rows


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--max-players", type=int, default=150,
                    help="top-N hitters and top-N pitchers by volume")
    ap.add_argument("--min-games", type=int, default=15)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    acc: dict[str, StatAccumulator] = defaultdict(StatAccumulator)

    async with MLBStatsClient() as client:
        hitters = await _top_players(client, "hitting", args.season, "plateAppearances", args.max_players)
        pitchers = await _top_players(client, "pitching", args.season, "battersFaced", args.max_players)
    print(f"Universe: {len(hitters)} hitters, {len(pitchers)} pitchers (season {args.season})")

    async def _do(pid: int, group: str) -> None:
        logs = await fetch_game_logs(pid, args.season, group)
        # gameLog is reverse-chronological from the API for some seasons; sort
        # ascending by date so prior = earlier games.
        logs = sorted(logs, key=lambda g: g.get("date", ""))
        if group == "pitching":
            _walk_player(logs, PITCHER_STATS, args.min_games, acc, _pitcher_profile, starts_only=True)
        else:
            _walk_player(logs, HITTER_STATS, args.min_games, acc, _hitter_profile, starts_only=False)

    # bounded concurrency (client caps at 5; gather in chunks of 20 tasks)
    tasks = [(_do, pid, "hitting") for pid in hitters] + [(_do, pid, "pitching") for pid in pitchers]
    for j in range(0, len(tasks), 20):
        await asyncio.gather(*(fn(pid, grp) for fn, pid, grp in tasks[j:j + 20]))
        print(f"  processed {min(j+20, len(tasks))}/{len(tasks)} players", end="\r")
    print()

    # report
    report: dict[str, Any] = {"season": args.season, "stats": {}}
    print(f"\n{'stat':16}{'n':>8}{'Brier(model)':>14}{'Brier(emp)':>12}{'Brier(base)':>12}{'acc':>7}")
    print("-" * 69)
    all_model: list[tuple[float, int]] = []
    for stat_type in ["strikeouts", "pitching_outs", "total_bases", "hits",
                      "home_runs", "rbis", "hits_runs_rbis"]:
        a = acc.get(stat_type)
        if not a or not a.model:
            continue
        all_model.extend(a.model)
        bm = _brier(a.model)
        be = _brier(a.emp)
        base_rate = sum(a.outcomes) / len(a.outcomes)
        bb = _brier([(base_rate, o) for o in a.outcomes])
        ac = _accuracy(a.model)
        print(f"{stat_type:16}{len(a.model):>8}{bm:>14.4f}"
              f"{(be if be is not None else float('nan')):>12.4f}{bb:>12.4f}{ac:>7.3f}")
        report["stats"][stat_type] = {
            "n": len(a.model),
            "brier_model": round(bm, 5),
            "brier_empirical": round(be, 5) if be is not None else None,
            "brier_baserate": round(bb, 5),
            "accuracy": round(ac, 4),
            "base_rate": round(base_rate, 4),
            "calibration": _calibration_deciles(a.model),
        }
    overall = _brier(all_model)
    print("-" * 69)
    print(f"{'OVERALL':16}{len(all_model):>8}{overall:>14.4f}")
    report["overall"] = {"n": len(all_model), "brier_model": round(overall, 5)}

    # per-stat calibration tables
    for stat_type, s in report["stats"].items():
        print(f"\n{stat_type} calibration (n={s['n']}, base={s['base_rate']}):")
        print(f"  {'bucket':>10}{'n':>8}{'pred':>8}{'actual':>8}")
        for r in s["calibration"]:
            print(f"  {r['bucket']:>10}{r['n']:>8}{r['pred']:>8.3f}{r['actual']:>8.3f}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved buckets → {out}")


if __name__ == "__main__":
    asyncio.run(main())
