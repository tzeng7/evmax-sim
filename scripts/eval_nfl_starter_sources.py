#!/usr/bin/env python3
"""Which pre-game source names the starting QB? (evmax.nfl_projections.live.pick_starter)

For every played game of --seasons, builds each team's pre-game QB evidence
(``live.starter_inputs``: nflverse depth chart before game day, the week's
injury report, the weekly roster, the last game's starter, the schedule's
projected starter) and scores several rules against the game's actual starter
(first-dropback passer):

  shipped        live.pick_starter: depth chart's highest QB not ruled out >
                 last start > schedule; a Questionable QB1 who did not start the
                 last game yields to that game's starter
  depth          shipped without the Questionable exception
  majority       depth, unless the schedule and the last start agree on another
                 available QB (rejected)
  schedule_first schedule > last start (the order before 2026-10-10)
  last_start     last game's starter (schedule when he is ruled out)

"changed" = games whose actual starter is not the team's last starter: the
games the choice matters for.

Caveats: the schedule's QB columns for completed seasons appear to be
backfilled with the actual starter (2020-23 agree 99.8-100%), so the schedule
rule is only a fair comparison on 2026. Depth charts are near-daily snapshots
from 2025 (the chart before game day is used); seasons <= 2024 have one weekly
chart, which named the starter less often than the last start did (92% vs
95-96%, 2023-24), so ``live.starter_inputs`` ignores it and those seasons
score the last-start fallback.

Data: EVMAX_NFL_PROJ_DATA=<cache dir>; downloads weekly rosters and reads
depth charts through nflreadpy.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.clients import nfl_depth_charts as dc  # noqa: E402
from evmax.nfl_projections import data, live, team_games  # noqa: E402
from evmax.nfl_projections.live import StarterInputs  # noqa: E402


def rule_depth(inp: StarterInputs) -> Optional[str]:
    return live.pick_starter(replace(inp, questionable=set()))[0]


def rule_schedule_first(inp: StarterInputs) -> Optional[str]:
    return inp.schedule or inp.last_start


def rule_last_start(inp: StarterInputs) -> Optional[str]:
    if inp.last_start and inp.last_start not in inp.ruled_out:
        return inp.last_start
    return inp.schedule or inp.last_start


def rule_majority(inp: StarterInputs) -> Optional[str]:
    """Depth chart, unless the schedule and the last start agree on another available QB."""
    pick = rule_depth(inp)
    if (inp.schedule and inp.schedule == inp.last_start and inp.schedule != pick
            and inp.schedule not in inp.ruled_out):
        return inp.schedule
    return pick


RULES = {
    "shipped": lambda inp: live.pick_starter(inp)[0],
    "depth": rule_depth,
    "majority": rule_majority,
    "schedule_first": rule_schedule_first,
    "last_start": rule_last_start,
}


def evaluate(seasons: list[int]) -> pd.DataFrame:
    tg = team_games.load_team_games(range(min(seasons) - 1, max(seasons) + 1))
    games = data.load_games()
    truth = tg.set_index(["game_id", "team"])["first_qb_id"]
    rows = []
    for season in seasons:
        data.ensure_rosters(season, max_age_hours=None if season < max(seasons) else 6.0)
        rosters, injuries = data.load_rosters([season]), data.load_injuries(season)
        chart = dc.load_qb_chart_rows(season)
        sched = games[(games["season"] == season) & games["home_score"].notna() & (games["game_type"] == "REG")]
        for (week, day), wk in sched.groupby(["week", "gameday"]):
            if week == 1:
                continue        # no last start in the season; prior-season carry-over is a separate question
            # pre-game information only: the chart before game day, no game-day inactive list
            inputs, _ = live.starter_inputs(season, int(week), wk, tg, day, injuries, rosters, qb_depth=chart,
                                            as_of=day.date(), inactive=False)
            for team, inp in inputs.items():
                gid = wk.loc[(wk["home_team"] == team) | (wk["away_team"] == team), "game_id"].iloc[0]
                actual = truth.get((gid, team))
                if not isinstance(actual, str):
                    continue
                row = {"season": season, "week": week, "game_id": gid, "team": team, "actual": actual,
                       "changed": actual != inp.last_start, "has_depth": bool(inp.depth)}
                for name, rule in RULES.items():
                    row[name] = rule(inp) == actual
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023-2026")
    ap.add_argument("--save", type=Path)
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seasons.split("-"))
    r = evaluate(list(range(lo, hi + 1)))
    if args.save:
        r.to_parquet(args.save, index=False)
    rules = list(RULES)
    for label, sub in (("all team-games", r), ("changed starter", r[r["changed"]])):
        print(f"\n{label}: accuracy by season (n)")
        g = sub.groupby("season")
        t = g[rules].mean().mul(100).round(1)
        t["n"] = g.size()
        print(t.to_string())
    ch = r[r["changed"]]
    for a, b in (("shipped", "depth"), ("shipped", "majority"), ("shipped", "schedule_first")):
        print(f"changed-starter games where {a} and {b} disagree: {a} right {int((ch[a] & ~ch[b]).sum())}, "
              f"{b} right {int((ch[b] & ~ch[a]).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
