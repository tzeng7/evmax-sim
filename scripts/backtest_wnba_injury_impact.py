#!/usr/bin/env python3
"""Walk-forward backtest — does availability-based injury impact improve the WNBA
model blend's out-of-sample Brier, and at what magnitude? (MODEL-13 validation.)

WHY this exists: WNBA moneyline is LIVE (real-bankroll), and the project's rule is
that a probability-moving change to a live sector must clear a walk-forward gate,
not calibration+tests alone. There is also strong standing evidence that injury
shoves are ~90% already priced by sharp (scripts/backtest_adjustment_scaling.py:
the Pinnacle close realizes only ~9% of the full injury shove). So the enhancement
thesis "make WNBA star-out impact bigger" is NOT assumed — it is tested here.

METHOD (mirrors the live mechanism, not a new model):
  1. run_walkforward("wnba", months) gives the genuinely-OOS model blend per game
     (elo/form update per game from empty state; efficiency/possession-sim use a
     FIXED prior-season seed, exactly like live opening day). We do NOT modify it.
  2. Per game, reconstruct player AVAILABILITY from the ESPN summary boxscore
     (per-player minutes; a rostered player at 0 min = DNP). A player is a scratch
     iff they are an established rotation member (>= MIN_APPEARANCES prior games,
     rolling role minutes >= ROTATION_MIN) who played 0 minutes this game.
  3. Map a scratch -> a win-prob impact exactly like the live injury agent:
       impact = OUT(0.045) x TIER_MULT(star1.5/starter1.0/rotation0.5) x staleness
     where staleness ramps a weeks-old absence to 0 (days since last played is the
     reported_at proxy) — the same _injury_staleness_multiplier logic. Per-team
     impact is capped, then scaled by a MAGNITUDE variant.
  4. Apply the adjustment to the blend the way apply_adjustments does
     (p_home + adj_home - adj_away, renormalized) and score Brier vs the outcome.

SIGNAL: paired dBrier(magnitude vs generic=1.0) on the INJURY SUBSET (games with a
detected scratch — injury-free games are identical across variants and only
dilute). The magnitude that minimizes held-out injury-subset Brier is the answer:
  M* <= 1.0  -> no enhancement (generic magnitude is already right / too big)
  M* > 1.0 and it holds out-of-sample -> enhancement validated at M*
Base-blend absolute quality (incl. any efficiency-seed lookahead) is held constant
across variants, so the paired magnitude ranking is robust to it.

Usage:
  # 1) seed efficiency to the PRIOR season first (avoids future lookahead):
  python scripts/seed_wnba_efficiency.py --year 2024      # for the 2425 (2025) run
  # 2) run the backtest (first pass fetches+caches ~200 games + boxscores):
  python scripts/backtest_wnba_injury_impact.py --season 2425
  python scripts/backtest_wnba_injury_impact.py --season 2425 --detail
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402

from evmax.agents.intelligence.injury_agent import (  # noqa: E402
    STATUS_IMPACT,
    TIER_MULTIPLIER,
    MAX_ADJ,
    _injury_staleness_multiplier,
)
from evmax.backtest.engine import WALKFORWARD_MONTHS  # noqa: E402
from evmax.backtest.metrics import brier_score  # noqa: E402
from evmax.backtest.sources.espn_walkforward import (  # noqa: E402
    fetch_espn_games,
    run_walkforward,
)
from evmax.matching.normalizer import NameNormalizer  # noqa: E402

# --- injury-mechanism constants (mirror the live injury agent) ---------------
_OUT_IMPACT = STATUS_IMPACT["out"]           # 0.045
_ADJ_CAP = 0.10                              # apply_adjustments _adj_cap
# WNBA rotation thresholds (40-minute game). Role = rolling avg minutes over the
# player's last _ROLL_K appearances; tier by that role.
_ROLL_K = 5
_MIN_APPEARANCES = 3
_ROTATION_MIN = 14.0
_STARTER_MIN = 22.0
_STAR_MIN = 30.0

_MAGNITUDES = [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary"
_BOX_CACHE = _REPO_ROOT / "data" / "backtest" / "cache" / "espn_wnba_boxscore"
_WF_CACHE = _REPO_ROOT / "data" / "backtest" / "wnba_injury"
_REQUEST_DELAY = 0.08


def _tier(role_min: float) -> str:
    if role_min >= _STAR_MIN:
        return "star"
    if role_min >= _STARTER_MIN:
        return "starter"
    return "rotation"


def _parse_minutes(val) -> float:
    """ESPN MIN column -> float. '--'/''/'DNP'/None -> 0.0."""
    try:
        s = str(val).strip()
        if not s or s in ("--", "-", "DNP", "NP"):
            return 0.0
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _fetch_boxscore(client: httpx.Client, event_id: str) -> Optional[dict]:
    """Cached ESPN WNBA summary fetch (settled boxscores never change)."""
    _BOX_CACHE.mkdir(parents=True, exist_ok=True)
    path = _BOX_CACHE / f"{event_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            pass
    try:
        r = client.get(_SUMMARY_URL, params={"event": event_id})
        r.raise_for_status()
        data = r.json()
    except Exception:  # noqa: BLE001 — a missing boxscore just means no injury signal
        return None
    time.sleep(_REQUEST_DELAY)
    try:
        path.write_text(json.dumps(data))
    except OSError:
        pass
    return data


def _team_minutes(summary: dict, norm: NameNormalizer) -> dict[str, dict[str, float]]:
    """Return {canonical_team_key: {player_displayName: minutes}} from a summary.

    Reads boxscore.players (2 team blocks); the minutes column is labelled 'MIN'.
    Team names are normalized to the same canonical keys the walk-forward uses.
    """
    out: dict[str, dict[str, float]] = {}
    for team_box in (summary or {}).get("boxscore", {}).get("players", []):
        raw = (team_box.get("team") or {}).get("displayName", "")
        if not raw:
            continue
        tname = norm.normalize(raw)
        players: dict[str, float] = {}
        for block in team_box.get("statistics", []):
            labels = block.get("labels", [])
            try:
                min_idx = labels.index("MIN")
            except ValueError:
                min_idx = 0
            for athlete in block.get("athletes", []):
                name = (athlete.get("athlete") or {}).get("displayName", "")
                stats = athlete.get("stats", [])
                if not name:
                    continue
                mins = _parse_minutes(stats[min_idx]) if min_idx < len(stats) else 0.0
                players[name] = mins
        out[tname] = players
    return out


def _team_scratch_impact(
    team_name: str,
    minutes_now: dict[str, float],
    history: dict[str, deque],
    last_played: dict[str, date],
    game_date: date,
) -> tuple[float, list[str]]:
    """Raw injury impact for one team's scratches this game (pre-magnitude, pre-cap).

    An established rotation member (>= _MIN_APPEARANCES prior appearances, rolling
    role minutes >= _ROTATION_MIN) who played 0 minutes is a scratch; its impact is
    OUT x tier-mult x staleness (staleness from days since last played).
    """
    impact = 0.0
    notes: list[str] = []
    for player, appearances in history.items():
        if len(appearances) < _MIN_APPEARANCES:
            continue
        role_min = sum(appearances) / len(appearances)
        if role_min < _ROTATION_MIN:
            continue
        if minutes_now.get(player, 0.0) > 0.0:
            continue  # played — not a scratch
        lp = last_played.get(player)
        stale = _injury_staleness_multiplier(lp.isoformat() if lp else None, game_date)
        if stale <= 0.0:
            continue
        tier = _tier(role_min)
        p_impact = _OUT_IMPACT * TIER_MULTIPLIER[tier] * stale
        impact += p_impact
        notes.append(f"{player}({tier},role{role_min:.0f},stale{stale:.2f})")
    return impact, notes


def _apply_variant(prob_home: float, raw_home: float, raw_away: float, magnitude: float) -> float:
    """Apply the injury adjustment to the blended home prob, mirroring
    InjuryReportAgent.apply_adjustments (cap per team, additive, renormalize)."""
    # report-level cap MAX_ADJ, then per-team effective cap after magnitude
    adj_home = -min(min(raw_home, MAX_ADJ) * magnitude, _ADJ_CAP)
    adj_away = -min(min(raw_away, MAX_ADJ) * magnitude, _ADJ_CAP)
    new_home = max(0.02, min(0.98, prob_home + adj_home - adj_away))
    new_away = max(0.02, min(0.98, (1.0 - prob_home) + adj_away - adj_home))
    return new_home / (new_home + new_away)


def _load_walkforward(season: str, refresh: bool) -> list:
    """run_walkforward (slow, network) cached to a pickle keyed by season."""
    _WF_CACHE.mkdir(parents=True, exist_ok=True)
    pkl = _WF_CACHE / f"wf_{season}.pkl"
    if pkl.exists() and not refresh:
        with open(pkl, "rb") as f:
            return pickle.load(f)
    months = WALKFORWARD_MONTHS[season]["wnba"]
    print(f"  running walk-forward for wnba {season} ({months}) — first pass fetches ~200 games …")
    report = run_walkforward("wnba", months)
    results = report.results
    print(f"  {report.n_games} games, {report.n_predicted} predicted")
    with open(pkl, "wb") as f:
        pickle.dump(results, f)
    return results


def _bootstrap_ci(deltas: list[float], n_boot: int = 5000) -> tuple[float, float]:
    """95% CI of the mean of paired deltas via a fixed-seed bootstrap (no RNG import
    — uses a simple LCG so the script is deterministic)."""
    if not deltas:
        return 0.0, 0.0
    n = len(deltas)
    means = []
    state = 123456789
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            total += deltas[state % n]
        means.append(total / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2425", help="Season code (2324=2024, 2425=2025).")
    ap.add_argument("--refresh", action="store_true", help="Ignore the walk-forward pickle cache.")
    ap.add_argument("--detail", action="store_true", help="Print per-scratch-game detail.")
    args = ap.parse_args()

    # Contamination guard: print the efficiency seed season so lookahead is visible.
    try:
        eff = json.loads((_REPO_ROOT / "data" / "models" / "wnba_efficiency_state.json").read_text())
        print(f"efficiency seed source_season = {eff.get('source_season')} "
              f"(should be the season BEFORE {args.season} to avoid lookahead)")
    except Exception:  # noqa: BLE001
        print("efficiency seed: (unreadable)")

    results = _load_walkforward(args.season, args.refresh)
    months = WALKFORWARD_MONTHS[args.season]["wnba"]
    norm = NameNormalizer("wnba")  # WalkForwardResult stores canonical keys, not displayNames
    games = fetch_espn_games("wnba", months)
    eid_by_key = {
        (g["date"].isoformat(), norm.normalize(g["home"]), norm.normalize(g["away"])): g["event_id"]
        for g in games
    }

    client = httpx.Client(timeout=15.0, headers={"User-Agent": "curl/8.7.1"})

    # Rolling per-team, per-player appearance history + last-played date.
    history: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=_ROLL_K)))
    last_played: dict[str, dict[str, date]] = defaultdict(dict)

    # Per-magnitude Brier accumulators (paired over the injury subset).
    subset_probs: dict[float, list[float]] = {m: [] for m in _MAGNITUDES}
    subset_actual: list[bool] = []
    all_probs: dict[float, list[float]] = {m: [] for m in _MAGNITUDES}
    all_actual: list[bool] = []
    n_scratch_games = 0
    detail_lines: list[str] = []

    for r in results:
        if r.ensemble_prob_home is None:
            continue
        key = (r.date.isoformat(), r.home, r.away)
        eid = eid_by_key.get(key)
        minutes = {}
        if eid is not None:
            box = _fetch_boxscore(client, eid)
            if box is not None:
                minutes = _team_minutes(box, norm)

        # Detect scratches for home & away using history BEFORE this game.
        raw_home, notes_home = _team_scratch_impact(
            r.home, minutes.get(r.home, {}), history[r.home], last_played[r.home], r.date
        )
        raw_away, notes_away = _team_scratch_impact(
            r.away, minutes.get(r.away, {}), history[r.away], last_played[r.away], r.date
        )
        is_scratch_game = (raw_home > 0.0 or raw_away > 0.0) and bool(minutes)

        for m in _MAGNITUDES:
            p = _apply_variant(r.ensemble_prob_home, raw_home, raw_away, m)
            all_probs[m].append(p)
            if is_scratch_game:
                subset_probs[m].append(p)
        all_actual.append(r.home_won)
        if is_scratch_game:
            subset_actual.append(r.home_won)
            n_scratch_games += 1
            if args.detail:
                detail_lines.append(
                    f"  {r.date} {r.away} @ {r.home} | home_won={r.home_won} "
                    f"base={r.ensemble_prob_home:.3f} rawH={raw_home:.3f} rawA={raw_away:.3f} "
                    f"| H:{notes_home} A:{notes_away}"
                )

        # Update history AFTER predicting (walk-forward).
        for team, pm in minutes.items():
            for player, mins in pm.items():
                if mins > 0.0:
                    history[team][player].append(mins)
                    last_played[team][player] = r.date

    client.close()

    # --- report ---
    print(f"\nSeason {args.season}: {len(all_actual)} games, {n_scratch_games} with a detected scratch.\n")
    if args.detail:
        print("Scratch games:")
        print("\n".join(detail_lines[:80]))
        print()

    def _brier(probs, actual):
        return brier_score(probs, actual) if actual else float("nan")

    generic_all = _brier(all_probs[1.0], all_actual)
    generic_sub = _brier(subset_probs[1.0], subset_actual)

    print(f"{'Magnitude':>10} | {'Brier(all)':>11} | {'Brier(inj)':>11} | "
          f"{'dBrier/1k vs generic (inj)':>28} | {'95% CI':>18}")
    print("-" * 92)
    for m in _MAGNITUDES:
        b_all = _brier(all_probs[m], all_actual)
        b_sub = _brier(subset_probs[m], subset_actual)
        # paired deltas vs generic on the injury subset (positive = generic better,
        # i.e. this magnitude is WORSE; we want negative to beat generic)
        deltas = [
            (subset_probs[m][i] - subset_actual[i]) ** 2
            - (subset_probs[1.0][i] - subset_actual[i]) ** 2
            for i in range(len(subset_actual))
        ]
        mean_d = (sum(deltas) / len(deltas)) if deltas else 0.0
        lo, hi = _bootstrap_ci(deltas)
        tag = " <- generic" if m == 1.0 else (" <- no-injury" if m == 0.0 else "")
        print(f"{m:>10.1f} | {b_all:>11.5f} | {b_sub:>11.5f} | "
              f"{mean_d * 1000:>+28.3f} | [{lo*1000:>+6.2f},{hi*1000:>+6.2f}]{tag}")

    print("\nRead: dBrier is (this magnitude) - (generic) on the injury subset, per 1000.")
    print("Negative + CI excluding 0 = this magnitude significantly BEATS generic ->")
    print("enhancement validated at that magnitude. Otherwise generic stands (or, if")
    print("M=0 is best, injuries do not help the model — consistent with sharp pricing them).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
