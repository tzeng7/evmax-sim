"""Phase 1 validation backtest for NFL efficiency agent.

Compares the OLD NFL blend (elo + form + poisson at class defaults) against
the NEW NFL blend (elo + form + nfl_efficiency at the per-sector override
weights) on a 3-season walk-forward.

How leak is avoided:
  - elo / form / poisson start from EMPTY state and are walk-forward updated
    as games are observed.
  - nfl_efficiency state is REGENERATED periodically during the walk-forward
    to simulate the live re-seed cadence. At each re-seed, we slice the
    pre-loaded PBP frame to rows where game_date < current cutoff and
    recompute opponent-adjusted EPA from that slice. This gives a per-cutoff
    "as-if-live-on-date-X" snapshot with zero future leak.

Re-seed cadence:
  Default every 7 days (mirrors a typical weekly re-seed that would run on
  off-day during the season). A first-game-of-season prediction always
  forces a re-seed using only prior-seasons data.

Usage:
    python scripts/backtest_nfl_efficiency.py
    python scripts/backtest_nfl_efficiency.py --reseed-days 14
    python scripts/backtest_nfl_efficiency.py --no-reseed  # frozen-state baseline
    python scripts/backtest_nfl_efficiency.py --restore-only  # restore backup
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import polars as pl
import nflreadpy as nfl

from scripts.seed_nfl_efficiency import filter_valid_pbp, compute_team_stats
from evmax.agents.models.nfl_efficiency_agent import (
    NflEfficiencyModelAgent,
    STATE_PATH as NFL_EFF_STATE_PATH,
)
from evmax.backtest.metrics import brier_score, accuracy_score, log_loss_score
from evmax.backtest.sources.espn_walkforward import (
    fetch_espn_games,
    _elo_predict,
    _form_predict,
    _poisson_predict,
    _elo_update,
    _form_update,
    _poisson_update,
)
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook

import asyncio

# Months for each evaluated season (mirrors WALKFORWARD_MONTHS in engine.py)
SEASON_MONTHS: dict[str, list[str]] = {
    "2324": [f"2023{m:02d}" for m in range(9, 13)] + [f"2024{m:02d}" for m in range(1, 3)],
    "2425": [f"2024{m:02d}" for m in range(9, 13)] + [f"2025{m:02d}" for m in range(1, 3)],
    "2526": [f"2025{m:02d}" for m in range(9, 13)] + [f"2026{m:02d}" for m in range(1, 3)],
}

# Old NFL blend (class defaults — what was running pre-Phase-0):
#   elo 0.35 · form 0.25 · poisson 0.30  (sum 0.90; normalized at blend time)
OLD_NFL_WEIGHTS = {"elo": 0.35, "form": 0.25, "poisson": 0.30, "nfl_efficiency": 0.0}

# Candidate new blends — sweep efficiency weight to find the best mix.
# Names appear in the metric tables exactly as written here.
CANDIDATE_BLENDS: dict[str, dict[str, float]] = {
    "NEW v1: nfl_eff 0.50 / elo 0.30 / form 0.20": {
        "elo": 0.30, "form": 0.20, "poisson": 0.0, "nfl_efficiency": 0.50},
    "NEW v2: nfl_eff 0.40 / elo 0.35 / form 0.25": {
        "elo": 0.35, "form": 0.25, "poisson": 0.0, "nfl_efficiency": 0.40},
    "NEW v3: nfl_eff 0.30 / elo 0.40 / form 0.30": {
        "elo": 0.40, "form": 0.30, "poisson": 0.0, "nfl_efficiency": 0.30},
    "NEW v4: nfl_eff 0.20 / elo 0.45 / form 0.35": {
        "elo": 0.45, "form": 0.35, "poisson": 0.0, "nfl_efficiency": 0.20},
    "ALT: elo+form only (no efficiency, no poisson)": {
        "elo": 0.55, "form": 0.45, "poisson": 0.0, "nfl_efficiency": 0.0},
}

# Backwards-compat: kept so any external code referencing this still works.
NEW_NFL_WEIGHTS = CANDIDATE_BLENDS["NEW v1: nfl_eff 0.50 / elo 0.30 / form 0.20"]


@dataclass
class GamePrediction:
    season: str
    game_date: date
    home: str
    away: str
    home_won: bool
    elo: Optional[float] = None
    form: Optional[float] = None
    poisson: Optional[float] = None
    nfl_efficiency: Optional[float] = None


def _ensemble(probs: dict[str, Optional[float]], weights: dict[str, float]) -> Optional[float]:
    """Weighted average of available model probs. Returns None if no models contributed."""
    total_w = 0.0
    total_p = 0.0
    for name, p in probs.items():
        w = weights.get(name, 0.0)
        if p is not None and w > 0:
            total_w += w
            total_p += p * w
    if total_w == 0:
        return None
    return total_p / total_w


def _nfl_efficiency_predict(
    agent: NflEfficiencyModelAgent,
    home: str,
    away: str,
) -> Optional[float]:
    """Run the agent on a synthetic market/sharp pair and return P(home)."""
    market = PredictionMarket(
        id="bt", market_id="bt", event_id=f"bt_{home}_{away}",
        sector="nfl", team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=0.55, no_price=0.45,
    )
    sharp = SharpOdds(
        event_id=f"bt_{home}_{away}", book=SharpBook.pinnacle, sector="nfl",
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=1.82, outcome_b_decimal=2.22,
        true_prob_a=0.55, true_prob_b=0.45,
    )
    try:
        pred = asyncio.run(agent.predict_pair(market, sharp))
        return pred.true_prob_a if pred else None
    except Exception:
        return None


def _backup_state() -> Optional[Path]:
    """Save the current NFL efficiency state file to a sibling .backup so we
    can restore it once the backtest is done. Returns the backup path."""
    if not NFL_EFF_STATE_PATH.exists():
        return None
    backup = NFL_EFF_STATE_PATH.with_suffix(".json.backtest_backup")
    shutil.copy(NFL_EFF_STATE_PATH, backup)
    print(f"[backup] saved {NFL_EFF_STATE_PATH.name} → {backup.name}")
    return backup


def _restore_state(backup: Optional[Path]) -> None:
    if backup is None or not backup.exists():
        print("[restore] no backup to restore (was state file missing before run?)")
        return
    shutil.move(backup, NFL_EFF_STATE_PATH)
    print(f"[restore] restored {NFL_EFF_STATE_PATH.name} from backup")


def _reseed(seasons: list[int]) -> None:
    """Re-run the seeder for the given season list, blocking until done."""
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "seed_nfl_efficiency.py"),
           "--seasons", ",".join(str(s) for s in seasons)]
    print(f"[seed] running: {' '.join(cmd[-3:])}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"seed failed with exit {res.returncode}")
    # Print just the top/bottom team summary lines from the seeder for context
    for line in res.stdout.splitlines():
        if "rows" in line or "Wrote" in line or "net=" in line.lower()[:40]:
            print(f"  {line}")


def _print_metric_row(label: str, brier: float, acc: float, ll: float, n: int) -> None:
    if n == 0:
        print(f"  {label:30s}  n={n:5d}  (no predictions)")
        return
    print(f"  {label:30s}  n={n:5d}  Brier={brier:.4f}  Acc={acc*100:5.1f}%  LogLoss={ll:.4f}")


def _evaluate(
    preds: list[GamePrediction],
    season_filter: Optional[set[str]] = None,
) -> dict[str, dict[str, float]]:
    """Compute per-model and per-ensemble metrics over the given season filter."""
    if season_filter is not None:
        preds = [p for p in preds if p.season in season_filter]

    out: dict[str, dict[str, float]] = {}

    for model_name in ["elo", "form", "poisson", "nfl_efficiency"]:
        ps = []
        actuals = []
        for p in preds:
            v = getattr(p, model_name)
            if v is not None:
                ps.append(v)
                actuals.append(p.home_won)
        out[model_name] = {
            "n": len(ps),
            "brier": brier_score(ps, actuals),
            "acc": accuracy_score(ps, actuals),
            "ll": log_loss_score(ps, actuals),
        }

    blends_to_eval: list[tuple[str, dict[str, float]]] = [
        ("OLD ensemble (elo+form+poisson)", OLD_NFL_WEIGHTS),
    ]
    blends_to_eval.extend(CANDIDATE_BLENDS.items())

    for blend_name, weights in blends_to_eval:
        ps = []
        actuals = []
        for p in preds:
            probs = {
                "elo": p.elo,
                "form": p.form,
                "poisson": p.poisson,
                "nfl_efficiency": p.nfl_efficiency,
            }
            ens = _ensemble(probs, weights)
            if ens is not None:
                ps.append(ens)
                actuals.append(p.home_won)
        out[blend_name] = {
            "n": len(ps),
            "brier": brier_score(ps, actuals),
            "acc": accuracy_score(ps, actuals),
            "ll": log_loss_score(ps, actuals),
        }

    # Baselines
    if preds:
        rate = sum(1 for p in preds if p.home_won) / len(preds)
        baseline_brier = sum((rate - float(p.home_won)) ** 2 for p in preds) / len(preds)
        out["baseline (always P=home_rate)"] = {
            "n": len(preds), "brier": baseline_brier, "acc": rate, "ll": 0.0,
        }
    return out


def _print_table(title: str, metrics: dict[str, dict[str, float]]) -> None:
    print()
    print(f"--- {title} ---")
    for name, m in metrics.items():
        _print_metric_row(name, m["brier"], m["acc"], m["ll"], int(m["n"]))


def _reseed_from_pbp(
    nfl_eff: NflEfficiencyModelAgent,
    pbp_filtered: pl.DataFrame,
    cutoff: date,
    current_season: int,
) -> int:
    """Re-derive nfl_eff state in-place from PBP rows strictly before cutoff.

    Returns the row count used (0 if the slice was empty — caller decides
    whether to keep the existing state or skip prediction).
    """
    slice_df = pbp_filtered.filter(pl.col("game_date") < pl.lit(cutoff))
    if len(slice_df) == 0:
        return 0
    teams = compute_team_stats(slice_df, current_season)
    nfl_eff._state["nfl"] = {
        "teams": teams,
        "fetched_at": cutoff.isoformat(),
        "seasons_used": sorted(slice_df["season"].unique().to_list()),
    }
    return len(slice_df)


def _season_of(game_date: date) -> int:
    """NFL season starts in September. A game on Jan/Feb belongs to the
    season that started in the prior calendar year."""
    return game_date.year if game_date.month >= 7 else game_date.year - 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 backtest validation for nfl_efficiency")
    ap.add_argument("--restore-only", action="store_true",
                    help="Just restore from .backtest_backup if one exists, then exit")
    ap.add_argument("--seasons", type=str, default="2324,2425,2526",
                    help="Comma-separated season codes (default: 2324,2425,2526)")
    ap.add_argument("--reseed-days", type=int, default=7,
                    help="Re-seed cadence in days during walk-forward (default 7 = weekly).")
    ap.add_argument("--no-reseed", action="store_true",
                    help="Disable intra-season re-seeding. State is frozen to a single "
                         "pre-eval-period snapshot per season. Useful as a baseline.")
    args = ap.parse_args()

    if args.restore_only:
        backup = NFL_EFF_STATE_PATH.with_suffix(".json.backtest_backup")
        _restore_state(backup if backup.exists() else None)
        return 0

    seasons = args.seasons.split(",")
    reseed_mode = "frozen-pre-season" if args.no_reseed else f"every {args.reseed_days}d"
    print(f"Phase 1 backtest — seasons: {seasons}  re-seed: {reseed_mode}")
    print(f"  evaluation gate: 2526 only (no leak — re-seeds use game_date < cutoff)")
    print()

    backup = _backup_state()

    try:
        # Load PBP for all relevant seasons ONCE — re-seeding is then a polars
        # filter-and-aggregate over this frame, no network round-trips per cutoff.
        # nflreadpy indexes by NFL season (2025 = Sep 2025-Feb 2026), so
        # season=2025 covers the entire 2526 evaluation period.
        pbp_seasons = list(range(2020, 2026))
        print(f"[pbp] loading PBP for seasons {pbp_seasons[0]}-{pbp_seasons[-1]}...")
        pbp_raw = nfl.load_pbp(seasons=pbp_seasons)
        print(f"  raw rows: {len(pbp_raw):,}")
        pbp = filter_valid_pbp(pbp_raw)
        # Ensure game_date is a Date type for filtering
        if pbp.schema["game_date"] != pl.Date:
            pbp = pbp.with_columns(pl.col("game_date").cast(pl.Date))
        print(f"  rows after filter: {len(pbp):,}")
        print()

        # Initialize the nfl_efficiency agent with empty state — first re-seed
        # populates it before any predictions are made.
        nfl_eff = NflEfficiencyModelAgent()
        nfl_eff._state = {}

        # Walk-forward fresh elo / form / poisson
        elo = EloModelAgent(); elo._state = {}
        form = FormModelAgent(); form._state = {}
        poisson = PoissonModelAgent(); poisson._state = {}

        norm = NameNormalizer("nfl")

        # Fetch ALL months across the seasons in one go, then iterate chronologically
        all_games_by_season: dict[str, list[dict]] = {}
        for s in seasons:
            months = SEASON_MONTHS.get(s, [])
            if not months:
                print(f"  [warn] no months mapped for season {s}")
                continue
            games = fetch_espn_games("nfl", months)
            all_games_by_season[s] = games
            print(f"[fetch] season {s}: {len(games)} games")

        all_preds: list[GamePrediction] = []
        last_seed_date: Optional[date] = None
        last_seed_season: Optional[str] = None
        n_reseeds = 0

        for s in seasons:
            for g in all_games_by_season.get(s, []):
                home = norm.normalize(g["home"])
                away = norm.normalize(g["away"])
                if not home or not away:
                    continue
                home_won = g["home_score"] > g["away_score"]
                game_date = g["date"]

                # Decide whether to re-seed before this prediction
                need_reseed = False
                if last_seed_date is None:
                    need_reseed = True
                elif s != last_seed_season:
                    # New season starts — always force a re-seed at the boundary
                    need_reseed = True
                elif not args.no_reseed and (game_date - last_seed_date).days >= args.reseed_days:
                    need_reseed = True

                if need_reseed:
                    used = _reseed_from_pbp(
                        nfl_eff, pbp, cutoff=game_date,
                        current_season=_season_of(game_date),
                    )
                    last_seed_date = game_date
                    last_seed_season = s
                    if used > 0:
                        n_reseeds += 1

                elo_p = _elo_predict(elo, "nfl", home, away)
                form_p = _form_predict(form, "nfl", home, away, game_date)
                pois_p = _poisson_predict(poisson, "nfl", home, away)
                nfl_eff_p = _nfl_efficiency_predict(nfl_eff, home, away)

                all_preds.append(GamePrediction(
                    season=s, game_date=game_date, home=home, away=away,
                    home_won=home_won,
                    elo=elo_p, form=form_p, poisson=pois_p, nfl_efficiency=nfl_eff_p,
                ))

                # Walk-forward updates (after prediction, before next game)
                _elo_update(elo, "nfl", home, away, home_won, game_date)
                _form_update(form, "nfl", home, away, home_won, game_date)
                _poisson_update(poisson, "nfl", home, away, g["home_score"], g["away_score"])

        print(f"\n[reseed] performed {n_reseeds} re-seeds across the walk-forward")

        print(f"\n[predict] {len(all_preds)} games predicted across {len(seasons)} seasons")

        # Per-season + combined metrics
        for s in seasons:
            metrics = _evaluate(all_preds, season_filter={s})
            _print_table(f"Season {s}", metrics)

        combined = _evaluate(all_preds)
        _print_table("Combined (all seasons)", combined)

        # Gate summary table — best blend per evaluation period
        print()
        print("=" * 78)
        print("PHASE 1 GATE — Δ vs OLD (positive = new variant beats old blend)")
        print("=" * 78)
        for label, sf in [("Combined (3 seasons)", None), ("2526 only (most recent)", {"2526"})]:
            metrics = _evaluate(all_preds, season_filter=sf)
            old_b = metrics["OLD ensemble (elo+form+poisson)"]["brier"]
            old_a = metrics["OLD ensemble (elo+form+poisson)"]["acc"]
            print(f"\n  {label}  (OLD: Brier {old_b:.4f}  Acc {old_a*100:.1f}%)")
            for name in CANDIDATE_BLENDS:
                m = metrics[name]
                d_brier = old_b - m["brier"]
                d_acc = m["acc"] - old_a
                marker = "  "
                if d_brier > 0.001 and d_acc >= -0.005:
                    marker = "✓ "
                elif d_brier > 0:
                    marker = "~ "
                else:
                    marker = "✗ "
                print(f"    {marker}{name:60s}  ΔBrier={d_brier:+.4f}  ΔAcc={d_acc*100:+.2f}pp")
        print("=" * 78)
        print("Legend: ✓ wins on both, ~ wins on Brier only, ✗ does not improve")

    finally:
        _restore_state(backup)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
