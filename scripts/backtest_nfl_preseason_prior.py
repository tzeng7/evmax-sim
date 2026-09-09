"""Walk-forward screen for the NFL efficiency preseason-prior ramp.

Weeks 1-N of a season the efficiency seed still maxes at last season, so today
the model BLANKS (nfl_state_is_stale_for_today) and the blend runs on elo+sharp
only. `PRESEASON_PRIOR_ENABLED` instead fires off the SEASON_DECAY-weighted
prior (2025-dominant early) at reduced confidence — the "carry a regressed prior
instead of going dark" approach strong early-season models use.

This screen asks: does that prior-only efficiency prediction have STANDALONE
skill in the early weeks, or is it noise? For each holdout season it builds the
"stale week-1 seed" (prior seasons only, exactly what the live state looks like
before the first in-season reseed), predicts each early-season game from it, and
scores Brier + accuracy against a home-field baseline.

Interpretation:
  * prior-seed Brier well below the baseline → firing the prior beats blanking;
    worth enabling and then CLV-gating on prod (the sharp anchor still prices
    preseason info, so promotion is on `cleanup shadow clv nfl`, NOT on Brier).
  * prior-seed Brier ≈ baseline → the prior adds nothing early; keep the blank.

Leak-free: the seed for season S uses ONLY seasons < S.

Usage:
    uv run python scripts/backtest_nfl_preseason_prior.py --holdout 2023,2024,2025 --weeks 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.nfl_efficiency_agent import (  # noqa: E402
    EPA_MARGIN_PTS, SR_MARGIN_PTS, HOME_EDGE_PTS, SCORE_STDEV,
    NFL_ABBREV_TO_NAME, _normal_cdf,
)


def _prior_seed(prior_seasons: list[int]):
    """Team stats from prior seasons only — the stale week-1 state."""
    import nflreadpy as nfl
    import polars as pl
    from scripts.seed_nfl_efficiency import compute_team_stats, filter_valid_pbp

    frames = []
    for s in prior_seasons:
        try:
            frames.append(nfl.load_pbp(seasons=[s]))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skip prior season {s}: {e}", file=sys.stderr)
    if not frames:
        return {}
    pbp = pl.concat(frames, how="diagonal")
    # current_season = the season we will PREDICT (one after the newest prior),
    # so SEASON_DECAY weights the newest prior at 0.45 exactly as a real week-1
    # seed would.
    return compute_team_stats(filter_valid_pbp(pbp), max(prior_seasons) + 1)


def _early_games(season: int, max_week: int):
    import nflreadpy as nfl
    import polars as pl
    pbp = nfl.load_pbp(seasons=[season])
    g = (
        pbp.filter((pl.col("season_type") == "REG") & (pl.col("week") <= max_week))
        .group_by(["game_id", "home_team", "away_team", "week"])
        .agg([pl.col("home_score").max().alias("hs"), pl.col("away_score").max().alias("as_")])
    )
    return [r for r in g.iter_rows(named=True)
            if r["hs"] is not None and r["as_"] is not None and r["hs"] != r["as_"]]


def _pred(seed: dict, home_abbr: str, away_abbr: str):
    sa = seed.get(NFL_ABBREV_TO_NAME.get(home_abbr, ""))
    sb = seed.get(NFL_ABBREV_TO_NAME.get(away_abbr, ""))
    if not sa or not sb:
        return None
    d_epa = (sa["off_epa_adj"] - sa["def_epa_adj"]) - (sb["off_epa_adj"] - sb["def_epa_adj"])
    d_sr = ((sa.get("off_success_adj", 0.0) - sa.get("def_success_adj", 0.0))
            - (sb.get("off_success_adj", 0.0) - sb.get("def_success_adj", 0.0)))
    margin = EPA_MARGIN_PTS * d_epa + SR_MARGIN_PTS * d_sr + HOME_EDGE_PTS
    return min(0.98, max(0.02, _normal_cdf(margin / SCORE_STDEV)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", default="2023,2024,2025")
    ap.add_argument("--weeks", type=int, default=4, help="early-season week cutoff (1..N)")
    ap.add_argument("--prior-seasons", type=int, default=6)
    args = ap.parse_args()

    holdout = [int(s) for s in args.holdout.split(",") if s.strip()]
    # Home-field baseline: leaguewide early-season home win rate ≈ 0.55.
    HOME_BASE = 0.55

    rows: list[tuple[float, float, int]] = []  # (model_p_home, base_p_home, home_won)
    covered = 0
    total = 0
    for season in holdout:
        priors = list(range(season - args.prior_seasons, season))
        seed = _prior_seed(priors)
        if not seed:
            print(f"  ! no prior seed for {season}", file=sys.stderr)
            continue
        games = _early_games(season, args.weeks)
        for gm in games:
            total += 1
            p = _pred(seed, gm["home_team"], gm["away_team"])
            if p is None:
                continue
            covered += 1
            rows.append((p, HOME_BASE, 1 if gm["hs"] > gm["as_"] else 0))
        print(f"  {season}: {len(games)} early games (weeks 1-{args.weeks})")

    if not rows:
        print("No scored games — abort", file=sys.stderr)
        return 1

    def brier(idx: int) -> float:
        return sum((r[idx] - r[2]) ** 2 for r in rows) / len(rows)
    def acc(idx: int) -> float:
        return sum(1 for r in rows if (r[idx] >= 0.5) == bool(r[2])) / len(rows)

    b_model, b_base = brier(0), brier(1)
    print(f"\n=== Preseason-prior standalone skill (weeks 1-{args.weeks}) ===")
    print(f"  n scored:          {len(rows)}  (coverage {covered}/{total})")
    print(f"  prior-seed Brier:  {b_model:.4f}   acc {acc(0)*100:.1f}%")
    print(f"  home-base Brier:   {b_base:.4f}   (constant {HOME_BASE:.2f} home)")
    print(f"  ΔBrier (base−prior): {(b_base-b_model)*1000:+.1f}/1000  "
          f"(positive → the prior has early-season skill)")
    print("\nEnabling: set PRESEASON_PRIOR_ENABLED=True only if this shows clear")
    print("skill AND it earns CLV on `evmax cleanup shadow clv nfl` — the sharp")
    print("line already prices preseason info, so Brier alone is not the gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
