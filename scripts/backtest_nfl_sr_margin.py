"""Walk-forward validation for the NFL efficiency SR-in-margin prototype.

Mirrors scripts/backtest_ncaaf_v2.py's protocol for the NFL analog: does adding
an opponent-adjusted **success-rate** term to the moneyline margin
(margin = EPA_PTS·Δnet_epa + SR_PTS·Δnet_sr + HOME_EDGE) beat the shipped
EPA-only margin out of sample?

Leak-free: for every game the team stats (opponent-adjusted net EPA + net SR)
are recomputed from PBP STRICTLY BEFORE that game's date, decayed across prior
seasons exactly as the live seed does (reuses seed_nfl_efficiency helpers).

Two modes:
  --fit-seasons  : no-intercept OLS margin ~ home + Δepa + Δsr on those seasons.
                   Prints the (HOME, EPA_PTS, SR_PTS) a fit would choose — the
                   candidate constants for nfl_efficiency_agent.
  (holdout)      : on the holdout seasons, paired Brier of EPA-only (the shipped
                   EPA_MARGIN_PTS/0/HOME_EDGE_PTS) vs EPA+SR (fitted), with z.

PROMOTION RULE (NCAAF v2 lesson): the sharp anchor absorbs most standalone-Brier
gain — a positive holdout ΔBrier here is necessary but NOT sufficient. Promote
(set EPA_MARGIN_PTS / SR_MARGIN_PTS in the agent) only if the change ALSO earns
CLV on the live shadow stream (`evmax cleanup shadow clv nfl`), never on Brier
alone. This script only measures the forecasting side.

Usage:
    uv run python scripts/backtest_nfl_sr_margin.py --fit-seasons 2022,2023,2024 --holdout 2025
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.nfl_efficiency_agent import (  # noqa: E402
    EPA_MARGIN_PTS, HOME_EDGE_PTS, MIN_GAMES, SCORE_STDEV,
    NFL_ABBREV_TO_NAME, _normal_cdf,
)


def _season_of(d: date) -> int:
    return d.year if d.month >= 3 else d.year - 1


def build_rows(seasons: list[int]) -> list[dict]:
    """One row per completed game with as-of (pre-game) net EPA + net SR diffs.

    Each row: {season, d_epa, d_sr, home (1/0 — always 1 here, NFL has a home
    side), margin (home−away), home_won}. Games where either team is below
    MIN_GAMES of prior data are skipped (mirrors the live gate).
    """
    import nflreadpy as nfl
    import polars as pl
    from scripts.seed_nfl_efficiency import compute_team_stats, filter_valid_pbp

    # Load PBP for the requested seasons plus the 3 trailing seasons each needs
    # for the decayed prior (so early-season games have history).
    load_seasons = sorted(set(seasons) | {s - k for s in seasons for k in (1, 2, 3)})
    frames = []
    for s in load_seasons:
        try:
            frames.append(nfl.load_pbp(seasons=[s]))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skip season {s}: {e}", file=sys.stderr)
    if not frames:
        print("No PBP loaded — abort", file=sys.stderr)
        return []
    pbp = pl.concat(frames, how="diagonal")
    pbp = pbp.with_columns(pl.col("game_date").cast(pl.Date, strict=False))
    valid = filter_valid_pbp(pbp)

    games = (
        pbp.group_by(["game_id", "home_team", "away_team", "season", "game_date"])
        .agg([pl.col("home_score").max().alias("hs"),
              pl.col("away_score").max().alias("as_")])
        .filter(pl.col("season").is_in(seasons))
        .sort("game_date")
    )

    rows: list[dict] = []
    stats_cache: dict[date, dict] = {}
    for g in games.iter_rows(named=True):
        hs, as_ = g["hs"], g["as_"]
        if hs is None or as_ is None or hs == as_:
            continue
        cutoff = g["game_date"]
        if cutoff not in stats_cache:
            slice_df = valid.filter(pl.col("game_date") < pl.lit(cutoff))
            stats_cache[cutoff] = (
                compute_team_stats(slice_df, _season_of(cutoff)) if len(slice_df) else {}
            )
        teams = stats_cache[cutoff]
        home = NFL_ABBREV_TO_NAME.get(g["home_team"])
        away = NFL_ABBREV_TO_NAME.get(g["away_team"])
        sa, sb = teams.get(home or ""), teams.get(away or "")
        if not sa or not sb:
            continue
        if sa.get("gp", 0) < MIN_GAMES or sb.get("gp", 0) < MIN_GAMES:
            continue
        net_epa = (sa["off_epa_adj"] - sa["def_epa_adj"]) - (sb["off_epa_adj"] - sb["def_epa_adj"])
        net_sr = (
            (sa.get("off_success_adj", 0.0) - sa.get("def_success_adj", 0.0))
            - (sb.get("off_success_adj", 0.0) - sb.get("def_success_adj", 0.0))
        )
        rows.append({
            "season": g["season"], "d_epa": net_epa, "d_sr": net_sr,
            "margin": float(hs - as_), "home_won": 1 if hs > as_ else 0,
        })
    return rows


def _brier(rows: list[dict], epa_pts: float, sr_pts: float, home_edge: float) -> tuple[float, int]:
    if not rows:
        return float("nan"), 0
    tot = 0.0
    for r in rows:
        m = epa_pts * r["d_epa"] + sr_pts * r["d_sr"] + home_edge
        p = min(0.98, max(0.02, _normal_cdf(m / SCORE_STDEV)))
        tot += (p - r["home_won"]) ** 2
    return tot / len(rows), len(rows)


def fit_constants(rows: list[dict]) -> tuple[float, float, float]:
    """No-intercept OLS: margin ~ home + Δepa + Δsr. Returns (HOME, EPA, SR)."""
    X = np.array([[1.0, r["d_epa"], r["d_sr"]] for r in rows], float)
    y = np.array([r["margin"] for r in rows], float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(b[0]), float(b[1]), float(b[2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-seasons", default="2022,2023,2024",
                    help="Comma-separated seasons to fit the OLS constants on.")
    ap.add_argument("--holdout", default="2025",
                    help="Comma-separated holdout seasons for the paired Brier test.")
    args = ap.parse_args()

    fit_seasons = [int(s) for s in args.fit_seasons.split(",") if s.strip()]
    holdout = [int(s) for s in args.holdout.split(",") if s.strip()]

    print(f"Building fit rows for {fit_seasons} …")
    fit_rows = build_rows(fit_seasons)
    print(f"  n={len(fit_rows)}")
    if len(fit_rows) < 50:
        print("Too few fit rows — abort", file=sys.stderr)
        return 1

    home_c, epa_c, sr_c = fit_constants(fit_rows)
    print("\n=== OLS fit (no intercept: margin ~ home + Δepa + Δsr) ===")
    print(f"  HOME_EDGE_PTS = {home_c:.2f}   (shipped {HOME_EDGE_PTS})")
    print(f"  EPA_MARGIN_PTS = {epa_c:.2f}   (shipped {EPA_MARGIN_PTS})")
    print(f"  SR_MARGIN_PTS  = {sr_c:.2f}   (shipped 0.0 — inert)")

    print(f"\nBuilding holdout rows for {holdout} …")
    hold = build_rows(holdout)
    print(f"  n={len(hold)}")
    if not hold:
        return 1

    # Shipped EPA-only margin vs the fitted EPA+SR margin, both on holdout.
    b_epa, n = _brier(hold, EPA_MARGIN_PTS, 0.0, HOME_EDGE_PTS)
    b_sr, _ = _brier(hold, epa_c, sr_c, home_c)
    delta = b_epa - b_sr  # positive → SR margin better

    # Paired z on the per-game squared-error difference.
    diffs = []
    for r in hold:
        m0 = EPA_MARGIN_PTS * r["d_epa"] + HOME_EDGE_PTS
        m1 = epa_c * r["d_epa"] + sr_c * r["d_sr"] + home_c
        p0 = min(0.98, max(0.02, _normal_cdf(m0 / SCORE_STDEV)))
        p1 = min(0.98, max(0.02, _normal_cdf(m1 / SCORE_STDEV)))
        diffs.append((p0 - r["home_won"]) ** 2 - (p1 - r["home_won"]) ** 2)
    arr = np.array(diffs)
    z = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))) if arr.std() > 0 else 0.0

    print("\n=== Holdout paired Brier ===")
    print(f"  EPA-only (shipped): {b_epa:.4f}")
    print(f"  EPA+SR   (fitted) : {b_sr:.4f}")
    print(f"  ΔBrier (epa−sr)   : {delta*1000:+.2f}/1000   z={z:+.2f}   n={n}")
    print("\nPromotion: a positive ΔBrier is necessary but NOT sufficient — the")
    print("sharp anchor absorbs most of it. Set EPA_MARGIN_PTS / SR_MARGIN_PTS in")
    print("nfl_efficiency_agent ONLY if this ALSO earns CLV on the live shadow")
    print("stream (`evmax cleanup shadow clv nfl`), never on Brier alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
