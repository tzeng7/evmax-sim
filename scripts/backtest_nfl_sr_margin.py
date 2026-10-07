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

    Each row: {season, game_date, d_epa, d_sr, home (1/0 — always 1 here, NFL has a home
    side), margin (home−away), home_won}. Games where either team is below
    MIN_GAMES of prior data, and games before the season's first PBP lands
    (Week 1), are skipped — both mirror the live gates.
    """
    import polars as pl
    from scripts.seed_nfl_efficiency import (
        DEFAULT_NUM_SEASONS, compute_team_stats, filter_valid_pbp, load_pbp_tolerant,
    )

    # Load PBP for the requested seasons plus the trailing seasons the live seed
    # decays over (DEFAULT_NUM_SEASONS in total), so every as-of state matches
    # what scripts/seed_nfl_efficiency.py would have written on that date.
    trailing = range(1, DEFAULT_NUM_SEASONS)
    load_seasons = sorted(set(seasons) | {s - k for s in seasons for k in trailing})
    pbp = load_pbp_tolerant(load_seasons)
    if pbp.is_empty():
        print("No PBP loaded — abort", file=sys.stderr)
        return []
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
            cur = _season_of(cutoff)
            slice_df = valid.filter(
                (pl.col("game_date") < pl.lit(cutoff))
                & (pl.col("season") > cur - DEFAULT_NUM_SEASONS)
            )
            # Live blanks the model until the seed holds current-season PBP
            # (nfl_state_is_stale_for_today), so Week-1 games are never priced.
            has_current = slice_df.filter(pl.col("season") == cur).height > 0
            stats_cache[cutoff] = (
                compute_team_stats(slice_df, cur) if has_current else {}
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
            "season": g["season"], "game_date": cutoff, "d_epa": net_epa, "d_sr": net_sr,
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


def fit_epa_only(rows: list[dict]) -> tuple[float, float]:
    """No-intercept OLS: margin ~ home + Δepa. Returns (HOME, EPA).

    The scale-only refit: SR stays inert, only EPA_MARGIN_PTS / HOME_EDGE_PTS
    move. The shipped EPA_MARGIN_PTS (= plays per game) assumes one EPA point
    is one scoreboard point; this measures what the results support.
    """
    X = np.array([[1.0, r["d_epa"]] for r in rows], float)
    y = np.array([r["margin"] for r in rows], float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(b[0]), float(b[1])


def fit_epa_scale_only(rows: list[dict]) -> float:
    """OLS for EPA_MARGIN_PTS with HOME_EDGE_PTS held at the shipped value."""
    x = np.array([r["d_epa"] for r in rows], float)
    y = np.array([r["margin"] for r in rows], float) - HOME_EDGE_PTS
    return float((x @ y) / (x @ x))


def _paired(rows: list[dict], base: tuple[float, float, float],
            cand: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Paired Brier of two (EPA_PTS, SR_PTS, HOME) margins on the same games.
    Returns (brier_base, brier_cand, Δ per 1000 (base − cand; + = cand better), z)."""
    diffs = []
    for r in rows:
        sq = []
        for epa_pts, sr_pts, home in (base, cand):
            m = epa_pts * r["d_epa"] + sr_pts * r["d_sr"] + home
            p = min(0.98, max(0.02, _normal_cdf(m / SCORE_STDEV)))
            sq.append((p - r["home_won"]) ** 2)
        diffs.append(sq)
    arr = np.array(diffs)
    d = arr[:, 0] - arr[:, 1]
    z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0.0
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(d.mean() * 1000), float(z)


def walk_forward_scale(eval_seasons: list[int], window: int) -> int:
    """Season-by-season out-of-sample test of the EPA scale refit.

    For each season S: fit EPA_MARGIN_PTS (HOME held at HOME_EDGE_PTS) on the
    `window` seasons before S, then score S against the shipped constant.
    Every scored game is out of sample for the constant that prices it.
    """
    all_seasons = sorted({s - k for s in eval_seasons for k in range(0, window + 1)})
    print(f"Building rows for {all_seasons[0]}–{all_seasons[-1]} …")
    rows = build_rows(all_seasons)
    by_season: dict[int, list[dict]] = {}
    for r in rows:
        by_season.setdefault(r["season"], []).append(r)
    shipped = (EPA_MARGIN_PTS, 0.0, HOME_EDGE_PTS)
    pooled_base: list[dict] = []
    pooled_pairs: list[tuple[dict, float]] = []
    print(f"\n=== Walk-forward scale refit (fit window {window} seasons, HOME {HOME_EDGE_PTS}) ===")
    for s in eval_seasons:
        train = [r for k in range(1, window + 1) for r in by_season.get(s - k, [])]
        test = by_season.get(s, [])
        if len(train) < 200 or not test:
            print(f"  {s}: skipped (train n={len(train)}, test n={len(test)})")
            continue
        k = fit_epa_scale_only(train)
        b0, b1, d, z = _paired(test, shipped, (k, 0.0, HOME_EDGE_PTS))
        print(f"  {s}: EPA_PTS fit={k:5.1f}  Brier {b0:.4f} → {b1:.4f}  Δ {d:+.2f}/1000  z={z:+.2f}  n={len(test)}")
        pooled_pairs.extend((r, k) for r in test)
    if pooled_pairs:
        diffs = []
        for r, k in pooled_pairs:
            sq = []
            for epa_pts in (EPA_MARGIN_PTS, k):
                m = epa_pts * r["d_epa"] + HOME_EDGE_PTS
                p = min(0.98, max(0.02, _normal_cdf(m / SCORE_STDEV)))
                sq.append((p - r["home_won"]) ** 2)
            diffs.append(sq[0] - sq[1])
        d = np.array(diffs)
        z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f"  POOLED: Δ {d.mean()*1000:+.2f}/1000  z={z:+.2f}  n={len(d)}  (+ = refit better)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-seasons", default="2022,2023,2024",
                    help="Comma-separated seasons to fit the OLS constants on.")
    ap.add_argument("--holdout", default="2025",
                    help="Comma-separated holdout seasons for the paired Brier test.")
    ap.add_argument("--walk-forward", default="",
                    help="Comma-separated seasons. Scale-only mode: each season is "
                         "scored with EPA_MARGIN_PTS fitted on the FIT_WINDOW seasons "
                         "before it (HOME kept), then the paired Δ is pooled.")
    ap.add_argument("--fit-window", type=int, default=4,
                    help="Seasons of history per walk-forward fit (default 4).")
    args = ap.parse_args()

    if args.walk_forward:
        return walk_forward_scale([int(s) for s in args.walk_forward.split(",") if s.strip()],
                                  args.fit_window)

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

    home_e, epa_e = fit_epa_only(fit_rows)
    print("\n=== OLS fit, EPA only (no intercept: margin ~ home + Δepa) ===")
    print(f"  HOME_EDGE_PTS = {home_e:.2f}   (shipped {HOME_EDGE_PTS})")
    print(f"  EPA_MARGIN_PTS = {epa_e:.2f}   (shipped {EPA_MARGIN_PTS})")
    print("  per fit season (stability):")
    for fs in fit_seasons:
        sub = [r for r in fit_rows if r["season"] == fs]
        if len(sub) >= 50:
            h, e = fit_epa_only(sub)
            print(f"    {fs}: HOME {h:5.2f}  EPA {e:6.2f}  n={len(sub)}")

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

    shipped = (EPA_MARGIN_PTS, 0.0, HOME_EDGE_PTS)
    print("\n=== Holdout paired Brier: scale refit vs shipped (+ = refit better) ===")
    for label, cand in (
        ("EPA-only refit (EPA + HOME)", (epa_e, 0.0, home_e)),
        ("EPA-only refit, HOME kept", (fit_epa_scale_only(fit_rows), 0.0, HOME_EDGE_PTS)),
    ):
        b0, b1, d, zz = _paired(hold, shipped, cand)
        print(f"  {label:30s} EPA={cand[0]:6.2f} HOME={cand[2]:4.2f}  "
              f"{b0:.4f} → {b1:.4f}  Δ {d:+.2f}/1000  z={zz:+.2f}")
        for hs in holdout:
            sub = [r for r in hold if r["season"] == hs]
            early = [r for r in sub if r["game_date"].month == 9 or
                     (r["game_date"].month == 10 and r["game_date"].day <= 15)]
            for tag, rr in ((str(hs), sub), (f"{hs} Sep–mid-Oct", early)):
                if len(rr) >= 30:
                    b0, b1, d, zz = _paired(rr, shipped, cand)
                    print(f"      {tag:18s} {b0:.4f} → {b1:.4f}  Δ {d:+.2f}/1000  z={zz:+.2f}  n={len(rr)}")
    print("\nPromotion: a positive ΔBrier is necessary but NOT sufficient — the")
    print("sharp anchor absorbs most of it. Set EPA_MARGIN_PTS / SR_MARGIN_PTS in")
    print("nfl_efficiency_agent ONLY if this ALSO earns CLV on the live shadow")
    print("stream (`evmax cleanup shadow clv nfl`), never on Brier alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
