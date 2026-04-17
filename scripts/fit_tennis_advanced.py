"""Fit logistic regression on advanced tennis stats to find optimal coefficients.

Downloads Sackmann match CSVs (2023-2025), computes per-player rolling stats,
builds per-match feature vectors, fits logistic regression, and reports
coefficients + validation accuracy.

Features:
  1. BP conversion rate differential (bp_won / bp_faced)
  2. Return points won % differential (rpw = 1 - opponent_spw)
  3. Unforced error rate differential (ue / total_points_played)
  4. Winners-to-UE ratio differential

Usage:
    uv run python scripts/fit_tennis_advanced.py
"""

from __future__ import annotations

import csv
import io
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
import numpy as np

REPO_BASE = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}
MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"

TRAIN_YEARS = [2023, 2024]
VAL_YEARS = [2025]
TOURS = ["atp", "wta"]
MIN_MATCHES = 15


def _fetch_csv(url: str) -> list[dict] | None:
    try:
        r = httpx.get(url, timeout=30.0)
    except httpx.HTTPError:
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def _safe_int(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def _safe_float(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


@dataclass
class PlayerStats:
    bp_won: int = 0
    bp_faced: int = 0
    return_pts: int = 0
    return_pts_won: int = 0
    aces: int = 0
    dfs: int = 0
    svpt: int = 0
    first_won: int = 0
    second_won: int = 0
    unforced: int = 0
    winners: int = 0
    total_pts: int = 0
    matches: int = 0


def aggregate_stats(match_rows: list[dict]) -> dict[str, PlayerStats]:
    """Aggregate per-player stats from Sackmann match rows."""
    stats: dict[str, PlayerStats] = defaultdict(PlayerStats)

    for row in match_rows:
        w_name = (row.get("winner_name") or "").strip().lower()
        l_name = (row.get("loser_name") or "").strip().lower()
        if not w_name or not l_name:
            continue

        w_svpt = _safe_int(row.get("w_svpt"))
        l_svpt = _safe_int(row.get("l_svpt"))
        if w_svpt <= 0 or l_svpt <= 0:
            continue

        w = stats[w_name]
        l = stats[l_name]

        # Winner's stats (as winner side)
        w_1w = _safe_int(row.get("w_1stWon"))
        w_2w = _safe_int(row.get("w_2ndWon"))
        w_ace = _safe_int(row.get("w_ace"))
        w_df = _safe_int(row.get("w_df"))
        w_bpf = _safe_int(row.get("w_bpFaced"))
        w_bps = _safe_int(row.get("w_bpSaved"))

        l_1w = _safe_int(row.get("l_1stWon"))
        l_2w = _safe_int(row.get("l_2ndWon"))
        l_ace = _safe_int(row.get("l_ace"))
        l_df = _safe_int(row.get("l_df"))
        l_bpf = _safe_int(row.get("l_bpFaced"))
        l_bps = _safe_int(row.get("l_bpSaved"))

        # Break points: bpFaced is how many BPs the opponent had on your serve.
        # bpSaved is how many you saved. So BP converted against you = bpFaced - bpSaved.
        # For the RETURNER: their BP conversion = opponent_bpFaced - opponent_bpSaved.
        # Winner's BP conversion on return = l_bpFaced - l_bpSaved (BPs the loser faced, winner converted)
        w.bp_won += (l_bpf - l_bps)  # winner broke loser's serve this many times
        w.bp_faced += l_bpf           # winner had this many BP opportunities
        l.bp_won += (w_bpf - w_bps)   # loser broke winner this many times
        l.bp_faced += w_bpf

        # Return points won: when you're returning, opponent is serving.
        # Winner returning against loser's serve: total = l_svpt, won = l_svpt - (l_1w + l_2w)
        w.return_pts += l_svpt
        w.return_pts_won += (l_svpt - l_1w - l_2w)
        l.return_pts += w_svpt
        l.return_pts_won += (w_svpt - w_1w - w_2w)

        # Serve stats
        w.svpt += w_svpt
        w.first_won += w_1w
        w.second_won += w_2w
        w.aces += w_ace
        w.dfs += w_df
        l.svpt += l_svpt
        l.first_won += l_1w
        l.second_won += l_2w
        l.aces += l_ace
        l.dfs += l_df

        # Total points played (serve + return)
        total_pts = w_svpt + l_svpt
        w.total_pts += total_pts
        l.total_pts += total_pts

        w.matches += 1
        l.matches += 1

    return dict(stats)


def augment_with_mcp(stats: dict[str, PlayerStats], tour: str) -> None:
    """Augment stats with MCP Overview data (winners, unforced errors)."""
    matches_url = f"{MCP_BASE}/charting-{'m' if tour == 'atp' else 'w'}-matches.csv"
    overview_url = f"{MCP_BASE}/charting-{'m' if tour == 'atp' else 'w'}-stats-Overview.csv"

    matches_rows = _fetch_csv(matches_url)
    overview_rows = _fetch_csv(overview_url)
    if not matches_rows or not overview_rows:
        print(f"  ! MCP data unavailable for {tour}", file=sys.stderr)
        return

    match_years: dict[str, int] = {}
    for row in matches_rows:
        mid = row.get("match_id", "")
        date_str = row.get("Date", "")
        if mid and len(date_str) >= 4:
            try:
                match_years[mid] = int(date_str[:4])
            except ValueError:
                pass

    count = 0
    for row in overview_rows:
        if row.get("set") != "Total":
            continue
        mid = row.get("match_id", "")
        yr = match_years.get(mid)
        if yr is None or yr < 2023:
            continue

        player = (row.get("player") or "").strip().lower()
        if not player or player not in stats:
            continue

        s = stats[player]
        s.winners += _safe_int(row.get("winners"))
        s.unforced += _safe_int(row.get("unforced"))
        count += 1

    print(f"  MCP {tour}: augmented {count} player-match records with winners/UE")


def build_match_features(
    match_rows: list[dict],
    player_stats: dict[str, PlayerStats],
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix from match rows using pre-computed player stats.

    Returns (X, y) where X has columns:
      [bp_conv_diff, rpw_diff, ue_rate_diff, w_ue_ratio_diff]
    and y is 1 if winner is player_a (alphabetically first), 0 otherwise.
    """
    X_rows = []
    y_rows = []

    for row in match_rows:
        w_name = (row.get("winner_name") or "").strip().lower()
        l_name = (row.get("loser_name") or "").strip().lower()
        if not w_name or not l_name:
            continue

        w_stats = player_stats.get(w_name)
        l_stats = player_stats.get(l_name)
        if not w_stats or not l_stats:
            continue
        if w_stats.matches < MIN_MATCHES or l_stats.matches < MIN_MATCHES:
            continue

        feats_w = _compute_features(w_stats)
        feats_l = _compute_features(l_stats)
        if feats_w is None or feats_l is None:
            continue

        # Alphabetical ordering for consistency
        a, b = sorted((w_name, l_name))
        if a == w_name:
            diff = [fw - fl for fw, fl in zip(feats_w, feats_l)]
            y = 1
        else:
            diff = [fl - fw for fw, fl in zip(feats_w, feats_l)]
            y = 0

        X_rows.append(diff)
        y_rows.append(y)

    return np.array(X_rows), np.array(y_rows)


def _compute_features(s: PlayerStats) -> list[float] | None:
    if s.bp_faced <= 0 or s.return_pts <= 0 or s.total_pts <= 0:
        return None

    bp_conv = s.bp_won / s.bp_faced
    rpw = s.return_pts_won / s.return_pts
    ue_rate = s.unforced / s.total_pts if s.total_pts > 0 else 0.0
    w_ue = (s.winners / max(s.unforced, 1)) if s.unforced > 0 else (s.winners if s.winners > 0 else 1.0)

    return [bp_conv, rpw, ue_rate, w_ue]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss

    all_stats: dict[str, PlayerStats] = defaultdict(PlayerStats)
    train_matches: list[dict] = []
    val_matches: list[dict] = []

    for tour in TOURS:
        base = REPO_BASE[tour]
        prefix = tour

        for year in TRAIN_YEARS + VAL_YEARS:
            url = f"{base}/{prefix}_matches_{year}.csv"
            print(f"Fetching {url} ...")
            rows = _fetch_csv(url)
            if not rows:
                print(f"  ! skipped (404 or error)")
                continue
            print(f"  {len(rows)} matches")

            if year in TRAIN_YEARS:
                train_matches.extend(rows)
            else:
                val_matches.extend(rows)

    print(f"\nAggregating stats from {len(train_matches)} training matches ...")
    train_stats = aggregate_stats(train_matches)
    print(f"  {len(train_stats)} players")

    for tour in TOURS:
        augment_with_mcp(train_stats, tour)

    print(f"\nBuilding feature matrix (train) ...")
    X_train, y_train = build_match_features(train_matches, train_stats)
    print(f"  {len(X_train)} usable matches, {y_train.mean():.3f} class balance")

    if len(X_train) < 100:
        print("ERROR: Not enough training data", file=sys.stderr)
        return

    print(f"\nFitting logistic regression ...")
    model = LogisticRegression(penalty=None, max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    feature_names = ["bp_conv_diff", "rpw_diff", "ue_rate_diff", "w_ue_ratio_diff"]
    print(f"\n{'='*60}")
    print(f"LOGISTIC REGRESSION COEFFICIENTS")
    print(f"{'='*60}")
    for name, coef in zip(feature_names, coefs):
        print(f"  {name:25s}  {coef:+.4f}")
    print(f"  {'intercept':25s}  {intercept:+.4f}")

    train_pred = model.predict_proba(X_train)[:, 1]
    print(f"\nTrain accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Train log-loss: {log_loss(y_train, train_pred):.4f}")
    print(f"Train Brier:    {np.mean((train_pred - y_train) ** 2):.4f}")

    # Validation
    if val_matches:
        print(f"\nBuilding feature matrix (validation — {VAL_YEARS}) ...")
        # Use train stats for features (simulating production: model knows 2023-24, predicts 2025)
        X_val, y_val = build_match_features(val_matches, train_stats)
        print(f"  {len(X_val)} usable matches")

        if len(X_val) > 0:
            val_pred = model.predict_proba(X_val)[:, 1]
            print(f"\nValidation accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
            print(f"Validation log-loss: {log_loss(y_val, val_pred):.4f}")
            print(f"Validation Brier:    {np.mean((val_pred - y_val) ** 2):.4f}")

            # Compare to baseline (50/50)
            baseline_brier = np.mean((0.5 - y_val) ** 2)
            print(f"Baseline Brier (50/50): {baseline_brier:.4f}")

            # Compare to SPW-only (simple model)
            print(f"\nBrier lift vs baseline: {(baseline_brier - np.mean((val_pred - y_val)**2)) / baseline_brier * 100:.1f}%")

    # Output the coefficients for hardcoding into the agent
    print(f"\n{'='*60}")
    print(f"COPY INTO TennisAdvancedStatsAgent:")
    print(f"{'='*60}")
    print(f"_COEFS = [{', '.join(f'{c:.4f}' for c in coefs)}]")
    print(f"_INTERCEPT = {intercept:.4f}")

    # Feature importance
    print(f"\nFeature importance (|coef| × std(feature)):")
    for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        col_idx = feature_names.index(name)
        std = X_train[:, col_idx].std()
        print(f"  {name:25s}  |coef|={abs(coef):.3f}  std={std:.4f}  impact={abs(coef)*std:.4f}")


if __name__ == "__main__":
    main()
