"""Robustness check on the WNBA spread lay-side first-anchored-sweep CLV.

`evmax cleanup listings-eval -s wnba -m spread` reports lay: n=81, mean CLV
+0.57pp, 55% positive — clearing the promotion gate (n>=30, mean>=0,
%pos>=55%) exactly at the threshold. That's the first non-"no edge" result
all session, and it deserves the same skepticism every other apparent edge
here got before anyone scopes the early-entry capture-and-place pipeline it
would take to actually harvest it.

Six checks, all built on evaluate_sector()/AnchoredEntry from
evmax.agents.cleanup.listings_eval — no new capture/eval logic:

  1. Game clustering  : n=81 counts LADDER STRIKES, not games. Strikes on the
                        same game aren't independent. Report distinct games +
                        a game-declustered summary (one vote per game).
  2. Time stability   : split the ~11-day watch-listings window in half;
                        a real edge should show in more than one slice.
  3. Pre/post-07-05   : both launchd plists have mtime 07-05 (this session's
                        root-cause finding); re-run with since=2026-07-05.
  4. Contested/lopsided: bucket by entry_price (0.35-0.65 vs not) — the
                        corrected backtest (PR #101) found real CLV
                        concentrated in contested strikes.
  5. Falsification    : pull ALL depth-adequate entries regardless of edge
                        (ev_min_pp very negative) as a control set, plus a
                        fade subset (edge_pp <= -2) from that same pull.
                        Positive CLV on control/fade too = market-wide drift
                        artifact, not EV-selection edge (same falsification
                        logic as backtest_wnba_spread_walkforward.py's CLV
                        CONTROL section).
  6. Significance     : binomial test of %positive vs 50% null, at both raw
                        n and the game-declustered n from (1).

Read-only. Run:  .venv/bin/python scripts/analyze_wnba_listings_lay_robustness.py
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

import structlog

# The falsification control (ev_min_pp=-999) forces SpreadDistributionModel to
# price every alt-line strike on the ladder, including ones many points from
# Pinnacle's primary line — each miss logs a per-strike DEBUG line, thousands
# of them, drowning the actual report. Silence below WARNING for this script.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The worktree carries its own near-empty data/archive.db; point at the main
# checkout's live archive (same pattern as scripts/diagnose_wnba_clv_zeros.py).
MAIN_REPO = Path("/Users/ktzeng/Projects/evmax")
import evmax.archiver as _archiver_mod  # noqa: E402
_archiver_mod.DB_PATH = MAIN_REPO / "data" / "archive.db"

from evmax.agents.cleanup.listings_eval import AnchoredEntry, evaluate_sector  # noqa: E402
from evmax.archiver import _get_connection as get_archive_conn  # noqa: E402

SECTOR = "wnba"
MARKET_TYPES = ("spread",)
EV_MIN = 2.0
DEPTH_MIN = 50.0
SINCE_CUTOFF = "2026-07-05"
CONTESTED_LO, CONTESTED_HI = 0.35, 0.65


def game_key(e: AnchoredEntry) -> tuple[str, date]:
    return (e.event_label, e.tip_at.date())


def clv_summary(entries: list[AnchoredEntry]) -> dict:
    clvs = [e.clv_pp for e in entries if e.clv_pp is not None]
    n = len(clvs)
    if n == 0:
        return {"n": 0, "mean": None, "pos_pct": None}
    return {
        "n": n,
        "mean": mean(clvs),
        "pos_pct": 100 * sum(1 for c in clvs if c > 0) / n,
    }


def game_declustered_summary(entries: list[AnchoredEntry]) -> dict:
    """One vote per game: mean-of-per-game-mean-CLV, %positive by game median."""
    by_game: dict[tuple[str, date], list[float]] = defaultdict(list)
    for e in entries:
        if e.clv_pp is not None:
            by_game[game_key(e)].append(e.clv_pp)
    if not by_game:
        return {"n_games": 0, "mean": None, "pos_pct": None}
    per_game_means = [mean(vs) for vs in by_game.values()]
    n = len(per_game_means)
    return {
        "n_games": n,
        "mean": mean(per_game_means),
        "pos_pct": 100 * sum(1 for m in per_game_means if m > 0) / n,
    }


def binom_p(n: int, pos: int) -> float | None:
    if n == 0:
        return None
    try:
        from scipy.stats import binomtest
        return binomtest(pos, n, 0.5, alternative="greater").pvalue
    except ImportError:
        # Normal approximation fallback (continuity-corrected one-sided z-test).
        from math import erf, sqrt
        p_hat = pos / n
        z = ((p_hat - 0.5) * n - 0.5) / sqrt(n * 0.25)
        return 1 - 0.5 * (1 + erf(z / sqrt(2)))


def report_bucket(label: str, entries: list[AnchoredEntry]) -> None:
    s = clv_summary(entries)
    if s["n"] == 0:
        print(f"  {label:<28} n=0")
        return
    pos = round(s["pos_pct"] / 100 * s["n"])
    p = binom_p(s["n"], pos)
    p_str = f"p={p:.3f}" if p is not None else "p=n/a"
    print(f"  {label:<28} n={s['n']:<4} mean={s['mean']:+.2f}pp  "
          f"%pos={s['pos_pct']:.0f}%  {p_str}")


def main() -> None:
    with get_archive_conn() as conn:
        stats = evaluate_sector(
            conn, SECTOR, ev_min_pp=EV_MIN, depth_min_usd=DEPTH_MIN,
            market_types=MARKET_TYPES,
        )
        control_stats = evaluate_sector(
            conn, SECTOR, ev_min_pp=-999.0, depth_min_usd=DEPTH_MIN,
            market_types=MARKET_TYPES,
        )
        stats_since05 = evaluate_sector(
            conn, SECTOR, ev_min_pp=EV_MIN, depth_min_usd=DEPTH_MIN,
            market_types=MARKET_TYPES, since=SINCE_CUTOFF,
        )

    lay = [e for e in stats.entries if e.side == "lay"]
    lay_control = [e for e in control_stats.entries if e.side == "lay"]
    lay_fade = [e for e in lay_control if e.edge_pp <= -EV_MIN]
    lay_since05 = [e for e in stats_since05.entries if e.side == "lay"]

    print(f"=== baseline (matches CLI) ===")
    report_bucket("lay (all)", lay)
    n_all = clv_summary(lay)["n"]

    print(f"\n=== 1. GAME CLUSTERING ===")
    games = {game_key(e) for e in lay}
    per_game_n = defaultdict(int)
    for e in lay:
        per_game_n[game_key(e)] += 1
    counts = sorted(per_game_n.values(), reverse=True)
    print(f"  distinct games: {len(games)}  (entries={n_all}, "
          f"{n_all/len(games):.1f} strikes/game avg)")
    print(f"  entries-per-game distribution: {counts}")
    decl = game_declustered_summary(lay)
    if decl["n_games"]:
        pos_g = round(decl["pos_pct"] / 100 * decl["n_games"])
        p_g = binom_p(decl["n_games"], pos_g)
        p_g_str = f"p={p_g:.3f}" if p_g is not None else "p=n/a"
        print(f"  game-declustered: n_games={decl['n_games']}  "
              f"mean(per-game mean)={decl['mean']:+.2f}pp  "
              f"%games-positive={decl['pos_pct']:.0f}%  {p_g_str}")

    print(f"\n=== 2. TIME STABILITY (split at median entry date) ===")
    dated = sorted(lay, key=lambda e: e.entry_at)
    with_clv = [e for e in dated if e.clv_pp is not None]
    if with_clv:
        mid = len(with_clv) // 2
        first_half, second_half = with_clv[:mid], with_clv[mid:]
        if first_half:
            print(f"  earliest entry: {with_clv[0].entry_at.date()}  "
                  f"latest: {with_clv[-1].entry_at.date()}")
        report_bucket("first half (by entry order)", first_half)
        report_bucket("second half (by entry order)", second_half)

    print(f"\n=== 3. PRE/POST 2026-07-05 (launchd migration date) ===")
    report_bucket(f"since={SINCE_CUTOFF}", lay_since05)
    pre = [e for e in lay if e.entry_at.date().isoformat() < SINCE_CUTOFF]
    report_bucket("before 2026-07-05", pre)

    print(f"\n=== 4. CONTESTED vs LOPSIDED (entry price 0.35-0.65) ===")
    contested = [e for e in lay if CONTESTED_LO <= e.entry_price <= CONTESTED_HI]
    lopsided = [e for e in lay if not (CONTESTED_LO <= e.entry_price <= CONTESTED_HI)]
    report_bucket("contested", contested)
    report_bucket("lopsided", lopsided)

    print(f"\n=== 5. FALSIFICATION (control = any depth-adequate entry) ===")
    report_bucket("actual +EV entries", lay)
    report_bucket("ALL depth-adequate (no EV gate)", lay_control)
    report_bucket("FADE (edge<=-2pp)", lay_fade)
    print("  <- if control/fade are ALSO positive, this is a market-wide "
          "drift, not selection edge")

    print(f"\n=== 6. SIGNIFICANCE SUMMARY ===")
    print(f"  raw n={n_all}: see baseline p-value above")
    print(f"  game-declustered n={decl['n_games']}: see game-clustering p-value above")
    print("  (p < 0.05 one-sided needed to call this distinguishable from a coin flip)")


if __name__ == "__main__":
    main()
