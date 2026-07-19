"""Full-season anchored-entry CLV verdict for WNBA spread/total ladders.

Replays the Kalshi candlestick backfill (scripts/backfill_kalshi_candles.py,
session_id 'candlebf-*') through the first-anchored-sweep entry rule and asks,
per market type and side: does entering at the first hour where a Pinnacle
anchor can price the ticker's line, gated on EV >= 2pp at the crossable candle
price, earn positive Kalshi CLV by the last pre-tip candle?

This is the decisive full-season version of the watch-listings readout (which
had only ~26 declustered games). Known limitation, stated in the report:
candles carry NO order-book depth, so the eval runs depth-free — section 2
bounds that bias on the July overlap where watchlist depth capture exists.

Stats discipline (from analyze_wnba_listings_lay_robustness.py, whose helpers
are copied verbatim): ladder strikes within one game are correlated, so every
number is reported both raw and GAME-DECLUSTERED (one vote per game), with a
one-sided binomial p vs the coin-flip null.

Verdict rule per side (Track C of the plan):
  PROMOTE-CANDIDATE  declustered n>=30, mean>0, %pos>=55, p<=0.05
  KILL               declustered n>=70 and (mean<=0 or p>0.10)
  CONTINUE           anything else (underpowered / ambiguous)

Read-only against archive.db except nothing — it never writes anywhere.
Usage: python scripts/eval_wnba_anchored_backfill.py [--ev-min 2.0] [--detail]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

import structlog

# The falsification control (ev_min_pp=-999) prices every alt-line strike,
# logging thousands of per-strike DEBUG lines — silence below WARNING.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The worktree carries its own near-empty data/archive.db; point at the main
# checkout's live archive (same pattern as analyze_wnba_listings_lay_robustness).
MAIN_REPO = Path("/Users/ktzeng/Projects/evmax")
import evmax.archiver as _archiver_mod  # noqa: E402
if (MAIN_REPO / "data" / "archive.db").exists():
    _archiver_mod.DB_PATH = MAIN_REPO / "data" / "archive.db"

from evmax.agents.cleanup.listings_eval import AnchoredEntry, evaluate_sector  # noqa: E402
from evmax.archiver import _get_connection as get_archive_conn  # noqa: E402

SECTOR = "wnba"
MARKET_TYPES = ("spread", "total")
DEPTH_MIN_WATCHLIST = 50.0
JULY_CUTOFF = "2026-07-01"
SIDES = (("spread", "lay"), ("spread", "take"), ("total", "over"), ("total", "under"))


# --- stat helpers copied verbatim from analyze_wnba_listings_lay_robustness ---

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

# --- end copied helpers ---


def report_bucket(label: str, entries: list[AnchoredEntry]) -> dict:
    s = clv_summary(entries)
    if s["n"] == 0:
        print(f"  {label:<34} n=0")
        return s
    pos = round(s["pos_pct"] / 100 * s["n"])
    p = binom_p(s["n"], pos)
    p_str = f"p={p:.3f}" if p is not None else "p=n/a"
    print(f"  {label:<34} n={s['n']:<5} mean={s['mean']:+.2f}pp  "
          f"%pos={s['pos_pct']:.0f}%  {p_str}")
    return s


def report_declustered(label: str, entries: list[AnchoredEntry]) -> tuple[dict, float | None]:
    d = game_declustered_summary(entries)
    if d["n_games"] == 0:
        print(f"  {label:<34} n_games=0")
        return d, None
    pos = round(d["pos_pct"] / 100 * d["n_games"])
    p = binom_p(d["n_games"], pos)
    p_str = f"p={p:.3f}" if p is not None else "p=n/a"
    print(f"  {label:<34} n_games={d['n_games']:<4} mean={d['mean']:+.2f}pp  "
          f"%games-pos={d['pos_pct']:.0f}%  {p_str}")
    return d, p


def side_entries(entries: list[AnchoredEntry], mt: str, side: str) -> list[AnchoredEntry]:
    return [e for e in entries if e.market_type == mt and e.side == side]


def verdict(d: dict, p: float | None) -> str:
    n, m, pos = d["n_games"], d["mean"], d["pos_pct"]
    if n >= 30 and m is not None and m > 0 and pos >= 55 and p is not None and p <= 0.05:
        return "PROMOTE-CANDIDATE"
    if n >= 70 and (m is None or m <= 0 or (p is not None and p > 0.10)):
        return "KILL"
    return "CONTINUE"


def candle_eval(conn, ev_min: float, **kw):
    return evaluate_sector(
        conn, SECTOR, ev_min_pp=ev_min, depth_min_usd=0.0,
        market_types=MARKET_TYPES, session_prefixes=("candlebf",),
        bid_from_no_price=True, **kw,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev-min", type=float, default=2.0)
    ap.add_argument("--detail", action="store_true",
                    help="Print every entry, not just the summaries.")
    args = ap.parse_args()

    with get_archive_conn() as conn:
        stats = candle_eval(conn, args.ev_min)
        control = candle_eval(conn, -999.0)
        stats_july = candle_eval(conn, args.ev_min, since=JULY_CUTOFF)
        watchlist_july = evaluate_sector(
            conn, SECTOR, ev_min_pp=args.ev_min, depth_min_usd=DEPTH_MIN_WATCHLIST,
            market_types=MARKET_TYPES, since=JULY_CUTOFF,
        )
        watchlist_july_d0 = evaluate_sector(
            conn, SECTOR, ev_min_pp=args.ev_min, depth_min_usd=0.0,
            market_types=MARKET_TYPES, since=JULY_CUTOFF,
        )

    print(f"WNBA anchored-entry backfill eval  ev>={args.ev_min}pp  depth-free "
          f"(candles carry no depth)\n"
          f"coverage: tickers={stats.tickers}  never-anchored={stats.never_anchored}  "
          f"anchored-no-entry={stats.anchored_no_entry}  entries={len(stats.entries)}")

    if args.detail:
        for e in sorted(stats.entries, key=lambda x: x.entry_at):
            clv = f"{e.clv_pp:+.1f}pp" if e.clv_pp is not None else "  — "
            print(f"  {e.entry_at.date()}  {e.event_label:<24} {e.outcome_label:<18} "
                  f"{e.market_type}/{e.side:<5} entry={e.entry_price:.2f} "
                  f"edge={e.edge_pp:+.1f}pp lead={e.entry_lead_h:.1f}h clv={clv}")

    verdicts: dict[str, str] = {}
    print("\n=== 1. FULL SEASON, PER SIDE (raw + game-declustered) ===")
    for mt, side in SIDES:
        sel = side_entries(stats.entries, mt, side)
        print(f"[{mt}/{side}]")
        report_bucket("raw strikes", sel)
        d, p = report_declustered("game-declustered", sel)
        verdicts[f"{mt.upper()}_{side.upper()}"] = verdict(d, p)

    print("\n=== 2. JULY OVERLAP VALIDITY (candles vs live watchlist capture) ===")
    print("  same window (since 2026-07-01); candle and watchlist reads should "
          "agree in sign/magnitude.")
    for mt, side in SIDES:
        print(f"[{mt}/{side}]")
        report_bucket("candles (depth-free)", side_entries(stats_july.entries, mt, side))
        report_bucket(f"watchlist depth>=${DEPTH_MIN_WATCHLIST:g}",
                      side_entries(watchlist_july.entries, mt, side))
        report_bucket("watchlist depth-free",
                      side_entries(watchlist_july_d0.entries, mt, side))
    print("  <- the depth>=$50 vs depth-free watchlist delta bounds the bias of "
          "the depth-free full-season number")

    print("\n=== 3. FALSIFICATION (control = every anchored strike, no EV gate) ===")
    for mt, side in SIDES:
        sel = side_entries(stats.entries, mt, side)
        ctrl = side_entries(control.entries, mt, side)
        fade = [e for e in ctrl if e.edge_pp <= -args.ev_min]
        print(f"[{mt}/{side}]")
        report_bucket("actual +EV entries", sel)
        report_bucket("ALL anchored (no EV gate)", ctrl)
        report_bucket(f"FADE (edge<=-{args.ev_min:g}pp)", fade)
    print("  <- if control/fade are as positive as the entries, the 'edge' is "
          "market-wide drift, not selection")

    print("\n=== 4. PRE-JULY vs POST-JULY (anchor cadence 2-4/day vs hourly) ===")
    for mt, side in SIDES:
        sel = side_entries(stats.entries, mt, side)
        pre = [e for e in sel if e.entry_at.date().isoformat() < JULY_CUTOFF]
        post = [e for e in sel if e.entry_at.date().isoformat() >= JULY_CUTOFF]
        print(f"[{mt}/{side}]")
        report_bucket("pre-July", pre)
        report_bucket("post-July", post)

    print("\n=== 5. VERDICTS (declustered gate: PROMOTE n>=30/mean>0/%pos>=55/"
          "p<=0.05; KILL n>=70 & (mean<=0 or p>0.10)) ===")
    for key, v in verdicts.items():
        print(f"VERDICT_{key}={v}")


if __name__ == "__main__":
    main()
