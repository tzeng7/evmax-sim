"""Walk-forward backtest of the NBA stat stack (efficiency, possession_sim,
shot_quality, matchup) — the models `run_walkforward("nba")` never fires.

Why a dedicated script: the generic walk-forward reads today's season-to-date
state, so all four agents abstain on historical games (see the module docstring
of evmax/backtest/sources/nba_stack_walkforward.py). This script rebuilds each
agent's inputs point-in-time from stats.nba.com (`DateTo` < game date, disk
cached) and runs the production agent code against them.

Reports, on regular-season games of the scored seasons:
  * standalone Brier / accuracy / coverage per model
  * three blends: generic (elo .35 / form .25 / poisson .30 — the 0.2199 figure),
    live_no_stack (live NBA weights with the stack abstaining) and live_full
  * paired live_full - live_no_stack Brier per 1000 with z (negative = the
    stack helps)

Model-side only: no Pinnacle history exists for these seasons, so this measures
calibration vs outcomes, not CLV. The matchup agent shifts a sharp probability;
here its base is the live-weighted blend of the other models.

Usage:
    uv run python scripts/backtest_nba_stack.py                       # warm 2324, score 2425+2526
    uv run python scripts/backtest_nba_stack.py --snapshot-every 1    # live cadence (slow cold)
    uv run python scripts/backtest_nba_stack.py --offline             # cached payloads only
    uv run python scripts/backtest_nba_stack.py --from-results data/backtest/nba_stack/results_*.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# possession_sim seeds its RNG from hash(str); pin it so reruns reproduce.
if os.environ.get("PYTHONHASHSEED") != "0" and "--no-reexec" not in sys.argv:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.backtest.sources.nba_stack_walkforward import (  # noqa: E402
    RESULTS_DIR,
    STACK_MODELS,
    load_records,
    run_nba_stack_walkforward,
    save_records,
    summarize,
)


def _fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def print_summary(s: dict) -> None:
    print(f"\nGames scored (regular season): {s['n_games']}   home win rate {s['home_win_rate']}   "
          f"always-home Brier {_fmt(s['always_home_brier'], 5)}")
    print("\nStandalone models")
    print(f"  {'model':<16}{'n':>6}{'coverage':>10}{'Brier':>9}{'acc':>8}")
    for m in ("elo", "form", "poisson") + STACK_MODELS:
        d = s["models"][m]
        cov = s["coverage"].get(m)
        print(f"  {m:<16}{d['n']:>6}{(_fmt(cov, 3) if cov is not None else '—'):>10}"
              f"{_fmt(d['brier'], 5):>9}{_fmt(d['accuracy'], 3):>8}")
    print("\nBlends")
    print(f"  {'blend':<16}{'n':>6}{'Brier':>9}{'acc':>8}")
    for b, d in s["blends"].items():
        print(f"  {b:<16}{d['n']:>6}{_fmt(d['brier'], 5):>9}{_fmt(d['accuracy'], 3):>8}")
    print("\nPaired: live_full - live_no_stack  (Brier per 1000, negative = stack helps)")
    for k, d in s["paired"].items():
        print(f"  {k:<18} n={d['n']:<6} delta={_fmt(d['delta_per_1000'], 3):>8}  z={_fmt(d['z'], 2)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmup-seasons", default="2324", help="comma list, trains elo/form/poisson only")
    ap.add_argument("--score-seasons", default="2425,2526")
    ap.add_argument("--snapshot-every", type=int, default=3,
                    help="game days between stats snapshots (1 = live cadence)")
    ap.add_argument("--offline", action="store_true", help="use cached nba_api payloads only")
    ap.add_argument("--pause", type=float, default=0.6, help="seconds between uncached nba_api calls")
    ap.add_argument("--include-playoffs", action="store_true", help="score playoff games too (sim has no playoff flag)")
    ap.add_argument("--from-results", help="re-summarize a saved results JSON, no fetching")
    ap.add_argument("--no-reexec", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    score = [s for s in args.score_seasons.split(",") if s]
    if args.from_results:
        records = load_records(Path(args.from_results))
    else:
        warm = [s for s in args.warmup_seasons.split(",") if s]
        records, fstats = run_nba_stack_walkforward(
            warm, score, snapshot_every=args.snapshot_every,
            offline=args.offline, pause_s=args.pause,
        )
        out = RESULTS_DIR / f"results_{'_'.join(score)}_every{args.snapshot_every}.json"
        save_records(records, out)
        print(f"saved {len(records)} game records -> {out}")
        print(f"nba_api: {fstats.network_calls} network calls, {fstats.cache_hits} cache hits, "
              f"{fstats.failures} failures")
        if fstats.failures:
            print("WARNING: fetch failures leave snapshots empty (stack abstains for those games); "
                  "rerun to fill gaps from a clean cache.")

    for code in score:
        print(f"\n===== season {code} =====")
        print_summary(summarize(records, regular_only=not args.include_playoffs, score_seasons=[code]))
    if len(score) > 1:
        print("\n===== pooled =====")
        print_summary(summarize(records, regular_only=not args.include_playoffs, score_seasons=score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
