"""Benchmark harness for the scan cycle — the measurement gate for scan-perf work.

Reports, for the current checkout:

  1. Full-cycle wall time (all in-season sectors, run N times, median)
  2. Isolated per-sector wall time for the sectors that matter
     (tennis = CPU pole, baseball = I/O pole, nba = starvation canary)
  3. Correctness counts per run (markets fetched / matched / gaps) so a
     "speedup" that changes outputs is caught immediately.

Protocol (see the scan-timing design doc): baseline before any change,
re-measure after each fix, keep a fix only if its target number moves AND
correctness counts are unchanged. Timings hit live APIs — expect a few
seconds of network variance between runs; that is why the full cycle runs
multiple times and the median is reported.

Run from the checkout under test (cwd determines which evmax is imported):

    python scripts/benchmark_scan_cycle.py [--runs 3] [--skip-isolated]

Read-only with respect to predictions.db: this calls run_cycle() directly,
which never persists gaps (log_gaps lives in the CLI layer). It DOES write
archive.db scan sessions and the steam cache, same as any scan.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

# Benchmark the checkout this script lives in, not whatever the venv's
# editable install points at. `python scripts/benchmark_scan_cycle.py` puts
# scripts/ (not the repo root) on sys.path, so without this line a worktree
# copy of the harness would silently measure the main checkout's code.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ISOLATED_SECTORS = ["tennis", "baseball", "nba", "soccer"]


def _fmt_result(res) -> str:
    return (
        f"fetched={res.markets_fetched} matched={res.markets_matched} "
        f"gaps={len(res.ev_gaps)}"
    )


def run_full_cycle() -> tuple[float, object, dict[str, float]]:
    """One full in-season cycle with a per-sector timer wrapped around
    _run_sector. Returns (wall_s, CycleResult, {sector: seconds})."""
    from evmax.agents.coordinator import AgentCoordinator
    from evmax.sectors.registry import ALL_SECTORS

    coord = AgentCoordinator(
        sectors=ALL_SECTORS,
        enable_models=True,
        enable_injuries=True,
        bankroll=250.0,
        kelly_fraction=0.5,
        respect_season_window=True,
    )

    timings: dict[str, float] = {}
    orig = coord._run_sector

    async def timed(sector: str, cid: str):
        t = time.perf_counter()
        try:
            return await orig(sector, cid)
        finally:
            timings[sector] = time.perf_counter() - t

    coord._run_sector = timed

    t0 = time.perf_counter()
    res = asyncio.run(coord.run_cycle())
    return time.perf_counter() - t0, res, timings


def run_isolated(sector: str) -> tuple[float, object]:
    """One single-sector cycle — no contention, so this is the sector's
    true cost. respect_season_window=False so offseason canaries still run."""
    from evmax.agents.coordinator import AgentCoordinator

    coord = AgentCoordinator(
        sectors=[sector],
        enable_models=True,
        enable_injuries=True,
        bankroll=250.0,
        kelly_fraction=0.5,
        respect_season_window=False,
    )
    t0 = time.perf_counter()
    res = asyncio.run(coord.run_cycle())
    return time.perf_counter() - t0, res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=3, help="full-cycle repetitions (median reported)")
    ap.add_argument("--skip-isolated", action="store_true", help="only run the full-cycle benchmark")
    ap.add_argument("--skip-full", action="store_true", help="only run the isolated benchmarks")
    args = ap.parse_args()

    import evmax

    print(f"benchmarking evmax at: {evmax.__file__}")

    if not args.skip_full:
        walls: list[float] = []
        for i in range(args.runs):
            wall, res, timings = run_full_cycle()
            walls.append(wall)
            slowest = sorted(timings.items(), key=lambda kv: -kv[1])[:4]
            slow_s = "  ".join(f"{s}={dt:.1f}s" for s, dt in slowest)
            print(f"full[{i + 1}/{args.runs}]  wall={wall:5.1f}s  {_fmt_result(res)}  slowest: {slow_s}")
        print(f"FULL CYCLE median of {len(walls)}: {statistics.median(walls):.1f}s")

    if not args.skip_isolated:
        for sector in ISOLATED_SECTORS:
            wall, res = run_isolated(sector)
            print(f"isolated {sector:10s} {wall:5.1f}s  {_fmt_result(res)}")


if __name__ == "__main__":
    main()
