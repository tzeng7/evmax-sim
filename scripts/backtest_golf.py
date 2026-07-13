#!/usr/bin/env python
"""Walk-forward backtest of the golf skill→simulate pipeline on ESPN history.

Fetches completed PGA seasons from ESPN (one call per season, cached to disk),
replays every tournament point-in-time (skill from strictly-prior events only),
and reports calibration: is the model's win / top-10 / make-cut probability
believable — does a 15%-model golfer win ~15% of the time — and is it
systematically under/over-confident?

    python scripts/backtest_golf.py                    # 2025 + 2026, cached
    python scripts/backtest_golf.py --seasons 2025     # one season
    python scripts/backtest_golf.py --refresh          # force re-fetch
    python scripts/backtest_golf.py --sims 20000

Free ESPN data only; no paid feed. This backtests the score-proxy skill path
(the only leak-free point-in-time skill signal) — see evmax/golf/backtest.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

# Allow running as a plain script from the repo root (editable-install parity).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.golf.backtest import WalkForwardConfig, run_backtest  # noqa: E402
from evmax.golf.data.espn import parse_scoreboard  # noqa: E402
from evmax.golf.simulator import SimConfig  # noqa: E402

_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/{tour}/scoreboard?dates={year}"
_UA = "Mozilla/5.0 (evmax-golf backtest)"
_CACHE = Path(__file__).resolve().parents[1] / "data" / "golf_backtest_cache"


def fetch_season(year: int, tour: str = "pga", refresh: bool = False) -> dict:
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / f"{tour}_{year}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    url = _SCOREBOARD.format(tour=tour, year=year)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=40) as resp:  # noqa: S310
        payload = json.load(resp)
    path.write_text(json.dumps(payload))
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2025,2026", help="comma-separated years")
    ap.add_argument("--tour", default="pga")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--min-prior", type=int, default=8, help="min prior events before scoring")
    ap.add_argument("--refresh", action="store_true", help="force re-fetch of ESPN data")
    args = ap.parse_args()

    years = [int(y) for y in args.seasons.split(",")]
    all_results = []
    for y in years:
        payload = fetch_season(y, args.tour, args.refresh)
        results = parse_scoreboard(payload)
        completed = [r for r in results if r.completed]
        print(f"season {y}: {len(results)} events, {len(completed)} completed")
        all_results.extend(completed)

    cfg = WalkForwardConfig(
        min_prior_events=args.min_prior,
        sim=SimConfig(n_sims=args.sims, seed=99),
    )
    print(f"\nrunning walk-forward over {len(all_results)} completed events "
          f"(sims={args.sims}, min_prior={args.min_prior})...\n")
    report = run_backtest(all_results, cfg)
    print(report.summary())

    print("\n" + "=" * 64)
    print("READING THE RESULT")
    print("=" * 64)
    print(
        "• skill > 0  → model beats the naive base-rate baseline on that market\n"
        "• calibration: pred ≈ actual per bin → well-calibrated;\n"
        "  actual > pred in the top bins → model is UNDER-confident (its\n"
        "  favourites win MORE than it says) — the opposite of the live-market\n"
        "  read, and the sign that the model *could* be sharpened.\n"
        "• winner_capture is the model's mean win prob on eventual winners;\n"
        "  compare to ~1/field_size (a flat model) to see if there is real signal."
    )


if __name__ == "__main__":
    main()
