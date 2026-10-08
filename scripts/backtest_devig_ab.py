"""A/B the sharp-anchor devig method (power / shin / multiplicative) on CLV/Brier.

The devig method turns Pinnacle's vigged decimal odds into the "true" sharp
probability that anchors every EV gap (settings.devig_method, default power).
Power already handles favorite/underdog asymmetry via its exponent; Shin models
an insider fraction and shades longshots down; multiplicative does neither. This
harness measures which one best predicts outcomes, per sector and market shape,
so a change is made on evidence — never on the shipped default.

For every resolved game (predictions.db `ev_outcomes`) it takes the closing
pre-tip Pinnacle game-winner line (archive.db `archived_sharp_odds`, which
stores the raw decimals), re-devigs the SAME decimals three ways and scores
each method's probability of the logged contract against its result. Lower
Brier is better. The sample comes from
``evmax.ev.devig_selection.collect_devig_observations``, the same function the
automatic selector reads: one observation per game, game-winner markets only,
no in-play snapshots (see that module for the 2026-10-07 NHL artifact).

This manual lens complements the AUTOMATIC selector
(``evmax/ev/devig_selection.py``, surfaced weekly by the integrity sweep,
applied via ``evmax cleanup devig promote``). Devig quality is calibration vs
OUTCOMES (Brier), not CLV — devigging EXTRACTS the book's own probability, it
isn't a forecast trying to beat the close. The tennis lesson survives only as
the SIGNIFICANCE guard: the auto-gate needs a material Brier delta (>=2/1000)
AND a paired z (>=1.64) AND n>=200 games, so a noise-floor edge never flips a
sector. Use this to size the effect and catch sign bugs by hand.

Read-only. Runs on the prod box (needs archive.db + predictions.db). Prints a
per-(sector, market-shape, method) Brier table; writes nothing.

Usage:
    uv run python scripts/backtest_devig_ab.py [--sectors nba,soccer] [--days 180]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _shape(has_draw: bool) -> str:
    return "3-way (soccer)" if has_draw else "2-way"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sectors", default="", help="Comma-separated sectors; empty = all.")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()
    sectors = {s.strip().lower() for s in args.sectors.split(",") if s.strip()}

    from evmax.ev.devig import DEVIG_METHODS
    from evmax.ev.devig_selection import collect_devig_observations

    observations = [
        o for o in collect_devig_observations(days=args.days)
        if not sectors or o.sector in sectors
    ]
    if not observations:
        print("No resolved game joined to an archived pre-tip Pinnacle line in window.\n"
              "(Expected on a fresh checkout; this harness runs on the prod DB.)")
        return 0

    # brier[(sector, shape, method)] = [(prob, won), ...] — one pair per game.
    brier: dict[tuple, list[tuple[float, int]]] = defaultdict(list)
    for o in observations:
        shape = _shape(o.three_way)
        for method, prob in o.probs.items():
            brier[(o.sector, shape, method)].append((prob, o.won))

    def _brier(pairs) -> float:
        return sum((p - o) ** 2 for p, o in pairs) / len(pairs) if pairs else float("nan")

    print(f"{'sector':10} {'shape':16} {'method':16} {'games':>6} {'Brier':>9}")
    print("-" * 62)
    keys = sorted({(s, sh) for (s, sh, _m) in brier})
    for sec, shape in keys:
        best_method, best_brier = None, float("inf")
        for method in DEVIG_METHODS:
            pairs = brier.get((sec, shape, method), [])
            if not pairs:
                continue
            b = _brier(pairs)
            if b < best_brier:
                best_method, best_brier = method, b
        for method in DEVIG_METHODS:
            pairs = brier.get((sec, shape, method), [])
            if not pairs:
                continue
            b = _brier(pairs)
            star = " *best" if method == best_method else ""
            print(f"{sec:10} {shape:16} {method:16} {len(pairs):>6} {b:>9.4f}{star}")
        print()
    print("Lower Brier is better. A sector/shape only warrants a non-power devig")
    print("when a method beats power by more than the noise floor here AND clears")
    print("the auto-gate (evmax cleanup devig show). Power ships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
