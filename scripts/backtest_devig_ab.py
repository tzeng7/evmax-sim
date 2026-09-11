"""A/B the sharp-anchor devig method (power / shin / multiplicative) on CLV/Brier.

The devig method turns Pinnacle's vigged decimal odds into the "true" sharp
probability that anchors every EV gap (settings.devig_method, default power).
Power already handles favorite/underdog asymmetry via its exponent; Shin models
an insider fraction and shades longshots down; multiplicative does neither. This
harness measures which one best predicts outcomes, per sector and market shape,
so a change is made on evidence — never on the shipped default.

For every archived Pinnacle line (archive.db `archived_sharp_odds`, which stores
the raw decimals) that later resolved (predictions.db `ev_outcomes`), it
re-devigs the SAME decimals three ways and scores each method's side-A
probability against whether side A actually won. Lower Brier is better.

This manual lens complements the AUTOMATIC selector
(``evmax/ev/devig_selection.py``, surfaced weekly by the integrity sweep,
applied via ``evmax cleanup devig promote``). Devig quality is calibration vs
OUTCOMES (Brier), not CLV — devigging EXTRACTS the book's own probability, it
isn't a forecast trying to beat the close. The tennis lesson survives only as
the SIGNIFICANCE guard: the auto-gate needs a material Brier delta (>=2/1000)
AND a paired z (>=1.64) AND n>=200, so a noise-floor edge never flips a sector.
Use this to size the effect and catch sign bugs by hand.

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

    # Source DBs: predictions.db (ev_outcomes) via get_connection, and
    # archive.db (archived_sharp_odds — the raw Pinnacle decimals) opened below.
    from evmax.agents.cleanup.db import get_connection
    from evmax.ev.devig import DEVIG_METHODS, devig

    # Resolved moneyline/3-way outcomes with a winning side, keyed by event_id.
    # ev_outcomes.outcome == 1 means the YES side (yes_team) won.
    with get_connection() as conn:
        outcomes = {
            r["event_id"]: (r["yes_team"], int(r["outcome"]), (r["sector"] or "").lower())
            for r in conn.execute(
                """
                SELECT event_id, yes_team, outcome, sector
                FROM ev_outcomes
                WHERE outcome IS NOT NULL AND event_id IS NOT NULL
                  AND resolved_at >= datetime('now', ?)
                """,
                (f"-{args.days} days",),
            )
        }
    if not outcomes:
        print("No resolved outcomes in window — nothing to A/B.\n"
              "(Expected on a fresh checkout; this harness runs on the prod DB.)")
        return 0

    # archive.db carries the raw Pinnacle decimals. Connect read-only.
    import sqlite3
    archive_path = _REPO_ROOT / "data" / "archive.db"
    if not archive_path.exists():
        print(f"archive.db not found at {archive_path} — run on the prod box.")
        return 0
    arc = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
    arc.row_factory = sqlite3.Row

    # brier[(sector, shape, method)] = [(prob_a, side_a_won), ...]
    brier: dict[tuple, list[tuple[float, int]]] = defaultdict(list)

    rows = arc.execute(
        """
        SELECT event_id, sector, outcome_a_label, outcome_b_label,
               outcome_a_decimal, outcome_b_decimal, outcome_draw_decimal
        FROM archived_sharp_odds
        WHERE outcome_a_decimal IS NOT NULL AND outcome_b_decimal IS NOT NULL
        """
    )
    for r in rows:
        ev = r["event_id"]
        if ev not in outcomes:
            continue
        sec = (r["sector"] or "").lower()
        if sectors and sec not in sectors:
            continue
        yes_team, won, _ = outcomes[ev]
        # Map the winning YES side onto side A/B via the archived labels.
        a_label = (r["outcome_a_label"] or "").lower()
        yes = (yes_team or "").lower()
        if yes and yes == a_label:
            side_a_won = won
        elif yes and yes == (r["outcome_b_label"] or "").lower():
            side_a_won = 1 - won
        else:
            continue  # label mismatch — don't guess a side
        draw = r["outcome_draw_decimal"]
        decimals = [r["outcome_a_decimal"], r["outcome_b_decimal"]]
        if draw:
            decimals.append(draw)
        shape = _shape(bool(draw))
        for method in DEVIG_METHODS:
            try:
                res = devig(decimals, method=method)
            except Exception:
                continue
            brier[(sec, shape, method)].append((res.true_probs[0], side_a_won))

    arc.close()

    if not brier:
        print("No archived sharp lines joined to a resolved outcome in window.")
        return 0

    def _brier(pairs) -> float:
        return sum((p - o) ** 2 for p, o in pairs) / len(pairs) if pairs else float("nan")

    print(f"{'sector':10} {'shape':16} {'method':16} {'n':>6} {'Brier':>9}")
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
    print("CLV under DEVIG_METHOD=<method> (evmax cleanup shadow clv). Power ships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
