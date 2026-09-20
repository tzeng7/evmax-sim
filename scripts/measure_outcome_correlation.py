"""Measure realized outcome correlation of same-game legs (Phase 4 evidence).

The exposure guard's ``same_side_kelly_discount`` (default 0.5) is a hand-set stand-in
for an assumed ρ ≈ 0.8 between correlated legs on one game (ML + same-team spread, alt
spread stacks). This script measures the REALIZED correlation from resolved rows so the
0.5 is grounded in data rather than assumed — or replaced by the joint-Kelly path, which
models the correlation structure directly.

Method: dedup to one resolved row per (game, market_type, yes_team, line) to strip the
scan/rung duplication that inflates naive pair counts, then correlate the binary outcomes
of every leg pair sharing a base event, split into same-side (identical yes_team) and
opposite-side, plus the specific ML + same-team-spread pair.

Caveat printed with the result: pooled binary-outcome correlation understates the
correlation of the Kelly RETURNS and is confounded by the spread magnitude mix; treat
the number as a floor and prefer joint Kelly for the principled fix.

Usage:
    uv run python scripts/measure_outcome_correlation.py [--days 365] [--sector nba]
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

DB = Path(__file__).resolve().parents[1] / "data" / "predictions.db"


def _corr(pairs: list[tuple[int, int]]) -> tuple[float | None, int]:
    if len(pairs) < 20:
        return None, len(pairs)
    x = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)
    if x.std() == 0 or y.std() == 0:
        return None, len(pairs)
    return float(np.corrcoef(x, y)[0, 1]), len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--sector", default=None)
    args = ap.parse_args()

    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    q = """
        SELECT p.event_id, p.sector, p.market_type, p.yes_team, p.line, o.outcome
        FROM ev_predictions p JOIN ev_outcomes o ON o.market_id = p.market_id
        WHERE o.outcome IS NOT NULL AND p.voided = 0
          AND p.scan_date >= date('now', ?)
    """
    params = [f"-{args.days} day"]
    if args.sector:
        q += " AND p.sector = ?"
        params.append(args.sector)
    q += " GROUP BY p.market_id"
    rows = con.execute(q, params).fetchall()
    con.close()

    # Dedup to one leg per (game, market_type, yes_team, line).
    dedup: dict[tuple, dict] = {}
    for r in rows:
        base = (r["event_id"] or "").split("::")[0]
        key = (base, r["market_type"], (r["yes_team"] or "").lower().strip(), r["line"] or 0)
        dedup.setdefault(key, r)
    by_game: dict[str, list] = defaultdict(list)
    for (base, *_), r in dedup.items():
        by_game[base].append(r)

    same, opp, ml_spread = [], [], []
    for base, legs in by_game.items():
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                a, b = legs[i], legs[j]
                ta = (a["yes_team"] or "").lower().strip()
                tb = (b["yes_team"] or "").lower().strip()
                pair = (a["outcome"], b["outcome"])
                if ta and tb and ta == tb:
                    same.append(pair)
                    if {a["market_type"], b["market_type"]} == {"moneyline", "spread"}:
                        ml_spread.append(pair)
                else:
                    opp.append(pair)

    print(f"resolved rows={len(rows)}  deduped legs={len(dedup)}  games={len(by_game)}"
          f"  sector={args.sector or 'ALL'}  days={args.days}\n")
    for name, pairs in [("same-side (same team, diff market/line)", same),
                        ("opposite-side", opp),
                        ("ML + same-team spread", ml_spread)]:
        c, n = _corr(pairs)
        cs = f"{c:+.3f}" if c is not None else "n/a"
        print(f"  {name:42s} corr={cs}  n={n}")

    c_same, _ = _corr(same)
    print("\nInterpretation:")
    print("  same_side_kelly_discount = 0.5 assumes ρ ≈ 0.8 (≈half of 1x combined Kelly).")
    if c_same is not None:
        # ρ→discount heuristic: discount ≈ 1/(1+ρ) pulls a 2-leg same-side position
        # back toward 1x effective Kelly. Report it as a data-grounded alternative.
        implied = 1.0 / (1.0 + max(0.0, c_same))
        print(f"  measured same-side outcome corr ≈ {c_same:+.3f} → implied discount ≈ {implied:.2f}")
        print("  NOTE: binary-outcome corr understates Kelly-return corr and is confounded by")
        print("  the spread-magnitude mix. Do not hand-set the discount from this alone —")
        print("  validate the joint-Kelly path (joint_kelly_enabled) against it before changing 0.5.")


if __name__ == "__main__":
    main()
