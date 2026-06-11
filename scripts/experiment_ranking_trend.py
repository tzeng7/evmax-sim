"""Fast offline experiments on the tennis ranking-trend formula.

The full walk-forward replays all 6 agents (~4 min). The trend agent's
prediction depends only on (player names, match date, ranking state), so
candidate formulas can be evaluated in seconds against the same
tennis-data.co.uk matches the walk-forward uses.

Candidates (all anchored at the match date, staleness-guarded):
  cur        — production: raw rank-spot momentum, k=0.007/spot, cap 0.40
  logmom     — log-rank momentum: k * (Δlog_a - Δlog_b); 100→50 == 4→2
  rank       — absolute log-rank differential: alpha * (ln r_b - ln r_a)
  rank+mom   — alpha * absolute + k * log-momentum

Params are swept on the TRAIN year (2024) and frozen before reading the
eval years (2025+2026), reported as standalone Brier/acc on fired rows and
as the production blend (sharp 0.85 + default ramp) delta vs sharp.

Usage:
    uv run python scripts/experiment_ranking_trend.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from math import exp, log
from pathlib import Path
from typing import Callable, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR.parent))

from backtest_tennis_walkforward import load_tennis_matches  # noqa: E402
from compare_tennis_blend_buckets import DEFAULT_RAMP, PROD_SHARP_WEIGHT  # noqa: E402
from backtest_tennis_walkforward import disagreement_ramp  # noqa: E402
from evmax.agents.models.tennis_ranking_trend_agent import (  # noqa: E402
    TennisRankingTrendAgent,
    _MIN_SNAPSHOTS,
    _STALE_WEEKS,
    _TREND_WEEKS,
)
from evmax.backtest.metrics import accuracy_score, brier_score  # noqa: E402

TRAIN_YEARS = {2024}
EVAL_YEARS = {2025, 2026}


def snapshots_asof(history: list[dict], as_of: date) -> Optional[list[dict]]:
    """Anchored, staleness-guarded snapshot list (same rules as the agent)."""
    snaps = [s for s in sorted(history, key=lambda r: r["date"])
             if date.fromisoformat(s["date"]) <= as_of]
    if len(snaps) < _MIN_SNAPSHOTS:
        return None
    latest_date = date.fromisoformat(snaps[-1]["date"])
    if (as_of - latest_date) > timedelta(weeks=_STALE_WEEKS):
        return None
    return snaps


def momentum(snaps: list[dict], log_space: bool) -> float:
    """Rank improvement over the trend window. Positive = improving."""
    latest = snaps[-1]
    cutoff = date.fromisoformat(latest["date"]) - timedelta(weeks=_TREND_WEEKS)
    baseline = snaps[0]
    for s in snaps:
        if date.fromisoformat(s["date"]) <= cutoff:
            baseline = s
        else:
            break
    if log_space:
        return log(float(baseline["rank"])) - log(float(latest["rank"]))
    return float(baseline["rank"]) - float(latest["rank"])


# Each candidate: (snaps_a, snaps_b) → logit for P(A wins)
def make_cur() -> Callable:
    def f(sa, sb):
        diff = momentum(sa, False) - momentum(sb, False)
        return max(-0.40, min(0.40, diff * 0.007))
    return f


def make_logmom(k: float, cap: float = 0.40) -> Callable:
    def f(sa, sb):
        diff = momentum(sa, True) - momentum(sb, True)
        return max(-cap, min(cap, k * diff))
    return f


def make_rank(alpha: float) -> Callable:
    def f(sa, sb):
        return alpha * (log(float(sb[-1]["rank"])) - log(float(sa[-1]["rank"])))
    return f


def make_rank_mom(alpha: float, k: float, cap: float = 0.40) -> Callable:
    rank_f, mom_f = make_rank(alpha), make_logmom(k, cap)
    def f(sa, sb):
        return rank_f(sa, sb) + mom_f(sa, sb)
    return f


def main() -> int:
    agent = TennisRankingTrendAgent()
    resolve_cache: dict[str, Optional[list[dict]]] = {}

    def histories(player: str) -> Optional[list[dict]]:
        key = player.lower().strip()
        if key not in resolve_cache:
            resolve_cache[key] = agent._resolve(key)
        return resolve_cache[key]

    years = sorted(TRAIN_YEARS | EVAL_YEARS)
    print(f"Loading matches {years}...")
    matches = load_tennis_matches(years)

    # Pre-resolve per match: (snaps_a, snaps_b, a_won, sharp_a, year)
    rows = []
    n_unresolved = n_guarded = 0
    for m in matches:
        if m.winner.lower() <= m.loser.lower():
            pa, pb, a_won, sharp_a = m.winner, m.loser, True, m.sharp_winner_prob
        else:
            pa, pb, a_won, sharp_a = m.loser, m.winner, False, 1.0 - m.sharp_winner_prob
        ha, hb = histories(pa), histories(pb)
        if ha is None or hb is None:
            n_unresolved += 1
            continue
        sa, sb = snapshots_asof(ha, m.date), snapshots_asof(hb, m.date)
        if sa is None or sb is None:
            n_guarded += 1
            continue
        rows.append((sa, sb, a_won, sharp_a, m.date.year))

    train = [r for r in rows if r[4] in TRAIN_YEARS]
    eval_ = [r for r in rows if r[4] in EVAL_YEARS]
    print(f"rows: {len(rows)} usable ({n_unresolved} unresolved, {n_guarded} guarded) "
          f"| train {len(train)}, eval {len(eval_)}")

    def metrics(rows_, logit_fn):
        ps = [1.0 / (1.0 + exp(-logit_fn(sa, sb))) for sa, sb, *_ in rows_]
        acts = [a for _, _, a, _, _ in rows_]
        sharp = [s for _, _, _, s, _ in rows_]
        blend = []
        for p, s in zip(ps, sharp):
            sw = disagreement_ramp(PROD_SHARP_WEIGHT, p, s, **DEFAULT_RAMP)
            blend.append(sw * s + (1 - sw) * p)
        return (brier_score(ps, acts), accuracy_score(ps, acts),
                brier_score(sharp, acts), brier_score(blend, acts))

    # ---- Sweep on TRAIN ----
    print(f"\n=== TRAIN ({sorted(TRAIN_YEARS)}) sweep — standalone Brier ===")
    sweep: list[tuple[str, Callable]] = [("cur (raw spots)", make_cur())]
    for k in (0.15, 0.30, 0.45, 0.60):
        sweep.append((f"logmom k={k}", make_logmom(k)))
    for a in (0.40, 0.55, 0.70, 0.85):
        sweep.append((f"rank alpha={a}", make_rank(a)))
    for a in (0.40, 0.55, 0.70):
        for k in (0.15, 0.30, 0.45):
            sweep.append((f"rank+mom a={a} k={k}", make_rank_mom(a, k)))

    print(f"{'candidate':26s}  {'Brier':>8s}  {'Acc':>7s}  {'blendΔ/1000':>12s}")
    train_results = []
    for name, fn in sweep:
        b, acc, b_sharp, b_blend = metrics(train, fn)
        train_results.append((name, fn, b, b_blend))
        print(f"  {name:24s}  {b:8.4f}  {acc*100:6.1f}%  {(b_sharp - b_blend)*1000:+12.3f}")

    # Freeze: best standalone Brier per family on train
    def best(prefix):
        cands = [(n, f, b, bb) for n, f, b, bb in train_results if n.startswith(prefix)]
        return min(cands, key=lambda t: t[2])

    chosen = [train_results[0]] + [best(p) for p in ("logmom", "rank alpha", "rank+mom")]

    # ---- Report on EVAL ----
    print(f"\n=== EVAL ({sorted(EVAL_YEARS)}) — frozen params ===")
    print(f"{'candidate':26s}  {'n':>5s}  {'Brier':>8s}  {'Acc':>7s}  "
          f"{'sharp':>8s}  {'blendΔ/1000':>12s}")
    for name, fn, *_ in chosen:
        b, acc, b_sharp, b_blend = metrics(eval_, fn)
        print(f"  {name:24s}  {len(eval_):5d}  {b:8.4f}  {acc*100:6.1f}%  "
              f"{b_sharp:8.4f}  {(b_sharp - b_blend)*1000:+12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
