"""Fast deterministic signal for the tennis ensemble-weight self-improve loop.

Phase-0 signal. The point-in-time per-match model probabilities are FIXED, so we
compute them once (`build`) and cache them; thereafter `measure` just re-combines
them under whatever weights currently live in
``EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["tennis"]`` — so editing that dict in
the source IS the iteration, and re-measuring is instant and deterministic.

Train = 2025 matches (optimize on this). Holdout = 2026 Q1 (never tuned on; guard).

    uv run python scripts/tennis_weight_signal.py build      # ~60s, once
    uv run python scripts/tennis_weight_signal.py            # prints train+holdout Brier

Caveat (documented in analyze_tennis_full_blend): surface Elo here is a crude
matchmx-replay and advanced is RPW-reduced, so those two weights are understated —
trust changes driven by the serve_return/form signal, not surface/advanced.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import analyze_tennis_full_blend as afb  # noqa: E402  (runs the _resolve_surface monkeypatch)
import backtest_tennis_matchmx as bt  # noqa: E402
from evmax.agents.models.ensemble_agent import EnsembleModelAgent  # noqa: E402
from evmax.clients.tennisabstract import fetch_matchmx  # noqa: E402
from evmax.ev.devig import devig_two_way  # noqa: E402

CACHE = Path("/tmp/evmax_tennis_weight_cache.json")
# Chronological 70/30 split: optimize on the earlier matches, validate on the later
# ones. (A calendar split at 2026 fails — tennis-data.co.uk has Pinnacle odds for
# almost no 2026 matches, leaving only ~71 there.)
TRAIN_FRAC = 0.70


async def build() -> None:
    afb.TMP.mkdir(exist_ok=True)
    mx = fetch_matchmx("atp")
    results = bt.load_atp_results((2024, 2025, 2026))
    mx_sorted = sorted(mx, key=lambda r: r["tourney_date"])

    months = []
    y, m = 2025, 1
    while (y, m) <= (2026, 4):
        months.append((y, m)); m += 1
        if m > 12:
            m = 1; y += 1

    agents = afb._new_agents()
    surface = agents["tennis_surface"]
    first = months[0][0] * 10000 + months[0][1] * 100 + 1
    afb._elo_replay(surface, [r for r in mx_sorted if afb._ymd(r["tourney_date"]) < first])

    rows = []  # [outcome_A, date_iso, p_mkt_A, {model: [prob, conf]}]
    for (yy, mm) in months:
        mstart = yy * 10000 + mm * 100 + 1
        afb._reseed_batch(agents, [r for r in mx_sorted if afb._ymd(r["tourney_date"]) < mstart])
        for r in [r for r in results if (r["date"].year, r["date"].month) == (yy, mm)]:
            if not r["psw"] or not r["psl"]:
                continue
            try:
                p_mkt_w, _, _ = devig_two_way(float(r["psw"]), float(r["psl"]))
            except Exception:
                continue
            a, b = sorted((r["winner"], r["loser"]))
            a_won = a == r["winner"]
            market = afb._mk_market(a, b, r["surface"], r["date"], r["best_of"])
            sharp = afb._mk_sharp(a, b, r["date"])
            preds = await asyncio.gather(*[afb._prob_a(agents[k], market, sharp) for k in afb.WEIGHTS])
            mp = {k: [preds[i][0], preds[i][1]] for i, k in enumerate(afb.WEIGHTS) if preds[i][0] is not None}
            rows.append([1.0 if a_won else 0.0, r["date"].isoformat(),
                         p_mkt_w if a_won else 1.0 - p_mkt_w, mp])
        afb._elo_replay(surface, [r for r in mx_sorted
                                  if mstart <= afb._ymd(r["tourney_date"]) < (mstart + 100)])

    CACHE.write_text(json.dumps(rows))
    print(f"cached {len(rows)} rows -> {CACHE}")


def _brier(subset, weights, sw):
    tot = 0.0
    for o, _d, p_mkt, mp in subset:
        parts = [(weights[k], c, p) for k, (p, c) in mp.items() if k in weights]
        den = sum(w * c for w, c, _ in parts)
        cons = (sum(w * c * p for w, c, p in parts) / den) if den else None
        p = p_mkt if cons is None else sw * p_mkt + (1 - sw) * cons
        tot += (p - o) ** 2
    return tot / len(subset) if subset else float("nan")


def measure() -> None:
    if not CACHE.exists():
        print("cache missing — run: uv run python scripts/tennis_weight_signal.py build", file=sys.stderr)
        sys.exit(2)
    rows = sorted(json.loads(CACHE.read_text()), key=lambda r: r[1])
    weights = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES["tennis"]  # the live, tunable weights
    sw = afb.SHARP_WEIGHT
    cut = int(len(rows) * TRAIN_FRAC)
    train, holdout = rows[:cut], rows[cut:]
    tb = _brier(train, weights, sw)
    hb = _brier(holdout, weights, sw)
    print(f"train_brier={tb:.6f} holdout_brier={hb:.6f} "
          f"n_train={len(train)} n_holdout={len(holdout)} "
          f"weights={ {k: weights[k] for k in afb.WEIGHTS} }")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        asyncio.run(build())
    else:
        measure()
