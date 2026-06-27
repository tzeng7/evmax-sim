"""Do the matchmx serve/return + advanced models pull their weight, or just ride sharp?

Reuses the point-in-time walk-forward from backtest_tennis_matchmx.py, but instead of
scoring each model alone, it asks the combination question directly:

1. DECORRELATION — correlation of each model's forecast residual with the sharp's
   residual (and with each other). A model adds value to a blend only if it is wrong
   in *different* places than the sharp (low/negative residual correlation).

2. MARGINAL BRIER — on the common match set (both models + market priced), the
   ensemble blend (real tennis weights, sharp_weight) vs sharp alone:
       consensus = Σ(weight_i·conf_i·p_i) / Σ(weight_i·conf_i)
       blend     = sw·p_sharp + (1-sw)·consensus
   reported for sharp, sharp+serve, sharp+advanced, sharp+both.

3. SHARP_WEIGHT SWEEP — the decisive test. Sweep sw in [0.5,1.0]; if the Brier-
   minimizing mix is at sw<1.0, the models carry information beyond the sharp.
   If the minimum sits at sw=1.0, they are dead weight on this (close-anchored) test.

Sharp = Pinnacle CLOSE (devigged) — the hardest bar, since the close already contains
information the live line we actually bet against does not. "Beats the close" is
strong; "ties the close" can still help live (timing/CLV), which this test can't see.

Neutral framing: player A = alphabetically-first name, so outcome has variance for the
correlation step (scoring only winners would pin outcome≡1).

Usage:
    uv run python scripts/analyze_tennis_blend_contribution.py
"""
from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evmax.agents.models import tennis_advanced_stats_agent as adv_mod  # noqa: E402
from evmax.agents.models import tennis_serve_return_agent as sr_mod  # noqa: E402
from evmax.clients.tennisabstract import fetch_matchmx  # noqa: E402
from evmax.ev.devig import devig_two_way  # noqa: E402
import backtest_tennis_matchmx as bt  # noqa: E402

# Real tennis ensemble weights (ensemble_agent.SECTOR_WEIGHT_OVERRIDES["tennis"]).
W_SERVE, W_ADV = 0.25, 0.15
SHARP_WEIGHT = 0.85  # CLI/config default


def _spw_with_conf(sr, player, surface, ref_date):
    rec = sr._get_spw(player, surface, ref_date)
    return rec  # (spw, weighted_pts) or None


def _serve_prob_A(sr, a, b, surface, ref_date, best_of):
    ra, rb = _spw_with_conf(sr, a, surface, ref_date), _spw_with_conf(sr, b, surface, ref_date)
    if ra is None or rb is None:
        return None, None
    p = sr_mod._match_win_prob(ra[0], rb[0], best_of=best_of)
    p = max(0.02, min(0.98, p))
    conf = min(0.85, 0.55 + min(ra[1], rb[1]) / 2000.0)
    return p, conf


def _adv_prob_A(adv, a, b):
    ra, rb = adv._lookup(a), adv._lookup(b)
    if ra is None or rb is None:
        return None, None
    if ra.get("matches", 0) < adv_mod._MIN_MATCHES or rb.get("matches", 0) < adv_mod._MIN_MATCHES:
        return None, None
    fa, exta = adv._compute_features(ra)
    fb, extb = adv._compute_features(rb)
    if not fa or not fb:
        return None, None
    if exta and extb:
        diff = [x - y for x, y in zip(fa, fb)]
        z = sum(c * d for c, d in zip(adv_mod._COEFS_FULL, diff)) + adv_mod._INTERCEPT_FULL
    else:
        z = adv_mod._COEFS_REDUCED[0] * (fa[1] - fb[1]) + adv_mod._INTERCEPT_REDUCED
    p = max(0.02, min(0.98, 1.0 / (1.0 + math.exp(-z))))
    conf = min(0.80, 0.50 + min(ra["matches"], rb["matches"]) / 500.0)
    return p, conf


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def _consensus(parts):
    """parts: list of (weight, conf, prob_A). Confidence-weighted mean of probs."""
    num = sum(w * c * p for w, c, p in parts)
    den = sum(w * c for w, c, _ in parts)
    return num / den if den else None


def main():
    print("Fetching matchmx (atp)...")
    mx = fetch_matchmx("atp")
    results = bt.load_atp_results((2024, 2025, 2026))

    months = []
    y, m = 2025, 1
    while (y, m) <= (2026, 4):
        months.append((y, m)); m += 1
        if m > 12:
            m = 1; y += 1

    sr, adv = bt._new_agents()
    rows = []  # (outcome_A, p_mkt_A, p_sr_A, conf_sr, p_adv_A, conf_adv)

    for (y, mo) in months:
        bt._seed_pointintime(sr, adv, mx, y * 10000 + mo * 100 + 1)
        for r in results:
            if (r["date"].year, r["date"].month) != (y, mo):
                continue
            if not r["psw"] or not r["psl"]:
                continue
            try:
                p_mkt_winner, _, _ = devig_two_way(float(r["psw"]), float(r["psl"]))
            except Exception:
                continue
            # neutral framing: A = alphabetically first
            a, b = sorted((r["winner"], r["loser"]))
            a_won = (a == r["winner"])
            outcome_A = 1.0 if a_won else 0.0
            p_mkt_A = p_mkt_winner if a_won else 1.0 - p_mkt_winner

            p_sr_A, c_sr = _serve_prob_A(sr, a, b, r["surface"], r["date"], r["best_of"])
            p_adv_A, c_adv = _adv_prob_A(adv, a, b)
            rows.append((outcome_A, p_mkt_A, p_sr_A, c_sr, p_adv_A, c_adv))

    common = [r for r in rows if r[2] is not None and r[4] is not None]
    print(f"\nrows with market: {len(rows)} | common (serve+adv+market): {len(common)}")

    # ---- 1. decorrelation ----
    out = [r[0] for r in common]
    res_mkt = [o - r[1] for o, r in zip(out, common)]
    res_sr = [o - r[2] for o, r in zip(out, common)]
    res_adv = [o - r[4] for o, r in zip(out, common)]
    print("\n=== residual correlation (lower = more decorrelated = more to add) ===")
    print(f"  serve/return vs sharp : {_pearson(res_sr, res_mkt):+.3f}")
    print(f"  advanced     vs sharp : {_pearson(res_adv, res_mkt):+.3f}")
    print(f"  serve/return vs advanced: {_pearson(res_sr, res_adv):+.3f}")

    # ---- 2. marginal Brier at the real weights ----
    def brier(probs):
        return sum((p - o) ** 2 for p, o in zip(probs, out)) / len(out)

    def blended(include_sr, include_adv, sw=SHARP_WEIGHT):
        probs = []
        for o, p_mkt, p_sr, c_sr, p_adv, c_adv in common:
            parts = []
            if include_sr:
                parts.append((W_SERVE, c_sr, p_sr))
            if include_adv:
                parts.append((W_ADV, c_adv, p_adv))
            cons = _consensus(parts)
            probs.append(p_mkt if cons is None else sw * p_mkt + (1 - sw) * cons)
        return probs

    print(f"\n=== marginal Brier on common set (sharp_weight={SHARP_WEIGHT}) ===")
    b_sharp = brier([r[1] for r in common])
    print(f"  sharp alone        : {b_sharp:.5f}")
    for label, isr, iadv in [("sharp + serve", True, False),
                             ("sharp + advanced", False, True),
                             ("sharp + both", True, True)]:
        b = brier(blended(isr, iadv))
        print(f"  {label:<18} : {b:.5f}  Δ {b - b_sharp:+.5f} ({(b-b_sharp)*1000:+.2f}/1000)")

    # ---- 3. sharp_weight sweep (decisive) ----
    print("\n=== sharp_weight sweep, sharp+both (argmin sw<1.0 => models add info) ===")
    best = (1.0, b_sharp)
    for i in range(11):
        sw = 0.50 + 0.05 * i
        b = brier(blended(True, True, sw=sw))
        mark = ""
        if b < best[1]:
            best = (sw, b)
        print(f"  sw={sw:.2f}  Brier={b:.5f}{mark}")
    print(f"  -> min at sw={best[0]:.2f} (Brier {best[1]:.5f}); "
          f"sw=1.00 is pure sharp = {brier([r[1] for r in common]):.5f}")


if __name__ == "__main__":
    main()
