"""Walk-forward value test for the matchmx-seeded serve/return + advanced models (ATP).

Validates the models that were re-sourced from Tennis Abstract's ``matchmx`` (see
``evmax/clients/tennisabstract.py`` + ``scripts/seed_tennis_models.py``) after
Sackmann's CSVs went offline.

Unlike the Elo page, ``matchmx`` is *per-match data with dates*, so we get a clean
point-in-time walk-forward with no Wayback snapshots: for each test month, the agents
are seeded ONLY from matches that finished before the month starts, then used to
predict that month's matches.

- Feature source: ``fetch_matchmx("atp")`` (serve/return point stats, dated).
- Test universe + market + outcomes: tennis-data.co.uk ATP files
  (``data/backtest/tennis/{year}.xlsx``: Winner/Loser/Surface/Date/PSW/PSL).
- Scored: each model's P(winner) vs the Pinnacle close and a coin-flip baseline.

The advanced model runs in its RPW-reduced mode here: matchmx supplies BP/RPW but the
winners/UE feature is a current-only leaderboard (no point-in-time history), so it is
left out — an honest test of the matchmx-derived features.

Names: tennis-data "Surname F." is normalized to "Surname" so the agents' surname
resolver matches matchmx's full names, mirroring live behavior.

Last run (2026-06-27, monthly walk-forward 2025-01 -> 2026-04):
    serve/return : Brier 0.2354  acc 60.8%  (n=3099)
    advanced     : Brier 0.2274  acc 63.9%  (n=2567, RPW-reduced)
    Pinnacle     : Brier 0.2091  acc 66.9%
    coin flip    : Brier 0.2500
  => both models carry genuine OOS skill (well below coin-flip), behind the sharp
     line as expected for 0.15/0.25-weight blend ingredients. serve/return is weak
     on clay (Brier 0.2456 / 57.3%), which reproduces the known reason its weight is
     kept low — i.e. the matchmx re-sourcing preserves the models' skill profile.

Usage:
    uv run python scripts/backtest_tennis_matchmx.py
    uv run python scripts/backtest_tennis_matchmx.py --from 202501 --to 202604
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import sys
from pathlib import Path

import openpyxl

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models import tennis_advanced_stats_agent as adv_mod  # noqa: E402
from evmax.agents.models import tennis_serve_return_agent as sr_mod  # noqa: E402
from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent  # noqa: E402
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent  # noqa: E402
from evmax.clients.tennisabstract import fetch_matchmx  # noqa: E402
from evmax.ev.devig import devig_two_way  # noqa: E402

# Reuse the production aggregators so the backtest seeds exactly like the seeder.
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from seed_tennis_models import aggregate_advanced_stats, aggregate_match_serve_stats  # noqa: E402

_TMP = Path("/tmp/evmax_matchmx_bt_state.json")


def _norm(name) -> str:
    return re.sub(r"\s+[A-Za-z]\.?\s*$", "", str(name)).strip()


def load_atp_results(years) -> list[dict]:
    rows = []
    for yr in years:
        p = _REPO_ROOT / "data" / "backtest" / "tennis" / f"{yr}.xlsx"
        if not p.exists():
            continue
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        hdr = [str(c).strip() if c else "" for c in next(it)]
        ix = {k: (hdr.index(k) if k in hdr else None)
              for k in ("Date", "Winner", "Loser", "Surface", "PSW", "PSL", "Best of")}
        for r in it:
            w, l = r[ix["Winner"]], r[ix["Loser"]]
            if not w or not l:
                continue
            d = r[ix["Date"]]
            if isinstance(d, dt.datetime):
                ds = d.date()
            else:
                try:
                    ds = dt.date.fromisoformat(str(d)[:10])
                except Exception:
                    continue
            surf = (r[ix["Surface"]] or "Hard").strip().lower()
            surf = "hard" if surf == "carpet" else surf
            bo = 3
            if ix["Best of"] is not None:
                try:
                    bo = int(r[ix["Best of"]])
                except (ValueError, TypeError):
                    bo = 3
            rows.append(dict(date=ds, winner=_norm(w).lower(), loser=_norm(l).lower(),
                             surface=surf, best_of=bo, psw=r[ix["PSW"]], psl=r[ix["PSL"]]))
        wb.close()
    return rows


def _new_agents():
    sr = TennisServeReturnAgent(); sr._state_path = _TMP; sr._state = {}
    adv = TennisAdvancedStatsAgent(); adv._state_path = _TMP; adv._state = {}
    return sr, adv


def _seed_pointintime(sr, adv, mx_rows, before_yyyymmdd: int) -> None:
    """Seed both agents from matchmx rows that finished strictly before a date."""
    prior = [r for r in mx_rows if _safe_int(str(r.get("tourney_date", ""))[:8]) < before_yyyymmdd]
    sr._state = {}
    sr.seed_match_serve_stats(aggregate_match_serve_stats(prior))
    adv._state = {}
    adv.seed_stats(aggregate_advanced_stats(prior))


def _safe_int(s) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def _serve_prob(sr, winner, loser, surface, ref_date, best_of):
    a = sr._get_spw(winner, surface, ref_date)
    b = sr._get_spw(loser, surface, ref_date)
    if a is None or b is None:
        return None
    p = sr_mod._match_win_prob(a[0], b[0], best_of=best_of)
    return max(0.02, min(0.98, p))


def _adv_prob(adv, winner, loser):
    ra, rb = adv._lookup(winner), adv._lookup(loser)
    if ra is None or rb is None:
        return None
    if ra.get("matches", 0) < adv_mod._MIN_MATCHES or rb.get("matches", 0) < adv_mod._MIN_MATCHES:
        return None
    fa, exta = adv._compute_features(ra)
    fb, extb = adv._compute_features(rb)
    if not fa or not fb:
        return None
    if exta and extb:
        diff = [x - y for x, y in zip(fa, fb)]
        z = sum(c * d for c, d in zip(adv_mod._COEFS_FULL, diff)) + adv_mod._INTERCEPT_FULL
    else:
        z = adv_mod._COEFS_REDUCED[0] * (fa[1] - fb[1]) + adv_mod._INTERCEPT_REDUCED
    p = 1.0 / (1.0 + math.exp(-z))
    return max(0.02, min(0.98, p))


class Acc:
    __slots__ = ("n", "brier", "correct")

    def __init__(self):
        self.n = 0; self.brier = 0.0; self.correct = 0

    def add(self, p):  # outcome is always 1 (the winner won)
        self.n += 1; self.brier += (p - 1) ** 2; self.correct += 1 if p > 0.5 else 0

    def row(self, label):
        if not self.n:
            return f"  {label:<16} (no coverage)"
        return f"  {label:<16} n={self.n:<5} Brier={self.brier/self.n:.4f} acc={self.correct/self.n:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="202501")
    ap.add_argument("--to", default="202604")
    args = ap.parse_args()

    print("Fetching matchmx (atp)...")
    mx = fetch_matchmx("atp")
    print(f"  {len(mx)} matches")
    results = load_atp_results((2024, 2025, 2026))
    print(f"ATP results: {len(results)} matches "
          f"({min(r['date'] for r in results)} -> {max(r['date'] for r in results)})")

    # months to walk
    y0, m0 = int(args.frm[:4]), int(args.frm[4:6])
    y1, m1 = int(args.to[:4]), int(args.to[4:6])
    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1

    sr, adv = _new_agents()
    A_sr, A_adv, A_mkt, A_both_mkt = Acc(), Acc(), Acc(), Acc()
    by_surf = {}
    window_total = 0

    for (y, m) in months:
        mstart = y * 10000 + m * 100 + 1
        _seed_pointintime(sr, adv, mx, mstart)
        for r in results:
            if (r["date"].year, r["date"].month) != (y, m):
                continue
            window_total += 1
            p_sr = _serve_prob(sr, r["winner"], r["loser"], r["surface"], r["date"], r["best_of"])
            p_adv = _adv_prob(adv, r["winner"], r["loser"])
            p_mkt = None
            if r["psw"] and r["psl"]:
                try:
                    p_mkt, _, _ = devig_two_way(float(r["psw"]), float(r["psl"]))
                except Exception:
                    p_mkt = None
            if p_sr is not None:
                A_sr.add(p_sr)
                s = by_surf.setdefault(r["surface"], Acc())
                s.add(p_sr)
            if p_adv is not None:
                A_adv.add(p_adv)
            if p_mkt is not None:
                A_mkt.add(p_mkt)
                # apples-to-apples: market on the subset serve/return also covered
                if p_sr is not None:
                    A_both_mkt.add(p_mkt)

    print(f"\nwindow matches: {window_total}")
    print("\n— serve/return model vs market (matches serve/return could price) —")
    print(A_sr.row("serve/return"))
    print(A_both_mkt.row("Pinnacle (same)"))
    print("\n— advanced model (RPW-reduced) —")
    print(A_adv.row("advanced"))
    print("\n— market overall + baseline —")
    print(A_mkt.row("Pinnacle close"))
    print(f"  {'coin flip 0.50':<16} Brier=0.2500")
    print("\nserve/return by surface:")
    for surf, acc in sorted(by_surf.items()):
        print(acc.row(surf))


if __name__ == "__main__":
    main()
