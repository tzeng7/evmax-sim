"""Walk-forward value test for Tennis Abstract surface Elo (ATP).

Validates that the surface Elo seeded from tennisabstract.com (see
``scripts/seed_tennis_abstract_elo.py``) has genuine out-of-sample forecasting
value before it is trusted live.

Method: Tennis Abstract publishes only the *current* Elo snapshot, so we pull
historical monthly snapshots from the Wayback Machine. For each snapshot dated D,
seed the model from it and predict every ATP match played in ``(D, D+WINDOW]`` —
genuinely out-of-sample, since the snapshot predates those results. Score Brier +
accuracy against the Pinnacle closing line (devigged) and a coin-flip baseline.

Ground truth + market come from the tennis-data.co.uk ATP files
(``data/backtest/tennis/{year}.xlsx``: Winner/Loser/Surface/Date/PSW/PSL). Player
names are normalized to the live-resolver form (drop the trailing initial,
"Sinner J." -> "Sinner") so coverage mirrors production.

NOTE: only ATP is covered — the cached tennis-data files are the men's tour. The
WTA pipeline is identical; its surface Elo went from a near-flat incumbent
(Elo spread ~110) to a real spread (~414), but is not walk-forward scored here for
lack of a cached WTA results file.

Last run (2026-06-27, 16 monthly snapshots, 2025-01 -> 2026-04, WINDOW=42d):
    scored 2865 matches (62% coverage)
      TA surface Elo : Brier 0.2239  acc 0.640
      Pinnacle close : Brier 0.2094  acc 0.675
      coin flip      : Brier 0.2500
    => strong skill vs baseline, within ~0.0145 Brier of the sharp close, and
       beats the old tennis-data.co.uk source by ~0.014 Brier / +10pp acc on an
       equal-vintage 2026-Q1 common set.

Usage:
    uv run python scripts/backtest_tennis_abstract_elo.py
    uv run python scripts/backtest_tennis_abstract_elo.py --window 35
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tempfile
from pathlib import Path

import httpx
import openpyxl

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.tennis_model_agent import TennisModelAgent  # noqa: E402
from evmax.clients.tennisabstract import parse_elo_html  # noqa: E402
from evmax.ev.devig import devig_two_way  # noqa: E402

# Wayback HTML snapshots are cached to a git-tracked path so cloud runs can
# read from committed snapshots rather than re-fetching from web.archive.org.
# Fallback: tempfile.gettempdir() for local dev.
CACHE = _REPO_ROOT / "data/backtest/tennis/wayback_cache"


def _get(url: str, timeout: float = 90.0, retries: int = 4) -> str | None:
    for _ in range(retries):
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
    return None


def wayback_snapshots(tour: str, frm: str, to: str) -> list[str]:
    txt = _get(
        f"http://web.archive.org/cdx/search/cdx?url=tennisabstract.com/reports/"
        f"{tour}_elo_ratings.html&output=json&from={frm}&to={to}&collapse=timestamp:6"
    )
    rows = json.loads(txt)[1:] if txt else []
    return sorted({x[1] for x in rows})


def snapshot_html(ts: str, tour: str) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{tour}_{ts}.html"
    if f.exists():
        return f.read_text()
    txt = _get(f"http://web.archive.org/web/{ts}id_/https://tennisabstract.com/reports/{tour}_elo_ratings.html")
    if txt:
        f.write_text(txt)
    return txt


def _norm(name) -> str:
    return re.sub(r"\s+[A-Za-z]\.?\s*$", "", str(name)).strip()


def load_atp_results(years: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    for yr in years:
        p = _REPO_ROOT / "data" / "backtest" / "tennis" / f"{yr}.xlsx"
        if not p.exists():
            continue
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        hdr = [str(c).strip() if c else "" for c in next(it)]
        ix = {k: (hdr.index(k) if k in hdr else None) for k in ("Date", "Winner", "Loser", "Surface", "PSW", "PSL")}
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
            rows.append(dict(date=ds, winner=_norm(w), loser=_norm(l), surface=surf,
                             psw=r[ix["PSW"]], psl=r[ix["PSL"]]))
        wb.close()
    return rows


def seed_from_snapshot(rows) -> TennisModelAgent:
    ag = TennisModelAgent()
    ag._state_path = CACHE / "_tmp_state.json"
    ag._state = {}
    rbs = {s: {} for s in ("overall", "hard", "clay", "grass", "indoor")}
    for p in rows:
        rbs["overall"][p.player] = p.elo
        if p.hard is not None:
            rbs["hard"][p.player] = p.hard
            rbs["indoor"][p.player] = p.hard
        if p.clay is not None:
            rbs["clay"][p.player] = p.clay
        if p.grass is not None:
            rbs["grass"][p.player] = p.grass
    ag.seed_precomputed_surface_elo(rbs)
    return ag


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward value test for TA surface Elo (ATP)")
    ap.add_argument("--window", type=int, default=42, help="Days after a snapshot to score (default 42)")
    ap.add_argument("--from", dest="frm", default="20250101")
    ap.add_argument("--to", default="20260420")
    args = ap.parse_args()

    results = load_atp_results((2025, 2026))
    if not results:
        print("No ATP results found in data/backtest/tennis — aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"ATP results: {len(results)} matches "
          f"({min(r['date'] for r in results)} -> {max(r['date'] for r in results)})")

    snaps = wayback_snapshots("atp", args.frm, args.to)
    if not snaps and CACHE.exists():
        # Wayback CDX is flaky; fall back to whatever snapshots are already cached.
        snaps = sorted(f.stem.split("_", 1)[1] for f in CACHE.glob("atp_*.html"))
        print("  (Wayback CDX unavailable — using cached snapshots)")
    print(f"Wayback snapshots: {len(snaps)}")

    n = 0
    bm = bk = bb = 0.0
    am = ak = 0
    window_total = unrated = noodds = 0
    by_surface: dict[str, list] = {}

    for ts in snaps:
        html = snapshot_html(ts, "atp")
        if not html:
            continue
        rows = parse_elo_html(html)
        if not rows:
            continue
        D = dt.date(int(ts[:4]), int(ts[4:6]), int(ts[6:8]))
        ag = seed_from_snapshot(rows)
        for m in results:
            if not (D < m["date"] <= D + dt.timedelta(days=args.window)):
                continue
            window_total += 1
            ew = ag.get_rating(m["winner"], m["surface"])
            el = ag.get_rating(m["loser"], m["surface"])
            if ew == 1500.0 or el == 1500.0:
                unrated += 1
                continue
            if not m["psw"] or not m["psl"]:
                noodds += 1
                continue
            try:
                pk, _pl, _mg = devig_two_way(float(m["psw"]), float(m["psl"]))
            except Exception:
                noodds += 1
                continue
            pm = 1.0 / (1.0 + 10.0 ** ((el - ew) / 400.0))
            n += 1
            bm += (pm - 1) ** 2
            bk += (pk - 1) ** 2
            bb += 0.25
            am += 1 if pm > 0.5 else 0
            ak += 1 if pk > 0.5 else 0
            s = by_surface.setdefault(m["surface"], [0, 0.0, 0.0, 0])
            s[0] += 1; s[1] += (pm - 1) ** 2; s[2] += (pk - 1) ** 2; s[3] += 1 if pm > 0.5 else 0

    if n == 0:
        print("No matches scored (Wayback unavailable?).", file=sys.stderr)
        sys.exit(1)

    print(f"\nwindow matches: {window_total} | scored: {n} | "
          f"unrated: {unrated} | no-odds: {noodds} (coverage {n/window_total*100:.0f}%)")
    print(f"\n{'':16}{'Brier':>10}{'Accuracy':>11}")
    print(f"{'TA surface Elo':16}{bm/n:>10.4f}{am/n:>11.3f}")
    print(f"{'Pinnacle close':16}{bk/n:>10.4f}{ak/n:>11.3f}")
    print(f"{'coin flip 0.50':16}{bb/n:>10.4f}{'-':>11}")
    print(f"\nΔBrier (model - market): {bm/n - bk/n:+.4f}  "
          f"(negative = model better; ~0 = competitive with the sharp line)")
    print("\nBy surface:")
    for surf, (c, sm, sk, ac) in sorted(by_surface.items()):
        print(f"  {surf:<7} n={c:<5} Brier model={sm/c:.4f} market={sk/c:.4f} | model acc={ac/c:.3f}")


if __name__ == "__main__":
    main()
