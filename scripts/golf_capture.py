#!/usr/bin/env python
"""Freeze golf (model vs market) before a tournament, score it after it resolves.

The one clean forward test for golf edge — see evmax/golf/capture.py.

CAPTURE (run before/at tournament start):
    # live, needs Kalshi RSA creds in settings:
    python scripts/golf_capture.py capture --event THOC26 --tournament "The Open"
    # or from a manually-pulled Kalshi markets JSON ({"markets": [...]}):
    python scripts/golf_capture.py capture --event THOC26 --tournament "The Open" \
        --markets-file open_kalshi.json
    # add --calibrate to also store calibrated model probs (fit on ESPN history)

SCORE (run after the event finishes, this weekend for the Open):
    python scripts/golf_capture.py score --snapshot data/golf_captures/THOC26_*.json
    # or offline with an ESPN results JSON:
    python scripts/golf_capture.py score --snapshot <file> --results-file espn_result.json

Snapshots are written to data/golf_captures/. Free ESPN skill + outcomes; the
only paid/authed piece is the live Kalshi price pull (skippable via --markets-file).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evmax.golf.backtest import (  # noqa: E402
    WalkForwardConfig, collect_predictions, compute_actuals,
)
from evmax.golf.calibration import IsotonicCalibrator  # noqa: E402
from evmax.golf.capture import (  # noqa: E402
    build_snapshot, load_snapshot, save_snapshot, score_snapshot,
)
from evmax.golf.data.espn import parse_scoreboard, result_to_sg_proxy  # noqa: E402
from evmax.golf.models import GolfMarketType  # noqa: E402
from evmax.golf.simulator import SimConfig, simulate_field  # noqa: E402
from evmax.golf.skill import SkillConfig, skill_from_records, to_sim_inputs  # noqa: E402
from evmax.golf.venues.kalshi_golf import parse_kalshi_markets  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_CACHE = _ROOT / "data" / "golf_backtest_cache"
_CAPTURES = _ROOT / "data" / "golf_captures"
_KALSHI_SERIES = ["KXPGATOUR", "KXPGAMAKECUT", "KXPGATOP5", "KXPGATOP10", "KXPGATOP20"]
_UA = "Mozilla/5.0 (evmax-golf capture)"


def _load_espn_results(years=(2025, 2026)):
    import urllib.request
    _CACHE.mkdir(parents=True, exist_ok=True)
    results = []
    for y in years:
        path = _CACHE / f"pga_{y}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            url = f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates={y}"
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=40) as resp:  # noqa: S310
                payload = json.load(resp)
            path.write_text(json.dumps(payload))
        results.extend(r for r in parse_scoreboard(payload) if r.completed)
    return results


def _build_calibrators() -> dict:
    results = _load_espn_results()
    preds, _, _ = collect_predictions(results, WalkForwardConfig(sim=SimConfig(n_sims=8000)))
    cals = {}
    for mk in (GolfMarketType.win, GolfMarketType.top_10, GolfMarketType.make_cut):
        pairs = [(p.model[mk.value], p.actual[mk.value]) for p in preds
                 if p.had_prior and p.model.get(mk.value) is not None
                 and p.actual.get(mk.value) is not None]
        if len(pairs) >= 200:
            cals[mk.value] = IsotonicCalibrator.fit([m for m, _ in pairs], [a for _, a in pairs])
    return cals


def _fetch_kalshi_live(event: str):
    from evmax.golf.venues.kalshi_golf import fetch_golf
    fields = []
    for s in _KALSHI_SERIES:
        try:
            fields.extend(fetch_golf(series_ticker=s, event_ticker=f"{s}-{event}"))
        except Exception as exc:  # pragma: no cover - live/creds
            logging.warning("kalshi fetch failed for %s: %s", s, exc)
    return fields


def cmd_capture(args):
    if args.markets_file:
        raw = json.loads(Path(args.markets_file).read_text())
        markets = raw.get("markets", raw) if isinstance(raw, dict) else raw
        fields = parse_kalshi_markets(markets)
    else:
        fields = _fetch_kalshi_live(args.event)
    fields = [f for f in fields if f.priced_markets()]
    if not fields:
        print("no priced markets captured — need Kalshi creds or a --markets-file "
              "with live prices (the public endpoint returns null prices).")
        return

    records = []
    for r in sorted(_load_espn_results(), key=lambda e: e.date):
        records.extend(result_to_sg_proxy(r, min_field=30))
    skills = skill_from_records(records, cfg=SkillConfig())

    roster = sorted({m.golfer for f in fields for m in f.markets})
    names, means, sds = to_sim_inputs(roster, skills, SkillConfig())
    sims = {r.player: r for r in simulate_field(
        names, means, sds, SimConfig(n_sims=30000, seed=41, cut_size=min(65, len(roster) // 2)))}

    calibrators = _build_calibrators() if args.calibrate else None
    ts = datetime.now(timezone.utc).isoformat()
    snap = build_snapshot(ts, args.tournament, args.event, fields, sims, calibrators)

    _CAPTURES.mkdir(parents=True, exist_ok=True)
    stamp = ts.replace(":", "").replace("-", "")[:15]
    out = _CAPTURES / f"{args.event}_{stamp}.json"
    save_snapshot(snap, out)
    print(f"captured {len(snap.contracts)} contracts across "
          f"{len({c.market_type for c in snap.contracts})} market types → {out}")
    for mt in sorted({c.market_type for c in snap.contracts}):
        n = sum(1 for c in snap.contracts if c.market_type == mt)
        print(f"  {mt}: {n} contracts")


def _result_for(snapshot, args):
    if args.results_file:
        payload = json.loads(Path(args.results_file).read_text())
    else:
        import urllib.request
        url = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            payload = json.load(resp)
    results = parse_scoreboard(payload)
    # match by ESPN event id if present, else by tournament-name substring
    for r in results:
        if r.event_id == snapshot.event_id or r.event_id.endswith(snapshot.event_id):
            return r
    key = snapshot.tournament.lower().split()[0]
    for r in results:
        if key in r.event.lower():
            return r
    return None


def cmd_score(args):
    files = sorted(glob.glob(args.snapshot))
    if not files:
        print(f"no snapshot matched {args.snapshot}")
        return
    snap = load_snapshot(files[-1])
    result = _result_for(snap, args)
    if result is None:
        print(f"could not find ESPN results for '{snap.tournament}' "
              f"(event {snap.event_id}). Pass --results-file, or wait until it "
              f"appears on the scoreboard.")
        return
    if not result.completed:
        print(f"'{result.event}' is not completed yet (state in progress) — "
              f"score again after final round.")
        return
    actuals = compute_actuals(result)
    report = score_snapshot(snap, actuals, edge_threshold=args.edge, min_market_prob=args.min_prob)
    print(report.summary())


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="freeze model vs market before the event")
    c.add_argument("--event", required=True, help="Kalshi event suffix, e.g. THOC26")
    c.add_argument("--tournament", required=True, help='display name, e.g. "The Open"')
    c.add_argument("--markets-file", help="raw Kalshi markets JSON (skip live/creds)")
    c.add_argument("--calibrate", action="store_true", help="also store calibrated probs")
    c.set_defaults(func=cmd_capture)

    s = sub.add_parser("score", help="score a snapshot after the event resolves")
    s.add_argument("--snapshot", required=True, help="snapshot path or glob")
    s.add_argument("--results-file", help="ESPN scoreboard JSON (offline scoring)")
    s.add_argument("--edge", type=float, default=0.03, help="flag bets with model−ask ≥ this")
    s.add_argument("--min-prob", type=float, default=0.0, help="ignore contracts below this fair prob")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
