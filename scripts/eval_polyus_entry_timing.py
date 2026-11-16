"""Entry timing and placement policy for Polymarket US candidates.

Reads the price-history cache built by scripts/eval_polyus_maker_replay.py
(run that first, or pass --fetch) and reports, per sector / market type:

  A. Edge decay: how fast our side's ask moves after the signal.
  B. Sampling interval: what a snapshot every k minutes misses.
  C. Placement policies: value per candidate, net of fees.
  D. Value by hours-to-tip at the signal.

Definitions: evmax/agents/cleanup/polyus_entry_timing.py. Read-only on
predictions.db.

Usage:
    python scripts/eval_polyus_entry_timing.py --pred-db data/predictions.db
    python scripts/eval_polyus_entry_timing.py --group nfl/spread --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from evmax.agents.cleanup import polyus_entry_timing as et  # noqa: E402
from evmax.agents.cleanup import polyus_maker_replay as mr  # noqa: E402
import eval_polyus_maker_replay as rp  # noqa: E402

DELAYS = (5, 15, 30, 60, 120, 180, 360, 720, 1440)
SAMPLES = (5, 15, 30, 60, 180)


def _ci(b: dict, nd: int = 2) -> str:
    if not b or b.get("mean") is None:
        return "—"
    s = f"{b['mean']:+.{nd}f}"
    if b.get("ci95"):
        s += f" [{b['ci95'][0]:+.{nd}f}, {b['ci95'][1]:+.{nd}f}]"
    return s


def report(label: str, paths: list[et.Path]) -> dict:
    games = len({p.game for p in paths})
    leads = sorted(p.lead_h for p in paths)
    print(f"\n{'=' * 100}\n{label}: {len(paths)} candidates / {games} games | "
          f"median lead {leads[len(leads) // 2]:.0f}h")
    out: dict = {"n": len(paths), "games": games}

    print("\n A. Edge decay after the signal (ask drift, pp; + = moved against us)")
    print(f"    {'delay':>8} {'n':>6} {'games':>6}   {'drift':<26} {'total move to close':<26} consumed")
    out["decay"] = et.decay_curve(paths, DELAYS)
    for d in out["decay"]:
        sc = "—" if d["share_consumed"] is None else f"{d['share_consumed'] * 100:5.0f}%"
        print(f"    {d['delay_min']:>6}m {d['n']:>6} {d['games']:>6}   "
              f"{_ci(d['drift_pp']):<26} {_ci(d['total_move_pp']):<26} {sc}")

    ttf = et.time_to_fill(paths)
    out["time_to_fill"] = ttf
    if ttf["n"]:
        within = "  ".join(f"≤{m}m {v * 100:.0f}%" for m, v in ttf["share_within"].items())
        print(f"\n    Resting bids that fill before tip: {ttf['n']} | median wait "
              f"{ttf['median_min']:.0f} min | {within}")

    print("\n B. What a point sample every k minutes sees (resting bid, rest to tip)")
    print(f"    {'sample':>10} {'fill rate':>10}   {'value if filled':<26} value per candidate")
    out["sampling"] = et.sampling_bias(paths, SAMPLES)
    for s in out["sampling"]:
        name = "bucket min" if s["sample_min"] is None else f"every {s['sample_min']}m"
        print(f"    {name:>10} {s['fill_rate'] * 100:>9.0f}%   "
              f"{_ci(s['value_if_filled_pp']):<26} {_ci(s['value_per_candidate_pp'])}")

    print("\n C. Placement policy: value per candidate (pp of contract price, net of fees)")
    out["policies"] = et.policy_table(paths)
    best = max(out["policies"], key=lambda r: r["mean"])
    for r in out["policies"]:
        mark = "  <-- best mean" if r is best else ""
        print(f"    {r['policy']:<36} {_ci(r)}{mark}")

    print("\n D. By hours-to-tip at the signal")
    print(f"    {'lead':>10} {'n':>6} {'games':>6} {'fill':>6}   {'taker gross':<24} "
          f"{'taker net':<24} maker net (rest to tip)")
    out["by_lead"] = et.by_lead(paths)
    for b in out["by_lead"]:
        lo, hi = b["lead_h"]
        name = f"{lo:g}-{hi:g}h" if hi < 1e8 else f"{lo:g}h+"
        print(f"    {name:>10} {b['n']:>6} {b['games']:>6} {b['fill_rate'] * 100:>5.0f}%   "
              f"{_ci(b['taker_gross_pp']):<24} {_ci(b['taker_net_pp']):<24} "
              f"{_ci(b['maker_net_pp'])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-db", default="data/predictions.db")
    ap.add_argument("--cache-dir", default=str(_REPO / "data" / "backtest" / "polyus_history"))
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument("--sectors", default="")
    ap.add_argument("--group", default="", help="only this sector/market_type, e.g. nfl/spread")
    ap.add_argument("--min-games", type=int, default=30)
    ap.add_argument("--fetch", action="store_true", help="fetch missing history first")
    ap.add_argument("--rps", type=float, default=8.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    sectors = {s.strip() for s in args.sectors.split(",") if s.strip()}
    rows = rp.load_rows(args.pred_db, args.since, sectors)
    cached = asyncio.run(rp.fetch_all(rows, rp.Cache(Path(args.cache_dir)), args.rps,
                                      offline=not args.fetch))
    cached.pop("__stats__", None)

    paths, skipped = [], Counter()
    for r in rows:
        parsed = mr.parse_market_id(r["market_id"])
        data = cached.get(parsed[0]) if parsed else None
        if not data or data.get("meta") is None:
            skipped["no_market_meta"] += 1
            continue
        meta = mr.parse_market_meta(parsed[0], data["meta"])
        buckets = sorted(b for day in data["days"].values() for b in day)
        p, why = et.build_path(r, meta, buckets)
        if p is None:
            skipped[why] += 1
        else:
            paths.append(p)
    print(f"candidates {len(rows)} | usable paths {len(paths)} | skipped {dict(skipped)}")
    if not paths:
        return 1

    groups: dict[str, list[et.Path]] = defaultdict(list)
    for p in paths:
        groups[f"{p.sector}/{p.market_type}"].append(p)
    out = {}
    if args.group:
        if args.group not in groups:
            print(f"no such group {args.group!r}; have {sorted(groups)}")
            return 1
        out[args.group] = report(args.group, groups[args.group])
    else:
        out["ALL"] = report("ALL", paths)
        for name, ps in sorted(groups.items(), key=lambda kv: -len({p.game for p in kv[1]})):
            if len({p.game for p in ps}) >= args.min_games:
                out[name] = report(name, ps)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
