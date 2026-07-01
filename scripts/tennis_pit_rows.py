"""Shared point-in-time (PIT) row builder for tennis offline experiments.

Produces leak-free per-match rows for walk-forward experiments:

    [outcome_a, date_iso, p_market_a, {model: [prob_a, confidence]}]

where (a, b) are the two players sorted alphabetically, outcome_a is 1.0 when
a won, and p_market_a is the devigged Pinnacle close for a.

Point-in-time seeding, per test month:
  - **surface Elo**: the real Tennis Abstract Elo leaderboard from the nearest
    Wayback snapshot strictly BEFORE the month (snapshots cached by
    scripts/backtest_tennis_abstract_elo.py under {tempdir}/evmax_ta_elo_wayback
    and/or /tmp/evmax_ta_elo_wayback) — matching how live seeds from TA.
  - **serve/return, advanced, form, h2h**: seeded from matchmx rows dated
    before the month (same aggregate functions live seeding uses).
Ground truth + market come from tennis-data.co.uk (data/backtest/tennis/*.xlsx).

Lifted from tennis_weight_optimize.py (2026-06-27) so the weight optimizer and
the calibration harness (fit_tennis_calibration.py) share one builder.

NOTE (2026-07-01): the game-count lookup fix in tennis_model_agent.py means
caches rebuilt after the fix carry surface confidence 0.61 (the synthetic
seeded counts now resolve for surname-form queries) instead of the flat 0.30
artifact baked into caches built before it. Production gates models at
confidence > 0.45, so a gate-faithful consensus computed over a PRE-fix cache
would wrongly exclude tennis_surface from every blend — rebuild before use.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import analyze_tennis_full_blend as afb  # noqa: E402  (runs the _resolve_surface monkeypatch)
import backtest_tennis_matchmx as bt  # noqa: E402
from evmax.clients.tennisabstract import fetch_matchmx, parse_elo_html  # noqa: E402
from evmax.ev.devig import devig_two_way  # noqa: E402

# Wayback snapshot caches: backtest_tennis_abstract_elo.py writes to
# tempfile.gettempdir() (=/var/folders/... on macOS); older runs also live in
# a literal /tmp dir. Check both.
WAYBACK_DIRS = (
    Path(tempfile.gettempdir()) / "evmax_ta_elo_wayback",
    Path("/tmp/evmax_ta_elo_wayback"),
)
MODELS = list(afb.WEIGHTS)  # surface, serve_return, form, advanced, h2h
DEFAULT_MONTH_RANGE = ((2025, 1), (2026, 4))


def wb_snapshots() -> list[tuple[int, Path]]:
    """[(yyyymmdd_int, path)] for the cached ATP Elo snapshots, ascending."""
    seen: dict[str, Path] = {}
    for d in WAYBACK_DIRS:
        if not d.is_dir():
            continue
        for f in d.glob("atp_*.html"):
            seen.setdefault(f.stem, f)
    return sorted((int(stem.split("_", 1)[1][:8]), path) for stem, path in seen.items())


def ratings_from_snapshot(path: Path) -> dict[str, dict[str, float]]:
    rows = parse_elo_html(path.read_text())
    rbs: dict[str, dict[str, float]] = {s: {} for s in ("overall", "hard", "clay", "grass", "indoor")}
    for p in rows:
        rbs["overall"][p.player] = p.elo
        if p.hard is not None:
            rbs["hard"][p.player] = p.hard
            rbs["indoor"][p.player] = p.hard
        if p.clay is not None:
            rbs["clay"][p.player] = p.clay
        if p.grass is not None:
            rbs["grass"][p.player] = p.grass
    return rbs


def month_range(first: tuple[int, int], last: tuple[int, int]) -> list[tuple[int, int]]:
    months = []
    y, m = first
    while (y, m) <= last:
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


async def build_rows(
    cache_path: Path,
    months: list[tuple[int, int]] | None = None,
) -> list[list]:
    """Build PIT rows month-by-month and write them to cache_path as JSON."""
    afb.TMP.mkdir(exist_ok=True)
    snaps = wb_snapshots()
    if not snaps:
        print("No Wayback snapshots — run backtest_tennis_abstract_elo.py first.", file=sys.stderr)
        sys.exit(2)
    months = months or month_range(*DEFAULT_MONTH_RANGE)
    mx = fetch_matchmx("atp")
    mx_sorted = sorted(mx, key=lambda r: r["tourney_date"])
    years = tuple(sorted({y for y, _ in months} | {months[0][0] - 1}))
    results = bt.load_atp_results(years)

    agents = afb._new_agents()
    surface = agents["tennis_surface"]
    rows: list[list] = []
    for (yy, mm) in months:
        mstart = yy * 10000 + mm * 100 + 1
        # surface Elo from the most recent Wayback snapshot strictly before the month
        prior_snaps = [p for d, p in snaps if d < mstart]
        if not prior_snaps:
            continue
        surface._state = {}
        surface.seed_precomputed_surface_elo(ratings_from_snapshot(prior_snaps[-1]))
        # other models from matchmx pre-month
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
            preds = await asyncio.gather(*[afb._prob_a(agents[k], market, sharp) for k in MODELS])
            mp = {k: [preds[i][0], preds[i][1]] for i, k in enumerate(MODELS) if preds[i][0] is not None}
            rows.append([1.0 if a_won else 0.0, r["date"].isoformat(),
                         p_mkt_w if a_won else 1.0 - p_mkt_w, mp])

    cache_path.write_text(json.dumps(rows))
    fire = {k: sum(1 for _, _, _, mp in rows if k in mp) for k in MODELS}
    print(f"cached {len(rows)} rows -> {cache_path}. fire rates:",
          {k: f"{v}/{len(rows)}" for k, v in fire.items()})
    return rows


def load_rows(cache_path: Path) -> list[list]:
    """Load cached PIT rows, sorted chronologically."""
    return sorted(json.loads(cache_path.read_text()), key=lambda r: r[1])
