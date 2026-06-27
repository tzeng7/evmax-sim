"""Full 6-model* tennis blend: point-in-time walk-forward, decorrelation + weight sweep.

Extends analyze_tennis_blend_contribution.py from 2 models to the full statistical
stack — surface Elo, serve/return, advanced, form, h2h — seeded point-in-time from
matchmx, to answer:

  1. Does the FULL model blend beat the sharp (Pinnacle close)? (the documented claim)
  2. Leave-one-out: how much does each model contribute *inside* the full blend?
  3. Is the serve/return (0.25) vs advanced (0.15) weighting backwards, as the 2-model
     slice suggested? (swap test)
  4. sharp_weight sweep for the full blend.

* ranking_trend (weight 0.05) is omitted: it needs a weekly ranking-snapshot series,
  which matchmx (per-match ranks only) can't reconstruct point-in-time. It structurally
  rarely fires anyway.

Each model's probability comes from its real `predict_pair` (so gating + confidence are
exact); surface is injected by monkeypatching `_resolve_surface` to read the known
surface off the market. Sharp = Pinnacle close (devigged) — the hardest bar.

Usage:
    uv run python scripts/analyze_tennis_full_blend.py
"""
from __future__ import annotations

import asyncio
import datetime as dt
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evmax.agents.models import tennis_model_agent as tm  # noqa: E402
from evmax.agents.models.tennis_model_agent import TennisModelAgent  # noqa: E402
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent  # noqa: E402
from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent  # noqa: E402
from evmax.agents.models.tennis_form_agent import TennisFormAgent  # noqa: E402
from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent  # noqa: E402
from evmax.models.market import MarketSource, MarketType, PredictionMarket  # noqa: E402
from evmax.models.odds import SharpBook, SharpOdds  # noqa: E402
from evmax.clients.tennisabstract import fetch_matchmx  # noqa: E402
from evmax.ev.devig import devig_two_way  # noqa: E402
import backtest_tennis_matchmx as bt  # noqa: E402
from seed_tennis_models import (  # noqa: E402
    aggregate_advanced_stats, aggregate_match_history,
    aggregate_match_serve_stats, aggregate_h2h,
)

# Inject the known surface (we monkeypatch the shared resolver to read market.competition).
tm.TennisModelAgent._resolve_surface = staticmethod(
    lambda competition=None, title=None: ((competition or "hard").lower(), False)
)

# tennis SECTOR_WEIGHT_OVERRIDES (ranking_trend dropped).
WEIGHTS = {"tennis_surface": 0.30, "tennis_serve_return": 0.25,
           "tennis_form": 0.20, "tennis_advanced": 0.15, "tennis_h2h": 0.05}
SHARP_WEIGHT = 0.85
TMP = Path("/tmp/evmax_fullblend")


def _mk_market(a, b, surface, date_, best_of):
    return PredictionMarket(
        id=f"kalshi:{a}_vs_{b}", source=MarketSource.kalshi, sector="tennis",
        market_type=MarketType.moneyline,
        title="Wimbledon" if best_of == 5 else "ATP Match",  # slam keyword => bo5
        ticker="KXATPMATCH", yes_price=0.5, no_price=0.5, volume_usd=1e4,
        team_home=a, team_away=b, yes_team=a,
        competition=surface,  # read by the patched _resolve_surface
        event_date=datetime(date_.year, date_.month, date_.day, 12, tzinfo=timezone.utc),
    )


def _mk_sharp(a, b, date_):
    return SharpOdds(
        event_id=f"tennis::{date_}::{a}_vs_{b}", book=SharpBook.pinnacle, sector="tennis",
        outcome_a_label=a, outcome_b_label=b,
        outcome_a_decimal=2.0, outcome_b_decimal=2.0, true_prob_a=0.5, true_prob_b=0.5,
        margin=0.04, event_date=datetime(date_.year, date_.month, date_.day, 12, tzinfo=timezone.utc),
    )


def _new_agents():
    agents = {
        "tennis_surface": TennisModelAgent(),
        "tennis_serve_return": TennisServeReturnAgent(),
        "tennis_form": TennisFormAgent(),
        "tennis_advanced": TennisAdvancedStatsAgent(),
        "tennis_h2h": TennisH2HAgent(),
    }
    for i, ag in enumerate(agents.values()):
        ag._state_path = TMP / f"{i}.json"
        ag._state = {}
        ag.save_state = lambda: None  # suppress disk writes during the sweep
    return agents


def _reseed_batch(agents, prior_rows):
    a = agents
    a["tennis_serve_return"]._state = {}
    a["tennis_serve_return"].seed_match_serve_stats(aggregate_match_serve_stats(prior_rows))
    a["tennis_advanced"]._state = {}
    a["tennis_advanced"].seed_stats(aggregate_advanced_stats(prior_rows))
    a["tennis_form"]._state = {}
    a["tennis_form"].seed_match_history(aggregate_match_history(prior_rows))
    a["tennis_h2h"]._state = {}
    a["tennis_h2h"].seed_h2h(aggregate_h2h(prior_rows))


def _elo_replay(surface_agent, rows):
    for r in rows:
        surface_agent.update(r["winner_name"], r["loser_name"], 1.0, 0.0, "tennis",
                             event_date=r["tourney_date"], surface=(r["surface"] or "hard").lower())


async def _prob_a(agent, market, sharp):
    pred = await agent.predict_pair(market, sharp)
    if pred is None:
        return None, None
    return pred.true_prob_a, pred.confidence


def _ymd(s):
    return bt._safe_int(str(s)[:8])


async def main():
    TMP.mkdir(exist_ok=True)
    print("Fetching matchmx (atp)...")
    mx = fetch_matchmx("atp")
    results = bt.load_atp_results((2024, 2025, 2026))
    mx_sorted = sorted(mx, key=lambda r: r["tourney_date"])

    months = []
    y, m = 2025, 1
    while (y, m) <= (2026, 4):
        months.append((y, m)); m += 1
        if m > 12:
            m = 1; y += 1

    agents = _new_agents()
    surface = agents["tennis_surface"]
    # warm up Elo with everything before the first test month
    first = months[0][0] * 10000 + months[0][1] * 100 + 1
    _elo_replay(surface, [r for r in mx_sorted if _ymd(r["tourney_date"]) < first])

    rows = []  # (outcome_A, p_mkt_A, {model: (prob_A, conf)})
    for (yy, mm) in months:
        mstart = yy * 10000 + mm * 100 + 1
        prior = [r for r in mx_sorted if _ymd(r["tourney_date"]) < mstart]
        _reseed_batch(agents, prior)
        month_matches = [r for r in results if (r["date"].year, r["date"].month) == (yy, mm)]
        for r in month_matches:
            if not r["psw"] or not r["psl"]:
                continue
            try:
                p_mkt_w, _, _ = devig_two_way(float(r["psw"]), float(r["psl"]))
            except Exception:
                continue
            a, b = sorted((r["winner"], r["loser"]))
            a_won = a == r["winner"]
            market = _mk_market(a, b, r["surface"], r["date"], r["best_of"])
            sharp = _mk_sharp(a, b, r["date"])
            preds = await asyncio.gather(*[_prob_a(agents[k], market, sharp) for k in WEIGHTS])
            mp = {k: preds[i] for i, k in enumerate(WEIGHTS) if preds[i][0] is not None}
            rows.append((1.0 if a_won else 0.0,
                         p_mkt_w if a_won else 1.0 - p_mkt_w, mp))
        # advance Elo with this month's matches
        _elo_replay(surface, [r for r in mx_sorted
                              if mstart <= _ymd(r["tourney_date"]) < (mstart + 100)])

    print(f"\nscored matches: {len(rows)}")
    fire = {k: sum(1 for _, _, mp in rows if k in mp) for k in WEIGHTS}
    print("model fire rate:", {k: f"{v}/{len(rows)}" for k, v in fire.items()})

    out = [r[0] for r in rows]
    n = len(rows)

    def brier(probs):
        return sum((p - o) ** 2 for p, o in zip(probs, out)) / n

    def consensus(mp, weights):
        parts = [(weights[k], c, p) for k, (p, c) in mp.items() if k in weights]
        den = sum(w * c for w, c, _ in parts)
        return (sum(w * c * p for w, c, p in parts) / den) if den else None

    def blend_probs(weights, sw=SHARP_WEIGHT):
        res = []
        for o, p_mkt, mp in rows:
            cons = consensus(mp, weights)
            res.append(p_mkt if cons is None else sw * p_mkt + (1 - sw) * cons)
        return res

    b_sharp = brier([r[1] for r in rows])
    b_full = brier(blend_probs(WEIGHTS))
    print("\n=== full blend vs sharp (close) ===")
    print(f"  sharp alone : {b_sharp:.5f}")
    print(f"  full blend  : {b_full:.5f}  Δ {(b_full-b_sharp)*1000:+.2f}/1000")

    print("\n=== leave-one-out (Δ when each model is dropped; +Brier = model was helping) ===")
    for k in WEIGHTS:
        w2 = {x: v for x, v in WEIGHTS.items() if x != k}
        b = brier(blend_probs(w2))
        print(f"  drop {k:<20}: {b:.5f}  Δ vs full {(b-b_full)*1000:+.2f}/1000")

    print("\n=== serve<->advanced weight swap ===")
    swapped = dict(WEIGHTS); swapped["tennis_serve_return"] = 0.15; swapped["tennis_advanced"] = 0.25
    b_sw = brier(blend_probs(swapped))
    print(f"  current (serve .25 / adv .15): {b_full:.5f}")
    print(f"  swapped (serve .15 / adv .25): {b_sw:.5f}  Δ {(b_sw-b_full)*1000:+.2f}/1000")

    print("\n=== sharp_weight sweep (full blend) ===")
    best = (1.0, b_sharp)
    for i in range(11):
        sw = 0.50 + 0.05 * i
        b = brier(blend_probs(WEIGHTS, sw=sw))
        if b < best[1]:
            best = (sw, b)
        print(f"  sw={sw:.2f}  Brier={b:.5f}")
    print(f"  -> min at sw={best[0]:.2f} (Brier {best[1]:.5f}); pure sharp (sw=1.0) {b_sharp:.5f}")


if __name__ == "__main__":
    asyncio.run(main())
