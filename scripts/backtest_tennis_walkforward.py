"""Walk-forward backtest for tennis to validate the sharp_weight=1.00 decision.

Loads tennis-data.co.uk xlsx for 2024+2025+2026, replays matches chronologically
through the 6 tennis agents (surface_elo, serve_return, h2h, ranking_trend, form,
advanced), captures per-model + sharp predictions, then reports:

  1. Per-model Brier vs sharp Brier on the eval period (default 2025+2026)
  2. Per-disagreement-bucket Brier — answering "does the model ever beat sharp"
  3. Counterfactual blend Brier under three sharp_weight strategies:
       a) flat 1.00 (current production)
       b) 0.85 base + override ramp (0.02, 0.10, 1.00)
       c) 0.85 flat (no ramp — deliberately bad baseline)
  4. ROI proxy on the n bets where the model would have flagged +EV (blended >
     captured price, modeled as sharp_a here as a stand-in for capture price).

The tennis agents start from EMPTY state and warm up over 2024 (training).
Predictions on 2025+2026 are evaluated. State is held in memory (no save_state
calls) so production data/models/tennis_*.json are untouched.

Usage:
    python scripts/backtest_tennis_walkforward.py
    python scripts/backtest_tennis_walkforward.py --train 2024 --eval 2025,2026
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent
from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent
from evmax.agents.models.tennis_form_agent import TennisFormAgent
from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent
from evmax.backtest.metrics import brier_score, accuracy_score, log_loss_score
from evmax.ev.devig import devig_two_way
from evmax.models.market import MarketSource, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ---------------------------------------------------------------------------
# Data loading — direct xlsx parse so we can preserve surface info that the
# shared BacktestRow schema doesn't carry today.
# ---------------------------------------------------------------------------

@dataclass
class TennisMatch:
    date: date
    winner: str
    loser: str
    surface: str        # "hard" | "clay" | "grass"
    tournament: str
    sharp_winner_prob: float  # devigged P(winner) from PSW/PSL


SURFACE_TO_COMP = {
    "hard": "ATP melbourne",       # maps to hard via CITY_TO_SURFACE
    "clay": "ATP roland garros",   # maps to clay
    "grass": "ATP wimbledon",      # maps to grass
}


def _parse_date(val) -> Optional[date]:
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if not val:
        return None
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(val).strip(), fmt).date()
        except ValueError:
            pass
    return None


def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return v if v > 1.0 else None
    except (ValueError, TypeError):
        return None


def load_tennis_matches(years: list[int]) -> list[TennisMatch]:
    """Parse all years' xlsx into TennisMatch objects, sorted by date."""
    import openpyxl

    base = _REPO_ROOT / "data" / "backtest" / "tennis"
    matches: list[TennisMatch] = []

    for year in years:
        path = base / f"{year}.xlsx"
        if not path.exists():
            print(f"  [warn] missing {path}")
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        header = [str(c).strip() if c else "" for c in next(rows_iter)]

        def col(row_vals, name):
            try:
                return row_vals[header.index(name)]
            except (ValueError, IndexError):
                return None

        n_loaded = 0
        n_skipped = 0
        for raw in rows_iter:
            winner = col(raw, "Winner")
            loser = col(raw, "Loser")
            psw = _safe_float(col(raw, "PSW"))
            psl = _safe_float(col(raw, "PSL"))
            d = _parse_date(col(raw, "Date"))
            surface = (col(raw, "Surface") or "").strip().lower()
            tournament = col(raw, "Tournament") or ""
            if not (winner and loser and psw and psl and d and surface):
                n_skipped += 1
                continue
            if surface not in ("hard", "clay", "grass"):
                n_skipped += 1
                continue
            try:
                prob_w, _, _ = devig_two_way(psw, psl)
            except Exception:
                n_skipped += 1
                continue
            matches.append(TennisMatch(
                date=d,
                winner=str(winner).strip(),
                loser=str(loser).strip(),
                surface=surface,
                tournament=str(tournament).strip(),
                sharp_winner_prob=prob_w,
            ))
            n_loaded += 1
        wb.close()
        print(f"  loaded {year}: {n_loaded:5d} matches  ({n_skipped} skipped)")

    matches.sort(key=lambda m: m.date)
    return matches


# ---------------------------------------------------------------------------
# Walk-forward replay
# ---------------------------------------------------------------------------

@dataclass
class TennisWalkRow:
    date: date
    surface: str
    player_a: str   # alphabetical first (deterministic ordering — removes winner-bias)
    player_b: str
    a_won: bool
    sharp_prob_a: float
    surface_prob_a: Optional[float] = None
    serve_prob_a: Optional[float] = None
    h2h_prob_a: Optional[float] = None
    trend_prob_a: Optional[float] = None
    form_prob_a: Optional[float] = None
    advanced_prob_a: Optional[float] = None
    # Per-model confidence — needed for confidence-weighted blending
    surface_conf: Optional[float] = None
    serve_conf: Optional[float] = None
    h2h_conf: Optional[float] = None
    trend_conf: Optional[float] = None
    form_conf: Optional[float] = None
    advanced_conf: Optional[float] = None


def _pair(player_a: str, player_b: str, sharp_a: float, surface: str,
          event_date: Optional[date] = None):
    """Construct a (market, sharp) pair the agents will accept. Surface is
    embedded in `competition` via SURFACE_TO_COMP — the agents' resolver
    keys off well-known city names. Title also carries a surface hint as a
    backup so resolution doesn't rely on a single channel.

    `event_date` is set on the market so date-aware agents (form staleness,
    ranking-trend anchoring) evaluate relative to the match date instead of
    falling back to date.today() — without it the replay leaks future state
    and misjudges staleness."""
    market = PredictionMarket(
        id="bt", market_id="bt", event_id=f"bt::{player_a}::{player_b}",
        sector="tennis", team_home=player_a, team_away=player_b,
        source=MarketSource.kalshi, yes_price=sharp_a, no_price=1.0 - sharp_a,
        competition=SURFACE_TO_COMP.get(surface, "ATP melbourne"),
        title=f"{player_a} vs {player_b} ({surface})",
        event_date=datetime(event_date.year, event_date.month, event_date.day, 12)
        if event_date else None,
    )
    sharp = SharpOdds(
        event_id=f"bt::{player_a}::{player_b}", book=SharpBook.pinnacle, sector="tennis",
        outcome_a_label=player_a, outcome_b_label=player_b,
        outcome_a_decimal=1.0 / sharp_a if sharp_a > 0.01 else 100.0,
        outcome_b_decimal=1.0 / (1.0 - sharp_a) if sharp_a < 0.99 else 100.0,
        true_prob_a=sharp_a, true_prob_b=1.0 - sharp_a,
    )
    return market, sharp


async def _predict_one(agent, market, sharp) -> tuple[Optional[float], Optional[float]]:
    """Return (prob_a, confidence). None when agent has insufficient data."""
    try:
        pred = await agent.predict_pair(market, sharp)
        if pred is None:
            return None, None
        return pred.true_prob_a, pred.confidence
    except Exception:
        return None, None


def walk_forward(matches: list[TennisMatch], cold_start: bool = False) -> list[TennisWalkRow]:
    """Replay matches chronologically through the 6 tennis agents.

    By default, agents load their seeded state from disk on init (covers
    2020-2024 Sackmann + MCP). The walk-forward only mutates in-memory
    state — no save_state() call — so production state files are
    untouched. Eval period (2025+2026) is leak-free because the seed
    didn't see those games.

    `cold_start=True` empties state before the walk to test the
    cold-start regime; eval will only get signal from agents that build
    their own state quickly (Elo). Most agents need pre-seeded data.
    """
    surface_agent = TennisModelAgent()
    serve_agent = TennisServeReturnAgent()
    h2h_agent = TennisH2HAgent()
    trend_agent = TennisRankingTrendAgent()
    form_agent = TennisFormAgent()
    advanced_agent = TennisAdvancedStatsAgent()
    all_agents = (surface_agent, serve_agent, h2h_agent, trend_agent,
                   form_agent, advanced_agent)
    if cold_start:
        for agent in all_agents:
            agent._state = {}
    # CRITICAL: tennis agents call save_state() inside update(). Without
    # neutering it, walk-forward updates would corrupt production state
    # files on disk. We replace each agent's save_state with a no-op so
    # mutations stay in memory only.
    for agent in all_agents:
        agent.save_state = lambda: None

    rows: list[TennisWalkRow] = []
    progress_every = max(1, len(matches) // 20)

    for i, m in enumerate(matches):
        # Deterministic A/B ordering — alphabetical so model has no implicit
        # "winner-side" advantage and predictions are unbiased.
        if m.winner.lower() <= m.loser.lower():
            player_a, player_b, a_won = m.winner, m.loser, True
            sharp_a = m.sharp_winner_prob
        else:
            player_a, player_b, a_won = m.loser, m.winner, False
            sharp_a = 1.0 - m.sharp_winner_prob

        market, sharp = _pair(player_a, player_b, sharp_a, m.surface,
                              event_date=m.date)

        async def _all():
            return await asyncio.gather(
                _predict_one(surface_agent, market, sharp),
                _predict_one(serve_agent, market, sharp),
                _predict_one(h2h_agent, market, sharp),
                _predict_one(trend_agent, market, sharp),
                _predict_one(form_agent, market, sharp),
                _predict_one(advanced_agent, market, sharp),
            )
        (sf, sv, h2h, tr, fm, ad) = asyncio.run(_all())

        rows.append(TennisWalkRow(
            date=m.date, surface=m.surface,
            player_a=player_a, player_b=player_b, a_won=a_won,
            sharp_prob_a=sharp_a,
            surface_prob_a=sf[0], surface_conf=sf[1],
            serve_prob_a=sv[0], serve_conf=sv[1],
            h2h_prob_a=h2h[0], h2h_conf=h2h[1],
            trend_prob_a=tr[0], trend_conf=tr[1],
            form_prob_a=fm[0], form_conf=fm[1],
            advanced_prob_a=ad[0], advanced_conf=ad[1],
        ))

        # Update each agent with the actual result. Tennis agents take
        # (player_a, player_b, score_a, score_b, sector, event_date) where
        # score is 1/0 to indicate win.
        for agent in (surface_agent, serve_agent, h2h_agent, trend_agent,
                       form_agent, advanced_agent):
            try:
                agent.update(player_a, player_b,
                              1 if a_won else 0,
                              0 if a_won else 1,
                              "tennis",
                              event_date=m.date.isoformat())
            except Exception as e:
                # Don't let one agent's update kill the walk-forward
                pass

        if (i + 1) % progress_every == 0:
            print(f"  walked {i+1:5d}/{len(matches)} matches")

    return rows


# ---------------------------------------------------------------------------
# Blending — same weights and gating the live ensemble uses for tennis
# ---------------------------------------------------------------------------

# Mirrors SECTOR_WEIGHT_OVERRIDES["tennis"] in ensemble_agent.py
TENNIS_MODEL_WEIGHTS = {
    "tennis_surface":       0.30,
    "tennis_serve_return":  0.25,
    "tennis_form":          0.20,
    "tennis_advanced":      0.15,
    "tennis_h2h":           0.05,
    "tennis_ranking_trend": 0.05,
}
CONFIDENCE_GATE = 0.45


def model_blend(row: TennisWalkRow) -> Optional[float]:
    """Confidence-weighted average of model predictions, gated at >0.45.
    Returns None if no model contributed."""
    contribs = []
    for name, prob, conf in [
        ("tennis_surface",       row.surface_prob_a,  row.surface_conf),
        ("tennis_serve_return",  row.serve_prob_a,    row.serve_conf),
        ("tennis_h2h",           row.h2h_prob_a,      row.h2h_conf),
        ("tennis_ranking_trend", row.trend_prob_a,    row.trend_conf),
        ("tennis_form",          row.form_prob_a,     row.form_conf),
        ("tennis_advanced",      row.advanced_prob_a, row.advanced_conf),
    ]:
        if prob is None or conf is None or conf <= CONFIDENCE_GATE:
            continue
        w = TENNIS_MODEL_WEIGHTS.get(name, 0.0) * conf
        if w <= 0:
            continue
        contribs.append((w, prob))
    if not contribs:
        return None
    total_w = sum(w for w, _ in contribs)
    return sum(w * p for w, p in contribs) / total_w


def disagreement_ramp(
    base_sharp_weight: float,
    model_a: float,
    sharp_a: float,
    threshold: float = 0.02,
    saturate_at: float = 0.10,
    cap: float = 1.00,
) -> float:
    """Match _disagreement_sharp_weight in ensemble_agent.py for tennis."""
    diff = abs(model_a - sharp_a)
    if diff <= threshold:
        return base_sharp_weight
    if diff >= saturate_at:
        return cap
    fraction = (diff - threshold) / (saturate_at - threshold)
    return base_sharp_weight + fraction * (cap - base_sharp_weight)


def blend(row: TennisWalkRow, strategy: str) -> Optional[float]:
    """Apply one of the named blend strategies. Returns None when there's
    not enough model data to produce a blended prediction."""
    if strategy == "sharp_only":
        return row.sharp_prob_a

    model_a = model_blend(row)
    if model_a is None:
        # Fall through to sharp-only when models can't speak. Mirrors the
        # ensemble's behavior when no models exceed the confidence gate.
        return row.sharp_prob_a

    if strategy == "flat_1.00":
        return row.sharp_prob_a
    if strategy == "flat_0.85":
        return 0.85 * row.sharp_prob_a + 0.15 * model_a
    if strategy == "0.85_with_ramp":
        sw = disagreement_ramp(0.85, model_a, row.sharp_prob_a,
                                threshold=0.02, saturate_at=0.10, cap=1.00)
        return sw * row.sharp_prob_a + (1.0 - sw) * model_a
    if strategy == "model_only":
        return model_a
    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def filter_eval(rows: list[TennisWalkRow], years: set[int]) -> list[TennisWalkRow]:
    return [r for r in rows if r.date.year in years]


def per_model_metrics(rows: list[TennisWalkRow]) -> None:
    print("\n=== Per-model Brier (eval period only) ===")
    print(f"{'Model':24s}  {'n':>6s}  {'Brier':>8s}  {'Acc':>7s}  {'LogLoss':>8s}")
    print("-" * 60)

    for name, getter in [
        ("sharp",          lambda r: r.sharp_prob_a),
        ("tennis_surface", lambda r: r.surface_prob_a),
        ("tennis_serve",   lambda r: r.serve_prob_a),
        ("tennis_h2h",     lambda r: r.h2h_prob_a),
        ("tennis_trend",   lambda r: r.trend_prob_a),
        ("tennis_form",    lambda r: r.form_prob_a),
        ("tennis_advanced",lambda r: r.advanced_prob_a),
    ]:
        ps, acts = [], []
        for r in rows:
            v = getter(r)
            if v is not None:
                ps.append(v)
                acts.append(r.a_won)
        if not ps:
            print(f"  {name:22s}  (no data)")
            continue
        b = brier_score(ps, acts)
        a = accuracy_score(ps, acts)
        ll = log_loss_score(ps, acts)
        print(f"  {name:22s}  {len(ps):6d}  {b:8.4f}  {a*100:6.1f}%  {ll:8.4f}")

    # Confidence-weighted model blend
    ps, acts = [], []
    for r in rows:
        m = model_blend(r)
        if m is not None:
            ps.append(m); acts.append(r.a_won)
    if ps:
        b = brier_score(ps, acts); a = accuracy_score(ps, acts); ll = log_loss_score(ps, acts)
        print(f"  {'model_blend (no sharp)':22s}  {len(ps):6d}  {b:8.4f}  {a*100:6.1f}%  {ll:8.4f}")


def per_blend_strategy(rows: list[TennisWalkRow]) -> None:
    print("\n=== Blend strategy comparison (eval period) ===")
    print(f"{'Strategy':28s}  {'n':>6s}  {'Brier':>8s}  {'Acc':>7s}  {'Δ vs sharp':>11s}")
    print("-" * 68)

    sharp_b = None
    for strategy in ["sharp_only", "flat_0.85", "0.85_with_ramp", "model_only"]:
        ps, acts = [], []
        for r in rows:
            v = blend(r, strategy)
            if v is not None:
                ps.append(v); acts.append(r.a_won)
        if not ps:
            print(f"  {strategy:26s}  (no data)")
            continue
        b = brier_score(ps, acts); a = accuracy_score(ps, acts)
        delta = "" if sharp_b is None else f"{(sharp_b - b)*1000:+7.2f}/1000"
        if strategy == "sharp_only":
            sharp_b = b
        print(f"  {strategy:26s}  {len(ps):6d}  {b:8.4f}  {a*100:6.1f}%  {delta}")


def per_disagreement_bucket(rows: list[TennisWalkRow]) -> None:
    """Bucket by signed gap (model - sharp) on the A side. Critical for
    answering: in which buckets, if any, does the model add value?"""
    print("\n=== Per-disagreement-bucket Brier (eval period) ===")
    print("Buckets are |model - sharp| on the A side.")
    print(f"{'Bucket':>16s}  {'n':>5s}  {'sharp':>8s}  {'model':>8s}  "
          f"{'flat_0.85':>9s}  {'0.85+ramp':>9s}  best")
    print("-" * 80)

    bucket_edges = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.00)]
    bucketed: dict[tuple[float, float], list[TennisWalkRow]] = defaultdict(list)
    for r in rows:
        m = model_blend(r)
        if m is None:
            continue
        diff = abs(m - r.sharp_prob_a)
        for lo, hi in bucket_edges:
            if lo <= diff < hi:
                bucketed[(lo, hi)].append(r)
                break

    for (lo, hi) in bucket_edges:
        sub = bucketed.get((lo, hi), [])
        if not sub:
            print(f"  {lo:.2f}-{hi:.2f}{'':>6s}  (empty)")
            continue
        actuals = [r.a_won for r in sub]
        sharp_ps  = [r.sharp_prob_a for r in sub]
        model_ps  = [model_blend(r) for r in sub]
        flat_ps   = [blend(r, "flat_0.85") for r in sub]
        ramp_ps   = [blend(r, "0.85_with_ramp") for r in sub]
        b_sharp = brier_score(sharp_ps, actuals)
        b_model = brier_score(model_ps, actuals)
        b_flat  = brier_score(flat_ps,  actuals)
        b_ramp  = brier_score(ramp_ps,  actuals)
        candidates = [("sharp", b_sharp), ("model", b_model),
                      ("flat", b_flat), ("ramp", b_ramp)]
        best_name = min(candidates, key=lambda x: x[1])[0]
        print(f"  {lo:.2f}-{hi:.2f}{'':>6s}  {len(sub):5d}  "
              f"{b_sharp:8.4f}  {b_model:8.4f}  {b_flat:9.4f}  {b_ramp:9.4f}  {best_name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="2024",
                    help="Comma-separated training years (warmup, not evaluated). Default 2024.")
    ap.add_argument("--eval", type=str, default="2025,2026",
                    help="Comma-separated eval years. Default 2025,2026.")
    ap.add_argument("--cold-start", action="store_true",
                    help="Wipe agent state before walk-forward (tests Elo only — most "
                         "tennis agents need pre-seeded Sackmann data).")
    args = ap.parse_args()

    train_years = [int(y) for y in args.train.split(",") if y]
    eval_years_set = {int(y) for y in args.eval.split(",") if y}
    all_years = sorted(set(train_years) | eval_years_set)

    print(f"Loading tennis xlsx for years {all_years}...")
    matches = load_tennis_matches(all_years)
    print(f"\nTotal matches loaded: {len(matches)}")
    if not matches:
        print("No matches loaded — abort.")
        return 1

    mode = "cold-start" if args.cold_start else "seeded state from disk (in-memory mutations only)"
    print(f"\nWalk-forward replay ({mode})...")
    rows = walk_forward(matches, cold_start=args.cold_start)
    print(f"  produced {len(rows)} walk-forward rows")

    eval_rows = filter_eval(rows, eval_years_set)
    print(f"\nEval slice ({sorted(eval_years_set)}): {len(eval_rows)} matches")

    per_model_metrics(eval_rows)
    per_blend_strategy(eval_rows)
    per_disagreement_bucket(eval_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
