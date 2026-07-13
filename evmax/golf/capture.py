"""Capture live golf prices vs the model, then score them when the event resolves.

This is the one clean forward test available for golf: there is no historical
golf-odds series for a CLV backtest and no sharp anchor for make-cut / top-N, so
the only honest way to settle "does the model beat the market" is to LOCK IN
(model probability, market price) pairs before a tournament and score them
against the actual outcomes afterwards.

  build_snapshot(...)  → freeze model + market for every priced contract
  save/load             → persist the frozen snapshot (JSON)
  score_snapshot(...)   → after the event: model vs market Brier, who-was-right
                          on the biggest disagreements, and the realised P&L of
                          betting the model's flagged +EV contracts at the ask.

The betting test is the decisive one: buy a YES contract for its ask ``a``; it
settles ``outcome`` ∈ {0,1}; profit = ``outcome − a`` per unit face. If the
contracts where the model claimed edge over the ask actually turn a profit, the
edge is real; if not, it was noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from evmax.golf.calibration import IsotonicCalibrator, brier
from evmax.golf.devig import devig_field
from evmax.golf.models import FieldPrices, GolfMarketType, SimResult


@dataclass
class ContractSnapshot:
    market_type: str
    golfer: str
    ticker: str
    venue: str
    model_prob: float            # raw simulator probability
    market_prob: float           # devigged (fair) market probability
    market_ask: Optional[float]  # what you actually pay (vigged)
    model_prob_cal: Optional[float] = None  # calibrated model prob, if available


@dataclass
class Snapshot:
    captured_at: str
    tournament: str
    event_id: str
    contracts: list[ContractSnapshot] = field(default_factory=list)


@dataclass
class MarketScore:
    market_type: str
    n: int
    model_brier: float
    market_brier: float
    model_cal_brier: Optional[float]
    # disagreement-who's-right (on contracts where |model − market| ≥ gap)
    n_disagreements: int
    model_closer: int
    market_closer: int


@dataclass
class BetScore:
    n_bets: int
    total_cost: float
    total_return: float
    roi: float
    hit_rate: float
    avg_ask: float


@dataclass
class ScoreReport:
    tournament: str
    by_market: list[MarketScore]
    bets: BetScore
    bets_cal: Optional[BetScore]

    def summary(self) -> str:
        lines = [f"scored: {self.tournament}", ""]
        lines.append(f"{'market':10s} {'n':>4s} {'modelBrier':>11s} "
                     f"{'mktBrier':>9s} {'calBrier':>9s}  who's-closer(disagree)")
        for m in self.by_market:
            cal = f"{m.model_cal_brier:.4f}" if m.model_cal_brier is not None else "   -   "
            edge = "model" if m.model_brier < m.market_brier else "market"
            lines.append(
                f"{m.market_type:10s} {m.n:4d} {m.model_brier:11.4f} "
                f"{m.market_brier:9.4f} {cal:>9s}  "
                f"model {m.model_closer}/{m.n_disagreements} vs mkt {m.market_closer} "
                f"[Brier→{edge}]"
            )
        b = self.bets
        lines += ["", "flagged +EV bets (model_prob − ask ≥ threshold), settled at the ask:",
                  f"  raw model: n={b.n_bets}  ROI={b.roi:+.1%}  hit={b.hit_rate:.1%}  "
                  f"avg_ask={b.avg_ask:.1%}  (cost {b.total_cost:.2f} → return {b.total_return:.2f})"]
        if self.bets_cal is not None:
            c = self.bets_cal
            lines.append(f"  calibrated: n={c.n_bets}  ROI={c.roi:+.1%}  hit={c.hit_rate:.1%}  "
                         f"avg_ask={c.avg_ask:.1%}")
        lines += ["",
                  "read: modelBrier < mktBrier → model more accurate on that market;",
                  "      positive ROI on flagged bets → the model's edges actually paid."]
        return "\n".join(lines)


def build_snapshot(
    captured_at: str,
    tournament: str,
    event_id: str,
    fields: list[FieldPrices],
    sims: dict[str, SimResult],
    calibrators: Optional[dict[str, IsotonicCalibrator]] = None,
) -> Snapshot:
    """Freeze (model, market) for every priced contract the model can price."""
    calibrators = calibrators or {}
    contracts: list[ContractSnapshot] = []
    for f in fields:
        if not f.devigged:
            devig_field(f)
        for m in f.priced_markets():
            sim = sims.get(m.golfer)
            if sim is None:
                continue
            mp = sim.prob_for(f.market_type)
            if mp is None:  # market family the base sim doesn't emit (round leaders)
                continue
            cal = calibrators.get(f.market_type.value)
            contracts.append(ContractSnapshot(
                market_type=f.market_type.value,
                golfer=m.golfer,
                ticker=m.ticker,
                venue=f.venue,
                model_prob=mp,
                market_prob=f.devigged.get(m.golfer, 0.0),
                market_ask=m.implied_raw,
                model_prob_cal=(cal.transform_one(mp) if cal is not None else None),
            ))
    return Snapshot(captured_at, tournament, event_id, contracts)


def save_snapshot(snapshot: Snapshot, path) -> None:
    from pathlib import Path

    Path(path).write_text(json.dumps(asdict(snapshot), indent=1))


def load_snapshot(path) -> Snapshot:
    from pathlib import Path

    d = json.loads(Path(path).read_text())
    contracts = [ContractSnapshot(**c) for c in d.pop("contracts", [])]
    return Snapshot(contracts=contracts, **d)


def _bet_score(rows, use_cal, edge_threshold) -> BetScore:
    """rows: list of (model_prob, model_cal, ask, outcome). Flag where the chosen
    model prob beats the ask by >= threshold; settle each at the ask."""
    cost = ret = hits = 0.0
    asks = []
    n = 0
    for model_prob, model_cal, ask, outcome in rows:
        p = model_cal if use_cal else model_prob
        if p is None or ask is None:
            continue
        if p - ask < edge_threshold:
            continue
        n += 1
        cost += ask
        ret += outcome
        hits += 1.0 if outcome >= 0.5 else 0.0
        asks.append(ask)
    roi = (ret - cost) / cost if cost > 0 else 0.0
    return BetScore(
        n_bets=n, total_cost=cost, total_return=ret, roi=roi,
        hit_rate=(hits / n if n else 0.0),
        avg_ask=(sum(asks) / len(asks) if asks else 0.0),
    )


def score_snapshot(
    snapshot: Snapshot,
    actuals: dict[str, dict],
    edge_threshold: float = 0.03,
    disagreement_gap: float = 0.03,
    min_market_prob: float = 0.0,
) -> ScoreReport:
    """Score a frozen snapshot against realised outcomes.

    ``actuals`` maps canonical golfer → {market_type_value: outcome ∈ {0,1}} (as
    produced by ``evmax.golf.backtest.compute_actuals``, extended to top-5/20).
    """
    # group contracts by market type
    by_mt: dict[str, list[ContractSnapshot]] = {}
    for c in snapshot.contracts:
        by_mt.setdefault(c.market_type, []).append(c)

    market_scores: list[MarketScore] = []
    all_bet_rows: list[tuple] = []
    has_cal = any(c.model_prob_cal is not None for c in snapshot.contracts)

    for mt, cs in sorted(by_mt.items()):
        m_probs, k_probs, cal_probs, outs = [], [], [], []
        n_dis = model_closer = market_closer = 0
        for c in cs:
            a = actuals.get(c.golfer, {}).get(mt)
            if a is None or c.market_prob < min_market_prob:
                continue
            m_probs.append(c.model_prob)
            k_probs.append(c.market_prob)
            outs.append(a)
            if c.model_prob_cal is not None:
                cal_probs.append(c.model_prob_cal)
            # disagreement test uses the fair market prob vs the model prob
            if abs(c.model_prob - c.market_prob) >= disagreement_gap:
                n_dis += 1
                if abs(c.model_prob - a) < abs(c.market_prob - a):
                    model_closer += 1
                elif abs(c.market_prob - a) < abs(c.model_prob - a):
                    market_closer += 1
            all_bet_rows.append((c.model_prob, c.model_prob_cal, c.market_ask, a))
        if not outs:
            continue
        market_scores.append(MarketScore(
            market_type=mt, n=len(outs),
            model_brier=brier(m_probs, outs),
            market_brier=brier(k_probs, outs),
            model_cal_brier=(brier(cal_probs, outs) if len(cal_probs) == len(outs) and cal_probs else None),
            n_disagreements=n_dis, model_closer=model_closer, market_closer=market_closer,
        ))

    bets = _bet_score(all_bet_rows, use_cal=False, edge_threshold=edge_threshold)
    bets_cal = _bet_score(all_bet_rows, use_cal=True, edge_threshold=edge_threshold) if has_cal else None
    return ScoreReport(snapshot.tournament, market_scores, bets, bets_cal)
