"""Capture + weekend-scoring harness tests: snapshot round-trip, model-vs-market
Brier, the disagreement-who's-right count, and the realised betting P&L math."""

import pytest

from evmax.golf.capture import (
    ContractSnapshot,
    Snapshot,
    build_snapshot,
    load_snapshot,
    save_snapshot,
    score_snapshot,
)
from evmax.golf.models import FieldPrices, GolfMarketType, GolferMarket, SimResult


def _contract(golfer, model, market, ask, mt="make_cut", cal=None):
    return ContractSnapshot(mt, golfer, golfer, "kalshi", model, market, ask, cal)


class TestBuildSnapshot:
    def test_prices_every_priced_contract(self):
        markets = [
            GolferMarket("kalshi", "T", GolfMarketType.make_cut, "a", 0.6, "a",
                         yes_ask=0.6, yes_bid=0.58, implied_no_raw=0.42),
            GolferMarket("kalshi", "T", GolfMarketType.make_cut, "b", 0.5, "b",
                         yes_ask=0.5, yes_bid=0.48, implied_no_raw=0.52),
        ]
        f = FieldPrices("kalshi", "T", GolfMarketType.make_cut, markets)
        sims = {"a": SimResult("a", 0.05, 0.2, 0.3, 0.5, 0.75, 20.0),
                "b": SimResult("b", 0.02, 0.1, 0.2, 0.4, 0.55, 40.0)}
        snap = build_snapshot("2026-07-16T00:00:00Z", "Open", "THOC26", [f], sims)
        assert len(snap.contracts) == 2
        a = next(c for c in snap.contracts if c.golfer == "a")
        assert a.model_prob == 0.75  # make_cut prob from the sim
        assert a.market_ask == 0.6

    def test_round_leader_contract_skipped(self):
        m = GolferMarket("polymarket_us", "T", GolfMarketType.r1_leader, "a", 0.1, "a",
                         yes_ask=0.1, yes_bid=0.08, implied_no_raw=0.9)
        f = FieldPrices("polymarket_us", "T", GolfMarketType.r1_leader, [m])
        sims = {"a": SimResult("a", 0.1, 0.2, 0.3, 0.4, 0.6, 30.0)}
        snap = build_snapshot("t", "T", "E", [f], sims)
        assert snap.contracts == []  # base sim emits no round-leader prob


class TestRoundTrip:
    def test_save_load(self, tmp_path):
        snap = Snapshot("t", "Open", "THOC26", [
            _contract("a", 0.75, 0.6, 0.62, cal=0.7),
            _contract("b", 0.55, 0.5, 0.52),
        ])
        p = tmp_path / "snap.json"
        save_snapshot(snap, p)
        loaded = load_snapshot(p)
        assert loaded.tournament == "Open"
        assert len(loaded.contracts) == 2
        assert loaded.contracts[0].model_prob_cal == 0.7


class TestScoring:
    def test_model_more_accurate_than_market(self):
        # model sits right on the outcomes; market is off.
        snap = Snapshot("t", "T", "E", [
            _contract("a", 0.9, 0.6, 0.62),  # outcome 1
            _contract("b", 0.1, 0.4, 0.42),  # outcome 0
        ])
        actuals = {"a": {"make_cut": 1.0}, "b": {"make_cut": 0.0}}
        rep = score_snapshot(snap, actuals)
        ms = rep.by_market[0]
        assert ms.model_brier < ms.market_brier

    def test_betting_pnl_math(self):
        # A: ask .62, wins (outcome 1) → profit +.38 ; B: ask .52, loses → −.52
        snap = Snapshot("t", "T", "E", [
            _contract("a", 0.80, 0.60, 0.62),
            _contract("b", 0.70, 0.50, 0.52),
        ])
        actuals = {"a": {"make_cut": 1.0}, "b": {"make_cut": 0.0}}
        rep = score_snapshot(snap, actuals, edge_threshold=0.03)
        assert rep.bets.n_bets == 2  # both clear model−ask ≥ .03
        assert rep.bets.total_cost == pytest.approx(1.14)
        assert rep.bets.total_return == pytest.approx(1.0)
        assert rep.bets.roi == pytest.approx((1.0 - 1.14) / 1.14)
        assert rep.bets.hit_rate == pytest.approx(0.5)

    def test_only_flagged_bets_counted(self):
        # A has edge (.8-.62=.18); B does not (.53-.52=.01 < .03)
        snap = Snapshot("t", "T", "E", [
            _contract("a", 0.80, 0.60, 0.62),
            _contract("b", 0.53, 0.50, 0.52),
        ])
        actuals = {"a": {"make_cut": 1.0}, "b": {"make_cut": 1.0}}
        rep = score_snapshot(snap, actuals, edge_threshold=0.03)
        assert rep.bets.n_bets == 1

    def test_disagreement_who_is_right(self):
        # model far from market on both; model right on a, wrong on b.
        snap = Snapshot("t", "T", "E", [
            _contract("a", 0.9, 0.5, 0.52),   # outcome 1 → model closer
            _contract("b", 0.9, 0.5, 0.52),   # outcome 0 → market closer
        ])
        actuals = {"a": {"make_cut": 1.0}, "b": {"make_cut": 0.0}}
        rep = score_snapshot(snap, actuals, disagreement_gap=0.03)
        ms = rep.by_market[0]
        assert ms.n_disagreements == 2
        assert ms.model_closer == 1
        assert ms.market_closer == 1

    def test_missing_actual_skipped(self):
        snap = Snapshot("t", "T", "E", [_contract("ghost", 0.5, 0.5, 0.5)])
        rep = score_snapshot(snap, {})  # no actuals at all
        assert rep.by_market == []
        assert rep.bets.n_bets == 0

    def test_calibrated_bets_scored_separately(self):
        snap = Snapshot("t", "T", "E", [
            _contract("a", 0.80, 0.60, 0.62, cal=0.65),  # cal-ask=.03 → flagged
        ])
        actuals = {"a": {"make_cut": 1.0}}
        rep = score_snapshot(snap, actuals, edge_threshold=0.03)
        assert rep.bets_cal is not None
        assert rep.bets_cal.n_bets == 1
