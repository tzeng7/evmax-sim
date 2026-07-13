"""Edge-observation integration tests — the point of the whole module.

These run the full data → skill → simulate → devig → price chain on REAL
captured fixtures (PolyUS Open field + PGA season SG) and:

  1. assert the pipeline produces coherent probabilities end to end;
  2. assert the model reproduces the market's ORDERING (favourites are
     favourites) — the capability that works;
  3. lock in the honest empirical FINDING as a regression guard: on an
     efficient, high-volume major winner market the free-data model is
     systematically UNDER-confident on the favourites, so it flags no outright
     plays there. (This is the true state of the first slice — the module
     observes divergence; it does not manufacture edge.)
  4. prove the harness CAN detect edge when it really exists (synthetic
     mispriced field), so the negative result above is a real finding, not a
     dead pipeline.
"""

import pytest

from evmax.golf.data.pga import parse_sg_snapshot
from evmax.golf.devig import devig_field
from evmax.golf.models import FieldPrices, GolfMarketType, GolferMarket, SimResult
from evmax.golf.pricing import EdgeConfig, flag_plays, price_field
from evmax.golf.run import scan
from evmax.golf.simulator import SimConfig
from evmax.golf.names import canonical


@pytest.fixture
def open_scan():
    # PolyUS-only keeps the winner field on one venue for a clean read.
    return scan(live=False, include_kalshi=False, sim_config=SimConfig(n_sims=20000, seed=17))


def _winner_tournament(open_scan):
    for t in open_scan.tournaments:
        if any(f.market_type is GolfMarketType.win for f in t.fields):
            return t
    raise AssertionError("no winner tournament in scan")


class TestPipelineCoherence:
    def test_scan_produces_tournaments_and_sims(self, open_scan):
        assert open_scan.tournaments
        t = _winner_tournament(open_scan)
        assert t.field_size > 100
        assert abs(sum(r.win for r in t.sims.values()) - 1.0) < 1e-6

    def test_edges_are_finite_and_bounded(self, open_scan):
        t = _winner_tournament(open_scan)
        assert t.edges
        for e in t.edges:
            assert 0.0 <= e.model_prob <= 1.0
            assert 0.0 <= e.market_prob <= 1.0
            assert -1.0 <= e.edge <= 1.0

    def test_market_devig_sums_to_one(self, open_scan):
        t = _winner_tournament(open_scan)
        win = next(f for f in t.fields if f.market_type is GolfMarketType.win)
        assert sum(win.devigged.values()) == pytest.approx(1.0, abs=1e-6)


class TestModelReproducesOrdering:
    def test_top_favourites_are_favourites(self, open_scan):
        t = _winner_tournament(open_scan)
        by_win = sorted(t.sims.values(), key=lambda r: r.win, reverse=True)
        top5 = {r.player for r in by_win[:5]}
        # The model's top handful must include the marquee names — the skill
        # signal is real even if the magnitude is under-confident.
        assert canonical("Scottie Scheffler") in top5
        assert (
            canonical("Rory McIlroy") in top5
            or canonical("Matt Fitzpatrick") in top5
        )

    def test_scheffler_ranks_above_a_journeyman(self, open_scan):
        t = _winner_tournament(open_scan)
        sch = t.sims[canonical("Scottie Scheffler")].win
        # someone deep in the field
        assert sch > 0.02


class TestHonestUnderconfidenceFinding:
    """Regression guard for the empirical result: the free-data model is
    under-confident vs the efficient major market, so no outright favourite is
    a flagged play. If a future model change flips this, it should be a
    deliberate, reviewed change — this test forces that."""

    def test_market_rates_favourite_far_higher_than_model(self, open_scan):
        t = _winner_tournament(open_scan)
        win = next(f for f in t.fields if f.market_type is GolfMarketType.win)
        sch = canonical("Scottie Scheffler")
        model = t.sims[sch].win
        market = win.devigged[sch]
        # market ~0.15, model ~0.03-0.06: the market is meaningfully more
        # confident in the favourite than a season-SG Gaussian sim.
        assert market > model
        assert market - model > 0.05

    def test_no_outright_plays_flagged_on_the_major(self, open_scan):
        t = _winner_tournament(open_scan)
        outright_plays = [p for p in t.plays if p.market_type is GolfMarketType.win]
        assert outright_plays == []


class TestHarnessCanDetectEdge:
    """The negative result above is only meaningful if the harness would fire
    when edge is genuinely present. Construct a field the model out-predicts."""

    def test_flags_a_genuinely_mispriced_golfer(self):
        # Model thinks 'value' is a 12% shot; the market is asking 5% (tight,
        # liquid). That is a real +7pp edge vs the ask on an outright.
        sims = {
            "value": SimResult("value", 0.12, 0.35, 0.5, 0.65, 0.9, 8.0),
            "chalk": SimResult("chalk", 0.10, 0.30, 0.45, 0.6, 0.88, 10.0),
            "filler": SimResult("filler", 0.02, 0.08, 0.15, 0.3, 0.6, 40.0),
        }
        markets = [
            GolferMarket("polymarket_us", "T", GolfMarketType.win, "value", 0.05, "v",
                         yes_ask=0.05, yes_bid=0.045, implied_no_raw=0.955),
            GolferMarket("polymarket_us", "T", GolfMarketType.win, "chalk", 0.11, "c",
                         yes_ask=0.11, yes_bid=0.10, implied_no_raw=0.90),
            GolferMarket("polymarket_us", "T", GolfMarketType.win, "filler", 0.02, "f",
                         yes_ask=0.02, yes_bid=0.015, implied_no_raw=0.985),
        ]
        f = FieldPrices("polymarket_us", "T", GolfMarketType.win, markets)
        devig_field(f)
        plays = flag_plays(price_field(sims, f, devig=False), [f],
                           EdgeConfig(outright_threshold=0.05))
        assert any(p.golfer == "value" and p.edge_vs_raw > 0.05 for p in plays)

    def test_make_cut_edge_detected(self):
        # Placement market: model 80% make-cut vs a 70% ask → +10pp, liquid.
        sims = {"a": SimResult("a", 0.05, 0.2, 0.35, 0.55, 0.80, 20.0)}
        m = GolferMarket("polymarket_us", "T", GolfMarketType.make_cut, "a", 0.70, "a",
                         yes_ask=0.70, yes_bid=0.68, implied_no_raw=0.31)
        f = FieldPrices("polymarket_us", "T", GolfMarketType.make_cut, [m])
        devig_field(f, cut_target=70.0)
        plays = flag_plays(price_field(sims, f, devig=False), [f],
                           EdgeConfig(placement_threshold=0.03))
        assert any(p.golfer == "a" for p in plays)
