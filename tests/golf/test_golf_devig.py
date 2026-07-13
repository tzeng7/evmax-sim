"""Field-devig tests: the two probability-mass shapes (exactly-one-winner vs
non-exclusive), plus the spread-aware robustness that stops illiquid longshots
from poisoning a winner field."""

import pytest

from evmax.golf.devig import devig_contract, devig_field
from evmax.golf.models import FieldPrices, GolfMarketType, GolferMarket


def _mkt(name, ask, bid=None, no=None, mt=GolfMarketType.win):
    return GolferMarket(
        venue="polymarket_us", tournament="T", market_type=mt, golfer=name,
        implied_raw=ask, ticker=name, yes_ask=ask, yes_bid=bid, implied_no_raw=no,
    )


class TestContractDevig:
    def test_removes_two_way_vig(self):
        # 0.15 / 0.87 → overround 1.02. The Power Method shrinks the longshot
        # side asymmetrically, so the devigged YES sits just below the raw ask
        # AND below the naive proportional value 0.15/1.02 = 0.147.
        p = devig_contract(0.15, 0.87)
        assert 0.13 < p < 0.15
        assert p < 0.15 / (0.15 + 0.87)  # favorite-longshot asymmetry

    def test_balanced_market_symmetric(self):
        # A symmetric 0.52/0.52 quote devigs to ~0.5.
        assert devig_contract(0.52, 0.52) == pytest.approx(0.5, abs=1e-6)

    def test_degenerate_quote_falls_back(self):
        # ask >= 1 is degenerate → proportional fallback, no exception.
        p = devig_contract(1.2, 0.5)
        assert 0.0 <= p <= 1.0


class TestWinnerFieldDevig:
    def test_sums_to_one(self):
        f = FieldPrices("polymarket_us", "T", GolfMarketType.win, [
            _mkt("a", 0.30, 0.28), _mkt("b", 0.22, 0.20), _mkt("c", 0.12, 0.10),
            _mkt("d", 0.06, 0.04),
        ])
        devig_field(f)
        assert sum(f.devigged.values()) == pytest.approx(1.0, abs=1e-9)
        # ordering preserved
        assert f.devigged["a"] > f.devigged["b"] > f.devigged["c"] > f.devigged["d"]

    def test_illiquid_longshot_does_not_poison_favourite(self):
        # Favourite tight at ~0.30; a no-hoper posts a placeholder 0.45 ask with
        # a 0.01 bid. Spread-aware devig must use the longshot's BID, so it can't
        # steal mass from the favourite.
        favourite = _mkt("fav", 0.31, 0.29)
        longshot = _mkt("dog", 0.45, 0.01)
        f = FieldPrices("polymarket_us", "T", GolfMarketType.win, [favourite, longshot])
        devig_field(f, max_spread=0.05)
        assert f.devigged["fav"] > 0.9  # favourite keeps almost all the mass
        assert f.devigged["dog"] < 0.05
        assert sum(f.devigged.values()) == pytest.approx(1.0, abs=1e-9)

    def test_round_leader_treated_as_winner_shape(self):
        f = FieldPrices("polymarket_us", "T", GolfMarketType.r1_leader, [
            _mkt("a", 0.10, 0.08, mt=GolfMarketType.r1_leader),
            _mkt("b", 0.05, 0.03, mt=GolfMarketType.r1_leader),
        ])
        devig_field(f)
        assert sum(f.devigged.values()) == pytest.approx(1.0, abs=1e-9)


class TestNonExclusiveFieldDevig:
    def test_top10_scales_to_ten_when_yes_only(self):
        # YES-only contracts with no NO side → scaled to the market mass (N=10).
        markets = [_mkt(f"p{i}", 0.20, mt=GolfMarketType.top_10) for i in range(30)]
        for m in markets:
            m.implied_no_raw = None  # force YES-only path
        f = FieldPrices("polymarket_us", "T", GolfMarketType.top_10, markets)
        devig_field(f)
        assert sum(f.devigged.values()) == pytest.approx(10.0, abs=1e-6)

    def test_make_cut_without_target_passes_raw(self):
        markets = [_mkt(f"p{i}", 0.5, mt=GolfMarketType.make_cut) for i in range(4)]
        for m in markets:
            m.implied_no_raw = None
        f = FieldPrices("polymarket_us", "T", GolfMarketType.make_cut, markets)
        devig_field(f, cut_target=None)
        # No known mass → raw YES passed through unchanged.
        assert all(v == pytest.approx(0.5) for v in f.devigged.values())

    def test_make_cut_with_target_scales(self):
        markets = [_mkt(f"p{i}", 0.5, mt=GolfMarketType.make_cut) for i in range(100)]
        for m in markets:
            m.implied_no_raw = None
        f = FieldPrices("polymarket_us", "T", GolfMarketType.make_cut, markets)
        devig_field(f, cut_target=70.0)
        assert sum(f.devigged.values()) == pytest.approx(70.0, abs=1e-6)

    def test_per_contract_devig_used_when_both_sides_present(self):
        markets = [_mkt(f"p{i}", 0.55, no=0.50, mt=GolfMarketType.top_20) for i in range(40)]
        f = FieldPrices("polymarket_us", "T", GolfMarketType.top_20, markets)
        devig_field(f)
        # Each devigged to its own 2-way fair value ~0.55/1.05, not scaled to 20.
        assert all(0.51 < v < 0.54 for v in f.devigged.values())


class TestDevigEdgeCases:
    def test_empty_field(self):
        f = FieldPrices("kalshi", "T", GolfMarketType.win, [])
        devig_field(f)
        assert f.devigged == {}
        assert f.overround is None

    def test_overround_recorded(self):
        f = FieldPrices("polymarket_us", "T", GolfMarketType.win,
                        [_mkt("a", 0.6, 0.58), _mkt("b", 0.6, 0.58)])
        devig_field(f)
        assert f.overround == pytest.approx(1.2)
