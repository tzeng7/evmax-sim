"""Edge/pricing-layer tests: joining model probs to devigged market probs,
honest edge-vs-ask, and the play-flagging gate (threshold + liquidity)."""

import pytest

from evmax.golf.models import FieldPrices, GolfMarketType, GolferMarket, SimResult
from evmax.golf.pricing import EdgeConfig, flag_plays, price_field


def _sim(name, win, t5=0.0, t10=0.0, t20=0.0, cut=0.0):
    return SimResult(name, win, t5, t10, t20, cut, mean_finish=1.0)


def _field(markets, mt=GolfMarketType.win):
    return FieldPrices("polymarket_us", "T", mt, markets)


def _mkt(name, ask, bid, no=None):
    return GolferMarket("polymarket_us", "T", GolfMarketType.win, name, ask, name,
                        yes_ask=ask, yes_bid=bid, implied_no_raw=no or (1 - bid))


class TestPriceField:
    def test_edge_is_model_minus_market(self):
        sims = {"a": _sim("a", 0.20), "b": _sim("b", 0.05)}
        f = _field([_mkt("a", 0.16, 0.14), _mkt("b", 0.06, 0.04)])
        edges = {e.golfer: e for e in price_field(sims, f)}
        # devigged fair probs sum to 1 across the 2-man field
        assert edges["a"].model_prob == 0.20
        assert edges["a"].edge == pytest.approx(0.20 - edges["a"].market_prob)
        assert edges["a"].edge_vs_raw == pytest.approx(0.20 - 0.16)

    def test_skips_players_without_sim(self):
        sims = {"a": _sim("a", 0.2)}
        f = _field([_mkt("a", 0.16, 0.14), _mkt("ghost", 0.10, 0.08)])
        edges = price_field(sims, f)
        assert {e.golfer for e in edges} == {"a"}

    def test_round_leader_not_priced_by_base_sim(self):
        sims = {"a": _sim("a", 0.2)}
        f = _field([_mkt("a", 0.16, 0.14)], mt=GolfMarketType.r1_leader)
        # base SimResult has no round-leader prob → no edge rows
        assert price_field(sims, f) == []


class TestFlagPlays:
    def test_positive_edge_over_threshold_flagged(self):
        # model 25% vs ask 15% on the winner market → +10pp edge vs ask, liquid.
        sims = {"a": _sim("a", 0.25), "b": _sim("b", 0.05)}
        f = _field([_mkt("a", 0.15, 0.14), _mkt("b", 0.06, 0.05)])
        edges = price_field(sims, f)
        plays = flag_plays(edges, [f], EdgeConfig(outright_threshold=0.05))
        assert any(p.golfer == "a" for p in plays)

    def test_negative_edge_not_flagged(self):
        # model 3% vs ask 15% → the observed real-world case; no play.
        sims = {"a": _sim("a", 0.03), "b": _sim("b", 0.02)}
        f = _field([_mkt("a", 0.15, 0.14), _mkt("b", 0.06, 0.05)])
        plays = flag_plays(price_field(sims, f), [f])
        assert plays == []

    def test_wide_spread_blocks_play(self):
        # big edge vs ask but the quote is illiquid (spread 0.30) → not flagged.
        sims = {"a": _sim("a", 0.60), "b": _sim("b", 0.05)}
        f = _field([_mkt("a", 0.20, 0.02), _mkt("b", 0.05, 0.03)])
        plays = flag_plays(price_field(sims, f), [f], EdgeConfig(max_spread=0.06))
        assert plays == []

    def test_placement_threshold_lower_than_outright(self):
        cfg = EdgeConfig(outright_threshold=0.05, placement_threshold=0.03)
        # +4pp edge: clears placement, not outright.
        sims = {"a": _sim("a", 0.0, t10=0.34), "b": _sim("b", 0.0, t10=0.30)}
        markets = [
            GolferMarket("polymarket_us", "T", GolfMarketType.top_10, "a", 0.30, "a",
                         yes_ask=0.30, yes_bid=0.29, implied_no_raw=0.71),
            GolferMarket("polymarket_us", "T", GolfMarketType.top_10, "b", 0.31, "b",
                         yes_ask=0.31, yes_bid=0.30, implied_no_raw=0.70),
        ]
        f = FieldPrices("polymarket_us", "T", GolfMarketType.top_10, markets)
        plays = flag_plays(price_field(sims, f), [f], cfg)
        assert any(p.golfer == "a" for p in plays)
