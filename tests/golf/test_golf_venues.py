"""Venue-adapter capability tests: Kalshi + PolyUS golf field parsing from real
(PolyUS) and synthetic (Kalshi) fixtures. Confirms we can actually read a field
of per-golfer YES contracts with prices from both venues."""

import pytest

from evmax.golf.models import GolfMarketType
from evmax.golf.names import canonical
from evmax.golf.venues.kalshi_golf import parse_kalshi_markets
from evmax.golf.venues.polymarket_golf import parse_golf_events


class TestPolymarketGolf:
    def test_parses_winner_field(self, polyus_golf_payload):
        fields = parse_golf_events(polyus_golf_payload)
        win = [f for f in fields if f.market_type is GolfMarketType.win]
        assert win, "expected a winner field"
        f = win[0]
        assert len(f.markets) > 100  # full Open field
        assert "winner" in f.tournament.lower()

    def test_yes_bid_ask_extracted(self, polyus_golf_payload):
        f = [x for x in parse_golf_events(polyus_golf_payload) if x.market_type is GolfMarketType.win][0]
        priced = f.priced_markets()
        assert len(priced) > 100
        # a favourite exists with a tight two-sided quote
        tight = [m for m in priced if m.spread is not None and m.spread < 0.05]
        assert tight, "expected at least one liquid two-sided market"
        for m in priced:
            assert 0.0 < m.implied_raw < 1.0

    def test_names_canonicalised(self, polyus_golf_payload):
        f = [x for x in parse_golf_events(polyus_golf_payload) if x.market_type is GolfMarketType.win][0]
        names = {m.golfer for m in f.markets}
        assert canonical("Scottie Scheffler") in names
        assert all(n == canonical(n) for n in names)

    def test_round_leader_market_type_detected(self, polyus_golf_payload):
        fields = parse_golf_events(polyus_golf_payload)
        types = {f.market_type for f in fields}
        # fixture carries winner + round-1 leader
        assert GolfMarketType.win in types
        assert GolfMarketType.r1_leader in types


class TestKalshiGolf:
    def test_parses_field_and_names(self, kalshi_golf_markets):
        fields = parse_kalshi_markets(kalshi_golf_markets, series_ticker="KXPGATOUR")
        assert len(fields) == 1
        f = fields[0]
        assert f.market_type is GolfMarketType.win
        assert canonical("Scottie Scheffler") in {m.golfer for m in f.markets}

    def test_cents_converted_to_probability(self, kalshi_golf_markets):
        f = parse_kalshi_markets(kalshi_golf_markets, series_ticker="KXPGATOUR")[0]
        sch = next(m for m in f.markets if m.golfer == canonical("Scottie Scheffler"))
        assert sch.yes_ask == pytest.approx(0.15)  # 15 cents
        assert sch.yes_bid == pytest.approx(0.12)
        assert sch.implied_no_raw == pytest.approx(0.88)  # 88-cent no ask

    def test_tournament_extracted_from_title(self, kalshi_golf_markets):
        f = parse_kalshi_markets(kalshi_golf_markets, series_ticker="KXPGATOUR")[0]
        assert "Open Championship" in f.tournament

    def test_unpriced_contract_excluded_from_priced(self, kalshi_golf_markets):
        f = parse_kalshi_markets(kalshi_golf_markets, series_ticker="KXPGATOUR")[0]
        # the null-priced Zecheng Dou row parses but is not "priced"
        assert canonical("Zecheng Dou") in {m.golfer for m in f.markets}
        assert canonical("Zecheng Dou") not in {m.golfer for m in f.priced_markets()}

    def test_unknown_series_yields_no_fields(self, kalshi_golf_markets):
        # A series we haven't mapped (e.g. cut-line) produces nothing.
        fields = parse_kalshi_markets(kalshi_golf_markets, series_ticker="KXPGACUTLINE")
        assert fields == []


class TestKalshiNonOutright:
    """Make-cut / top-N per-golfer markets — live on Kalshi for majors, mapping
    1:1 onto the simulator's outputs."""

    def _mk(self, series, mt_word):
        return [
            {"ticker": f"{series}-THOC26-SSCH", "yes_sub_title": "Scottie Scheffler",
             "title": f"The Open Championship: Will Scottie Scheffler {mt_word}?",
             "yes_bid": 60, "yes_ask": 63, "no_bid": 37, "no_ask": 40, "volume": 900},
            {"ticker": f"{series}-THOC26-RMCI", "yes_sub_title": "Rory McIlroy",
             "title": f"The Open Championship: Will Rory McIlroy {mt_word}?",
             "yes_bid": 55, "yes_ask": 58, "no_bid": 42, "no_ask": 45, "volume": 700},
        ]

    def test_make_cut_series_maps(self):
        f = parse_kalshi_markets(self._mk("KXPGAMAKECUT", "make the cut"))[0]
        assert f.market_type is GolfMarketType.make_cut
        assert canonical("Scottie Scheffler") in {m.golfer for m in f.markets}

    def test_top10_and_top5_map_distinctly(self):
        f10 = parse_kalshi_markets(self._mk("KXPGATOP10", "finish top 10"))[0]
        f5 = parse_kalshi_markets(self._mk("KXPGATOP5", "finish top 5"))[0]
        assert f10.market_type is GolfMarketType.top_10
        assert f5.market_type is GolfMarketType.top_5  # not shadowed by TOP10

    def test_make_cut_devig_is_per_contract_not_sum_to_one(self):
        from evmax.golf.devig import devig_field
        f = parse_kalshi_markets(self._mk("KXPGAMAKECUT", "make the cut"))[0]
        devig_field(f)
        # non-exclusive market: two ~60% make-cut contracts must BOTH stay ~60%,
        # NOT be normalised to sum to 1.
        assert all(0.5 < v < 0.7 for v in f.devigged.values())
        assert sum(f.devigged.values()) > 1.0
