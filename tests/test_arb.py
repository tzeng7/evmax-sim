"""Tests for evmax.arb — cross-venue arbitrage detection."""

from datetime import datetime, timezone

import pytest

from evmax.arb import ARB_LEAGUE_MAP, find_arbs
from evmax.clients.polymarket_us import POLYMARKET_US_LEAGUE_MAP
from evmax.models.market import MarketSource, MarketType, PredictionMarket

NOW = datetime(2026, 7, 9, 18, 0, tzinfo=timezone.utc)
# Kalshi-style noon-UTC date anchor for a July 10 game
KDATE = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
# PolyUS true start for the same July 10 game (8:05pm ET = 00:05 UTC July 11)
PDATE = datetime(2026, 7, 11, 0, 5, tzinfo=timezone.utc)


def mk(
    mid,
    source,
    yes_team,
    yes_price,
    no_price,
    market_type=MarketType.moneyline,
    line=None,
    event_date=KDATE,
    home="astros",
    away="rangers",
    sector="baseball",
    volume=100.0,
):
    return PredictionMarket(
        id=mid,
        source=source,
        sector=sector,
        market_type=market_type,
        title=f"{away} vs {home}",
        ticker=mid.split(":", 1)[-1],
        yes_price=yes_price,
        no_price=no_price,
        volume_usd=volume,
        team_home=home,
        team_away=away,
        line=line,
        event_date=event_date,
        yes_team=yes_team,
    )


class TestMoneylineArb:
    def test_cross_venue_arb_detected(self):
        pool = [
            mk("kalshi:T1", MarketSource.kalshi, "astros", 0.40, 0.62),
            mk("polymarket_us:t1", MarketSource.polymarket_us, "astros",
               0.55, 0.50, event_date=PDATE),
        ]
        arbs = find_arbs(pool, max_net_cost=1.0, now=NOW)
        assert len(arbs) == 1
        arb = arbs[0]
        # cheapest cover: astros K-yes @0.40 + rangers P-no @0.50 = 0.90
        assert arb.gross_cost == pytest.approx(0.90)
        assert arb.net_cost > arb.gross_cost  # fees added
        assert arb.net_edge > 0
        assert arb.cross_venue
        sides = {(leg.venue, leg.side) for leg in arb.legs}
        assert sides == {("kalshi", "yes"), ("polymarket_us", "no")}

    def test_no_arb_when_books_efficient(self):
        pool = [
            mk("kalshi:T1", MarketSource.kalshi, "astros", 0.51, 0.51),
            mk("polymarket_us:t1", MarketSource.polymarket_us, "astros",
               0.51, 0.51, event_date=PDATE),
        ]
        assert find_arbs(pool, max_net_cost=1.0, now=NOW) == []

    def test_series_game_date_trap_not_paired(self):
        """REGRESSION: MLB series play consecutive nights. Tonight's Kalshi
        market must never pair with tomorrow's PolyUS market for the same
        team pair — that manufactured fake +12pp 'arbs' in the first sweep."""
        tonight = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)   # Kalshi anchor, July 9 ET
        tomorrow = datetime(2026, 7, 11, 0, 5, tzinfo=timezone.utc)  # PolyUS start, July 10 ET
        pool = [
            # tonight's game trading in-play at a big discount
            mk("kalshi:T1", MarketSource.kalshi, "astros", 0.20, 0.85,
               event_date=tonight),
            # tomorrow's game pre-game near 50/50
            mk("polymarket_us:t1", MarketSource.polymarket_us, "astros",
               0.55, 0.50, event_date=tomorrow),
        ]
        arbs = find_arbs(pool, max_net_cost=2.0, include_in_play=True, now=NOW)
        # Each date-group only has one venue: single-market baskets exist but
        # no cross-venue pairing between the two games.
        assert all(not a.cross_venue for a in arbs)

    def test_in_play_excluded_by_default(self):
        started = datetime(2026, 7, 9, 17, 0, tzinfo=timezone.utc)  # before NOW
        pool = [
            mk("kalshi:T1", MarketSource.kalshi, "astros", 0.40, 0.62,
               event_date=datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)),
            mk("polymarket_us:t1", MarketSource.polymarket_us, "astros",
               0.55, 0.50, event_date=started),
        ]
        assert find_arbs(pool, max_net_cost=1.0, now=NOW) == []
        assert len(find_arbs(pool, max_net_cost=1.0, include_in_play=True, now=NOW)) == 1

    def test_yes_team_alignment(self):
        """A rangers-YES Kalshi market must contribute its YES ask to the
        rangers outcome and its NO ask to the astros outcome."""
        pool = [
            mk("kalshi:T2", MarketSource.kalshi, "rangers", 0.45, 0.58),
            mk("polymarket_us:t1", MarketSource.polymarket_us, "astros",
               0.40, 0.63, event_date=PDATE),
        ]
        arbs = find_arbs(pool, max_net_cost=1.0, now=NOW)
        assert len(arbs) == 1
        # astros P-yes @0.40 + rangers K-yes @0.45 = 0.85
        assert arbs[0].gross_cost == pytest.approx(0.85)


class TestThreeWay:
    def test_soccer_draw_basket(self):
        common = dict(home="seattle", away="portland", sector="soccer")
        pool = [
            mk("kalshi:S1", MarketSource.kalshi, "seattle", 0.30, 0.75, **common),
            mk("kalshi:S2", MarketSource.kalshi, "portland", 0.35, 0.70, **common),
            mk("kalshi:S3", MarketSource.kalshi, "tie", 0.28, 0.75, **common),
            mk("polymarket_us:s1", MarketSource.polymarket_us, "seattle",
               0.25, 0.72, event_date=PDATE, **common),
        ]
        arbs = find_arbs(pool, max_net_cost=1.0, now=NOW)
        assert len(arbs) == 1
        # seattle P @0.25 + portland K @0.35 + draw K @0.28 = 0.88
        assert arbs[0].gross_cost == pytest.approx(0.88)
        assert {leg.outcome for leg in arbs[0].legs} == {"seattle", "portland", "draw"}

    def test_missing_draw_market_never_degrades_to_two_way(self):
        """REGRESSION: a soccer event whose draw market failed to parse must
        NOT be treated as a 2-way — YES on both teams leaves the draw
        uncovered (caught live 2026-07-09 as a fake +6.2pp 'arb')."""
        common = dict(home="levski", away="borac", sector="soccer")
        pool = [
            mk("polymarket_us:m1", MarketSource.polymarket_us, "levski",
               0.59, 0.44, event_date=PDATE, **common),
            mk("polymarket_us:m2", MarketSource.polymarket_us, "borac",
               0.32, 0.70, event_date=PDATE, **common),
        ]
        assert find_arbs(pool, max_net_cost=1.0, now=NOW) == []

    def test_incomplete_three_way_no_basket(self):
        common = dict(home="seattle", away="portland", sector="soccer")
        pool = [
            mk("kalshi:S1", MarketSource.kalshi, "seattle", 0.30, 0.75, **common),
            mk("kalshi:S3", MarketSource.kalshi, "tie", 0.28, 0.75, **common),
        ]
        assert find_arbs(pool, max_net_cost=5.0, now=NOW) == []


class TestArbLeagueMap:
    """ARB_LEAGUE_MAP — arb coverage decoupled from betting coverage."""

    def test_superset_of_betting_map(self):
        for sector, leagues in POLYMARKET_US_LEAGUE_MAP.items():
            assert set(leagues) <= set(ARB_LEAGUE_MAP[sector])

    def test_tier1_extensions_present(self):
        assert ARB_LEAGUE_MAP["worldcup"] == ["fwc"]
        assert ARB_LEAGUE_MAP["lol"] == ["lol"]
        assert ARB_LEAGUE_MAP["cs2"] == ["cs2"]
        assert {"lal", "bun", "sea", "uefa"} <= set(ARB_LEAGUE_MAP["soccer"])

    def test_betting_map_not_mutated(self):
        """The arb map must be a copy — extending it can never flip
        worldcup/lol/cs2 on for the betting scan."""
        assert "worldcup" not in POLYMARKET_US_LEAGUE_MAP
        assert "lol" not in POLYMARKET_US_LEAGUE_MAP
        assert "cs2" not in POLYMARKET_US_LEAGUE_MAP
        assert "lal" not in POLYMARKET_US_LEAGUE_MAP["soccer"]

    def test_every_key_is_a_registered_category(self):
        from evmax.categories import all_categories
        keys = {c.key for c in all_categories()}
        assert set(ARB_LEAGUE_MAP) <= keys

    def test_single_venue_baskets_hidden_by_default(self):
        """A lone market's yes+no basket is structurally ≥ $1 — dropped
        unless cross_venue_only=False."""
        pool = [mk("kalshi:T1", MarketSource.kalshi, "astros", 0.99, 0.01)]
        assert find_arbs(pool, max_net_cost=1.5, now=NOW) == []
        arbs = find_arbs(pool, max_net_cost=1.5, cross_venue_only=False, now=NOW)
        assert len(arbs) == 1 and not arbs[0].cross_venue

    def test_worldcup_baskets_require_draw(self):
        """A worldcup knockout pair without a draw quote must not produce a
        2-way basket — same guarantee as club soccer."""
        common = dict(home="france", away="morocco", sector="worldcup")
        pool = [
            mk("kalshi:W1", MarketSource.kalshi, "france", 0.55, 0.50, **common),
            mk("polymarket_us:w1", MarketSource.polymarket_us, "morocco",
               0.30, 0.72, event_date=PDATE, **common),
        ]
        assert find_arbs(pool, max_net_cost=1.0, now=NOW) == []
        # with a draw leg the basket completes
        pool.append(mk("kalshi:W2", MarketSource.kalshi, "tie", 0.10, 0.92, **common))
        arbs = find_arbs(pool, max_net_cost=1.0, now=NOW)
        assert len(arbs) == 1
        assert arbs[0].gross_cost == pytest.approx(0.55 + 0.30 + 0.10)


class TestSpreadAndTotal:
    def test_spread_same_line_pairing(self):
        pool = [
            mk("kalshi:SP1", MarketSource.kalshi, "astros", 0.45, 0.60,
               market_type=MarketType.spread, line=-1.5),
            mk("polymarket_us:sp1", MarketSource.polymarket_us, "astros",
               0.55, 0.48, market_type=MarketType.spread, line=-1.5,
               event_date=PDATE),
        ]
        arbs = find_arbs(pool, max_net_cost=1.0, now=NOW)
        assert len(arbs) == 1
        assert arbs[0].gross_cost == pytest.approx(0.45 + 0.48)

    def test_spread_mirrored_line_accepted(self):
        pool = [
            mk("kalshi:SP1", MarketSource.kalshi, "astros", 0.45, 0.60,
               market_type=MarketType.spread, line=-1.5),
            mk("polymarket_us:sp2", MarketSource.polymarket_us, "rangers",
               0.48, 0.60, market_type=MarketType.spread, line=1.5,
               event_date=PDATE),
        ]
        arbs = find_arbs(pool, max_net_cost=1.0, now=NOW)
        # astros -1.5 YES @0.45 + rangers +1.5 YES @0.48 covers both outcomes
        assert any(a.gross_cost == pytest.approx(0.93) for a in arbs)

    def test_spread_different_lines_not_paired(self):
        pool = [
            mk("kalshi:SP1", MarketSource.kalshi, "astros", 0.45, 0.90,
               market_type=MarketType.spread, line=-1.5),
            mk("polymarket_us:sp1", MarketSource.polymarket_us, "astros",
               0.30, 0.90, market_type=MarketType.spread, line=-2.5,
               event_date=PDATE),
        ]
        arbs = find_arbs(pool, max_net_cost=5.0, now=NOW)
        assert all(not a.cross_venue for a in arbs)

    def test_total_pairing(self):
        pool = [
            mk("kalshi:TO1", MarketSource.kalshi, "over", 0.42, 0.62,
               market_type=MarketType.total, line=8.5),
            mk("polymarket_us:to1", MarketSource.polymarket_us, "over",
               0.55, 0.50, market_type=MarketType.total, line=8.5,
               event_date=PDATE),
        ]
        arbs = find_arbs(pool, max_net_cost=1.0, now=NOW)
        assert len(arbs) == 1
        # over K @0.42 + under P @0.50
        assert arbs[0].gross_cost == pytest.approx(0.92)


class TestFeeNetting:
    def test_gross_arb_killed_by_fees(self):
        """A 1pp gross edge at midpoint prices loses to ~3.2pp round-trip
        taker fees — must NOT be reported as net-positive."""
        pool = [
            mk("kalshi:T1", MarketSource.kalshi, "astros", 0.49, 0.55),
            mk("polymarket_us:t1", MarketSource.polymarket_us, "astros",
               0.55, 0.50, event_date=PDATE),
        ]
        arbs = find_arbs(pool, max_net_cost=1.05, now=NOW)
        target = [a for a in arbs if a.cross_venue]
        assert target
        assert target[0].gross_cost == pytest.approx(0.99)
        assert target[0].net_edge < 0
