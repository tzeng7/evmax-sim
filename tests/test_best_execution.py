"""Side-aware best-execution collapse for the display play list (GAP 3).

Same bet on two venues → one actionable row on the better book, alternative
annotated. Genuinely different bets (opposite ML sides, over vs under, different
lines) must NOT collapse. View-layer only — this never touches persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.ev.best_execution import apply_venue_cash_cap, collapse_best_execution


def _gap(
    market_id: str,
    event_id: str,
    *,
    venue: str = "kalshi",
    yes_team: str = "lakers",
    market_type: str = "moneyline",
    price: float = 0.45,
    ev: float = 0.05,
    kelly: float = 0.03,
    line: float | None = None,
) -> EVGap:
    return EVGap(
        market_id=market_id,
        event_id=event_id,
        sector="nba",
        yes_team=yes_team,
        market_type=market_type,
        kalshi_yes_price=price,
        sharp_true_prob=0.50,
        blended_true_prob=0.50,
        ev_pct=ev,
        kelly_full=kelly * 2,
        kelly_fraction=kelly,
        match_confidence=95.0,
        volume_usd=5_000.0,
        spread_pct=0.02,
        event_date=datetime(2026, 4, 30, 19, 0, tzinfo=timezone.utc),
        model_sources="sharp",
        line=line,
        event_title="Lakers vs Warriors",
        venue=venue,
    )


EID = "nba::2026-04-30::lakers_vs_warriors"


class TestCollapse:
    def test_same_bet_two_venues_collapses_to_higher_ev(self):
        """Same outcome + side on both venues → one row (higher EV / cheaper
        wins), the loser annotated as the alternative."""
        kalshi = _gap("kalshi:A", EID, venue="kalshi", ev=0.04, price=0.46)
        poly = _gap("polymarket_us:A", EID, venue="polymarket_us", ev=0.06, price=0.44)
        out = collapse_best_execution([kalshi, poly])
        assert len(out) == 1
        assert out[0].venue == "polymarket_us"        # higher EV survives
        assert out[0].alt_venue == "kalshi"
        assert out[0].alt_venue_price == pytest.approx(0.46)
        assert out[0].alt_venue_ev_pct == pytest.approx(0.04)

    def test_live_sized_beats_shadow_even_if_lower_ev(self):
        """A live-sized (kelly>0) leg beats a shadow-bound zero-Kelly leg even
        when the shadow leg shows a higher EV — you execute the book you can
        actually stake."""
        live = _gap("kalshi:A", EID, venue="kalshi", ev=0.03, kelly=0.03)
        shadow = _gap("polymarket_us:A", EID, venue="polymarket_us", ev=0.09, kelly=0.0)
        out = collapse_best_execution([live, shadow])
        assert len(out) == 1
        assert out[0].venue == "kalshi"
        assert out[0].alt_venue == "polymarket_us"

    def test_opposite_ml_sides_are_kept(self):
        """Team A YES on one venue and Team B YES on the other is a dislocation,
        not a duplicate — both stay, neither annotated."""
        a = _gap("kalshi:A", EID, venue="kalshi", yes_team="lakers")
        b = _gap("polymarket_us:B", EID, venue="polymarket_us", yes_team="warriors")
        out = collapse_best_execution([a, b])
        assert len(out) == 2
        assert all(g.alt_venue is None for g in out)

    def test_over_and_under_same_total_are_kept(self):
        over = _gap("kalshi:O", EID + "::total", venue="kalshi",
                    market_type="total", yes_team="over", line=220.5)
        under = _gap("polymarket_us:U", EID + "::total", venue="polymarket_us",
                     market_type="total", yes_team="under", line=220.5)
        out = collapse_best_execution([over, under])
        assert len(out) == 2

    def test_different_lines_do_not_collapse(self):
        """Different spread lines are different bets even on one side."""
        a = _gap("kalshi:S1", EID + "::spread", venue="kalshi",
                 market_type="spread", yes_team="lakers", line=-4.5)
        b = _gap("polymarket_us:S2", EID + "::spread", venue="polymarket_us",
                 market_type="spread", yes_team="lakers", line=-6.5)
        out = collapse_best_execution([a, b])
        assert len(out) == 2

    def test_single_venue_unchanged(self):
        g = _gap("kalshi:A", EID, venue="kalshi")
        out = collapse_best_execution([g])
        assert len(out) == 1
        assert out[0].alt_venue is None

    def test_preserves_first_appearance_order(self):
        g1 = _gap("kalshi:G1", "nba::2026-04-30::a_vs_b", venue="kalshi", yes_team="a")
        g2 = _gap("kalshi:G2", "nba::2026-04-30::c_vs_d", venue="kalshi", yes_team="c")
        out = collapse_best_execution([g1, g2])
        assert [g.market_id for g in out] == ["kalshi:G1", "kalshi:G2"]

    def test_does_not_mutate_inputs(self):
        kalshi = _gap("kalshi:A", EID, venue="kalshi", ev=0.06)
        poly = _gap("polymarket_us:A", EID, venue="polymarket_us", ev=0.04)
        collapse_best_execution([kalshi, poly])
        # originals keep alt_venue = None (replace produced a new object)
        assert kalshi.alt_venue is None
        assert poly.alt_venue is None


# ---------------------------------------------------------------------------
# venue_options (dropdown) — the full ranked leg set carried on the winner
# ---------------------------------------------------------------------------


class TestVenueOptions:
    def test_two_venues_attach_ranked_options(self):
        """A dual-venue bet carries every leg (winner first) so the display can
        offer a venue dropdown."""
        kalshi = _gap("kalshi:A", EID, venue="kalshi", ev=0.04, price=0.46)
        poly = _gap("polymarket_us:A", EID, venue="polymarket_us", ev=0.06, price=0.44)
        out = collapse_best_execution([kalshi, poly])
        assert len(out) == 1
        opts = out[0].venue_options
        assert opts is not None and len(opts) == 2
        assert opts[0].venue == "polymarket_us"          # best-execution winner first
        assert {o.venue for o in opts} == {"kalshi", "polymarket_us"}
        # Nested option legs must not themselves carry options (no recursion).
        assert all(o.venue_options is None for o in opts)

    def test_single_venue_has_no_options(self):
        out = collapse_best_execution([_gap("kalshi:A", EID, venue="kalshi")])
        assert out[0].venue_options is None

    def test_opposite_sides_have_no_options(self):
        """Different bets (opposite ML sides) stay as separate rows — no dropdown."""
        a = _gap("kalshi:A", EID, venue="kalshi", yes_team="lakers")
        b = _gap("polymarket_us:B", EID, venue="polymarket_us", yes_team="warriors")
        out = collapse_best_execution([a, b])
        assert all(g.venue_options is None for g in out)

    def test_options_include_maker_only_leg(self):
        """Mirrors the real dual-venue row: both legs zero-Kelly, one venue
        clears only as a maker. The dropdown must surface both, preserving each
        leg's maker_only flag so the display can tag the maker book."""
        kalshi = _gap("kalshi:A", EID, venue="kalshi", ev=0.015, kelly=0.0)
        kalshi.maker_only = True
        poly = _gap("polymarket_us:A", EID, venue="polymarket_us", ev=0.021, kelly=0.0)
        out = collapse_best_execution([kalshi, poly])
        assert out[0].venue == "polymarket_us"           # higher EV wins the tie of zeros
        opts = {o.venue: o for o in out[0].venue_options}
        assert opts["kalshi"].maker_only is True
        assert opts["polymarket_us"].maker_only is False

    def test_does_not_mutate_input_options(self):
        kalshi = _gap("kalshi:A", EID, venue="kalshi", ev=0.06)
        poly = _gap("polymarket_us:A", EID, venue="polymarket_us", ev=0.04)
        collapse_best_execution([kalshi, poly])
        assert kalshi.venue_options is None
        assert poly.venue_options is None


# ---------------------------------------------------------------------------
# apply_venue_cash_cap (GAP 2)
# ---------------------------------------------------------------------------


class TestVenueCashCap:
    def test_scales_venue_over_cash(self):
        # two kalshi legs: stake = (0.05 + 0.03) * $1000 = $80 > $40 cash
        # → scale 0.5 → 0.025 + 0.015
        a = _gap("kalshi:A", EID, venue="kalshi", yes_team="lakers", kelly=0.05)
        b = _gap("kalshi:B", EID + "::spread", venue="kalshi",
                 market_type="spread", yes_team="lakers", kelly=0.03, line=-4.5)
        out = apply_venue_cash_cap([a, b], bankroll=1000.0, cash_by_venue={"kalshi": 40.0})
        sized = {g.market_id: g.kelly_fraction for g in out}
        assert sized["kalshi:A"] == pytest.approx(0.025)
        assert sized["kalshi:B"] == pytest.approx(0.015)

    def test_under_cash_unchanged(self):
        a = _gap("kalshi:A", EID, venue="kalshi", kelly=0.03)  # $30 < $100 cash
        out = apply_venue_cash_cap([a], bankroll=1000.0, cash_by_venue={"kalshi": 100.0})
        assert out[0].kelly_fraction == pytest.approx(0.03)

    def test_missing_cash_venue_uncapped(self):
        a = _gap("kalshi:A", EID, venue="kalshi", kelly=0.05)  # $50, no kalshi cash entry
        out = apply_venue_cash_cap([a], bankroll=1000.0, cash_by_venue={"polymarket_us": 10.0})
        assert out[0].kelly_fraction == pytest.approx(0.05)

    def test_per_venue_independent(self):
        k = _gap("kalshi:A", EID, venue="kalshi", yes_team="lakers", kelly=0.05)  # $50 ≤ $50
        p = _gap("polymarket_us:A", "nba::2026-04-30::x_vs_y", venue="polymarket_us",
                 yes_team="x", kelly=0.05)  # $50 > $25 → half
        out = apply_venue_cash_cap(
            [k, p], bankroll=1000.0,
            cash_by_venue={"kalshi": 50.0, "polymarket_us": 25.0},
        )
        sized = {g.market_id: g.kelly_fraction for g in out}
        assert sized["kalshi:A"] == pytest.approx(0.05)
        assert sized["polymarket_us:A"] == pytest.approx(0.025)

    def test_empty_cash_map_is_noop(self):
        a = _gap("kalshi:A", EID, venue="kalshi", kelly=0.05)
        out = apply_venue_cash_cap([a], bankroll=1000.0, cash_by_venue={})
        assert out[0].kelly_fraction == pytest.approx(0.05)

    def test_zero_bankroll_is_noop(self):
        a = _gap("kalshi:A", EID, venue="kalshi", kelly=0.05)
        out = apply_venue_cash_cap([a], bankroll=0.0, cash_by_venue={"kalshi": 1.0})
        assert out[0].kelly_fraction == pytest.approx(0.05)

    def test_does_not_mutate_inputs(self):
        a = _gap("kalshi:A", EID, venue="kalshi", kelly=0.05)
        apply_venue_cash_cap([a], bankroll=1000.0, cash_by_venue={"kalshi": 10.0})
        assert a.kelly_fraction == pytest.approx(0.05)
