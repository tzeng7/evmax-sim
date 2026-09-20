"""Tests for the probability-space admission floor (favorite–longshot guard).

Covers ``dual_ev(edge_min_pp=)`` in evmax/ev/calculator.py, the per-sector
resolver + drop counter in evmax/agents/odds/ev_gap_agent.py, and the replay
harness ``make_gated_policy`` in evmax/backtest/sizing.py.

The floor SHIPS OFF (settings default 0.0, ``EDGE_MIN_PP_BY_SECTOR`` empty),
so the load-bearing tests are the identity ones: with the floor at 0 every
result must be byte-identical to the EV-only gate.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from evmax.agents.odds import ev_gap_agent as ega
from evmax.agents.odds.ev_gap_agent import EVGapAgent, resolve_edge_min_pp
from evmax.backtest.sizing import ResolvedRow, edge_pp_for_row, make_gated_policy, make_kelly_policy
from evmax.ev.calculator import dual_ev, effective_price
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ---------------------------------------------------------------------------
# dual_ev
# ---------------------------------------------------------------------------


class TestDualEvEdgeFloor:
    @pytest.mark.parametrize("price,prob", [(0.05, 0.07), (0.30, 0.34), (0.50, 0.55), (0.90, 0.93)])
    @pytest.mark.parametrize("venue", ["kalshi", "polymarket_us", None])
    def test_floor_zero_is_identity(self, price, prob, venue):
        base = dual_ev(price, prob, venue, 0.02)
        gated = dual_ev(price, prob, venue, 0.02, edge_min_pp=0.0)
        assert (gated.passes, gated.maker_only, gated.taker_ev, gated.maker_ev) == (
            base.passes, base.maker_only, base.taker_ev, base.maker_ev
        )
        assert gated.ev_passes == base.passes

    def test_edge_pp_fields_are_true_minus_effective_price(self):
        r = dual_ev(0.30, 0.34, "kalshi", 0.02)
        assert math.isclose(r.taker_edge_pp, (0.34 - effective_price(0.30, "kalshi")) * 100, abs_tol=1e-9)
        assert math.isclose(
            r.maker_edge_pp, (0.34 - effective_price(0.30, "kalshi", maker=True)) * 100, abs_tol=1e-9
        )
        assert r.maker_edge_pp >= r.taker_edge_pp

    def test_longshot_passes_ev_but_fails_pp_floor(self):
        # 5c contract, model 7%: EV ≈ +31% (clears 2%) but only ~1.7pp of
        # edge after the fee — exactly the row the EV% gate over-admits.
        r = dual_ev(0.05, 0.07, "kalshi", 0.02, edge_min_pp=2.0)
        assert r.ev_passes is True
        assert r.taker_ev > 0.20
        assert r.passes is False and r.maker_only is False

    def test_same_pp_edge_on_favorite_passes_both(self):
        # 70c contract with the same +3pp gross edge clears the pp floor; the
        # EV floor is what's tight there (~2.6% net of fee).
        r = dual_ev(0.70, 0.73, "kalshi", 0.02, edge_min_pp=2.0)
        assert r.passes is True and r.ev_passes is True

    def test_maker_only_can_be_created_by_the_pp_floor(self):
        # Taker pp edge just under the floor, maker pp edge (smaller fee) just
        # over: the play becomes fill-contingent, not dropped.
        price, prob = 0.30, 0.325
        taker_pp = (prob - effective_price(price, "kalshi")) * 100
        maker_pp = (prob - effective_price(price, "kalshi", maker=True)) * 100
        floor = (taker_pp + maker_pp) / 2
        assert taker_pp < floor < maker_pp
        r = dual_ev(price, prob, "kalshi", 0.0, edge_min_pp=floor)
        assert r.ev_passes is True and r.passes is True and r.maker_only is True

    def test_identity_holds_on_a_maker_only_case(self):
        # p=0.50, prob=0.525: taker EV ~1.4% (<2%), maker ~4% → maker_only in
        # the old gate; floor 0 must reproduce it exactly (the True branch).
        base = dual_ev(0.50, 0.525, "kalshi", 0.02)
        gated = dual_ev(0.50, 0.525, "kalshi", 0.02, edge_min_pp=0.0)
        assert base.maker_only is True
        assert (gated.passes, gated.maker_only) == (base.passes, base.maker_only)

    def test_negative_or_none_floor_treated_as_off(self):
        a = dual_ev(0.05, 0.07, "kalshi", 0.02, edge_min_pp=-3.0)
        b = dual_ev(0.05, 0.07, "kalshi", 0.02, edge_min_pp=None)  # type: ignore[arg-type]
        c = dual_ev(0.05, 0.07, "kalshi", 0.02)
        assert a.passes == b.passes == c.passes is True


class TestMakerLimitFloor:
    def test_floor_zero_is_identity(self):
        from evmax.ev.calculator import max_maker_limit_price

        for prob in (0.07, 0.34, 0.55, 0.93):
            assert max_maker_limit_price(prob, "kalshi", 0.02) == max_maker_limit_price(
                prob, "kalshi", 0.02, edge_min_pp=0.0
            )

    def test_ceiling_is_capped_at_the_pp_break_even(self):
        # A 7% longshot: the EV ceiling alone is ~6.9c, but a 2pp floor caps
        # the resting price so the maker edge at the ceiling is still >= 2pp.
        from evmax.ev.calculator import max_maker_limit_price

        ev_only = max_maker_limit_price(0.07, "kalshi", 0.02)
        capped = max_maker_limit_price(0.07, "kalshi", 0.02, edge_min_pp=2.0)
        assert capped is not None and capped < ev_only
        assert (0.07 - effective_price(capped, "kalshi", maker=True)) * 100 >= 2.0 - 1e-6

    def test_advertised_maker_price_never_below_gate(self):
        # The gate (dual_ev) and the advertised ceiling must agree: any rest
        # price at/below the ceiling would itself pass dual_ev with the floor.
        from evmax.ev.calculator import max_maker_limit_price

        prob, floor = 0.34, 2.0
        ceiling = max_maker_limit_price(prob, "kalshi", 0.02, edge_min_pp=floor)
        assert ceiling is not None
        assert dual_ev(ceiling, prob, "kalshi", 0.02, edge_min_pp=floor).passes


# ---------------------------------------------------------------------------
# resolver
# ---------------------------------------------------------------------------


class TestResolveEdgeMinPp:
    def test_ships_off(self):
        assert ega.EDGE_MIN_PP_BY_SECTOR == {}
        from evmax.settings import Settings

        assert resolve_edge_min_pp("nfl", Settings()) == 0.0

    def test_settings_value_used_when_no_override(self):
        s = MagicMock()
        s.edge_min_pp = 1.5
        assert resolve_edge_min_pp("nfl", s) == 1.5

    def test_sector_override_beats_settings(self, monkeypatch):
        monkeypatch.setitem(ega.EDGE_MIN_PP_BY_SECTOR, "ncaaf", 2.0)
        s = MagicMock()
        s.edge_min_pp = 1.0
        assert resolve_edge_min_pp("NCAAF", s) == 2.0
        assert resolve_edge_min_pp("nfl", s) == 1.0

    def test_mock_settings_attribute_is_off(self):
        # A bare MagicMock attribute is not a number → floor off, never a
        # MagicMock leaking into the gate arithmetic.
        assert resolve_edge_min_pp("nba", MagicMock()) == 0.0
        s = MagicMock()
        s.edge_min_pp = True
        assert resolve_edge_min_pp("nba", s) == 0.0


# ---------------------------------------------------------------------------
# agent-level
# ---------------------------------------------------------------------------


def _market(yes_price: float, yes_team: str = "pistons") -> PredictionMarket:
    no_price = round(min(0.99, max(0.01, (1.0 - yes_price) + 0.02)), 4)
    return PredictionMarket(
        id="kalshi:TEST-001", source=MarketSource.kalshi, sector="nba",
        market_type=MarketType.moneyline, yes_price=yes_price, no_price=no_price,
        volume_usd=10_000.0, yes_team=yes_team, team_home="pistons", team_away="warriors",
    )


def _sharp(true_prob_a: float) -> SharpOdds:
    return SharpOdds(
        event_id="nba::2026-03-20::pistons_vs_warriors", book=SharpBook.pinnacle,
        sector="nba", outcome_a_label="pistons", outcome_b_label="warriors",
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=round(true_prob_a, 4), true_prob_b=round(1 - true_prob_a, 4),
    )


def _agent(edge_min_pp: float = 0.0) -> EVGapAgent:
    with patch("evmax.agents.odds.ev_gap_agent.get_settings") as mock_settings:
        settings = MagicMock()
        settings.ev_threshold = 0.02
        settings.max_kelly_fraction = 0.05
        settings.chalk_price_ceiling = 0.90
        settings.edge_min_pp = edge_min_pp
        mock_settings.return_value = settings
        return EVGapAgent()


def _eval(agent: EVGapAgent, market, sharp):
    return agent._evaluate_pair(market, sharp, 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)


class TestAgentGate:
    def test_off_by_default_longshot_is_admitted(self):
        # 5c, sharp-only 7% → EV ≈ +31%: the EV-only gate admits it.
        gap = _eval(_agent(), _market(0.05), _sharp(0.07))
        assert gap is not None and gap.ev_pct > 0.2

    def test_floor_drops_longshot_and_counts_it(self):
        agent = _agent(edge_min_pp=2.0)
        gap = _eval(agent, _market(0.05), _sharp(0.07))
        assert gap is None
        assert agent._edge_floor_drops.get("nba") == 1

    def test_floor_leaves_favorite_with_real_edge_alone(self):
        agent = _agent(edge_min_pp=2.0)
        gap = _eval(agent, _market(0.60), _sharp(0.66))  # +6pp, EV ≈ +8%
        assert gap is not None
        assert agent._edge_floor_drops.get("nba", 0) == 0

    def test_sector_override_applies_without_settings(self, monkeypatch):
        monkeypatch.setitem(ega.EDGE_MIN_PP_BY_SECTOR, "nba", 2.0)
        agent = _agent(edge_min_pp=0.0)
        assert _eval(agent, _market(0.05), _sharp(0.07)) is None
        assert agent._edge_floor_drops.get("nba") == 1

    def test_row_failing_ev_floor_is_not_counted_as_pp_drop(self):
        agent = _agent(edge_min_pp=2.0)
        gap = _eval(agent, _market(0.55), _sharp(0.56))  # fails the 2% EV floor itself
        assert gap is None
        assert agent._edge_floor_drops.get("nba", 0) == 0

    def test_counter_is_keyed_on_the_request_sector(self):
        # market.sector is 'wnba' but the REQUEST sector ('nba') is what the
        # cycle summary logs/resets — the counter must land under the request key.
        agent = _agent(edge_min_pp=2.0)
        m = _market(0.05).model_copy(update={"sector": "wnba"})
        gap = agent._evaluate_pair(m, _sharp(0.07), 0.95, "nba", {}, {}, model_sources={}, kelly_base=0.5)
        assert gap is None
        assert agent._edge_floor_drops == {"nba": 1}

    def test_maker_limit_on_admitted_gap_respects_floor(self):
        agent = _agent(edge_min_pp=2.0)
        gap = _eval(agent, _market(0.30), _sharp(0.36))  # +6pp, admitted
        assert gap is not None and gap.maker_limit_price is not None
        assert (0.36 - effective_price(gap.maker_limit_price, "kalshi", maker=True)) * 100 >= 2.0 - 1e-6


class TestPropsGate:
    def _prop_market(self, yes_price: float) -> PredictionMarket:
        return PredictionMarket(
            id="kalshi:PROP-001", source=MarketSource.kalshi, sector="nba",
            market_type=MarketType.player_prop, yes_price=yes_price,
            no_price=round(min(0.99, (1.0 - yes_price) + 0.02), 4), volume_usd=5_000.0,
            yes_team="over", player_name="cade_cunningham", stat_type="points", line=24.5,
            team_home="pistons", team_away="warriors",
        )

    def _prop_sharp(self, over_prob: float) -> SharpOdds:
        return SharpOdds(
            event_id="nba::2026-03-20::pistons_vs_warriors::prop::cade_cunningham::points::24.5",
            book=SharpBook.pinnacle, sector="nba", outcome_a_label="over",
            outcome_b_label="under", outcome_a_decimal=2.0, outcome_b_decimal=2.0,
            true_prob_a=over_prob, true_prob_b=round(1 - over_prob, 4),
            true_prob_over=over_prob,  # the field _evaluate_prop_pair reads
        )

    def test_off_by_default_cheap_prop_admitted_then_dropped_by_floor(self):
        # 10c over with sharp 12%: EV ≈ +18% but only ~1.9pp net of fee.
        off = _agent()
        on = _agent(edge_min_pp=2.5)
        args = (self._prop_market(0.10), self._prop_sharp(0.12), 0.95, "nba")
        assert off._evaluate_prop_pair(*args, kelly_base=0.5) is not None
        assert on._evaluate_prop_pair(*args, kelly_base=0.5) is None
        assert on._edge_floor_drops.get("nba") == 1

    def test_prop_with_real_edge_passes_floor(self):
        on = _agent(edge_min_pp=2.5)
        gap = on._evaluate_prop_pair(
            self._prop_market(0.50), self._prop_sharp(0.58), 0.95, "nba", kelly_base=0.5
        )
        assert gap is not None
        assert on._edge_floor_drops.get("nba", 0) == 0


# ---------------------------------------------------------------------------
# replay harness
# ---------------------------------------------------------------------------


def _row(mid, blended, price, outcome=1):
    return ResolvedRow(
        market_id=mid, sector="nfl", market_type="moneyline",
        event_id=f"nfl::2026-09-14::{mid}", event_date="2026-09-14",
        blended=blended, price=price, outcome=outcome, ev_pct=0.05,
    )


class TestGatedPolicy:
    def test_edge_pp_matches_dual_ev_taker(self):
        r = _row("m", 0.34, 0.30)
        assert math.isclose(edge_pp_for_row(r), dual_ev(0.30, 0.34, "kalshi", 0.0).taker_edge_pp, abs_tol=1e-9)

    def test_floor_zero_is_identity(self):
        inner = make_kelly_policy(base_fraction=0.5, max_kelly=0.05)
        gated = make_gated_policy(inner, edge_min_pp=0.0)
        for r in (_row("a", 0.07, 0.05), _row("b", 0.55, 0.50), _row("c", 0.93, 0.90)):
            assert gated(r) == inner(r)

    def test_floor_zeroes_only_below_floor_rows(self):
        inner = make_kelly_policy(base_fraction=0.5, max_kelly=0.05)
        gated = make_gated_policy(inner, edge_min_pp=2.0)
        longshot = _row("a", 0.07, 0.05)   # ~1.7pp net of fee → sits out
        fav = _row("b", 0.66, 0.60)        # ~4.7pp → sized as before
        assert inner(longshot) > 0 and gated(longshot) == 0.0
        assert gated(fav) == inner(fav) > 0
