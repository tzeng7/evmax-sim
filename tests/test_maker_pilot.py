"""Maker-only pilot lane for shallow model-priced NFL spread rungs.

evmax/ev/maker_pilot.py re-opens the shallow (|line| <= 10.5) `spread_pmf`
rungs that the 2026-10-03 ladder-only rule hides, as ¼-Kelly MAKER plays on
Kalshi. These tests pin the eligibility boundary, the conversion, the per-game
cap, the checkpoint verdict and the wiring through EVGapAgent.
"""
from __future__ import annotations

import dataclasses

import pytest

import evmax.agents.odds.ev_gap_agent as ev_mod
import evmax.ev.maker_pilot as mp
import evmax.models_ml.spread_distribution as sd
from evmax.agents.cleanup import contamination, integrity
from evmax.agents.odds.ev_gap_agent import EVGap, EVGapAgent
from evmax.fees import venue_fee_prob
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


@pytest.fixture
def nfl_ladder(monkeypatch):
    """The shipped NFL ladder configuration with the pilot enabled."""
    monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", False)
    monkeypatch.setattr(sd, "SPREAD_LADDER_SECTORS", {"nfl"})
    monkeypatch.setattr(sd, "SPREAD_LADDER_EXACT_SECTORS", {"nfl"})
    monkeypatch.setattr(sd, "SPREAD_LADDER_ONLY_SECTORS", {"nfl"})
    monkeypatch.setattr(mp, "MAKER_PILOT_SECTORS", {"nfl"})


def _gap(**kw) -> EVGap:
    base = dict(
        market_id="kalshi:KXNFLSPREAD-26OCT11CHIGB-CHI4", event_id="nfl::2026-10-11::bears_vs_packers::spread",
        sector="nfl", yes_team="bears", market_type="spread", kalshi_yes_price=0.25,
        sharp_true_prob=0.27, blended_true_prob=0.27, ev_pct=0.01, kelly_full=0.03,
        kelly_fraction=0.012, match_confidence=95.0, volume_usd=1000.0, spread_pct=0.04,
        event_date=None, line=-3.5, model_sources="sharp+spread_pmf", full_blend=False,
        maker_bid_price=0.24, maker_bid_ev_pct=0.10, maker_bid_kelly_fraction=0.012,
        venue="kalshi",
    )
    base.update(kw)
    return EVGap(**base)


# --- eligibility -------------------------------------------------------------

class TestEligibility:
    @pytest.mark.parametrize("sector,mt,src,line,venue,expected", [
        ("nfl", "spread", "sharp+spread_pmf", -3.5, "kalshi", True),         # underdog lays
        ("nfl", "spread", "sharp+spread_pmf+no_side", 3.5, "kalshi", True),  # NO-side take
        ("nfl", "spread", "sharp+spread_pmf", -10.5, "kalshi", True),        # boundary
        ("nfl", "spread", "sharp+spread_pmf", -13.5, "kalshi", False),       # deep rung
        ("nfl", "spread", "sharp+sharp_ladder", -3.5, "kalshi", False),      # Pinnacle-priced
        ("nfl", "spread", "sharp+spread_dist", -3.5, "kalshi", False),       # superseded pricing
        ("nfl", "spread", "sharp+spread_pmf", -3.5, "polymarket_us", False), # Kalshi only
        ("nba", "spread", "sharp+spread_pmf", -3.5, "kalshi", False),        # NFL only
        ("nfl", "moneyline", "sharp+spread_pmf", None, "kalshi", False),
        ("nfl", "spread", "sharp+spread_pmf", None, "kalshi", False),
    ])
    def test_boundary(self, nfl_ladder, sector, mt, src, line, venue, expected):
        assert mp.maker_pilot_eligible(sector, mt, src, line, venue, yes_is_underdog=True) is expected

    @pytest.mark.parametrize("underdog,expected", [(True, True), (False, False), (None, False)])
    def test_underdog_only(self, nfl_ladder, underdog, expected):
        # Favorite rungs (e.g. Texans -10.5 off a -7.5 main) have no evidence behind
        # them; an unknown side fails closed.
        assert mp.maker_pilot_eligible(
            "nfl", "spread", "sharp+spread_pmf", -9.5, "kalshi", yes_is_underdog=underdog,
        ) is expected

    def test_underdog_only_switch_admits_favorites(self, nfl_ladder, monkeypatch):
        monkeypatch.setattr(mp, "MAKER_PILOT_UNDERDOG_ONLY", False)
        assert mp.maker_pilot_eligible("nfl", "spread", "sharp+spread_pmf", -9.5, "kalshi", False)

    def test_disabled_when_sector_set_is_empty(self, monkeypatch):
        monkeypatch.setattr(mp, "MAKER_PILOT_SECTORS", set())
        assert not mp.maker_pilot_eligible("nfl", "spread", "sharp+spread_pmf", -3.5, "kalshi", True)


# --- conversion --------------------------------------------------------------

class TestApplyMakerPilot:
    def test_converts_to_quarter_kelly_maker_play(self, nfl_ladder):
        out = mp.apply_maker_pilot(_gap(), yes_is_underdog=True)
        assert out.maker_only is True
        assert out.kelly_fraction == 0.0                      # no taker stake, ever
        assert out.maker_bid_kelly_fraction == pytest.approx(0.012 * mp.MAKER_PILOT_KELLY_MULT)
        assert out.model_sources == "sharp+spread_pmf+maker_pilot"
        assert out.full_blend is True                         # displayed as a MAKER play

    def test_full_blend_flag_agrees_with_has_full_blend(self, nfl_ladder):
        out = mp.apply_maker_pilot(_gap(), yes_is_underdog=True)
        assert ev_mod.has_full_blend("nfl", out.model_sources, "spread") is True
        assert ev_mod.has_full_blend("nfl", "sharp+spread_pmf", "spread") is False

    def test_is_idempotent(self, nfl_ladder):
        once = mp.apply_maker_pilot(_gap(), yes_is_underdog=True)
        assert mp.apply_maker_pilot(once, yes_is_underdog=True) == once

    @pytest.mark.parametrize("kw", [
        {"maker_bid_price": None},            # no actionable rest price
        {"maker_bid_kelly_fraction": None},
        {"maker_bid_kelly_fraction": 0.0},
        {"quarantined": True},
        {"line": -13.5},                       # deep rung stays backend-only
        {"venue": "polymarket_us"},
        {"model_sources": "sharp+sharp_ladder", "full_blend": True},
    ])
    def test_left_unchanged(self, nfl_ladder, kw):
        g = _gap(**kw)
        assert mp.apply_maker_pilot(g, yes_is_underdog=True) == g

    def test_favorite_rung_left_unchanged(self, nfl_ladder):
        g = _gap()
        assert mp.apply_maker_pilot(g, yes_is_underdog=False) == g
        assert mp.apply_maker_pilot(g) == g          # side unknown -> fail closed

    def test_pilot_token_is_non_model_everywhere(self):
        assert "maker_pilot" in ev_mod._NON_MODEL_TOKENS
        assert "maker_pilot" in contamination._NON_MODEL_TOKENS
        assert "maker_pilot" in integrity._NON_MODEL_SOURCE_TOKENS
        assert not contamination.is_contaminated("nfl", "spread", "sharp+spread_pmf+maker_pilot", -3.5)


# --- per-game cap ------------------------------------------------------------

class TestGameCap:
    def _pilot(self, game, kelly, rung):
        return _gap(event_id=f"nfl::2026-10-11::{game}::spread", market_id=f"k-{game}-{rung}",
                    model_sources="sharp+spread_pmf+maker_pilot", maker_only=True,
                    kelly_fraction=0.0, maker_bid_kelly_fraction=kelly, full_blend=True)

    def test_scales_an_over_cap_game_proportionally(self):
        gaps = [self._pilot("a", 0.012, 1), self._pilot("a", 0.006, 2), self._pilot("a", 0.012, 3)]
        out = mp.cap_maker_pilot_game_exposure(gaps, max_game_kelly=0.02)
        total = sum(g.maker_bid_kelly_fraction for g in out)
        assert total <= 0.02 + 1e-12 and total == pytest.approx(0.02, abs=4e-4)
        assert out[0].maker_bid_kelly_fraction == pytest.approx(2 * out[1].maker_bid_kelly_fraction, abs=2e-4)

    def test_leaves_under_cap_games_and_non_pilot_gaps_alone(self):
        other = _gap(model_sources="sharp+sharp_ladder", maker_bid_kelly_fraction=0.05, full_blend=True)
        gaps = [self._pilot("a", 0.012, 1), self._pilot("a", 0.012, 2),
                self._pilot("b", 0.005, 1), other]
        out = mp.cap_maker_pilot_game_exposure(gaps, max_game_kelly=0.02)
        assert [g.market_id for g in out] == [g.market_id for g in gaps]      # order kept
        assert out[2].maker_bid_kelly_fraction == 0.005                       # game b under cap
        assert out[3] is other                                                # not a pilot row

    def test_no_pilot_rows_is_a_no_op(self):
        gaps = [_gap(model_sources="sharp+sharp_ladder")]
        assert mp.cap_maker_pilot_game_exposure(gaps) is gaps


# --- checkpoint verdict --------------------------------------------------------

def _filled(game, clv_pp, price=0.24):
    return {"event_id": f"nfl::2026-10-{game:02d}::g{game}::spread", "kalshi_clv_pct": clv_pp,
            "placed_price": price, "venue": "kalshi"}


class TestCheckpointVerdict:
    def test_no_fills(self):
        assert mp.checkpoint_verdict([])["verdict"] == "COLLECTING"

    def test_collecting_below_checkpoint(self):
        v = mp.checkpoint_verdict([_filled(g, 2.0) for g in range(1, 10)])
        assert v["verdict"] == "COLLECTING" and v["games"] == 9

    def test_stop_on_negative_net_mean(self):
        # +0.2pp gross is negative once the ~0.32pp maker fee at 24c is taken off.
        v = mp.checkpoint_verdict([_filled(g, 0.2) for g in range(1, 16)])
        assert v["mean_net_pp"] < 0 and v["verdict"] == "STOP"

    def test_net_subtracts_the_maker_fee(self):
        v = mp.checkpoint_verdict([_filled(1, 1.0)])
        assert v["mean_net_pp"] == pytest.approx(1.0 - venue_fee_prob("kalshi", 0.24, maker=True) * 100, abs=1e-3)

    def test_step_up_and_hold(self):
        up = mp.checkpoint_verdict([_filled(g, 2.0) for g in range(1, 16)])
        assert up["verdict"] == "STEP-UP"
        # positive mean but only 1/3 of rows positive -> hold
        rows = [_filled(g, 6.0 if g % 3 == 0 else -0.5) for g in range(1, 16)]
        hold = mp.checkpoint_verdict(rows)
        assert hold["mean_net_pp"] > 0 and hold["frac_positive"] < 0.55 and hold["verdict"] == "HOLD"

    def test_full_size_at_gate(self):
        v = mp.checkpoint_verdict([_filled(g, 2.0) for g in range(1, 31)])
        assert v["verdict"] == "FULL-SIZE"

    def test_mean_is_per_game_not_per_row(self):
        # Game 1 has 9 rungs at +5, games 2-15 one rung each at -1 (net of fee).
        rows = [_filled(1, 5.0) for _ in range(9)] + [_filled(g, -0.68) for g in range(2, 16)]
        v = mp.checkpoint_verdict(rows)
        assert v["games"] == 15 and v["mean_net_pp"] < 0     # rung-weighted mean would be positive
        assert v["verdict"] == "STOP"


# --- wiring through EVGapAgent ---------------------------------------------------

def _main_line(cover_prob=0.50):
    """Pinnacle main line: patriots (outcome_a) -3.0. No alt rung at the market's line."""
    return SharpOdds(
        event_id="nfl::2026-11-15::pats_vs_hawks::spread",
        book=SharpBook.pinnacle, sector="nfl",
        outcome_a_label="patriots", outcome_b_label="seahawks",
        outcome_a_decimal=1.0 / cover_prob, outcome_b_decimal=1.0 / (1 - cover_prob),
        true_prob_a=cover_prob, true_prob_b=round(1 - cover_prob, 6),
        spread_line=-3.0, is_alternate=False, margin=0.04,
    )


def _market(yes_team, line, yes=0.05, no=0.97, source=MarketSource.kalshi):
    return PredictionMarket(
        id="k1", source=source, sector="nfl", market_type=MarketType.spread,
        yes_price=yes, no_price=no, team_home="patriots", team_away="seahawks",
        yes_team=yes_team, line=line,
    )


def _eval(market):
    return EVGapAgent()._evaluate_pair(
        market=market, sharp=_main_line(), confidence=95.0, sector="nfl",
        blended_preds={}, injuries={}, model_sources={}, kelly_base=0.25, steam_events=set())


class TestAgentWiring:
    """Main line: patriots (Pinnacle outcome_a, favorite) -3.0; seahawks the underdog."""

    def test_underdog_lay_rung_is_a_pilot_maker_play(self, nfl_ladder):
        gap = _eval(_market("seahawks", -3.5, yes=0.15, no=0.87))
        assert gap is not None
        assert "spread_pmf" in gap.model_sources and "maker_pilot" in gap.model_sources
        assert gap.maker_only is True and gap.kelly_fraction == 0.0
        assert gap.full_blend is True and gap.maker_bid_price is not None
        assert gap.maker_bid_kelly_fraction > 0

    def test_favorite_lay_rung_stays_hidden(self, nfl_ladder):
        gap = _eval(_market("patriots", -10.5))   # favorite laying extra points
        assert gap is not None and "maker_pilot" not in gap.model_sources
        assert gap.full_blend is False

    def test_pilot_off_restores_the_hidden_backend_row(self, nfl_ladder, monkeypatch):
        monkeypatch.setattr(mp, "MAKER_PILOT_SECTORS", set())
        gap = _eval(_market("seahawks", -3.5, yes=0.15, no=0.87))
        assert gap is not None and "maker_pilot" not in gap.model_sources
        assert gap.full_blend is False

    def test_polymarket_rung_stays_hidden(self, nfl_ladder):
        gap = _eval(_market("seahawks", -3.5, yes=0.15, no=0.87, source=MarketSource.polymarket_us))
        assert gap is not None and gap.full_blend is False
        assert "maker_pilot" not in gap.model_sources

    def _no_side(self, yes_team, line, yes, no, yes_is_outcome_b):
        return EVGapAgent()._build_no_side_spread_gap(
            _market(yes_team, line, yes=yes, no=no), _main_line(),
            {"blended_prob_yes": 0.05, "sharp_true_prob_yes": 0.05,
             "src": "sharp+spread_pmf", "yes_is_outcome_b": yes_is_outcome_b},
            95.0, "nfl")

    def test_no_side_underdog_take_is_a_pilot_play(self, nfl_ladder):
        # NO on "patriots win by 11+" = seahawks +10.5. Coherent book: NO ask 80c,
        # NO bid 78c (= 1 - YES ask 22c) -> rest at 79c.
        gap = self._no_side("patriots", -10.5, yes=0.22, no=0.80, yes_is_outcome_b=False)
        assert gap is not None and gap.yes_team == "seahawks" and gap.line == pytest.approx(10.5)
        assert "maker_pilot" in gap.model_sources and gap.maker_only is True
        assert gap.full_blend is True and gap.kelly_fraction == 0.0

    def test_no_side_favorite_take_stays_hidden(self, nfl_ladder):
        # NO on "seahawks win by 4+" = patriots +3.5: the favorite getting points.
        gap = self._no_side("seahawks", -3.5, yes=0.22, no=0.80, yes_is_outcome_b=True)
        assert gap is not None and gap.yes_team == "patriots"
        assert "maker_pilot" not in gap.model_sources and gap.full_blend is False

    def test_coordinator_cap_is_wired(self):
        from evmax.agents import coordinator
        assert coordinator.cap_maker_pilot_game_exposure is mp.cap_maker_pilot_game_exposure


def test_dataclass_replace_keeps_unrelated_fields(nfl_ladder):
    g = _gap(alt_venue="polymarket_us", alt_venue_price=0.26)
    out = mp.apply_maker_pilot(g, yes_is_underdog=True)
    assert dataclasses.replace(out, maker_only=g.maker_only, kelly_fraction=g.kelly_fraction,
                               maker_bid_kelly_fraction=g.maker_bid_kelly_fraction,
                               model_sources=g.model_sources, full_blend=g.full_blend) == g
