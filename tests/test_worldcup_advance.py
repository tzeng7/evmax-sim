"""World Cup knockout "to advance" coverage (KXWCADVANCE).

Covers the full path added 2026-07-05:
  - Kalshi: KXWCADVANCE ticker parsing → MarketType.advance
  - Matching: ::advance event-key suffix + advance/regulation pool isolation
  - Pinnacle: the ::advance SharpOdds derived from the live regulation 3-way
    (Pinnacle's "To Reach X" specials cut off at the team's PREVIOUS kickoff
    and never reopen, so there is no live per-match advance special)
  - EV gap: derive_advance_prob math + advance/regulation type guards +
    model derivation from the base 3-way blend
  - Resolver: advance resolves on the ESPN winner flag (incl. ET/pens), and
    the regulation-market fix — STATUS_FINAL_AET/_PEN means the 90' result
    was a draw regardless of the ET-inclusive final score.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from evmax.agents.models.ensemble_agent import BlendedPrediction
from evmax.agents.odds.ev_gap_agent import EVGapAgent, derive_advance_prob
from evmax.clients.esports_pinnacle import PinnacleGuestClient
from evmax.clients.kalshi import KalshiClient
from evmax.matching.engine import MatchingEngine
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


KICKOFF = datetime(2026, 7, 9, 20, 0, tzinfo=timezone.utc)


def make_market(
    market_type: MarketType = MarketType.advance,
    yes_team: str = "morocco",
    yes_price: float = 0.24,
) -> PredictionMarket:
    return PredictionMarket(
        id="kalshi:KXWCADVANCE-26JUL09FRAMAR-MAR",
        source=MarketSource.kalshi,
        sector="worldcup",
        market_type=market_type,
        title="France vs Morocco: To Advance",
        ticker="KXWCADVANCE-26JUL09FRAMAR-MAR",
        yes_price=yes_price,
        no_price=round(1.0 - yes_price + 0.02, 4),
        volume_usd=250_000.0,
        open_interest_usd=50_000.0,
        team_home="mar",
        team_away="fra",
        yes_team=yes_team,
        event_date=KICKOFF,
    )


def make_sharp(
    event_id: str = "worldcup::2026-07-09::france_vs_morocco::advance",
    prob_a: float = 0.78,
    prob_b: float = 0.22,
    team_a: str = "France",
    team_b: str = "Morocco",
    prob_draw: float | None = None,
) -> SharpOdds:
    return SharpOdds(
        event_id=event_id,
        book=SharpBook.pinnacle,
        sector="worldcup",
        outcome_a_label=team_a,
        outcome_b_label=team_b,
        outcome_a_decimal=1.0 / prob_a,
        outcome_b_decimal=1.0 / prob_b,
        true_prob_a=prob_a,
        true_prob_b=prob_b,
        true_prob_draw=prob_draw,
        margin=0.05,
        event_date=KICKOFF,
    )


# ---------------------------------------------------------------------------
# Kalshi parsing
# ---------------------------------------------------------------------------
class TestKalshiAdvanceParsing:
    def _client(self) -> KalshiClient:
        return KalshiClient.__new__(KalshiClient)

    def test_ticker_teams(self):
        home, away = self._client()._extract_teams_from_ticker(
            "KXWCADVANCE-26JUL09FRAMAR-MAR", "worldcup"
        )
        assert (home, away) == ("mar", "fra")

    def test_parse_market_is_advance_type(self):
        raw = {
            "ticker": "KXWCADVANCE-26JUL09FRAMAR-MAR",
            "title": "France vs Morocco: To Advance",
            "yes_ask": 24,
            "yes_bid": 23,
            "no_ask": 77,
            "no_bid": 76,
            "volume": 290_000,
            "open_interest": 280_000,
            "status": "active",
        }
        market = self._client()._parse_market(raw, "worldcup")
        assert market is not None
        assert market.market_type == MarketType.advance
        assert market.line is None
        assert market.team_home == "mar"
        assert market.team_away == "fra"
        assert (market.event_date.year, market.event_date.month, market.event_date.day) == (
            2026, 7, 9,
        )

    def test_game_series_still_moneyline(self):
        raw = {
            "ticker": "KXWCGAME-26JUL09FRAMAR-MAR",
            "title": "France vs Morocco Winner?",
            "yes_ask": 20,
            "yes_bid": 19,
            "no_ask": 81,
            "no_bid": 80,
            "volume": 100_000,
            "open_interest": 90_000,
            "status": "active",
        }
        market = self._client()._parse_market(raw, "worldcup")
        assert market is not None
        assert market.market_type == MarketType.moneyline


# ---------------------------------------------------------------------------
# Matching: key suffix + pool isolation
# ---------------------------------------------------------------------------
class TestAdvanceMatching:
    def test_advance_key_has_suffix(self):
        engine = MatchingEngine()
        key = engine.build_market_key(make_market())
        assert key is not None
        assert key.endswith("::advance")

    def test_advance_market_only_matches_advance_record(self):
        engine = MatchingEngine()
        ml = make_sharp(event_id="worldcup::2026-07-09::france_vs_morocco",
                        prob_a=0.55, prob_b=0.20, prob_draw=0.25)
        adv = make_sharp()
        result = engine.match(make_market(), [ml, adv])
        assert result is not None
        assert result[0].event_id.endswith("::advance")

    def test_advance_market_never_matches_regulation_record(self):
        # Only the 3-way regulation record exists — advance market must NOT
        # fuzzy-match it (the probs price different events).
        engine = MatchingEngine()
        ml = make_sharp(event_id="worldcup::2026-07-09::france_vs_morocco",
                        prob_a=0.55, prob_b=0.20, prob_draw=0.25)
        assert engine.match(make_market(), [ml]) is None

    def test_regulation_market_never_matches_advance_record(self):
        engine = MatchingEngine()
        adv = make_sharp()
        ml_market = make_market(market_type=MarketType.moneyline)
        assert engine.match(ml_market, [adv]) is None


# ---------------------------------------------------------------------------
# Pinnacle: advance record derived from the regulation 3-way
# ---------------------------------------------------------------------------
class TestDeriveAdvanceOdds:
    def test_derived_record_shape(self):
        ml = make_sharp(
            event_id="worldcup::2026-07-09::france_vs_morocco",
            prob_a=0.55, prob_b=0.20, prob_draw=0.25,
        )
        adv = PinnacleGuestClient.derive_advance_odds(ml)
        assert adv is not None
        assert adv.event_id == "worldcup::2026-07-09::france_vs_morocco::advance"
        assert adv.outcome_a_label == "France"
        assert adv.outcome_b_label == "Morocco"
        # r = 0.55/0.75 = 0.7333; 0.55 + 0.25*(0.55*0.7333 + 0.45*0.5) = 0.7071
        assert adv.true_prob_a == pytest.approx(0.7071, abs=0.001)
        assert adv.true_prob_a + adv.true_prob_b == pytest.approx(1.0)
        assert adv.true_prob_draw is None

    def test_advance_prob_dominates_regulation(self):
        # Advancing includes the ET/pens path, so P(advance) > P(win in 90')
        # for BOTH sides whenever a draw is possible.
        ml = make_sharp(
            event_id="worldcup::2026-07-09::france_vs_morocco",
            prob_a=0.55, prob_b=0.20, prob_draw=0.25,
        )
        adv = PinnacleGuestClient.derive_advance_odds(ml)
        assert adv.true_prob_a > ml.true_prob_a
        assert adv.true_prob_b > ml.true_prob_b

    def test_no_draw_passthrough(self):
        # Degenerate 2-way input: advance prob equals the win prob.
        ml = make_sharp(
            event_id="worldcup::2026-07-09::france_vs_morocco",
            prob_a=0.78, prob_b=0.22, prob_draw=None,
        )
        adv = PinnacleGuestClient.derive_advance_odds(ml)
        assert adv is not None
        assert adv.true_prob_a == pytest.approx(0.78)


# ---------------------------------------------------------------------------
# EV gap: derivation math + type guards
# ---------------------------------------------------------------------------
class TestDeriveAdvanceProb:
    def test_draw_share_blends_et_edge_with_pens_coinflip(self):
        # r = 0.5/0.8 = 0.625; 0.5 + 0.2*(0.55*0.625 + 0.45*0.5) = 0.61375
        assert derive_advance_prob(0.5, 0.3, 0.2) == pytest.approx(0.61375)

    def test_probs_complement(self):
        a = derive_advance_prob(0.5, 0.3, 0.2)
        b = derive_advance_prob(0.3, 0.5, 0.2)
        assert a + b == pytest.approx(1.0)

    def test_no_draw_passthrough(self):
        assert derive_advance_prob(0.6, 0.4, None) == pytest.approx(0.6)

    def test_degenerate_inputs(self):
        assert derive_advance_prob(0.0, 0.0, 1.0) is None
        assert derive_advance_prob(None, 0.4, 0.2) is None


class TestEvGapAdvance:
    @pytest.fixture(autouse=True)
    def _enable_worldcup(self):
        # #185 disabled worldcup in production config, and _evaluate_pair returns
        # None for a disabled sector's market_type. These tests exercise the
        # advance-derivation LOGIC, which is independent of the deployment on/off
        # switch — force worldcup back on for the duration. The two rejects-tests
        # still return None via the advance/regulation mismatch guard.
        from evmax.modes import clear_runtime_overrides, set_runtime_overrides

        set_runtime_overrides({"worldcup": "shadow"})
        yield
        clear_runtime_overrides()

    def _blend(self, p_a=0.55, p_b=0.20, p_draw=0.25) -> BlendedPrediction:
        return BlendedPrediction(
            event_id="worldcup::2026-07-09::france_vs_morocco",
            true_prob_a=p_a,
            true_prob_b=p_b,
            true_prob_draw=p_draw,
            confidence=0.7,
            model_sources="elo+poisson+sharp",
            per_model={},
        )

    def test_advance_market_rejects_regulation_sharp(self):
        agent = EVGapAgent()
        ml_sharp = make_sharp(event_id="worldcup::2026-07-09::france_vs_morocco",
                              prob_a=0.55, prob_b=0.20, prob_draw=0.25)
        gap = agent._evaluate_pair(
            make_market(), ml_sharp, 95.0, "worldcup", {}, {},
        )
        assert gap is None

    def test_regulation_market_rejects_advance_sharp(self):
        agent = EVGapAgent()
        gap = agent._evaluate_pair(
            make_market(market_type=MarketType.moneyline, yes_price=0.15),
            make_sharp(), 95.0, "worldcup", {}, {},
        )
        assert gap is None

    def test_advance_derives_model_prob_from_base_blend(self):
        agent = EVGapAgent()
        # Sharp anchor: Morocco advances 22%. Model 3-way blend implies
        # r = 0.20/0.75 = 0.2667 → 0.20 + 0.25*(0.55*0.2667 + 0.45*0.5) =
        # 0.2929 — Kalshi asks 24¢, so the derived blend gives a positive gap.
        blended = {"worldcup::2026-07-09::france_vs_morocco": self._blend()}
        gap = agent._evaluate_pair(
            make_market(), make_sharp(), 95.0, "worldcup", blended, {},
            model_sources={},
        )
        assert gap is not None
        assert gap.market_type == "advance"
        assert gap.sharp_true_prob == pytest.approx(0.22)
        assert gap.blended_true_prob == pytest.approx(0.2929, abs=0.001)
        assert "advance_derived" in gap.model_sources

    def test_advance_without_base_blend_falls_back_to_sharp(self):
        agent = EVGapAgent()
        gap = agent._evaluate_pair(
            make_market(yes_price=0.15), make_sharp(), 95.0, "worldcup", {}, {},
            model_sources={},
        )
        assert gap is not None
        assert gap.blended_true_prob == pytest.approx(0.22)
        assert "advance_derived" not in gap.model_sources

    def test_advance_drift_capped_to_sharp(self):
        agent = EVGapAgent()
        # Base blend wildly off (Morocco 0.60 + draw share → derived ~0.79 vs
        # sharp 0.22 → ratio > 2) — must fall back to the sharp anchor.
        blended = {
            "worldcup::2026-07-09::france_vs_morocco": self._blend(
                p_a=0.10, p_b=0.60, p_draw=0.30,
            ),
        }
        gap = agent._evaluate_pair(
            make_market(yes_price=0.15), make_sharp(), 95.0, "worldcup", blended, {},
            model_sources={},
        )
        assert gap is not None
        assert gap.blended_true_prob == pytest.approx(0.22)
        assert gap.model_sources == "sharp(capped)"


# ---------------------------------------------------------------------------
# Resolver: advance + regulation on ET/pens knockout games
# ---------------------------------------------------------------------------
def _score(
    home="Australia", away="Egypt", hs=1, as_=1,
    home_winner=False, away_winner=True, went_extra_time=True,
) -> dict:
    return {
        "home_name": home,
        "away_name": away,
        "home_score": hs,
        "away_score": as_,
        "home_won": hs > as_,
        "home_winner": home_winner,
        "away_winner": away_winner,
        "went_extra_time": went_extra_time,
        "game_date": "2026-07-02",
    }


def _pred(yes_team: str, market_type: str) -> dict:
    # Advance rows carry the ::advance event-key suffix in production —
    # _slug_teams reads the teams segment (parts[2]) so the suffix is inert.
    suffix = "::advance" if market_type == "advance" else ""
    return {
        "event_id": f"worldcup::2026-07-02::australia_vs_egypt{suffix}",
        "yes_team": yes_team,
        "market_type": market_type,
        "event_date": "2026-07-02",
        "line": None,
    }


class TestResolverAdvance:
    def test_shootout_winner_advances(self):
        from evmax.agents.cleanup.resolver import _match_espn
        # Australia 1-1 Egypt, Egypt won on pens (winner flag set).
        assert _match_espn(_pred("egypt", "advance"), [_score()]) == 1
        assert _match_espn(_pred("australia", "advance"), [_score()]) == 0

    def test_regular_time_winner_advances(self):
        from evmax.agents.cleanup.resolver import _match_espn
        score = _score(hs=2, as_=0, home_winner=True, away_winner=False,
                       went_extra_time=False)
        assert _match_espn(_pred("australia", "advance"), [score]) == 1
        assert _match_espn(_pred("egypt", "advance"), [score]) == 0

    def test_missing_winner_flags_fall_back_to_score(self):
        from evmax.agents.cleanup.resolver import _match_espn
        score = _score(hs=3, as_=1, home_winner=None, away_winner=None,
                       went_extra_time=False)
        assert _match_espn(_pred("australia", "advance"), [score]) == 1

    def test_missing_flags_and_level_score_unresolvable(self):
        from evmax.agents.cleanup.resolver import _match_espn
        score = _score(home_winner=None, away_winner=None)
        assert _match_espn(_pred("egypt", "advance"), [score]) is None


class TestResolverRegulationExtraTime:
    """Kalshi's soccer game markets settle on the 90' result — a knockout game
    that went to AET/pens was drawn in regulation regardless of the ESPN
    (ET-inclusive) final score. Regression: Belgium 3-2 AET vs Senegal would
    previously have resolved 'Belgium ML' YES."""

    def test_aet_game_is_regulation_draw(self):
        from evmax.agents.cleanup.resolver import _match_espn
        score = _score(home="Belgium", away="Senegal", hs=3, as_=2,
                       home_winner=True, away_winner=False, went_extra_time=True)
        pred = {
            "event_id": "worldcup::2026-07-01::belgium_vs_senegal",
            "yes_team": "belgium", "market_type": "moneyline",
            "event_date": "2026-07-01", "line": None,
        }
        tie_pred = dict(pred, yes_team="tie")
        assert _match_espn(pred, [score]) == 0
        assert _match_espn(dict(pred, yes_team="senegal"), [score]) == 0
        assert _match_espn(tie_pred, [score]) == 1

    def test_regulation_win_unaffected(self):
        from evmax.agents.cleanup.resolver import _match_espn
        score = _score(home="Colombia", away="Ghana", hs=1, as_=0,
                       home_winner=True, away_winner=False, went_extra_time=False)
        pred = {
            "event_id": "worldcup::2026-07-02::colombia_vs_ghana",
            "yes_team": "colombia", "market_type": "moneyline",
            "event_date": "2026-07-02", "line": None,
        }
        assert _match_espn(pred, [score]) == 1
        assert _match_espn(dict(pred, yes_team="tie"), [score]) == 0

    def test_advance_market_still_resolves_on_aet_game(self):
        from evmax.agents.cleanup.resolver import _match_espn
        score = _score(home="Belgium", away="Senegal", hs=3, as_=2,
                       home_winner=True, away_winner=False, went_extra_time=True)
        pred = {
            "event_id": "worldcup::2026-07-01::belgium_vs_senegal",
            "yes_team": "belgium", "market_type": "advance",
            "event_date": "2026-07-01", "line": None,
        }
        assert _match_espn(pred, [score]) == 1


class TestFetchEspnKnockoutFields:
    def test_aet_and_pen_statuses_flagged(self):
        from evmax.agents.cleanup.resolver import _fetch_espn_scores

        def _event(status_name, home_winner, away_winner, hs, as_, shootout=None):
            return {
                "competitions": [{
                    "status": {"type": {"completed": True, "name": status_name}},
                    "competitors": [
                        {"homeAway": "home", "score": str(hs), "winner": home_winner,
                         "team": {"displayName": "Australia"}},
                        {"homeAway": "away", "score": str(as_), "winner": away_winner,
                         "team": {"displayName": "Egypt"}},
                    ],
                }],
            }

        payload = {"events": [
            _event("STATUS_FULL_TIME", True, False, 2, 1),
            _event("STATUS_FINAL_AET", True, False, 3, 2),
            _event("STATUS_FINAL_PEN", False, True, 1, 1),
        ]}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        client = AsyncMock()
        client.get = AsyncMock(return_value=_Resp())
        scores = asyncio.run(
            _fetch_espn_scores(client, "soccer", "fifa.world", "20260702")
        )
        assert [s["went_extra_time"] for s in scores] == [False, True, True]
        assert scores[2]["away_winner"] is True
        assert scores[2]["home_winner"] is False
