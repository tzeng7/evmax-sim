"""Golden fixtures for the alt-spread ladder (evmax/models_ml/spread_distribution
SPREAD_LADDER_ENABLED). Validates the parse → match → price loop across sectors
on synthetic Pinnacle `markets/straight` payloads shaped exactly like the live
one (matchupId / type / period / status / isAlternate / prices[designation,
points, price] — the fields _parse_spread already reads).

The sign/convention golden table (which team covers, at what line) is the risk
CLAUDE.md named when it declined this feature; these tests pin it. Everything is
inert while the flag is False (asserted), so the flag-off suites remain the
regression guard.
"""
from __future__ import annotations

import asyncio

import pytest

import evmax.models_ml.spread_distribution as sd
import evmax.agents.odds.ev_gap_agent as ev_mod
from evmax.agents.odds.ev_gap_agent import EVGapAgent
from evmax.clients.esports_pinnacle import PinnacleGuestClient
from evmax.matching.engine import MatchingEngine
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


@pytest.fixture(autouse=True)
def _global_flag_semantics(monkeypatch):
    """The suites below pin the GLOBAL-flag behaviour, so clear the per-sector
    sets (NFL ships in them). TestNflLadderOnly sets them back explicitly."""
    monkeypatch.setattr(sd, "SPREAD_LADDER_SECTORS", set())
    monkeypatch.setattr(sd, "SPREAD_LADDER_EXACT_SECTORS", set())
    monkeypatch.setattr(sd, "SPREAD_LADDER_ONLY_SECTORS", set())


# --- Pinnacle payload builders (real shape) ---------------------------------

def _spread_market(matchup_id, home_points, home_price, away_price, is_alt):
    return {
        "matchupId": matchup_id, "type": "spread", "period": 0, "status": "open",
        "isAlternate": is_alt,
        "prices": [
            {"designation": "home", "points": home_points, "price": home_price},
            {"designation": "away", "points": -home_points, "price": away_price},
        ],
    }


def _matchup(matchup_id, home, away):
    return {
        "id": matchup_id,
        "participants": [
            {"name": home, "alignment": "home"},
            {"name": away, "alignment": "away"},
        ],
        "startTime": "2026-11-15T18:00:00Z",
    }


# Per-sector fixture: (sector, home, away, main_pts, [alt_pts...]). Home is the
# favorite (negative points). Prices are illustrative American odds.
SECTORS = [
    ("nfl", "New England Patriots", "Seattle Seahawks", -3.0, [-7.0, -16.5]),
    ("nba", "Boston Celtics", "Brooklyn Nets", -6.5, [-10.5, -14.5]),
    ("ncaab", "Duke Blue Devils", "Elon Phoenix", -12.5, [-18.5, -24.5]),
]


def _payload(matchup_id, main_pts, alt_pts):
    mkts = [_spread_market(matchup_id, main_pts, -110, -110, False)]
    for i, pts in enumerate(alt_pts):
        # deeper favorite line → longer + price on the favorite, shorter on dog
        mkts.append(_spread_market(matchup_id, pts, 180 + 120 * i, -240 - 200 * i, True))
    return mkts


# --- Phase 1: client emits + tags rungs, correct sign --------------------------

class TestClientEmitsLadder:
    @pytest.mark.parametrize("sector,home,away,main_pts,alt_pts", SECTORS)
    def test_flag_on_emits_all_rungs_with_sign(self, sector, home, away, main_pts, alt_pts, monkeypatch):
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", True)
        client = PinnacleGuestClient()
        payload = _payload("m1", main_pts, alt_pts)
        out = asyncio.run(client._fetch_matchup_odds(_matchup("m1", home, away), sector, markets_override=payload))
        spreads = [o for o in (out or []) if o.spread_line is not None]

        # one record per rung (main + alternates)
        assert len(spreads) == 1 + len(alt_pts)
        main = [s for s in spreads if not s.is_alternate]
        alts = [s for s in spreads if s.is_alternate]
        assert len(main) == 1 and len(alts) == len(alt_pts)

        # sign/convention: the home favorite (negative points) is the covering
        # side (outcome_a); the line carries its negative value.
        m = main[0]
        assert m.event_id.endswith("::spread")           # bare id (back-compat)
        assert m.spread_line == pytest.approx(main_pts)  # negative
        assert m.true_prob_a > 0 and m.true_prob_a < 1
        for a in alts:
            assert a.event_id.endswith(f"::spread::{a.spread_line}")
            assert a.spread_line in [pytest.approx(p) for p in alt_pts]
            # deeper favorite line → lower cover prob
        # monotonic: cover prob falls as the favorite line deepens
        by_line = sorted(spreads, key=lambda s: s.spread_line)  # most-negative first
        probs = [s.true_prob_a for s in by_line]
        assert probs == sorted(probs), "P(favorite covers) must rise as the line eases"

    def test_flag_off_emits_only_main_line(self, monkeypatch):
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", False)
        client = PinnacleGuestClient()
        payload = _payload("m1", -3.0, [-7.0, -16.5])
        out = asyncio.run(client._fetch_matchup_odds(_matchup("m1", "New England Patriots", "Seattle Seahawks"), "nfl", markets_override=payload))
        spreads = [o for o in (out or []) if o.spread_line is not None]
        assert len(spreads) == 1 and not spreads[0].is_alternate


class TestCaptureOnlyOverride:
    """The capture-only ``include_alternate_spreads`` override emits the ladder
    even while the global SPREAD_LADDER_ENABLED flag is OFF, so watch-listings
    can archive the book's rungs without changing live pricing."""

    def test_override_emits_ladder_with_global_flag_off(self, monkeypatch):
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", False)
        client = PinnacleGuestClient()
        payload = _payload("m1", -3.0, [-7.0, -16.5])
        out = asyncio.run(client._fetch_matchup_odds(
            _matchup("m1", "New England Patriots", "Seattle Seahawks"),
            "nfl", markets_override=payload, include_alternate_spreads=True,
        ))
        spreads = [o for o in (out or []) if o.spread_line is not None]
        assert len(spreads) == 3, "override must emit main + both alt rungs"
        alts = [s for s in spreads if s.is_alternate]
        assert len(alts) == 2
        for a in alts:
            assert a.event_id.endswith(f"::spread::{a.spread_line}")

    def test_override_default_off_leaves_main_only(self, monkeypatch):
        # Global flag off AND no override → main line only (live-pricing path).
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", False)
        client = PinnacleGuestClient()
        payload = _payload("m1", -3.0, [-7.0, -16.5])
        out = asyncio.run(client._fetch_matchup_odds(
            _matchup("m1", "New England Patriots", "Seattle Seahawks"),
            "nfl", markets_override=payload,
        ))
        spreads = [o for o in (out or []) if o.spread_line is not None]
        assert len(spreads) == 1 and not spreads[0].is_alternate


# --- Phase 2a: matcher prefers the exact-line rung -----------------------------

class TestMatcherNearestSpread:
    def _sharps(self, sector, home, away, main_pts, alt_pts):
        """Build the sharp rung list with event_ids the matcher will reconstruct."""
        eng = MatchingEngine()
        norm = eng._get_normalizer(sector)
        # event_date → same day the matcher derives
        from datetime import datetime, timezone
        ed = datetime(2026, 11, 15, 18, 0, tzinfo=timezone.utc)
        from evmax.clients.time_util import kalshi_game_day
        base = norm.normalize_event_key(home, away, kalshi_game_day(ed, sector), sector)
        recs = []
        for pts, is_alt in [(main_pts, False)] + [(p, True) for p in alt_pts]:
            eid = f"{base}::spread::{pts}" if is_alt else f"{base}::spread"
            recs.append(SharpOdds(
                event_id=eid, book=SharpBook.pinnacle, sector=sector,
                outcome_a_label=home, outcome_b_label=away,
                outcome_a_decimal=2.0, outcome_b_decimal=2.0,
                true_prob_a=0.5, true_prob_b=0.5, spread_line=pts, is_alternate=is_alt,
            ))
        return recs, ed

    @pytest.mark.parametrize("sector,home,away,main_pts,alt_pts", SECTORS)
    def test_flag_on_matches_exact_line_rung(self, sector, home, away, main_pts, alt_pts, monkeypatch):
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", True)
        recs, ed = self._sharps(sector, home, away, main_pts, alt_pts)
        target = alt_pts[-1]  # deepest rung
        eng = MatchingEngine()
        norm = eng._get_normalizer(sector)
        market = PredictionMarket(
            id="k1", source=MarketSource.kalshi, sector=sector,
            market_type=MarketType.spread, yes_price=0.1, no_price=0.92,
            team_home=home, team_away=away,
            yes_team=home, line=target, event_date=ed,
        )
        res = eng.match(market, recs)
        assert res is not None
        so, conf = res
        assert so.spread_line == pytest.approx(target)   # the deep rung, not the main line
        assert so.is_alternate is True
        assert conf >= 90.0

    def test_flag_off_matches_main_line(self, monkeypatch):
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", False)
        recs, ed = self._sharps("nfl", "New England Patriots", "Seattle Seahawks", -3.0, [-16.5])
        eng = MatchingEngine()
        norm = eng._get_normalizer("nfl")
        market = PredictionMarket(
            id="k1", source=MarketSource.kalshi, sector="nfl",
            market_type=MarketType.spread, yes_price=0.05, no_price=0.97,
            team_home="New England Patriots", team_away="Seattle Seahawks",
            yes_team="New England Patriots", line=-16.5, event_date=ed,
        )
        res = eng.match(market, recs)
        # only the main-line exact match applies; the -16.5 rung is never picked
        assert res is not None and res[0].spread_line == pytest.approx(-3.0)


# --- Phase 2b: ev_gap prices off the rung (ladder) vs extrapolates (CDF) --------

class TestEvGapLadderPricing:
    def _rung(self, sector, spread_line, cover_prob, is_alt):
        return SharpOdds(
            event_id=f"{sector}::2026-11-15::pats_vs_hawks::spread" + (f"::{spread_line}" if is_alt else ""),
            book=SharpBook.pinnacle, sector=sector,
            outcome_a_label="patriots", outcome_b_label="seahawks",
            outcome_a_decimal=1.0 / cover_prob, outcome_b_decimal=1.0 / (1 - cover_prob),
            true_prob_a=cover_prob, true_prob_b=round(1 - cover_prob, 6),
            spread_line=spread_line, is_alternate=is_alt, margin=0.04,
        )

    def _market(self, line):
        return PredictionMarket(
            id="k1", source=MarketSource.kalshi, sector="nfl",
            market_type=MarketType.spread, yes_price=0.05, no_price=0.97,
            team_home="patriots", team_away="seahawks", yes_team="patriots", line=line,
        )

    def test_ladder_uses_book_devig_not_cdf(self, monkeypatch):
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", True)
        agent = EVGapAgent()
        # matched record IS the -16.5 rung; its devigged cover prob is 0.077.
        rung = self._rung("nfl", -16.5, 0.077, is_alt=True)
        gap = agent._evaluate_pair(
            market=self._market(-16.5), sharp=rung, confidence=95.0, sector="nfl",
            blended_preds={}, injuries={}, model_sources={}, kelly_base=0.25, steam_events=set(),
        )
        assert gap is not None
        assert gap.blended_true_prob == pytest.approx(0.077, abs=0.002)  # book's price, no CDF
        assert "sharp_ladder" in gap.model_sources
        assert "spread_dist" not in gap.model_sources

    def test_cdf_extrapolation_overstates_vs_ladder(self, monkeypatch):
        # Flag off: the matched record is the MAIN line (-3); the CDF extrapolates
        # to -16.5. NFL prices with the key-number PMF by default (_PMF_SECTORS),
        # so disable it here to pin the normal-CDF path this test is about.
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", False)
        monkeypatch.setattr(sd, "_PMF_SECTORS", set())
        agent = EVGapAgent()
        main = self._rung("nfl", -3.0, 0.50, is_alt=False)  # pick'em-ish favorite at -3
        gap = agent._evaluate_pair(
            market=self._market(-16.5), sharp=main, confidence=95.0, sector="nfl",
            blended_preds={}, injuries={}, model_sources={}, kelly_base=0.25, steam_events=set(),
        )
        assert gap is not None
        assert "spread_dist" in gap.model_sources        # CDF path, not the ladder
        # The extrapolated cover prob differs from the ladder test's synthetic
        # 0.077 rung: the two paths are distinct. (The favorite -16.5 fair off a
        # -3 main is ~0.15-0.16, 2003-25 empirical 0.159; 0.077 is closer to the
        # UNDERDOG winning by 17+.)
        assert gap.blended_true_prob > 0.077


# --- NFL: ladder per sector, exact line, model-only rungs are backend-only ------

@pytest.fixture
def nfl_ladder(monkeypatch):
    monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", False)
    monkeypatch.setattr(sd, "SPREAD_LADDER_SECTORS", {"nfl"})
    monkeypatch.setattr(sd, "SPREAD_LADDER_EXACT_SECTORS", {"nfl"})
    monkeypatch.setattr(sd, "SPREAD_LADDER_ONLY_SECTORS", {"nfl"})


class TestNflLadderOnly:
    def test_helpers_are_per_sector(self, nfl_ladder):
        assert sd.spread_ladder_enabled("nfl") and not sd.spread_ladder_enabled("nba")
        assert sd.spread_ladder_tolerance("nfl") < 0.1
        assert sd.spread_ladder_tolerance("nba") == sd.SPREAD_LADDER_LINE_TOLERANCE
        assert sd.spread_ladder_required("nfl") and not sd.spread_ladder_required("nba")

    def test_required_needs_ladder_enabled(self, monkeypatch):
        monkeypatch.setattr(sd, "SPREAD_LADDER_ENABLED", False)
        monkeypatch.setattr(sd, "SPREAD_LADDER_SECTORS", set())
        monkeypatch.setattr(sd, "SPREAD_LADDER_ONLY_SECTORS", {"nfl"})
        assert not sd.spread_ladder_required("nfl")  # no ladder → nothing to require

    def test_client_emits_nfl_ladder_without_global_flag_but_not_nba(self, nfl_ladder):
        client = PinnacleGuestClient()
        out = asyncio.run(client._fetch_matchup_odds(
            _matchup("m1", "New England Patriots", "Seattle Seahawks"), "nfl",
            markets_override=_payload("m1", -3.0, [-7.0, -16.5])))
        assert len([o for o in out if o.spread_line is not None]) == 3
        out = asyncio.run(client._fetch_matchup_odds(
            _matchup("m2", "Boston Celtics", "Brooklyn Nets"), "nba",
            markets_override=_payload("m2", -6.5, [-10.5])))
        assert len([o for o in out if o.spread_line is not None]) == 1

    def test_nfl_matcher_needs_the_same_line(self, nfl_ladder):
        helper = TestMatcherNearestSpread()
        recs, ed = helper._sharps("nfl", "New England Patriots", "Seattle Seahawks", -3.0, [-7.0])

        def match(line):
            m = PredictionMarket(
                id="k1", source=MarketSource.kalshi, sector="nfl",
                market_type=MarketType.spread, yes_price=0.2, no_price=0.82,
                team_home="New England Patriots", team_away="Seattle Seahawks",
                yes_team="New England Patriots", line=line, event_date=ed)
            return MatchingEngine().match(m, recs)

        res = match(-7.0)
        assert res and res[0].spread_line == pytest.approx(-7.0)   # same line → rung
        res = match(-7.5)   # Pinnacle has -7.0 (half a point away, NOT the same line)
        assert res and res[0].spread_line == pytest.approx(-3.0)   # falls back to main

    @pytest.mark.parametrize("src,market_type,expected", [
        ("sharp+sharp_ladder", "spread", True),
        ("sharp+sharp_ladder+no_side", "spread", True),
        ("sharp+spread_pmf", "spread", False),
        ("sharp+spread_pmf+no_side", "spread", False),
        ("sharp+spread_dist", "spread", False),
        ("elo+form+sharp", "moneyline", True),     # moneyline untouched
        ("sharp", "total", True),                  # totals untouched
        ("sharp+spread_pmf", None, False),         # fail-closed without a market type
    ])
    def test_has_full_blend_requires_ladder_for_nfl_spread(self, nfl_ladder, src, market_type, expected):
        assert ev_mod.has_full_blend("nfl", src, market_type) is expected

    def test_other_sectors_spreads_stay_visible(self, nfl_ladder):
        assert ev_mod.has_full_blend("nba", "sharp+spread_dist", "spread") is True

    def _market(self, line):
        return PredictionMarket(
            id="k1", source=MarketSource.kalshi, sector="nfl",
            market_type=MarketType.spread, yes_price=0.05, no_price=0.97,
            team_home="patriots", team_away="seahawks", yes_team="patriots", line=line,
        )

    def _eval(self, market, sharp):
        return EVGapAgent()._evaluate_pair(
            market=market, sharp=sharp, confidence=95.0, sector="nfl",
            blended_preds={}, injuries={}, model_sources={}, kelly_base=0.25, steam_events=set())

    def test_rung_priced_off_pinnacle_is_a_visible_play(self, nfl_ladder):
        rung = TestEvGapLadderPricing()._rung("nfl", -10.5, 0.20, is_alt=True)
        gap = self._eval(self._market(-10.5), rung)
        assert gap is not None
        assert "sharp_ladder" in gap.model_sources and gap.full_blend is True
        assert gap.blended_true_prob == pytest.approx(0.20, abs=0.002)

    def test_rung_without_a_pinnacle_price_is_model_only_and_hidden(self, nfl_ladder):
        main = TestEvGapLadderPricing()._rung("nfl", -3.0, 0.50, is_alt=False)
        gap = self._eval(self._market(-10.5), main)   # no -10.5 rung → PMF path
        assert gap is not None
        assert "spread_pmf" in gap.model_sources and "sharp_ladder" not in gap.model_sources
        assert gap.full_blend is False                # backend-only: shadow-logged, never displayed

    def test_no_side_gap_inherits_the_hidden_flag(self, nfl_ladder):
        agent = EVGapAgent()
        market = self._market(-10.5)
        market.no_price = 0.80
        sharp = TestEvGapLadderPricing()._rung("nfl", -3.0, 0.50, is_alt=False)

        def build(src):
            return agent._build_no_side_spread_gap(
                market, sharp,
                {"blended_prob_yes": 0.05, "sharp_true_prob_yes": 0.05,
                 "src": src, "yes_is_outcome_b": False},
                95.0, "nfl")

        hidden = build("sharp+spread_pmf")
        shown = build("sharp+sharp_ladder")
        assert hidden is not None and hidden.full_blend is False
        assert shown is not None and shown.full_blend is True


# --- Orientation: a rung prices the YES contract only for the team that LAYS it --

class TestLadderOrientation:
    """Pinnacle's outcome_a is always the team laying points at a rung, so the
    ``-16.5`` rung offers "Patriots -16.5 / Seahawks +16.5". A venue YES contract
    "Seahawks win by over 16.5" (the underdog LAYING 16.5) is NOT on that rung.
    Matching by |line| alone priced it at the +16.5 side's ~55% against a 3c ask
    (the 2026-10-03 dashboard screenshot: +1000-2000% EV)."""

    HOME, AWAY = "New England Patriots", "Seattle Seahawks"

    def _recs(self):
        return TestMatcherNearestSpread()._sharps("nfl", self.HOME, self.AWAY, -3.0, [-16.5])

    def _market(self, yes_team, line):
        _, ed = self._recs()
        return PredictionMarket(
            id="k1", source=MarketSource.kalshi, sector="nfl",
            market_type=MarketType.spread, yes_price=0.05, no_price=0.97,
            team_home=self.HOME, team_away=self.AWAY,
            yes_team=yes_team, line=line, event_date=ed)

    def test_matcher_picks_rung_for_the_laying_team(self, nfl_ladder):
        recs, _ = self._recs()
        res = MatchingEngine().match(self._market(self.HOME, -16.5), recs)
        assert res and res[0].spread_line == pytest.approx(-16.5)

    def test_matcher_skips_rung_when_yes_team_is_the_other_side(self, nfl_ladder):
        recs, _ = self._recs()
        res = MatchingEngine().match(self._market(self.AWAY, -16.5), recs)
        # underdog "wins by over 16.5": no Pinnacle rung → main-line record (PMF path)
        assert res and res[0].spread_line == pytest.approx(-3.0)
        assert res[0].is_alternate is False

    def test_matcher_hits_rung_for_points_getting_side_with_positive_line(self, nfl_ladder):
        # Polymarket US convention: YES = long side at its own signed handicap.
        recs, _ = self._recs()
        res = MatchingEngine().match(self._market(self.AWAY, +16.5), recs)
        assert res and res[0].spread_line == pytest.approx(-16.5)

    def _eval(self, market, rung):
        return EVGapAgent()._evaluate_pair(
            market=market, sharp=rung, confidence=95.0, sector="nfl",
            blended_preds={}, injuries={}, model_sources={}, kelly_base=0.25, steam_events=set())

    def _rung(self):
        r = TestEvGapLadderPricing()._rung("nfl", -16.5, 0.45, is_alt=True)
        return r.model_copy(update={"outcome_a_label": "patriots", "outcome_b_label": "seahawks"})

    def test_underdog_laying_the_line_is_never_priced_off_the_rung(self, nfl_ladder):
        gap = self._eval(self._market("seahawks", -16.5), self._rung())
        assert gap is None or "sharp_ladder" not in gap.model_sources
        assert gap is None or gap.blended_true_prob < 0.2   # not the +16.5 side's 55%

    def test_favorite_laying_the_line_is_priced_off_the_rung(self, nfl_ladder):
        gap = self._eval(self._market("patriots", -16.5), self._rung())
        assert gap is not None and "sharp_ladder" in gap.model_sources
        assert gap.blended_true_prob == pytest.approx(0.45, abs=0.002)

    def test_underdog_getting_points_uses_the_other_side_of_the_rung(self, nfl_ladder):
        gap = self._eval(self._market("seahawks", +16.5), self._rung())
        assert gap is not None and "sharp_ladder" in gap.model_sources
        assert gap.blended_true_prob == pytest.approx(0.55, abs=0.002)
