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
        monkeypatch.setattr(ev_mod, "SPREAD_LADDER_ENABLED", True)
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
        # to -16.5 and (per the documented tail bias) overstates the cover prob.
        monkeypatch.setattr(ev_mod, "SPREAD_LADDER_ENABLED", False)
        agent = EVGapAgent()
        main = self._rung("nfl", -3.0, 0.50, is_alt=False)  # pick'em-ish favorite at -3
        gap = agent._evaluate_pair(
            market=self._market(-16.5), sharp=main, confidence=95.0, sector="nfl",
            blended_preds={}, injuries={}, model_sources={}, kelly_base=0.25, steam_events=set(),
        )
        assert gap is not None
        assert "spread_dist" in gap.model_sources        # CDF path, not the ladder
        # the extrapolated cover prob is materially above the book's 0.077 rung —
        # this gap between the two paths IS the phantom-EV the ladder removes.
        assert gap.blended_true_prob > 0.077
