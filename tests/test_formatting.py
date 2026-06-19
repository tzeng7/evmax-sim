"""Canonical outcome-label formatter (evmax/formatting.py) — the single source
that replaced four divergent _display_label copies. EVGap.display_label and the
web/CLI surfaces all delegate here."""

from __future__ import annotations

from evmax.formatting import format_outcome_label, format_outcome_label_for_row


class TestGameMarkets:
    def test_moneyline(self):
        assert format_outcome_label(yes_team="lakers", market_type="moneyline") == "Lakers ML"

    def test_negative_spread_no_plus(self):
        assert format_outcome_label(yes_team="pistons", market_type="spread", line=-7.5) == "Pistons -7.5"

    def test_positive_spread_gets_plus(self):
        # The bug this fixes: a +9.5 dog cover must NOT render as "Warriors 9.5".
        assert format_outcome_label(yes_team="warriors", market_type="spread", line=9.5) == "Warriors +9.5"

    def test_total_over_under_by_side(self):
        assert format_outcome_label(yes_team="over", market_type="total", line=220.5) == "Over 220.5"
        assert format_outcome_label(yes_team="under", market_type="total", line=220.5) == "Under 220.5"

    def test_unknown_market_falls_back_to_team(self):
        assert format_outcome_label(yes_team="lakers", market_type="garbage") == "Lakers"

    def test_missing_team(self):
        assert format_outcome_label(yes_team=None, market_type="moneyline") == "? ML"


class TestProps:
    def test_prop_with_player_and_threshold(self):
        assert format_outcome_label(
            yes_team="bennedict_mathurin", market_type="player_prop",
            prop_player_name="bennedict_mathurin", prop_stat_type="assists", prop_threshold=4,
        ) == "Bennedict Mathurin 4+ AST"

    def test_prop_threshold_falls_back_to_line(self):
        assert format_outcome_label(
            yes_team="x", market_type="player_prop",
            prop_player_name="luka doncic", prop_stat_type="points", line=24.5,
        ) == "Luka Doncic 24.5+ PTS"


class TestRowAdapter:
    def test_row_dict(self):
        row = {"yes_team": "cavaliers", "market_type": "spread", "line": 9.5}
        assert format_outcome_label_for_row(row) == "Cavaliers +9.5"

    def test_row_prop(self):
        row = {
            "yes_team": "x", "market_type": "player_prop",
            "prop_player_name": "tyrese haliburton", "prop_stat_type": "assists",
            "prop_threshold": 10,
        }
        assert format_outcome_label_for_row(row) == "Tyrese Haliburton 10+ AST"


def test_evgap_delegates_to_formatter():
    from datetime import datetime, date
    from evmax.agents.odds.ev_gap_agent import EVGap

    gap = EVGap(
        market_id="m", event_id="e", sector="nba", yes_team="warriors",
        market_type="spread", kalshi_yes_price=0.5, sharp_true_prob=0.5,
        blended_true_prob=0.5, ev_pct=0.03, kelly_full=0.02, kelly_fraction=0.01,
        match_confidence=0.9, volume_usd=1000.0, spread_pct=0.02,
        event_date=datetime.combine(date.today(), datetime.min.time()),
        model_sources="elo", line=9.5, event_title="GSW vs LAL",
    )
    assert gap.display_label == "Warriors +9.5"
