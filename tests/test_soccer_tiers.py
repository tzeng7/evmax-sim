"""Tests for the soccer league tier lookup that drives per-league sharp_weight."""
from __future__ import annotations

from evmax.sectors import soccer_tiers


def test_top_tier_leagues_get_elevated_sharp_weight():
    """Top-5 Euro + UEFA competitions should all return the top-tier weight."""
    for series in (
        "KXEPLGAME",
        "KXLALIGAGAME",
        "KXBUNDESLIGAGAME",
        "KXSERIEAGAME",
        "KXLIGUE1GAME",
        "KXUCLGAME",
        "KXUELGAME",
    ):
        ticker = f"{series}-26APR24MANCHE-CHE"
        assert soccer_tiers.sharp_weight_for_ticker(ticker) == 0.85, (
            f"{series} should be top-tier"
        )


def test_secondary_league_stays_at_default():
    """MLS is configured in the secondary tier with the default weight."""
    assert soccer_tiers.sharp_weight_for_ticker("KXMLSGAME-26APR24LAXNYC-NYC") == 0.40


def test_unknown_ticker_falls_back_to_default():
    """Tickers not in any tier use default_sharp_weight."""
    assert soccer_tiers.sharp_weight_for_ticker("KXFAKELEAGUE-26APR24XXXYYY-YYY") == 0.40


def test_empty_ticker_returns_default():
    assert soccer_tiers.sharp_weight_for_ticker("") == 0.40
    assert soccer_tiers.sharp_weight_for_ticker(None) == 0.40


def test_ticker_with_source_prefix_is_stripped():
    """`kalshi:KXEPLGAME-...` should still resolve to the EPL tier."""
    assert soccer_tiers.sharp_weight_for_ticker("kalshi:KXEPLGAME-26APR24MANCHE-CHE") == 0.85


def test_case_insensitive_series_match():
    """Lowercase ticker still maps."""
    assert soccer_tiers.sharp_weight_for_ticker("kxeplgame-26apr24manche-che") == 0.85


def test_default_sharp_weight_helper():
    assert soccer_tiers.default_sharp_weight() == 0.40
