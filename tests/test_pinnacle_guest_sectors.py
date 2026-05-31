"""Sector-coverage gates for the Pinnacle guest client.

This file does not (yet) cover the network or parsing paths in
``evmax/clients/esports_pinnacle.py`` — those have known test debt called
out in CLAUDE.md. What it locks down here is the set of sectors that
``_parse_event_to_odds`` is willing to attach a ``::total::{line}`` SharpOdds
row for. Without WNBA in that set, every KXWNBATOTAL alternate on Kalshi
fell through the matching engine as ``match_failed`` because there was no
sharp ``::total::*`` event_id to match against.
"""

from __future__ import annotations

from evmax.clients.esports_pinnacle import TOTALS_SECTORS, NAME_MATCHED_SECTORS


def test_wnba_eligible_for_totals():
    assert "wnba" in TOTALS_SECTORS


def test_scoring_team_sports_eligible():
    # Anything that posts a game-total market on Pinnacle.
    for sector in ("nba", "wnba", "nfl", "ncaab", "ncaaw", "soccer", "baseball", "nhl"):
        assert sector in TOTALS_SECTORS, f"{sector} missing from TOTALS_SECTORS"


def test_non_scoring_sports_excluded():
    # Esports / UFC / F1 / tennis don't have a "total" concept that maps to
    # Kalshi's totals tickers — keep them out so we don't waste time parsing.
    for sector in NAME_MATCHED_SECTORS:
        assert sector not in TOTALS_SECTORS, f"{sector} should not be in TOTALS_SECTORS"
