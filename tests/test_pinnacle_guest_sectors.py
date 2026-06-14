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

from evmax.clients.esports_pinnacle import (
    NAME_MATCHED_SECTORS,
    TOTALS_MAIN_LINE_ONLY_SECTORS,
    TOTALS_SECTORS,
)


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


def test_baseball_is_main_total_line_only():
    # Baseball prices only the main total line — its alt O/U ladder floods the
    # scan and isn't bettable signal (skewed run distribution, thin alt books).
    assert "baseball" in TOTALS_MAIN_LINE_ONLY_SECTORS
    # Main-line-only is a subset of totals-eligible sectors.
    assert TOTALS_MAIN_LINE_ONLY_SECTORS <= TOTALS_SECTORS


def test_live_totals_sectors_keep_their_alt_ladder():
    # Sectors with live totals (NBA/NFL/WNBA) must NOT be main-line-only, or
    # their Kalshi alt strikes would lose exact-line matching.
    for sector in ("nba", "nfl", "wnba"):
        assert sector not in TOTALS_MAIN_LINE_ONLY_SECTORS
