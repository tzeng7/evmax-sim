"""Tests for the Tennis Abstract Elo leaderboard client/parser.

The parser must skip the page's prose intro (a single-cell row whose text happens
to contain the words 'players' and 'hElo'), key columns by their header labels,
unwrap player links + &nbsp;, and tolerate missing surface values and stray rows.
"""
from __future__ import annotations

import httpx
import pytest

from evmax.clients.tennisabstract import (
    PlayerElo,
    fetch_elo_ratings,
    parse_elo_html,
    parse_last_update,
)

# Mirrors the real page shape: an intro <table> (prose, single cell) followed by
# the data <table> with a header row then player rows. Column order matches the
# live page: Elo Rank, Player, Age, Elo, _, hElo Rank, hElo, cElo Rank, cElo,
# gElo Rank, gElo, _, Peak Elo, Peak Month, _, ATP Rank, Log diff.
FIXTURE = """
<html><body>
<table><tr><td>Current Elo ratings for the ATP tour. This list includes only those
players who have completed 10+ matches. Surface Elos hElo, cElo, gElo are blends.
Updated weekly(ish). Last update: 2026-06-22</td></tr></table>
<table>
<tr><td align="right">Elo Rank</td><td>Player</td><td>Age</td><td>Elo</td><td></td>
<td>hElo Rank</td><td>hElo</td><td>cElo Rank</td><td>cElo</td><td>gElo Rank</td><td>gElo</td>
<td></td><td>Peak Elo</td><td>Peak Month</td><td></td><td>ATP Rank</td><td>Log diff</td></tr>
<tr><td align="right">1</td><td><a href="x">Jannik&nbsp;Sinner</a></td><td>24.7</td>
<td align="right">2319.8</td><td></td><td>1</td><td align="right">2263.2</td>
<td>1</td><td align="right">2215.7</td><td>1</td><td align="right">2088.3</td>
<td></td><td>2339.8</td><td>2026-05</td><td></td><td>1</td><td>0</td></tr>
<tr><td align="right">2</td><td><a href="x">Carlos&nbsp;Alcaraz</a></td><td>22.9</td>
<td align="right">2161.8</td><td></td><td>2</td><td align="right">2088.3</td>
<td>2</td><td align="right">2101.6</td><td>3</td><td align="right">2029.2</td>
<td></td><td>2210.0</td><td>2026-05</td><td></td><td>2</td><td>0</td></tr>
<tr><td align="right">3</td><td><a href="x">Grass&nbsp;Hater</a></td><td>30.0</td>
<td align="right">1800.0</td><td></td><td>50</td><td align="right">1820.0</td>
<td>40</td><td align="right">1790.0</td><td></td><td align="right"></td>
<td></td><td>1850.0</td><td>2024-01</td><td></td><td></td><td>0</td></tr>
</table>
</body></html>
"""


def test_parses_rows_and_maps_surfaces():
    rows = parse_elo_html(FIXTURE)
    assert len(rows) == 3
    sinner = rows[0]
    assert sinner.player == "Jannik Sinner"      # &nbsp; collapsed, <a> stripped
    assert sinner.elo == 2319.8
    assert sinner.hard == 2263.2
    assert sinner.clay == 2215.7
    assert sinner.grass == 2088.3
    assert sinner.official_rank == 1


def test_skips_prose_intro_row():
    # The intro paragraph mentions 'players' and 'hElo' but is not a data row.
    rows = parse_elo_html(FIXTURE)
    assert all(isinstance(r, PlayerElo) for r in rows)
    assert "Current Elo" not in {r.player for r in rows}


def test_tolerates_missing_surface():
    rows = parse_elo_html(FIXTURE)
    grass_hater = rows[2]
    assert grass_hater.grass is None         # empty gElo cell -> None
    assert grass_hater.hard == 1820.0
    assert grass_hater.official_rank is None  # empty ATP Rank cell -> None


def test_parse_last_update():
    assert parse_last_update(FIXTURE) == "2026-06-22"


def test_empty_or_headerless_html_returns_empty():
    assert parse_elo_html("<html><body><p>nothing here</p></body></html>") == []
    assert parse_elo_html("") == []


def test_fetch_uses_injected_client():
    """fetch_elo_ratings should parse whatever the injected transport returns."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert "atp_elo_ratings" in str(request.url)
        return httpx.Response(200, text=FIXTURE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = fetch_elo_ratings("atp", client=client)
    assert rows[0].player == "Jannik Sinner"
    assert len(rows) == 3


def test_fetch_rejects_unknown_tour():
    with pytest.raises(ValueError):
        fetch_elo_ratings("padel")
