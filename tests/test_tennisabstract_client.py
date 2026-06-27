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
    fetch_matchmx,
    parse_elo_html,
    parse_last_update,
    parse_matchmx,
    parse_winners_errors_html,
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


# ---------------------------------------------------------------------------
# matchmx (per-match serve/return matrix)
# ---------------------------------------------------------------------------

def _mx_row(**vals) -> list:
    """Build a 45-column matchmx row, filling the given indices by field name."""
    from evmax.clients.tennisabstract import _MX
    row = [""] * 45
    for field, v in vals.items():
        row[_MX[field]] = v
    return row


def _matchmx_js(rows) -> str:
    import json
    return "var matchmx = " + json.dumps(rows) + ";\n"


def test_parse_matchmx_keeps_only_winner_rows_and_maps_fields():
    win = _mx_row(
        date="20240115", surf="Hard", level="G", wl="W",
        player="Jannik Sinner", rank="1", opp="Daniil Medvedev", orank="3", time="218",
        pts="144", fwon="95", swon="20", saved="4", chances="6",
        opts="150", ofwon="80", oswon="25", osaved="8", ochances="12",
    )
    # the same match from the loser's perspective — must be dropped
    loss = _mx_row(date="20240115", surf="Hard", wl="L",
                   player="Daniil Medvedev", opp="Jannik Sinner")
    recs = parse_matchmx(_matchmx_js([win, loss]))
    assert len(recs) == 1
    r = recs[0]
    assert r["winner_name"] == "Jannik Sinner"
    assert r["loser_name"] == "Daniil Medvedev"
    assert r["surface"] == "Hard"
    assert r["tourney_date"] == "20240115"
    # winner serve from own columns; loser serve from the opponent (o*) columns
    assert (r["w_svpt"], r["w_1stWon"], r["w_2ndWon"]) == ("144", "95", "20")
    assert (r["w_bpSaved"], r["w_bpFaced"]) == ("4", "6")
    assert (r["l_svpt"], r["l_1stWon"], r["l_2ndWon"]) == ("150", "80", "25")
    assert (r["l_bpSaved"], r["l_bpFaced"]) == ("8", "12")
    assert (r["winner_rank"], r["loser_rank"], r["minutes"]) == ("1", "3", "218")


def test_parse_matchmx_skips_rows_missing_names():
    bad = _mx_row(date="20240115", wl="W", player="", opp="Someone")
    assert parse_matchmx(_matchmx_js([bad])) == []


def test_parse_matchmx_empty_when_no_var():
    assert parse_matchmx("var other = [];") == []


def test_fetch_matchmx_unions_segments_and_dedupes():
    win = _mx_row(date="20240115", surf="Clay", wl="W",
                  player="Carlos Alcaraz", opp="Alexander Zverev",
                  pts="100", fwon="60", swon="20", opts="98", ofwon="55", oswon="18")
    js = _matchmx_js([win])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=js)  # every segment returns the same match

    client = httpx.Client(transport=httpx.MockTransport(handler))
    recs = fetch_matchmx("atp", client=client)
    # 3 ATP segments all return the same match -> deduped to one
    assert len(recs) == 1
    assert recs[0]["winner_name"] == "Carlos Alcaraz"


# ---------------------------------------------------------------------------
# winners / unforced errors
# ---------------------------------------------------------------------------

WE_FIXTURE = """
<table>
<tr><td>Player</td><td>Matches</td><td>Winners</td><td>UFEs</td><td>Ratio</td><td>Wnr/Pt</td></tr>
<tr><td><a href="x">Arthur Fils</a></td><td>17</td><td>479</td><td>472</td><td>1.0</td><td>20.7%</td></tr>
<tr><td>Jesper De Jong</td><td>13</td><td>470</td><td>351</td><td>1.3</td><td>19.1%</td></tr>
</table>
"""


def test_parse_winners_errors():
    we = parse_winners_errors_html(WE_FIXTURE)
    assert we["arthur fils"] == {"winners": 479, "unforced": 472}
    assert we["jesper de jong"] == {"winners": 470, "unforced": 351}


def test_parse_winners_errors_empty_without_header():
    assert parse_winners_errors_html("<table><tr><td>nope</td></tr></table>") == {}
