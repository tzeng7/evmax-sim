"""College basketball teams are named from the Kalshi EVENT title.

Payloads below are trimmed copies of live Kalshi responses (2026-11-15). The
per-team market title is the short "{Team} wins", so without the event join a
team is only its ticker code, and a code resolves only when it has been added
to the alias map by hand.
"""

from __future__ import annotations

import pytest

from evmax.clients.kalshi import (
    KalshiClient,
    _EVENT_TITLE_SECTORS,
    _event_code_names,
    _expand_trailing_state,
    _name_for_code,
)
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketType

GAME_EV = "KXNCAAMBGAME-26NOV14UNCGCOR"
SPREAD_EV = "KXNCAAMBSPREAD-26NOV14UNCGCOR"
TOTAL_EV = "KXNCAAMBTOTAL-26NOV14UNCGCOR"
TITLES = {
    GAME_EV: "UNC Greensboro vs Cornell",
    SPREAD_EV: "UNC Greensboro vs Cornell: Spread",
    TOTAL_EV: "UNC Greensboro vs Cornell: Point Total",
}
CODES = {ev: ("uncg", "cor") for ev in TITLES}


def _raw(ticker, event_ticker, title, yes_sub, floor=None):
    return {
        "ticker": ticker, "event_ticker": event_ticker, "title": title,
        "yes_sub_title": yes_sub, "floor_strike": floor, "status": "active",
        "yes_ask_dollars": "0.40", "no_ask_dollars": "0.62",
        "yes_bid_dollars": "0.38", "no_bid_dollars": "0.60",
        "yes_ask": 40, "no_ask": 62, "yes_bid": 38, "no_bid": 60,
        "volume": 100, "open_interest": 50,
        "close_time": "2026-11-15T03:00:00Z",
    }


def _parse(raw, sector="ncaab", titles=TITLES, codes=CODES):
    client = KalshiClient.__new__(KalshiClient)
    return client._parse_market(raw, sector, competitions={}, event_titles=titles,
                                event_codes=codes, event_fighters=None, event_teams=None)


def _norm(sector="ncaab"):
    return NameNormalizer(sector).normalize


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_college_sectors_fetch_events():
    assert {"ncaab", "ncaaw"} <= _EVENT_TITLE_SECTORS


@pytest.mark.parametrize("raw, expected", [
    ("Missouri St.", "Missouri State"),
    ("Long Beach St", "Long Beach State"),
    ("St. Thomas", "St. Thomas"),              # leading St. = Saint
    ("St. John's", "St. John's"),
    ("Mount St. Mary's", "Mount St. Mary's"),  # mid-name
    ("Michigan State", "Michigan State"),
    ("Cornell", "Cornell"),
])
def test_only_a_trailing_st_becomes_state(raw, expected):
    assert _expand_trailing_state(raw) == expected


def test_event_code_names_pairs_codes_with_names_in_order():
    n = _norm()
    names = _event_code_names("UNC Greensboro vs Cornell", ("uncg", "cor"), n)
    assert names == {"uncg": n("UNC Greensboro"), "cor": n("Cornell")}
    # the ": Spread" / ": Point Total" suffix is not part of the second name
    for suffix in (": Spread", ": Point Total"):
        assert _event_code_names("UNC Greensboro vs Cornell" + suffix, ("uncg", "cor"), n) == names


def test_event_code_names_expands_state():
    n = _norm()
    names = _event_code_names("Abilene Christian vs Missouri St.", ("ac", "most"), n)
    assert names["most"] == n("Missouri State")


@pytest.mark.parametrize("title, codes", [
    (None, ("a", "b")),
    ("UNC Greensboro vs Cornell", None),
    ("UNC Greensboro at Cornell", ("uncg", "cor")),     # no " vs "
    ("Cornell vs Cornell", ("cor", "cor")),             # not two teams
    ("UNC Greensboro vs Cornell", ("uncg", "")),
])
def test_event_code_names_refuses_a_partial_join(title, codes):
    assert _event_code_names(title, codes, _norm()) is None


def test_curated_alias_outranks_the_event_title():
    n = _norm()
    # "ku" is a curated alias for kansas; the event title must not override it.
    assert _name_for_code("ku", {"ku": "some other spelling"}, n) == "kansas"
    # an unknown code takes the event's name
    assert _name_for_code("uncg", {"uncg": "unc greensboro"}, n) == "unc greensboro"
    # unknown code, no event name: the old behaviour (the bare code)
    assert _name_for_code("zzzq", {}, n) == "zzzq"
    assert _name_for_code(None, {}, n) is None


# ---------------------------------------------------------------------------
# parser, on live-shaped payloads
# ---------------------------------------------------------------------------


def test_moneyline_teams_and_yes_side_come_from_the_event():
    n = _norm()
    m = _parse(_raw(GAME_EV + "-UNCG", GAME_EV, "UNC Greensboro wins", "UNC Greensboro"))
    assert m.market_type == MarketType.moneyline
    assert m.yes_team == n("UNC Greensboro")
    assert {m.team_home, m.team_away} == {n("UNC Greensboro"), n("Cornell")}
    assert m.team_away == n("UNC Greensboro") and m.team_home == n("Cornell")
    other = _parse(_raw(GAME_EV + "-COR", GAME_EV, "Cornell wins", "Cornell"))
    assert other.yes_team == n("Cornell")
    assert (other.team_home, other.team_away) == (m.team_home, m.team_away)


def test_spread_rung_keeps_its_line_and_names_the_covering_team():
    n = _norm()
    m = _parse(_raw(SPREAD_EV + "-COR24", SPREAD_EV, "Cornell wins by over 23.5 points",
                    "Cornell wins by over 23.5 points", floor=23.5))
    assert m.market_type == MarketType.spread
    assert m.line == pytest.approx(-23.5)
    assert m.yes_team == n("Cornell")
    assert (m.team_away, m.team_home) == (n("UNC Greensboro"), n("Cornell"))


def test_total_gets_both_teams_despite_a_variable_width_code_pair():
    # "UNCGCOR" cannot be split 3+3 ("unc" / "gcor"); the event names fix it.
    n = _norm()
    m = _parse(_raw(TOTAL_EV + "-129", TOTAL_EV, "Full Game: Over 128.5 points scored",
                    "Over 128.5 points", floor=128.5))
    assert m.market_type == MarketType.total
    assert m.yes_team == "over" and m.line == pytest.approx(128.5)
    assert (m.team_away, m.team_home) == (n("UNC Greensboro"), n("Cornell"))


def test_without_the_event_join_the_old_code_path_still_runs():
    m = _parse(_raw(GAME_EV + "-UNCG", GAME_EV, "UNC Greensboro wins", "UNC Greensboro"),
               titles=None, codes=None)
    assert m is not None and m.yes_team == _norm()("uncg")


def test_womens_series_is_named_the_same_way():
    ev = "KXNCAAWBGAME-26NOV14LBSUPORT"
    n = _norm("ncaaw")
    m = _parse(_raw(ev + "-LBSU", ev, "Long Beach St. wins", "Long Beach St."), sector="ncaaw",
               titles={ev: "Long Beach St. vs Portland"}, codes={ev: ("lbsu", "port")})
    assert m.yes_team == n("Long Beach State")
    assert m.team_home == n("Portland")


def test_other_sectors_are_untouched():
    ev = "KXNBAGAME-26NOV16CHICLE"
    with_join = _parse(_raw(ev + "-CHI", ev, "Chicago wins", "Chicago"), sector="nba",
                       titles={ev: "Chicago vs Cleveland"}, codes={ev: ("chi", "cle")})
    without = _parse(_raw(ev + "-CHI", ev, "Chicago wins", "Chicago"), sector="nba",
                     titles=None, codes=None)
    assert (with_join.yes_team, with_join.team_home, with_join.team_away) == (
        without.yes_team, without.team_home, without.team_away)


def test_pinnacles_own_state_abbreviation_is_aliased():
    assert _norm()("Tarleton St") == _norm()("Tarleton State")


# ---------------------------------------------------------------------------
# mode: the join multiplies matched NCAAB rungs, so they ship in shadow
# ---------------------------------------------------------------------------


def test_ncaab_spread_and_total_ship_shadow_moneyline_stays_live():
    from datetime import date

    from evmax.categories import get_category
    from evmax.modes import get_mode

    assert set(get_category("ncaab").shadow_market_types) == {"spread", "total"}
    in_season = date(2026, 11, 20)
    assert get_mode("ncaab", "moneyline", today=in_season) == "live"
    assert get_mode("ncaab", "spread", today=in_season) == "shadow"
    assert get_mode("ncaab", "total", today=in_season) == "shadow"
