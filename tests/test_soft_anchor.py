"""Soft-book anchor: record shapes, book preference, ESPN parsing, demotion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from evmax.agents.models import model_anchor as ma
from evmax.agents.odds.ev_gap_agent import _NON_MODEL_TOKENS, has_full_blend
from evmax.clients import soft_anchor as sa
from evmax.models.odds import SharpBook

NOW = datetime(2026, 11, 16, 18, 0, tzinfo=timezone.utc)
START = "2026-11-17T01:00Z"          # 20:00 ET on 11-16


def _by_kind(recs):
    out = {}
    for r in recs:
        kind = "spread" if "::spread" in r.event_id else "total" if "::total" in r.event_id else "ml"
        out[kind] = r
    return out


# --------------------------------------------------------------------- records

def test_build_records_mirror_the_pinnacle_shapes():
    recs = _by_kind(sa.build_records(
        "ncaaw", SharpBook.draftkings, "Central Michigan", "Michigan", _dt(START),
        ml=(15.0, 1.02), spread=(38.5, 1.91, 1.91), total=(142.5, 1.87, 1.95)))
    base = "ncaaw::2026-11-16::central_michigan_vs_michigan"
    assert recs["ml"].event_id == base
    assert recs["ml"].true_prob_a + recs["ml"].true_prob_b == pytest.approx(1.0)
    assert recs["ml"].true_prob_a < 0.1
    # outcome_a is the team LAYING points; the line is its (negative) handicap.
    sp = recs["spread"]
    assert sp.event_id == f"{base}::spread"
    assert sp.outcome_a_label == "Michigan" and sp.spread_line == -38.5
    assert sp.true_prob_a == pytest.approx(0.5)
    tot = recs["total"]
    assert tot.event_id == f"{base}::total::142.5" and tot.total_line == 142.5
    assert tot.true_prob_over > 0.5 > tot.true_prob_under
    assert all(r.book == SharpBook.draftkings for r in recs.values())


def test_home_favorite_spread_keeps_home_as_outcome_a():
    sp = _by_kind(sa.build_records("ncaaw", SharpBook.draftkings, "Michigan", "Ohio",
                                   _dt(START), spread=(-6.5, 1.95, 1.87)))["spread"]
    assert sp.outcome_a_label == "Michigan" and sp.spread_line == -6.5
    assert sp.true_prob_a < 0.5


def test_bad_prices_and_pickem_spread_produce_nothing():
    assert sa.build_records("ncaaw", SharpBook.draftkings, "A", "B", _dt(START),
                            ml=(1.0, 2.0), spread=(0.0, 1.9, 1.9), total=(0.0, 1.9, 1.9)) == []


def _dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------- The Odds API

def _api_event(books):
    return {"home_team": "Central Michigan", "away_team": "Michigan",
            "commence_time": START, "bookmakers": books}


def _bm(key, h2h=None, spread=None, total=None):
    markets = []
    if h2h:
        markets.append({"key": "h2h", "outcomes": [
            {"name": "Central Michigan", "price": h2h[0]}, {"name": "Michigan", "price": h2h[1]}]})
    if spread:
        markets.append({"key": "spreads", "outcomes": [
            {"name": "Central Michigan", "price": spread[1], "point": spread[0]},
            {"name": "Michigan", "price": spread[2], "point": -spread[0]}]})
    if total:
        markets.append({"key": "totals", "outcomes": [
            {"name": "Over", "price": total[1], "point": total[0]},
            {"name": "Under", "price": total[2], "point": total[0]}]})
    return {"key": key, "markets": markets}


def test_odds_api_prefers_the_sharpest_book_per_market():
    ev = _api_event([
        _bm("draftkings", h2h=(12.0, 1.03), spread=(37.5, 1.91, 1.91), total=(141.5, 1.91, 1.91)),
        _bm("pinnacle", h2h=(15.0, 1.02)),
        _bm("unknownbook", h2h=(9.0, 1.05)),
    ])
    recs = _by_kind(sa.parse_odds_api_event(ev, "ncaaw", now=NOW))
    assert recs["ml"].book == SharpBook.pinnacle            # sharpest for the market it has
    assert recs["spread"].book == SharpBook.draftkings      # falls through per market
    assert recs["total"].total_line == 141.5
    assert len(recs) == 3


def test_odds_api_skips_started_and_malformed_events():
    ev = _api_event([_bm("pinnacle", h2h=(15.0, 1.02))])
    assert sa.parse_odds_api_event(ev, "ncaaw", now=_dt(START) + timedelta(minutes=1)) == []
    assert sa.parse_odds_api_event({"home_team": "A"}, "ncaaw", now=NOW) == []
    bad = _api_event([{"key": "pinnacle", "markets": [{"key": "spreads", "outcomes": [
        {"name": "Central Michigan", "price": 1.9}, {"name": "Michigan", "price": 1.9}]}]}])
    assert sa.parse_odds_api_event(bad, "ncaaw", now=NOW) == []


# ------------------------------------------------------------------------ ESPN

def _espn_event(state="pre", provider="DraftKings", ml=("OFF", "OFF")):
    return {
        "date": START, "status": {"type": {"state": state}},
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"location": "Central Michigan"}},
                {"homeAway": "away", "team": {"location": "Michigan"}}],
            "odds": [{
                "provider": {"name": provider}, "overUnder": 142.5, "spread": 38.5,
                "moneyline": {"home": {"close": {"odds": ml[0]}}, "away": {"close": {"odds": ml[1]}}},
                "pointSpread": {"home": {"close": {"line": "+38.5", "odds": "-110"}},
                                "away": {"close": {"line": "-38.5", "odds": "-110"}}},
                "total": {"over": {"close": {"line": "o142.5", "odds": "-115"}},
                          "under": {"close": {"line": "u142.5", "odds": "-105"}}},
            }],
        }],
    }


def test_espn_event_with_moneyline_off_still_yields_spread_and_total():
    recs = _by_kind(sa.parse_espn_event(_espn_event(), "ncaaw", now=NOW))
    assert set(recs) == {"spread", "total"}
    assert recs["spread"].outcome_a_label == "Michigan" and recs["spread"].spread_line == -38.5
    assert recs["total"].true_prob_over > 0.5
    assert recs["spread"].book == SharpBook.draftkings


def test_espn_moneyline_is_parsed_when_posted():
    recs = _by_kind(sa.parse_espn_event(_espn_event(ml=("+1400", "-5000")), "ncaaw", now=NOW))
    assert recs["ml"].true_prob_b > 0.9


@pytest.mark.parametrize("kwargs", [
    {"state": "in"}, {"state": "post"}, {"provider": "DraftKings Live"}, {"provider": "Some Book"},
])
def test_espn_started_live_or_unknown_book_is_skipped(kwargs):
    assert sa.parse_espn_event(_espn_event(**kwargs), "ncaaw", now=NOW) == []


def test_espn_event_without_odds_is_skipped():
    ev = _espn_event()
    ev["competitions"][0]["odds"] = []
    assert sa.parse_espn_event(ev, "ncaaw", now=NOW) == []


@pytest.mark.parametrize("raw,dec", [("-110", 1.9091), ("+250", 3.5), ("EVEN", 2.0)])
def test_american_prices(raw, dec):
    assert sa._american(raw) == pytest.approx(dec, abs=1e-3)


@pytest.mark.parametrize("raw", ["OFF", "", None, "0", "50"])
def test_american_rejects_non_prices(raw):
    assert sa._american(raw) is None


# -------------------------------------------------------------------- gating

def test_only_configured_sectors_are_enabled():
    assert sa.soft_anchor_enabled("ncaaw") and sa.soft_anchor_enabled("NCAAW")
    assert not sa.soft_anchor_enabled("ncaab") and not sa.soft_anchor_enabled(None)


def test_soft_anchor_rows_can_never_be_live_eligible():
    assert "soft_anchor" in _NON_MODEL_TOKENS
    assert not has_full_blend("ncaaw", "elo+ncaaw_efficiency+soft_anchor", "moneyline")
    assert not has_full_blend("ncaaw", "sharp_ladder+soft_anchor", "spread")


def test_retag_with_the_soft_token():
    assert ma.retag_sources("elo+sharp", "soft_anchor") == "elo+soft_anchor"
    assert ma.retag_sources("sharp+sharp_ladder", "soft_anchor") == "sharp_ladder+soft_anchor"
    assert ma.retag_sources("elo+model_anchor", "soft_anchor") == "elo+soft_anchor"


def test_demote_anchored_gaps_tags_every_market_type_and_zeroes_kelly():
    base = "ncaaw::2026-11-16::central_michigan_vs_michigan"
    gaps = [
        SimpleNamespace(event_id=base, model_sources="elo+sharp", full_blend=True,
                        kelly_fraction=0.02, kelly_full=0.08),
        SimpleNamespace(event_id=f"{base}::total::142.5", model_sources="sharp+sharp_ladder",
                        full_blend=True, kelly_fraction=0.01, kelly_full=0.04),
        SimpleNamespace(event_id="ncaaw::2026-11-16::a_vs_b", model_sources="elo+sharp",
                        full_blend=True, kelly_fraction=0.02, kelly_full=0.08),
    ]
    assert ma.demote_anchored_gaps(gaps, {base: "soft_anchor"}) == 2
    assert gaps[0].model_sources == "elo+soft_anchor"
    assert gaps[1].model_sources == "sharp_ladder+soft_anchor"
    assert not gaps[0].full_blend and gaps[1].kelly_fraction == 0.0 and gaps[1].kelly_full == 0.0
    assert gaps[2].model_sources == "elo+sharp" and gaps[2].full_blend      # untouched


@pytest.mark.asyncio
async def test_entry_point_uses_espn_without_a_key_and_dedupes(monkeypatch):
    from evmax.models.market import MarketSource, MarketType, PredictionMarket

    async def fake_espn(sector, days):
        return [_espn_event(), _espn_event()]

    async def boom(*a, **k):
        raise AssertionError("no key: the Odds API must not be called")

    monkeypatch.setattr(sa, "_fetch_espn", fake_espn)
    monkeypatch.setattr(sa, "_fetch_odds_api", boom)
    monkeypatch.setattr(sa, "parse_espn_event",
                        lambda e, s, now=None: sa.build_records(
                            s, SharpBook.draftkings, "Central Michigan", "Michigan", _dt(START),
                            total=(142.5, 1.87, 1.95)))
    monkeypatch.setattr("evmax.settings.get_settings",
                        lambda: SimpleNamespace(odds_api_key="", soft_anchor_cache_ttl_secs=0))
    market = PredictionMarket(id="k", source=MarketSource.kalshi, sector="ncaaw",
                              market_type=MarketType.moneyline, yes_price=0.5, no_price=0.5,
                              team_home="central michigan", team_away="michigan",
                              yes_team="michigan", event_date=_dt(START))
    recs = await sa.soft_anchor_odds("ncaaw", [market])
    assert len(recs) == 1 and recs[0].total_line == 142.5
    assert await sa.soft_anchor_odds("ncaab", [market]) == []


def test_odds_api_names_map_to_the_school_name_and_extra_books_parse():
    ev = {"home_team": "Central Michigan Chippewas", "away_team": "Michigan Wolverines",
          "commence_time": START, "bookmakers": [{"key": "betrivers", "markets": [
              {"key": "h2h", "outcomes": [{"name": "Central Michigan Chippewas", "price": 15.0},
                                          {"name": "Michigan Wolverines", "price": 1.02}]}]}]}
    names = sa.espn_display_names([{"competitions": [{"competitors": [
        {"team": {"displayName": "Central Michigan Chippewas", "location": "Central Michigan"}},
        {"team": {"displayName": "Michigan Wolverines", "location": "Michigan"}}]}]}])
    recs = sa.parse_odds_api_event(ev, "ncaaw", now=NOW, names=names)
    assert len(recs) == 1 and recs[0].book == SharpBook.betrivers
    assert recs[0].event_id == "ncaaw::2026-11-16::central_michigan_vs_michigan"
    assert recs[0].outcome_a_label == "Central Michigan"


def test_settings_accept_the_legacy_key_name(monkeypatch):
    from evmax.settings import Settings
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setenv("THE_ODDS_API_KEY", "abc")
    assert Settings(_env_file=None).odds_api_key == "abc"
    monkeypatch.setenv("ODDS_API_KEY", "new")
    assert Settings(_env_file=None).odds_api_key == "new"
