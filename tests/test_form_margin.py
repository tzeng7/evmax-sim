"""FormModelAgent margin-based form — records carry margins; when a sector is in
MARGIN_FORM_SECTORS it blends on them, everything else keeps W/L form, and
mixed legacy state falls back. NFL margin form was backtest-REJECTED
(2026-09-03, see form_agent.py) so the set ships empty; the autouse fixture
switches NFL on for these tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from evmax.agents.models import form_agent as fm
from evmax.agents.models.form_agent import FormModelAgent, GameRecord
from evmax.models.market import MarketSource, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


def _agent():
    a = FormModelAgent()
    a._state = {}
    return a


@pytest.fixture(autouse=True)
def _nfl_margin_form_on():
    """NFL margin form is backtest-REJECTED and ships OFF; these tests exercise
    the machinery by switching it on explicitly (the same toggle the
    walk-forward harness uses to score the `form_margin` column)."""
    fm.MARGIN_FORM_SECTORS.add("nfl")
    try:
        yield
    finally:
        fm.MARGIN_FORM_SECTORS.discard("nfl")


def test_margin_form_ships_off_for_every_sector():
    fm.MARGIN_FORM_SECTORS.discard("nfl")
    try:
        assert fm.MARGIN_FORM_SECTORS == set()
    finally:
        fm.MARGIN_FORM_SECTORS.add("nfl")


def _predict(agent, sector, home, away, when="2026-10-15"):
    dt = datetime.fromisoformat(when).replace(tzinfo=timezone.utc)
    market = PredictionMarket(id="t", market_id="t", event_id="e", sector=sector, team_home=home, team_away=away,
                              source=MarketSource.kalshi, yes_price=0.5, no_price=0.5, event_date=dt)
    sharp = SharpOdds(event_id="e", book=SharpBook.pinnacle, sector=sector, outcome_a_label=home,
                      outcome_b_label=away, outcome_a_decimal=2.0, outcome_b_decimal=2.0,
                      true_prob_a=0.5, true_prob_b=0.5)
    return asyncio.run(agent.predict_pair(market, sharp))


def _feed(agent, sector, games):
    """games: (date, home, away, hs, as)"""
    for d, h, a, hs, as_ in games:
        agent.update(h, a, hs, as_, sector, event_date=d)


def test_update_stores_signed_margin_for_both_sides():
    a = _agent()
    a.update("chiefs", "broncos", 27, 13, "nfl", event_date="2026-09-13")
    kc = a._state["nfl"]["chiefs"][0]
    den = a._state["nfl"]["broncos"][0]
    assert kc["margin"] == 14.0 and kc["won"] is True and kc["home"] is True
    assert den["margin"] == -14.0 and den["won"] is False and den["home"] is False
    assert GameRecord(**kc).margin == 14.0
    assert GameRecord(date="2025-01-01", won=True, opp="x", home=True).margin is None   # legacy rows load


def test_margin_rate_is_home_edge_adjusted_and_none_when_any_margin_missing():
    params = fm.MARGIN_FORM_PARAMS["nfl"]
    recs = [GameRecord("2026-09-20", True, "x", True, margin=10.0),   # home +10 → adj +8
            GameRecord("2026-09-13", False, "y", False, margin=-2.0)]  # away −2 → adj 0
    r = FormModelAgent._margin_rate(recs, params)
    assert r == pytest.approx((8.0 * 1.0 + 0.0 * fm.DECAY) / (1.0 + fm.DECAY))
    recs.append(GameRecord("2026-09-06", True, "z", True))            # no margin → fallback signal
    assert FormModelAgent._margin_rate(recs, params) is None


def test_nfl_margin_form_beats_wl_form_on_blowouts_vs_squeakers():
    """Two 3-0 teams: one wins by 20 a game, the other by 1. W/L form calls
    them equal; margin form must not."""
    a = _agent()
    _feed(a, "nfl", [
        ("2026-09-13", "a", "p", 30, 10), ("2026-09-20", "a", "q", 31, 10), ("2026-09-27", "a", "r", 33, 13),
        ("2026-09-13", "b", "s", 21, 20), ("2026-09-20", "b", "t", 17, 16), ("2026-09-27", "b", "u", 20, 19),
    ])
    pred = _predict(a, "nfl", "a", "b", "2026-10-04")
    assert pred is not None and pred.true_prob_a > 0.6
    assert "margin_a=" in pred.notes and "pred=" in pred.notes
    # swap home/away → the strong team is still favoured; the two views differ
    # only by the home edge applied twice (2 pts each way ≈ 0.09 in P at σ=13.5)
    rev = _predict(a, "nfl", "b", "a", "2026-10-04")
    assert rev.true_prob_a < 0.35
    assert abs(rev.true_prob_a - (1 - pred.true_prob_a)) < 0.12


def test_nfl_falls_back_to_wl_form_when_records_lack_margins():
    a = _agent()
    # legacy-shaped records (no margin key)
    a._state = {"nfl": {
        "a": [{"date": d, "won": True, "opp": "x", "home": True} for d in ("2026-09-27", "2026-09-20", "2026-09-13")],
        "b": [{"date": d, "won": False, "opp": "y", "home": False} for d in ("2026-09-27", "2026-09-20", "2026-09-13")],
    }}
    pred = _predict(a, "nfl", "a", "b", "2026-10-04")
    assert pred is not None and "form_a=" in pred.notes and "margin_a=" not in pred.notes


def test_non_nfl_sector_ignores_margins():
    a = _agent()
    _feed(a, "nba", [("2026-11-01", "a", "p", 130, 90), ("2026-11-03", "a", "q", 120, 80), ("2026-11-05", "a", "r", 125, 85),
                     ("2026-11-01", "b", "s", 101, 100), ("2026-11-03", "b", "t", 99, 98), ("2026-11-05", "b", "u", 110, 109)])
    pred = _predict(a, "nba", "a", "b", "2026-11-07")
    assert pred is not None and "margin_a=" not in pred.notes
    # equal W/L records → log5 says 50/50 plus the NBA home bump
    assert pred.true_prob_a == pytest.approx(0.5 + fm.HOME_ADJ["nba"], abs=1e-6)


def test_margin_form_toggle_is_the_backtest_switch():
    a = _agent()
    _feed(a, "nfl", [("2026-09-13", "a", "p", 30, 10), ("2026-09-20", "a", "q", 31, 10), ("2026-09-27", "a", "r", 33, 13),
                     ("2026-09-13", "b", "s", 21, 20), ("2026-09-20", "b", "t", 17, 16), ("2026-09-27", "b", "u", 20, 19)])
    fm.MARGIN_FORM_SECTORS.discard("nfl")
    try:
        wl = _predict(a, "nfl", "a", "b", "2026-10-04")
        assert "margin_a=" not in wl.notes
    finally:
        fm.MARGIN_FORM_SECTORS.add("nfl")
    assert "margin_a=" in _predict(a, "nfl", "a", "b", "2026-10-04").notes


def test_walkforward_form_predict_runs_the_margin_path():
    """The replay harness must exercise the live form path — before 2026-09-03
    it re-implemented log5 itself and silently bypassed margin form."""
    from datetime import date
    from evmax.backtest.sources.espn_walkforward import _form_update, _form_predict
    a = _agent()
    for d, h, aw, hs, as_ in [("2026-09-13", "a", "p", 30, 10), ("2026-09-20", "a", "q", 31, 10), ("2026-09-27", "a", "r", 33, 13),
                              ("2026-09-13", "b", "s", 21, 20), ("2026-09-20", "b", "t", 17, 16), ("2026-09-27", "b", "u", 20, 19)]:
        _form_update(a, "nfl", h, aw, hs > as_, date.fromisoformat(d), home_score=hs, away_score=as_)
    assert a._state["nfl"]["a"][0]["margin"] == 20.0 and a._state["nfl"]["p"][0]["margin"] == -20.0
    p_margin = _form_predict(a, "nfl", "a", "b", date(2026, 10, 4))
    fm.MARGIN_FORM_SECTORS.discard("nfl")
    try:
        p_wl = _form_predict(a, "nfl", "a", "b", date(2026, 10, 4))
    finally:
        fm.MARGIN_FORM_SECTORS.add("nfl")
    assert p_wl == pytest.approx(0.5 + fm.HOME_ADJ["nfl"], abs=1e-6)   # equal 3-0 records
    assert p_margin > 0.6 and p_margin != p_wl
