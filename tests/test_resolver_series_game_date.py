"""A row must be graded on ITS OWN game, never on the previous game of a series.

Regression (2026-09-22): the ESPN resolve path pools the event_date−1 / event_date /
event_date+1 scoreboards to absorb stored-date off-by-ones, and the scoreboard
fetch returned COMPLETED games only. When a row was resolved before its own
game finished (the morning resolve picks up rows scanned yesterday for games
today), the only matchup left in the pool was YESTERDAY's completed game of
the same series, and the row was graded on it — 192 rows (191 baseball), e.g.
row #75181 resolved at 14:33 UTC before its 18:40 ET first pitch.
"""

from __future__ import annotations

import asyncio

import pytest

from evmax.agents.cleanup import resolver
from evmax.agents.cleanup.resolver import _fetch_espn_scores, _match_espn


def _score(home, hs, away, as_, game_date, completed=True):
    return {
        "home_name": home, "away_name": away,
        "home_abbr": "", "away_abbr": "",
        "home_score": hs if completed else None,
        "away_score": as_ if completed else None,
        "home_won": (hs > as_) if completed else None,
        "completed": completed,
        "game_date": game_date,
    }


def _pred(yes="reds", event_date="2026-06-02"):
    return {
        "market_id": f"kalshi:KXMLBGAME-{yes}",
        "event_id": f"baseball::{event_date}::reds_vs_royals",
        "sector": "baseball",
        "yes_team": yes,
        "event_date": event_date,
        "market_type": "moneyline",
    }


YESTERDAY_REDS_WIN = _score("Cincinnati Reds", 7, "Kansas City Royals", 2, "2026-06-01")


def test_pending_own_day_game_blocks_yesterdays_result():
    today_pending = _score("Cincinnati Reds", 0, "Kansas City Royals", 0, "2026-06-02",
                           completed=False)
    assert _match_espn(_pred(), [YESTERDAY_REDS_WIN, today_pending]) is None


def test_completed_own_day_game_beats_yesterdays():
    today_royals_win = _score("Cincinnati Reds", 1, "Kansas City Royals", 5, "2026-06-02")
    assert _match_espn(_pred(), [YESTERDAY_REDS_WIN, today_royals_win]) == 0


def test_off_by_one_stored_date_still_resolves_without_an_own_day_game():
    """The ±1-day window stays for a genuinely mis-dated row: no game between
    these teams exists on its own date at all."""
    assert _match_espn(_pred(event_date="2026-06-02"), [YESTERDAY_REDS_WIN]) == 1


def test_incomplete_games_never_grade_anything():
    only_pending = _score("Cincinnati Reds", 0, "Kansas City Royals", 0, "2026-06-01",
                          completed=False)
    assert _match_espn(_pred(), [only_pending]) is None


# --- fetch: incomplete games are opt-in and share one cached slate ----------

class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def _event(done: bool, hs: str, as_: str) -> dict:
    return {"season": {"type": 2}, "competitions": [{
        "status": {"type": {"completed": done, "name": "STATUS_FINAL" if done else "STATUS_SCHEDULED"}},
        "competitors": [
            {"homeAway": "home", "score": hs, "team": {"displayName": "Cincinnati Reds", "abbreviation": "CIN"}},
            {"homeAway": "away", "score": as_, "team": {"displayName": "Kansas City Royals", "abbreviation": "KC"}},
        ]}]}


class _Client:
    def __init__(self):
        self.calls = 0

    async def get(self, url, params=None):
        self.calls += 1
        return _Resp({"events": [_event(True, "3", "1"), _event(False, "0", "0")]})


def test_fetch_filters_incomplete_by_default_and_caches_one_slate():
    client, cache = _Client(), {}

    async def run():
        default = await _fetch_espn_scores(client, "baseball", "mlb", "20260602", cache=cache)
        full = await _fetch_espn_scores(client, "baseball", "mlb", "20260602", cache=cache,
                                        include_incomplete=True)
        return default, full

    default, full = asyncio.run(run())
    assert [g["completed"] for g in default] == [True]
    assert [g["completed"] for g in full] == [True, False]
    assert full[1]["home_score"] is None and full[1]["home_won"] is None
    assert client.calls == 1            # one fetch serves both views
    assert resolver  # module import sanity


# --- repair script: re-grade only onto the row's OWN-date game ---------------

def _repair_module():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "repair_series_game_outcomes",
        Path(__file__).resolve().parents[1] / "scripts" / "repair_series_game_outcomes.py",
    )
    import sys

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module at class creation
    spec.loader.exec_module(mod)
    return mod


def _stored_row(stored: int, event_date="2026-06-02"):
    return {**_pred(event_date=event_date), "stored": stored, "mode": "live",
            "placed": 0, "event_title": "Cincinnati Reds vs Kansas City Royals"}


def test_repair_regrades_a_row_graded_on_yesterdays_game():
    mod = _repair_module()
    today_royals_win = _score("Cincinnati Reds", 1, "Kansas City Royals", 5, "2026-06-02")
    # stored 1 = graded on yesterday's Reds win; its own game was a Reds loss
    fix = mod.plan_fix(_stored_row(1), [YESTERDAY_REDS_WIN, today_royals_win])
    assert fix is not None and (fix.stored, fix.correct) == (1, 0)


def test_repair_leaves_correct_and_misdated_rows_alone():
    mod = _repair_module()
    today_royals_win = _score("Cincinnati Reds", 1, "Kansas City Royals", 5, "2026-06-02")
    assert mod.plan_fix(_stored_row(0), [YESTERDAY_REDS_WIN, today_royals_win]) is None
    # no own-date game exists: the adjacent-day grade is never "repaired"
    assert mod.plan_fix(_stored_row(0), [YESTERDAY_REDS_WIN]) is None


@pytest.mark.parametrize("market_id,expected", [
    ("kalshi:KXMLBGAME-26JUL11TORSD-TOR", "2026-07-11"),
    ("kalshi:KXMLBGAME-26JUN021910KCCIN-KC", "2026-06-02"),
    ("polymarket_us:asc-mlb-tor-sd-2026-07-10-neg-1pt5:no", "2026-07-10"),
    ("kalshi:NODATE", None),
    (None, None),
])
def test_repair_contract_date(market_id, expected):
    assert _repair_module().contract_date(market_id) == expected


def test_repair_skips_rows_whose_contract_is_another_game():
    """A row priced against the adjacent game (ticker dated event_date+1)
    settles on THAT game — re-grading it onto event_date would overwrite a
    correct grade (review finding on #322)."""
    mod = _repair_module()
    row = {**_stored_row(1), "market_id": "kalshi:KXMLBGAME-26JUN03CINKC-CIN"}
    assert mod.contract_date_mismatch(row) is True
    assert mod.contract_date_mismatch(_stored_row(1) | {"market_id": "kalshi:KXMLBGAME-26JUN02CINKC-CIN"}) is False


def test_postponed_own_day_game_is_pending_and_warned(caplog):
    postponed = {**_score("Cincinnati Reds", 0, "Kansas City Royals", 0, "2026-06-02",
                          completed=False), "status": "STATUS_POSTPONED"}
    assert _match_espn(_pred(), [YESTERDAY_REDS_WIN, postponed]) is None
