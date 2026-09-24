"""A market may only match a sharp event on its OWN game day.

Regression for the series cross-match (2026-04 → 09): the fuzzy fallback
accepted any sharp key within ±1 day (and any date at all when that window was
empty), so Kalshi's NEXT game of a series matched Pinnacle's current game as
soon as Pinnacle had not yet posted the next one. The match_all dedup then kept
the wrong market for late starts, because the noon-UTC ticker anchor of the
NEXT day sits closer to a 20:40 ET (00:40Z) first pitch than the correct
day's anchor does. 154 Kalshi baseball moneyline rows were priced this way.

For ET-dated sectors (`uses_et_game_day`) both sides already sit on the
US/Eastern game day, so the date must match exactly. Other sectors compare a
local/US ticker date with a UTC start date and keep their tolerance.
"""

from datetime import datetime, timezone

import pytest

from evmax.clients.time_util import kalshi_game_day, uses_et_game_day
from evmax.matching.engine import MatchingEngine
from evmax.matching.fuzzy import fuzzy_match_event_keys
from evmax.matching.normalizer import NameNormalizer
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds

UTC = timezone.utc


def kalshi_market(ticker: str, sector: str, home: str, away: str,
                  y: int, mo: int, d: int, yes_team: str | None = None) -> PredictionMarket:
    """A Kalshi game market as the client builds it: noon-UTC ticker anchor."""
    return PredictionMarket(
        id=f"kalshi:{ticker}",
        source=MarketSource.kalshi,
        sector=sector,
        ticker=ticker,
        yes_price=0.50,
        no_price=0.52,
        team_home=home,
        team_away=away,
        yes_team=yes_team,
        event_date=datetime(y, mo, d, 12, 0, tzinfo=UTC),
    )


def pinnacle_event(sector: str, home: str, away: str, start: datetime,
                   suffix: str = "") -> SharpOdds:
    """A Pinnacle record keyed the way the guest client keys it."""
    key = NameNormalizer(sector).normalize_event_key(
        home, away, kalshi_game_day(start, sector), sector,
    )
    return SharpOdds(
        event_id=key + suffix,
        book=SharpBook.pinnacle,
        sector=sector,
        outcome_a_label=home,
        outcome_b_label=away,
        outcome_a_decimal=1.91,
        outcome_b_decimal=2.00,
        true_prob_a=0.51,
        true_prob_b=0.49,
        margin=0.03,
        event_date=start,
    )


# Pinnacle has only tonight's late game: AZ @ COL, 2026-09-23 20:40 ET.
LATE_START = datetime(2026, 9, 24, 0, 40, tzinfo=UTC)


class TestBaseballSeriesBackToBack:
    """The exact live failure: KXMLBGAME-26SEP241510AZCOL-COL was priced off
    baseball::2026-09-23::rockies_vs_diamondbacks (the previous game)."""

    def _sharp(self):
        return pinnacle_event("baseball", "Colorado Rockies", "Arizona Diamondbacks", LATE_START)

    def test_sharp_key_sits_on_the_et_game_day(self):
        assert self._sharp().event_id == "baseball::2026-09-23::rockies_vs_diamondbacks"

    def test_next_game_market_does_not_match_tonights_game(self):
        engine = MatchingEngine()
        nxt = kalshi_market("KXMLBGAME-26SEP241510AZCOL-COL", "baseball", "COL", "AZ", 2026, 9, 24)
        assert engine.match(nxt, [self._sharp()]) is None

    def test_next_game_market_with_swapped_teams_does_not_match(self):
        """Swapped home/away forces the fuzzy path — the path that crossed."""
        engine = MatchingEngine()
        nxt = kalshi_market("KXMLBGAME-26SEP241510AZCOL-COL", "baseball", "AZ", "COL", 2026, 9, 24)
        assert engine.match(nxt, [self._sharp()]) is None

    def test_previous_game_market_does_not_match_tomorrows_game(self):
        """The reverse direction (ticker a day EARLIER than Pinnacle's game)."""
        engine = MatchingEngine()
        tomorrow = pinnacle_event("baseball", "Colorado Rockies", "Arizona Diamondbacks",
                                  datetime(2026, 9, 24, 19, 10, tzinfo=UTC))
        today = kalshi_market("KXMLBGAME-26SEP232040AZCOL-COL", "baseball", "AZ", "COL", 2026, 9, 23)
        assert engine.match(today, [tomorrow]) is None

    def test_same_game_market_still_matches_exactly(self):
        engine = MatchingEngine()
        same = kalshi_market("KXMLBGAME-26SEP232040AZCOL-COL", "baseball", "COL", "AZ", 2026, 9, 23)
        so, conf = engine.match(same, [self._sharp()])
        assert so.event_id == "baseball::2026-09-23::rockies_vs_diamondbacks"
        assert conf == 100.0

    def test_same_game_market_still_fuzzy_matches_with_swapped_teams(self):
        """The legit fuzzy use — reversed team order on the SAME game day."""
        engine = MatchingEngine()
        same = kalshi_market("KXMLBGAME-26SEP232040AZCOL-COL", "baseball", "AZ", "COL", 2026, 9, 23)
        so, conf = engine.match(same, [self._sharp()])
        assert so.event_id == "baseball::2026-09-23::rockies_vs_diamondbacks"
        assert conf >= 88

    def test_match_all_keeps_the_same_game_market(self):
        """Both series markets listed, only tonight's game posted: the result is
        tonight's market. Before the fix match_all returned ONLY tomorrow's
        market (fuzzy ±1 matched it, the noon-anchor dedup preferred it)."""
        engine = MatchingEngine()
        today = kalshi_market("KXMLBGAME-26SEP232040AZCOL-COL", "baseball", "COL", "AZ",
                              2026, 9, 23, yes_team="rockies")
        nxt = kalshi_market("KXMLBGAME-26SEP241510AZCOL-COL", "baseball", "COL", "AZ",
                            2026, 9, 24, yes_team="rockies")
        results = engine.match_all([nxt, today], [self._sharp()])
        assert [m.id for m, _, _ in results] == ["kalshi:KXMLBGAME-26SEP232040AZCOL-COL"]

    def test_both_games_posted_each_market_gets_its_own_game(self):
        engine = MatchingEngine()
        tomorrow = pinnacle_event("baseball", "Colorado Rockies", "Arizona Diamondbacks",
                                  datetime(2026, 9, 24, 19, 10, tzinfo=UTC))
        today = kalshi_market("KXMLBGAME-26SEP232040AZCOL-COL", "baseball", "AZ", "COL", 2026, 9, 23)
        nxt = kalshi_market("KXMLBGAME-26SEP241510AZCOL-COL", "baseball", "AZ", "COL", 2026, 9, 24)
        results = {m.id: s.event_id for m, s, _ in engine.match_all([today, nxt], [self._sharp(), tomorrow])}
        assert results == {
            "kalshi:KXMLBGAME-26SEP232040AZCOL-COL": "baseball::2026-09-23::rockies_vs_diamondbacks",
            "kalshi:KXMLBGAME-26SEP241510AZCOL-COL": "baseball::2026-09-24::rockies_vs_diamondbacks",
        }


# Series / back-to-back fixtures for every ET-dated sector: Kalshi lists the
# NEXT meeting of the same pair while Pinnacle only has the current one.
# (sector, pinnacle home, pinnacle away, kalshi home code, kalshi away code,
#  pinnacle start, next-meeting kalshi date)
SERIES_FIXTURES = [
    # NHL playoff back-to-back: G1 Fri 21:00 ET, G2 Sat.
    ("nhl", "Dallas Stars", "Edmonton Oilers", "EDM", "DAL",
     datetime(2026, 5, 23, 1, 0, tzinfo=UTC), (2026, 5, 23)),
    # NBA playoffs: G2 two days later — the old any-date fallback crossed this
    # whenever no other NBA game sat within ±1 day of G2.
    ("nba", "Oklahoma City Thunder", "Denver Nuggets", "DEN", "OKC",
     datetime(2026, 5, 7, 1, 30, tzinfo=UTC), (2026, 5, 8)),
    # WNBA two-game set on consecutive days.
    ("wnba", "Seattle Storm", "Golden State Valkyries", "GS", "SEA",
     datetime(2026, 7, 11, 2, 0, tzinfo=UTC), (2026, 7, 11)),
    # MLB day-after game (early start, so the dedup alone would not save it).
    ("baseball", "St. Louis Cardinals", "Cleveland Guardians", "CLE", "STL",
     datetime(2026, 4, 14, 23, 45, tzinfo=UTC), (2026, 4, 15)),
    # NCAAB is ET-dated too; a next-day market must not borrow a game.
    ("ncaab", "Duke", "North Carolina", "UNC", "DUKE",
     datetime(2026, 3, 13, 23, 0, tzinfo=UTC), (2026, 3, 14)),
]


@pytest.mark.parametrize(
    "sector,p_home,p_away,k_home,k_away,start,next_day", SERIES_FIXTURES,
    ids=[f[0] for f in SERIES_FIXTURES],
)
class TestEtSectorSeriesNeverCrossMatch:
    def test_next_meeting_is_not_matched(self, sector, p_home, p_away, k_home, k_away, start, next_day):
        engine = MatchingEngine()
        sharp = pinnacle_event(sector, p_home, p_away, start)
        for home, away in ((k_home, k_away), (k_away, k_home)):
            nxt = kalshi_market(f"KX-{sector}-NEXT", sector, home, away, *next_day)
            assert engine.match(nxt, [sharp]) is None, (sector, home, away)

    def test_same_game_day_still_matches(self, sector, p_home, p_away, k_home, k_away, start, next_day):
        engine = MatchingEngine()
        sharp = pinnacle_event(sector, p_home, p_away, start)
        game_day = datetime.fromisoformat(kalshi_game_day(start, sector))
        for home, away in ((k_home, k_away), (k_away, k_home)):
            same = kalshi_market(f"KX-{sector}-SAME", sector, home, away,
                                 game_day.year, game_day.month, game_day.day)
            result = engine.match(same, [sharp])
            assert result is not None, (sector, home, away)
            assert result[0].event_id == sharp.event_id


class TestMatchAllDedupRanksGameDayFirst:
    """Defense in depth: whatever path lets two game days reach one sharp
    record, the dedup keeps the market on the sharp event's game day."""

    def test_late_start_keeps_same_day_market_not_nearer_noon_anchor(self, monkeypatch):
        engine = MatchingEngine()
        sharp = pinnacle_event("baseball", "Colorado Rockies", "Arizona Diamondbacks", LATE_START)
        today = kalshi_market("KXMLBGAME-26SEP232040AZCOL-COL", "baseball", "COL", "AZ",
                              2026, 9, 23, yes_team="rockies")
        nxt = kalshi_market("KXMLBGAME-26SEP241510AZCOL-COL", "baseball", "COL", "AZ",
                            2026, 9, 24, yes_team="rockies")
        # Tomorrow's noon anchor is nearer the 00:40Z start (11h20m vs 12h40m).
        assert abs((nxt.event_date - LATE_START).total_seconds()) < abs(
            (today.event_date - LATE_START).total_seconds())
        monkeypatch.setattr(engine, "match", lambda market, sharp_list: (sharp, 100.0))
        for order in ([nxt, today], [today, nxt]):
            results = engine.match_all(order, [sharp])
            assert [m.id for m, _, _ in results] == ["kalshi:KXMLBGAME-26SEP232040AZCOL-COL"]

    def test_equal_game_day_falls_back_to_time_then_confidence(self, monkeypatch):
        engine = MatchingEngine()
        sharp = pinnacle_event("baseball", "Colorado Rockies", "Arizona Diamondbacks", LATE_START)
        a = kalshi_market("KXMLBGAME-26SEP231310AZCOL-COL", "baseball", "COL", "AZ",
                          2026, 9, 23, yes_team="rockies")
        b = kalshi_market("KXMLBGAME-26SEP232040AZCOL-COL", "baseball", "COL", "AZ",
                          2026, 9, 23, yes_team="rockies")
        conf = {a.id: 90.0, b.id: 100.0}
        monkeypatch.setattr(engine, "match", lambda market, sharp_list: (sharp, conf[market.id]))
        results = engine.match_all([a, b], [sharp])
        assert [m.id for m, _, _ in results] == [b.id]


class TestNonEtSectorsKeepTheirTolerance:
    """Sectors whose Kalshi date and Pinnacle date sit on different calendars
    keep the ±1-day window — a strict rule would drop these same-game matches
    (29 soccer + 44 worldcup logged rows were exactly this)."""

    def test_mls_evening_kickoff_matches_across_the_utc_day(self):
        # RSL vs San Diego, Sat 2026-04-18 19:30 MDT = 2026-04-19 01:30Z.
        engine = MatchingEngine()
        sharp = pinnacle_event("soccer", "San Diego FC", "Real Salt Lake",
                               datetime(2026, 4, 19, 1, 30, tzinfo=UTC))
        assert sharp.event_id.startswith("soccer::2026-04-19::")
        market = kalshi_market("KXMLSGAME-26APR18RSLSD-RSL", "soccer",
                               "Real Salt Lake", "San Diego FC", 2026, 4, 18)
        result = engine.match(market, [sharp])
        assert result is not None and result[0].event_id == sharp.event_id

    def test_esports_rematch_prefers_the_same_date(self):
        """Two meetings of the same pair inside the window: the nearest date
        wins, not whichever key is listed first."""
        keys = [
            "lol::2026-07-03::top_vs_g2",
            "lol::2026-07-04::top_vs_g2",
        ]
        result = fuzzy_match_event_keys("lol::2026-07-04::g2_vs_top", keys)
        assert result is not None and result[0] == "lol::2026-07-04::top_vs_g2"


class TestFuzzyDateFilter:
    def test_et_sector_requires_the_exact_date(self):
        keys = ["nba::2026-05-06::nuggets_vs_thunder"]
        assert fuzzy_match_event_keys("nba::2026-05-06::thunder_vs_nuggets", keys) is not None
        assert fuzzy_match_event_keys("nba::2026-05-07::thunder_vs_nuggets", keys) is None
        assert fuzzy_match_event_keys("nba::2026-05-05::thunder_vs_nuggets", keys) is None

    def test_et_sector_has_no_any_date_fallback(self):
        """Old behavior: nothing within ±1 day → try every date in the sector."""
        keys = ["nba::2026-05-06::nuggets_vs_thunder"]
        assert fuzzy_match_event_keys("nba::2026-05-10::thunder_vs_nuggets", keys) is None

    def test_et_sector_unknown_date_never_matches(self):
        keys = ["baseball::unknown::rockies_vs_diamondbacks"]
        assert fuzzy_match_event_keys("baseball::2026-09-23::diamondbacks_vs_rockies", keys) is None
        assert fuzzy_match_event_keys(
            "baseball::unknown::diamondbacks_vs_rockies",
            ["baseball::2026-09-23::rockies_vs_diamondbacks"],
        ) is None

    def test_suffix_keys_use_the_date_segment(self):
        keys = ["worldcup::2026-07-06::england_vs_mexico::advance"]
        result = fuzzy_match_event_keys("worldcup::2026-07-05::mexico_vs_england::advance", keys)
        assert result is not None  # worldcup: local/US ticker day vs UTC key day


class TestUsesEtGameDay:
    @pytest.mark.parametrize("sector", ["nba", "wnba", "nfl", "baseball", "mlb", "nhl",
                                        "ncaab", "ncaaw", "ncaaf", "ufc"])
    def test_et_sectors(self, sector):
        assert uses_et_game_day(sector) is True

    @pytest.mark.parametrize("sector", ["soccer", "worldcup", "tennis", "lol", "cs2", "valorant", ""])
    def test_other_sectors(self, sector):
        assert uses_et_game_day(sector) is False
