"""Tests for the player-prop pipeline.

Covers:
  - PropMatcher (kalshi market ↔ sharp odds matching for props)
  - nba_props_cache pure functions (lookup_player_team, _opponent_adjustment,
    compute_prop_prob_cached)
  - prop_resolver pure helpers (_extract_stat, _normalize_for_match)

Network-dependent code paths (refresh_props_cache, fetch_player_stats,
resolve_prop_observations) are intentionally not tested here — they hit
stats.nba.com / ESPN and live inside the "network" tier.
"""

from __future__ import annotations

import time

import pytest

from evmax.agents.cleanup import prop_resolver
from evmax.agents.cleanup.prop_resolver import (
    _extract_stat,
    _normalize_for_match,
)
from evmax.clients import nba_props_cache
from evmax.clients.nba_props_cache import (
    PropResult,
    _opponent_adjustment,
    compute_prop_prob_cached,
    lookup_player_team,
)
from evmax.matching.prop_matcher import (
    PLAYER_MATCH_THRESHOLD,
    THRESHOLD_TOLERANCE,
    PropMatcher,
)
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


# ===========================================================================
# PropMatcher
# ===========================================================================


def _prop_market(
    player="LeBron James",
    stat="points",
    threshold=24.5,
) -> PredictionMarket:
    return PredictionMarket(
        id=f"kalshi:{player}-{stat}",
        source=MarketSource.kalshi,
        sector="nba",
        market_type=MarketType.player_prop,
        title=f"{player} {threshold}+ {stat}",
        ticker=f"KXNBAPTS-{player.upper()}",
        yes_price=0.48,
        no_price=0.52,
        volume_usd=5_000.0,
        player_name=player,
        stat_type=stat,
        threshold=threshold,
    )


def _prop_sharp(
    player="LeBron James",
    stat="points",
    line=24.5,
    prob_over=0.55,
) -> SharpOdds:
    return SharpOdds(
        event_id=f"nba::2026-03-20::prop::{player}::{stat}::{line}",
        book=SharpBook.pinnacle,
        sector="nba",
        outcome_a_label="over",
        outcome_b_label="under",
        outcome_a_decimal=1.0,
        outcome_b_decimal=1.0,
        true_prob_a=0.0,
        true_prob_b=0.0,
        true_prob_over=prob_over,
        true_prob_under=1.0 - prob_over,
        total_line=line,
        margin=0.0,
        prop_player_name=player,
        prop_stat_type=stat,
        prop_l15_games=15,
    )


class TestPropMatcher:
    def test_exact_match_returns_high_confidence(self):
        m = _prop_market()
        s = _prop_sharp()
        result = PropMatcher().match(m, [s])
        assert result is not None
        sharp, conf = result
        assert sharp is s
        assert conf == pytest.approx(1.0, abs=1e-6)

    def test_no_sharps_returns_none(self):
        assert PropMatcher().match(_prop_market(), []) is None

    def test_stat_mismatch_returns_none(self):
        result = PropMatcher().match(
            _prop_market(stat="points"),
            [_prop_sharp(stat="rebounds")],
        )
        assert result is None

    def test_threshold_outside_tolerance_returns_none(self):
        # 24.5 vs 26.0 is 1.5 apart, > THRESHOLD_TOLERANCE (0.5)
        result = PropMatcher().match(
            _prop_market(threshold=24.5),
            [_prop_sharp(line=26.0)],
        )
        assert result is None
        assert THRESHOLD_TOLERANCE == 0.5  # sanity pin

    def test_threshold_inside_tolerance_matches(self):
        result = PropMatcher().match(
            _prop_market(threshold=24.5),
            [_prop_sharp(line=24.0)],
        )
        assert result is not None

    def test_market_missing_player_returns_none(self):
        m = _prop_market()
        m = m.model_copy(update={"player_name": None})
        assert PropMatcher().match(m, [_prop_sharp()]) is None

    def test_sharp_missing_total_line_skipped(self):
        s = _prop_sharp()
        s.total_line = None  # type: ignore[misc]
        assert PropMatcher().match(_prop_market(), [s]) is None

    def test_low_name_similarity_rejected(self):
        # Completely different name → below PLAYER_MATCH_THRESHOLD
        result = PropMatcher().match(
            _prop_market(player="LeBron James"),
            [_prop_sharp(player="Nikola Jokic")],
        )
        assert result is None
        assert PLAYER_MATCH_THRESHOLD >= 80  # sanity pin

    def test_near_name_match_accepted(self):
        """Minor typo above the fuzzy threshold still matches."""
        result = PropMatcher().match(
            _prop_market(player="Stephen Curry"),
            [_prop_sharp(player="Stephen Currry")],  # one-letter typo
        )
        assert result is not None

    def test_picks_best_of_multiple_candidates(self):
        exact = _prop_sharp(player="LeBron James")
        near = _prop_sharp(player="Lebron Jaems")  # typo
        result = PropMatcher().match(_prop_market(player="LeBron James"), [near, exact])
        assert result is not None
        sharp, _ = result
        assert sharp is exact

    def test_match_all_returns_all_matches(self):
        markets = [
            _prop_market(player="LeBron James", stat="points"),
            _prop_market(player="Stephen Curry", stat="threes", threshold=4.5),
        ]
        sharps = [
            _prop_sharp(player="LeBron James", stat="points"),
            _prop_sharp(player="Stephen Curry", stat="threes", line=4.5),
            _prop_sharp(player="Nikola Jokic", stat="assists", line=8.5),  # noise
        ]
        results = PropMatcher().match_all(markets, sharps)
        assert len(results) == 2
        players = {m.player_name for m, _, _ in results}
        assert players == {"LeBron James", "Stephen Curry"}


# ===========================================================================
# nba_props_cache — lookup_player_team
# ===========================================================================


@pytest.fixture
def fake_cache(monkeypatch):
    """Install a minimal fake props cache into the module-level memo."""
    data = {
        "fetched_at": time.time(),
        "players": {
            "LeBron James": {
                "team": "LAL",
                "n_games": 15,
                "stats": {
                    "PTS": [30, 28, 25, 27, 22, 33, 31, 26, 24, 29, 32, 27, 25, 28, 30],
                    "REB": [8, 7, 9, 6, 8, 10, 7, 9, 8, 7, 6, 9, 8, 7, 8],
                    "AST": [8, 9, 7, 8, 10, 9, 7, 8, 9, 10, 8, 7, 9, 8, 9],
                    "MIN": [35, 34, 36, 33, 35, 37, 36, 34, 35, 36, 34, 35, 36, 35, 35],
                },
            },
            "Jayson Tatum": {
                "team": "BOS",
                "n_games": 15,
                "stats": {
                    "PTS": [28, 30, 25, 27, 32, 26, 29, 24, 28, 30, 27, 25, 31, 29, 28],
                    "MIN": [36] * 15,
                },
            },
            "Rookie Guy": {
                "team": "LAL",
                "n_games": 3,  # below _MIN_GAMES
                "stats": {"PTS": [5, 7, 6], "MIN": [12, 10, 14]},
            },
        },
        "team_stats": {},
        "league_avg": {},
        "schedule": [],
    }
    monkeypatch.setattr(nba_props_cache, "_mem_cache", data)
    monkeypatch.setattr(nba_props_cache, "_mem_cache_time", time.monotonic())
    return data


class TestLookupPlayerTeam:
    def test_known_player_returns_nickname(self, fake_cache):
        assert lookup_player_team("LeBron James") == "lakers"
        assert lookup_player_team("Jayson Tatum") == "celtics"

    def test_unknown_player_returns_none(self, fake_cache):
        assert lookup_player_team("Nobody") is None

    def test_empty_cache_returns_none(self, monkeypatch):
        monkeypatch.setattr(nba_props_cache, "_mem_cache", None)
        monkeypatch.setattr(nba_props_cache, "_mem_cache_time", 0.0)
        # Also make sure the on-disk path doesn't accidentally hit a real file.
        monkeypatch.setattr(
            nba_props_cache,
            "_CACHE_PATH",
            nba_props_cache._CACHE_PATH.parent / "nonexistent_cache.json",
        )
        assert lookup_player_team("LeBron James") is None

    def test_unmapped_abbreviation_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            nba_props_cache,
            "_mem_cache",
            {
                "fetched_at": time.time(),
                "players": {"X": {"team": "ZZZ", "n_games": 15, "stats": {}}},
            },
        )
        monkeypatch.setattr(nba_props_cache, "_mem_cache_time", time.monotonic())
        assert lookup_player_team("X") is None


# ===========================================================================
# nba_props_cache — compute_prop_prob_cached
# ===========================================================================


class TestComputePropProbCached:
    def test_unknown_player_returns_none(self, fake_cache):
        assert compute_prop_prob_cached("Nobody", "points", 25.5) is None

    def test_too_few_games_returns_none(self, fake_cache):
        assert compute_prop_prob_cached("Rookie Guy", "points", 5.5) is None

    def test_unknown_stat_returns_none(self, fake_cache):
        assert compute_prop_prob_cached("LeBron James", "bogus_stat", 25.5) is None

    def test_higher_threshold_lowers_probability(self, fake_cache):
        low = compute_prop_prob_cached("LeBron James", "points", 20.5)
        high = compute_prop_prob_cached("LeBron James", "points", 35.5)
        assert low is not None and high is not None
        assert low.prob > high.prob

    def test_probability_bounded(self, fake_cache):
        result = compute_prop_prob_cached("LeBron James", "points", 25.5)
        assert result is not None
        assert 0.01 <= result.prob <= 0.99

    def test_returns_sample_size(self, fake_cache):
        result = compute_prop_prob_cached("LeBron James", "points", 25.5)
        assert result is not None
        assert result.n_games == 15

    def test_returns_prop_result(self, fake_cache):
        result = compute_prop_prob_cached("LeBron James", "points", 25.5)
        assert isinstance(result, PropResult)
        assert not result.minutes_volatile  # LeBron has consistent 34-37 min
        assert result.avg_minutes > 30

    def test_pra_combo_stat(self, fake_cache):
        """Points+Rebounds+Assists combo requires all three series."""
        result = compute_prop_prob_cached(
            "LeBron James", "points_rebounds_assists", 40.5
        )
        assert result is not None
        assert 0.01 <= result.prob <= 0.99

    def test_pra_missing_series_returns_none(self, fake_cache):
        # Jayson Tatum has no REB/AST series in the fixture
        assert compute_prop_prob_cached("Jayson Tatum", "points_rebounds_assists", 40.5) is None

    def test_minutes_volatility_detected(self, monkeypatch):
        """Player with wildly inconsistent minutes should flag as volatile."""
        data = {
            "fetched_at": time.time(),
            "players": {
                "Bench Player": {
                    "team": "IND",
                    "n_games": 10,
                    "stats": {
                        # Normally 15 min, but 2 games at 35+ min (starter rest days)
                        "PTS": [22, 8, 10, 25, 7, 9, 12, 8, 11, 9],
                        "AST": [8, 2, 3, 7, 1, 2, 3, 2, 3, 2],
                        "MIN": [35, 15, 14, 38, 16, 14, 15, 13, 16, 14],
                    },
                },
            },
            "team_stats": {},
            "league_avg": {},
            "schedule": [],
        }
        monkeypatch.setattr(nba_props_cache, "_mem_cache", data)
        monkeypatch.setattr(nba_props_cache, "_mem_cache_time", time.monotonic())

        result = compute_prop_prob_cached("Bench Player", "points", 15.5)
        assert result is not None
        assert result.minutes_volatile is True
        assert result.volatile_games >= 2

    def test_minutes_volatility_discounts_prob(self, monkeypatch):
        """Volatile minutes should produce lower probability than stable minutes."""
        stable_data = {
            "fetched_at": time.time(),
            "players": {
                "Stable Player": {
                    "team": "IND",
                    "n_games": 10,
                    "stats": {
                        "AST": [6, 5, 7, 6, 5, 6, 7, 5, 6, 7],
                        "MIN": [32, 33, 31, 34, 32, 33, 31, 32, 33, 32],
                    },
                },
            },
            "team_stats": {},
            "league_avg": {},
            "schedule": [],
        }
        monkeypatch.setattr(nba_props_cache, "_mem_cache", stable_data)
        monkeypatch.setattr(nba_props_cache, "_mem_cache_time", time.monotonic())
        stable_result = compute_prop_prob_cached("Stable Player", "assists", 4.5)

        volatile_data = {
            "fetched_at": time.time(),
            "players": {
                "Volatile Player": {
                    "team": "IND",
                    "n_games": 10,
                    "stats": {
                        # Same assist numbers but with wildly varying minutes
                        "AST": [6, 5, 7, 6, 5, 6, 7, 5, 6, 7],
                        "MIN": [38, 15, 14, 40, 16, 14, 15, 13, 16, 14],
                    },
                },
            },
            "team_stats": {},
            "league_avg": {},
            "schedule": [],
        }
        monkeypatch.setattr(nba_props_cache, "_mem_cache", volatile_data)
        monkeypatch.setattr(nba_props_cache, "_mem_cache_time", time.monotonic())
        volatile_result = compute_prop_prob_cached("Volatile Player", "assists", 4.5)

        assert stable_result is not None and volatile_result is not None
        assert not stable_result.minutes_volatile
        assert volatile_result.minutes_volatile
        # Volatile player should have discounted probability
        assert volatile_result.prob < stable_result.prob


# ===========================================================================
# nba_props_cache — _opponent_adjustment
# ===========================================================================


class TestOpponentAdjustment:
    def test_neutral_when_opponent_equals_league_avg(self):
        league = {"def_rating": 112.0}
        adj = _opponent_adjustment("points", {"def_rating": 112.0}, league)
        assert adj == pytest.approx(1.0, abs=1e-6)

    def test_weaker_defense_increases_points_adjustment(self):
        """def_rating above league avg = defense gives up more points → boost."""
        league = {"def_rating": 112.0}
        adj = _opponent_adjustment("points", {"def_rating": 118.0}, league)
        assert adj > 1.0

    def test_stronger_defense_decreases_points_adjustment(self):
        league = {"def_rating": 112.0}
        adj = _opponent_adjustment("points", {"def_rating": 105.0}, league)
        assert adj < 1.0

    def test_clamped_to_max_opp_adj(self):
        """Adjustment must stay within ±_MAX_OPP_ADJ (0.15)."""
        league = {"def_rating": 100.0}
        adj_high = _opponent_adjustment("points", {"def_rating": 200.0}, league)
        adj_low = _opponent_adjustment("points", {"def_rating": 10.0}, league)
        assert adj_high <= 1.0 + nba_props_cache._MAX_OPP_ADJ + 1e-9
        assert adj_low >= 1.0 - nba_props_cache._MAX_OPP_ADJ - 1e-9

    def test_unknown_stat_returns_neutral(self):
        assert _opponent_adjustment("bogus", {}, {}) == pytest.approx(1.0, abs=1e-6)

    def test_handles_missing_fields(self):
        # Missing both opp_stats and league_avg fields → fall through defaults
        adj = _opponent_adjustment("rebounds", {}, {})
        assert 1.0 - nba_props_cache._MAX_OPP_ADJ <= adj <= 1.0 + nba_props_cache._MAX_OPP_ADJ


# ===========================================================================
# prop_resolver._normalize_for_match
# ===========================================================================


class TestNormalizeForMatch:
    def test_lowercases(self):
        assert _normalize_for_match("LeBron James") == "lebron james"

    def test_strips_accents(self):
        assert _normalize_for_match("Luka Dončić") == "luka doncic"
        assert _normalize_for_match("Müller") == "muller"

    def test_underscores_to_spaces(self):
        assert _normalize_for_match("lebron_james") == "lebron james"

    def test_strips_jr_sr_suffix(self):
        assert _normalize_for_match("Kenyon Martin Jr.") == "kenyon martin"
        assert _normalize_for_match("Tim Hardaway Sr") == "tim hardaway"

    def test_strips_roman_numeral_suffix(self):
        assert _normalize_for_match("Larry Nance III") == "larry nance"
        assert _normalize_for_match("Gary Payton II") == "gary payton"


# ===========================================================================
# prop_resolver._extract_stat
# ===========================================================================


class TestExtractStat:
    def test_basic_stat_lookup(self):
        keys = {"points": 0, "rebounds": 1, "assists": 2}
        assert _extract_stat(["28", "9", "7"], keys, "points") == 28.0
        assert _extract_stat(["28", "9", "7"], keys, "rebounds") == 9.0

    def test_missing_key_returns_none(self):
        keys = {"rebounds": 0}
        assert _extract_stat(["5"], keys, "points") is None

    def test_non_numeric_value_returns_none(self):
        keys = {"points": 0}
        assert _extract_stat(["DNP"], keys, "points") is None

    def test_threes_parses_made_attempted_string(self):
        keys = {"threePointFieldGoalsMade-threePointFieldGoalsAttempted": 0}
        assert _extract_stat(["3-7"], keys, "threes") == 3.0
        assert _extract_stat(["0-5"], keys, "threes") == 0.0

    def test_threes_missing_returns_none(self):
        assert _extract_stat(["3-7"], {}, "threes") is None

    def test_pra_sums_components(self):
        keys = {"points": 0, "rebounds": 1, "assists": 2}
        assert _extract_stat(["30", "10", "8"], keys, "points_rebounds_assists") == 48.0

    def test_pra_missing_component_returns_none(self):
        keys = {"points": 0, "rebounds": 1}  # no assists
        assert _extract_stat(["30", "10"], keys, "points_rebounds_assists") is None

    def test_unknown_stat_type_returns_none(self):
        keys = {"points": 0}
        assert _extract_stat(["30"], keys, "bogus") is None

    def test_stat_index_out_of_range_returns_none(self):
        keys = {"points": 5}  # index beyond stats list
        assert _extract_stat(["30", "10"], keys, "points") is None

    # NFL stat keys — added for MODEL-9 resolver support
    def test_nfl_passing_yards_from_passing_group(self):
        # A "passing" stat group has passingYards in its keys
        keys = {"passingYards": 2, "passingTouchdowns": 4}
        assert _extract_stat(["20/30", "66.7", "312", "10.4", "2"], keys, "passing_yards") == 312.0
        assert _extract_stat(["20/30", "66.7", "312", "10.4", "2"], keys, "passing_tds") == 2.0

    def test_nfl_rushing_yards_from_rushing_group(self):
        keys = {"rushingYards": 1, "rushingTouchdowns": 3}
        assert _extract_stat(["18", "94", "5.2", "1"], keys, "rushing_yards") == 94.0
        assert _extract_stat(["18", "94", "5.2", "1"], keys, "rushing_tds") == 1.0

    def test_nfl_receiving_group_extracts_receptions_yards_tds(self):
        keys = {"receptions": 0, "receivingYards": 2, "receivingTouchdowns": 4}
        assert _extract_stat(["6", "41.2", "72", "12.0", "1"], keys, "receptions") == 6.0
        assert _extract_stat(["6", "41.2", "72", "12.0", "1"], keys, "receiving_yards") == 72.0
        assert _extract_stat(["6", "41.2", "72", "12.0", "1"], keys, "receiving_tds") == 1.0

    def test_nfl_anytime_td_not_derivable_from_single_group(self):
        # anytime_td is computed post-merge in fetch_player_stats; the
        # per-group _extract_stat path must return None for it.
        keys = {"rushingTouchdowns": 1}
        assert _extract_stat(["18", "94", "1"], keys, "anytime_td") is None


# ===========================================================================
# prop_resolver.fetch_player_stats — sector gating only (no network)
# ===========================================================================


class TestFetchPlayerStatsGating:
    def test_unsupported_sector_returns_empty(self):
        from datetime import date
        assert prop_resolver.fetch_player_stats("tennis", date(2026, 3, 15)) == {}


# ===========================================================================
# prop_resolver.fetch_player_stats NFL merge-across-groups
# ===========================================================================
# This test exercises the MODEL-9 refactor that makes fetch_player_stats
# accumulate per-player stats across multiple stat_groups. For NFL a QB
# appears in both the passing group and the rushing group — the merged
# row must contain passing_yards AND rushing_yards AND anytime_td (derived
# post-merge from rushing_tds + receiving_tds).


class TestFetchPlayerStatsNflMerge:
    """Feed a minimal ESPN NFL summary response through fetch_player_stats
    and assert the merge + anytime_td derivation produce correct rows."""

    NFL_SUMMARY_RESPONSE = {
        "boxscore": {
            "players": [
                {
                    "statistics": [
                        # passing group — one QB
                        {
                            "name": "passing",
                            "keys": [
                                "completions-attempts",
                                "completionPercentage",
                                "passingYards",
                                "yardsPerPassAttempt",
                                "passingTouchdowns",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Patrick Mahomes"},
                                    "stats": ["25/35", "71.4", "312", "8.9", "2"],
                                },
                            ],
                        },
                        # rushing group — same QB (rushing stats) + RB
                        {
                            "name": "rushing",
                            "keys": [
                                "rushingAttempts",
                                "rushingYards",
                                "yardsPerRushAttempt",
                                "rushingTouchdowns",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Patrick Mahomes"},
                                    "stats": ["3", "18", "6.0", "1"],
                                },
                                {
                                    "athlete": {"displayName": "Isiah Pacheco"},
                                    "stats": ["18", "94", "5.2", "0"],
                                },
                            ],
                        },
                        # receiving group — WR with a TD
                        {
                            "name": "receiving",
                            "keys": [
                                "receptions",
                                "receivingYards",
                                "yardsPerReception",
                                "receivingTouchdowns",
                            ],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Travis Kelce"},
                                    "stats": ["6", "72", "12.0", "1"],
                                },
                                {
                                    "athlete": {"displayName": "Isiah Pacheco"},
                                    "stats": ["3", "22", "7.3", "0"],
                                },
                            ],
                        },
                    ]
                }
            ]
        }
    }

    def test_qb_row_has_passing_and_rushing_stats_after_merge(self, monkeypatch):
        """A QB who shows up in both passing and rushing groups must have
        both keys set on the merged row — the pre-refactor behavior was
        to overwrite row per group, losing passing stats."""
        import httpx
        from datetime import date

        class _FakeResponse:
            def __init__(self, payload):
                self._payload = payload
            def json(self):
                return self._payload

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url, params=None):
                if "scoreboard" in url:
                    return _FakeResponse({"events": [{"id": "401547435"}]})
                if "summary" in url:
                    return _FakeResponse(
                        TestFetchPlayerStatsNflMerge.NFL_SUMMARY_RESPONSE
                    )
                return _FakeResponse({})

        monkeypatch.setattr(httpx, "Client", _FakeClient)
        rows = prop_resolver.fetch_player_stats("nfl", date(2026, 1, 7))

        assert "patrick mahomes" in rows
        mahomes = rows["patrick mahomes"]
        assert mahomes["passing_yards"] == 312.0
        assert mahomes["passing_tds"] == 2.0
        assert mahomes["rushing_yards"] == 18.0
        assert mahomes["rushing_tds"] == 1.0
        # anytime_td = rushing_tds + receiving_tds (passing TDs excluded — he
        # threw them, didn't score them). Mahomes scored 1 rushing TD.
        assert mahomes["anytime_td"] == 1.0

    def test_rb_with_receiving_stats_merges_both_groups(self, monkeypatch):
        import httpx
        from datetime import date

        class _FakeResponse:
            def __init__(self, payload):
                self._payload = payload
            def json(self):
                return self._payload

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url, params=None):
                if "scoreboard" in url:
                    return _FakeResponse({"events": [{"id": "x"}]})
                return _FakeResponse(
                    TestFetchPlayerStatsNflMerge.NFL_SUMMARY_RESPONSE
                )

        monkeypatch.setattr(httpx, "Client", _FakeClient)
        rows = prop_resolver.fetch_player_stats("nfl", date(2026, 1, 7))

        pacheco = rows["isiah pacheco"]
        assert pacheco["rushing_yards"] == 94.0
        assert pacheco["receiving_yards"] == 22.0
        assert pacheco["receptions"] == 3.0
        # He didn't score a TD — anytime_td should be present but 0
        assert pacheco["anytime_td"] == 0.0

    def test_receiver_with_td_has_correct_anytime_td(self, monkeypatch):
        import httpx
        from datetime import date

        class _FakeResponse:
            def __init__(self, payload):
                self._payload = payload
            def json(self):
                return self._payload

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url, params=None):
                if "scoreboard" in url:
                    return _FakeResponse({"events": [{"id": "x"}]})
                return _FakeResponse(
                    TestFetchPlayerStatsNflMerge.NFL_SUMMARY_RESPONSE
                )

        monkeypatch.setattr(httpx, "Client", _FakeClient)
        rows = prop_resolver.fetch_player_stats("nfl", date(2026, 1, 7))

        kelce = rows["travis kelce"]
        assert kelce["receiving_yards"] == 72.0
        assert kelce["receptions"] == 6.0
        assert kelce["receiving_tds"] == 1.0
        assert kelce["anytime_td"] == 1.0


class TestLogPropFromSharp:
    """Test that log_prop_from_sharp logs raw SharpOdds+Market pairs."""

    def test_log_prop_from_sharp_inserts(self, tmp_path, monkeypatch):
        import sqlite3
        from datetime import date, datetime

        from evmax.agents.cleanup.logger import log_prop_from_sharp
        from evmax.models.market import MarketSource, MarketType, PredictionMarket
        from evmax.models.odds import SharpBook, SharpOdds

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE IF NOT EXISTS prop_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scan_date TEXT,
            event_date TEXT,
            sector TEXT,
            player_name TEXT,
            stat_type TEXT,
            line REAL,
            kalshi_price REAL,
            sharp_prob REAL,
            ev_pct REAL,
            l15_games INTEGER,
            market_id TEXT,
            event_id TEXT,
            event_title TEXT,
            actual_value REAL,
            outcome TEXT,
            resolved_at TIMESTAMP,
            UNIQUE(scan_date, player_name, stat_type, line)
        )""")
        conn.commit()
        conn.close()

        # Monkeypatch get_connection to use our temp DB
        import contextlib

        @contextlib.contextmanager
        def _fake_conn():
            c = sqlite3.connect(str(db_path))
            try:
                yield c
            finally:
                c.close()

        monkeypatch.setattr("evmax.agents.cleanup.logger.get_connection", _fake_conn)

        sharp = SharpOdds(
            event_id="nba::2026-04-15::prop::LeBron James::points::24.5",
            book=SharpBook.pinnacle,
            sector="nba",
            outcome_a_label="over",
            outcome_b_label="under",
            outcome_a_decimal=1.0,
            outcome_b_decimal=1.0,
            true_prob_a=0.0,
            true_prob_b=0.0,
            true_prob_over=0.42,
            true_prob_under=0.58,
            total_line=24.5,
            margin=0.0,
            prop_player_name="LeBron James",
            prop_stat_type="points",
            prop_l15_games=15,
            event_date=datetime(2026, 4, 15),
        )
        market = PredictionMarket(
            id="kalshi:KXNBAPTS-LEBRON",
            source=MarketSource.kalshi,
            sector="nba",
            market_type=MarketType.player_prop,
            title="LeBron James 24.5+ points",
            yes_price=0.55,
            no_price=0.45,
            player_name="LeBron James",
            stat_type="points",
            threshold=24.5,
        )

        n = log_prop_from_sharp([(sharp, market)], scan_date=date(2026, 4, 15))
        assert n == 1

        # Verify what was written
        c = sqlite3.connect(str(db_path))
        row = c.execute("SELECT * FROM prop_observations").fetchone()
        c.close()
        assert row is not None
        # Find column values by name using a dict cursor
        c2 = sqlite3.connect(str(db_path))
        c2.row_factory = sqlite3.Row
        row2 = c2.execute("SELECT * FROM prop_observations").fetchone()
        c2.close()
        assert row2["sharp_prob"] == pytest.approx(0.42, abs=1e-6)
        assert row2["kalshi_price"] == pytest.approx(0.55, abs=1e-6)
        assert row2["player_name"] == "LeBron James"
        assert row2["stat_type"] == "points"
