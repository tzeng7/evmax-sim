"""Indexed matching must return exactly what the per-market scans returned.

2026-11-20: MatchingEngine.match_all rescanned every sharp record 5-6 times
per market (advance filter, exact loop, fuzzy date filter, result lookup,
nearest-line loops), and the fuzzy step recomputed every candidate's team
string per query. Soccer (1,039 markets x 1,718 Pinnacle records) took
0.84 s per pass, and the scan makes two passes (coordinator + EVGapAgent),
all on the event loop. match_all now builds one index per batch.

The contract is BEHAVIORAL IDENTITY: same record, same confidence, same
order. Ties are decided by list order (rapidfuzz extractOne and the nearest-
date tie-break both keep the first candidate), so every index list keeps the
original order. These tests pin that against the unindexed reference.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from evmax.matching import fuzzy
from evmax.matching.engine import MatchingEngine, _SharpIndex
from evmax.matching.fuzzy import (
    FuzzyKeyIndex,
    fuzzy_match_event_keys,
    fuzzy_match_event_keys_indexed,
)
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds

UTC = timezone.utc

# ---------------------------------------------------------------------------
# fuzzy: indexed filter == linear filter
# ---------------------------------------------------------------------------

_TEAMS = [
    "arsenal_vs_chelsea", "chelsea_vs_arsenal", "man_city_vs_man_united",
    "manchester_city_vs_manchester_united", "nets_vs_celtics", "nuggets_vs_celtics",
    "lakers_vs_warriors", "warriors_vs_lakers", "real_madrid_vs_barcelona",
    "real_madrid_vs_barca", "bayern_vs_dortmund", "bayern_munich_vs_dortmund",
]
_DATES = [
    "2026-11-19", "2026-11-20", "2026-11-21", "2026-11-22", "2026-11-23",
    "", "unknown", "not-a-date", "20261121",
]
_SECTORS = ["soccer", "tennis", "nba", "baseball"]  # non-ET and ET sectors
_SUFFIXES = ["", "", "::spread", "::total::2.5", "::advance"]


def _random_key(rng: random.Random) -> str:
    return (
        f"{rng.choice(_SECTORS)}::{rng.choice(_DATES)}::{rng.choice(_TEAMS)}"
        f"{rng.choice(_SUFFIXES)}"
    )


class TestFuzzyIndexEquivalence:
    @pytest.mark.parametrize("seed", range(40))
    def test_random_candidates_and_queries(self, seed: int) -> None:
        rng = random.Random(seed)
        keys = [_random_key(rng) for _ in range(rng.randint(0, 60))]
        # Odd keys the linear filter must also skip or accept.
        keys += ["nodelimiter", "soccer", "soccer::"][: rng.randint(0, 3)]
        rng.shuffle(keys)
        index = FuzzyKeyIndex(keys)
        for _ in range(30):
            query = (
                f"{rng.choice(_SECTORS)}::{rng.choice(_DATES)}::{rng.choice(_TEAMS)}"
            )
            assert fuzzy_match_event_keys_indexed(query, index) == \
                fuzzy_match_event_keys(query, keys), (query, keys)

    def test_window_is_plus_minus_one_day_with_undated_keys(self) -> None:
        keys = [
            "soccer::2026-11-18::arsenal_vs_chelsea",  # 2 days out: excluded
            "soccer::unknown::arsenal_vs_chelsea",     # undated: always passes
            "soccer::2026-11-21::arsenal_vs_chelsea",
        ]
        index = FuzzyKeyIndex(keys)
        got = [k for _, k, _ in index.within_one_day("soccer", "2026-11-20")]
        assert got == keys[1:]  # original order kept

    def test_unknown_query_date_accepts_whole_sector(self) -> None:
        keys = ["soccer::2026-01-01::a_vs_b", "tennis::2026-01-01::a_vs_b"]
        index = FuzzyKeyIndex(keys)
        assert [k for _, k, _ in index.within_one_day("soccer", "unknown")] == keys[:1]

    def test_et_sector_needs_the_exact_day(self) -> None:
        keys = ["nba::2026-11-21::lakers_vs_warriors"]
        index = FuzzyKeyIndex(keys)
        assert fuzzy_match_event_keys_indexed(
            "nba::2026-11-20::lakers_vs_warriors", index) is None
        assert fuzzy_match_event_keys_indexed(
            "nba::2026-11-21::warriors_vs_lakers", index)[0] == keys[0]

    def test_duplicate_team_strings_scored_once_same_winner(self, monkeypatch) -> None:
        seen: list[list[str]] = []
        real = fuzzy.fuzzy_match

        def spy(query, candidates, threshold=fuzzy.DEFAULT_THRESHOLD):
            seen.append(list(candidates))
            return real(query, candidates, threshold)

        monkeypatch.setattr(fuzzy, "fuzzy_match", spy)
        keys = [
            "soccer::2026-11-21::arsenal_vs_chelsea",
            "soccer::2026-11-21::arsenal_vs_chelsea::spread",
            "soccer::2026-11-21::arsenal_vs_chelsea::total::2.5",
        ]
        got = fuzzy_match_event_keys_indexed(
            "soccer::2026-11-21::chelsea_vs_arsenal", FuzzyKeyIndex(keys))
        assert seen == [["arsenal vs chelsea"]]
        # The FIRST key carrying the winning teams still wins.
        assert got is not None and got[0] == keys[0]


# ---------------------------------------------------------------------------
# engine: match_all (indexed) == per-market match (reference scans)
# ---------------------------------------------------------------------------


def _market(mid: str, sector: str, home: str, away: str, day: int,
            mtype: MarketType = MarketType.moneyline, line: float | None = None,
            yes_team: str | None = None) -> PredictionMarket:
    return PredictionMarket(
        id=mid, source=MarketSource.kalshi, sector=sector, ticker=mid,
        yes_price=0.5, no_price=0.52, team_home=home, team_away=away,
        yes_team=yes_team or home, market_type=mtype, line=line,
        event_date=datetime(2026, 11, day, 12, 0, tzinfo=UTC),
    )


def _sharp(event_id: str, sector: str, home: str, away: str, day: int,
           prob: float = 0.5, total_line: float | None = None,
           spread_line: float | None = None) -> SharpOdds:
    return SharpOdds(
        event_id=event_id, book=SharpBook.pinnacle, sector=sector,
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=1.9, outcome_b_decimal=1.9,
        true_prob_a=prob, true_prob_b=1 - prob, margin=0.03,
        total_line=total_line, spread_line=spread_line,
        event_date=datetime(2026, 11, day, 20, 0, tzinfo=UTC),
    )


class _ReferenceEngine(MatchingEngine):
    """match_all through the single-market ``match`` (a fresh index per
    market): the shared batch index must not leak state between markets.
    Identity with the pre-index linear scans was checked on 3,747 real matched
    pairs (11 captured scan batches, see the PR); the fuzzy filter's identity
    is pinned against the linear reference above."""

    def match_all(self, markets, sharp_odds_list):
        out = []
        for m in markets:
            r = self.match(m, sharp_odds_list)
            if r:
                out.append((m.id, r[0].event_id, r[1]))
        return out


def _pairs(result):
    return [(m.id, s.event_id, c) for m, s, c in result]


class TestEngineIndex:
    def test_duplicate_event_ids_return_the_first_record(self) -> None:
        first = _sharp("nba::2026-11-21::lakers_vs_warriors", "nba", "Lakers", "Warriors", 21, 0.6)
        second = _sharp("nba::2026-11-21::lakers_vs_warriors", "nba", "Lakers", "Warriors", 21, 0.4)
        m = _market("m1", "nba", "Lakers", "Warriors", 21)
        [(_, so, conf)] = MatchingEngine().match_all([m], [first, second])
        assert so is first and conf == 100.0

    def test_advance_partition_is_kept(self) -> None:
        reg = _sharp("worldcup::2026-11-21::brazil_vs_france", "worldcup", "Brazil", "France", 21)
        adv = _sharp("worldcup::2026-11-21::brazil_vs_france::advance", "worldcup",
                     "Brazil", "France", 21)
        ml = _market("ml", "worldcup", "Brazil", "France", 21)
        ad = _market("ad", "worldcup", "Brazil", "France", 21, mtype=MarketType.advance)
        got = {m.id: s.event_id for m, s, _ in MatchingEngine().match_all([ml, ad], [adv, reg])}
        assert got == {"ml": reg.event_id, "ad": adv.event_id}

    def test_nearest_total_only_sees_its_own_game(self) -> None:
        base = "nba::2026-11-21::lakers_vs_warriors"
        other = "nba::2026-11-21::celtics_vs_knicks"
        sharp = [
            _sharp(f"{other}::total::221.0", "nba", "Celtics", "Knicks", 21, total_line=221.0),
            _sharp(f"{base}::total::223.0", "nba", "Lakers", "Warriors", 21, total_line=223.0),
            _sharp(f"{base}::total::220.0", "nba", "Lakers", "Warriors", 21, total_line=220.0),
        ]
        m = _market("t", "nba", "Lakers", "Warriors", 21, mtype=MarketType.total, line=221.5)
        [(_, so, conf)] = MatchingEngine().match_all([m], sharp)
        # 223.0 and 220.0 are both 1.5 away; the strict "<" keeps the first.
        assert so.event_id == f"{base}::total::223.0" and conf == pytest.approx(87.5)

    def test_empty_sharp_list(self) -> None:
        assert MatchingEngine().match_all([_market("m", "nba", "A", "B", 21)], []) == []

    def test_partition_groups_spread_main_and_alt_rungs_in_order(self) -> None:
        base = "nfl::2026-11-22::chiefs_vs_bills"
        recs = [
            _sharp(f"{base}::spread::-6.5", "nfl", "Chiefs", "Bills", 22, spread_line=-6.5),
            _sharp(f"{base}::spread", "nfl", "Chiefs", "Bills", 22, spread_line=-3.0),
            _sharp(f"{base}::total::47.5", "nfl", "Chiefs", "Bills", 22, total_line=47.5),
            _sharp(base, "nfl", "Chiefs", "Bills", 22),
        ]
        part = _SharpIndex(recs).partition(advance=False)
        assert [s.event_id for s in part.spreads_by_base[base]] == [
            f"{base}::spread::-6.5", f"{base}::spread"]
        assert [s.event_id for s in part.totals_by_base[base]] == [f"{base}::total::47.5"]

    @pytest.mark.parametrize("seed", range(15))
    def test_batch_index_equals_single_market_match(self, seed: int) -> None:
        rng = random.Random(seed)
        teams = [("Arsenal", "Chelsea"), ("Man City", "Man United"),
                 ("Real Madrid", "Barcelona"), ("Bayern", "Dortmund")]
        sharp, markets = [], []
        for i in range(rng.randint(5, 25)):
            h, a = rng.choice(teams)
            if rng.random() < 0.5:
                h, a = a, h
            day = rng.choice([20, 21, 22])
            key = f"soccer::2026-11-{day}::{h.lower().replace(' ', '_')}_vs_{a.lower().replace(' ', '_')}"
            sharp.append(_sharp(key + rng.choice(["", "", "::total::2.5", "::total::3.5"]),
                                "soccer", h, a, day, total_line=rng.choice([2.5, 3.5])))
        for i in range(rng.randint(5, 25)):
            h, a = rng.choice(teams)
            mtype = rng.choice([MarketType.moneyline, MarketType.total])
            markets.append(_market(
                f"m{i}", "soccer", h, a, rng.choice([19, 20, 21, 22, 23]), mtype=mtype,
                line=rng.choice([2.5, 3.0, 4.0]) if mtype == MarketType.total else None,
                yes_team=rng.choice([h, a]),
            ))
        ref = _ReferenceEngine().match_all(markets, sharp)
        got = _pairs(MatchingEngine().match_all(markets, sharp))
        # match_all also dedups; every surviving pair must be the reference's
        # match for that market, and every reference match must survive or be
        # a dedup of an identical (event, side, line) key.
        ref_by_market = {mid: (eid, c) for mid, eid, c in ref}
        for mid, eid, c in got:
            assert ref_by_market[mid] == (eid, c)
