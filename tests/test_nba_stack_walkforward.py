"""Tests for the point-in-time NBA stat-stack walk-forward (no network)."""
from __future__ import annotations

import json
import types
from datetime import date, timedelta

import pytest

import evmax.backtest.sources.nba_stack_walkforward as nsw


# --- snapshot schedule: the no-leak rule ---------------------------------

def _days(n, start=date(2024, 10, 22)):
    return [start + timedelta(days=i) for i in range(n)]


class TestSnapshotSchedule:
    def test_snapshot_is_strictly_before_the_game(self):
        days = _days(40)
        for every in (1, 2, 3, 7):
            sched = nsw.snapshot_schedule(days, every)
            for g, as_of in sched.items():
                assert as_of is None or as_of < g

    def test_opening_day_has_no_snapshot(self):
        days = _days(10)
        assert nsw.snapshot_schedule(days, 3)[days[0]] is None

    def test_daily_cadence_uses_previous_game_day(self):
        days = _days(5)
        sched = nsw.snapshot_schedule(days, 1)
        assert sched[days[3]] == days[2]

    def test_coarser_cadence_lags_but_never_leaks(self):
        days = _days(10)
        sched = nsw.snapshot_schedule(days, 3)  # grid: days 0, 3, 6, 9
        assert sched[days[1]] == days[0]
        assert sched[days[3]] == days[0]   # day 3 is itself a grid day -> not usable yet
        assert sched[days[4]] == days[3]
        assert sched[days[9]] == days[6]

    def test_duplicate_dates_collapse(self):
        d = _days(3)
        sched = nsw.snapshot_schedule(d + d, 1)
        assert set(sched) == set(d)

    def test_rejects_bad_cadence(self):
        with pytest.raises(ValueError):
            nsw.snapshot_schedule(_days(3), 0)


def test_season_code_label_and_regular_season_cut():
    assert nsw.season_code_label("2425") == "2024-25"
    assert nsw.is_regular_season(date(2025, 4, 13), "2425")
    assert not nsw.is_regular_season(date(2025, 4, 20), "2425")


def test_drop_preseason_removes_exhibitions_only():
    games = [{"date": date(2024, 10, 7)}, {"date": date(2024, 10, 22)}, {"date": date(2024, 12, 1)}]
    assert [g["date"] for g in nsw.drop_preseason(games, "2425")] == [date(2024, 10, 22), date(2024, 12, 1)]
    assert len(nsw.drop_preseason(games, "9999")) == 3  # unknown season: untouched


# --- date-bound, cached endpoints -----------------------------------------

class _FakeEndpoint:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)

    def get_dict(self):
        return {"resultSets": [{"headers": ["TEAM_NAME"], "rowSet": []}]}


def _fake_module():
    class A(_FakeEndpoint):
        calls = []

    class B(_FakeEndpoint):
        calls = []

    return types.SimpleNamespace(LeagueDashTeamStats=A, LeagueDashTeamShotLocations=B), A, B


class TestDateBoundEndpoints:
    def test_injects_date_to_and_caches(self, tmp_path):
        mod, A, _ = _fake_module()
        stats = nsw.FetchStats()
        as_of = date(2024, 12, 1)
        with nsw.date_bound_nba_api(as_of, stats, cache_dir=tmp_path, pause_s=0, module=mod):
            ep = mod.LeagueDashTeamStats(season="2024-25", season_type_all_star="Regular Season")
            ep.get_dict()
            mod.LeagueDashTeamStats(season="2024-25", season_type_all_star="Regular Season").get_dict()
        assert A.calls == [{"season": "2024-25", "season_type_all_star": "Regular Season",
                            "date_to_nullable": "12/01/2024"}]
        assert (stats.network_calls, stats.cache_hits) == (1, 1)

    def test_cache_key_depends_on_as_of(self, tmp_path):
        mod, A, _ = _fake_module()
        stats = nsw.FetchStats()
        for d in (date(2024, 12, 1), date(2024, 12, 2)):
            with nsw.date_bound_nba_api(d, stats, cache_dir=tmp_path, pause_s=0, module=mod):
                mod.LeagueDashTeamStats(season="2024-25").get_dict()
        assert stats.network_calls == 2
        assert len(A.calls) == 2

    def test_originals_restored(self, tmp_path):
        mod, A, B = _fake_module()
        with nsw.date_bound_nba_api(date(2024, 12, 1), nsw.FetchStats(), cache_dir=tmp_path,
                                    pause_s=0, module=mod):
            assert mod.LeagueDashTeamStats is not A
        assert mod.LeagueDashTeamStats is A and mod.LeagueDashTeamShotLocations is B

    def test_offline_without_cache_fails_and_counts(self, tmp_path):
        mod, A, _ = _fake_module()
        stats = nsw.FetchStats()
        with nsw.date_bound_nba_api(date(2024, 12, 1), stats, cache_dir=tmp_path, offline=True,
                                    pause_s=0, module=mod):
            with pytest.raises(RuntimeError):
                mod.LeagueDashTeamStats(season="2024-25").get_dict()
        assert stats.failures == 1 and A.calls == []

    def test_retries_then_reports_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nsw.time, "sleep", lambda s: None)
        mod, A, _ = _fake_module()

        def boom(self):
            raise ConnectionError("blocked")

        A.get_dict = boom
        stats = nsw.FetchStats()
        with nsw.date_bound_nba_api(date(2024, 12, 1), stats, cache_dir=tmp_path, pause_s=0, module=mod):
            with pytest.raises(RuntimeError):
                mod.LeagueDashTeamStats(season="2024-25").get_dict()
        assert stats.failures == 1
        assert list(tmp_path.iterdir()) == []  # failures are never cached


# --- blending and scoring --------------------------------------------------

def test_weighted_blend_skips_missing_and_zero_weight():
    probs = {"elo": 0.6, "form": None, "poisson": 0.9}
    assert nsw.weighted_blend(probs, nsw.LIVE_NBA_WEIGHTS) == pytest.approx(0.6)
    assert nsw.weighted_blend({}, nsw.LIVE_NBA_WEIGHTS) is None


def test_compute_blends_live_no_stack_is_elo_form_only():
    probs = {"elo": 0.7, "form": 0.5, "poisson": 0.1, "efficiency": None}
    b = nsw.compute_blends(probs)
    assert b["live_no_stack"] == pytest.approx(0.6)
    assert b["live_full"] == pytest.approx(0.6)  # nothing else fired
    assert b["generic"] == pytest.approx((0.35 * 0.7 + 0.25 * 0.5 + 0.30 * 0.1) / 0.9)


def _rec(y, probs, season="2425", rs=True):
    return nsw.GameRecord("2024-12-01", season, "a", "b", y, rs, probs, nsw.compute_blends(probs))


def test_paired_delta_sign_and_z():
    # Stack is exactly right, base is a coin flip -> full beats no_stack.
    recs = []
    for i in range(60):
        y = i % 2
        recs.append(_rec(y, {"elo": 0.5, "form": 0.5, "efficiency": (0.9 if y else 0.1) + (0.03 if i % 3 else -0.03)}))
    d = nsw.paired_delta(recs, "live_full", "live_no_stack")
    assert d["n"] == 60 and d["delta_per_1000"] < 0 and d["z"] < -5


def test_paired_delta_needs_two_games():
    assert nsw.paired_delta([], "live_full", "live_no_stack")["delta_per_1000"] is None


def test_summarize_filters_playoffs_and_seasons_and_reports_coverage():
    base = {"elo": 0.6, "form": 0.6, "poisson": 0.6}
    recs = [
        _rec(1, {**base, "efficiency": 0.7, "possession_sim": 0.7, "shot_quality": 0.6}),
        _rec(0, dict(base)),
        _rec(1, dict(base), rs=False),            # playoff: dropped by default
        _rec(1, dict(base), season="2526"),       # other season
    ]
    s = nsw.summarize(recs, score_seasons=["2425"])
    assert s["n_games"] == 2
    assert s["coverage"]["efficiency"] == 0.5
    assert s["models"]["efficiency"]["n"] == 1
    assert s["paired"]["stack_all_fired"]["n"] == 1
    assert nsw.summarize(recs, regular_only=False, score_seasons=["2425"])["n_games"] == 3


def test_records_round_trip(tmp_path):
    recs = [_rec(1, {"elo": 0.6, "form": None})]
    p = tmp_path / "r.json"
    nsw.save_records(recs, p)
    back = nsw.load_records(p)
    assert back[0].probs == recs[0].probs and back[0].home_won == 1


# --- agents priced point-in-time -------------------------------------------

def _eff_state(season="2024-25", gp=40):
    teams = {
        "celtics": {"ortg": 120.0, "drtg": 108.0, "pace": 99.0, "gp": gp, "tov_pct": 0.13},
        "wizards": {"ortg": 108.0, "drtg": 120.0, "pace": 100.0, "gp": gp, "tov_pct": 0.14},
    }
    return {"season": season, "league_avg_ortg": 114.0, "league_avg_drtg": 114.0,
            "league_avg_pace": 99.5, "teams": teams, "fetched_at": "2000-01-01"}


@pytest.mark.asyncio
async def test_efficiency_and_sim_price_a_historical_game_from_injected_state():
    """The reason this module exists: a 2024-25 game must be priced from a 2024-25
    snapshot even though the calendar (and the live state file) say 2026."""
    from evmax.agents.models.efficiency_agent import EfficiencyModelAgent
    from evmax.agents.models.possession_sim_agent import PossessionSimAgent

    state = _eff_state()
    eff = EfficiencyModelAgent()
    eff.save_state = nsw._noop
    eff._ensure_stats = lambda sector, _s=state: nsw._return(_s)
    sim = PossessionSimAgent()
    sim._load_efficiency_state = lambda _s=state: _s
    snap = nsw.StackSnapshot(as_of=date(2024, 12, 1), efficiency=eff, possession_sim=sim)

    with nsw.frozen_sim_seed():
        out = await nsw.predict_stack(snap, "celtics", "wizards", date(2024, 12, 2), {})
    assert out["efficiency"] > 0.8
    assert out["possession_sim"] > 0.8
    assert out["shot_quality"] is None and out["matchup"] is None


@pytest.mark.asyncio
async def test_snapshot_from_another_season_abstains():
    from evmax.agents.models.efficiency_agent import EfficiencyModelAgent

    state = _eff_state(season="2023-24")
    eff = EfficiencyModelAgent()
    eff._ensure_stats = lambda sector, _s=state: nsw._return(_s)
    snap = nsw.StackSnapshot(as_of=date(2024, 12, 1), efficiency=eff)
    out = await nsw.predict_stack(snap, "celtics", "wizards", date(2024, 12, 2), {})
    assert out["efficiency"] is None


@pytest.mark.asyncio
async def test_min_games_floor_still_applies():
    from evmax.agents.models.efficiency_agent import EfficiencyModelAgent

    state = _eff_state(gp=5)
    eff = EfficiencyModelAgent()
    eff._ensure_stats = lambda sector, _s=state: nsw._return(_s)
    snap = nsw.StackSnapshot(as_of=date(2024, 11, 1), efficiency=eff)
    out = await nsw.predict_stack(snap, "celtics", "wizards", date(2024, 11, 2), {})
    assert out["efficiency"] is None


@pytest.mark.asyncio
async def test_no_snapshot_means_everything_abstains():
    out = await nsw.predict_stack(None, "celtics", "wizards", date(2024, 10, 22), {"elo": 0.6})
    assert all(out[m] is None for m in nsw.STACK_MODELS)


@pytest.mark.asyncio
async def test_matchup_base_is_the_blend_of_other_models():
    """matchup shifts its base; the base here is the live-weighted blend, not a
    fixed 0.55."""
    from evmax.agents.models.matchup_agent import MatchupAgent

    seen = {}

    class _Pred:
        def __init__(self, p):
            self.true_prob_a = p

    class _Matchup:
        async def predict_pair(self, market, sharp):
            seen["base"] = sharp.true_prob_a
            return _Pred(sharp.true_prob_a + 0.02)

    snap = nsw.StackSnapshot(as_of=date(2024, 12, 1), matchup=_Matchup())
    out = await nsw.predict_stack(snap, "celtics", "wizards", date(2024, 12, 2),
                                  {"elo": 0.70, "form": 0.70})
    assert seen["base"] == pytest.approx(0.70)
    assert out["matchup_shift"] == pytest.approx(0.02)
    assert out["matchup"] == pytest.approx(0.72)
    assert MatchupAgent  # imported for parity with production wiring


def test_frozen_sim_seed_restores_date():
    import evmax.agents.models.possession_sim_agent as psa

    original = psa.date
    with nsw.frozen_sim_seed():
        assert psa.date.today() == date(2000, 1, 1)
    assert psa.date is original
