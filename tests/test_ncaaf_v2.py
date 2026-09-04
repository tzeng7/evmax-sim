"""ncaaf_efficiency_v2 tests — success-rate ratings, the FPI-mixed prior, the
v2 margin model, the agent's v2 path + v1-schema refusal, the seed assembly /
FPI freeze, and the contamination rule that dates v1 rows out.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from evmax.agents.cleanup.contamination import is_contaminated
from evmax.agents.models import _cfb_efficiency as E

_spec = importlib.util.spec_from_file_location(
    "seed_ncaaf_efficiency",
    Path(__file__).resolve().parents[1] / "scripts" / "seed_ncaaf_efficiency.py",
)
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)


# ---------------------------------------------------------------------------
# math core
# ---------------------------------------------------------------------------


def _drive(gid, off, dfn, gains, start=50, period=1):
    """A 1st-and-10 drive of `gains` yards each play (no scores)."""
    plays = []
    ytg = start
    for g in gains:
        plays.append({
            "game_id": gid, "half": 1, "period": period, "off_team": off, "def_team": dfn,
            "down": 1, "distance": 10, "yards_to_goal": ytg,
            "end_down": 1 if g >= 10 else 2, "end_distance": 10 if g >= 10 else 10 - g,
            "end_yards_to_goal": max(1, ytg - g), "end_team": off,
            "yards_gained": g, "type": "Rush",
            "score_points": 0, "score_team": None, "score_off": False, "off_margin_pre": 0,
        })
        ytg = max(1, ytg - g)
    return plays


def test_aggregate_game_sides_emits_success_rate():
    plays = _drive("g1", "A", "B", [10] * 12) + _drive("g1", "B", "A", [2] * 12)
    ep = E.build_ep_table(plays)
    games = {"g1": {"home": {"id": "A"}, "away": {"id": "B"}, "neutral": False}}
    sides, _ = E.aggregate_game_sides(plays, {"A", "B"}, games, ep, min_plays=5)
    by_off = {s["off_id"]: s for s in sides}
    assert by_off["A"]["sr"] == 1.0          # every 10-yd gain on 1st-and-10 is a success
    assert by_off["B"]["sr"] == 0.0          # 2-yd gains never reach 50%
    assert 0.0 <= by_off["A"]["sr"] <= 1.0


def test_opponent_adjust_value_key_sr_recovers_ordering():
    sides = []
    for g in range(6):
        # X's offense succeeds a lot vs everyone, Z's barely — Y in between
        sides += [
            {"off_id": "X", "def_id": "Y", "sr": 0.55, "plays": 60, "side": 1.0},
            {"off_id": "Y", "def_id": "X", "sr": 0.40, "plays": 60, "side": -1.0},
            {"off_id": "Z", "def_id": "Y", "sr": 0.30, "plays": 60, "side": -1.0},
            {"off_id": "Y", "def_id": "Z", "sr": 0.45, "plays": 60, "side": 1.0},
        ]
    r = E.opponent_adjust_epa(sides, value_key="sr")
    assert r["off"]["X"] > r["off"]["Y"] > r["off"]["Z"]
    assert r["converged"]


def test_build_team_ratings_carries_sr_fields():
    plays = _drive("g1", "A", "B", [10] * 12) + _drive("g1", "B", "A", [2] * 12)
    ep = E.build_ep_table(plays)
    games = {"g1": {"home": {"id": "A"}, "away": {"id": "B"}, "neutral": False}}
    out = E.build_team_ratings(plays, {"A", "B"}, games, ep, min_plays=5)
    for tid in ("A", "B"):
        for k in ("off_epa_adj", "def_epa_adj", "off_sr_adj", "def_sr_adj", "gp"):
            assert k in out["teams"][tid]
    assert out["teams"]["A"]["off_sr_adj"] > out["teams"]["B"]["off_sr_adj"]
    assert "league_mean_sr" in out and "hfa_sr" in out


def test_epa_prior_net_mixes_fpi_when_present():
    base = {"off_epa_prior": 0.10, "def_epa_prior": -0.10}
    assert E.epa_prior_net(base, 0.5, 70.0) == pytest.approx(0.20)          # no fpi -> EPA prior
    with_fpi = dict(base, fpi_prior=28.0)                                    # 28 pts / 70 = 0.40
    assert E.epa_prior_net(with_fpi, 0.5, 70.0) == pytest.approx(0.30)      # 50/50 mix
    assert E.epa_prior_net(with_fpi, 1.0, 70.0) == pytest.approx(0.40)      # pure FPI
    assert E.epa_prior_net(with_fpi, 0.0, 70.0) == pytest.approx(0.20)      # share 0 -> EPA prior
    assert E.epa_prior_net(dict(base, fpi_prior=None), 0.5, 70.0) == pytest.approx(0.20)


def test_blended_component_ramp_and_sr_path():
    stats = {"off_epa_adj": 0.30, "def_epa_adj": -0.10, "off_epa_prior": 0.02, "def_epa_prior": 0.0,
             "off_sr_adj": 0.06, "def_sr_adj": -0.02, "off_sr_prior": 0.01, "def_sr_prior": -0.01,
             "fpi_prior": 14.0, "gp": 0}
    # gp=0 -> pure prior (EPA: 0.5*0.02 + 0.5*14/70 = 0.11; SR: 0.02)
    assert E.blended_component(stats, 3.0, "epa", 0.5, 70.0) == pytest.approx(0.11)
    assert E.blended_component(stats, 3.0, "sr") == pytest.approx(0.02)
    stats["gp"] = 3000  # ~all in-season
    assert E.blended_component(stats, 3.0, "epa", 0.5, 70.0) == pytest.approx(0.40, abs=1e-3)
    assert E.blended_component(stats, 3.0, "sr") == pytest.approx(0.08, abs=1e-3)


def test_project_win_prob_v2_signs_and_neutral():
    p_fav, m = E.project_win_prob_v2(0.10, 0.05, 35.0, 70.0, 3.5, 16.5)
    assert p_fav > 0.5 and m == pytest.approx(3.5 + 3.5 + 3.5)
    p_even_home, _ = E.project_win_prob_v2(0.0, 0.0, 35.0, 70.0, 3.5, 16.5)
    assert p_even_home > 0.5
    p_even_neutral, m0 = E.project_win_prob_v2(0.0, 0.0, 35.0, 70.0, 3.5, 16.5, neutral=True)
    assert p_even_neutral == pytest.approx(0.5) and m0 == 0.0
    p_dog, _ = E.project_win_prob_v2(-0.10, -0.05, 35.0, 70.0, 3.5, 16.5, neutral=True)
    assert p_dog == pytest.approx(1.0 - E.project_win_prob_v2(0.10, 0.05, 35.0, 70.0, 3.5, 16.5, neutral=True)[0])
    assert 0.02 <= E.project_win_prob_v2(5.0, 5.0, 35.0, 70.0, 3.5, 16.5)[0] <= 0.98


def test_state_has_v2_schema():
    assert E.state_has_v2_schema({"schema_version": 2}) is True
    assert E.state_has_v2_schema({"schema_version": "2"}) is True
    assert E.state_has_v2_schema({}) is False
    assert E.state_has_v2_schema({"schema_version": "x"}) is False


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------


def _v2_state(with_fpi=True, schema=2):
    strong = {"off_epa_adj": 0.20, "def_epa_adj": -0.20, "off_sr_adj": 0.05, "def_sr_adj": -0.04,
              "off_epa_prior": 0.10, "def_epa_prior": -0.08, "off_sr_prior": 0.02, "def_sr_prior": -0.02,
              "gp": 10}
    weak = {"off_epa_adj": -0.25, "def_epa_adj": 0.22, "off_sr_adj": -0.05, "def_sr_adj": 0.04,
            "off_epa_prior": -0.10, "def_epa_prior": 0.09, "off_sr_prior": -0.02, "def_sr_prior": 0.02,
            "gp": 10}
    if with_fpi:
        strong["fpi_prior"] = 22.0
    state = {"ncaaf": {"source_season": 2026, "teams": {"ohio state": strong, "kent state": weak}}}
    if schema:
        state["ncaaf"]["schema_version"] = schema
    return state


def _agent(state, monkeypatch):
    import evmax.agents.models._cfb_efficiency as core
    from evmax.agents.models.ncaaf_efficiency_agent import NcaafEfficiencyModelAgent

    monkeypatch.setattr(core, "state_is_stale_for_today", lambda *a, **k: False)
    agent = NcaafEfficiencyModelAgent()
    agent._state = state
    return agent


def _market(home, away):
    from evmax.models.market import MarketSource, MarketType, PredictionMarket
    from evmax.models.odds import SharpBook, SharpOdds

    m = PredictionMarket(
        id=f"k:{home}", source=MarketSource.kalshi, sector="ncaaf",
        ticker="KXNCAAFGAME-X", title=f"{home} vs {away}", team_home=home, team_away=away,
        market_type=MarketType.moneyline, yes_price=0.5, no_price=0.5,
    )
    s = SharpOdds(
        event_id=f"e:{home}:{away}", sector="ncaaf", book=SharpBook.pinnacle,
        market_type=MarketType.moneyline, outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=2.0, outcome_b_decimal=2.0, true_prob_a=0.5, true_prob_b=0.5,
    )
    return m, s


def test_agent_v2_predicts_favorite_and_is_named_v2(monkeypatch):
    agent = _agent(_v2_state(), monkeypatch)
    pred = asyncio.run(agent.predict_pair(*_market("Ohio State", "Kent State")))
    assert pred is not None
    assert pred.model_name == "ncaaf_efficiency_v2"
    assert pred.true_prob_a > 0.85
    assert "sr=" in pred.notes and "fpi=yn" in pred.notes


def test_agent_refuses_v1_schema_state(monkeypatch):
    agent = _agent(_v2_state(schema=None), monkeypatch)
    assert asyncio.run(agent.predict_pair(*_market("Ohio State", "Kent State"))) is None


def test_agent_fires_without_any_fpi(monkeypatch):
    agent = _agent(_v2_state(with_fpi=False), monkeypatch)
    pred = asyncio.run(agent.predict_pair(*_market("Ohio State", "Kent State")))
    assert pred is not None and pred.true_prob_a > 0.8
    assert "fpi=nn" in pred.notes


def test_agent_fpi_prior_alone_clears_week0_gate(monkeypatch):
    state = _v2_state()
    for t in state["ncaaf"]["teams"].values():
        t["gp"] = 0
        for k in ("off_epa_prior", "def_epa_prior"):
            t[k] = 0.0
    state["ncaaf"]["teams"]["kent state"]["fpi_prior"] = -15.0
    agent = _agent(state, monkeypatch)
    pred = asyncio.run(agent.predict_pair(*_market("Ohio State", "Kent State")))
    assert pred is not None and pred.true_prob_a > 0.7 and pred.confidence == 0.52
    # and with neither prior on one side, week 0 is too thin
    del state["ncaaf"]["teams"]["kent state"]["fpi_prior"]
    assert asyncio.run(agent.predict_pair(*_market("Ohio State", "Kent State"))) is None


# ---------------------------------------------------------------------------
# seed assembly + FPI freeze
# ---------------------------------------------------------------------------


def _ratings(off_epa, def_epa, off_sr, def_sr, gp):
    return {"off_epa_adj": off_epa, "def_epa_adj": def_epa, "off_sr_adj": off_sr,
            "def_sr_adj": def_sr, "gp": gp, "off_success_rate": 0.45, "def_success_rate": 0.42}


def _two_seasons():
    prior = {"teams": {"1": _ratings(0.20, -0.10, 0.04, -0.02, 12), "2": _ratings(-0.10, 0.10, -0.03, 0.03, 12)},
             "league_mean_epa": 0.0, "hfa_epa": 0.05}
    inseason = {"teams": {"1": _ratings(0.05, 0.0, 0.01, 0.0, 1), "3": _ratings(0.0, 0.0, 0.0, 0.0, 1)},
                "league_mean_epa": 0.01, "hfa_epa": 0.04}
    names = {"1": "ohio state", "2": "kent state", "3": "new fbs"}
    abbr = {"1": "OSU", "2": "KENT", "3": "NEW"}
    return prior, inseason, names, abbr


def test_assemble_state_live_fpi_and_regressed_priors():
    prior, inseason, names, abbr = _two_seasons()
    fpi = {"1": {"fpi": 20.0, "name": "Ohio State"}, "2": {"fpi": -10.0, "name": "Kent State"}}
    st = seed.assemble_state(2026, 2025, 0.5, inseason, names, abbr, prior, names, abbr,
                             in_games=40, fpi_by_id=fpi, fpi_source="live", fetched_at="2026-09-03")
    s = st["ncaaf"]
    assert s["schema_version"] == 2 and s["fpi_source"] == "live" and s["fpi_season"] == 2026
    osu, kent, new = s["teams"]["ohio state"], s["teams"]["kent state"], s["teams"]["new fbs"]
    assert osu["off_epa_prior"] == pytest.approx(0.10) and osu["def_sr_prior"] == pytest.approx(-0.01)
    assert osu["off_sr_adj"] == pytest.approx(0.01) and osu["gp"] == 1
    assert kent["gp"] == 0 and kent["off_epa_adj"] == 0.0        # prior-only week-0 row
    # FPI centred on the rated members (20, -10 -> mean 5): +15 / -15; unrated team has no key
    assert osu["fpi_prior"] == pytest.approx(15.0) and kent["fpi_prior"] == pytest.approx(-15.0)
    assert "fpi_prior" not in new
    assert s["fpi_teams"] == 2 and s["source_season"] == 2026 and s["seasons_used"] == [2026]


def test_assemble_state_frozen_fpi_copies_by_name():
    prior, inseason, names, abbr = _two_seasons()
    existing = {"fpi_season": 2026, "fpi_frozen_at": "2026-08-20",
                "teams": {"ohio state": {"fpi_prior": 12.5}, "kent state": {"fpi_prior": -3.0}}}
    st = seed.assemble_state(2026, 2025, 0.5, inseason, names, abbr, prior, names, abbr,
                             in_games=40, fpi_by_id={}, fpi_source="frozen", existing=existing)
    s = st["ncaaf"]
    assert s["fpi_source"] == "frozen" and s["fpi_frozen_at"] == "2026-08-20"
    assert s["teams"]["ohio state"]["fpi_prior"] == 12.5
    assert s["teams"]["kent state"]["fpi_prior"] == -3.0
    assert "fpi_prior" not in s["teams"]["new fbs"]


def test_assemble_state_no_fpi_is_v1_prior_shape():
    prior, inseason, names, abbr = _two_seasons()
    st = seed.assemble_state(2026, 2025, 0.5, inseason, names, abbr, prior, names, abbr,
                             in_games=0, fpi_by_id={}, fpi_source="none")
    s = st["ncaaf"]
    assert s["fpi_source"] == "none" and s["fpi_teams"] == 0
    assert all("fpi_prior" not in t for t in s["teams"].values())
    assert s["seasons_used"] == [2025]         # no in-season games -> prior season labelled


def test_resolve_fpi_freeze_semantics():
    calls = []

    def fetch(season):
        calls.append(season)
        return {"1": {"fpi": 1.0, "name": "x"}}

    frozen = {"fpi_season": 2026, "teams": {"a": {"fpi_prior": 1.0}}}
    assert seed.resolve_fpi(frozen, 2026, fetch_fn=fetch) == ({}, "frozen") and calls == []
    # a different season's freeze is not reused
    out, src = seed.resolve_fpi({"fpi_season": 2025, "teams": {"a": {"fpi_prior": 1.0}}}, 2026, fetch_fn=fetch)
    assert src == "live" and out and calls == [2026]
    # refresh forces a fetch even when frozen
    assert seed.resolve_fpi(frozen, 2026, refresh=True, fetch_fn=fetch)[1] == "live"
    # fetch failure -> none
    assert seed.resolve_fpi({}, 2026, fetch_fn=lambda s: {}) == ({}, "none")


# ---------------------------------------------------------------------------
# contamination
# ---------------------------------------------------------------------------


def test_ncaaf_contamination_dates_v1_rows_only():
    assert is_contaminated("ncaaf", "moneyline", "elo+ncaaf_efficiency+sharp") is True
    assert is_contaminated("ncaaf", "moneyline", "elo+ncaaf_efficiency_v2+sharp") is False
    assert is_contaminated("ncaaf", "moneyline", "elo+sharp") is False
    assert is_contaminated("ncaaf", "spread", "elo+ncaaf_efficiency+sharp") is False


# ---------------------------------------------------------------------------
# seed: early-season FBS universe must come from the schedule, not completed games
# ---------------------------------------------------------------------------


def test_fbs_universe_from_schedule_counts_upcoming_games():
    """Two weeks in, no team has 6 COMPLETED games; the schedule-based universe
    (used by the in-season pass) still classifies the 12-game FBS teams."""
    def game(gid, h, a, done):
        return {"game_id": gid, "completed": done,
                "home": {"id": h, "location": f"Team {h}", "abbr": f"T{h}"},
                "away": {"id": a, "location": f"Team {a}", "abbr": f"T{a}"}}
    sched = []
    for w in range(12):  # A plays B every week, C (FCS) shows up once
        sched.append(game(f"g{w}", "A", "B", done=w < 2))
    sched.append(game("fcs", "A", "C", done=True))
    fbs_done, _, _ = seed._fbs_universe([g for g in sched if g["completed"]])
    fbs_sched, _, _ = seed._fbs_universe(sched)
    assert fbs_done == set()                 # the failure mode
    assert fbs_sched == {"A", "B"}           # C (1 appearance) stays FCS-pooled


def test_fetch_season_plays_with_games_list_uses_completed_only(monkeypatch):
    from evmax.clients import cfb_espn as C

    seen = []
    monkeypatch.setattr(C, "fetch_game_summary", lambda gid, client=None, use_cache=True: seen.append(gid) or None)
    monkeypatch.setattr(C, "fetch_season_games", lambda *a, **k: (_ for _ in ()).throw(AssertionError("walk must be skipped")))
    sched = [{"game_id": "1", "completed": True, "home": {"id": "A"}, "away": {"id": "B"}},
             {"game_id": "2", "completed": False, "home": {"id": "A"}, "away": {"id": "B"}}]
    plays, games = C.fetch_season_plays(2026, games=sched, max_workers=1)
    assert plays == [] and [g["game_id"] for g in games] == ["1"] and seen == ["1"]
