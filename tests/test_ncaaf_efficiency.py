"""NCAAF EPA efficiency model tests (Phase 2).

Covers the math core (_cfb_efficiency) and the agent: EP monotonicity, EPA sign,
the additive opponent-adjustment solver's recovery + FCS pooling, the preseason-
prior ramp, the staleness guard, and predict_pair resolution/gating.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from evmax.agents.models import _cfb_efficiency as E


# ---------------------------------------------------------------------------
# Expected points + EPA
# ---------------------------------------------------------------------------


def _synthetic_scoring_plays():
    """Two half-drives: one that ends in an offense TD, one that punts.

    Plays march from midfield toward the end zone so EP should rise as
    yards_to_goal falls.
    """
    plays = []
    # game 1, half 1: OFF=A drives 50→0 and scores 7 (TD on the last play)
    ytg = 50
    for i in range(5):
        end = ytg - 10
        scoring = i == 4
        plays.append({
            "game_id": "g1", "half": 1, "period": 1, "off_team": "A", "def_team": "B",
            "down": 1, "distance": 10, "yards_to_goal": ytg,
            "end_down": (1 if not scoring else -1), "end_distance": 10,
            "end_yards_to_goal": (end if not scoring else 0), "end_team": "A",
            "yards_gained": 10, "type": "Rush",
            "score_points": 7 if scoring else 0, "score_team": "A" if scoring else None,
            "score_off": scoring, "off_margin_pre": 0,
        })
        ytg = end
    # game 2, half 1: OFF=C stalls and no one scores the rest of the half
    for ytg in (80, 70, 60):
        plays.append({
            "game_id": "g2", "half": 1, "period": 1, "off_team": "C", "def_team": "D",
            "down": 1, "distance": 10, "yards_to_goal": ytg,
            "end_down": 2, "end_distance": 8, "end_yards_to_goal": ytg - 2, "end_team": "C",
            "yards_gained": 2, "type": "Rush",
            "score_points": 0, "score_team": None, "score_off": False, "off_margin_pre": 0,
        })
    return plays


def test_ep_table_monotonic_in_field_position():
    # Build a richer synthetic set: closer to the end zone => higher next-score EP.
    plays = []
    gid = 0
    for base_ytg, scores in [(10, True), (30, True), (60, False), (90, False)]:
        gid += 1
        plays.append({
            "game_id": f"g{gid}", "half": 1, "period": 1, "off_team": "A", "def_team": "B",
            "down": 1, "distance": 10, "yards_to_goal": base_ytg,
            "end_down": -1 if scores else 2, "end_distance": 10,
            "end_yards_to_goal": 0 if scores else base_ytg - 3, "end_team": "A",
            "yards_gained": 3, "type": "Rush",
            "score_points": 7 if scores else 0, "score_team": "A" if scores else None,
            "score_off": scores, "off_margin_pre": 0,
        })
    table = E.build_ep_table(plays)
    ep_near = E.ep_value(table, 1, 10, 10)   # close to end zone
    ep_far = E.ep_value(table, 1, 90, 10)    # backed up
    assert ep_near > ep_far


def test_play_epa_scoring_and_gain_positive():
    table = E.build_ep_table(_synthetic_scoring_plays())
    # A scoring play: EPA = points - ep_start, should be strongly positive.
    scoring = {
        "down": 1, "distance": 10, "yards_to_goal": 5, "off_team": "A",
        "score_points": 7, "score_off": True,
        "end_down": -1, "end_yards_to_goal": 0, "end_team": "A", "end_distance": 10,
    }
    assert E.play_epa(table, scoring) > 0


def test_play_epa_turnover_is_negative():
    table = E.build_ep_table(_synthetic_scoring_plays())
    # Possession flips to the opponent near midfield with no score → negative EPA
    turnover = {
        "down": 2, "distance": 8, "yards_to_goal": 50, "off_team": "A",
        "score_points": 0, "score_off": False,
        "end_down": 1, "end_yards_to_goal": 50, "end_team": "B", "end_distance": 10,
    }
    assert E.play_epa(table, turnover) < 0


def test_play_epa_none_on_non_scrimmage_state():
    table = E.build_ep_table(_synthetic_scoring_plays())
    kickoff = {"down": 0, "distance": 0, "yards_to_goal": 65, "off_team": "A"}
    assert E.play_epa(table, kickoff) is None


def test_is_success_thresholds():
    assert E.is_success(1, 10, 5) is True     # 50% on 1st
    assert E.is_success(1, 10, 4) is False
    assert E.is_success(2, 10, 7) is True     # 70% on 2nd
    assert E.is_success(3, 10, 9) is False    # need 100% on 3rd
    assert E.is_success(3, 10, 10) is True


def test_is_scrimmage_play_filter():
    assert E.is_scrimmage_play("Rush") is True
    assert E.is_scrimmage_play("Pass Reception") is True
    assert E.is_scrimmage_play("Punt") is False
    assert E.is_scrimmage_play("Kickoff Return (Offense)") is False
    assert E.is_scrimmage_play("Field Goal Good") is False


# ---------------------------------------------------------------------------
# Additive opponent adjustment
# ---------------------------------------------------------------------------


def test_opponent_adjust_recovers_ordering():
    # Ground-truth ratings; every team plays every other home and away so the
    # incidence graph is fully connected (identifiable up to ridge shrink).
    off = {"A": 0.30, "B": 0.00, "C": -0.30}
    dfn = {"A": -0.20, "B": 0.00, "C": 0.20}  # def_allowed: lower = better D
    mu = 0.05
    game_sides = []
    for o in off:
        for d in off:
            if o == d:
                continue
            for side in (1.0, -1.0):
                game_sides.append({
                    "off_id": o, "def_id": d,
                    "epa": mu + off[o] + dfn[d] + 0.04 * side / 2.0,
                    "plays": 60, "side": side,
                })
    res = E.opponent_adjust_epa(game_sides, ridge=0.5)
    # league mean recovered
    assert abs(res["league_mean_epa"] - mu) < 0.02
    # offensive ordering preserved
    assert res["off"]["A"] > res["off"]["B"] > res["off"]["C"]
    # def_allowed ordering preserved (A best defense = lowest allowed)
    assert res["def_allowed"]["A"] < res["def_allowed"]["B"] < res["def_allowed"]["C"]
    # home-field edge positive
    assert res["hfa"] > 0


def test_opponent_adjust_empty():
    res = E.opponent_adjust_epa([])
    assert res["off"] == {} and res["converged"] is True


def test_build_team_ratings_pools_fcs():
    # Two FBS teams (A, B) each blow out an FCS side; A also beats B.
    games_by_id = {
        "g1": {"game_id": "g1", "neutral": False, "home": {"id": "A"}, "away": {"id": "F1"}},
        "g2": {"game_id": "g2", "neutral": False, "home": {"id": "B"}, "away": {"id": "F2"}},
        "g3": {"game_id": "g3", "neutral": False, "home": {"id": "A"}, "away": {"id": "B"}},
    }
    plays = []
    for gid, off, dfn, epa_each in [
        ("g1", "A", "F1", 0.4), ("g2", "B", "F2", 0.3), ("g3", "A", "B", 0.2),
    ]:
        for _ in range(15):
            plays.append({
                "game_id": gid, "half": 1, "period": 1, "off_team": off, "def_team": dfn,
                "down": 1, "distance": 10, "yards_to_goal": 50,
                "end_down": 1, "end_distance": 10, "end_yards_to_goal": 45, "end_team": off,
                "yards_gained": 5, "type": "Rush", "score_points": 0, "score_team": None,
                "score_off": False, "off_margin_pre": 0,
            })
    # trivial EP table so play_epa returns finite numbers
    table = {"ep": {"1:5": 2.0, "1:4": 2.3}, "n": {}, "default_by_bucket": {5: 2.0, 4: 2.3}}
    fbs = {"A", "B"}
    out = E.build_team_ratings(plays, fbs, games_by_id, table, min_plays=5)
    assert set(out["teams"]) == {"A", "B"}          # FCS not rated as a team
    assert out["teams"]["A"]["gp"] >= 1


# ---------------------------------------------------------------------------
# Prior ramp + projection
# ---------------------------------------------------------------------------


def test_prior_ramp_weight():
    assert E.prior_ramp_weight(0, 3.0) == 0.0
    assert abs(E.prior_ramp_weight(3, 3.0) - 0.5) < 1e-9
    assert E.prior_ramp_weight(9, 3.0) > E.prior_ramp_weight(3, 3.0)


def test_blended_net_ramp():
    stats = {"off_epa_adj": 0.4, "def_epa_adj": -0.1,   # in-season net = 0.5
             "off_epa_prior": 0.0, "def_epa_prior": 0.0, "gp": 0}
    assert E.blended_net(stats, k=3.0) == pytest.approx(0.0)   # gp=0 → all prior
    stats["gp"] = 30
    assert E.blended_net(stats, k=3.0) > 0.4   # mostly in-season now


def test_project_win_prob_signs():
    p_fav, m = E.project_win_prob(0.3, -0.1, 70, 2.5, 16.5)
    assert p_fav > 0.5 and m > 0
    p_neu, m_neu = E.project_win_prob(0.1, 0.1, 70, 2.5, 16.5, neutral=True)
    assert p_neu == pytest.approx(0.5, abs=1e-6) and m_neu == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Staleness guard
# ---------------------------------------------------------------------------


def test_staleness_guard():
    state = {"ncaaf": {"source_season": 2024}}
    # In-season (November) with a 2-season-old seed → stale
    assert E.state_is_stale_for_today(state, dt.date(2026, 11, 1)) is True
    # Off-season (June) → guard relaxed
    assert E.state_is_stale_for_today(state, dt.date(2026, 6, 1)) is False
    # Current-season seed → not stale
    assert E.state_is_stale_for_today({"ncaaf": {"source_season": 2026}}, dt.date(2026, 11, 1)) is False
    # Missing source_season → guard disabled
    assert E.state_is_stale_for_today({"ncaaf": {}}, dt.date(2026, 11, 1)) is False


# ---------------------------------------------------------------------------
# Agent predict_pair
# ---------------------------------------------------------------------------


def _agent_with_state(source_season):
    from evmax.agents.models.ncaaf_efficiency_agent import NcaafEfficiencyModelAgent

    agent = NcaafEfficiencyModelAgent()
    agent._state = {
        "ncaaf": {
            "source_season": source_season,
            "teams": {
                "ohio state": {"off_epa_adj": 0.20, "def_epa_adj": -0.20,
                               "off_epa_prior": 0.10, "def_epa_prior": -0.08, "gp": 10},
                "kent state": {"off_epa_adj": -0.25, "def_epa_adj": 0.22,
                               "off_epa_prior": -0.10, "def_epa_prior": 0.09, "gp": 10},
            },
        }
    }
    return agent


def _mk_market_sharp(home, away):
    from evmax.models.market import PredictionMarket, MarketType, MarketSource
    from evmax.models.odds import SharpOdds, SharpBook

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


def test_agent_predicts_favorite(monkeypatch):
    import evmax.agents.models._cfb_efficiency as core

    # Force the state to look current so the staleness guard passes regardless
    # of the real calendar date the test runs on.
    monkeypatch.setattr(core, "state_is_stale_for_today", lambda *a, **k: False)
    agent = _agent_with_state(2026)
    m, s = _mk_market_sharp("Ohio State", "Kent State")
    pred = asyncio.run(agent.predict_pair(m, s))
    assert pred is not None
    assert pred.true_prob_a > 0.85       # OSU heavy favorite
    assert pred.model_name == "ncaaf_efficiency"


def test_agent_none_on_unresolved_team(monkeypatch):
    import evmax.agents.models._cfb_efficiency as core

    monkeypatch.setattr(core, "state_is_stale_for_today", lambda *a, **k: False)
    agent = _agent_with_state(2026)
    m, s = _mk_market_sharp("Ohio State", "Nonexistent College")
    pred = asyncio.run(agent.predict_pair(m, s))
    assert pred is None


def test_agent_none_when_stale():
    # Real guard active: a 2024 seed must not fire during an in-season month.
    agent = _agent_with_state(2024)
    m, s = _mk_market_sharp("Ohio State", "Kent State")
    # Only assert the stale path returns None when we are actually in-season;
    # otherwise the guard relaxes and the agent may legitimately predict.
    if dt.date.today().month in E.CFB_SEASON_MONTHS:
        pred = asyncio.run(agent.predict_pair(m, s))
        assert pred is None
