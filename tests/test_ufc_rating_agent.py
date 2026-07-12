"""UFCRatingAgent tests — rating updates, feature layer, name resolution."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from evmax.agents.models.ufc_rating_agent import (
    FEATURE_NAMES,
    FighterState,
    UFCRatingAgent,
    apply_fight_result,
    blend_probability,
    compute_features,
    replay_history,
)
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds


def _market(sector="ufc", home="pimblett", away="saint-denis", event_date=None):
    return PredictionMarket(
        id="kalshi:KXUFCFIGHT-26JUL11SAIPIM-SAI",
        source=MarketSource.kalshi,
        sector=sector,
        market_type=MarketType.moneyline,
        title="Will Benoit Saint-Denis win the Saint-Denis vs Pimblett professional MMA fight scheduled for Jul 11, 2026?",
        ticker="KXUFCFIGHT-26JUL11SAIPIM-SAI",
        yes_price=0.58,
        no_price=0.45,
        volume_usd=1500.0,
        team_home=home,
        team_away=away,
        event_date=event_date or datetime(2026, 7, 11, 12, tzinfo=timezone.utc),
        yes_team="saint-denis",
    )


def _sharp(label_a="Paddy Pimblett", label_b="Benoit Saint-Denis"):
    return SharpOdds(
        event_id="ufc::2026-07-11::pimblett_vs_saint-denis",
        book=SharpBook.pinnacle,
        sector="ufc",
        outcome_a_label=label_a,
        outcome_b_label=label_b,
        outcome_a_decimal=1.8,
        outcome_b_decimal=2.1,
        true_prob_a=0.55,
        true_prob_b=0.45,
        margin=0.04,
        event_date=datetime(2026, 7, 12, 2, 50, tzinfo=timezone.utc),
    )


def _agent(tmp_path, fighters: dict, last_updated="2026-07-05", feature_model=None):
    agent = UFCRatingAgent()
    agent._state_path = tmp_path / "ufc_rating_state.json"
    agent._state = {
        "fighters": fighters,
        "last_updated": last_updated,
    }
    if feature_model is not None:
        agent._state["feature_model"] = feature_model
    return agent


def _fighter(rating=1500.0, n=10, **kw) -> dict:
    base = FighterState(rating=rating, rd=90.0, n_fights=n,
                        wins=max(n - 2, 0), losses=min(n, 2),
                        last_fight="2026-01-10")
    d = base.to_dict()
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# apply_fight_result — rating update math + bookkeeping
# ---------------------------------------------------------------------------


class TestApplyFightResult:
    def test_winner_up_loser_down(self):
        fighters = {}
        apply_fight_result(fighters, "a guy", "b guy", 1.0, "2024-01-01")
        assert fighters["a guy"].rating > 1500 > fighters["b guy"].rating

    def test_bookkeeping(self):
        fighters = {}
        apply_fight_result(fighters, "a", "b", 1.0, "2024-01-01", method="ko_tko")
        a, b = fighters["a"], fighters["b"]
        assert (a.n_fights, a.wins, a.losses) == (1, 1, 0)
        assert (b.n_fights, b.wins, b.losses) == (1, 0, 1)
        assert a.last3 == ["w"] and b.last3 == ["l"]
        assert a.finish_wins == 1
        assert b.ko_losses == 1 and b.last_ko_loss == "2024-01-01"
        assert a.last_fight == b.last_fight == "2024-01-01"

    def test_decision_win_is_not_a_finish(self):
        fighters = {}
        apply_fight_result(fighters, "a", "b", 1.0, "2024-01-01", method="decision")
        assert fighters["a"].finish_wins == 0
        assert fighters["b"].ko_losses == 0

    def test_draw(self):
        fighters = {}
        apply_fight_result(fighters, "a", "b", 0.5, "2024-01-01", method="draw")
        assert fighters["a"].draws == 1 and fighters["b"].draws == 1
        assert fighters["a"].last3 == ["d"]

    def test_last3_rolls(self):
        fighters = {}
        for i, s in enumerate((1.0, 1.0, 0.0, 1.0)):
            apply_fight_result(fighters, "a", "b", s, f"2024-0{i+1}-01")
        # w, w, l, w → most-recent-last window of 3 = [w, l, w]
        assert fighters["a"].last3 == ["w", "l", "w"]

    def test_layoff_inflates_rd_before_update(self):
        """A fighter returning from a long layoff should carry a larger RD
        into the bout than one who just fought."""
        fresh, rusty = {}, {}
        apply_fight_result(fresh, "a", "b", 1.0, "2024-01-01")
        apply_fight_result(rusty, "a", "b", 1.0, "2024-01-01")
        # second bout: fresh 2 weeks later, rusty 2 years later
        apply_fight_result(fresh, "a", "c", 1.0, "2024-01-15")
        apply_fight_result(rusty, "a", "c", 1.0, "2026-01-01")
        assert rusty["a"].rd > fresh["a"].rd


# ---------------------------------------------------------------------------
# compute_features — point-in-time differentials
# ---------------------------------------------------------------------------


class TestComputeFeatures:
    def test_age_diff(self):
        a = FighterState(dob="1990-01-01")
        b = FighterState(dob="1995-01-01")
        feats = compute_features(a, b, "2025-01-01")
        assert feats["age"] == pytest.approx(5.0, abs=0.05)

    def test_age_missing_dob_contributes_zero(self):
        feats = compute_features(FighterState(dob="1990-01-01"), FighterState(), "2025-01-01")
        assert feats["age"] == 0.0

    def test_layoff_log_scale(self):
        a = FighterState(last_fight="2024-12-01")   # 31 days out
        b = FighterState(last_fight="2023-01-01")   # ~2 years out
        feats = compute_features(a, b, "2025-01-01")
        assert feats["layoff"] == pytest.approx(math.log1p(31) - math.log1p(731), abs=1e-6)
        assert feats["layoff"] < 0

    def test_recent_ko_window(self):
        recently_koed = FighterState(last_ko_loss="2024-10-01")
        long_ago_koed = FighterState(last_ko_loss="2020-01-01")
        clean = FighterState()
        assert compute_features(recently_koed, clean, "2025-01-01")["recent_ko"] == 1.0
        assert compute_features(long_ago_koed, clean, "2025-01-01")["recent_ko"] == 0.0
        assert compute_features(clean, recently_koed, "2025-01-01")["recent_ko"] == -1.0

    def test_reach_missing_contributes_zero(self):
        feats = compute_features(FighterState(reach_in=80.0), FighterState(), "2025-01-01")
        assert feats["reach"] == 0.0

    def test_all_feature_names_present(self):
        feats = compute_features(FighterState(), FighterState(), "2025-01-01")
        assert set(feats) == set(FEATURE_NAMES)

    def test_symmetric_inputs_are_zero(self):
        f = dict(dob="1992-05-05", last_fight="2024-11-01", reach_in=72.0,
                 height_in=70.0)
        feats = compute_features(FighterState(**f), FighterState(**f), "2025-01-01")
        assert all(abs(v) < 1e-9 for v in feats.values())


# ---------------------------------------------------------------------------
# blend_probability
# ---------------------------------------------------------------------------


class TestBlendProbability:
    def test_no_model_passthrough(self):
        assert blend_probability(0.63, {}, None) == 0.63

    def test_identity_model_reproduces_rating(self):
        fm = {"intercept": 0.0, "rating_coef": 1.0, "coefs": {}}
        assert blend_probability(0.63, {"age": 5.0}, fm) == pytest.approx(0.63)

    def test_positive_coef_shifts_up(self):
        fm = {"intercept": 0.0, "rating_coef": 1.0, "coefs": {"winrate": 2.0}}
        assert blend_probability(0.5, {"winrate": 0.3}, fm) > 0.5


# ---------------------------------------------------------------------------
# replay_history — walk-forward integrity
# ---------------------------------------------------------------------------


def _fight(a, b, winner, d, method="decision", aid="", bid="", comp="1"):
    return {
        "fighter_a": a, "fighter_b": b, "fighter_a_id": aid, "fighter_b_id": bid,
        "winner": winner, "date": d, "method": method,
        "event_id": "e", "comp_id": comp,
    }


class TestReplayHistory:
    def test_ratings_accumulate(self):
        fights = [
            _fight("Al Ace", "Bo Bell", "a", "2023-01-01", comp="1"),
            _fight("Al Ace", "Cy Cole", "a", "2023-06-01", comp="2"),
        ]
        fighters, _ = replay_history(fights, {})
        assert fighters["al ace"].rating > 1500
        assert fighters["al ace"].n_fights == 2

    def test_nc_skipped_entirely(self):
        fights = [_fight("Al Ace", "Bo Bell", "none", "2023-01-01", method="nc")]
        fighters, rows = replay_history(fights, {}, collect=True)
        assert fighters == {}
        assert rows == []

    def test_draw_updates_at_half(self):
        fights = [_fight("Al Ace", "Bo Bell", "none", "2023-01-01", method="draw")]
        fighters, _ = replay_history(fights, {})
        assert fighters["al ace"].draws == 1
        assert fighters["al ace"].rating == pytest.approx(1500, abs=1e-6)

    def test_collected_rows_are_pre_fight(self):
        """The prediction row for a bout must NOT contain that bout's result:
        two debutants → p_rating exactly 0.5, n_a == n_b == 0."""
        fights = [_fight("Al Ace", "Bo Bell", "a", "2023-01-01")]
        _, rows = replay_history(fights, {}, collect=True)
        assert len(rows) == 1
        assert rows[0]["p_rating"] == pytest.approx(0.5)
        assert rows[0]["n_a"] == 0 and rows[0]["n_b"] == 0

    def test_second_meeting_uses_first_result(self):
        fights = [
            _fight("Al Ace", "Bo Bell", "a", "2023-01-01", comp="1"),
            _fight("Al Ace", "Bo Bell", "a", "2023-06-01", comp="2"),
        ]
        _, rows = replay_history(fights, {}, collect=True)
        assert rows[1]["p_rating"] > 0.5  # Al established as better pre-fight

    def test_bios_attach(self):
        fights = [_fight("Al Ace", "Bo Bell", "a", "2023-01-01", aid="11", bid="22")]
        bios = {"11": {"dob": "1990-01-01", "height_in": "70", "reach_in": "72"}}
        fighters, _ = replay_history(fights, bios)
        assert fighters["al ace"].dob == "1990-01-01"
        assert fighters["al ace"].reach_in == 72.0
        assert fighters["bo bell"].dob is None

    def test_name_collision_keys_apart(self):
        """Two DIFFERENT athletes sharing a display name must not merge."""
        fights = [
            _fight("Bruno Silva", "Al Ace", "a", "2023-01-01", aid="1", bid="9", comp="1"),
            _fight("Bruno Silva", "Bo Bell", "b", "2023-02-01", aid="2", bid="8", comp="2"),
        ]
        fighters, _ = replay_history(fights, {})
        silvas = [k for k in fighters if "bruno silva" in k]
        assert len(silvas) == 2
        assert fighters["bruno silva"].n_fights == 1


# ---------------------------------------------------------------------------
# UFCRatingAgent.predict_pair
# ---------------------------------------------------------------------------


class TestPredictPair:
    @pytest.mark.asyncio
    async def test_happy_path_favors_higher_rating(self, tmp_path):
        agent = _agent(tmp_path, {
            "paddy pimblett": _fighter(rating=1650),
            "benoit saint-denis": _fighter(rating=1550),
        })
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.true_prob_a > 0.5          # outcome_a = Pimblett, higher rated
        assert pred.true_prob_a + pred.true_prob_b == pytest.approx(1.0)
        assert pred.confidence >= 0.45          # clears the ensemble gate at n=10

    @pytest.mark.asyncio
    async def test_surname_labels_resolve(self, tmp_path):
        """Kalshi-side surname labels must resolve to full-name state keys."""
        agent = _agent(tmp_path, {
            "paddy pimblett": _fighter(rating=1650),
            "benoit saint-denis": _fighter(rating=1550),
        })
        pred = await agent.predict_pair(
            _market(), _sharp(label_a="Pimblett", label_b="Saint-Denis"),
        )
        assert pred is not None
        assert pred.true_prob_a > 0.5

    @pytest.mark.asyncio
    async def test_same_surname_ambiguity_returns_none(self, tmp_path):
        """Two distinct fighters named Silva — a bare 'silva' query must NOT
        guess (the tennis name-resolution rule)."""
        agent = _agent(tmp_path, {
            "bruno silva": _fighter(rating=1650),
            "douglas silva": _fighter(rating=1450),
            "paddy pimblett": _fighter(),
        })
        pred = await agent.predict_pair(
            _market(), _sharp(label_a="Silva", label_b="Paddy Pimblett"),
        )
        assert pred is None

    @pytest.mark.asyncio
    async def test_unknown_fighter_returns_none(self, tmp_path):
        agent = _agent(tmp_path, {"paddy pimblett": _fighter()})
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is None

    @pytest.mark.asyncio
    async def test_non_ufc_sector_returns_none(self, tmp_path):
        agent = _agent(tmp_path, {
            "paddy pimblett": _fighter(), "benoit saint-denis": _fighter(),
        })
        pred = await agent.predict_pair(_market(sector="tennis"), _sharp())
        assert pred is None

    @pytest.mark.asyncio
    async def test_stale_state_returns_none(self, tmp_path):
        agent = _agent(tmp_path, {
            "paddy pimblett": _fighter(), "benoit saint-denis": _fighter(),
        }, last_updated="2026-01-01")   # 191 days before the July event
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is None

    @pytest.mark.asyncio
    async def test_debutant_confidence_below_gate(self, tmp_path):
        agent = _agent(tmp_path, {
            "paddy pimblett": _fighter(n=0),
            "benoit saint-denis": _fighter(),
        })
        # n=0 path: fighter dict with 0 fights
        fighters = agent._state["fighters"]
        fighters["paddy pimblett"]["n_fights"] = 0
        pred = await agent.predict_pair(_market(), _sharp())
        assert pred is not None
        assert pred.confidence < 0.45

    @pytest.mark.asyncio
    async def test_feature_model_shifts_probability(self, tmp_path):
        fighters = {
            "paddy pimblett": _fighter(rating=1600),
            "benoit saint-denis": _fighter(rating=1600),
        }
        base = _agent(tmp_path, dict(fighters))
        p0 = (await base.predict_pair(_market(), _sharp())).true_prob_a
        # Pimblett recently KO'd — with a fitted negative recent_ko coef the
        # blend must drop his probability below the rating-only line.
        fighters["paddy pimblett"]["last_ko_loss"] = "2026-05-01"
        fm = {"intercept": 0.0, "rating_coef": 1.0, "coefs": {"recent_ko": -0.6}}
        agent = _agent(tmp_path, fighters, feature_model=fm)
        p1 = (await agent.predict_pair(_market(), _sharp())).true_prob_a
        assert p1 < p0


class TestUpdate:
    def test_update_moves_ratings_and_stamps(self, tmp_path):
        agent = _agent(tmp_path, {
            "paddy pimblett": _fighter(rating=1600),
            "benoit saint-denis": _fighter(rating=1600),
        }, last_updated="2026-07-01")
        agent.update("saint-denis", "pimblett", 1.0, 0.0, "ufc", "2026-07-11")
        fighters = agent._state["fighters"]
        assert fighters["benoit saint-denis"]["rating"] > 1600
        assert fighters["paddy pimblett"]["rating"] < 1600
        assert agent._state["last_updated"] == "2026-07-11"

    def test_non_ufc_ignored(self, tmp_path):
        agent = _agent(tmp_path, {"paddy pimblett": _fighter(rating=1600)})
        agent.update("pimblett", "someone", 1.0, 0.0, "tennis", "2026-07-11")
        assert agent._state["fighters"]["paddy pimblett"]["rating"] == 1600


class TestSeedFromHistory:
    def test_empty_history_aborts_without_writing(self, tmp_path):
        agent = _agent(tmp_path, {"paddy pimblett": _fighter()})
        n = agent.seed_from_history([], {})
        assert n == 0
        assert "paddy pimblett" in agent._state["fighters"]  # untouched

    def test_seed_full_replaces(self, tmp_path):
        agent = _agent(tmp_path, {"ghost fighter": _fighter()})
        fights = [_fight("Al Ace", "Bo Bell", "a", "2026-07-01")]
        n = agent.seed_from_history(fights, {})
        assert n == 2
        assert "ghost fighter" not in agent._state["fighters"]
        assert agent._state["last_updated"] == "2026-07-01"
