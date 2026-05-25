"""Early-season K-factor boost on EloModelAgent.

The boost addresses a real problem: after the offseason regression shrinks
WNBA ratings ~35% toward 1500, the first month of games at baseline K=20
moves elo too slowly to overcome the regressed prior. Without the boost,
the WNBA blend was effectively pricing late May games on April's roster
projections.

See `EARLY_K_BOOST` / `EARLY_K_DECAY_GAMES` in elo_agent.py for the
sector configuration.
"""

from __future__ import annotations

import pytest

from evmax.agents.models.elo_agent import (
    EARLY_K_BOOST,
    EARLY_K_DECAY_GAMES,
    EloModelAgent,
    K_FACTORS,
    early_season_multiplier,
)


class TestEarlySeasonMultiplier:
    def test_wnba_full_boost_at_season_start(self) -> None:
        assert early_season_multiplier("wnba", 0) == pytest.approx(3.0)

    def test_wnba_decays_to_one_at_full_decay(self) -> None:
        decay = EARLY_K_DECAY_GAMES["wnba"]
        assert early_season_multiplier("wnba", decay) == pytest.approx(1.0)

    def test_wnba_floor_after_decay(self) -> None:
        # Past the decay window the multiplier never goes below 1.0.
        decay = EARLY_K_DECAY_GAMES["wnba"]
        assert early_season_multiplier("wnba", decay * 3) == pytest.approx(1.0)

    def test_wnba_linear_decay(self) -> None:
        # Halfway through decay → halfway between BOOST and 1.0.
        boost = EARLY_K_BOOST["wnba"]
        decay = EARLY_K_DECAY_GAMES["wnba"]
        expected = boost - (boost - 1.0) * 0.5
        assert early_season_multiplier("wnba", decay // 2) == pytest.approx(expected)

    def test_unconfigured_sector_returns_one(self) -> None:
        # NBA / NFL / baseball are not in the boost config — their elo state
        # never gets offseason-regressed, so the boost must NOT fire.
        assert early_season_multiplier("nba", 0) == 1.0
        assert early_season_multiplier("nfl", 0) == 1.0
        assert early_season_multiplier("baseball", 0) == 1.0


class TestEloAgentBoostIntegration:
    def _agent(self) -> EloModelAgent:
        a = EloModelAgent()
        a._state = {}  # isolate from disk-loaded state
        return a

    def test_wnba_first_game_uses_boosted_k(self) -> None:
        # Fresh state: both teams at 1500, season_games=0 → boost=3.0.
        # Underdog win vs equal prior should move elo by K_eff*0.5 ≈ 30
        # instead of the baseline 10.
        a = self._agent()
        a.update("aces", "sun", score_a=100, score_b=80, sector="wnba")
        # Aces won by 20 from a tied prior — at K=60 (boosted) plus MOV/SoS
        # multipliers this should push them well past +20 elo. Baseline K=20
        # alone (without MOV/SoS) would yield ~+10.
        aces = a._get_rating("wnba", "aces")
        sun = a._get_rating("wnba", "sun")
        assert aces - 1500 > 20, (
            f"early-K boost not firing: aces only moved to {aces} from 1500"
        )
        # Zero-sum: A's gain == B's loss
        assert (aces - 1500) == pytest.approx(-(sun - 1500), abs=0.01)

    def test_wnba_past_decay_uses_baseline_k(self) -> None:
        # Seed both teams' season_games at the decay threshold → multiplier=1.0
        a = self._agent()
        a._sector_state("wnba")  # initialize structure
        decay = EARLY_K_DECAY_GAMES["wnba"]
        a._state["wnba"]["season_games"] = {"aces": decay, "sun": decay}
        a._state["wnba"]["ratings"] = {"aces": 1500.0, "sun": 1500.0}
        a.update("aces", "sun", score_a=100, score_b=80, sector="wnba")
        boosted_delta = a._get_rating("wnba", "aces") - 1500.0

        # Now repeat at full boost
        b = self._agent()
        b._sector_state("wnba")
        b._state["wnba"]["season_games"] = {"aces": 0, "sun": 0}
        b._state["wnba"]["ratings"] = {"aces": 1500.0, "sun": 1500.0}
        b.update("aces", "sun", score_a=100, score_b=80, sector="wnba")
        boost_delta = b._get_rating("wnba", "aces") - 1500.0

        # Boosted delta must be ~3x baseline (BOOST=3.0)
        assert boost_delta == pytest.approx(boosted_delta * 3.0, rel=0.05)

    def test_nba_unaffected_by_boost_config(self) -> None:
        # NBA isn't in the boost config — first game must use baseline K.
        a = self._agent()
        a.update("lakers", "warriors", score_a=110, score_b=100, sector="nba")
        # Baseline NBA K=20, MOV+SoS multipliers ~1.0 for an evenly-matched
        # 10pt win. Delta should be in the ~5-10 range, not 30+.
        lakers = a._get_rating("nba", "lakers")
        assert (lakers - 1500) < 15, (
            f"NBA boost leaked: lakers moved to {lakers} from 1500 after one game"
        )

    def test_update_increments_both_counters(self) -> None:
        a = self._agent()
        a.update("aces", "sun", score_a=100, score_b=80, sector="wnba")
        state = a._state["wnba"]
        assert state["game_counts"]["aces"] == 1
        assert state["game_counts"]["sun"] == 1
        assert state["season_games"]["aces"] == 1
        assert state["season_games"]["sun"] == 1

    def test_legacy_state_without_season_games_is_treated_past_decay(self) -> None:
        # Existing on-disk NBA state has no `season_games` field. Loading it
        # must not accidentally fire the boost — every team should be seeded
        # at the decay threshold. (NBA's boost config is empty so this is
        # moot for the multiplier itself, but the WNBA-state-on-disk case
        # before this rev would also trip without the back-compat init.)
        a = self._agent()
        a._state = {"wnba": {
            "ratings": {"aces": 1550.0, "sun": 1450.0},
            "game_counts": {"aces": 50, "sun": 48},
            "h2h": {},
            # no season_games key
        }}
        a._sector_state("wnba")  # triggers back-compat init
        sg = a._state["wnba"]["season_games"]
        decay = EARLY_K_DECAY_GAMES["wnba"]
        assert sg["aces"] == decay
        assert sg["sun"] == decay


class TestOffseasonRegressionResetsCounter:
    def test_apply_offseason_zeros_season_games(self) -> None:
        from scripts.wnba_offseason_regress import apply_offseason

        state = {
            "wnba": {
                "ratings": {"aces": 1593.51, "sun": 1451.56},
                "game_counts": {"aces": 58, "sun": 46},
                "season_games": {"aces": 58, "sun": 46},  # stale
                "h2h": {},
            },
        }
        cfg = {
            "regression_coefficient": 0.65,
            "moves": [],
            "expansion_priors": {"fire": 1470.0},
        }
        apply_offseason(state, cfg)
        sg = state["wnba"]["season_games"]
        assert sg["aces"] == 0
        assert sg["sun"] == 0
        assert sg["fire"] == 0  # expansion teams reset too
        # lifetime game_counts must NOT be touched
        assert state["wnba"]["game_counts"]["aces"] == 58
