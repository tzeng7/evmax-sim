"""Tests for NflQbEloModelAgent — Elo with QB-rating layer for NFL."""

from __future__ import annotations

import asyncio

import pytest

from evmax.agents.models.nfl_qb_elo_agent import (
    NflQbEloModelAgent,
    apply_qb_elo_update,
    DEFAULT_ELO,
    HOME_ADVANTAGE_ELO,
    K_FACTOR,
    QB_UPDATE_SHARE,
    QB_DELTA_CLAMP,
)
from evmax.models.market import PredictionMarket, MarketSource
from evmax.models.odds import SharpOdds, SharpBook


def _pair(home: str, away: str, sector: str = "nfl") -> tuple[PredictionMarket, SharpOdds]:
    market = PredictionMarket(
        id="t", market_id="t", event_id=f"t_{home}_{away}",
        sector=sector, team_home=home, team_away=away,
        source=MarketSource.kalshi, yes_price=0.5, no_price=0.5,
    )
    sharp = SharpOdds(
        event_id=f"t_{home}_{away}", book=SharpBook.pinnacle, sector=sector,
        outcome_a_label=home, outcome_b_label=away,
        outcome_a_decimal=2.0, outcome_b_decimal=2.0,
        true_prob_a=0.5, true_prob_b=0.5,
    )
    return market, sharp


def _agent_with(state: dict) -> NflQbEloModelAgent:
    agent = NflQbEloModelAgent()
    agent._state = {"nfl": state}
    return agent


# ── Sector gating ──────────────────────────────────────────────────────────

class TestSectorGating:
    def test_non_nfl_returns_none(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills", sector="nba")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None

    def test_empty_state_returns_none(self):
        agent = NflQbEloModelAgent()
        agent._state = {}
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        assert asyncio.run(agent.predict_pair(market, sharp)) is None


# ── Team / QB resolution ───────────────────────────────────────────────────

class TestResolution:
    def test_full_name_lookup(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
            "qb_deltas": {"P.Mahomes": 100.0, "J.Allen": 80.0},
            "current_starters": {"kansas city chiefs": "P.Mahomes", "buffalo bills": "J.Allen"},
            "qb_games": {"P.Mahomes": 100, "J.Allen": 90},
        })
        market, sharp = _pair("kansas city chiefs", "buffalo bills")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        # Both elite QBs, both teams above-average → P(home) reasonable
        assert 0.40 < result.true_prob_a < 0.85

    def test_nickname_resolves(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
            "qb_deltas": {"P.Mahomes": 100.0, "J.Allen": 80.0},
            "current_starters": {"kansas city chiefs": "P.Mahomes", "buffalo bills": "J.Allen"},
            "qb_games": {"P.Mahomes": 100, "J.Allen": 90},
        })
        market, sharp = _pair("chiefs", "bills")  # Nickname-only
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None

    def test_qb_delta_lookup_case_insensitive(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1500.0},
            "qb_deltas": {"P.Mahomes": 100.0},
        })
        # Stored as "P.Mahomes" — agent should find it even on mixed case
        delta = agent._qb_delta(agent._state["nfl"]["qb_deltas"], "p.mahomes")
        assert delta == 100.0


# ── QB-swap behavior (the whole point of this agent) ──────────────────────

class TestQbSwap:
    def test_starter_change_shifts_probability(self):
        """Swapping the starter to a much weaker QB should drop P(home) materially."""
        state = {
            "team_base": {"miami dolphins": 1520.0, "buffalo bills": 1540.0},
            "qb_deltas": {"T.Tagovailoa": 60.0, "Backup.QB": -80.0, "J.Allen": 90.0},
            "current_starters": {"miami dolphins": "T.Tagovailoa", "buffalo bills": "J.Allen"},
            "qb_games": {"T.Tagovailoa": 70, "Backup.QB": 5, "J.Allen": 90},
        }
        agent = _agent_with(state)
        market, sharp = _pair("miami dolphins", "buffalo bills")
        with_starter = asyncio.run(agent.predict_pair(market, sharp))

        # Now flip Miami's starter to the backup
        state["current_starters"]["miami dolphins"] = "Backup.QB"
        agent_swap = _agent_with(state)
        with_backup = asyncio.run(agent_swap.predict_pair(market, sharp))

        assert with_starter is not None and with_backup is not None
        # Tua → backup represents a 60 - (-80) = 140 Elo point swing for Miami
        # That should clearly show in P(home) — at least ~12pp shift.
        assert with_starter.true_prob_a - with_backup.true_prob_a > 0.10

    def test_unknown_starter_falls_back_to_zero_delta(self):
        """If a starter is in current_starters but missing from qb_deltas, treat
        as average (delta=0) rather than crashing or returning None."""
        state = {
            "team_base": {"miami dolphins": 1520.0, "buffalo bills": 1540.0},
            "qb_deltas": {"J.Allen": 90.0},  # Tua not in this map
            "current_starters": {"miami dolphins": "T.Tagovailoa", "buffalo bills": "J.Allen"},
            "qb_games": {"J.Allen": 90},
        }
        agent = _agent_with(state)
        market, sharp = _pair("miami dolphins", "buffalo bills")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        # Tua treated as 0 delta → home_eff = 1520 + 0 + 48 = 1568,
        # away_eff = 1540 + 90 = 1630 → P(home) ≈ 0.41
        assert 0.35 < result.true_prob_a < 0.50

    def test_no_starter_assigned_lower_confidence(self):
        """Missing starter info should knock confidence down."""
        state = {
            "team_base": {"chicago bears": 1490.0, "detroit lions": 1560.0},
            "qb_deltas": {},
            "current_starters": {},  # Neither team has a known starter
            "qb_games": {},
        }
        agent = _agent_with(state)
        market, sharp = _pair("chicago bears", "detroit lions")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        # Confidence tier when starter info missing
        assert result.confidence == 0.45


# ── Probability semantics ──────────────────────────────────────────────────

class TestProbabilitySemantics:
    def test_even_teams_no_qbs_get_home_edge(self):
        state = {
            "team_base": {"team a": 1500.0, "team b": 1500.0},
            "qb_deltas": {},
            "current_starters": {},
            "qb_games": {},
        }
        agent = _agent_with(state)
        market, sharp = _pair("team a", "team b")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        # 48 Elo home advantage → P(home) ≈ 0.567
        expected = 1.0 / (1.0 + 10 ** (-HOME_ADVANTAGE_ELO / 400))
        assert result.true_prob_a == pytest.approx(expected, abs=0.001)

    def test_probabilities_clamped(self):
        """Even with extreme rating differentials the output stays in [0.02, 0.98]."""
        state = {
            "team_base": {"strong": 2000.0, "weak": 1000.0},
            "qb_deltas": {"Elite.QB": 200.0, "Bad.QB": -200.0},
            "current_starters": {"strong": "Elite.QB", "weak": "Bad.QB"},
            "qb_games": {"Elite.QB": 100, "Bad.QB": 100},
        }
        agent = _agent_with(state)
        market, sharp = _pair("strong", "weak")
        result = asyncio.run(agent.predict_pair(market, sharp))
        assert result is not None
        assert 0.02 <= result.true_prob_a <= 0.98


# ── apply_qb_elo_update — the walk-forward Elo math ───────────────────────

class TestApplyUpdate:
    def test_winner_qb_gains_loser_qb_loses(self):
        state = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            state,
            home_team="kansas city chiefs", away_team="denver broncos",
            home_starter="P.Mahomes", away_starter="B.Nix",
            home_won=True,
        )
        # Both QBs should have moved relative to 0
        assert state["qb_deltas"]["P.Mahomes"] > 0
        assert state["qb_deltas"]["B.Nix"] < 0
        assert state["qb_games"]["P.Mahomes"] == 1
        assert state["qb_games"]["B.Nix"] == 1

    def test_qb_delta_clamped_after_many_wins(self):
        """A QB on a long winning streak should be clamped to QB_DELTA_CLAMP."""
        state = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        for _ in range(200):  # 200 consecutive wins is more than enough to hit cap
            apply_qb_elo_update(
                state,
                home_team="winners", away_team="losers",
                home_starter="GOAT.QB", away_starter="Sad.QB",
                home_won=True,
            )
        assert state["qb_deltas"]["GOAT.QB"] <= QB_DELTA_CLAMP
        assert state["qb_deltas"]["Sad.QB"] >= -QB_DELTA_CLAMP

    def test_team_base_share_plus_qb_share_equals_one(self):
        """Total Elo delta should split correctly: QB_UPDATE_SHARE + (1-...) = 1."""
        state = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            state,
            home_team="A", away_team="B",
            home_starter="qb_a", away_starter="qb_b",
            home_won=True,
        )
        # Home team gains team_share of delta; home QB gains qb_share of delta.
        # The sum (excluding home advantage residual) should reconstruct full delta.
        team_gain = state["team_base"]["A"] - DEFAULT_ELO
        qb_gain = state["qb_deltas"]["qb_a"]
        total = team_gain + qb_gain
        # Equal & opposite for away
        team_loss = DEFAULT_ELO - state["team_base"]["B"]
        qb_loss = -state["qb_deltas"]["qb_b"]
        assert team_gain == pytest.approx(team_loss, abs=0.001)
        assert qb_gain == pytest.approx(qb_loss, abs=0.001)
        # Per-side total should equal full Elo delta (K-factor × prediction error)
        # Just sanity check it's positive and bounded by K_FACTOR
        assert 0 < total < K_FACTOR

    def test_missing_starter_only_updates_team_base(self):
        """If we don't know who the starter was, the QB rating dict shouldn't gain
        a None or empty-string key — only team_base should move."""
        state = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            state,
            home_team="A", away_team="B",
            home_starter=None, away_starter=None,
            home_won=True,
        )
        assert state["team_base"]["A"] > DEFAULT_ELO
        assert state["team_base"]["B"] < DEFAULT_ELO
        assert "" not in state["qb_deltas"]
        assert None not in state["qb_deltas"]
        assert state["qb_games"] == {}


# ── MOV scaling (opt-in, default off) ─────────────────────────────────────

class TestApplyUpdateMov:
    def test_no_score_call_reduces_to_baseline(self):
        """Without scores (or with use_mov=False) the update is identical to
        the prior no-MOV behaviour — regression guard for the existing tests."""
        base = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        mov = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            base, home_team="A", away_team="B",
            home_starter="qb_a", away_starter="qb_b", home_won=True,
        )
        apply_qb_elo_update(
            mov, home_team="A", away_team="B",
            home_starter="qb_a", away_starter="qb_b", home_won=True,
            home_score=27, away_score=24, use_mov=False,
        )
        assert base["team_base"]["A"] == pytest.approx(mov["team_base"]["A"])
        assert base["qb_deltas"]["qb_a"] == pytest.approx(mov["qb_deltas"]["qb_a"])

    def test_blowout_moves_more_than_close_win(self):
        """With use_mov=True a 28-point win should move Elo more than a 1-pt win
        for the same matchup (even pre-game ratings)."""
        close = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        blowout = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            close, home_team="A", away_team="B",
            home_starter="qb_a", away_starter="qb_b", home_won=True,
            home_score=24, away_score=23, use_mov=True,
        )
        apply_qb_elo_update(
            blowout, home_team="A", away_team="B",
            home_starter="qb_a", away_starter="qb_b", home_won=True,
            home_score=42, away_score=14, use_mov=True,
        )
        assert blowout["qb_deltas"]["qb_a"] > close["qb_deltas"]["qb_a"]
        assert blowout["team_base"]["A"] > close["team_base"]["A"]

    def test_mov_disabled_ignores_scores(self):
        """use_mov defaults False, so passing scores without the flag has no
        effect — the multiplier is strictly opt-in."""
        with_scores = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        without = {"team_base": {}, "qb_deltas": {}, "qb_games": {}}
        apply_qb_elo_update(
            with_scores, home_team="A", away_team="B",
            home_starter="qb_a", away_starter="qb_b", home_won=True,
            home_score=42, away_score=0,
        )
        apply_qb_elo_update(
            without, home_team="A", away_team="B",
            home_starter="qb_a", away_starter="qb_b", home_won=True,
        )
        assert with_scores["qb_deltas"]["qb_a"] == pytest.approx(without["qb_deltas"]["qb_a"])


# ── Staleness guard (Gap-3: prior-season seed must not leak in-season) ──────

class TestNflQbEloStalenessGuard:
    """A frozen prior-season seed must not keep firing once a new NFL season
    starts. Both NFL agents have a no-op update() + seed-driven state, so
    without this guard a stale seed silently persists across the offseason —
    the same failure mode that caused +24pp WNBA chalk bias.
    """

    def test_stale_state_returns_none_during_season(self):
        from datetime import date
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
            "seasons_used": [2023, 2024],
        })
        import evmax.agents.models.nfl_qb_elo_agent as mod
        orig = mod.nfl_state_is_stale_for_today
        # Pin "today" to Nov 2025 (active season 2025 > max(seasons_used)=2024)
        mod.nfl_state_is_stale_for_today = lambda s, today=date(2025, 11, 1): orig(s, today)
        try:
            market, sharp = _pair("kansas city chiefs", "buffalo bills")
            assert asyncio.run(agent.predict_pair(market, sharp)) is None
        finally:
            mod.nfl_state_is_stale_for_today = orig

    def test_fresh_state_still_predicts(self):
        from datetime import date
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1600.0, "buffalo bills": 1500.0},
            "seasons_used": [2024, 2025],
        })
        import evmax.agents.models.nfl_qb_elo_agent as mod
        orig = mod.nfl_state_is_stale_for_today
        mod.nfl_state_is_stale_for_today = lambda s, today=date(2025, 11, 1): orig(s, today)
        try:
            market, sharp = _pair("kansas city chiefs", "buffalo bills")
            pred = asyncio.run(agent.predict_pair(market, sharp))
            assert pred is not None
            assert pred.true_prob_a > 0.5  # higher-rated home favored
        finally:
            mod.nfl_state_is_stale_for_today = orig


# ── nfl_state_is_stale_for_today — the freshness helper ────────────────────

class TestNflStaleHelper:
    def test_stale_during_active_window(self):
        from datetime import date
        from evmax.agents.models.nfl_efficiency_agent import nfl_state_is_stale_for_today
        state = {"nfl": {"seasons_used": [2023, 2024]}}
        # Sep-Feb window, max(seasons_used)=2024 < active season 2025
        assert nfl_state_is_stale_for_today(state, today=date(2025, 9, 15)) is True
        assert nfl_state_is_stale_for_today(state, today=date(2025, 12, 1)) is True

    def test_january_belongs_to_prior_season(self):
        from datetime import date
        from evmax.agents.models.nfl_efficiency_agent import nfl_state_is_stale_for_today
        # Jan 2026 is part of the 2025 season — a 2025-seed is FRESH even though
        # 2025 < 2026 calendar year (the WNBA single-year logic would mis-flag this).
        state = {"nfl": {"seasons_used": [2024, 2025]}}
        assert nfl_state_is_stale_for_today(state, today=date(2026, 1, 20)) is False

    def test_fresh_state_not_stale(self):
        from datetime import date
        from evmax.agents.models.nfl_efficiency_agent import nfl_state_is_stale_for_today
        state = {"nfl": {"seasons_used": [2024, 2025]}}
        assert nfl_state_is_stale_for_today(state, today=date(2025, 11, 1)) is False

    def test_offseason_does_not_trigger(self):
        from datetime import date
        from evmax.agents.models.nfl_efficiency_agent import nfl_state_is_stale_for_today
        # Mar-Aug: no NFL games, prior-season state is fine.
        state = {"nfl": {"seasons_used": [2023, 2024]}}
        assert nfl_state_is_stale_for_today(state, today=date(2025, 3, 15)) is False
        assert nfl_state_is_stale_for_today(state, today=date(2025, 7, 1)) is False
        assert nfl_state_is_stale_for_today(state, today=date(2025, 8, 31)) is False

    def test_missing_or_bad_seasons_used_does_not_block(self):
        from datetime import date
        from evmax.agents.models.nfl_efficiency_agent import nfl_state_is_stale_for_today
        assert nfl_state_is_stale_for_today({}, today=date(2025, 11, 1)) is False
        assert nfl_state_is_stale_for_today({"nfl": {}}, today=date(2025, 11, 1)) is False
        assert nfl_state_is_stale_for_today(
            {"nfl": {"seasons_used": []}}, today=date(2025, 11, 1)) is False
        assert nfl_state_is_stale_for_today(
            {"nfl": {"seasons_used": ["bad"]}}, today=date(2025, 11, 1)) is False


# ── update() is no-op (state is seed-driven) ──────────────────────────────

class TestUpdateNoop:
    def test_update_does_not_mutate(self):
        agent = _agent_with({
            "team_base": {"kansas city chiefs": 1550.0, "buffalo bills": 1540.0},
            "qb_deltas": {"P.Mahomes": 100.0},
            "current_starters": {"kansas city chiefs": "P.Mahomes"},
            "qb_games": {"P.Mahomes": 100},
        })
        before = dict(agent._state["nfl"])
        agent.update("kansas city chiefs", "buffalo bills", 27, 24, "nfl", "2025-09-08")
        after = dict(agent._state["nfl"])
        assert before == after


class TestPregameStarters:
    """Depth-chart / injury-aware PRE-game starter overrides (2026-09-03).

    `current_starters` is the LAST game's passer, so a mid-week change was
    invisible until after the game it mattered for. Overrides win when set,
    fall back to `current_starters` when absent, and a failed refresh must
    leave the model exactly as it was (fail-soft)."""

    @staticmethod
    def _agent():
        from evmax.agents.models.nfl_qb_elo_agent import NflQbEloModelAgent
        a = NflQbEloModelAgent()
        a._state = {"nfl": {
            "team_base": {"kansas city chiefs": 1560.0, "denver broncos": 1500.0},
            "qb_deltas": {"P.Mahomes": 120.0, "G.Minshew": -20.0, "B.Nix": 30.0},
            "qb_games": {"P.Mahomes": 80, "G.Minshew": 30, "B.Nix": 30},
            "current_starters": {"kansas city chiefs": "P.Mahomes", "denver broncos": "B.Nix"},
            "seasons_used": [2020, 2021, 2022, 2023, 2024, 2025, 2026],
        }}
        return a

    @staticmethod
    def _predict(agent, home="kansas city chiefs", away="denver broncos"):
        import asyncio
        from evmax.models.market import MarketSource, PredictionMarket
        from evmax.models.odds import SharpBook, SharpOdds
        market = PredictionMarket(id="t", market_id="t", event_id="e", sector="nfl", team_home=home, team_away=away,
                                  source=MarketSource.kalshi, yes_price=0.5, no_price=0.5)
        sharp = SharpOdds(event_id="e", book=SharpBook.pinnacle, sector="nfl", outcome_a_label=home,
                          outcome_b_label=away, outcome_a_decimal=2.0, outcome_b_decimal=2.0,
                          true_prob_a=0.5, true_prob_b=0.5)
        return asyncio.run(agent.predict_pair(market, sharp))

    def test_override_replaces_last_game_passer_and_is_visible_in_notes(self):
        a = self._agent()
        base = self._predict(a)
        assert "home=P.Mahomes(+120,pbp)" in base.notes
        a.set_pregame_starters({"Kansas City Chiefs": "G.Minshew"})
        swapped = self._predict(a)
        assert "home=G.Minshew(-20,depth)" in swapped.notes and "away=B.Nix(+30,pbp)" in swapped.notes
        assert swapped.true_prob_a < base.true_prob_a
        a.clear_pregame_starters()
        assert self._predict(a).true_prob_a == base.true_prob_a

    def test_refresh_uses_client_and_reports_changes(self, monkeypatch):
        a = self._agent()
        import evmax.clients.nfl_depth_charts as dcmod
        monkeypatch.setattr(dcmod, "resolve_pregame_starters", lambda season, **kw: {
            "kansas city chiefs": {"starter": "G.Minshew", "full_name": "Gardner Minshew II", "rank": 2, "skipped": ["Patrick Mahomes"]},
            "denver broncos": {"starter": "B.Nix", "full_name": "Bo Nix", "rank": 1, "skipped": []},
        })
        assert a.refresh_pregame_starters({"kansas city chiefs": {"Patrick Mahomes"}}) == 2
        assert a._pregame_starters == {"kansas city chiefs": "G.Minshew", "denver broncos": "B.Nix"}
        assert a._pregame_starters_source == "depth_chart"

    def test_refresh_is_fail_soft(self, monkeypatch):
        a = self._agent()
        a.set_pregame_starters({"kansas city chiefs": "G.Minshew"})
        import evmax.clients.nfl_depth_charts as dcmod
        def boom(season, **kw): raise RuntimeError("nflverse down")
        monkeypatch.setattr(dcmod, "resolve_pregame_starters", boom)
        assert a.refresh_pregame_starters() == 0
        assert a._pregame_starters == {}
        assert "home=P.Mahomes(+120,pbp)" in self._predict(a).notes

    def test_backup_with_no_history_lowers_confidence(self):
        a = self._agent()
        a.set_pregame_starters({"kansas city chiefs": "C.Oladokun"})   # not in qb_deltas / qb_games
        pred = self._predict(a)
        assert "home=C.Oladokun(+0,depth)" in pred.notes
        assert pred.confidence <= 0.55
