"""Tests for the NCAAB / NCAAW opponent-adjusted efficiency stack.

Covers:
- The iterative opponent-adjustment solver (convergence, schedule-strength
  separation, unbeaten/winless edge cases, HCA estimation, empty input)
- D1 universe filtering
- Empirical-Bayes shrinkage + smooth confidence
- Season staleness guard (Nov–Apr window, year-wrapping seasons)
- College-safe team resolution (no last-word fallback; ambiguity → None)
- Agent predict_pair happy path + below-MIN_GAMES + wrong sector + stale state
- Possession sim agent happy path (vectorized core)
"""

from __future__ import annotations

from datetime import date

import pytest

from evmax.agents.models._college_efficiency import (
    college_season_start_year,
    dean_oliver_possessions,
    filter_d1_games,
    opponent_adjust,
    resolve_team,
    shrink_team_stats,
    smooth_confidence,
    state_is_stale_for_today,
)


# ---------------------------------------------------------------------------
# Synthetic game builders
# ---------------------------------------------------------------------------


def _game(home, away, home_oe, away_oe, poss=70.0, neutral=True):
    """Build a solver row from per-100 efficiencies at a fixed possession count."""
    return {
        "home_id": home, "away_id": away, "neutral": neutral,
        "home_pts": home_oe * poss / 100.0, "away_pts": away_oe * poss / 100.0,
        "home_poss": poss, "away_poss": poss,
    }


def _true_oe(o_team, d_opp, league=100.0):
    """Expected OE of a team with true adjO o_team against true adjD d_opp."""
    return o_team * d_opp / league


class TestOpponentAdjustSolver:
    def test_recovers_known_strengths_round_robin(self):
        """Full round-robin with noiseless multiplicative games -> exact recovery."""
        true = {"a": (110.0, 92.0), "b": (104.0, 98.0), "c": (98.0, 104.0), "d": (88.0, 106.0)}
        teams = list(true)
        games = []
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                for _ in range(3):
                    games.append(_game(
                        h, a,
                        _true_oe(true[h][0], true[a][1]),
                        _true_oe(true[a][0], true[h][1]),
                    ))
        out = opponent_adjust(games, estimate_hca=False)
        assert out["converged"]
        for t, (o, d) in true.items():
            assert out["teams"][t]["adj_o"] == pytest.approx(o, abs=1.5)
            assert out["teams"][t]["adj_d"] == pytest.approx(d, abs=1.5)

    def test_schedule_strength_separation(self):
        """Two equal teams, one on a brutal slate, one on cupcakes.

        This is the whole point of the solver: raw efficiency ranks the
        cupcake-slate team far higher; adjusted efficiency must rank them
        (nearly) equal. Includes intra-tier games (s-s, w-w — the college
        "conference play") so the offense-defense incidence graph is
        connected, as it always is on real schedules.
        """
        # a and b are both true (100, 100). a plays elite opponents
        # (110 O / 90 D), b plays weak ones (90 O / 110 D).
        games = []
        for i in range(6):
            strong, weak = f"s{i}", f"w{i}"
            s_next, w_next = f"s{(i + 1) % 6}", f"w{(i + 1) % 6}"
            for _ in range(3):  # each opponent appears often -> clears min_app
                games.append(_game("a", strong, _true_oe(100, 90), _true_oe(110, 100)))
                games.append(_game("b", weak, _true_oe(100, 110), _true_oe(90, 100)))
                games.append(_game(strong, weak, _true_oe(110, 110), _true_oe(90, 90)))
                # conference play within each tier
                games.append(_game(strong, s_next, _true_oe(110, 90), _true_oe(110, 90)))
                games.append(_game(weak, w_next, _true_oe(90, 110), _true_oe(90, 110)))
        out = opponent_adjust(games, estimate_hca=False)
        a, b = out["teams"]["a"], out["teams"]["b"]
        # Raw ratings are wildly apart...
        assert b["raw_o"] - a["raw_o"] > 15
        # ...adjusted ratings converge to (nearly) equal.
        assert abs(a["adj_o"] - b["adj_o"]) < 2.0
        assert abs(a["adj_d"] - b["adj_d"]) < 2.0

    def test_pathological_disconnected_schedule_stays_bounded(self):
        """No intra-tier games -> the incidence graph is disconnected.

        The ridge resolves the resulting degeneracy with a bounded compromise:
        the adjusted gap must stay small even though a raw comparison is off
        by 20 points. (Real schedules never look like this — conference play
        always connects the graph.)
        """
        games = []
        for i in range(6):
            strong, weak = f"s{i}", f"w{i}"
            for _ in range(3):
                games.append(_game("a", strong, _true_oe(100, 90), _true_oe(110, 100)))
                games.append(_game("b", weak, _true_oe(100, 110), _true_oe(90, 100)))
                games.append(_game(strong, weak, _true_oe(110, 110), _true_oe(90, 90)))
        out = opponent_adjust(games, estimate_hca=False)
        a, b = out["teams"]["a"], out["teams"]["b"]
        assert b["raw_o"] - a["raw_o"] > 15
        assert abs(a["adj_o"] - b["adj_o"]) < 3.5
        assert abs((a["adj_o"] - a["adj_d"]) - (b["adj_o"] - b["adj_d"])) < 7.0

    def test_unbeaten_and_winless_teams_stay_finite(self):
        """Efficiency is score-based: extreme records must not blow up."""
        games = []
        others = ["x", "y", "z"]
        for o in others:
            for _ in range(4):
                games.append(_game("unbeaten", o, 120.0, 80.0))
                games.append(_game("winless", o, 80.0, 120.0))
                games.append(_game(o, others[(others.index(o) + 1) % 3], 100.0, 100.0))
        out = opponent_adjust(games, estimate_hca=False)
        ub = out["teams"]["unbeaten"]
        wl = out["teams"]["winless"]
        assert out["converged"]
        for v in (ub["adj_o"], ub["adj_d"], wl["adj_o"], wl["adj_d"]):
            assert 40 < v < 180  # finite and sane
        assert ub["adj_o"] - ub["adj_d"] > 20
        assert wl["adj_o"] - wl["adj_d"] < -20

    def test_hca_estimation(self):
        """Equal teams, home side always +5 per 100 -> hca_eff ~ 5."""
        teams = ["a", "b", "c", "d"]
        games = []
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                for _ in range(3):
                    # home-and-home so venue and strength are not confounded
                    games.append(_game(h, a, 102.5, 97.5, neutral=False))
                    games.append(_game(a, h, 102.5, 97.5, neutral=False))
        out = opponent_adjust(games)
        assert out["hca_eff"] == pytest.approx(5.0, abs=0.8)
        # Ratings stay ~equal — the venue effect is absorbed by hca, not ratings.
        os = [t["adj_o"] for t in out["teams"].values()]
        assert max(os) - min(os) < 1.5

    def test_neutral_games_do_not_feed_hca(self):
        games = [
            _game("a", "b", 105.0, 95.0, neutral=True)
            for _ in range(10)
        ]
        out = opponent_adjust(games)
        assert out["hca_eff"] == 0.0

    def test_empty_input(self):
        out = opponent_adjust([])
        assert out["teams"] == {}
        assert out["converged"] is True

    def test_zero_possession_rows_skipped(self):
        games = [_game("a", "b", 100.0, 100.0)]
        games.append({
            "home_id": "a", "away_id": "b", "neutral": True,
            "home_pts": 70, "away_pts": 70, "home_poss": 0.0, "away_poss": 70.0,
        })
        out = opponent_adjust(games, estimate_hca=False)
        assert out["teams"]["a"]["gp"] == 1


class TestD1Filter:
    def test_low_appearance_teams_dropped_with_their_games(self):
        games = []
        for i in range(10):
            games.append(_game("d1_a", "d1_b", 100, 100))
        games.append(_game("d1_a", "exhibition_team", 130, 60))
        kept, d1_ids = filter_d1_games(games, min_appearances=8)
        assert "exhibition_team" not in d1_ids
        assert len(kept) == 10
        assert all(g["away_id"] != "exhibition_team" for g in kept)

    def test_dean_oliver_possessions(self):
        assert dean_oliver_possessions(60, 20, 12, 10) == pytest.approx(60 + 8.8 + 12 - 10)


class TestShrinkageAndConfidence:
    LEAGUE = {"league_avg_ortg": 104.0, "league_avg_drtg": 104.0, "league_avg_pace": 68.0}

    def test_shrinkage_pulls_small_sample_toward_mean(self):
        stats = {"ortg": 120.0, "drtg": 90.0, "pace": 72.0, "gp": 5}
        out = shrink_team_stats(stats, self.LEAGUE, k=10.0)
        # weight = 5/15 = 1/3 on raw
        assert out["ortg"] == pytest.approx((5 * 120 + 10 * 104) / 15, abs=0.01)
        assert 104.0 < out["ortg"] < 120.0
        assert 90.0 < out["drtg"] < 104.0

    def test_shrinkage_fades_with_sample(self):
        stats = {"ortg": 120.0, "drtg": 90.0, "pace": 72.0, "gp": 100}
        out = shrink_team_stats(stats, self.LEAGUE, k=10.0)
        assert out["ortg"] > 118.0

    def test_zero_gp_passthrough(self):
        stats = {"ortg": 120.0, "drtg": 90.0, "pace": 72.0, "gp": 0}
        out = shrink_team_stats(stats, self.LEAGUE, k=10.0)
        assert out["ortg"] == 120.0

    def test_smooth_confidence_college_ramp(self):
        assert smooth_confidence(0, full_gp=30) == pytest.approx(0.40)
        assert smooth_confidence(30, full_gp=30) == pytest.approx(0.80)
        assert smooth_confidence(99, full_gp=30) == pytest.approx(0.80)
        # Clears the 0.45 ensemble gate at MIN_GAMES=4
        assert smooth_confidence(4, full_gp=30) > 0.45
        assert smooth_confidence(3, full_gp=30) <= 0.45


class TestStalenessGuard:
    def test_season_start_year(self):
        assert college_season_start_year(date(2025, 11, 20)) == 2025
        assert college_season_start_year(date(2026, 2, 10)) == 2025
        assert college_season_start_year(date(2026, 8, 1)) == 2026
        assert college_season_start_year(date(2026, 7, 11)) == 2025

    def test_prior_season_state_is_stale_in_november(self):
        # The WNBA lesson: a fresh season must not open on last year's ratings.
        state = {"source_season": "2025"}
        assert state_is_stale_for_today(state, today=date(2026, 11, 5)) is True
        assert state_is_stale_for_today(state, today=date(2027, 1, 10)) is True

    def test_current_season_state_is_fresh_across_year_wrap(self):
        state = {"source_season": "2025"}
        assert state_is_stale_for_today(state, today=date(2025, 11, 20)) is False
        assert state_is_stale_for_today(state, today=date(2026, 2, 10)) is False
        assert state_is_stale_for_today(state, today=date(2026, 4, 5)) is False

    def test_offseason_relaxes_guard(self):
        state = {"source_season": "2024"}
        assert state_is_stale_for_today(state, today=date(2026, 7, 11)) is False
        assert state_is_stale_for_today(state, today=date(2026, 10, 15)) is False

    def test_missing_or_bad_source_season_never_blocks(self):
        assert state_is_stale_for_today({}, today=date(2026, 12, 1)) is False
        assert state_is_stale_for_today({"source_season": None}, today=date(2026, 12, 1)) is False
        assert state_is_stale_for_today({"source_season": "n/a"}, today=date(2026, 12, 1)) is False


class TestResolveTeam:
    TEAMS = {
        "duke": {"full_name": "duke blue devils", "gp": 20},
        "north carolina": {"full_name": "north carolina tar heels", "gp": 20},
        "north carolina central": {"full_name": "north carolina central eagles", "gp": 20},
        "michigan state": {"full_name": "michigan state spartans", "gp": 20},
    }

    def test_exact_key(self):
        assert resolve_team(self.TEAMS, "Duke")["full_name"] == "duke blue devils"

    def test_full_display_name(self):
        assert resolve_team(self.TEAMS, "Michigan State Spartans")["gp"] == 20

    def test_longest_key_wins_on_prefix(self):
        got = resolve_team(self.TEAMS, "north carolina central eagles")
        assert got is self.TEAMS["north carolina central"]

    def test_no_last_word_fallback(self):
        # "Kentucky Wildcats" must NOT match some other team via "wildcats".
        assert resolve_team(self.TEAMS, "kentucky wildcats") is None

    def test_empty_and_unknown(self):
        assert resolve_team(self.TEAMS, "") is None
        assert resolve_team(self.TEAMS, "gonzaga") is None


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def _mk_state(season="2025"):
    return {
        "source_season": season,
        "league_avg_ortg": 104.0,
        "league_avg_drtg": 104.0,
        "league_avg_pace": 68.0,
        "league_avg_tov_pct": 0.17,
        "hca_eff": 4.0,
        "fetched_at": date.today().isoformat(),
        "teams": {
            "duke": {"ortg": 118.0, "drtg": 92.0, "pace": 68.0, "gp": 25,
                     "tov_pct": 0.15, "full_name": "duke blue devils"},
            "wagner": {"ortg": 95.0, "drtg": 110.0, "pace": 65.0, "gp": 25,
                       "tov_pct": 0.20, "full_name": "wagner seahawks"},
            "newbie": {"ortg": 112.0, "drtg": 95.0, "pace": 68.0, "gp": 2,
                       "tov_pct": 0.16, "full_name": "newbie team"},
        },
    }


def _mk_pair(sector, home, away):
    from evmax.models.market import MarketSource, MarketType, PredictionMarket
    from evmax.models.odds import SharpBook, SharpOdds

    market = PredictionMarket(
        id="t1", event_id=f"{sector}::2026-01-15::{home}_vs_{away}",
        ticker="TEST", title=f"{home} vs {away}",
        yes_price=0.55, no_price=0.45, sector=sector,
        source=MarketSource.kalshi, market_type=MarketType.moneyline,
        team_home=home, team_away=away, yes_team=home,
    )
    sharp = SharpOdds(
        event_id=f"{sector}::2026-01-15::{home}_vs_{away}",
        book=SharpBook.pinnacle, sector=sector,
        team_a=home, team_b=away,
        true_prob_a=0.6, true_prob_b=0.4,
        outcome_a_decimal=1.67, outcome_b_decimal=2.50,
        outcome_a_label=home, outcome_b_label=away,
    )
    return market, sharp


class TestNcaabEfficiencyAgent:
    @pytest.mark.asyncio
    async def test_happy_path_favors_strong_home(self):
        from evmax.agents.models.ncaab_efficiency_agent import NcaabEfficiencyModelAgent

        agent = NcaabEfficiencyModelAgent()
        agent._state = _mk_state(season=str(college_season_start_year(date.today())))
        market, sharp = _mk_pair("ncaab", "duke", "wagner")
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a > 0.85       # elite vs weak — heavy favorite
        assert pred.true_prob_a <= 0.98      # clamped
        assert pred.true_prob_a + pred.true_prob_b == pytest.approx(1.0)
        assert pred.confidence > 0.45        # clears the ensemble gate at gp=25

    @pytest.mark.asyncio
    async def test_below_min_games_returns_none(self):
        from evmax.agents.models.ncaab_efficiency_agent import NcaabEfficiencyModelAgent

        agent = NcaabEfficiencyModelAgent()
        agent._state = _mk_state(season=str(college_season_start_year(date.today())))
        market, sharp = _mk_pair("ncaab", "duke", "newbie")
        assert await agent.predict_pair(market, sharp) is None

    @pytest.mark.asyncio
    async def test_wrong_sector_returns_none(self):
        from evmax.agents.models.ncaab_efficiency_agent import NcaabEfficiencyModelAgent

        agent = NcaabEfficiencyModelAgent()
        agent._state = _mk_state(season=str(college_season_start_year(date.today())))
        market, sharp = _mk_pair("ncaaw", "duke", "wagner")
        assert await agent.predict_pair(market, sharp) is None

    @pytest.mark.asyncio
    async def test_stale_state_returns_none_in_season(self, monkeypatch):
        from evmax.agents.models import _college_efficiency as core
        from evmax.agents.models.ncaab_efficiency_agent import NcaabEfficiencyModelAgent

        class FrozenDate(date):
            @classmethod
            def today(cls):
                return date(2026, 11, 20)

        monkeypatch.setattr(core, "date", FrozenDate)
        agent = NcaabEfficiencyModelAgent()
        agent._state = _mk_state(season="2025")   # prior season in Nov 2026
        market, sharp = _mk_pair("ncaab", "duke", "wagner")
        assert await agent.predict_pair(market, sharp) is None


class TestNcaawEfficiencyAgent:
    @pytest.mark.asyncio
    async def test_happy_path_and_sector_gate(self):
        from evmax.agents.models.ncaaw_efficiency_agent import NcaawEfficiencyModelAgent

        agent = NcaawEfficiencyModelAgent()
        agent._state = _mk_state(season=str(college_season_start_year(date.today())))
        market, sharp = _mk_pair("ncaaw", "duke", "wagner")
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a > 0.80

        market_b, sharp_b = _mk_pair("ncaab", "duke", "wagner")
        assert await agent.predict_pair(market_b, sharp_b) is None


class TestPossessionSimAgents:
    @pytest.mark.asyncio
    async def test_ncaab_sim_happy_path(self):
        from evmax.agents.models.ncaab_possession_sim_agent import NcaabPossessionSimAgent

        agent = NcaabPossessionSimAgent()
        agent._efficiency_data = _mk_state(season=str(college_season_start_year(date.today())))
        market, sharp = _mk_pair("ncaab", "duke", "wagner")
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert pred.true_prob_a > 0.80
        # Margin/total distributions cached for spread/total pricing.
        assert sharp.event_id in agent._margin_cache
        assert sharp.event_id in agent._total_cache
        assert agent._total_cache[sharp.event_id].mean() > 100  # sane college total

    @pytest.mark.asyncio
    async def test_ncaab_sim_below_min_games(self):
        from evmax.agents.models.ncaab_possession_sim_agent import NcaabPossessionSimAgent

        agent = NcaabPossessionSimAgent()
        agent._efficiency_data = _mk_state(season=str(college_season_start_year(date.today())))
        market, sharp = _mk_pair("ncaab", "duke", "newbie")
        assert await agent.predict_pair(market, sharp) is None

    @pytest.mark.asyncio
    async def test_ncaaw_sim_happy_path(self):
        from evmax.agents.models.ncaaw_possession_sim_agent import NcaawPossessionSimAgent

        agent = NcaawPossessionSimAgent()
        agent._efficiency_data = _mk_state(season=str(college_season_start_year(date.today())))
        market, sharp = _mk_pair("ncaaw", "duke", "wagner")
        pred = await agent.predict_pair(market, sharp)
        assert pred is not None
        assert 0.02 <= pred.true_prob_a <= 0.98

    def test_sim_agents_read_separate_state_files(self):
        from evmax.agents.models.ncaab_possession_sim_agent import (
            EFFICIENCY_STATE_PATH as B_PATH,
        )
        from evmax.agents.models.ncaaw_possession_sim_agent import (
            EFFICIENCY_STATE_PATH as W_PATH,
        )
        assert B_PATH != W_PATH
        assert B_PATH.name == "ncaab_efficiency_state.json"
        assert W_PATH.name == "ncaaw_efficiency_state.json"

    def test_college_stack_does_not_import_nba_or_wnba(self):
        """Parallel-stack rule: no imports from the NBA/WNBA stacks."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1] / "evmax" / "agents" / "models"
        for fname in (
            "_college_efficiency.py",
            "ncaab_efficiency_agent.py", "ncaab_possession_sim_agent.py",
            "ncaaw_efficiency_agent.py", "ncaaw_possession_sim_agent.py",
        ):
            src = (root / fname).read_text()
            for forbidden in ("models.efficiency_agent", "wnba_efficiency",
                              "models.possession_sim_agent", "wnba_possession",
                              "_nba_freshness"):
                assert forbidden not in src, f"{fname} references the NBA/WNBA stack"


class TestCollegeWiring:
    """Phase 2 wiring: registry, ensemble overrides, MODEL-2 elo constants."""

    def test_known_models_registered(self):
        from evmax.categories import KNOWN_MODELS

        for name in ("ncaab_efficiency", "ncaab_possession_sim",
                     "ncaaw_efficiency", "ncaaw_possession_sim"):
            assert name in KNOWN_MODELS

    def test_category_model_lists(self):
        from evmax.categories import get_category

        ncaab = get_category("ncaab").models
        assert "ncaab_efficiency" in ncaab
        assert "ncaab_possession_sim" in ncaab
        # Stale drift removed: poisson has been agent-gated to soccer/worldcup
        # since 2026-06-01 and never fired for basketball.
        assert "poisson" not in ncaab

        ncaaw = get_category("ncaaw").models
        assert "ncaaw_efficiency" in ncaaw
        assert "ncaaw_possession_sim" in ncaaw
        assert "elo" in ncaaw   # MODEL-2: elo now genuinely contributes

    def test_sector_weight_overrides(self):
        from evmax.agents.models.ensemble_agent import EnsembleModelAgent

        for sector, eff_name, sim_name in (
            ("ncaab", "ncaab_efficiency", "ncaab_possession_sim"),
            ("ncaaw", "ncaaw_efficiency", "ncaaw_possession_sim"),
        ):
            w = EnsembleModelAgent.SECTOR_WEIGHT_OVERRIDES[sector]
            assert w[eff_name] == pytest.approx(0.60)
            assert w[sim_name] == pytest.approx(0.20)
            assert w["elo"] == pytest.approx(0.10)
            assert w["form"] == pytest.approx(0.10)
            # No listed model at zero weight (the tennis gotcha: a 0-weight
            # model vanishes from model_sources).
            assert all(v > 0 for v in w.values())

    def test_ncaaw_elo_constants_calibrated(self):
        """MODEL-2: NCAAW no longer rides the uncalibrated NBA defaults."""
        from evmax.agents.models.elo_agent import HOME_ADVANTAGE_ELO, K_FACTORS

        assert K_FACTORS["ncaaw"] == 35.0
        assert HOME_ADVANTAGE_ELO["ncaaw"] == 80.0

    def test_shipped_state_files_exist_and_are_current_season(self):
        import json
        import pathlib

        models_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "models"
        for fname in ("ncaab_efficiency_state.json", "ncaaw_efficiency_state.json"):
            path = models_dir / fname
            assert path.exists(), f"{fname} must ship seeded"
            state = json.loads(path.read_text())
            assert len(state["teams"]) > 300          # full D1 universe
            assert state["source_season"] == "2025"   # the completed 2025-26 season
            assert state["solver_converged"] is True
            assert state["hca_eff"] > 0               # home edge estimated

    def test_agents_share_shrink_constants_with_their_sims(self):
        """Both consumers of a state file must see identical regularization."""
        from evmax.agents.models import ncaab_efficiency_agent as beff
        from evmax.agents.models import ncaab_possession_sim_agent as bsim
        from evmax.agents.models import ncaaw_efficiency_agent as weff
        from evmax.agents.models import ncaaw_possession_sim_agent as wsim

        assert bsim.SHRINK_K == beff.SHRINK_K
        assert bsim.MIN_GAMES == beff.MIN_GAMES
        assert wsim.SHRINK_K == weff.SHRINK_K
        assert wsim.MIN_GAMES == weff.MIN_GAMES
