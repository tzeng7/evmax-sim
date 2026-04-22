"""Tests for the PointProjectionModel conviction tool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from evmax.agents.intelligence.injury_agent import InjuredPlayer, InjuryReport
from evmax.agents.models.possession_sim_agent import PossessionSimAgent
from evmax.models_ml.point_projection import (
    GameProjection,
    PointProjectionModel,
    compute_injury_ortg_delta,
    is_nba_playoff_date,
)


# ── Synthetic state fixtures ──────────────────────────────────────────────


def _stub_efficiency_state() -> dict:
    return {
        "nba": {
            "league_avg_ortg": 114.5,
            "league_avg_drtg": 114.5,
            "league_avg_pace": 100.0,
            "teams": {
                "celtics": {
                    "ortg": 120.0, "drtg": 110.0, "pace": 98.0,
                    "tov_pct": 0.13, "oreb_pct": 0.30,
                    "gp": 82, "full_name": "boston celtics",
                },
                "hawks": {
                    "ortg": 113.0, "drtg": 116.0, "pace": 104.0,
                    "tov_pct": 0.14, "oreb_pct": 0.28,
                    "gp": 82, "full_name": "atlanta hawks",
                },
                "rookies": {  # insufficient-GP team for the gp-gate test
                    "ortg": 110.0, "drtg": 115.0, "pace": 99.0,
                    "tov_pct": 0.14, "oreb_pct": 0.28,
                    "gp": 10, "full_name": "rookie team",
                },
            },
        }
    }


def _stub_elo_state() -> dict:
    return {
        "nba": {"ratings": {"celtics": 1650.0, "hawks": 1520.0}},
        "nfl": {"ratings": {"patriots": 1580.0, "bills": 1600.0}},
    }


def _stub_poisson_state() -> dict:
    return {
        "nfl": {
            "league_avg": {"home": 23.5, "away": 21.5},
            "teams": {
                "patriots": {"attack": 1.10, "defense": 0.95, "games": 17, "full_name": "new england patriots"},
                "bills":    {"attack": 1.20, "defense": 1.00, "games": 17, "full_name": "buffalo bills"},
            },
        },
    }


@pytest.fixture
def stub_model(monkeypatch):
    """A PointProjectionModel with synthetic state pre-loaded (no disk reads)."""

    def _init(self):
        self._poisson_state = _stub_poisson_state()
        self._elo_state = _stub_elo_state()
        self._efficiency_state = _stub_efficiency_state()
        self._possession_agent = None

    monkeypatch.setattr(PointProjectionModel, "_load_state", _init)
    m = PointProjectionModel()

    agent = PossessionSimAgent()
    agent._efficiency_data = _stub_efficiency_state()["nba"]
    monkeypatch.setattr(agent, "_load_efficiency_state", lambda: _stub_efficiency_state()["nba"])
    m._possession_agent = agent
    return m


# ── NBA path (PossessionSim engine) ───────────────────────────────────────


def test_project_nba_uses_possession_sim(stub_model):
    proj = stub_model.project("celtics", "hawks", "nba")
    assert proj is not None
    assert isinstance(proj, GameProjection)
    assert proj.engine == "possession_sim"
    # Celtics are the stronger team at home → negative spread (home favored)
    assert proj.projected_spread < 0
    # Totals should land in a plausible range (stub ORTGs are high → loose upper bound)
    assert 180 < proj.projected_total < 290
    # Sim variance should be positive and in-range
    assert 5 < proj.margin_sigma < 20
    assert 10 < proj.total_sigma < 35
    # Win prob home ∈ (0, 1)
    assert 0.0 < proj.win_prob_home < 1.0
    # Elo populated from stub
    assert proj.home_elo == 1650.0
    assert proj.away_elo == 1520.0


def test_project_nba_returns_none_when_team_missing(stub_model):
    assert stub_model.project("celtics", "unknown_team", "nba") is None
    assert stub_model.project("unknown_team", "hawks", "nba") is None


def test_project_nba_returns_none_when_gp_insufficient(stub_model):
    # "rookies" has gp=10 < 20 → simulate_matchup must reject
    assert stub_model.project("celtics", "rookies", "nba") is None


def test_project_nba_works_without_elo_state(monkeypatch, stub_model):
    # Clear elo state; sim should still run, elo fields just go None
    stub_model._elo_state = {}
    proj = stub_model.project("celtics", "hawks", "nba")
    assert proj is not None
    assert proj.home_elo is None
    assert proj.away_elo is None


# ── Non-NBA path (Poisson+Elo blend) ──────────────────────────────────────


def test_project_nfl_uses_poisson_elo(stub_model):
    proj = stub_model.project("patriots", "bills", "nfl")
    assert proj is not None
    assert proj.engine == "poisson_elo"
    assert proj.home_elo == 1580.0
    assert proj.away_elo == 1600.0
    # Sanity: spread + total + sigmas are populated
    assert proj.margin_sigma > 0
    assert proj.total_sigma > 0


def test_project_nfl_returns_none_when_poisson_missing(stub_model):
    # Not in Poisson state → must return None (no silent league-avg fallback)
    assert stub_model.project("patriots", "unknown_team", "nfl") is None


def test_project_nfl_returns_none_when_elo_missing(stub_model):
    # Strip Elo for one team → must return None
    stub_model._elo_state = {"nfl": {"ratings": {"patriots": 1580.0}}}
    assert stub_model.project("patriots", "bills", "nfl") is None


def test_project_unknown_sector_returns_none(stub_model):
    assert stub_model.project("whatever", "whatever", "cricket") is None


# ── project_from_sharp edge detection ─────────────────────────────────────


def test_project_from_sharp_flags_spread_edge(stub_model):
    # Model projects Celtics as significant favorites; book spread much smaller
    # → spread_play should surface home team.
    out = stub_model.project_from_sharp(
        "celtics", "hawks", "nba",
        book_spread=-1.5,  # book has home favored by only 1.5
    )
    assert out is not None
    proj = out["projection"]
    # If the model's spread is meaningfully more negative than -1.5, home is the play
    if proj.projected_spread < -2.5:  # at least 1 pt edge required
        assert out["spread_play"] == "celtics"
    assert out["spread_edge"] is not None


def test_project_from_sharp_no_edge_when_close(stub_model):
    out = stub_model.project_from_sharp("celtics", "hawks", "nba")
    assert out is not None
    assert out["spread_play"] is None  # no book line → no play
    assert out["total_play"] is None


# ── PossessionSim.simulate_matchup sync helper ────────────────────────────


def test_simulate_matchup_returns_distributions(monkeypatch):
    agent = PossessionSimAgent()
    monkeypatch.setattr(agent, "_load_efficiency_state", lambda: _stub_efficiency_state()["nba"])
    sim = agent.simulate_matchup("celtics", "hawks")
    assert sim is not None
    assert isinstance(sim["margin"], np.ndarray)
    assert isinstance(sim["total"], np.ndarray)
    assert len(sim["margin"]) == len(sim["total"])
    assert 0.0 < sim["prob_a"] < 1.0


def test_simulate_matchup_rejects_low_gp(monkeypatch):
    agent = PossessionSimAgent()
    monkeypatch.setattr(agent, "_load_efficiency_state", lambda: _stub_efficiency_state()["nba"])
    assert agent.simulate_matchup("celtics", "rookies") is None


def test_simulate_matchup_returns_none_when_state_empty(monkeypatch):
    agent = PossessionSimAgent()
    monkeypatch.setattr(agent, "_load_efficiency_state", lambda: {})
    assert agent.simulate_matchup("celtics", "hawks") is None


# ── Playoff adjustment ────────────────────────────────────────────────────


def test_is_nba_playoff_date_detects_window():
    assert is_nba_playoff_date("2026-04-20") is True
    assert is_nba_playoff_date("2026-05-15") is True
    assert is_nba_playoff_date("2026-06-05") is True
    assert is_nba_playoff_date("2026-03-31") is False
    assert is_nba_playoff_date("2026-07-01") is False
    assert is_nba_playoff_date("2026-01-01") is False
    assert is_nba_playoff_date(None) is False
    assert is_nba_playoff_date("not-a-date") is False


def test_playoff_projection_lowers_total(stub_model):
    """Playoff games must project a lower total than the same teams in regular season."""
    proj_reg = stub_model.project("celtics", "hawks", "nba", game_date="2026-02-15")
    proj_po = stub_model.project("celtics", "hawks", "nba", game_date="2026-04-20")

    assert proj_reg is not None and proj_po is not None
    assert proj_reg.is_playoff is False
    assert proj_po.is_playoff is True

    # ORTG factor ~0.962 → total shrinks ~3.8%. On a ~230 total that's ~8-10 pts.
    delta = proj_reg.projected_total - proj_po.projected_total
    assert 5.0 < delta < 15.0, f"Expected ~9pt playoff drop, got {delta:.2f}"


def test_playoff_projection_preserves_margin_direction(stub_model):
    """Both teams shrink proportionally → spread sign should stay the same."""
    proj_reg = stub_model.project("celtics", "hawks", "nba", game_date="2026-02-15")
    proj_po = stub_model.project("celtics", "hawks", "nba", game_date="2026-04-20")
    assert proj_reg.projected_spread < 0  # Celtics favored
    assert proj_po.projected_spread < 0   # Celtics still favored in playoffs


# ── Injury ORTG adjustment ─────────────────────────────────────────────────


def _mk_report(team: str, players: list[tuple[str, str, str]]) -> InjuryReport:
    """Build an InjuryReport from (name, status, tier) triples."""
    report = InjuryReport(team=team, sector="nba")
    for name, status, tier in players:
        report.players.append(InjuredPlayer(
            name=name, position="PG", status=status, tier=tier,
            impact=0.045, injury_type="Hamstring", notes="",
        ))
    return report


def test_compute_injury_ortg_delta_star_out():
    """A single star OUT should drop ORTG by 4 points."""
    report = _mk_report("lakers", [("Luka Doncic", "Out", "star")])
    delta, notes = compute_injury_ortg_delta(report)
    assert delta == -4.0
    assert any("Luka" in n for n in notes)


def test_compute_injury_ortg_delta_two_stars_caps_at_minus_eight():
    """Two stars out = −8 ORTG (hits the cap exactly)."""
    report = _mk_report("lakers", [
        ("Luka Doncic", "Out", "star"),
        ("Austin Reaves", "Out", "star"),
    ])
    delta, _ = compute_injury_ortg_delta(report)
    assert delta == -8.0


def test_compute_injury_ortg_delta_cap_prevents_runaway():
    """Three stars out must not blow past the −8 cap."""
    report = _mk_report("lakers", [
        ("A", "Out", "star"),
        ("B", "Out", "star"),
        ("C", "Out", "star"),
    ])
    delta, _ = compute_injury_ortg_delta(report)
    assert delta == -8.0


def test_compute_injury_ortg_delta_day_to_day_half_impact():
    """Day-to-day star = half of the OUT impact (−2.0 instead of −4.0)."""
    report = _mk_report("lakers", [("X", "Day-To-Day", "star")])
    delta, _ = compute_injury_ortg_delta(report)
    assert delta == -2.0


def test_compute_injury_ortg_delta_rotation_ignored():
    """Rotation players don't move team ORTG at all."""
    report = _mk_report("lakers", [("Bench Guy", "Out", "rotation")])
    delta, notes = compute_injury_ortg_delta(report)
    assert delta == 0.0
    assert notes == []


def test_compute_injury_ortg_delta_questionable_ignored():
    """Questionable status is too noisy to adjust — leave as 0."""
    report = _mk_report("lakers", [("X", "Questionable", "star")])
    delta, _ = compute_injury_ortg_delta(report)
    assert delta == 0.0


def test_compute_injury_ortg_delta_no_report():
    """No report / empty report → delta 0."""
    assert compute_injury_ortg_delta(None) == (0.0, [])
    assert compute_injury_ortg_delta(_mk_report("x", [])) == (0.0, [])


def test_project_applies_injury_ortg_delta(stub_model):
    """When home team loses stars, projected home score should drop materially."""
    proj_healthy = stub_model.project("celtics", "hawks", "nba", game_date="2026-02-15")
    assert proj_healthy is not None

    reports = {
        "boston celtics": _mk_report("boston celtics", [
            ("Jayson Tatum", "Out", "star"),
            ("Jaylen Brown", "Out", "star"),
        ])
    }
    proj_injured = stub_model.project(
        "celtics", "hawks", "nba",
        game_date="2026-02-15",
        injury_reports=reports,
    )
    assert proj_injured is not None

    # The Celtics lose ~8 ORTG → ~8 points off their side of the score.
    # Accept a generous window to tolerate sim noise.
    drop = proj_healthy.home_points - proj_injured.home_points
    assert 3.0 < drop < 12.0, f"Expected home drop from star injuries, got {drop:.2f}"

    # The spread should shift toward home (less negative) because the home
    # team's offense shrunk — the home team is now less favored.
    assert proj_injured.projected_spread > proj_healthy.projected_spread
    assert proj_injured.home_ortg_adj == -8.0
    assert proj_injured.home_injury_notes is not None
    assert "Tatum" in proj_injured.home_injury_notes
    assert "star/" in proj_injured.home_injury_notes.lower()


def test_project_injury_reports_none_is_noop(stub_model):
    """Passing injury_reports=None (or empty) should behave like before."""
    proj_a = stub_model.project("celtics", "hawks", "nba", game_date="2026-02-15")
    proj_b = stub_model.project("celtics", "hawks", "nba", game_date="2026-02-15", injury_reports={})
    assert proj_a.home_points == proj_b.home_points
    assert proj_b.home_ortg_adj == 0.0
    assert proj_b.home_injury_notes is None


def test_simulate_matchup_accepts_ortg_adj(monkeypatch):
    """PossessionSim should accept ortg_adj_a/b and produce lower team totals."""
    agent = PossessionSimAgent()
    monkeypatch.setattr(agent, "_load_efficiency_state", lambda: _stub_efficiency_state()["nba"])

    sim_healthy = agent.simulate_matchup("celtics", "hawks")
    sim_injured = agent.simulate_matchup("celtics", "hawks", ortg_adj_a=-8.0)
    assert sim_healthy is not None and sim_injured is not None

    mean_a_healthy = sim_healthy["scores_a"].mean()
    mean_a_injured = sim_injured["scores_a"].mean()
    assert mean_a_injured < mean_a_healthy
    # −8 ORTG × ~100 possessions ≈ −8 points on the team's score
    diff = mean_a_healthy - mean_a_injured
    assert 3.0 < diff < 14.0, f"Expected ~8pt drop, got {diff:.2f}"
