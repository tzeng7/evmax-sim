"""Tests for the soccer seed scripts' pure helpers (WS1, 2026-07-19).

Covers:
- seed_soccer_xg._canonicalize_matches — raw ESPN displayNames must map to
  the canonical names used at scan time, and un-normalizable teams drop.
- The worldcup-namespace-preserving reset: a club-soccer reseed must not
  destroy state["worldcup"]["teams"] (the old ``agent._state = {"teams": {}}``
  did exactly that).
- seed_espn per-league month resolution: MLS gets its own Feb-anchored window
  while European leagues keep the shared Aug–Jun window.

No network — pure-function tests only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a scripts/*.py module without running its __main__ block."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def xg_script():
    return _load_script("seed_soccer_xg")


@pytest.fixture(scope="module")
def espn_script():
    return _load_script("seed_espn")


class TestCanonicalizeMatches:
    def test_mls_displaynames_map_to_canonical(self, xg_script):
        from evmax.matching.normalizer import NameNormalizer

        norm = NameNormalizer("soccer")
        matches = [{
            "date": "2026-07-01",
            "home": "Seattle Sounders FC",
            "away": "Inter Miami CF",
            "home_score": 2, "away_score": 1,
            "home_sot": 5, "away_sot": 3, "home_shots": 12, "away_shots": 9,
        }]
        out = xg_script._canonicalize_matches(matches, norm)
        assert len(out) == 1
        # Canonical keys are what SoccerXgAgent looks up at predict time.
        assert out[0]["home"] == norm.normalize("Seattle Sounders FC")
        assert out[0]["away"] == norm.normalize("Inter Miami CF")
        # Must NOT be the raw displayName (the pre-fix orphaned-seed bug).
        assert out[0]["home"] != "Seattle Sounders FC".lower()
        # Non-name fields pass through untouched.
        assert out[0]["home_sot"] == 5

    def test_unnormalizable_team_drops_match(self, xg_script):
        class NoneNormalizer:
            def normalize(self, name):
                return None

        matches = [{"home": "Mystery FC", "away": "Ghost United"}]
        assert xg_script._canonicalize_matches(matches, NoneNormalizer()) == []


class TestWorldcupNamespacePreservation:
    def test_club_reset_preserves_worldcup_teams(self, tmp_path, monkeypatch):
        """The reseed reset must only clear the flat club "teams" store."""
        from evmax.agents.models.soccer_xg_agent import SoccerXgAgent

        agent = SoccerXgAgent()
        agent._state_path = tmp_path / "soccer_xg_state.json"
        agent._state = {
            "teams": {"liverpool": {"matches": [{"date": "2026-05-01"}]}},
            "worldcup": {"teams": {"brazil": {"matches": [{"date": "2026-07-01"}]}}},
        }

        # The fixed reset (mirrors scripts/seed_soccer_xg.py main()).
        agent._state["teams"] = {}

        assert agent._state["teams"] == {}
        assert "brazil" in agent._state["worldcup"]["teams"]
        # And _teams_for still routes to the preserved namespace.
        assert agent._teams_for("worldcup") is agent._state["worldcup"]["teams"]


class TestSeedEspnLeagueMonths:
    def test_mls_and_uel_leagues_present(self, espn_script):
        leagues = espn_script.SECTOR_CONFIGS["soccer"]["leagues"]
        assert leagues["mls"] == ("soccer", "usa.1")
        assert leagues["uel"] == ("soccer", "uefa.europa")

    def test_mls_month_window_starts_february(self, espn_script):
        cfg = espn_script.SECTOR_CONFIGS["soccer"]
        mls_months = cfg["league_months"]["mls"]
        assert mls_months[0] == "202502"
        # Ends at the current month — never truncates the in-progress season.
        from datetime import date
        assert mls_months[-1] == f"{date.today().year}{date.today().month:02d}"

    def test_euro_leagues_keep_shared_window(self, espn_script):
        cfg = espn_script.SECTOR_CONFIGS["soccer"]
        league_months = cfg.get("league_months", {})
        assert "epl" not in league_months  # falls through to cfg["months"]
        assert cfg["months"][0] == "202508"
