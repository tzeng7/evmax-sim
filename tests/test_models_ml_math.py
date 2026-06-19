"""The shared normal_cdf primitive (models_ml/_math.py) is the single source
that replaced six byte-identical inline copies across the model agents."""

from __future__ import annotations

import pytest

from evmax.models_ml._math import normal_cdf


def test_boundary():
    assert normal_cdf(0.0) == pytest.approx(0.5, abs=1e-3)
    assert normal_cdf(6.0) == pytest.approx(1.0, abs=1e-3)
    assert normal_cdf(-6.0) == pytest.approx(0.0, abs=1e-3)


def test_tail_clamps():
    # Beyond ±6 the approximation hard-clamps.
    assert normal_cdf(-7.0) == 0.0
    assert normal_cdf(7.0) == 1.0


def test_symmetry():
    for x in (0.25, 1.0, 2.5):
        assert normal_cdf(x) + normal_cdf(-x) == pytest.approx(1.0, abs=1e-3)


def test_matches_agent_reexports():
    # The agents re-export this under the legacy private name; they must be the
    # exact same callable so numerics (and the boundary tests) stay identical.
    from evmax.agents.models.efficiency_agent import _normal_cdf as eff
    from evmax.agents.models.nhl_xg_agent import _normal_cdf as nhl
    assert eff is normal_cdf
    assert nhl is normal_cdf
