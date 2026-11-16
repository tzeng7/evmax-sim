"""Tests for the anchored alternate-line distributions (models_ml/line_pmf.py)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from evmax.models_ml import line_pmf as lp
from evmax.models_ml.line_pmf import FitConfig, LinePMF, fit_line_pmf


def spread_model(**kw):
    base = dict(sector="nfl", market="spread", k_min=-70, k_max=70, family="normal",
                s0=13.5, line_ref=0.0)
    base.update(kw)
    return LinePMF(**base)


def total_model(**kw):
    base = dict(sector="nba", market="total", k_min=120, k_max=330, family="normal",
                s0=18.0, line_ref=225.0)
    base.update(kw)
    return LinePMF(**base)


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


def test_pmf_is_a_distribution_centred_on_mu():
    m = spread_model()
    p = m.pmf(3.0, 3.0)
    assert p.sum() == pytest.approx(1.0)
    assert float((p * m.ks).sum()) == pytest.approx(3.0, abs=0.05)
    assert math.sqrt(float((p * (m.ks - 3.0) ** 2).sum())) == pytest.approx(13.5, abs=0.2)


def test_t_family_has_fatter_tails_than_normal():
    n = spread_model(family="normal")
    t = spread_model(family="t", nu=4.0)
    far = n.ks > 40
    assert t.pmf(0, 0)[far].sum() > 3 * n.pmf(0, 0)[far].sum()


def test_key_multiplier_raises_one_outcome_only():
    plain = spread_model()
    keyed = spread_model(beta={3: math.log(2.5)})
    p0, p1 = plain.pmf(3.0, 3.0), keyed.pmf(3.0, 3.0)
    i3, i4 = int(3 - plain.k_min), int(4 - plain.k_min)
    assert p1[i3] / p1[i4] == pytest.approx(2.5 * p0[i3] / p0[i4])
    assert p1[i3] > p0[i3] and p1[i4] < p0[i4]          # renormalized


def test_exclude_zero_removes_ties():
    m = spread_model(sector="nba", exclude_zero=True)
    assert m.pmf(0.5, 0.5)[int(0 - m.k_min)] == 0.0
    assert m.pmf(0.5, 0.5).sum() == pytest.approx(1.0)


def test_scale_grows_with_line_and_total_and_has_a_floor():
    m = spread_model(s0=12.0, s1=0.1, line_ref=5.0, s2=0.05, total_ref=220.0)
    assert m.scale(5.0) == pytest.approx(12.0)
    assert m.scale(15.0) == pytest.approx(13.0)
    assert m.scale(5.0, total=240.0) == pytest.approx(13.0)
    assert spread_model(s0=0.2).scale(0.0) == 1.0


def test_overtime_component_adds_a_right_tail():
    plain = total_model()
    ot = total_model(p_ot=0.06, d_ot=20.0)
    hi = plain.ks > 225 + 30
    assert ot.pmf(225, 225)[hi].sum() > plain.pmf(225, 225)[hi].sum()
    lo = plain.ks < 225 - 30
    assert ot.pmf(225, 225)[lo].sum() < plain.pmf(225, 225)[lo].sum()


# ---------------------------------------------------------------------------
# anchoring and pricing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line, p", [(3.0, 0.50), (3.0, 0.47), (7.5, 0.55), (0.0, 0.52)])
def test_anchor_reproduces_the_main_line_exactly(line, p):
    m = spread_model(beta={3: math.log(2.5), 7: math.log(1.6)})
    mu = m.anchor(line, p)
    assert m.conditional_survival(mu, line, line) == pytest.approx(p, abs=1e-6)


def test_anchor_rejects_impossible_probabilities():
    m = spread_model()
    assert m.anchor(3.0, 0.0) is None and m.anchor(3.0, 1.0) is None


def test_cover_probability_at_the_main_line_is_the_input():
    m = spread_model()
    assert m.cover_probability(-6.5, 0.53, -6.5) == pytest.approx(0.53, abs=1e-6)
    # the underdog at the mirrored handicap is the complement
    assert m.cover_probability(-6.5, 0.53, 6.5, yes_is_underdog=True) == pytest.approx(
        0.47, abs=1e-6)


def test_cover_probability_is_monotone_in_the_line():
    m = spread_model(beta={3: math.log(2.5)})
    fav = [m.cover_probability(-3.5, 0.5, -t) for t in (0.5, 2.5, 3.5, 6.5, 10.5)]
    assert fav == sorted(fav, reverse=True)
    dog = [m.cover_probability(-3.5, 0.5, t, yes_is_underdog=True) for t in (0.5, 3.5, 7.5)]
    assert dog == sorted(dog)


def test_underdog_wins_by_more_is_not_the_favorite_complement():
    m = spread_model()
    # "underdog wins by over 3.5" = underdog at −3.5: covers iff F < −3.5
    dog_minus = m.cover_probability(-3.5, 0.5, -3.5, yes_is_underdog=True)
    fav_minus = m.cover_probability(-3.5, 0.5, -3.5)
    assert dog_minus < 0.35 and dog_minus + fav_minus < 0.9


def test_a_key_number_makes_the_half_point_across_it_worth_more():
    plain, keyed = spread_model(), spread_model(beta={3: math.log(2.5)})

    def step(m):   # value of moving the favorite from −3.5 to −2.5
        return m.cover_probability(-6.5, 0.5, -2.5) - m.cover_probability(-6.5, 0.5, -3.5)

    assert step(keyed) > 1.5 * step(plain)


def test_whole_number_target_is_push_conditional_and_sides_sum_to_one():
    m = spread_model(beta={3: math.log(2.5)})
    fav = m.cover_probability(-3.5, 0.5, -3.0)
    dog = m.cover_probability(-3.5, 0.5, 3.0, yes_is_underdog=True)
    assert fav + dog == pytest.approx(1.0)


def test_over_probability_and_market_guards():
    m = total_model(p_ot=0.06, d_ot=20.0)
    assert m.over_probability(225.5, 0.5, 225.5) == pytest.approx(0.5, abs=1e-6)
    assert m.over_probability(225.5, 0.5, 215.5) > 0.5 > m.over_probability(225.5, 0.5, 235.5)
    with pytest.raises(ValueError):
        m.cover_probability(-3.5, 0.5, -3.5)
    with pytest.raises(ValueError):
        spread_model().over_probability(225.5, 0.5, 225.5)


def test_overtime_skew_makes_far_overs_worth_more_than_far_unders():
    m = total_model(p_ot=0.08, d_ot=22.0)
    over_far = m.over_probability(225.5, 0.5, 250.5)
    under_far = 1.0 - m.over_probability(225.5, 0.5, 200.5)
    assert over_far > under_far


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def test_artifact_round_trip(tmp_path, monkeypatch):
    m = spread_model(family="t", nu=9.0, s1=0.02, beta={3: 0.9, -3: 0.7, 7: 0.4},
                     meta={"n_games": 123})
    art = json.loads(json.dumps(m.to_artifact()))
    back = LinePMF.from_artifact(art)
    assert back.cover_probability(-3.5, 0.5, -6.5) == pytest.approx(
        m.cover_probability(-3.5, 0.5, -6.5))
    assert back.meta == {"n_games": 123}

    monkeypatch.setattr(lp, "ARTIFACT_DIR", tmp_path)
    lp.clear_line_pmf_cache()
    (tmp_path / "nfl_spread.json").write_text(json.dumps(art))
    assert lp.load_line_pmf("nfl", "spread") is not None
    assert lp.load_line_pmf("nba", "spread") is None            # no artifact
    (tmp_path / "nba_total.json").write_text("{not json")
    assert lp.load_line_pmf("nba", "total") is None             # invalid: never raises
    lp.clear_line_pmf_cache()


@pytest.mark.parametrize("patch", [
    {"schema_version": 99}, {"market": "moneyline"}, {"family": "cauchy"},
    {"k_min": 10, "k_max": 5}, {"p_ot": 1.5},
])
def test_bad_artifacts_are_rejected(patch):
    art = spread_model().to_artifact()
    art.update(patch)
    with pytest.raises(lp.LinePMFArtifactError):
        LinePMF.from_artifact(art)


# ---------------------------------------------------------------------------
# fitting recovers a known distribution
# ---------------------------------------------------------------------------


def _sample(model: LinePMF, lines: np.ndarray, rng) -> np.ndarray:
    out = np.empty(len(lines))
    for i, a in enumerate(lines):
        out[i] = rng.choice(model.ks, p=model.pmf(a, a))
    return out


def test_fit_recovers_scale_and_key_numbers():
    rng = np.random.default_rng(0)
    truth = spread_model(s0=13.0, beta={3: math.log(2.5), 7: math.log(1.7)})
    lines = rng.choice([1.5, 2.5, 3.0, 3.5, 6.5, 7.0, 9.5], size=6000)
    x = _sample(truth, lines, rng)
    fit = fit_line_pmf("nfl", "spread", x, lines,
                       config=FitConfig(family="normal", beta_window=(-12, 12), lam=5.0))
    assert fit.s0 == pytest.approx(13.0, abs=0.7)
    assert math.exp(fit.beta[3]) == pytest.approx(2.5, rel=0.25)
    assert math.exp(fit.beta[7]) == pytest.approx(1.7, rel=0.25)
    assert abs(fit.beta.get(5, 0.0)) < 0.25                    # no spike where none exists
    assert fit.meta["n_games"] == 6000


def test_fit_recovers_overtime_skew_on_totals():
    rng = np.random.default_rng(1)
    truth = total_model(s0=16.0, p_ot=0.07, d_ot=22.0)
    lines = rng.choice([215.5, 222.5, 228.5, 235.5], size=6000)
    x = _sample(truth, lines, rng)
    fit = fit_line_pmf("nba", "total", x, lines,
                       config=FitConfig(family="normal", overtime=True))
    assert 0.02 < fit.p_ot < 0.15 and 12.0 < fit.d_ot < 32.0
    plain = fit_line_pmf("nba", "total", x, lines, config=FitConfig(family="normal"))
    assert fit.meta["nll"] < plain.meta["nll"]


def test_stronger_ridge_shrinks_the_key_multipliers():
    rng = np.random.default_rng(2)
    truth = spread_model(beta={3: math.log(2.5)})
    lines = np.full(3000, 3.5)
    x = _sample(truth, lines, rng)
    loose = fit_line_pmf("nfl", "spread", x, lines,
                         config=FitConfig(family="normal", beta_window=(-10, 10), lam=2.0))
    tight = fit_line_pmf("nfl", "spread", x, lines,
                         config=FitConfig(family="normal", beta_window=(-10, 10), lam=500.0))
    assert abs(tight.beta[3]) < abs(loose.beta[3])


def test_fit_needs_enough_games():
    with pytest.raises(ValueError):
        fit_line_pmf("nfl", "spread", np.zeros(50), np.zeros(50))
