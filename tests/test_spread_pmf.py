"""Key-number margin PMF for NFL alt-spread pricing (evmax/models_ml/spread_pmf.py).

Pins: the anchor reproduces Pinnacle's main-line devig price (including the
push-conditional whole-number case); rung monotonicity; the key-number jump
(crossing 3 or 7 moves the price far more than crossing an off-key number); the
sign/convention table for favorite and underdog YES, identical to
SpreadDistributionModel.predict; the research phantom table; the true-distance
gate; the missing/invalid-artifact fallback to the normal CDF; the model_sources
token; the NFL contamination rule; and that non-NFL sectors price byte-for-byte
as before. The fit script's data prep, round trip and --dry-run are covered at
the end.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import evmax.models_ml.spread_distribution as sd
import evmax.models_ml.spread_pmf as sp
from evmax.agents.cleanup.contamination import is_contaminated
from evmax.agents.odds.ev_gap_agent import _NON_MODEL_TOKENS, EVGapAgent
from evmax.models.market import MarketSource, MarketType, PredictionMarket
from evmax.models.odds import SharpBook, SharpOdds
from evmax.models_ml.spread_distribution import SpreadDistributionModel

_spec = importlib.util.spec_from_file_location(
    "fit_nfl_margin_pmf",
    Path(__file__).resolve().parents[1] / "scripts" / "fit_nfl_margin_pmf.py",
)
fit_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_script)


@pytest.fixture(autouse=True)
def _fresh_pmf_cache():
    """Each test sees the artifact as it is on disk (or as it monkeypatches it)."""
    sp.clear_margin_pmf_cache()
    yield
    sp.clear_margin_pmf_cache()


def _sharp(spread_line: float, p: float, sector: str = "nfl", is_alt: bool = False) -> SharpOdds:
    # outcome_a = favorite (negative spread_line), as _parse_spread emits it.
    return SharpOdds(
        event_id=f"{sector}::2026-11-15::patriots_vs_seahawks::spread",
        book=SharpBook.pinnacle, sector=sector,
        outcome_a_label="patriots", outcome_b_label="seahawks",
        outcome_a_decimal=1.0 / p, outcome_b_decimal=1.0 / (1.0 - p),
        true_prob_a=p, true_prob_b=1.0 - p, margin=0.03,
        spread_line=spread_line, is_alternate=is_alt,
    )


def _old_normal(spread_line, p, target_line, yes_is_underdog, sigma):
    """The pre-PMF SpreadDistributionModel formula, verbatim."""
    mu = abs(spread_line) - norm.ppf(1.0 - p) * sigma
    if not yes_is_underdog:
        prob = float(1.0 - norm.cdf((-target_line - mu) / sigma))
    else:
        prob = float(norm.cdf((target_line - mu) / sigma))
    return max(0.01, min(0.99, prob)), mu


def _pmf():
    pmf = sp.load_margin_pmf("nfl")
    assert pmf is not None, "shipped artifact data/models/nfl_margin_pmf.json must load"
    return pmf


# ---------------------------------------------------------------------------
# Shipped artifact
# ---------------------------------------------------------------------------

class TestShippedArtifact:
    def test_loads_from_disk_with_metadata(self):
        """Guards the 'declared, not derived' state-file lesson (ncaaf v2): the
        runtime must load the artifact actually committed at the shipped path."""
        path = sp.artifact_path("nfl")
        assert path == Path(__file__).resolve().parents[1] / "data" / "models" / "nfl_margin_pmf.json"
        art = json.loads(path.read_text())
        assert art["schema_version"] == sp.SCHEMA_VERSION
        assert art["sector"] == "nfl"
        assert art["fit_seasons"][0] == 2003
        assert art["created"]
        v = art["validation"]
        assert v["holdout_seasons"][0] == 2019
        assert v["holdout"]["dbrier_x1000_vs_normal14_juice"] < 0
        assert v["walkforward_summary"]["seasons_better"] == v["walkforward_summary"]["seasons"]
        pmf = _pmf()
        assert pmf.sector == "nfl"
        assert pmf.pmf(3.0).sum() == pytest.approx(1.0)

    def test_key_numbers_carry_extra_mass(self):
        pmf = _pmf()
        p = pmf.pmf(3.0)
        idx = {int(k): i for i, k in enumerate(pmf.ks)}
        # 3 and 7 dominate their neighbours; a tie (0) is rare in the NFL.
        assert p[idx[3]] > 2.5 * p[idx[2]] and p[idx[3]] > 2.5 * p[idx[4]]
        assert p[idx[7]] > 1.5 * p[idx[8]]
        assert p[idx[0]] < 0.5 * p[idx[1]]


# ---------------------------------------------------------------------------
# Anchoring
# ---------------------------------------------------------------------------

class TestAnchor:
    @pytest.mark.parametrize("main_line,p", [
        (-1.5, 0.52), (-2.5, 0.514), (-3.5, 0.48), (-6.5, 0.505), (-7.5, 0.46),
        (-10.5, 0.53), (-13.5, 0.50),
    ])
    def test_half_point_anchor_reproduces_devig(self, main_line, p):
        pmf = _pmf()
        mu = pmf.anchor(main_line, p)
        assert mu is not None
        up, dn = pmf.split(mu, abs(main_line), pmf.bucket(abs(main_line)))
        assert up + dn == pytest.approx(1.0)          # half point: no push mass
        assert up == pytest.approx(p, abs=1e-7)

    @pytest.mark.parametrize("main_line,p", [(-3.0, 0.55), (-7.0, 0.465), (-1.0, 0.50), (-10.0, 0.52)])
    def test_whole_number_anchor_is_push_conditional(self, main_line, p):
        pmf = _pmf()
        mu = pmf.anchor(main_line, p)
        up, dn = pmf.split(mu, abs(main_line), pmf.bucket(abs(main_line)))
        push = 1.0 - up - dn
        assert push > 0.02                             # real mass on the integer
        assert up / (up + dn) == pytest.approx(p, abs=1e-7)
        assert up < p                                  # unconditional P(F > a) is lower

    def test_predict_at_main_line_returns_devig_both_sides(self):
        m = SpreadDistributionModel()
        for main_line, p in [(-3.0, 0.55), (-6.5, 0.52)]:
            s = _sharp(main_line, p)
            fav = m.predict(s, target_line=main_line, sector="nfl", yes_is_underdog=False)
            dog = m.predict(s, target_line=-main_line, sector="nfl", yes_is_underdog=True)
            assert fav.method == dog.method == "spread_pmf"
            assert fav.true_prob == pytest.approx(p, abs=1e-6)
            assert dog.true_prob == pytest.approx(1.0 - p, abs=1e-6)

    def test_anchor_without_root_returns_none(self):
        pmf = _pmf()
        assert pmf.anchor(-3.0, 0.999) is None
        assert pmf.anchor(-3.0, 0.0) is None
        assert pmf.anchor(-3.0, 1.0) is None


# ---------------------------------------------------------------------------
# Shape: monotonicity + key-number jumps
# ---------------------------------------------------------------------------

class TestShape:
    def test_favorite_cover_falls_as_the_line_deepens(self):
        m = SpreadDistributionModel()
        s = _sharp(-3.5, 0.50)
        lines = [-x / 2 for x in range(1, 33)]          # -0.5 .. -16.0, halves and wholes
        probs = [m.predict(s, target_line=L, sector="nfl").true_prob for L in lines]
        assert all(a >= b for a, b in zip(probs, probs[1:]))
        assert probs[0] > probs[-1]

    def test_underdog_cover_rises_with_points_received(self):
        m = SpreadDistributionModel()
        s = _sharp(-3.5, 0.50)
        lines = [x / 2 for x in range(-18, 34)]        # dog -9 .. dog +16.5
        probs = [m.predict(s, target_line=L, sector="nfl", yes_is_underdog=True).true_prob
                 for L in lines]
        assert all(a <= b for a, b in zip(probs, probs[1:]))

    @pytest.mark.parametrize("main_line", [-2.5, -3.5, -6.5])
    def test_crossing_3_moves_more_than_crossing_an_off_key_number(self, main_line):
        m = SpreadDistributionModel()
        s = _sharp(main_line, 0.50)

        def fav(L):
            return m.predict(s, target_line=L, sector="nfl").true_prob

        jump3 = fav(-2.5) - fav(-3.5)     # crosses 3
        jump5 = fav(-4.5) - fav(-5.5)     # crosses 5 (off-key)
        jump7 = fav(-6.5) - fav(-7.5)     # crosses 7
        jump9 = fav(-8.5) - fav(-9.5)     # crosses 9 (off-key)
        assert jump3 > 2.5 * jump5
        assert jump7 > 1.5 * jump9
        # the normal CDF spreads the same move evenly (~2.8pp per point)
        n3 = (_old_normal(main_line, .5, -2.5, False, 14.0)[0]
              - _old_normal(main_line, .5, -3.5, False, 14.0)[0])
        assert jump3 > 2.5 * n3


# ---------------------------------------------------------------------------
# Sign conventions + golden phantom table
# ---------------------------------------------------------------------------

# (main_line, devig p, yes_is_underdog, venue target_line, PMF lo, PMF hi, normal σ=14)
# The favorite rows are the research phantom table (2003-25 fit): the normal
# over-prices "favorite lays more across 3/7" and under-prices "favorite gets
# back across 3". NOTE -3 @ p=0.5 → fav -7.5 is 0.317 here, not the research
# sketch's 0.306: the sketch keyed the gamma bucket on the anchored mu, whose
# root at mu=3.0 sits on the bucket jump, so its main line did not reproduce
# p=0.5. This implementation keys the bucket on the main line (exact anchor).
GOLDEN = [
    (-3.0, 0.500, False, -7.5, 0.310, 0.325, 0.374),    # fav -7.5 (crosses 7)
    (-3.0, 0.500, False, -3.5, 0.450, 0.465, 0.486),    # fav -3.5 (hook off 3)
    (-3.5, 0.500, False, -2.5, 0.585, 0.600, 0.528),    # fav -2.5 (buys 3)
    (-2.5, 0.514, False, -7.5, 0.285, 0.297, 0.374),    # broncos/rams ladder game
    (-7.0, 0.500, False, -2.5, 0.698, 0.714, 0.626),
    (-3.0, 0.500, False, -16.5, 0.152, 0.165, 0.167),   # fav wins by 17+
    (-3.0, 0.500, True, 7.5, 0.675, 0.690, 0.626),      # dog +7.5 = 1 - fav -7.5
    (-3.0, 0.500, True, -3.5, 0.285, 0.299, 0.321),     # dog WINS by 4+ (F < -3.5)
    (-3.0, 0.500, False, 3.5, 0.701, 0.715, 0.679),     # fav +3.5 = 1 - (dog -3.5)
]


class TestSignConventionGolden:
    @pytest.mark.parametrize("main,p,dog,target,lo,hi,normal", GOLDEN)
    def test_golden_table(self, main, p, dog, target, lo, hi, normal):
        pred = SpreadDistributionModel().predict(
            _sharp(main, p), target_line=target, sector="nfl", yes_is_underdog=dog)
        assert pred is not None and pred.method == "spread_pmf"
        assert lo <= pred.true_prob <= hi
        assert _old_normal(main, p, target, dog, 14.0)[0] == pytest.approx(normal, abs=1e-3)

    def test_minus_16_5_off_a_minus_3_main_by_side(self):
        """The old CLAUDE.md example ('-16.5 off a -3 main: CDF 0.167 vs book
        0.077') mixed sides: the favorite -16.5 fair is ~0.16 (2003-25 pooled
        empirical 0.159), while ~0.07-0.08 is the UNDERDOG winning by 17+.
        cover_probability is ungated; predict() gates the dog rung (19.5 pts)."""
        pmf = _pmf()
        fav = pmf.cover_probability(-3.0, 0.5, -16.5, yes_is_underdog=False)
        dog_by_17 = pmf.cover_probability(-3.0, 0.5, -16.5, yes_is_underdog=True)
        assert 0.152 <= fav <= 0.165
        assert 0.064 <= dog_by_17 <= 0.080

    def test_reproduces_pooled_empirical_tail_for_minus_3_favorites(self):
        """2003-25 closing -3 favorites (n=986): P(F>3 | F!=3) = 0.474,
        P(F > 7.5) = 0.295, P(F > 3.5) = 0.432. Anchored at the pooled cover
        rate, the PMF reproduces the pooled tail; the normal over-prices it."""
        m = SpreadDistributionModel()
        s = _sharp(-3.0, 0.474)
        assert 0.290 <= m.predict(s, target_line=-7.5, sector="nfl").true_prob <= 0.301
        assert 0.427 <= m.predict(s, target_line=-3.5, sector="nfl").true_prob <= 0.440
        assert _old_normal(-3.0, 0.474, -7.5, False, 14.0)[0] > 0.34

    @pytest.mark.parametrize("L", [0.5, 2.5, 3.0, 3.5, 7.0, 7.5, 10.5, 13.5])
    def test_favorite_minus_L_and_underdog_plus_L_are_complements(self, L):
        m = SpreadDistributionModel()
        s = _sharp(-3.5, 0.47)
        fav = m.predict(s, target_line=-L, sector="nfl", yes_is_underdog=False).true_prob
        dog = m.predict(s, target_line=L, sector="nfl", yes_is_underdog=True).true_prob
        assert fav + dog == pytest.approx(1.0, abs=1e-12)

    def test_underdog_wins_by_X_is_not_the_complement_of_favorite_minus_X(self):
        """Kalshi 'underdog wins by over 3.5' is dog at -3.5: F < -3.5. It is the
        complement of favorite at +3.5, not of favorite at -3.5."""
        m = SpreadDistributionModel()
        s = _sharp(-3.0, 0.50)
        dog_wins_by_4 = m.predict(s, target_line=-3.5, sector="nfl", yes_is_underdog=True).true_prob
        fav_minus = m.predict(s, target_line=-3.5, sector="nfl").true_prob
        fav_plus = m.predict(s, target_line=3.5, sector="nfl").true_prob
        assert dog_wins_by_4 + fav_plus == pytest.approx(1.0, abs=1e-12)
        assert dog_wins_by_4 < 1.0 - fav_minus - 0.2

    def test_cover_probability_matches_predict(self):
        pmf = _pmf()
        s = _sharp(-6.5, 0.52)
        for dog, L in [(False, -9.5), (True, 3.5), (True, -1.5), (False, 2.5)]:
            pred = SpreadDistributionModel().predict(s, target_line=L, sector="nfl", yes_is_underdog=dog)
            raw = pmf.cover_probability(-6.5, 0.52, L, dog)
            assert pred.true_prob == pytest.approx(min(0.99, max(0.01, raw)))

    def test_prediction_fields(self):
        pmf = _pmf()
        pred = SpreadDistributionModel().predict(_sharp(-3.0, 0.5), target_line=-7.5, sector="nfl")
        assert pred.implied_mean == pytest.approx(pmf.anchor(-3.0, 0.5))
        assert pred.sigma == pytest.approx(pmf.sigma(pred.implied_mean))
        assert 12.0 < pred.sigma < 15.0


# ---------------------------------------------------------------------------
# Gate: true distance on the favorite-margin axis
# ---------------------------------------------------------------------------

class TestTrueDistanceGate:
    def test_underdog_deep_rung_rejected_by_true_distance(self):
        """Dog wins by over 16.5 off a -3 main: t=-16.5, 19.5 points away. The
        folded |abs(target)-abs(main)| distance is 13.5, which the normal path
        lets through; the PMF path rejects it."""
        s = _sharp(-3.0, 0.5)
        assert SpreadDistributionModel().predict(
            s, target_line=-16.5, sector="nfl", yes_is_underdog=True) is None

    def test_favorite_deep_rung_inside_gate_is_priced(self):
        s = _sharp(-3.0, 0.5)
        pred = SpreadDistributionModel().predict(s, target_line=-16.5, sector="nfl")
        assert pred is not None and pred.method == "spread_pmf"   # 13.5 from the main line

    def test_gate_boundary_favorite_side(self):
        s = _sharp(-3.5, 0.5)
        m = SpreadDistributionModel()
        assert m.predict(s, target_line=-17.5, sector="nfl") is not None          # t=17.5: 14.0
        assert m.predict(s, target_line=-18.0, sector="nfl") is None              # t=18.0: 14.5
        assert m.predict(s, target_line=10.5, sector="nfl") is not None           # t=-10.5: 14.0
        assert m.predict(s, target_line=11.0, sector="nfl") is None               # t=-11.0: 14.5

    def test_gate_boundary_underdog_side(self):
        s = _sharp(-3.5, 0.5)
        m = SpreadDistributionModel()
        assert m.predict(s, target_line=17.5, sector="nfl", yes_is_underdog=True) is not None  # t=17.5
        assert m.predict(s, target_line=-10.5, sector="nfl", yes_is_underdog=True) is not None  # t=-10.5: 14.0
        assert m.predict(s, target_line=-11.0, sector="nfl", yes_is_underdog=True) is None      # t=-11.0: 14.5

    def test_spread_max_sigma_still_applies(self, monkeypatch):
        monkeypatch.setattr(sd, "SPREAD_MAX_SIGMA", 0.5)
        s = _sharp(-3.0, 0.5)
        assert SpreadDistributionModel().predict(s, target_line=-10.5, sector="nfl") is None  # 7.5 > 7
        assert SpreadDistributionModel().predict(s, target_line=-9.5, sector="nfl") is not None


# ---------------------------------------------------------------------------
# Fallback + switch
# ---------------------------------------------------------------------------

class TestFallback:
    def test_missing_artifact_falls_back_to_normal(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sp, "MODELS_DIR", tmp_path)
        s = _sharp(-3.0, 0.5)
        pred = SpreadDistributionModel().predict(s, target_line=-7.5, sector="nfl")
        assert pred.method == "spread_dist"
        assert pred.sigma == 14.0
        assert pred.true_prob == _old_normal(-3.0, 0.5, -7.5, False, 14.0)[0]
        # the failure is cached (one warning per process, not one per rung)
        assert "nfl" in sp._PMF_CACHE and sp._PMF_CACHE["nfl"] is None

    def test_missing_artifact_logs_once(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sp, "MODELS_DIR", tmp_path)
        calls = []
        monkeypatch.setattr(sp.logger, "warning", lambda event, **kw: calls.append(event))
        m = SpreadDistributionModel()
        for L in (-3.5, -7.5, -10.5):
            m.predict(_sharp(-3.0, 0.5), target_line=L, sector="nfl")
        assert calls == ["spread_pmf_artifact_missing"]

    @pytest.mark.parametrize("mutate", [
        lambda a: a.update(schema_version=2),
        lambda a: a.update(beta=a["beta"][:-1]),
        lambda a: a.update(gamma=a["gamma"][:2]),
        lambda a: a.update(sector="nba"),
        lambda a: a.pop("s0"),
    ])
    def test_invalid_artifact_falls_back_to_normal(self, monkeypatch, tmp_path, mutate):
        art = json.loads(sp.artifact_path("nfl").read_text())
        mutate(art)
        (tmp_path / "nfl_margin_pmf.json").write_text(json.dumps(art))
        monkeypatch.setattr(sp, "MODELS_DIR", tmp_path)
        pred = SpreadDistributionModel().predict(_sharp(-3.0, 0.5), target_line=-7.5, sector="nfl")
        assert pred.method == "spread_dist"

    def test_corrupt_json_falls_back(self, monkeypatch, tmp_path):
        (tmp_path / "nfl_margin_pmf.json").write_text("{not json")
        monkeypatch.setattr(sp, "MODELS_DIR", tmp_path)
        pred = SpreadDistributionModel().predict(_sharp(-3.0, 0.5), target_line=-7.5, sector="nfl")
        assert pred.method == "spread_dist"

    def test_anchor_failure_falls_back_for_that_row(self):
        s = _sharp(-3.0, 0.999)
        pred = SpreadDistributionModel().predict(s, target_line=-3.5, sector="nfl")
        assert pred is not None and pred.method == "spread_dist"

    def test_alternate_rung_anchor_uses_normal(self):
        s = _sharp(-7.0, 0.40, is_alt=True)
        pred = SpreadDistributionModel().predict(s, target_line=-7.5, sector="nfl")
        assert pred.method == "spread_dist"

    def test_removing_sector_from_switch_reverts_to_normal(self, monkeypatch):
        monkeypatch.setattr(sd, "_PMF_SECTORS", set())
        pred = SpreadDistributionModel().predict(_sharp(-3.0, 0.5), target_line=-7.5, sector="nfl")
        assert pred.method == "spread_dist"
        assert pred.true_prob == _old_normal(-3.0, 0.5, -7.5, False, 14.0)[0]
        # the normal path gates on the TRUE axis too (2026-09-22): "dog wins by
        # 16.5" off a −3 main is 19.5 pts out → not priced
        dog = SpreadDistributionModel().predict(
            _sharp(-3.0, 0.5), target_line=-16.5, sector="nfl", yes_is_underdog=True)
        assert dog is None


# ---------------------------------------------------------------------------
# Non-NFL sectors: byte-identical
# ---------------------------------------------------------------------------

class TestOtherSectorsUnchanged:
    @pytest.mark.parametrize("sector,sigma", [("nba", 12.5), ("wnba", 12.5), ("ncaab", 12.5),
                                              ("ncaaw", 11.5), ("ncaaf", 15.0)])
    def test_exact_old_formula_and_token(self, sector, sigma, monkeypatch):
        def _boom(*a, **k):  # the PMF loader must never run for these sectors
            raise AssertionError("PMF consulted for a non-PMF sector")

        monkeypatch.setattr(sp, "load_margin_pmf", _boom)
        m = SpreadDistributionModel()
        for main, p in [(-3.5, 0.52), (-7.0, 0.47), (-10.5, 0.55)]:
            s = _sharp(main, p, sector=sector)
            for dog in (False, True):
                for L in (-12.5, -7.5, -3.5, -1.5, 1.5, 3.5, 7.5, 12.5):
                    pred = m.predict(s, target_line=L, sector=sector, yes_is_underdog=dog)
                    # Gate on the TRUE favorite-margin axis (2026-09-22): a
                    # "dog wins by X" rung (dog, negative L) sits at t = L.
                    t_fav = L if dog else -L
                    if abs(t_fav - abs(main)) > sigma:
                        assert pred is None
                        continue
                    exp_p, exp_mu = _old_normal(main, p, L, dog, sigma)
                    assert pred.true_prob == exp_p           # exact, not approx
                    assert pred.implied_mean == exp_mu
                    assert pred.sigma == sigma
                    assert pred.method == "spread_dist"

    def test_only_nfl_in_switch(self):
        assert sd._PMF_SECTORS == {"nfl"}


# ---------------------------------------------------------------------------
# EV gap token, non-model token, contamination
# ---------------------------------------------------------------------------

def _market(sector: str, line: float, yes_team: str = "patriots") -> PredictionMarket:
    return PredictionMarket(
        id="k1", source=MarketSource.kalshi, sector=sector,
        market_type=MarketType.spread, yes_price=0.20, no_price=0.82,
        team_home="patriots", team_away="seahawks", yes_team=yes_team, line=line,
    )


class TestPipelineWiring:
    def _gap(self, sector, line=-7.5):
        agent = EVGapAgent()
        return agent._evaluate_pair(
            market=_market(sector, line), sharp=_sharp(-3.0, 0.50, sector=sector),
            confidence=95.0, sector=sector, blended_preds={}, injuries={},
            model_sources={}, kelly_base=0.25, steam_events=set(),
        )

    def test_nfl_gap_carries_spread_pmf_token(self):
        gap = self._gap("nfl")
        assert gap is not None
        assert "spread_pmf" in gap.model_sources
        assert "spread_dist" not in gap.model_sources
        assert gap.sharp_true_prob == pytest.approx(0.3169, abs=0.002)

    def test_nba_gap_keeps_spread_dist_token(self):
        gap = self._gap("nba")
        assert gap is not None
        assert "spread_dist" in gap.model_sources
        assert "spread_pmf" not in gap.model_sources

    def test_spread_pmf_is_a_non_model_token(self):
        assert "spread_pmf" in _NON_MODEL_TOKENS
        from evmax.agents.cleanup import contamination, integrity
        assert "spread_pmf" in contamination._NON_MODEL_TOKENS
        assert "spread_pmf" in integrity._NON_MODEL_SOURCE_TOKENS

    @pytest.mark.parametrize("src,mt,expected", [
        ("sharp+spread_dist", "spread", True),
        ("sharp+spread_dist+no_side", "spread", True),
        ("sharp+spread_dist+anchored_entry", "spread", True),
        ("sharp+spread_pmf", "spread", False),
        ("sharp+spread_pmf+no_side+anchored_entry", "spread", False),
        ("sharp+sharp_ladder", "spread", False),
        ("sharp+elo+form+nfl_efficiency", "moneyline", False),
        ("sharp+total_dist", "total", False),
    ])
    def test_nfl_contamination_rule(self, src, mt, expected):
        assert is_contaminated("nfl", mt, src, -7.5) is expected

    def test_other_sectors_spread_dist_not_contaminated(self):
        for sector in ("nba", "wnba", "ncaab", "ncaaw"):
            assert not is_contaminated(sector, "spread", "sharp+spread_dist", -5.5)


# ---------------------------------------------------------------------------
# Fit script
# ---------------------------------------------------------------------------

def _synthetic_schedules(seasons=range(2003, 2021), n_per=220, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    gid = 0
    for season in seasons:
        for _ in range(n_per):
            a = float(rng.choice([1.5, 2.5, 3.0, 3.5, 6.5, 7.0, 9.5]))
            home_fav = bool(rng.integers(0, 2))
            # favorite margin: discrete normal plus extra mass on +/-3 and +/-7
            f = int(np.round(rng.normal(a, 13.5)))
            if rng.random() < 0.12:
                f = int(rng.choice([3, 3, 7, -3, -7]))
            spread_line = a if home_fav else -a
            result = f if home_fav else -f
            rows.append({"game_id": f"g{gid}", "season": season, "spread_line": spread_line,
                         "result": float(result), "home_spread_odds": -110.0,
                         "away_spread_odds": -110.0})
            gid += 1
    return pd.DataFrame(rows)


class TestFitScript:
    def test_prepare_games_sign_convention(self):
        df = pd.DataFrame([
            # home favored by 3 (nflverse spread_line > 0), home wins by 7 → F = +7
            {"season": 2020, "spread_line": 3.0, "result": 7.0,
             "home_spread_odds": -120.0, "away_spread_odds": 100.0},
            # away favored by 6.5, home wins by 2 → favorite (away) margin F = -2
            {"season": 2020, "spread_line": -6.5, "result": 2.0,
             "home_spread_odds": -110.0, "away_spread_odds": -110.0},
            {"season": 2020, "spread_line": None, "result": 1.0,
             "home_spread_odds": -110.0, "away_spread_odds": -110.0},
        ])
        g = fit_script.prepare_games(df, 2020, 2020)
        assert list(g.a) == [3.0, 6.5]
        assert list(g.F) == [7, -2]
        assert g.p_fav.iloc[0] > 0.5                    # favorite juiced to -120
        assert g.p_fav.iloc[1] == pytest.approx(0.5)

    def test_last_complete_season(self):
        from datetime import date
        assert fit_script.last_complete_season(date(2026, 9, 22)) == 2025
        assert fit_script.last_complete_season(date(2027, 2, 1)) == 2025
        assert fit_script.last_complete_season(date(2027, 3, 1)) == 2026

    def test_fit_round_trips_into_runtime_pmf(self):
        games = fit_script.prepare_games(_synthetic_schedules(), 2003, 2020)
        fitter = fit_script.fit_frozen(games)
        assert fitter.fit_ok
        pmf = fitter.to_margin_pmf()
        s0, s1, beta, gam = fitter.unpack(fitter.th)
        # the runtime reproduces the fitter's own distribution
        for mu in (1.0, 3.0, 5.0, 9.0):
            L = -((fit_script.KS - mu) ** 2) / (2 * (s0 + s1 * mu) ** 2) + beta
            L = L + gam[int(fitter.bucket(np.array([mu]))[0])]
            ref = np.exp(L - L.max())
            ref /= ref.sum()
            assert np.allclose(pmf.pmf(mu), ref, atol=1e-12)
        # the synthetic key-number bump is learned
        idx3 = int(3 - pmf.ks[0])
        assert pmf.pmf(3.0)[idx3] > 1.5 * pmf.pmf(3.0)[idx3 + 1]

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        parquet = tmp_path / "sched.parquet"
        _synthetic_schedules().to_parquet(parquet)
        out = tmp_path / "out.json"
        monkeypatch.setattr("sys.argv", [
            "fit_nfl_margin_pmf.py", "--schedules", str(parquet), "--fit-end", "2020",
            "--holdout-start", "2019", "--no-eval", "--out", str(out), "--dry-run",
        ])
        fit_script.main()
        assert not out.exists()

    def test_write_produces_loadable_artifact(self, tmp_path, monkeypatch):
        parquet = tmp_path / "sched.parquet"
        _synthetic_schedules().to_parquet(parquet)
        out = tmp_path / "nfl_margin_pmf.json"
        monkeypatch.setattr("sys.argv", [
            "fit_nfl_margin_pmf.py", "--schedules", str(parquet), "--fit-end", "2020",
            "--holdout-start", "2019", "--no-eval", "--out", str(out),
        ])
        fit_script.main()
        art = json.loads(out.read_text())
        assert art["fit_seasons"] == [2003, 2020]
        assert art["validation"]["holdout_seasons"] == [2019, 2020]
        monkeypatch.setattr(sp, "MODELS_DIR", tmp_path)
        assert sp.load_margin_pmf("nfl") is not None
