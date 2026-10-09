"""scripts/eval_nfl_scalar_props.py — Kalshi NFL Ladder / Escalator replication math."""

import math
import sys
from pathlib import Path

import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_nfl_scalar_props as ev  # noqa: E402


@pytest.mark.parametrize("s,expected", [
    (-8, 0.0), (0, 0.0), (9, 0.0), (15, 0.0001), (29, 0.0010),
    (65, 0.0270),   # float floor trap: (60/200)**3 = 0.026999..; Kalshi pays 0.0270
    (125, 0.2160), (199, 0.8573), (200, 1.0), (260, 1.0),
])
def test_escalator_yards_matches_kalshi_schedule(s, expected):
    assert ev.escalator_payoff("recyds", s) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("s,expected", [(0, 0.0), (1, 0.0003), (3, 0.0098), (7, 0.1250),
                                        (13, 0.8006), (14, 1.0), (17, 1.0)])
def test_escalator_receptions_matches_kalshi_schedule(s, expected):
    assert ev.escalator_payoff("rec", s) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("stat,s,expected", [
    ("recyds", 80, 0.20), ("recyds", 200, 0.50), ("recyds", 400, 1.0), ("recyds", 520, 1.0),
    ("rshyds", -3, 0.0), ("rec", 4, 0.20), ("rec", 25, 1.0),
])
def test_ladder_matches_kalshi_schedule(stat, s, expected):
    assert ev.ladder_payoff(stat, s) == pytest.approx(expected)


def test_taker_fee_is_quadratic():
    assert ev.taker_fee(0.5) == pytest.approx(0.0175)
    assert ev.taker_fee(0.03) == pytest.approx(ev.taker_fee(0.97))


def _poisson_survival(lam):
    return lambda y: float(stats.poisson.sf(math.ceil(y) - 1, lam))


def _discrete_uniform_survival(hi):
    # Y uniform on {0, ..., hi}
    return lambda y: max(0.0, min(1.0, (hi + 1 - math.ceil(y)) / (hi + 1)))


@pytest.mark.parametrize("product", ["ladder", "escalator"])
def test_fair_value_equals_direct_expectation_receptions(product):
    lam = 5.0
    direct = sum(stats.poisson.pmf(k, lam) * ev.payoff(product, "rec", k) for k in range(0, 60))
    assert ev.fair_value(product, "rec", _poisson_survival(lam)) == pytest.approx(direct, abs=1e-9)


@pytest.mark.parametrize("product", ["ladder", "escalator"])
def test_fair_value_equals_direct_expectation_yards(product):
    hi = 230
    direct = sum(ev.payoff(product, "recyds", y) for y in range(0, hi + 1)) / (hi + 1)
    assert ev.fair_value(product, "recyds", _discrete_uniform_survival(hi)) == pytest.approx(direct, abs=1e-9)


def test_escalator_is_convex_relative_to_ladder():
    # A wider distribution with the same center raises the escalator (convex) more than the ladder.
    narrow = ev.Survival([(40, 0.9), (60, 0.5), (80, 0.1)], "recyds")
    wide = ev.Survival([(20, 0.9), (60, 0.5), (100, 0.1)], "recyds")
    assert ev.fair_value("escalator", "recyds", wide) > 1.5 * ev.fair_value("escalator", "recyds", narrow)


def test_survival_interpolates_knots_and_is_monotone():
    pts = [(40, 0.86), (60, 0.69), (80, 0.52), (100, 0.345), (130, 0.16)]
    S = ev.Survival(pts, "recyds")
    for t, s in pts:
        assert S(t) == pytest.approx(s, abs=1e-9)
    vals = [S(y) for y in range(0, 300)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
    assert S(0) <= 0.97 and S(299) < 0.01


def test_survival_repairs_non_monotone_quotes():
    S = ev.Survival([(40, 0.80), (50, 0.82), (60, 0.60)], "recyds")
    assert S(40) >= S(50) >= S(60)


def test_isotonic_pools_violators():
    assert ev.isotonic_decreasing([0.9, 0.5, 0.6, 0.2]) == pytest.approx([0.9, 0.55, 0.55, 0.2])


def test_quote_takes_last_candle_at_or_before_ts_and_drops_empty_sides():
    candles = [
        {"end_period_ts": 100, "yes_bid": {"close_dollars": "0.10"}, "yes_ask": {"close_dollars": "0.12"}},
        {"end_period_ts": 200, "yes_bid": {"close_dollars": "0.0000"}, "yes_ask": {"close_dollars": "1.0000"}},
        {"end_period_ts": 300, "yes_bid": {"close_dollars": "0.30"}, "yes_ask": {"close_dollars": "0.31"}},
    ]
    assert ev._quote(candles, 150) == (0.10, 0.12)
    assert ev._quote(candles, 250) == (None, None)
    assert ev._quote(candles, 50) == (None, None)


def test_bias_shift_by_bucket():
    table = [(0.10, 0.02), (0.50, -0.01), (1.0, 0.0)]
    assert ev._shift(0.05, table) == pytest.approx(0.07)
    assert ev._shift(0.30, table) == pytest.approx(0.29)
    assert ev._shift(0.30, None) == 0.30


def test_clustered_mean_matches_hand_computation():
    m, se, G = ev.clustered_mean([1.0, 3.0, 2.0, 6.0], ["a", "a", "b", "b"])
    assert m == pytest.approx(3.0) and G == 2
    # cluster residual sums: a = -2, b = +2 -> sqrt(8)/4 * sqrt(2/1)
    assert se == pytest.approx(math.sqrt(8) / 4 * math.sqrt(2))
