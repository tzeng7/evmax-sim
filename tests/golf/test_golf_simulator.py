"""Capability tests for the Monte-Carlo field simulator.

The simulator is the data-agnostic heart of the module: given per-player mean
and SD it must produce a coherent probability distribution across every market
(win / top-N / make-cut) from ONE set of sims. These tests assert that
coherence and the modelling invariants, plus reproducibility.
"""

import numpy as np
import pytest

from evmax.golf.simulator import SimConfig, simulate_field


def _even_field(p=60, seed=0):
    rng = np.random.default_rng(seed)
    skills = np.sort(rng.normal(0, 0.8, p))[::-1]  # best first
    names = [f"P{i:03d}" for i in range(p)]
    means = list(-skills)  # lower score = better
    sds = [2.8] * p
    return names, means, sds


class TestSimulatorDistribution:
    def test_win_probs_sum_to_one(self):
        names, means, sds = _even_field(80)
        res = simulate_field(names, means, sds, SimConfig(n_sims=20000, seed=1))
        assert abs(sum(r.win for r in res) - 1.0) < 1e-9

    def test_make_cut_sum_equals_cut_size(self):
        names, means, sds = _even_field(100)
        cfg = SimConfig(n_sims=20000, seed=2, cut_size=70)
        res = simulate_field(names, means, sds, cfg)
        # Every sim advances exactly cut_size players (+ ties are rare with
        # continuous scores), so the field's make-cut mass ≈ cut_size.
        assert abs(sum(r.make_cut for r in res) - 70) < 1.0

    def test_top_n_sums_match_n(self):
        names, means, sds = _even_field(90)
        res = simulate_field(names, means, sds, SimConfig(n_sims=20000, seed=3))
        assert abs(sum(r.top_5 for r in res) - 5) < 0.5
        assert abs(sum(r.top_10 for r in res) - 10) < 0.5
        assert abs(sum(r.top_20 for r in res) - 20) < 0.5

    def test_monotonicity_win_le_top5_le_top10_le_top20_le_cut(self):
        names, means, sds = _even_field(70)
        res = simulate_field(names, means, sds, SimConfig(n_sims=20000, seed=4))
        for r in res:
            assert 0.0 <= r.win <= r.top_5 + 1e-9
            assert r.top_5 <= r.top_10 + 1e-9
            assert r.top_10 <= r.top_20 + 1e-9
            assert r.top_20 <= r.make_cut + 1e-9
            assert r.make_cut <= 1.0 + 1e-9

    def test_favourite_beats_longshot(self):
        # A clear favourite (2 SG better) must out-win a replacement player.
        names = ["fav", "mid", "dog"]
        means = [-2.0, 0.0, 2.0]
        sds = [2.6, 2.8, 3.0]
        res = {r.player: r for r in simulate_field(names, means, sds, SimConfig(n_sims=30000, seed=6, cut_size=2))}
        assert res["fav"].win > res["mid"].win > res["dog"].win
        assert res["fav"].make_cut > res["dog"].make_cut

    def test_mean_finish_orders_by_skill(self):
        names, means, sds = _even_field(50)
        res = simulate_field(names, means, sds, SimConfig(n_sims=15000, seed=7))
        # names are best-first; mean_finish should be roughly increasing.
        finishes = [r.mean_finish for r in res]
        # best player finishes better (lower) than the worst
        assert finishes[0] < finishes[-1]


class TestSimulatorReproducibility:
    def test_deterministic_with_seed(self):
        names, means, sds = _even_field(40)
        a = simulate_field(names, means, sds, SimConfig(n_sims=10000, seed=42))
        b = simulate_field(names, means, sds, SimConfig(n_sims=10000, seed=42))
        assert all(abs(x.win - y.win) < 1e-12 for x, y in zip(a, b))

    def test_chunking_is_statistically_equivalent(self):
        names, means, sds = _even_field(40)
        chunked = simulate_field(names, means, sds, SimConfig(n_sims=40000, chunk=2500, seed=9))
        whole = simulate_field(names, means, sds, SimConfig(n_sims=40000, chunk=40000, seed=9))
        # Different chunk sizes consume the rng stream in a different order, so
        # results are not bit-identical — but they must agree within Monte-Carlo
        # noise, and both must be valid distributions.
        assert abs(sum(r.win for r in chunked) - 1.0) < 1e-9
        assert abs(sum(r.win for r in whole) - 1.0) < 1e-9
        assert max(abs(x.win - y.win) for x, y in zip(chunked, whole)) < 0.01


class TestSimulatorEdgeCases:
    def test_empty_field(self):
        assert simulate_field([], [], []) == []

    def test_single_player_rejected(self):
        with pytest.raises(ValueError):
            simulate_field(["solo"], [0.0], [2.8])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            simulate_field(["a", "b"], [0.0], [2.8, 2.8])

    def test_nonpositive_sd_rejected(self):
        with pytest.raises(ValueError):
            simulate_field(["a", "b"], [0.0, 0.0], [2.8, 0.0])

    def test_cut_size_larger_than_field_clamps(self):
        names, means, sds = _even_field(10)
        res = simulate_field(names, means, sds, SimConfig(n_sims=5000, seed=1, cut_size=65))
        # Everyone makes the cut when cut_size >= field.
        assert all(r.make_cut > 0.99 for r in res)

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            SimConfig(round_corr=1.0)
        with pytest.raises(ValueError):
            SimConfig(cut_after_round=4, n_rounds=4)
        with pytest.raises(ValueError):
            SimConfig(n_sims=0)
