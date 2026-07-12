"""Glicko-2 math tests — anchored to the worked example in Glickman (2013)."""

from __future__ import annotations

import pytest

from evmax.models_ml.glicko2 import (
    DEFAULT_RD,
    GlickoRating,
    inflate_rd,
    update,
    win_probability,
)


class TestPaperExample:
    """Example from http://www.glicko.net/glicko/glicko2.pdf (tau=0.5):
    a 1500/200/0.06 player beats 1400/30, loses to 1550/100 and 1700/300
    → new rating 1464.06, RD 151.52, sigma ≈ 0.05999."""

    def test_matches_published_values(self):
        p = GlickoRating(1500, 200, 0.06)
        results = [
            (GlickoRating(1400, 30, 0.06), 1.0),
            (GlickoRating(1550, 100, 0.06), 0.0),
            (GlickoRating(1700, 300, 0.06), 0.0),
        ]
        new = update(p, results, tau=0.5)
        assert new.rating == pytest.approx(1464.06, abs=0.1)
        assert new.rd == pytest.approx(151.52, abs=0.1)
        assert new.sigma == pytest.approx(0.05999, abs=1e-4)


class TestUpdate:
    def test_winner_gains_loser_loses(self):
        a, b = GlickoRating(), GlickoRating()
        new_a = update(a, [(b, 1.0)])
        new_b = update(b, [(a, 0.0)])
        assert new_a.rating > 1500 > new_b.rating

    def test_rd_shrinks_with_evidence(self):
        a = GlickoRating()
        new_a = update(a, [(GlickoRating(), 1.0)])
        assert new_a.rd < a.rd

    def test_upset_moves_more_than_expected_win(self):
        favorite = GlickoRating(1700, 80)
        underdog = GlickoRating(1400, 80)
        fav_after_win = update(favorite, [(underdog, 1.0)])
        fav_after_loss = update(favorite, [(underdog, 0.0)])
        gain = fav_after_win.rating - favorite.rating
        loss = favorite.rating - fav_after_loss.rating
        assert loss > gain  # losing as a favorite costs more than winning pays

    def test_draw_leaves_equal_players_unchanged(self):
        a = GlickoRating(1500, 100)
        new_a = update(a, [(GlickoRating(1500, 100), 0.5)])
        assert new_a.rating == pytest.approx(1500, abs=1e-6)

    def test_no_results_is_identity(self):
        a = GlickoRating(1600, 90, 0.05)
        assert update(a, []) == a


class TestWinProbability:
    def test_symmetric(self):
        a, b = GlickoRating(1600, 60), GlickoRating(1450, 90)
        assert win_probability(a, b) + win_probability(b, a) == pytest.approx(1.0)

    def test_equal_ratings_are_even(self):
        assert win_probability(GlickoRating(), GlickoRating()) == pytest.approx(0.5)

    def test_higher_rating_favored(self):
        assert win_probability(GlickoRating(1700, 50), GlickoRating(1500, 50)) > 0.7

    def test_uncertainty_shrinks_edge_toward_half(self):
        confident = win_probability(GlickoRating(1700, 40), GlickoRating(1500, 40))
        uncertain = win_probability(GlickoRating(1700, 300), GlickoRating(1500, 300))
        assert 0.5 < uncertain < confident


class TestInflateRd:
    def test_inactivity_grows_rd(self):
        r = GlickoRating(1500, 60, 0.06)
        assert inflate_rd(r, 360).rd > 60

    def test_capped_at_default_ceiling(self):
        r = GlickoRating(1500, 340, 0.20)
        assert inflate_rd(r, 100000).rd == DEFAULT_RD

    def test_zero_days_is_identity(self):
        r = GlickoRating(1500, 60, 0.06)
        assert inflate_rd(r, 0) == r

    def test_rating_untouched(self):
        r = GlickoRating(1622, 60, 0.06)
        assert inflate_rd(r, 500).rating == 1622
