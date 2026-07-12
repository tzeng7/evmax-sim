"""Glicko-2 rating system (Glickman 2013) — used by the UFC rating agent.

Glicko-2 over Elo for MMA: fighters have SPARSE schedules (2-3 bouts/year,
long layoffs), so a per-player rating deviation (RD) that grows with
inactivity and shrinks with evidence is a materially better fit than a fixed
K-factor. The volatility term (sigma) lets a fighter whose results turn
erratic (decline cliffs are common past ~33) move faster.

Implementation follows the standard published algorithm:
  http://www.glicko.net/glicko/glicko2.pdf
with the usual sparse-sport adaptation: each bout is its own rating period,
and the pre-fight RD is inflated for the time elapsed since the fighter's
previous bout (fractional rating periods of ``PERIOD_DAYS``).

Pure functions + a small dataclass — no I/O — so the math is unit-testable
against the worked example in Glickman's paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

# Glicko-2 constants
SCALE = 173.7178          # rating-points per internal unit
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_SIGMA = 0.06
TAU = 0.5                 # volatility constraint (0.3–1.2 typical; 0.5 standard)
EPS = 1e-6

# One nominal rating period for inactivity-based RD inflation. UFC fighters
# average ~2.5 bouts/year; 120 days ≈ one "period" of decay between bouts.
PERIOD_DAYS = 120.0
MAX_RD = 350.0


@dataclass(frozen=True)
class GlickoRating:
    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    sigma: float = DEFAULT_SIGMA

    @property
    def mu(self) -> float:
        return (self.rating - DEFAULT_RATING) / SCALE

    @property
    def phi(self) -> float:
        return self.rd / SCALE


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))


def _expected(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def win_probability(a: GlickoRating, b: GlickoRating) -> float:
    """P(a beats b), discounting for BOTH players' uncertainty.

    Uses the combined deviation form: g(sqrt(phi_a² + phi_b²)) applied to the
    rating gap, which shrinks confident gaps between uncertain fighters
    toward 50%.
    """
    combined_phi = math.sqrt(a.phi ** 2 + b.phi ** 2)
    return 1.0 / (1.0 + math.exp(-_g(combined_phi) * (a.mu - b.mu)))


def inflate_rd(r: GlickoRating, days_inactive: float) -> GlickoRating:
    """Grow RD for time since the player's last rated bout.

    ``phi' = sqrt(phi² + sigma²·t)`` with ``t`` in fractional rating periods,
    capped at the unrated ceiling (350).
    """
    if days_inactive <= 0:
        return r
    t = days_inactive / PERIOD_DAYS
    phi = math.sqrt(r.phi ** 2 + (r.sigma ** 2) * t)
    return replace(r, rd=min(phi * SCALE, MAX_RD))


def _new_sigma(phi: float, sigma: float, v: float, delta: float, tau: float) -> float:
    """Illinois-method iteration for the new volatility (paper step 5)."""
    a = math.log(sigma ** 2)

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta ** 2 - phi ** 2 - v - ex)
        den = 2.0 * (phi ** 2 + v + ex) ** 2
        return num / den - (x - a) / (tau ** 2)

    big_a = a
    if delta ** 2 > phi ** 2 + v:
        big_b = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        big_b = a - k * tau

    fa, fb = f(big_a), f(big_b)
    while abs(big_b - big_a) > EPS:
        big_c = big_a + (big_a - big_b) * fa / (fb - fa)
        fc = f(big_c)
        if fc * fb <= 0:
            big_a, fa = big_b, fb
        else:
            fa = fa / 2.0
        big_b, fb = big_c, fc
    return math.exp(big_a / 2.0)


def update(
    player: GlickoRating,
    results: list[tuple[GlickoRating, float]],
    tau: float = TAU,
) -> GlickoRating:
    """One Glicko-2 rating-period update.

    Args:
        player:  pre-period rating (RD already inflated for inactivity).
        results: list of ``(opponent_rating, score)`` with score 1.0 win,
                 0.5 draw, 0.0 loss.
        tau:     volatility constraint.

    Returns the post-period rating. With no results, only RD grows (handled
    by the caller via :func:`inflate_rd`); we return the input unchanged.
    """
    if not results:
        return player

    mu, phi, sigma = player.mu, player.phi, player.sigma

    v_inv = 0.0
    delta_sum = 0.0
    for opp, score in results:
        g_j = _g(opp.phi)
        e_j = _expected(mu, opp.mu, opp.phi)
        v_inv += g_j * g_j * e_j * (1.0 - e_j)
        delta_sum += g_j * (score - e_j)
    v = 1.0 / v_inv
    delta = v * delta_sum

    sigma_new = _new_sigma(phi, sigma, v, delta, tau)
    phi_star = math.sqrt(phi ** 2 + sigma_new ** 2)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star ** 2) + 1.0 / v)
    mu_new = mu + phi_new ** 2 * delta_sum

    return GlickoRating(
        rating=DEFAULT_RATING + SCALE * mu_new,
        rd=phi_new * SCALE,
        sigma=sigma_new,
    )
