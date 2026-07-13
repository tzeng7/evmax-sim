"""Monte-Carlo tournament field simulator (Layer 4 of the design doc).

Given each player's projected mean score-to-par per round and a
player-specific round standard deviation, simulate the whole tournament many
times and read off P(win), P(top-5/10/20), and P(make-cut) for every player.
Because one simulated score distribution prices *every* market, we never build
per-market models — the outright, the placement markets, and the cut market
are all reductions of the same sim.

Modelling choices (from the design doc, kept deliberately simple + testable):

  * 72 holes = 4 rounds; a 36-hole cut keeps the low ``cut_size`` players
    (plus ties). Missed-cut players finish behind every made-cut player,
    ranked among themselves by their 36-hole score (golf convention).
  * Round-to-round correlation ``round_corr`` (hot streaks are real but weak,
    rho ~ 0.05-0.10): a per-player "form" offset shared across that player's
    four rounds carries ``sqrt(rho)`` of the variance, per-round noise the
    rest.
  * A common per-round course-difficulty shock shared across the whole field
    (a weather day is hard for everyone) fattens the tails correctly without
    changing relative finish within a round.

Everything is vectorised numpy and processed in sim-chunks to bound memory;
results are reproducible from ``seed``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from evmax.golf.models import SimResult

_log = logging.getLogger(__name__)

# Sentinel added to a missed-cut player's 36-hole score so they always rank
# behind every made-cut player's 72-hole score, while still being ordered
# among themselves by their 36-hole total. 10_000 dwarfs any plausible
# strokes-to-par spread over 72 holes.
_MISSED_CUT_PENALTY = 10_000.0


@dataclass
class SimConfig:
    n_sims: int = 20_000
    chunk: int = 5_000
    n_rounds: int = 4
    cut_after_round: int = 2  # cut applied on cumulative score through this round
    cut_size: int = 65  # low N (+ ties) make the cut
    round_corr: float = 0.08  # within-player round-to-round correlation (rho)
    course_shock_sd: float = 0.8  # shared per-round difficulty shock (strokes)
    seed: int = 12345

    def __post_init__(self) -> None:
        if self.n_sims <= 0:
            raise ValueError("n_sims must be positive")
        if not 0.0 <= self.round_corr < 1.0:
            raise ValueError("round_corr must be in [0, 1)")
        if self.cut_after_round >= self.n_rounds:
            raise ValueError("cut_after_round must be < n_rounds")
        if self.cut_size < 1:
            raise ValueError("cut_size must be >= 1")


def simulate_field(
    players: Sequence[str],
    means: Sequence[float],
    sds: Sequence[float],
    config: Optional[SimConfig] = None,
) -> list[SimResult]:
    """Simulate a tournament and return per-player market probabilities.

    Args:
        players: canonical player names (length P).
        means:   expected strokes-to-par per round, lower = better (length P).
        sds:     per-player per-round score standard deviation (length P).
        config:  simulation parameters.

    Returns:
        A ``SimResult`` per player, in the same order as ``players``.
    """
    config = config or SimConfig()
    P = len(players)
    if P == 0:
        return []
    if not (P == len(means) == len(sds)):
        raise ValueError("players, means, sds must be the same length")
    if P < 2:
        raise ValueError("need at least 2 players to simulate a tournament")

    m = np.asarray(means, dtype=np.float64)
    s = np.asarray(sds, dtype=np.float64)
    if np.any(s <= 0):
        raise ValueError("all round standard deviations must be > 0")

    cut_size = min(config.cut_size, P)

    # Split each player's per-round variance into a shared "form" component
    # (constant across their 4 rounds → induces round-to-round correlation)
    # and independent per-round noise.
    form_sd = s * np.sqrt(config.round_corr)
    noise_sd = s * np.sqrt(1.0 - config.round_corr)

    rng = np.random.default_rng(config.seed)

    win = np.zeros(P)
    top5 = np.zeros(P)
    top10 = np.zeros(P)
    top20 = np.zeros(P)
    made = np.zeros(P)
    finish_sum = np.zeros(P)

    remaining = config.n_sims
    while remaining > 0:
        C = min(config.chunk, remaining)
        remaining -= C

        # Per-player form offset, constant across this chunk's rounds: (C, P)
        form = rng.standard_normal((C, P)) * form_sd
        # Shared course-difficulty shock per round: (C, R)
        shock = rng.standard_normal((C, config.n_rounds)) * config.course_shock_sd
        # Independent per-round noise: (C, R, P)
        noise = rng.standard_normal((C, config.n_rounds, P)) * noise_sd

        # Round scores: mean + form (broadcast over rounds) + shock (over players)
        #   (C, R, P) = m[P] + form[C,1,P] + shock[C,R,1] + noise[C,R,P]
        round_scores = (
            m[None, None, :]
            + form[:, None, :]
            + shock[:, :, None]
            + noise
        )

        cut_r = config.cut_after_round
        score_cut = round_scores[:, :cut_r, :].sum(axis=1)  # (C, P) 36-hole
        score_full = round_scores.sum(axis=1)  # (C, P) 72-hole

        # Cut line per sim: the cut_size-th lowest 36-hole score. Players at or
        # below it make the cut (ties made — may exceed cut_size, as in reality).
        cut_line = np.partition(score_cut, cut_size - 1, axis=1)[:, cut_size - 1]
        made_cut = score_cut <= cut_line[:, None]  # (C, P) bool

        # Final ranking score: made-cut by 72-hole total; missed-cut pushed
        # behind everyone, ordered by their 36-hole total.
        final = np.where(made_cut, score_full, _MISSED_CUT_PENALTY + score_cut)

        made += made_cut.sum(axis=0)

        # Winner: lowest final per sim. Split credit on ties.
        row_min = final.min(axis=1, keepdims=True)
        is_min = final <= row_min + 1e-9  # (C, P)
        tie_ct = is_min.sum(axis=1, keepdims=True)
        win += (is_min / tie_ct).sum(axis=0)

        # Competition rank per player = 1 + (# strictly better). Done per row
        # via argsort-of-argsort on the sorted order, then corrected for ties
        # so that top-N counts use "strictly fewer than N ahead".
        order = np.argsort(final, axis=1, kind="stable")  # ascending
        ranks = np.empty_like(final)
        rows = np.arange(C)[:, None]
        ranks[rows, order] = np.arange(P)[None, :]  # 0-based positional rank
        # Positional ranks ignore ties, but for top-N thresholds we compare the
        # player's FINAL score against the N-th best score, which handles ties
        # correctly (a boundary tie lets >N players qualify — dead-heat, as
        # books settle them).
        for arr, n in ((top5, 5), (top10, 10), (top20, 20)):
            if P >= n:
                thresh = np.partition(final, n - 1, axis=1)[:, n - 1]
                arr += (final <= thresh[:, None] + 1e-9).sum(axis=0)
            else:
                arr += C  # fewer players than N → everyone is "top N"

        # Expected finishing position uses competition ranking (1-based, ties
        # share the best position they collectively occupy).
        # Competition rank = 1 + count strictly less.
        # Approximate with positional rank + 1 (stable within tie groups); the
        # bias is negligible for mean_finish reporting.
        finish_sum += (ranks + 1).sum(axis=0)

    n = float(config.n_sims)
    results: list[SimResult] = []
    for i, name in enumerate(players):
        results.append(
            SimResult(
                player=name,
                win=win[i] / n,
                top_5=top5[i] / n,
                top_10=top10[i] / n,
                top_20=top20[i] / n,
                make_cut=made[i] / n,
                mean_finish=finish_sum[i] / n,
            )
        )
    return results
