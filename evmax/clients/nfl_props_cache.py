"""NFL player-prop probability model — pure compute layer.

Mirrors the NBA model in `nba_props_cache.py` but adapted for NFL's
weekly cadence and different stat distributions. Stage 4 of the NFL prop
backtest plan provides only the pure computation; Stage 5 will wrap this
with a disk-backed cache (`data/nfl_props_cache.json`) matching the NBA
schema, so the coordinator can call one helper regardless of data source.

Model stages for yardage props (passing/rushing/receiving yards):

  1. Filter the player's prior-week game log (point-in-time).
  2. Exponential decay weights (most recent game = 1.0, decay=0.80).
  3. Weighted mean + weighted std.
  4. Normal CDF with continuity correction → model probability.
  5. Empirical hit rate (60/40 blend with model).
  6. Margin adjustment (±8% cap).
  7. Streak adjustment from last 3 games.
  8. Multiplicative opponent adjustment (±15% cap).

Model for touchdown props (anytime TD, passing TDs 1.5+/2.5+):

  - Weighted-mean expected TDs (λ) then Poisson tail.
  - Opponent adjustment scales λ.
  - Streak/margin adjustments do not apply — noisy at low counts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm, poisson

# Constants — CLAUDE.md testing policy requires tests; tune values live in
# tests/test_nfl_props_cache.py which calls the pure functions directly.
LAST_N_GAMES = 8
MIN_GAMES = 4
DECAY = 0.80
MAX_OPP_ADJ = 0.15
MODEL_EMPIRICAL_BLEND = 0.60  # 60% model / 40% empirical hit rate
EMPIRICAL_DISAGREE_CAP = 0.15 # if empirical disagrees with Gaussian by more than
                              # this, the empirical signal is being driven by
                              # 1-2 outlier games. Cap the blend contribution.
MARGIN_SCALE = 0.01           # per-unit margin → probability nudge
MARGIN_CAP = 0.08
MIN_STD = 10.0                # floor on yard std to avoid overconfident Gaussians

YARDAGE_STATS = {"passing_yards", "rushing_yards", "receiving_yards"}
TD_STATS = {"anytime_td", "passing_tds", "receptions"}
# NOTE: receptions is not a TD stat — it's a count stat handled like yards
# but with integer thresholds. We override via the dispatch table below.

STAT_TO_BRANCH = {
    "passing_yards": "yardage",
    "rushing_yards": "yardage",
    "receiving_yards": "yardage",
    "receptions": "count",     # uses yardage branch with MIN_STD=1.5
    "passing_tds": "poisson",  # ≥ 1.5 or ≥ 2.5
    "anytime_td": "poisson",   # ≥ 0.5
}

MIN_STD_BY_STAT = {
    # Floors doubled from initial Stage 4 values — initial floors assumed
    # a tighter per-game distribution than NFL actually produces (game
    # script, weather, opponent pressure add variance the L8 rolling std
    # fails to capture). Raising these flattens the model's tails, which
    # is the dominant source of the Stage 4 ROI failure. MODEL-8 step 1.
    "passing_yards": 70.0,
    "rushing_yards": 30.0,
    "receiving_yards": 24.0,
    "receptions": 2.0,
}


def _exp_weights(n: int) -> np.ndarray:
    w = np.array([DECAY ** i for i in range(n)], dtype=float)
    return w / w.sum()


def _yardage_prob(
    threshold: float,
    values: np.ndarray,
    stat_type: str,
    opp_adj: float,
) -> float:
    """Normal-CDF model for a continuous yardage stat.

    `values` is ordered most-recent-first and has already been clipped to
    LAST_N_GAMES. Returns a probability in [0.01, 0.99].
    """
    n = len(values)
    weights = _exp_weights(n)

    wmean = float(np.dot(weights, values))
    wvar = float(np.dot(weights, (values - wmean) ** 2))
    wstd = max(float(np.sqrt(wvar)), MIN_STD_BY_STAT.get(stat_type, MIN_STD))

    # Continuity correction — mirror NBA (threshold - 0.5)
    model_prob = float(1.0 - norm.cdf(threshold - 0.5, wmean, wstd))
    model_prob = max(0.01, min(0.99, model_prob))

    # Empirical hit-rate blend — MODEL-8 step 3 caps disagreement.
    # When the empirical hit rate is far from the Gaussian prediction,
    # it's usually because 1-2 outlier games are dominating the small
    # sample. Clip the empirical signal to stay within ±EMPIRICAL_DISAGREE_CAP
    # of the Gaussian so it can't amplify the model into the miscalibrated tails.
    hits = (values >= threshold).astype(float)
    weighted_hit_rate = float(np.dot(weights, hits))
    clipped_empirical = float(
        np.clip(
            weighted_hit_rate,
            model_prob - EMPIRICAL_DISAGREE_CAP,
            model_prob + EMPIRICAL_DISAGREE_CAP,
        )
    )
    blended = MODEL_EMPIRICAL_BLEND * model_prob + (1 - MODEL_EMPIRICAL_BLEND) * clipped_empirical

    # Margin adjustment: average margin over the line tilts the probability
    avg_margin = float(np.mean(values - threshold))
    scale = MARGIN_SCALE / max(MIN_STD_BY_STAT.get(stat_type, MIN_STD) / 10.0, 0.1)
    margin_adj = float(np.clip(avg_margin * scale, -MARGIN_CAP, MARGIN_CAP))
    blended += margin_adj

    # MODEL-8 step 2: streak adjustment removed. 3-game streaks at NFL's
    # weekly cadence are mostly noise and they compound with the empirical
    # blend to push the model into the miscalibrated tail buckets.

    prob = blended * opp_adj
    return max(0.01, min(0.99, float(prob)))


def _poisson_prob(
    threshold: float,
    values: np.ndarray,
    opp_adj: float,
) -> float:
    """Poisson tail for count-based TD props.

    Uses weighted-mean expected value as λ. The standard NFL thresholds
    are 0.5 (anytime), 1.5 (passing 2+), 2.5 (passing 3+) so we evaluate
    P(X ≥ ceil(threshold)) = 1 - poisson.cdf(k-1, λ) where k = ceil.
    """
    n = len(values)
    weights = _exp_weights(n)
    lam = float(np.dot(weights, values)) * opp_adj
    lam = max(0.01, lam)

    # k = smallest integer strictly greater than threshold
    # For threshold=0.5 → k=1 → P(X≥1); for 1.5 → k=2; for 2.5 → k=3
    k = int(np.ceil(threshold))
    if k <= 0:
        k = 1

    prob = float(1.0 - poisson.cdf(k - 1, lam))
    return max(0.01, min(0.99, prob))


def compute_nfl_prop_prob(
    stat_type: str,
    threshold: float,
    values: np.ndarray | list[float],
    opp_adj: float = 1.0,
) -> Optional[tuple[float, int]]:
    """Compute P(stat ≥ threshold) for an NFL player prop.

    Args:
      stat_type: canonical stat key (see STAT_TO_BRANCH).
      threshold: the prop line (e.g. 249.5 for "250+ passing yards").
      values: per-game observed stat values, ordered most-recent-first.
              The function will use up to LAST_N_GAMES of these.
      opp_adj: multiplicative opponent adjustment factor. 1.0 = neutral.

    Returns:
      (probability, n_games_used) or None if sample too thin.
    """
    branch = STAT_TO_BRANCH.get(stat_type)
    if branch is None:
        return None

    arr = np.asarray(values, dtype=float)[:LAST_N_GAMES]
    n = len(arr)
    if n < MIN_GAMES:
        return None

    if branch in ("yardage", "count"):
        prob = _yardage_prob(threshold, arr, stat_type, opp_adj)
    elif branch == "poisson":
        prob = _poisson_prob(threshold, arr, opp_adj)
    else:
        return None

    return prob, n


def compute_opponent_adjustment(
    stat_type: str,
    opponent_allowed_pg: float,
    league_avg_allowed_pg: float,
) -> float:
    """Multiplicative adjustment factor based on opponent defense.

    A team allowing more than league average → higher player prob
    (factor > 1.0). A team allowing less → lower (factor < 1.0).
    Clipped to [1 - MAX_OPP_ADJ, 1 + MAX_OPP_ADJ].
    """
    if league_avg_allowed_pg <= 0:
        return 1.0
    ratio = opponent_allowed_pg / league_avg_allowed_pg
    return float(np.clip(ratio, 1.0 - MAX_OPP_ADJ, 1.0 + MAX_OPP_ADJ))
