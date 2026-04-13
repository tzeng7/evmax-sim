# evmax/models_ml

Statistical sub-models that support the model agents in `agents/models/`.

These modules contain pure math/statistics — no agent lifecycle, no state files, no async.
They are called from within agent `predict()` methods.

## Active modules

### `spread_distribution.py`
Converts a point spread + standard deviation into a win probability using the Normal CDF.
Used by `EloModelAgent` and `PoissonModelAgent` for spread market cover probabilities.

### `total_distribution.py`
Converts projected total points + standard deviation into over/under probability.
Uses Normal CDF. Used by `PoissonModelAgent` for totals markets.

### `live_win_prob.py`
In-game win probability model. Takes prior Elo rating + current score differential + time
remaining → outputs real-time win probability. Used by `pipeline/live_scanner.py`.

### `point_projection.py`
Projects points-per-game for each team using attack/defense ratings from `poisson_state.json`.
Used as input to `total_distribution.py` for over/under markets.

## Legacy (not used in the live pipeline)

### `sharp_only.py`
Phase 1 placeholder model that returned the Pinnacle sharp line as the "prediction".
Superseded by the full `EnsembleModelAgent` blend. Only imported by `pipeline/runner.py`
(also legacy). Safe to delete once runner.py is removed.
