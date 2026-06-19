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

### `point_projection.py`
Projects points-per-game for each team using attack/defense ratings from `poisson_state.json`.
Used as input to `total_distribution.py` for over/under markets, and by the standalone
`evmax project` projection workflow.

### `base.py`
`ModelBase` ABC + `ModelPrediction` dataclass — the prediction interface that
`agents/models/base.py::ModelAgent` mixes in.
