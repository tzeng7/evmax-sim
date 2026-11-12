# evmax/models_ml

Statistical sub-models that support the model agents in `agents/models/`.

These modules contain pure math/statistics — no agent lifecycle, no state files, no async.
They are called from within agent `predict()` methods.

## Active modules

### `spread_distribution.py`
Converts a point spread + standard deviation into a win probability using the Normal CDF.
Used by `EVGapAgent` (`agents/odds/ev_gap_agent.py`) to price spread rungs off Pinnacle's main line,
and by the matcher / Pinnacle client / contamination + listings-eval lenses for the ladder and gate helpers.
Sectors in `_POSTED_RUNG_ONLY_SECTORS` (NHL) price only Pinnacle's own posted rung, so the normal
never extrapolates there; those rows carry the `puck_line` token.

### `spread_pmf.py`
Key-number-aware discrete margin PMF for NFL alt-spread pricing (`MarginPMF`). Loaded from
`data/models/nfl_margin_pmf.json` (refit with `scripts/fit_nfl_margin_pmf.py`) and anchored on
Pinnacle's devigged main line. `spread_distribution.py` delegates to it for sectors in
`_PMF_SECTORS` and falls back to the normal CDF when the artifact is missing or invalid.

### `total_distribution.py`
Converts projected total points + standard deviation into over/under probability.
Uses Normal CDF. Used by `EVGapAgent` (`agents/odds/ev_gap_agent.py`) to price alternate total lines.

### `point_projection.py`
Projects points-per-game for each team using attack/defense ratings from `poisson_state.json`.
Used as input to `total_distribution.py` for over/under markets, and by the standalone
`evmax project` projection workflow.

### `glicko2.py`
Glicko-2 rating system (Glickman 2013) — pure functions + a small dataclass, validated
against the paper's worked example in `tests/test_glicko2.py`. Sparse-schedule adaptation:
each bout is its own rating period and `inflate_rd` grows a fighter's RD for inactivity.
Used by `agents/models/ufc_rating_agent.py` (UFC fight-winner probabilities).

### `base.py`
`ModelBase` ABC + `ModelPrediction` dataclass — the prediction interface that
`agents/models/base.py::ModelAgent` mixes in.

### Other modules

- `_math.py` — shared numeric primitives for model agents (dependency leaf, no import cycle).
- `baseball_props.py` — MLB player-prop projection model; wraps per-stat expectations in the
  `ev/prop_pricing.py` distribution family so model and Pinnacle-anchor probabilities compare directly.
- `bullpen.py` — bullpen quality + availability (fatigue) for `pitcher_v2` and its walk-forward backtest.
- `spread_ladder_eval.py` — pure join/scoring helpers for `scripts/backtest_spread_ladder_replay.py`
  (CDF extrapolation vs Pinnacle's own alt-spread rung).
