"""Statistical model agents — predict true probabilities from historical data.

All models share the ModelAgent interface (predict / update / save_state / load_state).
EnsembleModelAgent blends their outputs weighted by confidence; models below the 0.45
gate are excluded from the blend entirely.

Model weights and state files:
  EloModelAgent              weight=0.35  data/models/elo_state.json
  FormModelAgent             weight=0.25  data/models/form_state.json
  PoissonModelAgent          weight=0.30  data/models/poisson_state.json
  TennisModelAgent           weight=0.35  data/models/tennis_surface_state.json        (tennis only)
  TennisServeReturnAgent     weight=0.15  data/models/tennis_serve_return_state.json   (tennis only)
  TennisFormAgent            weight=0.15  data/models/tennis_form_state.json           (tennis only)
  TennisH2HAgent             weight=0.10  data/models/tennis_h2h_state.json            (tennis only)
  TennisRankingTrendAgent    weight=0.10  data/models/tennis_ranking_trend_state.json  (tennis only)
  PitcherModelAgent          weight=0.50  data/models/pitcher_v2_state.json            (baseball only)
  UFCRatingAgent             weight=1.0   data/models/ufc_rating_state.json            (ufc only)

Sharp weight (Pinnacle) is separate from the model blend — controlled by sharp_weight
in EnsembleModelAgent, auto-tuned weekly via Brier score in agents/cleanup/metrics.py.
"""

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.agents.models.ensemble_agent import EnsembleModelAgent
from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent
from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent
from evmax.agents.models.tennis_form_agent import TennisFormAgent
from evmax.agents.models.pitcher_agent import PitcherModelAgent
from evmax.agents.models.ufc_rating_agent import UFCRatingAgent

__all__ = [
    "ModelAgent",
    "ModelAgentPrediction",
    "EloModelAgent",
    "FormModelAgent",
    "PoissonModelAgent",
    "EnsembleModelAgent",
    "TennisModelAgent",
    "TennisServeReturnAgent",
    "TennisH2HAgent",
    "TennisRankingTrendAgent",
    "TennisFormAgent",
    "PitcherModelAgent",
    "UFCRatingAgent",
]
