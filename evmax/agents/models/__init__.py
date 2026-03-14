"""Statistical model agents — predict true probabilities from historical data."""

from evmax.agents.models.base import ModelAgent, ModelAgentPrediction
from evmax.agents.models.elo_agent import EloModelAgent
from evmax.agents.models.form_agent import FormModelAgent
from evmax.agents.models.poisson_agent import PoissonModelAgent
from evmax.agents.models.ensemble_agent import EnsembleModelAgent

__all__ = [
    "ModelAgent",
    "ModelAgentPrediction",
    "EloModelAgent",
    "FormModelAgent",
    "PoissonModelAgent",
    "EnsembleModelAgent",
]
