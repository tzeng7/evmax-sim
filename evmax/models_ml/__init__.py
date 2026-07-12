"""Statistical sub-models used by the model agents.

Active (called from agents/models/ and cli/commands/project.py):
- spread_distribution.py  — Normal CDF win probability from spread + std dev
- total_distribution.py   — Over/under probability from projected total + std dev
- point_projection.py     — Points projected per team (standalone projection workflow)
- glicko2.py              — Glicko-2 rating math (pure functions, paper-validated);
                            used by agents/models/ufc_rating_agent.py
"""
