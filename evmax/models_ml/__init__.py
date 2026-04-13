"""Statistical sub-models used by the model agents.

Active (called from agents/models/):
- spread_distribution.py  — Normal CDF win probability from spread + std dev
- total_distribution.py   — Over/under probability from projected total + std dev
- live_win_prob.py        — In-game live model: prior Elo + current score/time state
- point_projection.py     — Points projected per team (used by Poisson agent)

Legacy / unused:
- sharp_only.py           — Phase 1 placeholder; never called in the live pipeline
"""
