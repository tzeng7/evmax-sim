"""Kalshi ↔ Pinnacle market matching.

- engine.py — MatchingEngine: canonical key match → rapidfuzz fallback (threshold=88).
              Canonical keys are "{sector}::{date}::{team_a}_vs_{team_b}".
              Underscores replaced with spaces before fuzzy scoring.
              UCL home/away reversal handled as a known exception.
"""
