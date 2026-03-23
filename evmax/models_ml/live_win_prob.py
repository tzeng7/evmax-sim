"""Live win probability model using game state + pre-game Elo prior.

Uses a logistic adjustment formula standard in sports analytics:

    live_prob = sigmoid(logit(prior) + k * score_diff_adj)

where:
    score_diff_adj = score_diff / sqrt(time_remaining_pct)
    k = sector-specific calibration constant

k converts a score-difference unit into log-odds space.  Values are calibrated
from historical data; e.g. for soccer a 1-goal lead at halftime (50% remaining)
yields adj = 1 / sqrt(0.5) = 1.41, shifting log-odds by 0.15 * 1.41 ≈ +0.21.
"""

from __future__ import annotations

import math


class LiveWinProbModel:
    """Estimate team_a win probability from live game state.

    Args:
        sector:           Sport sector key (e.g. "nba", "soccer").
        prior_prob:       Pre-game Elo win probability for team_a (0–1).
        score_diff:       team_a_score - team_b_score (positive = team_a winning).
        time_elapsed_pct: Fraction of game elapsed [0, 1].
    """

    # k values: effect of one score-diff unit on log-odds per sqrt(time_remaining)
    SECTOR_K: dict[str, float] = {
        "soccer": 0.15,   # one goal is decisive but not overwhelming
        "nba": 0.08,      # points are cheap; large deficits overcome often
        "nfl": 0.10,      # TD = 6-7 pts, meaningful but comebacks happen
        "ncaab": 0.07,    # similar to NBA
        "lol": 0.20,      # map wins are binary, strong signal
        "cs2": 0.20,      # same as LoL
    }

    def predict(
        self,
        sector: str,
        prior_prob: float,
        score_diff: int,
        time_elapsed_pct: float,
    ) -> float:
        """Return updated win probability for team_a after observing game state.

        Returns a probability clamped to [0.01, 0.99].
        Falls back to prior_prob if inputs are degenerate.
        """
        if not (0 < prior_prob < 1):
            return max(0.01, min(0.99, prior_prob))

        k = self.SECTOR_K.get(sector, 0.12)
        time_remaining_pct = max(1.0 - time_elapsed_pct, 0.001)
        score_diff_adj = score_diff / (time_remaining_pct ** 0.5)

        logit_prior = math.log(prior_prob / (1.0 - prior_prob))
        live_logit = logit_prior + k * score_diff_adj
        live_prob = 1.0 / (1.0 + math.exp(-live_logit))
        return max(0.01, min(0.99, live_prob))


def parse_time_elapsed(sector: str, period: int, clock_secs: float) -> float:
    """Convert ESPN period + remaining clock to fraction of game elapsed.

    ESPN clock semantics:
      - Soccer:        status.clock = seconds ELAPSED in current half (counts up)
      - NBA/NFL/NCAAB: status.clock = seconds REMAINING in current period (counts down)
      - LoL/CS2:       no clock; period = current map number (1-indexed)

    Returns elapsed fraction [0, 1].
    """
    if sector == "soccer":
        # First half = 0–45 min, second half = 45–90 min
        # clock_secs is elapsed seconds in current half
        elapsed = clock_secs
        if period == 1:
            return min(elapsed / (90 * 60), 0.5)
        elif period == 2:
            return min((45 * 60 + elapsed) / (90 * 60), 1.0)
        else:
            return 1.0  # extra time — treat as nearly over

    elif sector == "nba":
        period_len = 12 * 60
        total = 4 * period_len
        elapsed_in_period = max(0.0, period_len - clock_secs)
        elapsed_total = (period - 1) * period_len + elapsed_in_period
        return min(elapsed_total / total, 1.0)

    elif sector == "nfl":
        period_len = 15 * 60
        total = 4 * period_len
        elapsed_in_period = max(0.0, period_len - clock_secs)
        elapsed_total = (period - 1) * period_len + elapsed_in_period
        return min(elapsed_total / total, 1.0)

    elif sector == "ncaab":
        half_len = 20 * 60
        total = 2 * half_len
        elapsed_in_period = max(0.0, half_len - clock_secs)
        elapsed_total = (period - 1) * half_len + elapsed_in_period
        return min(elapsed_total / total, 1.0)

    elif sector in ("lol", "cs2"):
        # Map number as proxy: map 1 ≈ 33%, map 2 ≈ 66%, map 3 ≈ 85%
        return min(period * 0.33, 1.0)

    else:
        # Generic fallback: assume 3-period sport, use (period-0.5)/3
        return min((period - 0.5) / 3.0, 1.0)


def _parse_display_clock(clock_str: str) -> float:
    """Parse ESPN displayClock string to seconds.

    Handles formats:
      "8:23"  → 503.0 seconds remaining (NBA/NFL)
      "67:00" → 4020.0 seconds elapsed  (soccer, but parsed same way)
      "67'"   → 4020.0 (soccer elapsed minutes with apostrophe)
      "0.0"   → 0.0
    """
    if not clock_str:
        return 0.0
    clock_str = clock_str.strip().rstrip("'")
    try:
        if ":" in clock_str:
            parts = clock_str.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(clock_str) * 60  # bare minutes
    except (ValueError, IndexError):
        return 0.0
