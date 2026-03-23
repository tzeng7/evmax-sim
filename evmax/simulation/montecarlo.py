"""Monte Carlo bankroll simulation.

Resamples from historical ev_predictions + ev_outcomes to simulate bankroll
trajectories under uncertainty.  Returns percentile paths and risk metrics.

Usage:
    sim = MonteCarloSimulator()
    result = sim.run(bets, trials=10_000, starting_bankroll=1000.0)
    print(result.summary())
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation run."""

    trials: int
    starting_bankroll: float
    median_final: float
    p5_final: float      # 5th percentile (bad run)
    p25_final: float
    p75_final: float
    p95_final: float     # 95th percentile (great run)
    mean_final: float
    ruin_probability: float   # fraction of trials ending below 20% of starting bankroll
    max_drawdown_median: float  # median max drawdown fraction across trials
    total_bets: int
    total_wagered: float  # median total wagered across all bets in the trial

    # Percentile bankroll paths (sampled at 10 time points for display)
    path_p5: list[float] = field(default_factory=list)
    path_p50: list[float] = field(default_factory=list)
    path_p95: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Monte Carlo ({self.trials:,} trials, {self.total_bets} bets, "
            f"start ${self.starting_bankroll:,.0f})",
            f"  Median final:  ${self.median_final:,.2f}  "
            f"({(self.median_final/self.starting_bankroll - 1)*100:+.1f}%)",
            f"  5th / 95th:    ${self.p5_final:,.2f}  /  ${self.p95_final:,.2f}",
            f"  Mean final:    ${self.mean_final:,.2f}",
            f"  Ruin prob:     {self.ruin_probability*100:.1f}%  "
            f"(bankroll < 20% of start)",
            f"  Max drawdown:  {self.max_drawdown_median*100:.1f}% (median)",
        ]
        return "\n".join(lines)


class MonteCarloSimulator:
    """Simulate bankroll trajectories by resampling from historical bets.

    Each trial:
      1. Shuffle the bet order (path-order uncertainty).
      2. For each bet, determine outcome by sampling from a Bernoulli(true_prob).
      3. Apply fractional Kelly sizing: stake = bankroll × kelly_fraction × kelly_full.
      4. Track bankroll after each bet.

    Returns percentile final bankrolls and risk metrics.
    """

    RUIN_THRESHOLD = 0.20  # < 20% of starting bankroll = ruin

    def run(
        self,
        bets: list[dict],
        trials: int = 10_000,
        starting_bankroll: float = 1_000.0,
        kelly_fraction: float = 0.25,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation.

        Args:
            bets:               List of bet dicts with keys:
                                  true_prob (float), payout_decimal (float or None),
                                  kalshi_yes_price (float), kelly_fraction (float)
            trials:             Number of simulation trials.
            starting_bankroll:  Starting bankroll in USD.
            kelly_fraction:     Kelly multiplier (overrides per-bet kelly_fraction if > 0).
            seed:               Optional RNG seed for reproducibility.
        """
        if not bets:
            return MonteCarloResult(
                trials=trials,
                starting_bankroll=starting_bankroll,
                median_final=starting_bankroll,
                p5_final=starting_bankroll,
                p25_final=starting_bankroll,
                p75_final=starting_bankroll,
                p95_final=starting_bankroll,
                mean_final=starting_bankroll,
                ruin_probability=0.0,
                max_drawdown_median=0.0,
                total_bets=0,
                total_wagered=0.0,
            )

        rng = random.Random(seed)

        final_bankrolls: list[float] = []
        max_drawdowns: list[float] = []
        total_wagered_list: list[float] = []

        # Sample paths at 10 checkpoints for sparkline display
        n_checkpoints = min(10, len(bets))
        checkpoint_indices = [int(i * len(bets) / n_checkpoints) for i in range(n_checkpoints)]

        path_snapshots: list[list[float]] = [[] for _ in range(n_checkpoints)]

        for _ in range(trials):
            trial_bets = bets[:]
            rng.shuffle(trial_bets)

            bankroll = starting_bankroll
            peak = bankroll
            max_dd = 0.0
            total_wagered = 0.0

            for i, bet in enumerate(trial_bets):
                if bankroll <= 0:
                    break

                true_prob = float(bet.get("true_prob", bet.get("blended_true_prob", 0.5)))
                yes_price = float(bet.get("kalshi_yes_price", 0.5))
                payout = 1.0 / yes_price if yes_price > 0 else 2.0

                # Use per-bet kelly_fraction if stored; otherwise use override
                bet_kelly = float(bet.get("kelly_fraction", kelly_fraction))
                stake = bankroll * bet_kelly
                stake = max(0.0, min(stake, bankroll))  # clamp

                total_wagered += stake

                # Sample outcome
                won = rng.random() < true_prob
                if won:
                    bankroll += stake * (payout - 1.0)
                else:
                    bankroll -= stake

                if bankroll > peak:
                    peak = bankroll
                dd = (peak - bankroll) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

                # Record checkpoint
                if i in checkpoint_indices:
                    idx = checkpoint_indices.index(i)
                    path_snapshots[idx].append(bankroll)

            final_bankrolls.append(bankroll)
            max_drawdowns.append(max_dd)
            total_wagered_list.append(total_wagered)

        # Sort for percentile calculation
        final_bankrolls.sort()
        max_drawdowns.sort()
        total_wagered_list.sort()

        def pct(lst: list[float], p: float) -> float:
            idx = int(len(lst) * p / 100)
            return lst[min(idx, len(lst) - 1)]

        ruin_count = sum(1 for b in final_bankrolls if b < starting_bankroll * self.RUIN_THRESHOLD)

        # Build percentile paths
        path_p5 = [pct(sorted(snap), 5) for snap in path_snapshots if snap]
        path_p50 = [pct(sorted(snap), 50) for snap in path_snapshots if snap]
        path_p95 = [pct(sorted(snap), 95) for snap in path_snapshots if snap]

        return MonteCarloResult(
            trials=trials,
            starting_bankroll=starting_bankroll,
            median_final=pct(final_bankrolls, 50),
            p5_final=pct(final_bankrolls, 5),
            p25_final=pct(final_bankrolls, 25),
            p75_final=pct(final_bankrolls, 75),
            p95_final=pct(final_bankrolls, 95),
            mean_final=sum(final_bankrolls) / len(final_bankrolls),
            ruin_probability=ruin_count / trials,
            max_drawdown_median=pct(max_drawdowns, 50),
            total_bets=len(bets),
            total_wagered=pct(total_wagered_list, 50),
            path_p5=path_p5,
            path_p50=path_p50,
            path_p95=path_p95,
        )
