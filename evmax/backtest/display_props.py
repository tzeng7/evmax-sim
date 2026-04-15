"""Rich display helpers for NFL prop backtest reports."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evmax.backtest.models import PropBacktestReport

console = Console()


def _brier_color(bs: float, baseline: float) -> str:
    if bs < baseline * 0.92:
        return "green"
    if bs < baseline:
        return "cyan"
    return "red"


def summary_panel(report: PropBacktestReport) -> Panel:
    lines: list[str] = []
    random_baseline = 0.25
    prior_baseline = report.baseline_mean_prior
    market_baseline = report.baseline_market

    brier_color = _brier_color(report.brier_score, prior_baseline or random_baseline)

    lines.append(f"  [bold]Markets scored:[/bold] {report.n_scored:,} / {report.n_markets:,}")
    lines.append(
        f"  [bold]Brier score:[/bold]  [{brier_color}]{report.brier_score:.4f}[/{brier_color}]"
    )
    lines.append(
        f"  [bold]  vs prior:[/bold]    {prior_baseline:.4f} "
        f"({'better' if report.brier_score < prior_baseline else 'worse'})"
    )
    if market_baseline > 0:
        diff = market_baseline - report.brier_score
        color = "green" if diff > 0 else "red"
        lines.append(
            f"  [bold]  vs market:[/bold]   {market_baseline:.4f} "
            f"([{color}]Δ {diff:+.4f}[/{color}])"
        )
    lines.append(f"  [bold]Log loss:[/bold]     {report.log_loss:.4f}")
    lines.append(f"  [bold]Accuracy:[/bold]     {report.accuracy:.1%}")
    if report.min_volume > 0:
        lines.append(f"  [bold]Volume gate:[/bold]  ≥ {report.min_volume:.0f}")

    return Panel(
        "\n".join(lines),
        title=f"[bold]NFL prop backtest[/bold]",
        box=box.ROUNDED,
        border_style="cyan",
    )


def per_stat_table(report: PropBacktestReport) -> Table:
    t = Table(
        title="Per-stat breakdown",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Stat", width=18)
    t.add_column("N", justify="right")
    t.add_column("Yes rate", justify="right")
    t.add_column("Brier", justify="right")
    t.add_column("Log loss", justify="right")
    t.add_column("Accuracy", justify="right")
    for stat, m in sorted(report.by_stat.items()):
        t.add_row(
            stat,
            f"{m['n']:,}",
            f"{m['yes_rate']:.1%}",
            f"{m['brier']:.4f}",
            f"{m['log_loss']:.4f}",
            f"{m['accuracy']:.1%}",
        )
    return t


def calibration_table(report: PropBacktestReport) -> Table:
    t = Table(
        title="Calibration (all stats pooled)",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Prob range", width=12)
    t.add_column("Mean pred", justify="right")
    t.add_column("Actual %", justify="right")
    t.add_column("N", justify="right")
    t.add_column("95% CI", width=18)
    for b in report.calibration_bins:
        if b.n == 0:
            continue
        delta = b.actual_rate - b.mean_pred
        color = "green" if abs(delta) < 0.03 else ("cyan" if delta > 0 else "red")
        t.add_row(
            f"{b.prob_low:.0%}–{b.prob_high:.0%}",
            f"{b.mean_pred:.1%}",
            f"[{color}]{b.actual_rate:.1%}[/{color}]",
            f"{b.n:,}",
            f"[{b.ci_low:.1%}, {b.ci_high:.1%}]",
        )
    return t


def roi_table(report: PropBacktestReport) -> Table:
    t = Table(
        title="ROI at EV thresholds (YES side only, flat $1 units)",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Scenario", width=26)
    t.add_column("Bets", justify="right")
    t.add_column("Win rate", justify="right")
    t.add_column("ROI", justify="right")
    t.add_column("PnL", justify="right")
    for label, m in report.roi_by_ev_threshold.items():
        color = "green" if m["roi"] > 0 else "red"
        t.add_row(
            label,
            f"{m['bets']:,}",
            f"{m['win_rate']:.1%}",
            f"[{color}]{m['roi']:+.2%}[/{color}]",
            f"{m['pnl']:+.1f}",
        )
    return t


def print_prop_report(report: PropBacktestReport) -> None:
    console.print()
    console.print(summary_panel(report))
    console.print()
    console.print(per_stat_table(report))
    console.print()
    console.print(calibration_table(report))
    console.print()
    console.print(roi_table(report))
    console.print()
