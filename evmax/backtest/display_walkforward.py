"""Rich display for walk-forward backtest reports."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evmax.backtest.sources.espn_walkforward import (
    WalkForwardReport,
    SpreadBacktestReport,
    TotalsBacktestReport,
    TOTALS_LINES,
)

console = Console()


def _brier_color(brier: float, baseline: float) -> str:
    if brier < baseline * 0.85:
        return "green"
    if brier < baseline:
        return "yellow"
    return "red"


def _delta_color(mean_pred: float, actual_rate: float, n: int) -> str:
    if n < 10:
        return "dim"
    delta = actual_rate - mean_pred
    if abs(delta) < 0.03:
        return "green"
    return "red" if delta < 0 else "cyan"


def summary_panel(report: WalkForwardReport) -> Panel:
    lines: list[str] = []

    lines.append(f"  [bold]Total Games:[/bold]         {report.n_games}")
    lines.append(f"  [bold]Games w/ Predictions:[/bold] {report.n_predicted}")
    lines.append(f"  [bold]Home Win Rate:[/bold]        {report.home_win_rate:.1%}")
    lines.append("")

    # Baseline
    bc = _brier_color(report.baseline_always_home_brier, 0.25)
    lines.append(f"  [bold]Baseline (always {report.home_win_rate:.1%}):[/bold]  [{bc}]Brier {report.baseline_always_home_brier:.4f}[/{bc}]")
    lines.append("")

    # Per-model table
    lines.append("  [bold underline]Model Performance[/bold underline]")
    models = [
        ("Elo", report.elo_brier, report.elo_accuracy, report.elo_n),
        ("Form", report.form_brier, report.form_accuracy, report.form_n),
        ("Poisson", report.poisson_brier, report.poisson_accuracy, report.poisson_n),
        ("Efficiency", report.efficiency_brier, report.efficiency_accuracy, report.efficiency_n),
        ("PossessionSim", report.possession_sim_brier, report.possession_sim_accuracy, report.possession_sim_n),
        ("ShotQuality", report.shot_quality_brier, report.shot_quality_accuracy, report.shot_quality_n),
        ("Matchup", report.matchup_brier, report.matchup_accuracy, report.matchup_n),
        ("Ensemble", report.ensemble_brier, report.ensemble_accuracy, report.ensemble_n),
    ]
    for name, brier, acc, n in models:
        if n == 0:
            lines.append(f"  {name:12s}  [dim]no predictions (insufficient data)[/dim]")
            continue
        bc = _brier_color(brier, report.baseline_always_home_brier)
        beat = "  [green]✓ beats baseline[/green]" if brier < report.baseline_always_home_brier else ""
        lines.append(
            f"  {name:12s}  Brier: [{bc}]{brier:.4f}[/{bc}]  |  Acc: {acc:.1%}  |  N={n}{beat}"
        )

    content = "\n".join(lines)
    title = (
        f"[bold]WALK-FORWARD BACKTEST — {report.sector.upper()}[/bold]  |  "
        f"{report.n_games} games  |  ESPN data"
    )
    return Panel(content, title=title, border_style="cyan")


def calibration_table(report: WalkForwardReport) -> Table:
    t = Table(
        title=f"Ensemble Calibration — {report.sector.upper()} (N={report.ensemble_n})",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Prob Range", width=12)
    t.add_column("Mean Pred", justify="right", width=10)
    t.add_column("Actual %", justify="right", width=10)
    t.add_column("N", justify="right", width=7)
    t.add_column("95% CI", width=16)
    t.add_column("Delta", justify="right", width=8)

    for b in report.calibration_bins:
        if b.n == 0:
            continue
        delta = b.actual_rate - b.mean_pred
        color = _delta_color(b.mean_pred, b.actual_rate, b.n)
        t.add_row(
            f"{b.prob_low:.0%}–{b.prob_high:.0%}",
            f"{b.mean_pred:.1%}",
            f"{b.actual_rate:.1%}",
            str(b.n),
            f"[{b.ci_low:.1%}, {b.ci_high:.1%}]",
            f"[{color}]{delta:+.1%}[/{color}]",
        )
    return t


def model_convergence_table(report: WalkForwardReport) -> Table:
    """Show how quickly each model starts producing predictions."""
    t = Table(
        title="Model Warm-Up (games until first prediction)",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Model", width=12)
    t.add_column("First Pred At Game #", justify="right", width=22)
    t.add_column("Total Predictions", justify="right", width=18)
    t.add_column("Coverage", justify="right", width=10)

    models = [
        ("Elo", "elo_prob_home"),
        ("Form", "form_prob_home"),
        ("Poisson", "poisson_prob_home"),
        ("Efficiency", "efficiency_prob_home"),
        ("PossessionSim", "possession_sim_prob_home"),
        ("ShotQuality", "shot_quality_prob_home"),
        ("Matchup", "matchup_prob_home"),
        ("Ensemble", "ensemble_prob_home"),
    ]

    for name, attr in models:
        first = None
        count = 0
        for i, r in enumerate(report.results):
            if getattr(r, attr) is not None:
                if first is None:
                    first = i + 1
                count += 1
        n = len(report.results)
        coverage = f"{count/n:.0%}" if n else "0%"
        first_str = str(first) if first else "—"
        t.add_row(name, first_str, str(count), coverage)

    return t


def spread_summary_panel(report: SpreadBacktestReport) -> Panel:
    lines: list[str] = []
    lines.append(f"  [bold]Games:[/bold]       {report.n_games}")
    lines.append(f"  [bold]Predictions:[/bold] {report.n_predictions}  ({len(SPREAD_LINES)} lines × games with sim data)")
    lines.append("")

    sim_color = "green" if report.sim_brier < report.cdf_brier else "red"
    cdf_color = "green" if report.cdf_brier < report.sim_brier else "red"
    lines.append("  [bold underline]Overall Spread Prediction Accuracy[/bold underline]")
    lines.append(f"  {'PossessionSim':16s}  Brier: [{sim_color}]{report.sim_brier:.4f}[/{sim_color}]  |  Acc: {report.sim_accuracy:.1%}")
    lines.append(f"  {'Normal CDF':16s}  Brier: [{cdf_color}]{report.cdf_brier:.4f}[/{cdf_color}]  |  Acc: {report.cdf_accuracy:.1%}")
    delta = report.cdf_brier - report.sim_brier
    if delta > 0:
        lines.append(f"  [green]  Sim beats CDF by {delta:.4f} Brier ({delta/report.cdf_brier:.1%} improvement)[/green]")
    else:
        lines.append(f"  [red]  CDF beats Sim by {-delta:.4f} Brier[/red]")

    content = "\n".join(lines)
    return Panel(content, title="[bold]SPREAD BACKTEST — NBA[/bold]  |  PossessionSim vs Normal CDF", border_style="cyan")


SPREAD_LINES = [-1.5, -3.5, -5.5, -7.5, -9.5, -11.5, -13.5]


def spread_per_line_table(report: SpreadBacktestReport) -> Table:
    t = Table(
        title="Per-Line Breakdown",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Line", width=8)
    t.add_column("N", justify="right", width=6)
    t.add_column("Cover %", justify="right", width=9)
    t.add_column("Sim Prob", justify="right", width=10)
    t.add_column("CDF Prob", justify="right", width=10)
    t.add_column("Sim Brier", justify="right", width=10)
    t.add_column("CDF Brier", justify="right", width=10)
    t.add_column("Winner", width=10)

    for line in SPREAD_LINES:
        data = report.per_line.get(line)
        if not data:
            continue
        winner = "Sim" if data["sim_brier"] < data["cdf_brier"] else "CDF"
        w_color = "green" if winner == "Sim" else "yellow"
        t.add_row(
            f"{line:+.1f}",
            str(data["n"]),
            f"{data['actual_cover_rate']:.1%}",
            f"{data['sim_mean_prob']:.1%}",
            f"{data['cdf_mean_prob']:.1%}",
            f"{data['sim_brier']:.4f}",
            f"{data['cdf_brier']:.4f}",
            f"[{w_color}]{winner}[/{w_color}]",
        )
    return t


def spread_calibration_table(report: SpreadBacktestReport) -> Table:
    t = Table(
        title="PossessionSim Spread Calibration",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Prob Range", width=12)
    t.add_column("Mean Pred", justify="right", width=10)
    t.add_column("Actual %", justify="right", width=10)
    t.add_column("N", justify="right", width=7)
    t.add_column("Delta", justify="right", width=8)

    for b in report.calibration_bins:
        if b.n == 0:
            continue
        delta = b.actual_rate - b.mean_pred
        color = _delta_color(b.mean_pred, b.actual_rate, b.n)
        t.add_row(
            f"{b.prob_low:.0%}–{b.prob_high:.0%}",
            f"{b.mean_pred:.1%}",
            f"{b.actual_rate:.1%}",
            str(b.n),
            f"[{color}]{delta:+.1%}[/{color}]",
        )
    return t


def print_spread_backtest(report: SpreadBacktestReport) -> None:
    console.print()
    console.print(spread_summary_panel(report))
    console.print()
    console.print(spread_per_line_table(report))
    console.print()
    if report.calibration_bins:
        console.print(spread_calibration_table(report))
        console.print()


def totals_summary_panel(report: TotalsBacktestReport) -> Panel:
    lines: list[str] = []
    lines.append(f"  [bold]Games:[/bold]       {report.n_games}")
    lines.append(
        f"  [bold]Predictions:[/bold] {report.n_predictions}  "
        f"({len(TOTALS_LINES)} lines × games with sim data)"
    )
    lines.append("")

    sim_color = "green" if report.sim_brier < report.cdf_brier else "red"
    cdf_color = "green" if report.cdf_brier < report.sim_brier else "red"
    lines.append("  [bold underline]Overall Totals Prediction Accuracy[/bold underline]")
    lines.append(
        f"  {'PossessionSim':16s}  Brier: [{sim_color}]{report.sim_brier:.4f}[/{sim_color}]"
        f"  |  Acc: {report.sim_accuracy:.1%}"
    )
    lines.append(
        f"  {'Normal CDF (σ20)':16s}  Brier: [{cdf_color}]{report.cdf_brier:.4f}[/{cdf_color}]"
        f"  |  Acc: {report.cdf_accuracy:.1%}"
    )
    delta = report.cdf_brier - report.sim_brier
    if delta > 0:
        lines.append(
            f"  [green]  Sim beats CDF by {delta:.4f} Brier ({delta/report.cdf_brier:.1%} improvement)[/green]"
        )
    else:
        lines.append(f"  [red]  CDF beats Sim by {-delta:.4f} Brier[/red]")

    content = "\n".join(lines)
    return Panel(
        content,
        title="[bold]TOTALS BACKTEST — NBA[/bold]  |  PossessionSim vs Normal CDF",
        border_style="cyan",
    )


def totals_per_line_table(report: TotalsBacktestReport) -> Table:
    t = Table(title="Per-Line Breakdown", box=box.SIMPLE, header_style="bold cyan")
    t.add_column("Line", width=8)
    t.add_column("N", justify="right", width=6)
    t.add_column("Over %", justify="right", width=9)
    t.add_column("Sim Prob", justify="right", width=10)
    t.add_column("CDF Prob", justify="right", width=10)
    t.add_column("Sim Brier", justify="right", width=10)
    t.add_column("CDF Brier", justify="right", width=10)
    t.add_column("Winner", width=10)

    for line in TOTALS_LINES:
        data = report.per_line.get(line)
        if not data:
            continue
        winner = "Sim" if data["sim_brier"] < data["cdf_brier"] else "CDF"
        w_color = "green" if winner == "Sim" else "yellow"
        t.add_row(
            f"{line:.1f}",
            str(data["n"]),
            f"{data['actual_over_rate']:.1%}",
            f"{data['sim_mean_prob']:.1%}",
            f"{data['cdf_mean_prob']:.1%}",
            f"{data['sim_brier']:.4f}",
            f"{data['cdf_brier']:.4f}",
            f"[{w_color}]{winner}[/{w_color}]",
        )
    return t


def totals_calibration_table(report: TotalsBacktestReport) -> Table:
    t = Table(
        title="PossessionSim Totals Calibration",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Prob Range", width=12)
    t.add_column("Mean Pred", justify="right", width=10)
    t.add_column("Actual %", justify="right", width=10)
    t.add_column("N", justify="right", width=7)
    t.add_column("Delta", justify="right", width=8)

    for b in report.calibration_bins:
        if b.n == 0:
            continue
        delta = b.actual_rate - b.mean_pred
        color = _delta_color(b.mean_pred, b.actual_rate, b.n)
        t.add_row(
            f"{b.prob_low:.0%}–{b.prob_high:.0%}",
            f"{b.mean_pred:.1%}",
            f"{b.actual_rate:.1%}",
            str(b.n),
            f"[{color}]{delta:+.1%}[/{color}]",
        )
    return t


def print_totals_backtest(report: TotalsBacktestReport) -> None:
    console.print()
    console.print(totals_summary_panel(report))
    console.print()
    console.print(totals_per_line_table(report))
    console.print()
    if report.calibration_bins:
        console.print(totals_calibration_table(report))
        console.print()


def print_walkforward_report(report: WalkForwardReport) -> None:
    console.print()
    console.print(summary_panel(report))
    console.print()
    if report.calibration_bins:
        console.print(calibration_table(report))
        console.print()
    console.print(model_convergence_table(report))
    console.print()
