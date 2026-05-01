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
from evmax.backtest.sources.soccer_walkforward import SoccerWalkForwardReport

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
        ("Pitcher", report.pitcher_brier, report.pitcher_accuracy, report.pitcher_n),
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


def soccer_walkforward_summary_panel(report: SoccerWalkForwardReport) -> Panel:
    random_3way = 2.0 / 3.0  # Brier 3-way random baseline
    random_bin = 0.25
    home_bin = (1 - report.home_win_rate) ** 2 * report.home_win_rate + (
        report.home_win_rate ** 2
    ) * (1 - report.home_win_rate)

    lines: list[str] = []
    lines.append(f"  [bold]Games:[/bold]           {report.n_games}")
    lines.append(f"  [bold]Home win rate:[/bold]   {report.home_win_rate:.1%}")
    lines.append(f"  [bold]Draw rate:[/bold]       {report.draw_rate:.1%}")
    lines.append("")
    lines.append(
        f"  [bold]Baselines:[/bold]  always-home Brier (bin) = {home_bin:.4f}  |  "
        f"random 3-way Brier = {random_3way:.4f}"
    )
    lines.append("")
    lines.append("  [bold underline]Per-model Brier (walk-forward, unfiltered)[/bold underline]")
    lines.append(
        f"  {'Model':14s}  {'N':>6s}  {'Brier(H)':>9s}  {'Brier(3w)':>10s}  "
        f"{'Acc':>6s}  {'μ pred H':>9s}  {'μ actual H':>11s}"
    )
    order = ["sharp", "blended_ensemble", "blended_baseline", "stat_ensemble", "elo", "form", "poisson", "xg"]
    for name in order:
        m = report.by_model.get(name)
        if not m or m.n == 0:
            lines.append(f"  {name:14s}  [dim]no predictions[/dim]")
            continue
        bc = _brier_color(m.brier_binary, random_bin)
        tc = _brier_color(m.brier_3way, random_3way)
        lines.append(
            f"  {name:14s}  {m.n:>6d}  "
            f"[{bc}]{m.brier_binary:>9.4f}[/{bc}]  "
            f"[{tc}]{m.brier_3way:>10.4f}[/{tc}]  "
            f"{m.accuracy:>6.1%}  "
            f"{m.mean_pred_home:>9.1%}  {m.mean_actual_home:>11.1%}"
        )

    return Panel(
        "\n".join(lines),
        title="[bold]SOCCER WALK-FORWARD — per-model Brier[/bold]  |  football-data.co.uk",
        border_style="cyan",
    )


def soccer_walkforward_league_table(report: SoccerWalkForwardReport) -> Table:
    t = Table(
        title="Per-League Brier (binary, P(home)) — blended_ensemble is live blend w/ disagreement bump",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("League", width=14)
    t.add_column("N", justify="right", width=5)
    columns = ("sharp", "blended_ensemble", "blended_baseline", "elo", "form", "poisson")
    headers = {
        "sharp": "Sharp",
        "blended_ensemble": "Blend+Bump",
        "blended_baseline": "Blend Base",
        "elo": "Elo",
        "form": "Form",
        "poisson": "Poisson",
    }
    for model in columns:
        t.add_column(headers[model], justify="right", width=11)

    for league in sorted(report.by_league.keys()):
        data = report.by_league[league]
        n = max((data[m]["n"] for m in data), default=0)
        row = [league, str(n)]
        sharp_brier = data.get("sharp", {}).get("brier_binary")
        for model in columns:
            mdata = data.get(model)
            if not mdata or mdata["n"] == 0:
                row.append("—")
                continue
            val = mdata["brier_binary"]
            # Green if blended beats sharp (unlikely but possible in some leagues);
            # yellow if within 0.005; red if significantly worse.
            if model == "blended_ensemble" and sharp_brier is not None:
                delta = val - sharp_brier
                if delta <= 0:
                    color = "green"
                elif delta < 0.005:
                    color = "yellow"
                else:
                    color = "red"
                row.append(f"[{color}]{val:.4f}[/{color}]")
            else:
                row.append(f"{val:.4f}")
        t.add_row(*row)
    return t


def soccer_walkforward_draw_table(report: SoccerWalkForwardReport) -> Table:
    t = Table(
        title="Draw allocation — predicted vs actual 25.2% baseline",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Model", width=14)
    t.add_column("μ pred D", justify="right", width=10)
    t.add_column("μ actual D", justify="right", width=11)
    t.add_column("Delta", justify="right", width=8)
    for name in ("sharp", "stat_ensemble", "elo", "form", "poisson"):
        m = report.by_model.get(name)
        if not m or m.n == 0:
            continue
        delta = m.mean_actual_draw - m.mean_pred_draw
        color = "green" if abs(delta) < 0.02 else ("red" if delta < 0 else "cyan")
        t.add_row(
            name,
            f"{m.mean_pred_draw:.1%}",
            f"{m.mean_actual_draw:.1%}",
            f"[{color}]{delta:+.1%}[/{color}]",
        )
    return t


def soccer_walkforward_calibration_table(
    report: SoccerWalkForwardReport, model_name: str
) -> Table:
    t = Table(
        title=f"{model_name} P(home) calibration — predicted vs actual by bucket",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    t.add_column("Bucket", width=12)
    t.add_column("N", justify="right", width=6)
    t.add_column("Mean Pred", justify="right", width=11)
    t.add_column("Actual %", justify="right", width=10)
    t.add_column("Delta", justify="right", width=8)
    m = report.by_model.get(model_name)
    if not m:
        return t
    for b in m.calibration:
        delta = b.actual_rate - b.mean_pred
        if b.n < 10:
            color = "dim"
        elif abs(delta) < 0.03:
            color = "green"
        else:
            color = "red" if delta < 0 else "cyan"
        t.add_row(
            f"{b.prob_low:.0%}–{b.prob_high:.0%}",
            str(b.n),
            f"{b.mean_pred:.1%}",
            f"{b.actual_rate:.1%}",
            f"[{color}]{delta:+.1%}[/{color}]",
        )
    return t


def soccer_walkforward_disagree_panel(report: SoccerWalkForwardReport) -> Panel:
    n = report.n_disagree
    if n == 0:
        return Panel("no disagreements recorded", border_style="dim")
    stat = report.stat_correct_on_disagree
    sharp = report.sharp_correct_on_disagree
    lines = [
        f"  [bold]Games where stat-ensemble disagrees with sharp:[/bold] {n} ({n / report.n_games:.1%} of corpus)",
        f"  [bold]Stat correct:[/bold]  {stat} ({stat/n:.1%})",
        f"  [bold]Sharp correct:[/bold] {sharp} ({sharp/n:.1%})",
    ]
    if sharp > stat:
        lines.append(
            f"  [red]  → Sharp wins by {sharp - stat} games when they disagree. "
            f"Stat ensemble is a net drag on these.[/red]"
        )
    elif stat > sharp:
        lines.append(
            f"  [green]  → Stat wins by {stat - sharp} games when they disagree — "
            f"blending adds real signal on these.[/green]"
        )
    else:
        lines.append("  [yellow]  → Break-even; blending is noise-neutral.[/yellow]")
    return Panel("\n".join(lines), title="[bold]Sharp vs Stat-Ensemble Disagreement[/bold]", border_style="cyan")


def print_soccer_walkforward_report(report: SoccerWalkForwardReport) -> None:
    console.print()
    console.print(soccer_walkforward_summary_panel(report))
    console.print()
    if report.by_league:
        console.print(soccer_walkforward_league_table(report))
        console.print()
    console.print(soccer_walkforward_draw_table(report))
    console.print()
    for model in ("elo", "poisson", "form", "sharp"):
        console.print(soccer_walkforward_calibration_table(report, model))
        console.print()
    console.print(soccer_walkforward_disagree_panel(report))
    console.print()
