"""Rich display helpers for backtest reports."""

from __future__ import annotations

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from evmax.backtest.models import BacktestReport

console = Console()

KALSHI_CAVEAT = (
    "[yellow]⚠  CAVEAT:[/yellow] Kalshi EV figures use [bold]last_price_dollars[/bold] from resolved markets,\n"
    "   which converge toward 1.0/0.0 near settlement — [bold]NOT opening prices[/bold].\n"
    "   These figures likely [bold]underestimate[/bold] actual opening-price edge.\n"
    "   Use for directional signal only, not precise P&L."
)


def _delta_color(mean_pred: float, actual_rate: float, n: int) -> str:
    if n < 10:
        return "dim"
    delta = actual_rate - mean_pred
    if abs(delta) < 0.03:
        return "green"
    return "red" if delta < 0 else "cyan"


def calibration_table(report: BacktestReport, title: str = "") -> Table:
    t = Table(
        title=title or f"Calibration — {report.sector.upper()} (N={report.n_games:,})",
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


def summary_panel(report: BacktestReport) -> Panel:
    lines: list[str] = []

    # Header metrics
    brier_color = "green" if report.brier_score < report.random_brier_baseline * 0.85 else "yellow"
    lines.append(
        f"  [bold]Brier Score:[/bold]  [{brier_color}]{report.brier_score:.4f}[/{brier_color}]"
        f"  (random baseline: {report.random_brier_baseline:.3f})"
    )
    lines.append(f"  [bold]Log Loss:[/bold]    {report.log_loss:.4f}")
    lines.append(
        f"  [bold]Accuracy:[/bold]    {report.accuracy:.1%}"
        f"  (always-favourite baseline: {report.baseline_accuracy:.1%})"
    )
    lines.append(f"  [bold]Pinnacle Avg Margin:[/bold] {report.mean_pinnacle_margin:.2%}")
    lines.append(f"  [bold]Games:[/bold]       {report.n_games:,}")

    if report.n_kalshi_matched > 0:
        lines.append("")
        lines.append(KALSHI_CAVEAT)
        lines.append("")
        pct_match = report.n_kalshi_matched / report.n_games
        lines.append(f"  Kalshi matched: {report.n_kalshi_matched:,} / {report.n_games:,} ({pct_match:.1%})")
        lines.append(
            f"  Games with theoretical EV ≥ {report.ev_threshold:.0%}: "
            f"[bold green]{report.n_above_threshold:,}[/bold green]"
            f"  ({report.n_above_threshold/report.n_kalshi_matched:.1%} of matched)"
        )
        if report.mean_ev_positive > 0:
            lines.append(f"  Mean theoretical EV (positive only): [bold cyan]{report.mean_ev_positive:.1%}[/bold cyan]")

    content = "\n".join(lines)
    title = (
        f"[bold]BACKTEST — {report.sector.upper()}[/bold]  |  "
        f"Seasons: {', '.join(report.seasons)}  |  "
        f"Leagues: {len(report.leagues)}"
    )
    return Panel(content, title=title, border_style="cyan")


def ev_distribution_table(report: BacktestReport) -> Table:
    t = Table(
        title="Theoretical EV Distribution (Kalshi last_price — see caveat above)",
        box=box.SIMPLE,
        header_style="bold yellow",
    )
    t.add_column("EV Bucket", width=12)
    t.add_column("Count", justify="right", width=8)
    t.add_column("% of Matched", justify="right", width=14)
    t.add_column("Bar", width=30)

    total = report.n_kalshi_matched or 1
    bucket_order = ["< 0%", "0–2%", "2–5%", "5–10%", "> 10%"]
    bucket_colors = {"< 0%": "red", "0–2%": "dim", "2–5%": "yellow", "5–10%": "green", "> 10%": "bold green"}

    for label in bucket_order:
        count = report.ev_buckets.get(label, 0)
        pct = count / total
        bar_len = int(pct * 30)
        color = bucket_colors.get(label, "white")
        t.add_row(
            f"[{color}]{label}[/{color}]",
            str(count),
            f"{pct:.1%}",
            f"[{color}]{'█' * bar_len}[/{color}]",
        )
    return t


def league_table(report: BacktestReport) -> Table:
    t = Table(title="By League / Tournament", box=box.SIMPLE, header_style="bold cyan")
    t.add_column("League", min_width=20)
    t.add_column("N", justify="right", width=7)
    t.add_column("Brier", justify="right", width=8)
    t.add_column("Accuracy", justify="right", width=10)

    for league, stats in sorted(report.by_league.items(), key=lambda x: -x[1]["n"]):
        t.add_row(
            league[:28],
            str(stats["n"]),
            f"{stats['brier']:.4f}",
            f"{stats['accuracy']:.1%}",
        )
    return t


def print_report(report: BacktestReport) -> None:
    console.print()
    console.print(summary_panel(report))
    console.print()
    console.print(calibration_table(report))
    console.print()
    if report.n_kalshi_matched > 0:
        console.print(ev_distribution_table(report))
        console.print()
    if report.by_league:
        console.print(league_table(report))
        console.print()
