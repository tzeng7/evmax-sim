"""CLI commands for historical backtest."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Historical model calibration backtest using Pinnacle odds.")
console = Console()


@app.command("run")
def run(
    sectors: str = typer.Option(
        "soccer,tennis", "--sectors", "-s",
        help="Comma-separated sectors: soccer,tennis,nfl_props",
    ),
    seasons: str = typer.Option(
        "2425,2526", "--seasons",
        help="Soccer seasons (e.g. '2425,2526'). Also used to derive tennis years.",
    ),
    leagues: Optional[str] = typer.Option(
        None, "--leagues", "-l",
        help="Soccer league codes to include (e.g. 'E0,SP1'). Default: all 5 leagues.",
    ),
    kalshi: bool = typer.Option(
        False, "--kalshi",
        help="Fetch resolved Kalshi markets and compute theoretical EV (requires API auth).",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-download cached CSV/XLSX files.",
    ),
    ev_threshold: float = typer.Option(
        0.02, "--ev-threshold",
        help="Minimum EV to count as positive edge (default 2%%).",
    ),
    min_volume: float = typer.Option(
        0.0, "--min-volume",
        help="NFL props: minimum Kalshi volume to include a market (default: all).",
    ),
    stats: Optional[str] = typer.Option(
        None, "--stats",
        help="NFL props: comma-separated stat types to include (e.g. 'passing_yards,passing_tds'). Default: all stats.",
    ),
    walkforward: bool = typer.Option(
        False, "--walkforward",
        help="Soccer: replay matches chronologically through Elo/Form/Poisson agents "
             "and report per-model Brier. Unfiltered (every game, not just +EV picks).",
    ),
) -> None:
    """
    Run historical backtest. Supports Pinnacle-odds backtests (soccer, tennis) and
    ESPN walk-forward backtests (wnba, nba, ncaab, baseball, nhl, nfl, ncaaw).

    Examples:

      evmax backtest run --sectors soccer

      evmax backtest run --sectors wnba --seasons 2425

      evmax backtest run --sectors soccer,tennis --seasons 2425,2526

      evmax backtest run --sectors soccer --kalshi   (requires API keys)
    """
    from evmax.backtest.display import print_report
    from evmax.backtest.display_props import print_prop_report
    from evmax.backtest.display_walkforward import (
        print_walkforward_report,
        print_soccer_walkforward_report,
    )
    from evmax.backtest.engine import run_backtest
    from evmax.backtest.models import PropBacktestReport
    from evmax.backtest.sources.espn_walkforward import WalkForwardReport
    from evmax.backtest.sources.soccer_walkforward import SoccerWalkForwardReport

    sector_list = [s.strip().lower() for s in sectors.split(",") if s.strip()]
    season_list = [s.strip() for s in seasons.split(",") if s.strip()]
    league_list = [l.strip() for l in leagues.split(",") if l.strip()] if leagues else None
    stats_list = [s.strip() for s in stats.split(",") if s.strip()] if stats else None

    console.print(
        f"\n[bold cyan]evmax backtest[/bold cyan] — sectors: {', '.join(sector_list)}"
        f"  |  seasons: {', '.join(season_list)}"
        + (f"  |  leagues: {', '.join(league_list)}" if league_list else "")
        + ("\n[yellow]  + Kalshi EV analysis enabled[/yellow]" if kalshi else "")
    )

    with console.status("[bold green]Downloading historical data and computing metrics...[/bold green]"):
        reports = run_backtest(
            sectors=sector_list,
            seasons=season_list,
            leagues=league_list,
            fetch_kalshi=kalshi,
            force_refresh=force,
            ev_threshold=ev_threshold,
            min_volume=min_volume,
            stats_filter=stats_list,
            walkforward=walkforward,
        )

    if not reports:
        console.print("[red]No data returned. Check sector names and seasons.[/red]")
        raise typer.Exit(1)

    for report in reports:
        if isinstance(report, SoccerWalkForwardReport):
            print_soccer_walkforward_report(report)
        elif isinstance(report, WalkForwardReport):
            print_walkforward_report(report)
        elif isinstance(report, PropBacktestReport):
            print_prop_report(report)
        else:
            print_report(report)


@app.command("spread")
def spread(
    months: str = typer.Option(
        None, "--months", "-m",
        help="Comma-separated YYYYMM months (e.g. '202410,202411,...'). Default: Oct 2025 – Apr 2026.",
    ),
) -> None:
    """Backtest NBA spread predictions: PossessionSim margin distribution vs Normal CDF.

    Tests cover probability at standard spread lines (-1.5 to -13.5) against
    actual game margins from ESPN historical data.

    Examples:

      evmax backtest spread

      evmax backtest spread --months 202501,202502,202503
    """
    from evmax.backtest.display_walkforward import print_spread_backtest
    from evmax.backtest.sources.espn_walkforward import run_spread_backtest

    if months:
        month_list = [m.strip() for m in months.split(",") if m.strip()]
    else:
        month_list = [f"2025{m:02d}" for m in range(10, 13)] + [f"2026{m:02d}" for m in range(1, 5)]

    console.print(
        f"\n[bold cyan]evmax backtest spread[/bold cyan] — months: {', '.join(month_list)}"
        "\n  PossessionSim vs Normal CDF (σ=11.5)"
    )

    with console.status("[bold green]Running spread backtest (PossessionSim 10K sims per game)...[/bold green]"):
        report = run_spread_backtest(month_list)

    if report.n_predictions == 0:
        console.print("[red]No spread predictions generated. Check PossessionSim data.[/red]")
        raise typer.Exit(1)

    print_spread_backtest(report)


@app.command("totals")
def totals(
    months: str = typer.Option(
        None, "--months", "-m",
        help="Comma-separated YYYYMM months (e.g. '202410,202411,...'). Default: Oct 2025 – Apr 2026.",
    ),
) -> None:
    """Backtest NBA totals predictions: PossessionSim total distribution vs Normal CDF.

    Tests P(total > line) at standard lines (210.5 – 240.5) against actual
    game totals from ESPN historical data.

    Examples:

      evmax backtest totals

      evmax backtest totals --months 202501,202502,202503
    """
    from evmax.backtest.display_walkforward import print_totals_backtest
    from evmax.backtest.sources.espn_walkforward import run_totals_backtest

    if months:
        month_list = [m.strip() for m in months.split(",") if m.strip()]
    else:
        month_list = [f"2025{m:02d}" for m in range(10, 13)] + [f"2026{m:02d}" for m in range(1, 5)]

    console.print(
        f"\n[bold cyan]evmax backtest totals[/bold cyan] — months: {', '.join(month_list)}"
        "\n  PossessionSim vs Normal CDF (σ=20)"
    )

    with console.status("[bold green]Running totals backtest (PossessionSim 10K sims per game)...[/bold green]"):
        report = run_totals_backtest(month_list)

    if report.n_predictions == 0:
        console.print("[red]No totals predictions generated. Check PossessionSim data.[/red]")
        raise typer.Exit(1)

    print_totals_backtest(report)
