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
    from evmax.backtest.display_walkforward import print_walkforward_report
    from evmax.backtest.engine import run_backtest
    from evmax.backtest.models import PropBacktestReport
    from evmax.backtest.sources.espn_walkforward import WalkForwardReport

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
        )

    if not reports:
        console.print("[red]No data returned. Check sector names and seasons.[/red]")
        raise typer.Exit(1)

    for report in reports:
        if isinstance(report, WalkForwardReport):
            print_walkforward_report(report)
        elif isinstance(report, PropBacktestReport):
            print_prop_report(report)
        else:
            print_report(report)
