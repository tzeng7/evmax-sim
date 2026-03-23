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
        help="Comma-separated sectors: soccer,tennis",
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
) -> None:
    """
    Run historical backtest using football-data.co.uk (soccer) and tennis-data.co.uk (tennis).

    Validates Pinnacle devigged model calibration against actual match outcomes.
    Optionally joins resolved Kalshi markets for theoretical EV analysis.

    Examples:

      evmax backtest run --sectors soccer

      evmax backtest run --sectors soccer,tennis --seasons 2425,2526

      evmax backtest run --sectors soccer --kalshi   (requires API keys)
    """
    from evmax.backtest.display import print_report
    from evmax.backtest.engine import run_backtest

    sector_list = [s.strip().lower() for s in sectors.split(",") if s.strip()]
    season_list = [s.strip() for s in seasons.split(",") if s.strip()]
    league_list = [l.strip() for l in leagues.split(",") if l.strip()] if leagues else None

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
        )

    if not reports:
        console.print("[red]No data returned. Check sector names and seasons.[/red]")
        raise typer.Exit(1)

    for report in reports:
        print_report(report)
