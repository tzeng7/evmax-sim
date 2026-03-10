"""evmax scan command — run the EV scanning pipeline."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import typer
from rich.console import Console

from evmax.cli.display import print_ev_bets, print_scan_header
from evmax.pipeline.runner import PipelineRunner
from evmax.sectors.registry import ALL_SECTORS

console = Console()
app = typer.Typer()


@app.command("scan")
def scan(
    sectors: Optional[str] = typer.Option(
        None,
        "--sectors",
        "-s",
        help=f"Comma-separated sectors to scan. Available: {', '.join(ALL_SECTORS)}",
    ),
    once: bool = typer.Option(
        False,
        "--once",
        help="Run a single scan cycle and exit.",
    ),
    interval: int = typer.Option(
        300,
        "--interval",
        "-i",
        help="Seconds between scan cycles (default: 300).",
    ),
    no_sim: bool = typer.Option(
        False,
        "--no-sim",
        help="Disable simulation (don't place paper bets).",
    ),
) -> None:
    """Scan prediction markets for +EV opportunities."""
    sector_list = (
        [s.strip().lower() for s in sectors.split(",")]
        if sectors
        else ALL_SECTORS
    )

    # Validate sectors
    invalid = [s for s in sector_list if s not in ALL_SECTORS]
    if invalid:
        console.print(f"[red]Unknown sectors: {invalid}. Available: {ALL_SECTORS}[/red]")
        raise typer.Exit(1)

    print_scan_header(sector_list)

    runner = PipelineRunner(sectors=sector_list, simulate=not no_sim)

    if once:
        asyncio.run(_run_once(runner))
    else:
        asyncio.run(_run_loop(runner, interval))


async def _run_once(runner: PipelineRunner) -> None:
    """Run one scan cycle."""
    with console.status("[bold green]Scanning markets...[/bold green]"):
        result = await runner.run_cycle()

    console.print(
        f"\n[bold]Cycle complete[/bold]: {result.markets_fetched} markets fetched, "
        f"{result.markets_matched} matched, "
        f"[green bold]{result.ev_bets_found}[/green bold] +EV found "
        f"([dim]{result.cycle_duration_s:.1f}s[/dim])\n"
    )
    print_ev_bets(result.ev_bets)


async def _run_loop(runner: PipelineRunner, interval: int) -> None:
    """Continuous scan loop."""
    cycle = 0
    while True:
        cycle += 1
        console.print(f"\n[dim]--- Cycle #{cycle} ---[/dim]")
        try:
            await _run_once(runner)
        except Exception as e:
            console.print(f"[red]Cycle error: {e}[/red]")

        console.print(f"[dim]Next scan in {interval}s. Ctrl+C to stop.[/dim]")
        await asyncio.sleep(interval)
