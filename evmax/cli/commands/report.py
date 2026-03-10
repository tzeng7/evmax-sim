"""evmax report command — P&L and performance reporting."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from sqlalchemy import select

from evmax.cli.display import console, metrics_panel, ev_table
from evmax.db import AsyncSessionLocal
from evmax.models.bankroll import BankrollSnapshotORM
from evmax.models.ev_bet import EVBetORM, EVBet
from evmax.models.simulated_bet import SimulatedBet, SimulatedBetORM
from evmax.settings import get_settings
from evmax.simulation.metrics import calculate_metrics

app = typer.Typer()


@app.command("report")
def report(
    sector: Optional[str] = typer.Option(
        None,
        "--sector",
        help="Filter report by sector.",
    ),
    last_n: int = typer.Option(
        50,
        "--last-n",
        "-n",
        help="Number of most recent bets to include.",
    ),
    bankroll: bool = typer.Option(
        False,
        "--bankroll",
        help="Show bankroll history.",
    ),
) -> None:
    """Show P&L report and performance metrics."""
    if bankroll:
        asyncio.run(_show_bankroll())
    else:
        asyncio.run(_show_report(sector, last_n))


async def _show_report(sector_filter: Optional[str], last_n: int) -> None:
    settings = get_settings()

    async with AsyncSessionLocal() as session:
        query = select(SimulatedBetORM).order_by(SimulatedBetORM.id.desc()).limit(last_n)
        if sector_filter:
            query = query.where(SimulatedBetORM.sector == sector_filter.lower())

        result = await session.execute(query)
        bet_rows = result.scalars().all()

        # Get latest bankroll
        br_result = await session.execute(
            select(BankrollSnapshotORM).order_by(BankrollSnapshotORM.id.desc()).limit(1)
        )
        latest_snapshot = br_result.scalar_one_or_none()
        current_bankroll = (
            latest_snapshot.balance_usd if latest_snapshot else settings.initial_bankroll
        )

    bets = [SimulatedBet.model_validate(r.__dict__) for r in bet_rows]

    if not bets:
        console.print(f"[yellow]No bets found{f' for sector {sector_filter}' if sector_filter else ''}.[/yellow]")
        console.print(f"[dim]Starting bankroll: ${settings.initial_bankroll:,.2f}[/dim]")
        return

    metrics = calculate_metrics(bets, initial_bankroll=settings.initial_bankroll)
    console.print(metrics_panel(metrics, bankroll=current_bankroll))

    # Show recent EV bets (opportunities that were found)
    async with AsyncSessionLocal() as session:
        ev_query = select(EVBetORM).order_by(EVBetORM.id.desc()).limit(20)
        if sector_filter:
            ev_query = ev_query.where(EVBetORM.sector == sector_filter.lower())
        ev_result = await session.execute(ev_query)
        ev_rows = ev_result.scalars().all()

    if ev_rows:
        ev_bets = [EVBet.model_validate(r.__dict__) for r in reversed(ev_rows)]
        console.print()
        console.print(ev_table(ev_bets, title="Recent EV Opportunities (last 20)"))


async def _show_bankroll() -> None:
    settings = get_settings()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(BankrollSnapshotORM).order_by(BankrollSnapshotORM.recorded_at.asc()).limit(100)
        )
        snapshots = result.scalars().all()

    if not snapshots:
        console.print(f"[yellow]No bankroll history. Initial: ${settings.initial_bankroll:,.2f}[/yellow]")
        return

    from rich.table import Table
    from rich import box

    table = Table(
        title="Bankroll History",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Time", style="dim")
    table.add_column("Balance", justify="right", style="cyan")
    table.add_column("P&L", justify="right")
    table.add_column("ROI%", justify="right")
    table.add_column("Open", justify="right")
    table.add_column("W/L", justify="right")

    for snap in snapshots[-25:]:  # Last 25 snapshots
        pnl_style = "green" if snap.total_pnl_usd >= 0 else "red"
        roi_style = "green" if snap.roi_pct >= 0 else "red"
        table.add_row(
            snap.recorded_at.strftime("%m/%d %H:%M"),
            f"${snap.balance_usd:,.2f}",
            f"[{pnl_style}]${snap.total_pnl_usd:+,.2f}[/{pnl_style}]",
            f"[{roi_style}]{snap.roi_pct:+.2f}%[/{roi_style}]",
            str(snap.open_bets_count),
            f"{snap.won_bets_count}W/{snap.lost_bets_count}L",
        )

    console.print(table)
