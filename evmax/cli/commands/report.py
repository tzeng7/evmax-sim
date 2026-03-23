"""evmax report command — P&L and performance reporting."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from sqlalchemy import select

from sqlalchemy import func

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
        0,
        "--last-n",
        "-n",
        help="Number of most recent bets to include (0 = all).",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Only include bets placed on or after this date (YYYY-MM-DD).",
    ),
    until: Optional[str] = typer.Option(
        None,
        "--until",
        help="Only include bets placed on or before this date (YYYY-MM-DD).",
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
        asyncio.run(_show_report(sector, last_n, since, until))


async def _show_report(
    sector_filter: Optional[str],
    last_n: int,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> None:
    from datetime import date, timedelta
    from sqlalchemy import and_

    settings = get_settings()

    # Parse date filters
    since_dt = until_dt = None
    try:
        if since:
            since_dt = datetime.fromisoformat(since).replace(hour=0, minute=0, second=0)
        if until:
            until_dt = datetime.fromisoformat(until).replace(hour=23, minute=59, second=59)
    except ValueError as e:
        console.print(f"[red]Invalid date format: {e}. Use YYYY-MM-DD.[/red]")
        return

    async with AsyncSessionLocal() as session:
        query = select(SimulatedBetORM).order_by(SimulatedBetORM.id.desc())
        if last_n > 0:
            query = query.limit(last_n)
        if sector_filter:
            query = query.where(SimulatedBetORM.sector == sector_filter.lower())
        if since_dt:
            query = query.where(SimulatedBetORM.placed_at >= since_dt)
        if until_dt:
            query = query.where(SimulatedBetORM.placed_at <= until_dt)

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

    # Calibration: join settled bets with their ev_bet to get true_prob at placement
    ev_bet_ids = [b.ev_bet_id for b in bets if b.status.value in ("won", "lost")]
    if ev_bet_ids:
        async with AsyncSessionLocal() as session:
            cal_result = await session.execute(
                select(EVBetORM.true_prob).where(EVBetORM.id.in_(ev_bet_ids))
            )
            true_probs = [row[0] for row in cal_result.all()]
        if true_probs:
            avg_true_prob = sum(true_probs) / len(true_probs)
            metrics.avg_true_prob = avg_true_prob
            metrics.calibration_error = abs(avg_true_prob - metrics.win_rate)

    console.print(metrics_panel(metrics, bankroll=current_bankroll))

    # Show EV opportunities — one row per unique market (latest scan only)
    async with AsyncSessionLocal() as session:
        # Subquery: most recent id per (market_id, outcome) = one row per unique contract
        dedup_subq = (
            select(func.max(EVBetORM.id))
            .group_by(EVBetORM.market_id, EVBetORM.outcome)
        )
        if sector_filter:
            dedup_subq = dedup_subq.where(EVBetORM.sector == sector_filter.lower())
        if since_dt:
            dedup_subq = dedup_subq.where(EVBetORM.scanned_at >= since_dt)
        if until_dt:
            dedup_subq = dedup_subq.where(EVBetORM.scanned_at <= until_dt)

        ev_query = (
            select(EVBetORM)
            .where(EVBetORM.id.in_(dedup_subq))
            .order_by(EVBetORM.ev.desc())
        )
        if last_n > 0:
            ev_query = ev_query.limit(last_n)

        ev_result = await session.execute(ev_query)
        ev_rows = ev_result.scalars().all()

    if ev_rows:
        ev_bets = [EVBet.model_validate(r.__dict__) for r in ev_rows]
        title = f"EV Opportunities — {len(ev_bets)} unique markets"
        if last_n > 0:
            title = f"EV Opportunities — top {last_n} by EV"
        console.print()
        console.print(ev_table(ev_bets, title=title, dedup=False))


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
