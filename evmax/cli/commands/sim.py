"""evmax sim commands — list and resolve paper bets."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from sqlalchemy import select

from evmax.cli.display import console, sim_bets_table
from evmax.clients.kalshi import KalshiClient
from evmax.db import AsyncSessionLocal
from evmax.models.simulated_bet import BetStatus, SimulatedBetORM, SimulatedBet
from evmax.simulation.engine import SimulationEngine
from evmax.simulation.resolver import BetResolver

app = typer.Typer()


@app.command("list")
def sim_list(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        help="Filter by status: open | won | lost | void",
    ),
    last_n: int = typer.Option(
        50,
        "--last-n",
        "-n",
        help="Show last N bets.",
    ),
) -> None:
    """List simulated bets."""
    asyncio.run(_list_bets(status, last_n))


async def _list_bets(status_filter: Optional[str], last_n: int) -> None:
    async with AsyncSessionLocal() as session:
        query = select(SimulatedBetORM).order_by(SimulatedBetORM.id.desc()).limit(last_n)

        if status_filter:
            try:
                status_enum = BetStatus(status_filter.lower())
                query = query.where(SimulatedBetORM.status == status_enum)
            except ValueError:
                console.print(f"[red]Invalid status: {status_filter}. Use: open, won, lost, void[/red]")
                return

        result = await session.execute(query)
        rows = result.scalars().all()

    if not rows:
        console.print("[yellow]No bets found.[/yellow]")
        return

    bets = [SimulatedBet.model_validate(r.__dict__) for r in rows]
    bets.reverse()  # Show oldest first
    console.print(sim_bets_table(bets, title=f"Simulated Bets (last {last_n})"))

    total_pnl = sum(b.pnl_usd for b in bets if b.status in (BetStatus.won, BetStatus.lost))
    pnl_style = "green" if total_pnl >= 0 else "red"
    console.print(f"\n[bold]Total P&L (shown):[/bold] [{pnl_style}]${total_pnl:+.2f}[/{pnl_style}]")


@app.command("resolve")
def sim_resolve(
    market_id: Optional[str] = typer.Option(
        None,
        "--market",
        help="Resolve specific market ID.",
    ),
    yes_price: Optional[float] = typer.Option(
        None,
        "--yes-price",
        help="Final YES price for resolution (0.0 or 1.0).",
    ),
) -> None:
    """Resolve open simulated bets. Without args, auto-fetches results from Kalshi."""
    if market_id and yes_price is not None:
        asyncio.run(_resolve_market(market_id, yes_price))
    else:
        asyncio.run(_auto_resolve())


async def _resolve_market(market_id: str, yes_price: float) -> None:
    resolver = BetResolver()
    resolved = await resolver.resolve_all_settled({market_id: yes_price})

    if not resolved:
        console.print(f"[yellow]No open bets found for market: {market_id}[/yellow]")
        return

    console.print(f"[green]Resolved {len(resolved)} bet(s):[/green]")
    for bet in resolved:
        status_style = "green" if bet.status == BetStatus.won else "red"
        console.print(
            f"  Bet #{bet.id}: [{status_style}]{bet.status.value.upper()}[/{status_style}]  "
            f"P&L: ${bet.pnl_usd:+.2f}"
        )


async def _auto_resolve() -> None:
    """Fetch current prices from Kalshi for all open bets and resolve settled ones."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SimulatedBetORM).where(SimulatedBetORM.status == BetStatus.open)
        )
        open_bets = result.scalars().all()

    if not open_bets:
        console.print("[yellow]No open bets to resolve.[/yellow]")
        return

    console.print(f"Checking {len(open_bets)} open bet(s) against Kalshi...")

    settled: dict[str, float] = {}
    async with KalshiClient() as kalshi:
        for bet in open_bets:
            # Only Kalshi markets can be auto-resolved this way
            if not bet.market_id.startswith("kalshi:"):
                continue
            ticker = bet.market_id.split(":", 1)[1]
            price = await kalshi.get_market_price(ticker)
            if price is not None and (price >= 0.99 or price <= 0.01):
                settled[bet.market_id] = price
                result_label = "YES" if price >= 0.99 else "NO"
                console.print(f"  [bold]{ticker}[/bold] → settled {result_label}")

    if not settled:
        console.print("[yellow]No settled markets found — bets remain open.[/yellow]")
        return

    resolver = BetResolver()
    resolved = await resolver.resolve_all_settled(settled)

    # Update bankroll snapshot after resolution
    engine = SimulationEngine()
    async with AsyncSessionLocal() as session:
        await engine._save_snapshot(session)
        await session.commit()

    console.print(f"\n[green]Resolved {len(resolved)} bet(s):[/green]")
    total_pnl = 0.0
    for bet in resolved:
        status_style = "green" if bet.status == BetStatus.won else "red"
        console.print(
            f"  Bet #{bet.id} [{bet.sector}] {bet.market_id.split(':',1)[-1][:35]:35s}  "
            f"[{status_style}]{bet.status.value.upper()}[/{status_style}]  "
            f"P&L: [bold]${bet.pnl_usd:+.2f}[/bold]"
        )
        total_pnl += bet.pnl_usd

    pnl_style = "green" if total_pnl >= 0 else "red"
    console.print(f"\n[bold]Session P&L: [{pnl_style}]${total_pnl:+.2f}[/{pnl_style}][/bold]")
