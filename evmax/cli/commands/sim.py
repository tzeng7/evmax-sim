"""evmax sim commands — list and resolve paper bets."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table
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


@app.command("montecarlo")
def montecarlo(
    trials: int = typer.Option(10_000, "--trials", "-t", help="Number of simulation trials."),
    bankroll: float = typer.Option(1_000.0, "--bankroll", "-b", help="Starting bankroll in USD."),
    kelly: float = typer.Option(0.25, "--kelly", "-k", help="Kelly fraction (default: quarter Kelly)."),
    since: Optional[str] = typer.Option(None, "--since", help="Only use bets since YYYY-MM-DD."),
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Filter by sector."),
    min_ev: float = typer.Option(0.02, "--min-ev", help="Minimum EV threshold for included bets."),
) -> None:
    """Run Monte Carlo bankroll simulation on historical bets with known outcomes."""
    from evmax.agents.cleanup.db import get_connection
    from evmax.simulation.montecarlo import MonteCarloSimulator

    conn = get_connection()
    query = """
        SELECT p.kalshi_yes_price, p.blended_true_prob, p.kelly_fraction,
               p.ev_pct, p.sector, o.outcome
        FROM ev_predictions p
        INNER JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE o.outcome IS NOT NULL AND p.voided = 0
    """
    params: list = []

    if since:
        query += " AND p.scan_date >= ?"
        params.append(since)
    if sector:
        query += " AND p.sector = ?"
        params.append(sector.lower())
    if min_ev:
        query += " AND p.ev_pct >= ?"
        params.append(min_ev)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        console.print(
            "[yellow]No resolved bets found.[/yellow] "
            "Run [bold]evmax cleanup resolve[/bold] first to log outcomes."
        )
        return

    bets = [
        {
            "kalshi_yes_price": r["kalshi_yes_price"],
            "true_prob": r["blended_true_prob"],
            "kelly_fraction": r["kelly_fraction"] * kelly / 0.5,  # rescale to requested fraction
            "ev_pct": r["ev_pct"],
            "sector": r["sector"],
            "outcome": r["outcome"],
        }
        for r in rows
    ]

    console.print(
        f"\n[bold cyan]Monte Carlo Simulation[/bold cyan] — "
        f"{len(bets)} resolved bets | {trials:,} trials | "
        f"${bankroll:,.0f} bankroll | {kelly:.0%} Kelly\n"
    )

    sim = MonteCarloSimulator()
    result = sim.run(bets, trials=trials, starting_bankroll=bankroll, kelly_fraction=kelly)

    # Summary table
    table = Table(title="Monte Carlo Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    pnl_style = "green" if result.median_final >= bankroll else "red"
    p5_style = "green" if result.p5_final >= bankroll else "red"
    p95_style = "green" if result.p95_final >= bankroll else "red"

    def _fmt(val: float) -> str:
        pct = (val / bankroll - 1) * 100
        return f"${val:,.2f}  ({pct:+.1f}%)"

    table.add_row("Median final bankroll", f"[{pnl_style}]{_fmt(result.median_final)}[/{pnl_style}]")
    table.add_row("Mean final bankroll", _fmt(result.mean_final))
    table.add_row("5th percentile (bad run)", f"[{p5_style}]{_fmt(result.p5_final)}[/{p5_style}]")
    table.add_row("25th percentile", _fmt(result.p25_final))
    table.add_row("75th percentile", _fmt(result.p75_final))
    table.add_row("95th percentile (great run)", f"[{p95_style}]{_fmt(result.p95_final)}[/{p95_style}]")
    ruin_style = "red bold" if result.ruin_probability > 0.1 else "yellow" if result.ruin_probability > 0.03 else "green"
    table.add_row("Ruin probability (<20% bankroll)", f"[{ruin_style}]{result.ruin_probability*100:.1f}%[/{ruin_style}]")
    table.add_row("Median max drawdown", f"{result.max_drawdown_median*100:.1f}%")
    table.add_row("Total bets", str(result.total_bets))
    table.add_row("Median total wagered", f"${result.total_wagered:,.2f}")

    console.print(table)

    # Sparkline path chart
    if result.path_p50:
        console.print("\n[bold]Bankroll path (p5 / median / p95 at checkpoints):[/bold]")
        for i, (lo, mid, hi) in enumerate(
            zip(result.path_p5, result.path_p50, result.path_p95)
        ):
            bar_len = max(1, int((mid / bankroll) * 30))
            bar = "█" * bar_len
            mid_style = "green" if mid >= bankroll else "red"
            console.print(
                f"  [{i*10:>3}%] "
                f"[dim]${lo:>8,.0f}[/dim]  "
                f"[{mid_style}]{bar:<32}[/{mid_style}]  "
                f"[dim]${hi:>8,.0f}[/dim]"
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
