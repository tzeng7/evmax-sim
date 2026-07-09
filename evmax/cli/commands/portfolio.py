"""CLI commands for portfolio management and scheduled scanning."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import typer
from rich import box
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("list")
def list_portfolios(
    active_only: bool = typer.Option(True, "--active/--all", help="Show only active portfolios"),
) -> None:
    """List all portfolios."""
    from evmax.portfolios import list_portfolios as _list, get_portfolio_stats

    portfolios = _list(active_only=active_only)
    if not portfolios:
        console.print("[yellow]No portfolios found. Run 'evmax portfolio create-defaults' to get started.[/yellow]")
        return

    table = Table(title="Portfolios", box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Scenario")
    table.add_column("Sectors")
    table.add_column("Bankroll", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Bets", justify="right")
    table.add_column("Win%", justify="right")

    for p in portfolios:
        stats = get_portfolio_stats(p.id)
        pnl = stats.get("total_pnl", 0)
        pnl_str = f"[green]+{pnl:.2f}[/green]" if pnl >= 0 else f"[red]{pnl:.2f}[/red]"
        curr = stats.get("current_bankroll", p.current_bankroll)
        curr_str = f"[green]${curr:.2f}[/green]" if curr >= p.initial_bankroll else f"[red]${curr:.2f}[/red]"

        table.add_row(
            p.id, p.name, p.scenario,
            ", ".join(p.sectors),
            f"${p.initial_bankroll:.0f}",
            curr_str, pnl_str,
            str(stats.get("total_bets", 0)),
            f"{stats.get('win_rate', 0):.0f}%",
        )

    console.print(table)


@app.command("create-defaults")
def create_defaults() -> None:
    """Create default portfolios (3 scenarios x 7 sector groups)."""
    from evmax.portfolios import create_default_portfolios

    created = create_default_portfolios()
    console.print(f"[green]Created {len(created)} portfolios[/green]")
    for p in created:
        console.print(f"  {p.id}: {p.name} — ${p.initial_bankroll:.0f} / {p.kelly_fraction*100:.0f}% Kelly")


@app.command("create")
def create(
    portfolio_id: str = typer.Argument(..., help="Portfolio ID (e.g. 'nba_custom')"),
    name: str = typer.Option(..., "--name", "-n"),
    sectors: str = typer.Option(..., "--sectors", "-s", help="Comma-separated sectors"),
    bankroll: float = typer.Option(250.0, "--bankroll", "-b"),
    kelly: float = typer.Option(0.5, "--kelly", "-k"),
    scenario: str = typer.Option("moderate", "--scenario"),
) -> None:
    """Create a custom portfolio."""
    from evmax.portfolios import create_portfolio

    sector_list = [s.strip() for s in sectors.split(",")]
    p = create_portfolio(portfolio_id, name, sector_list, bankroll, kelly, scenario)
    console.print(f"[green]Created portfolio '{p.id}': {p.name}[/green]")


@app.command("show")
def show(
    portfolio_id: str = typer.Argument(..., help="Portfolio ID"),
) -> None:
    """Show detail for a portfolio."""
    from evmax.portfolios import get_portfolio_stats, get_portfolio_bets

    stats = get_portfolio_stats(portfolio_id)
    if not stats:
        console.print(f"[red]Portfolio '{portfolio_id}' not found[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{stats['name']}[/bold] ({stats['scenario']})")
    console.print(f"  Sectors: {', '.join(stats['sectors'])}")
    console.print(f"  Initial: ${stats['initial_bankroll']:.2f}  Current: ${stats['current_bankroll']:.2f}")

    pnl = stats["total_pnl"]
    pnl_str = f"[green]+${pnl:.2f}[/green]" if pnl >= 0 else f"[red]-${abs(pnl):.2f}[/red]"
    console.print(f"  P&L: {pnl_str}  ROI: {stats['roi_pct']:.1f}%")
    console.print(f"  Bets: {stats['total_bets']} ({stats['open_bets']} open, {stats['settled_bets']} settled)")
    console.print(f"  Wins: {stats['wins']}  Losses: {stats['losses']}  Win Rate: {stats['win_rate']:.0f}%")

    bets = get_portfolio_bets(portfolio_id)
    if not bets:
        console.print("\n  [dim]No bets yet[/dim]")
        return

    table = Table(box=box.SIMPLE)
    table.add_column("Date")
    table.add_column("Event")
    table.add_column("Outcome")
    table.add_column("Sector")
    table.add_column("EV%", justify="right")
    table.add_column("Stake", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Result")

    for b in bets[:30]:
        label = b.get("display_label") or b.get("yes_team") or "?"
        pnl_val = b.get("pnl")
        pnl_s = f"{pnl_val:+.2f}" if pnl_val is not None else "-"
        pnl_color = "green" if (pnl_val or 0) >= 0 else "red"
        result = "WON" if b["outcome"] == 1 else "LOST" if b["outcome"] == 0 else "OPEN"
        result_color = "green" if result == "WON" else "red" if result == "LOST" else "dim"

        table.add_row(
            b.get("event_date") or b.get("scan_date") or "",
            (b.get("event_title") or "")[:35],
            label,
            b.get("sector") or "",
            f"{(b.get('ev_pct') or 0)*100:.1f}%",
            f"${b.get('stake') or 0:.2f}",
            f"[{pnl_color}]{pnl_s}[/{pnl_color}]",
            f"[{result_color}]{result}[/{result_color}]",
        )

    console.print(table)


@app.command("scan")
def scan_portfolios(
    portfolio_id: str = typer.Option("", "--portfolio", "-p", help="Scan specific portfolio (default: all)"),
) -> None:
    """Run a scan and log bets to matching portfolios."""
    asyncio.run(_scan_portfolios(portfolio_id or None))


async def _scan_portfolios(portfolio_id: str | None) -> None:
    from evmax.agents.coordinator import AgentCoordinator
    from evmax.portfolios import (
        list_portfolios, log_portfolio_bet, is_excluded_from_portfolio,
        select_best_execution_plays,
    )

    portfolios = list_portfolios(active_only=True)
    if portfolio_id:
        portfolios = [p for p in portfolios if p.id == portfolio_id]

    if not portfolios:
        console.print("[red]No matching portfolios found[/red]")
        return

    all_sectors = sorted(set(s for p in portfolios for s in p.sectors))
    console.print(f"Scanning {len(portfolios)} portfolio(s) across sectors: {', '.join(all_sectors)}")

    coord = AgentCoordinator(sectors=all_sectors, bankroll=500, kelly_fraction=0.5)
    cycle = await coord.run_cycle()
    today_str = date.today().isoformat()

    # plays() drops partial-blend (shadow, $0.00-stake) gaps + map_handicap —
    # the canonical actionable-play definition shared with the CLI scan,
    # dashboard, and cron portfolio scanner. Best-execution collapse: the same
    # market quoted on both Kalshi and Polymarket US is one bet, not two.
    playable = select_best_execution_plays(
        cycle.plays(require_full_blend=True, drop_map_handicap=True)
    )

    total_logged = 0
    for portfolio in portfolios:
        count = 0
        for g in playable:
            if g.sector not in portfolio.sectors:
                continue
            if is_excluded_from_portfolio(g.sector, g.market_type):
                continue

            event_date_str = str(g.event_date.astimezone().strftime("%Y-%m-%d") if g.event_date else "")
            # Drop only strictly-past games (keep today + future). The old
            # "== today" silently dropped Finals/series games Kalshi lists 2-4
            # days out; the web fan-out has no date floor at all.
            if event_date_str and event_date_str < today_str:
                continue

            gap_dict = {
                "market_id": g.market_id or "",
                "event_id": g.event_id or "",
                "event_title": g.event_title or "",
                "yes_team": g.yes_team or "",
                "market_type": g.market_type or "",
                "display_label": getattr(g, "display_label", "") or "",
                "sector": g.sector or "",
                "kalshi_yes_price": round(g.kalshi_yes_price, 4),
                "blended_true_prob": round(g.blended_true_prob, 4),
                "ev_pct": round(g.ev_pct, 4),
                "kelly_fraction": round(g.kelly_fraction, 4),
                "event_date": event_date_str,
                "volume_usd": g.volume_usd or 0,
                "model_sources": g.model_sources or "",
                "scan_date": today_str,
            }
            if log_portfolio_bet(
                portfolio.id, gap_dict, portfolio.current_bankroll, portfolio.kelly_fraction
            ):
                count += 1

        total_logged += count
        console.print(f"  [cyan]{portfolio.id}[/cyan]: {count} bets logged")

    console.print(f"\n[green]Done — {total_logged} total bets across {len(portfolios)} portfolios[/green]")


@app.command("sync")
def sync_outcomes() -> None:
    """Sync portfolio bet outcomes from ev_outcomes."""
    from evmax.portfolios import sync_portfolio_outcomes

    count = sync_portfolio_outcomes()
    console.print(f"[green]Resolved {count} portfolio bet(s)[/green]")


@app.command("delete")
def delete(
    portfolio_id: str = typer.Argument(..., help="Portfolio ID to deactivate"),
) -> None:
    """Deactivate a portfolio."""
    from evmax.portfolios import delete_portfolio

    ok = delete_portfolio(portfolio_id)
    if ok:
        console.print(f"[green]Deactivated '{portfolio_id}'[/green]")
    else:
        console.print(f"[red]Portfolio '{portfolio_id}' not found[/red]")
