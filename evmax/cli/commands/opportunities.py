"""evmax opps — unified pre-game + live +EV opportunity viewer with interactive bet placement."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from evmax.models.ev_bet import EVBet
from evmax.pipeline.live_scanner import LiveScanner
from evmax.pipeline.runner import PipelineRunner
from evmax.sectors.registry import ALL_SECTORS
from evmax.settings import get_settings

console = Console()
app = typer.Typer()


@app.command("opps")
def opportunities(
    sectors: Optional[str] = typer.Option(
        None,
        "--sectors",
        "-s",
        help=f"Comma-separated sectors. Available: {', '.join(ALL_SECTORS)}",
    ),
    threshold: float = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Minimum EV threshold (default: from settings, usually 2%).",
    ),
    no_sim: bool = typer.Option(
        False,
        "--no-sim",
        help="Display only — do not prompt to place bets.",
    ),
) -> None:
    """View all +EV opportunities (pre-game and live) and choose which bets to place."""
    sector_list = (
        [s.strip().lower() for s in sectors.split(",")]
        if sectors
        else ALL_SECTORS
    )
    invalid = [s for s in sector_list if s not in ALL_SECTORS]
    if invalid:
        console.print(f"[red]Unknown sectors: {invalid}. Available: {ALL_SECTORS}[/red]")
        raise typer.Exit(1)

    ev_threshold = threshold if threshold is not None else get_settings().ev_threshold

    asyncio.run(_run(sector_list, ev_threshold, no_sim))


async def _run(sectors: list[str], threshold: float, no_sim: bool) -> None:
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")

    with console.status("[bold green]Scanning pre-game and live markets...[/bold green]"):
        pre_runner = PipelineRunner(sectors=sectors, simulate=False, dry_run=True)
        live_scanner = LiveScanner()

        pre_result, live_bets = await asyncio.gather(
            pre_runner.run_cycle(),
            live_scanner.scan(sectors, threshold=threshold),
        )

    pre_bets = pre_result.ev_bets

    if not pre_bets and not live_bets:
        console.print(f"\n[yellow]No +EV opportunities found.[/yellow] (threshold: {threshold*100:.0f}%)\n")
        return

    # Build combined indexed list: (index, EVBet, is_live)
    entries: list[tuple[int, EVBet, bool]] = []
    idx = 1
    for bet in live_bets:
        entries.append((idx, bet, True))
        idx += 1
    for bet in pre_bets:
        entries.append((idx, bet, False))
        idx += 1

    _print_table(entries, threshold, now_str)

    if no_sim:
        return

    # Interactive selection
    console.print()
    raw = console.input(
        "[bold]Place bets[/bold] [dim](comma-separated #s, 'all', or Enter to skip):[/dim] "
    ).strip()

    if not raw:
        console.print("[dim]No bets placed.[/dim]")
        return

    if raw.lower() == "all":
        selected = entries
    else:
        try:
            chosen_indices = {int(x.strip()) for x in raw.split(",")}
        except ValueError:
            console.print("[red]Invalid input — no bets placed.[/red]")
            return
        selected = [e for e in entries if e[0] in chosen_indices]

    if not selected:
        console.print("[yellow]No matching bets found.[/yellow]")
        return

    # Place selected bets via PipelineRunner (includes live price re-check)
    runner = PipelineRunner(sectors=sectors, simulate=True, dry_run=False)

    with console.status("[bold green]Placing bets...[/bold green]"):
        await runner.place_bets([bet for _, bet, _ in selected])

    console.print(f"\n[green bold]Placed {len(selected)} bet(s).[/green bold]")
    for num, bet, is_live in selected:
        label = "[red]LIVE[/red]" if is_live else "[cyan]PRE[/cyan]"
        console.print(
            f"  {label} #{num}  {bet.market_id.split(':', 1)[-1]}  "
            f"[green]{bet.ev*100:.1f}% EV[/green]  {bet.payout_decimal:.2f}x"
        )


def _print_table(
    entries: list[tuple[int, EVBet, bool]],
    threshold: float,
    now_str: str,
) -> None:
    table = Table(
        title=f"+EV Opportunities  [{now_str}]  (threshold: {threshold*100:.0f}%)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )

    table.add_column("#", style="bold dim", no_wrap=True, width=3)
    table.add_column("Type", no_wrap=True, width=6)
    table.add_column("Sector", style="yellow", no_wrap=True)
    table.add_column("Market", max_width=42)
    table.add_column("Price", justify="right", style="cyan")
    table.add_column("True", justify="right", style="green")
    table.add_column("EV%", justify="right")
    table.add_column("Payout", justify="right", style="blue")
    table.add_column("Kelly%", justify="right", style="magenta")
    table.add_column("Vol $", justify="right", style="dim")

    for num, bet, is_live in entries:
        ev_color = "green bold" if bet.ev >= 0.10 else "green" if bet.ev >= 0.05 else "yellow"
        type_label = Text("LIVE", style="red bold") if is_live else Text("PRE", style="cyan")
        market_label = bet.market_id.split(":", 1)[-1][:40]

        table.add_row(
            str(num),
            type_label,
            bet.sector.upper(),
            market_label,
            f"{bet.market_implied_prob:.3f}",
            f"{bet.true_prob:.3f}",
            Text(f"{bet.ev*100:.1f}%", style=ev_color),
            f"{bet.payout_decimal:.2f}x",
            f"{bet.suggested_units*100:.2f}%",
            f"${bet.volume_usd:,.0f}",
        )

    console.print(table)
    console.print()

    # Per-bet EV explanation
    for num, bet, is_live in entries:
        _print_bet_explanation(num, bet, is_live)


def _print_bet_explanation(num: int, bet: EVBet, is_live: bool) -> None:
    """Print a concise statistical explanation of why this bet is +EV."""
    from rich.panel import Panel

    type_tag = "[red bold]LIVE[/red bold]" if is_live else "[cyan]PRE-GAME[/cyan]"
    is_spread = "SPREAD" in bet.market_id.upper()

    kalshi_implied = bet.market_implied_prob
    true_prob = bet.true_prob
    payout = bet.payout_decimal
    edge_pp = (true_prob - kalshi_implied) * 100
    ev_pct = bet.ev * 100

    # Pinnacle fair decimal odds (1 / true_prob)
    fair_decimal = 1.0 / true_prob if true_prob > 0 else 0.0
    # Convert to American for familiarity
    if fair_decimal >= 2.0:
        american = f"+{int((fair_decimal - 1) * 100)}"
    else:
        american = f"{int(-100 / (fair_decimal - 1))}"

    # Kalshi American odds
    if payout >= 2.0:
        kalshi_american = f"+{int((payout - 1) * 100)}"
    else:
        kalshi_american = f"{int(-100 / (payout - 1))}"

    market_short = bet.market_id.split(":", 1)[-1]

    lines = [
        f"[bold]#{num}[/bold]  {type_tag}  [yellow]{bet.sector.upper()}[/yellow]  {market_short}",
        "",
    ]

    if is_spread:
        lines += [
            f"  [bold]Model:[/bold] SpreadDistributionModel (normal distribution of point margins)",
            f"  [bold]Sharp consensus:[/bold] Pinnacle devigged probability for YES side to cover",
            f"  [bold]Sharp probability:[/bold]  [green]{true_prob*100:.1f}%[/green]  (fair line: {american})",
            f"  [bold]Kalshi price:[/bold]       [cyan]{kalshi_implied*100:.1f}%[/cyan] implied  ({payout:.2f}x payout / {kalshi_american})",
            f"  [bold]Probability edge:[/bold]   [yellow]+{edge_pp:.1f}pp[/yellow]  ({true_prob*100:.1f}% − {kalshi_implied*100:.1f}%)",
        ]
    else:
        lines += [
            f"  [bold]Sharp source:[/bold]  Pinnacle via TheOddsAPI (Power Method devigged)",
            f"  [bold]Sharp probability:[/bold]  [green]{true_prob*100:.1f}%[/green]  (fair line: {american})",
            f"  [bold]Kalshi price:[/bold]       [cyan]{kalshi_implied*100:.1f}%[/cyan] implied  ({payout:.2f}x payout / {kalshi_american})",
            f"  [bold]Probability edge:[/bold]   [yellow]+{edge_pp:.1f}pp[/yellow]  ({true_prob*100:.1f}% − {kalshi_implied*100:.1f}%)",
        ]

    lines += [
        f"  [bold]EV formula:[/bold]        {true_prob*100:.1f}% × {payout:.2f}x − 1  =  [green bold]+{ev_pct:.1f}%[/green bold]",
        f"  [bold]Kelly sizing:[/bold]      {bet.suggested_units*100:.2f}% of bankroll",
        f"  [bold]Volume (liquidity):[/bold] ${bet.volume_usd:,.0f}",
    ]

    if is_live:
        lines.append(
            f"  [dim]⚠  Live bet — price confirmed against Kalshi API. "
            f"Pinnacle's live line reflects current game state.[/dim]"
        )
    else:
        lines.append(
            f"  [dim]✓  Pre-game — bet is valid until game start. "
            f"Price will be re-confirmed at placement.[/dim]"
        )

    console.print(Panel("\n".join(lines), border_style="dim", padding=(0, 1)))
