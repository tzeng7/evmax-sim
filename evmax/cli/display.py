"""Rich display helpers — tables, panels, formatting."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from evmax.models.ev_bet import EVBet
from evmax.models.simulated_bet import BetStatus, SimulatedBet
from evmax.simulation.metrics import PerformanceMetrics

console = Console()


def _readable_event(event_id: str, market_id: str) -> str:
    """Derive a human-readable event name from event_id or fall back to market_id.

    event_id format: "{sector}::{date}::{team_a}_vs_{team_b}[::spread|::total|::prop::...]"
    Returns e.g. "Leverkusen vs Bayer Munich"
    """
    parts = event_id.split("::")
    if len(parts) >= 3:
        matchup = parts[2]  # e.g. "leverkusen_vs_bayer_munich"
        return matchup.replace("_", " ").title()
    # Fall back to ticker portion of market_id
    return market_id.split(":", 1)[-1][:38]


def _market_type_label(event_id: str) -> str:
    """Derive a short market type label from event_id suffix.

    event_id format: "{sector}::{date}::{matchup}[::spread|::total|::prop|...]"
    Returns: "ML", "SPD", "O/U", "PROP", or "?"
    """
    parts = event_id.split("::")
    if len(parts) >= 4:
        suffix = parts[3].lower()
        if suffix == "spread":
            return "SPD"
        if suffix in ("total", "ou"):
            return "O/U"
        if suffix == "prop":
            return "PROP"
        return suffix.upper()[:4]
    return "ML"


def _dedup_ev_bets(ev_bets: list[EVBet]) -> list[EVBet]:
    """Deduplicate EV bets from multiple scan cycles.

    Keeps only the most recent scan per (market_id, outcome) pair,
    since repeat scans of the same market produce near-identical rows.
    """
    seen: dict[tuple[str, str], EVBet] = {}
    for bet in ev_bets:
        key = (bet.market_id, bet.outcome)
        existing = seen.get(key)
        if existing is None or bet.scanned_at > existing.scanned_at:
            seen[key] = bet
    # Return in original relative order (preserves chronological sort from caller)
    ordered = {id(b): i for i, b in enumerate(ev_bets)}
    return sorted(seen.values(), key=lambda b: ordered[id(b)])


def ev_table(ev_bets: list[EVBet], title: str = "EV Opportunities", dedup: bool = True) -> Table:
    """Build a Rich table for +EV bets."""
    display_bets = _dedup_ev_bets(ev_bets) if dedup else ev_bets

    table = Table(
        title=f"{title} ({len(display_bets)} unique)" if dedup and len(display_bets) != len(ev_bets) else title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )

    table.add_column("Sector", style="yellow", no_wrap=True)
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Market", style="white", max_width=32)
    table.add_column("Ticker", style="dim", max_width=22)
    table.add_column("Side", style="bold", no_wrap=True)
    table.add_column("PM Price", justify="right", style="cyan")
    table.add_column("True Prob", justify="right", style="green")
    table.add_column("EV%", justify="right")
    table.add_column("Payout", justify="right", style="blue")
    table.add_column("Kelly%", justify="right", style="magenta")
    table.add_column("Vol $", justify="right", style="dim")

    for bet in display_bets:
        ev_color = "green" if bet.ev >= 0.05 else "yellow" if bet.ev >= 0.02 else "white"
        mtype = _market_type_label(bet.event_id)
        # Show the distinctive part of the Kalshi ticker so same-matchup contracts are distinguishable
        ticker = bet.market_id.split(":")[-1][:20] if ":" in bet.market_id else bet.market_id[:20]

        table.add_row(
            bet.sector.upper(),
            mtype,
            _readable_event(bet.event_id, bet.market_id),
            ticker,
            bet.outcome.upper(),
            f"{bet.market_implied_prob:.3f}",
            f"{bet.true_prob:.3f}",
            Text(f"{bet.ev * 100:.1f}%", style=ev_color),
            f"{bet.payout_decimal:.2f}x",
            f"{bet.suggested_units * 100:.2f}%",
            f"${bet.volume_usd:,.0f}",
        )

    return table


def sim_bets_table(bets: list[SimulatedBet], title: str = "Simulated Bets") -> Table:
    """Build a Rich table for simulated bets."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )

    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Sector", style="yellow", no_wrap=True)
    table.add_column("Market", max_width=35)
    table.add_column("Side", no_wrap=True)
    table.add_column("Stake $", justify="right", style="cyan")
    table.add_column("Odds", justify="right", style="blue")
    table.add_column("Status", justify="center")
    table.add_column("P&L $", justify="right")
    table.add_column("Placed", no_wrap=True, style="dim")

    status_styles = {
        BetStatus.open: "yellow",
        BetStatus.won: "green bold",
        BetStatus.lost: "red",
        BetStatus.void: "dim",
    }

    for bet in bets:
        status_style = status_styles.get(bet.status, "white")
        pnl_style = "green" if bet.pnl_usd > 0 else "red" if bet.pnl_usd < 0 else "dim"
        pnl_str = f"${bet.pnl_usd:+.2f}" if bet.status != BetStatus.open else "—"

        table.add_row(
            str(bet.id or ""),
            bet.sector.upper(),
            bet.market_id.split(":", 1)[-1][:33],
            bet.outcome.upper(),
            f"${bet.stake_usd:.2f}",
            f"{bet.odds_decimal:.2f}x",
            Text(bet.status.value.upper(), style=status_style),
            Text(pnl_str, style=pnl_style),
            bet.placed_at.strftime("%m/%d %H:%M"),
        )

    return table


def metrics_panel(metrics: PerformanceMetrics, bankroll: float) -> Panel:
    """Build a Rich panel for performance metrics."""
    roi_style = "green" if metrics.roi_pct >= 0 else "red"
    pnl_style = "green" if metrics.total_pnl_usd >= 0 else "red"

    lines = [
        f"[bold cyan]Bankroll:[/bold cyan] ${bankroll:,.2f}",
        f"[bold]Total P&L:[/bold] [{pnl_style}]${metrics.total_pnl_usd:+,.2f}[/{pnl_style}]  "
        f"ROI: [{roi_style}]{metrics.roi_pct:+.2f}%[/{roi_style}]",
        "",
        f"[bold]Bets:[/bold] {metrics.total_bets} total  "
        f"([green]{metrics.won_bets}W[/green] / [red]{metrics.lost_bets}L[/red] / "
        f"[yellow]{metrics.open_bets} open[/yellow]"
        + (f" / [dim]{metrics.void_bets} void[/dim]" if metrics.void_bets else "")
        + ")",
        f"Win Rate: {metrics.win_rate * 100:.1f}%  |  Wagered: ${metrics.total_wagered_usd:,.2f}",
        "",
        f"Max Drawdown: [red]{metrics.max_drawdown_pct:.1f}%[/red]",
        f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "Sharpe Ratio: N/A (<5 settled bets)",
        "",
        f"Best Win: [green]${metrics.best_win_usd:+.2f}[/green]  "
        f"Worst Loss: [red]${metrics.worst_loss_usd:.2f}[/red]",
    ]

    # Model calibration block (only shown once enough settled bets exist)
    if metrics.avg_true_prob is not None and metrics.calibration_error is not None:
        cal_err = metrics.calibration_error
        cal_style = "green" if cal_err < 0.05 else "yellow" if cal_err < 0.10 else "red"
        lines += [
            "",
            "[bold]Model Calibration[/bold]",
            f"  Avg predicted prob: [cyan]{metrics.avg_true_prob * 100:.1f}%[/cyan]  "
            f"Actual win rate: [cyan]{metrics.win_rate * 100:.1f}%[/cyan]",
            f"  Calibration error: [{cal_style}]{cal_err * 100:.1f}pp[/{cal_style}]"
            + ("  [green](well-calibrated)[/green]" if cal_err < 0.05 else
               "  [yellow](slight overconfidence)[/yellow]" if metrics.avg_true_prob > metrics.win_rate else
               "  [yellow](slight underconfidence)[/yellow]"),
        ]

    content = "\n".join(lines)
    return Panel(content, title="[bold]Performance Summary[/bold]", border_style="cyan")


def print_ev_bets(ev_bets: list[EVBet]) -> None:
    """Print EV bets table to console."""
    if not ev_bets:
        console.print("[yellow]No +EV opportunities found.[/yellow]")
        return
    console.print(ev_table(ev_bets))
    console.print(f"[dim]Found {len(ev_bets)} opportunity(ies)[/dim]")


def print_scan_header(sectors: list[str]) -> None:
    """Print scan header."""
    console.print(
        Panel(
            f"Scanning sectors: [cyan]{', '.join(s.upper() for s in sectors)}[/cyan]",
            title="[bold green]evmax[/bold green]",
            border_style="green",
        )
    )
