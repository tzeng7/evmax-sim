"""CLI commands for cross-venue arbitrage scanning.

Subcommands:
  evmax arb scan   — fetch live Kalshi + Polymarket US books, find baskets
                     that cover every outcome of an event market for less
                     than $1.00 after taker fees. Read-only: nothing is
                     persisted and no orders are placed.

Steady-state arbs between the venues essentially don't exist (combined vig
runs ~1pp against ~1.75pp+1.5pp of round-trip taker fees), so treat any hit
as a transient dislocation to verify by hand — the asks shown are REST
snapshots without depth confirmation.
"""

from __future__ import annotations

import asyncio

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from evmax.arb import ArbOpportunity, fetch_arb_markets, find_arbs
from evmax.clients.polymarket_us import POLYMARKET_US_LEAGUE_MAP

app = typer.Typer(help="Cross-venue (Kalshi vs Polymarket US) arbitrage scanner.")
console = Console()


def _fmt_legs(arb: ArbOpportunity) -> str:
    parts = []
    for leg in arb.legs:
        venue = "K" if leg.venue == "kalshi" else "P"
        parts.append(f"{venue}:{leg.side} {leg.outcome} @{leg.ask:.2f}")
    return " + ".join(parts)


@app.command("scan")
def scan(
    sectors: str = typer.Option(
        "",
        "--sectors",
        "-s",
        help="Comma-separated sectors (default: every sector both venues carry).",
    ),
    max_cost: float = typer.Option(
        1.02,
        "--max-cost",
        help="Show baskets whose fee-inclusive cost is at most this (1.0 = true arbs only; the default keeps 2pp near-misses visible).",
    ),
    include_in_play: bool = typer.Option(
        False,
        "--include-in-play",
        help="Include games that have already started (quotes go stale fast in-play; expect false positives).",
    ),
    notify: bool = typer.Option(
        False,
        "--notify",
        help="Send net-positive arbs to the configured Slack/Discord webhooks.",
    ),
) -> None:
    """Scan both venues for complete-outcome baskets costing under $1."""
    sector_list = (
        [s.strip().lower() for s in sectors.split(",") if s.strip()]
        if sectors
        else list(POLYMARKET_US_LEAGUE_MAP.keys())
    )
    unknown = [s for s in sector_list if s not in POLYMARKET_US_LEAGUE_MAP]
    if unknown:
        console.print(
            f"[yellow]No Polymarket US product for: {', '.join(unknown)} — "
            "scanning them Kalshi-only (self-arb baskets only).[/yellow]"
        )

    with console.status("Fetching Kalshi + Polymarket US books..."):
        pools = asyncio.run(fetch_arb_markets(sector_list))

    all_arbs: list[ArbOpportunity] = []
    for sector, pool in pools.items():
        n_k = sum(1 for m in pool if m.source.value == "kalshi")
        n_p = len(pool) - n_k
        arbs = find_arbs(
            pool, max_net_cost=max_cost, include_in_play=include_in_play
        )
        all_arbs.extend(arbs)
        console.print(
            f"[dim]{sector}: kalshi={n_k} polyus={n_p} baskets≤{max_cost:.2f}: {len(arbs)}[/dim]"
        )

    if not all_arbs:
        console.print("\n[green]No baskets under the cost ceiling.[/green]")
        return

    all_arbs.sort(key=lambda a: a.net_cost)

    table = Table(box=box.SIMPLE_HEAVY, title="Cross-venue arb baskets")
    table.add_column("Event", no_wrap=False, min_width=24)
    table.add_column("Outcome", no_wrap=False)
    table.add_column("Legs", no_wrap=False)
    table.add_column("Gross", justify="right")
    table.add_column("Net", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("Start (UTC)")
    table.add_column("MinVol$", justify="right")

    for arb in all_arbs:
        edge = arb.net_edge
        style = "bold green" if edge > 0 else ("yellow" if arb.gross_cost < 1.0 else "dim")
        table.add_row(
            f"[{arb.sector}] {arb.event_title}",
            arb.market_desc,
            _fmt_legs(arb),
            f"{arb.gross_cost:.3f}",
            f"{arb.net_cost:.3f}",
            f"{edge*100:+.2f}pp",
            arb.event_date.strftime("%m-%d %H:%M") if arb.event_date else "?",
            f"{min(leg.volume_usd for leg in arb.legs):.0f}",
            style=style,
        )
    console.print(table)
    console.print(
        "[dim]Asks are REST snapshots; verify depth/price live before acting. "
        "Fees: Kalshi taker 0.07·p·(1−p), PolyUS taker 0.06·p·(1−p).[/dim]"
    )

    positives = [a for a in all_arbs if a.net_edge > 0]
    if notify and positives:
        from evmax.notifications import Notifier

        lines = [f"*evmax arb* — {len(positives)} net-positive basket(s)"]
        for a in positives[:10]:
            lines.append(
                f"• [{a.sector.upper()}] {a.event_title} {a.market_desc} — "
                f"cost {a.net_cost:.3f}, edge {a.net_edge*100:+.2f}pp: {_fmt_legs(a)}"
            )
        Notifier.from_settings().send_text("\n".join(lines))
        console.print(f"[green]Notified {len(positives)} net-positive basket(s).[/green]")
