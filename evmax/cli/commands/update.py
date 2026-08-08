"""evmax update commands — fetch ESPN scores and update model ratings.

Usage:
  evmax update scores [--date YYYY-MM-DD] [--sectors ...] [--dry-run]
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from evmax.agents.cleanup.model_updater import ESPN_MODEL_UPDATE_SECTORS

app = typer.Typer(no_args_is_help=True)
console = Console()

# Default --sectors: the one canonical ESPN-fed sector list, shared with the
# `evmax cleanup resolve` model-update hook so the two can't drift apart.
_DEFAULT_SECTORS = ",".join(ESPN_MODEL_UPDATE_SECTORS)


@app.command("scores")
def update_scores(
    target_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Date to fetch scores for (YYYY-MM-DD). Defaults to yesterday.",
    ),
    sectors: str = typer.Option(
        _DEFAULT_SECTORS,
        "--sectors",
        "-s",
        help=(
            "Comma-separated sectors to update. ESPN-supported: "
            "soccer, worldcup, nba, wnba, nfl, ncaab, ncaaw, ncaaf, nhl, baseball."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be updated without actually updating models.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Re-apply games already recorded in the applied-games ledger. "
            "Only for a deliberate re-derivation onto state that does not "
            "already contain them — otherwise this double-counts results."
        ),
    ),
) -> None:
    """Fetch ESPN completed scores and feed results into Elo/Form/Poisson models."""
    if target_date:
        try:
            fetch_date = date.fromisoformat(target_date)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {target_date!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        fetch_date = date.today() - timedelta(days=1)

    sector_list = [s.strip().lower() for s in sectors.split(",") if s.strip()]

    console.print(
        f"\n[bold cyan]evmax update scores[/bold cyan] — "
        f"date: {fetch_date}  sectors: {', '.join(sector_list)}"
        + ("  [yellow](dry run)[/yellow]" if dry_run else "")
        + ("  [red](force)[/red]" if force else "")
        + "\n"
    )

    asyncio.run(_run_update(sector_list, fetch_date, dry_run, force))


async def _run_update(
    sector_list: list[str],
    fetch_date: date,
    dry_run: bool,
    force: bool = False,
) -> None:
    """Thin CLI wrapper: run the shared updater, render its results as a table.

    All fetch/match/dedup/state-mutation logic lives in
    `evmax.agents.cleanup.model_updater`; this layer owns presentation only.
    """
    from evmax.agents.cleanup.model_updater import update_models_for_date

    table = Table(
        title=f"Model Update Results — {fetch_date}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Sector", style="dim", width=7)
    table.add_column("Home", min_width=20)
    table.add_column("Away", min_width=20)
    table.add_column("Score", justify="center", width=9)
    table.add_column("Status", justify="center", width=12)

    result = await update_models_for_date(
        sector_list, fetch_date, dry_run=dry_run, force=force
    )

    sectors_with_scores = {g.sector for g in result.games}
    for sector in sector_list:
        if sector not in sectors_with_scores:
            console.print(f"[dim]  {sector.upper()}: no ESPN scores found[/dim]")

    for game in result.games:
        if game.error:
            console.print(
                f"[yellow]  Warning: update failed for "
                f"{game.team_a} vs {game.team_b}: {game.error}[/yellow]"
            )
            status_str = "[red]failed[/red]"
        elif game.already_applied:
            status_str = "[dim]already fed[/dim]"
        elif dry_run:
            status_str = "[yellow]dry run[/yellow]"
        else:
            status_str = "[green]updated[/green]"

        table.add_row(
            game.sector.upper(),
            game.home_name[:22],
            game.away_name[:22],
            f"{game.score_a}–{game.score_b}",
            status_str,
        )

    console.print(table)
    if not dry_run:
        console.print(
            f"\n[green]Updated {result.updated} game(s) across "
            f"{len(sector_list)} sector(s).[/green]"
        )
    else:
        console.print(f"\n[yellow]Dry run — no models updated.[/yellow]")
    if result.skipped:
        console.print(
            f"[dim]Skipped {result.skipped} game(s) already fed into model "
            f"state (use --force to re-apply).[/dim]"
        )
