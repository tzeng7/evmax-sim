"""evmax CLI root application."""

from __future__ import annotations

import logging

import structlog
import typer
from rich.console import Console

from evmax.cli.commands.sim import app as sim_app
from evmax.cli.commands.report import app as report_app
from evmax.cli.commands.agents import app as agents_app
from evmax.cli.commands.cleanup import app as cleanup_app
from evmax.cli.commands.backtest import app as backtest_app
from evmax.cli.commands.archive import app as archive_app
from evmax.cli.commands.update import app as update_app

console = Console()

app = typer.Typer(
    name="evmax",
    help=(
        "[bold green]evmax[/bold green] — EV maximizer for Kalshi prediction markets.\n\n"
        "[bold]Standard pipeline:[/bold]\n"
        "  1. [cyan]evmax agents scan[/cyan]           — find +EV plays (logs automatically)\n"
        "  2. [cyan]evmax agents pick[/cyan]           — confirm which bets you're placing\n"
        "  3. [cyan]evmax update scores[/cyan]         — next morning: update models from ESPN\n"
        "  4. [cyan]evmax cleanup resolve --date[/cyan] — resolve yesterday's bets via ESPN\n"
        "  5. [cyan]evmax cleanup show[/cyan]          — view P&L and outcomes\n"
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# ── Core pipeline ──────────────────────────────────────────────────────────────
app.add_typer(agents_app,  name="agents",  help="[PRIMARY] Scan, pick, and manage model agents.")
app.add_typer(cleanup_app, name="cleanup", help="[PRIMARY] Resolve outcomes, view P&L, calibrate models.")
app.add_typer(update_app,  name="update",  help="Fetch ESPN scores and update Elo/Form/Poisson ratings.")

# ── Simulation / analytics ─────────────────────────────────────────────────────
app.add_typer(sim_app,     name="sim",     help="Monte Carlo simulation and paper-bet management.")
app.add_typer(report_app,  name="report",  help="P&L report from simulation database.")
app.add_typer(backtest_app, name="backtest", help="Historical model calibration backtest.")
app.add_typer(archive_app, name="archive", help="Manage historical Pinnacle + Kalshi snapshots.")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """evmax — +EV prediction market betting system."""
    level = "DEBUG" if verbose else "INFO"
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level)
        ),
    )


if __name__ == "__main__":
    app()
