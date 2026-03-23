"""evmax CLI root application."""

from __future__ import annotations

import logging

import structlog
import typer
from rich.console import Console

from evmax.cli.commands.scan import app as scan_app
from evmax.cli.commands.sim import app as sim_app
from evmax.cli.commands.report import app as report_app
from evmax.cli.commands.opportunities import app as opps_app
from evmax.cli.commands.agents import app as agents_app
from evmax.cli.commands.cleanup import app as cleanup_app
from evmax.cli.commands.backtest import app as backtest_app
from evmax.cli.commands.archive import app as archive_app
from evmax.cli.commands.update import app as update_app

console = Console()

app = typer.Typer(
    name="evmax",
    help="[bold green]evmax[/bold green] — EV maximizer for Kalshi/Polymarket prediction markets.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Register sub-apps
app.add_typer(scan_app, name="scan", help="Scan markets for +EV opportunities.")
app.add_typer(sim_app, name="sim", help="Manage simulated bets.")
app.add_typer(report_app, name="report", help="View performance reports.")
app.add_typer(opps_app, name="opps", help="View pre-game + live +EV opportunities and place bets.")
app.add_typer(agents_app, name="agents", help="Agent pipeline: odds checkers + statistical model agents.")
app.add_typer(cleanup_app, name="cleanup", help="Log +EV bets, resolve outcomes, compute Brier scores, adjust sharp_weight.")
app.add_typer(backtest_app, name="backtest", help="Historical model calibration backtest (soccer + tennis).")
app.add_typer(archive_app, name="archive", help="Manage historical data archive (Pinnacle + Kalshi snapshots).")
app.add_typer(update_app, name="update", help="Fetch ESPN scores and update Elo/Form/Poisson models.")


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
