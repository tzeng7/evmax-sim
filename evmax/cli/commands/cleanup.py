"""CLI commands for prediction logging, outcome resolution, and model calibration.

Commands:
  evmax cleanup show     — show logged +EV bets and their outcomes
  evmax cleanup resolve  — fetch actual outcomes for a given date
  evmax cleanup metrics  — compute Brier scores and display calibration report
  evmax cleanup adjust   — auto-adjust sharp_weight based on Brier scores
  evmax cleanup train    — re-seed model agents from live data (ESPN + bo3.gg)
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("show")
def show(
    days: int = typer.Option(7, "--days", "-d", help="How many days back to display."),
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Filter by sector."),
    resolved_only: bool = typer.Option(False, "--resolved", help="Only show resolved bets."),
) -> None:
    """Show logged +EV predictions and their WIN/LOSS outcomes."""
    from evmax.agents.cleanup.db import get_connection

    since = (date.today() - timedelta(days=days)).isoformat()
    conn = get_connection()

    where_parts = ["p.scan_date >= ?"]
    params: list = [since]
    if sector:
        where_parts.append("p.sector = ?")
        params.append(sector.lower())
    if resolved_only:
        where_parts.append("o.outcome IS NOT NULL")

    where = " AND ".join(where_parts)
    rows = conn.execute(
        f"""SELECT p.scan_date, p.sector, p.yes_team, p.event_title,
                   p.kalshi_yes_price, p.blended_true_prob, p.ev_pct,
                   p.model_sources, p.sharp_weight_used,
                   o.outcome, o.result_source
            FROM ev_predictions p
            LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE {where}
            ORDER BY p.scan_date DESC, p.ev_pct DESC
            LIMIT 200""",
        params,
    ).fetchall()
    conn.close()

    if not rows:
        console.print(f"[yellow]No predictions in the last {days} day(s).[/yellow]")
        return

    pending = sum(1 for r in rows if r["outcome"] is None)
    wins    = sum(1 for r in rows if r["outcome"] == 1)
    losses  = sum(1 for r in rows if r["outcome"] == 0)

    table = Table(
        title=f"+EV Predictions (last {days}d) | {wins}W / {losses}L / {pending} pending",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("Date", style="dim", width=10)
    table.add_column("Sector", style="dim", width=6)
    table.add_column("Bet", min_width=20)
    table.add_column("Kalshi", justify="right", width=7)
    table.add_column("TrueP", justify="right", width=7)
    table.add_column("EV%", justify="right", width=7)
    table.add_column("SW", justify="right", width=5, style="dim")
    table.add_column("Sources", style="dim", width=12)
    table.add_column("Result", justify="center", width=8)

    for r in rows:
        ev_color = "green" if r["ev_pct"] >= 0.05 else "yellow"
        if r["outcome"] is None:
            result_str = "[dim]pending[/dim]"
        elif r["outcome"] == 1:
            result_str = "[bold green]WIN[/bold green]"
        else:
            result_str = "[bold red]LOSS[/bold red]"

        sw = f"{r['sharp_weight_used']:.2f}" if r["sharp_weight_used"] is not None else ""
        table.add_row(
            r["scan_date"] or "",
            (r["sector"] or "").upper(),
            (r["yes_team"] or "?")[:22],
            f"{r['kalshi_yes_price']:.2f}",
            f"{r['blended_true_prob']:.3f}",
            f"[{ev_color}]{r['ev_pct']*100:+.1f}%[/{ev_color}]",
            sw,
            (r["model_sources"] or "")[:14],
            result_str,
        )

    console.print(table)


@app.command("resolve")
def resolve(
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Date to resolve (YYYY-MM-DD). Defaults to yesterday.",
    ),
) -> None:
    """Fetch actual game outcomes for +EV predictions logged on a date."""
    from evmax.agents.cleanup.resolver import resolve_outcomes_for_date

    if target_date:
        try:
            d = date.fromisoformat(target_date)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {target_date!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        d = date.today() - timedelta(days=1)

    console.print(f"[cyan]Resolving outcomes for[/cyan] {d.isoformat()} ...")
    result = asyncio.run(resolve_outcomes_for_date(d))

    console.print(
        f"  [green]Resolved:[/green] {result['resolved']}  "
        f"[yellow]Unmatched:[/yellow] {result['failed']}"
    )
    if result["resolved"] == 0:
        console.print(
            "  [dim]Tip: run [bold]evmax cleanup show[/bold] to check logged bets, "
            "or try [bold]--date YYYY-MM-DD[/bold] to target a specific game date.[/dim]"
        )


@app.command("metrics")
def metrics(
    weeks: int = typer.Option(1, "--weeks", "-w", help="Look-back window in weeks."),
) -> None:
    """Show Brier score calibration report for logged predictions."""
    from evmax.agents.cleanup.metrics import compute_brier_scores, load_config

    cfg = load_config()
    scores = compute_brier_scores(weeks=weeks)

    if scores is None:
        console.print(
            f"[yellow]No resolved predictions in the last {weeks} week(s).[/yellow]"
        )
        console.print(
            "  Run [bold]evmax cleanup resolve[/bold] first to fetch outcomes."
        )
        return

    bm = scores["brier_model"]
    bs = scores["brier_sharp"]
    improvement = (bs - bm) / bs * 100 if bs > 0 else 0.0

    table = Table(title="Brier Score Report", box=box.SIMPLE)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Period", f"{scores['period_start']} → {scores['period_end']}")
    table.add_row("Predictions resolved", str(scores["n"]))
    table.add_row("Brier — model blend", f"{bm:.5f}")
    table.add_row("Brier — sharp only", f"{bs:.5f}")
    table.add_row(
        "Model vs Sharp",
        (f"[green]+{improvement:.1f}% (model better)[/green]"
         if improvement > 0
         else f"[red]{improvement:.1f}% (model worse)[/red]"),
    )
    table.add_row("Current sharp_weight", f"{cfg['sharp_weight']:.2f}")
    table.add_row("Last adjusted", cfg.get("last_adjusted") or "never")

    console.print(table)

    if scores["n"] < 30:
        console.print(
            f"  [dim]Need 30+ resolved bets to trigger auto-adjustment (have {scores['n']}).[/dim]"
        )


@app.command("adjust")
def adjust(
    force: bool = typer.Option(
        False, "--force", help="Override 7-day cooldown."
    ),
) -> None:
    """Auto-adjust sharp_weight based on Brier score comparison."""
    from evmax.agents.cleanup.metrics import adjust_sharp_weight

    result = adjust_sharp_weight(force=force)

    if not result["adjusted"]:
        console.print(f"[yellow]No adjustment:[/yellow] {result.get('reason', '')}")
        console.print(f"  sharp_weight remains [bold]{result['sharp_weight']:.2f}[/bold]")
        return

    console.print(
        f"[green]sharp_weight adjusted[/green]  "
        f"Brier model={result['brier_model']:.5f} vs sharp={result['brier_sharp']:.5f} "
        f"({result['improvement_pct']:+.1f}% improvement)"
    )
    console.print(
        f"  {result['direction']}  →  "
        f"[bold cyan]sharp_weight = {result['sharp_weight']:.2f}[/bold cyan]"
    )
    console.print(
        "  [dim]Run [bold]evmax agents scan[/bold] — it will pick up the new weight automatically.[/dim]"
    )


@app.command("train")
def train(
    sectors: str = typer.Option(
        "lol,cs2,valorant",
        "--sectors", "-s",
        help="Comma-separated sectors to re-seed (lol, cs2, valorant, nba, nfl, ncaab, soccer).",
    ),
    since: str = typer.Option(
        "2025-01-01", "--since", help="Seed from this date (YYYY-MM-DD)."
    ),
) -> None:
    """Re-seed statistical models from live data (ESPN + bo3.gg)."""
    sector_list = [s.strip().lower() for s in sectors.split(",") if s.strip()]
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"

    esports = [s for s in sector_list if s in ("lol", "cs2", "valorant")]
    sports   = [s for s in sector_list if s in ("nba", "nfl", "ncaab", "soccer")]

    if esports:
        console.print(f"\n[cyan]Seeding esports:[/cyan] {', '.join(esports)}")
        result = subprocess.run(
            [sys.executable, str(scripts_dir / "seed_esports.py"),
             "--sectors", ",".join(esports), "--since", since],
        )
        if result.returncode != 0:
            console.print("[red]Esports seeding encountered errors.[/red]")

    if sports:
        console.print(f"\n[cyan]Seeding sports:[/cyan] {', '.join(sports)}")
        seed_espn = scripts_dir / "seed_espn.py"
        if not seed_espn.exists():
            console.print(f"[yellow]seed_espn.py not found at {seed_espn} — skipping.[/yellow]")
        else:
            result = subprocess.run(
                [sys.executable, str(seed_espn),
                 "--sectors", ",".join(sports), "--since", since],
            )
            if result.returncode != 0:
                console.print("[red]Sports seeding encountered errors.[/red]")

    console.print("\n[green]Training complete.[/green]")
    console.print(
        "  [dim]Run [bold]evmax cleanup adjust[/bold] to update sharp_weight if enough data.[/dim]"
    )
