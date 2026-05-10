"""CLI commands for prediction logging, outcome resolution, and model calibration.

Commands:
  evmax cleanup show          — show logged +EV bets and their outcomes
  evmax cleanup resolve       — fetch actual outcomes for a given date
  evmax cleanup metrics       — compute Brier scores and display calibration report
  evmax cleanup adjust        — auto-adjust sharp_weight based on Brier scores
  evmax cleanup train         — re-seed model agents from live data (ESPN + bo3.gg)
  evmax cleanup props         — show logged prop observations and outcomes
  evmax cleanup resolve-props — fetch ESPN boxscores and fill prop_observations outcomes
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(no_args_is_help=True)
console = Console()

# ARCH-11 shadow-mode subcommand group: `evmax cleanup shadow show/metrics/promote`
from evmax.cli.commands.shadow import app as shadow_app  # noqa: E402

app.add_typer(
    shadow_app,
    name="shadow",
    help="Inspect shadow-mode predictions (show / metrics) and promote categories to live.",
)


@app.command("show")
def show(
    since: Optional[str] = typer.Option(None, "--since", help="Start date YYYY-MM-DD (default: 7 days ago)."),
    until: Optional[str] = typer.Option(None, "--until", help="End date YYYY-MM-DD (default: today)."),
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Filter by sector."),
    resolved_only: bool = typer.Option(False, "--resolved", help="Only show resolved bets."),
    placed_only: bool = typer.Option(False, "--placed", help="Only show bets you manually placed via 'pick'."),
    bankroll: float = typer.Option(250.0, "--bankroll", "-b", help="Bankroll in USD for P&L calculation."),
    game_date: Optional[str] = typer.Option(None, "--date", help="Filter by exact game date (YYYY-MM-DD)."),
) -> None:
    """Show logged +EV predictions and their WIN/LOSS outcomes."""
    from evmax.agents.cleanup.db import get_connection

    for label, val in [("since", since), ("until", until), ("date", game_date)]:
        if val:
            try:
                date.fromisoformat(val)
            except ValueError:
                console.print(f"[red]Invalid {label}:[/red] {val!r} — use YYYY-MM-DD")
                raise typer.Exit(1)

    since_date = since or (date.today() - timedelta(days=7)).isoformat()
    until_date = until or date.today().isoformat()
    conn = get_connection()

    where_parts = ["p.scan_date >= ?", "p.scan_date <= ?", "p.voided = 0", "p.mode = 'live'"]
    params: list = [since_date, until_date]
    if game_date:
        where_parts.append("p.event_date = ?")
        params.append(game_date)
    if sector:
        where_parts.append("p.sector = ?")
        params.append(sector.lower())
    if resolved_only:
        where_parts.append("o.outcome IS NOT NULL")
    if placed_only:
        where_parts.append("p.placed = 1")

    where = " AND ".join(where_parts)
    rows = conn.execute(
        f"""SELECT p.scan_date, p.event_date, p.sector, p.yes_team, p.event_title,
                   p.market_type, p.line,
                   p.kalshi_yes_price, p.sharp_true_prob, p.blended_true_prob, p.ev_pct,
                   p.kelly_fraction, p.volume_usd, p.model_sources,
                   p.placed, p.placed_price, p.placed_stake,
                   o.outcome, o.pinnacle_close_prob
            FROM ev_predictions p
            INNER JOIN (
                SELECT market_id,
                       COALESCE(
                           MAX(CASE WHEN placed = 1 THEN scan_date END),
                           MAX(scan_date)
                       ) AS latest_scan
                FROM ev_predictions
                WHERE voided = 0 AND mode = 'live'
                GROUP BY market_id
            ) latest ON p.market_id = latest.market_id
                    AND p.scan_date = latest.latest_scan
            LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE {where}
            ORDER BY p.event_date DESC, p.ev_pct DESC
            LIMIT 200""",
        params,
    ).fetchall()
    conn.close()

    if not rows:
        msg = "No placed bets" if placed_only else "No predictions"
        console.print(f"[yellow]{msg} for {since_date} → {until_date}.[/yellow]")
        return

    pending = sum(1 for r in rows if r["outcome"] is None)
    wins    = sum(1 for r in rows if r["outcome"] == 1)
    losses  = sum(1 for r in rows if r["outcome"] == 0)

    # P&L split: real (placed) vs simulated (all unpicked resolved bets)
    real_staked = 0.0
    real_pnl    = 0.0
    sim_staked  = 0.0
    sim_pnl     = 0.0
    for r in rows:
        if r["outcome"] is None:
            continue
        price = r["placed_price"] if r["placed_price"] else r["kalshi_yes_price"]
        if r["placed"] and r["placed_stake"]:
            stake = r["placed_stake"]
            real_staked += stake
            real_pnl += stake * (1.0 / price - 1.0) if r["outcome"] == 1 else -stake
        else:
            stake = bankroll * (r["kelly_fraction"] or 0.0)
            sim_staked += stake
            sim_pnl += stake * (1.0 / price - 1.0) if r["outcome"] == 1 else -stake

    total_pnl    = real_pnl + sim_pnl
    total_staked = real_staked + sim_staked

    pnl_color = "green" if total_pnl >= 0 else "red"
    pnl_sign  = "+" if total_pnl >= 0 else ""

    title_date = f"game date {game_date}" if game_date else f"{since_date} → {until_date}"
    placed_note = " | placed only" if placed_only else ""
    placed_total = sum(1 for r in rows if r["placed"])
    table = Table(
        title=(
            f"+EV Predictions ({title_date}{placed_note}) | {wins}W / {losses}L / {pending} pending"
            f" | Sim P&L [{pnl_color}]{pnl_sign}${total_pnl:.2f}[/{pnl_color}]"
            + (f" | {placed_total} placed" if not placed_only and placed_total else "")
        ),
        box=box.SIMPLE,
        show_lines=False,
    )
    table.add_column("Date",    style="dim", width=10)
    table.add_column("Sect",    style="dim", width=5)
    table.add_column("Event",   style="dim", min_width=18, no_wrap=False)
    table.add_column("Outcome", style="bold white", min_width=12, no_wrap=False)
    table.add_column("Kalshi",  justify="right", width=6)
    table.add_column("TrueP",   justify="right", width=6)
    table.add_column("CLV",     justify="right", width=7)
    table.add_column("EV%",     justify="right", width=6)
    table.add_column("Stake$",  justify="right", width=7)
    table.add_column("P&L$",    justify="right", width=8)
    table.add_column("Src",     style="dim", width=10)
    table.add_column("Result",  justify="center", width=7)

    def _display_label(yes_team: str, market_type: str, line) -> str:
        team = (yes_team or "?").capitalize()
        if market_type == "moneyline":
            return f"{team} ML"
        if market_type == "spread" and line is not None:
            line_str = f"{line:.1f}".rstrip("0").rstrip(".")
            return f"{team} {line_str}"
        if market_type in ("over_under", "total") and line is not None:
            return f"O/U {line:.1f}"
        return team

    for i, r in enumerate(rows, 1):
        ev_color = "bold green" if r["ev_pct"] >= 0.10 else "green" if r["ev_pct"] >= 0.05 else "yellow"
        # Use actual placed stake/price if available, else estimated
        stake = r["placed_stake"] if r["placed_stake"] else bankroll * (r["kelly_fraction"] or 0.0)
        price = r["placed_price"] if r["placed_price"] else r["kalshi_yes_price"]
        placed_marker = " [cyan]●[/cyan]" if r["placed"] else ""

        is_sim = not r["placed"]
        if r["outcome"] is None:
            result_str = "[dim]pending[/dim]"
            pnl_str = "[dim]—[/dim]"
        elif r["outcome"] == 1:
            result_str = "[bold green]WIN[/bold green]" + ("[dim] sim[/dim]" if is_sim else "")
            profit = stake * (1.0 / price - 1.0)
            pnl_str = f"[green]+${profit:.2f}[/green]" + ("[dim]*[/dim]" if is_sim else "")
        else:
            result_str = "[bold red]LOSS[/bold red]" + ("[dim] sim[/dim]" if is_sim else "")
            pnl_str = f"[red]-${stake:.2f}[/red]" + ("[dim]*[/dim]" if is_sim else "")

        # CLV = entry sharp prob - Pinnacle closing prob (positive = bought value)
        close_prob = r["pinnacle_close_prob"]
        entry_prob = r["sharp_true_prob"]
        if close_prob is not None and entry_prob is not None:
            clv = entry_prob - close_prob
            clv_color = "green" if clv >= 0.01 else "red" if clv <= -0.01 else "dim"
            clv_str = f"[{clv_color}]{clv*100:+.1f}pp[/{clv_color}]"
        else:
            clv_str = "[dim]—[/dim]"

        table.add_row(
            r["event_date"] or r["scan_date"] or "",
            (r["sector"] or "").upper(),
            (r["event_title"] or "")[:24],
            _display_label(r["yes_team"], r["market_type"] or "moneyline", r["line"]) + placed_marker,
            f"{price:.2f}",
            f"{r['blended_true_prob']:.3f}",
            clv_str,
            f"[{ev_color}]{r['ev_pct']*100:+.1f}%[/{ev_color}]",
            f"${stake:.2f}",
            pnl_str,
            (r["model_sources"] or "")[:14],
            result_str,
        )

    console.print(table)

    # Footer: split real vs simulated
    if real_staked > 0:
        real_color = "green" if real_pnl >= 0 else "red"
        real_sign  = "+" if real_pnl >= 0 else ""
        real_roi   = real_pnl / real_staked * 100 if real_staked else 0.0
        current_bankroll = bankroll + real_pnl
        console.print(
            f"  [bold]Placed bets[/bold]  [{real_color}]{real_sign}${real_pnl:.2f}[/{real_color}]"
            f"  |  Staked: ${real_staked:.2f}"
            f"  |  ROI: [{real_color}]{real_sign}{real_roi:.1f}%[/{real_color}]"
            f"  |  Bankroll: ${bankroll:.0f} → [bold {real_color}]${current_bankroll:.2f}[/bold {real_color}]"
        )

    if sim_staked > 0:
        sim_color = "green" if sim_pnl >= 0 else "red"
        sim_sign  = "+" if sim_pnl >= 0 else ""
        sim_roi   = sim_pnl / sim_staked * 100 if sim_staked else 0.0
        console.print(
            f"  [bold]Simulated (all)[/bold]  [{sim_color}]{sim_sign}${sim_pnl:.2f}[/{sim_color}]"
            f"  |  Staked: ${sim_staked:.2f}"
            f"  |  ROI: [{sim_color}]{sim_sign}{sim_roi:.1f}%[/{sim_color}]"
            f"  [dim](hypothetical Kelly stakes — not real money)[/dim]"
        )

    if real_staked == 0 and sim_staked == 0:
        console.print(f"  [dim]No resolved bets in this period yet.[/dim]")
    print()


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

    try:
        from evmax.agents.cleanup.resolver import backfill_clv
        clv = backfill_clv()
        if clv["updated"]:
            avg = clv["avg_clv"]
            color = "green" if avg > 0 else "red"
            console.print(
                f"  [green]CLV backfilled:[/green] {clv['updated']} bet(s)  "
                f"avg [{color}]{avg:+.1f}pp[/{color}]  "
                f"[dim]skipped: {clv['skipped']} (no archived close)[/dim]"
            )
    except Exception as _clv_err:
        console.print(f"  [yellow]Warning: CLV backfill failed:[/yellow] {_clv_err}")
    unmatched = result.get("unmatched", [])
    if unmatched:
        console.print(f"\n  [yellow]Unmatched event IDs ({len(unmatched)}):[/yellow]")
        for eid in unmatched[:30]:
            console.print(f"    [dim]{eid}[/dim]")
        if len(unmatched) > 30:
            console.print(f"    [dim]... and {len(unmatched) - 30} more[/dim]")
    if result["resolved"] == 0 and not unmatched:
        console.print(
            "  [dim]Tip: run [bold]evmax cleanup show[/bold] to check logged bets, "
            "or try [bold]--date YYYY-MM-DD[/bold] to target a specific game date.[/dim]"
        )


@app.command("close-lines")
def close_lines(
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Date to capture closing lines for (YYYY-MM-DD). Defaults to today.",
    ),
) -> None:
    """Capture Pinnacle closing lines for unresolved markets.

    Run this at or just before game start time. Stores pinnacle_close_prob in
    ev_outcomes so CLV (entry prob vs closing line) can be computed later.

    CLV = sharp_true_prob (at scan time) - pinnacle_close_prob (at close).
    Positive CLV means you got better odds than the sharpest book offered at close.
    """
    import asyncio
    from evmax.agents.cleanup.db import get_connection
    from evmax.clients.esports_pinnacle import PinnacleGuestClient

    if target_date:
        try:
            d = date.fromisoformat(target_date)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {target_date!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        d = date.today()

    date_str = d.isoformat()
    conn = get_connection()

    # Get unresolved markets for this date that don't yet have a closing line
    rows = conn.execute(
        """SELECT o.market_id, o.event_id, o.sector, p.sharp_true_prob
           FROM ev_outcomes o
           JOIN ev_predictions p ON o.market_id = p.market_id
           WHERE o.event_date = ?
             AND o.outcome IS NULL
             AND o.pinnacle_close_prob IS NULL""",
        (date_str,),
    ).fetchall()

    if not rows:
        console.print(f"[yellow]No unresolved markets without closing lines for {date_str}.[/yellow]")
        conn.close()
        return

    console.print(f"[cyan]Capturing closing lines for {len(rows)} market(s) on {date_str}...[/cyan]")

    # Group by sector to batch Pinnacle fetches
    by_sector: dict[str, list] = {}
    for r in rows:
        by_sector.setdefault(r["sector"], []).append(dict(r))

    async def _fetch_and_store() -> int:
        updated = 0
        async with PinnacleGuestClient() as client:
            for sector, markets in by_sector.items():
                sharp_odds = await client.get_odds(sector)
                sharp_by_event: dict[str, float] = {}
                for so in sharp_odds:
                    # Store true_prob_a as the closing prob reference (YES-aligned per entry)
                    sharp_by_event[so.event_id] = so.true_prob_a

                for m in markets:
                    close_prob = sharp_by_event.get(m["event_id"])
                    if close_prob is None:
                        continue
                    conn.execute(
                        "UPDATE ev_outcomes SET pinnacle_close_prob = ? WHERE market_id = ?",
                        (close_prob, m["market_id"]),
                    )
                    updated += 1

        conn.commit()
        return updated

    updated = asyncio.run(_fetch_and_store())
    conn.close()

    console.print(f"  [green]Stored closing lines for {updated}/{len(rows)} market(s).[/green]")
    if updated < len(rows):
        console.print(
            f"  [dim]{len(rows) - updated} market(s) had no matching Pinnacle event "
            f"(may have already started or Pinnacle delisted).[/dim]"
        )


@app.command("watch-closes")
def watch_closes(
    lookahead: int = typer.Option(
        15, "--lookahead", "-l",
        help="Capture Pinnacle close for any event tipping off within the next N minutes.",
    ),
    interval: int = typer.Option(
        300, "--interval", "-i",
        help="Seconds to sleep between sweeps (default 5 min).",
    ),
    once: bool = typer.Option(
        False, "--once",
        help="Run a single sweep and exit (skip the loop).",
    ),
) -> None:
    """Watch upcoming events and capture Pinnacle close ~lookahead min before each tip-off.

    Self-sufficient — pulls each cycle's queue from the DB, so you can scan at any
    time of day and this loop will pick up the new events on the next sweep.

    Recommended setup: run this as a launchd/systemd service that's always up.
    Idempotent: only writes to ev_outcomes rows still missing pinnacle_close_prob.
    """
    import asyncio
    import time
    from evmax.agents.cleanup.db import get_connection
    from evmax.archiver import _get_connection as get_archive_conn
    from evmax.clients.esports_pinnacle import PinnacleGuestClient

    async def _sweep() -> tuple[int, int]:
        # 1. From archive.db: events tipping off in [now, now + lookahead]
        with get_archive_conn() as ac:
            upcoming = ac.execute(
                """SELECT DISTINCT event_id, sector, event_date
                   FROM archived_sharp_odds
                   WHERE event_date IS NOT NULL
                     AND datetime(event_date) >= datetime('now')
                     AND datetime(event_date) <= datetime('now', ?)""",
                (f"+{lookahead} minutes",),
            ).fetchall()

        if not upcoming:
            return (0, 0)

        event_ids = [r["event_id"] for r in upcoming]

        # 2. From predictions.db: which of those still need a close
        conn = get_connection()
        placeholders = ",".join("?" * len(event_ids))
        rows = conn.execute(
            f"""SELECT o.market_id, o.event_id, o.sector
                FROM ev_outcomes o
                WHERE o.event_id IN ({placeholders})
                  AND o.outcome IS NULL
                  AND o.pinnacle_close_prob IS NULL""",
            event_ids,
        ).fetchall()

        if not rows:
            conn.close()
            return (0, 0)

        # 3. Group by sector, fetch Pinnacle once per sector
        by_sector: dict[str, list] = {}
        for r in rows:
            by_sector.setdefault(r["sector"], []).append(dict(r))

        updated = 0
        async with PinnacleGuestClient() as client:
            for sector, markets in by_sector.items():
                try:
                    sharp_odds = await client.get_odds(sector)
                except Exception as fetch_err:
                    console.print(f"[yellow]  [{sector}] fetch failed: {fetch_err}[/yellow]")
                    continue
                close_by_event = {so.event_id: so.true_prob_a for so in sharp_odds}
                for m in markets:
                    cp = close_by_event.get(m["event_id"])
                    if cp is None:
                        continue
                    conn.execute(
                        "UPDATE ev_outcomes SET pinnacle_close_prob = ? WHERE market_id = ?",
                        (cp, m["market_id"]),
                    )
                    updated += 1

        conn.commit()
        conn.close()
        return (updated, len(rows))

    if once:
        captured, queued = asyncio.run(_sweep())
        console.print(f"[green]Captured {captured}/{queued} close(s).[/green]")
        return

    console.print(
        f"[cyan]watch-closes running[/cyan]  lookahead={lookahead}m  interval={interval}s  "
        f"[dim](Ctrl+C to stop)[/dim]"
    )
    try:
        while True:
            ts = datetime.now().strftime("%H:%M:%S")
            try:
                captured, queued = asyncio.run(_sweep())
                if queued:
                    console.print(f"[dim]{ts}[/dim]  captured {captured}/{queued}")
            except Exception as sweep_err:
                console.print(f"[red]{ts}  sweep failed:[/red] {sweep_err}")
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]watch-closes stopped.[/dim]")


@app.command("void")
def void(
    before: Optional[str] = typer.Option(
        None, "--before",
        help="Void predictions with event_date before this date (YYYY-MM-DD). "
             "Defaults to 5 days ago.",
    ),
    stale_days: int = typer.Option(
        5, "--days", "-d",
        help="Void predictions older than this many days with no outcome (used when --before is omitted).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be voided without making changes."
    ),
) -> None:
    """Void stale unmatched predictions (cancelled/postponed games, untraceable markets).

    Voided bets are excluded from P&L and Brier score calculations but kept
    in the database for audit purposes.
    """
    from evmax.agents.cleanup.db import get_connection

    if before:
        try:
            cutoff = date.fromisoformat(before)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {before!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        cutoff = date.today() - timedelta(days=stale_days)

    cutoff_str = cutoff.isoformat()
    conn = get_connection()

    # Find predictions that are: past the cutoff, still unresolved, and not already voided
    candidates = conn.execute(
        """
        SELECT p.market_id, p.event_id, p.sector, p.yes_team,
               p.event_date, p.scan_date, p.ev_pct
        FROM   ev_predictions p
        LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE  p.voided = 0
          AND  o.market_id IS NULL
          AND  COALESCE(p.event_date, p.scan_date) < ?
        ORDER  BY COALESCE(p.event_date, p.scan_date) DESC
        """,
        (cutoff_str,),
    ).fetchall()

    if not candidates:
        console.print(f"[green]No stale unmatched predictions before {cutoff_str}.[/green]")
        conn.close()
        return

    table = Table(
        title=f"{'[DRY RUN] ' if dry_run else ''}Voiding {len(candidates)} stale prediction(s) before {cutoff_str}",
        box=box.SIMPLE,
    )
    table.add_column("Event Date", style="dim", width=10)
    table.add_column("Sector", style="dim", width=6)
    table.add_column("Event ID", style="dim", min_width=32)
    table.add_column("YES Team", style="dim", min_width=14)
    table.add_column("EV%", justify="right", width=6)

    for r in candidates:
        table.add_row(
            r["event_date"] or r["scan_date"] or "",
            (r["sector"] or "").upper(),
            r["event_id"],
            r["yes_team"],
            f"{r['ev_pct'] * 100:+.1f}%",
        )

    console.print(table)

    if dry_run:
        console.print(f"\n  [dim]Dry run — no changes made. Remove --dry-run to apply.[/dim]")
        conn.close()
        return

    market_ids = [r["market_id"] for r in candidates]
    conn.execute(
        f"UPDATE ev_predictions SET voided = 1 WHERE market_id IN ({','.join('?' * len(market_ids))})",
        market_ids,
    )
    conn.commit()
    conn.close()

    console.print(f"\n  [cyan]Voided {len(candidates)} prediction(s).[/cyan] "
                  f"They are excluded from P&L but kept for audit.")


@app.command("metrics")
def metrics(
    weeks: int = typer.Option(1, "--weeks", "-w", help="Look-back window in weeks."),
    by_sector: bool = typer.Option(
        False, "--by-sector", help="Also print a per-sector Brier breakdown."
    ),
) -> None:
    """Show Brier score calibration report for logged predictions."""
    from evmax.agents.cleanup.metrics import (
        compute_brier_scores,
        compute_brier_scores_by_sector,
        load_config,
    )

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

    # Per-tier breakdown
    tiers = scores.get("tiers", [])
    if any(t["n"] > 0 for t in tiers):
        tier_table = Table(title="Brier Score by Probability Tier", box=box.SIMPLE)
        tier_table.add_column("Prob Tier",    style="dim", width=10)
        tier_table.add_column("Bets",         justify="right", width=6)
        tier_table.add_column("Brier Model",  justify="right", width=12)
        tier_table.add_column("Brier Sharp",  justify="right", width=12)
        tier_table.add_column("Model vs Sharp", justify="right", width=16)
        tier_table.add_column("Note", style="dim", min_width=20)

        for t in tiers:
            if t["n"] == 0:
                tier_table.add_row(t["label"], "0", "—", "—", "—", "[dim]insufficient data[/dim]")
                continue
            bm, bs = t["brier_model"], t["brier_sharp"]
            imp = (bs - bm) / bs * 100 if bs and bs > 0 else 0.0
            imp_str = (
                f"[green]+{imp:.1f}%[/green]" if imp > 0
                else f"[red]{imp:.1f}%[/red]"
            )
            note = ""
            if t["label"] == "< 20%" and t["n"] < 50:
                note = "[yellow]low vol — high variance[/yellow]"
            elif t["n"] < 15:
                note = "[dim]< 15 samples[/dim]"
            tier_table.add_row(
                t["label"],
                str(t["n"]),
                f"{bm:.5f}",
                f"{bs:.5f}",
                imp_str,
                note,
            )
        console.print(tier_table)

    if scores["n"] < 30:
        console.print(
            f"  [dim]Need 30+ resolved bets to trigger auto-adjustment (have {scores['n']}).[/dim]"
        )

    if by_sector:
        by = compute_brier_scores_by_sector(weeks=weeks)
        if not by:
            console.print(f"[dim]No per-sector data in the last {weeks} week(s).[/dim]")
            return
        sw_map = cfg.get("sharp_weight_by_sector") or {}
        global_sw = float(cfg.get("sharp_weight", 0.85))

        sec_table = Table(title=f"Brier by Sector — last {weeks}w", box=box.SIMPLE)
        sec_table.add_column("Sector", style="bold")
        sec_table.add_column("N", justify="right")
        sec_table.add_column("Brier Model",  justify="right")
        sec_table.add_column("Brier Sharp",  justify="right")
        sec_table.add_column("Edge",         justify="right")
        sec_table.add_column("sharp_w",      justify="right")
        sec_table.add_column("Suggests",     style="dim")

        for row in by:
            edge = row["edge_pct"]
            edge_str = (
                f"[green]+{edge:.1f}%[/green]" if edge > 0
                else f"[red]{edge:.1f}%[/red]"
            )
            sw = float(sw_map.get(row["sector"], global_sw))
            # Rule of thumb: if model is > 3% better, lower sharp_weight by 0.05;
            # if sharp is > 3% better, raise it by 0.05. Clamped 0.40–0.95.
            if row["n"] < 20:
                suggest = "[dim]insufficient data[/dim]"
            elif edge > 3:
                suggest = f"lower → {max(0.40, sw - 0.05):.2f}"
            elif edge < -3:
                suggest = f"raise → {min(0.95, sw + 0.05):.2f}"
            else:
                suggest = "hold"
            sec_table.add_row(
                row["sector"],
                str(row["n"]),
                f"{row['brier_model']:.5f}",
                f"{row['brier_sharp']:.5f}",
                edge_str,
                f"{sw:.2f}",
                suggest,
            )
        console.print(sec_table)


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


@app.command("dedup-ev")
def dedup_ev(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be deleted without making changes."
    ),
) -> None:
    """Remove duplicate EV scan records, keeping only the latest scan per market.

    Each time the scanner runs it stores a new ev_bets row for every +EV market found.
    This command keeps only the most recent scan per (market_id, outcome) and deletes
    the rest, so evmax report shows clean per-market data without scan-cycle duplicates.
    """
    import asyncio
    from sqlalchemy import text
    from evmax.db import AsyncSessionLocal

    async def _run() -> tuple[int, int]:
        async with AsyncSessionLocal() as session:
            # Count total rows
            total_result = await session.execute(text("SELECT COUNT(*) FROM ev_bets"))
            total = total_result.scalar()

            # Count unique (market_id, outcome) pairs
            unique_result = await session.execute(
                text("SELECT COUNT(*) FROM (SELECT DISTINCT market_id, outcome FROM ev_bets)")
            )
            unique = unique_result.scalar()

            duplicates = total - unique

            if not dry_run and duplicates > 0:
                # Delete all rows except the highest id per (market_id, outcome)
                await session.execute(text("""
                    DELETE FROM ev_bets
                    WHERE id NOT IN (
                        SELECT MAX(id)
                        FROM ev_bets
                        GROUP BY market_id, outcome
                    )
                """))
                await session.commit()

        return total, unique

    total, unique = asyncio.run(_run())
    duplicates = total - unique

    if duplicates == 0:
        console.print("[green]No duplicate EV scan records found.[/green]")
        console.print(f"  {total} rows, all unique (market_id, outcome) pairs.")
        return

    if dry_run:
        console.print(
            f"[yellow][DRY RUN][/yellow] Would delete [bold]{duplicates}[/bold] duplicate scan records "
            f"({total} total → {unique} unique markets)."
        )
        console.print("  Remove [bold]--dry-run[/bold] to apply.")
    else:
        console.print(
            f"[green]Deleted {duplicates} duplicate EV scan records.[/green]  "
            f"{total} rows → {unique} unique markets remaining."
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
    sports   = [s for s in sector_list if s in ("nba", "nfl", "ncaab", "soccer", "baseball", "ufc", "f1")]

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


@app.command("resolve-props")
def resolve_props(
    game_date: Optional[str] = typer.Option(None, "--date", "-d", help="YYYY-MM-DD (default: yesterday)"),
    sectors: str = typer.Option("nba,nfl", "--sectors", "-s", help="Comma-separated sectors to resolve"),
) -> None:
    """Fetch ESPN boxscores and fill in prop_observations actual values + outcomes."""
    from evmax.agents.cleanup.prop_resolver import resolve_prop_observations

    target = date.fromisoformat(game_date) if game_date else date.today() - timedelta(days=1)
    sector_list = [s.strip().lower() for s in sectors.split(",")]

    console.print(f"\n[bold]Resolving prop observations for {target}[/bold]")
    total_resolved = total_unmatched = 0

    for sector in sector_list:
        result = resolve_prop_observations(sector, target)
        resolved = result["resolved"]
        unmatched = result["unmatched"]
        total_resolved += resolved
        total_unmatched += unmatched
        if resolved or unmatched:
            console.print(
                f"  [cyan]{sector.upper()}[/cyan]  resolved={resolved}  unmatched={unmatched}"
            )

    console.print(
        f"\n  [green]Total resolved: {total_resolved}[/green]  "
        f"[yellow]Unmatched: {total_unmatched}[/yellow]"
    )


@app.command("props")
def show_props(
    days: int = typer.Option(14, "--days", "-d", help="How many days back to show"),
    sector: Optional[str] = typer.Option(None, "--sector", "-s"),
    stat_type: Optional[str] = typer.Option(None, "--stat", help="Filter by stat type (points, rebounds, etc.)"),
    player: Optional[str] = typer.Option(None, "--player", "-p", help="Filter by player name (partial)"),
    resolved_only: bool = typer.Option(False, "--resolved", help="Only show resolved observations"),
) -> None:
    """Show logged prop observations and their resolved outcomes."""
    from evmax.agents.cleanup.db import get_connection

    since = (date.today() - timedelta(days=days)).isoformat()
    conn = get_connection()

    where = ["scan_date >= ?", "mode = 'live'"]
    params: list = [since]
    if sector:
        where.append("sector = ?")
        params.append(sector.lower())
    if stat_type:
        where.append("stat_type = ?")
        params.append(stat_type.lower())
    if player:
        where.append("player_name LIKE ?")
        params.append(f"%{player}%")
    if resolved_only:
        where.append("outcome IS NOT NULL")

    rows = conn.execute(
        f"""SELECT scan_date, event_date, sector, player_name, stat_type, line,
                   kalshi_price, sharp_prob, ev_pct, l15_games,
                   actual_value, outcome
            FROM prop_observations
            WHERE {' AND '.join(where)}
            ORDER BY scan_date DESC, player_name""",
        params,
    ).fetchall()

    if not rows:
        console.print(f"[yellow]No prop observations in the last {days} day(s).[/yellow]")
        return

    table = Table(title=f"Prop Observations — last {days} days", box=box.ROUNDED, show_lines=True)
    table.add_column("Date", width=10)
    table.add_column("Sector", width=6)
    table.add_column("Player", min_width=18, no_wrap=False)
    table.add_column("Stat", width=8)
    table.add_column("Line", justify="right", width=6)
    table.add_column("K Prob", justify="right", width=7)
    table.add_column("Sharp", justify="right", width=7)
    table.add_column("EV%", justify="right", width=7)
    table.add_column("L15", justify="right", width=4)
    table.add_column("Actual", justify="right", width=7)
    table.add_column("Result", justify="center", width=7)

    over_count = under_count = unresolved = 0
    for r in rows:
        outcome = r["outcome"]
        if outcome == 1:
            result_str = "[green]OVER[/green]"
            over_count += 1
        elif outcome == 0:
            result_str = "[red]UNDER[/red]"
            under_count += 1
        else:
            result_str = "[dim]—[/dim]"
            unresolved += 1

        ev_color = "green" if (r["ev_pct"] or 0) >= 0.02 else "dim"
        table.add_row(
            r["event_date"] or r["scan_date"] or "—",
            (r["sector"] or "").upper(),
            r["player_name"] or "—",
            r["stat_type"] or "—",
            f"{r['line']:.1f}" if r["line"] else "—",
            f"{r['kalshi_price']:.1%}" if r["kalshi_price"] else "—",
            f"{r['sharp_prob']:.1%}" if r["sharp_prob"] else "—",
            f"[{ev_color}]{(r['ev_pct'] or 0)*100:+.1f}%[/{ev_color}]",
            str(r["l15_games"]) if r["l15_games"] else "—",
            f"{r['actual_value']:.1f}" if r["actual_value"] is not None else "—",
            result_str,
        )

    console.print(table)
    resolved = over_count + under_count
    hit_rate = over_count / resolved if resolved else 0
    console.print(
        f"\n  {len(rows)} observations  |  "
        f"Resolved: {resolved}  "
        f"([green]OVER: {over_count}[/green] / [red]UNDER: {under_count}[/red])  "
        + (f"Over rate: {hit_rate:.1%}  |  " if resolved else "")
        + f"Pending: {unresolved}"
    )


@app.command("prop-calibration")
def prop_calibration(
    weeks: int = typer.Option(4, "--weeks", "-w", help="Look-back window in weeks."),
    min_samples: int = typer.Option(5, "--min-samples", help="Minimum resolved rows per bucket to show."),
    stat: Optional[str] = typer.Option(None, "--stat", help="Filter to one stat type."),
) -> None:
    """Show player prop model calibration: predicted hit rate vs actual hit rate.

    Groups resolved prop_observations by (stat_type, probability bucket) and
    shows where the model is over- or under-estimating. Use this after running
    evmax cleanup resolve-props to detect systematic bias and tune nba_stats.py.

    Bias = model_prob - actual_hit_rate. Positive = over-estimating probability.
    """
    from evmax.agents.cleanup.db import get_connection

    since = (date.today() - timedelta(weeks=weeks)).isoformat()
    conn = get_connection()

    where = ["scan_date >= ?", "outcome IS NOT NULL", "sharp_prob IS NOT NULL", "mode = 'live'"]
    params: list = [since]
    if stat:
        where.append("stat_type = ?")
        params.append(stat.lower())

    rows = conn.execute(
        f"""SELECT stat_type, line, sharp_prob, kalshi_price, outcome, l15_games
            FROM prop_observations
            WHERE {' AND '.join(where)}
            ORDER BY stat_type, sharp_prob""",
        params,
    ).fetchall()
    conn.close()

    if not rows:
        console.print(f"[yellow]No resolved prop observations in the last {weeks} week(s).[/yellow]")
        console.print("  Run [bold]evmax cleanup resolve-props[/bold] first.")
        return

    # Probability buckets
    BUCKETS = [
        ("< 30%",    0.00, 0.30),
        ("30–40%",   0.30, 0.40),
        ("40–50%",   0.40, 0.50),
        ("50–60%",   0.50, 0.60),
        ("60–70%",   0.60, 0.70),
        ("> 70%",    0.70, 1.01),
    ]

    # Group by stat_type
    from collections import defaultdict
    by_stat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_stat[r["stat_type"]].append(r)

    # Overall summary table
    summary_table = Table(title=f"Prop Model Calibration — last {weeks}w ({len(rows)} resolved)", box=box.SIMPLE)
    summary_table.add_column("Stat Type", min_width=14)
    summary_table.add_column("N", justify="right", width=5)
    summary_table.add_column("Avg Model%", justify="right", width=11)
    summary_table.add_column("Actual Hit%", justify="right", width=12)
    summary_table.add_column("Bias (pp)", justify="right", width=10)
    summary_table.add_column("Brier", justify="right", width=8)
    summary_table.add_column("Signal", justify="center", width=14)

    all_biases: list[float] = []
    for stat_type in sorted(by_stat):
        stat_rows = by_stat[stat_type]
        n = len(stat_rows)
        avg_model = sum(r["sharp_prob"] for r in stat_rows) / n
        hit_rate = sum(r["outcome"] for r in stat_rows) / n
        bias = avg_model - hit_rate  # positive = over-estimating
        brier = sum((r["sharp_prob"] - r["outcome"]) ** 2 for r in stat_rows) / n
        all_biases.append(bias)

        bias_pp = bias * 100
        bias_color = "red" if bias > 0.05 else "yellow" if bias > 0.02 else "green" if bias > -0.02 else "cyan"
        signal = (
            "[red]OVER-EST[/red]" if bias > 0.05 else
            "[yellow]slight over[/yellow]" if bias > 0.02 else
            "[cyan]slight under[/cyan]" if bias < -0.02 else
            "[green]calibrated[/green]"
        )
        summary_table.add_row(
            stat_type,
            str(n),
            f"{avg_model:.1%}",
            f"{hit_rate:.1%}",
            f"[{bias_color}]{bias_pp:+.1f}pp[/{bias_color}]",
            f"{brier:.4f}",
            signal,
        )

    console.print(summary_table)

    # Reliability diagram per bucket (overall)
    bucket_table = Table(title="Reliability by Probability Bucket (all stats)", box=box.SIMPLE)
    bucket_table.add_column("Model Prob", min_width=10)
    bucket_table.add_column("N", justify="right", width=5)
    bucket_table.add_column("Actual Hit%", justify="right", width=12)
    bucket_table.add_column("Bias (pp)", justify="right", width=10)
    bucket_table.add_column("Calibration", justify="center", width=14)

    for label, lo, hi in BUCKETS:
        bucket_rows = [r for r in rows if lo <= r["sharp_prob"] < hi]
        if len(bucket_rows) < min_samples:
            bucket_table.add_row(label, str(len(bucket_rows)), "—", "—", "[dim]too few[/dim]")
            continue
        mid = (lo + hi) / 2
        hit_rate = sum(r["outcome"] for r in bucket_rows) / len(bucket_rows)
        bias = mid - hit_rate
        bias_color = "red" if abs(bias) > 0.08 else "yellow" if abs(bias) > 0.04 else "green"
        cal_str = (
            "[red]over-est[/red]" if bias > 0.08 else
            "[yellow]slight over[/yellow]" if bias > 0.04 else
            "[cyan]slight under[/cyan]" if bias < -0.04 else
            "[green]good[/green]"
        )
        bucket_table.add_row(
            label,
            str(len(bucket_rows)),
            f"{hit_rate:.1%}",
            f"[{bias_color}]{bias*100:+.1f}pp[/{bias_color}]",
            cal_str,
        )

    console.print(bucket_table)

    # Recommendations
    avg_bias = sum(all_biases) / len(all_biases) if all_biases else 0
    console.print()
    if abs(avg_bias) < 0.02:
        console.print("[green]Model is well-calibrated overall.[/green]")
    elif avg_bias > 0.05:
        console.print(
            f"[red]Model over-estimates by avg {avg_bias*100:.1f}pp.[/red]  "
            "Consider: lower _DECAY in nba_stats.py (e.g. 0.80) to reduce hot-streak inflation, "
            "or raise the spread_pct filter to cut thin markets."
        )
    elif avg_bias > 0.02:
        console.print(
            f"[yellow]Slight over-estimation ({avg_bias*100:.1f}pp avg bias).[/yellow]  "
            "Monitor another week — may self-correct with more samples."
        )
    elif avg_bias < -0.05:
        console.print(
            f"[cyan]Model under-estimates by avg {avg_bias*100:.1f}pp.[/cyan]  "
            "Consider: raise _DECAY in nba_stats.py (e.g. 0.90) to weight recent form more, "
            "or loosen the spread_pct filter."
        )

    console.print(
        f"\n  [dim]Tune nba_stats.py constants: _DECAY (recency), _MAX_OPP_ADJ (opponent cap). "
        f"Run resolve-props → prop-calibration weekly.[/dim]"
    )


@app.command("replay-props")
def replay_props(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print results without updating DB."),
) -> None:
    """Replay resolved prop observations through the current model.

    Runs compute_prop_prob_cached for every resolved prop_observations row and
    backfills sharp_prob + ev_pct columns. This enables prop-calibration to work.

    NOTE: Uses the CURRENT player cache, not the historical one from scan-time.
    Best run shortly after games while the cache is still close to scan-time state.
    """
    from evmax.agents.cleanup.db import get_connection
    from evmax.clients.nba_props_cache import compute_prop_prob_cached, is_cache_fresh

    if not is_cache_fresh():
        console.print("[yellow]Props cache is stale. Run a scan first to refresh, or results may be inaccurate.[/yellow]")

    conn = get_connection()
    rows = conn.execute(
        """SELECT id, player_name, stat_type, line, kalshi_price, event_date, outcome
           FROM prop_observations
           WHERE outcome IS NOT NULL
           ORDER BY id""",
    ).fetchall()

    if not rows:
        console.print("[yellow]No resolved prop observations to replay.[/yellow]")
        conn.close()
        return

    updated = 0
    skipped = 0
    model_brier_sum = 0.0
    kalshi_brier_sum = 0.0
    model_probs: list[tuple[float, int]] = []  # (model_prob, outcome)
    n_volatile = 0

    for r in rows:
        row_id = r[0] if isinstance(r, (tuple, list)) else r["id"]
        player = r[1] if isinstance(r, (tuple, list)) else r["player_name"]
        stat = r[2] if isinstance(r, (tuple, list)) else r["stat_type"]
        line = r[3] if isinstance(r, (tuple, list)) else r["line"]
        ask = r[4] if isinstance(r, (tuple, list)) else r["kalshi_price"]
        edate = r[5] if isinstance(r, (tuple, list)) else r["event_date"]
        outcome = r[6] if isinstance(r, (tuple, list)) else r["outcome"]

        result = compute_prop_prob_cached(player, stat, line, edate)
        if result is None:
            skipped += 1
            continue

        model_prob = result.prob
        ev = (model_prob / ask - 1.0) if ask > 0 else 0.0
        model_brier_sum += (model_prob - outcome) ** 2
        kalshi_brier_sum += (ask - outcome) ** 2
        model_probs.append((model_prob, outcome))
        if result.minutes_volatile:
            n_volatile += 1

        if not dry_run:
            conn.execute(
                "UPDATE prop_observations SET sharp_prob = ?, ev_pct = ? WHERE id = ?",
                (round(model_prob, 5), round(ev, 5), row_id),
            )
            updated += 1

    if not dry_run:
        conn.commit()
    conn.close()

    n = len(model_probs)
    if n == 0:
        console.print("[yellow]Model had no cached data for any resolved observations.[/yellow]")
        return

    model_brier = model_brier_sum / n
    kalshi_brier = kalshi_brier_sum / n

    console.print(f"\n[bold cyan]Prop Model Replay — {n} observations ({skipped} skipped, {n_volatile} volatile)[/bold cyan]\n")

    # Overall comparison
    summary = Table(title="Model vs Market", box=box.SIMPLE)
    summary.add_column("Source", min_width=12)
    summary.add_column("Brier", justify="right", width=8)
    summary.add_column("Avg Prob", justify="right", width=10)
    summary.add_column("Actual Hit%", justify="right", width=12)

    avg_model = sum(p for p, _ in model_probs) / n
    avg_kalshi = sum(r[4] if isinstance(r, (tuple, list)) else r["kalshi_price"]
                     for r in rows if compute_prop_prob_cached(
                         r[1] if isinstance(r, (tuple, list)) else r["player_name"],
                         r[2] if isinstance(r, (tuple, list)) else r["stat_type"],
                         r[3] if isinstance(r, (tuple, list)) else r["line"],
                         r[5] if isinstance(r, (tuple, list)) else r["event_date"],
                     ) is not None) / n
    actual_hit = sum(o for _, o in model_probs) / n

    m_color = "green" if model_brier < kalshi_brier else "red"
    k_color = "green" if kalshi_brier < model_brier else "red"
    summary.add_row("Our Model", f"[{m_color}]{model_brier:.4f}[/{m_color}]",
                     f"{avg_model:.1%}", f"{actual_hit:.1%}")
    summary.add_row("Kalshi Price", f"[{k_color}]{kalshi_brier:.4f}[/{k_color}]",
                     f"{avg_kalshi:.1%}", f"{actual_hit:.1%}")
    console.print(summary)

    # Calibration by model prob bucket
    BUCKETS = [
        ("< 30%",    0.00, 0.30),
        ("30–50%",   0.30, 0.50),
        ("50–70%",   0.50, 0.70),
        ("70–85%",   0.70, 0.85),
        ("> 85%",    0.85, 1.01),
    ]
    cal = Table(title="Model Calibration by Predicted Probability", box=box.SIMPLE)
    cal.add_column("Model Prob", min_width=10)
    cal.add_column("N", justify="right", width=5)
    cal.add_column("Predicted", justify="right", width=10)
    cal.add_column("Actual", justify="right", width=10)
    cal.add_column("Bias", justify="right", width=8)

    for label, lo, hi in BUCKETS:
        bucket = [(p, o) for p, o in model_probs if lo <= p < hi]
        if not bucket:
            cal.add_row(label, "0", "—", "—", "—")
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_o = sum(o for _, o in bucket) / len(bucket)
        bias = avg_p - avg_o
        color = "red" if bias > 0.10 else "yellow" if bias > 0.05 else "green" if abs(bias) < 0.05 else "cyan"
        cal.add_row(label, str(len(bucket)), f"{avg_p:.1%}", f"{avg_o:.1%}",
                     f"[{color}]{bias*100:+.1f}pp[/{color}]")

    console.print(cal)

    if not dry_run:
        console.print(f"\n  [green]Updated {updated} rows with model probabilities.[/green]")
        console.print("  Run [bold]evmax cleanup prop-calibration[/bold] for detailed analysis.")
    else:
        console.print("\n  [dim]Dry run — no DB changes made.[/dim]")

    # Diagnosis
    bias = avg_model - actual_hit
    if bias > 0.10:
        console.print(
            f"\n  [bold red]Model over-estimates by {bias*100:.1f}pp.[/bold red]"
            f"\n  The model thinks overs hit {avg_model:.0%} of the time but they actually hit {actual_hit:.0%}."
            f"\n  This is why 'high EV' props lose money — the edge is illusory."
        )


@app.command("backfill-clv")
def backfill_clv_cmd(
    since: str = typer.Option(
        None, "--since", help="YYYY-MM-DD start date (default: all time)."
    ),
    until: str = typer.Option(
        None, "--until", help="YYYY-MM-DD end date (default: today)."
    ),
) -> None:
    """Backfill CLV (Closing Line Value) for resolved bets.

    Compares entry price vs Pinnacle closing line. Positive CLV = real edge.
    """
    from datetime import date as _date
    from evmax.agents.cleanup.resolver import backfill_clv

    since_date = _date.fromisoformat(since) if since else None
    until_date = _date.fromisoformat(until) if until else None

    result = backfill_clv(since=since_date, until=until_date)
    updated = result["updated"]
    skipped = result["skipped"]
    avg_clv = result["avg_clv"]

    if updated:
        clv_color = "green" if avg_clv > 0 else "red"
        console.print(
            f"\n  Backfilled CLV for [bold]{updated}[/bold] bets.  "
            f"Avg CLV: [{clv_color}]{avg_clv:+.1f}%[/{clv_color}]  "
            f"Skipped: {skipped}"
        )
    else:
        console.print("[dim]No resolved bets found missing CLV.[/dim]")
