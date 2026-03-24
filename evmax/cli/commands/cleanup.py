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
    placed_only: bool = typer.Option(False, "--placed", help="Only show bets you manually placed via 'pick'."),
    bankroll: float = typer.Option(250.0, "--bankroll", "-b", help="Bankroll in USD for P&L calculation."),
    game_date: Optional[str] = typer.Option(None, "--date", help="Filter by game date (YYYY-MM-DD)."),
) -> None:
    """Show logged +EV predictions and their WIN/LOSS outcomes."""
    from evmax.agents.cleanup.db import get_connection

    if game_date:
        try:
            date.fromisoformat(game_date)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {game_date!r} — use YYYY-MM-DD")
            raise typer.Exit(1)

    since = (date.today() - timedelta(days=days)).isoformat()
    conn = get_connection()

    where_parts = ["p.scan_date >= ?", "p.voided = 0"]
    params: list = [since]
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
                SELECT market_id, MAX(scan_date) AS latest_scan
                FROM ev_predictions
                WHERE voided = 0
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
        console.print(f"[yellow]{msg} in the last {days} day(s).[/yellow]")
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

    title_date = f"game date {game_date}" if game_date else f"last {days}d"
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
    from evmax.clients.pinnacle import PinnacleClient

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
        async with PinnacleClient() as client:
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
