"""CLI commands for the historical data archive.

Commands:
  evmax archive stats     — show archive size, session counts, row counts by sector
  evmax archive resolve   — fetch Kalshi outcomes for a date, store in archived_outcomes
  evmax archive backtest  — compute EV/P&L/Brier over archived history
  evmax archive export    — dump archived odds/markets to JSONL or CSV
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "archive.db"


def _conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        console.print("[red]Archive DB not found.[/red] Run a scan first: [cyan]evmax agents scan[/cyan]")
        raise typer.Exit(1)
    from evmax.archiver import _get_connection
    return _get_connection()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@app.command("stats")
def stats() -> None:
    """Show archive statistics: session counts, row totals, date range, file size."""
    conn = _conn()

    row = conn.execute(
        "SELECT COUNT(*) as n, MIN(started_at) as oldest, MAX(started_at) as newest"
        " FROM scan_sessions"
    ).fetchone()

    sharp_total  = conn.execute("SELECT COUNT(*) FROM archived_sharp_odds").fetchone()[0]
    kalshi_total = conn.execute("SELECT COUNT(*) FROM archived_kalshi_markets").fetchone()[0]
    outcomes_total = conn.execute("SELECT COUNT(*) FROM archived_outcomes WHERE result IS NOT NULL").fetchone()[0]

    sharp_by_sector = conn.execute(
        "SELECT sector, COUNT(*) as n FROM archived_sharp_odds GROUP BY sector ORDER BY n DESC"
    ).fetchall()
    kalshi_by_sector = conn.execute(
        "SELECT sector, COUNT(*) as n FROM archived_kalshi_markets GROUP BY sector ORDER BY n DESC"
    ).fetchall()

    db_size_mb = DB_PATH.stat().st_size / 1024 / 1024

    console.print()
    console.print("[bold cyan]Archive Stats[/bold cyan]")
    console.print(f"  DB path:  {DB_PATH}")
    console.print(f"  DB size:  {db_size_mb:.2f} MB")
    console.print(f"  Sessions: {row['n']}  (oldest: {(row['oldest'] or 'n/a')[:19]}  newest: {(row['newest'] or 'n/a')[:19]})")
    console.print(f"  Pinnacle odds rows:   {sharp_total:,}")
    console.print(f"  Kalshi market rows:   {kalshi_total:,}")
    console.print(f"  Resolved outcomes:    {outcomes_total:,}")
    console.print()

    t = Table(title="Pinnacle odds by sector", box=box.SIMPLE)
    t.add_column("Sector", style="cyan")
    t.add_column("Rows", justify="right")
    for r in sharp_by_sector:
        t.add_row(r["sector"], f"{r['n']:,}")
    console.print(t)

    t2 = Table(title="Kalshi markets by sector", box=box.SIMPLE)
    t2.add_column("Sector", style="cyan")
    t2.add_column("Rows", justify="right")
    for r in kalshi_by_sector:
        t2.add_row(r["sector"], f"{r['n']:,}")
    console.print(t2)


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

@app.command("resolve")
def resolve(
    target_date: str = typer.Option(
        ..., "--date", "-d", help="Game date to resolve (YYYY-MM-DD)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be fetched without storing."),
) -> None:
    """Fetch Kalshi settlement results for all archived markets on a date.

    Run this the day after games finish. Kalshi settles markets within ~24h of
    game end and returns result='yes' or result='no' on the market endpoint.
    """
    try:
        date.fromisoformat(target_date)
    except ValueError:
        console.print(f"[red]Invalid date:[/red] {target_date!r} — use YYYY-MM-DD")
        raise typer.Exit(1)

    from evmax.archiver import DataArchiver
    archiver = DataArchiver()
    tickers = archiver.get_unresolved_tickers(target_date)

    if not tickers:
        console.print(f"[green]No unresolved archived markets for {target_date}.[/green]")
        return

    console.print(f"[cyan]Fetching outcomes for {len(tickers)} markets on {target_date}...[/cyan]")

    if dry_run:
        for t in tickers[:20]:
            console.print(f"  [dim]{t}[/dim]")
        if len(tickers) > 20:
            console.print(f"  [dim]... and {len(tickers) - 20} more[/dim]")
        console.print("[dim]Dry run — no changes made.[/dim]")
        return

    outcomes = asyncio.run(_fetch_kalshi_results(tickers))

    settled   = [(t, r) for t, r in outcomes if r is not None]
    still_open = [(t, r) for t, r in outcomes if r is None]

    if settled:
        stored = archiver.store_outcomes(settled)
        yes_won = sum(1 for _, r in settled if r == 1)
        no_won  = sum(1 for _, r in settled if r == 0)
        console.print(
            f"  [green]Stored {stored} outcomes[/green]  "
            f"(YES won: {yes_won}  NO won: {no_won})"
        )
    if still_open:
        console.print(f"  [yellow]{len(still_open)} markets still open / not yet settled.[/yellow]")
        console.print("  [dim]Re-run tomorrow once Kalshi settles all markets.[/dim]")


async def _fetch_kalshi_results(tickers: list[str]) -> list[tuple[str, int | None]]:
    """Concurrently fetch Kalshi /markets/{ticker} and parse result field."""
    import httpx
    from evmax.clients.kalshi import KalshiClient

    results: list[tuple[str, int | None]] = []

    async with KalshiClient() as client:
        batch_size = 20
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            tasks = [_get_result(client, ticker) for ticker in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for ticker, outcome in zip(batch, batch_results):
                if isinstance(outcome, Exception):
                    results.append((ticker, None))
                else:
                    results.append((ticker, outcome))

    return results


async def _get_result(client, ticker: str) -> int | None:
    """Return 1 (YES won), 0 (NO won), or None (still open / error)."""
    try:
        data = await client._get(f"/markets/{ticker}")
        market = data.get("market", {})
        result = market.get("result", "")
        if result == "yes":
            return 1
        if result == "no":
            return 0
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@app.command("backtest")
def backtest(
    since: str = typer.Option(..., "--since", help="Start date YYYY-MM-DD (event_date)."),
    until: Optional[str] = typer.Option(None, "--until", help="End date YYYY-MM-DD (event_date, default: today)."),
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Filter by sector (e.g. soccer, nba)."),
    ev_threshold: float = typer.Option(0.02, "--ev-threshold", help="Minimum EV to count as a flagged bet."),
    bankroll: float = typer.Option(250.0, "--bankroll", "-b", help="Simulated bankroll for P&L calc."),
    kelly: float = typer.Option(0.25, "--kelly", "-k", help="Kelly fraction used for stake sizing."),
    min_volume: float = typer.Option(0.0, "--min-volume", help="Minimum Kalshi volume_usd filter."),
) -> None:
    """Backtest EV strategy over archived history.

    Joins archived Kalshi prices + Pinnacle odds + resolved outcomes.
    Uses the FIRST captured price for each market (entry price at detection time).
    Lets you retroactively test different EV thresholds against real results.
    """
    try:
        date.fromisoformat(since)
    except ValueError:
        console.print(f"[red]Invalid --since:[/red] {since!r} — use YYYY-MM-DD")
        raise typer.Exit(1)

    until_str = until or date.today().isoformat()

    conn = _conn()

    # -----------------------------------------------------------------------
    # 1. Fetch first Kalshi snapshot per ticker in date range
    # -----------------------------------------------------------------------
    sector_filter = "AND k.sector = ?" if sector else ""
    sector_params = [sector.lower()] if sector else []

    rows = conn.execute(
        f"""
        SELECT k.ticker, k.sector, k.yes_team, k.yes_price, k.event_date,
               k.event_id, k.market_type, k.title, k.line, k.volume_usd,
               s.true_prob_a, s.true_prob_b, s.true_prob_draw,
               s.outcome_a_label, s.outcome_b_label,
               o.result
        FROM archived_kalshi_markets k
        JOIN archived_sharp_odds s ON k.event_id = s.event_id
        LEFT JOIN archived_outcomes o ON k.ticker = o.ticker
        WHERE k.event_date BETWEEN ? AND ?
          AND k.event_id IS NOT NULL
          {sector_filter}
        ORDER BY k.fetched_at ASC
        """,
        [since, until_str] + sector_params,
    ).fetchall()

    if not rows:
        console.print(f"[yellow]No archived data found for {since} → {until_str}.[/yellow]")
        console.print("  Tip: run [bold]evmax agents scan[/bold] to start building the archive.")
        return

    # -----------------------------------------------------------------------
    # 2. Deduplicate to first snapshot per ticker (ORDER BY fetched_at ASC above)
    # -----------------------------------------------------------------------
    seen: set[str] = set()
    deduped = []
    for r in rows:
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            deduped.append(dict(r))

    # -----------------------------------------------------------------------
    # 3. Align YES-team probability and compute EV
    # -----------------------------------------------------------------------
    bets = []
    for r in deduped:
        if r["volume_usd"] is not None and r["volume_usd"] < min_volume:
            continue

        true_prob = _align_prob(
            r["yes_team"], r["event_id"],
            r["true_prob_a"], r["true_prob_b"], r["true_prob_draw"],
        )
        if true_prob is None or true_prob <= 0:
            continue

        yes_price = r["yes_price"]
        if yes_price <= 0 or yes_price >= 1:
            continue

        payout = 1.0 / yes_price
        ev = true_prob * payout - 1.0

        bets.append({
            **r,
            "true_prob": true_prob,
            "ev": ev,
            "payout": payout,
        })

    if not bets:
        console.print("[yellow]No matched market pairs found after alignment.[/yellow]")
        return

    # -----------------------------------------------------------------------
    # 4. Compute metrics at the chosen EV threshold
    # -----------------------------------------------------------------------
    flagged   = [b for b in bets if b["ev"] >= ev_threshold]
    resolved  = [b for b in flagged if b["result"] is not None]
    pending   = [b for b in flagged if b["result"] is None]
    wins      = [b for b in resolved if b["result"] == 1]
    losses    = [b for b in resolved if b["result"] == 0]

    total_staked = 0.0
    total_pnl    = 0.0
    for b in resolved:
        stake = bankroll * min(kelly * max(0.0, b["ev"]), 0.05)
        total_staked += stake
        if b["result"] == 1:
            total_pnl += stake * (b["payout"] - 1.0)
        else:
            total_pnl -= stake

    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (len(wins) / len(resolved) * 100) if resolved else 0.0

    # Brier score over resolved flagged bets
    brier = (
        sum((b["true_prob"] - b["result"]) ** 2 for b in resolved) / len(resolved)
        if resolved else None
    )
    # Naive Brier (always predict 0.5)
    brier_naive = 0.25

    # -----------------------------------------------------------------------
    # 5. Summary table
    # -----------------------------------------------------------------------
    pnl_color = "green" if total_pnl >= 0 else "red"
    pnl_sign  = "+" if total_pnl >= 0 else ""
    roi_color = "green" if roi >= 0 else "red"

    console.print()
    console.print(
        f"[bold cyan]Archive Backtest[/bold cyan]  "
        f"{since} → {until_str}"
        + (f"  |  sector: {sector.upper()}" if sector else "")
        + f"  |  EV ≥ {ev_threshold*100:.0f}%"
    )
    console.print()

    summary = Table(title="Summary", box=box.SIMPLE, show_header=False)
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", justify="right")
    summary.add_row("Total markets archived",     f"{len(bets):,}")
    summary.add_row(f"Flagged (EV ≥ {ev_threshold*100:.0f}%)", f"{len(flagged):,}")
    summary.add_row("Resolved",                   f"{len(resolved):,}")
    summary.add_row("Pending / not settled",       f"{len(pending):,}")
    summary.add_row("Wins",                        f"[green]{len(wins)}[/green]")
    summary.add_row("Losses",                      f"[red]{len(losses)}[/red]")
    summary.add_row("Win rate",                    f"{win_rate:.1f}%")
    summary.add_row("Total staked",               f"${total_staked:.2f}")
    summary.add_row("Net P&L",                    f"[{pnl_color}]{pnl_sign}${total_pnl:.2f}[/{pnl_color}]")
    summary.add_row("ROI",                        f"[{roi_color}]{pnl_sign}{roi:.1f}%[/{roi_color}]")
    if brier is not None:
        brier_color = "green" if brier < brier_naive else "red"
        summary.add_row("Brier score (flagged)",  f"[{brier_color}]{brier:.4f}[/{brier_color}]")
        summary.add_row("Brier naive (0.5 always)", f"{brier_naive:.4f}")
    console.print(summary)

    # -----------------------------------------------------------------------
    # 6. Threshold comparison table
    # -----------------------------------------------------------------------
    console.print()
    thresh_table = Table(title="Results at different EV thresholds (resolved bets only)", box=box.SIMPLE)
    thresh_table.add_column("EV ≥",   justify="right", width=8)
    thresh_table.add_column("Bets",   justify="right", width=6)
    thresh_table.add_column("Wins",   justify="right", width=6)
    thresh_table.add_column("Losses", justify="right", width=7)
    thresh_table.add_column("Win%",   justify="right", width=7)
    thresh_table.add_column("ROI",    justify="right", width=8, style="bold")
    thresh_table.add_column("P&L",    justify="right", width=10)

    for thresh in (0.02, 0.03, 0.05, 0.08, 0.10):
        t_flagged  = [b for b in bets if b["ev"] >= thresh]
        t_resolved = [b for b in t_flagged if b["result"] is not None]
        if not t_resolved:
            thresh_table.add_row(f"{thresh*100:.0f}%", str(len(t_flagged)), "—", "—", "—", "—", "—")
            continue
        t_wins   = sum(1 for b in t_resolved if b["result"] == 1)
        t_losses = len(t_resolved) - t_wins
        t_staked = 0.0
        t_pnl    = 0.0
        for b in t_resolved:
            stake   = bankroll * min(kelly * max(0.0, b["ev"]), 0.05)
            t_staked += stake
            if b["result"] == 1:
                t_pnl += stake * (b["payout"] - 1.0)
            else:
                t_pnl -= stake
        t_roi = (t_pnl / t_staked * 100) if t_staked > 0 else 0.0
        t_wr  = t_wins / len(t_resolved) * 100
        roi_c = "green" if t_roi >= 0 else "red"
        pnl_c = "green" if t_pnl >= 0 else "red"
        thresh_table.add_row(
            f"{thresh*100:.0f}%",
            str(len(t_resolved)),
            f"[green]{t_wins}[/green]",
            f"[red]{t_losses}[/red]",
            f"{t_wr:.1f}%",
            f"[{roi_c}]{'+' if t_roi>=0 else ''}{t_roi:.1f}%[/{roi_c}]",
            f"[{pnl_c}]{'+' if t_pnl>=0 else ''}${t_pnl:.2f}[/{pnl_c}]",
        )
    console.print(thresh_table)

    # -----------------------------------------------------------------------
    # 7. P&L by sector
    # -----------------------------------------------------------------------
    if not sector:
        sector_stats: dict[str, dict] = defaultdict(lambda: {"w": 0, "l": 0, "staked": 0.0, "pnl": 0.0})
        for b in resolved:
            s = b["sector"]
            stake = bankroll * min(kelly * max(0.0, b["ev"]), 0.05)
            sector_stats[s]["staked"] += stake
            if b["result"] == 1:
                sector_stats[s]["w"] += 1
                sector_stats[s]["pnl"] += stake * (b["payout"] - 1.0)
            else:
                sector_stats[s]["l"] += 1
                sector_stats[s]["pnl"] -= stake

        sect_table = Table(title="P&L by sector (resolved bets)", box=box.SIMPLE)
        sect_table.add_column("Sector",  style="cyan", width=10)
        sect_table.add_column("W",   justify="right", width=5)
        sect_table.add_column("L",   justify="right", width=5)
        sect_table.add_column("Win%",    justify="right", width=7)
        sect_table.add_column("Staked",  justify="right", width=9)
        sect_table.add_column("P&L",     justify="right", width=10)
        sect_table.add_column("ROI",     justify="right", width=8)

        for sec, st in sorted(sector_stats.items(), key=lambda x: -abs(x[1]["pnl"])):
            total_s = st["w"] + st["l"]
            wr_s    = st["w"] / total_s * 100 if total_s else 0
            roi_s   = st["pnl"] / st["staked"] * 100 if st["staked"] else 0
            c       = "green" if st["pnl"] >= 0 else "red"
            sect_table.add_row(
                sec.upper(),
                f"[green]{st['w']}[/green]",
                f"[red]{st['l']}[/red]",
                f"{wr_s:.1f}%",
                f"${st['staked']:.2f}",
                f"[{c}]{'+' if st['pnl']>=0 else ''}${st['pnl']:.2f}[/{c}]",
                f"[{c}]{'+' if roi_s>=0 else ''}{roi_s:.1f}%[/{c}]",
            )
        console.print(sect_table)

    # -----------------------------------------------------------------------
    # 8. Calibration bins (flagged + resolved bets)
    # -----------------------------------------------------------------------
    if len(resolved) >= 10:
        console.print()
        _print_calibration(resolved)

    if pending:
        console.print(
            f"\n  [dim]{len(pending)} pending markets not yet resolved. "
            f"Run [bold]evmax archive resolve --date {until_str}[/bold] to fetch outcomes.[/dim]"
        )


def _align_prob(
    yes_team: Optional[str],
    event_id: str,
    true_prob_a: float,
    true_prob_b: float,
    true_prob_draw: Optional[float],
) -> Optional[float]:
    """Map Kalshi YES team → correct sharp probability side."""
    if not yes_team:
        return true_prob_a

    yt = yes_team.lower().strip()

    # Draw market
    if yt in ("tie", "draw", "x") and true_prob_draw is not None:
        return true_prob_draw

    # Extract team_a from event_id: "sector::date::team_a_vs_team_b"
    parts = event_id.split("::")
    if len(parts) >= 3 and "_vs_" in parts[2]:
        team_a_slug = parts[2].split("_vs_")[0].replace("_", " ")
        # Simple token overlap check (both already normalized)
        if yt in team_a_slug or team_a_slug in yt or _overlap(yt, team_a_slug) >= 0.6:
            return true_prob_a
        return true_prob_b

    return true_prob_a


def _overlap(a: str, b: str) -> float:
    """Fraction of tokens in a that appear in b."""
    ta, tb = set(a.split()), set(b.split())
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _print_calibration(resolved: list[dict]) -> None:
    """Print a 10-bin calibration table for resolved bets."""
    bins: list[dict] = []
    for low in [i / 10 for i in range(0, 10)]:
        high = low + 0.1
        bucket = [b for b in resolved if low <= b["true_prob"] < high]
        if not bucket:
            bins.append({"label": f"{low:.0%}–{high:.0%}", "n": 0, "mean_pred": 0, "actual": 0})
            continue
        mean_pred  = sum(b["true_prob"] for b in bucket) / len(bucket)
        actual_rate = sum(b["result"] for b in bucket) / len(bucket)
        bins.append({
            "label": f"{low:.0%}–{high:.0%}",
            "n": len(bucket),
            "mean_pred": mean_pred,
            "actual": actual_rate,
        })

    cal_table = Table(title="Calibration (predicted prob vs actual win rate)", box=box.SIMPLE)
    cal_table.add_column("Prob bin",    width=10, style="dim")
    cal_table.add_column("N",           justify="right", width=5)
    cal_table.add_column("Mean pred",   justify="right", width=10)
    cal_table.add_column("Actual rate", justify="right", width=12)
    cal_table.add_column("Δ",           justify="right", width=8)
    cal_table.add_column("Bar", width=20)

    for b in bins:
        if b["n"] == 0:
            cal_table.add_row(b["label"], "0", "—", "—", "—", "")
            continue
        delta = b["actual"] - b["mean_pred"]
        delta_color = "green" if abs(delta) <= 0.05 else "yellow" if abs(delta) <= 0.10 else "red"
        bar_len = int(b["actual"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        cal_table.add_row(
            b["label"],
            str(b["n"]),
            f"{b['mean_pred']:.3f}",
            f"{b['actual']:.3f}",
            f"[{delta_color}]{delta:+.3f}[/{delta_color}]",
            f"[cyan]{bar}[/cyan]",
        )

    console.print(cal_table)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@app.command("export")
def export(
    sector: str = typer.Option(..., "--sector", "-s", help="Sector to export, e.g. 'soccer'"),
    source: str = typer.Option("both", "--source", help="'pinnacle', 'kalshi', or 'both'"),
    since: Optional[str] = typer.Option(None, "--since", help="Start date YYYY-MM-DD (filters event_date)"),
    until: Optional[str] = typer.Option(None, "--until", help="End date YYYY-MM-DD (filters event_date)"),
    fmt: str = typer.Option("jsonl", "--format", "-f", help="Output format: 'jsonl' or 'csv'"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output file path (default: stdout)"),
) -> None:
    """Export archived data for a sector to JSONL or CSV."""
    conn = _conn()

    fh = open(out, "w") if out else sys.stdout
    writer = csv.writer(fh) if fmt == "csv" else None
    header_written = False

    def _write(row_dict: dict) -> None:
        nonlocal header_written
        if fmt == "jsonl":
            fh.write(json.dumps(row_dict) + "\n")
        else:
            if not header_written:
                writer.writerow(list(row_dict.keys()))
                header_written = True
            writer.writerow(list(row_dict.values()))

    def _date_filter(col: str) -> tuple[str, list]:
        clauses, params = [], []
        if since:
            clauses.append(f"{col} >= ?")
            params.append(since)
        if until:
            clauses.append(f"{col} <= ?")
            params.append(until)
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    total = 0

    if source in ("pinnacle", "both"):
        date_sql, date_params = _date_filter("event_date")
        rows = conn.execute(
            f"SELECT * FROM archived_sharp_odds WHERE sector=?{date_sql} ORDER BY fetched_at",
            [sector] + date_params,
        ).fetchall()
        for r in rows:
            _write(dict(r))
        total += len(rows)

    if source in ("kalshi", "both"):
        date_sql, date_params = _date_filter("event_date")
        rows = conn.execute(
            f"SELECT * FROM archived_kalshi_markets WHERE sector=?{date_sql} ORDER BY fetched_at",
            [sector] + date_params,
        ).fetchall()
        for r in rows:
            _write(dict(r))
        total += len(rows)

    if out:
        fh.close()
        console.print(f"Exported {total:,} rows to {out}")
