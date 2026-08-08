"""CLI commands for prediction logging, outcome resolution, and model calibration.

Commands:
  evmax cleanup show          — show logged +EV bets and their outcomes
  evmax cleanup resolve       — fetch actual outcomes for a given date
  evmax cleanup metrics       — compute Brier scores and display calibration report
  evmax cleanup value-audit   — per-sector model-blend VALUE audit (Brier vs sharp AND
                                vs close, CLV, calibration) with significance + verdict
  evmax cleanup adjust        — auto-adjust sharp_weight based on Brier scores
  evmax cleanup train         — re-seed model agents from live data (ESPN + bo3.gg)
  evmax cleanup props         — show logged prop observations and outcomes
  evmax cleanup resolve-props — fetch ESPN boxscores and fill prop_observations outcomes
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from evmax.formatting import format_outcome_label
from evmax.models.market import venue_label

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
                   p.market_type, p.line, p.venue,
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
    table.add_column("Ven",     style="dim", width=6)
    table.add_column("Event",   style="dim", min_width=18, no_wrap=False)
    table.add_column("Outcome", style="bold white", min_width=12, no_wrap=False)
    table.add_column("Ask",     justify="right", width=6)
    table.add_column("TrueP",   justify="right", width=6)
    table.add_column("CLV",     justify="right", width=7)
    table.add_column("EV%",     justify="right", width=6)
    table.add_column("Stake$",  justify="right", width=7)
    table.add_column("P&L$",    justify="right", width=8)
    table.add_column("Src",     style="dim", width=10)
    table.add_column("Result",  justify="center", width=7)

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

        # Display Pinnacle drift here (legacy "CLV" column). This is NOT a
        # clean edge metric for our system — see backfill_clv docstring for
        # the selection-bias argument. The new kalshi_clv_pct column is the
        # primary edge signal but isn't yet shown in this table.
        close_prob = r["pinnacle_close_prob"]
        entry_price = r["kalshi_yes_price"]
        if close_prob is not None and entry_price is not None:
            drift = close_prob - entry_price
            clv_color = "green" if drift >= 0.01 else "red" if drift <= -0.01 else "dim"
            clv_str = f"[{clv_color}]{drift*100:+.1f}pp[/{clv_color}]"
        else:
            clv_str = "[dim]—[/dim]"

        table.add_row(
            r["event_date"] or r["scan_date"] or "",
            (r["sector"] or "").upper(),
            venue_label(r["venue"] if "venue" in r.keys() else None),
            (r["event_title"] or "")[:24],
            format_outcome_label(
                yes_team=r["yes_team"],
                market_type=r["market_type"] or "moneyline",
                line=r["line"],
            ) + placed_marker,
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


# Default sectors refreshed by the resolve-time model-update hook. Mirrors the
# `evmax update scores` default; ESPN-supported game sectors only.
_UPDATE_HOOK_SECTORS = ["soccer", "worldcup", "nba", "wnba", "nfl", "ncaab", "nhl", "baseball"]


@app.command("resolve")
def resolve(
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Date to resolve (YYYY-MM-DD). Defaults to yesterday.",
    ),
    update_models: bool = typer.Option(
        True, "--update-models/--no-update-models",
        help=(
            "After resolving, feed the date's completed ESPN scores into the "
            "Elo/Form/Poisson/xG models so in-season state stays fresh. On by "
            "default — this is what keeps the daily cron from letting states go "
            "stale. Failures here never break resolution."
        ),
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
    # One scoreboard cache shared across the resolve phase and the model-update
    # hook so both reuse a single ESPN fetch per (sport, league, date) instead
    # of pulling the same dates twice. Plain dict — no event-loop affinity, so
    # the two separate asyncio.run() calls below can share it.
    espn_cache: dict = {}
    _t0 = time.perf_counter()
    result = asyncio.run(resolve_outcomes_for_date(d, espn_cache=espn_cache))
    _resolve_s = time.perf_counter() - _t0

    console.print(
        f"  [green]Resolved:[/green] {result['resolved']}  "
        f"[yellow]Unmatched:[/yellow] {result['failed']}"
        + (f"  [dim]Voided:[/dim] {result['voided']}" if result.get("voided") else "")
        + f"  [dim]({_resolve_s:.1f}s)[/dim]"
    )

    try:
        from evmax.agents.cleanup.resolver import backfill_clv
        _t0 = time.perf_counter()
        clv = backfill_clv()
        _clv_s = time.perf_counter() - _t0
        if clv["updated"]:
            avg_kc = clv.get("avg_kalshi_clv", 0.0)
            kc_color = "green" if avg_kc > 0 else "red" if avg_kc < 0 else "dim"
            console.print(
                f"  [green]CLV backfilled:[/green] {clv['updated']} bet(s)  "
                f"Kalshi-CLV [{kc_color}]{avg_kc:+.1f}pp[/{kc_color}]  "
                f"[dim]Pinn-drift {clv.get('avg_pinn_drift', 0.0):+.1f}pp  "
                f"skipped: {clv['skipped']}  ({_clv_s:.1f}s)[/dim]"
            )
    except Exception as _clv_err:
        console.print(f"  [yellow]Warning: CLV backfill failed:[/yellow] {_clv_err}")

    # Refresh model state from the same date's completed scores. Wrapped in
    # try/except so a fetch/model failure can never break resolution (the cron
    # depends on resolve always exiting cleanly). Off via --no-update-models.
    if update_models:
        try:
            from evmax.agents.cleanup.model_updater import update_models_for_date

            _t0 = time.perf_counter()
            upd = asyncio.run(
                update_models_for_date(_UPDATE_HOOK_SECTORS, d, espn_cache=espn_cache)
            )
            _upd_s = time.perf_counter() - _t0
            skipped_note = (
                f", {upd.skipped} already fed" if upd.skipped else ""
            )
            console.print(
                f"  [green]Model state refreshed:[/green] {upd.updated} game(s) "
                f"across {len(_UPDATE_HOOK_SECTORS)} sector(s){skipped_note}  "
                f"[dim]({_upd_s:.1f}s)[/dim]"
            )
        except Exception as _upd_err:
            console.print(
                f"  [yellow]Warning: model update failed:[/yellow] {_upd_err}"
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


def near_tip_snapshot_candidates(conn, event_ids: list[str]) -> list:
    """Unresolved, unvoided bets on the given events that deserve a near-tip
    venue snapshot (Kalshi or Polymarket US) from the watch-closes sweep.

    Includes BOTH live and shadow rows (widened 2026-07-05): the laddered
    categories (WNBA spread/total) sit in shadow mode, and their promotion
    gate — `cleanup shadow clv --side lay` — anchors CLV against the last
    archived Kalshi price before tip. With the old ``mode = 'live'`` filter,
    shadow bets never got a watch-closes snapshot, so their "close" was
    whatever hourly watch-listings sweep happened to land last (median ~5h
    before tip, worse across Mac-sleep holes). Disabled-category rows never
    reach ev_predictions, so 'live'/'shadow' covers everything persisted.
    """
    if not event_ids:
        return []
    placeholders = ",".join("?" * len(event_ids))
    return conn.execute(
        f"""SELECT DISTINCT p.market_id, p.sector, p.event_id, p.event_date,
                   p.market_type, p.yes_team, p.line
            FROM ev_predictions p
            LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
            WHERE p.event_id IN ({placeholders})
              AND p.mode IN ('live', 'shadow') AND p.voided = 0
              AND (o.outcome IS NULL OR o.id IS NULL)""",
        event_ids,
    ).fetchall()


def snapshot_targets_by_venue(bet_rows: list) -> dict[str, dict[str, dict]]:
    """Group near-tip snapshot candidates by venue → {archive_ticker: bet meta}.

    market_id conventions: ``kalshi:TICKER[:no]`` and
    ``polymarket_us:slug[:side][:no]``. The Kalshi branch needs the RAW
    ticker for the orderbook fetch; the Polymarket US branch keeps the full
    venue-prefixed id — it is both the live-market lookup key
    (PredictionMarket.id) and the ticker the snapshot is archived under
    (see resolver.close_lookup_ticker). The live book is the same YES
    market for both sides of a bet, so the NO-side ``:no`` suffix is
    stripped. Rows from any unrecognized venue are skipped.
    """
    targets: dict[str, dict[str, dict]] = {}
    for r in bet_rows:
        mid = r["market_id"]
        if not mid:
            continue
        ticker = mid.removesuffix(":no")
        if ticker.startswith("kalshi:"):
            venue, key = "kalshi", ticker.removeprefix("kalshi:")
        elif ticker.startswith("polymarket_us:"):
            venue, key = "polymarket_us", ticker
        else:
            continue
        targets.setdefault(venue, {}).setdefault(key, dict(r))
    return targets


async def capture_polymarket_us_closes(targets: dict[str, dict]) -> int:
    """Snapshot live Polymarket US asks for unresolved bets near tip-off.

    ``targets`` maps the venue-prefixed market id ("polymarket_us:slug[:side]")
    → bet meta (see snapshot_targets_by_venue). Prices come from the same
    league-events fetch the scanner uses — one call per sector, indexed by
    PredictionMarket.id, which is exactly the id ev_predictions stores.
    Snapshots land in archived_kalshi_markets under the PREFIXED id so
    backfill_clv's close lookup hits them without a schema change; lowercase
    PolyUS slugs can never collide with Kalshi's uppercase tickers.

    The venue firewall (polymarket_us_live) is irrelevant here — shadow rows
    are exactly the ones the promotion gate needs closes for — but the venue
    kill-switch (polymarket_us_enabled) is honored. Failures never abort the
    sweep.
    """
    if not targets:
        return 0
    from evmax.settings import get_settings

    if not get_settings().polymarket_us_enabled:
        return 0

    from evmax.archiver import DataArchiver
    from evmax.clients.polymarket_us import PolymarketUSClient

    sectors = sorted({m["sector"] for m in targets.values()})
    markets_by_id: dict[str, object] = {}
    try:
        async with PolymarketUSClient() as client:
            batches = await asyncio.gather(*(client.get_markets(sec) for sec in sectors))
        for batch in batches:
            for m in batch:
                markets_by_id[m.id] = m
    except Exception as perr:  # noqa: BLE001 — never abort the sweep
        console.print(f"[yellow]  polymarket_us snapshot fetch failed: {perr}[/yellow]")
        return 0

    snapshots_by_sector: dict[str, list[dict]] = {}
    for mid, meta in targets.items():
        market = markets_by_id.get(mid)
        # Missing = delisted near tip; ≥0.99 mirrors the Kalshi empty-book filter.
        if market is None or market.yes_price is None or market.yes_price >= 0.99:
            continue
        snapshots_by_sector.setdefault(meta["sector"], []).append({
            "ticker": mid,  # venue-prefixed on purpose — see docstring
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "event_id": meta.get("event_id"),
            "event_date": meta.get("event_date"),
            "market_type": meta.get("market_type"),
            "yes_team": meta.get("yes_team"),
            "line": meta.get("line"),
        })
    if not snapshots_by_sector:
        return 0

    # Fresh session per sweep — UNIQUE(session_id, ticker) would IGNORE
    # a re-used session's snapshot.
    session_id = "watchclose-pmus-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    archiver = DataArchiver()
    captured = 0
    for sec, snaps in snapshots_by_sector.items():
        captured += archiver.archive_kalshi_snapshot(session_id, sec, snaps)
    return captured


@app.command("watch-closes")
def watch_closes(
    lookahead: int = typer.Option(
        45, "--lookahead", "-l",
        help="Capture closes for any event tipping off within the next N minutes. "
             "Default 45 so a near-tip Kalshi snapshot lands inside the [T-60, T-30] "
             "window a typical pre-game fill is placed in.",
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
    """Watch upcoming events and capture closing lines ~lookahead min before each tip-off.

    Captures two things per sweep:
      1. Pinnacle close → ev_outcomes.pinnacle_close_prob (for the diagnostic drift).
      2. A near-tip venue yes-ask snapshot → archived_kalshi_markets, for BOTH
         venues (Kalshi tickers + Polymarket US prefixed ids). This is what gives
         placed/shadow-bet venue CLV a genuine post-entry close to anchor against —
         without it, get_kalshi_close_price only sees the night-before scan row, so
         close == entry and CLV collapses to ~0. PolyUS shadow rows need this for
         the venue promotion gate (entry→close CLV on the venue's own book).

    Self-sufficient — pulls each cycle's queue from the DB, so you can scan at any
    time of day and this loop will pick up the new events on the next sweep.

    Recommended setup: run this as a launchd/systemd service that's always up so the
    [T-45, T-0] window is swept for every game. Idempotent for Pinnacle (only writes
    rows still missing pinnacle_close_prob); each Kalshi sweep writes a fresh
    timestamped snapshot so the close anchor has a real price trail to pick from.
    """
    import asyncio
    import time
    from evmax.agents.cleanup.db import get_connection
    from evmax.archiver import DataArchiver
    from evmax.archiver import _get_connection as get_archive_conn
    from evmax.clients.esports_pinnacle import PinnacleGuestClient
    from evmax.clients.kalshi import KalshiClient

    async def _capture_venue_closes(event_ids: list[str]) -> int:
        """Snapshot live venue asks (Kalshi + Polymarket US) for unresolved bets.

        This is what gives placed/shadow-bet venue CLV a genuine POST-ENTRY close
        to anchor against — without a near-tip snapshot, get_kalshi_close_price
        only sees the night-before scan row (close == entry → CLV ≈ 0). Independent
        of the Pinnacle-close queue: we keep snapshotting until tip-off regardless
        of whether the Pinnacle close has landed. Failures here never abort the
        sweep, and each venue fails independently of the other.
        """
        conn = get_connection()
        bet_rows = near_tip_snapshot_candidates(conn, event_ids)
        conn.close()
        if not bet_rows:
            return 0

        targets = snapshot_targets_by_venue(bet_rows)
        pmus_captured = await capture_polymarket_us_closes(
            targets.get("polymarket_us", {})
        )

        meta_by_ticker = targets.get("kalshi", {})
        if not meta_by_ticker:
            return pmus_captured

        tickers = list(meta_by_ticker)
        try:
            async with KalshiClient() as client:
                asks = await client.get_market_asks_batch(tickers)
        except Exception as kerr:  # noqa: BLE001 — never abort the sweep
            console.print(f"[yellow]  kalshi snapshot fetch failed: {kerr}[/yellow]")
            return pmus_captured

        snapshots: list[dict] = []
        for ticker, meta in meta_by_ticker.items():
            yes_ask = asks.get(ticker)
            # 99c asks = empty orderbook; not a real close.
            if yes_ask is None or yes_ask >= 0.99:
                continue
            snapshots.append({
                "ticker": ticker,
                "yes_price": yes_ask,
                "event_id": meta.get("event_id"),
                "event_date": meta.get("event_date"),
                "market_type": meta.get("market_type"),
                "yes_team": meta.get("yes_team"),
                "line": meta.get("line"),
            })
        if not snapshots:
            return pmus_captured

        # Fresh session per sweep — UNIQUE(session_id, ticker) means a reused
        # session would IGNORE the new snapshot.
        session_id = "watchclose-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archiver = DataArchiver()
        # snapshots may span sectors; archive per sector for the sector column.
        captured = 0
        by_sector: dict[str, list[dict]] = {}
        for snap in snapshots:
            sec = meta_by_ticker[snap["ticker"]]["sector"]
            by_sector.setdefault(sec, []).append(snap)
        for sec, snaps in by_sector.items():
            captured += archiver.archive_kalshi_snapshot(session_id, sec, snaps)
        return pmus_captured + captured

    async def _sweep() -> tuple[int, int, int]:
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
            return (0, 0, 0)

        event_ids = [r["event_id"] for r in upcoming]

        # 1b. Near-tip venue snapshots — Kalshi + Polymarket US — for
        # placed/shadow-bet CLV close anchoring.
        kalshi_captured = await _capture_venue_closes(event_ids)

        # 2. From predictions.db: which of those still need a close.
        # Pull yes_team via ev_predictions so the close prob can be
        # aligned to the YES side actually bet on.
        conn = get_connection()
        placeholders = ",".join("?" * len(event_ids))
        rows = conn.execute(
            f"""SELECT o.market_id, o.event_id, o.sector, p.yes_team, p.market_type
                FROM ev_outcomes o
                JOIN ev_predictions p ON o.market_id = p.market_id
                WHERE o.event_id IN ({placeholders})
                  AND o.outcome IS NULL
                  AND o.pinnacle_close_prob IS NULL""",
            event_ids,
        ).fetchall()

        if not rows:
            conn.close()
            return (0, 0, kalshi_captured)

        # 3. Group by sector, fetch Pinnacle once per sector
        by_sector: dict[str, list] = {}
        for r in rows:
            by_sector.setdefault(r["sector"], []).append(dict(r))

        updated = 0
        from evmax.agents.cleanup.resolver import yes_aligned_close_prob
        async with PinnacleGuestClient() as client:
            for sector, markets in by_sector.items():
                try:
                    sharp_odds = await client.get_odds(sector)
                except Exception as fetch_err:
                    console.print(f"[yellow]  [{sector}] fetch failed: {fetch_err}[/yellow]")
                    continue
                so_by_event = {so.event_id: so for so in sharp_odds}
                for m in markets:
                    so = so_by_event.get(m["event_id"])
                    if so is None:
                        continue
                    mt = (m.get("market_type") or "").lower()
                    if mt == "total":
                        side = (m.get("yes_team") or "").strip().lower()
                        if side == "under":
                            cp = so.true_prob_under
                        elif side == "over":
                            cp = so.true_prob_over
                        else:
                            cp = None
                    else:
                        cp = yes_aligned_close_prob(
                            yes_team=m.get("yes_team"),
                            outcome_a_label=so.outcome_a_label,
                            outcome_b_label=so.outcome_b_label,
                            true_prob_a=so.true_prob_a,
                            true_prob_b=so.true_prob_b,
                            true_prob_draw=so.true_prob_draw,
                        )
                    if cp is None:
                        continue
                    conn.execute(
                        "UPDATE ev_outcomes SET pinnacle_close_prob = ? WHERE market_id = ?",
                        (cp, m["market_id"]),
                    )
                    updated += 1

        conn.commit()
        conn.close()
        return (updated, len(rows), kalshi_captured)

    if once:
        captured, queued, kalshi = asyncio.run(_sweep())
        console.print(
            f"[green]Captured {captured}/{queued} Pinnacle close(s); "
            f"{kalshi} venue snapshot(s).[/green]"
        )
        return

    console.print(
        f"[cyan]watch-closes running[/cyan]  lookahead={lookahead}m  interval={interval}s  "
        f"[dim](Ctrl+C to stop)[/dim]"
    )
    try:
        while True:
            ts = datetime.now().strftime("%H:%M:%S")
            try:
                captured, queued, kalshi = asyncio.run(_sweep())
                if queued or kalshi:
                    console.print(
                        f"[dim]{ts}[/dim]  pinnacle {captured}/{queued}  venue snaps {kalshi}"
                    )
            except Exception as sweep_err:
                console.print(f"[red]{ts}  sweep failed:[/red] {sweep_err}")
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]watch-closes stopped.[/dim]")


def listing_window_markets(
    markets: list,
    market_types: set[str],
    window_hours: int,
    now: "datetime | None" = None,
) -> list:
    """Filter fetched Kalshi markets to the listing→tip capture window.

    Keeps markets whose type is in ``market_types`` (empty set = all) and whose
    event_date falls in [now − 24h, now + window_hours]. The 24h lower bound is
    deliberate slack: Kalshi ticker dates anchor at noon UTC, so a same-evening
    game's event_date sits hours BEFORE the actual tip — a tight `>= now` bound
    would drop today's still-open pre-game markets. Already-settled markets
    never reach here (get_markets fetches status=open only).
    """
    now = now or datetime.now(timezone.utc)
    lo = now - timedelta(hours=24)
    hi = now + timedelta(hours=window_hours)
    out = []
    for m in markets:
        mt = m.market_type.value if hasattr(m.market_type, "value") else str(m.market_type)
        if market_types and mt not in market_types:
            continue
        ed = m.event_date
        if ed is None:
            continue
        if ed.tzinfo is None:
            ed = ed.replace(tzinfo=timezone.utc)
        if lo <= ed <= hi:
            out.append(m)
    return out


def resolve_watch_sectors(sectors: str) -> list[str]:
    """Expand the --sectors option: 'all' → every game sector in SECTOR_SERIES_MAP.

    Prop sectors are excluded — they have no laddered spread/total markets and
    their listing dynamics (props list days early) are a different product.
    Offseason sectors cost one cheap Kalshi series fetch per sweep and archive
    nothing (no open markets in the window), so 'all' is safe year-round; the
    Pinnacle anchor is only fetched for sectors that actually have markets.
    """
    if sectors.strip().lower() == "all":
        from evmax.clients.kalshi import SECTOR_SERIES_MAP
        return [s for s in SECTOR_SERIES_MAP if not s.endswith("_props")]
    return [s.strip().lower() for s in sectors.split(",") if s.strip()]


@app.command("watch-listings")
def watch_listings(
    sectors: str = typer.Option(
        "all", "--sectors", "-s",
        help="Comma-separated sectors to watch, or 'all' for every game sector "
             "(props excluded). Default: all.",
    ),
    market_types: str = typer.Option(
        "spread,total", "--market-types", "-m",
        help="Comma-separated market types to capture (default: spread,total — "
             "the laddered markets judged on CLV; empty string = all types).",
    ),
    window: int = typer.Option(
        72, "--window", "-w",
        help="Capture markets whose event_date is within the next N hours.",
    ),
    interval: int = typer.Option(
        3600, "--interval", "-i",
        help="Seconds to sleep between sweeps (default 1h).",
    ),
    once: bool = typer.Option(
        False, "--once",
        help="Run a single sweep and exit (skip the loop).",
    ),
    log_entries: bool = typer.Option(
        False, "--log-entries",
        help="After archiving, log anchored-entry shadow rows to predictions.db "
             "for --entry-sectors (first sweep with a Pinnacle anchor + EV/depth "
             "gates at the crossable price; see anchored_entry.py).",
    ),
    entry_sectors: str = typer.Option(
        "wnba", "--entry-sectors",
        help="Comma-separated sectors eligible for --log-entries.",
    ),
) -> None:
    """Capture the listing→scan window: prices + order-book DEPTH + sharp anchor.

    WHY (2026-07-01 WNBA spread audit): Kalshi lists spread ladders ~2 days
    pre-tip with MM placeholder quotes (85% have $0 volume/OI at first
    snapshot). The whole harvestable price move (~17pp on the laying side)
    happens between listing and the daily scan, so scan-entry CLV is ~0 by
    construction. This watcher archives, per sweep:

      1. Kalshi market snapshots  → archived_kalshi_markets  (price trail)
      2. Order-book depth metrics → archived_orderbook_depth (fillability)
      3. Devigged Pinnacle odds   → archived_sharp_odds      (as-of EV anchor)

    Together these let the depth-aware entry rule be evaluated offline: the
    first sweep where EV ≥ threshold at the live ask AND ask depth clears a
    floor is the honest "entry" the lay-side CLV promotion gate should score
    (see `cleanup shadow clv --side lay`). Without --log-entries this is pure
    capture — writes archive.db only, never predictions.db, so it cannot
    contaminate the shadow stream. With --log-entries (Track B of the WNBA
    anchored-entry plan), qualifying entries for --entry-sectors ALSO land in
    predictions.db as mode='shadow' rows with captured_yes_price = the
    crossable order-book price, tagged model_sources '+anchored_entry';
    UNIQUE(market_id) freeze-on-first-insert makes the first qualifying sweep
    the entry and later sweeps no-ops.

    Run hourly via a scheduled task / launchd during WNBA season, or --once.
    """
    import asyncio
    import time
    from evmax.archiver import DataArchiver
    from evmax.clients.esports_pinnacle import PinnacleGuestClient
    from evmax.clients.kalshi import KalshiClient

    sector_list = resolve_watch_sectors(sectors)
    type_set = {t.strip().lower() for t in market_types.split(",") if t.strip()}
    entry_set = (
        {s.strip().lower() for s in entry_sectors.split(",") if s.strip()}
        if log_entries else set()
    )

    async def _sweep() -> dict[str, tuple[int, int, int]]:
        """Returns {sector: (markets_archived, depth_rows, sharp_rows)}."""
        session_id = "watchlist-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archiver = DataArchiver()
        stats: dict[str, tuple[int, int, int]] = {}
        for sector in sector_list:
            # 1. Kalshi markets in the listing window → price snapshot
            try:
                async with KalshiClient() as client:
                    markets = await client.get_markets(sector)
                    wanted = listing_window_markets(markets, type_set, window)
                    if not wanted:
                        stats[sector] = (0, 0, 0)
                        continue
                    n_mkts = archiver.archive_kalshi_markets(session_id, sector, wanted)
                    # 2. Order-book depth for the same tickers
                    books = await client.get_market_books_batch(
                        [m.ticker for m in wanted if m.ticker]
                    )
            except Exception as kerr:  # noqa: BLE001 — never abort the sweep
                console.print(f"[yellow]  [{sector}] kalshi sweep failed: {kerr}[/yellow]")
                stats[sector] = (0, 0, 0)
                continue
            depth_rows = [
                {"ticker": t, **m} for t, m in books.items() if m is not None
            ]
            n_depth = archiver.archive_orderbook_depth(session_id, sector, depth_rows)

            # 3. As-of sharp anchor (ML + spread devigged) for EV-at-entry evaluation
            n_sharp = 0
            odds = []
            try:
                async with PinnacleGuestClient() as pclient:
                    odds = await pclient.get_odds(sector)
                if odds:
                    n_sharp = archiver.archive_sharp_odds(session_id, sector, odds)
            except Exception as perr:  # noqa: BLE001
                console.print(f"[yellow]  [{sector}] pinnacle fetch failed: {perr}[/yellow]")

            # 4. Anchored-entry shadow logging (opt-in, never aborts the sweep)
            if sector in entry_set and odds:
                try:
                    from evmax.agents.cleanup.anchored_entry import build_anchored_entries
                    from evmax.agents.cleanup.logger import log_gaps

                    entries = build_anchored_entries(wanted, books, odds, sector)
                    if entries:
                        n_logged = log_gaps(
                            entries, mode_resolver=lambda cat: "shadow",
                        )
                        if n_logged:
                            console.print(
                                f"[green]  [{sector}] anchored entries logged: "
                                f"{n_logged}[/green]"
                            )
                except Exception as eerr:  # noqa: BLE001
                    console.print(
                        f"[yellow]  [{sector}] anchored-entry logging failed: {eerr}[/yellow]"
                    )

            stats[sector] = (n_mkts, n_depth, n_sharp)
        return stats

    def _print_stats(stats: dict[str, tuple[int, int, int]]) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        active = {s: c for s, c in stats.items() if any(c)}
        if not active:
            console.print(f"[dim]{ts}  no open markets in window "
                          f"({len(stats)} sectors swept)[/dim]")
            return
        for sector, (n_mkts, n_depth, n_sharp) in active.items():
            console.print(
                f"[dim]{ts}[/dim]  [{sector}] markets {n_mkts}  "
                f"depth {n_depth}  sharp {n_sharp}"
            )

    if once:
        _print_stats(asyncio.run(_sweep()))
        return

    console.print(
        f"[cyan]watch-listings running[/cyan]  sectors={','.join(sector_list)}  "
        f"types={','.join(sorted(type_set)) or 'all'}  window={window}h  "
        f"interval={interval}s  [dim](Ctrl+C to stop)[/dim]"
    )
    try:
        while True:
            try:
                _print_stats(asyncio.run(_sweep()))
            except Exception as sweep_err:  # noqa: BLE001
                ts = datetime.now().strftime("%H:%M:%S")
                console.print(f"[red]{ts}  sweep failed:[/red] {sweep_err}")
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]watch-listings stopped.[/dim]")


@app.command("listings-eval")
def listings_eval(
    sector: str = typer.Option(
        "wnba", "--sector", "-s",
        help="Sector to evaluate (one at a time — the anchor windows differ).",
    ),
    market_types: str = typer.Option(
        "spread,total", "--market-types", "-m",
        help="Comma-separated laddered market types to evaluate.",
    ),
    ev_min: float = typer.Option(
        2.0, "--ev-min",
        help="Entry gate: edge vs the sharp anchor at the crossable price, in pp.",
    ),
    depth_min: float = typer.Option(
        50.0, "--depth-min",
        help="Entry gate: resting $ at the crossable price level.",
    ),
    since: Optional[str] = typer.Option(
        None, "--since",
        help="Only consider snapshots fetched on/after this ISO date.",
    ),
    detail: bool = typer.Option(
        False, "--detail",
        help="Also print every hypothetical entry, not just the side summary.",
    ),
    session_prefix: list[str] = typer.Option(
        ["watchlist"], "--session-prefix",
        help="Capture stream(s) to replay by archived session_id prefix — "
        "'watchlist' = hourly sweeps (default), 'candlebf' = candlestick "
        "backfill. Repeatable.",
    ),
    bid_from_no_price: bool = typer.Option(
        False, "--bid-from-no-price",
        help="Derive the YES bid as 1 - no_price when a snapshot has no "
        "order-book depth row (needed for candlestick-backfill sessions, "
        "which carry bid/ask but no depth).",
    ),
) -> None:
    """Replay watch-listings captures through the first-anchored-sweep entry rule.

    For each laddered ticker: find the first hourly snapshot with a concurrent
    Pinnacle anchor able to price its line (Kalshi lists ~6-24h BEFORE Pinnacle
    posts, so raw listing time is unanchored noise), enter hypothetically when
    edge >= --ev-min at the crossable price AND depth >= --depth-min, and score
    Kalshi CLV to the last pre-tip snapshot. This is the offline promotion lens
    for laddered markets (`cleanup shadow clv --side lay` is the live one).
    Read-only — never writes to archive.db or predictions.db.
    """
    from evmax.agents.cleanup.listings_eval import evaluate_sector, summarize
    from evmax.archiver import _get_connection as get_archive_conn

    types = tuple(t.strip().lower() for t in market_types.split(",") if t.strip())
    with get_archive_conn() as conn:
        stats = evaluate_sector(
            conn, sector.lower(), ev_min_pp=ev_min, depth_min_usd=depth_min,
            market_types=types, since=since,
            session_prefixes=tuple(session_prefix),
            bid_from_no_price=bid_from_no_price,
        )

    console.print(
        f"[cyan]{sector} anchored-entry eval[/cyan]  ev>={ev_min}pp  depth>=${depth_min:g}  "
        f"tickers={stats.tickers}  never-anchored={stats.never_anchored}  "
        f"anchored-no-entry={stats.anchored_no_entry}  entries={len(stats.entries)}"
    )
    if not stats.entries:
        console.print("[dim]No qualifying entries — accumulate more capture or relax gates.[/dim]")
        return

    if detail:
        dt = Table(title="Hypothetical anchored entries", box=box.SIMPLE)
        dt.add_column("Event", no_wrap=False, min_width=24)
        dt.add_column("Outcome", no_wrap=False)
        dt.add_column("Entry (T-h)", justify="right")
        dt.add_column("Price", justify="right")
        dt.add_column("Depth$", justify="right")
        dt.add_column("Fair", justify="right")
        dt.add_column("Edge", justify="right")
        dt.add_column("Close", justify="right")
        dt.add_column("CLV", justify="right")
        for e in sorted(stats.entries, key=lambda x: x.entry_at):
            clv = f"{e.clv_pp:+.1f}pp" if e.clv_pp is not None else "—"
            close = f"{e.close_price:.2f}" if e.close_price is not None else "—"
            dt.add_row(
                e.event_label, e.outcome_label, f"{e.entry_lead_h:.1f}",
                f"{e.entry_price:.2f}", f"{e.entry_depth_usd:,.0f}",
                f"{e.fair_prob:.2f}", f"{e.edge_pp:+.1f}pp", close, clv,
            )
        console.print(dt)

    st = Table(title=f"{sector} — CLV by market/side (anchored entries)", box=box.SIMPLE)
    st.add_column("Market")
    st.add_column("Side")
    st.add_column("n", justify="right")
    st.add_column("mean edge", justify="right")
    st.add_column("mean CLV", justify="right")
    st.add_column("med CLV", justify="right")
    st.add_column("%CLV>0", justify="right")
    st.add_column("med depth$", justify="right")
    st.add_column("med lead", justify="right")
    for s in summarize(stats):
        st.add_row(
            s.market_type, s.side, str(s.n),
            f"{s.mean_edge_pp:+.1f}pp",
            f"{s.mean_clv_pp:+.2f}pp" if s.mean_clv_pp is not None else "—",
            f"{s.median_clv_pp:+.2f}pp" if s.median_clv_pp is not None else "—",
            f"{s.pct_clv_pos:.0f}%" if s.pct_clv_pos is not None else "—",
            f"{s.median_depth_usd:,.0f}",
            f"{s.median_lead_h:.1f}h",
        )
    console.print(st)


def _stale_unmatched_candidates(conn, cutoff_str: str):
    """Live, unresolved, not-yet-voided predictions before the cutoff.

    The ``mode = 'live'`` guard is load-bearing: without it, ``void`` would
    sweep up stale unresolved SHADOW rows too, shrinking the MODEL-9 promotion
    sample (shadow markets that go unresolved past the cutoff before their ESPN
    result lands would be silently voided out of the validation set).
    """
    return conn.execute(
        """
        SELECT p.market_id, p.event_id, p.sector, p.yes_team,
               p.event_date, p.scan_date, p.ev_pct
        FROM   ev_predictions p
        LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE  p.voided = 0
          AND  p.mode = 'live'
          AND  o.market_id IS NULL
          AND  COALESCE(p.event_date, p.scan_date) < ?
        ORDER  BY COALESCE(p.event_date, p.scan_date) DESC
        """,
        (cutoff_str,),
    ).fetchall()


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

    # Find predictions that are: past the cutoff, still unresolved, not already
    # voided, and live (never void shadow rows — see helper docstring).
    candidates = _stale_unmatched_candidates(conn, cutoff_str)

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


@app.command("value-audit")
def value_audit(
    weeks: int = typer.Option(12, "--weeks", "-w", help="Look-back window in weeks."),
    sector: Optional[str] = typer.Option(None, "--sector", help="Restrict to one sector."),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON (for agents/scripts)."),
) -> None:
    """Per-sector model-blend VALUE audit: Brier vs entry-sharp AND vs close, with
    significance (paired z / 95% CI), CLV, calibration bias, and an actionability verdict.

    Unlike `metrics`, this distinguishes a real model gap from noise: a sector is only
    flagged `actionable` when the blend is significantly worse than the sharp line, or has
    a systematic calibration bias — both fixable in the MODELS, never by gating plays.
    """
    from evmax.agents.cleanup.value_audit import compute_value_audit

    audit = compute_value_audit(weeks=weeks)
    if sector:
        audit = [a for a in audit if a["sector"] == sector.lower()]

    if as_json:
        import json as _json
        console.print_json(_json.dumps(audit))
        return

    if not audit:
        console.print(f"[yellow]No resolved live predictions in the last {weeks} week(s).[/yellow]")
        return

    table = Table(title=f"Model-Blend Value Audit — last {weeks}w", box=box.SIMPLE)
    table.add_column("Sector", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Brier\nModel", justify="right")
    table.add_column("Brier\nSharp", justify="right")
    table.add_column("Brier\nClose", justify="right")
    table.add_column("vs Sharp\n(z)", justify="right")
    table.add_column("vs Close\n(z)", justify="right")
    table.add_column("CLV pp\n(%+)", justify="right")
    table.add_column("Calib\nbias pp", justify="right")
    table.add_column("Calib\nECE pp", justify="right")
    table.add_column("Verdict", style="bold", no_wrap=False)

    def _z(edge):
        if not edge:
            return "—"
        col = "green" if edge["z"] > 0 else "red"
        return f"[{col}]{edge['z']:+.2f}[/{col}]"

    def _bn(v):
        return f"{v:.4f}" if v is not None else "—"

    for a in audit:
        clv = a["clv"]
        clv_str = (f"{clv['mean_pp']:+.2f}\n({clv['frac_positive']*100:.0f}%)"
                   if clv else "—")
        v = a["verdict"]
        vcol = {
            "model_subtracts": "red",
            "calibration_bias": "yellow",
            "adds_value": "green",
            "neutral": "dim",
            "insufficient": "dim",
        }.get(v["tag"], "white")
        verdict_str = f"[{vcol}]{v['tag']}[/{vcol}]" + (" ⚑" if v["actionable"] else "")
        table.add_row(
            a["sector"], str(a["n"]),
            _bn(a["brier_model"]), _bn(a["brier_sharp"]), _bn(a["brier_close"]),
            _z(a["edge_vs_sharp"]), _z(a["edge_vs_close"]),
            clv_str,
            f"{a['calibration']['signed_bias_pp']:+.2f}",
            f"{a['calibration'].get('ece_pp', 0.0):.2f}",
            verdict_str,
        )
    console.print(table)

    actionable = [a for a in audit if a["verdict"]["actionable"]]
    console.print(
        f"\n[bold]{len(actionable)}[/bold] sector(s) with a model-actionable value gap "
        f"(⚑). Positive z = blend beats benchmark; close-Brier near zero is EXPECTED "
        f"(beating the close is rare). CLV is context only — a fine-Brier / negative-CLV "
        f"sector is a timing/selection issue, not a model-blend fix."
    )
    for a in actionable:
        console.print(f"  [yellow]⚑ {a['sector']}[/yellow]: {a['verdict']['reason']}")
        if a["verdict"]["tag"] == "calibration_bias":
            console.print(
                f"     [dim]→ recalibrate: [bold]evmax cleanup recalibrate --sector {a['sector']}[/bold]"
                f"  (leakage-safe alt: scripts/fit_{a['sector']}_calibration.py)[/dim]"
            )


@app.command("recalibrate")
def recalibrate(
    sector: Optional[str] = typer.Option(
        None, "--sector", help="Restrict the refit to one sector (e.g. 'nba')."
    ),
    min_n: int = typer.Option(
        50, "--min-n", help="Minimum resolved LIVE rows per sector before refitting."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report fit quality without writing calibration.json."
    ),
) -> None:
    """Refit per-sector isotonic ensemble calibration from resolved LIVE predictions.

    Fits the ``{sector}_ensemble`` curve the ensemble actually applies at predict
    time, from resolved ``mode='live'`` rows (deduped to the latest scan/market).
    This is the closed-loop fix for a `value-audit` sector tagged `calibration_bias`.

    In-sample post-hoc recalibration: the blended probs were genuine out-of-sample
    forecasts, and the curve is applied FORWARD, so this is not lookahead. For thin
    or early-season sectors, prefer the walk-forward, promotion-gated point-in-time
    scripts (scripts/fit_<sector>_calibration.py), which this command does not replace.
    """
    from evmax.agents.models.calibration import ModelCalibrator

    report = ModelCalibrator().retrain_all_from_db(
        min_per_sector=min_n, sector=sector, dry_run=dry_run
    )
    if not report:
        console.print(
            "[yellow]No resolved live predictions found to recalibrate.[/yellow]"
        )
        return

    table = Table(
        title=("Ensemble recalibration " + ("(DRY RUN — nothing written)" if dry_run else "")),
        box=box.SIMPLE,
    )
    table.add_column("Calibration key", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Brier before", justify="right")
    table.add_column("Brier after", justify="right")
    table.add_column("Δ/1000", justify="right")
    table.add_column("Status", no_wrap=False)

    for key, r in report.items():
        if r["brier_before"] is not None and r["brier_after"] is not None:
            delta = (r["brier_before"] - r["brier_after"]) * 1000.0
            dcol = "green" if delta >= 0 else "red"
            status = ("[green]written[/green]" if r["updated"]
                      else "[dim]not written (dry run)[/dim]")
            table.add_row(
                key, str(r["n"]),
                f"{r['brier_before']:.4f}", f"{r['brier_after']:.4f}",
                f"[{dcol}]{delta:+.2f}[/{dcol}]", status,
            )
        else:
            table.add_row(
                key, str(r["n"]), "—", "—", "—",
                f"[yellow]skipped[/yellow] — {r.get('reason', '')}",
            )
    console.print(table)
    console.print(
        "\n[dim]Brier before/after is IN-SAMPLE (fit quality, not a held-out gain). "
        "The curve is applied to FUTURE predictions via "
        "EnsembleModelAgent._apply_sector_calibration.[/dim]"
    )


@app.command("calibration-alert")
def calibration_alert(
    weeks: int = typer.Option(8, "--weeks", "-w", help="Look-back window in weeks."),
    notify: bool = typer.Option(
        True, "--notify/--no-notify",
        help="Send a Slack/Discord alert when a sector is flagged (no-op if no webhook).",
    ),
) -> None:
    """Tripwire: alert ONLY when a sector shows a real, consistent calibration bias.

    Runs the value-audit and fires an alert for any sector tagged `calibration_bias`
    — which already requires n>=30 resolved rows, a consistent over/under-confidence
    direction, and >=4pp mean signed error. Stays silent otherwise, so it is safe to
    run unattended (launchd com.evmax.calibration-alert). Each flagged sector's fix is
    `evmax cleanup recalibrate --sector <sector>`.
    """
    from evmax.agents.cleanup.value_audit import (
        compute_value_audit, format_calibration_alert,
    )

    audit = compute_value_audit(weeks=weeks)
    flagged = [a for a in audit if a["verdict"]["tag"] == "calibration_bias"]
    msg = format_calibration_alert(flagged)

    if msg is None:
        console.print(
            f"[green]No calibration bias across {len(audit)} sector(s) "
            f"in the last {weeks}w — nothing to alert.[/green]"
        )
        return

    console.print(f"[yellow]{msg}[/yellow]")
    if notify:
        from evmax.notifications import Notifier
        notifier = Notifier.from_settings()
        if notifier.is_configured():
            notifier.send_text(msg)
            console.print("[dim]Alert sent to configured webhook(s).[/dim]")
        else:
            console.print(
                "[dim]No webhook configured (SLACK_WEBHOOK_URL / DISCORD_WEBHOOK_URL) "
                "— printed only.[/dim]"
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
    sectors: str = typer.Option(
        "nba,nfl,baseball",
        "--sectors",
        "-s",
        help="Comma-separated sectors to resolve (nba/nfl via ESPN, baseball via MLB Stats API)",
    ),
) -> None:
    """Fill in prop_observations actual values + outcomes.

    NBA/NFL stats come from ESPN boxscores; baseball comes from the MLB Stats
    API via the baseball props cache. Resolution is mode-agnostic — shadow rows
    resolve too, since validation needs their outcomes.
    """
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


@app.command("backfill-clv")
def backfill_clv_cmd(
    since: str = typer.Option(
        None, "--since", help="YYYY-MM-DD start date (default: all time)."
    ),
    until: str = typer.Option(
        None, "--until", help="YYYY-MM-DD end date (default: today)."
    ),
) -> None:
    """Backfill CLV metrics for resolved bets.

    Populates two columns on ev_predictions:
      - pinnacle_drift_pct  (Pinnacle pre-tipoff close vs Kalshi entry)
                            Diagnostic only — biased positive for our system.
      - kalshi_clv_pct      (Kalshi T-30-min snapshot vs Kalshi entry)
                            Primary edge signal — Pinnacle-independent.
    """
    from datetime import date as _date
    from evmax.agents.cleanup.resolver import backfill_clv

    since_date = _date.fromisoformat(since) if since else None
    until_date = _date.fromisoformat(until) if until else None

    result = backfill_clv(since=since_date, until=until_date)
    updated = result["updated"]
    skipped = result["skipped"]
    avg_pd = result.get("avg_pinn_drift", 0.0)
    avg_kc = result.get("avg_kalshi_clv", 0.0)
    n_pd = result.get("n_pinn", 0)
    n_kc = result.get("n_kalshi", 0)

    if updated:
        kc_color = "green" if avg_kc > 0 else "red" if avg_kc < 0 else "dim"
        pd_color = "yellow"  # always dim — selection-biased metric
        console.print(
            f"\n  Backfilled CLV for [bold]{updated}[/bold] bet(s)  "
            f"(skipped: {skipped})"
        )
        console.print(
            f"  [{kc_color}]Kalshi-CLV    {avg_kc:+.2f}pp[/{kc_color}]  "
            f"(n={n_kc})  ← primary edge signal"
        )
        console.print(
            f"  [{pd_color}]Pinnacle drift {avg_pd:+.2f}pp[/{pd_color}]  "
            f"(n={n_pd})  ← diagnostic only (selection-biased)"
        )
    else:
        console.print("[dim]No resolved bets found missing CLV.[/dim]")
