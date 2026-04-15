"""CLI commands for the agent pipeline and model management.

Commands:
  evmax agents scan  — run the agent coordinator for a single cycle and print gaps
  evmax agents seed  — seed Elo / Poisson / Form models from a JSON file
  evmax agents ratings — display current Elo ratings for a sector
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(no_args_is_help=True)
console = Console()


def _display_label(yes_team: str, market_type: str, line) -> str:
    """Mirror EVGap.display_label: team + market type + line."""
    team = (yes_team or "?").capitalize()
    mt = (market_type or "").lower()
    if mt == "moneyline":
        return f"{team} ML"
    if mt == "spread" and line is not None:
        line_str = f"{line:.1f}".rstrip("0").rstrip(".")
        return f"{team} {line_str}"
    if mt in ("over_under", "total") and line is not None:
        return f"O/U {line:.1f}"
    return team


def _american(prob: float) -> str:
    """Convert implied probability to American odds string (+185, -108)."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        return f"{-round(prob / (1 - prob) * 100)}"
    return f"+{round((1 - prob) / prob * 100)}"


async def _scan_loop(
    coordinator,
    sector_list: list[str],
    min_ev: float,
    min_prob: float,
    top: int,
    bankroll: float,
    kelly: float,
    date_filter,
    sharp_weight: float,
) -> None:
    """Run agent scan continuously with adaptive intervals."""
    cycle = 0
    while True:
        cycle += 1
        console.print(f"\n[dim]--- Cycle #{cycle} ---[/dim]")
        try:
            result = await coordinator.run_cycle()
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped.[/yellow]")
            return
        except Exception as e:
            console.print(f"[red]Cycle error: {e}[/red]")
            await asyncio.sleep(60)
            continue

        # Compute adaptive interval from EV gaps' event dates
        interval = coordinator.next_scan_interval_seconds(result)

        if interval <= 90:
            interval_label = "[bold red]90s (LIVE)[/bold red]"
        elif interval <= 180:
            interval_label = f"[yellow]{interval}s (<1h to kickoff)[/yellow]"
        elif interval <= 600:
            interval_label = f"[cyan]{interval}s (1–4h to kickoff)[/cyan]"
        else:
            interval_label = f"[dim]{interval}s (>4h / no games)[/dim]"

        console.print(f"[dim]Next scan in[/dim] {interval_label}[dim]. Ctrl+C to stop.[/dim]")
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            console.print("\n[yellow]Stopped.[/yellow]")
            return


@app.command("scan")
def scan(
    sectors: str = typer.Option(
        "nba,ncaab,ncaaw,soccer,lol,cs2,tennis,baseball",
        "--sectors",
        "-s",
        help="Comma-separated sector list, e.g. 'nba,soccer'",
    ),
    no_models: bool = typer.Option(False, "--no-models", help="Skip model agents (sharp probs only)."),
    no_injuries: bool = typer.Option(False, "--no-injuries", help="Skip injury report agent."),
    sharp_weight: float = typer.Option(0.85, "--sharp-weight", help="Weight for Pinnacle in ensemble blend."),
    bankroll: float = typer.Option(250.0, "--bankroll", "-b", help="Current bankroll in USD."),
    kelly: float = typer.Option(0.5, "--kelly", "-k", help="Kelly fraction (0.5=half, 0.25=quarter)."),
    min_ev: float = typer.Option(0.02, "--min-ev", help="Base minimum EV threshold (scaled up automatically for low-prob bets)."),
    min_prob: float = typer.Option(0.15, "--min-prob", help="Minimum true probability floor. Bets below this are excluded regardless of EV."),
    top: int = typer.Option(25, "--top", "-n", help="Max plays to show."),
    max_props: int = typer.Option(10, "--max-props", help="Max player prop plays to show (prevents prop spam)."),
    date_filter: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Only show games on this date (YYYY-MM-DD). Defaults to today.",
    ),
    loop: bool = typer.Option(
        False, "--loop",
        help="Run continuously with smart adaptive scan intervals based on game times.",
    ),
    live_override: Optional[str] = typer.Option(
        None, "--live",
        help="Comma-separated category keys to force into LIVE mode for this run "
             "(e.g. 'nfl_props'). Overrides data/categories.yaml.",
    ),
    shadow_override: Optional[str] = typer.Option(
        None, "--shadow",
        help="Comma-separated category keys to force into SHADOW mode for this run. "
             "Logged with mode='shadow', does NOT touch bankroll.",
    ),
    disabled_override: Optional[str] = typer.Option(
        None, "--disabled",
        help="Comma-separated category keys to force into DISABLED mode for this run. "
             "Scanner skips persistence entirely for these categories.",
    ),
) -> None:
    """Run the full agent pipeline for one cycle and display +EV plays."""
    from evmax.agents.coordinator import AgentCoordinator
    from evmax.settings import get_settings as _get_settings
    from evmax.sectors.registry import ALL_SECTORS

    sector_list = [s.strip().lower() for s in sectors.split(",") if s.strip()]

    # Validate sector names upfront
    invalid = [s for s in sector_list if s not in ALL_SECTORS]
    if invalid:
        console.print(f"[red]Unknown sector(s):[/red] {', '.join(invalid)}")
        console.print(f"[dim]Valid sectors: {', '.join(ALL_SECTORS)}[/dim]")
        raise typer.Exit(1)

    # ARCH-11 category-mode overrides. Compose a single runtime-override
    # dict and install it before any scan output or logging happens so
    # every downstream call to evmax.modes.get_mode sees the right mode.
    mode_overrides: dict[str, str] = {}
    for raw, target_mode in (
        (live_override, "live"),
        (shadow_override, "shadow"),
        (disabled_override, "disabled"),
    ):
        if not raw:
            continue
        for key in (k.strip() for k in raw.split(",") if k.strip()):
            if key in mode_overrides:
                console.print(
                    f"[red]Category {key!r} appears in more than one of "
                    f"--live / --shadow / --disabled.[/red]"
                )
                raise typer.Exit(1)
            mode_overrides[key] = target_mode
    if mode_overrides:
        from evmax.modes import set_runtime_overrides
        set_runtime_overrides(mode_overrides)
        override_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(mode_overrides.items())
        )
        console.print(
            f"[yellow]Mode overrides for this run:[/yellow] {override_summary}"
        )

    # Warn on missing API keys before spinning up the pipeline
    _missing_keys = _get_settings().warn_missing_keys()
    for _key_warn in _missing_keys:
        console.print(f"[yellow]⚠ Missing:[/yellow] {_key_warn}")

    # Read persisted sharp_weight from model_config.json (overrides CLI default 0.85)
    from evmax.agents.cleanup.metrics import load_config as _load_cfg
    _cfg = _load_cfg()
    if _cfg.get("sharp_weight") and sharp_weight == 0.85:
        sharp_weight = _cfg["sharp_weight"]

    coordinator = AgentCoordinator(
        sectors=sector_list,
        enable_models=not no_models,
        sharp_weight=sharp_weight,
        enable_injuries=not no_injuries,
        bankroll=bankroll,
        kelly_fraction=kelly,
    )

    console.print(f"\n[bold cyan]evmax agent scan[/bold cyan] — sectors: {', '.join(sector_list)}\n")

    if loop:
        asyncio.run(_scan_loop(coordinator, sector_list, min_ev, min_prob, top, bankroll, kelly, date_filter, sharp_weight))
        return


    result = asyncio.run(coordinator.run_cycle())

    # Run maintenance checks after logging
    try:
        from evmax.agents.cleanup.maintenance import run_maintenance, MaintenanceReport
        maint = run_maintenance()
        console.print(f"[dim]  {maint.summary()}[/dim]")
        if maint.errors:
            for issue in maint.errors:
                console.print(f"[red]  [MAINT ERROR][/red] [{issue.check}] {issue.event_id}: {issue.detail}")
                if issue.fix:
                    console.print(f"[dim]               → {issue.fix}[/dim]")
        if maint.warnings:
            for issue in maint.warnings:
                console.print(f"[yellow]  [MAINT WARN][/yellow]  [{issue.check}] {issue.event_id}: {issue.detail}")
                if issue.fix:
                    console.print(f"[dim]               → {issue.fix}[/dim]")
    except Exception as _maint_err:
        console.print(f"[dim yellow]  Warning: maintenance check failed: {_maint_err}[/dim yellow]")

    # Parse date filter (default: earliest date with gaps, starting from today)
    if date_filter:
        try:
            target_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[red]Invalid date format:[/red] {date_filter!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        target_date = date.today()

    def _matches_date(g) -> bool:
        if g.event_date is None:
            return True  # no date info — include by default
        # datetime subclasses date, so always call .date() to strip time/tz
        ed = g.event_date.date() if hasattr(g.event_date, "date") else g.event_date
        return ed == target_date

    def _tiered_min_ev(true_prob: float) -> float:
        """Scale minimum EV up for low-probability bets.

        Formula: base_min_ev + max(0, min_prob_floor - true_prob) * 0.5
        Examples (base=2%, floor=15%):
          true_prob=0.08 → min_ev = 2% + (0.15-0.08)*0.5 = 5.5%
          true_prob=0.12 → min_ev = 2% + (0.15-0.12)*0.5 = 3.5%
          true_prob=0.15 → min_ev = 2% (floor, no scaling)
          true_prob=0.50 → min_ev = 2% (unchanged)
        """
        return min_ev + max(0.0, min_prob - true_prob) * 0.5

    # All gaps that pass display filters (no top-N cap yet) — these are what get logged.
    # Logging uses the same date + min_prob + tiered EV as the display so that
    # verify only shows bets the user actually saw (or would have seen) in the scan.
    qualifying_gaps = [
        g for g in result.top_gaps
        if g.blended_true_prob >= min_prob
        and g.ev_pct >= _tiered_min_ev(g.blended_true_prob)
        and _matches_date(g)
    ]

    # Exclude events that have already been placed via pick
    try:
        from evmax.agents.cleanup.db import get_connection as _get_conn
        _pconn = _get_conn()
        _placed_rows = _pconn.execute(
            "SELECT DISTINCT event_title FROM ev_predictions "
            "WHERE placed = 1 AND event_date = ? AND mode = 'live'",
            (str(target_date),),
        ).fetchall()
        _pconn.close()
        _placed_events = {r["event_title"] for r in _placed_rows if r["event_title"]}
        if _placed_events:
            qualifying_gaps = [g for g in qualifying_gaps if g.event_title not in _placed_events]
    except Exception:
        pass  # DB unavailable — show all

    # Log game-level +EV gaps to ev_predictions (bankroll tracking)
    loggable_gaps = [g for g in qualifying_gaps if "::prop::" not in g.event_id]
    if loggable_gaps:
        try:
            from evmax.agents.cleanup.logger import log_gaps as _log_gaps
            n_logged = _log_gaps(loggable_gaps, sharp_weight_used=sharp_weight, bankroll_used=bankroll)
            if n_logged:
                console.print(f"[dim]  Logged {n_logged} new prediction(s) to predictions.db[/dim]")
        except Exception as _log_err:
            console.print(f"[bold red]  ERROR: Failed to log predictions to DB: {_log_err}[/bold red]")
            console.print(f"[red]  Plays above were NOT saved — resolve will not find them.[/red]")

    # Log ALL prop lines (not just +EV) to prop_observations for model training
    all_prop_gaps = [g for g in result.top_gaps if "::prop::" in g.event_id]
    if all_prop_gaps:
        try:
            from evmax.agents.cleanup.logger import log_prop_observations as _log_props
            n_props = _log_props(all_prop_gaps)
            if n_props:
                console.print(f"[dim]  Logged {n_props} prop line(s) to prop_observations[/dim]")
        except Exception as _log_err:
            logger.warning("prop_log_failed", error=str(_log_err))

    # Enforce per-type cap: at most max_props prop plays, rest are game markets
    prop_gaps = [g for g in qualifying_gaps if g.market_type == "player_prop"]
    game_gaps = [g for g in qualifying_gaps if g.market_type != "player_prop"]
    gaps = (game_gaps + prop_gaps[:max_props])[:top]

    # Print injury summary — only for teams involved in the displayed plays
    if result.injury_reports and gaps:
        # Build set of team name fragments from event titles + yes_team
        teams_in_plays: set[str] = set()
        for g in gaps:
            teams_in_plays.add(g.yes_team.lower().strip())
            # event_title format: "Team A vs Team B"
            for part in g.event_title.lower().replace(" vs ", " ").split():
                if len(part) > 3:
                    teams_in_plays.add(part)

        inj_lines = []
        for sector, team_reports in result.injury_reports.items():
            sig = [
                r for r in team_reports.values()
                if r.has_significant_injuries
                and any(t in r.team or r.team in t for t in teams_in_plays)
            ]
            if sig:
                inj_lines.append(f"[bold]{sector.upper()}[/bold]: " + ", ".join(
                    f"{r.team} ({len(r.players)} out/dtd, adj={r.probability_adjustment:+.1%})"
                    for r in sig
                ))
        if inj_lines:
            console.print("\n[bold yellow]Injury Impact:[/bold yellow]")
            for line in inj_lines:
                console.print(f"  {line}")

    if not gaps:
        console.print(f"\n[yellow]No +EV plays found at EV >= {min_ev*100:.0f}% threshold.[/yellow]")
        console.print(f"Scanned {result.markets_fetched} markets, matched {result.markets_matched}.")
        if result.errors:
            console.print(f"[red]Errors:[/red] {', '.join(result.errors)}")
        return

    table = Table(
        title=(
            f"+EV Plays — {len(gaps)} found | {target_date} | Bankroll ${bankroll:.0f} | {kelly:.0%} Kelly"
            f" | min prob {min_prob:.0%} | base EV {min_ev:.0%} (tiered)"
        ),
        box=box.ROUNDED,
        show_lines=True,
        expand=False,
    )
    table.add_column("#", style="dim", width=2)
    table.add_column("Sec", style="dim", width=5)
    table.add_column("Event", style="dim", no_wrap=False, max_width=26)
    table.add_column("Outcome", style="bold white", no_wrap=False, max_width=20)
    table.add_column("Ask", justify="right", width=6)
    table.add_column("Fair", justify="right", width=6)
    table.add_column("Edge", justify="right", width=5)
    table.add_column("EV%", justify="right", style="green bold", width=6)
    table.add_column("K%", justify="right", width=6)
    table.add_column("Stake", justify="right", style="cyan bold", width=7)
    table.add_column("N", justify="right", width=3)
    table.add_column("Bk", justify="right", width=2)
    table.add_column("Stm", justify="center", width=5)
    table.add_column("Vol", justify="right", width=8)
    table.add_column("Cf", justify="center", width=4)

    total_stake = 0.0
    for i, gap in enumerate(gaps, 1):
        stake = result.stake_for(gap)
        total_stake += stake
        is_prop = gap.market_type == "player_prop"
        # Flag suspiciously high EV on props (likely small sample / model artifact)
        ev_suspicious = is_prop and gap.ev_pct > 0.30
        ev_color = (
            "bold red" if ev_suspicious
            else "bold green" if gap.ev_pct >= 0.10
            else "green" if gap.ev_pct >= 0.05
            else "yellow"
        )
        ask_odds = _american(gap.kalshi_yes_price)
        fair_odds = _american(gap.blended_true_prob)
        # Edge in cents: how much cheaper the ask is vs fair value
        # e.g. ask=42¢, fair=48¢ → edge=6¢ — you can pay up to 48¢ and still be +EV
        edge_cents = round((gap.blended_true_prob - gap.kalshi_yes_price) * 100)
        edge_color = "green" if edge_cents >= 3 else "yellow"
        odds_ok = gap.blended_true_prob >= gap.kalshi_yes_price * 1.02
        odds_color = "green" if odds_ok else "red"
        ev_str = f"[{ev_color}]{gap.ev_pct*100:+.1f}%{'?' if ev_suspicious else ''}[/{ev_color}]"
        l15_str = str(gap.prop_l15_games) if is_prop and gap.prop_l15_games else "[dim]—[/dim]"
        bks = getattr(gap, "book_count", 1)
        bks_str = f"[green]{bks}[/green]" if bks > 1 else f"[dim]{bks}[/dim]"
        # Line velocity / steam flag
        vel_flag = getattr(gap, "velocity_flag", None)
        if vel_flag == "STEAM":
            steam_str = "[bold red]⚡STEAM[/bold red]"
        elif vel_flag == "STALE":
            steam_str = "[dim]STALE[/dim]"
        else:
            steam_str = "[dim]—[/dim]"
        table.add_row(
            str(i),
            gap.sector.upper(),
            gap.event_title[:28],
            gap.display_label[:22],
            f"[{odds_color}]{ask_odds}[/{odds_color}]",
            f"[bold]{fair_odds}[/bold]",
            f"[{edge_color}]{edge_cents:+d}¢[/{edge_color}]",
            ev_str,
            f"{gap.kelly_fraction*100:.2f}%",
            f"${stake:.2f}",
            l15_str,
            bks_str,
            steam_str,
            f"${gap.volume_usd:,.0f}",
            gap.stars_display,
        )

    console.print(f"\n[bold cyan]evmax agent scan — {', '.join(sector_list).upper()}[/bold cyan]\n")
    console.print(table)
    guard_note = (
        f"  [yellow]⚠ {result.exposure_guard_dropped} play(s) dropped/capped by exposure guard "
        f"(>8% per game)[/yellow]" if result.exposure_guard_dropped else ""
    )
    # TheOddsAPI quota
    from evmax.clients.pinnacle import PinnacleClient
    quota = PinnacleClient.get_quota()
    quota_str = ""
    if quota["remaining"] is not None:
        remaining = quota["remaining"]
        used = quota["used"] or 0
        color = "green" if remaining > 100 else "yellow" if remaining > 25 else "red"
        quota_str = f"  |  TheOddsAPI: [{color}]{remaining:,} remaining[/{color}] ({used:,} used)"

    console.print(
        f"\n  [bold]Total at risk:[/bold] ${total_stake:.2f} / ${bankroll:.0f} "
        f"({total_stake/bankroll*100:.1f}%)  |  "
        f"Matched {result.markets_matched}/{result.markets_fetched} markets"
        + quota_str
        + (f"\n{guard_note}" if guard_note else "") + "\n"
    )

    if result.errors:
        console.print(f"[red]Errors:[/red] {', '.join(result.errors)}")


@app.command("verify")
def verify(
    date_filter: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Check bets logged on this date (YYYY-MM-DD). Defaults to today.",
    ),
    min_ev: float = typer.Option(0.02, "--min-ev", help="Base EV threshold for 'still live' check."),
    min_prob: float = typer.Option(0.15, "--min-prob", help="Minimum true probability floor."),
    bankroll: Optional[float] = typer.Option(None, "--bankroll", "-b", help="Bankroll for stake re-calc (default: value used at scan time)."),
    kelly: float = typer.Option(0.5, "--kelly", "-k", help="Kelly fraction."),
) -> None:
    """Re-fetch live Kalshi ask prices and check which +EV bets are still actionable."""
    from evmax.agents.cleanup.db import get_connection
    from evmax.clients.kalshi import KalshiClient
    from evmax.ev.calculator import calculate_ev
    from evmax.ev.kelly import compute_kelly

    if date_filter:
        try:
            target_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[red]Invalid date format:[/red] {date_filter!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        target_date = date.today()

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT p.id, p.market_id, p.event_title, p.yes_team, p.sector,
               p.market_type, p.kalshi_yes_price, p.sharp_true_prob,
               p.blended_true_prob, p.ev_pct, p.kelly_fraction,
               p.volume_usd, p.model_sources, p.line, p.bankroll_used
        FROM ev_predictions p
        INNER JOIN (
            SELECT market_id, MAX(scan_date) AS latest_scan
            FROM ev_predictions WHERE voided = 0 AND mode = 'live' GROUP BY market_id
        ) latest ON p.market_id = latest.market_id AND p.scan_date = latest.latest_scan
        LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE (p.event_date = ? OR (p.event_date IS NULL AND p.scan_date = ?))
          AND p.voided = 0
          AND p.mode = 'live'
          AND (o.outcome IS NULL OR o.id IS NULL)
        ORDER BY p.ev_pct DESC
        """,
        (str(target_date), str(target_date)),
    ).fetchall()
    conn.close()

    if not rows:
        console.print(f"[yellow]No open (unresolved, non-voided) predictions for games on {target_date}.[/yellow]")
        return

    # Resolve bankroll: CLI override > stored scan value > fallback 250
    stored_bankroll = next((dict(r)["bankroll_used"] for r in rows if dict(r).get("bankroll_used")), None)
    effective_bankroll = bankroll if bankroll is not None else (stored_bankroll or 250.0)

    console.print(f"\n[bold cyan]evmax verify[/bold cyan] — re-checking {len(rows)} bets for games on {target_date} ...\n")

    def _tiered_min_ev(true_prob: float) -> float:
        return min_ev + max(0.0, min_prob - true_prob) * 0.5

    # Fetch all live ask prices via WebSocket (one connection for all tickers)
    # with automatic REST fallback for any ticker missed by WS.
    async def _fetch_asks(tickers: list[str]) -> dict[str, Optional[float]]:
        async with KalshiClient() as client:
            return await client.get_market_asks_batch(tickers)

    tickers = [dict(r)["market_id"] for r in rows]
    live_prices = asyncio.run(_fetch_asks(tickers))

    bankroll = effective_bankroll
    table = Table(
        title=f"Live Price Check — {target_date} | Bankroll ${bankroll:.0f} | {kelly:.0%} Kelly",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Sector", style="dim", width=6)
    table.add_column("Event", style="dim", min_width=20)
    table.add_column("Outcome", style="bold white", min_width=18)
    table.add_column("Scan Ask", justify="right", width=9)
    table.add_column("Live Ask", justify="right", width=9)
    table.add_column("Δ Price", justify="right", width=8)
    table.add_column("True P", justify="right", width=7)
    table.add_column("Scan EV%", justify="right", width=9)
    table.add_column("Live EV%", justify="right", style="bold", width=9)
    table.add_column("Stake $", justify="right", style="cyan", width=8)
    table.add_column("Status", justify="center", width=10)

    still_live = 0
    total_stake = 0.0

    for i, row in enumerate(rows, 1):
        r = dict(row)
        market_id = r["market_id"]
        blended_prob = r["blended_true_prob"]
        scan_ask = r["kalshi_yes_price"]
        live_ask = live_prices.get(market_id)

        if live_ask is None:
            # Settled or fetch error
            table.add_row(
                str(i), r["sector"].upper(),
                (r["event_title"] or "")[:22],
                _display_label(r["yes_team"], r["market_type"], r["line"])[:20],
                f"{scan_ask:.3f}", "—", "—",
                f"{blended_prob:.3f}",
                f"{r['ev_pct']*100:+.1f}%", "—",
                "—", "[dim]settled/err[/dim]",
            )
            continue

        # 99¢ asks = empty orderbook (no real liquidity), skip entirely
        if live_ask >= 0.99:
            table.add_row(
                str(i), r["sector"].upper(),
                (r["event_title"] or "")[:22],
                _display_label(r["yes_team"], r["market_type"], r["line"])[:20],
                f"{scan_ask:.3f}", f"{live_ask:.3f}", "—",
                f"{blended_prob:.3f}",
                "—", "—",
                "—", "[dim]no book[/dim]",
            )
            continue

        live_ev, _ = calculate_ev(live_ask, blended_prob)
        threshold = _tiered_min_ev(blended_prob)
        is_live = live_ev >= threshold and blended_prob >= min_prob

        # Kelly stake at live price
        if is_live:
            from evmax.settings import get_settings
            settings = get_settings()
            payout = 1.0 / live_ask
            k = compute_kelly(
                true_prob=blended_prob,
                payout_decimal=payout,
                edge_pct=live_ev,
                spread_pct=0.0,
                base_fraction=kelly,
                max_kelly=settings.max_kelly_fraction,
            )
            stake = bankroll * k.kelly_fraction
            total_stake += stake
            stake_str = f"${stake:.2f}"
            still_live += 1
        else:
            stake_str = "—"

        price_delta = live_ask - scan_ask
        delta_color = "red" if price_delta > 0.01 else "green" if price_delta < -0.01 else "dim"
        ev_color = "bold green" if is_live and live_ev >= 0.10 else "green" if is_live else "red"
        status = "[green]LIVE[/green]" if is_live else "[red]STALE[/red]"

        table.add_row(
            str(i), r["sector"].upper(),
            (r["event_title"] or "")[:22],
            _display_label(r["yes_team"], r["market_type"], r["line"])[:20],
            f"{scan_ask:.3f}",
            f"[{delta_color}]{live_ask:.3f}[/{delta_color}]",
            f"[{delta_color}]{price_delta:+.3f}[/{delta_color}]",
            f"{blended_prob:.3f}",
            f"{r['ev_pct']*100:+.1f}%",
            f"[{ev_color}]{live_ev*100:+.1f}%[/{ev_color}]",
            stake_str,
            status,
        )

    console.print(table)
    console.print(
        f"\n  [bold]{still_live}[/bold] of {len(rows)} bets still +EV at live prices  |  "
        f"Total stake if all placed: [cyan]${total_stake:.2f}[/cyan]\n"
    )


@app.command("pick")
def pick(
    date_filter: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Check bets logged on this date (YYYY-MM-DD). Defaults to today.",
    ),
    sectors: Optional[str] = typer.Option(None, "--sectors", "-s", help="Filter by sectors (comma-separated, e.g. 'nba,tennis'). Default: all."),
    min_ev: float = typer.Option(0.02, "--min-ev", help="Base EV threshold for 'still live' check."),
    min_prob: float = typer.Option(0.15, "--min-prob", help="Minimum true probability floor."),
    bankroll: Optional[float] = typer.Option(None, "--bankroll", "-b", help="Bankroll for stake calculation (default: value used at scan time)."),
    kelly: float = typer.Option(0.5, "--kelly", "-k", help="Kelly fraction."),
    show_stale: bool = typer.Option(False, "--show-stale", help="Also show bets whose edge has evaporated."),
) -> None:
    """Interactively select which +EV bets you're placing. Records placed bets in the database."""
    from evmax.agents.cleanup.db import get_connection
    from evmax.clients.kalshi import KalshiClient
    from evmax.ev.calculator import calculate_ev
    from evmax.ev.kelly import compute_kelly
    from evmax.settings import get_settings
    import questionary
    from datetime import datetime as _dt

    if date_filter:
        try:
            target_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[red]Invalid date format:[/red] {date_filter!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        target_date = date.today()

    # Parse sector filter
    sector_filter = None
    if sectors:
        sector_filter = [s.strip().lower() for s in sectors.split(",") if s.strip()]

    conn = get_connection()
    query = """
        SELECT p.id, p.market_id, p.event_id, p.event_title, p.yes_team, p.sector,
               p.market_type, p.kalshi_yes_price, p.sharp_true_prob,
               p.blended_true_prob, p.ev_pct, p.kelly_fraction,
               p.volume_usd, p.model_sources, p.line,
               p.placed, p.placed_price, p.placed_stake, p.bankroll_used
        FROM ev_predictions p
        INNER JOIN (
            SELECT market_id, MAX(scan_date) AS latest_scan
            FROM ev_predictions WHERE voided = 0 AND mode = 'live' GROUP BY market_id
        ) latest ON p.market_id = latest.market_id AND p.scan_date = latest.latest_scan
        LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE (p.event_date = ? OR (p.event_date IS NULL AND p.scan_date = ?))
          AND p.voided = 0
          AND p.mode = 'live'
          AND (o.outcome IS NULL OR o.id IS NULL)
        ORDER BY p.ev_pct DESC
    """
    params: list = [str(target_date), str(target_date)]
    all_rows = conn.execute(query, params).fetchall()
    conn.close()

    # Build set of events that have ANY placed bet — exclude all rows for those events
    placed_events: set[str] = set()
    for r in all_rows:
        d = dict(r)
        if d["placed"]:
            # Key by event: sector + event_title (covers different market_ids for same game)
            placed_events.add(f"{d['sector']}::{d['event_title']}")

    rows = [r for r in all_rows if f"{dict(r)['sector']}::{dict(r)['event_title']}" not in placed_events and not dict(r)["placed"]]

    # Deduplicate by event+outcome (same game can have multiple market_ids across scans)
    seen_keys: set[str] = set()
    deduped_rows = []
    for r in rows:
        d = dict(r)
        dedup_key = f"{d['sector']}::{d['event_title']}::{d['yes_team']}::{d['market_type']}::{d.get('line')}"
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            deduped_rows.append(r)
    rows = deduped_rows

    # Apply sector filter
    if sector_filter:
        rows = [r for r in rows if dict(r)["sector"] in sector_filter]

    if not rows:
        console.print(f"[yellow]No open predictions for games on {target_date}.[/yellow]")
        console.print(f"[dim]Tip: scan_date is when the scan ran; --date filters by game date.[/dim]")
        return

    # Resolve bankroll: CLI override > stored scan value > fallback 250
    stored_bankroll = next((dict(r)["bankroll_used"] for r in rows if dict(r).get("bankroll_used")), None)
    bankroll = bankroll if bankroll is not None else (stored_bankroll or 250.0)

    console.print(f"\n[bold cyan]evmax pick[/bold cyan] — {len(rows)} bets from scan\n")

    def _tiered_min_ev(true_prob: float) -> float:
        return min_ev + max(0.0, min_prob - true_prob) * 0.5

    settings = get_settings()

    # Build enriched bet list using scan-time prices (no live re-fetch)
    bets = []
    for r in [dict(r) for r in rows]:
        blended_prob = r["blended_true_prob"]
        scan_ask = r["kalshi_yes_price"]

        scan_stake = bankroll * r["kelly_fraction"]
        threshold = _tiered_min_ev(blended_prob)
        is_live = r["ev_pct"] >= threshold and blended_prob >= min_prob

        bets.append({
            **r,
            "live_ask": scan_ask,
            "live_ev": r["ev_pct"],
            "is_live": is_live,
            "stake": scan_stake,
        })

    def _display_label(yes_team, market_type, line):
        team = (yes_team or "?").capitalize()
        if market_type == "moneyline":
            return f"{team} ML"
        if market_type == "spread" and line is not None:
            line_str = f"{line:.1f}".rstrip("0").rstrip(".")
            return f"{team} {line_str}"
        if market_type in ("over_under", "total") and line is not None:
            return f"O/U {line:.1f}"
        return team

    # Show a summary table first
    table = Table(
        title=f"Live Bets — {target_date} | Bankroll ${bankroll:.0f} | {kelly:.0%} Kelly",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Sector", style="dim", width=6)
    table.add_column("Event", style="dim", min_width=20)
    table.add_column("Outcome", style="bold white", min_width=16)
    table.add_column("Ask", justify="right", width=7)
    table.add_column("Fair", justify="right", width=7)
    table.add_column("Edge", justify="right", width=7)
    table.add_column("EV%", justify="right", width=8)
    table.add_column("Stake $", justify="right", style="cyan", width=8)
    table.add_column("Status", justify="center", width=10)

    displayable = [b for b in bets if b["is_live"] or show_stale]

    for i, b in enumerate(displayable, 1):
        ev_color = "bold green" if b["live_ev"] >= 0.10 else "green" if b["is_live"] else "red dim"
        live_ask_american = _american(b["live_ask"]) if b["live_ask"] else "—"
        fair_american = _american(b["blended_true_prob"])
        # Edge in cents (ask prob vs fair prob)
        edge_cents = f"{(b['blended_true_prob'] - (b['live_ask'] or b['kalshi_yes_price'])) * 100:+.0f}\u00a2" if b["live_ask"] else "—"
        if b["is_live"]:
            status = "[green]LIVE[/green]"
        else:
            status = "[red]STALE[/red]"
        table.add_row(
            str(i),
            b["sector"].upper(),
            (b["event_title"] or "")[:22],
            _display_label(b["yes_team"], b["market_type"], b["line"])[:18],
            live_ask_american,
            fair_american,
            edge_cents,
            f"[{ev_color}]{b['live_ev']*100:+.1f}%[/{ev_color}]",
            f"${b['stake']:.2f}",
            status,
        )

    console.print(table)

    live_bets = [b for b in displayable if b["is_live"]]
    if not live_bets:
        console.print("\n[yellow]No live bets to select.[/yellow]\n")
        return

    # Build questionary choices — only show live, unplaced bets
    choices = []
    for b in live_bets:
        label = _display_label(b["yes_team"], b["market_type"], b["line"])
        event = (b["event_title"] or b["yes_team"] or "?")[:30]
        ask_odds = _american(b["live_ask"]) if b["live_ask"] else "?"
        fair_odds = _american(b["blended_true_prob"])
        choices.append(questionary.Choice(
            title=f"{b['sector'].upper():6s} | {event:30s} | {label:18s} | Ask {ask_odds:>5s}  Fair {fair_odds:>5s}  EV={b['live_ev']*100:+.1f}%  ${b['stake']:.2f}",
            value=b,
            checked=True,  # default: all live bets selected
        ))

    console.print()
    selected = questionary.checkbox(
        "Select the bets you're placing (space to toggle, enter to confirm):",
        choices=choices,
        style=questionary.Style([
            ("checkbox-selected", "fg:cyan bold"),
            ("selected", "fg:cyan"),
            ("pointer", "fg:cyan bold"),
        ]),
    ).ask()

    if selected is None or len(selected) == 0:
        console.print("\n[dim]No bets selected. Nothing recorded.[/dim]\n")
        return

    # Ask for actual fill price and stake per bet
    console.print("\n[bold]For each bet, enter your fill odds and stake. Press Enter to use defaults.[/bold]\n")
    filled_bets = []
    for b in selected:
        label = _display_label(b["yes_team"], b["market_type"], b["line"])
        event = (b["event_title"] or "?")[:35]
        default_odds = _american(b["live_ask"]) if b["live_ask"] else "N/A"
        default_stake = b["stake"]

        fill_input = questionary.text(
            f"  {event} | {label}\n    Odds [default {default_odds}]:",
            default="",
        ).ask()
        if fill_input is None:
            console.print("\n[dim]Cancelled. Nothing recorded.[/dim]\n")
            return

        stake_input = questionary.text(
            f"    Stake [default ${default_stake:.2f}]:",
            default="",
        ).ask()
        if stake_input is None:
            console.print("\n[dim]Cancelled. Nothing recorded.[/dim]\n")
            return

        # Parse odds
        fill_input = fill_input.strip()
        if fill_input:
            try:
                odds_val = int(fill_input.replace("+", ""))
                if odds_val > 0:
                    fill_prob = 100.0 / (odds_val + 100.0)
                else:
                    fill_prob = abs(odds_val) / (abs(odds_val) + 100.0)
                fill_odds_str = fill_input
            except ValueError:
                console.print(f"[yellow]  Invalid odds '{fill_input}', using live ask.[/yellow]")
                fill_prob = b["live_ask"]
                fill_odds_str = default_odds
        else:
            fill_prob = b["live_ask"]
            fill_odds_str = default_odds

        # Parse stake
        stake_input = stake_input.strip().replace("$", "")
        if stake_input:
            try:
                fill_stake = float(stake_input)
            except ValueError:
                console.print(f"[yellow]  Invalid stake '{stake_input}', using default.[/yellow]")
                fill_stake = default_stake
        else:
            fill_stake = default_stake

        filled_bets.append({**b, "fill_price": fill_prob, "fill_odds": fill_odds_str, "fill_stake": fill_stake})

    # Write placed records to DB — mark ALL rows for the same market_id
    # (covers re-scans that create duplicate rows for the same market)
    now_str = _dt.now(timezone.utc).isoformat()
    conn = get_connection()
    placed_count = 0
    for b in filled_bets:
        conn.execute(
            """
            UPDATE ev_predictions
            SET placed = 1, placed_at = ?, placed_price = ?, placed_stake = ?
            WHERE market_id = ?
            """,
            (now_str, b["fill_price"], b["fill_stake"], b["market_id"]),
        )
        placed_count += 1
    conn.commit()
    conn.close()

    # Summary
    console.print()
    summary_table = Table(box=box.SIMPLE, show_header=True, title="Placed Bets")
    summary_table.add_column("Event", min_width=24)
    summary_table.add_column("Outcome", width=18)
    summary_table.add_column("Fill Odds", justify="right", width=10)
    summary_table.add_column("Stake", justify="right", width=8)
    for b in filled_bets:
        summary_table.add_row(
            (b["event_title"] or "?")[:30],
            _display_label(b["yes_team"], b["market_type"], b["line"]),
            b["fill_odds"],
            f"${b['fill_stake']:.2f}",
        )
    console.print(summary_table)

    total = sum(b["fill_stake"] for b in filled_bets)
    console.print(f"\n[bold green]Recorded {placed_count} bet(s)[/bold green] — total at risk: [cyan]${total:.2f}[/cyan]\n")


@app.command("resolve")
def resolve(
    target_date: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Date to resolve outcomes for (YYYY-MM-DD). Defaults to yesterday.",
    ),
) -> None:
    """Resolve yesterday's game outcomes via ESPN and update P&L.

    Alias for: evmax cleanup resolve --date YYYY-MM-DD

    \b
    Example:
      evmax agents resolve              # resolves yesterday
      evmax agents resolve --date 2026-03-21
    """
    from datetime import date as _date, timedelta
    from evmax.agents.cleanup.resolver import resolve_outcomes_for_date

    if target_date:
        try:
            d = _date.fromisoformat(target_date)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {target_date!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        d = _date.today() - timedelta(days=1)

    console.print(f"[cyan]Resolving outcomes for[/cyan] {d.isoformat()} via ESPN...")
    result = asyncio.run(resolve_outcomes_for_date(d))

    console.print(
        f"  [green]Resolved:[/green] {result['resolved']}  "
        f"[yellow]Unmatched:[/yellow] {result['failed']}"
    )
    unmatched = result.get("unmatched", [])
    if unmatched:
        console.print(f"\n  [yellow]Unmatched ({len(unmatched)}):[/yellow]")
        for eid in unmatched[:20]:
            console.print(f"    [dim]{eid}[/dim]")
        if len(unmatched) > 20:
            console.print(f"    [dim]... and {len(unmatched) - 20} more[/dim]")

    if result["resolved"] > 0:
        console.print(
            f"\n  [dim]Run [bold]evmax cleanup show[/bold] to view updated P&L.[/dim]"
        )


@app.command("seed")
def seed(
    model: str = typer.Argument(..., help="Model to seed: elo | form | poisson"),
    sector: str = typer.Option(..., "--sector", "-s", help="Sector, e.g. 'nba'"),
    file: Path = typer.Option(..., "--file", "-f", help="JSON file with seed data"),
) -> None:
    """Seed a model agent with historical data from a JSON file.

    JSON format by model type:

    elo:
      {"ratings": {"lakers": 1550.0, "celtics": 1620.0, ...}}

    form:
      {"results": [
        {"date": "2026-01-15", "home": "lakers", "away": "celtics",
         "score_home": 112, "score_away": 108},
        ...
      ]}

    poisson:
      {
        "league_avg": {"home": 1.55, "away": 1.15},
        "teams": {
          "manchester city": {"attack": 1.42, "defense": 0.65, "games": 28},
          ...
        }
      }
    """
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    data = json.loads(file.read_text())

    if model == "elo":
        from evmax.agents.models.elo_agent import EloModelAgent
        agent = EloModelAgent()
        ratings = data.get("ratings", data)  # allow top-level dict too
        agent.seed_ratings(sector, ratings)
        console.print(f"[green]Seeded Elo for {sector}:[/green] {len(ratings)} teams")

    elif model == "form":
        from evmax.agents.models.form_agent import FormModelAgent
        agent = FormModelAgent()
        results = data.get("results", data)
        agent.seed_results(sector, results)
        console.print(f"[green]Seeded Form for {sector}:[/green] {len(results)} results")

    elif model == "poisson":
        from evmax.agents.models.poisson_agent import PoissonModelAgent
        agent = PoissonModelAgent()
        agent.seed_team_stats(
            sector=sector,
            team_stats=data.get("teams", {}),
            league_avg=data.get("league_avg"),
        )
        console.print(f"[green]Seeded Poisson for {sector}:[/green] {len(data.get('teams', {}))} teams")

    else:
        console.print(f"[red]Unknown model:[/red] {model}. Choose: elo | form | poisson")
        raise typer.Exit(1)


@app.command("ratings")
def ratings(
    sector: str = typer.Argument(..., help="Sector, e.g. 'nba'"),
    top: int = typer.Option(20, "--top", "-n", help="Show top N teams by Elo."),
) -> None:
    """Display current Elo ratings for a sector."""
    from evmax.agents.models.elo_agent import EloModelAgent
    agent = EloModelAgent()
    all_r = agent.all_ratings(sector)

    if not all_r:
        console.print(f"[yellow]No Elo ratings found for {sector}.[/yellow] Use [bold]evmax agents seed elo[/bold] to load data.")
        return

    sorted_teams = sorted(all_r.items(), key=lambda x: x[1], reverse=True)[:top]

    table = Table(title=f"Elo Ratings — {sector.upper()}", box=box.SIMPLE)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Team", style="bold")
    table.add_column("Elo", justify="right", style="cyan")
    table.add_column("Games", justify="right", style="dim")

    for rank, (team, elo) in enumerate(sorted_teams, 1):
        games = agent._get_count(sector, team)
        table.add_row(str(rank), team.title(), f"{elo:.0f}", str(games))

    console.print(table)


@app.command("update")
def update_result(
    sector: str = typer.Option(..., "--sector", "-s", help="Sector"),
    team_a: str = typer.Option(..., "--home", help="Home / outcome_a team name"),
    team_b: str = typer.Option(..., "--away", help="Away / outcome_b team name"),
    score_a: float = typer.Option(..., "--score-home", help="Final score for home team"),
    score_b: float = typer.Option(..., "--score-away", help="Final score for away team"),
    date: Optional[str] = typer.Option(None, "--date", help="Game date ISO (YYYY-MM-DD)"),
    surface: Optional[str] = typer.Option(None, "--surface", help="Court surface (hard/clay/grass). Required for tennis."),
) -> None:
    """Feed a completed game result into all model agents (updates Elo + Form + Poisson)."""
    if surface and surface.lower() == "indoor":
        console.print(
            "[red]--surface indoor is no longer valid.[/red] "
            "Indoor is a court-condition modifier on hard, not a surface. "
            "Use --surface hard for indoor hard events. See TODO.md MODEL-6 "
            "for the planned court-adjustment factor."
        )
        raise typer.Exit(1)
    from evmax.agents.coordinator import AgentCoordinator
    c = AgentCoordinator(sectors=[sector], enable_models=True)
    c.update_models(team_a, team_b, score_a, score_b, sector, date, surface=surface or "overall")
    console.print(
        f"[green]Updated models:[/green] {team_a} {score_a:.0f} – {score_b:.0f} {team_b} ({sector})"
    )


@app.command("seed-tennis")
def seed_tennis(
    what: str = typer.Argument(..., help="What to seed: rankings | surface"),
    file: Path = typer.Option(..., "--file", "-f", help="JSON file"),
    surface: Optional[str] = typer.Option(None, "--surface", help="Surface for 'surface' seeding: hard/clay/grass"),
) -> None:
    """Seed tennis rankings or surface-specific Elo ratings.

    JSON format for rankings:
      {"sinner": 1, "alcaraz": 2, "djokovic": 3, ...}

    JSON format for surface ratings:
      {"djokovic": 1860.0, "nadal": 1820.0, ...}
    """
    from evmax.agents.models.tennis_model_agent import TennisModelAgent
    import json

    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    agent = TennisModelAgent()
    data = json.loads(file.read_text())

    if what == "rankings":
        agent.seed_rankings(data)
        console.print(f"[green]Seeded rankings:[/green] {len(data)} players")
    elif what == "surface":
        if not surface:
            console.print("[red]--surface required for surface seeding[/red]")
            raise typer.Exit(1)
        if surface.lower() == "indoor":
            console.print(
                "[red]--surface indoor is no longer valid.[/red] "
                "Indoor is a court-condition modifier on hard, not a surface. "
                "See TODO.md MODEL-6 for the planned court-adjustment factor."
            )
            raise typer.Exit(1)
        agent.seed_surface_ratings(surface, data)
        console.print(f"[green]Seeded {surface} ratings:[/green] {len(data)} players")
    else:
        console.print(f"[red]Unknown:[/red] {what}. Choose: rankings | surface")
        raise typer.Exit(1)
