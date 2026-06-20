"""Shadow-mode bet inspection and promotion commands.

Registered as `evmax cleanup shadow <subcommand>` via cleanup.py.

Three commands:
  show     — list recent shadow predictions + resolved outcomes
  metrics  — Brier / ROI / win-rate for shadow predictions (per-category)
  promote  — flip a category from `shadow` to `live` in data/categories.yaml

Shadow mode is the ARCH-11 feature that lets the scanner log predictions
for a category without touching the bankroll. It's the validation path
for MODEL-9 (NFL props): run in shadow during the 2026 NFL regular
season, capture pre-game prices at scan time, resolve outcomes via
ESPN boxscore, compare ROI against the Stage 4 backtest number. If the
edge holds, `promote` flips the category to `live` and real bets start.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from evmax.models.market import is_prop_event

app = typer.Typer(
    help="Inspect shadow-mode predictions and promote categories to live."
)
console = Console()


_CATEGORIES_YAML = Path(__file__).resolve().parents[3] / "data" / "categories.yaml"


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command("show")
def show(
    days: int = typer.Option(7, "--days", "-d", help="Look back this many days."),
    category: Optional[str] = typer.Option(
        None, "--category", "-c",
        help="Filter by category key (e.g. 'nfl_props'). Default: all.",
    ),
    resolved_only: bool = typer.Option(
        False, "--resolved", help="Only show rows that have a settled outcome."
    ),
) -> None:
    """Show recent shadow predictions with model_prob, captured_yes_price,
    resolved outcome (if any), and edge."""
    from evmax.agents.cleanup.db import get_connection

    since = (date.today() - timedelta(days=days)).isoformat()
    where = ["p.mode = 'shadow'", "p.scan_date >= ?"]
    params: list = [since]
    if category:
        # For game categories the sector matches the category key directly;
        # for prop categories the key is `{sector}_props` so we match both.
        if category.endswith("_props"):
            where.append("(p.sector = ? AND p.event_id LIKE '%::prop::%')")
            params.append(category[: -len("_props")])
        else:
            where.append("p.sector = ?")
            params.append(category)
    if resolved_only:
        where.append("o.outcome IS NOT NULL")

    sql = f"""
        SELECT p.scan_date, p.event_date, p.sector, p.event_title,
               p.yes_team, p.market_type, p.line,
               p.captured_yes_price, p.blended_true_prob, p.ev_pct,
               p.kelly_fraction, p.model_version,
               o.outcome
        FROM ev_predictions p
        LEFT JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE {' AND '.join(where)}
        ORDER BY p.scan_date DESC, p.ev_pct DESC
        LIMIT 200
    """
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        console.print(
            f"[yellow]No shadow predictions in the last {days} day(s)"
            + (f" for {category}" if category else "")
            + ".[/yellow]"
        )
        return

    table = Table(
        title=f"Shadow predictions — last {days} days",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Scan", width=10)
    table.add_column("Game", width=10)
    table.add_column("Sector", width=8)
    table.add_column("Event", min_width=22, no_wrap=False)
    table.add_column("Bet", width=14)
    table.add_column("Captured", justify="right", width=10)
    table.add_column("Model", justify="right", width=8)
    table.add_column("EV%", justify="right", width=7)
    table.add_column("Outcome", width=10)

    n_resolved = n_wins = 0
    for r in rows:
        outcome = r["outcome"]
        if outcome is None:
            outcome_str = "[dim]pending[/dim]"
        elif outcome == 1:
            outcome_str = "[green]WIN[/green]"
            n_resolved += 1
            n_wins += 1
        else:
            outcome_str = "[red]LOSS[/red]"
            n_resolved += 1

        captured = r["captured_yes_price"]
        captured_str = f"{captured:.3f}" if captured is not None else "—"
        model_prob = r["blended_true_prob"]
        model_str = f"{model_prob:.3f}" if model_prob is not None else "—"
        ev_pct = r["ev_pct"]
        ev_str = f"{ev_pct*100:+.1f}%" if ev_pct is not None else "—"

        bet_label = r["yes_team"] or ""
        if r["market_type"]:
            bet_label += f" {r['market_type']}"
        if r["line"] is not None:
            bet_label += f" {r['line']:+.1f}"

        table.add_row(
            (r["scan_date"] or "")[-5:],
            (r["event_date"] or "")[-5:] if r["event_date"] else "—",
            r["sector"] or "",
            r["event_title"] or "",
            bet_label.strip(),
            captured_str,
            model_str,
            ev_str,
            outcome_str,
        )

    console.print(table)
    console.print(
        f"[dim]{len(rows)} rows · "
        f"{n_resolved} resolved · "
        f"{n_wins} wins ({(n_wins/n_resolved*100) if n_resolved else 0:.0f}%)[/dim]"
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@app.command("metrics")
def metrics(
    days: int = typer.Option(30, "--days", "-d", help="Look back this many days."),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category key. Default: all shadow categories."
    ),
    include_contaminated: bool = typer.Option(
        False,
        "--include-contaminated",
        help="Include rows produced by a superseded code state (default: excluded). "
        "Off by default so the promotion gate only scores current-code rows.",
    ),
) -> None:
    """Compute Brier score, accuracy, and ROI for shadow predictions.

    Used by MODEL-9 validation — compare these numbers against the
    Stage 4 backtest to decide whether a category's edge is real or
    was retrospective leakage.

    By default, rows produced by a superseded code state are excluded (see
    evmax/agents/cleanup/contamination.py) so a contaminated sample can't drive
    a wrong promote decision. The count of excluded rows is reported; pass
    --include-contaminated to score the raw sample instead.
    """
    from evmax.agents.cleanup.contamination import is_contaminated
    from evmax.agents.cleanup.db import get_connection

    since = (date.today() - timedelta(days=days)).isoformat()
    where = ["p.mode = 'shadow'", "p.scan_date >= ?", "o.outcome IS NOT NULL"]
    params: list = [since]
    if category:
        if category.endswith("_props"):
            where.append("(p.sector = ? AND p.event_id LIKE '%::prop::%')")
            params.append(category[: -len("_props")])
        else:
            where.append("p.sector = ?")
            params.append(category)

    sql = f"""
        SELECT p.sector, p.event_id, p.market_type, p.model_sources, p.line,
               p.captured_yes_price, p.blended_true_prob,
               p.ev_pct, p.kelly_fraction, p.volume_usd, o.outcome
        FROM ev_predictions p
        INNER JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE {' AND '.join(where)}
    """
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    # Drop rows from superseded code states unless explicitly included, and
    # tally how many were excluded per category for transparency.
    excluded_by_category: dict[str, int] = {}
    if not include_contaminated:
        kept = []
        for r in rows:
            if is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"]):
                key = r["sector"] or "unknown"
                if is_prop_event(r["event_id"]):
                    key = f"{key}_props"
                excluded_by_category[key] = excluded_by_category.get(key, 0) + 1
            else:
                kept.append(r)
        rows = kept

    if not rows:
        msg = (
            f"[yellow]No resolved shadow predictions in the last {days} day(s)"
            + (f" for {category}" if category else "")
            + ".[/yellow]"
        )
        console.print(msg)
        if excluded_by_category:
            n_excl = sum(excluded_by_category.values())
            console.print(
                f"[dim]({n_excl} contaminated row(s) excluded — superseded code "
                f"state. Pass --include-contaminated to score them.)[/dim]"
            )
        return

    # Group by category key (game vs prop)
    by_category: dict[str, list] = {}
    for r in rows:
        key = r["sector"] or "unknown"
        if is_prop_event(r["event_id"]):
            key = f"{key}_props"
        by_category.setdefault(key, []).append(r)

    table = Table(
        title=f"Shadow metrics — last {days} days",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Category", width=14)
    table.add_column("N", justify="right", width=6)
    table.add_column("Excl", justify="right", width=6)
    table.add_column("Accuracy", justify="right", width=10)
    table.add_column("Brier", justify="right", width=9)
    table.add_column("LogLoss", justify="right", width=9)
    table.add_column("ROI (flat)", justify="right", width=11)
    table.add_column("WinRate", justify="right", width=9)

    for cat_key in sorted(by_category.keys()):
        cat_rows = by_category[cat_key]
        n = len(cat_rows)
        hits = sum(1 for r in cat_rows if r["outcome"] == 1)
        predictions = [r["blended_true_prob"] for r in cat_rows]
        outcomes = [r["outcome"] for r in cat_rows]

        brier = sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n
        eps = 1e-7
        log_loss = -sum(
            o * math.log(max(eps, min(1 - eps, p))) + (1 - o) * math.log(max(eps, min(1 - eps, 1 - p)))
            for p, o in zip(predictions, outcomes)
        ) / n
        # Directional accuracy — model picks the side it thinks is more likely
        acc = sum(1 for p, o in zip(predictions, outcomes) if (p >= 0.5) == (o == 1)) / n

        # ROI at flat $1/bet using captured_yes_price. Only rows with a
        # captured price participate. Positive if actual outcome (1/0)
        # is the YES side we modelled.
        stake = 0.0
        pnl = 0.0
        wins_roi = 0
        for r in cat_rows:
            price = r["captured_yes_price"]
            if price is None or price <= 0 or price >= 1:
                continue
            # We only "bet" when the model has an edge at ev_threshold >= 2%
            if r["ev_pct"] is None or r["ev_pct"] < 0.02:
                continue
            stake += 1.0
            if r["outcome"] == 1:
                wins_roi += 1
                pnl += (1.0 / price) - 1.0
            else:
                pnl -= 1.0
        roi = pnl / stake if stake > 0 else 0.0
        win_rate = wins_roi / stake if stake > 0 else 0.0

        roi_color = "green" if roi > 0 else "red"
        brier_color = "green" if brier < 0.22 else ("yellow" if brier < 0.25 else "red")

        excl = excluded_by_category.get(cat_key, 0)
        excl_str = f"[yellow]{excl}[/yellow]" if excl else "[dim]0[/dim]"

        table.add_row(
            cat_key,
            str(n),
            excl_str,
            f"{acc * 100:.1f}%",
            f"[{brier_color}]{brier:.4f}[/{brier_color}]",
            f"{log_loss:.4f}",
            f"[{roi_color}]{roi * 100:+.1f}%[/{roi_color}]",
            f"{win_rate * 100:.1f}%",
        )

    # Categories that were *entirely* contaminated have no surviving rows, so
    # they never reach the table above — surface them explicitly so a wiped-out
    # sample doesn't look like "no data."
    for cat_key, n_excl in sorted(excluded_by_category.items()):
        if cat_key not in by_category:
            table.add_row(
                cat_key, "0", f"[yellow]{n_excl}[/yellow]",
                "—", "—", "—", "—", "—",
            )

    console.print(table)
    footer = (
        "[dim]ROI uses captured_yes_price at scan time on rows with EV ≥ 2%. "
        "Compare against the Stage 4 backtest number in TODO.md MODEL-9."
    )
    if excluded_by_category and not include_contaminated:
        n_excl = sum(excluded_by_category.values())
        footer += (
            f"\nExcl = rows from a superseded code state, dropped from scoring "
            f"({n_excl} total). Pass --include-contaminated to score them."
        )
    console.print(footer + "[/dim]")


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


# Minimum current-code (non-contaminated) resolved rows required before a
# category may be promoted to live. The MODEL-9 / WNBA gates cite n≥75 (ML) or
# ~30 for a model already backtested; we use 30 as a floor that catches the
# "3 clean rows out of 49" baseball case while staying out of the way of a
# genuinely validated category. Override with --force.
MIN_CLEAN_RESOLVED = 30


def _clean_resolved_count(category: str) -> int:
    """Count resolved shadow rows for a category that survive the contamination
    filter — i.e. were produced by the current code state."""
    from evmax.agents.cleanup.contamination import is_contaminated
    from evmax.agents.cleanup.db import get_connection

    where = ["p.mode = 'shadow'", "o.outcome IS NOT NULL"]
    params: list = []
    if category.endswith("_props"):
        where.append("(p.sector = ? AND p.event_id LIKE '%::prop::%')")
        params.append(category[: -len("_props")])
    else:
        where.append("p.sector = ?")
        params.append(category)
    sql = f"""
        SELECT p.sector, p.market_type, p.model_sources, p.line
        FROM ev_predictions p
        INNER JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE {' AND '.join(where)}
    """
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return sum(
        1 for r in rows
        if not is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"])
    )


# ---------------------------------------------------------------------------
# CLV gate — the metric that actually predicts +EV for laddered markets
# ---------------------------------------------------------------------------
# For spreads/totals (and any market where the model trails Kalshi-at-close on
# Brier by construction), Brier is the wrong promotion yardstick: the edge is
# beating the CLOSING price, not out-forecasting the outcome. We gate on
# kalshi_clv_pct = kalshi_close − kalshi_entry (the conventional sharp-bettor
# CLV; pinnacle_drift_pct is positive BY CONSTRUCTION — our own selection rule —
# and is NOT line-aligned for alt-strikes, so it is never used as a gate).
#
# A market "clears" when, on enough current-code resolved bets, the average
# Kalshi entry→close move is positive AND a clear majority moved our way.
MIN_CLV_RESOLVED = 30
CLV_MIN_MEAN_PP = 0.0       # mean kalshi CLV must be ≥ 0 (beat the close)
CLV_MIN_FRAC_POSITIVE = 0.55  # clear majority of bets gained CLV (not a coin flip)


def clv_clears(n: int, mean_clv_pp: float, frac_positive: float) -> bool:
    """Whether a CLV sample passes the promotion gate.

    Needs a big-enough current-code sample, a non-negative mean entry→close move
    (we beat the close on average) AND a clear majority of bets moving our way —
    a small positive mean carried by a few outliers (frac_positive ≈ 0.44) is
    noise, not edge, so it must NOT clear.
    """
    return (
        n >= MIN_CLV_RESOLVED
        and mean_clv_pp >= CLV_MIN_MEAN_PP
        and frac_positive >= CLV_MIN_FRAC_POSITIVE
    )


def clv_stats(
    category: str,
    market_type: Optional[str] = None,
    mode: Optional[str] = None,
    since: Optional[str] = None,
) -> dict:
    """Aggregate kalshi_clv_pct for a category's current-code resolved bets.

    Returns {n, mean_clv_pp, frac_positive, clears}. Filters out rows from a
    superseded code state via the same contamination rules the count gate uses,
    so we never validate retired pricing. `mode=None` scores every mode (use to
    judge a currently-LIVE market type's retention); `mode='shadow'` scores the
    shadow validation set (use before a shadow→live promote). `since` (YYYY-MM-DD)
    excludes older rows — use it to drop a stale-modelling period that the
    contamination rules don't yet flag (e.g. a pricing change with no SHA stamp).
    """
    from evmax.agents.cleanup.contamination import is_contaminated
    from evmax.agents.cleanup.db import get_connection

    where = ["p.kalshi_clv_pct IS NOT NULL", "o.outcome IS NOT NULL"]
    params: list = []
    if category.endswith("_props"):
        where.append("(p.sector = ? AND p.event_id LIKE '%::prop::%')")
        params.append(category[: -len("_props")])
    else:
        where.append("p.sector = ?")
        params.append(category)
    if market_type is not None:
        where.append("p.market_type = ?")
        params.append(market_type)
    if mode is not None:
        where.append("p.mode = ?")
        params.append(mode)
    if since is not None:
        where.append("p.scan_date >= ?")
        params.append(since)
    sql = f"""
        SELECT p.sector, p.market_type, p.model_sources, p.line, p.kalshi_clv_pct
        FROM ev_predictions p
        INNER JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE {' AND '.join(where)}
    """
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    clvs = [
        r["kalshi_clv_pct"] for r in rows
        if not is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"])
    ]
    n = len(clvs)
    if n == 0:
        return {"n": 0, "mean_clv_pp": 0.0, "frac_positive": 0.0, "clears": False}
    mean_clv = sum(clvs) / n
    frac_pos = sum(1 for c in clvs if c > 0) / n
    return {
        "n": n,
        "mean_clv_pp": round(mean_clv, 3),
        "frac_positive": round(frac_pos, 3),
        "clears": clv_clears(n, mean_clv, frac_pos),
    }


@app.command("clv")
def clv(
    category: str = typer.Argument(..., help="Category key (e.g. 'wnba')."),
    market_type: Optional[str] = typer.Option(
        None, "--market-type", "-m",
        help="Restrict to one market type (moneyline/spread/total).",
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", help="Restrict to one mode (live/shadow). Default: all."
    ),
    since: Optional[str] = typer.Option(
        None, "--since",
        help="Only score rows scanned on/after YYYY-MM-DD (drop a stale-model period).",
    ),
) -> None:
    """Report Kalshi CLV (entry→close) — the +EV signal for laddered markets.

    This is the right lens for spreads/totals where the model trails Kalshi-at-
    close on Brier by construction. Positive mean CLV with a clear majority of
    bets moving our way = we are beating the close = genuine edge.
    """
    s = clv_stats(category, market_type=market_type, mode=mode, since=since)
    label = f"{category}" + (f" / {market_type}" if market_type else "")
    label += f" [{mode}]" if mode else " [all modes]"
    label += f" since {since}" if since else ""
    if s["n"] == 0:
        console.print(f"[yellow]No current-code resolved CLV rows for {label}.[/yellow]")
        return
    verdict = "[green]CLEARS[/green]" if s["clears"] else "[red]does NOT clear[/red]"
    console.print(
        f"\n[bold]CLV — {label}[/bold]  (current-code resolved, n={s['n']})\n"
        f"  mean kalshi CLV = {s['mean_clv_pp']:+.2f}pp\n"
        f"  % bets with +CLV = {s['frac_positive']*100:.0f}%\n"
        f"  gate: n≥{MIN_CLV_RESOLVED}, mean≥{CLV_MIN_MEAN_PP:+.1f}pp, "
        f"%pos≥{CLV_MIN_FRAC_POSITIVE*100:.0f}%  →  {verdict}"
    )


@app.command("promote")
def promote(
    category: str = typer.Argument(..., help="Category key to promote (e.g. 'nfl_props')."),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Promote even if the clean (current-code) resolved sample is below "
        f"the {MIN_CLEAN_RESOLVED}-row floor.",
    ),
) -> None:
    """Flip a category from `shadow` to `live` in data/categories.yaml.

    Uses targeted text replacement to preserve comments and formatting.
    Validates current mode is 'shadow' before flipping — errors out if
    the category is already live or if mode detection fails.

    Refuses to promote when fewer than MIN_CLEAN_RESOLVED resolved rows
    survive the contamination filter (so a sample dominated by superseded
    code — e.g. baseball's 3-clean-of-49 — can't trigger a live flip).
    Override with --force.

    After this command runs:
      - `evmax categories show <category>` reports mode=live
      - Next scan will log the category with mode='live' and apply
        full Kelly sizing against the bankroll
    """
    if not _CATEGORIES_YAML.exists():
        console.print(f"[red]Not found:[/red] {_CATEGORIES_YAML}")
        raise typer.Exit(1)

    text = _CATEGORIES_YAML.read_text()
    new_text, old_mode = _flip_mode_in_yaml(text, category, new_mode="live")
    if old_mode is None:
        console.print(
            f"[red]Could not find a `mode:` line for category {category!r} "
            f"in {_CATEGORIES_YAML}.[/red]"
        )
        raise typer.Exit(1)

    if old_mode == "live":
        console.print(
            f"[yellow]{category} is already in `live` mode. No change.[/yellow]"
        )
        raise typer.Exit(0)

    if old_mode != "shadow":
        console.print(
            f"[red]{category} is currently in `{old_mode}` mode, not `shadow`. "
            f"Refusing to auto-promote — edit data/categories.yaml manually "
            f"if this is intentional.[/red]"
        )
        raise typer.Exit(1)

    # Contamination gate: only promote on a sample that reflects current code.
    try:
        clean_n = _clean_resolved_count(category)
    except Exception as _count_err:  # noqa: BLE001 — DB shape varies in tests
        clean_n = None
    if clean_n is not None and clean_n < MIN_CLEAN_RESOLVED:
        msg = (
            f"[red]Only {clean_n} current-code resolved row(s) for "
            f"{category} (need ≥ {MIN_CLEAN_RESOLVED}).[/red] "
            f"The rest are from a superseded code state — promoting now would "
            f"validate old code. Accumulate clean shadow data first "
            f"(see `evmax cleanup shadow metrics --category {category}`)."
        )
        if not force:
            console.print(msg)
            console.print("[dim]Override with --force if this is intentional.[/dim]")
            raise typer.Exit(1)
        console.print(msg.replace("[red]", "[yellow]").replace("[/red]", "[/yellow]"))
        console.print("[yellow]--force set — promoting anyway.[/yellow]")

    # CLV gate: for the metric that actually predicts +EV. We only enforce it
    # when CLV has been backfilled for enough shadow rows (n≥MIN_CLV_RESOLVED);
    # categories without CLV data yet (e.g. props pre-resolution) fall through
    # to the count gate alone, unchanged.
    try:
        cstats = clv_stats(category, mode="shadow")
    except Exception:  # noqa: BLE001 — DB shape varies in tests
        cstats = {"n": 0, "clears": False}
    if cstats["n"] >= MIN_CLV_RESOLVED and not cstats["clears"]:
        cmsg = (
            f"[red]CLV does not clear for {category}:[/red] "
            f"mean={cstats['mean_clv_pp']:+.2f}pp, "
            f"%pos={cstats['frac_positive']*100:.0f}% on n={cstats['n']} "
            f"(need mean≥{CLV_MIN_MEAN_PP:+.1f}pp & %pos≥{CLV_MIN_FRAC_POSITIVE*100:.0f}%). "
            f"Beating the close — not Brier — is the +EV signal; this sample isn't."
        )
        if not force:
            console.print(cmsg)
            console.print(
                f"[dim]Inspect with `evmax cleanup shadow clv {category}`. "
                f"Override with --force if intentional.[/dim]"
            )
            raise typer.Exit(1)
        console.print(cmsg.replace("[red]", "[yellow]").replace("[/red]", "[/yellow]"))
        console.print("[yellow]--force set — promoting despite weak CLV.[/yellow]")

    if not yes:
        console.print(
            f"\nAbout to flip [bold]{category}[/bold] from "
            f"[yellow]shadow[/yellow] → [green]live[/green] in "
            f"{_CATEGORIES_YAML}."
        )
        confirmed = typer.confirm("Proceed?", default=False)
        if not confirmed:
            console.print("Aborted.")
            raise typer.Exit(0)

    _CATEGORIES_YAML.write_text(new_text)

    # Verify by re-validating the registry
    from evmax.categories import reload_registry, validate_registry

    try:
        reload_registry()
        validate_registry()
    except Exception as e:
        console.print(f"[red]❌ Post-promote validation failed:[/red] {e}")
        console.print(
            f"[red]The YAML was updated but is now invalid. "
            f"Please inspect {_CATEGORIES_YAML} and restore manually.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"[green]✓ {category} promoted to live.[/green] "
        f"Next scan will log rows with mode='live' and size Kelly against "
        f"bankroll."
    )


def _flip_mode_in_yaml(
    text: str, category: str, new_mode: str
) -> tuple[str, Optional[str]]:
    """Find the `mode:` line under `category:` and replace it with
    `mode: new_mode`. Returns (updated_text, old_mode or None).

    Uses line-based scanning rather than a full YAML parse so comments
    and formatting survive. Expects the shipped YAML format:
      category_key:
        display_name: "..."
        mode: live
        ...
    """
    lines = text.splitlines(keepends=True)
    category_header = f"{category}:"
    in_block = False
    old_mode: Optional[str] = None
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n").rstrip("\r")
        # Category header starts at column 0, no leading whitespace
        if not in_block:
            if stripped == category_header or stripped.startswith(category_header + " "):
                in_block = True
            continue
        # Still in block if indented (at least one space) or blank
        if stripped == "" or line.startswith(" ") or line.startswith("\t"):
            if line.lstrip().startswith("mode:"):
                # Extract existing value
                after = line.split("mode:", 1)[1].strip()
                # Strip trailing comments
                if "#" in after:
                    after = after.split("#", 1)[0].strip()
                old_mode = after.strip()
                # Preserve leading whitespace and trailing newline
                leading = line[: len(line) - len(line.lstrip())]
                trailing = "\n" if line.endswith("\n") else ""
                lines[i] = f"{leading}mode: {new_mode}{trailing}"
                return "".join(lines), old_mode
            continue
        # Dedented back to column 0 — left the category block without
        # finding a mode line
        break
    return text, None
