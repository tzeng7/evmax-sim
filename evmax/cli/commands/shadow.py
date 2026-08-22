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


def _format_why(diagnostics_json: Optional[str]) -> str:
    """Compact one-line rendering of a model_diagnostics JSON blob.

    ``missing:elo,form · gated:tennis_surface(0.30)`` — missing means the
    lookup found nothing (player/team absent from state, recoverable by
    seeding); gated means the agent predicted but fell at the confidence
    gate (known but thin).
    """
    if not diagnostics_json:
        return "[dim]—[/dim]"
    import json as _json

    try:
        diag = _json.loads(diagnostics_json)
    except (ValueError, TypeError):
        return "[dim]?[/dim]"
    parts = []
    missing = diag.get("missing") or []
    if missing:
        parts.append("missing:" + ",".join(missing))
    gated = diag.get("gated") or {}
    if gated:
        parts.append(
            "gated:" + ",".join(
                f"{name}({rec.get('conf', '?')})" for name, rec in gated.items()
            )
        )
    return " · ".join(parts) if parts else "[dim]full[/dim]"


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
    why: bool = typer.Option(
        False, "--why",
        help="Add a Why column from model_diagnostics: which models were "
             "missing (player/team absent from state) or confidence-gated "
             "(known but thin) on each row.",
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
               p.kelly_fraction, p.model_version, p.model_diagnostics,
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
    if why:
        table.add_column("Why (missing / gated)", min_width=20, no_wrap=False)

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

        row_cells = [
            (r["scan_date"] or "")[-5:],
            (r["event_date"] or "")[-5:] if r["event_date"] else "—",
            r["sector"] or "",
            r["event_title"] or "",
            bet_label.strip(),
            captured_str,
            model_str,
            ev_str,
            outcome_str,
        ]
        if why:
            row_cells.append(_format_why(r["model_diagnostics"]))
        table.add_row(*row_cells)

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


def _fetch_clv_rows(
    category: str,
    market_type: Optional[str] = None,
    mode: Optional[str] = None,
    since: Optional[str] = None,
    side: Optional[str] = None,
    venue: Optional[str] = None,
    max_staleness_h: Optional[float] = None,
    sources_token: Optional[str] = None,
) -> tuple[list, int]:
    """Fetch a category's current-code resolved CLV rows.

    Shared row-fetch behind ``clv_stats`` and the ``clv-tiers`` segmentation:
    applies the identical category / market_type / mode / since / side / venue /
    staleness filters and the contamination guard, so every CLV lens scores the
    same row set. Each returned row also carries ``event_title`` for downstream
    grouping. See ``clv_stats`` for the per-parameter semantics. Returns
    ``(kept_rows, excluded_stale)``.
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
    if side is not None:
        if side not in ("lay", "take"):
            raise ValueError(f"side must be 'lay' or 'take', got {side!r}")
        where.append("p.line < 0" if side == "lay" else "p.line > 0")
    if venue is not None:
        if venue not in ("kalshi", "polymarket_us"):
            raise ValueError(
                f"venue must be 'kalshi' or 'polymarket_us', got {venue!r}"
            )
        where.append("p.venue = ?")
        params.append(venue)
    if max_staleness_h is not None and max_staleness_h < 0:
        raise ValueError(f"max_staleness_h must be >= 0, got {max_staleness_h!r}")
    if sources_token is not None:
        where.append("p.model_sources LIKE ?")
        params.append(f"%{sources_token}%")
    sql = f"""
        SELECT p.sector, p.market_type, p.model_sources, p.line, p.kalshi_clv_pct,
               p.event_id, p.event_title, p.venue, p.placed, p.placed_at,
               p.market_id
        FROM ev_predictions p
        INNER JOIN ev_outcomes o ON p.market_id = o.market_id
        WHERE {' AND '.join(where)}
    """
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    kept = [
        r for r in rows
        if not is_contaminated(r["sector"], r["market_type"], r["model_sources"], r["line"])
    ]

    excluded_stale = 0
    if max_staleness_h is not None:
        from evmax.agents.cleanup.resolver import close_lookup_ticker
        from evmax.archiver import DataArchiver

        archiver = DataArchiver()
        fresh = []
        for r in kept:
            # Staleness only applies to Kalshi close snapshots; PolyUS rows have
            # no Kalshi close and are passed through untouched.
            if (r["venue"] or "kalshi") != "kalshi":
                fresh.append(r)
                continue
            ticker, _is_no = close_lookup_ticker(r["market_id"])
            if not ticker:
                excluded_stale += 1  # no anchor => untrustworthy close, drop
                continue
            not_before = r["placed_at"] if (r["placed"] and r["placed_at"]) else None
            staleness = archiver.get_kalshi_close_staleness_h(
                ticker, r["event_id"], not_before=not_before
            )
            if staleness is None or staleness > max_staleness_h:
                excluded_stale += 1
                continue
            fresh.append(r)
        kept = fresh

    return kept, excluded_stale


def _aggregate_clv(kept: list, excluded_stale: int = 0) -> dict:
    """Aggregate kept CLV rows into the clv_stats result dict."""
    clvs = [r["kalshi_clv_pct"] for r in kept]
    n = len(clvs)
    if n == 0:
        return {"n": 0, "mean_clv_pp": 0.0, "frac_positive": 0.0,
                "clears": False, "excluded_stale": excluded_stale}
    mean_clv = sum(clvs) / n
    frac_pos = sum(1 for c in clvs if c > 0) / n
    return {
        "n": n,
        "mean_clv_pp": round(mean_clv, 3),
        "frac_positive": round(frac_pos, 3),
        "clears": clv_clears(n, mean_clv, frac_pos),
        "excluded_stale": excluded_stale,
    }


def clv_stats(
    category: str,
    market_type: Optional[str] = None,
    mode: Optional[str] = None,
    since: Optional[str] = None,
    side: Optional[str] = None,
    venue: Optional[str] = None,
    max_staleness_h: Optional[float] = None,
    sources_token: Optional[str] = None,
) -> dict:
    """Aggregate kalshi_clv_pct for a category's current-code resolved bets.

    Returns {n, mean_clv_pp, frac_positive, clears, excluded_stale}. Filters out
    rows from a superseded code state via the same contamination rules the count
    gate uses, so we never validate retired pricing. `mode=None` scores every
    mode (use to judge a currently-LIVE market type's retention); `mode='shadow'`
    scores the shadow validation set (use before a shadow→live promote). `since`
    (YYYY-MM-DD) excludes older rows — use it to drop a stale-modelling period
    that the contamination rules don't yet flag (e.g. a pricing change with no
    SHA stamp).
    `side` splits laddered markets by the bet's direction relative to the line:
    'lay' = yes_team gives points (line < 0), 'take' = yes_team gets points
    (line > 0). Rows without a line are excluded when side is set. The 2026-07
    WNBA spread audit found the two sides behave as different products — laying
    is ~breakeven CLV while scan-time taking bets buy a NO-side run-up and
    mean-revert — so promotion must be judged per side, not pooled.
    `venue` ('kalshi' / 'polymarket_us') restricts to one exchange. Pooling is
    the default, but the venues sit behind separate shadow firewalls with their
    own promotion criteria — a single thin PolyUS market must not be able to
    flip the sign of a Kalshi-sized sample (2026-07-12 WNBA spread audit: pooled
    lay n=34 read −0.35pp only because one PolyUS row at −18pp dragged a n=33
    Kalshi sample of +0.18pp negative). Judge each venue's CLV on its own book.
    `max_staleness_h` excludes rows whose archived "close" snapshot sits more
    than this many hours before the T-30 target — a watch-closes capture gap,
    not a genuine flat market (2026-07-12 audit: 68% of the exact-zero WNBA
    spread CLV rows had a close snapshot 3-21h stale; those zeros drag
    frac_positive down without measuring real near-tip price action). Only
    'kalshi'-venue rows carry a Kalshi close snapshot, so this filter is
    Kalshi-only; PolyUS rows are left untouched. Staleness is read from
    archive.db per surviving row, so the filter is off by default (no archive
    access, unchanged behaviour); pass a threshold to activate it.
    `sources_token` keeps only rows whose model_sources contains the token —
    the separation mechanism for strategy sub-streams sharing a category, e.g.
    'anchored_entry' isolates the watch-listings anchored-entry rows from
    historical scan-time rows without --since gymnastics.
    """
    kept, excluded_stale = _fetch_clv_rows(
        category, market_type=market_type, mode=mode, since=since,
        side=side, venue=venue, max_staleness_h=max_staleness_h,
        sources_token=sources_token,
    )
    return _aggregate_clv(kept, excluded_stale)


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
    side: Optional[str] = typer.Option(
        None, "--side",
        help="Restrict laddered bets by direction: 'lay' (line<0, giving points) "
             "or 'take' (line>0, getting points). Judge spread promotion per side.",
    ),
    venue: Optional[str] = typer.Option(
        None, "--venue",
        help="Restrict to one exchange: 'kalshi' or 'polymarket_us'. The venues "
             "have separate shadow firewalls — judge each on its own book so a "
             "thin PolyUS market can't flip a Kalshi-sized sample.",
    ),
    max_staleness_h: Optional[float] = typer.Option(
        None, "--max-staleness-h",
        help="Exclude Kalshi rows whose archived 'close' snapshot is more than "
             "this many hours before the T-30 target (a watch-closes capture "
             "gap, not a flat market). 68%% of exact-zero WNBA spread CLV rows "
             "have a 3-21h stale close that drags frac_positive down.",
    ),
    sources_token: Optional[str] = typer.Option(
        None, "--sources-token",
        help="Keep only rows whose model_sources contains this token — e.g. "
             "'anchored_entry' isolates the watch-listings anchored-entry "
             "stream from historical scan-time rows.",
    ),
) -> None:
    """Report Kalshi CLV (entry→close) — the +EV signal for laddered markets.

    This is the right lens for spreads/totals where the model trails Kalshi-at-
    close on Brier by construction. Positive mean CLV with a clear majority of
    bets moving our way = we are beating the close = genuine edge.
    """
    s = clv_stats(
        category, market_type=market_type, mode=mode, since=since,
        side=side, venue=venue, max_staleness_h=max_staleness_h,
        sources_token=sources_token,
    )
    label = f"{category}" + (f" / {market_type}" if market_type else "")
    label += f" [{mode}]" if mode else " [all modes]"
    label += f" since {since}" if since else ""
    label += f" side={side}" if side else ""
    label += f" venue={venue}" if venue else ""
    label += f" fresh≤{max_staleness_h:g}h" if max_staleness_h is not None else ""
    label += f" sources~{sources_token}" if sources_token else ""
    if s["n"] == 0:
        stale_note = ""
        if max_staleness_h is not None and s.get("excluded_stale"):
            stale_note = f" ({s['excluded_stale']} excluded as stale-capture)"
        console.print(
            f"[yellow]No current-code resolved CLV rows for {label}.{stale_note}[/yellow]"
        )
        return
    verdict = "[green]CLEARS[/green]" if s["clears"] else "[red]does NOT clear[/red]"
    stale_line = ""
    if max_staleness_h is not None:
        stale_line = (
            f"\n  excluded (stale close) = {s.get('excluded_stale', 0)} "
            f"(close snapshot > {max_staleness_h:g}h before T-30)"
        )
    console.print(
        f"\n[bold]CLV — {label}[/bold]  (current-code resolved, n={s['n']})\n"
        f"  mean kalshi CLV = {s['mean_clv_pp']:+.2f}pp\n"
        f"  % bets with +CLV = {s['frac_positive']*100:.0f}%{stale_line}\n"
        f"  gate: n≥{MIN_CLV_RESOLVED}, mean≥{CLV_MIN_MEAN_PP:+.1f}pp, "
        f"%pos≥{CLV_MIN_FRAC_POSITIVE*100:.0f}%  →  {verdict}"
    )


@app.command("clv-tiers")
def clv_tiers(
    category: str = typer.Argument(
        "ncaaf", help="Category to segment (only 'ncaaf' is tiered)."
    ),
    market_type: Optional[str] = typer.Option(
        None, "--market-type", "-m", help="Restrict to one market type."
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", help="Restrict to one mode (live/shadow). Default: all."
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Only score rows scanned on/after YYYY-MM-DD."
    ),
    venue: Optional[str] = typer.Option(
        None, "--venue",
        help="Restrict to one exchange: 'kalshi' or 'polymarket_us'.",
    ),
    max_staleness_h: Optional[float] = typer.Option(
        None, "--max-staleness-h",
        help="Exclude Kalshi rows whose archived close is > this many h before T-30.",
    ),
    sources_token: Optional[str] = typer.Option(
        None, "--sources-token",
        help="Keep only rows whose model_sources contains this token.",
    ),
) -> None:
    """Segment resolved CLV by conference tier — the CFB soft-market edge test.

    Groups a category's current-code resolved CLV rows by matchup tier (G5 = both
    Group-of-Five, cross = one of each, P4 = both Power-4, FCS = a buy game or an
    unmapped team) and reports per-tier CLV against the same promotion gate that
    ``clv`` uses. The thesis is that softer, less-watched G5 games carry more
    entry->close edge than efficiently-priced marquee P4 games; this is where the
    live sample proves or kills it. Only 'ncaaf' carries a conference-tier map.
    """
    if category != "ncaaf":
        console.print(
            "[red]clv-tiers only supports 'ncaaf' — it is the only sector with a "
            "conference-tier map.[/red]"
        )
        raise typer.Exit(1)

    from evmax.sectors.ncaaf_tiers import TIER_ORDER, matchup_tier, split_event_title

    kept, excluded_stale = _fetch_clv_rows(
        category, market_type=market_type, mode=mode, since=since,
        venue=venue, max_staleness_h=max_staleness_h, sources_token=sources_token,
    )

    buckets: dict[str, list] = {tier: [] for tier in TIER_ORDER}
    for r in kept:
        a, b = split_event_title(r["event_title"])
        buckets[matchup_tier(a, b)].append(r)

    label = f"{category}" + (f" / {market_type}" if market_type else "")
    label += f" [{mode}]" if mode else " [all modes]"
    if since:
        label += f" since {since}"
    if venue:
        label += f" venue={venue}"

    total = sum(len(v) for v in buckets.values())
    if total == 0:
        note = ""
        if max_staleness_h is not None and excluded_stale:
            note = f" ({excluded_stale} excluded as stale-capture)"
        console.print(
            f"[yellow]No current-code resolved CLV rows for {label} yet.{note}[/yellow]\n"
            "Tier segmentation is wired and will populate as ncaaf games resolve "
            "with backfilled CLV (watch-closes + backfill_clv over the season)."
        )
        return

    _TIER_DESC = {
        "G5": "G5 · both Group-of-Five (softest)",
        "cross": "cross · P4 vs G5",
        "P4": "P4 · both Power-4 (hardest)",
        "FCS": "FCS · buy game / unmapped",
    }
    table = Table(title=f"NCAAF CLV by conference tier — {label}", box=box.SIMPLE)
    table.add_column("Tier", no_wrap=True)
    table.add_column("Games", justify="right")
    table.add_column("CLV rows", justify="right")
    table.add_column("mean CLV", justify="right")
    table.add_column("% +CLV", justify="right")
    table.add_column("gate", justify="center")
    for tier in TIER_ORDER:
        rows = buckets[tier]
        games = len({r["event_id"] for r in rows})
        s = _aggregate_clv(rows)
        if s["n"] == 0:
            table.add_row(_TIER_DESC[tier], str(games), "0", "—", "—", "—")
            continue
        gate = "[green]✓[/green]" if s["clears"] else "[red]✗[/red]"
        table.add_row(
            _TIER_DESC[tier], str(games), str(s["n"]),
            f"{s['mean_clv_pp']:+.2f}pp", f"{s['frac_positive']*100:.0f}%", gate,
        )
    console.print(table)
    console.print(
        f"  gate: n≥{MIN_CLV_RESOLVED}, mean≥{CLV_MIN_MEAN_PP:+.1f}pp, "
        f"%pos≥{CLV_MIN_FRAC_POSITIVE*100:.0f}%  ·  thesis: G5 CLV > P4 CLV"
        + (f"  ·  {excluded_stale} stale-excluded" if excluded_stale else "")
    )


@app.command("board")
def board(
    days: int = typer.Option(30, "--days", "-d", help="Trailing window on game date."),
    sector: Optional[str] = typer.Option(
        None, "--sector", "-s", help="Restrict to one sector."
    ),
    staleness_h: float = typer.Option(
        3.0, "--staleness-h",
        help="CLV stale-close filter (hours before T-30). 0 disables.",
    ),
) -> None:
    """Promotion scoreboard — per (sector, market type, venue) health.

    One row per group: sample counts, Brier blend-vs-sharp, CLV gate status,
    and blend divergence (mean |blended − sharp| pp) — the sharp-passthrough
    detector. This is the 'which sectors can I rely on' view; also served at
    GET /api/promotion-board on the dashboard.
    """
    from evmax.agents.cleanup.promotion_board import compute_promotion_board

    rows = compute_promotion_board(
        days=days,
        staleness_h=staleness_h if staleness_h > 0 else None,
        sector=sector,
    )
    if not rows:
        console.print("[yellow]No prediction rows in the window.[/yellow]")
        return

    table = Table(
        title=f"Promotion board — last {days} days (game date)",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
    )
    _MKT_ABBR = {"moneyline": "ML", "spread": "spr", "total": "tot", "advance": "adv"}
    _VEN_ABBR = {"kalshi": "kal", "polymarket_us": "pmus"}

    table.add_column("Sector", width=9)
    table.add_column("Mkt", width=4)
    table.add_column("Ven", width=4)
    table.add_column("Mode", width=6)
    table.add_column("n c/r/l", justify="right", width=11)
    table.add_column("ΔBr/1k", justify="right", width=6)
    table.add_column("CLV mean/%pos(n)", justify="right", width=16)
    table.add_column("Div", justify="right", width=5)
    table.add_column("Gate", width=4)
    table.add_column("Verdict", min_width=15, no_wrap=False)

    for r in rows:
        clv = r["clv"]
        clv_str = (
            f"{clv['mean_clv_pp']:+.2f}/{clv['frac_positive']*100:.0f}%({clv['n']})"
            if clv["n"] else "—"
        )
        delta = r["brier_delta_per_1000"]
        delta_str = f"{delta:+.1f}" if delta is not None else "—"
        div = r["blend_divergence_pp"]
        if div is None:
            div_str = "—"
        elif r["sharp_passthrough"]:
            div_str = f"[dim]{div:.2f}[/dim]"
        else:
            div_str = f"[green]{div:.2f}[/green]"
        gates = r["gates"]
        gate_str = "".join(
            "[green]✓[/green]" if gates[k]["ok"] else "[red]✗[/red]"
            for k in ("clean_n", "clv_n", "clv_mean", "clv_frac_pos")
        )
        verdict = r["verdict"]
        style = {
            "SHARP-PASSTHROUGH": "red",
            "LIVE-DEGRADING": "red",
            "FAILING-CLV": "yellow",
            "PROMOTE-READY": "green",
            "LIVE-HEALTHY": "green",
        }.get(verdict, "dim")
        verdict_cell = f"[{style}]{verdict}[/{style}]"
        if r["top_blockers"]:
            verdict_cell += f"\n[dim]{' '.join(r['top_blockers'])}[/dim]"

        table.add_row(
            r["sector"],
            _MKT_ABBR.get(r["market_type"], r["market_type"][:4]),
            _VEN_ABBR.get(r["venue"], r["venue"][:4]),
            (r["mode"] or "?")[:6],
            f"{r['n_clean_resolved']}/{r['n_resolved']}/{r['n_logged']}",
            delta_str,
            clv_str,
            div_str,
            gate_str,
            verdict_cell,
        )

    console.print(table)
    console.print(
        "[dim]Div pp = mean |blended − sharp|; moneyline groups under "
        f"{0.5:.1f}pp are sharp-passthrough (no independent model signal). "
        "Gates: clean-n≥30 · CLV n≥30 · mean≥0 · %pos≥55.[/dim]"
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
