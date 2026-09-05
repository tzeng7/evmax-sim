"""Discord embed builders — the dashboard's tables, rendered as monospace.

Every table here is a column-for-column port of a React dashboard panel
(``frontend/src/components/*.tsx``), in the panel's column order, with the
panel's own cell formatting (``probToCents`` → :func:`cents`, ``toFixed`` →
:func:`to_fixed`, ``Math.round`` → :func:`js_round`) and the panel's row
selection/order left to the caller (``evmax.web.playlist`` for the scan view,
``evmax.web.app._open_bets`` / ``_settled_bets`` for the others). What the
dashboard shows as a badge (``shadow`` mode, ``MAKER``, ``NEW``, ``LIVE``)
is a text tag in the same column.

Discord has no table markup, so each table is a code block inside an embed
description. Long tables are split across embeds (each chunk repeats the
header) and the :mod:`evmax.discord_bot.client` batches embeds into messages
under Discord's per-message caps.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Optional, Sequence

from evmax.discord_bot.client import (
    EMBED_DESCRIPTION_MAX,
    EMBED_FOOTER_MAX,
    EMBED_TITLE_MAX,
)

# Embed accent colors (dashboard palette).
COLOR_PLAYS = 0x3FB950      # green — the EV column color
COLOR_EMPTY = 0x8B949E      # muted
COLOR_INFO = 0x58A6FF
COLOR_WARNING = 0xE0B341    # the dashboard's shadow-badge amber
COLOR_CRITICAL = 0xF85149
COLOR_MAKER = 0xC678DD      # the dashboard's maker purple

SEVERITY_COLORS = {"info": COLOR_INFO, "warning": COLOR_WARNING, "critical": COLOR_CRITICAL}
SEVERITY_EMOJI = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}

# Room left in a 4096-char description for the code-block fences.
TABLE_CHUNK_MAX = EMBED_DESCRIPTION_MAX - 16
# Dashboard constants (frontend/src/components/OpenPositions.tsx).
SCAN_KELLY_BASELINE = 0.5
OPEN_POSITIONS_ROW_CAP = 40
RECENT_SETTLED_ROW_CAP = 50
KELLY_PER_BET_CAP = 0.05


# ---------------------------------------------------------------------------
# JS-parity number formatting (frontend/src/lib/odds.ts)
# ---------------------------------------------------------------------------

def to_fixed(x: Any, digits: int) -> str:
    """``Number.prototype.toFixed`` — rounds the exact binary value of ``x``
    half AWAY from zero (JS negates first, then picks the larger n on a tie),
    which is not Python's ``round``/``format`` (half-to-even)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "NaN"
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(v).quantize(q, rounding=ROUND_HALF_UP))


def js_round(x: float) -> int:
    """``Math.round`` — half toward +infinity."""
    return int(math.floor(x + 0.5))


def cents(p: Any) -> str:
    """``probToCents``: ``0.12 → "12¢"``, ``0.127 → "12.7¢"``; ``"-"`` for
    null / non-finite / outside (0, 1)."""
    if p is None:
        return "-"
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v) or v <= 0 or v >= 1:
        return "-"
    s = to_fixed(v * 100, 1)
    if s.endswith(".0"):
        s = s[:-2]
    return s + "¢"


def pct1(x: Any) -> str:
    """``x.toFixed(1) + '%'``."""
    return f"{to_fixed(x, 1)}%"


def venue_short(v: Optional[str]) -> str:
    if v == "polymarket_us":
        return "Poly"
    if v == "kalshi":
        return "Kalshi"
    return v or ""


# ---------------------------------------------------------------------------
# Scan Results (frontend/src/components/ScanResults.tsx)
# ---------------------------------------------------------------------------

def legs_of(g: dict[str, Any]) -> list[dict[str, Any]]:
    """The venue legs for a row: the full best-execution option set when the
    same bet is quoted on multiple venues (winner first), else the row."""
    opts = g.get("venue_options")
    return list(opts) if opts and len(opts) > 1 else [g]


def venue_option_label(leg: dict[str, Any]) -> str:
    """One entry of the dashboard's venue dropdown: venue + taker EV, with a
    ``mkr`` tag when the leg only clears as a maker."""
    label = f"{venue_short(leg.get('venue'))} · {to_fixed(leg.get('ev_pct') or 0.0, 1)}%"
    return label + (" mkr" if leg.get("maker_only") else "")


def recomputed_stake(g: dict[str, Any], bankroll: float, kelly: float, scan_kelly: float) -> float:
    """Dashboard ``recomputedStake``: rescale the scan-time Kelly fraction to
    the current Kelly knob and re-apply the 5% per-bet cap."""
    frac = float(g.get("kelly_fraction") or 0.0)
    scaled = frac * (kelly / scan_kelly) if scan_kelly > 0 else frac
    return bankroll * min(scaled, KELLY_PER_BET_CAP)


def default_fill(
    g: dict[str, Any], bankroll: float, kelly: float, scan_kelly: float
) -> tuple[str, str]:
    """Dashboard ``defaultFill``: the Fill ¢ / Stake ($) a row is seeded with.
    Maker-only plays seed to the actionable resting bid and the maker-sized
    Kelly stake; everything else to the taker ask and the taker stake."""
    if g.get("maker_only") and g.get("maker_bid_price") is not None:
        frac = float(g.get("maker_bid_kelly_fraction") or 0.0)
        return cents(g["maker_bid_price"]), to_fixed(bankroll * min(frac, KELLY_PER_BET_CAP), 2)
    return cents(g.get("kalshi_price")), to_fixed(recomputed_stake(g, bankroll, kelly, scan_kelly), 2)


SCAN_HEADERS: tuple[str, ...] = (
    "Date", "Sector", "Venue", "Event", "Outcome",
    "Ask", "Fair Value", "Model", "EV", "Maker EV", "Limit ¢", "Bid ¢",
    "Fill ¢", "Stake ($)", "Models",
)
# "<" left / ">" right — the dashboard's `.num` columns are right-aligned.
SCAN_ALIGN: tuple[str, ...] = (
    "<", "<", "<", "<", "<",
    ">", ">", ">", ">", ">", ">", ">",
    ">", ">", "<",
)


def scan_row(
    g: dict[str, Any],
    bankroll: float,
    kelly: Optional[float] = None,
    scan_kelly: Optional[float] = None,
) -> list[str]:
    """One Scan Results row (winner leg — the dashboard's default before any
    venue-dropdown interaction), cell by cell in column order."""
    kelly = SCAN_KELLY_BASELINE if kelly is None else kelly
    scan_kelly = kelly if scan_kelly is None else scan_kelly
    leg = g
    legs = legs_of(g)
    has_dropdown = len(legs) > 1
    mode = leg.get("mode") or "live"
    is_live = mode == "live"
    is_maker = bool(leg.get("maker_only"))

    sector = str(g.get("sector") or "")
    if not is_live:
        sector += f" {mode}"
    if is_maker:
        sector += " MAKER"

    if has_dropdown:
        venue = " | ".join(venue_option_label(l) for l in legs)
    else:
        venue = venue_short(leg.get("venue"))

    outcome = str(g.get("display_label") or "")
    if not has_dropdown and g.get("alt_venue"):
        outcome += f" · also {venue_short(g['alt_venue'])}"
        if g.get("alt_venue_price") is not None:
            outcome += f" {cents(g['alt_venue_price'])}"

    fill_odds, fill_stake = default_fill(leg, bankroll, kelly, scan_kelly)
    maker_ev = leg.get("maker_ev_pct")
    return [
        str(g.get("event_date") or ""),
        sector,
        venue,
        str(g.get("event_title") or ""),
        outcome,
        cents(leg.get("kalshi_price")),
        cents(leg.get("true_prob")),
        pct1(float(leg.get("true_prob") or 0.0) * 100),
        pct1(leg.get("ev_pct") or 0.0),
        pct1(maker_ev) if maker_ev is not None else "—",
        cents(leg["maker_limit_price"]) if leg.get("maker_limit_price") is not None else "—",
        cents(leg["maker_bid_price"]) if leg.get("maker_bid_price") is not None else "—",
        fill_odds,
        fill_stake,
        str(leg.get("model_sources") or ""),
    ]


def scan_rows(
    gaps: Sequence[dict[str, Any]],
    bankroll: float,
    kelly: Optional[float] = None,
    scan_kelly: Optional[float] = None,
) -> list[list[str]]:
    return [scan_row(g, bankroll, kelly, scan_kelly) for g in gaps]


def scan_title(n_plays: int, markets_fetched: int, markets_matched: int) -> str:
    """The Scan Results panel header, verbatim."""
    return f"Scan Results — {n_plays} plays ({markets_fetched} markets, {markets_matched} matched)"


def scan_result_embeds(
    gaps: Sequence[dict[str, Any]],
    *,
    markets_fetched: int,
    markets_matched: int,
    bankroll: float,
    kelly: float,
    sectors: Optional[Sequence[str]] = None,
    date_from: str = "",
    date_to: str = "",
    duration_s: Optional[float] = None,
    errors: Optional[Sequence[str]] = None,
    source: Optional[str] = None,
) -> list[dict[str, Any]]:
    """The Scan Results panel as embeds: title = the panel header, body = the
    table (chunked across embeds when long, header repeated per chunk),
    footer = the scan's bankroll / Kelly / sectors / window / duration."""
    title = scan_title(len(gaps), markets_fetched, markets_matched)
    footer_bits = [f"Bankroll ${bankroll:,.2f}", f"Kelly {kelly:g}"]
    if sectors:
        footer_bits.append("sectors: " + ", ".join(sectors))
    if date_from or date_to:
        footer_bits.append(f"window: {date_from or '…'} → {date_to or '…'}")
    else:
        footer_bits.append("window: today + tomorrow")
    if duration_s is not None:
        footer_bits.append(f"{duration_s:.1f}s")
    if source:
        footer_bits.append(source)
    footer = " · ".join(footer_bits)
    if errors:
        footer += "\nerrors: " + "; ".join(str(e) for e in errors)

    if not gaps:
        return [_embed(title, "No +EV plays found.", COLOR_EMPTY, footer)]

    rows = scan_rows(gaps, bankroll, kelly, kelly)
    chunks = table_chunks(SCAN_HEADERS, rows, SCAN_ALIGN, TABLE_CHUNK_MAX)
    return _table_embeds(title, chunks, COLOR_PLAYS, footer)


# ---------------------------------------------------------------------------
# Open Positions (frontend/src/components/OpenPositions.tsx)
# ---------------------------------------------------------------------------

OPEN_HEADERS: tuple[str, ...] = (
    "", "Date", "Sector", "Venue", "Event", "Outcome", "Ask", "Fair Value", "EV", "Stake",
)
OPEN_ALIGN: tuple[str, ...] = ("<", "<", "<", "<", "<", "<", ">", ">", ">", ">")


def preview_stake(b: dict[str, Any], bankroll: float, kelly: float) -> float:
    """Dashboard ``previewStake``: stored Kelly fraction rescaled from the
    0.5 scan baseline to the current knob, capped at 5%."""
    stored = float(b.get("kelly_fraction") or 0.0)
    scaled = stored * (kelly / SCAN_KELLY_BASELINE)
    return bankroll * min(scaled, KELLY_PER_BET_CAP)


def open_position_row(
    b: dict[str, Any], bankroll: float, kelly: float, scan_mids: Optional[set[str]] = None
) -> list[str]:
    tags: list[str] = []
    if scan_mids and b.get("market_id") in scan_mids:
        tags.append("NEW")
    if b.get("status") == "in_progress":
        tags.append("LIVE")
    return [
        " ".join(tags),
        str(b.get("event_date") or ""),
        str(b.get("sector") or ""),
        venue_short(b.get("venue")),
        str(b.get("event_title") or ""),
        str(b.get("display_label") or ""),
        cents(b.get("kalshi_yes_price")),
        cents(b.get("blended_true_prob")),
        pct1(float(b.get("ev_pct") or 0.0) * 100),
        f"${to_fixed(preview_stake(b, bankroll, kelly), 2)}",
    ]


def open_positions_embeds(
    bets: Sequence[dict[str, Any]],
    *,
    bankroll: float,
    kelly: float,
    scan_mids: Optional[set[str]] = None,
    sector: Optional[str] = None,
    source: Optional[str] = None,
) -> list[dict[str, Any]]:
    """The Open Positions panel: unresolved, unplaced live rows (the dashboard
    caps the render at 40). ``source`` (e.g. ``bankroll live:kalshi``) is
    appended to the footer when the bankroll came from a live balance."""
    rows_in = [b for b in bets if not sector or b.get("sector") == sector]
    title = f"Open Positions ({len(rows_in)})"
    footer = f"Bankroll ${bankroll:,.2f} · Kelly {kelly:g}"
    if source:
        footer += f" · {source}"
    if not rows_in:
        return [_embed(title, "No open positions.", COLOR_EMPTY, footer)]
    rows = [open_position_row(b, bankroll, kelly, scan_mids) for b in rows_in[:OPEN_POSITIONS_ROW_CAP]]
    if len(rows_in) > OPEN_POSITIONS_ROW_CAP:
        footer += f" · showing {OPEN_POSITIONS_ROW_CAP} of {len(rows_in)}"
    chunks = table_chunks(OPEN_HEADERS, rows, OPEN_ALIGN, TABLE_CHUNK_MAX)
    return _table_embeds(title, chunks, COLOR_INFO, footer)


# ---------------------------------------------------------------------------
# Recent Settled Bets (frontend/src/components/RecentSettled.tsx)
# ---------------------------------------------------------------------------

SETTLED_HEADERS: tuple[str, ...] = (
    "Date", "Sector", "Venue", "Event", "Outcome", "Ask", "Model", "EV", "Result", "P&L",
)
SETTLED_ALIGN: tuple[str, ...] = ("<", "<", "<", "<", "<", ">", ">", ">", "<", ">")


def settled_pnl(b: dict[str, Any]) -> float:
    """The panel's row P&L (``pnl`` in RecentSettled.tsx): gross of fees."""
    stake = b.get("placed_stake") or (b.get("bankroll_used") or 250.0) * (b.get("kelly_fraction") or 0.0)
    price = b.get("placed_price") or b.get("kalshi_yes_price") or 0.5
    if price <= 0 or price >= 1:
        return 0.0
    return stake * (1.0 / price - 1.0) if b.get("outcome") == 1 else -stake


def settled_row(b: dict[str, Any]) -> list[str]:
    p = settled_pnl(b)
    return [
        str(b.get("event_date") or ""),
        str(b.get("sector") or ""),
        venue_short(b.get("venue")),
        str(b.get("event_title") or ""),
        str(b.get("display_label") or ""),
        f"{js_round(float(b.get('kalshi_yes_price') or 0.0) * 100)}c",
        f"{js_round(float(b.get('blended_true_prob') or 0.0) * 100)}%",
        pct1(float(b.get("ev_pct") or 0.0) * 100),
        "WON" if b.get("outcome") == 1 else "LOST",
        f"${to_fixed(p, 2)}",
    ]


def recent_settled_embeds(
    bets: Sequence[dict[str, Any]],
    *,
    summary: Optional[dict[str, Any]] = None,
    placed_only: bool = False,
) -> list[dict[str, Any]]:
    """The Recent Settled Bets panel (newest first, 50-row cap like the
    dashboard payload) with the KPI summary in the footer."""
    rows_in = [b for b in bets if b.get("placed") == 1] if placed_only else list(bets)
    title = "Recent Settled Bets" + (" — Placed Only" if placed_only else "")
    footer = ""
    if summary:
        footer = (
            f"{summary.get('total_bets', 0)} bets · {summary.get('wins', 0)}W / "
            f"{summary.get('losses', 0)}L · win rate {summary.get('win_rate', 0.0)}% · "
            f"P&L ${summary.get('total_pnl', 0.0):,.2f} · ROI {summary.get('roi_pct', 0.0)}% · "
            f"avg EV {summary.get('avg_ev', 0.0)}%"
        )
    if not rows_in:
        return [_embed(title, "No settled bets.", COLOR_EMPTY, footer)]
    rows = [settled_row(b) for b in rows_in[:RECENT_SETTLED_ROW_CAP]]
    chunks = table_chunks(SETTLED_HEADERS, rows, SETTLED_ALIGN, TABLE_CHUNK_MAX)
    return _table_embeds(title, chunks, COLOR_INFO, footer)


# ---------------------------------------------------------------------------
# Alerts + status
# ---------------------------------------------------------------------------

def alert_embed(title: str, message: str, *, severity: str = "warning") -> dict[str, Any]:
    """An operational alert (Notifier.notify_alert) as one colored embed."""
    emoji = SEVERITY_EMOJI.get(severity, "⚠️")
    return _embed(
        f"{emoji} evmax {severity} — {title}",
        message,
        SEVERITY_COLORS.get(severity, COLOR_WARNING),
        None,
    )


def status_embed(report: dict[str, Any]) -> dict[str, Any]:
    """``run_heartbeat`` output as an embed: green all-clear or the issue list."""
    issues = list(report.get("issues") or [])
    if not issues:
        return _embed("✅ evmax status — all clear", "No pipeline health issues.", COLOR_PLAYS, None)
    worst = "critical" if any(i.get("severity") == "critical" for i in issues) else "warning"
    body = "\n".join(f"• [{i.get('severity')}] {i.get('detail')}" for i in issues)
    return _embed(
        f"{SEVERITY_EMOJI[worst]} evmax status — {len(issues)} issue(s)",
        body,
        SEVERITY_COLORS[worst],
        None,
    )


# ---------------------------------------------------------------------------
# Monospace table rendering
# ---------------------------------------------------------------------------

def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Optional[Sequence[str]] = None,
    *,
    widths: Optional[Sequence[int]] = None,
) -> str:
    """Fixed-width text table: header, rule, rows. Columns are sized to their
    widest cell (or ``widths``) and separated by two spaces; ``aligns`` is
    ``"<"``/``">"`` per column."""
    n = len(headers)
    aligns = list(aligns) if aligns else ["<"] * n
    if widths is None:
        widths = [len(h) for h in headers]
        for r in rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))
    widths = list(widths)

    def fmt(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = widths[i]
            parts.append(cell.rjust(w) if aligns[i] == ">" else cell.ljust(w))
        return "  ".join(parts).rstrip()

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines)


def table_chunks(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Optional[Sequence[str]],
    max_chars: int,
) -> list[str]:
    """Split a table into chunks of ≤``max_chars`` rendered characters. Every
    chunk carries the header + rule, and every chunk shares the SAME column
    widths (computed over the whole table) so pages line up."""
    n = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(n):
            widths[i] = max(widths[i], len(r[i]))
    head = render_table(headers, [], aligns, widths=widths)
    chunks: list[str] = []
    cur: list[Sequence[str]] = []
    cur_len = len(head)
    for r in rows:
        line = render_table(headers, [r], aligns, widths=widths).split("\n", 2)[2]
        if cur and cur_len + 1 + len(line) > max_chars:
            chunks.append(render_table(headers, cur, aligns, widths=widths))
            cur, cur_len = [], len(head)
        cur.append(r)
        cur_len += 1 + len(line)
    if cur or not chunks:
        chunks.append(render_table(headers, cur, aligns, widths=widths))
    return chunks


def code_block(text: str) -> str:
    return f"```\n{text}\n```"


def _embed(title: str, description: str, color: int, footer: Optional[str]) -> dict[str, Any]:
    e: dict[str, Any] = {
        "title": title[:EMBED_TITLE_MAX],
        "description": description[:EMBED_DESCRIPTION_MAX],
        "color": color,
    }
    if footer:
        e["footer"] = {"text": footer[:EMBED_FOOTER_MAX]}
    return e


def _table_embeds(
    title: str, chunks: list[str], color: int, footer: Optional[str]
) -> list[dict[str, Any]]:
    """One embed per table chunk; continuation embeds are titled ``(cont. i/n)``
    and only the last carries the footer."""
    out: list[dict[str, Any]] = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        t = title if i == 1 else f"{title} (cont. {i}/{total})"
        out.append(_embed(t, code_block(chunk), color, footer if i == total else None))
    return out


TableBuilder = Callable[..., list[dict[str, Any]]]
