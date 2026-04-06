"""Backtest and cross-reference NBA player prop lines.

For each live Kalshi NBA prop market, computes:
  - Season hit rate (actual historical performance)
  - L15 hit rate (recent form)
  - Model probability (normal distribution fitted to L15)
  - EV vs Kalshi price using each source

Signal classification:
  STRONG  — both model AND actual history agree it's +EV (>2%)
  MODEL+  — model says +EV, history is neutral/unclear
  HIST+   — historical hit rate says +EV, model diverges
  PASS    — both sources say negative EV

Usage:
    .venv/bin/python scripts/backtest_props.py
    .venv/bin/python scripts/backtest_props.py --min-ev 0.05
    .venv/bin/python scripts/backtest_props.py --stat points
    .venv/bin/python scripts/backtest_props.py --signal STRONG
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any, Optional

# Silence structlog INFO noise from evmax internals
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

from scipy.stats import norm
from rich import box
from rich.console import Console
from rich.table import Table

from evmax.clients.nba_stats import _find_player_id, STAT_COL, _MIN_GAMES, _SEASON

console = Console()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PropResult:
    player_name: str       # underscore-joined normalized name
    display_name: str      # "Title Case" for display
    stat_type: str
    threshold: float
    kalshi_price: float

    season_hits: int
    season_games: int
    full_hit_rate: float   # season_hits / season_games

    l15_hits: int
    l15_games: int         # ≤15 (some players have <15 games this season)
    l15_rate: float        # l15_hits / l15_games

    model_prob: float      # normal dist P(stat >= threshold) from L15 mean/std
    ev_model: float        # model_prob / kalshi_price - 1
    ev_actual: float       # full_hit_rate / kalshi_price - 1

    signal: str            # STRONG / MODEL+ / HIST+ / PASS


# ---------------------------------------------------------------------------
# Game log fetching (full season, uncached — separate from nba_stats cache)
# ---------------------------------------------------------------------------

_gamelog_cache: dict[int, Any] = {}


def _fetch_full_season(player_id: int):
    """Fetch the full 2025-26 season game log for a player. Session-cached."""
    if player_id in _gamelog_cache:
        return _gamelog_cache[player_id]
    try:
        from nba_api.stats.endpoints import playergamelogs
        df = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=_SEASON,
            last_n_games_nullable=0,  # 0 = all games this season
        ).get_data_frames()[0]
        # Guarantee newest-first sort (nba_api default, but defensive)
        if "GAME_DATE" in df.columns:
            df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
        _gamelog_cache[player_id] = df
        return df
    except Exception as exc:
        _gamelog_cache[player_id] = None
        return None


def _build_series(df, stat_type: str):
    """Return per-game stat Series, or None if stat unavailable."""
    col = STAT_COL.get(stat_type)
    if col is None:
        return None
    if col == "__pra__":
        return df["PTS"] + df["REB"] + df["AST"]
    if col == "__bs__":
        return df.get("BLK", 0) + df.get("STL", 0)
    if col not in df.columns:
        return None
    return df[col]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_prop_result(
    player_name: str,
    stat_type: str,
    threshold: float,
    yes_price: float,
) -> Optional[PropResult]:
    """Compute backtest statistics for one (player, stat, threshold) triple."""
    player_id = _find_player_id(player_name)
    if player_id is None:
        return None

    df = _fetch_full_season(player_id)
    if df is None or len(df) < _MIN_GAMES:
        return None

    series = _build_series(df, stat_type)
    if series is None:
        return None

    # Full-season stats
    season_games = len(series)
    # "over threshold" means stat > threshold (e.g., 20+ points = score > 19)
    season_hits = int((series >= threshold).sum())
    full_hit_rate = season_hits / season_games

    # Last-15 stats (slice to at most 15 newest games)
    l15 = series.head(15)
    l15_games = len(l15)
    l15_hits = int((l15 >= threshold).sum())
    l15_rate = l15_hits / l15_games if l15_games > 0 else 0.0

    # Model probability (normal distribution fitted to L15, continuity correction)
    mean_l15 = float(l15.mean())
    std_l15 = max(float(l15.std()) if l15_games > 1 else 0.0, 0.5)
    model_prob = max(0.01, min(0.99, float(1 - norm.cdf(threshold - 0.5, mean_l15, std_l15))))

    # EV relative to Kalshi implied probability
    ev_model  = model_prob    / yes_price - 1
    ev_actual = full_hit_rate / yes_price - 1

    # Signal
    if ev_model > 0.02 and ev_actual > 0.02:
        signal = "STRONG"
    elif ev_model > 0.02:
        signal = "MODEL+"
    elif ev_actual > 0.02:
        signal = "HIST+"
    else:
        signal = "PASS"

    display_name = " ".join(p.capitalize() for p in player_name.split("_"))

    return PropResult(
        player_name=player_name,
        display_name=display_name,
        stat_type=stat_type,
        threshold=threshold,
        kalshi_price=yes_price,
        season_hits=season_hits,
        season_games=season_games,
        full_hit_rate=full_hit_rate,
        l15_hits=l15_hits,
        l15_games=l15_games,
        l15_rate=l15_rate,
        model_prob=model_prob,
        ev_model=ev_model,
        ev_actual=ev_actual,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_SIGNAL_STYLE = {
    "STRONG": "bold green",
    "MODEL+": "cyan",
    "HIST+":  "yellow",
    "PASS":   "dim",
}


def display_table(results: list[PropResult], min_ev: float, signal_filter: Optional[str]) -> None:
    filtered = [
        r for r in results
        if max(r.ev_model, r.ev_actual) >= min_ev
        and (signal_filter is None or r.signal == signal_filter)
    ]
    if not filtered:
        console.print(f"[yellow]No props pass filters (min_ev={min_ev:.1%}, signal={signal_filter or 'all'}).[/yellow]")
        return

    filtered.sort(key=lambda r: r.ev_model, reverse=True)

    title = (
        f"NBA Prop Backtest — {len(filtered)} lines | Season {_SEASON}"
        + (f" | min_ev {min_ev:.0%}" if min_ev else "")
        + (f" | signal {signal_filter}" if signal_filter else "")
    )

    table = Table(title=title, box=box.ROUNDED, show_lines=True)

    # Per CLAUDE.md: Event and Outcome columns must come first
    table.add_column("Player (Event)",    min_width=22, no_wrap=False)
    table.add_column("Stat + Line (Outcome)", min_width=18, no_wrap=False)
    table.add_column("K Price", justify="right", width=8)
    table.add_column("Season",  justify="center", width=9)   # hits/games
    table.add_column("Hit%",    justify="right",  width=7)
    table.add_column("L15",     justify="center", width=7)   # hits/15
    table.add_column("L15%",    justify="right",  width=7)
    table.add_column("Model%",  justify="right",  width=8)
    table.add_column("EV(Mdl)", justify="right",  width=9)
    table.add_column("EV(Hist)",justify="right",  width=9)
    table.add_column("Signal",  justify="center", width=8)

    for r in filtered:
        stat_disp = r.stat_type.replace("_", " ").title()
        outcome = f"{stat_disp} O{r.threshold:.1f}"

        ev_m_color = ("bold green" if r.ev_model > 0.10
                      else "green" if r.ev_model > 0.02
                      else "yellow" if r.ev_model > -0.05
                      else "dim red")
        ev_a_color = ("bold green" if r.ev_actual > 0.10
                      else "green" if r.ev_actual > 0.02
                      else "yellow" if r.ev_actual > -0.05
                      else "dim red")
        sig_style = _SIGNAL_STYLE.get(r.signal, "dim")

        table.add_row(
            r.display_name,
            outcome,
            f"{r.kalshi_price:.3f}",
            f"{r.season_hits}/{r.season_games}",
            f"{r.full_hit_rate:.1%}",
            f"{r.l15_hits}/{r.l15_games}",
            f"{r.l15_rate:.1%}",
            f"{r.model_prob:.1%}",
            f"[{ev_m_color}]{r.ev_model:+.1%}[/{ev_m_color}]",
            f"[{ev_a_color}]{r.ev_actual:+.1%}[/{ev_a_color}]",
            f"[{sig_style}]{r.signal}[/{sig_style}]",
        )

    console.print(table)


def display_calibration(results: list[PropResult]) -> None:
    """Print model calibration metrics across all lines."""
    if not results:
        return

    model_probs  = [r.model_prob    for r in results]
    actual_rates = [r.full_hit_rate for r in results]

    n = len(results)
    mae  = sum(abs(m - a) for m, a in zip(model_probs, actual_rates)) / n
    bias = sum(m - a       for m, a in zip(model_probs, actual_rates)) / n

    # Correlation
    try:
        import statistics
        if n >= 2:
            cov = sum((m - sum(model_probs)/n) * (a - sum(actual_rates)/n)
                      for m, a in zip(model_probs, actual_rates)) / (n - 1)
            std_m = statistics.stdev(model_probs)
            std_a = statistics.stdev(actual_rates)
            corr  = cov / (std_m * std_a) if std_m > 0 and std_a > 0 else float("nan")
        else:
            corr = float("nan")
    except Exception:
        corr = float("nan")

    signal_counts = {s: sum(1 for r in results if r.signal == s)
                     for s in ("STRONG", "MODEL+", "HIST+", "PASS")}

    bias_dir = "overestimates" if bias > 0.005 else "underestimates" if bias < -0.005 else "well-calibrated"

    console.print(f"\n[bold cyan]Calibration Summary[/bold cyan] — {n} lines")
    console.print(f"  MAE  (model vs actual):  [bold]{mae:.3f}[/bold]  ({mae:.1%})")
    console.print(f"  Bias (model − actual):   [bold]{bias:+.3f}[/bold]  → model {bias_dir}")
    if not (corr != corr):  # not NaN
        console.print(f"  Correlation (r):         [bold]{corr:.3f}[/bold]")
    console.print(
        f"  Signals — "
        f"[bold green]STRONG[/bold green]: {signal_counts['STRONG']}  "
        f"[cyan]MODEL+[/cyan]: {signal_counts['MODEL+']}  "
        f"[yellow]HIST+[/yellow]: {signal_counts['HIST+']}  "
        f"[dim]PASS[/dim]: {signal_counts['PASS']}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _fetch_markets():
    from evmax.clients.kalshi import KalshiClient
    async with KalshiClient() as kalshi:
        return await kalshi.get_markets("nba_props")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Kalshi NBA player prop lines.")
    parser.add_argument("--min-ev",  type=float, default=0.0,
                        help="Minimum EV (model or actual) to include in table (default: 0.0 = all)")
    parser.add_argument("--stat",    type=str,   default=None,
                        help="Filter to a specific stat type (points, rebounds, assists, ...)")
    parser.add_argument("--signal",  type=str,   default=None,
                        choices=["STRONG", "MODEL+", "HIST+", "PASS"],
                        help="Filter to a specific signal type")
    parser.add_argument("--player",  type=str,   default=None,
                        help="Filter to a player (partial name match, case-insensitive)")
    args = parser.parse_args()

    console.print("\n[bold cyan]Fetching Kalshi NBA prop markets...[/bold cyan]")
    markets = asyncio.run(_fetch_markets())

    if not markets:
        console.print("[red]No NBA prop markets found.[/red]")
        sys.exit(1)

    # Deduplicate by (player, stat, threshold) — keep lowest (most conservative) price
    seen: dict[tuple, float] = {}
    for m in markets:
        if not (m.player_name and m.stat_type and m.threshold is not None):
            continue
        key = (m.player_name, m.stat_type, m.threshold)
        # Take the lowest Kalshi ask (most conservative EV estimate)
        if key not in seen or m.yes_price < seen[key]:
            seen[key] = m.yes_price

    # Optional filters
    if args.stat:
        seen = {k: v for k, v in seen.items() if k[1] == args.stat.lower()}
    if args.player:
        seen = {k: v for k, v in seen.items()
                if args.player.lower() in k[0].replace("_", " ")}

    console.print(f"[dim]Fetched {len(markets)} markets → {len(seen)} unique lines after dedup. Computing...[/dim]\n")

    results: list[PropResult] = []
    skipped = 0

    for (player_name, stat_type, threshold), yes_price in seen.items():
        r = compute_prop_result(player_name, stat_type, threshold, yes_price)
        if r is None:
            skipped += 1
        else:
            results.append(r)

    console.print(f"[dim]Computed {len(results)} results, skipped {skipped} (player not found / insufficient data).[/dim]\n")

    display_table(results, min_ev=args.min_ev, signal_filter=args.signal)
    display_calibration(results)
    console.print()


if __name__ == "__main__":
    main()
