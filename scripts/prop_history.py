"""Track historical prop line win rates across the NBA season.

For each live Kalshi NBA prop line, shows:
  - Month-by-month hit rate (Oct → current)
  - Rolling 15-game win rate
  - Early-season vs last-15 comparison
  - Trend direction (improving / declining / stable)

Usage:
    .venv/bin/python scripts/prop_history.py
    .venv/bin/python scripts/prop_history.py --stat rebounds
    .venv/bin/python scripts/prop_history.py --player "derik queen"
    .venv/bin/python scripts/prop_history.py --signal STRONG     # only STRONG backtest lines
    .venv/bin/python scripts/prop_history.py --min-games 10      # require at least N season games
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

import pandas as pd
from scipy.stats import norm
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from evmax.clients.nba_stats import _find_player_id, STAT_COL, _SEASON

console = Console()

# NBA season months in order (2025-26 season)
SEASON_MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
MONTH_MAP = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}

_gamelog_cache: dict[int, Any] = {}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_full_season(player_id: int) -> Optional[pd.DataFrame]:
    if player_id in _gamelog_cache:
        return _gamelog_cache[player_id]
    try:
        from nba_api.stats.endpoints import playergamelogs
        df = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=_SEASON,
            last_n_games_nullable=0,
        ).get_data_frames()[0]
        if df is not None and "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
        _gamelog_cache[player_id] = df
        return df
    except Exception:
        _gamelog_cache[player_id] = None
        return None


def _build_series(df: pd.DataFrame, stat_type: str) -> Optional[pd.Series]:
    col = STAT_COL.get(stat_type)
    if col is None:
        return None
    if col == "__pra__":
        return df["PTS"] + df["REB"] + df["AST"]
    if col == "__bs__":
        return df.get("BLK", 0) + df.get("STL", 0)
    return df[col] if col in df.columns else None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

@dataclass
class MonthStats:
    month: str
    games: int
    hits: int

    @property
    def hit_rate(self) -> Optional[float]:
        return self.hits / self.games if self.games > 0 else None

    @property
    def display(self) -> str:
        if self.games == 0:
            return "[dim]—[/dim]"
        pct = self.hits / self.games
        color = "bold green" if pct >= 0.55 else "green" if pct >= 0.40 else "yellow" if pct >= 0.25 else "dim red"
        return f"[{color}]{self.hits}/{self.games}[/{color}]"


@dataclass
class PropHistory:
    player_name: str
    display_name: str
    stat_type: str
    threshold: float
    kalshi_price: float

    season_games: int
    season_hits: int

    monthly: list[MonthStats] = field(default_factory=list)   # ordered by SEASON_MONTHS

    l15_hits: int = 0
    l15_games: int = 0

    model_prob: float = 0.0
    ev_model: float = 0.0
    ev_actual: float = 0.0
    trend: str = "→"    # ↑ improving  ↓ declining  → stable


def _classify_trend(early_rate: Optional[float], recent_rate: Optional[float], delta_threshold: float = 0.10) -> str:
    """Compare first half of season to L15."""
    if early_rate is None or recent_rate is None:
        return "→"
    delta = recent_rate - early_rate
    if delta >= delta_threshold:
        return "↑"
    if delta <= -delta_threshold:
        return "↓"
    return "→"


def compute_prop_history(
    player_name: str,
    stat_type: str,
    threshold: float,
    kalshi_price: float,
    min_games: int = 5,
) -> Optional[PropHistory]:
    player_id = _find_player_id(player_name)
    if player_id is None:
        return None

    df = _fetch_full_season(player_id)
    if df is None or len(df) < min_games:
        return None

    series = _build_series(df, stat_type)
    if series is None:
        return None

    # Attach month labels (df is sorted newest-first, but month grouping is order-independent)
    months = df["GAME_DATE"].dt.month.map(MONTH_MAP)
    hit_flags = (series >= threshold).values

    season_games = len(series)
    season_hits  = int(hit_flags.sum())

    # Monthly breakdown
    monthly_data: dict[str, MonthStats] = {m: MonthStats(m, 0, 0) for m in SEASON_MONTHS}
    for month_label, hit in zip(months, hit_flags):
        if month_label in monthly_data:
            monthly_data[month_label].games += 1
            if hit:
                monthly_data[month_label].hits += 1

    # L15 (head = most recent, since df is newest-first)
    l15_flags = hit_flags[:15]
    l15_games  = len(l15_flags)
    l15_hits   = int(l15_flags.sum())
    l15_rate   = l15_hits / l15_games if l15_games > 0 else 0.0

    # Early-season rate (everything beyond L15)
    early_flags = hit_flags[15:]
    early_rate  = float(early_flags.mean()) if len(early_flags) >= 5 else None

    trend = _classify_trend(early_rate, l15_rate if l15_games >= 5 else None)

    # Model probability from L15
    l15_values = series.values[:l15_games].astype(float)
    mean_l15 = float(l15_values.mean()) if l15_games > 0 else 0.0
    std_l15  = max(float(l15_values.std()) if l15_games > 1 else 0.0, 0.5)
    model_prob = max(0.01, min(0.99, float(1 - norm.cdf(threshold - 0.5, mean_l15, std_l15))))

    ev_model  = model_prob   / kalshi_price - 1
    ev_actual = (season_hits / season_games) / kalshi_price - 1

    display_name = " ".join(p.capitalize() for p in player_name.split("_"))

    return PropHistory(
        player_name=player_name,
        display_name=display_name,
        stat_type=stat_type,
        threshold=threshold,
        kalshi_price=kalshi_price,
        season_games=season_games,
        season_hits=season_hits,
        monthly=[monthly_data[m] for m in SEASON_MONTHS],
        l15_hits=l15_hits,
        l15_games=l15_games,
        model_prob=model_prob,
        ev_model=ev_model,
        ev_actual=ev_actual,
        trend=trend,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _ev_cell(ev: float) -> str:
    color = "bold green" if ev > 0.10 else "green" if ev > 0.02 else "yellow" if ev > -0.05 else "dim"
    return f"[{color}]{ev:+.0%}[/{color}]"


def display_history_table(
    histories: list[PropHistory],
    min_games: int,
    signal_filter: Optional[str],
) -> None:
    # Determine which months have any games across all players
    active_months = [
        m for m in SEASON_MONTHS
        if any(h.monthly[i].games > 0 for h in histories
               for i, ms in enumerate(SEASON_MONTHS) if ms == m)
    ]

    # Signal filter
    if signal_filter:
        def _sig(h: PropHistory) -> str:
            if h.ev_model > 0.02 and h.ev_actual > 0.02:   return "STRONG"
            elif h.ev_model > 0.02:                          return "MODEL+"
            elif h.ev_actual > 0.02:                        return "HIST+"
            else:                                           return "PASS"
        histories = [h for h in histories if _sig(h) == signal_filter]

    if not histories:
        console.print("[yellow]No lines match the filters.[/yellow]")
        return

    # Sort: STRONG first, then by ev_model descending
    def _sort_key(h: PropHistory):
        strong = (h.ev_model > 0.02 and h.ev_actual > 0.02)
        return (not strong, -h.ev_model)

    histories.sort(key=_sort_key)

    title = f"NBA Prop Win Rate History — {len(histories)} lines | Season {_SEASON}"
    if signal_filter:
        title += f" | signal={signal_filter}"

    table = Table(title=title, box=box.ROUNDED, show_lines=True)

    # CLAUDE.md: Event + Outcome first
    table.add_column("Player (Event)",        min_width=20, no_wrap=False)
    table.add_column("Stat + Line (Outcome)", min_width=16, no_wrap=False)
    table.add_column("K Price",  justify="right", width=8)

    # Monthly win rate columns
    for month in active_months:
        table.add_column(month, justify="center", width=7)

    table.add_column("Season",  justify="center", width=9)  # hits/games
    table.add_column("Hit%",    justify="right",  width=7)
    table.add_column("L15",     justify="center", width=7)
    table.add_column("L15%",    justify="right",  width=7)
    table.add_column("Trend",   justify="center", width=6)
    table.add_column("EV(Mdl)", justify="right",  width=8)
    table.add_column("EV(Hist)",justify="right",  width=8)

    for h in histories:
        stat_disp = h.stat_type.replace("_", " ").title()
        outcome   = f"{stat_disp} O{h.threshold:.1f}"
        hit_pct   = h.season_hits / h.season_games
        l15_pct   = h.l15_hits / h.l15_games if h.l15_games > 0 else 0.0

        hit_color = ("bold green" if hit_pct >= 0.55 else "green" if hit_pct >= 0.40
                     else "yellow" if hit_pct >= 0.25 else "dim")
        l15_color = ("bold green" if l15_pct >= 0.55 else "green" if l15_pct >= 0.40
                     else "yellow" if l15_pct >= 0.25 else "dim")
        trend_color = "green" if h.trend == "↑" else "red" if h.trend == "↓" else "dim"

        month_cells = []
        for i, month in enumerate(SEASON_MONTHS):
            if month in active_months:
                month_cells.append(h.monthly[i].display)

        table.add_row(
            h.display_name,
            outcome,
            f"{h.kalshi_price:.3f}",
            *month_cells,
            f"{h.season_hits}/{h.season_games}",
            f"[{hit_color}]{hit_pct:.1%}[/{hit_color}]",
            f"{h.l15_hits}/{h.l15_games}",
            f"[{l15_color}]{l15_pct:.1%}[/{l15_color}]",
            f"[{trend_color}]{h.trend}[/{trend_color}]",
            _ev_cell(h.ev_model),
            _ev_cell(h.ev_actual),
        )

    console.print(table)


def display_summary(histories: list[PropHistory]) -> None:
    """Print aggregate win rate stats and notable trends."""
    if not histories:
        return

    total_lines = len(histories)
    strong      = sum(1 for h in histories if h.ev_model > 0.02 and h.ev_actual > 0.02)
    improving   = sum(1 for h in histories if h.trend == "↑")
    declining   = sum(1 for h in histories if h.trend == "↓")

    avg_ev_model  = sum(h.ev_model  for h in histories) / total_lines
    avg_ev_actual = sum(h.ev_actual for h in histories) / total_lines

    # Players with the most STRONG lines
    from collections import Counter
    strong_players = Counter(
        h.display_name for h in histories if h.ev_model > 0.02 and h.ev_actual > 0.02
    )

    console.print(f"\n[bold cyan]Season Win Rate Summary[/bold cyan] — {total_lines} lines")
    console.print(f"  STRONG (both +EV):  [bold green]{strong}[/bold green]")
    console.print(f"  Trend ↑ improving:  [green]{improving}[/green]  |  Trend ↓ declining: [red]{declining}[/red]")
    console.print(f"  Avg EV (model):     {avg_ev_model:+.1%}  |  Avg EV (actual):  {avg_ev_actual:+.1%}")

    if strong_players:
        top = strong_players.most_common(5)
        console.print(f"  Top players (STRONG lines): " +
                      ", ".join(f"[bold]{name}[/bold] ({n})" for name, n in top))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _fetch_markets():
    from evmax.clients.kalshi import KalshiClient
    async with KalshiClient() as kalshi:
        return await kalshi.get_markets("nba_props")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NBA prop line win rate history.")
    parser.add_argument("--stat",       type=str, default=None,
                        help="Filter to stat type (points, rebounds, assists, threes, ...)")
    parser.add_argument("--player",     type=str, default=None,
                        help="Filter to player (partial name match)")
    parser.add_argument("--signal",     type=str, default=None,
                        choices=["STRONG", "MODEL+", "HIST+", "PASS"],
                        help="Filter by backtest signal")
    parser.add_argument("--min-games",  type=int, default=5,
                        help="Minimum season games to include player (default: 5)")
    parser.add_argument("--min-ev",     type=float, default=0.0,
                        help="Minimum EV (model or actual) to include in table")
    args = parser.parse_args()

    console.print("\n[bold cyan]Fetching Kalshi NBA prop markets...[/bold cyan]")
    markets = asyncio.run(_fetch_markets())

    if not markets:
        console.print("[red]No NBA prop markets found.[/red]")
        sys.exit(1)

    # Deduplicate (player, stat, threshold) — keep lowest price
    seen: dict[tuple, float] = {}
    for m in markets:
        if not (m.player_name and m.stat_type and m.threshold is not None):
            continue
        key = (m.player_name, m.stat_type, m.threshold)
        if key not in seen or m.yes_price < seen[key]:
            seen[key] = m.yes_price

    # Optional pre-filters
    if args.stat:
        seen = {k: v for k, v in seen.items() if k[1] == args.stat.lower()}
    if args.player:
        seen = {k: v for k, v in seen.items()
                if args.player.lower() in k[0].replace("_", " ")}

    console.print(f"[dim]{len(markets)} markets → {len(seen)} unique lines. Computing season history...[/dim]\n")

    histories: list[PropHistory] = []
    skipped = 0

    for (player_name, stat_type, threshold), yes_price in seen.items():
        h = compute_prop_history(
            player_name=player_name,
            stat_type=stat_type,
            threshold=threshold,
            kalshi_price=yes_price,
            min_games=args.min_games,
        )
        if h is None:
            skipped += 1
        else:
            # Apply EV filter
            if max(h.ev_model, h.ev_actual) < args.min_ev:
                skipped += 1
                continue
            histories.append(h)

    console.print(f"[dim]{len(histories)} computed, {skipped} skipped (not found / < {args.min_games} games / EV filter).[/dim]\n")

    if not histories:
        console.print("[yellow]No results. Try --min-games 3 or broaden filters.[/yellow]")
        sys.exit(0)

    display_history_table(histories, args.min_games, args.signal)
    display_summary(histories)
    console.print()


if __name__ == "__main__":
    main()
