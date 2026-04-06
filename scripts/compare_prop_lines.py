"""Compare NBA prop lines: Kalshi vs FanDuel/DraftKings vs our nba_stats model.

For each live Kalshi NBA prop line, shows:
  - FanDuel devigged implied probability
  - DraftKings devigged implied probability
  - Our L15 model probability
  - Kalshi implied price
  - EV gap: model vs each source

This lets you see whether our model is in the ballpark of US sportsbooks,
and whether Kalshi is systematically mispriced relative to FD/DK consensus.

Note on API credits: player props cost 4 credits per event per bookmaker on
TheOddsAPI. Running this against all 7 stat markets costs roughly
  7 markets × ~8 NBA game-days × 2 bookmakers × 4 credits ≈ 450 credits.
Use --stat to limit to one stat type (~65 credits per run).

Usage:
    .venv/bin/python scripts/compare_prop_lines.py
    .venv/bin/python scripts/compare_prop_lines.py --stat points
    .venv/bin/python scripts/compare_prop_lines.py --stat rebounds --min-ev 0.03
    .venv/bin/python scripts/compare_prop_lines.py --player "luka doncic"
    .venv/bin/python scripts/compare_prop_lines.py --signal STRONG
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Optional

import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

from rich import box
from rich.console import Console
from rich.table import Table

from evmax.clients.nba_stats import _find_player_id, STAT_COL, _SEASON

console = Console()

# Stat market keys on TheOddsAPI → canonical stat type
_STAT_MARKET_MAP = {
    "player_points":                    "points",
    "player_rebounds":                  "rebounds",
    "player_assists":                   "assists",
    "player_threes":                    "threes",
    "player_steals":                    "steals",
    "player_blocks":                    "blocks",
    "player_points_rebounds_assists":   "points_rebounds_assists",
}
_STAT_TO_MARKET = {v: k for k, v in _STAT_MARKET_MAP.items()}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PropComparison:
    player_name: str        # underscore-joined normalized
    display_name: str
    stat_type: str
    threshold: float

    kalshi_price: float     # Kalshi YES price (implied prob)

    fd_prob: Optional[float] = None    # FanDuel devigged over prob
    dk_prob: Optional[float] = None    # DraftKings devigged over prob
    model_prob: float = 0.0            # Our L15 normal-dist model

    @property
    def consensus_prob(self) -> Optional[float]:
        """Average of FD and DK if both available, else whichever exists."""
        probs = [p for p in (self.fd_prob, self.dk_prob) if p is not None]
        return sum(probs) / len(probs) if probs else None

    @property
    def ev_vs_fd(self) -> Optional[float]:
        return self.fd_prob / self.kalshi_price - 1 if self.fd_prob else None

    @property
    def ev_vs_dk(self) -> Optional[float]:
        return self.dk_prob / self.kalshi_price - 1 if self.dk_prob else None

    @property
    def ev_vs_consensus(self) -> Optional[float]:
        c = self.consensus_prob
        return c / self.kalshi_price - 1 if c else None

    @property
    def ev_vs_model(self) -> Optional[float]:
        return self.model_prob / self.kalshi_price - 1

    @property
    def model_vs_consensus_delta(self) -> Optional[float]:
        """How far our model is from the FD/DK consensus."""
        c = self.consensus_prob
        return self.model_prob - c if c else None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

async def _fetch_kalshi_markets():
    from evmax.clients.kalshi import KalshiClient
    async with KalshiClient() as k:
        return await k.get_markets("nba_props")


async def _fetch_us_book_props(stat_markets: list[str]) -> dict[tuple[str, str, float], dict[str, float]]:
    """Fetch FD + DK prop lines via TheOddsAPI.

    Returns dict: (player_norm, stat_type, threshold) → {"fanduel": prob, "draftkings": prob}
    """
    from evmax.clients.pinnacle import PinnacleClient
    from evmax.players import normalize_player_name
    from evmax.ev.devig import devig_two_way

    results: dict[tuple, dict[str, float]] = {}

    async with PinnacleClient() as client:
        for stat_mkt in stat_markets:
            stat_type = _STAT_MARKET_MAP.get(stat_mkt, stat_mkt)
            # Fetch FD and DK separately so we can label each
            for book_key in ["fanduel", "draftkings"]:
                try:
                    props = await client._fetch_player_props(
                        "basketball_nba", stat_mkt, "nba", book_key
                    )
                    for p in props:
                        player = p.prop_player_name or ""
                        thr = p.total_line
                        prob = p.true_prob_over
                        if not player or thr is None or prob is None:
                            continue
                        key = (player, stat_type, thr)
                        if key not in results:
                            results[key] = {}
                        results[key][book_key] = prob
                except Exception as e:
                    console.print(f"[dim]Warning: {book_key} {stat_mkt} failed: {e}[/dim]")

    return results


def _compute_model_prob(player_name: str, stat_type: str, threshold: float) -> float:
    """Compute L15 normal-distribution model probability."""
    from scipy.stats import norm
    import pandas as pd
    from nba_api.stats.endpoints import playergamelogs

    player_id = _find_player_id(player_name)
    if player_id is None:
        return 0.0

    try:
        df = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=_SEASON,
            last_n_games_nullable=15,
        ).get_data_frames()[0]

        if df is None or df.empty:
            return 0.0

        col = STAT_COL.get(stat_type)
        if col is None:
            return 0.0

        if col == "__pra__":
            series = df["PTS"] + df["REB"] + df["AST"]
        elif col == "__bs__":
            series = df.get("BLK", 0) + df.get("STL", 0)
        elif col in df.columns:
            series = df[col]
        else:
            return 0.0

        vals = series.astype(float)
        mean = float(vals.mean())
        std = max(float(vals.std()) if len(vals) > 1 else 0.0, 0.5)
        return max(0.01, min(0.99, float(1 - norm.cdf(threshold - 0.5, mean, std))))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _ev_cell(ev: Optional[float]) -> str:
    if ev is None:
        return "[dim]—[/dim]"
    color = "bold green" if ev > 0.10 else "green" if ev > 0.02 else "yellow" if ev > -0.05 else "dim red"
    return f"[{color}]{ev:+.0%}[/{color}]"


def _prob_cell(prob: Optional[float]) -> str:
    if prob is None:
        return "[dim]—[/dim]"
    return f"{prob:.1%}"


def _signal(c: PropComparison) -> str:
    ev_c = c.ev_vs_consensus
    ev_m = c.ev_vs_model
    if ev_c is not None and ev_c > 0.02 and ev_m is not None and ev_m > 0.02:
        return "[bold green]STRONG[/bold green]"
    if ev_c is not None and ev_c > 0.02:
        return "[cyan]CONSENS+[/cyan]"
    if ev_m is not None and ev_m > 0.02:
        return "[yellow]MODEL+[/yellow]"
    return "[dim]PASS[/dim]"


def display_comparison_table(comparisons: list[PropComparison], signal_filter: Optional[str]) -> None:
    # Signal filtering
    def _sig_raw(c: PropComparison) -> str:
        ev_c = c.ev_vs_consensus
        ev_m = c.ev_vs_model
        if ev_c is not None and ev_c > 0.02 and ev_m is not None and ev_m > 0.02:
            return "STRONG"
        if ev_c is not None and ev_c > 0.02:
            return "CONSENS+"
        if ev_m is not None and ev_m > 0.02:
            return "MODEL+"
        return "PASS"

    if signal_filter:
        comparisons = [c for c in comparisons if _sig_raw(c) == signal_filter]

    if not comparisons:
        console.print("[yellow]No lines match the filters.[/yellow]")
        return

    # Sort: STRONG first, then by consensus EV desc
    comparisons.sort(key=lambda c: (
        _sig_raw(c) != "STRONG",
        -(c.ev_vs_consensus or -99),
    ))

    table = Table(
        title=f"NBA Prop Line Comparison — {len(comparisons)} lines | FanDuel / DraftKings / Model vs Kalshi",
        box=box.ROUNDED,
        show_lines=True,
    )

    table.add_column("Player (Event)",        min_width=20, no_wrap=False)
    table.add_column("Stat + Line (Outcome)", min_width=16, no_wrap=False)
    table.add_column("K Price",   justify="right", width=8)
    table.add_column("FD Prob",   justify="right", width=8)
    table.add_column("DK Prob",   justify="right", width=8)
    table.add_column("Consensus", justify="right", width=9)
    table.add_column("Model",     justify="right", width=8)
    table.add_column("Δ Mdl−Csn", justify="right", width=9)  # model vs consensus
    table.add_column("EV(FD)",    justify="right", width=8)
    table.add_column("EV(DK)",    justify="right", width=8)
    table.add_column("EV(Csn)",   justify="right", width=8)
    table.add_column("EV(Mdl)",   justify="right", width=8)
    table.add_column("Signal",    justify="center", width=10)

    for c in comparisons:
        stat_disp = c.stat_type.replace("_", " ").title()
        outcome = f"{stat_disp} O{c.threshold:.1f}"

        delta = c.model_vs_consensus_delta
        delta_str = (
            f"[green]{delta:+.1%}[/green]" if delta is not None and delta > 0.03
            else f"[red]{delta:+.1%}[/red]" if delta is not None and delta < -0.03
            else f"[dim]{delta:+.1%}[/dim]" if delta is not None
            else "[dim]—[/dim]"
        )

        table.add_row(
            c.display_name,
            outcome,
            f"{c.kalshi_price:.3f}",
            _prob_cell(c.fd_prob),
            _prob_cell(c.dk_prob),
            _prob_cell(c.consensus_prob),
            _prob_cell(c.model_prob) if c.model_prob > 0 else "[dim]—[/dim]",
            delta_str,
            _ev_cell(c.ev_vs_fd),
            _ev_cell(c.ev_vs_dk),
            _ev_cell(c.ev_vs_consensus),
            _ev_cell(c.ev_vs_model),
            _signal(c),
        )

    console.print(table)


def display_summary(comparisons: list[PropComparison]) -> None:
    if not comparisons:
        return

    with_consensus = [c for c in comparisons if c.consensus_prob is not None]
    if not with_consensus:
        console.print("\n[yellow]No lines with FD/DK data available.[/yellow]")
        return

    strong = sum(1 for c in with_consensus
                 if (c.ev_vs_consensus or 0) > 0.02 and c.ev_vs_model > 0.02)
    consens_only = sum(1 for c in with_consensus
                       if (c.ev_vs_consensus or 0) > 0.02 and not c.ev_vs_model > 0.02)
    model_only = sum(1 for c in comparisons if (c.ev_vs_model or 0) > 0.02
                     and (c.ev_vs_consensus is None or c.ev_vs_consensus <= 0.02))

    # Model calibration vs consensus
    deltas = [c.model_vs_consensus_delta for c in with_consensus
              if c.model_vs_consensus_delta is not None]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    mae = sum(abs(d) for d in deltas) / len(deltas) if deltas else 0.0

    # Average Kalshi vs FD/DK gap
    csn_evs = [c.ev_vs_consensus for c in with_consensus if c.ev_vs_consensus is not None]
    avg_csn_ev = sum(csn_evs) / len(csn_evs) if csn_evs else 0.0

    console.print(f"\n[bold cyan]Comparison Summary[/bold cyan] — {len(with_consensus)}/{len(comparisons)} lines matched to FD/DK")
    console.print(f"  STRONG (consensus + model +EV): [bold green]{strong}[/bold green]")
    console.print(f"  CONSENS+ only:  [cyan]{consens_only}[/cyan]  |  MODEL+ only: [yellow]{model_only}[/yellow]")
    console.print(f"  Avg Kalshi EV vs FD/DK consensus: {avg_csn_ev:+.1%}")
    console.print(f"  Model vs consensus — avg delta: {avg_delta:+.1%}  MAE: {mae:.1%}")
    if abs(avg_delta) < 0.02:
        console.print("  [green]Model is well-aligned with FD/DK consensus[/green]")
    elif avg_delta > 0.02:
        console.print("  [yellow]Model systematically higher than FD/DK consensus[/yellow]")
    else:
        console.print("  [yellow]Model systematically lower than FD/DK consensus[/yellow]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare NBA prop lines: Kalshi vs FD/DK vs model.")
    parser.add_argument("--stat",    type=str, default=None,
                        help="Filter to stat type (points, rebounds, assists, ...)")
    parser.add_argument("--player",  type=str, default=None,
                        help="Filter to player (partial name match)")
    parser.add_argument("--signal",  type=str, default=None,
                        choices=["STRONG", "CONSENS+", "MODEL+", "PASS"],
                        help="Filter by signal")
    parser.add_argument("--min-ev",  type=float, default=None,
                        help="Min EV vs FD/DK consensus to show")
    args = parser.parse_args()

    # Determine which stat markets to fetch
    if args.stat:
        stat_norm = args.stat.lower()
        mkt_key = _STAT_TO_MARKET.get(stat_norm)
        if not mkt_key:
            console.print(f"[red]Unknown stat '{args.stat}'. Use: {', '.join(_STAT_TO_MARKET.keys())}[/red]")
            sys.exit(1)
        stat_markets = [mkt_key]
    else:
        stat_markets = list(_STAT_MARKET_MAP.keys())

    console.print("\n[bold cyan]Fetching Kalshi NBA prop markets...[/bold cyan]")
    kalshi_raw = asyncio.run(_fetch_kalshi_markets())
    if not kalshi_raw:
        console.print("[red]No Kalshi NBA prop markets found.[/red]")
        sys.exit(1)

    # Deduplicate Kalshi: (player, stat, threshold) → lowest price
    kalshi_map: dict[tuple[str, str, float], float] = {}
    for m in kalshi_raw:
        if not (m.player_name and m.stat_type and m.threshold is not None):
            continue
        key = (m.player_name, m.stat_type, m.threshold)
        if key not in kalshi_map or m.yes_price < kalshi_map[key]:
            kalshi_map[key] = m.yes_price

    # Optional pre-filters
    if args.stat:
        kalshi_map = {k: v for k, v in kalshi_map.items() if k[1] == args.stat.lower()}
    if args.player:
        kalshi_map = {k: v for k, v in kalshi_map.items()
                      if args.player.lower() in k[0].replace("_", " ")}

    console.print(f"[dim]{len(kalshi_raw)} Kalshi markets → {len(kalshi_map)} unique lines[/dim]")

    console.print("[bold cyan]Fetching FanDuel + DraftKings prop lines via TheOddsAPI...[/bold cyan]")
    console.print("[dim](This uses TheOddsAPI credits — player props cost 4 credits/event/bookmaker)[/dim]\n")
    us_book_data = asyncio.run(_fetch_us_book_props(stat_markets))
    console.print(f"[dim]US book data: {len(us_book_data)} (player, stat, threshold) entries[/dim]\n")

    # Build comparisons
    console.print("[dim]Computing model probabilities via nba_api...[/dim]\n")
    comparisons: list[PropComparison] = []

    for (player_name, stat_type, threshold), kalshi_price in kalshi_map.items():
        book_data = us_book_data.get((player_name, stat_type, threshold), {})

        # Try ±0.5 threshold fuzzy match if exact miss (books sometimes use 24.5 vs Kalshi 25)
        if not book_data:
            for (p, s, t), bd in us_book_data.items():
                if p == player_name and s == stat_type and abs(t - threshold) <= 0.5:
                    book_data = bd
                    break

        model_prob = _compute_model_prob(player_name, stat_type, threshold)

        display_name = " ".join(part.capitalize() for part in player_name.split("_"))

        c = PropComparison(
            player_name=player_name,
            display_name=display_name,
            stat_type=stat_type,
            threshold=threshold,
            kalshi_price=kalshi_price,
            fd_prob=book_data.get("fanduel"),
            dk_prob=book_data.get("draftkings"),
            model_prob=model_prob,
        )

        # Apply min-ev filter
        if args.min_ev is not None:
            best_ev = max(
                ev for ev in (c.ev_vs_consensus, c.ev_vs_model) if ev is not None
            ) if any(ev is not None for ev in (c.ev_vs_consensus, c.ev_vs_model)) else -99
            if best_ev < args.min_ev:
                continue

        comparisons.append(c)

    console.print(f"[dim]{len(comparisons)} comparisons built.[/dim]\n")

    if not comparisons:
        console.print("[yellow]No results. Try broadening filters.[/yellow]")
        sys.exit(0)

    display_comparison_table(comparisons, args.signal)
    display_summary(comparisons)
    console.print()


if __name__ == "__main__":
    main()
