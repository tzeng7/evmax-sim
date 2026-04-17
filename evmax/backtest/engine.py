"""BacktestEngine — orchestrates data loading, matching, and metrics computation."""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog

from evmax.backtest.loader import SOCCER_LEAGUES
from evmax.backtest.metrics import compute_report
from evmax.backtest.metrics_props import compute_prop_report
from evmax.backtest.models import BacktestReport, BacktestRow, PropBacktestReport
from evmax.backtest.sources.espn_walkforward import WalkForwardReport, ESPN_SECTORS

logger = structlog.get_logger(__name__)


def run_backtest(
    sectors: list[str],
    seasons: Optional[list[str]] = None,
    leagues: Optional[list[str]] = None,
    fetch_kalshi: bool = False,
    force_refresh: bool = False,
    ev_threshold: float = 0.02,
    min_volume: float = 0.0,
    stats_filter: Optional[list[str]] = None,
) -> list[BacktestReport | PropBacktestReport]:
    """
    Run historical backtest for the given sectors.

    Args:
        sectors: List of sector names, e.g. ["soccer", "tennis", "nfl_props"]
        seasons: Soccer seasons to include, e.g. ["2425", "2526"]
        leagues: Soccer league codes to include, e.g. ["E0", "SP1"]. None = all.
        fetch_kalshi: If True, fetch resolved Kalshi markets and join for EV analysis.
        force_refresh: Re-download cached CSV/XLSX files.
        ev_threshold: Minimum EV% to count as a positive edge (default 2%).
        min_volume: For prop sectors, exclude markets with volume below this.

    Returns:
        One report per sector — either BacktestReport (match markets) or
        PropBacktestReport (player props).
    """
    if seasons is None:
        seasons = ["2425", "2526"]

    reports: list[BacktestReport | PropBacktestReport | WalkForwardReport] = []

    for sector in sectors:
        try:
            if sector == "soccer":
                report = _run_soccer(seasons, leagues, fetch_kalshi, force_refresh, ev_threshold)
            elif sector == "tennis":
                years = _seasons_to_tennis_years(seasons)
                report = _run_tennis(years, fetch_kalshi, force_refresh, ev_threshold)
            elif sector == "nfl_props":
                report = _run_nfl_props(min_volume, stats_filter)
            elif sector in ESPN_SECTORS:
                report = _run_walkforward(sector, seasons)
            else:
                logger.warning("backtest_unsupported_sector", sector=sector)
                continue
            reports.append(report)
        except Exception as e:
            logger.error("backtest_sector_failed", sector=sector, error=str(e))

    return reports


WALKFORWARD_MONTHS: dict[str, dict[str, list[str]]] = {
    # season code → sector → months
    "2425": {
        "wnba": [f"2025{m:02d}" for m in range(5, 10)],  # 2025 season May-Sep
        "nba": [f"2024{m:02d}" for m in range(10, 13)] + [f"2025{m:02d}" for m in range(1, 7)],
        "ncaab": [f"2024{m:02d}" for m in range(11, 13)] + [f"2025{m:02d}" for m in range(1, 4)],
        "ncaaw": [f"2024{m:02d}" for m in range(11, 13)] + [f"2025{m:02d}" for m in range(1, 4)],
        "baseball": [f"2025{m:02d}" for m in range(3, 11)],
        "nhl": [f"2024{m:02d}" for m in range(10, 13)] + [f"2025{m:02d}" for m in range(1, 7)],
        "nfl": [f"2024{m:02d}" for m in range(9, 13)] + [f"2025{m:02d}" for m in range(1, 3)],
    },
    "2526": {
        "wnba": [f"2026{m:02d}" for m in range(5, 10)],
        "nba": [f"2025{m:02d}" for m in range(10, 13)] + [f"2026{m:02d}" for m in range(1, 7)],
        "ncaab": [f"2025{m:02d}" for m in range(11, 13)] + [f"2026{m:02d}" for m in range(1, 4)],
        "ncaaw": [f"2025{m:02d}" for m in range(11, 13)] + [f"2026{m:02d}" for m in range(1, 4)],
        "baseball": [f"2025{m:02d}" for m in range(3, 11)] + [f"2026{m:02d}" for m in range(3, 11)],
        "nhl": [f"2025{m:02d}" for m in range(10, 13)] + [f"2026{m:02d}" for m in range(1, 7)],
        "nfl": [f"2025{m:02d}" for m in range(9, 13)] + [f"2026{m:02d}" for m in range(1, 3)],
    },
}


def _run_walkforward(sector: str, seasons: list[str]) -> WalkForwardReport:
    """Run ESPN walk-forward backtest for any ESPN-supported sector."""
    from evmax.backtest.sources.espn_walkforward import run_walkforward

    # Collect months for the requested seasons
    months: list[str] = []
    for s in seasons:
        season_months = WALKFORWARD_MONTHS.get(s, {}).get(sector, [])
        months.extend(season_months)

    if not months:
        logger.warning("walkforward_no_months", sector=sector, seasons=seasons)
        # Fallback: if seasons don't map, use all of 2025
        if sector == "wnba":
            months = [f"2025{m:02d}" for m in range(5, 10)]

    # Deduplicate and sort
    months = sorted(set(months))
    logger.info("walkforward_run", sector=sector, months=len(months))
    return run_walkforward(sector, months)


def _run_nfl_props(
    min_volume: float,
    stats_filter: Optional[list[str]] = None,
) -> PropBacktestReport:
    from evmax.backtest.sources.nfl_props import load_nfl_props

    logger.info("backtest_nfl_props_start", stats_filter=stats_filter or "all")
    rows = load_nfl_props(stats_filter=stats_filter)
    logger.info("backtest_nfl_props_loaded", n_rows=len(rows))
    return compute_prop_report(rows, min_volume=min_volume)


def _run_soccer(
    seasons: list[str],
    leagues: Optional[list[str]],
    fetch_kalshi: bool,
    force_refresh: bool,
    ev_threshold: float,
) -> BacktestReport:
    from evmax.backtest.sources.soccer_csv import load_soccer

    logger.info("backtest_soccer_start", seasons=seasons, leagues=leagues or "all")
    rows = load_soccer(seasons, leagues=leagues, force=force_refresh)
    logger.info("backtest_soccer_loaded", n_rows=len(rows))

    if fetch_kalshi:
        rows = _join_kalshi(rows, "soccer")

    league_codes = leagues or list(SOCCER_LEAGUES.keys())
    league_names = [SOCCER_LEAGUES.get(c, c) for c in league_codes]

    return compute_report(rows, "soccer", league_names, seasons, ev_threshold)


def _run_tennis(
    years: list[int],
    fetch_kalshi: bool,
    force_refresh: bool,
    ev_threshold: float,
) -> BacktestReport:
    from evmax.backtest.sources.tennis_xlsx import load_tennis

    logger.info("backtest_tennis_start", years=years)
    rows = load_tennis(years, force=force_refresh)
    logger.info("backtest_tennis_loaded", n_rows=len(rows))

    if fetch_kalshi:
        rows = _join_kalshi(rows, "tennis")

    seasons = [str(y) for y in years]
    tournaments = sorted(set(r.league for r in rows))

    return compute_report(rows, "tennis", tournaments, seasons, ev_threshold)


def _join_kalshi(rows: list[BacktestRow], sector: str) -> list[BacktestRow]:
    """Fetch resolved Kalshi markets and join to rows."""
    from evmax.backtest.kalshi_history import fetch_resolved_markets
    from evmax.backtest.matcher import (
        apply_kalshi_joins,
        match_soccer,
        match_tennis,
    )

    logger.info("backtest_kalshi_fetch_start", sector=sector)
    resolved = asyncio.run(fetch_resolved_markets(sector))
    logger.info("backtest_kalshi_fetched", sector=sector, count=len(resolved))

    if sector == "soccer":
        matches = match_soccer(rows, resolved)
    elif sector == "tennis":
        matches = match_tennis(rows, resolved)
    else:
        return rows

    return apply_kalshi_joins(rows, matches)


def _seasons_to_tennis_years(seasons: list[str]) -> list[int]:
    """Convert season codes like '2425', '2526' to tennis years [2024, 2025, 2026]."""
    years: set[int] = set()
    for s in seasons:
        if len(s) == 4:
            # e.g. "2425" → 2024, 2025
            years.add(2000 + int(s[:2]))
            years.add(2000 + int(s[2:]))
    return sorted(years)
