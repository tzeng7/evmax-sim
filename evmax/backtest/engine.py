"""BacktestEngine — orchestrates data loading, matching, and metrics computation."""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog

from evmax.backtest.loader import SOCCER_LEAGUES
from evmax.backtest.metrics import compute_report
from evmax.backtest.models import BacktestReport, BacktestRow

logger = structlog.get_logger(__name__)


def run_backtest(
    sectors: list[str],
    seasons: Optional[list[str]] = None,
    leagues: Optional[list[str]] = None,
    fetch_kalshi: bool = False,
    force_refresh: bool = False,
    ev_threshold: float = 0.02,
) -> list[BacktestReport]:
    """
    Run historical backtest for the given sectors.

    Args:
        sectors: List of sector names, e.g. ["soccer", "tennis"]
        seasons: Soccer seasons to include, e.g. ["2425", "2526"]
        leagues: Soccer league codes to include, e.g. ["E0", "SP1"]. None = all.
        fetch_kalshi: If True, fetch resolved Kalshi markets and join for EV analysis.
        force_refresh: Re-download cached CSV/XLSX files.
        ev_threshold: Minimum EV% to count as a positive edge (default 2%).

    Returns:
        One BacktestReport per sector.
    """
    if seasons is None:
        seasons = ["2425", "2526"]

    reports: list[BacktestReport] = []

    for sector in sectors:
        try:
            if sector == "soccer":
                report = _run_soccer(seasons, leagues, fetch_kalshi, force_refresh, ev_threshold)
            elif sector == "tennis":
                years = _seasons_to_tennis_years(seasons)
                report = _run_tennis(years, fetch_kalshi, force_refresh, ev_threshold)
            else:
                logger.warning("backtest_unsupported_sector", sector=sector)
                continue
            reports.append(report)
        except Exception as e:
            logger.error("backtest_sector_failed", sector=sector, error=str(e))

    return reports


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
