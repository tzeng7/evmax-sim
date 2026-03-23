"""Match historical CSV rows to resolved Kalshi markets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import structlog

from evmax.backtest.kalshi_history import ResolvedMarket
from evmax.backtest.models import BacktestRow
from evmax.matching.normalizer import NameNormalizer

logger = structlog.get_logger(__name__)


@dataclass
class KalshiJoin:
    ticker: str
    last_price: float   # price on YES side
    yes_team: Optional[str]
    result: str         # "yes"/"no"
    theoretical_ev: float  # (true_prob_yes / last_price) - 1


def _norm(name: str, sector: str) -> str:
    return NameNormalizer(sector).normalize(name)


def _event_key(team_a: str, team_b: str, d: date, sector: str) -> str:
    na = _norm(team_a, sector).replace(" ", "_")
    nb = _norm(team_b, sector).replace(" ", "_")
    return f"{sector}::{d}::{na}_vs_{nb}"


def match_soccer(
    rows: list[BacktestRow],
    resolved: list[ResolvedMarket],
    date_tolerance: int = 1,
) -> dict[int, KalshiJoin]:
    """
    Match soccer CSV rows to resolved Kalshi markets.
    Returns {row_index: KalshiJoin} for matched rows.
    """
    # Build lookup: event_key → list[ResolvedMarket]
    kalshi_lookup: dict[str, list[ResolvedMarket]] = {}
    for rm in resolved:
        if not rm.team_home or not rm.event_date:
            continue
        for delta in range(-date_tolerance, date_tolerance + 1):
            d = rm.event_date + timedelta(days=delta)
            key = _event_key(rm.team_home, rm.team_away or "", d, "soccer")
            kalshi_lookup.setdefault(key, []).append(rm)

    matches: dict[int, KalshiJoin] = {}
    for i, row in enumerate(rows):
        key = _event_key(row.team_home, row.team_away, row.date, "soccer")
        candidates = kalshi_lookup.get(key, [])
        if not candidates:
            continue

        # Pick the best candidate: prefer YES side that matches the actual winner
        for rm in candidates:
            yes_norm = _norm(rm.yes_team or "", "soccer")
            home_norm = _norm(row.team_home, "soccer")
            away_norm = _norm(row.team_away, "soccer")

            # Determine which true_prob corresponds to the YES side
            if yes_norm == home_norm or yes_norm in home_norm or home_norm in yes_norm:
                true_prob_yes = row.true_prob_home
            elif yes_norm == away_norm or yes_norm in away_norm or away_norm in yes_norm:
                true_prob_yes = row.true_prob_away
            elif yes_norm in ("tie", "draw", "x"):
                true_prob_yes = row.true_prob_draw or 0.0
            else:
                continue  # can't determine YES alignment

            if rm.last_price <= 0 or rm.last_price >= 1:
                continue

            ev = (true_prob_yes / rm.last_price) - 1.0

            matches[i] = KalshiJoin(
                ticker=rm.ticker,
                last_price=rm.last_price,
                yes_team=rm.yes_team,
                result=rm.result,
                theoretical_ev=ev,
            )
            break

    logger.info("soccer_kalshi_matched", total=len(rows), matched=len(matches))
    return matches


def match_tennis(
    rows: list[BacktestRow],
    resolved: list[ResolvedMarket],
    date_tolerance: int = 1,
) -> dict[int, KalshiJoin]:
    """
    Match tennis CSV rows to resolved Kalshi markets by player pair + date.
    Returns {row_index: KalshiJoin} for matched rows.
    """
    # Build Kalshi lookup: frozenset({p1_norm, p2_norm}) + date → ResolvedMarket
    kalshi_lookup: dict[tuple, list[ResolvedMarket]] = {}
    for rm in resolved:
        if not rm.team_home or not rm.event_date:
            continue
        p1 = _norm(rm.team_home, "tennis")
        p2 = _norm(rm.team_away or "", "tennis")
        pair = frozenset({p1, p2})
        for delta in range(-date_tolerance, date_tolerance + 1):
            d = rm.event_date + timedelta(days=delta)
            key = (pair, d)
            kalshi_lookup.setdefault(key, []).append(rm)

    matches: dict[int, KalshiJoin] = {}
    for i, row in enumerate(rows):
        winner_norm = _norm(row.team_home, "tennis")
        loser_norm = _norm(row.team_away, "tennis")
        pair = frozenset({winner_norm, loser_norm})

        candidates = kalshi_lookup.get((pair, row.date), [])
        if not candidates:
            continue

        for rm in candidates:
            yes_norm = _norm(rm.yes_team or "", "tennis")

            # Determine YES side true_prob
            if yes_norm == winner_norm or winner_norm.startswith(yes_norm) or yes_norm.startswith(winner_norm):
                true_prob_yes = row.true_prob_home   # home = winner in tennis
            elif yes_norm == loser_norm or loser_norm.startswith(yes_norm) or yes_norm.startswith(loser_norm):
                true_prob_yes = row.true_prob_away
            else:
                continue

            if rm.last_price <= 0 or rm.last_price >= 1:
                continue

            ev = (true_prob_yes / rm.last_price) - 1.0

            matches[i] = KalshiJoin(
                ticker=rm.ticker,
                last_price=rm.last_price,
                yes_team=rm.yes_team,
                result=rm.result,
                theoretical_ev=ev,
            )
            break

    logger.info("tennis_kalshi_matched", total=len(rows), matched=len(matches))
    return matches


def apply_kalshi_joins(
    rows: list[BacktestRow],
    matches: dict[int, KalshiJoin],
) -> list[BacktestRow]:
    """Apply Kalshi join data back onto the rows (mutates in place, returns same list)."""
    for i, join in matches.items():
        row = rows[i]
        row.kalshi_ticker = join.ticker
        row.kalshi_last_price = join.last_price
        row.kalshi_yes_team = join.yes_team
        row.kalshi_result = join.result
        row.theoretical_ev = join.theoretical_ev
    return rows
