"""Parse football-data.co.uk CSV files into BacktestRows."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import structlog

from evmax.backtest.loader import SOCCER_LEAGUES, fetch_soccer_csv
from evmax.backtest.models import BacktestRow
from evmax.ev.devig import devig_three_way

logger = structlog.get_logger(__name__)


def _parse_date(raw: str) -> Optional[date]:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            pass
    return None


def _safe_float(val: str) -> Optional[float]:
    try:
        v = float(val.strip())
        return v if v > 0 else None
    except (ValueError, AttributeError):
        return None


def parse_soccer_csv(path: Path, league_code: str, season: str) -> list[BacktestRow]:
    """Parse one football-data.co.uk CSV into BacktestRows."""
    rows: list[BacktestRow] = []
    league_name = SOCCER_LEAGUES.get(league_code, league_code)
    skipped = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            date_str = raw.get("Date", "").strip()
            home = raw.get("HomeTeam", "").strip()
            away = raw.get("AwayTeam", "").strip()
            ftr = raw.get("FTR", "").strip()

            if not all([date_str, home, away, ftr]):
                skipped += 1
                continue

            event_date = _parse_date(date_str)
            if not event_date:
                skipped += 1
                continue

            psh = _safe_float(raw.get("PSH", ""))
            psd = _safe_float(raw.get("PSD", ""))
            psa = _safe_float(raw.get("PSA", ""))

            if not all([psh, psd, psa]):
                skipped += 1
                continue

            try:
                prob_h, prob_a, prob_d, margin = devig_three_way(psh, psa, psd)
            except Exception:
                skipped += 1
                continue

            row = BacktestRow(
                sector="soccer",
                league=league_name,
                date=event_date,
                team_home=home,
                team_away=away,
                pinnacle_home_dec=psh,
                pinnacle_away_dec=psa,
                pinnacle_draw_dec=psd,
                result=ftr,
                true_prob_home=prob_h,
                true_prob_away=prob_a,
                true_prob_draw=prob_d,
                home_won=(ftr == "H"),
                draw=(ftr == "D"),
            )
            rows.append(row)

    logger.info(
        "soccer_csv_parsed",
        league=league_name,
        season=season,
        rows=len(rows),
        skipped=skipped,
    )
    return rows


def load_soccer(
    seasons: list[str],
    leagues: Optional[list[str]] = None,
    force: bool = False,
) -> list[BacktestRow]:
    """Download and parse soccer data for the given seasons and leagues."""
    league_codes = leagues or list(SOCCER_LEAGUES.keys())
    all_rows: list[BacktestRow] = []

    for season in seasons:
        for code in league_codes:
            try:
                path = fetch_soccer_csv(season, code, force=force)
                rows = parse_soccer_csv(path, code, season)
                all_rows.extend(rows)
            except Exception as e:
                logger.warning("soccer_load_failed", season=season, league=code, error=str(e))

    return all_rows
