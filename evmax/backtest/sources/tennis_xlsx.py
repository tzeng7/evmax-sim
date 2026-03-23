"""Parse tennis-data.co.uk XLSX files into BacktestRows."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import structlog

from evmax.backtest.loader import fetch_tennis_xlsx
from evmax.backtest.models import BacktestRow
from evmax.ev.devig import devig_two_way

logger = structlog.get_logger(__name__)


def _parse_date(val) -> Optional[date]:
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if not val:
        return None
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(val).strip(), fmt).date()
        except ValueError:
            pass
    return None


def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return v if v > 1.0 else None  # Pinnacle decimal odds always > 1
    except (ValueError, TypeError):
        return None


def parse_tennis_xlsx(path: Path, year: int, surfaces: Optional[list[str]] = None) -> list[BacktestRow]:
    """Parse one tennis-data.co.uk XLSX into BacktestRows."""
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError("openpyxl is required for tennis backtest: pip install openpyxl")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    rows_iter = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c else "" for c in next(rows_iter)]

    def col(row_vals, name: str):
        try:
            return row_vals[header.index(name)]
        except (ValueError, IndexError):
            return None

    rows: list[BacktestRow] = []
    skipped = 0

    for raw in rows_iter:
        winner = col(raw, "Winner")
        loser = col(raw, "Loser")
        psw = col(raw, "PSW")
        psl = col(raw, "PSL")
        date_val = col(raw, "Date")
        surface = col(raw, "Surface") or ""
        tournament = col(raw, "Tournament") or ""

        if not winner or not loser:
            skipped += 1
            continue

        if surfaces and surface.lower() not in [s.lower() for s in surfaces]:
            continue

        event_date = _parse_date(date_val)
        if not event_date:
            skipped += 1
            continue

        winner_dec = _safe_float(psw)
        loser_dec = _safe_float(psl)

        if not winner_dec or not loser_dec:
            skipped += 1
            continue

        try:
            prob_w, prob_l, margin = devig_two_way(winner_dec, loser_dec)
        except Exception:
            skipped += 1
            continue

        row = BacktestRow(
            sector="tennis",
            league=tournament,
            date=event_date,
            team_home=str(winner).strip(),   # home = Winner for tennis
            team_away=str(loser).strip(),
            pinnacle_home_dec=winner_dec,
            pinnacle_away_dec=loser_dec,
            pinnacle_draw_dec=None,
            result="W",
            true_prob_home=prob_w,
            true_prob_away=prob_l,
            true_prob_draw=None,
            home_won=True,   # Winner always wins
            draw=False,
        )
        rows.append(row)

    wb.close()
    logger.info("tennis_xlsx_parsed", year=year, rows=len(rows), skipped=skipped)
    return rows


def load_tennis(
    years: list[int],
    surfaces: Optional[list[str]] = None,
    force: bool = False,
) -> list[BacktestRow]:
    """Download and parse tennis data for the given years."""
    all_rows: list[BacktestRow] = []

    for year in years:
        try:
            path = fetch_tennis_xlsx(year, force=force)
            rows = parse_tennis_xlsx(path, year, surfaces=surfaces)
            all_rows.extend(rows)
        except Exception as e:
            logger.warning("tennis_load_failed", year=year, error=str(e))

    return all_rows
