"""Parse football-data.co.uk CSV files into BacktestRows."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import structlog

from evmax.backtest.loader import (
    SOCCER_EXTRA_LEAGUES,
    SOCCER_LEAGUES,
    fetch_soccer_csv,
    fetch_soccer_extra_csv,
)
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

            # Goal counts (FTHG/FTAG) are optional — absent in a handful of
            # older rows. Keep them as int or None; Poisson walk-forward skips
            # rows without goals.
            try:
                home_goals = int(raw.get("FTHG", "").strip())
                away_goals = int(raw.get("FTAG", "").strip())
            except (ValueError, AttributeError):
                home_goals = None
                away_goals = None

            def _opt_int(key: str) -> Optional[int]:
                try:
                    return int(raw.get(key, "").strip())
                except (ValueError, AttributeError):
                    return None

            home_shots = _opt_int("HS")
            away_shots = _opt_int("AS")
            home_sot = _opt_int("HST")
            away_sot = _opt_int("AST")

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
                home_goals=home_goals,
                away_goals=away_goals,
                home_shots=home_shots,
                away_shots=away_shots,
                home_sot=home_sot,
                away_sot=away_sot,
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


def _season_code_to_year(code: str) -> int:
    """Map a European season code to the calendar year it ends in.

    For calendar-year leagues (MLS runs Feb–Nov), "the season that ends in
    spring 2026" is the 2026 season — so "2526" → 2026, "2425" → 2025.
    """
    return 2000 + int(code[2:])


def parse_soccer_extra_csv(
    path: Path, league_code: str, seasons: list[str]
) -> list[BacktestRow]:
    """Parse a football-data.co.uk "new leagues" all-seasons CSV.

    Schema differs from the top-5 per-season files: Season/Home/Away/HG/AG/Res
    columns, closing odds only, and no shot columns (xG legs stay None and
    the walk-forward renormalizes over the remaining models).

    Odds anchor fallback chain, per row: Pinnacle closing (PSCH/PSCD/PSCA)
    → pre-closing Pinnacle (PH/PD/PA, present in some extra files) → market-
    average closing (AvgCH/AvgCD/AvgCA). The fallback matters in practice:
    USA.csv carries NO Pinnacle columns for the in-progress season (2026 rows
    are AvgC*/B365C*-only), so without it the holdout year is unscorable.
    Anchor-source counts are logged so a run can see how much of the corpus
    rode the consensus-average proxy.
    """
    rows: list[BacktestRow] = []
    league_name = SOCCER_EXTRA_LEAGUES.get(league_code, league_code)
    want_years = {str(_season_code_to_year(s)) for s in seasons}
    skipped = 0
    anchor_counts = {"pinnacle_close": 0, "pinnacle": 0, "avg_close": 0}

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            season = (raw.get("Season") or "").strip()
            if season not in want_years:
                continue

            date_str = (raw.get("Date") or "").strip()
            home = (raw.get("Home") or "").strip()
            away = (raw.get("Away") or "").strip()
            res = (raw.get("Res") or "").strip()

            if not all([date_str, home, away, res]):
                skipped += 1
                continue

            event_date = _parse_date(date_str)
            if not event_date:
                skipped += 1
                continue

            psh = psd = psa = None
            for keys, label in (
                (("PSCH", "PSCD", "PSCA"), "pinnacle_close"),
                (("PH", "PD", "PA"), "pinnacle"),
                (("AvgCH", "AvgCD", "AvgCA"), "avg_close"),
            ):
                h = _safe_float(raw.get(keys[0], ""))
                d = _safe_float(raw.get(keys[1], ""))
                a = _safe_float(raw.get(keys[2], ""))
                if all([h, d, a]):
                    psh, psd, psa = h, d, a
                    anchor_counts[label] += 1
                    break

            if not all([psh, psd, psa]):
                skipped += 1
                continue

            try:
                prob_h, prob_a, prob_d, margin = devig_three_way(psh, psa, psd)
            except Exception:
                skipped += 1
                continue

            try:
                home_goals = int((raw.get("HG") or "").strip())
                away_goals = int((raw.get("AG") or "").strip())
            except (ValueError, AttributeError):
                home_goals = None
                away_goals = None

            rows.append(BacktestRow(
                sector="soccer",
                league=league_name,
                date=event_date,
                team_home=home,
                team_away=away,
                pinnacle_home_dec=psh,
                pinnacle_away_dec=psa,
                pinnacle_draw_dec=psd,
                result=res,
                true_prob_home=prob_h,
                true_prob_away=prob_a,
                true_prob_draw=prob_d,
                home_won=(res == "H"),
                draw=(res == "D"),
                home_goals=home_goals,
                away_goals=away_goals,
                home_shots=None,
                away_shots=None,
                home_sot=None,
                away_sot=None,
            ))

    logger.info(
        "soccer_extra_csv_parsed",
        league=league_name,
        seasons=sorted(want_years),
        rows=len(rows),
        skipped=skipped,
        anchors=anchor_counts,
    )
    return rows


def load_soccer(
    seasons: list[str],
    leagues: Optional[list[str]] = None,
    force: bool = False,
) -> list[BacktestRow]:
    """Download and parse soccer data for the given seasons and leagues.

    League codes split into the top-5 per-season files (E0, SP1, ...) and the
    "new leagues" all-seasons files (USA, ...); either kind may be mixed in
    ``leagues``. Default remains the 5 European leagues.
    """
    league_codes = leagues or list(SOCCER_LEAGUES.keys())
    standard = [c for c in league_codes if c not in SOCCER_EXTRA_LEAGUES]
    extra = [c for c in league_codes if c in SOCCER_EXTRA_LEAGUES]
    all_rows: list[BacktestRow] = []

    for season in seasons:
        for code in standard:
            try:
                path = fetch_soccer_csv(season, code, force=force)
                rows = parse_soccer_csv(path, code, season)
                all_rows.extend(rows)
            except Exception as e:
                logger.warning("soccer_load_failed", season=season, league=code, error=str(e))

    for code in extra:
        try:
            path = fetch_soccer_extra_csv(code, force=force)
            rows = parse_soccer_extra_csv(path, code, seasons)
            all_rows.extend(rows)
        except Exception as e:
            logger.warning("soccer_load_failed", seasons=seasons, league=code, error=str(e))

    return all_rows
