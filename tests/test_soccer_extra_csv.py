"""Tests for the football-data.co.uk "new leagues" (extra) CSV parser (WS1).

The extra-league files (USA.csv = MLS) carry every season in one file with a
different schema from the top-5 per-season files: Season/Home/Away/HG/AG/Res
columns, PH/PD/PA Pinnacle odds with PSCH/PSCD/PSCA closing odds in recent
rows, and no shot columns.
"""

from __future__ import annotations

from pathlib import Path

from evmax.backtest.sources.soccer_csv import (
    _season_code_to_year,
    parse_soccer_extra_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "soccer_extra_usa.csv"


def test_season_code_maps_to_ending_year():
    # Calendar-year leagues: "the season ending in spring 2026" = MLS 2026.
    assert _season_code_to_year("2526") == 2026
    assert _season_code_to_year("2425") == 2025


def test_parses_selected_season_only():
    rows = parse_soccer_extra_csv(FIXTURE, "USA", ["2425"])
    # 2025 rows only: Inter Miami + LA Galaxy games; 2024/2026 filtered out.
    assert len(rows) == 2
    assert all(r.date.year == 2025 for r in rows)
    assert rows[0].league == "MLS"
    assert rows[0].sector == "soccer"


def test_multi_season_load():
    rows = parse_soccer_extra_csv(FIXTURE, "USA", ["2425", "2526"])
    years = {r.date.year for r in rows}
    assert years == {2025, 2026}


def test_prefers_closing_odds_when_present():
    rows = parse_soccer_extra_csv(FIXTURE, "USA", ["2425"])
    miami = next(r for r in rows if r.team_home == "Inter Miami")
    # PSCH=1.62 preferred over PH=1.65.
    assert miami.pinnacle_home_dec == 1.62
    assert miami.pinnacle_draw_dec == 4.25
    assert miami.pinnacle_away_dec == 5.10


def test_falls_back_to_ph_when_closing_absent():
    rows = parse_soccer_extra_csv(FIXTURE, "USA", ["2526"])
    austin = next(r for r in rows if r.team_home == "Austin FC")
    # PSCH empty → PH=2.10 used.
    assert austin.pinnacle_home_dec == 2.10


def test_no_shot_columns_yield_none():
    rows = parse_soccer_extra_csv(FIXTURE, "USA", ["2425"])
    assert all(r.home_shots is None and r.away_sot is None for r in rows)


def test_missing_goals_kept_with_result():
    rows = parse_soccer_extra_csv(FIXTURE, "USA", ["2526"])
    portland = next(r for r in rows if r.team_home == "Portland Timbers")
    assert portland.home_goals is None
    assert portland.home_won is True  # Res=H still parsed


def test_row_without_any_pinnacle_odds_skipped():
    rows = parse_soccer_extra_csv(FIXTURE, "USA", ["2526"])
    # Nashville row has neither PSC* nor PH/PD/PA → skipped.
    assert not any(r.team_home == "Nashville SC" for r in rows)


def test_devig_probs_populated():
    rows = parse_soccer_extra_csv(FIXTURE, "USA", ["2425"])
    for r in rows:
        assert 0 < r.true_prob_home < 1
        assert 0 < r.true_prob_draw < 1
        assert abs(r.true_prob_home + r.true_prob_away + r.true_prob_draw - 1.0) < 1e-6
