"""AgentCoordinator's seasonality gate.

The coordinator drops out-of-season sectors from `self._sectors` so we
don't burn Kalshi rate-limit tokens (one HTTP call per series prefix per
sector) and Pinnacle /matchups calls on dead leagues. Explicit callers
(CLI `--sectors nfl`, one-off scripts, regression tests) can opt out via
`respect_season_window=False`.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from evmax.agents.coordinator import AgentCoordinator


def _patched_today(year: int, month: int, day: int):
    """Pin `date.today()` as seen from evmax.categories to a fixed value.

    The coordinator's season gate calls `spec.is_in_season()` which
    bottoms out in `SeasonWindow.contains(date.today())` inside
    evmax.categories. Patching the `date` symbol there is enough.
    """

    class _D(date):
        @classmethod
        def today(cls):
            return date(year, month, day)

    return patch("evmax.categories.date", _D)


def test_default_scan_drops_out_of_season_sectors_in_may():
    """May 23 → nfl / nfl_props / ncaab / ncaaw must be dropped."""
    with _patched_today(2026, 5, 23):
        c = AgentCoordinator(
            sectors=["nba", "nfl", "ncaab", "ncaaw", "baseball", "tennis"],
            respect_season_window=True,
        )
    assert "nfl" not in c._sectors
    assert "ncaab" not in c._sectors
    assert "ncaaw" not in c._sectors
    assert "nba" in c._sectors           # NBA Finals run into late June
    assert "baseball" in c._sectors
    assert "tennis" in c._sectors        # year-round
    assert set(c._skipped_off_season) == {"nfl", "ncaab", "ncaaw"}


def test_default_scan_drops_baseball_in_january():
    """In Jan, MLB is out and NFL/NCAAB are in."""
    with _patched_today(2026, 1, 15):
        c = AgentCoordinator(
            sectors=["nfl", "ncaab", "baseball", "nba"],
            respect_season_window=True,
        )
    assert "baseball" not in c._sectors
    assert "nfl" in c._sectors
    assert "ncaab" in c._sectors
    assert "nba" in c._sectors


def test_explicit_override_bypasses_gate():
    """respect_season_window=False keeps NFL even in May."""
    with _patched_today(2026, 5, 23):
        c = AgentCoordinator(
            sectors=["nfl", "ncaab"],
            respect_season_window=False,
        )
    assert c._sectors == ["nfl", "ncaab"]
    assert c._skipped_off_season == []


def test_unknown_sector_passes_through_unmodified():
    """Latent / unregistered sectors aren't second-guessed by the gate —
    they hit the same downstream KeyError they always would."""
    with _patched_today(2026, 5, 23):
        c = AgentCoordinator(
            sectors=["nba", "ufc"],  # ufc is latent — not in categories.yaml
            respect_season_window=True,
        )
    assert "ufc" in c._sectors
    assert "nba" in c._sectors
