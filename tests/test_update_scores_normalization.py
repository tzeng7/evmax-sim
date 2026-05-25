"""Regression: `evmax update scores` must normalize ESPN team names.

Bug: when there was no matching `ev_predictions` row to draw a canonical
slug from, the score-update path fell back to `home_name.lower()` straight
from ESPN — producing duplicate elo state entries (e.g. both "fever" and
"indiana fever" for the same team). Fix is in
`evmax/cli/commands/update.py` (`_normalizer = NameNormalizer(sector)`).

This test exists so that an alias-file regression — removing the
"indiana fever" entry from wnba.yaml, for example — would fail loudly
instead of silently re-splitting the state on the next WNBA backfill.
"""

from __future__ import annotations

import pytest

from evmax.matching.normalizer import NameNormalizer


# ESPN scoreboard returns team names in their full "City Nickname" form
# for WNBA; the live scan / seeded state uses single-word slugs. Every
# entry below was emitted by ESPN during the 2026-05-08 → 2026-05-24
# backfill that surfaced the bug.
_WNBA_ESPN_NAME_TO_SLUG = {
    "Indiana Fever":           "fever",
    "Atlanta Dream":           "dream",
    "New York Liberty":        "liberty",
    "Minnesota Lynx":          "lynx",
    "Las Vegas Aces":          "aces",
    "Connecticut Sun":         "sun",
    "Los Angeles Sparks":      "sparks",
    "Phoenix Mercury":         "mercury",
    "Seattle Storm":           "storm",
    "Chicago Sky":             "sky",
    "Dallas Wings":            "wings",
    "Washington Mystics":      "mystics",
    "Golden State Valkyries":  "valkyries",
    "Portland Fire":           "fire",
    "Toronto Tempo":           "tempo",
}


@pytest.mark.parametrize("espn_name,slug", _WNBA_ESPN_NAME_TO_SLUG.items())
def test_wnba_espn_name_normalizes_to_slug(espn_name: str, slug: str) -> None:
    n = NameNormalizer("wnba")
    assert n.normalize(espn_name) == slug, (
        f"Alias gap in wnba.yaml: {espn_name!r} → {n.normalize(espn_name)!r} "
        f"(expected {slug!r}). evmax update scores will create a duplicate "
        f"elo entry on the next WNBA backfill."
    )
