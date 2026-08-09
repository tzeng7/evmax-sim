"""Injury-staleness decay tests.

An injury adjustment corrects the model rating for information it does NOT yet
contain: a normally-playing contributor who is out for THIS game. Once a player
has been out for weeks — or permanently, like a mid-season overseas departure —
the season-long elo/efficiency/form ratings (and the sharp line) already reflect
the absence, so re-applying the full starter-OUT penalty double-counts it.

Regression anchor: Jovana Nogic (Phoenix Mercury) left for Russia in June 2026,
tagged ESPN status "Out", `date` 2026-06-28, `injury_type` "Not Injury Related".
The old code scored her at the full −4.5pp starter-OUT impact, which
manufactured a phantom ~+7.5pp edge on Mercury's opponents.
"""

from datetime import date

from evmax.agents.intelligence.injury_agent import (
    INJURY_FRESH_DAYS,
    INJURY_STALE_DAYS,
    InjuryReportAgent,
    _injury_staleness_multiplier,
)


def _entry(reported_at: str, status: str = "Out", position: str = "G", name: str = "Test Player"):
    """Minimal ESPN-shaped injury entry."""
    return {
        "athlete": {"displayName": name, "position": {"abbreviation": position}},
        "status": status,
        "details": {"type": "Knee"},
        "shortComment": "",
        "date": reported_at,
    }


class TestStalenessMultiplier:
    def test_fresh_absence_full_impact(self):
        ref = date(2026, 8, 9)
        assert _injury_staleness_multiplier("2026-08-08T00:00Z", ref) == 1.0

    def test_boundary_fresh_days_full_impact(self):
        ref = date(2026, 8, 9)
        reported = "2026-08-06T00:00Z"  # exactly INJURY_FRESH_DAYS=3 days out
        assert _injury_staleness_multiplier(reported, ref) == 1.0

    def test_stale_absence_zero_impact(self):
        ref = date(2026, 8, 9)
        # 42 days out — well past INJURY_STALE_DAYS
        assert _injury_staleness_multiplier("2026-06-28T00:25Z", ref) == 0.0

    def test_boundary_stale_days_zero_impact(self):
        ref = date(2026, 8, 9)
        reported = "2026-07-26T00:00Z"  # exactly INJURY_STALE_DAYS=14 days out
        assert _injury_staleness_multiplier(reported, ref) == 0.0

    def test_midpoint_partial_decay(self):
        ref = date(2026, 8, 9)
        # 8.5 days would be exact 0.5; use the integer midpoint of the ramp
        mid_days = (INJURY_FRESH_DAYS + INJURY_STALE_DAYS) / 2  # 8.5
        # pick 8 and 9 days and assert the ramp is decreasing and within (0,1)
        eight = _injury_staleness_multiplier("2026-08-01T00:00Z", ref)   # 8 days
        nine = _injury_staleness_multiplier("2026-07-31T00:00Z", ref)    # 9 days
        assert 0.0 < nine < eight < 1.0
        assert mid_days == 8.5  # ramp midpoint sanity

    def test_missing_date_fails_open(self):
        ref = date(2026, 8, 9)
        assert _injury_staleness_multiplier(None, ref) == 1.0

    def test_unparseable_date_fails_open(self):
        ref = date(2026, 8, 9)
        assert _injury_staleness_multiplier("not-a-date", ref) == 1.0


class TestParsePlayerStaleness:
    def test_nogic_permanent_absence_zeroed(self):
        """The real-world regression case: a June departure scored on an August
        game must decay to zero impact (and thus be dropped by the caller)."""
        agent = InjuryReportAgent()
        entry = _entry(
            "2026-06-28T00:25Z", status="Out", position="G", name="Jovana Nogic"
        )
        player = agent._parse_player(
            entry, sector="wnba", reference_date=date(2026, 8, 9)
        )
        # Impact is fully decayed. The fetch loop drops players with impact <= 0.
        assert player is not None
        assert player.impact == 0.0

    def test_fresh_scratch_keeps_full_impact(self):
        """A same-week OUT is genuine new information — full starter impact."""
        agent = InjuryReportAgent()
        entry = _entry(
            "2026-08-08T18:00Z", status="Out", position="G", name="Fresh Scratch"
        )
        player = agent._parse_player(
            entry, sector="wnba", reference_date=date(2026, 8, 9)
        )
        assert player is not None
        # OUT (0.045) × starter (1.0) × non-NFL position weight (1.0)
        assert player.impact == 0.045

    def test_stale_and_fresh_ordering(self):
        """The same OUT status at the same position must score higher when
        fresh than when stale — the decay is monotonic in age."""
        agent = InjuryReportAgent()
        ref = date(2026, 8, 9)
        fresh = agent._parse_player(
            _entry("2026-08-07T00:00Z"), sector="wnba", reference_date=ref
        )
        stale = agent._parse_player(
            _entry("2026-06-01T00:00Z"), sector="wnba", reference_date=ref
        )
        assert fresh.impact > stale.impact
        assert stale.impact == 0.0
