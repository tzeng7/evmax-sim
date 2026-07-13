"""Data-adapter capability tests: ESPN field/results/score-proxy and PGA SG
snapshot parsing, all against captured real fixtures (offline)."""

import pytest

from evmax.golf.data.espn import (
    parse_scoreboard,
    field_from_scoreboard,
    result_to_sg_proxy,
)
from evmax.golf.data.pga import parse_sg_snapshot, parse_stat_rows
from evmax.golf.names import canonical


class TestEspnParsing:
    def test_parses_events_and_field(self, espn_scoreboard_payload):
        results = parse_scoreboard(espn_scoreboard_payload)
        assert len(results) >= 1
        r = results[0]
        assert r.event
        assert len(r.entries) >= 20
        # every entry has a canonical name
        assert all(e.player == canonical(e.display_name) for e in r.entries)

    def test_field_names_nonempty(self, espn_scoreboard_payload):
        names = field_from_scoreboard(espn_scoreboard_payload)
        assert len(names) >= 20
        assert all(isinstance(n, str) and n for n in names)

    def test_round_scores_and_totals_present(self, espn_scoreboard_payload):
        r = parse_scoreboard(espn_scoreboard_payload)[0]
        scored = [e for e in r.entries if e.to_par is not None]
        assert scored, "expected at least one scored competitor"
        # a completed event carries multi-round linescores
        assert any(e.rounds_played >= 2 for e in scored)


class TestScoreProxy:
    def test_completed_event_yields_sg_records(self, espn_scoreboard_payload):
        completed = [r for r in parse_scoreboard(espn_scoreboard_payload) if r.completed]
        assert completed, "fixture should contain a completed event"
        recs = result_to_sg_proxy(completed[0])
        assert len(recs) >= 20
        # SG-total proxy is centred on the field: it must have both signs and
        # average ~0 across the field.
        sgs = [r.sg_total for r in recs]
        assert min(sgs) < 0 < max(sgs)
        assert abs(sum(sgs) / len(sgs)) < 0.5
        assert all(r.source == "espn_proxy" for r in recs)

    def test_leader_has_positive_sg(self, espn_scoreboard_payload):
        completed = [r for r in parse_scoreboard(espn_scoreboard_payload) if r.completed][0]
        recs = result_to_sg_proxy(completed)
        # the record with the best (lowest) finish should carry positive SG
        winners = [r for r in recs if r.finish_pos == 1]
        assert winners and winners[0].sg_total > 0

    def test_tiny_field_returns_nothing(self):
        from evmax.golf.data.espn import GolfEntry, GolfResult

        res = GolfResult("1", "Tiny", True, [
            GolfEntry("a a", "A A", -5, 4, [67, 68]),
            GolfEntry("b b", "B B", -3, 4, [69, 68]),
        ])
        assert result_to_sg_proxy(res, min_field=20) == []


class TestPgaSgParsing:
    def test_snapshot_has_all_categories(self, pga_sg_payload):
        snaps = parse_sg_snapshot(pga_sg_payload)
        assert len(snaps) > 100
        withcat = [s for s in snaps.values() if s.has_categories]
        assert len(withcat) > 100  # full fixture → everyone has 4 categories

    def test_scheffler_is_top_total(self, pga_sg_payload):
        snaps = parse_sg_snapshot(pga_sg_payload)
        top = max(snaps.values(), key=lambda s: s.sg_total or -9)
        assert "scheffler" in top.player
        assert top.sg_total > 1.5  # elite SG/round

    def test_measured_rounds_populated(self, pga_sg_payload):
        snaps = parse_sg_snapshot(pga_sg_payload)
        assert any(s.measured_rounds and s.measured_rounds > 20 for s in snaps.values())

    def test_stat_row_value_parsing(self, pga_sg_payload):
        rows = parse_stat_rows(pga_sg_payload["total"])
        assert rows
        # values are floats (handles ".422", "-6.328")
        for avg, rounds in list(rows.values())[:10]:
            assert avg is None or isinstance(avg, float)
