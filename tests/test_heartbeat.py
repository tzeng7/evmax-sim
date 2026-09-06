"""Tests for the pipeline heartbeat (evmax/agents/cleanup/heartbeat.py).

Pure cadence/seed-state logic is tested directly; run_heartbeat's alert path is
tested with the DB/notifier readers mocked out.
"""
from __future__ import annotations

from datetime import datetime

from evmax.agents.cleanup import heartbeat as hb
from evmax.notifications import Notifier

NOW = datetime(2026, 8, 28, 12, 0, 0)


class TestAgeHours:
    def test_parses_space_and_t_separators(self):
        assert hb._age_hours("2026-08-28 06:00:00", NOW) == 6.0
        assert hb._age_hours("2026-08-28T06:00:00", NOW) == 6.0

    def test_date_only(self):
        assert hb._age_hours("2026-08-27", NOW) == 36.0

    def test_bad_and_empty(self):
        assert hb._age_hours(None, NOW) is None
        assert hb._age_hours("not-a-date", NOW) is None


class TestCadenceIssues:
    def test_all_fresh_no_issues(self):
        assert hb._cadence_issues(
            "2026-08-28 06:00:00", "2026-08-28", NOW,
            max_resolve_age_h=36.0, max_scan_age_days=1,
        ) == []

    def test_stale_resolve_is_critical(self):
        issues = hb._cadence_issues(
            "2026-08-25 06:00:00", "2026-08-28", NOW,
            max_resolve_age_h=36.0, max_scan_age_days=1,
        )
        assert len(issues) == 1
        assert issues[0]["check"] == "resolve"
        assert issues[0]["severity"] == "critical"

    def test_missing_resolve_is_critical(self):
        issues = hb._cadence_issues(None, "2026-08-28", NOW, 36.0, 1)
        assert issues[0]["check"] == "resolve"
        assert issues[0]["severity"] == "critical"

    def test_stale_scan_is_warning(self):
        issues = hb._cadence_issues(
            "2026-08-28 06:00:00", "2026-08-25", NOW, 36.0, 1
        )
        assert len(issues) == 1
        assert issues[0]["check"] == "scan"
        assert issues[0]["severity"] == "warning"

    def test_both_stale_reports_both(self):
        issues = hb._cadence_issues(
            "2026-08-20 06:00:00", "2026-08-20", NOW, 36.0, 1
        )
        assert {i["check"] for i in issues} == {"resolve", "scan"}


class TestSeedStateIssue:
    def test_fresh_in_season_ok(self):
        assert hb._seed_state_issue(
            "ufc_rating", "2026-08-24", NOW, stale_days=10, in_season=True
        ) is None

    def test_stale_in_season_warns(self):
        # The UFC freeze shape: stamp stuck ~27 days.
        issue = hb._seed_state_issue(
            "ufc_rating", "2026-08-01", NOW, stale_days=10, in_season=True
        )
        assert issue is not None
        assert issue["severity"] == "warning"
        assert "ufc_rating" in issue["check"]

    def test_stale_off_season_ignored(self):
        assert hb._seed_state_issue(
            "tennis_surface", "2026-08-01", NOW, stale_days=10, in_season=False
        ) is None

    def test_missing_stamp_in_season_warns(self):
        issue = hb._seed_state_issue(
            "ufc_rating", None, NOW, stale_days=10, in_season=True
        )
        assert issue is not None
        assert "no last_updated" in issue["detail"]


class TestResolveStamp:
    def test_top_level_stamp(self):
        assert hb._resolve_stamp({"last_updated": "2026-08-24"}, ["last_updated"]) == "2026-08-24"

    def test_nested_stamp(self):
        data = {"nfl": {"fetched_at": "2026-09-01", "teams": {}}}
        assert hb._resolve_stamp(data, ["nfl", "fetched_at"]) == "2026-09-01"

    def test_missing_path_returns_none(self):
        assert hb._resolve_stamp({"nfl": {}}, ["nfl", "fetched_at"]) is None
        assert hb._resolve_stamp({}, ["nfl", "fetched_at"]) is None
        # non-str leaf → None (never a bogus stamp)
        assert hb._resolve_stamp({"nfl": {"fetched_at": 12}}, ["nfl", "fetched_at"]) is None


class TestCheckSeedStatesNestedStamp:
    def test_stale_nested_nfl_stamp_warns_in_season(self, tmp_path, monkeypatch):
        """A frozen NFL efficiency reseed (nested nfl.fetched_at) is flagged —
        the gap the old top-level-only check missed for NFL/WNBA/NCAAF."""
        monkeypatch.setattr(hb, "_MODELS_DIR", tmp_path)
        # only the NFL efficiency check should fire; write just that file
        (tmp_path / "nfl_efficiency_state.json").write_text(
            '{"nfl": {"fetched_at": "2026-08-01", "teams": {}}}'
        )
        # NOW is 2026-08-28, in NFL's Sep-Feb window? NFL season is 09-04→02-15,
        # so Aug 28 is OUT of season → no alert. Use an in-season date.
        in_season_now = datetime(2026, 9, 20, 12, 0, 0)
        issues = hb.check_seed_states(now=in_season_now, today=in_season_now.date())
        labels = {i["check"] for i in issues}
        assert any("nfl_efficiency" in c for c in labels), labels

    def test_fresh_nested_nfl_stamp_ok(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_MODELS_DIR", tmp_path)
        in_season_now = datetime(2026, 9, 20, 12, 0, 0)
        (tmp_path / "nfl_efficiency_state.json").write_text(
            '{"nfl": {"fetched_at": "2026-09-19", "teams": {}}}'
        )
        issues = hb.check_seed_states(now=in_season_now, today=in_season_now.date())
        assert not any("nfl_efficiency" in i["check"] for i in issues)


class _FakeNotifier:
    def __init__(self, result=True):
        self.calls = []
        self._result = result

    def notify_alert(self, title, message, *, severity="warning"):
        self.calls.append((title, message, severity))
        return self._result


class TestRunHeartbeat:
    def test_healthy_no_alert(self, monkeypatch):
        monkeypatch.setattr(hb, "check_cadence", lambda **k: [])
        monkeypatch.setattr(hb, "check_seed_states", lambda **k: [])
        fake = _FakeNotifier()
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))

        result = hb.run_heartbeat(notify=True)
        assert result["ok"] is True
        assert result["notified"] is False
        assert fake.calls == []

    def test_critical_issue_alerts_critical(self, monkeypatch):
        monkeypatch.setattr(
            hb, "check_cadence",
            lambda **k: [{"check": "resolve", "severity": "critical", "detail": "stopped"}],
        )
        monkeypatch.setattr(hb, "check_seed_states", lambda **k: [])
        fake = _FakeNotifier(result=True)
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))

        result = hb.run_heartbeat(notify=True)
        assert result["ok"] is False
        assert result["notified"] is True
        assert fake.calls[0][2] == "critical"

    def test_only_warnings_alerts_warning(self, monkeypatch):
        monkeypatch.setattr(hb, "check_cadence", lambda **k: [])
        monkeypatch.setattr(
            hb, "check_seed_states",
            lambda **k: [{"check": "state:ufc_rating", "severity": "warning", "detail": "stale"}],
        )
        fake = _FakeNotifier()
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))

        result = hb.run_heartbeat(notify=True)
        assert result["notified"] is True
        assert fake.calls[0][2] == "warning"

    def test_issues_not_pushed_without_notify(self, monkeypatch):
        monkeypatch.setattr(
            hb, "check_cadence",
            lambda **k: [{"check": "resolve", "severity": "critical", "detail": "stopped"}],
        )
        monkeypatch.setattr(hb, "check_seed_states", lambda **k: [])
        fake = _FakeNotifier()
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))

        result = hb.run_heartbeat(notify=False)
        assert result["ok"] is False
        assert result["notified"] is False
        assert fake.calls == []


class TestPinnacleCheck:
    def test_ok_probe_no_issue(self):
        assert hb._pinnacle_issue({"ok": True, "status": 200, "reason": "ok"}) is None

    def test_down_probe_is_critical(self):
        issue = hb._pinnacle_issue({"ok": False, "status": 403, "reason": "geo_block"})
        assert issue["severity"] == "critical"
        assert issue["check"] == "pinnacle"
        assert "geo_block" in issue["detail"]

    def test_opt_in_probe_included_when_enabled(self, monkeypatch):
        monkeypatch.setattr(hb, "check_cadence", lambda **k: [])
        monkeypatch.setattr(hb, "check_seed_states", lambda **k: [])
        monkeypatch.setattr(
            hb, "check_pinnacle",
            lambda: [{"check": "pinnacle", "severity": "critical", "detail": "down"}],
        )
        fake = _FakeNotifier()
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))

        result = hb.run_heartbeat(check_pinnacle_reachability=True, notify=True)
        assert result["ok"] is False
        assert any(i["check"] == "pinnacle" for i in result["issues"])
        assert fake.calls[0][2] == "critical"

    def test_probe_skipped_by_default(self, monkeypatch):
        monkeypatch.setattr(hb, "check_cadence", lambda **k: [])
        monkeypatch.setattr(hb, "check_seed_states", lambda **k: [])
        called = {"n": 0}

        def _tracked():
            called["n"] += 1
            return []
        monkeypatch.setattr(hb, "check_pinnacle", _tracked)

        hb.run_heartbeat(check_pinnacle_reachability=False)
        assert called["n"] == 0  # no network probe unless explicitly asked
