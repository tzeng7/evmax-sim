"""Tests for the consolidated integrity sweep (evmax/agents/cleanup/integrity.py).

Pure ``_*_issues`` functions are tested on hand-built rows; ``run_integrity``'s
orchestration + single-alert path is tested with every check runner mocked so
no DB, archive, network or launchctl is involved.
"""
from __future__ import annotations

from datetime import date

from evmax.agents.cleanup import integrity as ig
from evmax.notifications import Notifier

TODAY = date(2026, 9, 5)


def _row(**kw) -> dict:
    base = {
        "sector": "soccer", "event_title": "Brentford vs Sunderland", "yes_team": "brentford",
        "market_id": "kalshi:KXEPLGAME-26SEP05BRESUN-BRE", "ev_pct": 0.03,
        "minutes_to_tipoff": 120, "logged_at": "2026-09-05 14:10:07",
    }
    base.update(kw)
    return base


class TestInplay:
    def test_absurd_ev_is_critical_regardless_of_timing(self):
        issues = ig._inplay_issues([_row(ev_pct=0.349, minutes_to_tipoff=180)])
        assert len(issues) == 1
        assert issues[0]["check"] == "inplay" and issues[0]["severity"] == "critical"
        assert "+35%" in issues[0]["detail"]

    def test_at_tip_with_moderate_ev_is_flagged(self):
        issues = ig._inplay_issues([_row(ev_pct=0.12, minutes_to_tipoff=0)])
        assert len(issues) == 1
        assert "minutes_to_tipoff=0" in issues[0]["detail"]

    def test_at_tip_with_small_ev_and_normal_rows_are_silent(self):
        rows = [
            _row(ev_pct=0.04, minutes_to_tipoff=0),      # T-0 scan, ordinary edge
            _row(ev_pct=0.08, minutes_to_tipoff=90),     # ordinary pre-match row
            _row(ev_pct=0.20, minutes_to_tipoff=None),   # unknown timing, below absurd
        ]
        assert ig._inplay_issues(rows) == []


class TestModelMissing:
    @staticmethod
    def _rows(sector, scan_date, n, missing=(), sources=("elo", "form", "sharp")):
        return [
            {"sector": sector, "scan_date": scan_date, "missing": list(missing),
             "sources": list(sources)}
            for _ in range(n)
        ]

    def test_regime_change_via_diagnostics_is_critical(self):
        rows = (
            self._rows("ncaaf", "2026-08-30", 20, missing=[])
            + self._rows("ncaaf", "2026-09-04", 6, missing=["ncaaf_efficiency_v2"])
            + self._rows("ncaaf", "2026-09-05", 6, missing=["ncaaf_efficiency_v2"])
        )
        issues = ig._model_missing_issues(rows, TODAY)
        assert len(issues) == 1
        assert "ncaaf_efficiency_v2" in issues[0]["detail"]
        assert issues[0]["severity"] == "critical"

    def test_structurally_missing_model_stays_silent(self):
        # h2h missing on 90% of rows in BOTH windows → no regime change.
        rows = (
            self._rows("tennis", "2026-08-30", 20, missing=["tennis_h2h"])
            + self._rows("tennis", "2026-09-05", 10, missing=["tennis_h2h"])
        )
        assert ig._model_missing_issues(rows, TODAY) == []

    def test_fire_rate_drop_via_model_sources(self):
        # Diagnostics never mention the model (the 2026-09-04 surface case),
        # but it vanished from model_sources.
        rows = (
            self._rows("tennis", "2026-08-30", 20,
                       sources=["tennis_surface", "tennis_form", "sharp"])
            + self._rows("tennis", "2026-09-05", 6, sources=["tennis_form", "sharp"])
        )
        issues = ig._model_missing_issues(rows, TODAY)
        assert len(issues) == 1
        assert "tennis_surface" in issues[0]["detail"] and "fired on 100%" in issues[0]["detail"]

    def test_non_model_tokens_never_flag(self):
        rows = (
            self._rows("soccer", "2026-08-30", 20, sources=["elo", "injury", "sharp"])
            + self._rows("soccer", "2026-09-05", 6, sources=["elo", "sharp"])
        )
        assert ig._model_missing_issues(rows, TODAY) == []

    def test_thin_recent_sample_or_no_baseline_is_silent(self):
        thin = (
            self._rows("wnba", "2026-08-30", 20)
            + self._rows("wnba", "2026-09-05", 2, missing=["wnba_efficiency"])
        )
        assert ig._model_missing_issues(thin, TODAY) == []
        no_base = self._rows("nhl", "2026-09-05", 12, missing=["nhl_xg"])
        assert ig._model_missing_issues(no_base, TODAY) == []


class TestMatchRate:
    def test_fetched_but_matched_zero_is_critical(self):
        today = {"ncaaf": {"fetched": 500, "matched": 0}}
        base = {"ncaaf": [{"fetched": 480, "matched": 70}, {"fetched": 510, "matched": 65}]}
        issues = ig._match_rate_issues(today, base)
        assert len(issues) == 1 and issues[0]["severity"] == "critical"
        assert "matched 0" in issues[0]["detail"]

    def test_ratio_collapse_is_warning(self):
        today = {"soccer": {"fetched": 200, "matched": 6}}      # 3%
        base = {"soccer": [{"fetched": 190, "matched": 130}] * 5}  # ~68%
        issues = ig._match_rate_issues(today, base)
        assert len(issues) == 1 and issues[0]["severity"] == "warning"

    def test_no_history_or_offseason_zero_is_silent(self):
        # Brand-new sector: nothing to compare against.
        assert ig._match_rate_issues({"ufc": {"fetched": 30, "matched": 0}}, {}) == []
        # Sector whose history is ALSO zero-matched (genuinely nothing to match).
        base = {"lol": [{"fetched": 10, "matched": 0}] * 3}
        assert ig._match_rate_issues({"lol": {"fetched": 12, "matched": 0}}, base) == []
        # Healthy day.
        base = {"nfl": [{"fetched": 100, "matched": 60}] * 3}
        assert ig._match_rate_issues({"nfl": {"fetched": 110, "matched": 58}}, base) == []


class TestSimpleChecks:
    def test_resolution_backlog_threshold(self):
        assert ig._resolution_issues({"tennis": 17, "soccer": 2}) == [
            ig._issue("resolution", "warning", ig._resolution_issues({"tennis": 17})[0]["detail"])
        ]
        assert ig._resolution_issues({"soccer": 4}) == []

    def test_close_capture_coverage_and_archive_age(self):
        cov = {"wnba": {"n": 40, "null": 20}, "nba": {"n": 5, "null": 5}, "nfl": {"n": 30, "null": 3}}
        issues = ig._close_capture_issues(cov, archive_age_h=30.0)
        sectors = [i["detail"].split(":")[0] for i in issues if ": " in i["detail"][:8]]
        assert "wnba" in sectors            # 50% null on n=40
        assert "nba" not in sectors         # n<10 ignored
        assert "nfl" not in sectors         # 10% null fine
        assert any("30h old" in i["detail"] for i in issues)
        assert ig._close_capture_issues({}, archive_age_h=2.0) == []
        assert ig._close_capture_issues({}, archive_age_h=None) == []

    def test_drawdown_floor(self):
        rows = [
            {"sector": "soccer", "n": 21, "wins": 4, "roi": -0.497},
            {"sector": "tennis", "n": 206, "wins": 82, "roi": 0.037},
            {"sector": "wnba", "n": 8, "wins": 1, "roi": -0.8},   # n < 20
        ]
        issues = ig._drawdown_issues(rows)
        assert [i["detail"].split(":")[0] for i in issues] == ["soccer"]

    def test_launchd_parsing(self):
        listing = (
            "PID\tStatus\tLabel\n"
            "-\t0\tcom.evmax.watch-closes\n"
            "35156\t-15\tcom.evmax.discord-bot\n"     # SIGTERM restart — not a failure
            "-\t1\tcom.evmax.heartbeat\n"
            "-\t78\tcom.apple.something\n"            # not ours
        )
        issues = ig._launchd_issues(listing)
        assert len(issues) == 1
        assert "com.evmax.heartbeat" in issues[0]["detail"] and "status 1" in issues[0]["detail"]


class TestBoardDerived:
    @staticmethod
    def _brow(verdict, mode="live", div=2.0, clv=None):
        return {
            "sector": "tennis", "market_type": "moneyline", "venue": "kalshi", "mode": mode,
            "verdict": verdict, "blend_divergence_pp": div, "n_clean_resolved": 40,
            "clv": clv or {"n": 40, "mean_clv_pp": -0.4, "frac_positive": 0.4},
        }

    def test_degrading_passthrough_and_gate(self):
        board = [
            self._brow("LIVE-DEGRADING"),
            self._brow("SHARP-PASSTHROUGH", div=0.15),
            self._brow("SHARP-PASSTHROUGH", mode="shadow", div=0.1),   # shadow: no bankroll at risk
            self._brow("PROMOTE-READY", mode="shadow"),
            self._brow("LIVE-HEALTHY"),
        ]
        daily = ig._board_issues(board, include_gates=False)
        assert [i["check"] for i in daily] == ["clv", "passthrough"]
        weekly = ig._board_issues(board, include_gates=True)
        assert [i["check"] for i in weekly] == ["clv", "passthrough", "gate"]
        assert weekly[-1]["severity"] == "info"

    def test_gate_watch_only_speaks_when_cleared(self):
        spec = ig.GATE_WATCHES[0]
        assert ig._gate_watch_issues([(spec, {"n": 20, "clears": False})]) == []
        out = ig._gate_watch_issues([(spec, {"n": 40, "clears": True, "mean_clv_pp": 0.8, "frac_positive": 0.6})])
        assert len(out) == 1 and out[0]["severity"] == "info" and "n=40" in out[0]["detail"]


class _FakeNotifier:
    def __init__(self, ok=True):
        self.ok, self.calls = ok, []

    def notify_alert(self, title, message, *, severity="warning"):
        self.calls.append((title, message, severity))
        return self.ok


def _mock_all_checks(monkeypatch, overrides: dict):
    """Make every runner return [] except those in ``overrides``."""
    names = {
        "check_inplay", "check_model_missing", "check_match_rate", "check_resolution",
        "check_close_capture", "check_board", "check_drawdown", "check_launchd",
        "check_calibration", "check_gate_watches",
    }
    for n in names:
        monkeypatch.setattr(ig, n, (lambda _n=n: (lambda **k: overrides.get(_n, [])))())
    monkeypatch.setattr(ig._hb, "check_cadence", lambda **k: overrides.get("check_cadence", []))
    monkeypatch.setattr(ig._hb, "check_seed_states", lambda **k: overrides.get("check_seed_states", []))
    monkeypatch.setattr(ig._hb, "check_pinnacle", lambda **k: overrides.get("check_pinnacle", []))


class TestRunIntegrity:
    def test_clean_run_no_alert(self, monkeypatch):
        _mock_all_checks(monkeypatch, {})
        fake = _FakeNotifier()
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))
        res = ig.run_integrity(notify=True)
        assert res["ok"] and res["issues"] == [] and not res["notified"]
        assert set(res["ran"]) == set(ig.DAILY_CHECKS) and res["failed"] == []
        assert fake.calls == []

    def test_weekly_adds_checks_and_pinnacle_is_opt_in(self, monkeypatch):
        _mock_all_checks(monkeypatch, {})
        assert "calibration" not in ig.run_integrity()["ran"]
        assert "pinnacle" not in ig.run_integrity()["ran"]
        ran = ig.run_integrity(weekly=True, check_pinnacle=True)["ran"]
        assert {"calibration", "gates", "pinnacle"} <= set(ran)

    def test_one_alert_at_worst_severity_sorted_worst_first(self, monkeypatch):
        _mock_all_checks(monkeypatch, {
            "check_drawdown": [ig._issue("drawdown", "warning", "soccer bleeding")],
            "check_inplay": [ig._issue("inplay", "critical", "brentford in-play")],
            "check_gate_watches": [ig._issue("gate", "info", "wnba lay cleared")],
        })
        fake = _FakeNotifier()
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))
        res = ig.run_integrity(weekly=True, notify=True)
        assert not res["ok"]
        assert [i["severity"] for i in res["issues"]] == ["critical", "warning", "info"]
        assert len(fake.calls) == 1
        title, message, severity = fake.calls[0]
        assert severity == "critical" and "2 issue(s)" in title and "1 gate clearance" in title
        assert "brentford" in message and "wnba lay" in message
        assert res["notified"]

    def test_info_only_is_still_ok_but_notifies(self, monkeypatch):
        _mock_all_checks(monkeypatch, {"check_gate_watches": [ig._issue("gate", "info", "cleared")]})
        fake = _FakeNotifier()
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))
        res = ig.run_integrity(weekly=True, notify=True)
        assert res["ok"] and res["notified"] and fake.calls[0][2] == "info"

    def test_crashing_check_is_reported_not_fatal(self, monkeypatch):
        _mock_all_checks(monkeypatch, {"check_drawdown": [ig._issue("drawdown", "warning", "x")]})

        def _boom(**k):
            raise RuntimeError("archive locked")

        monkeypatch.setattr(ig, "check_close_capture", _boom)
        res = ig.run_integrity()
        assert res["failed"] == ["close_capture"]
        assert any("crashed" in i["detail"] for i in res["issues"])
        assert any(i["check"] == "drawdown" for i in res["issues"])

    def test_only_restricts_runners(self, monkeypatch):
        _mock_all_checks(monkeypatch, {})
        res = ig.run_integrity(only={"launchd", "inplay"})
        assert set(res["ran"]) == {"launchd", "inplay"}

    def test_no_notify_without_flag(self, monkeypatch):
        _mock_all_checks(monkeypatch, {"check_inplay": [ig._issue("inplay", "critical", "x")]})
        fake = _FakeNotifier()
        monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))
        res = ig.run_integrity()
        assert not res["ok"] and not res["notified"] and fake.calls == []
