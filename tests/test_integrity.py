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


def _days(*triples, start: int = 1) -> list[dict]:
    """(fetched, matched, sharp) per consecutive day from 2026-09-{start}."""
    return [
        {"scan_date": f"2026-09-{start + i:02d}", "fetched": f, "matched": m, "sharp": s}
        for i, (f, m, s) in enumerate(triples)
    ]


class TestZeroMatchStreak:
    """The absolute tripwire: no baseline needed (the UFC short-title break
    predated the ledger, so its 14d median was 0 and the relative check
    stayed silent for weeks)."""

    def test_ufc_shape_is_critical_without_any_baseline(self):
        # Every day in the ledger is a zero — exactly what UFC looked like.
        daily = {"ufc": _days((262, 0, 18), (324, 0, 18), (180, 0, 16), (170, 0, 18))}
        issues = ig._zero_match_streak_issues(daily, {"ufc": "shadow"}, TODAY)
        assert len(issues) == 1
        i = issues[0]
        assert i["check"] == "match_rate" and i["severity"] == "critical"
        assert "ufc (shadow)" in i["detail"] and "last 4 scan days since 2026-09-01" in i["detail"]
        assert "Pinnacle had up to 18" in i["detail"]

    def test_live_sector_is_critical_and_unknown_sharp_counts_as_posted(self):
        daily = {"nfl": _days((200, 0, None), (210, 0, None), (190, 0, None))}
        issues = ig._zero_match_streak_issues(daily, {"nfl": "live"}, TODAY)
        assert len(issues) == 1 and issues[0]["severity"] == "critical"
        assert "count not logged" in issues[0]["detail"]

    def test_disabled_or_unregistered_sector_only_warns(self):
        daily = {
            "worldcup": _days((40, 0, 5), (40, 0, 5), (40, 0, 5)),
            "valorant": _days((10, 0, 3), (10, 0, 3), (10, 0, 3)),
        }
        issues = ig._zero_match_streak_issues(
            daily, {"worldcup": "disabled", "valorant": None}, TODAY,
        )
        assert [i["severity"] for i in issues] == ["warning", "warning"]
        assert "valorant (unregistered)" in issues[0]["detail"]

    def test_pinnacle_posted_nothing_downgrades_to_warning(self):
        # NBA 2026-09: Kalshi lists opening night (Oct 20), Pinnacle has no
        # NBA board yet — also the stale-league-id fingerprint, so still warn.
        daily = {"nba": _days((30, 0, 0), (54, 0, 0), (48, 0, None))}
        issues = ig._zero_match_streak_issues(daily, {"nba": "live"}, TODAY)
        assert len(issues) == 1 and issues[0]["severity"] == "warning"
        assert "check_pinnacle_leagues.py -s nba" in issues[0]["detail"]

    def test_any_match_in_the_window_or_short_history_is_silent(self):
        modes = {"cs2": "shadow", "ufc": "shadow"}
        # cs2: zero, zero, then a matched day — streak broken at the end.
        assert ig._zero_match_streak_issues(
            {"cs2": _days((700, 0, 9), (450, 0, 9), (580, 12, 9))}, modes, TODAY,
        ) == []
        # Only two fetch days so far.
        assert ig._zero_match_streak_issues(
            {"ufc": _days((100, 0, 9), (100, 0, 9))}, modes, TODAY,
        ) == []

    def test_non_fetch_days_do_not_break_or_count_toward_the_streak(self):
        # Off days (fetched 0) between card weeks are skipped, not "healthy".
        daily = {"ufc": _days((100, 0, 9), (0, 0, 0), (120, 0, 9), (0, 0, 0), (90, 0, 9))}
        issues = ig._zero_match_streak_issues(daily, {"ufc": "shadow"}, TODAY)
        assert len(issues) == 1 and issues[0]["severity"] == "critical"
        assert ig._zero_match_streak_issues(
            {"ufc": _days((100, 0, 9), (0, 0, 0), (120, 0, 9))}, {"ufc": "shadow"}, TODAY,
        ) == []

    def test_stale_streak_from_a_finished_season_is_silent(self):
        daily = {"wnba": _days((271, 0, 4), (930, 0, 4), (500, 0, 4), start=1)}
        later = date(2026, 9, 20)
        assert ig._zero_match_streak_issues(daily, {"wnba": "live"}, later) == []

    def test_skip_suppresses_sectors_already_flagged(self):
        daily = {"ufc": _days((100, 0, 9), (100, 0, 9), (100, 0, 9))}
        assert ig._zero_match_streak_issues(
            daily, {"ufc": "shadow"}, TODAY, skip={"ufc"},
        ) == []


class TestCheckMatchRateWiring:
    """check_match_rate over a real (temp) scan_sector_stats ledger."""

    def _ledger(self, tmp_path, monkeypatch, rows):
        import sqlite3

        from evmax.agents.cleanup import db as db_module

        path = tmp_path / "predictions.db"
        monkeypatch.setattr(db_module, "DB_PATH", path)
        with db_module.get_connection() as conn:  # creates schema + migrations
            conn.executemany(
                "INSERT INTO scan_sector_stats (scan_date, source, sector, markets_fetched, "
                "markets_matched, ev_gaps, error, sharp_events) VALUES (?, 'cli', ?, ?, ?, 0, ?, ?)",
                rows,
            )
            conn.commit()
        return sqlite3

    def test_streak_fires_and_relative_check_is_not_duplicated(self, tmp_path, monkeypatch):
        rows = []
        for day in ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05"):
            # Two cycles per day; the second predates the sharp_events column.
            rows.append((day, "ufc", 150, 0, None, 18))
            rows.append((day, "ufc", 150, 0, None, None))
            rows.append((day, "nba", 30, 0, None, 0))
            rows.append((day, "soccer", 200, 120, None, 90))
        # soccer: history matched, TODAY zero → the RELATIVE critical owns it.
        rows[-1] = ("2026-09-05", "soccer", 200, 0, None, 90)
        # An errored cycle never counts.
        rows.append(("2026-09-05", "lol", 0, 0, "timed out", None))
        self._ledger(tmp_path, monkeypatch, rows)

        issues = ig.check_match_rate(today=TODAY)
        by_sector = {i["detail"].split(" ", 1)[0].rstrip(":"): i for i in issues}
        assert set(by_sector) == {"ufc", "nba", "soccer"}
        assert by_sector["ufc"]["severity"] == "critical"      # shadow, Pinnacle had 18
        # Per-cycle MAX, NULL cycles ignored — never a sum across cycles.
        assert "Pinnacle had up to 18" in by_sector["ufc"]["detail"]
        assert "up to 150 fetched per scan" in by_sector["ufc"]["detail"]
        assert by_sector["nba"]["severity"] == "warning"       # live, Pinnacle posted 0
        assert by_sector["soccer"]["severity"] == "critical"
        assert "14d median" in by_sector["soccer"]["detail"]   # relative, not the streak
        assert sum(1 for i in issues if i["detail"].startswith("soccer")) == 1


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

    def test_nfl_spread_lay_and_take_watches_present(self):
        """NFL spread is a shadow_market_type judged per side — both lay and take
        must be watched, keyed to nfl spread, so promotion evidence surfaces."""
        nfl = [w for w in ig.GATE_WATCHES
               if w["category"] == "nfl" and w["market_type"] == "spread"]
        assert {w["side"] for w in nfl} == {"lay", "take"}
        # regular scan rows, NOT the WNBA anchored-entry laddered stream
        assert all("sources_token" not in w for w in nfl)


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
