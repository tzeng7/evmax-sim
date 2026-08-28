"""Tests for the CLV tripwire (evmax/agents/cleanup/clv_monitor.py).

The tripwire reuses the promotion board's LIVE-DEGRADING verdict and pushes a
warning — it must never touch model parameters (CLV is a timing signal). These
mock the board + notifier so no DB or network is involved.
"""
from __future__ import annotations

from evmax.agents.cleanup import clv_monitor
from evmax.notifications import Notifier


def _row(sector, mt, venue, verdict, mean_clv, n=40, frac=0.4):
    return {
        "sector": sector,
        "market_type": mt,
        "venue": venue,
        "verdict": verdict,
        "clv": {"n": n, "mean_clv_pp": mean_clv, "frac_positive": frac},
    }


def _patch_board(monkeypatch, rows):
    monkeypatch.setattr(
        "evmax.agents.cleanup.promotion_board.compute_promotion_board",
        lambda days=30, staleness_h=3.0, sector=None: rows,
    )


class _FakeNotifier:
    def __init__(self, result=True):
        self.calls = []
        self._result = result

    def notify_alert(self, title, message, *, severity="warning"):
        self.calls.append((title, message, severity))
        return self._result


def _patch_notifier(monkeypatch, fake):
    monkeypatch.setattr(Notifier, "from_settings", staticmethod(lambda: fake))


def test_find_degrading_filters_and_sorts(monkeypatch):
    _patch_board(monkeypatch, [
        _row("wnba", "moneyline", "kalshi", "LIVE-HEALTHY", 0.40),
        _row("soccer", "moneyline", "kalshi", "LIVE-DEGRADING", -1.20),
        _row("baseball", "moneyline", "kalshi", "SHARP-PASSTHROUGH", 0.00),
        _row("nba", "total", "kalshi", "LIVE-DEGRADING", -0.30),
    ])
    out = clv_monitor.find_degrading_groups()
    assert [r["sector"] for r in out] == ["soccer", "nba"]  # worst CLV first
    assert all(r["verdict"] == "LIVE-DEGRADING" for r in out)


def test_no_degrading_returns_empty(monkeypatch):
    _patch_board(monkeypatch, [
        _row("wnba", "moneyline", "kalshi", "LIVE-HEALTHY", 0.40),
        _row("baseball", "moneyline", "kalshi", "COLLECTING 12/30", 0.10),
    ])
    assert clv_monitor.find_degrading_groups() == []


def test_format_alert_contents():
    rows = [_row("soccer", "moneyline", "polymarket_us", "LIVE-DEGRADING", -1.25, n=44, frac=0.41)]
    title, message = clv_monitor.format_clv_alert(rows, days=30)
    assert "1 live book" in title
    assert "soccer" in message
    assert "-1.25pp" in message
    assert "polymarket_us" in message
    assert "n=44" in message
    # The guidance must steer to entry-timing, not the model.
    assert "model" in message.lower()


def test_tripwire_notifies_when_degrading(monkeypatch):
    _patch_board(monkeypatch, [
        _row("soccer", "moneyline", "kalshi", "LIVE-DEGRADING", -1.20),
    ])
    fake = _FakeNotifier(result=True)
    _patch_notifier(monkeypatch, fake)

    result = clv_monitor.run_clv_tripwire(notify=True)
    assert result["notified"] is True
    assert len(result["degrading"]) == 1
    assert len(fake.calls) == 1
    assert fake.calls[0][2] == "warning"  # severity


def test_tripwire_silent_when_clean(monkeypatch):
    _patch_board(monkeypatch, [
        _row("wnba", "moneyline", "kalshi", "LIVE-HEALTHY", 0.40),
    ])
    fake = _FakeNotifier()
    _patch_notifier(monkeypatch, fake)

    result = clv_monitor.run_clv_tripwire(notify=True)
    assert result["notified"] is False
    assert result["degrading"] == []
    assert fake.calls == []  # nothing to report → no push


def test_tripwire_no_push_without_notify_flag(monkeypatch):
    _patch_board(monkeypatch, [
        _row("soccer", "moneyline", "kalshi", "LIVE-DEGRADING", -1.20),
    ])
    fake = _FakeNotifier()
    _patch_notifier(monkeypatch, fake)

    result = clv_monitor.run_clv_tripwire(notify=False)
    assert len(result["degrading"]) == 1  # still reported to the caller
    assert result["notified"] is False
    assert fake.calls == []  # but not pushed


def test_delivery_failure_reports_not_notified(monkeypatch):
    _patch_board(monkeypatch, [
        _row("soccer", "moneyline", "kalshi", "LIVE-DEGRADING", -1.20),
    ])
    fake = _FakeNotifier(result=False)  # webhook delivery fails
    _patch_notifier(monkeypatch, fake)

    result = clv_monitor.run_clv_tripwire(notify=True)
    assert result["notified"] is False
    assert len(fake.calls) == 1  # it tried
