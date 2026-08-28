"""Tests for the webhook notifier: delivery retry, permanent-failure fast-fail,
and the operational ``notify_alert`` primitive. No real network I/O.

Previously the notifier was fire-and-forget with zero coverage — a broken
webhook silently swallowed every alert. These lock in the retry/return contract
the S4 heartbeat relies on to know an alert actually got out.
"""
from __future__ import annotations

import urllib.error
import urllib.request

import pytest

import evmax.notifications as notif
from evmax.notifications import Notifier


class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # Never actually sleep during backoff.
    monkeypatch.setattr(notif.time, "sleep", lambda _s: None)


def _patch_urlopen(monkeypatch, responses):
    """responses: list of either _FakeResp or Exception, consumed per call."""
    calls = {"n": 0}
    seq = list(responses)

    def fake(req, timeout=10):
        calls["n"] += 1
        item = seq[min(calls["n"] - 1, len(seq) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return calls


def _slack_notifier() -> Notifier:
    return Notifier(slack_url="http://slack.test/hook")


class TestDelivery:
    def test_delivers_on_first_success(self, monkeypatch):
        calls = _patch_urlopen(monkeypatch, [_FakeResp(200)])
        assert _slack_notifier()._post("http://x", {"text": "hi"}) is True
        assert calls["n"] == 1

    def test_204_counts_as_success(self, monkeypatch):
        calls = _patch_urlopen(monkeypatch, [_FakeResp(204)])
        assert _slack_notifier()._post("http://x", {"text": "hi"}) is True
        assert calls["n"] == 1

    def test_retries_then_succeeds(self, monkeypatch):
        calls = _patch_urlopen(
            monkeypatch,
            [urllib.error.URLError("boom"), urllib.error.URLError("boom"), _FakeResp(200)],
        )
        assert _slack_notifier()._post("http://x", {"text": "hi"}) is True
        assert calls["n"] == 3

    def test_gives_up_after_max_retries(self, monkeypatch):
        calls = _patch_urlopen(monkeypatch, [urllib.error.URLError("down")])
        assert _slack_notifier()._post("http://x", {"text": "hi"}) is False
        assert calls["n"] == notif._MAX_RETRIES

    def test_permanent_4xx_not_retried(self, monkeypatch):
        calls = _patch_urlopen(monkeypatch, [_http_error(404)])
        assert _slack_notifier()._post("http://x", {"text": "hi"}) is False
        assert calls["n"] == 1  # 404 is permanent — no retry

    def test_429_is_retried(self, monkeypatch):
        calls = _patch_urlopen(
            monkeypatch, [_http_error(429), _http_error(429), _FakeResp(200)]
        )
        assert _slack_notifier()._post("http://x", {"text": "hi"}) is True
        assert calls["n"] == 3

    def test_500_is_retried(self, monkeypatch):
        calls = _patch_urlopen(monkeypatch, [_http_error(503)])
        assert _slack_notifier()._post("http://x", {"text": "hi"}) is False
        assert calls["n"] == notif._MAX_RETRIES


class TestSendAggregation:
    def test_send_false_if_any_webhook_fails(self, monkeypatch):
        n = Notifier(slack_url="http://slack.test", discord_url="http://discord.test")

        def fake(req, timeout=10):
            # Slack ok, Discord fails.
            if "slack" in req.full_url:
                return _FakeResp(200)
            raise urllib.error.URLError("discord down")

        monkeypatch.setattr(urllib.request, "urlopen", fake)
        assert n._send("hi") is False

    def test_send_true_when_all_succeed(self, monkeypatch):
        n = Notifier(slack_url="http://slack.test", discord_url="http://discord.test")
        _patch_urlopen(monkeypatch, [_FakeResp(200)])
        assert n._send("hi") is True


class TestNotifyAlert:
    def test_not_configured_is_noop(self, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )
        n = Notifier()  # no webhooks
        assert n.is_configured() is False
        assert n.notify_alert("data source down", "Pinnacle 403") is False
        assert called["n"] == 0  # never touched the network

    def test_alert_carries_severity_and_body(self, monkeypatch):
        captured = {}

        def _capture(text):
            captured["text"] = text
            return True

        n = _slack_notifier()
        monkeypatch.setattr(n, "_send", _capture)
        assert n.notify_alert("Pinnacle down", "403 BAD_LOCATION", severity="critical") is True
        assert "🚨" in captured["text"]
        assert "critical" in captured["text"]
        assert "Pinnacle down" in captured["text"]
        assert "403 BAD_LOCATION" in captured["text"]

    def test_alert_returns_delivery_result(self, monkeypatch):
        n = _slack_notifier()
        _patch_urlopen(monkeypatch, [urllib.error.URLError("down")])
        # Delivery fails on every retry → alert reports False so ops can react.
        assert n.notify_alert("x", "y") is False
