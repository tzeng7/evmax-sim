"""Discord bot: REST transport, dashboard-parity embeds, the scan feed, the
Notifier wiring, and the framework-free slash-command handlers.

No network I/O. discord.py is only required by the last class (skipped when
the optional extra is absent, e.g. in CI's `uv sync --extra dev`).
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import pytest

import evmax.discord_bot.client as dclient
from evmax.agents.coordinator import CycleResult
from evmax.agents.odds.ev_gap_agent import EVGap
from evmax.discord_bot import embeds as E
from evmax.discord_bot.client import (
    EMBED_DESCRIPTION_MAX,
    MESSAGE_EMBED_CHARS_MAX,
    MESSAGE_EMBEDS_MAX,
    DiscordBotClient,
    _split_text,
    batch_embeds,
    embed_char_count,
)
from evmax.discord_bot.feed import (
    build_scan_feed,
    post_scan_feed,
    scan_feed_suppressed,
    suppress_scan_feed,
)
from evmax.discord_bot.handlers import HELP_TEXT, CommandHandlers, Reply
from evmax.notifications import Notifier
from evmax.web.playlist import dashboard_play_dicts, filter_scan_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_local_noon() -> datetime:
    return datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0)


def _gap(
    *,
    market_id: str = "KXTEST-1",
    venue: str = "kalshi",
    ev: float = 0.07,
    kelly: float = 0.02,
    yes_team: str = "lakers",
    market_type: str = "moneyline",
    full_blend: bool = True,
    event_date: datetime | None = None,
    **extra,
) -> EVGap:
    return EVGap(
        market_id=market_id,
        # One event per market id: the dashboard collapses the SAME bet quoted
        # twice into one best-execution row, so parity tests need distinct bets.
        event_id=f"nba::2026-07-08::lakers_vs_warriors_{market_id}",
        sector="nba",
        yes_team=yes_team,
        market_type=market_type,
        kalshi_yes_price=0.45,
        sharp_true_prob=0.55,
        blended_true_prob=0.55,
        ev_pct=ev,
        kelly_full=0.10,
        kelly_fraction=kelly,
        match_confidence=0.95,
        volume_usd=1000.0,
        spread_pct=0.02,
        event_title="Lakers vs Warriors",
        event_date=event_date or _today_local_noon(),
        model_sources="sharp,elo,form",
        full_blend=full_blend,
        venue=venue,
        **extra,
    )


def _row(**over) -> dict:
    base = {
        "event_date": "2026-09-04", "sector": "nfl", "venue": "kalshi",
        "event_title": "Kansas City Chiefs vs Los Angeles Chargers",
        "display_label": "Chiefs ML", "kalshi_price": 0.45, "true_prob": 0.49,
        "ev_pct": 4.1, "maker_ev_pct": 5.2, "maker_only": False,
        "maker_limit_price": 0.48, "maker_bid_price": 0.46, "maker_bid_ev_pct": 6.0,
        "maker_bid_kelly_fraction": 0.03, "kelly_fraction": 0.02, "stake": 5.0,
        "model_sources": "sharp,elo,form", "market_id": "KX1", "mode": "live",
        "alt_venue": None, "alt_venue_price": None, "venue_options": None,
    }
    base.update(over)
    return base


class _FakeResp:
    def __init__(self, status: int, body: bytes = b"{}") -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code: int, body: dict | None = None) -> urllib.error.HTTPError:
    fp = io.BytesIO(json.dumps(body).encode()) if body is not None else None
    return urllib.error.HTTPError("http://x", code, "err", {}, fp)


class _RecordingBot:
    """Stands in for DiscordBotClient in Notifier / feed tests."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.embeds_calls: list[list[dict]] = []
        self.text_calls: list[str] = []

    def is_configured(self) -> bool:
        return True

    def post_embeds(self, embeds, *, content=None, channel_id=None) -> bool:
        self.embeds_calls.append(list(embeds))
        return self.ok

    def post_text(self, text, *, channel_id=None) -> bool:
        self.text_calls.append(text)
        return self.ok


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(dclient.time, "sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------

class TestClient:
    def _client(self) -> DiscordBotClient:
        return DiscordBotClient("tok", "123")

    def _patch(self, monkeypatch, responses):
        calls: dict = {"n": 0, "reqs": []}
        seq = list(responses)

        def fake(req, timeout=15):
            calls["n"] += 1
            calls["reqs"].append(req)
            item = seq[min(calls["n"] - 1, len(seq) - 1)]
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(urllib.request, "urlopen", fake)
        return calls

    def test_not_configured_never_posts(self, monkeypatch):
        calls = self._patch(monkeypatch, [_FakeResp(200)])
        c = DiscordBotClient("", "")
        assert c.is_configured() is False
        assert c.post("hi") is False
        assert calls["n"] == 0

    def test_post_sends_bot_auth_to_channel_endpoint(self, monkeypatch):
        calls = self._patch(monkeypatch, [_FakeResp(200)])
        assert self._client().post("hello", [{"title": "t"}]) is True
        req = calls["reqs"][0]
        assert req.full_url.endswith("/channels/123/messages")
        assert req.get_header("Authorization") == "Bot tok"
        payload = json.loads(req.data)
        assert payload == {"content": "hello", "embeds": [{"title": "t"}]}

    def test_content_truncated_to_cap(self, monkeypatch):
        calls = self._patch(monkeypatch, [_FakeResp(200)])
        assert self._client().post("x" * 5000) is True
        assert len(json.loads(calls["reqs"][0].data)["content"]) == 2000

    def test_429_honours_retry_after_then_succeeds(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr(dclient.time, "sleep", slept.append)
        calls = self._patch(
            monkeypatch, [_http_error(429, {"retry_after": 0.25}), _FakeResp(200)],
        )
        assert self._client().post("hi") is True
        assert calls["n"] == 2
        assert slept == [0.25]

    def test_permanent_4xx_not_retried(self, monkeypatch):
        calls = self._patch(monkeypatch, [_http_error(403, {"message": "Missing Access"})])
        assert self._client().post("hi") is False
        assert calls["n"] == 1

    def test_5xx_and_network_retried_then_fail(self, monkeypatch):
        calls = self._patch(monkeypatch, [_http_error(503)])
        assert self._client().post("hi") is False
        assert calls["n"] == dclient._MAX_RETRIES
        calls = self._patch(monkeypatch, [urllib.error.URLError("down")])
        assert self._client().post("hi") is False
        assert calls["n"] == dclient._MAX_RETRIES

    def test_post_embeds_batches_and_stops_on_first_failure(self, monkeypatch):
        big = {"title": "t", "description": "d" * 2900}
        calls = self._patch(monkeypatch, [_FakeResp(200)])
        assert self._client().post_embeds([big, big, big]) is True
        assert calls["n"] == 2  # 6000-char cap → 2 per message (2 × 2901), then 1
        calls = self._patch(monkeypatch, [_FakeResp(200), _http_error(404)])
        assert self._client().post_embeds([big, big, big]) is False
        assert calls["n"] == 2  # stopped after the failing second message

    def test_post_text_splits_on_lines(self, monkeypatch):
        calls = self._patch(monkeypatch, [_FakeResp(200)])
        text = "\n".join("line %03d %s" % (i, "y" * 90) for i in range(60))
        assert self._client().post_text(text) is True
        assert calls["n"] == len(_split_text(text, 2000)) >= 3
        for req in calls["reqs"]:
            assert len(json.loads(req.data)["content"]) <= 2000

    def test_dm_target_opens_dm_channel_once_and_posts_there(self, monkeypatch):
        calls = self._patch(monkeypatch, [_FakeResp(200, b'{"id": "777"}'), _FakeResp(200), _FakeResp(200)])
        c = DiscordBotClient("tok", dm_user_id="42")
        assert c.is_configured() and c.describe_targets() == "DM to user 42"
        assert c.post("hi") is True and c.post("again") is True
        paths = [r.full_url.rsplit("/api/v10", 1)[1] for r in calls["reqs"]]
        assert paths == ["/users/@me/channels", "/channels/777/messages", "/channels/777/messages"]
        assert json.loads(calls["reqs"][0].data) == {"recipient_id": "42"}

    def test_channel_and_dm_both_receive(self, monkeypatch):
        calls = self._patch(monkeypatch, [_FakeResp(200, b'{"id": "777"}'), _FakeResp(200)])
        c = DiscordBotClient("tok", "123", dm_user_id="42")
        assert c.post("hi") is True
        paths = [r.full_url.rsplit("/api/v10", 1)[1] for r in calls["reqs"]]
        # DM channel is resolved up front, then the message goes to both targets.
        assert paths == ["/users/@me/channels", "/channels/123/messages", "/channels/777/messages"]
        assert c.describe_targets() == "channel 123 + DM to user 42"

    def test_dm_open_failure_reports_false_without_channel(self, monkeypatch):
        calls = self._patch(monkeypatch, [_http_error(403, {"message": "Cannot send messages to this user"})])
        c = DiscordBotClient("tok", dm_user_id="42")
        assert c.post("hi") is False
        assert calls["n"] == 1  # never tried a message send

    def test_from_settings_none_without_config(self, monkeypatch):
        from evmax.settings import get_settings
        s = get_settings()
        monkeypatch.setattr(s, "discord_bot_token", "")
        monkeypatch.setattr(s, "discord_channel_id", "")
        monkeypatch.setattr(s, "discord_dm_user_id", "")
        assert DiscordBotClient.from_settings() is None
        monkeypatch.setattr(s, "discord_bot_token", "t")
        assert DiscordBotClient.from_settings() is None  # token alone is not a target
        monkeypatch.setattr(s, "discord_dm_user_id", "42")
        c = DiscordBotClient.from_settings()
        assert c is not None and c.channel_id == "" and c.dm_user_id == "42"
        monkeypatch.setattr(s, "discord_channel_id", "9")
        c = DiscordBotClient.from_settings()
        assert c is not None and c.channel_id == "9"


class TestBatching:
    def test_embed_char_count_sums_every_text_part(self):
        e = {"title": "ab", "description": "cde", "footer": {"text": "f"},
             "fields": [{"name": "n", "value": "vv"}]}
        assert embed_char_count(e) == 2 + 3 + 1 + 1 + 2

    def test_batches_respect_count_and_char_caps(self):
        small = [{"title": f"t{i}"} for i in range(23)]
        batches = batch_embeds(small)
        assert [len(b) for b in batches] == [10, 10, 3]
        big = [{"description": "x" * 2500} for _ in range(5)]
        batches = batch_embeds(big)
        assert all(len(b) <= MESSAGE_EMBEDS_MAX for b in batches)
        assert all(sum(embed_char_count(e) for e in b) <= MESSAGE_EMBED_CHARS_MAX for b in batches)
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_split_text_hard_splits_long_lines(self):
        chunks = _split_text("a" * 4500, 2000)
        assert [len(c) for c in chunks] == [2000, 2000, 500]


# ---------------------------------------------------------------------------
# Embeds — column-for-column parity with the dashboard panels
# ---------------------------------------------------------------------------

class TestJsFormatting:
    @pytest.mark.parametrize("p,expected", [
        (0.12, "12¢"), (0.127, "12.7¢"), (0.5, "50¢"), (0.005, "0.5¢"),
        (None, "-"), (0.0, "-"), (1.0, "-"), (1.2, "-"), ("x", "-"), (float("nan"), "-"),
    ])
    def test_cents_matches_probToCents(self, p, expected):
        assert E.cents(p) == expected

    def test_to_fixed_rounds_half_away_from_zero_like_js(self):
        # Python's format() would give "4.2" / "-4.2" (half-to-even).
        assert E.to_fixed(4.25, 1) == "4.3"
        assert E.to_fixed(-4.25, 1) == "-4.3"
        assert E.to_fixed(0.125, 2) == "0.13"
        assert E.to_fixed(5, 2) == "5.00"
        assert E.to_fixed(float("nan"), 1) == "NaN"

    def test_js_round_half_toward_plus_infinity(self):
        assert E.js_round(2.5) == 3
        assert E.js_round(-2.5) == -2
        assert E.js_round(44.4) == 44

    def test_venue_short(self):
        assert E.venue_short("kalshi") == "Kalshi"
        assert E.venue_short("polymarket_us") == "Poly"
        assert E.venue_short(None) == ""


class TestScanTable:
    def test_column_order_is_the_dashboard_header(self):
        assert E.SCAN_HEADERS == (
            "Date", "Sector", "Venue", "Event", "Outcome",
            "Ask", "Fair Value", "Model", "EV", "Maker EV", "Limit ¢", "Bid ¢",
            "Fill ¢", "Stake ($)", "Models",
        )
        assert len(E.SCAN_ALIGN) == len(E.SCAN_HEADERS)

    def test_live_row_cells(self):
        row = E.scan_row(_row(), bankroll=250.0, kelly=0.5, scan_kelly=0.5)
        assert row == [
            "2026-09-04", "nfl", "Kalshi",
            "Kansas City Chiefs vs Los Angeles Chargers", "Chiefs ML",
            "45¢", "49¢", "49.0%", "4.1%", "5.2%", "48¢", "46¢",
            "45¢",          # Fill ¢ = ask for a taker play
            "5.00",         # Stake = bankroll × min(kelly_fraction, 5%)
            "sharp,elo,form",
        ]

    def test_shadow_and_maker_tags_ride_in_the_sector_cell(self):
        row = E.scan_row(_row(mode="shadow", maker_only=True), 250.0)
        assert row[1] == "nfl shadow MAKER"
        # Maker-only rows seed Fill ¢ / Stake to the resting bid + maker Kelly.
        assert row[12] == "46¢"
        assert row[13] == "7.50"   # 250 × min(0.03, 0.05)

    def test_maker_columns_dash_when_absent(self):
        row = E.scan_row(_row(maker_ev_pct=None, maker_limit_price=None, maker_bid_price=None), 250.0)
        assert row[9:12] == ["—", "—", "—"]

    def test_alt_venue_annotation_on_outcome(self):
        row = E.scan_row(_row(alt_venue="polymarket_us", alt_venue_price=0.47), 250.0)
        assert row[4] == "Chiefs ML · also Poly 47¢"

    def test_dual_venue_row_shows_the_dropdown_options_and_no_annotation(self):
        legs = [_row(venue="kalshi", ev_pct=4.1), _row(venue="polymarket_us", ev_pct=3.25, maker_only=True)]
        row = E.scan_row(_row(venue_options=legs, alt_venue="polymarket_us", alt_venue_price=0.47), 250.0)
        assert row[2] == "Kalshi · 4.1% | Poly · 3.3% mkr"
        assert row[4] == "Chiefs ML"

    def test_stake_rescales_with_kelly_knob_and_caps_at_5pct(self):
        row = E.scan_row(_row(kelly_fraction=0.04), 250.0, kelly=1.0, scan_kelly=0.5)
        assert row[13] == "12.50"  # 0.04 × 2 = 0.08 → capped 0.05 × 250

    def test_title_is_the_panel_header_and_empty_body(self):
        out = E.scan_result_embeds(
            [], markets_fetched=600, markets_matched=71, bankroll=250, kelly=0.5, sectors=["nfl"],
        )
        assert len(out) == 1
        assert out[0]["title"] == "Scan Results — 0 plays (600 markets, 71 matched)"
        assert out[0]["description"] == "No +EV plays found."
        assert "sectors: nfl" in out[0]["footer"]["text"]
        assert "window: today + tomorrow" in out[0]["footer"]["text"]

    def test_table_in_code_block_with_footer_and_errors(self):
        out = E.scan_result_embeds(
            [_row()], markets_fetched=10, markets_matched=5, bankroll=250, kelly=0.5,
            date_from="2026-09-04", date_to="2026-09-05", duration_s=3.31,
            errors=["nhl: timed out"], source="bankroll live:kalshi",
        )
        assert len(out) == 1
        d = out[0]["description"]
        assert d.startswith("```\n") and d.endswith("\n```")
        lines = d.strip("`\n").split("\n")
        assert lines[0].split() == ["Date", "Sector", "Venue", "Event", "Outcome", "Ask",
                                    "Fair", "Value", "Model", "EV", "Maker", "EV",
                                    "Limit", "¢", "Bid", "¢", "Fill", "¢", "Stake", "($)", "Models"]
        assert "Chiefs ML" in lines[2] and lines[2].startswith("2026-09-04")
        f = out[0]["footer"]["text"]
        assert "window: 2026-09-04 → 2026-09-05" in f and "3.3s" in f
        assert "errors: nhl: timed out" in f and "bankroll live:kalshi" in f

    def test_long_table_chunks_under_description_cap_with_header_repeated(self):
        rows = [_row(market_id=f"KX{i}", event_title=f"Game number {i} with a long title") for i in range(200)]
        out = E.scan_result_embeds(rows, markets_fetched=1, markets_matched=1, bankroll=250, kelly=0.5)
        assert len(out) > 1
        for i, e in enumerate(out, 1):
            assert len(e["description"]) <= EMBED_DESCRIPTION_MAX
            first = e["description"].split("\n")[1]
            assert first.startswith("Date")
            if i > 1:
                assert e["title"].endswith(f"(cont. {i}/{len(out)})")
        assert "footer" in out[-1] and all("footer" not in e for e in out[:-1])
        # Every row appears exactly once, in input order.
        body = "\n".join(e["description"] for e in out)
        positions = [body.index(f"Game number {i} ") for i in range(200)]
        assert positions == sorted(positions)
        # And the messages the client would send respect the 6000-char cap.
        for batch in batch_embeds(out):
            assert sum(embed_char_count(e) for e in batch) <= MESSAGE_EMBED_CHARS_MAX


class TestOpenAndSettledTables:
    def test_open_position_row_matches_panel(self):
        b = {"market_id": "M1", "status": "in_progress", "event_date": "2026-09-04", "sector": "wnba",
             "venue": "polymarket_us", "event_title": "Aces vs Liberty", "display_label": "Aces ML",
             "kalshi_yes_price": 0.61, "blended_true_prob": 0.667, "ev_pct": 0.0425, "kelly_fraction": 0.03}
        row = E.open_position_row(b, bankroll=250.0, kelly=0.5, scan_mids={"M1"})
        assert row == ["NEW LIVE", "2026-09-04", "wnba", "Poly", "Aces vs Liberty", "Aces ML",
                       "61¢", "66.7¢", "4.3%", "$7.50"]
        assert E.OPEN_HEADERS == ("", "Date", "Sector", "Venue", "Event", "Outcome", "Ask", "Fair Value", "EV", "Stake")

    def test_open_positions_embeds_cap_and_sector_filter(self):
        bets = [{"market_id": f"M{i}", "sector": "nba" if i % 2 else "nfl", "event_date": "2026-09-04",
                 "kelly_fraction": 0.01} for i in range(50)]
        out = E.open_positions_embeds(bets, bankroll=250, kelly=0.5, sector="nba")
        assert out[0]["title"] == "Open Positions (25)"
        out = E.open_positions_embeds(bets, bankroll=250, kelly=0.5)
        assert out[0]["title"] == "Open Positions (50)"
        assert "showing 40 of 50" in out[-1]["footer"]["text"]
        assert E.open_positions_embeds([], bankroll=250, kelly=0.5)[0]["description"] == "No open positions."

    def test_settled_row_matches_panel_including_gross_pnl(self):
        b = {"event_date": "2026-09-03", "sector": "nba", "venue": "kalshi", "event_title": "A vs B",
             "display_label": "A ML", "kalshi_yes_price": 0.445, "blended_true_prob": 0.494,
             "ev_pct": 0.0525, "outcome": 1, "placed_stake": 10.0, "placed_price": 0.5}
        assert E.settled_row(b) == ["2026-09-03", "nba", "Kalshi", "A vs B", "A ML",
                                    "45c", "49%", "5.3%", "WON", "$10.00"]
        b.update(outcome=0)
        assert E.settled_row(b)[-2:] == ["LOST", "$-10.00"]
        assert E.SETTLED_HEADERS == ("Date", "Sector", "Venue", "Event", "Outcome", "Ask", "Model", "EV", "Result", "P&L")

    def test_settled_embeds_summary_footer_and_placed_filter(self):
        bets = [{"placed": 1, "outcome": 1, "event_date": "2026-09-01", "kalshi_yes_price": 0.5},
                {"placed": 0, "outcome": 0, "event_date": "2026-09-02", "kalshi_yes_price": 0.5}]
        summary = {"total_bets": 2, "wins": 1, "losses": 1, "win_rate": 50.0, "total_pnl": -1.5,
                   "roi_pct": -7.5, "avg_ev": 4.0}
        out = E.recent_settled_embeds(bets, summary=summary)
        assert out[0]["title"] == "Recent Settled Bets"
        assert "2 bets · 1W / 1L · win rate 50.0% · P&L $-1.50 · ROI -7.5% · avg EV 4.0%" == out[0]["footer"]["text"]
        out = E.recent_settled_embeds(bets, summary=summary, placed_only=True)
        assert out[0]["title"].endswith("Placed Only")
        lines = out[0]["description"].strip("`\n").split("\n")
        assert len(lines) == 3  # header + rule + one row


class TestAlertAndStatus:
    def test_alert_embed_severity_color_and_text(self):
        e = E.alert_embed("Pinnacle down", "403 BAD_LOCATION", severity="critical")
        assert e["color"] == E.COLOR_CRITICAL
        assert "🚨" in e["title"] and "critical" in e["title"] and "Pinnacle down" in e["title"]
        assert e["description"] == "403 BAD_LOCATION"

    def test_status_embed_all_clear_and_issues(self):
        assert E.status_embed({"ok": True, "issues": []})["color"] == E.COLOR_PLAYS
        e = E.status_embed({"ok": False, "issues": [
            {"severity": "warning", "detail": "scan stale"},
            {"severity": "critical", "detail": "resolve stopped"},
        ]})
        assert e["color"] == E.COLOR_CRITICAL
        assert "2 issue(s)" in e["title"]
        assert "• [critical] resolve stopped" in e["description"]


# ---------------------------------------------------------------------------
# Scan feed — what the coordinator hook posts
# ---------------------------------------------------------------------------

class TestScanFeed:
    def _result(self, gaps) -> CycleResult:
        r = CycleResult(bankroll=500.0, kelly_fraction=0.5)
        r.ev_gaps = list(gaps)
        r.sectors_scanned = ["nba"]
        r.markets_fetched = 40
        r.markets_matched = 12
        r.cycle_duration_s = 2.5
        return r

    def test_feed_rows_are_the_dashboard_rows_in_dashboard_order(self, monkeypatch):
        monkeypatch.setattr("evmax.web.playlist.placed_market_ids", lambda: set())
        gaps = [
            _gap(market_id="A", ev=0.03),
            _gap(market_id="B", ev=0.09),
            _gap(market_id="C", ev=0.05, full_blend=False),                     # partial blend → not a play
            _gap(market_id="D", ev=0.04, event_date=_today_local_noon() + timedelta(days=5)),  # outside window
            _gap(market_id="E", ev=0.06, market_type="map_handicap"),           # dropped by the view
        ]
        res = self._result(gaps)
        plays, embeds = build_scan_feed(res, placed_mids=set())
        expected = filter_scan_view(dashboard_play_dicts(res, 500.0), placed_mids=set())
        assert plays == expected
        assert [p["market_id"] for p in plays] == ["B", "A"]
        assert embeds[0]["title"] == "Scan Results — 2 plays (40 markets, 12 matched)"
        assert "Lakers vs Warriors" in embeds[0]["description"]
        assert "Bankroll $500.00" in embeds[0]["footer"]["text"]

    def test_placed_markets_are_excluded_like_the_dashboard(self):
        res = self._result([_gap(market_id="A"), _gap(market_id="B", ev=0.02)])
        plays, _ = build_scan_feed(res, placed_mids={"A"})
        assert [p["market_id"] for p in plays] == ["B"]

    def test_post_skips_empty_unless_post_empty(self):
        bot = _RecordingBot()
        res = self._result([])
        assert post_scan_feed(bot, res) is False
        assert bot.embeds_calls == []
        assert post_scan_feed(bot, res, post_empty=True) is True
        assert bot.embeds_calls[0][0]["description"] == "No +EV plays found."

    def test_post_sends_the_table_and_reports_delivery(self):
        bot = _RecordingBot()
        res = self._result([_gap(market_id="A")])
        assert post_scan_feed(bot, res, date_from="2000-01-01", date_to="2100-01-01") is True
        assert "Lakers ML" in bot.embeds_calls[0][0]["description"]
        assert post_scan_feed(_RecordingBot(ok=False), res, date_from="2000-01-01", date_to="2100-01-01") is False

    def test_build_failure_is_swallowed(self, monkeypatch):
        bot = _RecordingBot()
        monkeypatch.setattr("evmax.discord_bot.feed.build_scan_feed", lambda *a, **k: 1 / 0)
        assert post_scan_feed(bot, self._result([_gap()])) is False
        assert bot.embeds_calls == []

    def test_suppression_contextvar_follows_the_task_and_threads(self):
        assert scan_feed_suppressed() is False
        with suppress_scan_feed():
            assert scan_feed_suppressed() is True

            async def _inner():
                await asyncio.sleep(0)
                return scan_feed_suppressed(), await asyncio.to_thread(scan_feed_suppressed)

            assert asyncio.run(_inner()) == (True, True)
        assert scan_feed_suppressed() is False


# ---------------------------------------------------------------------------
# Notifier wiring
# ---------------------------------------------------------------------------

class TestNotifierBotTransport:
    def _result(self, ev: float = 0.07) -> CycleResult:
        r = CycleResult(bankroll=250.0, kelly_fraction=0.5)
        r.ev_gaps = [_gap(ev=ev)]
        return r

    def test_bot_alone_makes_notifier_configured(self):
        assert Notifier().is_configured() is False
        assert Notifier(discord_bot=_RecordingBot()).is_configured() is True

    def test_notify_cycle_posts_feed_via_bot_without_min_ev_gate(self, monkeypatch):
        posted: list = []
        monkeypatch.setattr(
            "evmax.discord_bot.feed.post_scan_feed",
            lambda client, result, **kw: posted.append((client, result, kw)) or True,
        )
        bot = _RecordingBot()
        n = Notifier(discord_bot=bot, post_empty_scans=True)
        n.notify_cycle(self._result(ev=0.03))  # below the 5% webhook gate
        assert len(posted) == 1
        assert posted[0][0] is bot and posted[0][2] == {"post_empty": True}
        assert bot.text_calls == []  # the feed, not the plain-text alert

    def test_feed_off_or_suppressed_skips_bot(self, monkeypatch):
        posted: list = []
        monkeypatch.setattr("evmax.discord_bot.feed.post_scan_feed", lambda *a, **k: posted.append(1))
        Notifier(discord_bot=_RecordingBot(), scan_feed=False).notify_cycle(self._result())
        with suppress_scan_feed():
            Notifier(discord_bot=_RecordingBot()).notify_cycle(self._result())
        assert posted == []

    def test_webhook_text_still_gated_and_sent_alongside_feed(self, monkeypatch):
        sent: list = []
        monkeypatch.setattr("evmax.discord_bot.feed.post_scan_feed", lambda *a, **k: True)
        n = Notifier(slack_url="http://slack.test", discord_bot=_RecordingBot())
        monkeypatch.setattr(n, "_post", lambda url, payload: sent.append((url, payload)) or True)
        n.notify_cycle(self._result(ev=0.03))
        assert sent == []
        n.notify_cycle(self._result(ev=0.07))
        assert len(sent) == 1 and "evmax" in sent[0][1]["text"]

    def test_notify_alert_goes_to_bot_as_embed_and_reports_delivery(self, monkeypatch):
        bot = _RecordingBot()
        n = Notifier(discord_url="http://discord.test", discord_bot=bot)
        monkeypatch.setattr(n, "_post", lambda url, payload: True)
        assert n.notify_alert("Pinnacle down", "403", severity="critical") is True
        assert bot.embeds_calls[0][0]["color"] == E.COLOR_CRITICAL
        assert "Pinnacle down" in bot.embeds_calls[0][0]["title"]
        bot.ok = False
        assert n.notify_alert("x", "y") is False

    def test_send_text_reaches_bot_channel(self):
        bot = _RecordingBot()
        Notifier(discord_bot=bot).send_text("arb: cheap basket")
        assert bot.text_calls == ["arb: cheap basket"]

    def test_from_settings_builds_bot_when_configured(self, monkeypatch):
        from evmax.settings import get_settings
        s = get_settings()
        monkeypatch.setattr(s, "discord_bot_token", "tok")
        monkeypatch.setattr(s, "discord_channel_id", "42")
        monkeypatch.setattr(s, "discord_dm_user_id", "7")
        monkeypatch.setattr(s, "discord_scan_feed", False)
        n = Notifier.from_settings()
        assert n.discord_bot is not None and n.discord_bot.channel_id == "42"
        assert n.discord_bot.dm_user_id == "7"
        assert n._scan_feed is False
        monkeypatch.setattr(s, "discord_bot_token", "")
        assert Notifier.from_settings().discord_bot is None


# ---------------------------------------------------------------------------
# Slash-command handlers (framework-free)
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


class TestHandlers:
    def test_allow_list(self):
        h = CommandHandlers(allowed_user_ids=frozenset({7}))
        assert h.is_allowed(7) and not h.is_allowed(8)
        assert h.denied().ephemeral is True
        assert CommandHandlers().is_allowed(12345)

    def test_help(self):
        r = _run(CommandHandlers().help())
        assert r.content == HELP_TEXT and r.ephemeral

    @pytest.mark.parametrize("kw,needle", [
        ({"bankroll": 0}, "bankroll"),
        ({"kelly": 1.5}, "kelly"),
        ({"date_from": "2026/09/04"}, "date_from"),
        ({"date_from": "2026-09-05", "date_to": "2026-09-04"}, "after"),
        ({"sectors": "nba,curling"}, "curling"),
    ])
    def test_scan_validation_replies_ephemerally_without_scanning(self, monkeypatch, kw, needle):
        called = []
        monkeypatch.setattr("evmax.web.app.run_dashboard_scan", lambda **k: called.append(k))
        r = _run(CommandHandlers().scan(**kw))
        assert r.ephemeral and needle in r.content
        assert called == []

    def test_scan_runs_dashboard_scan_suppressed_and_renders_table(self, monkeypatch):
        seen: dict = {}

        async def fake_scan(**kw):
            seen["kw"] = kw
            seen["suppressed"] = scan_feed_suppressed()
            return {"gaps": [_row()], "markets_fetched": 9, "markets_matched": 4,
                    "sectors": ["nfl"], "portfolio_results": [], "bankroll": 250.0,
                    "bankroll_source": "live:kalshi"}

        monkeypatch.setattr("evmax.web.app.run_dashboard_scan", fake_scan)
        r = _run(CommandHandlers().scan(sectors="NFL", date_from="2026-09-04", date_to="2026-09-05"))
        assert seen["suppressed"] is True
        assert seen["kw"]["sectors_str"] == "nfl"
        assert seen["kw"]["bankroll"] == 250.0 and seen["kw"]["kelly"] == 0.5
        assert seen["kw"]["fan_out_portfolios"] is True
        assert r.embeds[0]["title"] == "Scan Results — 1 plays (9 markets, 4 matched)"
        assert "Chiefs ML" in r.embeds[0]["description"]
        assert "bankroll live:kalshi" in r.embeds[0]["footer"]["text"]
        assert scan_feed_suppressed() is False

    def test_concurrent_scan_is_refused(self, monkeypatch):
        gate = asyncio.Event()

        async def slow_scan(**kw):
            await gate.wait()
            return {"gaps": [], "markets_fetched": 0, "markets_matched": 0, "sectors": [],
                    "portfolio_results": [], "bankroll": 250.0, "bankroll_source": "manual"}

        monkeypatch.setattr("evmax.web.app.run_dashboard_scan", slow_scan)

        async def _go():
            h = CommandHandlers()
            first = asyncio.create_task(h.scan())
            await asyncio.sleep(0)
            assert h.scan_running
            second = await h.scan()
            gate.set()
            return second, await first

        second, first = _run(_go())
        assert second.ephemeral and "already running" in second.content
        assert first.embeds[0]["description"] == "No +EV plays found."

    def test_plays_uses_open_bets(self, monkeypatch):
        bets = [{"market_id": "M1", "sector": "nba", "event_date": "2026-09-04", "kelly_fraction": 0.02,
                 "event_title": "A vs B", "display_label": "A ML", "kalshi_yes_price": 0.4,
                 "blended_true_prob": 0.5, "ev_pct": 0.05, "venue": "kalshi", "status": "upcoming"}]
        monkeypatch.setattr("evmax.web.app._open_bets", lambda: bets)
        r = _run(CommandHandlers().plays())
        assert r.embeds[0]["title"] == "Open Positions (1)"
        assert "A ML" in r.embeds[0]["description"]
        r = _run(CommandHandlers().plays(sector="NFL"))
        assert r.embeds[0]["title"] == "Open Positions (0)"

    def test_settled_uses_settled_bets_newest_first_with_summary(self, monkeypatch):
        settled = [
            {"placed": 0, "outcome": 1, "event_date": "2026-09-01", "kalshi_yes_price": 0.5,
             "kelly_fraction": 0.02, "bankroll_used": 250, "ev_pct": 0.04, "sector": "nba"},
            {"placed": 1, "outcome": 0, "event_date": "2026-09-02", "kalshi_yes_price": 0.5,
             "placed_stake": 10, "placed_price": 0.5, "ev_pct": 0.06, "sector": "nba"},
        ]
        monkeypatch.setattr("evmax.web.app._settled_bets", lambda: settled)
        r = _run(CommandHandlers().settled())
        d = r.embeds[0]["description"]
        assert d.index("2026-09-02") < d.index("2026-09-01")
        assert "2 bets · 1W / 1L" in r.embeds[0]["footer"]["text"]
        r = _run(CommandHandlers().settled(placed_only=True))
        assert "1 bets · 0W / 1L" in r.embeds[0]["footer"]["text"]
        assert "2026-09-01" not in r.embeds[0]["description"]

    def test_status_uses_heartbeat_without_notifying(self, monkeypatch):
        seen = {}

        def fake_hb(**kw):
            seen.update(kw)
            return {"ok": False, "issues": [{"severity": "warning", "detail": "scan stale"}], "notified": False}

        monkeypatch.setattr("evmax.agents.cleanup.heartbeat.run_heartbeat", fake_hb)
        r = _run(CommandHandlers().status(probe_pinnacle=False))
        assert seen == {"check_pinnacle_reachability": False, "notify": False}
        assert "1 issue(s)" in r.embeds[0]["title"]


# ---------------------------------------------------------------------------
# discord.py wiring (only when the optional extra is installed)
# ---------------------------------------------------------------------------

class TestGatewayBot:
    def test_build_bot_registers_every_command(self):
        discord = pytest.importorskip("discord")
        from evmax.discord_bot.bot import COMMAND_NAMES, build_bot
        from evmax.settings import Settings

        s = Settings(discord_bot_token="tok", discord_channel_id="1", discord_guild_id="22",
                     discord_allowed_user_ids="5")
        bot = build_bot(s)
        names = {c.name for c in bot.tree.get_commands()}
        assert names == set(COMMAND_NAMES)
        assert bot.guild_id == 22
        assert bot.handlers.is_allowed(5) and not bot.handlers.is_allowed(6)
        assert isinstance(bot, discord.Client)

    def test_build_bot_requires_token(self):
        pytest.importorskip("discord")
        from evmax.discord_bot.bot import build_bot
        from evmax.settings import Settings

        with pytest.raises(RuntimeError, match="DISCORD_BOT_TOKEN"):
            build_bot(Settings(discord_bot_token=""))

    def test_send_reply_batches_embeds_across_followups(self):
        discord = pytest.importorskip("discord")
        from evmax.discord_bot.bot import send_reply

        class _Followup:
            def __init__(self):
                self.calls = []

            async def send(self, **kw):
                self.calls.append(kw)

        class _Interaction:
            def __init__(self):
                self.followup = _Followup()

        big = {"title": "t", "description": "d" * 2900}
        inter = _Interaction()
        asyncio.run(send_reply(inter, Reply(content="hi", embeds=[big, big, big]), discord))
        assert len(inter.followup.calls) == 2
        assert inter.followup.calls[0]["content"] == "hi" and "content" not in inter.followup.calls[1]
        assert all(isinstance(e, discord.Embed) for e in inter.followup.calls[0]["embeds"])

        inter = _Interaction()
        asyncio.run(send_reply(inter, Reply(content="nope", ephemeral=True), discord))
        assert inter.followup.calls == [{"content": "nope", "ephemeral": True}]


class TestDefaultBankrollVenue:
    """``DISCORD_BANKROLL_VENUE``: the bot assumes a venue's live balance when a
    command is run without an explicit bankroll."""

    def test_scan_defaults_to_venue_when_no_bankroll_given(self, monkeypatch):
        seen: dict = {}

        async def fake_scan(**kw):
            seen.update(kw)
            return {"gaps": [], "markets_fetched": 0, "markets_matched": 0, "sectors": [],
                    "portfolio_results": [], "bankroll": 1664.84, "bankroll_source": "live:kalshi"}

        monkeypatch.setattr("evmax.web.app.run_dashboard_scan", fake_scan)
        h = CommandHandlers(default_bankroll_venue="Kalshi ")
        r = _run(h.scan())
        assert seen["bankroll_venue"] == "kalshi"
        assert "Bankroll $1,664.84" in r.embeds[0]["footer"]["text"]
        assert "bankroll live:kalshi" in r.embeds[0]["footer"]["text"]

    def test_scan_explicit_bankroll_means_manual(self, monkeypatch):
        seen: dict = {}

        async def fake_scan(**kw):
            seen.update(kw)
            return {"gaps": [], "markets_fetched": 0, "markets_matched": 0, "sectors": [],
                    "portfolio_results": [], "bankroll": 300.0, "bankroll_source": "manual"}

        monkeypatch.setattr("evmax.web.app.run_dashboard_scan", fake_scan)
        h = CommandHandlers(default_bankroll_venue="kalshi")
        _run(h.scan(bankroll=300))
        assert seen["bankroll_venue"] is None and seen["bankroll"] == 300.0
        # An explicit venue still wins over the default.
        _run(h.scan(bankroll_venue="both"))
        assert seen["bankroll_venue"] == "both"

    def test_scan_without_default_keeps_manual_250(self, monkeypatch):
        seen: dict = {}

        async def fake_scan(**kw):
            seen.update(kw)
            return {"gaps": [], "markets_fetched": 0, "markets_matched": 0, "sectors": [],
                    "portfolio_results": [], "bankroll": 250.0, "bankroll_source": "manual"}

        monkeypatch.setattr("evmax.web.app.run_dashboard_scan", fake_scan)
        _run(CommandHandlers().scan())
        assert seen["bankroll_venue"] is None and seen["bankroll"] == 250.0

    def test_plays_sizes_stake_against_live_balance(self, monkeypatch):
        from evmax.clients.balances import BankrollPlan

        calls: list = []

        async def fake_plan(bankroll, selection):
            calls.append((bankroll, selection))
            return BankrollPlan(1000.0, "live:kalshi", ["kalshi"], {"kalshi": 900.0})

        monkeypatch.setattr("evmax.clients.balances.resolve_bankroll_plan", fake_plan)
        bets = [{"market_id": "M1", "sector": "nba", "event_date": "2026-09-04", "kelly_fraction": 0.02,
                 "event_title": "A vs B", "display_label": "A ML", "kalshi_yes_price": 0.4,
                 "blended_true_prob": 0.5, "ev_pct": 0.05, "venue": "kalshi", "status": "upcoming"}]
        monkeypatch.setattr("evmax.web.app._open_bets", lambda: bets)
        r = _run(CommandHandlers(default_bankroll_venue="kalshi").plays())
        assert calls == [(250.0, "kalshi")]
        footer = r.embeds[0]["footer"]["text"]
        assert "Bankroll $1,000.00" in footer and "bankroll live:kalshi" in footer
        assert "$20.00" in r.embeds[0]["description"]  # 0.02 × 1000 at kelly 0.5

    def test_plays_explicit_bankroll_skips_live_fetch(self, monkeypatch):
        async def boom(bankroll, selection):
            raise AssertionError("must not fetch a balance")

        monkeypatch.setattr("evmax.clients.balances.resolve_bankroll_plan", boom)
        monkeypatch.setattr("evmax.web.app._open_bets", lambda: [])
        r = _run(CommandHandlers(default_bankroll_venue="kalshi").plays(bankroll=500))
        assert "Bankroll $500.00" in r.embeds[0]["footer"]["text"]
        assert "live:" not in r.embeds[0]["footer"]["text"]

    def test_plays_fail_soft_when_balance_unavailable(self, monkeypatch):
        from evmax.clients.balances import BankrollPlan

        async def fallback(bankroll, selection):
            return BankrollPlan(bankroll, "manual_fallback", ["kalshi"], {})

        monkeypatch.setattr("evmax.clients.balances.resolve_bankroll_plan", fallback)
        monkeypatch.setattr("evmax.web.app._open_bets", lambda: [])
        r = _run(CommandHandlers(default_bankroll_venue="kalshi").plays())
        footer = r.embeds[0]["footer"]["text"]
        assert "Bankroll $250.00" in footer and "manual_fallback" in footer

    def test_settings_field_and_bot_wiring(self, monkeypatch):
        from evmax.settings import get_settings
        s = get_settings()
        assert hasattr(s, "discord_bankroll_venue")
        monkeypatch.setattr(s, "discord_bankroll_venue", "both")
        # build_bot needs discord.py; only check the handler construction path.
        h = CommandHandlers(default_bankroll_venue=s.discord_bankroll_venue)
        assert h._default_bankroll_venue == "both"

    def test_option_descriptions_fit_discord_limit(self):
        import re
        src = open("evmax/discord_bot/bot.py", encoding="utf-8").read()
        for m in re.finditer(r'^\s+\w+="([^"]+)",?$', src, re.M):
            assert len(m.group(1)) <= 100, m.group(1)
