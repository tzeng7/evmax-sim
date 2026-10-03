"""Kalshi order-history parsing, fill-rate and fee-multiplier summaries."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from evmax.agents.cleanup.maker_orders import (
    OrderAttempt,
    fill_summary,
    hours_to_fill,
    implied_fee_multipliers,
    load_attempts,
    parse_order,
    parse_orders,
    series_of,
    upsert_attempts,
)


def _raw(**kw) -> dict:
    base = {
        "order_id": "o1", "ticker": "KXNFLSPREAD-26OCT05DENKC-DEN6",
        "status": "executed", "outcome_side": "yes",
        "yes_price_dollars": "0.2600", "no_price_dollars": "0.7400",
        "initial_count_fp": "100.00", "fill_count_fp": "100.00",
        "remaining_count_fp": "0.00",
        "maker_fill_cost_dollars": "26.0000", "taker_fill_cost_dollars": "0.0000",
        "maker_fees_dollars": "0.3400", "taker_fees_dollars": "0.0000",
        "created_time": "2026-10-04T12:00:00Z", "last_update_time": "2026-10-04T15:30:00Z",
    }
    base.update(kw)
    return base


class TestParse:
    def test_yes_order(self):
        a = parse_order(_raw())
        assert a.series == "KXNFLSPREAD" and a.outcome_side == "yes"
        assert a.limit_price == pytest.approx(0.26)
        assert a.initial_count == 100 and a.fill_count == 100 and a.kind == "maker"

    def test_no_order_uses_no_price(self):
        a = parse_order(_raw(outcome_side="no"))
        assert a.limit_price == pytest.approx(0.74)

    def test_legacy_integer_cent_fields(self):
        raw = _raw(outcome_side="yes")
        for k in ("yes_price_dollars", "no_price_dollars", "initial_count_fp", "fill_count_fp"):
            raw.pop(k)
        raw.update(yes_price=26, initial_count=10, fill_count=0)
        a = parse_order(raw)
        assert a.limit_price == pytest.approx(0.26) and a.initial_count == 10 and not a.filled

    @pytest.mark.parametrize("bad", [
        {"order_id": "x"},                                  # no ticker/status
        {"outcome_side": "maybe"},                          # unknown side
        {"yes_price_dollars": "oops"},                      # unreadable price
        {"yes_price_dollars": "1.0000"},                    # degenerate price
    ])
    def test_unreadable_orders_are_skipped(self, bad):
        raw = _raw()
        raw.update(bad)
        if "ticker" not in bad and set(bad) == {"order_id"}:
            raw = {"order_id": "x"}
        assert parse_order(raw) is None

    def test_parse_orders_counts_skips(self):
        got, skipped = parse_orders([_raw(), {"order_id": "z"}, _raw(order_id="o2")])
        assert len(got) == 2 and skipped == 1

    def test_kind_classification(self):
        assert parse_order(_raw(maker_fill_cost_dollars="0", taker_fill_cost_dollars="26")).kind == "taker"
        assert parse_order(_raw(taker_fill_cost_dollars="5")).kind == "mixed"
        assert parse_order(_raw(fill_count_fp="0", maker_fill_cost_dollars="0")).kind == "none"

    def test_series_of(self):
        assert series_of("kxnflgame-26OCT05-X") == "KXNFLGAME"


class TestFillSummary:
    def _a(self, oid, status, fill, init=10.0, **kw):
        raw = _raw(order_id=oid, status=status, fill_count_fp=f"{fill:.2f}",
                   initial_count_fp=f"{init:.2f}",
                   maker_fill_cost_dollars=str(fill * 0.26), **kw)
        return parse_order(raw)

    def test_resting_orders_excluded_from_denominator(self):
        s = fill_summary([
            self._a("a", "executed", 10), self._a("b", "canceled", 0),
            self._a("c", "canceled", 4), self._a("d", "resting", 0),
        ])["KXNFLSPREAD"]
        assert s["orders"] == 4 and s["resting"] == 1 and s["resolved_maker_orders"] == 3
        assert s["filled_orders"] == 2
        assert s["fill_rate_orders"] == pytest.approx(2 / 3)
        assert s["fill_rate_contracts"] == pytest.approx((10 + 0 + 4) / 30)

    def test_taker_crossing_not_counted_as_maker_fill(self):
        crossed = parse_order(_raw(order_id="t", maker_fill_cost_dollars="0",
                                   taker_fill_cost_dollars="26"))
        s = fill_summary([crossed, self._a("a", "canceled", 0)])["KXNFLSPREAD"]
        assert s["crossed_as_taker"] == 1 and s["resolved_maker_orders"] == 1
        assert s["fill_rate_orders"] == 0.0

    def test_no_data_rates_are_none(self):
        s = fill_summary([self._a("d", "resting", 0)])["KXNFLSPREAD"]
        assert s["fill_rate_orders"] is None and s["median_hours_to_fill"] is None

    def test_hours_to_fill(self):
        assert hours_to_fill(parse_order(_raw())) == pytest.approx(3.5)
        assert hours_to_fill(parse_order(_raw(status="canceled"))) is None
        assert hours_to_fill(parse_order(_raw(last_update_time="2026-10-04T11:00:00Z"))) is None


class TestImpliedFeeMultiplier:
    @staticmethod
    def _order(kind, mult, price=0.26, n=1000):
        rate = 0.0175 if kind == "maker" else 0.07
        fee = round(rate * mult * n * price * (1 - price), 4)
        cost = price * n
        return parse_order(_raw(
            order_id=f"{kind}{mult}", initial_count_fp=str(n), fill_count_fp=str(n),
            maker_fill_cost_dollars=str(cost if kind == "maker" else 0),
            taker_fill_cost_dollars=str(cost if kind == "taker" else 0),
            maker_fees_dollars=str(fee if kind == "maker" else 0),
            taker_fees_dollars=str(fee if kind == "taker" else 0)))

    @pytest.mark.parametrize("kind", ["maker", "taker"])
    @pytest.mark.parametrize("mult", [1.0, 0.5])
    def test_recovers_the_billed_multiplier(self, kind, mult):
        out = implied_fee_multipliers([self._order(kind, mult)])
        assert out[f"KXNFLSPREAD:{kind}"]["implied_multiplier"] == pytest.approx(mult, rel=1e-3)

    def test_mixed_and_unfilled_orders_excluded(self):
        mixed = parse_order(_raw(order_id="m", taker_fill_cost_dollars="5"))
        none = parse_order(_raw(order_id="n", fill_count_fp="0", maker_fill_cost_dollars="0"))
        assert implied_fee_multipliers([mixed, none]) == {}


class TestStore:
    def test_upsert_refreshes_changed_order(self):
        conn = sqlite3.connect(":memory:")
        resting = parse_order(_raw(status="resting", fill_count_fp="0",
                                   maker_fill_cost_dollars="0", maker_fees_dollars="0"))
        assert upsert_attempts(conn, [resting]) == 1
        assert load_attempts(conn)[0].status == "resting"
        upsert_attempts(conn, [parse_order(_raw())])          # same order_id, now executed
        rows = load_attempts(conn)
        assert len(rows) == 1 and rows[0].status == "executed" and rows[0].fill_count == 100

    def test_roundtrip_preserves_fields(self):
        conn = sqlite3.connect(":memory:")
        a = parse_order(_raw())
        upsert_attempts(conn, [a])
        assert load_attempts(conn) == [a]


class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


class TestClientGetOrders:
    @pytest.mark.asyncio
    async def test_pages_until_cursor_exhausted(self):
        from evmax.clients.kalshi import KalshiClient
        pages = [{"orders": [{"order_id": "1"}], "cursor": "c1"},
                 {"orders": [{"order_id": "2"}], "cursor": ""}]
        c = KalshiClient()
        get = AsyncMock(side_effect=[_Resp(p) for p in pages])
        c._client = type("H", (), {"get": get})()
        with patch.object(c, "_sign_request", return_value={"KALSHI-ACCESS-KEY": "k"}):
            got = await c.get_orders(status="executed", min_ts=100)
        assert [o["order_id"] for o in got] == ["1", "2"]
        assert get.await_args_list[1].kwargs["params"]["cursor"] == "c1"
        assert get.await_args_list[0].kwargs["params"]["status"] == "executed"
        assert get.await_args_list[0].kwargs["params"]["min_ts"] == 100

    @pytest.mark.asyncio
    async def test_no_credentials_returns_none_without_request(self):
        from evmax.clients.kalshi import KalshiClient
        c = KalshiClient()
        get = AsyncMock()
        c._client = type("H", (), {"get": get})()
        with patch.object(c, "_sign_request", return_value={}):
            assert await c.get_orders() is None
        get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_midway_returns_none_not_partial(self):
        from evmax.clients.kalshi import KalshiClient
        c = KalshiClient()
        get = AsyncMock(side_effect=[_Resp({"orders": [{"order_id": "1"}], "cursor": "c1"}),
                                     RuntimeError("boom")])
        c._client = type("H", (), {"get": get})()
        with patch.object(c, "_sign_request", return_value={"KALSHI-ACCESS-KEY": "k"}):
            assert await c.get_orders() is None


class TestCli:
    @staticmethod
    def _conn(tmp_path):
        path = tmp_path / "pred.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE ev_predictions (market_id TEXT, venue TEXT)")
        conn.commit()
        conn.close()
        return path

    def _invoke(self, path, args, orders=None):
        from typer.testing import CliRunner
        from evmax.cli.commands.cleanup import app

        class _Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get_orders(self, **kw): return orders

        with patch("evmax.agents.cleanup.db.get_connection", side_effect=lambda: sqlite3.connect(path)), \
             patch("evmax.clients.kalshi.KalshiClient", _Client):
            return CliRunner().invoke(app, ["maker-orders", *args])

    def test_report_with_no_orders(self, tmp_path):
        res = self._invoke(self._conn(tmp_path), ["--no-sync"])
        assert res.exit_code == 0 and "No stored maker orders" in res.output

    def test_sync_without_credentials_exits_nonzero_and_stores_nothing(self, tmp_path):
        path = self._conn(tmp_path)
        res = self._invoke(path, ["--sync"], orders=None)
        assert res.exit_code == 1 and "Could not fetch" in res.output
        assert load_attempts(sqlite3.connect(path)) == []

    def test_sync_keeps_only_scanned_tickers_then_reports(self, tmp_path):
        path = self._conn(tmp_path)
        with sqlite3.connect(path) as c:
            c.execute("INSERT INTO ev_predictions VALUES ('kalshi:KXNFLSPREAD-26OCT05DENKC-DEN6:no','kalshi')")
        orders = [_raw(order_id="mine"),
                  _raw(order_id="other", ticker="KXNBAGAME-26OCT05-X")]
        res = self._invoke(path, ["--sync"], orders=orders)
        assert res.exit_code == 0, res.output
        assert [a.order_id for a in load_attempts(sqlite3.connect(path))] == ["mine"]
        assert "KXNFLSPREAD" in res.output and "Implied" in res.output

    def test_all_tickers_keeps_everything(self, tmp_path):
        path = self._conn(tmp_path)
        orders = [_raw(order_id="mine"), _raw(order_id="other", ticker="KXNBAGAME-26OCT05-X")]
        res = self._invoke(path, ["--sync", "--all-tickers"], orders=orders)
        assert res.exit_code == 0
        assert {a.order_id for a in load_attempts(sqlite3.connect(path))} == {"mine", "other"}
