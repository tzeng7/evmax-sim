"""Tests for recompute_at_price / reblend_with_fresh_sharp — the pure helpers
that let `evmax agents pick` gate and size a bet at CURRENT prices instead of
the stale scan-time snapshot.

This is the entry-timing fix: the fattest scan-time edges are the most likely to
have reverted by the time you place, so a bet only counts as live if its edge
survives to the live price. recompute_at_price handles the Kalshi side (the
ask); reblend_with_fresh_sharp handles the Pinnacle side (blended_true_prob) —
without it, EV/Kelly at pick time were computed from a sharp line frozen at
scan time even though Pinnacle drifts meaningfully (median ~0.8pp / p90 ~4pp
on resolved bets) between an early scan and a T-60 pick window, flipping the
bet/no-bet decision on ~19% of rows in a backtest over resolved live bets.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evmax.cli.commands.agents import (
    fetch_polymarket_live_asks,
    polymarket_live_ask,
    recompute_at_price,
    reblend_with_fresh_sharp,
)

_COMMON = dict(base_kelly=0.5, max_kelly=0.05, bankroll=1000.0, min_ev=0.02, min_prob=0.15)


def test_live_edge_survives_is_live_with_stake():
    # Fair 0.60 vs a 0.50 ask → ~20% edge, well above threshold.
    rc = recompute_at_price(blended_prob=0.60, price=0.50, **_COMMON)
    assert rc["is_live"] is True
    assert rc["ev"] > 0.02
    assert rc["stake"] > 0
    assert 0 < rc["kelly_fraction"] <= 0.05  # capped at max_kelly


def test_edge_eroded_to_zero_is_not_live():
    # Same fair 0.60 but the ask has risen to 0.60 → no edge → drop the bet.
    rc = recompute_at_price(blended_prob=0.60, price=0.60, **_COMMON)
    assert rc["is_live"] is False
    assert rc["stake"] == 0.0
    assert rc["kelly_fraction"] == 0.0


def test_price_past_fair_is_not_live():
    # Ask above fair → negative EV → never live.
    rc = recompute_at_price(blended_prob=0.55, price=0.70, **_COMMON)
    assert rc["is_live"] is False
    assert rc["stake"] == 0.0


def test_none_price_is_not_live():
    rc = recompute_at_price(blended_prob=0.60, price=None, **_COMMON)
    assert rc["is_live"] is False
    assert rc["ev"] is None
    assert rc["stake"] == 0.0


@pytest.mark.parametrize("price", [0.0, 1.0, 0.99, 1.5, -0.1])
def test_degenerate_prices_are_not_live(price):
    # Settled (0/1), empty-book (>=0.99), or out-of-range prices yield no bet.
    rc = recompute_at_price(blended_prob=0.60, price=price, **_COMMON)
    assert rc["is_live"] is False
    assert rc["stake"] == 0.0


def test_stake_scales_with_bankroll():
    small = recompute_at_price(blended_prob=0.60, price=0.50, **{**_COMMON, "bankroll": 500.0})
    big = recompute_at_price(blended_prob=0.60, price=0.50, **{**_COMMON, "bankroll": 2000.0})
    assert big["kelly_fraction"] == pytest.approx(small["kelly_fraction"])
    assert big["stake"] == pytest.approx(4 * small["stake"])


def test_low_prob_longshot_gated_out():
    # blended_prob below min_prob is excluded even if nominal EV is positive.
    rc = recompute_at_price(blended_prob=0.10, price=0.05, **_COMMON)
    assert rc["is_live"] is False


class TestReblendWithFreshSharp:
    """reblend_with_fresh_sharp — re-derives blended_true_prob from a fresh
    Pinnacle line using the same linear formula EnsembleModelAgent uses
    (blended = sharp_weight*sharp + (1-sharp_weight)*model), holding the
    model component fixed."""

    def test_sharp_moved_up_shifts_blend_by_weight(self):
        # Pinnacle moved +10pp (0.40 -> 0.50) with sharp_weight=0.80 -> blend
        # should shift by 0.80*0.10 = +0.08pp.
        updated = reblend_with_fresh_sharp(
            blended_prob=0.45, scan_sharp_prob=0.40, fresh_sharp_prob=0.50, sharp_weight=0.80,
        )
        assert updated == pytest.approx(0.45 + 0.08)

    def test_sharp_moved_down_shifts_blend_down(self):
        updated = reblend_with_fresh_sharp(
            blended_prob=0.45, scan_sharp_prob=0.40, fresh_sharp_prob=0.30, sharp_weight=0.80,
        )
        assert updated == pytest.approx(0.45 - 0.08)

    def test_sharp_weight_zero_no_change(self):
        # A fully model-driven blend shouldn't move even if Pinnacle drifted.
        updated = reblend_with_fresh_sharp(
            blended_prob=0.45, scan_sharp_prob=0.40, fresh_sharp_prob=0.70, sharp_weight=0.0,
        )
        assert updated == pytest.approx(0.45)

    def test_sharp_weight_one_is_pure_passthrough(self):
        # sharp_weight=1.0 means blended == scan_sharp_prob at scan time, so
        # re-blending should land exactly on the fresh sharp prob.
        updated = reblend_with_fresh_sharp(
            blended_prob=0.40, scan_sharp_prob=0.40, fresh_sharp_prob=0.63, sharp_weight=1.0,
        )
        assert updated == pytest.approx(0.63)

    @pytest.mark.parametrize("missing_field", ["scan_sharp_prob", "fresh_sharp_prob", "sharp_weight"])
    def test_missing_input_falls_back_to_frozen_blend(self, missing_field):
        kwargs = dict(blended_prob=0.45, scan_sharp_prob=0.40, fresh_sharp_prob=0.55, sharp_weight=0.75)
        kwargs[missing_field] = None
        assert reblend_with_fresh_sharp(**kwargs) == 0.45

    def test_clamped_to_valid_probability_range(self):
        # A large upward drift on an already-high blend must not exceed 1.
        high = reblend_with_fresh_sharp(
            blended_prob=0.97, scan_sharp_prob=0.50, fresh_sharp_prob=0.99, sharp_weight=1.0,
        )
        assert high <= 0.999
        # A large downward drift on an already-low blend must not go below 0.
        low = reblend_with_fresh_sharp(
            blended_prob=0.03, scan_sharp_prob=0.50, fresh_sharp_prob=0.01, sharp_weight=1.0,
        )
        assert low >= 0.001


def _mkt(mid: str, yes: float | None, no: float | None):
    """Minimal PredictionMarket stand-in for polymarket_live_ask lookups."""
    return SimpleNamespace(id=mid, yes_price=yes, no_price=no)


class TestPolymarketLiveAsk:
    """polymarket_live_ask — resolve a stored PolyUS row id to its live ask in a
    fresh {market.id: PredictionMarket} map. YES rows read yes_price; NO rows
    (':no' suffix) read the market's own no_price (exact, not a 1-yes fudge)."""

    def test_yes_moneyline_reads_yes_price(self):
        mid = "polymarket_us:aec-wnba-min-conn-2026-07-08:conn"
        assert polymarket_live_ask(mid, {mid: _mkt(mid, 0.62, 0.40)}) == pytest.approx(0.62)

    def test_no_spread_reads_no_price_directly(self):
        # The ':no' row shares the market's base id; NO ask = no_price exactly.
        base = "polymarket_us:asc-wnba-gsv-conn-2026-07-10-pos-11pt5"
        assert polymarket_live_ask(base + ":no", {base: _mkt(base, 0.55, 0.47)}) == pytest.approx(0.47)

    def test_yes_spread_reads_yes_price(self):
        base = "polymarket_us:asc-wnba-gsv-conn-2026-07-10-pos-11pt5"
        assert polymarket_live_ask(base, {base: _mkt(base, 0.55, 0.47)}) == pytest.approx(0.55)

    def test_unlisted_market_returns_none(self):
        # Settled / closed markets drop out of the fresh map → None → scan-price fallback.
        assert polymarket_live_ask("polymarket_us:gone:x", {}) is None

    def test_unquoted_side_returns_none(self):
        mid = "polymarket_us:m:x"
        assert polymarket_live_ask(mid, {mid: _mkt(mid, None, 0.5)}) is None

    def test_price_clamped_to_unit_interval(self):
        mid = "polymarket_us:m:x"
        assert polymarket_live_ask(mid, {mid: _mkt(mid, 1.4, 0.5)}) == 1.0


class TestFetchPolymarketLiveAsks:
    """fetch_polymarket_live_asks — group rows by sector, re-fetch each sector's
    live markets (cache-bypassed), map every row id to its live ask, and isolate
    a failed sector so its rows fall back to None rather than sinking the batch."""

    def _patch_client(self, monkeypatch, markets_by_sector, calls=None):
        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get_markets(self, sector, fresh=False):
                if calls is not None:
                    calls.append((sector, fresh))
                result = markets_by_sector[sector]
                if isinstance(result, Exception):
                    raise result
                return result

        monkeypatch.setattr(
            "evmax.clients.polymarket_us.PolymarketUSClient", _FakeClient
        )

    @pytest.mark.asyncio
    async def test_maps_yes_and_no_rows_across_sectors(self, monkeypatch):
        wnba_base = "polymarket_us:asc-wnba-gsv-conn-pos-11pt5"
        atp_id = "polymarket_us:aec-atp-zsopir-damdzu:zsopir"
        calls: list = []
        self._patch_client(
            monkeypatch,
            {
                "wnba": [_mkt(wnba_base, 0.55, 0.47)],
                "tennis": [_mkt(atp_id, 0.71, 0.31)],
            },
            calls,
        )
        rows = [
            {"market_id": wnba_base + ":no", "sector": "wnba"},
            {"market_id": atp_id, "sector": "tennis"},
        ]
        out = await fetch_polymarket_live_asks(rows)
        assert out[wnba_base + ":no"] == pytest.approx(0.47)  # NO → no_price
        assert out[atp_id] == pytest.approx(0.71)             # YES → yes_price
        # Cache is bypassed on the refresh path.
        assert all(fresh is True for _, fresh in calls)

    @pytest.mark.asyncio
    async def test_failed_sector_isolated_to_none(self, monkeypatch):
        good = "polymarket_us:good:x"
        bad = "polymarket_us:bad:y"
        self._patch_client(
            monkeypatch,
            {
                "tennis": [_mkt(good, 0.60, 0.42)],
                "wnba": RuntimeError("gateway 503"),
            },
        )
        rows = [
            {"market_id": good, "sector": "tennis"},
            {"market_id": bad, "sector": "wnba"},
        ]
        out = await fetch_polymarket_live_asks(rows)
        assert out[good] == pytest.approx(0.60)
        assert out[bad] is None  # failed sector's rows fall back, batch survives

    @pytest.mark.asyncio
    async def test_empty_rows_no_fetch(self, monkeypatch):
        calls: list = []
        self._patch_client(monkeypatch, {}, calls)
        assert await fetch_polymarket_live_asks([]) == {}
        assert calls == []  # no sectors → no client call
