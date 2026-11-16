"""Tests for the Polymarket US maker-fill replay (polyus_maker_replay.py).

Pure-logic tests on synthetic price history: bucketing, side orientation,
the no-look-ahead entry rule, touch vs thru fills, in-play exclusion, the
limit price, and the per-candidate maker statistic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evmax.agents.cleanup import polyus_maker_replay as mr

TIP = datetime(2026, 11, 16, 1, 0, tzinfo=timezone.utc)
T0 = TIP - timedelta(hours=6)          # row logged 6h before tip
B = mr.BUCKET_S


def ts(dt: datetime) -> int:
    return int(dt.timestamp())


def meta(tick=0.01, tip=TIP, sides=(("chi", "Bulls", True), ("cle", "Cavaliers", False))):
    return mr.MarketMeta("slug", tip, tick, tuple(sides))


def flat_path(long_ask, long_bid, start=T0 - timedelta(hours=1), end=TIP + timedelta(hours=1)):
    """One bucket every 5 min with a constant long-side quote."""
    out, t = [], ts(start) - ts(start) % B
    while t < ts(end):
        out.append([t, long_ask, 1 - long_bid, long_ask, 1 - long_bid])
        t += B
    return out


def set_bucket(buckets, when, *, long_ask=None, long_bid=None, long_min=None, short_min=None):
    key = ts(when) - ts(when) % B
    for b in buckets:
        if b[0] == key:
            if long_ask is not None:
                b[1] = long_ask
                b[3] = min(b[3], long_ask)
            if long_bid is not None:
                b[2] = 1 - long_bid
                b[4] = min(b[4], 1 - long_bid)
            if long_min is not None:
                b[3] = long_min
            if short_min is not None:
                b[4] = short_min
            return
    raise AssertionError("bucket not found")


def row(market_id="polymarket_us:slug:chi", logged=T0, **kw):
    base = {"market_id": market_id, "logged_at": logged.strftime("%Y-%m-%d %H:%M:%S"),
            "minutes_to_tipoff": 360, "sector": "nba", "market_type": "moneyline",
            "game": "g1", "outcome": 1, "blended_true_prob": 0.35, "entry_price": None}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# bucketize / day_windows
# ---------------------------------------------------------------------------


def test_bucketize_keeps_last_and_minimum_per_bucket():
    t = ts(T0) - ts(T0) % B
    hist = [
        {"timestamp": t + 10, "longPrice": 0.31, "shortPrice": 0.71},
        {"timestamp": t + 70, "longPrice": 0.29, "shortPrice": 0.72},   # dip
        {"timestamp": t + 200, "longPrice": 0.30, "shortPrice": 0.70},
        {"timestamp": t + B + 5, "longPrice": 0.33, "shortPrice": 0.69},
        {"timestamp": t + 20, "longPrice": None, "shortPrice": 0.5},     # dropped
        {"timestamp": t + 30, "longPrice": 1.4, "shortPrice": 0.5},      # dropped
    ]
    first, second = mr.bucketize(hist)
    assert first == [t, 0.30, 0.70, 0.29, 0.70]
    assert second == [t + B, 0.33, 0.69, 0.33, 0.69]


def test_day_windows_cover_both_ends_in_utc():
    a = datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc)
    assert mr.day_windows(a, a + timedelta(hours=30)) == ["2026-11-13", "2026-11-14", "2026-11-15"]
    assert mr.day_windows(a, a) == ["2026-11-13"]
    assert mr.day_windows(a, a - timedelta(days=1)) == []


# ---------------------------------------------------------------------------
# metadata + side orientation
# ---------------------------------------------------------------------------


def test_parse_market_meta_reads_tip_tick_and_sides():
    m = mr.parse_market_meta("s", {
        "gameStartTime": "2026-09-13T17:00:00Z", "orderPriceMinTickSize": 0.005,
        "marketSides": [
            {"description": "Jets", "long": True, "team": {"abbreviation": "nyj"}},
            {"description": "Titans", "long": False, "team": {"abbreviation": "ten"}},
        ],
    })
    assert m.tip == datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
    assert m.tick == pytest.approx(0.005)
    assert m.sides == (("nyj", "Jets", True), ("ten", "Titans", False))


def test_parse_market_meta_bad_values_fall_back():
    m = mr.parse_market_meta("s", {"gameStartTime": "soon", "orderPriceMinTickSize": 3})
    assert m.tip is None and m.tick == mr.DEFAULT_TICK and m.sides == ()


def test_parse_market_id():
    assert mr.parse_market_id("polymarket_us:a-b:chi") == ("a-b", "chi", False)
    assert mr.parse_market_id("polymarket_us:a-b-total-49pt5:no") == ("a-b-total-49pt5", None, True)
    assert mr.parse_market_id("polymarket_us:a-b:chi:no") == ("a-b", "chi", True)
    assert mr.parse_market_id("kalshi:KXNBAGAME-X") is None


def test_our_side_is_long_per_team_spread_total_and_no():
    m = meta()
    assert mr.our_side_is_long("polymarket_us:slug:chi", m) is True
    assert mr.our_side_is_long("polymarket_us:slug:cle", m) is False
    assert mr.our_side_is_long("polymarket_us:slug:chi:no", m) is False   # NO of long = short
    assert mr.our_side_is_long("polymarket_us:slug:zzz", m) is None       # unknown team

    spread = meta(sides=((None, "-3.50", True), (None, "+3.50", False)))
    assert mr.our_side_is_long("polymarket_us:slug", spread) is True
    assert mr.our_side_is_long("polymarket_us:slug:no", spread) is False

    over_short = meta(sides=((None, "Over", False), (None, "Under", True)))
    assert mr.our_side_is_long("polymarket_us:slug", over_short) is False  # YES=Over is short
    assert mr.our_side_is_long("polymarket_us:slug:no", over_short) is True


def test_side_quote_long_and_short_are_complements():
    b = [0, 0.30, 0.7025, 0.29, 0.70]          # long ask .30, long bid .2975
    assert mr.side_quote(b, True) == (0.30, pytest.approx(0.2975), 0.29)
    ask, bid, low = mr.side_quote(b, False)
    assert ask == pytest.approx(0.7025) and bid == pytest.approx(0.70) and low == 0.70


# ---------------------------------------------------------------------------
# replay_row
# ---------------------------------------------------------------------------


def test_unfilled_when_the_ask_never_reaches_the_limit():
    res, why = mr.replay_row(row(), meta(), flat_path(0.32, 0.30))
    assert why == "ok"
    assert res["ask"] == pytest.approx(0.32) and res["bid"] == pytest.approx(0.30)
    assert res["limit"] == pytest.approx(0.31)            # bid + one tick
    assert res["touch"] is False and res["thru"] is False
    assert res["close_ask"] == pytest.approx(0.32)
    assert res["close_mid"] == pytest.approx(0.31)
    assert res["lead_h"] == pytest.approx(6.0)


def test_touch_is_not_thru():
    path = flat_path(0.32, 0.30)
    set_bucket(path, T0 + timedelta(hours=2), long_min=0.31)       # ask touches the limit
    res, _ = mr.replay_row(row(), meta(), path)
    assert res["touch"] is True and res["thru"] is False
    set_bucket(path, T0 + timedelta(hours=3), long_min=0.305)      # trades through it
    res, _ = mr.replay_row(row(), meta(), path)
    assert res["touch"] is True and res["thru"] is True


def test_limit_never_crosses_a_one_tick_spread():
    res, _ = mr.replay_row(row(), meta(tick=0.01), flat_path(0.31, 0.30))
    assert res["limit"] == pytest.approx(0.30)            # join the bid, never the ask
    res, _ = mr.replay_row(row(), meta(tick=0.005), flat_path(0.31, 0.30))
    assert res["limit"] == pytest.approx(0.305)


def test_entry_uses_the_last_completed_bucket_never_a_later_one():
    path = flat_path(0.32, 0.30)
    # The quote jumps in the bucket that CONTAINS logged_at: entry must not see it.
    set_bucket(path, T0, long_ask=0.40, long_bid=0.38)
    res, _ = mr.replay_row(row(), meta(), path)
    assert res["ask"] == pytest.approx(0.32) and res["limit"] == pytest.approx(0.31)


def test_a_dip_before_entry_or_after_tip_is_not_a_fill():
    path = flat_path(0.32, 0.30)
    set_bucket(path, T0 - timedelta(minutes=20), long_min=0.20)    # before the order rests
    set_bucket(path, TIP + timedelta(minutes=10), long_min=0.05)   # in play
    res, _ = mr.replay_row(row(), meta(), path)
    assert res["touch"] is False


def test_a_dip_in_the_last_pre_tip_bucket_is_a_fill():
    # TIP sits on a bucket boundary, so the bucket ending at TIP is fully pre-tip.
    path = flat_path(0.32, 0.30)
    set_bucket(path, TIP - timedelta(minutes=2), long_min=0.05)
    res, _ = mr.replay_row(row(), meta(), path)
    assert res["touch"] is True and res["thru"] is True


def test_bucket_straddling_tip_is_excluded_from_fill_and_close():
    tip = TIP + timedelta(minutes=2)                       # tip inside a bucket
    path = flat_path(0.32, 0.30)
    set_bucket(path, TIP + timedelta(minutes=1), long_ask=0.10, long_bid=0.08, long_min=0.10)
    res, _ = mr.replay_row(row(), meta(tip=tip), path)
    assert res["touch"] is False
    assert res["close_ask"] == pytest.approx(0.32)         # last FULLY pre-tip bucket


def test_short_side_row_is_tested_on_the_short_ask():
    path = flat_path(0.32, 0.30)                           # short: ask .70, bid .68
    res, _ = mr.replay_row(row("polymarket_us:slug:cle"), meta(), path)
    assert res["ask"] == pytest.approx(0.70) and res["bid"] == pytest.approx(0.68)
    assert res["limit"] == pytest.approx(0.69)
    assert res["touch"] is False
    # The LONG ask collapsing does not fill a short-side bid...
    set_bucket(path, T0 + timedelta(hours=1), long_min=0.05)
    assert mr.replay_row(row("polymarket_us:slug:cle"), meta(), path)[0]["touch"] is False
    # ...the short ask dropping does.
    set_bucket(path, T0 + timedelta(hours=2), short_min=0.69)
    assert mr.replay_row(row("polymarket_us:slug:cle"), meta(), path)[0]["touch"] is True


def test_logged_ask_diff_is_the_orientation_check():
    res, _ = mr.replay_row(row(entry_price=0.33), meta(), flat_path(0.32, 0.30))
    assert res["logged_ask_diff"] == pytest.approx(-0.01)


@pytest.mark.parametrize("kwargs, path, reason", [
    (dict(market_id="polymarket_us:slug:zzz"), None, "side_unknown"),
    (dict(logged=TIP - timedelta(minutes=5)), None, "logged_at_or_after_tip"),
    (dict(logged=TIP + timedelta(hours=1)), None, "logged_at_or_after_tip"),
    (dict(), [], "no_entry_quote"),
    (dict(), flat_path(0.30, 0.30), "bad_entry_book"),                     # zero spread
    (dict(), flat_path(0.32, 0.30, end=T0), "no_pre_tip_path"),
    # entry quote older than 30 minutes
    (dict(), flat_path(0.32, 0.30, start=T0 - timedelta(hours=3),
                       end=T0 - timedelta(hours=1)), "no_entry_quote"),
])
def test_skip_reasons(kwargs, path, reason):
    res, why = mr.replay_row(row(**kwargs), meta(), flat_path(0.32, 0.30) if path is None else path)
    assert res is None and why == reason


def test_tip_falls_back_to_logged_at_plus_minutes_to_tipoff():
    res, why = mr.replay_row(row(), meta(tip=None), flat_path(0.32, 0.30))
    assert why == "ok" and res["lead_h"] == pytest.approx(6.0)
    res, why = mr.replay_row(row(minutes_to_tipoff=None), meta(tip=None), flat_path(0.32, 0.30))
    assert res is None and why == "no_tip"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def _res(game, *, touch, thru=False, ask=0.32, limit=0.31, close_ask=0.34, close_mid=0.335,
         outcome=1):
    return {"game": game, "touch": touch, "thru": thru, "ask": ask, "bid": 0.30,
            "spread": 0.02, "limit": limit, "close_ask": close_ask, "close_mid": close_mid,
            "taker_fee": 0.015, "maker_fee": 0.0, "lead_h": 6.0, "logged_ask_diff": 0.0,
            "outcome": outcome}


def test_summarize_counts_an_unfilled_order_as_zero():
    rs = [_res("g1", touch=True), _res("g2", touch=False),
          _res("g3", touch=False), _res("g4", touch=True, close_ask=0.30, close_mid=0.295)]
    s = mr.summarize(rs, "touch")
    assert s["n"] == 4 and s["games"] == 4 and s["fill_rate"] == pytest.approx(0.5)
    # filled edges: +3pp and −1pp; unfilled orders earn nothing.
    assert s["maker_clv_filled_ask"]["mean"] == pytest.approx(1.0)
    assert s["maker_net_per_candidate"]["mean"] == pytest.approx((3.0 - 1.0) / 4)
    assert s["maker_net_per_candidate_mid"]["mean"] == pytest.approx((2.5 - 1.5) / 4)
    # taker: every candidate crosses and pays the fee.
    assert s["taker_net_per_candidate"]["mean"] == pytest.approx(
        ((0.34 - 0.32) * 3 + (0.30 - 0.32)) / 4 * 100 - 1.5)
    assert s["unfilled_taker_clv_gross"]["mean"] == pytest.approx(2.0)


def test_summarize_thru_is_a_subset_and_realized_roi_uses_the_limit():
    rs = [_res("g1", touch=True, thru=True, outcome=1),
          _res("g2", touch=True, thru=False, outcome=0),
          _res("g3", touch=False, outcome=1)]
    touch, thru = mr.summarize(rs, "touch"), mr.summarize(rs, "thru")
    assert touch["fill_rate"] == pytest.approx(2 / 3)
    assert thru["fill_rate"] == pytest.approx(1 / 3)
    assert touch["filled_realized"]["n"] == 2
    assert touch["filled_realized"]["win_rate"] == pytest.approx(0.5)
    # two fills at 0.31: one wins (+0.69), one loses (−0.31) on 0.62 staked
    assert touch["filled_realized"]["roi_net_pct"] == pytest.approx((0.69 - 0.31) / 0.62 * 100)


def test_summarize_ci_is_clustered_by_game():
    # Twenty rungs of ONE game are one observation: no CI below five games.
    one_game = [_res("g1", touch=True) for _ in range(20)]
    assert mr.summarize(one_game, "touch")["maker_net_per_candidate"]["ci95"] is None
    many = [_res(f"g{i}", touch=bool(i % 2), close_ask=0.30 + 0.01 * (i % 5)) for i in range(30)]
    lo, hi = mr.summarize(many, "touch")["maker_net_per_candidate"]["ci95"]
    assert lo <= mr.summarize(many, "touch")["maker_net_per_candidate"]["mean"] <= hi


def test_summarize_empty_and_bad_fill():
    assert mr.summarize([], "touch") is None
    with pytest.raises(ValueError):
        mr.summarize([_res("g", touch=True)], "maybe")


def test_fees_come_from_the_venue_fee_model():
    from evmax.fees import polymarket_us_fee_prob

    res, _ = mr.replay_row(row(), meta(), flat_path(0.32, 0.30))
    assert res["taker_fee"] == pytest.approx(polymarket_us_fee_prob(0.32))
    assert res["maker_fee"] == pytest.approx(polymarket_us_fee_prob(0.31, maker=True))
    assert res["taker_fee"] > res["maker_fee"]
