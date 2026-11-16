"""Tests for the Polymarket US entry-timing analysis (polyus_entry_timing.py).

Synthetic own-side paths. Covers the no-look-ahead quote rule, bucket-minimum
vs point-sample fills, each placement policy's payoff, the fallback that keeps
every policy scored on the same candidates, and the decay / sampling tables.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evmax.agents.cleanup import polyus_entry_timing as et
from evmax.agents.cleanup import polyus_maker_replay as mr
from evmax.fees import polymarket_us_fee_prob as fee

B = mr.BUCKET_S
TIP = datetime(2026, 11, 16, 1, 0, tzinfo=timezone.utc)


def make_path(asks, *, bid_gap=0.02, lows=None, lead_min=None, tick=0.01, game="g1"):
    """Path whose quotes[0] ends exactly at t0, then one bucket per 5 min.

    ``asks[0]`` is the entry ask; ``lows[i]`` overrides bucket i's minimum ask.
    """
    n = len(asks)
    lead_min = (n - 1) * 5 if lead_min is None else lead_min
    tip = TIP.timestamp()
    t0 = tip - lead_min * 60
    quotes = []
    for i, a in enumerate(asks):
        low = a if lows is None or lows[i] is None else lows[i]
        quotes.append((t0 + i * B, a, round(a - bid_gap, 6), low))
    close_ask, close_bid = quotes[-1][1], quotes[-1][2]
    return et.Path("polymarket_us:s:chi", "nba", "moneyline", game, t0, tip, tick,
                   tuple(quotes), close_ask, (close_ask + close_bid) / 2)


def flat(n=73, ask=0.32):          # 6 hours of 5-minute buckets
    return [ask] * n


# ---------------------------------------------------------------------------
# build_path
# ---------------------------------------------------------------------------


def _buckets(long_ask, long_bid, start, end):
    out, t = [], int(start.timestamp()) - int(start.timestamp()) % B
    while t < int(end.timestamp()):
        out.append([t, long_ask, 1 - long_bid, long_ask, 1 - long_bid])
        t += B
    return out


def test_build_path_matches_the_replay_entry_and_close():
    t0 = TIP - timedelta(hours=2)
    meta = mr.MarketMeta("s", TIP, 0.01, (("chi", "Bulls", True), ("cle", "Cavs", False)))
    buckets = _buckets(0.32, 0.30, t0 - timedelta(hours=1), TIP + timedelta(hours=1))
    row = {"market_id": "polymarket_us:s:chi", "logged_at": t0.strftime("%Y-%m-%d %H:%M:%S"),
           "minutes_to_tipoff": 120, "sector": "nba", "market_type": "moneyline", "game": "g"}
    p, why = et.build_path(row, meta, buckets)
    assert why == "ok"
    assert p.quotes[0][0] == pytest.approx(t0.timestamp())     # entry bucket ends at t0
    assert p.quotes[0][1:3] == (0.32, pytest.approx(0.30))
    assert p.quotes[-1][0] == pytest.approx(TIP.timestamp())   # last fully pre-tip bucket
    assert len(p.quotes) == 1 + 24                              # 2h of 5-min buckets
    assert p.lead_h == pytest.approx(2.0)
    assert et.limit_price(p) == pytest.approx(0.31)

    replay, _ = mr.replay_row(row, meta, buckets)
    assert et.limit_price(p) == pytest.approx(replay["limit"])
    assert p.close_ask == pytest.approx(replay["close_ask"])

    short, _ = et.build_path(dict(row, market_id="polymarket_us:s:cle"), meta, buckets)
    assert short.quotes[0][1:3] == (pytest.approx(0.70), pytest.approx(0.68))

    assert et.build_path(dict(row, market_id="polymarket_us:s:zzz"), meta, buckets) == (
        None, "side_unknown")
    assert et.build_path(row, meta, [])[1] == "no_entry_quote"


# ---------------------------------------------------------------------------
# quote_at / first_fill
# ---------------------------------------------------------------------------


def test_quote_at_never_looks_ahead():
    p = make_path([0.32, 0.34, 0.36])
    assert et.quote_at(p, p.t0)[0] == pytest.approx(0.32)
    assert et.quote_at(p, p.t0 + B - 1)[0] == pytest.approx(0.32)   # 0.34 not complete yet
    assert et.quote_at(p, p.t0 + B)[0] == pytest.approx(0.34)
    assert et.quote_at(p, p.t0 - 1) is None
    # a quote older than the staleness limit is a gap, not a price
    assert et.quote_at(p, p.t0 + 2 * B + mr.ENTRY_MAX_STALENESS_S + 1) is None


def test_first_fill_uses_the_bucket_minimum():
    lows = [None] * 13
    lows[4] = 0.31                                   # dips to the limit, then recovers
    p = make_path(flat(13), lows=lows)
    assert et.first_fill(p, 0.31) == pytest.approx(p.t0 + 4 * B)
    assert et.first_fill(p, 0.31, thru=True) is None  # touched, never traded through
    assert et.first_fill(p, 0.31, until=p.t0 + 3 * B) is None       # cancelled first
    assert et.first_fill(p, 0.31, until=p.t0 + 4 * B) is not None


def test_entry_bucket_minimum_is_not_a_fill():
    lows = [0.10] + [None] * 5                       # the dip was BEFORE the order rested
    assert et.first_fill(make_path(flat(6), lows=lows), 0.31) is None


def test_point_sampling_misses_a_dip_that_recovers_but_sees_one_that_persists():
    lows = [None] * 37
    lows[5] = 0.30                                   # transient dip inside one bucket
    transient = make_path(flat(37), lows=lows)
    assert et.first_fill(transient, 0.31) is not None
    for k in (5, 15, 60):                            # the standing quote never moved
        assert et.first_fill(transient, 0.31, sample_s=k * 60) is None

    persisted = make_path([0.32] * 6 + [0.30] * 31)  # the line moved and stayed
    assert et.first_fill(persisted, 0.31, sample_s=60 * 60) == pytest.approx(
        persisted.t0 + 12 * B)                       # first hourly sample
    assert et.first_fill(persisted, 0.31) == pytest.approx(persisted.t0 + 6 * B)


# ---------------------------------------------------------------------------
# policy payoffs
# ---------------------------------------------------------------------------


def test_taker_value_now_and_after_a_delay():
    p = make_path([0.32] + [0.34] * 36)              # runs away, closes 0.34
    assert et.taker_value(p) == pytest.approx((0.34 - 0.32 - fee(0.32)) * 100)
    assert et.taker_value(p, 30 * 60) == pytest.approx((0.34 - 0.34 - fee(0.34)) * 100)


def test_taker_delay_past_tip_falls_back_to_the_last_entry_time():
    p = make_path([0.32] * 6 + [0.36] * 7)           # 1h lead
    late = et.taker_value(p, 6 * 3600)               # asked to wait 6h
    assert late == pytest.approx((0.36 - 0.36 - fee(0.36)) * 100)
    assert et.drift(p, 6 * 3600) is None             # but drift is undefined there


def test_maker_rest_to_tip_filled_and_unfilled():
    lows = [None] * 37
    lows[10] = 0.31
    filled = make_path([0.32] * 20 + [0.35] * 17, lows=lows)
    assert et.maker_value(filled) == pytest.approx((0.35 - 0.31 - fee(0.31, maker=True)) * 100)
    assert et.maker_value(filled, thru=True) == 0.0
    # mid basis charges the maker half the closing spread
    assert et.maker_value(filled, mid_basis=True) == pytest.approx(
        (0.34 - 0.31 - fee(0.31, maker=True)) * 100)
    assert et.maker_value(make_path([0.32] + [0.36] * 36)) == 0.0     # ran away: no fill


def test_rest_then_cancel_and_rest_then_cross():
    runaway = make_path([0.32] * 7 + [0.36] * 30)    # ask jumps at +30 min, closes 0.36
    assert et.maker_value(runaway, rest_s=3600) == 0.0
    crossed = et.maker_value(runaway, rest_s=3600, then_cross=True)
    assert crossed == pytest.approx((0.36 - 0.36 - fee(0.36)) * 100)   # paid the new ask
    early = et.maker_value(runaway, rest_s=15 * 60, then_cross=True)
    assert early == pytest.approx((0.36 - 0.32 - fee(0.32)) * 100)     # crossed before the jump

    lows = [None] * 37
    lows[2] = 0.31
    quick = make_path(flat(37), lows=lows)           # fills 10 minutes in
    for h in (15, 60):
        assert et.maker_value(quick, rest_s=h * 60, then_cross=True) == pytest.approx(
            (0.32 - 0.31 - fee(0.31, maker=True)) * 100)
    assert et.maker_value(quick, rest_s=5 * 60, then_cross=True) == pytest.approx(
        (0.32 - 0.32 - fee(0.32)) * 100)             # cancelled before the fill


def test_drift_sign_and_pre_tip_limit():
    p = make_path([0.32] * 6 + [0.35] * 31)
    assert et.drift(p, 15 * 60) == pytest.approx(0.0)
    assert et.drift(p, 60 * 60) == pytest.approx(3.0)        # moved 3pp against us
    assert et.drift(p, 175 * 60) == pytest.approx(3.0)       # 5 min before tip: allowed
    assert et.drift(p, 180 * 60) is None                     # at tip: not an entry


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def test_decay_curve_uses_the_same_cohort_for_drift_and_total():
    long_lead = make_path([0.30] * 13 + [0.34] * 60, game="a")     # 6h lead, +4pp at 1h
    short_lead = make_path([0.30, 0.31] + [0.32] * 5, game="b")    # 30 min lead, +2pp total
    rows = {d["delay_min"]: d for d in et.decay_curve([long_lead, short_lead], (5, 120))}
    assert rows[5]["n"] == 2
    assert rows[5]["drift_pp"]["mean"] == pytest.approx((0.0 + 1.0) / 2)
    assert rows[5]["total_move_pp"]["mean"] == pytest.approx((4.0 + 2.0) / 2)
    assert rows[120]["n"] == 1                                     # only the long-lead game
    assert rows[120]["drift_pp"]["mean"] == pytest.approx(4.0)
    assert rows[120]["share_consumed"] == pytest.approx(1.0)


def test_sampling_bias_reference_row_and_coarse_row():
    lows = [None] * 37
    lows[5] = 0.30
    transient = make_path([0.32] * 37, lows=lows, game="a")        # good fill, value +1pp
    persisted = make_path([0.32] * 6 + [0.29] * 31, game="b")      # bad fill, value −2pp
    ref, hourly = et.sampling_bias([transient, persisted], (60,))
    assert ref["sample_min"] is None and ref["fill_rate"] == pytest.approx(1.0)
    assert ref["value_if_filled_pp"]["mean"] == pytest.approx((1.0 - 2.0) / 2)
    assert hourly["fill_rate"] == pytest.approx(0.5)               # misses the good one
    assert hourly["value_if_filled_pp"]["mean"] == pytest.approx(-2.0)
    assert hourly["value_per_candidate_pp"]["mean"] == pytest.approx(-1.0)


def test_time_to_fill():
    lows = [None] * 37
    lows[2] = 0.31
    quick = make_path(flat(37), lows=lows, game="a")               # 10 min
    never = make_path([0.32] + [0.36] * 36, game="b")
    ttf = et.time_to_fill([quick, never])
    assert ttf["n"] == 1 and ttf["median_min"] == pytest.approx(10.0)
    assert ttf["share_within"][15] == pytest.approx(1.0)
    assert et.time_to_fill([never]) == {"n": 0}


def test_policy_table_scores_every_policy_on_every_candidate():
    paths = [make_path([0.32] * 7 + [0.36] * 30, game=f"g{i}") for i in range(6)]
    paths.append(make_path([0.32, 0.33, 0.34], game="short"))      # 10 min lead
    table = et.policy_table(paths, rest_minutes=(15, 60), wait_minutes=(15,))
    assert {r["n"] for r in table} == {7}
    by = {r["policy"]: r["mean"] for r in table}
    assert by["rest 15m, then cross"] > by["rest 60m, then cross"]   # crossed before the jump
    assert by["rest to tip (touch)"] == pytest.approx(0.0)
    # The 10-minute-lead game cannot rest 15 minutes: it crosses at the last
    # entry time instead (ask 0.33, not 0.32), so it is scored, not dropped.
    six = (0.36 - 0.32 - fee(0.32)) * 100
    assert by["cross now"] == pytest.approx((6 * six + (0.34 - 0.32 - fee(0.32)) * 100) / 7)
    assert by["rest 15m, then cross"] == pytest.approx(
        (6 * six + (0.34 - 0.33 - fee(0.33)) * 100) / 7)


def test_by_lead_buckets():
    out = et.by_lead([make_path(flat(13), game="a"),               # 1h
                      make_path(flat(73), game="b")])              # 6h
    assert [b["lead_h"] for b in out] == [(1, 3), (6, 12)]
    assert all(b["n"] == 1 for b in out)
