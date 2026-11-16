"""Tests for the Pinnacle-vs-venue line-movement lens (line_moves.py).

Synthetic archive.db rows. Covers the pairing rules (nearest Pinnacle snapshot
within tolerance, strictly pre-tip, bad quotes dropped, same-ET-day matching,
opponent-contract complement, venue split) and each statistic's sign.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from evmax.agents.cleanup import line_moves as lm
from evmax.archiver import _get_connection

TIP = datetime(2026, 11, 13, 0, 30, tzinfo=timezone.utc)  # 19:30 ET on 11-12
EVENT = "nba::2026-11-12::hawks_vs_bulls"
ET_DAY = "2026-11-12"


def _ident(name: str) -> str:
    return (name or "").strip().lower()


@pytest.fixture
def conn(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.db"
    monkeypatch.setattr("evmax.archiver.DB_PATH", db_path)
    with _get_connection():  # creates schema
        pass
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    yield c
    c.close()


_seq = iter(range(10**6))


def pin(conn, hours_out, prob_a, *, event=EVENT, tip=TIP, a="hawks", b="bulls",
        sector="nba", offset_min=0.0, spread_line=None):
    t = tip - timedelta(hours=hours_out) + timedelta(minutes=offset_min)
    conn.execute(
        "INSERT INTO archived_sharp_odds (session_id, fetched_at, sector, event_id, book, "
        "outcome_a_label, outcome_b_label, outcome_a_decimal, outcome_b_decimal, "
        "true_prob_a, true_prob_b, margin, spread_line, event_date) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (f"s{next(_seq)}", t.isoformat(), sector, event, "pinnacle", a, b, 1.9, 1.9,
         prob_a, 1 - prob_a, 0.03, spread_line, tip.isoformat()),
    )


def ven(conn, hours_out, ask, *, team="hawks", day=ET_DAY, tip=TIP, bid=None,
        ticker=None, sector="nba", market_type="moneyline", offset_min=0.0):
    t = tip - timedelta(hours=hours_out) + timedelta(minutes=offset_min)
    bid = ask - 0.01 if bid is None else bid
    conn.execute(
        "INSERT INTO archived_kalshi_markets (session_id, fetched_at, sector, ticker, "
        "market_type, yes_price, no_price, yes_team, event_date) VALUES (?,?,?,?,?,?,?,?,?)",
        (f"s{next(_seq)}", t.isoformat(), sector,
         ticker or f"KXNBAGAME-X-{team.upper()}", market_type, ask, 1 - bid, team,
         f"{day}T12:00:00+00:00"),
    )


def both(conn, hours_out, prob, ask, **kw):
    pin(conn, hours_out, prob)
    ven(conn, hours_out, ask, **kw)


def load(conn, **kw):
    return lm.load_series(conn, "nba", normalize=_ident, **kw)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_pairs_snapshots_far_to_near_with_mid_and_spread(conn):
    both(conn, 10, 0.60, 0.59)
    both(conn, 5, 0.62, 0.61)
    both(conn, 0.5, 0.65, 0.66)
    conn.commit()
    (g,) = load(conn)
    assert (g.team, g.opponent) == ("hawks", "bulls")
    assert [round(p.hours_to_tip, 1) for p in g.points] == [10.0, 5.0, 0.5]
    first = g.points[0]
    assert first.pinnacle == pytest.approx(0.60)
    assert first.venue_mid == pytest.approx(0.585)  # ask 0.59, bid 0.58
    assert first.venue_spread == pytest.approx(0.01)


def test_in_play_and_bad_quotes_are_dropped(conn):
    both(conn, 10, 0.60, 0.59)
    both(conn, 5, 0.62, 0.61)
    both(conn, 1, 0.63, 0.62)
    pin(conn, 3, 0.62)
    ven(conn, 3, 0.99)                 # settled / empty book
    pin(conn, 2, 0.62)
    ven(conn, 2, 0.70, bid=0.40)       # 30c-wide placeholder quote
    both(conn, -1, 0.90, 0.92)         # one hour AFTER tip
    conn.commit()
    (g,) = load(conn)
    assert [round(p.hours_to_tip) for p in g.points] == [10, 5, 1]


def test_venue_snapshot_needs_a_pinnacle_snapshot_within_tolerance(conn):
    both(conn, 10, 0.60, 0.59)
    both(conn, 5, 0.62, 0.61)
    both(conn, 1, 0.63, 0.62)
    ven(conn, 7.5, 0.50)               # 2.5h from the nearest Pinnacle print
    ven(conn, 5, 0.615, offset_min=10)  # 10 min after the 5h Pinnacle print
    conn.commit()
    (g,) = load(conn, pair_tol_min=20)
    leads = [round(p.hours_to_tip, 2) for p in g.points]
    assert 7.5 not in leads
    assert len(g.points) == 4
    (g,) = load(conn, pair_tol_min=5)
    assert len(g.points) == 3


def test_et_sector_never_pairs_the_next_days_game(conn):
    # Series: same two teams play again the next ET day. Only the NEXT game's
    # venue market is archived; today's Pinnacle line must not price it.
    for h, p in ((10, 0.60), (5, 0.62), (1, 0.63)):
        pin(conn, h, p)
        ven(conn, h, p, day="2026-11-13")
    conn.commit()
    assert load(conn) == []


def test_opponent_contract_is_read_as_its_complement(conn):
    for h, p in ((10, 0.60), (5, 0.62), (1, 0.63)):
        pin(conn, h, p)
        ven(conn, h, round(1 - p + 0.005, 3), team="bulls")  # mid == 1 - p
    conn.commit()
    (g,) = load(conn)
    assert g.team == "hawks"
    for pt in g.points:
        assert pt.venue_mid == pytest.approx(pt.pinnacle)


def test_spread_records_and_non_moneyline_markets_are_ignored(conn):
    both(conn, 10, 0.60, 0.59)
    both(conn, 5, 0.62, 0.61)
    both(conn, 1, 0.63, 0.62)
    pin(conn, 5, 0.10, spread_line=-4.5)  # a spread record on the same id
    ven(conn, 5, 0.20, market_type="spread", ticker="KXNBASPREAD-X-HAWKS4")
    conn.commit()
    (g,) = load(conn)
    assert len(g.points) == 3
    assert all(p.pinnacle > 0.5 for p in g.points)


def test_venue_filter_separates_kalshi_and_polymarket(conn):
    for h, p in ((10, 0.60), (5, 0.62), (1, 0.63)):
        pin(conn, h, p)
        ven(conn, h, p)
        ven(conn, h, p + 0.05, ticker="polymarket_us:aec-nba-atl-chi-2026-11-12:atl")
    conn.commit()
    (k,) = load(conn, venue="kalshi")
    (pm,) = load(conn, venue="polymarket_us")
    assert k.points[0].venue_mid == pytest.approx(0.595)
    assert pm.points[0].venue_mid == pytest.approx(0.645)


def test_min_points_and_since_filter(conn):
    both(conn, 10, 0.60, 0.59)
    both(conn, 5, 0.62, 0.61)
    conn.commit()
    assert load(conn) == []                       # 2 points < min_points=3
    assert len(load(conn, min_points=2)) == 1
    assert load(conn, min_points=2, since="2026-12-01") == []


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _series(eid, pts):
    return lm.GameSeries(eid, "a", "b", TIP,
                         tuple(lm.Point(h, p, v, 0.01) for h, p, v in pts))


def test_gap_by_bucket_uses_last_snapshot_per_bucket_and_needs_a_close():
    g = _series("g1", [(20, 0.60, 0.58), (14, 0.61, 0.60), (4, 0.63, 0.63), (0.5, 0.65, 0.64)])
    no_close = _series("g2", [(20, 0.50, 0.40), (14, 0.50, 0.40), (4, 0.50, 0.40)])
    rows = {r["bucket"]: r for r in lm.gap_by_bucket([g, no_close])}
    b = rows["12-24h"]
    assert b["games"] == 1                         # g2 has no usable close
    assert b["abs_gap_pp"] == pytest.approx(1.0)   # the 14h point, not the 20h one
    assert b["signed_gap_pp"] == pytest.approx(-1.0)
    assert b["pinnacle_to_close_pp"] == pytest.approx(4.0)
    assert b["venue_to_close_pp"] == pytest.approx(4.0)
    assert rows["<1h"]["pinnacle_to_close_pp"] == pytest.approx(0.0)
    assert "6-12h" not in rows


def test_gap_closure_identifies_the_follower():
    # Pinnacle is flat; the venue closes half of the gap every step.
    series = []
    for i, gap0 in enumerate((0.04, -0.06, 0.02, -0.03)):
        pts, v = [], 0.50 - gap0
        for h in (10, 8, 6, 4):
            pts.append((h, 0.50, v))
            v += (0.50 - v) * 0.5
        series.append(_series(f"g{i}", pts))
    c = lm.gap_closure(series)
    assert c["venue_toward_pinnacle"] == pytest.approx(0.5)
    assert c["pinnacle_toward_venue"] == pytest.approx(0.0, abs=1e-12)
    assert c["games"] == 4
    # No interval starts under 5h from tip (the last point has no next move).
    assert lm.gap_closure(series, 0.0, 5.0) is None


def test_gap_closure_window_and_interval_limits():
    far = _series("g1", [(30, 0.50, 0.46), (10, 0.50, 0.48), (9, 0.50, 0.49)])
    near = _series("g2", [(3, 0.50, 0.54), (2, 0.50, 0.52), (1.9, 0.50, 0.51), (1, 0.50, 0.505)])
    # g1's 30h -> 10h step (20h apart) and g2's 2h -> 1.9h step (6 min) are
    # not "next moves": one usable interval each.
    assert sum(1 for _ in lm._intervals(far)) == 1
    assert sum(1 for _ in lm._intervals(near)) == 2
    assert lm.gap_closure([far, near], 6.0, float("inf")) is None  # one game only


def test_lead_lag_detects_a_venue_that_follows_one_step_late():
    series = []
    for i, moves in enumerate(([0.02, -0.01, 0.03, 0.0], [-0.03, 0.02, 0.01, -0.02],
                               [0.01, 0.04, -0.02, 0.01])):
        p, v, pts, prev = 0.50, 0.50, [(10, 0.50, 0.50)], 0.0
        for k, m in enumerate(moves):
            p += m
            v += prev          # venue repeats Pinnacle's PREVIOUS move
            prev = m
            pts.append((10 - 2 * (k + 1), p, v))
        series.append(_series(f"g{i}", pts))
    ll = lm.lead_lag(series)
    assert ll["pinnacle_leads_venue"]["slope"] == pytest.approx(1.0)
    assert abs(ll["venue_leads_pinnacle"]["slope"]) < 0.9


def test_drift_is_signed_toward_the_opening_favorite():
    fav_shortens = _series("g1", [(12, 0.60, 0.60), (6, 0.62, 0.62), (0.5, 0.66, 0.65)])
    dog_side_listed = _series("g2", [(12, 0.40, 0.40), (6, 0.38, 0.38), (0.5, 0.36, 0.37)])
    late_first = _series("g3", [(4, 0.50, 0.50), (2, 0.60, 0.60), (0.5, 0.70, 0.70)])
    d = lm.drift([fav_shortens, dog_side_listed, late_first])
    assert d["games"] == 2                        # g3 first seen under 6h out
    # g1 favorite +6pp; g2 lists the underdog falling 4pp = favorite +4pp.
    assert d["favorite_drift_pp"] == pytest.approx(5.0)
    assert d["favorite_shortened_pct"] == pytest.approx(100.0)
    assert d["moves_ge_3pp"] == 2
    assert d["largest"][0]["event_id"] == "g1"


def test_clustered_slope_degenerate_inputs_return_none():
    assert lm.clustered_slope([], [], []) is None
    assert lm.clustered_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], ["a", "b", "c"]) is None
    assert lm.clustered_slope([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], ["a", "a", "a"]) is None


def test_analyze_on_empty_archive_is_json_serializable(conn):
    rep = lm.analyze(conn, "nba", normalize=_ident)
    assert rep["games"] == 0 and rep["buckets"] == [] and rep["drift"] is None
    json.dumps(rep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prints_event_and_outcome_columns(conn):
    from evmax.cli.commands.cleanup import app

    for day_shift, eid in ((0, EVENT), (1, "nba::2026-11-13::hawks_vs_bulls")):
        tip = TIP + timedelta(days=day_shift)
        day = (datetime(2026, 11, 12) + timedelta(days=day_shift)).date().isoformat()
        for h, p in ((12, 0.60), (8, 0.62), (4, 0.63), (0.5, 0.66 + 0.02 * day_shift)):
            pin(conn, h, p, event=eid, tip=tip)
            ven(conn, h, p, day=day, tip=tip)
    conn.commit()
    runner = CliRunner()
    res = runner.invoke(app, ["line-moves", "-s", "nba", "--since", "2026-11-01"])
    assert res.exit_code == 0, res.output
    assert "2 games" in res.output
    assert "Event" in res.output and "Outcome" in res.output
    assert "Hawks vs Bulls" in res.output and "Hawks ML" in res.output

    res = runner.invoke(app, ["line-moves", "-s", "nba", "--since", "2026-11-01", "--json"])
    assert json.loads(res.output)["games"] == 2

    res = runner.invoke(app, ["line-moves", "--venue", "betfair"])
    assert res.exit_code == 2
