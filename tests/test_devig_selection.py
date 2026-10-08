"""Automatic per-sector devig selection: gate logic + persisted state + resolve."""
from __future__ import annotations

import json

import pytest

import evmax.ev.devig as d
from evmax.ev.devig import resolve_devig_method
from evmax.ev.devig_selection import (
    DEVIG_SELECT_MIN_BRIER_DELTA,
    DEVIG_SELECT_MIN_N,
    DevigRecommendation,
    clear_selected_method,
    evaluate_sector,
    save_selected_method,
)


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Point the persisted-state file at a temp path and clear the cache."""
    state = tmp_path / "devig_method_state.json"
    monkeypatch.setattr(d, "DEVIG_METHOD_STATE_PATH", state)
    monkeypatch.setattr("evmax.ev.devig_selection.DEVIG_METHOD_STATE_PATH", state)
    monkeypatch.setattr(d, "DEVIG_METHOD_BY_SECTOR", {})
    d.invalidate_selected_methods_cache()
    yield
    d.invalidate_selected_methods_cache()


# ---------------------------------------------------------------------------
# evaluate_sector — the three-part gate
# ---------------------------------------------------------------------------

def _pairs(probs, outcomes):
    return list(zip(probs, outcomes))


def test_no_challenger_when_power_best():
    # Power already perfectly calibrated; shin worse → recommend nothing.
    n = 300
    power = _pairs([0.5] * n, [1, 0] * (n // 2))
    shin = _pairs([0.9] * n, [1, 0] * (n // 2))  # badly off
    rec = evaluate_sector("soccer", {"power": power, "shin": shin})
    assert rec is not None
    assert rec.best_method == "power"
    assert rec.clears_gate is False
    assert rec.is_actionable is False


def test_gate_clears_when_challenger_significantly_better():
    # Construct a case where shin is materially + significantly better: outcomes
    # follow shin's probs (0.6 wins ~60%), power is a flat 0.5 mismatch.
    n = 400
    outcomes = ([1] * 6 + [0] * 4) * (n // 10)  # 60% win rate
    power = _pairs([0.5] * n, outcomes)
    shin = _pairs([0.6] * n, outcomes)  # closer to the 0.6 base rate
    rec = evaluate_sector("soccer", {"power": power, "shin": shin})
    assert rec.best_method == "shin"
    assert rec.delta >= DEVIG_SELECT_MIN_BRIER_DELTA
    assert rec.z >= 1.64
    assert rec.n >= DEVIG_SELECT_MIN_N
    assert rec.clears_gate is True
    assert rec.is_actionable is True


def test_below_min_n_does_not_clear():
    n = 50  # under DEVIG_SELECT_MIN_N
    outcomes = ([1] * 6 + [0] * 4) * (n // 10)
    power = _pairs([0.5] * n, outcomes)
    shin = _pairs([0.6] * n, outcomes)
    rec = evaluate_sector("soccer", {"power": power, "shin": shin})
    assert rec.n < DEVIG_SELECT_MIN_N
    assert rec.clears_gate is False


def test_marginal_edge_below_delta_does_not_clear():
    # Tiny, noise-floor improvement — the significance/margin guard must reject it
    # (the tennis lesson: a marginal Brier edge never triggers a flip).
    n = 400
    outcomes = ([1, 0]) * (n // 2)  # 50% base rate
    power = _pairs([0.50] * n, outcomes)
    shin = _pairs([0.499] * n, outcomes)  # essentially identical
    rec = evaluate_sector("soccer", {"power": power, "shin": shin})
    assert rec.delta < DEVIG_SELECT_MIN_BRIER_DELTA
    assert rec.clears_gate is False


def test_none_when_power_absent():
    assert evaluate_sector("soccer", {"shin": _pairs([0.5], [1])}) is None


def test_mismatched_lengths_skip_challenger():
    # A challenger with a different line count can't be paired → ignored.
    power = _pairs([0.5] * 300, [1, 0] * 150)
    shin = _pairs([0.6] * 200, [1, 0] * 100)
    rec = evaluate_sector("soccer", {"power": power, "shin": shin})
    assert rec.best_method == "power"


# ---------------------------------------------------------------------------
# persisted state + resolve precedence
# ---------------------------------------------------------------------------

def test_default_is_power_when_no_state():
    assert resolve_devig_method("soccer", "power") == "power"


def test_promote_persists_and_resolves():
    save_selected_method("soccer", "shin")
    assert resolve_devig_method("soccer") == "shin"
    assert resolve_devig_method("nba") == "power"  # untouched sectors


def test_promote_writes_methods_block(tmp_path):
    save_selected_method("soccer", "shin")
    state = json.loads(d.DEVIG_METHOD_STATE_PATH.read_text())
    assert state["methods"]["soccer"] == "shin"


def test_clear_reverts_to_power():
    save_selected_method("soccer", "shin")
    assert resolve_devig_method("soccer") == "shin"
    clear_selected_method("soccer")
    assert resolve_devig_method("soccer") == "power"
    # power stores nothing — the entry is removed, not written as "power".
    state = json.loads(d.DEVIG_METHOD_STATE_PATH.read_text())
    assert "soccer" not in state["methods"]


def test_code_override_wins_over_persisted(monkeypatch):
    save_selected_method("soccer", "shin")
    monkeypatch.setattr(d, "DEVIG_METHOD_BY_SECTOR", {"soccer": "power"})
    d.invalidate_selected_methods_cache()
    assert resolve_devig_method("soccer") == "power"  # code hard override wins


def test_save_rejects_unknown_method():
    with pytest.raises(ValueError):
        save_selected_method("soccer", "bogus")


def test_corrupt_state_degrades_to_power():
    d.DEVIG_METHOD_STATE_PATH.write_text("{ not json")
    d.invalidate_selected_methods_cache()
    assert resolve_devig_method("soccer", "power") == "power"


def test_unknown_method_in_state_is_dropped():
    d.DEVIG_METHOD_STATE_PATH.write_text(json.dumps({"methods": {"soccer": "bogus"}}))
    d.invalidate_selected_methods_cache()
    assert resolve_devig_method("soccer", "power") == "power"


def test_recommendation_is_actionable_only_when_unapplied():
    r = DevigRecommendation(
        sector="soccer", current_method="shin", best_method="shin",
        power_brier=0.20, best_brier=0.19, delta=0.01, z=3.0, n=400, clears_gate=True,
    )
    # Already applied (current == best) → not actionable, nothing to surface.
    assert r.is_actionable is False


# ---------------------------------------------------------------------------
# collect_devig_observations — which lines and outcomes enter the A/B
# ---------------------------------------------------------------------------

import sqlite3
from datetime import datetime, timedelta, timezone

from evmax.ev.devig import devig
from evmax.ev.devig_selection import (
    collect_devig_observations,
    is_game_winner_event_id,
    recommend_devig_methods,
    scored_contract,
)

TOR, MTL = "Toronto Maple Leafs", "Montreal Canadiens"


def test_game_winner_event_ids():
    assert is_game_winner_event_id("nhl::2026-09-29::toronto_maple_leafs_vs_montreal_canadiens")
    assert not is_game_winner_event_id("nhl::2026-09-29::toronto_maple_leafs_vs_montreal_canadiens::spread")
    assert not is_game_winner_event_id("nfl::2026-09-20::a_vs_b::total::44.5")
    assert not is_game_winner_event_id("worldcup::2026-07-04::a_vs_b::advance")
    assert not is_game_winner_event_id("nba::2026-03-01::a_vs_b::prop::x::points")
    assert not is_game_winner_event_id(None)


class TestScoredContract:
    def test_side_a_market(self):
        assert scored_contract([("toronto maple leafs", 1)], TOR, MTL, three_way=False) == ("a", 1)

    def test_only_side_b_market_scores_b(self):
        assert scored_contract([("montreal canadiens", 0)], TOR, MTL, three_way=False) == ("b", 0)

    def test_a_preferred_over_b_whatever_the_result(self):
        rows = [("montreal canadiens", 1), ("toronto maple leafs", 0)]
        assert scored_contract(rows, TOR, MTL, three_way=False) == ("a", 0)

    def test_short_label_maps_by_token_subset(self):
        assert scored_contract(
            [("fever", 1)], "Indiana Fever", "Las Vegas Aces", three_way=False,
        ) == ("a", 1)

    def test_ambiguous_label_is_ignored(self):
        # "maria" is inside BOTH names — never guess a side.
        assert scored_contract(
            [("maria", 1)], "Maria Sakkari", "Tatjana Maria", three_way=False,
        ) is None

    def test_three_way_b_loss_scores_b_not_an_a_win(self):
        # A side-B loss leaves "A won" and "draw" open on a 3-way line. It must
        # score B's own contract, never infer an A win.
        assert scored_contract([("montreal canadiens", 0)], TOR, MTL, three_way=True) == ("b", 0)

    def test_three_way_tie_market(self):
        assert scored_contract([("tie", 1)], TOR, MTL, three_way=True) == ("draw", 1)

    def test_draw_row_ignored_on_two_way_line(self):
        assert scored_contract([("tie", 0)], TOR, MTL, three_way=False) is None

    def test_contradictions_return_none(self):
        both_won = [("toronto maple leafs", 1), ("montreal canadiens", 1)]
        assert scored_contract(both_won, TOR, MTL, three_way=False) is None
        flip = [("toronto maple leafs", 1), ("toronto maple leafs", 0)]
        assert scored_contract(flip, TOR, MTL, three_way=False) is None
        both_lost = [("toronto maple leafs", 0), ("montreal canadiens", 0)]
        assert scored_contract(both_lost, TOR, MTL, three_way=False) is None

    def test_no_mappable_row(self):
        assert scored_contract([("boston bruins", 1)], TOR, MTL, three_way=False) is None


_TIP = datetime(2026, 9, 29, 23, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture()
def dbs(tmp_path):
    """(predictions conn, archive path, add_outcome, add_snapshot) with the real archive schema."""
    from evmax.archiver import _MIGRATIONS, SCHEMA

    archive_path = tmp_path / "archive.db"
    arc = sqlite3.connect(archive_path, isolation_level=None)  # autocommit
    arc.executescript(SCHEMA)
    for migration in _MIGRATIONS:
        try:
            arc.execute(migration)
        except sqlite3.OperationalError:
            pass  # column already in SCHEMA
    pred = sqlite3.connect(":memory:")
    pred.execute(
        "CREATE TABLE ev_outcomes (market_id TEXT UNIQUE, event_id TEXT, yes_team TEXT,"
        " outcome INTEGER, sector TEXT, resolved_at TEXT)"
    )

    def add_outcome(market_id, event_id, yes_team, outcome):
        pred.execute(
            "INSERT INTO ev_outcomes VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (market_id, event_id, yes_team, outcome, event_id.split("::")[0]),
        )

    def add_snapshot(event_id, fetched_at, a_dec, b_dec, *, a=TOR, b=MTL, draw=None,
                     spread_line=None, total_line=None, tip=_TIP):
        arc.execute(
            "INSERT INTO archived_sharp_odds (session_id, fetched_at, sector, event_id, book,"
            " outcome_a_label, outcome_b_label, outcome_a_decimal, outcome_b_decimal,"
            " outcome_draw_decimal, true_prob_a, true_prob_b, margin, spread_line,"
            " total_line, event_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"s-{fetched_at}", _iso(fetched_at), event_id.split("::")[0], event_id,
             "pinnacle", a, b, a_dec, b_dec, draw, 0.5, 0.5, 0.03, spread_line,
             total_line, _iso(tip)),
        )

    yield pred, archive_path, add_outcome, add_snapshot
    arc.close()
    pred.close()


def _collect(dbs_tuple):
    pred, archive_path, *_ = dbs_tuple
    return collect_devig_observations(120, pred_conn=pred, archive_path=archive_path)


def test_one_observation_per_game_from_the_last_pretip_snapshot(dbs):
    pred, archive_path, add_outcome, add_snapshot = dbs
    ev = "nhl::2026-09-29::toronto_maple_leafs_vs_montreal_canadiens"
    add_outcome("kalshi:TOR", ev, "toronto maple leafs", 1)
    add_outcome("kalshi:MTL", ev, "montreal canadiens", 0)
    add_outcome("polymarket_us:tor", ev, "toronto maple leafs", 1)
    add_snapshot(ev, _TIP - timedelta(hours=48), 1.90, 1.95)
    add_snapshot(ev, _TIP - timedelta(hours=2), 1.80, 2.08)
    add_snapshot(ev, _TIP - timedelta(minutes=10), 1.75, 2.15)  # the close
    add_snapshot(ev, _TIP + timedelta(minutes=30), 1.10, 7.00)  # in-play — never scored
    obs = _collect(dbs)
    assert len(obs) == 1
    o = obs[0]
    assert (o.contract, o.won, o.three_way) == ("a", 1, False)
    for method, prob in o.probs.items():
        assert prob == pytest.approx(devig([1.75, 2.15], method=method).true_probs[0])


def test_three_way_scores_the_logged_contract(dbs):
    pred, archive_path, add_outcome, add_snapshot = dbs
    ev = "soccer::2026-09-29::toronto_maple_leafs_vs_montreal_canadiens"
    add_outcome("kalshi:TIE", ev, "tie", 0)
    add_outcome("kalshi:MTL", ev, "montreal canadiens", 0)
    add_snapshot(ev, _TIP - timedelta(hours=1), 2.10, 3.60, draw=3.40)
    obs = _collect(dbs)
    assert len(obs) == 1
    o = obs[0]
    assert (o.contract, o.won, o.three_way) == ("b", 0, True)
    assert o.probs["power"] == pytest.approx(
        devig([2.10, 3.60, 3.40], method="power").true_probs[1]
    )


def test_spread_and_total_records_never_enter_the_ab(dbs):
    pred, archive_path, add_outcome, add_snapshot = dbs
    base = "nhl::2026-09-29::toronto_maple_leafs_vs_montreal_canadiens"
    add_outcome("kalshi:SPREAD", f"{base}::spread", "toronto maple leafs", 1)
    add_snapshot(f"{base}::spread", _TIP - timedelta(hours=1), 3.2, 1.38, spread_line=-1.5)
    add_outcome("kalshi:TOTAL", f"{base}::total", "over", 1)
    add_snapshot(f"{base}::total::6.5", _TIP - timedelta(hours=1), 2.0, 1.85,
                 a="over", b="under", total_line=6.5)
    assert _collect(dbs) == []


def test_game_without_pretip_snapshot_is_skipped(dbs):
    pred, archive_path, add_outcome, add_snapshot = dbs
    ev = "nhl::2026-09-29::toronto_maple_leafs_vs_montreal_canadiens"
    add_outcome("kalshi:TOR", ev, "toronto maple leafs", 1)
    add_snapshot(ev, _TIP + timedelta(minutes=5), 1.50, 2.70)
    assert _collect(dbs) == []


def test_nhl_spread_ladder_artifact_regression(dbs):
    """The 2026-10-07 false NHL recommendation (multiplicative, z=32).

    Each game's ``::spread`` record is Pinnacle's main line with the −1.5 side
    as outcome_a. The resolved ``::spread`` market on the same team is a +1.5
    contract that covers ~80% of the time. Scored together, multiplicative (the
    method that gives the longshot the most) beat power on every snapshot. The
    game-winner lines are devigged exactly as Pinnacle prices them, so no method
    has a real edge. The fixed recommender must not flag nhl, and n must count
    games, not snapshots.
    """
    pred, archive_path, add_outcome, add_snapshot = dbs
    n_games, n_snaps = 240, 12
    for g in range(n_games):
        tip = _TIP + timedelta(days=g)
        base = f"nhl::{tip.date()}::toronto_maple_leafs_vs_montreal_canadiens"
        covered = int(g % 5 != 0)  # the +1.5 side covers 80% of the time
        add_outcome(f"kalshi:SPREAD-{g}:no", f"{base}::spread", "toronto maple leafs", covered)
        won = g % 2  # coin-flip game results on even lines
        add_outcome(f"kalshi:ML-{g}", base, "toronto maple leafs", won)
        for k in range(n_snaps):
            at = tip - timedelta(hours=n_snaps - k)
            add_snapshot(f"{base}::spread", at, 3.2, 1.38, spread_line=-1.5, tip=tip)
            add_snapshot(base, at, 1.95, 1.95, tip=tip)
    recs = recommend_devig_methods(120, pred_conn=pred, archive_path=archive_path)
    nhl = [r for r in recs if r.sector == "nhl"]
    assert len(nhl) == 1
    assert nhl[0].n == n_games
    assert nhl[0].clears_gate is False
    assert nhl[0].is_actionable is False

