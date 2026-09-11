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
