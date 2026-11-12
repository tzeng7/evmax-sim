"""Deterministic orchestration of `.claude/workflows/model-improve.js`.

The graph keeps its gates in plain JavaScript: which triggers may queue work
(value_audit always; the improve-triggers set only when a run enables them), the
JS re-check of every trigger threshold, the allowed change kinds per trigger,
the denied-file guard, integrity before signal, and the per-trigger signal rule
(Brier for model changes; rows priced / CLV coverage for coverage and data
fixes, never Brier). These tests run the graph under tests/workflow_harness.mjs
with stubbed agents and pin that logic, plus its parity with
evmax/agents/cleanup/improve_triggers.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evmax.agents.cleanup import improve_triggers as it

REPO = Path(__file__).resolve().parents[1]
GRAPH = ".claude/workflows/model-improve.js"
HARNESS = REPO / "tests" / "workflow_harness.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required for the workflow harness")


def _harness(tmp_path: Path, scenario: dict) -> dict:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({"script": GRAPH, **scenario}))
    cp = subprocess.run(["node", str(HARNESS), str(path)], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert cp.returncode == 0, cp.stderr
    return json.loads(cp.stdout)


def _call(tmp_path: Path, fn: str, *argsets: list) -> list:
    return _harness(tmp_path, {"mode": "call", "calls": [{"fn": fn, "args": a} for a in argsets]})["results"]


def _run(tmp_path: Path, responses: dict, args: dict | None = None) -> dict:
    return _harness(tmp_path, {"mode": "run", "args": args or {}, "responses": responses})


# ---- fixtures ---------------------------------------------------------------

def _tf(trigger: str, sector: str, **over) -> dict:
    """A trigger finding shaped like `evmax cleanup improve-triggers --json` output."""
    base = {
        "passthrough": dict(market_type="moneyline", venue="polymarket_us", n=23, value=0.16, threshold=0.5, priority=0.34),
        "model_coverage": dict(model="ncaab_efficiency", n=25, value=1.0, threshold=0.5, priority=1.0),
        "close_capture": dict(venue="kalshi", n=15, value=0.333, threshold=0.3, priority=0.333),
        "calibration_bucket": dict(market_type="moneyline", bucket="10-20", n=200, value=8.0, threshold=3.0, z=4.1, priority=4.1),
    }[trigger]
    f = {"trigger": trigger, "sector": sector, "market_type": None, "venue": None, "model": None, "bucket": None,
         "z": None, "fires": True, "judge_metric": it.JUDGE_METRIC[trigger], "evidence": f"{trigger} {sector}"}
    f.update(base)
    f.update(over)
    return f


AUDIT_QUIET = {"findings": [{"sector": "nfl", "gap_type": "neutral", "actionable": False, "evidence": "within noise"}]}
AUDIT_GAP = {"findings": [
    {"sector": "nba", "gap_type": "model_subtracts", "z_score": -2.1, "actionable": True, "evidence": "worse"},
    {"sector": "nfl", "gap_type": "neutral", "actionable": False, "evidence": "within noise"},
]}
COVERAGE_TRIGGERS = {"findings": [
    _tf("model_coverage", "ncaab"),
    _tf("model_coverage", "ncaab", model="ncaab_possession_sim", priority=0.9),
    _tf("passthrough", "ncaab"),
    _tf("passthrough", "tennis", value=0.17, priority=0.33),
    _tf("close_capture", "ncaab"),
], "ran": list(it.TRIGGERS), "failed": {}}

SPEC_COVERAGE = {"sector": "ncaab", "kind": "coverage_fix", "params": "alias X", "novel": True, "target": "aliases"}
IMPL_OK = {"committed": True, "branch": "model-improve/ncaab-coverage_fix", "files_changed": ["evmax/sectors/aliases/ncaab.yaml"],
           "diff_summary": "alias"}
COV_PASS = {"ran": True, "coverage_before": 0.0, "coverage_after": 0.8, "newly_priced_n": 40, "harm_z": 0.3, "note": "replayed"}
ACCEPT = {"accept": True, "reason": "genuine alias fix"}
KEPT = {"action": "kept", "pr_url": "https://github.com/x/y/pull/1", "ledger_written": True}
REVERTED = {"action": "reverted", "ledger_written": True}


# ---- parity with the Python trigger module ------------------------------------

def test_trigger_rules_match_python_constants(tmp_path):
    [cfg] = _call(tmp_path, "triggerConfig", [])
    rules = cfg["TRIGGER_RULES"]
    assert cfg["DEFAULT_TRIGGERS"] == ["value_audit"]
    assert set(cfg["TRIGGER_ORDER"]) == set(it.TRIGGERS)
    for t in it.TRIGGERS:
        assert rules[t]["judge"] == it.JUDGE_METRIC[t]
    assert rules["passthrough"]["min_n"] == it.PASSTHROUGH_MIN_N
    assert rules["passthrough"]["max_value"] == it.PASSTHROUGH_MAX_DIVERGENCE_PP
    assert rules["model_coverage"]["min_n"] == it.COVERAGE_MIN_N
    assert rules["model_coverage"]["min_value"] == it.COVERAGE_MIN_UNPRICED_FRAC
    assert rules["close_capture"]["min_n"] == it.CLOSE_CAPTURE_MIN_N
    assert rules["close_capture"]["min_value_exclusive"] == it.CLOSE_CAPTURE_MAX_NULL_FRAC
    assert rules["calibration_bucket"]["min_n"] == it.CALIB_BUCKET_MIN_N
    assert rules["calibration_bucket"]["min_abs_value"] == it.CALIB_BUCKET_MIN_ERR_PP
    assert rules["calibration_bucket"]["min_abs_z"] == it.CALIB_BUCKET_MIN_Z
    # coverage / data fixes may never be judged on Brier
    assert {rules[t]["judge"] for t in ("passthrough", "model_coverage", "close_capture")}.isdisjoint({"holdout_brier"})
    # passthrough must never allow a weight change (sharp_weight is owned by `cleanup adjust`)
    assert rules["passthrough"]["kinds"] == ["coverage_fix"]


def test_python_findings_pass_the_js_recheck(tmp_path):
    """A finding the Python module emits must be eligible in JS; otherwise the two drift."""
    board = [{"sector": "ncaab", "market_type": "moneyline", "venue": "kalshi", "mode": "live",
              "blend_divergence_pp": 0.3, "n_logged": it.PASSTHROUGH_MIN_N}]
    rows = [{"sector": "ncaab", "model_diagnostics": json.dumps({"missing": ["ncaab_efficiency"]})}] * it.COVERAGE_MIN_N
    close = [{"sector": "ncaaf", "market_type": "moneyline", "market_id": f"K{i}",
              "kalshi_clv_pct": None if i < 4 else 1.0} for i in range(12)]
    found = it.passthrough_findings(board) + it.coverage_findings(rows) + it.close_capture_findings(close)
    assert len(found) == 3
    assert [r["ok"] for r in _call(tmp_path, "triggerEligible", *[[f] for f in found])] == [True, True, True]


# ---- pure gates ---------------------------------------------------------------

def test_resolve_triggers_always_keeps_value_audit(tmp_path):
    res = _call(tmp_path, "resolveTriggers", [None], [{}], [{"triggers": ["model_coverage", "bogus"]}],
                [{"triggers": "model_coverage"}])
    assert res[0] == res[1] == {"enabled": ["value_audit"], "unknown": []}
    assert res[2] == {"enabled": ["value_audit", "model_coverage"], "unknown": ["bogus"]}
    assert res[3]["enabled"] == ["value_audit"]   # a non-array is ignored, never widened


def test_trigger_eligible_rechecks_thresholds(tmp_path):
    cases = [
        _tf("passthrough", "ncaab"),
        _tf("passthrough", "ncaab", value=0.5),                    # not strictly below
        _tf("passthrough", "ncaab", n=9),
        _tf("model_coverage", "ncaab", value=0.49),
        _tf("close_capture", "ncaab", value=0.3),                  # must be > 0.30
        _tf("calibration_bucket", "nba", z=2.4),
        _tf("calibration_bucket", "nba", value=-8.0, z=-4.0),      # under-confidence counts too
        _tf("model_coverage", "ncaab", fires=False),
        _tf("model_coverage", "ncaab", judge_metric="holdout_brier"),
        {**_tf("model_coverage", "ncaab"), "trigger": "value_audit"},
        {**_tf("model_coverage", "ncaab"), "sector": ""},
        None,
    ]
    ok = [r["ok"] for r in _call(tmp_path, "triggerEligible", *[[c] for c in cases])]
    assert ok == [True, False, False, False, False, False, True, False, False, False, False, False]


def test_build_queue_default_keeps_new_triggers_in_shadow(tmp_path):
    [q] = _call(tmp_path, "buildQueue", [AUDIT_GAP["findings"], COVERAGE_TRIGGERS["findings"], ["value_audit"], 3])
    assert [(f["trigger"], f["sector"]) for f in q["queue"]] == [("value_audit", "nba")]
    assert {(f["trigger"], f["sector"]) for f in q["shadow"]} == {
        ("model_coverage", "ncaab"), ("passthrough", "ncaab"), ("passthrough", "tennis"), ("close_capture", "ncaab")}


def test_build_queue_enabled_dedupes_and_round_robins(tmp_path):
    enabled = ["value_audit", "model_coverage", "passthrough", "close_capture"]
    [q] = _call(tmp_path, "buildQueue", [AUDIT_GAP["findings"], COVERAGE_TRIGGERS["findings"], enabled, 10])
    order = [(f["trigger"], f["sector"]) for f in q["queue"]]
    # value audit leads; ncaab passthrough rides along the ncaab coverage finding; one per (trigger, sector)
    assert order == [("value_audit", "nba"), ("model_coverage", "ncaab"), ("passthrough", "tennis"), ("close_capture", "ncaab")]
    cov = q["queue"][1]
    assert cov["model"] == "ncaab_efficiency"
    assert cov["related"] == ["model_coverage ncaab", "passthrough ncaab"]
    assert q["shadow"] == []


def test_build_queue_caps_and_reports_rejected(tmp_path):
    bad = _tf("close_capture", "nhl", n=3)
    [q] = _call(tmp_path, "buildQueue", [[], COVERAGE_TRIGGERS["findings"] + [bad], ["model_coverage", "passthrough"], 1])
    assert len(q["queue"]) == 1
    assert q["rejected"] == [{"trigger": "close_capture", "sector": "nhl", "reason": "n=3 < 10"}]


def test_kind_allowed_and_denied_files(tmp_path):
    kinds = _call(tmp_path, "kindAllowed", ["passthrough", "reweight"], ["passthrough", "coverage_fix"],
                  ["close_capture", "data_fix"], ["close_capture", "coverage_fix"], ["value_audit", "add_feature"],
                  ["calibration_bucket", "reweight"], ["nope", "reweight"])
    assert kinds == [False, True, True, False, True, False, False]
    [denied] = _call(tmp_path, "deniedFiles", [["./data/categories.yaml", "evmax/sectors/aliases/ncaab.yaml",
                                                "data/model_config.json", "evmax/settings.py"]])
    assert denied == ["data/categories.yaml", "data/model_config.json", "evmax/settings.py"]


def test_signal_verdict_per_judge(tmp_path):
    cases = [
        ["value_audit", {"ran": True, "baseline_holdout_brier": 0.21, "candidate_holdout_brier": 0.207}],
        ["value_audit", {"ran": True, "baseline_holdout_brier": 0.21, "candidate_holdout_brier": 0.2095}],
        ["model_coverage", COV_PASS],
        ["model_coverage", {**COV_PASS, "coverage_after": 0.05}],                 # gain below 10 points
        ["model_coverage", {**COV_PASS, "newly_priced_n": 5}],
        ["model_coverage", {**COV_PASS, "harm_z": 2.0}],                         # newly priced rows worse
        ["model_coverage", {**COV_PASS, "coverage_after": 80}],                  # percent, not fraction
        ["model_coverage", {k: v for k, v in COV_PASS.items() if k != "harm_z"}],  # guard missing
        # a coverage fix carrying ONLY a Brier gain is not a coverage win
        ["passthrough", {"ran": True, "baseline_holdout_brier": 0.21, "candidate_holdout_brier": 0.1}],
        ["close_capture", {"ran": True, "coverage_before": 0.6, "coverage_after": 0.9, "newly_priced_n": 6, "rule_violations": 0}],
        ["close_capture", {"ran": True, "coverage_before": 0.6, "coverage_after": 0.9, "newly_priced_n": 6, "rule_violations": 1}],
        ["calibration_bucket", {"ran": True, "bucket_err_before_pp": 8, "bucket_err_after_pp": 3,
                                "baseline_holdout_brier": 0.21, "candidate_holdout_brier": 0.211}],
        ["calibration_bucket", {"ran": True, "bucket_err_before_pp": 8, "bucket_err_after_pp": 3,
                                "baseline_holdout_brier": 0.21, "candidate_holdout_brier": 0.215}],
        ["model_coverage", {"ran": False, "note": "no data"}],
        ["bogus", COV_PASS],
    ]
    res = _call(tmp_path, "signalVerdict", *cases)
    assert [r["pass"] for r in res] == [True, False, True, False, False, False, False, False, False,
                                        True, False, True, False, False, False]


# ---- full graph runs ------------------------------------------------------------

def test_default_run_stops_but_reports_shadow_queue(tmp_path):
    out = _run(tmp_path, {"value-audit": [AUDIT_QUIET], "improve-triggers": [COVERAGE_TRIGGERS]})
    res = out["result"]
    assert res["stopped"] == "no_actionable_gap" and res["enabled"] == ["value_audit"]
    assert {(f["trigger"], f["sector"]) for f in res["shadow_queue"]} >= {("model_coverage", "ncaab"), ("close_capture", "ncaab")}
    assert [c["label"] for c in out["calls"]] == ["value-audit", "improve-triggers"]   # nothing acted on


def test_dry_run_returns_queue_without_hypothesizing(tmp_path):
    out = _run(tmp_path, {"value-audit": [AUDIT_QUIET], "improve-triggers": [COVERAGE_TRIGGERS]},
               {"dry_run": True, "triggers": ["model_coverage", "close_capture"]})
    res = out["result"]
    assert res["decision"] == "dry_run"
    assert [(f["trigger"], f["sector"]) for f in res["queue"]] == [("model_coverage", "ncaab"), ("close_capture", "ncaab")]
    assert [(f["trigger"], f["sector"]) for f in res["shadow_queue"]] == [("passthrough", "tennis")]
    assert all(not c["label"].startswith("hypothesize") for c in out["calls"])


def test_enabled_coverage_trigger_keeps_on_rows_priced(tmp_path):
    out = _run(tmp_path, {
        "value-audit": [AUDIT_QUIET], "improve-triggers": [COVERAGE_TRIGGERS],
        "hypothesize": [SPEC_COVERAGE], "implement": [IMPL_OK], "measure": [COV_PASS],
        "integrity": [ACCEPT], "dispose:keep": [KEPT],
    }, {"triggers": ["model_coverage"]})
    res = out["result"]
    assert res["decision"] == "kept" and res["trigger"] == "model_coverage"
    assert res["judge_metric"] == "rows_priced_frac" and res["delta"] == pytest.approx(0.8)
    assert res["holdout_delta"] is None
    prompts = out["prompts"]
    assert "judged on rows priced, NOT on Brier" in prompts["measure:model_coverage:ncaab#1"]
    assert "loosening a confidence gate" in prompts["integrity:model_coverage:ncaab#1"]
    assert "Do NOT propose" not in prompts["hypothesize:model_coverage:ncaab#1"]  # passthrough-only text
    assert "Loosening a gate" in prompts["hypothesize:model_coverage:ncaab#1"]


def test_disallowed_kind_is_skipped_before_implement(tmp_path):
    out = _run(tmp_path, {
        "value-audit": [AUDIT_QUIET], "improve-triggers": [COVERAGE_TRIGGERS],
        "hypothesize": [{**SPEC_COVERAGE, "kind": "reweight"}],
    }, {"triggers": ["passthrough"]})
    assert out["result"]["attempts"] == [{"trigger": "passthrough", "sector": "ncaab", "gap": "passthrough", "outcome": "kind_not_allowed"},
                                         {"trigger": "passthrough", "sector": "tennis", "gap": "passthrough", "outcome": "kind_not_allowed"}]
    assert not any(c["label"].startswith("implement") for c in out["calls"])


def test_denied_file_reverts_without_measuring(tmp_path):
    out = _run(tmp_path, {
        "value-audit": [AUDIT_QUIET], "improve-triggers": [COVERAGE_TRIGGERS],
        "hypothesize": [SPEC_COVERAGE], "implement": [{**IMPL_OK, "files_changed": ["data/model_config.json"]}],
        "dispose:revert": [REVERTED],
    }, {"triggers": ["model_coverage"]})
    labels = [c["label"] for c in out["calls"]]
    assert not any(lbl.startswith(("measure", "integrity", "dispose:keep")) for lbl in labels)
    assert "dispose:revert:model_coverage:ncaab#1" in labels
    assert out["result"]["decision"] == "no_validated_change"


def test_integrity_reject_beats_a_passing_signal(tmp_path):
    out = _run(tmp_path, {
        "value-audit": [AUDIT_QUIET], "improve-triggers": [COVERAGE_TRIGGERS],
        "hypothesize": [SPEC_COVERAGE], "implement": [IMPL_OK], "measure": [COV_PASS],
        "integrity": [{"accept": False, "reason": "lowered MIN_GAMES"}], "dispose:revert": [REVERTED],
    }, {"triggers": ["model_coverage"]})
    assert out["result"]["attempts"][0]["outcome"] == "reverted"


def test_value_audit_path_unchanged_and_survives_dead_trigger_agent(tmp_path):
    out = _run(tmp_path, {
        "value-audit": [AUDIT_GAP],   # improve-triggers unmatched → returns null (dead agent)
        "hypothesize": [{"sector": "nba", "kind": "reweight", "params": "eff 0.30→0.35", "novel": True}],
        "implement": [{**IMPL_OK, "branch": "model-improve/nba-reweight", "files_changed": ["evmax/agents/models/ensemble_agent.py"]}],
        "measure": [{"ran": True, "baseline_holdout_brier": 0.2100, "candidate_holdout_brier": 0.2070, "note": "wf"}],
        "integrity": [ACCEPT], "dispose:keep": [KEPT],
    })
    res = out["result"]
    assert res["decision"] == "kept" and res["trigger"] == "value_audit"
    assert res["holdout_delta"] == pytest.approx(0.003)
    assert "improve-triggers" in out["unmatched"]
    assert any("trigger findings unavailable" in m for m in out["logs"])
