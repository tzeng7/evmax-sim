"""Deterministic orchestration of the opportunity workflows.

`.claude/workflows/opportunity-scout.js` and `opportunity-build.js` keep every gate
(hard fails, pre-registration, score floor, one revision round, top-K/top-N caps,
leakage and units checks, integrity before signal, fix-loop limit, final review,
G4 reproduction) in plain JavaScript so the orchestrator — not an LLM — decides
each transition. These tests run the graphs under tests/workflow_harness.mjs with
stubbed agents and pin that logic, plus its parity with the Python ledger.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.opportunity_ledger import (
    deny_matches,
    load_deny_paths,
    load_metric_rules,
    parse_metric_rules,
    preregistration_errors,
    reproduces,
)

REPO = Path(__file__).resolve().parents[1]
SCOUT = ".claude/workflows/opportunity-scout.js"
BUILD = ".claude/workflows/opportunity-build.js"
HARNESS = REPO / "tests" / "workflow_harness.mjs"
DATE = "2026-10-10"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required for the workflow harness")

HARD_FAILS = [
    "requires_auth_or_account", "requires_antibot_bypass", "requires_tos_violation_or_paywall",
    "touches_live_pricing_without_flag", "changes_bankroll_or_mode", "no_measurable_signal",
    "in_graveyard_without_new_evidence", "edits_eval_or_holdout",
]
SCORE_KEYS = ["edge_mechanism", "net_of_fee", "data_availability", "time_to_evidence",
              "build_size", "architecture_fit", "upside"]
CLEAN = {"utc_et_day": True, "point_in_time": True, "no_future_close": True, "declustered_by_game": True}


def _harness(tmp_path: Path, scenario: dict) -> dict:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    cp = subprocess.run(["node", str(HARNESS), str(path)], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert cp.returncode == 0, cp.stderr
    return json.loads(cp.stdout)


def _call(tmp_path: Path, script: str, fn: str, *argsets: list) -> list:
    out = _harness(tmp_path, {"script": script, "mode": "call",
                              "calls": [{"fn": fn, "args": a} for a in argsets]})
    return out["results"]


def _prereg(metric="clv_pp_net_fee", **over) -> dict:
    p = {"metric": metric, "threshold": 0.0, "z_min": 1.64, "min_n_games": 30,
         "train_window": "2026-09-04..2026-09-30", "holdout_window": "2026-10-01..2026-10-09",
         "comparator": "scan-time entry", "command": "evmax cleanup shadow clv nfl -m total",
         "declustering": "game"}
    p.update(over)
    return p


def _brief(title: str, rank: float = 6, **over) -> dict:
    b = {"title": title, "lever": "execution", "sectors": ["nfl"], "market_types": ["total"],
         "venues": ["kalshi"], "hypothesis": "h", "edge_mechanism": "stale venue quote",
         "sources": [{"kind": "internal", "ref": "evmax cleanup shadow clv nfl -m total"}],
         "data": [{"name": "archive.db", "public": True, "auth_required": False}],
         "preregistration": _prereg(), "size": "S", "graveyard_check": {"matched_ids": []},
         "rank_score": rank, "risks": ["thin sample"]}
    b.update(over)
    return b


def _verdict(oid: str, verdict: str = "allow", score: int = 2, **fails) -> dict:
    return {"id": oid, "verdict": verdict,
            "hard_fails": {k: bool(fails.get(k, False)) for k in HARD_FAILS},
            "scores": {k: score for k in SCORE_KEYS}, "required_changes": ["tighten"], "reason": "r"}


def _metrics(value=0.8, z=2.1, n=40, ran=True, metric="clv_pp_net_fee", **over) -> dict:
    m = {"ran": ran, "metric": metric, "value": value, "z_improvement": z, "n_games": n,
         "ci_low": 0.3, "ci_high": 1.3, "command": "cmd", "note": "n", "leakage_checks": dict(CLEAN)}
    m.update(over)
    return m


def _id(title: str, seq: int = 1) -> str:
    stem = "-".join(title.lower().split())
    return f"{stem}{'-r' + str(seq) if seq > 1 else ''}-20261010"


def _scout_args(**over) -> dict:
    a = {"date": DATE, "focus": "nfl", "snapshot_path": "/tmp/ctx.json", "db_dir": "/tmp/data",
         "web_context": {"graveyard": [{"id": "elo-h2h-layer", "idea": "Elo H2H"}], "categories": [{"key": "nfl"}]},
         "git_baseline_path": "/tmp/baseline.txt"}
    a.update(over)
    return a


def _proposers(n_candidates: int = 1) -> dict:
    cand = {"title": "c", "lever": "execution", "hypothesis": "h", "edge_mechanism": "m", "refs": ["x"]}
    out = {"candidates": [cand] * n_candidates, "notes": ""}
    return {"propose:research": [out], "propose:competitive": [out], "propose:modeling": [out]}


# ---------------------------------------------------------------------------
# Single sources of truth shared by JS and Python
# ---------------------------------------------------------------------------

def test_metric_rules_identical_in_both_workflows():
    scout = parse_metric_rules((REPO / SCOUT).read_text())
    build = parse_metric_rules((REPO / BUILD).read_text())
    assert scout == build == load_metric_rules(REPO)
    assert scout["clv_pp_net_fee"]["min_n_games"] == 30
    assert scout["brier_delta_per_1000"] == {**scout["brier_delta_per_1000"], "direction": "lower", "threshold": -2.0}
    assert all("units" in r and len(r["value_range"]) == 2 for r in scout.values())


PREREG_CASES = [
    _prereg(),                                                   # valid
    _prereg(threshold=0.5, z_min=2.5, min_n_games=60),           # tightened
    _prereg(threshold=-0.2),                                     # looser threshold
    _prereg(z_min=1.0),                                          # looser z
    _prereg(min_n_games=12),                                     # looser n
    _prereg(metric="brier_delta_per_1000", threshold=-1.0, min_n_games=200, promotion_plan="clv"),  # looser (lower-is-better)
    _prereg(metric="brier_delta_per_1000", threshold=-2.5, min_n_games=400, promotion_plan="clv"),
    _prereg(metric="brier_delta_per_1000", threshold=-0.002, min_n_games=200, promotion_plan="clv"),  # raw units: looser
    _prereg(metric="brier_delta_per_1000", threshold=-2.5, min_n_games=200),  # no promotion plan
    _prereg(metric="match_rate", threshold=None),                # null-default metric needs a threshold
    _prereg(metric="match_rate", threshold=0.9),
    _prereg(metric="match_rate", threshold=0),                   # outside (0, 1)
    _prereg(metric="clv_pp_net_fee", threshold=80),              # outside plausible range
    _prereg(metric="sharpe"),                                    # unknown metric
    _prereg(metric="constructor"),                               # prototype key must not pass
    _prereg(command="", comparator="", train_window="", declustering=""),
    {k: v for k, v in _prereg().items() if k != "metric"},
    None,
]


def test_prereg_errors_match_python_message_for_message(tmp_path):
    rules = load_metric_rules(REPO)
    js = _call(tmp_path, SCOUT, "preregErrors", *[[c] for c in PREREG_CASES])
    py = [preregistration_errors(c, rules) for c in PREREG_CASES]
    assert js == py
    assert [bool(e) for e in js] == [False, False, True, True, True, True, False, True, True, True, False,
                                     True, True, True, True, True, True, True]


def test_deny_list_identical_in_js_and_python(tmp_path):
    corpus = ["evmax/x.py", "data/models/elo_state.json", "./data/model_config.json", "data/predictions.db-wal",
              "data/archive.db", "tests/test_x.py", "data/categories.yaml", ".claude/workflows/opportunity-scout.js",
              ".env", "config/.env.local", "scripts/opportunity_ledger.py", "scripts/seed_x.py",
              "docs/opportunities/graveyard.yaml", "docs/other.md", "tests/test_opportunity_ledger.py",
              "tests/workflow_harness.mjs", "tests/fixtures/ncaaf_holdout_2025.json"]
    [js_bad] = _call(tmp_path, BUILD, "forbiddenFiles", [corpus])
    assert js_bad == deny_matches(corpus, load_deny_paths(REPO))
    assert set(js_bad) == set(corpus) - {"evmax/x.py", "tests/test_x.py", "data/categories.yaml",
                                         "scripts/seed_x.py", "docs/other.md"}


def test_reproduces_matches_python(tmp_path):
    rules = load_metric_rules(REPO)
    pre_ev = {"value": 0.8, "ci_low": 0.3, "ci_high": 1.3, "n_games": 40}
    pre = _prereg()
    cases = [
        ["shadow_feature", pre, pre_ev, _metrics(value=0.7)],
        ["shadow_feature", pre, pre_ev, _metrics(value=2.0)],                 # outside CI
        ["shadow_feature", pre, pre_ev, _metrics(value=-0.1)],                # wrong side
        ["shadow_feature", pre, pre_ev, _metrics(value=0.7, n=10)],           # too few games
        ["shadow_feature", pre, {"value": 0.8}, _metrics(value=0.75)],        # no CI → 25% of margin
        ["shadow_feature", pre, {"value": 0.8}, _metrics(value=0.5)],
        ["shadow_feature", pre, pre_ev, _metrics(value=0.7, metric="roi_net_fee")],
        ["shadow_feature", pre, pre_ev, _metrics(value=0.7, leakage_checks={**CLEAN, "point_in_time": False})],
        ["shadow_feature", pre, pre_ev, _metrics(value=70.0)],                # implausible units
        ["shadow_collect", pre, None, {"shadow_collect_ok": True}],
        ["shadow_collect", pre, None, {"shadow_collect_ok": False, "note": "no rows"}],
        ["shadow_feature", pre, pre_ev, None],
        ["live", pre, pre_ev, _metrics(value=0.7)],
    ]
    js = [r["ok"] for r in _call(tmp_path, BUILD, "reproduces", *cases)]
    py = [reproduces(*c, rules)[0] for c in cases]
    assert js == py == [True, False, False, False, True, False, False, False, False, True, False, False, False]


# ---------------------------------------------------------------------------
# Pure gates
# ---------------------------------------------------------------------------

def test_signal_verdict_branches(tmp_path):
    pre = _prereg()
    brier = _prereg(metric="brier_delta_per_1000", threshold=-2.0, min_n_games=200, promotion_plan="clv")
    res = _call(tmp_path, SCOUT, "signalVerdict",
                [pre, _metrics()],                                   # SUPPORTED
                [pre, _metrics(n=20)],                               # right direction, n short
                [pre, _metrics(value=-3.0, z=-2.5, n=12)],           # wrong direction, n short
                [pre, _metrics(value=-0.4, z=-1.0)],                 # wrong side, enough games
                [pre, _metrics(z=1.2)],                              # z short
                [pre, _metrics(metric="roi_net_fee")],               # metric switch
                [pre, _metrics(metric="")],                          # empty metric string is a switch too
                [pre, _metrics(ran=False)],
                [pre, _metrics(z=None)],                             # z required
                [pre, _metrics(value=0.0)],                          # must strictly beat 0
                [pre, _metrics(leakage_checks={**CLEAN, "declustered_by_game": False})],
                [pre, _metrics(leakage_checks={})],                  # unreported leakage = not passed
                [pre, _metrics(value=75.0)],                         # implausible units
                [brier, _metrics(metric="brier_delta_per_1000", value=-2.6, z=1.8, n=500)],
                [brier, _metrics(metric="brier_delta_per_1000", value=-1.1, z=1.8, n=500)],   # −1.1/1000 fails −2
                [brier, _metrics(metric="brier_delta_per_1000", value=-0.0011, z=1.8, n=500)],  # raw units: refuted, not supported
                [_prereg(threshold=-5.0), _metrics(value=-1.0)])     # loosened threshold is clamped to 0
    assert [r["status"] for r in res] == [
        "SUPPORTED", "UNDERPOWERED", "INCONCLUSIVE", "REFUTED", "UNDERPOWERED", "INVALID", "INVALID",
        "NOT_RUN", "INVALID", "REFUTED", "INVALID", "INVALID", "INVALID", "SUPPORTED", "REFUTED", "REFUTED",
        "REFUTED",
    ]


def test_scope_gate_overrides_validator(tmp_path):
    b = _brief("x")
    auth = _brief("x", data=[{"name": "novig api", "public": False, "auth_required": True}])
    gy = _brief("x", graveyard_check={"matched_ids": ["elo-h2h-layer"], "why_different": " "})
    malformed_hf = {**_verdict("a"), "hard_fails": {**_verdict("a")["hard_fails"], "no_measurable_signal": "false"}}
    missing_hf = {k: v for k, v in _verdict("a").items() if k != "hard_fails"}
    bad_scores = {**_verdict("a"), "scores": {**_verdict("a")["scores"], "upside": 7}}
    res = _call(tmp_path, SCOUT, "applyScopeGate",
                [_verdict("a"), [], 1, b],
                [_verdict("a", requires_auth_or_account=True), [], 1, b],          # hard fail beats allow
                [_verdict("a", in_graveyard_without_new_evidence=True), [], 1, b],
                [_verdict("a"), [], 1, auth],                                     # auth derived from the brief
                [malformed_hf, [], 1, b],                                         # "false" string fails closed
                [missing_hf, [], 1, b],
                [bad_scores, [], 1, b],
                [{**_verdict("a"), "scores": {**_verdict("a")["scores"], "net_of_fee": 0}}, [], 1, b],
                [{**_verdict("a"), "scores": {**_verdict("a")["scores"], "edge_mechanism": 1}}, [], 1, b],
                [_verdict("a"), [], 1, gy],                                       # graveyard match, no why_different
                [_verdict("a"), ["threshold looser"], 1, b],                       # prereg error forces revise
                [_verdict("a", "revise"), [], 2, b],                               # second revise → reject
                [None, [], 1, b],
                [{**_verdict("a"), "verdict": "maybe"}, [], 1, b])
    assert [(r["decision"], r["status"]) for r in res] == [
        ("allow", "ALLOWED_UNTESTED"),
        ("reject", "REJECTED_SCOPE"),
        ("reject", "REJECTED_NOVELTY"),
        ("reject", "REJECTED_SCOPE"),
        ("reject", "REJECTED_SCOPE"),
        ("reject", "REJECTED_SCOPE"),
        ("reject", "REJECTED_SCOPE"),
        ("reject", "REJECTED_SCOPE"),
        ("revise", "PROPOSED"),
        ("revise", "PROPOSED"),
        ("revise", "PROPOSED"),
        ("reject", "REJECTED_SCOPE"),
        ("reject", "REJECTED_SCOPE"),
        ("reject", "REJECTED_SCOPE"),
    ]
    assert "requires_auth_or_account" in res[3]["reason"]
    assert "malformed" in res[4]["reason"] and "malformed" in res[5]["reason"]


def test_assign_ids_deterministic_unique_and_run_scoped(tmp_path):
    [ids, ids2] = _call(tmp_path, SCOUT, "assignIds",
                        [[{"title": "NFL Total Lag!"}, {"title": "nfl total lag"}, {"title": ""}], DATE],
                        [[{"title": "NFL Total Lag"}], DATE, [], 2])
    assert [b["id"] for b in ids] == ["nfl-total-lag-20261010", "nfl-total-lag-2-20261010", "opportunity-20261010"]
    assert [b["id"] for b in ids2] == ["nfl-total-lag-r2-20261010"]


def test_build_contradictory_outputs_do_not_count(tmp_path):
    ok_review = {"verdict": "ACCEPT", "live_behavior_unchanged": True, "issues": [], "reason": "ok"}
    res = _call(tmp_path, BUILD, "reviewAccepted",
                [ok_review],
                [{**ok_review, "issues": [{"severity": "blocker", "problem": "p"}]}],
                [{**ok_review, "live_behavior_unchanged": False}],
                [{**ok_review, "reward_hack_suspected": True}],
                [None])
    assert res == [True, False, False, False, False]
    res = _call(tmp_path, BUILD, "testPassed",
                [{"passed": True, "failures": []}], [{"passed": True, "failures": ["x"]}], [{"passed": False, "failures": []}], [None])
    assert res == [True, False, False, False]


# ---------------------------------------------------------------------------
# Discovery graph end to end (stubbed agents)
# ---------------------------------------------------------------------------

def test_scout_happy_path(tmp_path):
    a, b = "Kalshi total lag", "Maker NFL rungs"
    out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": _scout_args(), "responses": {
        **_proposers(),
        "synthesize": [{"briefs": [_brief(a, 7), _brief(b, 5)],
                        "dropped": [{"title": "Elo H2H", "reason": "rejected", "graveyard_id": "elo-h2h-layer"}]}],
        "scope": [{"verdicts": [_verdict(_id(a)), _verdict(_id(b))]}],
        f"backtest:{_id(a)}": [_metrics()],
        f"backtest:{_id(b)}": [_metrics(value=-0.5, z=-1.1)],
        "integrity:": [{"accept": True, "reason": "honest"}],
    }})
    res = out["result"]
    status = {x["id"]: x["status"] for x in res["briefs"]}
    assert status == {_id(a): "SUPPORTED", _id(b): "REFUTED"}
    assert res["dropped"][0]["id"] == "elo-h2h-20261010"
    assert res["dropped"][0]["graveyard_id"] == "elo-h2h-layer"
    assert len(out["calls"]) == 9                       # within the <10-agent guideline
    types = {c["label"]: c["agentType"] for c in out["calls"]}
    assert types["propose:research"] == "opportunity-researcher"
    assert types["scope"] == "opportunity-validator"
    assert types[f"backtest:{_id(a)}"] == "evmax-backtester"
    assert types[f"integrity:{_id(a)}"] == "iteration-reviewer"
    assert not any(t in ("implementer", "change-validator") for t in types.values())   # discovery never builds
    assert out["unmatched"] == []
    # Web-facing agents get the inline digest, never the snapshot path; the others get the path.
    for label in ("propose:research", "propose:competitive"):
        assert "/tmp/ctx.json" not in out["prompts"][label] and "elo-h2h-layer" in out["prompts"][label]
    assert "/tmp/ctx.json" in out["prompts"]["propose:modeling"]
    assert "/tmp/baseline.txt" in out["prompts"][f"integrity:{_id(a)}"]


def test_scout_hard_fail_and_integrity_reject(tmp_path):
    a, b = "Novig feed", "Prop news lag"
    out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": _scout_args(), "responses": {
        **_proposers(),
        "synthesize": [{"briefs": [_brief(a, 9), _brief(b, 5)], "dropped": []}],
        "scope": [{"verdicts": [_verdict(_id(a), requires_auth_or_account=True), _verdict(_id(b))]}],
        "backtest:": [_metrics()],
        "integrity:": [{"accept": False, "reason": "holdout window changed"}],
    }})
    briefs = {x["id"]: x for x in out["result"]["briefs"]}
    assert briefs[_id(a)]["status"] == "REJECTED_SCOPE"
    assert "requires_auth_or_account" in briefs[_id(a)]["status_reason"]
    assert briefs[_id(b)]["status"] == "INVALID"           # G2a runs before G2b and wins
    assert not any(c["label"] == f"backtest:{_id(a)}" for c in out["calls"])


def test_scout_integrity_agent_failure_keeps_evidence(tmp_path):
    a = "Kalshi total lag"
    for integrity in ([{"__throw__": "api error"}], None):
        responses = {**_proposers(),
                     "synthesize": [{"briefs": [_brief(a)], "dropped": []}],
                     "scope": [{"verdicts": [_verdict(_id(a))]}],
                     "backtest:": [_metrics()]}
        if integrity:
            responses["integrity:"] = integrity
        out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": _scout_args(), "responses": responses})
        [brief] = out["result"]["briefs"]
        assert brief["status"] == "INVALID" and "integrity reviewer returned nothing" in brief["status_reason"]
        assert brief["evidence"]["value"] == 0.8             # stage-1 metrics survive


def test_scout_revise_round_then_allow_and_second_revise_rejects(tmp_path):
    a, b = "Loose prereg", "Still vague"
    loose = _brief(a, 8, preregistration=_prereg(min_n_games=10))
    fixed = {**_brief(a, 8), "id": _id(a)}
    out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": _scout_args(), "responses": {
        **_proposers(),
        "synthesize": [{"briefs": [loose, _brief(b, 6)], "dropped": []}],
        "scope": [{"verdicts": [_verdict(_id(a)), _verdict(_id(b), "revise")]}],
        "synthesize:revise": [{"briefs": [fixed, {**_brief(b, 6), "id": _id(b)}], "dropped": []}],
        "scope:revise": [{"verdicts": [_verdict(_id(a)), _verdict(_id(b), "revise")]}],
        "backtest:": [_metrics()],
        "integrity:": [{"accept": True, "reason": "ok"}],
    }})
    status = {x["id"]: x["status"] for x in out["result"]["briefs"]}
    assert status == {_id(a): "SUPPORTED", _id(b): "REJECTED_SCOPE"}
    labels = [c["label"] for c in out["calls"]]
    assert labels.count("scope") == 1 and labels.count("scope:revise") == 1   # exactly one revision round


def test_scout_revise_round_with_no_synthesizer_output_rejects(tmp_path):
    a = "Needs work"
    out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": _scout_args(), "responses": {
        **_proposers(),
        "synthesize": [{"briefs": [_brief(a)], "dropped": []}],
        "scope": [{"verdicts": [_verdict(_id(a), "revise")]}],
        "synthesize:revise": [None],
    }})
    [brief] = out["result"]["briefs"]
    assert brief["status"] == "REJECTED_SCOPE" and "did not return a revision" in brief["status_reason"]
    assert "scope:revise" not in [c["label"] for c in out["calls"]]


def test_scout_caps_are_logged(tmp_path):
    titles = [f"Idea {i}" for i in range(4)]
    out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": _scout_args(top_k=3, top_n=1), "responses": {
        **_proposers(),
        "synthesize": [{"briefs": [_brief(t, 9 - i) for i, t in enumerate(titles)], "dropped": []}],
        "scope": [{"verdicts": [_verdict(_id(t), score=3 - (i % 2)) for i, t in enumerate(titles[:3])]}],
        "backtest:": [_metrics()],
        "integrity:": [{"accept": True, "reason": "ok"}],
    }})
    res = out["result"]
    status = {x["id"]: x["status"] for x in res["briefs"]}
    assert status == {_id("Idea 0"): "SUPPORTED", _id("Idea 1"): "ALLOWED_UNTESTED", _id("Idea 2"): "ALLOWED_UNTESTED"}
    assert [d["id"] for d in res["dropped"]] == [_id("Idea 3")]
    assert any("Top-K cap" in m for m in out["logs"]) and any("Top-N cap" in m for m in out["logs"])


def test_scout_run_seq_and_missing_web_context(tmp_path):
    a = "Kalshi total lag"
    args = {k: v for k, v in _scout_args(run_seq=3).items() if k != "web_context"}
    out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": args, "responses": {
        **_proposers(),
        "synthesize": [{"briefs": [_brief(a)], "dropped": []}],
        "scope": [{"verdicts": [_verdict(_id(a, 3))]}],
        "backtest:": [_metrics()],
        "integrity:": [{"accept": True, "reason": "ok"}],
    }})
    assert [b["id"] for b in out["result"]["briefs"]] == [_id(a, 3)]
    assert any("web_context missing" in n for n in out["result"]["notes"])


def test_scout_no_candidates_and_missing_args(tmp_path):
    empty = {"candidates": [], "notes": "nothing new"}
    out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": _scout_args(), "responses": {
        "propose:research": [empty], "propose:competitive": [empty], "propose:modeling": [None],
    }})
    assert out["result"]["briefs"] == []
    assert "modeling agent returned nothing" in out["result"]["notes"]
    assert [c["label"] for c in out["calls"]] == ["propose:research", "propose:competitive", "propose:modeling"]

    bad = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": {"date": DATE}, "responses": {}})
    assert "required" in bad["result"]["error"] and bad["calls"] == []


def test_scout_two_research_lenses(tmp_path):
    out = _harness(tmp_path, {"script": SCOUT, "mode": "run", "args": _scout_args(research_agents=2), "responses": {
        "propose:": [{"candidates": [], "notes": ""}],
    }})
    assert {c["label"] for c in out["calls"]} == {"propose:research", "propose:research_2",
                                                  "propose:competitive", "propose:modeling"}


# ---------------------------------------------------------------------------
# Build graph end to end (stubbed agents)
# ---------------------------------------------------------------------------

def _build_args(mode: str = "shadow_feature") -> dict:
    return {"date": DATE, "opp_id": "kalshi-total-lag-20261010", "build_mode": mode, "title": "Kalshi total lag",
            "row_path": "/tmp/row.json", "worktree": "/tmp/wt", "branch": "opp/kalshi-total-lag-20261010",
            "db_dir": "/tmp/data",
            "gate": {"preregistration": _prereg(),
                     "evidence": {"metric": "clv_pp_net_fee", "value": 0.8, "z_improvement": 2.1, "n_games": 40,
                                  "ci_low": 0.3, "ci_high": 1.3, "command": "cmd"},
                     "brief_summary": {"hypothesis": "h", "edge_mechanism": "m", "lever": "execution",
                                       "sectors": ["nfl"], "risks": ["thin"]}}}


IMPL = {"files_changed": ["evmax/x.py", "tests/test_x.py"], "diff_summary": "d", "shadow_guarantee": "flag off",
        "flags_added": ["X_ENABLED"]}
ACCEPT = {"verdict": "ACCEPT", "live_behavior_unchanged": True, "issues": [], "files_reviewed": ["evmax/x.py"], "reason": "ok"}
REJECT = {"verdict": "REJECT", "live_behavior_unchanged": True, "files_reviewed": ["evmax/x.py"],
          "issues": [{"severity": "major", "problem": "missing edge-case test"}], "reason": "no"}
PASS = {"passed": True, "commands_run": ["pytest tests/ -q"], "tests_added": ["tests/test_y.py"], "failures": []}
VERIFY_OK = {**_metrics(value=0.7), "code_path_used": "evmax.x.f"}


def test_build_happy_path_reviewer_runs_last(tmp_path):
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": _build_args(), "responses": {
        "implement": [IMPL], "test:": [PASS], "verify:": [VERIFY_OK], "review:": [ACCEPT],
    }})
    res = out["result"]
    assert res["ready_to_ship"] is True and res["status"] == "READY" and res["build_mode"] == "shadow_feature"
    assert [c["label"] for c in out["calls"]] == ["implement", "test:1", "verify:1", "review:1"]
    assert [c["agentType"] for c in out["calls"]] == ["implementer", "test-runner", "evmax-backtester", "change-validator"]
    assert res["allowed_files"] == ["evmax/x.py", "tests/test_x.py", "tests/test_y.py"]
    assert res["pr_title"].startswith("opportunity(shadow): ")
    assert "Generated with [Claude Code]" in res["pr_body"] and "This PR promotes nothing" in res["pr_body"]
    for label in ("implement", "test:1", "verify:1", "review:1"):
        assert "git stash" in out["prompts"][label] and "cd /tmp/wt" in out["prompts"][label]


def test_build_fix_loop_then_accept(tmp_path):
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": _build_args(), "responses": {
        "implement": [IMPL], "test:": [PASS], "verify:": [VERIFY_OK], "review:1": [REJECT], "review:2": [ACCEPT],
    }})
    assert out["result"]["ready_to_ship"] is True
    assert [c["label"] for c in out["calls"]] == ["implement", "test:1", "verify:1", "review:1",
                                                  "implement:fix-1", "test:2", "verify:2", "review:2"]


def test_build_accept_with_blocker_and_pass_with_failures_loop(tmp_path):
    blocker_accept = {**ACCEPT, "issues": [{"severity": "blocker", "problem": "weakened assertion"}]}
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": _build_args(), "responses": {
        "implement": [IMPL], "test:1": [{**PASS, "failures": ["test_a"]}], "test:": [PASS],
        "verify:": [VERIFY_OK], "review:2": [blocker_accept], "review:": [ACCEPT],
    }})
    # loop 1: tests report failures despite passed=true → fix; loop 2: ACCEPT with a blocker → fix;
    # loop 3: clean ACCEPT → ready. Neither contradictory output counted as a pass.
    labels = [c["label"] for c in out["calls"]]
    assert labels == ["implement", "test:1", "implement:fix-1", "test:2", "verify:2", "review:2",
                      "implement:fix-2", "test:3", "verify:3", "review:3"]
    assert out["result"]["ready_to_ship"] is True
    history = json.dumps(out["result"]["history"])
    assert "passed=true with failures" in history and "internally inconsistent" in history


def test_build_forbidden_file_blocks_after_fix_limit(tmp_path):
    bad_impl = {**IMPL, "files_changed": ["evmax/x.py", "data/models/elo_state.json"]}
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": _build_args(), "responses": {
        "implement": [bad_impl], "test:": [PASS], "verify:": [VERIFY_OK], "review:": [ACCEPT],
    }})
    res = out["result"]
    assert res["ready_to_ship"] is False and res["status"] == "BLOCKED"
    labels = [c["label"] for c in out["calls"]]
    assert labels == ["implement", "implement:fix-1", "implement:fix-2"]
    assert "forbidden path" in json.dumps(res["history"])


def test_build_live_behavior_unconfirmed_never_ships(tmp_path):
    unsure = {**ACCEPT, "live_behavior_unchanged": False}
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": _build_args(), "responses": {
        "implement": [IMPL], "test:": [PASS], "verify:": [VERIFY_OK], "review:": [unsure],
    }})
    assert out["result"]["status"] == "BLOCKED"
    assert [c["label"] for c in out["calls"]].count("review:3") == 1


def test_build_g4_not_reproduced_blocks_without_iterating(tmp_path):
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": _build_args(), "responses": {
        "implement": [IMPL], "test:": [PASS], "verify:": [{**VERIFY_OK, "value": 2.5}], "review:": [ACCEPT],
    }})
    res = out["result"]
    assert res["ready_to_ship"] is False and "outside pre-build CI" in res["reason"]
    assert [c["label"] for c in out["calls"]] == ["implement", "test:1", "verify:1"]


def test_build_shadow_collect(tmp_path):
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": _build_args("shadow_collect"), "responses": {
        "implement": [IMPL], "test:": [PASS], "review:": [ACCEPT],
        "verify:": [{"ran": True, "command": "pytest", "code_path_used": "log_gaps", "note": "ok", "shadow_collect_ok": True}],
    }})
    assert out["result"]["ready_to_ship"] is True
    assert out["result"]["pr_title"].startswith("opportunity(shadow-collect): ")


def test_build_rejects_bad_args(tmp_path):
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": {**_build_args(), "build_mode": "live"}, "responses": {}})
    assert out["result"]["status"] == "BLOCKED" and out["calls"] == []
    bad_metric = _build_args()
    bad_metric["gate"] = {**bad_metric["gate"], "preregistration": _prereg(metric="constructor")}
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": bad_metric, "responses": {}})
    assert out["result"]["status"] == "BLOCKED" and out["calls"] == []
    out = _harness(tmp_path, {"script": BUILD, "mode": "run", "args": {"date": DATE}, "responses": {}})
    assert "missing args" in out["result"]["reason"]
