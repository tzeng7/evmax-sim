"""scripts/opportunity_ledger.py — the one writer for opportunity-scout output."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest
import yaml

from scripts import opportunity_ledger as L

REPO = Path(__file__).resolve().parents[1]
RUN_DATE = "2026-10-10"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A throwaway repo root with the scout workflow (for the metric rules) and a seed graveyard."""
    (tmp_path / L.SCOUT_WORKFLOW).parent.mkdir(parents=True)
    shutil.copy(REPO / L.SCOUT_WORKFLOW, tmp_path / L.SCOUT_WORKFLOW)
    shutil.copy(REPO / L.BUILD_WORKFLOW, tmp_path / L.BUILD_WORKFLOW)
    (tmp_path / L.GRAVEYARD).parent.mkdir(parents=True)
    (tmp_path / L.GRAVEYARD).write_text(
        "# header comment\n\n- id: existing\n  idea: x\n  lever: model\n  sectors: [nba]\n"
        "  verdict: REJECTED\n  date: 2026-01-01\n  evidence: e\n  revisit_if: r\n"
    )
    return tmp_path


def _prereg(**over):
    p = {"metric": "clv_pp_net_fee", "threshold": 0.0, "z_min": 1.64, "min_n_games": 30,
         "train_window": "a", "holdout_window": "b", "comparator": "scan entry", "command": "cmd",
         "declustering": "game"}
    p.update(over)
    return p


def _brief(oid="kalshi-total-lag-20261010", status="SUPPORTED", **over):
    b = {"id": oid, "title": "Kalshi total lag", "lever": "execution", "sectors": ["nfl"],
         "hypothesis": "h", "edge_mechanism": "m", "size": "S", "preregistration": _prereg(),
         "status": status, "status_reason": "why",
         "scope": {"verdict": "allow", "hard_fails": {}, "scores": {"upside": 2}, "reason": "ok"},
         "evidence": {"metric": "clv_pp_net_fee", "value": 0.8, "z_improvement": 2.1, "n_games": 40,
                      "ci_low": 0.3, "ci_high": 1.3, "command": "cmd"},
         "integrity": {"accept": True, "reason": "ok"}}
    b.update(over)
    return b


# ---------------------------------------------------------------------------
# Rules + validation
# ---------------------------------------------------------------------------

def test_metric_rules_load_from_workflow(root):
    rules = L.load_metric_rules(root)
    assert set(rules) == {"clv_pp_net_fee", "roi_net_fee", "open_close_slope", "brier_delta_per_1000",
                          "match_rate", "coverage"}


def test_parse_metric_rules_requires_markers():
    with pytest.raises(ValueError):
        L.parse_metric_rules("const METRIC_RULES = {}")


@pytest.mark.parametrize("over,n_errors", [
    ({}, 0),
    ({"threshold": 0.4, "z_min": 2.0, "min_n_games": 50}, 0),
    ({"threshold": -0.1}, 1),
    ({"z_min": 1.0}, 1),
    ({"min_n_games": 10}, 1),
    ({"min_n_games": "30"}, 1),
    ({"threshold": True}, 1),
    ({"command": "", "comparator": None}, 2),
    ({"metric": "match_rate", "threshold": None}, 1),
    ({"metric": "brier_delta_per_1000", "threshold": -1.0, "min_n_games": 200, "promotion_plan": "clv"}, 1),
    ({"metric": "brier_delta_per_1000", "threshold": -2.5, "min_n_games": 200, "promotion_plan": "clv"}, 0),
    ({"metric": "brier_delta_per_1000", "threshold": -2.5, "min_n_games": 200}, 1),    # no promotion plan
    ({"metric": "match_rate", "threshold": 1.0}, 1),                                     # cannot beat 1.0
    ({"threshold": 60}, 1),                                                              # implausible pp
    ({"train_window": "", "declustering": None}, 2),
])
def test_preregistration_errors(root, over, n_errors):
    errs = L.preregistration_errors(_prereg(**over), L.load_metric_rules(root))
    assert len(errs) == n_errors, errs


def test_preregistration_unknown_metric_and_missing(root):
    rules = L.load_metric_rules(root)
    assert "not one of" in L.preregistration_errors(_prereg(metric="sharpe"), rules)[0]
    assert L.preregistration_errors(None, rules) == ["preregistration missing"]


def test_validate_brief(root):
    rules = L.load_metric_rules(root)
    assert L.validate_brief(_brief(), rules) == []
    errs = L.validate_brief(_brief(id="Bad Id", lever="vibes", size="XL"), rules)
    assert len(errs) == 3
    assert L.validate_brief("nope", rules) == ["brief is not an object"]


# ---------------------------------------------------------------------------
# Ledger + build gate
# ---------------------------------------------------------------------------

def test_append_rejects_unknown_status(root):
    with pytest.raises(ValueError):
        L.append_ledger([{"id": "x", "status": "MAYBE"}], root)


def test_latest_by_id_carries_only_sticky_fields(root):
    L.append_ledger([
        {"ts": "2026-10-01", "id": "a", "status": "SUPPORTED", "status_reason": "beat threshold",
         "brief": {"x": 1}, "evidence": {"value": 1}, "evidence_status": "SUPPORTED"},
        {"ts": "2026-10-02", "id": "a", "status": "BUILDING", "branch": "opp/a"},
        {"ts": "2026-10-02", "id": "b", "status": "REFUTED"},
    ], root)
    latest = L.latest_by_id(L.read_ledger(root))
    a = latest["a"]
    assert a["status"] == "BUILDING" and a["branch"] == "opp/a"
    assert a["brief"] == {"x": 1} and a["evidence"] == {"value": 1}
    assert "status_reason" not in a                   # not inherited from the older row
    assert latest["b"]["status"] == "REFUTED"


@pytest.mark.parametrize("row,today,ok,why", [
    (None, "2026-10-10", False, "unknown"),
    ({"status": "REFUTED", "brief": {"a": 1}, "ts": "2026-10-09"}, "2026-10-10", False, "status"),
    ({"status": "SUPPORTED", "ts": "2026-10-09"}, "2026-10-10", False, "no brief"),
    ({"status": "SUPPORTED", "brief": {"a": 1}, "evidence_date": "2026-09-01"}, "2026-10-10", False, "days old"),
    ({"status": "SUPPORTED", "brief": {"a": 1}, "evidence_date": "2026-10-01"}, "2026-10-10", True, "shadow_feature"),
    ({"status": "UNDERPOWERED", "brief": {"a": 1}, "evidence_date": "2026-10-10"}, "2026-10-10", True, "shadow_collect"),
    ({"status": "BLOCKED", "evidence_status": "SUPPORTED", "brief": {"a": 1}, "evidence_date": "2026-10-08"},
     "2026-10-10", True, "shadow_feature"),
    ({"status": "BUILDING", "evidence_status": "SUPPORTED", "brief": {"a": 1}, "evidence_date": "2026-10-08"},
     "2026-10-10", True, "shadow_feature"),                     # a build whose session died may be retried
    ({"status": "SUPPORTED", "brief": {"a": 1}, "evidence_date": "2026-10-09", "schema_errors": ["min_n_games 5"]},
     "2026-10-10", False, "schema errors"),
    ({"status": "INCONCLUSIVE", "brief": {"a": 1}, "evidence_date": "2026-10-09"}, "2026-10-10", False, "status"),
])
def test_buildable(row, today, ok, why):
    got_ok, got_why = L.buildable(row, date.fromisoformat(today))
    assert got_ok is ok and why in got_why


def test_build_gate_is_compact(root):
    row = {"id": "a", "title": "T", "brief": _brief(), "evidence": _brief()["evidence"]}
    gate = L.build_gate(row, "shadow_feature")
    assert set(gate) == {"opp_id", "title", "build_mode", "preregistration", "evidence", "brief_summary"}
    assert gate["preregistration"]["metric"] == "clv_pp_net_fee"
    assert set(gate["evidence"]) == {"metric", "value", "z_improvement", "n_games", "ci_low", "ci_high", "command"}


# ---------------------------------------------------------------------------
# Graveyard
# ---------------------------------------------------------------------------

def test_graveyard_add_appends_and_dedupes(root):
    entry = {"id": "new-idea", "idea": "i", "lever": "pricing", "sectors": ["nfl"], "verdict": "REJECTED",
             "date": RUN_DATE, "evidence": "e: with colon", "revisit_if": "r"}
    assert L.graveyard_add(entry, root) is True
    assert L.graveyard_add(entry, root) is False
    text = (root / L.GRAVEYARD).read_text()
    assert text.startswith("# header comment")       # existing content and comments untouched
    ids = [e["id"] for e in yaml.safe_load(text)]
    assert ids == ["existing", "new-idea"]


def test_graveyard_add_validates(root):
    with pytest.raises(ValueError):
        L.graveyard_add({"id": "x", "idea": "i", "lever": "model", "sectors": ["a"], "verdict": "MEH",
                         "evidence": "e", "revisit_if": "r"}, root)
    with pytest.raises(ValueError):
        L.graveyard_add({"id": "x"}, root)


def test_graveyard_entry_mapping():
    refuted = L.graveyard_entry_for(_brief(status="REFUTED"), RUN_DATE)
    assert refuted["verdict"] == "REJECTED" and "clv_pp_net_fee = 0.8" in refuted["evidence"]
    blocked = L.graveyard_entry_for(_brief(status="REJECTED_SCOPE", scope={
        "hard_fails": {"requires_auth_or_account": True, "no_measurable_signal": False}}), RUN_DATE)
    assert blocked["verdict"] == "NOT_SUPPORTABLE" and "requires_auth_or_account" in blocked["evidence"]
    soft = _brief(status="REJECTED_SCOPE", scope={"hard_fails": {"changes_bankroll_or_mode": True}})
    assert L.graveyard_entry_for(soft, RUN_DATE) is None
    for status in ("SUPPORTED", "UNDERPOWERED", "INCONCLUSIVE", "INVALID", "NOT_RUN", "ALLOWED_UNTESTED"):
        assert L.graveyard_entry_for(_brief(status=status), RUN_DATE) is None


def test_shipped_graveyard_is_well_formed():
    entries = L.read_graveyard(REPO)
    assert len(entries) >= 40
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    for e in entries:
        assert e["verdict"] in L.GRAVEYARD_VERDICTS, e["id"]
        assert e["lever"] in L.LEVERS, e["id"]
        for key in ("idea", "sectors", "evidence", "revisit_if"):
            assert e.get(key), (e["id"], key)


# ---------------------------------------------------------------------------
# Journal, landscape, report, ingest
# ---------------------------------------------------------------------------

def test_append_journal_dedupes_by_normalized_url(root):
    entries = [{"url": "https://www.example.org/paper/", "title": "P", "fetched": True, "claim": "c"},
               {"url": "http://example.org/paper", "title": "P dup", "fetched": True},
               {"url": "", "title": "no url"}]
    assert L.append_journal(entries, RUN_DATE, root) == 1
    assert L.append_journal(entries, "2026-10-11", root) == 0
    month = (root / L.JOURNAL_DIR / "2026-10.md").read_text()
    assert "Research journal — 2026-10" in month and "**P**" in month


def test_unwrap_workflow_output():
    wrapped = {"summary": "s", "agentCount": 9, "logs": ["Top-K cap dropped 1"], "result": {"briefs": [], "notes": ["a"]}}
    assert L.unwrap_workflow_output(wrapped) == {"briefs": [], "notes": ["a", "log: Top-K cap dropped 1"]}
    plain = {"briefs": []}
    assert L.unwrap_workflow_output(plain) is plain


def test_ingest_writes_every_artifact(root):
    result = {
        "run": {"date": RUN_DATE, "focus": "nfl", "snapshot_path": "/tmp/ctx.json"},
        "proposers": {
            "research": {"candidates": [{}], "notes": "searched",
                         "journal_entries": [{"url": "https://arxiv.org/abs/1", "title": "Paper", "fetched": True}]},
            "competitive": {"candidates": [], "notes": "",
                            "landscape": {"competitors": [{"name": "Tool | A", "url": "https://a.example"}],
                                          "venue_gaps": [{"market": "KXGOLF", "venue": "kalshi"}],
                                          "diff_vs_previous": "first snapshot"}},
            "modeling": {"candidates": [{}, {}], "notes": ""},
        },
        "briefs": [
            _brief(),
            _brief(oid="dead-idea-20261010", status="REFUTED", title="Dead idea"),
            _brief(oid="novig-20261010", status="REJECTED_SCOPE", title="Novig",
                   scope={"verdict": "reject", "hard_fails": {"requires_auth_or_account": True}, "reason": "auth"},
                   evidence=None, integrity=None),
        ],
        "dropped": [{"id": "elo-h2h-20261010", "title": "Elo H2H", "reason": "rejected before",
                     "graveyard_id": "elo-h2h-layer"},
                    {"id": "idea-9-20261010", "title": "Idea 9", "reason": "top-K cap (5)"}],
        "notes": ["note"],
    }
    summary = L.ingest(result, RUN_DATE, root)
    assert summary["report"] == f"docs/opportunities/{RUN_DATE}.md"
    assert summary["ledger_rows"] == 5
    assert summary["graveyard_added"] == ["dead-idea-20261010", "novig-20261010"]
    assert summary["journal_sources_added"] == 1 and summary["landscape_written"] is True
    assert summary["schema_errors"] == {}

    latest = L.latest_by_id(L.read_ledger(root))
    assert latest["kalshi-total-lag-20261010"]["evidence_status"] == "SUPPORTED"
    assert latest["elo-h2h-20261010"]["status"] == "REJECTED_NOVELTY"
    assert latest["idea-9-20261010"]["status"] == "DROPPED"
    assert "status" not in latest["kalshi-total-lag-20261010"]["brief"]

    report = (root / summary["report"]).read_text()
    assert "| 1 | Kalshi total lag (`kalshi-total-lag-20261010`)" in report
    assert "/opportunities build <id>" in report and "`kalshi-total-lag-20261010`" in report
    assert "elo-h2h-layer" in report
    assert "Tool \\| A" in (root / L.LANDSCAPE).read_text()

    # A second ingest the same day gets its own report file.
    assert L.ingest({**result, "briefs": [], "dropped": []}, RUN_DATE, root)["report"].endswith(f"{RUN_DATE}-2.md")


def test_ingest_flags_schema_errors(root):
    bad = _brief(preregistration=_prereg(min_n_games=5))
    summary = L.ingest({"run": {}, "briefs": [bad], "dropped": []}, RUN_DATE, root)
    assert "min_n_games 5 is looser" in summary["schema_errors"]["kalshi-total-lag-20261010"][0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_get_gate_and_record(root, tmp_path, capsys):
    L.ingest({"run": {}, "briefs": [_brief()], "dropped": []}, RUN_DATE, root)
    rc = L.main(["--root", str(root), "get", "--id", "kalshi-total-lag-20261010", "--gate",
                 "--today", "2026-10-12", "--row-out", str(tmp_path / "row.json")])
    assert rc == 0
    gate = json.loads(capsys.readouterr().out)
    assert gate["build_mode"] == "shadow_feature" and gate["evidence"]["value"] == 0.8
    assert json.loads((tmp_path / "row.json").read_text())["id"] == "kalshi-total-lag-20261010"

    assert L.main(["--root", str(root), "get", "--id", "kalshi-total-lag-20261010", "--gate",
                   "--today", "2026-11-30"]) == 2          # evidence too old
    assert L.main(["--root", str(root), "record", "--id", "kalshi-total-lag-20261010",
                   "--status", "PR_OPEN", "--pr-url", "https://github.com/x/y/pull/1", "--date", "2026-10-12"]) == 0
    assert L.main(["--root", str(root), "record", "--id", "ghost", "--status", "BLOCKED"]) == 1
    capsys.readouterr()
    assert L.main(["--root", str(root), "status", "--json"]) == 0
    view = json.loads(capsys.readouterr().out)
    assert view == [{"id": "kalshi-total-lag-20261010", "status": "PR_OPEN", "title": "Kalshi total lag",
                     "ts": "2026-10-12", "pr_url": "https://github.com/x/y/pull/1"}]


def test_cli_validate_brief(root, tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_brief()))
    assert L.main(["--root", str(root), "validate-brief", str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_brief(preregistration=_prereg(threshold=-1))))
    assert L.main(["--root", str(root), "validate-brief", str(bad)]) == 1


def test_cli_ingest_refuses_unexpected_payload(root, tmp_path, capsys):
    bad = tmp_path / "out.json"
    bad.write_text(json.dumps({"summary": "s", "agentCount": 0, "logs": [], "result": {"probe": True}}))
    assert L.main(["--root", str(root), "ingest", str(bad), "--date", RUN_DATE]) == 1
    assert "no 'briefs' list" in capsys.readouterr().err
    assert not (root / L.LEDGER).exists()


# ---------------------------------------------------------------------------
# verify-build: the deterministic ship gate
# ---------------------------------------------------------------------------

def test_live_widening():
    base_c = {"nfl": {"mode": "shadow", "shadow_market_types": ["spread"], "disabled_market_types": ["total"],
                      "shadow_venue_market_types": {"kalshi": ["total"]}},
              "nba": {"mode": "live"}}
    assert L.live_widening(base_c, base_c, {"shadow_leagues": ["ligamx"]}, {"shadow_leagues": ["ligamx"]}, "") == []
    tightened = {**base_c, "nba": {"mode": "shadow"}, "golf": {"mode": "shadow"}}
    assert L.live_widening(base_c, tightened, {}, {}, "") == []
    widened = {"nfl": {"mode": "live", "shadow_market_types": [], "disabled_market_types": [],
                       "shadow_venue_market_types": {}},
               "nba": {"mode": "live"}, "golf": {"mode": "live"}, "odd": {"mode": "weird"}}
    reasons = L.live_widening(base_c, widened, {"shadow_leagues": ["ligamx", "jleague"]},
                              {"shadow_leagues": ["ligamx"]},
                              "--- a/evmax/settings.py\n+++ b/evmax/settings.py\n-    novig_live: bool = False\n"
                              "+    novig_live: bool = True\n+    some_other: int = 1\n")
    text = "\n".join(reasons)
    for needle in ("nfl mode shadow -> live", "shadow_market_types lost", "disabled_market_types lost",
                   "shadow_venue_market_types.kalshi lost", "new category golf", "new category odd",
                   "shadow_leagues lost ['jleague']", "novig_live: bool = True"):
        assert needle in text, needle
    assert "some_other" not in text


def _git(repo: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                        capture_output=True, text=True, check=True)
    return cp.stdout


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    wt = tmp_path / "wt"
    (wt / "data").mkdir(parents=True)
    (wt / "evmax").mkdir()
    (wt / "data" / "categories.yaml").write_text(
        "nfl:\n  mode: shadow\n  shadow_market_types: [spread]\nnba:\n  mode: live\n")
    (wt / "data" / "soccer_league_tiers.yaml").write_text("shadow_leagues: [ligamx]\n")
    (wt / "evmax" / "settings.py").write_text("novig_live: bool = False\n")
    (wt / "evmax" / "old.py").write_text("x = 1\n")
    _git(wt, "init", "-q")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-qm", "base")
    (wt / "evmax" / "x.py").write_text("FLAG = False\n")
    (wt / "tests").mkdir()
    (wt / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    return wt


def _ready_result(**over) -> dict:
    r = {"ready_to_ship": True, "build_mode": "shadow_feature",
         "allowed_files": ["evmax/x.py", "tests/test_x.py"],
         "verification": {"ran": True, "metric": "clv_pp_net_fee", "value": 0.7, "z_improvement": 2.0,
                          "n_games": 40, "leakage_checks": {"utc_et_day": True}}}
    r.update(over)
    return r


def _build_row(**over) -> dict:
    row = {"id": "kalshi-total-lag-20261010", "status": "BUILDING", "evidence_status": "SUPPORTED",
           "evidence_date": RUN_DATE, "brief": _brief(), "evidence": _brief()["evidence"], "schema_errors": []}
    row.update(over)
    return row


def test_verify_build_ok(root, worktree):
    rep = L.verify_build(_ready_result(), _build_row(), worktree, date(2026, 10, 11), root)
    assert rep == {"ok": True, "reasons": [], "files": ["evmax/x.py", "tests/test_x.py"], "build_mode": "shadow_feature"}


def test_verify_build_catches_undeclared_and_denied_files(root, worktree):
    (worktree / "evmax" / "old.py").write_text("x = 2\n")                 # edited, never declared
    (worktree / "data" / "models").mkdir()
    (worktree / "data" / "models" / "elo_state.json").write_text("{}")    # live state
    rep = L.verify_build(_ready_result(), _build_row(), worktree, date(2026, 10, 11), root)
    text = "\n".join(rep["reasons"])
    assert not rep["ok"]
    assert "deny-listed paths changed: ['data/models/elo_state.json']" in text
    assert "evmax/old.py" in text and "nobody declared" in text


def test_verify_build_catches_renames_and_deletions(root, worktree):
    _git(worktree, "mv", "evmax/old.py", "evmax/renamed.py")
    rep = L.verify_build(_ready_result(), _build_row(), worktree, date(2026, 10, 11), root)
    assert {"evmax/old.py", "evmax/renamed.py"} <= set(rep["files"]) and not rep["ok"]


def test_verify_build_catches_live_widening(root, worktree):
    (worktree / "data" / "categories.yaml").write_text("nfl:\n  mode: live\nnba:\n  mode: live\n")
    (worktree / "evmax" / "settings.py").write_text("novig_live: bool = True\n")
    allowed = ["evmax/x.py", "tests/test_x.py", "data/categories.yaml", "evmax/settings.py"]
    rep = L.verify_build(_ready_result(allowed_files=allowed), _build_row(), worktree, date(2026, 10, 11), root)
    text = "\n".join(rep["reasons"])
    assert "category nfl mode shadow -> live" in text and "shadow_market_types lost" in text
    assert "novig_live" in text


def test_verify_build_recomputes_g4_and_mode_from_the_ledger(root, worktree):
    # The workflow claims ready, but the verification value is outside the LEDGER's pre-build CI.
    bad = _ready_result(verification={**_ready_result()["verification"], "value": 2.4})
    rep = L.verify_build(bad, _build_row(), worktree, date(2026, 10, 11), root)
    assert any("G4 (recomputed from the ledger)" in r for r in rep["reasons"])
    # A mistyped build_mode (shadow_collect would reduce G4 to a boolean) is caught.
    rep = L.verify_build(_ready_result(build_mode="shadow_collect"), _build_row(), worktree, date(2026, 10, 11), root)
    assert any("build_mode 'shadow_collect' != ledger mode 'shadow_feature'" in r for r in rep["reasons"])
    # Stale evidence fails the ledger gate.
    rep = L.verify_build(_ready_result(), _build_row(), worktree, date(2026, 12, 1), root)
    assert any(r.startswith("ledger gate:") for r in rep["reasons"])


def test_verify_build_session_checkout_must_be_unchanged(root, worktree, tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _git(session, "init", "-q")
    baseline = _git(session, "status", "--porcelain=v1", "--untracked-files=all")
    rep = L.verify_build(_ready_result(), _build_row(), worktree, date(2026, 10, 11), root, session, baseline)
    assert rep["ok"], rep["reasons"]
    (session / "stray.py").write_text("oops")
    rep = L.verify_build(_ready_result(), _build_row(), worktree, date(2026, 10, 11), root, session, baseline)
    assert any("session checkout" in r for r in rep["reasons"])


def test_cli_verify_build_exit_codes(root, worktree, tmp_path, capsys):
    L.append_ledger([_build_row()], root)
    out = tmp_path / "output.json"
    out.write_text(json.dumps({"summary": "s", "agentCount": 4, "logs": [], "result": _ready_result()}))
    files = tmp_path / "files.txt"
    rc = L.main(["--root", str(root), "verify-build", str(out), "--id", "kalshi-total-lag-20261010",
                 "--worktree", str(worktree), "--today", "2026-10-11", "--files-out", str(files)])
    assert rc == 0, capsys.readouterr().out
    assert files.read_text().split() == ["evmax/x.py", "tests/test_x.py"]
    out.write_text(json.dumps({"summary": "s", "agentCount": 4, "logs": [], "result": _ready_result(ready_to_ship=False)}))
    assert L.main(["--root", str(root), "verify-build", str(out), "--id", "kalshi-total-lag-20261010",
                   "--worktree", str(worktree), "--today", "2026-10-11"]) == 3
