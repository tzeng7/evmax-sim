"""scripts/opportunity_ledger.py — the one writer for opportunity-scout output."""

from __future__ import annotations

import json
import shutil
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
    assert set(rules) == {"clv_pp_net_fee", "roi_net_fee", "open_close_slope", "brier_paired_vs_sharp",
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
    ({"metric": "brier_paired_vs_sharp", "threshold": -0.001, "min_n_games": 200}, 1),
    ({"metric": "brier_paired_vs_sharp", "threshold": -0.003, "min_n_games": 200}, 0),
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
     "2026-10-10", False, "status"),
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
    for status in ("SUPPORTED", "UNDERPOWERED", "INVALID", "NOT_RUN", "ALLOWED_UNTESTED"):
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
