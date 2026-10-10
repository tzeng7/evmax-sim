"""scripts/opportunity_context.py — the shared snapshot every opportunity agent reads."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from evmax import db_location
from scripts import opportunity_context as C

REPO = Path(__file__).resolve().parents[1]


def _isolate_env(monkeypatch) -> None:
    """build_snapshot sets EVMAX_DB_* in os.environ on purpose (one-shot CLI).

    ``monkeypatch.delenv(raising=False)`` registers no undo for an absent
    variable, so the value build_snapshot sets would leak into later tests.
    ``setenv`` always registers an undo; "" means "unset" to db_location.
    """
    monkeypatch.setenv(db_location.ENV_DB_DIR, "")
    monkeypatch.setenv(db_location.ENV_DB_READONLY, "")


def test_summarize_kalshi_series_buckets_and_ranks():
    our_map = {"nfl": ["KXNFLGAME", "KXNFLGONE"], "nba": ["KXNBAGAME"]}
    series = [
        {"ticker": "KXNFLGAME", "title": "NFL game", "tags": ["Football"], "last_updated_ts": "2026-10-01"},
        {"ticker": "KXNBAGAME", "title": "NBA game", "tags": ["Basketball"], "last_updated_ts": "2026-10-01"},
        {"ticker": "KXNFLGAMEHALF", "title": "NFL first half winner", "tags": ["Football"], "last_updated_ts": "2026-10-05"},
        {"ticker": "KXDARTS", "title": "Darts", "tags": ["Other"], "last_updated_ts": "2026-01-01"},
        {"ticker": "KXCRICKET", "title": "Cricket", "tags": ["Cricket"], "last_updated_ts": "2026-10-08"},
        {"ticker": "KXCHESS", "title": "Chess", "tags": [], "last_updated_ts": "2026-10-09"},
    ]
    out = C.summarize_kalshi_series(series, our_map, max_unwired=2)
    assert out["n_kalshi_sports_series"] == 6 and out["n_wired_ok"] == 2
    assert out["stale"] == [{"ticker": "KXNFLGONE", "sector": "nfl"}]
    assert [s["ticker"] for s in out["unwired_matching_our_sectors"]] == ["KXNFLGAMEHALF"]
    assert [s["ticker"] for s in out["unwired_other"]] == ["KXCHESS", "KXCRICKET"]   # newest first
    assert out["unwired_other_truncated"] == 1
    assert out["unwired_other_by_sport"] == {"untagged": 1, "Cricket": 1, "Other": 1}


def test_doc_index_reads_title_and_verdict(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a-eval.md").write_text("# A eval\n\nintro\n**Verdict: REJECTED.** because\n")
    (docs / "b.md").write_text("no heading\n")
    idx = C.doc_index(docs)
    assert idx == [
        {"path": "docs/a-eval.md", "title": "A eval", "verdict_line": "**Verdict: REJECTED.** because"},
        {"path": "docs/b.md", "title": None, "verdict_line": None},
    ]


def test_memory_index_path_slug():
    p = C.memory_index_path(Path("/Users/someone/Projects/evmax"))
    assert p.parts[-4:] == ("projects", "-Users-someone-Projects-evmax", "memory", "MEMORY.md")


def test_main_checkout_root_and_db_dir(tmp_path):
    main = C.main_checkout_root(REPO)
    assert (main / ".git").exists()
    assert C.main_checkout_root(tmp_path) == tmp_path            # outside git → falls back
    assert C.default_db_dir(tmp_path) == tmp_path / "data"       # nothing found → <main>/data
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "predictions.db").write_text("")
    assert C.default_db_dir(tmp_path) == tmp_path / "data"


def test_build_snapshot_offline_is_fail_soft_and_readonly(monkeypatch, tmp_path):
    _isolate_env(monkeypatch)
    heavy = {"categories", "promotion_board", "value_audit", "integrity", "recent_commits", "memory_index"}
    snap = C.build_snapshot(date(2026, 10, 10), "nfl", tmp_path / "data", offline=True, skip=heavy, root=REPO)
    meta = snap["meta"]
    assert meta["db_readonly"] is True and meta["db_dir"] == str(tmp_path / "data")
    assert meta["sections"]["kalshi_series"] == "skipped" and meta["sections"]["open_prs"] == "skipped"
    assert all(meta["sections"][s] == "skipped" for s in heavy)
    assert meta["sections"]["graveyard"].startswith("ok")
    assert len(snap["graveyard"]) >= 40
    assert {d["path"] for d in snap["eval_docs"]} >= {"docs/opportunity-workflow-scope.md"}
    assert "EVMAX_DB_READONLY=1" in meta["how_to_query_dbs"]
    assert db_location.readonly_enabled() and db_location.resolve_db_path("x.db", Path("/d")) == tmp_path / "data" / "x.db"
    json.dumps(snap, default=str)                                 # serializable


def test_build_snapshot_records_section_errors(monkeypatch, tmp_path):
    _isolate_env(monkeypatch)

    def boom(_root):
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(C, "section_ledger", boom)
    keep = {"ledger"}
    all_sections = {"categories", "promotion_board", "value_audit", "integrity", "kalshi_series", "open_prs",
                    "recent_commits", "eval_docs", "graveyard", "research_sources_seen", "landscape_previous",
                    "memory_index", "ledger"}
    snap = C.build_snapshot(date(2026, 10, 10), None, tmp_path, skip=all_sections - keep, root=REPO)
    assert snap["ledger"] == {"error": "RuntimeError: ledger exploded"}
    assert snap["meta"]["sections"]["ledger"].startswith("error: RuntimeError")


def test_cli_writes_snapshot(monkeypatch, tmp_path, capsys):
    _isolate_env(monkeypatch)
    out = tmp_path / "ctx.json"
    rc = C.main(["--date", "2026-10-10", "--out", str(out), "--db-dir", str(tmp_path), "--offline",
                 "--skip", "categories,promotion_board,value_audit,integrity,recent_commits,memory_index"])
    assert rc == 0
    assert capsys.readouterr().out.strip().endswith(str(out))
    data = json.loads(out.read_text())
    assert data["meta"]["date"] == "2026-10-10" and data["meta"]["focus"] is None
    web = tmp_path / "ctx.web.json"
    assert data["meta"]["web_digest_path"] == str(web)
    digest = json.loads(web.read_text())
    assert len(digest["graveyard"]) >= 40 and web.stat().st_size < 10_000   # small: passed inline as args


def test_web_digest_is_compact_and_skips_errored_sections():
    snap = {
        "meta": {"date": "2026-10-10", "focus": "nfl"},
        "categories": [{"key": "nfl", "effective_mode": "live", "market_types": ["moneyline"], "models": ["elo"],
                        "notes": "long internal notes " * 50}],
        "graveyard": [{"id": "elo-h2h-layer", "verdict": "REJECTED", "idea": "x" * 500, "evidence": "secret-ish detail"}],
        "research_sources_seen": [{"url": f"https://a.example/{i}"} for i in range(200)],
        "kalshi_series": {"error": "HTTPError: 503"},
        "landscape_previous": "| Name | Type |\n|---|---|\n| [Tool A](https://a.example) | tool |\n| Plain B | repo |\n",
    }
    d = C.web_digest(snap)
    assert d["categories"] == [{"key": "nfl", "mode": "live", "markets": ["moneyline"]}]
    assert d["graveyard"] == [{"id": "elo-h2h-layer", "verdict": "REJECTED", "idea": "x" * C.WEB_IDEA_CHARS}]
    assert len(d["research_urls_already_read"]) == C.WEB_MAX_SOURCES
    assert d["research_urls_already_read"][-1] == "https://a.example/199"      # most recent kept
    assert d["kalshi_series"]["unwired_matching_our_sectors"] == []            # errored section → empty
    assert d["previous_competitors"] == ["Tool A", "Plain B"]
    assert "notes" not in json.dumps(d) and "evidence" not in json.dumps(d)

