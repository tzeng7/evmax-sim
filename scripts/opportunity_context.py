#!/usr/bin/env python3
"""Build the opportunity-scout context snapshot (one JSON file every agent reads).

Run by the ``/opportunities`` command BEFORE the discovery workflow starts, so
all agents share one consistent picture of the project instead of each
re-deriving it. See docs/opportunity-workflow-scope.md §3 and §7.

Read-only by construction
    The script sets ``EVMAX_DB_DIR`` + ``EVMAX_DB_READONLY=1`` (evmax/db_location.py)
    before importing any evmax database module. It therefore reads the MAIN
    checkout's predictions.db / archive.db even when it runs from a worktree
    (which has no databases), and any accidental write raises instead of
    touching the file.

Sections (each fail-soft: an error is recorded in place, the rest still run)
    categories          base + effective mode, markets, models per category
    promotion_board     `cleanup shadow board` rows (CLV gates, divergence, verdicts)
    value_audit         `cleanup value-audit` rows (Brier vs sharp/close, CLV, calibration)
    integrity           `cleanup integrity` daily sweep result
    kalshi_series       Kalshi Sports series: wired / stale / unwired (network)
    open_prs            open pull requests (gh; network)
    recent_commits      last 40 commits on this checkout
    eval_docs           docs/*.md title + first verdict line
    graveyard           docs/opportunities/graveyard.yaml
    ledger              latest status per opportunity
    research_sources_seen   URLs already in docs/research-journal/sources.jsonl
    landscape_previous  docs/competitive/landscape.md (previous competitive snapshot)
    memory_index        the owner's Claude memory index for this project, if present

Usage
    uv run python scripts/opportunity_context.py --date 2026-10-09 [--focus "nfl props"]
        [--out PATH] [--db-dir DIR] [--offline] [--skip integrity,kalshi_series]

The last stdout line is the snapshot path. Next to it the script writes
``<date>.web.json`` — a compact digest (categories, graveyard ids + ideas, research
URLs already read, unwired Kalshi series, previous competitor names) that the
command passes INLINE to the two web-facing agents, which get no file access.
A per-section summary goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date as Date
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.opportunity_ledger import (  # noqa: E402  (sys.path set above)
    JOURNAL_SOURCES,
    LANDSCAPE,
    latest_by_id,
    read_graveyard,
    read_ledger,
)

CONTEXT_DIR = Path("docs/opportunities/context")
MAX_UNWIRED_SERIES = 300
WEB_MAX_SOURCES = 150
WEB_MAX_SERIES = 20
WEB_IDEA_CHARS = 70
WEB_TITLE_CHARS = 60
NETWORK_SECTIONS = {"kalshi_series", "open_prs"}


def _git(*args: str, cwd: Path = REPO) -> str:
    cp = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or f"git {' '.join(args)} failed")
    return cp.stdout.strip()


def main_checkout_root(repo: Path = REPO) -> Path:
    """The primary checkout that owns the shared .git directory.

    From a worktree, ``--git-common-dir`` is ``<main>/.git``; from the main
    checkout it is the same path. Falls back to ``repo`` outside git.
    """
    try:
        return Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=repo)).parent
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return repo


def default_db_dir(repo: Path = REPO) -> Path:
    """First of <main>/data, <repo>/data that holds predictions.db; else <main>/data."""
    main = main_checkout_root(repo)
    for cand in (main / "data", repo / "data"):
        if (cand / "predictions.db").exists():
            return cand
    return main / "data"


def memory_index_path(main_root: Path) -> Path:
    """Claude Code's per-project memory index (slug = absolute path with '/' → '-')."""
    slug = str(main_root).replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug / "memory" / "MEMORY.md"


def configure_readonly_env(db_dir: Path) -> None:
    """Point evmax at ``db_dir`` read-only. Must run before evmax DB modules import."""
    os.environ["EVMAX_DB_DIR"] = str(db_dir)
    os.environ["EVMAX_DB_READONLY"] = "1"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _val(x: Any) -> Any:
    return getattr(x, "value", x)


def section_categories(today: Date) -> list[dict[str, Any]]:
    from evmax.categories import all_categories, is_in_season
    from evmax.modes import effective_modes

    eff = effective_modes(today=today)
    out = []
    for c in all_categories():
        out.append({
            "key": c.key,
            "base_mode": _val(c.mode),
            "effective_mode": _val(eff.get(c.key)),
            "status": _val(c.status),
            "market_types": [_val(m) for m in c.market_types],
            "models": list(c.models),
            "resolver": c.resolver,
            "shadow_market_types": list(c.shadow_market_types),
            "disabled_market_types": list(c.disabled_market_types),
            "shadow_venue_market_types": {k: list(v) for k, v in c.shadow_venue_market_types.items()},
            "in_season": is_in_season(c.key, today),
            "notes": (c.notes or "")[:400] or None,
        })
    return out


def section_promotion_board(days: int) -> list[dict[str, Any]]:
    from evmax.agents.cleanup.promotion_board import compute_promotion_board

    return compute_promotion_board(days=days)


def section_value_audit(weeks: int) -> Any:
    from evmax.agents.cleanup.value_audit import compute_value_audit

    return compute_value_audit(weeks=weeks)


def section_integrity() -> dict[str, Any]:
    from evmax.agents.cleanup.integrity import run_integrity

    return run_integrity(weekly=False, check_pinnacle=False, notify=False)


def summarize_kalshi_series(
    series: list[dict[str, Any]],
    our_map: dict[str, list[str]],
    max_unwired: int = MAX_UNWIRED_SERIES,
) -> dict[str, Any]:
    """Pure: classify Kalshi Sports series against SECTOR_SERIES_MAP.

    Unwired series are counted per sport tag, and listed most-recently-updated
    first (``last_updated_ts``) so the truncated tail is the stale one-offs,
    not an alphabetical accident.
    """
    from collections import Counter

    from scripts.check_kalshi_series import classify_series

    by_ticker = {s["ticker"]: s for s in series if s.get("ticker")}
    ok, stale, new = classify_series(our_map, set(by_ticker))
    ours = {t for ts in our_map.values() for t in ts}
    new_set = {t for t, _ in new}

    def slim(ticker: str) -> dict[str, Any]:
        s = by_ticker.get(ticker, {})
        return {"ticker": ticker, "title": s.get("title"), "tags": s.get("tags") or [],
                "frequency": s.get("frequency"), "last_updated_ts": s.get("last_updated_ts")}

    def recency(ticker: str) -> str:
        return str(by_ticker.get(ticker, {}).get("last_updated_ts") or "")

    unwired = sorted((t for t in by_ticker if t not in ours and t not in new_set),
                     key=recency, reverse=True)
    sport_counts = Counter(
        ((by_ticker[t].get("tags") or ["untagged"])[0]) for t in unwired
    )
    return {
        "n_kalshi_sports_series": len(by_ticker),
        "n_wired_ok": len(ok),
        "stale": [{"ticker": t, "sector": s} for t, s in stale],
        "unwired_matching_our_sectors": [
            {**slim(t), "sector": s} for t, s in sorted(new, key=lambda ts: recency(ts[0]), reverse=True)
        ],
        "unwired_other_by_sport": dict(sport_counts.most_common()),
        "unwired_other": [slim(t) for t in unwired[:max_unwired]],
        "unwired_other_truncated": max(0, len(unwired) - max_unwired),
    }


def section_kalshi_series() -> dict[str, Any]:
    import httpx

    from evmax.clients.kalshi import SECTOR_SERIES_MAP
    from scripts.check_kalshi_series import KALSHI_API, TIMEOUT

    r = httpx.get(f"{KALSHI_API}/series", params={"category": "Sports", "limit": 5000}, timeout=TIMEOUT)
    r.raise_for_status()
    return summarize_kalshi_series(r.json().get("series", []), SECTOR_SERIES_MAP)


def section_open_prs() -> list[dict[str, Any]]:
    cp = subprocess.run(
        ["gh", "pr", "list", "--state", "open", "--limit", "40",
         "--json", "number,title,headRefName,createdAt,url"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or "gh pr list failed")
    return json.loads(cp.stdout or "[]")


def section_recent_commits() -> list[str]:
    return _git("log", "--oneline", "-40").splitlines()


def doc_index(docs_dir: Path) -> list[dict[str, Any]]:
    """Title + first verdict-bearing line of every top-level docs/*.md."""
    out = []
    for p in sorted(docs_dir.glob("*.md")):
        title, verdict = None, None
        for line in p.read_text(errors="replace").splitlines():
            if title is None and line.startswith("# "):
                title = line[2:].strip()
            if verdict is None and re.search(r"verdict|rejected|refuted|don.t build", line, re.I):
                verdict = line.strip()[:300]
            if title and verdict:
                break
        out.append({"path": str(p.relative_to(docs_dir.parent)), "title": title, "verdict_line": verdict})
    return out


def section_ledger(root: Path) -> dict[str, Any]:
    rows = read_ledger(root)
    latest = latest_by_id(rows)
    return {
        "n_rows": len(rows),
        "latest": [
            {"id": k, "status": v.get("status"), "title": v.get("title"), "lever": v.get("lever"),
             "ts": v.get("ts"), "status_reason": v.get("status_reason")}
            for k, v in sorted(latest.items(), key=lambda kv: str(kv[1].get("ts", "")))
        ],
    }


def section_research_seen(root: Path) -> list[dict[str, Any]]:
    path = root / JOURNAL_SOURCES
    if not path.exists():
        return []
    seen = []
    for line in path.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            seen.append({"url": e.get("url"), "title": e.get("title"), "date_read": e.get("date_read")})
    return seen


def section_landscape_previous(root: Path) -> str | None:
    path = root / LANDSCAPE
    return path.read_text() if path.exists() else None


def section_memory_index(main_root: Path) -> dict[str, Any] | None:
    path = memory_index_path(main_root)
    if not path.exists():
        return None
    bullets = [ln.strip() for ln in path.read_text().splitlines() if ln.strip().startswith("- ")]
    return {"path": str(path), "entries": bullets}


def web_digest(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The compact, non-secret context the web-facing agents receive inline.

    Those agents have no Read tool (an injected page cannot make them read
    .env and leak it through a URL), so this digest is all they see of the repo.
    Kept small because the command passes it as Workflow args.
    """
    def ok(section: str) -> Any:
        v = snapshot.get(section)
        return None if isinstance(v, dict) and "error" in v and len(v) == 1 else v

    cats = ok("categories") or []
    graveyard = ok("graveyard") or []
    seen = ok("research_sources_seen") or []
    ks = ok("kalshi_series") or {}
    landscape = ok("landscape_previous") or ""
    competitors = re.findall(r"^\| \[?([^\]|]+?)\]?(?:\([^)]*\))? \|", landscape, flags=re.M)
    return {
        "date": (snapshot.get("meta") or {}).get("date"),
        "focus": (snapshot.get("meta") or {}).get("focus"),
        "categories": [{"key": c.get("key"), "mode": c.get("effective_mode"),
                        "markets": c.get("market_types")} for c in cats],
        "graveyard": [{"id": g.get("id"), "verdict": g.get("verdict"),
                       "idea": str(g.get("idea") or "")[:WEB_IDEA_CHARS]} for g in graveyard],
        "research_urls_already_read": [s.get("url") for s in seen][-WEB_MAX_SOURCES:],
        "kalshi_series": {
            "unwired_by_sport": ks.get("unwired_other_by_sport"),
            "unwired_matching_our_sectors": [
                {"ticker": s.get("ticker"), "title": str(s.get("title") or "")[:WEB_TITLE_CHARS]}
                for s in (ks.get("unwired_matching_our_sectors") or [])[:WEB_MAX_SERIES]],
            "unwired_other_recent": [
                {"ticker": s.get("ticker"), "title": str(s.get("title") or "")[:WEB_TITLE_CHARS]}
                for s in (ks.get("unwired_other") or [])[:WEB_MAX_SERIES]],
        },
        "previous_competitors": [c.strip() for c in competitors if c.strip() not in ("Name", "Market")],
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_snapshot(
    run_date: Date,
    focus: str | None,
    db_dir: Path,
    *,
    offline: bool = False,
    skip: set[str] | None = None,
    board_days: int = 30,
    weeks: int = 12,
    root: Path = REPO,
) -> dict[str, Any]:
    configure_readonly_env(db_dir)
    main_root = main_checkout_root(root)
    skip = set(skip or ())
    if offline:
        skip |= NETWORK_SECTIONS

    plan: list[tuple[str, Callable[[], Any]]] = [
        ("categories", lambda: section_categories(run_date)),
        ("promotion_board", lambda: section_promotion_board(board_days)),
        ("value_audit", lambda: section_value_audit(weeks)),
        ("integrity", section_integrity),
        ("kalshi_series", section_kalshi_series),
        ("open_prs", section_open_prs),
        ("recent_commits", section_recent_commits),
        ("eval_docs", lambda: doc_index(root / "docs")),
        ("graveyard", lambda: read_graveyard(root)),
        ("ledger", lambda: section_ledger(root)),
        ("research_sources_seen", lambda: section_research_seen(root)),
        ("landscape_previous", lambda: section_landscape_previous(root)),
        ("memory_index", lambda: section_memory_index(main_root)),
    ]
    snapshot: dict[str, Any] = {}
    status: dict[str, str] = {}
    for name, fn in plan:
        if name in skip:
            status[name] = "skipped"
            continue
        t0 = time.monotonic()
        try:
            snapshot[name] = fn()
            status[name] = f"ok ({time.monotonic() - t0:.1f}s)"
        except Exception as exc:  # noqa: BLE001 — every section is fail-soft by design
            snapshot[name] = {"error": f"{type(exc).__name__}: {exc}"}
            status[name] = f"error: {type(exc).__name__}: {str(exc)[:160]}"

    try:
        sha, branch = _git("rev-parse", "HEAD", cwd=root), _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    except (RuntimeError, OSError, subprocess.SubprocessError):
        sha, branch = None, None
    meta = {
        "date": run_date.isoformat(),
        "focus": focus or None,
        "repo": str(root),
        "main_checkout": str(main_root),
        "db_dir": str(db_dir),
        "db_readonly": True,
        "git_sha": sha,
        "branch": branch,
        "sections": status,
        "how_to_query_dbs": (
            f"Prefix evmax commands with EVMAX_DB_DIR={db_dir} EVMAX_DB_READONLY=1 (DB writes then raise). "
            f"Scripts that take --archive-db/--pred-db open them READ-WRITE with plain sqlite3: pass "
            f"{db_dir}/archive.db or {db_dir}/predictions.db ONLY to eval_*/backtest_*/check_* scripts you "
            "have read and confirmed never write — never to restore_/backfill_/repair_/relabel_/seed_ "
            "scripts. In your own Python use sqlite3.connect('file:<path>?mode=ro', uri=True)."
        ),
    }
    return {"meta": meta, **snapshot}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--date", default=Date.today().isoformat(), help="Run date YYYY-MM-DD")
    p.add_argument("--focus", default=None, help="Optional focus text for the run")
    p.add_argument("--out", default=None, help="Output path (default: docs/opportunities/context/<date>.json)")
    p.add_argument("--db-dir", default=None, help="Directory with predictions.db + archive.db "
                                                  "(default: the main checkout's data/)")
    p.add_argument("--offline", action="store_true", help="Skip network sections (Kalshi, gh)")
    p.add_argument("--skip", default="", help="Comma-separated section names to skip")
    p.add_argument("--board-days", type=int, default=30)
    p.add_argument("--weeks", type=int, default=12)
    args = p.parse_args(argv)

    run_date = Date.fromisoformat(args.date)
    db_dir = Path(args.db_dir).expanduser() if args.db_dir else default_db_dir(REPO)
    snapshot = build_snapshot(
        run_date, args.focus, db_dir,
        offline=args.offline,
        skip={s.strip() for s in args.skip.split(",") if s.strip()},
        board_days=args.board_days, weeks=args.weeks,
    )
    out = Path(args.out) if args.out else REPO / CONTEXT_DIR / f"{run_date.isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    web_out = out.with_name(out.stem + ".web.json")
    snapshot["meta"]["web_digest_path"] = str(web_out)
    out.write_text(json.dumps(snapshot, indent=1, default=str, ensure_ascii=False))
    web_out.write_text(json.dumps(web_digest(snapshot), default=str, ensure_ascii=False, separators=(",", ":")))

    for name, st in snapshot["meta"]["sections"].items():
        print(f"  {name:<22} {st}", file=sys.stderr)
    print(f"  db_dir {db_dir} (read-only) · {out.stat().st_size / 1024:.0f} KB · "
          f"web digest {web_out.stat().st_size / 1024:.1f} KB → {web_out}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
