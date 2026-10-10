#!/usr/bin/env python3
"""Opportunity-scout ledger, graveyard, journal and report writer.

The opportunity-scout workflows (``.claude/workflows/opportunity-scout.js`` and
``opportunity-build.js``) return structured JSON and never write to the repo.
This script is the ONE writer for everything they produce, so every file
change is deterministic and reviewable. See docs/opportunity-workflow-scope.md.

Subcommands
    ingest RESULT.json      discovery result → ledger rows, graveyard additions,
                            research-journal sources, competitive landscape and
                            the dated run report (docs/opportunities/<date>.md)
    status [--id ID]        latest status per opportunity (``--json`` for agents)
    get --id ID             latest ledger row carrying the brief, as JSON;
                            ``--check-buildable --today D`` exits 2 when the
                            opportunity may not enter a build run
    record --id ID --status S   append one status transition (build runs)
    verify-build OUTPUT --id ID --worktree WT
                            deterministic ship gate for a build run: re-checks G4
                            against the ledger, the actual changed files against the
                            deny list and the build's allowlist, live-mode widening,
                            and (optionally) that the session checkout is untouched;
                            exits 3 when the build may not ship
    graveyard-add ...       append one graveyard entry by hand
    validate-brief FILE     check a brief's structure and pre-registration

Files (all tracked)
    docs/opportunities/ledger.jsonl        append-only status log
    docs/opportunities/graveyard.yaml      rejected / refuted / unsupportable ideas
    docs/opportunities/<date>.md           one report per discovery run
    docs/research-journal/sources.jsonl    every research source read
    docs/research-journal/<YYYY-MM>.md     human-readable journal entries
    docs/competitive/landscape.md          latest competitive snapshot
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import date as Date
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

REPO = Path(__file__).resolve().parents[1]
SCOUT_WORKFLOW = Path(".claude/workflows/opportunity-scout.js")
BUILD_WORKFLOW = Path(".claude/workflows/opportunity-build.js")
LEDGER = Path("docs/opportunities/ledger.jsonl")
GRAVEYARD = Path("docs/opportunities/graveyard.yaml")
REPORTS_DIR = Path("docs/opportunities")
JOURNAL_SOURCES = Path("docs/research-journal/sources.jsonl")
JOURNAL_DIR = Path("docs/research-journal")
LANDSCAPE = Path("docs/competitive/landscape.md")

STATUSES = (
    "PROPOSED",          # written by the synthesizer, not yet judged
    "DROPPED",           # removed by the synthesizer or the top-K cap
    "REJECTED_NOVELTY",  # G0: matches a graveyard entry without new evidence
    "REJECTED_SCOPE",    # G1: validator reject or a hard fail
    "ALLOWED_UNTESTED",  # G1 allow, beyond the top-N backtest cap
    "SUPPORTED",         # G2b: pre-registered threshold met
    "UNDERPOWERED",      # G2b: right direction, sample or significance short
    "INCONCLUSIVE",      # G2b: too few games AND the wrong direction (not buildable, not graveyard)
    "REFUTED",           # G2b: wrong side of the threshold
    "INVALID",           # G2a: integrity reviewer rejected the test
    "NOT_RUN",           # backtest could not run
    "BUILDING",          # build run started
    "BLOCKED",           # build run stopped at a gate
    "PR_OPEN",           # build run opened a pull request
    "MERGED",
    "ABANDONED",
)
BUILDABLE = {"SUPPORTED", "UNDERPOWERED"}
EVIDENCE_MAX_AGE_DAYS = 14

LEVERS = ("model", "pricing", "execution", "coverage", "venue", "data", "reliability", "sizing")
SIZES = ("S", "M", "L")
GRAVEYARD_VERDICTS = (
    "REJECTED", "REFUTED", "DONT_BUILD", "UNDERPOWERED", "NOT_SUPPORTABLE", "PARKED", "POLICY",
)
# Validator hard fails that describe a permanent access/policy block. A scope
# reject for one of these lands in the graveyard as NOT_SUPPORTABLE; any other
# scope reject (low upside, unclear mechanism) stays in the ledger only, so a
# better-framed version can come back without fighting the graveyard.
PERMANENT_HARD_FAILS = (
    "requires_auth_or_account",
    "requires_antibot_bypass",
    "requires_tos_violation_or_paywall",
)

_RULES_BEGIN = "// METRIC_RULES:BEGIN"
_RULES_END = "// METRIC_RULES:END"
_DENY_BEGIN = "// DENY_PATHS:BEGIN"
_DENY_END = "// DENY_PATHS:END"


# ---------------------------------------------------------------------------
# Metric rules (single source of truth: the scout workflow script)
# ---------------------------------------------------------------------------

def load_metric_rules(root: Path = REPO) -> dict[str, dict[str, Any]]:
    """Parse the strict-JSON METRIC_RULES block out of opportunity-scout.js.

    The workflow enforces gate G2b with these rules; reading them from the same
    file keeps the Python brief validator from drifting away from the gate.
    """
    return parse_metric_rules((root / SCOUT_WORKFLOW).read_text())


def _marked_json(text: str, begin: str, end: str, open_ch: str, close_ch: str) -> Any:
    try:
        block = text.split(begin, 1)[1].split(end, 1)[0]
    except IndexError as exc:
        raise ValueError(f"{begin} / {end} markers not found") from exc
    body = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("//"))
    return json.loads(body[body.index(open_ch): body.rindex(close_ch) + 1])


def parse_metric_rules(text: str) -> dict[str, dict[str, Any]]:
    """Extract the JSON object between the METRIC_RULES:BEGIN / END markers."""
    return _marked_json(text, _RULES_BEGIN, _RULES_END, "{", "}")


def load_deny_paths(root: Path = REPO) -> list[str]:
    """The build deny list (regexes) from opportunity-build.js — one list for JS and Python."""
    return _marked_json((root / BUILD_WORKFLOW).read_text(), _DENY_BEGIN, _DENY_END, "[", "]")


def _num(x: Any) -> str:
    """Render a number the way a JS template literal does (30.0 → "30")."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return (isinstance(x, int) and not isinstance(x, bool)) or (isinstance(x, float) and x.is_integer())


def preregistration_errors(prereg: Any, rules: dict[str, dict[str, Any]]) -> list[str]:
    """Return why a pre-registration is unusable. Empty list = usable.

    A brief may TIGHTEN a default (higher threshold for a higher-is-better
    metric, larger z_min, larger min_n_games) but never loosen one. A metric
    whose default threshold is null needs an explicit numeric threshold, every
    threshold must lie inside the metric's plausible ``value_range`` (catches
    unit mistakes), and a screening-only metric needs a ``promotion_plan``.
    Mirrors ``preregErrors`` in opportunity-scout.js message-for-message
    (tests/test_opportunity_workflows.py compares them).
    """
    if not isinstance(prereg, dict):
        return ["preregistration missing"]
    metric = prereg.get("metric")
    if not isinstance(metric, str) or metric not in rules:
        return [f"metric {json.dumps(metric, ensure_ascii=False)} is not one of {', '.join(sorted(rules))}"]
    rule = rules[metric]
    errors = [f"{key} missing" for key in
              ("train_window", "holdout_window", "comparator", "command", "declustering")
              if not prereg.get(key)]

    threshold = prereg.get("threshold")
    default_t = rule["threshold"]
    if threshold is None:
        if default_t is None:
            errors.append(f"{metric} needs an explicit numeric threshold")
    elif not _is_number(threshold) or not math.isfinite(threshold):
        errors.append("threshold must be a number")
    else:
        if default_t is not None:
            looser = threshold < default_t if rule["direction"] == "higher" else threshold > default_t
            if looser:
                errors.append(f"threshold {_num(threshold)} is looser than the default {_num(default_t)}")
        lo, hi = rule["value_range"]
        if not (lo < threshold < hi):
            errors.append(f"threshold {_num(threshold)} is outside the plausible range ({_num(lo)}, {_num(hi)})")

    z_min = prereg.get("z_min")
    if rule["z_min"] is not None and _is_number(z_min) and z_min < rule["z_min"]:
        errors.append(f"z_min {_num(z_min)} is looser than the default {_num(rule['z_min'])}")

    n_min = prereg.get("min_n_games")
    if not _is_int(n_min):
        errors.append("min_n_games must be an integer")
    elif n_min < rule["min_n_games"]:
        errors.append(f"min_n_games {_num(n_min)} is looser than the default {_num(rule['min_n_games'])}")
    if rule.get("needs_promotion_plan") and not prereg.get("promotion_plan"):
        errors.append(f"{metric} is screening only: promotion_plan is required")
    return errors


def effective_rule(prereg: dict[str, Any], rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mirror of ``effectiveRule`` (both graphs): defaults tightened, never loosened."""
    base = rules[prereg["metric"]]
    t = prereg.get("threshold") if _is_number(prereg.get("threshold")) else base["threshold"]
    if base["threshold"] is None:
        threshold = t
    elif base["direction"] == "higher":
        threshold = max(base["threshold"], t)
    else:
        threshold = min(base["threshold"], t)
    n = prereg.get("min_n_games")
    min_n = max(base["min_n_games"], int(n) if _is_int(n) else base["min_n_games"])
    return {"direction": base["direction"], "threshold": threshold, "min_n_games": min_n,
            "value_range": base["value_range"]}


def reproduces(mode: str, prereg: Any, pre: Any, post: Any,
               rules: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    """Python mirror of ``reproduces`` in opportunity-build.js (gate G4).

    verify-build recomputes G4 here from the LEDGER's pre-build evidence, so a
    mistyped Workflow arg can never make a non-reproducing build shippable.
    """
    if not isinstance(post, dict):
        return False, "verification agent returned nothing"
    if mode == "shadow_collect":
        ok = post.get("shadow_collect_ok") is True
        return ok, ("shadow-collect lane logs mode=shadow rows" if ok
                    else f"shadow-collect lane not confirmed: {post.get('note') or 'shadow_collect_ok is not true'}")
    if mode != "shadow_feature":
        return False, f"unknown build mode {mode}"
    if not isinstance(prereg, dict) or prereg.get("metric") not in rules:
        return False, "pre-registration missing or unknown metric"
    if post.get("ran") is not True:
        return False, f"post-build test did not run: {post.get('note') or ''}"
    if post.get("metric") != prereg["metric"]:
        return False, f"post-build metric {post.get('metric')!r} != pre-registered {prereg['metric']}"
    leaky = [k for k, v in (post.get("leakage_checks") or {}).items() if v is False]
    if leaky:
        return False, f"post-build leakage checks failed: {', '.join(leaky)}"
    if not isinstance(pre, dict) or not _is_number(pre.get("value")):
        return False, "no pre-build value to reproduce"
    value = post.get("value")
    if not _is_number(value) or not math.isfinite(value):
        return False, "post-build test returned no numeric value"
    rule = effective_rule(prereg, rules)
    lo, hi = rule["value_range"]
    if not (lo < value < hi):
        return False, f"post-build value {value} outside the plausible range ({lo}, {hi})"
    if rule["threshold"] is None:
        return False, "no threshold to compare against"
    beats = value > rule["threshold"] if rule["direction"] == "higher" else value < rule["threshold"]
    if not beats:
        return False, f"post-build {value} does not beat {rule['threshold']} (pre-build {pre['value']})"
    n = post.get("n_games")
    if not _is_int(n) or n < rule["min_n_games"]:
        return False, f"post-build n {n} < {rule['min_n_games']} games"
    if _is_number(pre.get("ci_low")) and _is_number(pre.get("ci_high")):
        if value < pre["ci_low"] or value > pre["ci_high"]:
            return False, f"post-build {value} outside pre-build CI [{pre['ci_low']}, {pre['ci_high']}]"
        return True, f"post-build {value} inside pre-build CI [{pre['ci_low']}, {pre['ci_high']}]"
    tol = 0.25 * abs(pre["value"] - rule["threshold"])
    if abs(value - pre["value"]) > tol:
        return False, f"post-build {value} differs from pre-build {pre['value']} by more than 25% of the margin ({tol})"
    return True, f"post-build {value} within {tol} of pre-build {pre['value']}"


def validate_brief(brief: Any, rules: dict[str, dict[str, Any]]) -> list[str]:
    """Structural + pre-registration check for one Opportunity Brief."""
    if not isinstance(brief, dict):
        return ["brief is not an object"]
    errors = [
        f"{key} missing"
        for key in ("id", "title", "lever", "sectors", "hypothesis", "edge_mechanism", "size")
        if not brief.get(key)
    ]
    if brief.get("lever") and brief["lever"] not in LEVERS:
        errors.append(f"lever {brief['lever']!r} not in {LEVERS}")
    if brief.get("size") and brief["size"] not in SIZES:
        errors.append(f"size {brief['size']!r} not in {SIZES}")
    if brief.get("id") and not re.fullmatch(r"[a-z0-9][a-z0-9-]*-\d{8}", str(brief["id"])):
        errors.append(f"id {brief['id']!r} is not <slug>-<YYYYMMDD>")
    errors += preregistration_errors(brief.get("preregistration"), rules)
    return errors


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def read_ledger(root: Path = REPO) -> list[dict[str, Any]]:
    path = root / LEDGER
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_ledger(rows: Iterable[dict[str, Any]], root: Path = REPO) -> int:
    path = root / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a") as fh:
        for row in rows:
            status = row.get("status")
            if status not in STATUSES:
                raise ValueError(f"unknown status {status!r} for {row.get('id')}")
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            n += 1
    return n


# Fields a later status row (BUILDING, PR_OPEN, ...) inherits from earlier
# rows when it does not carry them itself. Everything else — status,
# status_reason, ts, kind — always comes from the latest row alone.
_STICKY = ("title", "lever", "brief", "scope", "evidence", "integrity",
           "evidence_date", "evidence_status", "schema_errors", "branch", "pr_url")


def latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest row per id, with the sticky fields (brief, evidence, ...) carried forward."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        oid = row.get("id")
        if not oid:
            continue
        prev = latest.get(oid, {})
        merged = {k: prev[k] for k in _STICKY if prev.get(k) is not None}
        merged.update({k: v for k, v in row.items() if v is not None})
        latest[oid] = merged
    return latest


def buildable(row: Optional[dict[str, Any]], today: Date) -> tuple[bool, str]:
    """Gate G0 of the build run. Pure; exercised by tests."""
    if row is None:
        return False, "unknown opportunity id"
    status = row.get("status")
    evidence_status = row.get("evidence_status")
    # A BLOCKED build — or a BUILDING one whose session died — may be retried
    # while its discovery evidence is fresh.
    if status in ("BLOCKED", "BUILDING") and evidence_status in BUILDABLE:
        status = evidence_status
    if status not in BUILDABLE:
        return False, f"status {status} is not one of {sorted(BUILDABLE)}"
    if not row.get("brief"):
        return False, "no brief recorded"
    if row.get("schema_errors"):
        return False, f"brief has schema errors: {'; '.join(row['schema_errors'])}"
    ts = row.get("evidence_date") or row.get("ts")
    try:
        age = (today - Date.fromisoformat(str(ts)[:10])).days
    except ValueError:
        return False, f"unreadable evidence date {ts!r}"
    if age > EVIDENCE_MAX_AGE_DAYS:
        return False, f"evidence is {age} days old (> {EVIDENCE_MAX_AGE_DAYS}); re-run discovery"
    mode = "shadow_collect" if status == "UNDERPOWERED" else "shadow_feature"
    return True, mode


# ---------------------------------------------------------------------------
# Graveyard
# ---------------------------------------------------------------------------

def read_graveyard(root: Path = REPO) -> list[dict[str, Any]]:
    path = root / GRAVEYARD
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()) or []


def graveyard_add(entry: dict[str, Any], root: Path = REPO) -> bool:
    """Append one entry. Returns False (and writes nothing) when the id exists."""
    for key in ("id", "idea", "lever", "sectors", "verdict", "evidence", "revisit_if"):
        if not entry.get(key):
            raise ValueError(f"graveyard entry missing {key}")
    if entry["verdict"] not in GRAVEYARD_VERDICTS:
        raise ValueError(f"verdict {entry['verdict']!r} not in {GRAVEYARD_VERDICTS}")
    if any(e.get("id") == entry["id"] for e in read_graveyard(root)):
        return False
    ordered = {k: entry.get(k) for k in
               ("id", "idea", "lever", "sectors", "verdict", "date", "evidence", "revisit_if")}
    text = yaml.safe_dump([ordered], sort_keys=False, allow_unicode=True, width=100)
    path = root / GRAVEYARD
    existing = path.read_text() if path.exists() else ""
    sep = "" if not existing or existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    path.write_text(existing + sep + text)
    return True


def graveyard_entry_for(brief: dict[str, Any], run_date: str) -> Optional[dict[str, Any]]:
    """Map a finished discovery brief to a graveyard entry, or None.

    REFUTED → REJECTED with the measured numbers. A scope reject whose hard
    fails include a permanent access/policy block → NOT_SUPPORTABLE. Every
    other outcome stays out of the graveyard (UNDERPOWERED can still be
    shadow-collected; INVALID means the test was bad, not the idea).
    """
    status = brief.get("status")
    base = {
        "id": brief["id"],
        "idea": f"{brief.get('title', '')}. {brief.get('hypothesis', '')}".strip(),
        "lever": brief.get("lever") or "model",
        "sectors": brief.get("sectors") or ["all"],
        "date": run_date,
    }
    if status == "REFUTED":
        ev = brief.get("evidence") or {}
        pre = brief.get("preregistration") or {}
        base.update(
            verdict="REJECTED",
            evidence=(
                f"docs/opportunities/{run_date}.md. {pre.get('metric')} = {ev.get('value')} "
                f"(threshold {pre.get('threshold')}, z {ev.get('z_improvement')}, "
                f"n {ev.get('n_games')} games) via `{ev.get('command') or pre.get('command')}`."
            ),
            revisit_if=brief.get("revisit_if")
            or "New evidence that meets the pre-registered threshold on a fresh holdout.",
        )
        return base
    if status == "REJECTED_SCOPE":
        fails = [k for k, v in ((brief.get("scope") or {}).get("hard_fails") or {}).items() if v]
        permanent = [f for f in fails if f in PERMANENT_HARD_FAILS]
        if not permanent:
            return None
        base.update(
            verdict="NOT_SUPPORTABLE",
            evidence=f"docs/opportunities/{run_date}.md. Scope hard fail: {', '.join(permanent)}.",
            revisit_if="A public, unauthenticated, ToS-compliant way to get the same data.",
        )
        return base
    return None


# ---------------------------------------------------------------------------
# Research journal + competitive landscape
# ---------------------------------------------------------------------------

def _norm_url(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", (url or "").strip()).rstrip("/").lower()


def append_journal(entries: list[dict[str, Any]], run_date: str, root: Path = REPO) -> int:
    """Append new research sources (deduped by normalized URL). Returns the count added."""
    src_path = root / JOURNAL_SOURCES
    seen = set()
    if src_path.exists():
        for line in src_path.read_text().splitlines():
            if line.strip():
                seen.add(_norm_url(json.loads(line).get("url", "")))
    fresh = []
    for e in entries or []:
        key = _norm_url(e.get("url", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        fresh.append({**e, "date_read": run_date})
    if not fresh:
        return 0
    src_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("a") as fh:
        for e in fresh:
            fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")
    month = root / JOURNAL_DIR / f"{run_date[:7]}.md"
    lines = [] if month.exists() else [f"# Research journal — {run_date[:7]}", ""]
    lines += [f"## {run_date}", ""]
    for e in fresh:
        lines.append(f"- **{e.get('title', 'untitled')}** ({e.get('venue_year', '?')}) — {e.get('url')}")
        if e.get("claim"):
            lines.append(f"  - Claim: {e['claim']}")
        lines.append(
            f"  - Relevance: {e.get('relevance', '?')} · lever: {e.get('maps_to_lever', '?')}"
            f" · fetched: {e.get('fetched', '?')}"
        )
    lines.append("")
    with month.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(fresh)


def render_landscape(landscape: dict[str, Any], run_date: str) -> str:
    out = [f"# Competitive landscape — {run_date}", "",
           "Written by `scripts/opportunity_ledger.py ingest` from the opportunity-scout "
           "competitive-analysis agent. Each run overwrites this file; history lives in git.", ""]
    if landscape.get("diff_vs_previous"):
        out += ["## Changes since the previous snapshot", "", landscape["diff_vs_previous"], ""]
    comps = landscape.get("competitors") or []
    if comps:
        out += ["## Projects and tools", "",
                "| Name | Type | Approach | Has that evmax lacks | evmax has that it lacks |",
                "|---|---|---|---|---|"]
        for c in comps:
            name = f"[{c.get('name')}]({c['url']})" if c.get("url") else str(c.get("name"))
            out.append("| " + " | ".join(_cell(v) for v in (
                name, c.get("type"), c.get("approach"), c.get("has_we_lack"), c.get("we_have_they_lack"),
            )) + " |")
        out.append("")
    gaps = landscape.get("venue_gaps") or []
    if gaps:
        out += ["## Venue and market gaps", "", "| Market | Venue | Observed activity | Wiring cost |",
                "|---|---|---|---|"]
        for g in gaps:
            out.append("| " + " | ".join(_cell(g.get(k)) for k in
                                         ("market", "venue", "observed_activity", "wiring_cost")) + " |")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _cell(v: Any) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v)
    return str(v).replace("|", "\\|").replace("\n", " ")


def _result_cell(brief: dict[str, Any]) -> str:
    ev = brief.get("evidence") or {}
    if not ev or ev.get("value") is None:
        return "—"
    parts = [f"{ev['value']:+.4g}" if isinstance(ev["value"], (int, float)) else str(ev["value"])]
    if ev.get("z_improvement") is not None:
        parts.append(f"z {ev['z_improvement']:+.2f}")
    if ev.get("n_games") is not None:
        parts.append(f"n {ev['n_games']}g")
    return " · ".join(parts)


def render_report(result: dict[str, Any], run_date: str) -> str:
    run = result.get("run") or {}
    briefs = result.get("briefs") or []
    out = [
        f"# Opportunity scout — {run_date}", "",
        f"Focus: {run.get('focus') or 'whole project'} · snapshot: `{run.get('snapshot_path', '?')}` · "
        "workflow: `.claude/workflows/opportunity-scout.js`", "",
        "Evidence is offline and pre-registered. `SUPPORTED` means the pre-registered "
        "threshold was met on the stated holdout; it is not a promotion verdict. Every build "
        "enters as shadow or behind a default-off flag.", "",
        "## Summary", "",
        "| # | Opportunity | Lever | Sectors | Status | Pre-registered metric | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, b in enumerate(briefs, 1):
        pre = b.get("preregistration") or {}
        metric = f"{pre.get('metric')} vs {pre.get('threshold')}" if pre.get("metric") else "—"
        out.append("| " + " | ".join(_cell(v) for v in (
            i, f"{b.get('title')} (`{b.get('id')}`)", b.get("lever"), b.get("sectors"),
            b.get("status"), metric, _result_cell(b),
        )) + " |")
    out.append("")

    buildable_ids = [b["id"] for b in briefs if b.get("status") in BUILDABLE]
    if buildable_ids:
        out += ["**Next step:** `/opportunities build <id>` — "
                + ", ".join(f"`{i}`" for i in buildable_ids)
                + ". `UNDERPOWERED` builds are shadow-collect only.", ""]

    out += ["## Briefs", ""]
    for b in briefs:
        out += _render_brief(b)

    dropped = result.get("dropped") or []
    if dropped:
        out += ["## Dropped before validation", "", "| Candidate | Source | Reason | Graveyard match |",
                "|---|---|---|---|"]
        for d in dropped:
            out.append("| " + " | ".join(_cell(d.get(k)) for k in
                                         ("title", "source", "reason", "graveyard_id")) + " |")
        out.append("")

    props = result.get("proposers") or {}
    if props:
        out += ["## Proposer notes", ""]
        for name in ("research", "competitive", "modeling"):
            p = props.get(name)
            if not p:
                continue
            n = len(p.get("candidates") or [])
            out.append(f"- **{name}** — {n} candidate(s). {p.get('notes') or ''}".rstrip())
        out.append("")
    notes = result.get("notes") or []
    if notes:
        out += ["## Run notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(out)


def _render_brief(b: dict[str, Any]) -> list[str]:
    pre = b.get("preregistration") or {}
    scope = b.get("scope") or {}
    ev = b.get("evidence") or {}
    integ = b.get("integrity") or {}
    out = [f"### {b.get('title')} — `{b.get('status')}`", "",
           f"`{b.get('id')}` · lever **{b.get('lever')}** · size **{b.get('size')}** · "
           f"sectors {_cell(b.get('sectors'))} · markets {_cell(b.get('market_types'))} · "
           f"venues {_cell(b.get('venues'))}", ""]
    if b.get("status_reason"):
        out += [f"Status reason: {b['status_reason']}", ""]
    out += [f"- **Hypothesis:** {b.get('hypothesis')}",
            f"- **Edge mechanism:** {b.get('edge_mechanism')}"]
    for s in b.get("sources") or []:
        out.append(f"- Source ({s.get('kind')}): {s.get('ref')}")
    gc = b.get("graveyard_check") or {}
    if gc.get("matched_ids"):
        out.append(f"- Graveyard matches: {_cell(gc['matched_ids'])} — {gc.get('why_different', '')}")
    out.append("")
    out += ["**Pre-registration**", "",
            "| Metric | Threshold | z min | Min games | Train | Holdout | Comparator | Declustering |",
            "|---|---|---|---|---|---|---|---|",
            "| " + " | ".join(_cell(pre.get(k)) for k in (
                "metric", "threshold", "z_min", "min_n_games", "train_window", "holdout_window",
                "comparator", "declustering")) + " |", "",
            f"Command: `{pre.get('command', '—')}`", ""]
    if scope:
        fails = [k for k, v in (scope.get("hard_fails") or {}).items() if v]
        scores = scope.get("scores") or {}
        out += [f"**Scope (G1):** {scope.get('verdict')} — {scope.get('reason', '')}"]
        if fails:
            out.append(f"- Hard fails: {', '.join(fails)}")
        if scores:
            out.append("- Scores: " + ", ".join(f"{k} {v}" for k, v in scores.items()))
        for rc in scope.get("required_changes") or []:
            out.append(f"- Required change: {rc}")
        out.append("")
    if ev:
        out += [f"**Evidence (G2):** {_result_cell(b)}"
                + (f" · CI [{ev.get('ci_low')}, {ev.get('ci_high')}]" if ev.get("ci_low") is not None else ""),
                f"- Command run: `{ev.get('command', '—')}`",
                f"- Windows: train {ev.get('train_window', '—')} · holdout {ev.get('holdout_window', '—')}"]
        if ev.get("note"):
            out.append(f"- Note: {ev['note']}")
        for s in ev.get("secondary") or []:
            out.append(f"- Secondary (not gated): {s}")
        if integ:
            out.append(f"- Integrity (G2a): {'accept' if integ.get('accept') else 'REJECT'} — {integ.get('reason', '')}")
        out.append("")
    if b.get("build_plan") or b.get("files_likely_touched"):
        out += [f"**Build sketch:** {b.get('build_plan', '—')}",
                f"- Files likely touched: {_cell(b.get('files_likely_touched'))}",
                f"- Blast radius: {b.get('blast_radius', '—')}", ""]
    for r in b.get("risks") or []:
        out.append(f"- Risk: {r}")
    if b.get("risks"):
        out.append("")
    return out


def _report_path(run_date: str, root: Path) -> Path:
    path = root / REPORTS_DIR / f"{run_date}.md"
    n = 2
    while path.exists():
        path = root / REPORTS_DIR / f"{run_date}-{n}.md"
        n += 1
    return path


# ---------------------------------------------------------------------------
# verify-build (deterministic ship gate for a build run)
# ---------------------------------------------------------------------------

_MODE_RANK = {"disabled": 0, "shadow": 1, "live": 2}


def _git_out(repo: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or f"git {' '.join(args)} failed")
    return cp.stdout


def changed_paths(worktree: Path) -> list[str]:
    """Every path git sees as changed in ``worktree`` (tracked, untracked, both sides of renames)."""
    raw = _git_out(worktree, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    parts = raw.split("\0")
    out: list[str] = []
    i = 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        out.append(path)
        if "R" in code or "C" in code:          # -z: the original path follows as its own field
            if i < len(parts) and parts[i]:
                out.append(parts[i])
            i += 1
    return sorted(set(out))


def deny_matches(paths: Iterable[str], deny: list[str]) -> list[str]:
    res = [re.compile(r) for r in deny]
    return [p for p in paths if any(r.search(p.removeprefix("./")) for r in res)]


def live_widening(base_categories: Any, head_categories: Any,
                  base_tiers: Any, head_tiers: Any, settings_diff: str) -> list[str]:
    """Reasons a change makes any lane MORE live. Empty list = nothing widened.

    categories.yaml: a mode may not rise (disabled < shadow < live), a new
    category may not be live, and no entry may leave shadow_market_types /
    disabled_market_types / shadow_venue_market_types. soccer_league_tiers.yaml:
    no league may leave shadow_leagues. settings.py: no line naming a *_live
    switch may change.
    """
    out: list[str] = []
    base_c = base_categories if isinstance(base_categories, dict) else {}
    head_c = head_categories if isinstance(head_categories, dict) else {}
    for key, head in head_c.items():
        if not isinstance(head, dict):
            continue
        base = base_c.get(key) if isinstance(base_c.get(key), dict) else None
        h_rank = _MODE_RANK.get(str(head.get("mode")), 2)
        if base is None:
            if h_rank >= _MODE_RANK["live"]:
                out.append(f"new category {key} is not shadow/disabled (mode {head.get('mode')})")
            continue
        if h_rank > _MODE_RANK.get(str(base.get("mode")), 2):
            out.append(f"category {key} mode {base.get('mode')} -> {head.get('mode')}")
        for field in ("shadow_market_types", "disabled_market_types"):
            lost = set(base.get(field) or []) - set(head.get(field) or [])
            if lost:
                out.append(f"category {key} {field} lost {sorted(lost)}")
        bv, hv = base.get("shadow_venue_market_types") or {}, head.get("shadow_venue_market_types") or {}
        for venue, types in bv.items():
            lost = set(types or []) - set(hv.get(venue) or [])
            if lost:
                out.append(f"category {key} shadow_venue_market_types.{venue} lost {sorted(lost)}")
    lost_leagues = set((base_tiers or {}).get("shadow_leagues") or []) - set((head_tiers or {}).get("shadow_leagues") or [])
    if lost_leagues:
        out.append(f"soccer shadow_leagues lost {sorted(lost_leagues)}")
    for line in (settings_diff or "").splitlines():
        if line[:1] in "+-" and not line.startswith(("+++", "---")) and re.search(r"\w_live\b", line):
            out.append(f"settings.py live switch line changed: {line.strip()[:120]}")
    return out


def _yaml_at_head(worktree: Path, rel: str) -> Any:
    try:
        return yaml.safe_load(_git_out(worktree, "show", f"HEAD:{rel}")) or {}
    except RuntimeError:
        return {}


def _yaml_now(worktree: Path, rel: str) -> Any:
    path = worktree / rel
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def verify_build(result: dict[str, Any], row: Optional[dict[str, Any]], worktree: Path, today: Date,
                 root: Path = REPO, session_root: Optional[Path] = None,
                 session_baseline: Optional[str] = None) -> dict[str, Any]:
    """Every check that must pass before a build run's worktree may be shipped.

    Recomputed here, outside every LLM: the ledger's build gate, the build mode,
    G4 against the ledger's own pre-build evidence, the ACTUAL changed files
    (deny list + the build's declared allowlist), live-mode widening, and that
    the session checkout did not change during the build.
    """
    reasons: list[str] = []
    if not result.get("ready_to_ship"):
        reasons.append(f"workflow did not mark the build ready: {result.get('reason')}")
    ok, mode = buildable(row, today)
    if not ok:
        reasons.append(f"ledger gate: {mode}")
        mode = None
    if mode and result.get("build_mode") != mode:
        reasons.append(f"build_mode {result.get('build_mode')!r} != ledger mode {mode!r}")
    if mode:
        rules = load_metric_rules(root)
        g4_ok, g4_why = reproduces(mode, (row.get("brief") or {}).get("preregistration"),
                                   row.get("evidence"), result.get("verification"), rules)
        if not g4_ok:
            reasons.append(f"G4 (recomputed from the ledger): {g4_why}")

    files = changed_paths(worktree)
    if not files:
        reasons.append("no changed files in the worktree")
    denied = deny_matches(files, load_deny_paths(root))
    if denied:
        reasons.append(f"deny-listed paths changed: {denied}")
    allowed = {f.removeprefix("./") for f in (result.get("allowed_files") or [])}
    unexpected = [f for f in files if f not in allowed]
    if unexpected:
        reasons.append(f"changed files nobody declared (implementer files_changed / test tests_added): {unexpected}")

    widened = live_widening(
        _yaml_at_head(worktree, "data/categories.yaml"), _yaml_now(worktree, "data/categories.yaml"),
        _yaml_at_head(worktree, "data/soccer_league_tiers.yaml"), _yaml_now(worktree, "data/soccer_league_tiers.yaml"),
        _git_out(worktree, "diff", "HEAD", "--", "evmax/settings.py"),
    )
    reasons += [f"live widening: {w}" for w in widened]

    if session_root is not None and session_baseline is not None:
        now = _git_out(session_root, "status", "--porcelain=v1", "--untracked-files=all")
        if now.strip() != session_baseline.strip():
            reasons.append(f"session checkout {session_root} changed during the build")
    return {"ok": not reasons, "reasons": reasons, "files": files, "build_mode": mode}


# ---------------------------------------------------------------------------
# Ingest (discovery result → every file)
# ---------------------------------------------------------------------------

def ingest(result: dict[str, Any], run_date: str, root: Path = REPO) -> dict[str, Any]:
    """Write every artifact for one discovery result. Returns a summary dict."""
    rules = load_metric_rules(root)
    briefs = result.get("briefs") or []
    run_label = f"{run_date}:{(result.get('run') or {}).get('focus') or 'all'}"

    rows = []
    for b in briefs:
        errors = validate_brief(b, rules)
        rows.append({
            "ts": run_date,
            "kind": "discovery",
            "run": run_label,
            "id": b.get("id"),
            "title": b.get("title"),
            "lever": b.get("lever"),
            "status": b.get("status") or "PROPOSED",
            "status_reason": b.get("status_reason"),
            "evidence_date": run_date if b.get("evidence") else None,
            "evidence_status": b.get("status") if b.get("evidence") else None,
            "brief": {k: v for k, v in b.items()
                      if k not in ("status", "status_reason", "scope", "evidence", "integrity")},
            "scope": b.get("scope"),
            "evidence": b.get("evidence"),
            "integrity": b.get("integrity"),
            "schema_errors": errors,   # [] (not None) so a clean re-ingest clears old errors
        })
    for d in result.get("dropped") or []:
        if d.get("id"):
            status = "REJECTED_NOVELTY" if d.get("graveyard_id") else "DROPPED"
            reason = d.get("reason")
            if d.get("graveyard_id"):
                reason = f"graveyard:{d['graveyard_id']} — {reason or ''}".rstrip(" —")
            rows.append({"ts": run_date, "kind": "discovery", "run": run_label, "id": d["id"],
                         "title": d.get("title"), "lever": d.get("lever"), "status": status,
                         "status_reason": reason})
    n_ledger = append_ledger(rows, root)

    graveyard_added = []
    for b in briefs:
        entry = graveyard_entry_for(b, run_date)
        if entry and graveyard_add(entry, root):
            graveyard_added.append(entry["id"])

    props = result.get("proposers") or {}
    journal_entries = []
    for key in ("research", "research_2"):
        journal_entries += (props.get(key) or {}).get("journal_entries") or []
    n_journal = append_journal(journal_entries, run_date, root)

    landscape = (props.get("competitive") or {}).get("landscape")
    if landscape:
        path = root / LANDSCAPE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_landscape(landscape, run_date))

    report = _report_path(run_date, root)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(result, run_date))
    return {
        "report": str(report.relative_to(root)),
        "ledger_rows": n_ledger,
        "graveyard_added": graveyard_added,
        "journal_sources_added": n_journal,
        "landscape_written": bool(landscape),
        "schema_errors": {r["id"]: r["schema_errors"] for r in rows if r.get("schema_errors")},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def unwrap_workflow_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept either the workflow's return value or the Workflow task output file.

    The Workflow tool writes ``{"summary", "agentCount", "logs", "result", ...}``
    to its task output file; ingesting that file directly means nobody has to
    re-type the result JSON. Workflow ``log()`` lines are appended to ``notes``.
    """
    if isinstance(payload, dict) and "result" in payload and "agentCount" in payload:
        result = dict(payload.get("result") or {})
        logs = [f"log: {line}" for line in payload.get("logs") or []]
        if logs:
            result["notes"] = list(result.get("notes") or []) + logs
        return result
    return payload


def _cmd_ingest(args: argparse.Namespace) -> int:
    result = unwrap_workflow_output(json.loads(Path(args.result).read_text()))
    if result.get("error"):
        print(f"ingest: workflow returned an error: {result['error']}", file=sys.stderr)
        return 1
    if not isinstance(result.get("briefs"), list):
        print("ingest: unexpected payload (no 'briefs' list) — refusing to write an empty report", file=sys.stderr)
        return 1
    run_date = args.date or (result.get("run") or {}).get("date")
    if not run_date:
        print("ingest: pass --date or include run.date in the result", file=sys.stderr)
        return 1
    summary = ingest(result, run_date, Path(args.root))
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    latest = latest_by_id(read_ledger(Path(args.root)))
    if args.id:
        latest = {k: v for k, v in latest.items() if k == args.id}
    view = [{"id": k, "status": v.get("status"), "title": v.get("title"), "ts": v.get("ts"),
             "pr_url": v.get("pr_url")} for k, v in sorted(latest.items(), key=lambda kv: kv[1].get("ts", ""))]
    if args.json:
        print(json.dumps(view, indent=2))
    else:
        for r in view:
            print(f"{r['ts']}  {r['status']:<17} {r['id']}  {r['title'] or ''}"
                  + (f"  {r['pr_url']}" if r.get("pr_url") else ""))
    return 0


_GATE_EVIDENCE_KEYS = ("metric", "value", "z_improvement", "n_games", "ci_low", "ci_high", "command")
_GATE_SUMMARY_KEYS = ("hypothesis", "edge_mechanism", "lever", "sectors", "risks")


def build_gate(row: dict[str, Any], build_mode: str) -> dict[str, Any]:
    """The compact object the build workflow gates on (passed as Workflow args).

    Kept small on purpose: it is the only ledger data the deterministic G4 gate
    reads, so the command never has to retype the full row.
    """
    brief = row.get("brief") or {}
    evidence = row.get("evidence") or {}
    return {
        "opp_id": row.get("id"),
        "title": row.get("title") or brief.get("title"),
        "build_mode": build_mode,
        "preregistration": brief.get("preregistration"),
        "evidence": {k: evidence.get(k) for k in _GATE_EVIDENCE_KEYS},
        "brief_summary": {k: brief.get(k) for k in _GATE_SUMMARY_KEYS},
    }


def _cmd_get(args: argparse.Namespace) -> int:
    row = latest_by_id(read_ledger(Path(args.root))).get(args.id)
    build_mode = None
    if args.check_buildable or args.gate:
        today = Date.fromisoformat(args.today) if args.today else Date.today()
        ok, why = buildable(row, today)
        if not ok:
            print(f"not buildable: {why}", file=sys.stderr)
            return 2
        build_mode = why
        row = {**row, "build_mode": why}
    if row is None:
        print(f"unknown opportunity id {args.id}", file=sys.stderr)
        return 1
    if args.row_out:
        out = Path(args.row_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=2, default=str))
    if args.gate:
        print(json.dumps(build_gate(row, build_mode), default=str))
    else:
        print(json.dumps(row, indent=2, default=str))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    row = latest_by_id(read_ledger(Path(args.root))).get(args.id)
    if row is None and not args.force:
        print(f"unknown opportunity id {args.id} (use --force to record anyway)", file=sys.stderr)
        return 1
    append_ledger([{
        "ts": args.date or Date.today().isoformat(),
        "kind": "build" if args.status in ("BUILDING", "BLOCKED", "PR_OPEN") else "manual",
        "id": args.id,
        "title": (row or {}).get("title"),
        "status": args.status,
        "status_reason": args.note,
        "branch": args.branch,
        "pr_url": args.pr_url,
    }], Path(args.root))
    return 0


def _cmd_verify_build(args: argparse.Namespace) -> int:
    result = unwrap_workflow_output(json.loads(Path(args.output).read_text()))
    root = Path(args.root)
    row = latest_by_id(read_ledger(root)).get(args.id)
    today = Date.fromisoformat(args.today) if args.today else Date.today()
    baseline = Path(args.session_baseline).read_text() if args.session_baseline else None
    report = verify_build(result, row, Path(args.worktree), today, root,
                          Path(args.session_root) if args.session_root else None, baseline)
    if args.files_out:
        Path(args.files_out).write_text("\n".join(report["files"]) + ("\n" if report["files"] else ""))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 3


def _cmd_graveyard_add(args: argparse.Namespace) -> int:
    added = graveyard_add({
        "id": args.id, "idea": args.idea, "lever": args.lever,
        "sectors": [s.strip() for s in args.sectors.split(",") if s.strip()],
        "verdict": args.verdict, "date": args.date, "evidence": args.evidence,
        "revisit_if": args.revisit_if,
    }, Path(args.root))
    print("added" if added else f"id {args.id} already in the graveyard; nothing written")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    brief = json.loads(Path(args.file).read_text())
    errors = validate_brief(brief, load_metric_rules(Path(args.root)))
    for e in errors:
        print(f"- {e}")
    print("ok" if not errors else f"{len(errors)} problem(s)")
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=str(REPO), help="Repo root (default: this checkout)")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="Write every artifact for one discovery result")
    pi.add_argument("result", help="Workflow result JSON, or the Workflow task output file")
    pi.add_argument("--date", help="Run date YYYY-MM-DD (default: result.run.date)")
    pi.set_defaults(func=_cmd_ingest)

    ps = sub.add_parser("status", help="Latest status per opportunity")
    ps.add_argument("--id")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=_cmd_status)

    pg = sub.add_parser("get", help="Latest ledger row for one opportunity, as JSON")
    pg.add_argument("--id", required=True)
    pg.add_argument("--check-buildable", action="store_true",
                    help="Exit 2 unless the opportunity may enter a build run")
    pg.add_argument("--today", help="Date for the evidence-age check (default: today)")
    pg.add_argument("--row-out", help="Also write the full row to this path")
    pg.add_argument("--gate", action="store_true",
                    help="Print the compact build gate object (implies --check-buildable)")
    pg.set_defaults(func=_cmd_get)

    pr = sub.add_parser("record", help="Append one status transition")
    pr.add_argument("--id", required=True)
    pr.add_argument("--status", required=True, choices=STATUSES)
    pr.add_argument("--note")
    pr.add_argument("--branch")
    pr.add_argument("--pr-url")
    pr.add_argument("--date")
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(func=_cmd_record)

    pb = sub.add_parser("verify-build", help="Deterministic ship gate for a build run (exit 3 = do not ship)")
    pb.add_argument("output", help="Workflow task output file of the opportunity-build run")
    pb.add_argument("--id", required=True)
    pb.add_argument("--worktree", required=True)
    pb.add_argument("--today")
    pb.add_argument("--session-root", help="Checkout the command ran from (must be unchanged)")
    pb.add_argument("--session-baseline", help="`git status --porcelain=v1 --untracked-files=all` captured before the build")
    pb.add_argument("--files-out", help="Write the changed files to ship, one per line")
    pb.set_defaults(func=_cmd_verify_build)

    pa = sub.add_parser("graveyard-add", help="Append one graveyard entry")
    pa.add_argument("--id", required=True)
    pa.add_argument("--idea", required=True)
    pa.add_argument("--lever", required=True, choices=LEVERS)
    pa.add_argument("--sectors", required=True, help="Comma-separated sector keys, or 'all'")
    pa.add_argument("--verdict", required=True, choices=GRAVEYARD_VERDICTS)
    pa.add_argument("--evidence", required=True)
    pa.add_argument("--revisit-if", required=True)
    pa.add_argument("--date")
    pa.set_defaults(func=_cmd_graveyard_add)

    pv = sub.add_parser("validate-brief", help="Check one brief JSON file")
    pv.add_argument("file")
    pv.set_defaults(func=_cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
