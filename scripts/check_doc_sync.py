#!/usr/bin/env python3
"""Pre-commit hook: remind to keep docs, tests, AND TODO.md in sync.

Three parallel checks:
  1. doc-sync  — warn when source changes but related docs (CLAUDE.md, __init__.py,
     folder READMEs) are not also in the staged set.
  2. test-sync — warn when logic-bearing source changes but no test file in the
     staged set covers it. Source modules without ANY existing test file get a
     stronger warning ("zero coverage — add a test file before merging").
  3. todo-sync — warn when source changes but TODO.md is not in the staged set.
     Soft nudge: "did this commit knock anything off the TODO list?"

Exits 0 always — this is a reminder, not a blocker.
Usage (called by pre-commit framework):
    python3 scripts/check_doc_sync.py
Or directly against a list of staged files:
    python3 scripts/check_doc_sync.py file1.py file2.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Maps a changed file pattern → list of docs to check
# Key: string that must appear in the changed file path (checked with str.startswith)
# Value: list of (doc_path, hint) pairs
DOC_RULES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "evmax/clients/",
        [
            ("CLAUDE.md", "Key Pipeline step 2, Architecture tree, Key Implementation Details"),
            ("evmax/clients/__init__.py", "client descriptions"),
            ("evmax/clients/README.md", "active vs legacy client list"),
        ],
    ),
    (
        "evmax/agents/models/",
        [
            ("CLAUDE.md", "Modeling table (weights, confidence gate, state files)"),
            ("evmax/agents/models/__init__.py", "model list and weight comments"),
        ],
    ),
    (
        "evmax/agents/odds/",
        [
            ("CLAUDE.md", "Key Implementation Details (YES alignment, draw market, EV formula)"),
        ],
    ),
    (
        "evmax/agents/coordinator.py",
        [
            ("CLAUDE.md", "Key Pipeline, Daily Workflow, sharp_weight default"),
        ],
    ),
    (
        "evmax/agents/cleanup/",
        [
            ("CLAUDE.md", "Data Sources for Outcome Resolution"),
            ("evmax/agents/cleanup/__init__.py", "module list"),
        ],
    ),
    (
        "evmax/sectors/",
        [
            ("CLAUDE.md", "Key Sectors list, Architecture tree"),
            ("evmax/sectors/__init__.py", "active sectors list"),
        ],
    ),
    (
        "evmax/ev/",
        [
            ("CLAUDE.md", "Key Pipeline steps 5/8/9, Key Implementation Details"),
            ("evmax/ev/__init__.py", "module descriptions"),
        ],
    ),
    (
        "evmax/matching/",
        [
            ("CLAUDE.md", "Key Pipeline step 4, Key Implementation Details"),
            ("evmax/matching/__init__.py", "algorithm description"),
        ],
    ),
    (
        "evmax/models_ml/",
        [
            ("evmax/models_ml/__init__.py", "active vs legacy module list"),
            ("evmax/models_ml/README.md", "module descriptions"),
        ],
    ),
    (
        "evmax/pipeline/",
        [
            ("evmax/pipeline/__init__.py", "active vs legacy module note"),
        ],
    ),
    (
        "evmax/cli/commands/",
        [
            ("CLAUDE.md", "Daily Workflow, CLI Output Requirements"),
            ("README.md", "CLI usage examples"),
        ],
    ),
]

# Maps a changed source path prefix → list of test files that should ALSO be staged.
# Order matters only for the hint message; the check passes if ANY listed test file
# is in the staged set. If no listed test file exists on disk, we treat the module
# as "zero coverage" and print a stronger warning.
TEST_RULES: list[tuple[str, list[str], str]] = [
    (
        "evmax/agents/models/elo_agent.py",
        ["tests/test_agents.py"],
        "Elo agent — test win prob, rest adjustment, MOV multiplier",
    ),
    (
        "evmax/agents/models/form_agent.py",
        ["tests/test_agents.py"],
        "Form agent — test log5, decay, soccer draw allocation",
    ),
    (
        "evmax/agents/models/poisson_agent.py",
        ["tests/test_agents.py"],
        "Poisson agent — test win prob, Dixon-Coles, score matrix bounds",
    ),
    (
        "evmax/agents/models/ensemble_agent.py",
        ["tests/test_agents.py"],
        "Ensemble — test confidence gate, weight blending, normalization",
    ),
    (
        "evmax/agents/models/tennis_model_agent.py",
        ["tests/test_tennis_model.py"],
        "Tennis model — test surface lookup, Elo computation, fallback when player unknown",
    ),
    (
        "evmax/agents/models/pitcher_agent.py",
        ["tests/test_pitcher_agent.py"],
        "Pitcher model — test ERA lookup, park factor, Pythagorean win prob",
    ),
    (
        "evmax/agents/odds/ev_gap_agent.py",
        ["tests/test_injury_ev_gap.py", "tests/test_matching_and_ev.py"],
        "EVGap — test YES alignment, injury boost, exposure guard, EV math",
    ),
    (
        "evmax/agents/odds/sharp_agent.py",
        ["tests/test_agents.py"],
        "Sharp odds agent — test devig flow + sector dispatch",
    ),
    (
        "evmax/agents/odds/kalshi_agent.py",
        ["tests/test_agents.py"],
        "Kalshi agent — test market parsing + series fanout",
    ),
    (
        "evmax/agents/cleanup/resolver.py",
        ["tests/test_resolver.py", "tests/test_cleanup_pipeline.py"],
        "Resolver — test ESPN match, bo3 match, sector fall-through, draw handling",
    ),
    (
        "evmax/agents/cleanup/metrics.py",
        ["tests/test_cleanup_pipeline.py"],
        "Metrics — test Brier score calc, sharp_weight auto-tune bounds",
    ),
    (
        "evmax/ev/devig.py",
        ["tests/test_devig.py", "tests/test_robustness_pass1_math.py"],
        "Devig — test 2-way + 3-way Power Method against known-good fixtures",
    ),
    (
        "evmax/ev/calculator.py",
        ["tests/test_ev_calculator.py", "tests/test_core_math.py"],
        "EV calculator — test edge sign, payout formula, NaN guards",
    ),
    (
        "evmax/ev/kelly.py",
        ["tests/test_kelly.py"],
        "Kelly — test fraction bounds, liquidity discount, NaN/Inf guards",
    ),
    (
        "evmax/matching/engine.py",
        ["tests/test_matching.py", "tests/test_matching_and_ev.py", "tests/test_robustness_pass2_matching.py"],
        "Matcher — test canonical key, fuzzy fallback, UCL home/away reversal",
    ),
    (
        "evmax/clients/esports_pinnacle.py",
        ["tests/test_pinnacle_guest.py"],
        "PinnacleGuestClient — test response parsing + devigging on fixtures",
    ),
    (
        "evmax/clients/kalshi.py",
        ["tests/test_kalshi_client.py"],
        "Kalshi client — test ticker parsing, rate limiter, WS message handling",
    ),
    (
        "evmax/sectors/",
        ["tests/test_sectors.py"],
        "Sector handler — test alias resolution and canonical name normalization",
    ),
]


def get_staged_files() -> list[str]:
    """Return list of staged file paths relative to repo root."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def check_docs(changed_files: list[str]) -> None:
    reminders: dict[str, list[str]] = {}  # doc_path → list of reasons

    for changed in changed_files:
        for prefix, doc_hints in DOC_RULES:
            if changed.startswith(prefix) or changed == prefix.rstrip("/"):
                for doc_path, hint in doc_hints:
                    # Don't remind about the file that was just changed
                    if changed == doc_path:
                        continue
                    reminders.setdefault(doc_path, [])
                    reason = f"  changed: {changed} → check {hint}"
                    if reason not in reminders[doc_path]:
                        reminders[doc_path].append(reason)

    if not reminders:
        return

    print("\n\033[33m[doc-sync]\033[0m Reminder: the following docs may need updating:\n")
    for doc_path, reasons in reminders.items():
        print(f"  \033[36m{doc_path}\033[0m")
        for r in reasons:
            print(r)
    print()


def check_tests(changed_files: list[str], repo_root: Path) -> None:
    """Warn when source changes are not accompanied by test changes.

    Two flavours of warning:
      - "no staged test"  — the source has tests on disk but none staged in this commit
      - "ZERO COVERAGE"   — no test file exists for this source at all (worse)
    """
    staged_set = set(changed_files)
    # Any test file at all in the staged set counts as "tests touched somewhere".
    has_any_test_change = any(f.startswith("tests/") for f in changed_files)

    missing_tests: list[tuple[str, str, str]] = []  # (src, hint, severity)

    for changed in changed_files:
        for prefix, test_files, hint in TEST_RULES:
            matched = (
                changed == prefix
                or (prefix.endswith("/") and changed.startswith(prefix))
            )
            if not matched:
                continue

            staged_test_match = any(t in staged_set for t in test_files)
            if staged_test_match:
                continue  # a relevant test was staged — good

            test_files_exist = [t for t in test_files if (repo_root / t).exists()]
            if not test_files_exist:
                missing_tests.append((changed, hint, "ZERO COVERAGE"))
            elif not has_any_test_change:
                missing_tests.append((changed, hint, "no staged test"))
            # else: some test change was staged, even if not the canonical one — soft pass

    if not missing_tests:
        return

    print("\n\033[31m[test-sync]\033[0m Source changes without matching test changes:\n")
    for src, hint, severity in missing_tests:
        colour = "\033[31m" if severity == "ZERO COVERAGE" else "\033[33m"
        print(f"  {colour}[{severity}]\033[0m {src}")
        print(f"      → {hint}")
    print(
        "\n  Per CLAUDE.md testing policy: write or extend a test before this lands.\n"
        "  This reminder is advisory and does not block the commit.\n"
    )


def check_todo(changed_files: list[str]) -> None:
    """Soft nudge: if logic-bearing source changed, did you update TODO.md?

    We don't try to be smart about whether the change actually closes a TODO item —
    that requires understanding intent. Instead we just remind once per commit
    when source under evmax/ changes without TODO.md being staged. Easy to ignore
    if nothing was knocked off; hard to forget if something was.
    """
    if "TODO.md" in changed_files:
        return  # Already updated — nothing to nudge about

    src_changed = any(
        f.startswith("evmax/") and f.endswith(".py") for f in changed_files
    )
    if not src_changed:
        return

    print(
        "\n\033[36m[todo-sync]\033[0m Did this commit close, partially close, or"
        " invalidate any item in TODO.md?\n"
        "  If yes: move it to the 'Recently Shipped' section (or strike it through).\n"
        "  If no:  ignore this nudge.\n"
    )


def main() -> None:
    if len(sys.argv) > 1:
        # Called with explicit file list (e.g. from pre-commit args)
        changed_files = sys.argv[1:]
    else:
        changed_files = get_staged_files()

    if changed_files:
        repo_root = Path(__file__).resolve().parents[1]
        check_docs(changed_files)
        check_tests(changed_files, repo_root)
        check_todo(changed_files)

    sys.exit(0)  # Never block the commit


if __name__ == "__main__":
    main()
