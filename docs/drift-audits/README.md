# Drift audits

Timestamped reports from the **drift auditor** — the weekly engineering-hygiene loop that
finds and closes the gap between what evmax *says it does* (CLAUDE.md, README, MEMORY, `docs/`,
the registry) and what the code *actually does*.

- **Process spec / manual run:** [`.claude/commands/drift-audit.md`](../../.claude/commands/drift-audit.md) — run on demand with `/drift-audit` (add `--report-only` for a read-only sweep).
- **Schedule:** local scheduled task `weekly-drift-audit`, Mondays ~07:50 local (after the
  seasonal reseed at 07:04 and before model calibration at 08:07, so it audits a freshly-seeded tree).
- **Cadence rationale:** drift accumulates slowly; weekly catches it without noise.

## What it audits

Three classes of drift (see the command spec for the full taxonomy):

1. **Functionality drift** — documented behavior/constants/weights/modes/gates ≠ the code that implements them.
2. **Ownership drift** — the script/task documented as owning a process is gone, disabled, or superseded.
3. **Reference / structural drift** — a path, script, CLI command, or task id named in docs no longer resolves.

## What it does with findings

- **SAFE drift** (stale docs, dead references — code is authoritative) → auto-fixed on a
  `drift-audit/<date>` branch and bundled into a PR.
- **RISKY drift** (anything touching code logic, `categories.yaml`, model state, tests, or where
  it's unclear which side is correct) → **proposed only** in the PR body for the user to decide.

Each run writes `docs/drift-audits/<YYYY-MM-DD>.md` and opens at most one PR. A clean tree
produces "0 findings" and no PR.
