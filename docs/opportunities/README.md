# docs/opportunities

Output of the opportunity-scout workflow. Design: [../opportunity-workflow-scope.md](../opportunity-workflow-scope.md).

| Path | What | Written by |
|---|---|---|
| `<date>.md` | One report per discovery run: briefs, scope verdicts, pre-registrations, offline evidence | `scripts/opportunity_ledger.py ingest` |
| `ledger.jsonl` | Append-only status log, one row per opportunity per transition | `ingest`, `record` |
| `graveyard.yaml` | Ideas already tested and not shipped, each with a `revisit_if` | seeded by hand; `ingest` and `graveyard-add` append |
| `context/<date>.json` | The shared snapshot each run's agents read (gitignored, regenerated) | `scripts/opportunity_context.py` |

Related: `docs/research-journal/` (every research source read) and `docs/competitive/landscape.md`
(latest competitive snapshot).

```bash
uv run python scripts/opportunity_ledger.py status            # latest status per opportunity
uv run python scripts/opportunity_ledger.py get --id <id>     # full row: brief, scope, evidence
```

Statuses: `PROPOSED`, `DROPPED`, `REJECTED_NOVELTY`, `REJECTED_SCOPE`, `ALLOWED_UNTESTED`,
`SUPPORTED`, `UNDERPOWERED`, `REFUTED`, `INVALID`, `NOT_RUN`, `BUILDING`, `BLOCKED`, `PR_OPEN`,
`MERGED`, `ABANDONED`. Only `SUPPORTED` (shadow feature) and `UNDERPOWERED` (shadow-collect only)
can enter a build, and only while the evidence is at most 14 days old.
