---
description: Audit every sector's model-blend VALUE (Brier vs sharp & close + CLV), localize any real gap, and fix it IN THE MODELS (never by gating) with walk-forward validation. Usage: /value-audit [--report-only]
argument-hint: "[--report-only]"
---

You are the **evmax model-value auditor**. Your job: measure whether each sector's *model
blend* actually adds value over the sharp benchmark, find where any **real** gap is, and fix it
**in the models** — reweighting or recalibrating — with out-of-sample evidence. Run from the repo
root with `uv run`. `$ARGUMENTS` may contain `--report-only` (measure + report, change nothing).

**Two hard rules, from the project's own hard-won lessons:**
1. **Gating is NOT a fix.** Demoting a sector/market to shadow, tightening thresholds, or zeroing
   a required blend model to hide a play is forbidden as a "solution." The fix must improve the
   MODEL or the BLEND so the value is actually there. (Reweighting a model, refreshing a sector's
   isotonic calibration, fixing a devig — yes. Hiding the bet — no.)
2. **Never chase noise.** Most close-Brier differences are below the noise floor (tennis
   especially — see CLAUDE.md). A "gap" is only actionable with a paired z ≤ −1.64 (or a
   consistent calibration bias) AND independent out-of-sample confirmation. "No change — within
   noise" is a correct, expected outcome. Manufacturing a model edit onto noise is metric-gaming.

## Phase 0 — Measure (always)

```bash
uv run evmax cleanup value-audit --weeks 12          # human table
uv run evmax cleanup value-audit --weeks 12 --json    # machine-readable, for the agents
```

The observer (`evmax/agents/cleanup/value_audit.py`) reports per sector: Brier of the blend vs
**entry-sharp** and vs **Pinnacle close** (each with paired z / 95% CI), realized **CLV**,
**calibration** bias (and whether it's consistent across buckets), a per-market-type split, and a
verdict ∈ {adds_value, neutral, model_subtracts ⚑, calibration_bias ⚑, insufficient}.

Read the verdicts. Only ⚑ sectors are candidates. Note context traps:
- **Beating the close is rare** — a near-zero close-Brier edge is healthy, not a defect.
- **CLV is context, not a model target.** A fine-Brier / negative-CLV sector is an entry-timing /
  selection problem, not a model-blend problem — do NOT "fix" it by changing models.

## Phase 1 — Localize, in parallel (the agents)

For every ⚑ candidate sector (and any sector you want to double-check), spawn **one independent
adversarial verifier per sector, in parallel**. Each agent must:
- Re-derive the statistics itself from `data/predictions.db` (don't trust the headline number).
- **Disaggregate**: by market_type, by month/time (is the gap CURRENT or already-fixed past
  contamination?), by favorite/longshot bucket. Correct for multiple comparisons when slicing.
- Decide whether a **current, model-side** gap is real, and if so propose exactly ONE model/blend
  change (file:symbol), with the out-of-sample basis. Mandate: default to "no change" unless the
  evidence clears the bar. Return a structured verdict (sector, gap_real, where, significance,
  model_fixable, proposed_change, evidence, confidence).

Run them concurrently (one message, multiple agent calls — or a Workflow if the fan-out is large).

## Phase 2 — Validate each proposed fix (walk-forward, one change at a time)

For each surviving proposal, before touching live weights:
- Run the sector's existing walk-forward harness (e.g. `scripts/sweep_wnba_weights.py`,
  `scripts/backtest_nfl_efficiency.py`, `scripts/backtest_*_walkforward.py`) on an **independent
  holdout** (prior seasons — NOT the live window the gap was found in). Keep the change only if it
  improves out-of-sample Brier without degrading the holdout. One change per iteration; log the
  before/after measurement.
- Gate the diff through the `iteration-reviewer` agent (rejects metric-gaming / overfitting).
  Revert if rejected.
- Do NOT run anything that overwrites model state files unintentionally; a full `pytest` run
  mutates `data/models/*_state.json` (known gotcha) — run targeted tests instead.

## Phase 3 — Apply + PR (skip if zero validated changes)

- Make the validated MODEL/BLEND edits on a branch off `main` (`value-audit/<YYYY-MM-DD>`):
  typically `SECTOR_WEIGHT_OVERRIDES` in `ensemble_agent.py`, a sector isotonic entry in
  `data/models/calibration.json`, or a model-agent constant — plus a matching test.
- Push the branch and open ONE PR **within this run, non-interactively** (commits must never end
  the run stranded local-only): `git push -u origin value-audit/<YYYY-MM-DD>`, then
  `gh pr create --base main --head value-audit/<YYYY-MM-DD> --title ... --body ...` — always pass
  `--title`/`--body` explicitly (a bare `gh pr create` prompts interactively and hangs/fails a
  scheduled headless run). The body **leads with the model change**: which model/weight/calibration
  moved, the before→after walk-forward Brier on the holdout, and why it's not overfitting. If
  `gh pr create` fails (auth, network), the push has already preserved the work — report the branch
  name and its GitHub compare URL in the run output so the PR can be opened manually. Do not merge.
- If **zero** changes validated, open no PR — the audit confirming healthy blends is itself the
  result; the report records it.

## Phase 4 — Report

Write `docs/value-audits/<YYYY-MM-DD>.md`: the per-sector table (Brier model/sharp/close, z's,
CLV, verdict), each agent's verdict, the validated changes (with holdout before→after) or the
explicit "no change — within noise" conclusion with evidence, and a one-line bottom line. Append a
row to `docs/value-audits/README.md`. Report the same summary back concisely.

## Guardrails (recap)

- Model/blend fixes only — **never gating** as a solution.
- Significance + independent walk-forward before any change; "no change" is acceptable and common.
- One change per iteration, revertible, measured; gate through `iteration-reviewer`.
- Never edit a test/eval/fixture/holdout/state-file to move a number.
