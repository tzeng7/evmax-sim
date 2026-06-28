# Model-blend value audits

Reports from the **Brier/CLV value observer** — `evmax cleanup value-audit` — which measures,
per sector, whether our *model blend* actually adds value over the sharp benchmark, and whether
any apparent gap is real signal or noise.

- **Tool:** [`evmax/agents/cleanup/value_audit.py`](../../evmax/agents/cleanup/value_audit.py) · CLI `evmax cleanup value-audit [--weeks N] [--sector S] [--json]`
- **What it measures (per sector):** Brier of the blend vs **entry-sharp** (`sharp_true_prob`) and vs **Pinnacle close** (`pinnacle_close_prob`), each with a paired z-score / 95% CI; realized Kalshi **CLV**; **calibration** bias (signed, and whether consistent across buckets); a per-market-type split; and an **actionability verdict**.

## Reading the verdict

| Verdict | Meaning | Action |
|---|---|---|
| `adds_value` | blend significantly beats entry-sharp (z ≥ 1.64) | none — working as intended |
| `neutral` | within noise of the sharp line | none — **expected** for most sectors |
| `model_subtracts` ⚑ | blend significantly **worse** than entry-sharp | reweight / recalibrate the **models** |
| `calibration_bias` ⚑ | systematic, *consistent* over/under-confidence | isotonic recalibration |
| `insufficient` | < 30 resolved rows | wait for data |

Only ⚑ verdicts are model-actionable. **The fix is always model-side (reweight / recalibrate) — never gating plays.** Two cautions baked into the tool:

1. **Beating the *close* is rare and not expected.** The closing Pinnacle line is the sharpest public estimate; a near-zero close-Brier edge is healthy, not a defect. Per CLAUDE.md, close-Brier differences are routinely *below the noise floor* (tennis especially) — do not fine-tune weights to chase them.
2. **CLV is context, not a model target.** A sector with a fine Brier but negative CLV is an *entry-timing / selection* problem (stale Kalshi price), not a model-blend problem, so it is **not** tagged model-actionable.

## Audit history

| Date | Window | Verdict | Notes |
|---|---|---|---|
| [2026-06-27](2026-06-27.md) | 16w (Apr–Jun) | **0 actionable gaps** | All blends within noise of sharp. Verified by 4 independent adversarial agents. No model change made — see report. |
