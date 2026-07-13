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
| [2026-07-13](2026-07-13.md) | 12w (Apr–Jul) | **0 actionable gaps** | All blends within noise of sharp (no z ≤ −1.64, no consistent calib bias). 2 adversarial agents confirmed soccer +7.67pp = sparse-tail artifact (one n=1 bucket; N-weighted +1.68pp, flips to −1.50pp without it) and NBA-spread +6.42/1000 = noise (z=−1.47, 40/83 sharp-passthrough, blend beats close, stale window ends 05-25). No model change, no PR. |
| [2026-07-06](2026-07-06.md) | 12w (Apr–Jul) | **0 actionable gaps** | All blends within noise of sharp (no z ≤ −1.64, no consistent calib bias). 2 adversarial agents confirmed soccer +9.33pp = sparse-bucket artifact (N-weighted +2.52pp) and NBA-spread +5.11/1000 = noise (z=+1.15, stale window). No model change, no PR. |
| [2026-06-27](2026-06-27.md) | 16w (Apr–Jun) | **0 actionable gaps** | All blends within noise of sharp. Verified by 4 independent adversarial agents. No model change made — see report. |
