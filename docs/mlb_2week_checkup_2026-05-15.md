# MLB Pipeline 2-Week Checkup — 2026-05-15

## Pipeline Health

- **Calibration refit outcome:** FAILED (ESPN scoreboard API returning 403 Forbidden — see Red Flags below)
- **2024 calibrated Brier:** N/A — walk-forward produced 0 predictions due to ESPN 403; reference from 2026-05-01 commit: 0.2490 (target ~0.2490)
- **2025 calibrated Brier:** N/A — walk-forward produced 0 predictions due to ESPN 403; reference from 2026-05-01 commit: 0.2395 (target ~0.2393)
- **FIP reseed commits in last 2 weeks:** 1 (most recent: `dd034ff` on 2026-05-03 by Claude — this was the scheduled Sunday "Weekly Model Calibration" run at 08:00 PDT; Monday FIP reseeds for May 4 and May 11 produced no-ops because FIP/ERA values were unchanged, which is documented normal behavior)

## Yellow / Red Flags

### 🔴 RED: `baseball_ensemble` calibration accidentally wiped (2026-05-10)

The isotonic calibration fitted on 2026-05-01 (`ade02c3`, n=2536, Brier 0.24902 → 0.24413) was
**removed from `data/models/calibration.json`** in commit `f8678fc` ("nba: playoff regular-season
blend + per-stat prop calibration", 2026-05-10). The NBA playoff work rewrote `calibration.json`,
discarding the `baseball_ensemble` key in the process.

Impact: baseball predictions have been running without isotonic calibration since May 10. Without
the calibration, the ensemble is over-confident in high-probability ranges (the original bug the
calibration fixed — 2024 uncalibrated Brier 0.2552 vs calibrated 0.2490). Baseball was also demoted
to shadow mode on 2026-05-10 (`806e6c5`), so no bankroll exposure occurred — but shadow predictions
logged since May 10 carry uncalibrated probabilities, which will bias the Brier you measure locally
if you include the post-May-10 shadow rows.

**Recommended fix:** restore the `baseball_ensemble` key from the last known good state:

```bash
git show f9e6828:data/models/calibration.json > data/models/calibration.json
git add data/models/calibration.json
git commit -m "fix: restore baseball_ensemble calibration accidentally wiped in f8678fc"
```

The pre-wipe calibration had `n_samples=2536`, `brier_before=0.24902`, `brier_after=0.24413`.

### 🔴 RED: ESPN scoreboard API returning 403 for all MLB months (both seasons)

Every request to `site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard` returned
403 Forbidden during this run. This blocked the calibration refit entirely (0 predictions
collected across all 16 month-fetches). The same issue has likely caused the `mlb-calibration-refit`
scheduled routine (Mon 04:00 PDT) to fail silently on May 4 and May 11 as well — neither run
produced a commit.

This is a cloud-environment blocking issue (ESPN blocks datacenter IP ranges). The FIP seeder
(`seed_pitcher_fip.py`) uses Baseball Reference via pybaseball and is unaffected, but any
walk-forward or ESPN-based backtest cannot run in the remote environment.

**Recommended fix:** run the calibration refit locally where ESPN isn't blocked:

```bash
source .venv/bin/activate
python scripts/fit_baseball_calibration.py --train 2324+2425
# Expected: ~20 min; PROMOTE if the fit improves both seasons by ≥0.001
git add data/models/calibration.json
git commit -m "MLB: re-fit baseball calibration (2-week checkup)"
git push
```

### 🟡 YELLOW: `mlb-calibration-refit` routine may have been silently failing since May 4

Per `docs/SCHEDULED_RUNS.md`, the `mlb-calibration-refit` routine runs every Monday at 04:00 PDT.
No commit was found from May 4 or May 11 runs. The calibration script exits with code 1 (not 0)
when ESPN 403s prevent data collection — it's unclear whether the routine page shows these as
failures or no-ops. Check the routines page: <https://claude.ai/code/routines/trig_01U8USfUfcBdphqEo5aS6e7j>

### 🟡 YELLOW: baseball in shadow mode — Brier drift since May 10 is uncalibrated

Shadow rows logged from 2026-05-10 onward lack the isotonic correction. When you run
`evmax cleanup shadow metrics --category baseball`, split the window at May 10 if possible
to distinguish calibrated vs uncalibrated shadow predictions.

## Local Checklist

The upstream pipeline has **two issues requiring local action** before baseball can be promoted
back to live: (1) restore the wiped calibration and (2) re-run the calibration refit locally.
After fixing those, run these on your laptop where `predictions.db` lives:

```bash
# 0. Restore the wiped calibration first (remote can't do this — ESPN is 403-ing)
git pull
git show f9e6828:data/models/calibration.json > data/models/calibration.json
python scripts/fit_baseball_calibration.py --train 2324+2425
git add data/models/calibration.json && git commit -m "fix: restore + refit baseball calibration"
git push

# 1. Brier on resolved baseball bets in the last 2 weeks
evmax cleanup metrics --weeks 2 --sector baseball

# 2. Recent baseball bets and their outcomes
evmax cleanup show --days 14 --sector baseball

# 3. Open baseball positions
evmax sim list --status open

# 4. Shadow validation — note: rows since May 10 are uncalibrated (see Red Flags)
evmax cleanup shadow show --days 14 --category baseball
evmax cleanup shadow metrics --days 14 --category baseball
```

## Go / No-Go Criteria

Decide based on the local commands above:

- **✅ GO (keep current ensemble):** ROI on resolved baseball bets is positive AND Brier on resolved bets is roughly in line with backtest expectation (Brier ~0.24 in production). The model-side improvements are showing up in dollars.

- **⚠️ INVESTIGATE (don't change anything yet):** ROI is flat or slightly negative but Brier looks calibrated. Probably needs more sample size; revisit in another 2 weeks.

- **❌ NO-GO (consider rollback):** Brier on resolved bets is materially worse than backtest expectation (>0.005 worse) AND ROI is meaningfully negative. The backtest predictions aren't generalizing to live; revert to pre-2026-05-01 ensemble or reduce baseball Kelly fraction temporarily.

> **Note:** The GO/NO-GO evaluation should use only shadow rows from **before 2026-05-10** until
> the calibration is restored and re-fitted. Post-May-10 rows carry uncalibrated probabilities and
> will inflate Brier. Once the fix commit is merged, rows from May 15 onward will be calibrated again.

## Suggested Next Steps

1. **Immediately: restore the `baseball_ensemble` calibration.** Use `git show f9e6828:data/models/calibration.json` to recover the pre-wipe state, then re-run `fit_baseball_calibration.py` locally to get a fresh fit. The pre-wipe fit is fine to restore as a stopgap — it was fitted on 2536 games across two seasons and passed the promotion bar on both.

2. **Check the `mlb-calibration-refit` routine logs** at <https://claude.ai/code/routines/trig_01U8USfUfcBdphqEo5aS6e7j> to confirm whether May 4 and May 11 runs are logged as errors or silently suppressed. If they're silently swallowed, add a Slack alert for non-zero exit codes from this routine.

3. **Add a calibration key audit to the NBA playoff / per-sector calibration scripts.** The wipe happened because the NBA script rewrote `calibration.json` without preserving non-NBA keys. A simple guard — `current = json.loads(path.read_text()); current.update(new_keys); path.write_text(json.dumps(current))` instead of a full overwrite — would prevent cross-sector clobbering.
