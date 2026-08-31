# Season-Start Runbook — bringing a sector back online for +EV scanning

What has to be true, checked, and re-seeded before evmax can find +EV plays in a
sector whose season is (re)starting. Written 2026-07-08; the upcoming boundary
dates are for the 2026-27 cycle. Per-sector windows come from `season_window`
in [`data/categories.yaml`](../data/categories.yaml) — that file stays the
single source of truth for modes/models/resolvers; this doc is the *operational*
companion for the season boundary.

---

## 1. What happens automatically at a season boundary

Understand this first — most of the machinery self-heals, and the checklist
below is only the part that doesn't.

| Mechanism | Behavior at season start | Where |
|---|---|---|
| **`season_window` scan gate** | The coordinator drops out-of-season sectors from default scans and auto-reopens them on the window's start date. No action needed to resume scanning (explicit `--sectors` bypasses the gate year-round). | `evmax/categories.py::is_in_season`, consumed by the coordinator |
| **Elo staleness guard** | `EloModelAgent` returns `None` when the sector's `last_updated` is > `STALE_DAYS=60` older than the game's event date. Every sector with a real offseason (NFL, NBA, NHL, NCAAB/W) opens the season with Elo **gated out** — it re-enters the blend only after the first resolves refresh `last_updated`, and then fires with **un-regressed end-of-last-season ratings**. | `evmax/agents/models/elo_agent.py:143` |
| **Form staleness guard** | Same 60-day rule against each team's most recent record. Form silently skips opening week and re-enters as fresh results accumulate. This is self-healing by design — no action needed. | `evmax/agents/models/form_agent.py:51` |
| **Resolve-time model auto-update** | The daily `daily-resolve-and-model-update` scheduled task (07:32) runs `evmax update scores` + `cleanup resolve`, feeding completed ESPN scores into Elo/Form/Poisson state for the game sectors and shot stats into the xG agent for soccer/worldcup. This is the self-heal path that un-gates Elo/Form after the first few game days. | `evmax/agents/cleanup/model_updater.py` |
| **NBA stat models self-refresh** | `efficiency` / `shot_quality` / `matchup` / `possession_sim` fetch current-season `LeagueDashTeamStats` from `nba_api` at scan time (with an ESPN-driven freshness check and a circuit breaker). No manual seed exists or is needed for NBA. | `evmax/agents/models/_nba_freshness.py` |
| **Source-season staleness guards (NFL, WNBA)** | `nfl_state_is_stale_for_today` / `state_is_stale_for_today` blank `nfl_efficiency`, `nfl_qb_elo`, `wnba_efficiency`, `wnba_possession_sim` whenever the seeded state's season is behind the active season — a frozen prior-season seed cannot silently fire. NHL's `nhl_xg` has **no such guard** (see Gaps). | `nfl_efficiency_agent.py:84`, `wnba_efficiency_agent.py:117` |
| **`weekly-seasonal-model-reseed` task** | Monday 07:04, season-aware: WNBA efficiency (May–Oct), NFL efficiency + QB Elo (Sep–Feb), NCAAF efficiency EPA (Aug–Jan), MLB pitcher_v2 (Mar–Nov), UFC Glicko-2 ratings (weekly, no offseason). NHL, NCAAB/W, and soccer are **not** in it. | `~/.claude/scheduled-tasks/weekly-seasonal-model-reseed/SKILL.md`, [`docs/SCHEDULED_RUNS.md`](SCHEDULED_RUNS.md) |

**Net effect of the guards:** for roughly the first week of any restarted
season, the blend degrades toward sharp-passthrough (Pinnacle devig at
`sharp_weight`). That is intended — it's strictly better than firing frozen
ratings — but it means *opening-week "edges" are mostly Kalshi-vs-Pinnacle arb,
not model edge*. Treat them accordingly (the tennis full-blend gate exists for
exactly this failure mode; other sectors have no equivalent gate — see Gaps).

---

## 2. Universal pre-season checklist (every sector, ~2 weeks before opening day)

1. **Verify Kalshi series tickers are live.** Kalshi retires/renames series
   between seasons and new ones are invisible until `SECTOR_SERIES_MAP`
   (`evmax/clients/kalshi.py`) is edited by hand.
   ```bash
   python3 scripts/check_kalshi_series.py --probe   # exit 1 on STALE tickers; NEW = missed opportunity
   ```
2. **Verify Polymarket US league slugs** for the 8 sectors in
   `POLYMARKET_US_LEAGUE_MAP` (`evmax/clients/polymarket_us.py`) — same drift
   risk, no checker script exists yet (see Gaps).
3. **Update alias maps for team churn** in `evmax/sectors/aliases/<sector>.yaml`:
   expansion teams, relocations, promoted/relegated clubs, renamed franchises.
   An unmapped name fails canonical-key matching and silently drops the game
   from the pool (fuzzy fallback at threshold 88 catches some, not all).
4. **Confirm Pinnacle league coverage.** `PinnacleGuestClient` fetches by
   sport/league; a league id that changed over the summer yields zero sharp
   anchors and therefore zero computable EV. Also remember the measured posting
   windows (memory: Pinnacle posts ~T-17–24h for US majors) — early-listed
   Kalshi markets are unanchored noise until Pinnacle posts.
5. **Run the sector's seed scripts** (per-sector section below) and sanity-check
   the state file's season stamp.
6. **Decide the mode.** Anything with a new model, new market type, or a full
   offseason of roster churn should open in `shadow` (edit
   `data/categories.yaml`) and promote via
   `evmax cleanup shadow promote <category>` only after the standard gate
   (≥30 clean resolved rows, CLV ≥ 0). Judge laddered markets (spread/total)
   on `evmax cleanup shadow clv <sector> -m spread --side lay`, never pooled
   Brier.
7. **Re-enable any season-parked scheduled tasks** (e.g.
   `weekly-nba-props-shadow-metrics`, disabled 2026-07-01, flagged in
   SCHEDULED_RUNS.md for re-enable when NBA restarts).
8. **First week: watch, don't trust.** Run `evmax cleanup shadow show` /
   `cleanup show` daily for matching/devig errors, and expect thin
   `model_sources` until the guards release.

---

## 3. Per-sector checklists (2026-27 boundary order)

### Soccer (club) — European seasons restart ~mid-August 2026 · `live`

The sector never leaves season (no `season_window`; MLS bridges the summer), so
Elo/Form state stays warm. The season boundary work is **team churn**, not
staleness:

- [ ] Add promoted clubs (EPL, La Liga, Bundesliga, Serie A, Ligue 1
      promotions) to `evmax/sectors/aliases/soccer.yaml`.
- [ ] Promoted clubs have no Elo/Poisson/xG history → they enter at defaults
      with near-zero confidence. Optionally backfill their second-division
      results via `python scripts/seed_espn.py --sectors soccer` (ESPN
      coverage of lower divisions is spotty; if unseeded they just ride the
      sharp anchor for their first ~10 matches — acceptable).
- [ ] xG state (`soccer_xg_state.json`) self-heals via resolve-time
      `record_match`; verify it's flowing after week 1: new teams should have
      entries. One-off backfill: `python scripts/seed_soccer_xg.py --since <date>`.
- [ ] Confirm `KXEPLGAME` etc. tickers via `check_kalshi_series.py` (Kalshi
      has re-cut soccer series before).

### NFL — kickoff ~Sep 10, 2026 (window opens 09-04) · `live`

**Have:** `seed_nfl_efficiency.py`, `seed_nfl_qb_elo.py`, the Sep–Feb block of
`weekly-seasonal-model-reseed`, `nfl_state_is_stale_for_today` guard,
`season_window` auto-reopen.

- [ ] **Manually run both seeds the week of kickoff** — don't rely on the
      Monday task alone. Until 2026 PBP exists in `nflreadpy`, the seeds write
      `seasons_used` maxing at 2025 and the staleness guard keeps
      `nfl_efficiency`/`nfl_qb_elo` blanked, so **Week 1 runs on sharp-only**
      (Elo 60d-stale from February, Form stale, both NFL models guarded).
      That's by design; just know the first real model-informed week is Week 2,
      after the first Monday reseed ingests Week 1 PBP:
      ```bash
      uv run python scripts/seed_nfl_efficiency.py
      uv run python scripts/seed_nfl_qb_elo.py     # also refreshes current_starters
      ```
- [ ] Verify `current_starters` in `nfl_qb_elo_state.json` reflects offseason
      QB moves after the first reseed with 2026 data (the per-QB delta layer is
      the whole point of this model).
- [ ] Alias check for any franchise rename/relocation.
- [ ] Generic Elo (0.20 weight) carries raw February ratings into September —
      NFL is the #2 target for offseason regression (section 5; ⅓-toward-mean
      prior). `nfl_qb_elo` and `nfl_efficiency` don't need it (weekly full
      re-walk-forward / built-in season decay).
- [ ] Do **not** re-attempt the backtest-rejected levers (QB-Elo MOV,
      form-weight redistribution — form stays 0.30; see the NFL blend audit
      memory) without new evidence.

### NFL props — same window · `disabled` (PR #185, 2026-08-08), `status: blocked`

This is the biggest "produce" item of the fall (MODEL-9):

- [x] **Fix the phantom Kalshi series names.** Done 2026-04-15 (`58e6150`):
      `SECTOR_SERIES_MAP["nfl_props"]` now holds `KXNFLPASSYDS`, `KXNFLRSHYDS`,
      `KXNFLRECYDS`, `KXNFLANYTD`, `KXNFLPASSTDS`, `KXNFLREC`, and
      `check_kalshi_series.py` reports `STALE: 0` against the live Kalshi
      `/series` API. Re-run that checker once Kalshi lists 2026 NFL props to
      confirm the series still carry markets.
- [ ] Start the weekly feature refresh: `scripts/fetch_nfl_features.py`
      (repopulates `data/backtest/nfl_props/*.parquet` that
      `evmax/clients/nfl_props_cache.py` reads point-in-time). Needs a
      scheduled home — it's in no task today (see Gaps).
- [ ] **Re-enable `nfl_props` to `shadow` first** — it is `disabled` today, and
      a disabled category skips persistence entirely, so no rows accrue.
- [ ] Then accumulate 3–4 weeks of shadow rows against pre-game prices and apply
      the props promotion gate (predict-37%/Brier — see
      `project_props_validation_gate` memory). QB-only v1; NO-side ROI +79% in
      backtest is *not* a promotion basis on its own.
- [ ] Respect the nba_props lesson: gate out the cheap-longshot bucket before
      any live flip.

### NHL — puck drop ~early Oct 2026 · `shadow`

**Have:** `seed_nhl_xg.py` (MoneyPuck 5v5, no incremental path), form
self-heal, resolver wired. **This sector has the most missing scaffolding:**

- [ ] Seed at season open and **weekly thereafter**:
      `python scripts/seed_nhl_xg.py --season 2026` (MoneyPuck keys seasons by
      start year). Early-season note: the agent's confidence ramps over
      `LOW_CONF_GAMES=25` / `HIGH_CONF_GAMES=50`, so October predictions run
      at reduced confidence regardless — decide whether to seed with prior
      season data initially or accept the low-confidence ramp (document the
      choice in categories.yaml notes).
- [ ] **Produce: add an NHL block to `weekly-seasonal-model-reseed`**
      (Oct–Jun). Today nothing reseeds `nhl_xg` on a schedule — it will
      silently freeze exactly like the WNBA efficiency incident.
- [ ] **Produce: a source-season staleness guard for `nhl_xg`** (mirror
      `nfl_state_is_stale_for_today`; the state already carries
      `season_start_year`). Without it, a 2025-26 seed fires at full 0.30
      weight into October 2026 games.
- [ ] Check aliases for relocations (the Utah Mammoth precedent is already
      handled in the agent's abbreviation map — verify nothing new).
- [ ] Elo remains excluded from the NHL ensemble (uncalibrated K — MODEL-2 /
      SECTOR-1). Calibrating it is optional pre-season work, not a blocker.
- [ ] Promotion: still shadow — gate on Brier vs the ~0.225 sharp baseline
      once ≥30 clean resolved rows exist.

### NBA — opening night ~Oct 20, 2026 · `live`

**Have:** the best self-maintaining stack — efficiency/shot_quality/matchup/
possession_sim auto-fetch current-season stats from `nba_api`; playoff-blend
helper handles season-type transitions; Elo/Form self-heal via resolves.

- [ ] **No seed scripts to run.** Verify on opening week that `nba_api`
      returns 2026-27 rows (small-sample early season is handled by the
      agents' MIN_GAMES/confidence ramps) and that the circuit breaker isn't
      pinned (`data/models/.nba_api_breaker.json`).
- [ ] Elo opens gated (last update = June Finals) and re-enters after the
      first resolves **with un-regressed ratings** — but at 0.10 weight behind
      four self-refreshing NBA models this is marginal (section 5 verdict:
      regress for consistency or skip).
- [ ] Re-enable `weekly-nba-props-shadow-metrics` if nba_props promotion
      tracking resumes (`nba_props` stays shadow until the long-shot-bias
      root-cause work from its categories.yaml note happens; do not promote on
      a hot month).
- [ ] Alias check (rare in NBA, but free).

### NCAAB / NCAAW — Nov 1, 2026 · `live`

The highest roster-churn sectors (transfer portal turns over half a rotation)
with the *least* season-start tooling:

- [ ] Reseed Elo/Form/Poisson from ESPN before opening week:
      `python scripts/seed_espn.py --sectors ncaab,ncaaw` (despite the
      script's stale docstring, it covers both). This mostly refreshes
      `last_updated` and team coverage — ratings still encode last season's
      rosters.
- [ ] **NCAAB is the #1 target for offseason Elo regression** (section 5):
      with no `SECTOR_WEIGHT_OVERRIDES` entry and Poisson excluded from
      basketball, Elo is ~58% of the model blend, Form is stale-gated in
      opening weeks, and transfer-portal churn is the worst of any sector.
      Ship the generalized regress script for NCAAB before Nov 1.
- [ ] NCAAW: Elo contributes 0 (uncalibrated K / home-adv — MODEL-2 /
      SECTOR-2), so it opens as form+sharp and form is stale-gated → NCAAW is
      effectively **sharp-only until mid-November**. Calibrating NCAAW Elo
      before the season is the highest-leverage pre-season model task here.
- [ ] Conference realignment → alias updates (schools change names/leagues
      constantly; run one scan early and grep logs for match failures).
- [ ] Verify `KXNCAABGAME` / `KXNCAAWBGAME`-family tickers.

### Already mid-season (no action from this runbook)

WNBA, MLB (+ `baseball_props`), tennis (year-round, Tuesday TA refresh task),
worldcup (window closes 07-19 — after the final, its shadow record feeds the
promote/park decision), esports (year-round, sharp-only). The **WNBA playbook**
(`wnba_offseason_regress.py` + `seed_wnba_efficiency.py --year` + the
`state_is_stale_for_today` guard) is the reference implementation the gaps
below generalize from — its offseason-gap incident (+24pp chalk bias, May
2026) is the canonical example of what this runbook prevents.

---

## 4. What we need to produce (gap list, ranked)

1. **NHL reseed block + `nhl_xg` staleness guard** (before Oct). Smallest
   work, prevents a known-class silent failure. Add the block to
   `weekly-seasonal-model-reseed`, add a `season_start_year` guard mirroring
   `nfl_state_is_stale_for_today`, plus tests (Testing Policy applies —
   `evmax/agents/models/` change).
2. **NFL props MODEL-9 unblock** (Sep): real series tickers in
   `SECTOR_SERIES_MAP`, a scheduled home for `fetch_nfl_features.py`
   (weekly, Sep–Feb), then the shadow-validation clock starts. Without the
   ticker fix the category fetches nothing at all.
3. **Generalized offseason Elo regression — NCAAB first, NFL second** (NCAAB
   before Nov 1). Only WNBA regresses ratings across the offseason today;
   everywhere else Elo re-enters the blend after week 1 with raw
   end-of-last-season ratings. See section 5 for the full per-sector verdict —
   it is NOT needed everywhere. Generalize `wnba_offseason_regress.py` into
   `scripts/offseason_regress.py --sector <s> [--moves <yaml>]` — shrinkage
   toward 1500 with a per-sector coefficient; the roster-move YAML stays
   optional and human-reviewed. Keep it manual-by-design like WNBA's.
4. ~~**NCAAW Elo calibration** (before Nov, MODEL-2/SECTOR-2)~~ — **DONE
   2026-07-11.** Generic Elo now runs NCAAW-calibrated `K=35` /
   `HOME_ADVANTAGE_ELO=80` (`elo_agent.py`), swept on 2024-25 and held out on
   2025-26. The NHL half of MODEL-2 closed later, on 2026-07-18
   (`K=6` / `HOME_ADVANTAGE_ELO=48`); raising NHL's 0 ensemble weight is now an
   open blend decision, not a blocked calibration.
5. **Polymarket US league-slug checker** — `check_kalshi_series.py` equivalent
   for `POLYMARKET_US_LEAGUE_MAP`; both venues share the season-boundary drift
   risk (matters more once the venue promotion gate clears).
6. **Season-open sharp-passthrough visibility** (nice-to-have): during the
   guard window most sectors' `model_sources` shrink to `[sharp]`. Tennis
   demotes those rows to shadow via `REQUIRED_BLEND_MODELS`; other sectors
   just show inflated "edges". Even a `Models` column glance-check in the scan
   table, or a warning when `model_sources == ['sharp']` in a normally
   multi-model sector, would prevent opening-week overbetting.

## 5. Offseason Elo regression — per-sector verdict

Not every sector needs it. Two code facts frame the decision:

- **The agent-side machinery is already general.** `EloModelAgent` carries a
  `season_games` counter (reset by an offseason regression) and a 538-style
  `EARLY_K_BOOST` that amplifies K while post-regression games accumulate —
  and the comment at `elo_agent.py:72` explicitly says NBA/NFL would benefit
  but must not be enabled "until you also wire its offseason script to reset
  `season_games`". The build is a generalization of
  `scripts/wnba_offseason_regress.py`, not new model code.
- **Exposure varies enormously with the effective ensemble weight.** A sector
  where Elo is 10% of the blend barely notices un-regressed ratings; a sector
  where it's the dominant model gets the full WNBA-style chalk bias.

| Sector | Effective Elo weight | Verdict |
|---|---|---|
| **NCAAB** | ~0.35 class weight (**no `SECTOR_WEIGHT_OVERRIDES` entry**); with Poisson hard-excluded from basketball, the model side is just elo+form → **Elo ≈ 58% of the model blend after normalization** | **YES — strongest case by far.** Transfer-portal roster churn is the worst of any sector, and since Form is stale-gated in opening weeks, raw March ratings would be the *only* model firing into November games. The WNBA +24pp-chalk-bias pattern at higher weight. Ship before Nov 1. |
| **NFL** | 0.20 (generic elo only) | **YES, moderate.** `nfl_qb_elo` is re-walk-forwarded from PBP weekly and `nfl_efficiency` has season-decay 0.45 built in — only generic Elo carries raw February ratings. FiveThirtyEight's classic ⅓-toward-mean is the prior. |
| Baseball | 0.25 | Include in the script, defer the run — K=6 over 162 games self-corrects fast, the sector is shadow, and the next boundary is March 2027. |
| NBA | 0.10 | Marginal. The four NBA-specific models (0.80 combined) self-refresh from `nba_api`; regressing a 0.10-weight input barely moves the blend. Do it for consistency or skip. |
| Soccer (club) | 0.15 | **NO.** Subtlety: the staleness guard never trips (MLS keeps the sector's `last_updated` fresh through the European summer), so August ratings do fire un-regressed — but club soccer has the highest year-over-year squad persistence of any sport (ClubElo doesn't regress between seasons at all). The real August work is promoted-club aliases. |
| NHL, NCAAW | 0 (Elo gated out of both blends — uncalibrated K, MODEL-2) | **Moot** until the SECTOR-1/2 calibrations happen; add regression then. |
| Tennis, esports, worldcup | — | **NO** — per-player tennis Elo is reseeded weekly from Tennis Abstract; esports is sharp-only; worldcup is seeded from scratch each cycle. |
| WNBA | 0.15 | ✅ Already has it (`wnba_offseason_regress.py` + roster-move YAML + `EARLY_K_BOOST`). The template. |

**Implementation rules (house policy applies):**

1. Do NOT copy WNBA's 35% coefficient blind. Sweep the coefficient per sector
   via walk-forward replay — predict each of the last few seasons' opening
   ~6 weeks with vs. without regression, keep on Brier, one change per
   iteration. Priors: NFL ≈ ⅓ (538), NBA ≈ 25%, NCAAB likely *higher* than
   WNBA's 35% given portal churn.
2. Only add an `EARLY_K_BOOST` entry for a sector once its regression actually
   resets `season_games` — exactly as the `elo_agent.py` comment warns
   (boosting without the reset just amplifies late-season noise).
3. Roster-move deltas stay optional and human-reviewed (the YAML pattern);
   pure shrinkage is the automated part.

## 6. Season-start inventory (quick reference)

| Sector | Next start | Mode | Seed script(s) | Scheduled reseed | Staleness guard | Offseason Elo regression |
|---|---|---|---|---|---|---|
| soccer | ~Aug 15 | live | `seed_espn.py`, `seed_soccer_xg.py` (backfill only) | resolve-time auto | elo/form 60d (rarely trips — year-round) | n/a (no offseason) |
| nfl | Sep 4 window | live | `seed_nfl_efficiency.py`, `seed_nfl_qb_elo.py` | ✅ weekly task (Sep–Feb) | ✅ source-season + elo/form 60d | ❌ build — #2 target (§5) |
| nfl_props | Sep 4 window | shadow (blocked) | `fetch_nfl_features.py` | ❌ none | n/a (cache) | n/a |
| nhl | ~Oct 7 | shadow | `seed_nhl_xg.py` | ❌ none | ❌ none on nhl_xg | ❌ none (elo not in blend) |
| nba | ~Oct 20 | live | none needed (nba_api self-fetch) | auto at scan time | freshness helper + elo/form 60d | ❌ none |
| ncaab | Nov 1 window | live | `seed_espn.py --sectors ncaab` | ❌ none | elo/form 60d | ❌ build — #1 target (§5) |
| ncaaw | Nov 1 window | live | `seed_espn.py --sectors ncaaw` | ❌ none | elo/form 60d (elo weight 0 anyway) | ❌ none |
| wnba | May (template) | live (ML) | `seed_wnba_efficiency.py --year` | ✅ weekly task (May–Oct) | ✅ source_season | ✅ `wnba_offseason_regress.py` |
