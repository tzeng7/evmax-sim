# evmax Implementation Plan

> Checkpoints marked ✅ when complete. Work top-to-bottom.

---

## BATCH A — Quick Fixes (no DB changes, isolated)

### A1. Fix `utcnow()` deprecation warnings (#10)
**Files:** `evmax/models/ev_bet.py:64`, `evmax/models/simulated_bet.py:66`
**Change:** `datetime.utcnow()` → `datetime.now(timezone.utc)` in both Pydantic default fields. Add `timezone` to imports.
**Test:** `uv run pytest tests/ -q` — no deprecation warnings.
**Checkpoint:** ✅

---

### A2. Fix crontab missing `cd` prefix (#9)
**Current broken entries:**
```
0 23 * * * .venv/bin/evmax cleanup resolve ...    # no cd
0 1 * * *  .venv/bin/evmax agents scan ...        # no cd
```
**Fix:** Prepend `cd /Users/ktzeng/Projects/evmax &&` to every cron entry that lacks it.
**Expected final crontab (4 entries):** all 4 lines start with `cd /Users/ktzeng/Projects/evmax &&`.
**Checkpoint:** ✅

---

## BATCH B — Infrastructure (other features depend on these)

### B1. Pinnacle odds in-memory cache (#7)
**File:** `evmax/clients/pinnacle.py`
**Design:**
- Module-level `_CACHE: dict[str, tuple[float, list[SharpOdds]]] = {}` (key=sector, value=(timestamp, results))
- TTL = 300 seconds (5 min)
- Wrap `get_odds()`: if cached and age < TTL, return cached result; else fetch and store
- Log `pinnacle_cache_hit` vs `pinnacle_cache_miss` via structlog
- Cache is per-process (in-memory) — no file I/O overhead

**Test:** Verify second call returns same object without extra HTTP calls.
**Checkpoint:** ✅

---

## BATCH C — Database Schema Changes

### C1. CLV tracking — add `pinnacle_close_prob` column (#1)
**Files:** `evmax/agents/cleanup/db.py`, `evmax/cli/commands/cleanup.py`
**Design:**
1. Add `pinnacle_close_prob REAL` column to `ev_outcomes` schema
2. Add migration block in `get_connection()` (same pattern as existing `voided` migration)
3. New CLI command: `evmax cleanup close-lines [--date DATE]`
   - Queries `ev_outcomes` for rows with `outcome IS NULL` and `event_date = DATE`
   - For each unresolved market, looks up `sharp_true_prob` from `ev_predictions`
   - Fetches current Pinnacle lines via `PinnacleClient` for each sector
   - Matches by `event_id` and writes `pinnacle_close_prob` to `ev_outcomes`
   - CLV = `sharp_true_prob (at entry) - pinnacle_close_prob (at close)`
4. Update `show` command: add `CLV` column — shows `+X.Xpp` if `pinnacle_close_prob` available, else `—`

**Intent:** Run `evmax cleanup close-lines` at game start time (add cron at T-15min, or manually).
**Checkpoint:** ✅

---

## BATCH D — Core Agent Features

### D1. Confidence stars on EVGap (#8)
**Files:** `evmax/agents/odds/ev_gap_agent.py`, `evmax/cli/commands/agents.py`
**Design:**
- Add `confidence_stars` property to `EVGap` dataclass:
  ```
  +1 star: match_confidence >= 0.90
  +1 star: volume_usd >= 5000
  +1 star: model_sources not in ("sharp", "sharp(capped)")  — has real model signal
  Result: 0–3 stars, shown as ★★★ / ★★☆ / ★☆☆ / ☆☆☆
  ```
- Add `Stars` column to scan table in `agents.py` (between `Sources` and end of row)
- Also shown in `summary()` string

**Checkpoint:** ✅

---

### D2. Correlated exposure guard (#3)
**File:** `evmax/agents/coordinator.py`
**Design:**
- Add helper `_apply_exposure_guard(gaps, max_event_exposure=0.08)` in coordinator
- Groups gaps by base event key: strip `::spread`, `::total` etc. from event_id
- Sort by `ev_pct` descending (best plays get budget first)
- Iterate: if adding a gap's `kelly_fraction` would push event total over 8%, scale it down to fit remaining budget; if remaining < 0.5%, skip entirely
- Use `dataclasses.replace(gap, kelly_fraction=scaled)` to produce a capped copy
- Call after `run_cycle()` collects all ev_gaps, before returning result
- Log `exposure_guard_capped` when a gap is scaled down

**Test:** Two overlapping bets on same game: combined kelly never exceeds 8%.
**Checkpoint:** ✅

---

### D3. Steam move detection (#2)
**Files:** `evmax/agents/coordinator.py`, `evmax/agents/odds/ev_gap_agent.py`
**Design:**
- Add `steam_move: bool = False` field to `EVGap` dataclass
- Add `data/steam_cache.json` — stores `{event_id: sharp_true_prob}` from the previous scan
- In `_run_sector()`, after `SharpOddsAgent` returns `sharp_odds`:
  - Load cache; compare each event's current `true_prob_a` to cached value
  - Build `steam_events: set[str]` — event_ids where abs delta >= 0.02 (2pp)
  - Save updated probs to cache
  - Pass `steam_events` into `EVGapAgent` via `request.params`
- In `EVGapAgent._evaluate_pair()`: set `steam_move=True` if `sharp.event_id in steam_events`
- In `agents.py` table: show `⚡` prefix on EV% cell for steam moves

**Checkpoint:** ✅

---

## BATCH E — New Sector

### E1. MLB sector (#5)
**Files (4):**
1. `evmax/sectors/registry.py` — register `BaseballHandler`, add `"baseball"` to `ALL_SECTORS`
2. `evmax/clients/pinnacle.py` — add `"baseball": ["baseball_mlb"]` to `SECTOR_SPORT_KEYS`
3. `evmax/sectors/aliases/baseball.yaml` — common abbreviations (NYY, BOS, LAD, SF, etc.)
4. `evmax/agents/cleanup/resolver.py` — add `"baseball": ("baseball", "mlb", {})` to `ESPN_SPORT_MAP`

**Kalshi series:** Baseball Kalshi markets use `KXMLB` series prefix — `KalshiOddsAgent` already filters by sector keywords from the handler, no changes needed there.
**Checkpoint:** ✅

---

## BATCH F — Backtest Verification

### F1. Verify backtest CLI works end-to-end (#6)
**Existing files:** `evmax/backtest/engine.py`, `evmax/backtest/models.py`, `evmax/backtest/display.py`, `evmax/cli/commands/backtest.py`
**Steps:**
1. Read `backtest/engine.py` and `cli/commands/backtest.py` to confirm they're wired and runnable
2. Run `evmax backtest run --sectors soccer --seasons 2425` (dry run with `--help` first)
3. If broken: identify and fix the minimal wiring issue
4. If working: document the command in PLAN.md and update MEMORY.md

**Checkpoint:** ✅ — CLI loads, `evmax backtest run --help` works, engine is fully wired.

---

## Execution Order

| Order | Item | Batch | Risk | Est. files changed |
|-------|------|-------|------|-------------------|
| 1 | utcnow fix | A | none | 2 |
| 2 | crontab fix | A | low | crontab |
| 3 | Pinnacle cache | B | low | 1 |
| 4 | Confidence stars | D | low | 2 |
| 5 | CLV tracking | C | medium | 2 + new command |
| 6 | Correlated exposure guard | D | medium | 1 |
| 7 | Steam moves | D | medium | 2 |
| 8 | MLB sector | E | low | 4 |
| 9 | Backtest verify | F | low | 0–2 |

---

## Key Constraints / Decisions

- **CLV uses Pinnacle entry price vs Pinnacle close price** — not Kalshi (Kalshi goes to 0.99 at settlement, useless as a closing line signal)
- **Steam cache is per-process in `data/steam_cache.json`** — persists between cron runs; if the file is missing, first scan populates it silently (no steam flags)
- **Exposure guard scales down kelly, doesn't drop bets** — scaled bets are still logged and displayed; use `model_sources` suffix `+(capped)` to indicate
- **MLB Kalshi series:** `KalshiOddsAgent` already handles series-based filtering; `BaseballHandler` just needs to be in the registry
- **Backtest** already has its CLI command in `app.py` — just needs to be verified working
