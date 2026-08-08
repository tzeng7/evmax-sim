"""Cleanup agents — prediction logging, outcome resolution, and Brier score calibration.

Key modules:
- db.py          — predictions.db schema (raw sqlite3, NOT the SQLAlchemy ORM in evmax/db.py)
- logger.py      — writes EVGaps to ev_predictions (game bets) and prop_observations
                   (rows whose event_id contains '::prop::'). It does NOT touch
                   ev_outcomes — resolver.py is the sole writer of that table.
- resolver.py    — auto-resolves outcomes via ESPN / bo3.gg / Kalshi+PolyUS settlement.
                   `_write_outcome` is the ONLY code path that INSERTs a row into
                   ev_outcomes, and it always supplies a non-null `outcome` (so no row is
                   ever pending — see the close-lines note in docs/SCHEDULED_RUNS.md).
                   ev_outcomes.pinnacle_close_prob is later UPDATEd in place, both here
                   and from evmax/cli/commands/cleanup.py
- prop_resolver.py — resolves prop bet outcomes via ESPN boxscores
- metrics.py     — computes Brier scores; auto-tunes sharp_weight in data/model_config.json
- value_audit.py — per-sector model-blend VALUE audit (Brier vs sharp & close + CLV) with
                   paired significance + actionability verdict; read-only (`evmax cleanup value-audit`)
- listings_eval.py — offline first-anchored-sweep entry evaluator over archive.db
                   watch-listings captures (or candlestick-backfill sessions via
                   session_prefixes); promotion lens for laddered markets;
                   read-only (`evmax cleanup listings-eval`)
- anchored_entry.py — LIVE first-anchored-sweep trigger: turns one watch-listings
                   sweep's in-memory data into shadow EVGaps at crossable
                   order-book prices (model_sources '+anchored_entry'); wired
                   via `cleanup watch-listings --log-entries` (2026-07-19,
                   validated by the WNBA candlestick backfill: lay +1.32pp
                   over 116 declustered games, p<0.001)
- rescan_eval.py — near-close re-scan replay: paired actual-scan-entry vs simulated
                   near-tip-entry CLV over archived snapshots; read-only research
                   harness (`scripts/eval_near_close_rescan.py`), Phase 1 gate for
                   the proposed `agents rescan` workflow (2026-07-11: REJECTED,
                   see docs/near-close-rescan-eval.md)
- maintenance.py — prune old rows, vacuum DB
- adjustment.py  — manual outcome overrides and void handling

WARNING: predictions.db (sqlite3 here) and evmax.db (SQLAlchemy ORM in evmax/db.py) are
separate databases. The live pipeline only writes to predictions.db. The ORM models
(evmax/models/) are used only by the simulation/ module.
"""
