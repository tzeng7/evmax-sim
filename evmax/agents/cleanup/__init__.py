"""Cleanup agents — prediction logging, outcome resolution, and Brier score calibration.

Key modules:
- db.py          — predictions.db schema (raw sqlite3, NOT the SQLAlchemy ORM in evmax/db.py)
- logger.py      — writes EVGaps to ev_predictions + ev_outcomes after scan
- resolver.py    — auto-resolves outcomes via ESPN / bo3.gg; updates ev_outcomes.result
- prop_resolver.py — resolves prop bet outcomes via ESPN boxscores
- metrics.py     — computes Brier scores; auto-tunes sharp_weight in data/model_config.json
- value_audit.py — per-sector model-blend VALUE audit (Brier vs sharp & close + CLV) with
                   paired significance + actionability verdict; read-only (`evmax cleanup value-audit`)
- maintenance.py — prune old rows, vacuum DB
- adjustment.py  — manual outcome overrides and void handling

WARNING: predictions.db (sqlite3 here) and evmax.db (SQLAlchemy ORM in evmax/db.py) are
separate databases. The live pipeline only writes to predictions.db. The ORM models
(evmax/models/) are used only by the simulation/ module.
"""
