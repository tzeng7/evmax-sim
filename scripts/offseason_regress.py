"""Offseason Elo regression for a sector — shrink end-of-season Elo toward 1500.

Generalizes scripts/wnba_offseason_regress.py (which stays as the WNBA entry
point because it carries the reviewed roster-move YAML). For every other
sector the automated part is pure shrinkage: each team's rating moves toward
DEFAULT_ELO so a full offseason of roster churn is not carried into Week 1 at
end-of-season confidence, and the per-team `season_games` counter is reset so
the early-season K boost (where a sector has one) restarts.

`keep` is the fraction of the deviation from 1500 RETAINED across the boundary
(538's "regress one third toward the mean" is keep=0.667). Per-sector values
MUST come from a walk-forward sweep, never copied across sports
(docs/SEASON_START.md §5):

  nfl  0.667  scripts/backtest_nfl_elo_regression.py, 2026-09-02 — replay of
              2015–2025 with the production Elo update; keep swept over
              {1.0, 0.9, 0.8, 0.75, 0.667, 0.6, 0.5}. 0.667 was best on the
              RANK window (2019–23 opening-6-weeks Brier 0.2307 vs 0.2371 with
              no regression, +6.4/1000), held on CONFIRM 2024 (+6.9/1000) and
              on the untouched HOLDOUT 2025 (0.2403 vs 0.2533, +13.0/1000;
              full-season 0.2273 vs 0.2401). Every early-K boost variant
              (1.5/2/3 × decay 4/6/8) was WORSE on rank and confirm, so NFL
              gets no EARLY_K_BOOST entry.

Also drops non-team keys that exhibition games leave in the state (the NFL
Pro Bowl feeds `afc` / `nfc` into elo AND form state) so they can never be
matched or averaged into the league mean.

Usage:
    python scripts/offseason_regress.py --sector nfl --dry-run
    python scripts/offseason_regress.py --sector nfl --drop afc,nfc
    python scripts/offseason_regress.py --sector nba --keep 0.75 --moves data/models/nba_2027_offseason.yaml

The Elo (and, when keys are dropped, form) state files are backed up beside
themselves as `*.backup.{sector}_offseason_{timestamp}.json` before any write.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ELO_STATE_PATH = REPO_ROOT / "data" / "models" / "elo_state.json"
FORM_STATE_PATH = REPO_ROOT / "data" / "models" / "form_state.json"

DEFAULT_ELO = 1500.0

# Swept per sector — see the module docstring for provenance. A sector with no
# entry needs an explicit --keep (and a sweep behind it).
SECTOR_DEFAULT_KEEP: dict[str, float] = {
    "nfl": 0.667,
}


def shrink(elo: float, keep: float) -> float:
    """Move `elo` toward DEFAULT_ELO, keeping `keep` of its deviation."""
    return DEFAULT_ELO + keep * (elo - DEFAULT_ELO)


def load_moves(path: Path) -> dict:
    """Read a WNBA-style roster-move YAML: {regression_coefficient?, moves: [{team, delta}], expansion_priors?}."""
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg.get("moves", []), list):
        raise ValueError(f"{path}: 'moves' must be a list")
    return cfg


def _tally_deltas(moves: Optional[Iterable[dict]]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for move in moves or []:
        team = str(move["team"]).lower().strip()
        deltas[team] = deltas.get(team, 0.0) + float(move["delta"])
    return deltas


def apply_regression(
    state: dict,
    sector: str,
    keep: float,
    *,
    drop: Iterable[str] = (),
    moves: Optional[Iterable[dict]] = None,
    expansion: Optional[dict[str, float]] = None,
    today: Optional[date] = None,
) -> dict:
    """Regress `state[sector]` in place; return a before/after summary.

    - ratings: shrink(prior, keep) (+ summed roster deltas); expansion teams
      take their prior directly.
    - season_games: reset to 0 for every rated team.
    - drop: keys removed from ratings / game_counts / season_games and any
      h2h entry that references them.
    - stamps `state[sector]["offseason_regression"]` for provenance. Does NOT
      touch `last_updated` — the staleness guard keeps its own semantics.
    """
    if not (0.0 < keep <= 1.0):
        raise ValueError(f"keep must be in (0, 1], got {keep}")
    sec = state.setdefault(
        sector, {"ratings": {}, "game_counts": {}, "season_games": {}, "h2h": {}}
    )
    ratings = sec.setdefault("ratings", {})
    counts = sec.setdefault("game_counts", {})
    season_games = sec.setdefault("season_games", {})
    h2h = sec.setdefault("h2h", {})

    dropped: list[str] = []
    for key in drop:
        key = key.lower().strip()
        hit = False
        for store in (ratings, counts, season_games):
            if key in store:
                store.pop(key)
                hit = True
        for hk in list(h2h):
            if key in hk.split("::"):
                h2h.pop(hk)
                hit = True
        if hit:
            dropped.append(key)

    deltas = _tally_deltas(moves)
    expansion = {str(k).lower().strip(): float(v) for k, v in (expansion or {}).items()}

    before = dict(ratings)
    shrunk: dict[str, float] = {}
    after: dict[str, float] = {}
    for team in sorted(set(ratings) | set(deltas) | set(expansion)):
        prior = ratings.get(team, DEFAULT_ELO)
        base = expansion[team] if team in expansion else shrink(prior, keep)
        new = base + deltas.get(team, 0.0)
        shrunk[team] = round(base, 2)
        after[team] = round(new, 2)
        ratings[team] = after[team]
        season_games[team] = 0

    sec["offseason_regression"] = {
        "applied_on": (today or date.today()).isoformat(),
        "keep": keep,
        "dropped": dropped,
        "moves": len(list(moves or [])),
    }
    return {"before": before, "shrunk": shrunk, "after": after,
            "deltas": deltas, "dropped": dropped, "keep": keep}


def prune_form_state(form_state: dict, sector: str, drop: Iterable[str]) -> list[str]:
    """Remove `drop` keys from `form_state[sector]`; return the keys removed."""
    sec = form_state.get(sector)
    if not isinstance(sec, dict):
        return []
    removed = []
    for key in drop:
        key = key.lower().strip()
        if key in sec:
            sec.pop(key)
            removed.append(key)
    return removed


def print_table(summary: dict) -> None:
    before, shrunk, after, deltas = (summary[k] for k in ("before", "shrunk", "after", "deltas"))
    rows = sorted(
        ((t, before.get(t), shrunk[t], deltas.get(t, 0.0), after[t]) for t in after),
        key=lambda r: -r[4],
    )
    header = f"{'team':<14} {'prior':>7} {'shrunk':>7} {'delta':>6} {'new':>7}"
    print(header)
    print("-" * len(header))
    for team, b, s, d, a in rows:
        b_str = f"{b:.0f}" if b is not None else "—"
        sign = "+" if d > 0 else ""
        print(f"{team:<14} {b_str:>7} {s:>7.0f} {sign}{d:>5.0f} {a:>7.0f}")


def _backup(path: Path, sector: str, stamp: str) -> Path:
    backup = path.with_suffix(f".backup.{sector}_offseason_{stamp}.json")
    shutil.copy2(path, backup)
    return backup


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sector", required=True, help="Sector key in elo_state.json (e.g. nfl)")
    parser.add_argument("--keep", type=float, default=None,
                        help="Fraction of the deviation from 1500 kept (default: the sector's swept value)")
    parser.add_argument("--moves", type=Path, default=None,
                        help="Optional roster-move YAML (WNBA schema: moves[{team,delta}], expansion_priors)")
    parser.add_argument("--drop", default="",
                        help="Comma-separated non-team keys to remove from elo + form state (e.g. afc,nfc)")
    parser.add_argument("--dry-run", action="store_true", help="Print the table, write nothing")
    parser.add_argument("--state", type=Path, default=ELO_STATE_PATH)
    parser.add_argument("--form-state", type=Path, default=FORM_STATE_PATH)
    parser.add_argument("--no-form", action="store_true", help="Do not touch form_state.json even when --drop is set")
    args = parser.parse_args(argv)

    sector = args.sector.lower().strip()
    if not args.state.exists():
        print(f"error: {args.state} not found", file=sys.stderr)
        return 1

    cfg = load_moves(args.moves) if args.moves else {}
    keep = args.keep
    if keep is None:
        keep = cfg.get("regression_coefficient", SECTOR_DEFAULT_KEEP.get(sector))
    if keep is None:
        print(f"error: no swept keep for sector {sector!r} — pass --keep (and sweep it first, "
              f"see docs/SEASON_START.md §5)", file=sys.stderr)
        return 1
    drop = [d for d in (x.strip() for x in args.drop.split(",")) if d]

    state = json.loads(args.state.read_text())
    if sector not in state:
        print(f"error: sector {sector!r} not present in {args.state}", file=sys.stderr)
        return 1
    summary = apply_regression(
        state, sector, float(keep), drop=drop,
        moves=cfg.get("moves"), expansion=cfg.get("expansion_priors"),
    )

    form_removed: list[str] = []
    form_state: Optional[dict] = None
    if drop and not args.no_form and args.form_state.exists():
        form_state = json.loads(args.form_state.read_text())
        form_removed = prune_form_state(form_state, sector, drop)

    mode = "DRY RUN" if args.dry_run else "APPLYING"
    print(f"[{mode}] {sector} offseason Elo regression")
    print(f"  keep              = {float(keep):.3f}  (deviation from 1500 retained)")
    print(f"  teams             = {len(summary['after'])}")
    print(f"  dropped (elo)     = {summary['dropped'] or '—'}")
    print(f"  dropped (form)    = {form_removed or '—'}")
    print(f"  roster moves      = {len(list(cfg.get('moves') or []))}")
    print()
    print_table(summary)
    print()

    if args.dry_run:
        print("dry-run complete — no files modified")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = _backup(args.state, sector, stamp)
    args.state.write_text(json.dumps(state, indent=2))
    print(f"wrote {args.state}  (backup {backup.name})")
    if form_state is not None and form_removed:
        fbackup = _backup(args.form_state, sector, stamp)
        args.form_state.write_text(json.dumps(form_state, indent=2))
        print(f"wrote {args.form_state}  (backup {fbackup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
