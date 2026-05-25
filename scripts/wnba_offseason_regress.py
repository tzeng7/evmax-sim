"""Apply 2026 WNBA offseason regression + roster-move deltas to elo_state.json.

Reads the move list from data/models/wnba_2026_offseason.yaml, regresses each
team's 2025-end Elo toward 1500 by the configured coefficient, then applies
summed per-team roster deltas. Expansion teams get their prior written in
directly.

Usage:
    python scripts/wnba_offseason_regress.py --dry-run   # preview, no write
    python scripts/wnba_offseason_regress.py             # apply + backup

The original elo_state.json is copied to
    data/models/elo_state.backup.wnba_offseason_{timestamp}.json
before any in-place write, so re-running this script is always reversible.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ELO_STATE_PATH = REPO_ROOT / "data" / "models" / "elo_state.json"
OFFSEASON_YAML = REPO_ROOT / "data" / "models" / "wnba_2026_offseason.yaml"

DEFAULT_ELO = 1500.0


def load_offseason_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    required = {"regression_coefficient", "moves"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    if not (0 < cfg["regression_coefficient"] <= 1.0):
        raise ValueError(
            f"regression_coefficient must be in (0, 1], got {cfg['regression_coefficient']}"
        )
    return cfg


def shrink(elo: float, k: float) -> float:
    return DEFAULT_ELO + k * (elo - DEFAULT_ELO)


def apply_offseason(state: dict, cfg: dict) -> dict:
    """Return a new state dict with 2026 WNBA recalibration applied.

    `state` is mutated in place for the wnba section; callers that need
    the pre-change snapshot should copy first.
    """
    wnba = state.setdefault(
        "wnba",
        {"ratings": {}, "game_counts": {}, "season_games": {}, "h2h": {}},
    )
    ratings = wnba.setdefault("ratings", {})
    season_games = wnba.setdefault("season_games", {})
    k = cfg["regression_coefficient"]
    expansion = cfg.get("expansion_priors") or {}

    # Tally per-team roster deltas
    deltas: dict[str, float] = {}
    for move in cfg["moves"]:
        team = move["team"].lower().strip()
        deltas[team] = deltas.get(team, 0.0) + float(move["delta"])

    before: dict[str, float] = dict(ratings)
    shrunk: dict[str, float] = {}
    after: dict[str, float] = {}

    all_teams = set(ratings) | set(deltas) | set(expansion)
    for team in sorted(all_teams):
        prior_elo = ratings.get(team, DEFAULT_ELO)
        if team in expansion:
            shrunk_elo = float(expansion[team])
            new_elo = shrunk_elo + deltas.get(team, 0.0)
        else:
            shrunk_elo = shrink(prior_elo, k)
            new_elo = shrunk_elo + deltas.get(team, 0.0)
        shrunk[team] = round(shrunk_elo, 2)
        after[team] = round(new_elo, 2)
        ratings[team] = after[team]
        # Reset post-regression game counter so EloModelAgent's early-season
        # K boost kicks in for the new season. Without this the regressed
        # prior dominates the first ~month at baseline K=20.
        season_games[team] = 0

    return {"before": before, "shrunk": shrunk, "after": after, "deltas": deltas}


def print_table(summary: dict) -> None:
    before = summary["before"]
    shrunk = summary["shrunk"]
    after = summary["after"]
    deltas = summary["deltas"]

    rows = [
        (team, before.get(team, None), shrunk[team], deltas.get(team, 0.0), after[team])
        for team in after
    ]
    rows.sort(key=lambda r: -r[4])

    header = f"{'team':<12} {'2025 end':>10} {'shrunk':>8} {'delta':>7} {'2026':>7}"
    print(header)
    print("-" * len(header))
    for team, b, s, d, a in rows:
        b_str = f"{b:.0f}" if b is not None else "  —"
        sign = "+" if d > 0 else ""
        print(f"{team:<12} {b_str:>10} {s:>8.0f} {sign}{d:>6.0f} {a:>7.0f}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the before/after table, do not write files")
    parser.add_argument("--yaml", type=Path, default=OFFSEASON_YAML,
                        help=f"Offseason config YAML (default: {OFFSEASON_YAML})")
    parser.add_argument("--state", type=Path, default=ELO_STATE_PATH,
                        help=f"Elo state JSON (default: {ELO_STATE_PATH})")
    args = parser.parse_args(argv)

    if not args.yaml.exists():
        print(f"error: {args.yaml} not found", file=sys.stderr)
        return 1
    if not args.state.exists():
        print(f"error: {args.state} not found", file=sys.stderr)
        return 1

    cfg = load_offseason_config(args.yaml)
    state = json.loads(args.state.read_text())

    summary = apply_offseason(state, cfg)

    mode = "DRY RUN" if args.dry_run else "APPLYING"
    print(f"[{mode}] WNBA 2026 offseason recalibration")
    print(f"  regression_coefficient = {cfg['regression_coefficient']}")
    print(f"  expansion_priors       = {cfg.get('expansion_priors') or {}}")
    print(f"  move entries           = {len(cfg['moves'])}")
    print()
    print_table(summary)
    print()

    if args.dry_run:
        print("dry-run complete — no files modified")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.state.with_suffix(f".backup.wnba_offseason_{stamp}.json")
    shutil.copy2(args.state, backup)
    args.state.write_text(json.dumps(state, indent=2))
    print(f"wrote {args.state}")
    print(f"backup saved at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
