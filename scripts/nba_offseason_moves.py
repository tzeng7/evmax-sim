"""Apply NBA roster-move Elo deltas to the ALREADY-regressed elo_state.json.

Unlike scripts/wnba_offseason_regress.py this never shrinks toward 1500 and never
resets `season_games`: scripts/offseason_regress.py --sector nba (keep 0.50) owns
the regression, and the NBA sweep rejected any early-K boost. This step only adds
the summed per-team deltas from data/models/nba_2026_offseason_moves.yaml.

Idempotent: the applied moves' digest is stored under
elo_state['nba']['offseason_moves'], and a re-run with the same file refuses
unless --force is given (a double apply would double every delta).

Usage:
    python scripts/nba_offseason_moves.py --dry-run
    python scripts/nba_offseason_moves.py            # apply + backup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ELO_STATE_PATH = REPO_ROOT / "data" / "models" / "elo_state.json"
MOVES_YAML = REPO_ROOT / "data" / "models" / "nba_2026_offseason_moves.yaml"

STATE_KEY = "offseason_moves"


class AlreadyApplied(RuntimeError):
    pass


def load_moves(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw) or {}
    moves = cfg.get("moves")
    if not isinstance(moves, list):
        raise ValueError(f"{path} must define a `moves` list")
    for m in moves:
        if "team" not in m or "delta" not in m:
            raise ValueError(f"move missing team/delta: {m!r}")
        float(m["delta"])
    return moves, hashlib.sha256(raw).hexdigest()[:16]


def sum_deltas(moves: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in moves:
        team = str(m["team"]).lower().strip()
        out[team] = out.get(team, 0.0) + float(m["delta"])
    return out


def apply_moves(state: dict, moves: list[dict], digest: str, *, force: bool = False) -> dict:
    """Add summed deltas to state['nba']['ratings'] in place.

    Raises KeyError for a team absent from the ratings (a typo would otherwise
    mint a phantom team at 1500 + delta) and AlreadyApplied on a repeat digest.
    """
    nba = state["nba"]
    ratings = nba["ratings"]
    if nba.get(STATE_KEY, {}).get("digest") == digest and not force:
        raise AlreadyApplied(f"moves with digest {digest} already applied")
    deltas = sum_deltas(moves)
    unknown = sorted(t for t in deltas if t not in ratings)
    if unknown:
        raise KeyError(f"teams not in NBA Elo ratings: {unknown}")
    before = {t: ratings[t] for t in deltas}
    for team, d in deltas.items():
        ratings[team] = round(ratings[team] + d, 2)
    nba[STATE_KEY] = {
        "digest": digest,
        "applied_on": datetime.now().date().isoformat(),
        "deltas": deltas,
    }
    return {"before": before, "after": {t: ratings[t] for t in deltas}, "deltas": deltas}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-apply even if this digest was applied")
    p.add_argument("--yaml", type=Path, default=MOVES_YAML)
    p.add_argument("--state", type=Path, default=ELO_STATE_PATH)
    args = p.parse_args(argv)

    moves, digest = load_moves(args.yaml)
    state = json.loads(args.state.read_text())
    try:
        summary = apply_moves(state, moves, digest, force=args.force)
    except (AlreadyApplied, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"[{'DRY RUN' if args.dry_run else 'APPLYING'}] NBA offseason moves (digest {digest})")
    print(f"{'team':<12} {'before':>8} {'delta':>7} {'after':>8}")
    for t in sorted(summary["after"], key=lambda k: -summary["after"][k]):
        print(f"{t:<12} {summary['before'][t]:>8.1f} {summary['deltas'][t]:>+7.0f} {summary['after'][t]:>8.1f}")
    if args.dry_run:
        print("dry-run complete — no files modified")
        return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.state.with_suffix(f".backup.nba_moves_{stamp}.json")
    shutil.copy2(args.state, backup)
    args.state.write_text(json.dumps(state, indent=2))
    print(f"wrote {args.state}\nbackup saved at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
