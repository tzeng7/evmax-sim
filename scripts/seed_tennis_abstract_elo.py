"""Seed the surface-aware tennis Elo model from Tennis Abstract's Elo leaderboards.

Replaces the dead Sackmann match-CSV path for **surface Elo**. Jeff Sackmann's
``tennis_atp`` / ``tennis_wta`` GitHub repos went offline in 2026, but his site
tennisabstract.com still publishes weekly-updated Elo reports. This seeder pulls
the ATP + WTA Elo leaderboards and writes overall + hard/clay/grass ratings
(plus ``indoor`` = hard, since the source has no indoor split and predict-time
indoor events resolve to the hard surface) into ``tennis_surface_state.json``.

It is idempotent — each run clears the existing ratings/counts and re-seeds from
the current leaderboard — and also refreshes the ATP/WTA ranking priors used as a
fallback for unrated players.

NOTE: this only covers the surface Elo model. The serve/return and advanced-stats
models still depend on Sackmann's (now offline) point-level data and are not
re-sourced here.

Usage:
    uv run python scripts/seed_tennis_abstract_elo.py
    uv run python scripts/seed_tennis_abstract_elo.py --tours atp
    uv run python scripts/seed_tennis_abstract_elo.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Editable-install/worktree safety: ensure the repo root is importable when this
# script is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evmax.agents.models.tennis_model_agent import TennisModelAgent  # noqa: E402
from evmax.clients.tennisabstract import PlayerElo, fetch_elo_page  # noqa: E402

SURFACES = ("overall", "hard", "clay", "grass", "indoor")


def build_surface_ratings(
    rows_by_tour: dict[str, list[PlayerElo]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """Map Tennis Abstract rows into the model's surface buckets + ranking priors.

    Returns ``(ratings_by_surface, rankings_by_tour)`` where ratings_by_surface
    merges both tours (disjoint player sets) and rankings_by_tour is keyed
    ``{"atp": {...}, "wta": {...}}`` for the ranking-prior store.
    """
    ratings_by_surface: dict[str, dict[str, float]] = {s: {} for s in SURFACES}
    rankings_by_tour: dict[str, dict[str, int]] = {}

    for tour, rows in rows_by_tour.items():
        ranks: dict[str, int] = {}
        for p in rows:
            ratings_by_surface["overall"][p.player] = p.elo
            if p.hard is not None:
                ratings_by_surface["hard"][p.player] = p.hard
                ratings_by_surface["indoor"][p.player] = p.hard
            if p.clay is not None:
                ratings_by_surface["clay"][p.player] = p.clay
            if p.grass is not None:
                ratings_by_surface["grass"][p.player] = p.grass
            if p.official_rank is not None:
                ranks[p.player.lower().strip()] = p.official_rank
        rankings_by_tour[tour] = ranks

    return ratings_by_surface, rankings_by_tour


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed surface Elo from Tennis Abstract leaderboards"
    )
    parser.add_argument(
        "--tours", default="atp,wta",
        help="Comma-separated tours to pull (default: atp,wta)",
    )
    parser.add_argument(
        "--games-per-player", type=int, default=12,
        help="Synthetic per-surface game count stamped for confidence gating "
             "(default: 12 → ~0.61 model confidence)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch + summarize without writing state",
    )
    args = parser.parse_args()

    tours = [t.strip().lower() for t in args.tours.split(",") if t.strip()]

    rows_by_tour: dict[str, list[PlayerElo]] = {}
    for tour in tours:
        rows, last_update = fetch_elo_page(tour)
        if not rows:
            print(f"  ! {tour}: no rows parsed — page format may have changed", file=sys.stderr)
            continue
        rows_by_tour[tour] = rows
        print(f"  + {tour.upper()}: {len(rows)} players (page last update: {last_update})")

    if not rows_by_tour:
        print("No data fetched — aborting (state untouched).", file=sys.stderr)
        sys.exit(1)

    ratings_by_surface, rankings_by_tour = build_surface_ratings(rows_by_tour)

    print("\n--- Surface coverage ---")
    for surface in SURFACES:
        print(f"  {surface:<8}: {len(ratings_by_surface[surface])} players")

    if args.dry_run:
        print("\n[dry-run] No state written.")
        return

    agent = TennisModelAgent()
    seeded = agent.seed_precomputed_surface_elo(
        ratings_by_surface, games_per_player=args.games_per_player
    )
    for tour, ranks in rankings_by_tour.items():
        if ranks:
            agent.seed_rankings(ranks, tour=tour)

    print(f"\n[done] Seeded surface Elo: {seeded}")
    print("  data/models/tennis_surface_state.json")
    # Sanity: echo a couple of marquee ratings via the resolver.
    for name in ("jannik sinner", "carlos alcaraz", "aryna sabalenka"):
        print(
            f"  {name:<18} overall={agent.get_rating(name, 'overall'):.0f} "
            f"hard={agent.get_rating(name, 'hard'):.0f} "
            f"clay={agent.get_rating(name, 'clay'):.0f} "
            f"grass={agent.get_rating(name, 'grass'):.0f}"
        )


if __name__ == "__main__":
    main()
