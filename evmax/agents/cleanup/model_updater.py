"""Reusable model-state updater — feeds completed ESPN scores into the models.

Extracted from `evmax/cli/commands/update.py::_run_update` so the same logic can
be invoked from both the `evmax update scores` CLI command and the daily
`evmax cleanup resolve` cron hook. The CLI keeps its Rich-table presentation;
this module returns structured per-game results and performs the state
mutation (coordinator.update_models + soccer_xg record_match).

State staleness — `data/models/{elo,form}_state.json` — only advanced when this
runs, so wiring it into the resolve cron keeps in-season blends fresh instead
of decaying to sharp-passthrough once form's STALE_DAYS guard renormalizes a
sector's contribution away.

Module-level imports of `fetch_completed_scores` / `_slug_teams` /
`_fuzzy_team_match` / `get_connection` are deliberate: the established test
idiom is `patch.object(model_updater, "fetch_completed_scores", ...)` and
`patch.object(model_updater, "get_connection", ...)`. Keeping them function-
local (as the old CLI code did) would defeat those patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import structlog

from evmax.agents.cleanup.db import get_connection
from evmax.agents.cleanup.resolver import (
    fetch_completed_scores,
    _slug_teams,
    _fuzzy_team_match,
)
from evmax.matching.normalizer import NameNormalizer

logger = structlog.get_logger(__name__)

# Lenient threshold for slug↔ESPN team matching on the update path. Distinct
# from resolver.FUZZY_THRESHOLD (72): the update path only uses the match to
# pick canonical slug names for state keys, so a looser bar is acceptable here.
# Pinned at 65 to preserve the historical CLI behavior exactly.
FUZZY_THRESHOLD = 65


@dataclass
class GameUpdate:
    """One completed game fed (or to-be-fed) into the models."""

    sector: str
    home_name: str
    away_name: str
    team_a: str          # normalized slug used as model-state key
    team_b: str
    score_a: int
    score_b: int
    applied: bool        # False under dry_run / on error
    error: Optional[str] = None


@dataclass
class UpdateResult:
    """Aggregate outcome of an update run."""

    games: list[GameUpdate]
    updated: int         # number of games actually fed into the models


async def update_models_for_date(
    sectors: list[str],
    target_date: date,
    coordinator=None,
    *,
    dry_run: bool = False,
) -> UpdateResult:
    """Feed all completed ESPN scores for `target_date` into the model agents.

    For each sector: fetch ESPN completed scores, resolve each game to canonical
    slug team names via the DB event_id slugs (FUZZY_THRESHOLD=65 fuzzy match)
    with a NameNormalizer(sector) fallback, dedup on (team_a, team_b), then call
    `coordinator.update_models` (and `soccer_xg_agent.record_match` for soccer).

    Args:
        sectors: sector keys to update (ESPN-supported only do anything).
        target_date: game date to fetch and apply.
        coordinator: optional pre-built AgentCoordinator (lets callers inject an
            isolated coordinator in tests / reuse one across calls). When None a
            fresh `AgentCoordinator(sectors=..., enable_models=True)` is built.
        dry_run: when True, no model state is mutated — games are still returned
            with `applied=False` so callers can preview.

    Returns:
        UpdateResult with the per-game list and the count actually applied.
    """
    if coordinator is None:
        from evmax.agents.coordinator import AgentCoordinator

        coordinator = AgentCoordinator(sectors=sectors, enable_models=True)

    games: list[GameUpdate] = []
    updated = 0

    for sector in sectors:
        scores = await fetch_completed_scores(sector, target_date)
        if not scores:
            continue

        # Load this sector+date's event_ids so we can recover canonical slug
        # team names (consistent with seeded model state).
        conn = get_connection()
        try:
            db_rows = conn.execute(
                """SELECT DISTINCT event_id FROM ev_predictions
                   WHERE sector = ? AND event_date = ?""",
                (sector, target_date.isoformat()),
            ).fetchall()
        finally:
            conn.close()

        normalizer = NameNormalizer(sector)
        processed: set[tuple[str, str]] = set()

        for score in scores:
            home_n = score["home_name"]
            away_n = score["away_name"]
            home_s = int(score["home_score"])
            away_s = int(score["away_score"])

            # Default to normalizer slugs; upgrade to DB slug names on a match.
            team_a = normalizer.normalize(home_n) or home_n.lower()
            team_b = normalizer.normalize(away_n) or away_n.lower()

            for row in db_rows:
                slug_a, slug_b = _slug_teams(row["event_id"])
                if not slug_a:
                    continue
                if (
                    _fuzzy_team_match(slug_a, home_n) >= FUZZY_THRESHOLD
                    and _fuzzy_team_match(slug_b, away_n) >= FUZZY_THRESHOLD
                ):
                    team_a = slug_a
                    team_b = slug_b
                    break

            key = (team_a, team_b)
            if key in processed:
                continue
            processed.add(key)

            game = GameUpdate(
                sector=sector,
                home_name=home_n,
                away_name=away_n,
                team_a=team_a,
                team_b=team_b,
                score_a=home_s,
                score_b=away_s,
                applied=False,
            )

            if not dry_run:
                try:
                    coordinator.update_models(
                        team_a=team_a,
                        team_b=team_b,
                        score_a=home_s,
                        score_b=away_s,
                        sector=sector,
                        event_date=target_date.isoformat(),
                    )
                    game.applied = True
                    updated += 1

                    # Feed shot stats into xG agent (soccer + national-team WC,
                    # both of which carry shotsOnTarget/totalShots from ESPN).
                    if sector in ("soccer", "worldcup"):
                        home_sot = score.get("home_sot")
                        away_sot = score.get("away_sot")
                        home_shots = score.get("home_shots")
                        away_shots = score.get("away_shots")
                        if all(
                            v is not None
                            for v in (home_sot, away_sot, home_shots, away_shots)
                        ):
                            xg = coordinator.soccer_xg_agent
                            xg.record_match(
                                team=team_a, goals_for=home_s, goals_against=away_s,
                                shots_on_target=home_sot, total_shots=home_shots,
                                opponent_sot=away_sot, opponent_shots=away_shots,
                                match_date=target_date.isoformat(), is_home=True,
                                sector=sector,
                            )
                            xg.record_match(
                                team=team_b, goals_for=away_s, goals_against=home_s,
                                shots_on_target=away_sot, total_shots=away_shots,
                                opponent_sot=home_sot, opponent_shots=home_shots,
                                match_date=target_date.isoformat(), is_home=False,
                                sector=sector,
                            )
                            xg.save_state()
                except Exception as exc:  # noqa: BLE001 — one bad game must not abort the run
                    game.error = str(exc)
                    logger.warning(
                        "model_update_game_failed",
                        sector=sector,
                        team_a=team_a,
                        team_b=team_b,
                        error=str(exc),
                    )

            games.append(game)

    return UpdateResult(games=games, updated=updated)
