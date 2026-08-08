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

Idempotency — because both callers exist, a daily run of `update scores`
followed by `cleanup resolve --date D` used to feed every game of date D into
Elo/Form/Poisson TWICE (the in-loop `processed` set only dedups within one
invocation). Each applied game is now recorded in the `applied_model_games`
table and skipped on a later pass. `force=True` overrides, for a deliberate
re-derivation onto state that does not already contain the games.

Module-level imports of `fetch_completed_scores` / `_slug_teams` /
`_fuzzy_team_match` / `get_connection` are deliberate: the established test
idiom is `patch.object(model_updater, "fetch_completed_scores", ...)` and
`patch.object(model_updater, "get_connection", ...)`. Keeping them function-
local (as the old CLI code did) would defeat those patches.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Optional

import structlog

from evmax.agents.cleanup.db import get_connection
from evmax.agents.cleanup.resolver import (
    ESPN_SOCCER_LIKE_LEAGUES,
    fetch_completed_scores,
    _slug_teams,
    _fuzzy_team_match,
)
from evmax.matching.normalizer import NameNormalizer

logger = structlog.get_logger(__name__)

# ESPN season type 1 = preseason / exhibition. These games must never train
# Elo/Form: preseason rosters and effort are not the regular-season signal, and
# feeding them corrupts ratings right before the opener (the WNBA All-Star /
# exhibition contamination lesson). Regular season = 2, postseason = 3. Applied
# to the US-league sectors (ESPN_SPORT_MAP); the soccer-like path is exempt
# because ESPN uses `season.type` differently there (friendlies are handled at
# the seed layer, not here).
_PRESEASON_SEASON_TYPE = 1

# Lenient threshold for slug↔ESPN team matching on the update path. Distinct
# from resolver.FUZZY_THRESHOLD (72): the update path only uses the match to
# pick canonical slug names for state keys, so a looser bar is acceptable here.
# Pinned at 65 to preserve the historical CLI behavior exactly.
FUZZY_THRESHOLD = 65

# Canonical set of ESPN-fed game sectors whose elo/form/poisson/xg state is
# advanced by update_models_for_date. Both entry points reference THIS single
# list so they cannot drift apart:
#   - `evmax update scores` default --sectors (cli/commands/update.py)
#   - the `evmax cleanup resolve` model-update hook (cli/commands/cleanup.py)
# They previously diverged — ncaaw + ncaaf were missing from both (so the
# NCAAW Elo calibrated 2026-07-11 was never incrementally fed and ncaaf never
# entered elo/form state at all), and worldcup was missing from the CLI default
# despite the hook feeding it. Membership matches the ESPN-scored game sectors
# in resolver.ESPN_SPORT_MAP + resolver.ESPN_SOCCER_LIKE_LEAGUES (tennis/ufc
# resolve via Kalshi settlement and refresh their own agents, not this hook;
# f1 has no elo/form model). Off-season sectors are a no-op — fetch returns no
# scores and they are silently skipped.
ESPN_MODEL_UPDATE_SECTORS: list[str] = [
    "soccer",
    "worldcup",
    "nba",
    "wnba",
    "nfl",
    "ncaab",
    "ncaaw",
    "ncaaf",
    "nhl",
    "baseball",
]


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
    applied: bool        # False under dry_run / on error / when already applied
    error: Optional[str] = None
    already_applied: bool = False   # skipped: this game is already in model state


@dataclass
class UpdateResult:
    """Aggregate outcome of an update run."""

    games: list[GameUpdate]
    updated: int         # number of games actually fed into the models
    skipped: int = 0     # games skipped because they were already applied


def _load_applied_keys(conn, sector: str, target_date: date) -> set[tuple[str, str]]:
    """Return the (team_a, team_b) pairs already fed into state for this sector+date.

    Tolerates a missing `applied_model_games` table so a pre-existing database
    (or a hand-rolled test connection) degrades to "no ledger" instead of
    raising. `get_connection` re-runs SCHEMA on every open, so production
    always has the table.
    """
    try:
        rows = conn.execute(
            """SELECT team_a, team_b FROM applied_model_games
               WHERE sector = ? AND event_date = ?""",
            (sector, target_date.isoformat()),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001 — missing table must not abort the run
        logger.debug("applied_games_ledger_unavailable", sector=sector, error=str(exc))
        return set()
    return {(r["team_a"], r["team_b"]) for r in rows}


def _record_applied(conn, sector: str, target_date: date, team_a: str, team_b: str) -> None:
    """Mark one game as fed into model state. Best-effort; never aborts a run."""
    try:
        conn.execute(
            """INSERT OR IGNORE INTO applied_model_games (sector, event_date, team_a, team_b)
               VALUES (?, ?, ?, ?)""",
            (sector, target_date.isoformat(), team_a, team_b),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.debug("applied_games_record_failed", sector=sector, error=str(exc))


async def update_models_for_date(
    sectors: list[str],
    target_date: date,
    coordinator=None,
    *,
    dry_run: bool = False,
    force: bool = False,
    espn_cache: Optional[dict] = None,
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
        force: re-apply games even when the `applied_model_games` ledger already
            records them. Off by default: `evmax update scores` and the
            `evmax cleanup resolve` hook both call this function, so without the
            ledger a daily run of both double-counts every game into Elo/Form/
            Poisson. Use only for a deliberate re-derivation onto a state file
            that does not already contain the games.
        espn_cache: optional per-run scoreboard cache shared with
            `resolve_outcomes_for_date` so the model-update fetch reuses
            scoreboards the resolve phase already pulled for `target_date`.

    Returns:
        UpdateResult with the per-game list, the count actually applied, and the
        count skipped as already-applied.
    """
    if coordinator is None:
        from evmax.agents.coordinator import AgentCoordinator

        coordinator = AgentCoordinator(sectors=sectors, enable_models=True)

    games: list[GameUpdate] = []
    updated = 0
    skipped = 0

    # Prefetch every sector's completed scores concurrently (bounded by the
    # resolver's _ESPN_FETCH_SEM) before the serial apply loop below. The
    # model-state mutations and the SQLite ledger writes must stay serial, but
    # the ESPN round-trips they depend on don't — this collapses the old
    # per-sector serial fetch into one bounded fan-out. With a shared
    # espn_cache, sectors the resolve phase already fetched return instantly.
    fetched = await asyncio.gather(*(
        fetch_completed_scores(sector, target_date, cache=espn_cache)
        for sector in sectors
    ))
    scores_by_sector = dict(zip(sectors, fetched))

    for sector in sectors:
        scores = scores_by_sector[sector]
        if not scores:
            continue

        # Drop preseason / exhibition games before they reach the model feed.
        # Soccer-like sectors are exempt (see _PRESEASON_SEASON_TYPE). Missing
        # season_type (older cache entries) is treated as regular-season so the
        # guard is fail-open, never dropping real games.
        if sector not in ESPN_SOCCER_LIKE_LEAGUES:
            preseason = [
                s for s in scores if s.get("season_type") == _PRESEASON_SEASON_TYPE
            ]
            if preseason:
                scores = [
                    s
                    for s in scores
                    if s.get("season_type") != _PRESEASON_SEASON_TYPE
                ]
                logger.info(
                    "skipped_preseason_model_update",
                    sector=sector,
                    dropped=len(preseason),
                    date=str(target_date),
                )
            if not scores:
                continue

        # Held open for the whole sector: the applied-games ledger is written
        # as each game lands, not in a second pass.
        conn = get_connection()
        try:
            # Load this sector+date's event_ids so we can recover canonical slug
            # team names (consistent with seeded model state).
            db_rows = conn.execute(
                """SELECT DISTINCT event_id FROM ev_predictions
                   WHERE sector = ? AND event_date = ?""",
                (sector, target_date.isoformat()),
            ).fetchall()
            applied_keys = _load_applied_keys(conn, sector, target_date)

            normalizer = NameNormalizer(sector)
            processed: set[tuple[str, str]] = set()
            # Games applied this sector, ledgered in one pass AFTER the single
            # end-of-sector state flush — preserves the save-before-ledger
            # ordering the per-game path used (a crash before the flush leaves
            # these un-ledgered and re-appliable next run).
            sector_applied: list[tuple[str, str]] = []
            xg_dirty = False

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

                # Already in model state from an earlier pass over this date —
                # re-applying would double-count the result into Elo/Form/Poisson.
                if key in applied_keys and not force:
                    game.already_applied = True
                    skipped += 1
                    games.append(game)
                    continue

                if not dry_run:
                    try:
                        coordinator.update_models(
                            team_a=team_a,
                            team_b=team_b,
                            score_a=home_s,
                            score_b=away_s,
                            sector=sector,
                            event_date=target_date.isoformat(),
                            save=False,  # flushed once per sector below
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
                                xg_dirty = True  # flushed once per sector below

                        # Record for the post-flush ledger pass — a game is only
                        # ledgered after its state mutation is persisted.
                        sector_applied.append((team_a, team_b))
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

            # End-of-sector flush: persist the accumulated in-memory state ONCE
            # (elo/form/poisson + ufc/xg as applicable), THEN ledger the applied
            # games. Ordering matters — state must hit disk before the ledger, so
            # a crash between the two re-applies games rather than losing them.
            if not dry_run and sector_applied:
                coordinator.save_model_states(sector)
                if xg_dirty:
                    coordinator.soccer_xg_agent.save_state()
                for team_a, team_b in sector_applied:
                    _record_applied(conn, sector, target_date, team_a, team_b)
        finally:
            conn.close()

    return UpdateResult(games=games, updated=updated, skipped=skipped)
