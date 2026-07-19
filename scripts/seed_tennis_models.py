"""Seed tennis model agents from Tennis Abstract.

Jeff Sackmann's ``tennis_atp`` / ``tennis_wta`` GitHub CSVs went offline in 2026.
The per-match data now comes from tennisabstract.com's ``leadersource`` files
(``matchmx`` — full serve/return/break columns, the same granularity as the old
CSVs) and the winners/unforced-error leaderboards, via ``evmax/clients/tennisabstract.py``.

Computes and seeds (all five models, all from matchmx — no Sackmann/ESPN paths):
  1. tennis_serve_return  — per-match SPW with date + surface (recency-weighted at prediction time)
  2. tennis_h2h           — head-to-head win records (winner, loser counts)
  3. tennis_ranking_trend  — dated rank series from matchmx winner/loser rank columns
                             (one snapshot per match date; the trend agent reads its
                             12-week momentum off it, same shape the old weekly CSVs gave)
  4. tennis_advanced       — BP conversion, RPW from matchmx; UE rate + W/UE from the winners/errors page
  5. tennis_form           — recent match history with opponent rank, surface, minutes

Surface Elo is seeded separately by ``scripts/seed_tennis_abstract_elo.py``.

Usage:
    uv run python scripts/seed_tennis_models.py
    uv run python scripts/seed_tennis_models.py --years 2024,2025,2026
    uv run python scripts/seed_tennis_models.py --tours atp           # ATP only

matchmx covers roughly the trailing ~2.5 seasons (2024→present) for the full
bettable field across ranking segments.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta
from typing import Iterable, Optional

from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent
from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent
from evmax.agents.models.tennis_form_agent import TennisFormAgent
from evmax.agents.models.tennis_model_agent import TennisModelAgent
from evmax.clients.tennisabstract import fetch_matchmx, fetch_winners_errors

DEFAULT_YEARS = [2023, 2024, 2025, 2026]
DEFAULT_TOURS = ["atp", "wta"]


def _player_key(name: str) -> str:
    return name.strip().lower()


def _safe_int(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def _parse_date(raw: str) -> Optional[str]:
    """Convert Sackmann date format '20240615' to ISO '2024-06-15'."""
    raw = (raw or "").strip()
    if len(raw) >= 8:
        try:
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        except (ValueError, IndexError):
            pass
    return None


def _normalize_surface(surface: str) -> str:
    """Normalize surface names from Sackmann CSVs."""
    s = (surface or "").strip().lower()
    if s in ("hard", "clay", "grass", "carpet"):
        return s if s != "carpet" else "hard"  # carpet is extinct, treat as hard
    return "hard"  # default


# ---------------------------------------------------------------------------
# Per-match serve stats (for recency-weighted SPW)
# ---------------------------------------------------------------------------

def aggregate_match_serve_stats(match_rows: Iterable[dict]) -> dict[str, list[dict]]:
    """For each player, build per-match SPW entries with date and surface.

    Returns: {player → [{date, surface, won, svpt}, ...]}
    """
    entries: dict[str, list[dict]] = defaultdict(list)

    for row in match_rows:
        match_date = _parse_date(row.get("tourney_date", ""))
        surface = _normalize_surface(row.get("surface", ""))

        for side in ("w", "l"):
            name = row.get("winner_name" if side == "w" else "loser_name")
            if not name:
                continue
            try:
                first_won = _safe_int(row.get(f"{side}_1stWon"))
                second_won = _safe_int(row.get(f"{side}_2ndWon"))
                svpt = _safe_int(row.get(f"{side}_svpt"))
            except ValueError:
                continue
            if svpt <= 0:
                continue

            key = _player_key(name)
            entries[key].append({
                "date": match_date,
                "surface": surface,
                "won": first_won + second_won,
                "svpt": svpt,
            })

    return dict(entries)


# Legacy flat aggregate for backward compat
def aggregate_serve_stats(match_rows: Iterable[dict]) -> dict[str, tuple[float, int]]:
    """For each player, compute career SPW from match rows."""
    won: dict[str, int] = defaultdict(int)
    pts: dict[str, int] = defaultdict(int)

    for row in match_rows:
        for side in ("w", "l"):
            name = row.get(f"{side}inner_name" if side == "w" else "loser_name")
            if not name:
                continue
            try:
                first = int(row.get(f"{side}_1stWon") or 0)
                second = int(row.get(f"{side}_2ndWon") or 0)
                svpt = int(row.get(f"{side}_svpt") or 0)
            except ValueError:
                continue
            if svpt <= 0:
                continue
            key = _player_key(name)
            won[key] += first + second
            pts[key] += svpt

    return {
        player: (won[player] / pts[player], pts[player])
        for player in pts
        if pts[player] > 0
    }


# ---------------------------------------------------------------------------
# Match history for form agent (opponent rank, surface, minutes)
# ---------------------------------------------------------------------------

def aggregate_match_history(match_rows: Iterable[dict]) -> dict[str, list[dict]]:
    """Build per-player match history for the form agent.

    Returns: {player → [{date, won, opp_rank, surface, minutes, tourney_level}, ...]}
    """
    history: dict[str, list[dict]] = defaultdict(list)

    for row in match_rows:
        match_date = _parse_date(row.get("tourney_date", ""))
        surface = _normalize_surface(row.get("surface", ""))
        minutes = _safe_int(row.get("minutes"))
        tourney_level = (row.get("tourney_level") or "").strip()

        winner = (row.get("winner_name") or "").strip()
        loser = (row.get("loser_name") or "").strip()
        if not winner or not loser:
            continue

        winner_rank = _safe_int(row.get("winner_rank")) or None
        loser_rank = _safe_int(row.get("loser_rank")) or None

        w_key = _player_key(winner)
        l_key = _player_key(loser)

        # Winner's entry: won=True, opp_rank = loser's rank
        history[w_key].append({
            "date": match_date,
            "won": True,
            "opp_rank": loser_rank,
            "surface": surface,
            "minutes": minutes if minutes > 0 else None,
            "tourney_level": tourney_level,
        })

        # Loser's entry: won=False, opp_rank = winner's rank
        history[l_key].append({
            "date": match_date,
            "won": False,
            "opp_rank": winner_rank,
            "surface": surface,
            "minutes": minutes if minutes > 0 else None,
            "tourney_level": tourney_level,
        })

    # Sort each player's history by date
    for player in history:
        history[player].sort(key=lambda m: m.get("date") or "")

    return dict(history)


# ---------------------------------------------------------------------------
# Advanced stats (BP conversion, RPW, UE, winners)
# ---------------------------------------------------------------------------

def aggregate_advanced_stats(match_rows: Iterable[dict]) -> dict[str, dict]:
    """Aggregate per-player advanced stats from Sackmann match rows."""
    stats: dict[str, dict] = defaultdict(lambda: {
        "bp_won": 0, "bp_faced": 0,
        "return_pts": 0, "return_pts_won": 0,
        "total_pts": 0, "matches": 0,
        "unforced": 0, "winners": 0,
    })

    for row in match_rows:
        w_name = (row.get("winner_name") or "").strip().lower()
        l_name = (row.get("loser_name") or "").strip().lower()
        if not w_name or not l_name:
            continue

        w_svpt = _safe_int(row.get("w_svpt"))
        l_svpt = _safe_int(row.get("l_svpt"))
        if w_svpt <= 0 or l_svpt <= 0:
            continue

        w_1w = _safe_int(row.get("w_1stWon"))
        w_2w = _safe_int(row.get("w_2ndWon"))
        l_1w = _safe_int(row.get("l_1stWon"))
        l_2w = _safe_int(row.get("l_2ndWon"))
        w_bpf = _safe_int(row.get("w_bpFaced"))
        w_bps = _safe_int(row.get("w_bpSaved"))
        l_bpf = _safe_int(row.get("l_bpFaced"))
        l_bps = _safe_int(row.get("l_bpSaved"))

        total = w_svpt + l_svpt

        w = stats[w_name]
        w["bp_won"] += l_bpf - l_bps
        w["bp_faced"] += l_bpf
        w["return_pts"] += l_svpt
        w["return_pts_won"] += l_svpt - l_1w - l_2w
        w["total_pts"] += total
        w["matches"] += 1

        l = stats[l_name]
        l["bp_won"] += w_bpf - w_bps
        l["bp_faced"] += w_bpf
        l["return_pts"] += w_svpt
        l["return_pts_won"] += w_svpt - w_1w - w_2w
        l["total_pts"] += total
        l["matches"] += 1

    return dict(stats)


def augment_advanced_with_winners_errors(stats: dict[str, dict], tour: str) -> None:
    """Fill winners + unforced errors from Tennis Abstract's winners/errors leaderboard.

    Player names share Tennis Abstract's full-name format, so they key directly into
    ``stats`` (built from matchmx). Coverage is sparse (charted matches only); players
    not on the leaderboard keep winners/unforced at 0 and fall to the RPW-reduced model.
    """
    try:
        we = fetch_winners_errors(tour)
    except Exception as e:  # noqa: BLE001 — network/parse; advanced degrades gracefully
        print(f"  ! {tour} winners/errors unavailable: {e}")
        return
    count = 0
    for player, rec in we.items():
        if player not in stats:
            continue
        stats[player]["winners"] += rec["winners"]
        stats[player]["unforced"] += rec["unforced"]
        count += 1
    print(f"  + {tour} winners/errors: augmented {count} players")


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def aggregate_h2h(match_rows: Iterable[dict]) -> list[dict]:
    """Build H2H wins_a / wins_b records keyed by alphabetically-sorted pair."""
    counts: dict[tuple[str, str], list[int]] = {}
    for row in match_rows:
        w = row.get("winner_name")
        l = row.get("loser_name")
        if not w or not l:
            continue
        wk, lk = _player_key(w), _player_key(l)
        if wk == lk:
            continue
        a, b = sorted((wk, lk))
        rec = counts.setdefault((a, b), [0, 0])
        if wk == a:
            rec[0] += 1
        else:
            rec[1] += 1

    return [
        {"player_a": a, "player_b": b, "wins_a": rec[0], "wins_b": rec[1]}
        for (a, b), rec in counts.items()
    ]


# ---------------------------------------------------------------------------
# Ranking snapshots
# ---------------------------------------------------------------------------

# A player needs at least this many dated rank points to be worth storing; the
# trend agent itself gates on _MIN_SNAPSHOTS=4 after anchoring at the match date,
# so anything below that can never fire.
_MIN_RANK_SNAPSHOTS = 2


def aggregate_ranking_history(match_rows: Iterable[dict]) -> dict[str, list[dict]]:
    """Build a dated rank series per player from matchmx rank columns.

    Tennis Abstract's matchmx carries each player's official ATP/WTA rank at the
    time of every match (``winner_rank`` / ``loser_rank``). Collecting those
    across a player's matches yields the same ``{date → rank}`` shape the old
    Sackmann weekly ranking CSVs produced — just sampled at match dates instead
    of release Mondays. Players play far more than once per 12 weeks, so the
    series is dense enough for the trend agent's momentum window, and matchmx
    runs current (latest matches within days), keeping the staleness guard fed.

    Returns: ``{player → [{"date": "YYYY-MM-DD", "rank": int}, ...]}`` sorted by
    date, one snapshot per calendar day (a player's rank is fixed within a day).
    """
    by_player: dict[str, dict[str, int]] = defaultdict(dict)  # player → {iso_date: rank}

    for row in match_rows:
        match_date = _parse_date(row.get("tourney_date", ""))
        if not match_date:
            continue
        for name_field, rank_field in (
            ("winner_name", "winner_rank"),
            ("loser_name", "loser_rank"),
        ):
            name = (row.get(name_field) or "").strip()
            rank = _safe_int(row.get(rank_field))
            if not name or rank <= 0:
                continue
            # Last write per (player, date) wins; rank is constant within a day.
            by_player[_player_key(name)][match_date] = rank

    history: dict[str, list[dict]] = {}
    for player, date_rank in by_player.items():
        if len(date_rank) < _MIN_RANK_SNAPSHOTS:
            continue
        history[player] = [
            {"date": d, "rank": r} for d, r in sorted(date_rank.items())
        ]
    return history


# Players whose last matchmx appearance is older than this don't get a
# ranking prior — matchmx only carries players who played, and the recency
# filter keeps retirees from lingering as stale priors (the failure the
# seed_rankings full-replace fixed on 2026-07-01).
_RANK_PRIOR_MAX_AGE_DAYS = 90


def latest_rank_snapshots(
    history: dict[str, list[dict]],
    *,
    max_age_days: int = _RANK_PRIOR_MAX_AGE_DAYS,
    today: Optional[date] = None,
) -> dict[str, int]:
    """Latest official rank per recently-active player.

    Input is :func:`aggregate_ranking_history` output (date-sorted snapshots).
    Returns ``{player → latest_rank}`` restricted to players whose most recent
    snapshot is within ``max_age_days`` of ``today``. Feeds
    ``TennisModelAgent.merge_ranking_priors`` — the supplement that keeps the
    surface model's 0.48-confidence ranking fallback alive for established
    players the weekly TA Elo leaderboard churned off.
    """
    cutoff = (today or date.today()) - timedelta(days=max_age_days)
    out: dict[str, int] = {}
    for player, snaps in history.items():
        if not snaps:
            continue
        last = snaps[-1]  # date-sorted by aggregate_ranking_history
        try:
            last_date = date.fromisoformat(last["date"])
        except (KeyError, TypeError, ValueError):
            continue
        if last_date < cutoff:
            continue
        rank = last.get("rank")
        if isinstance(rank, int) and rank > 0:
            out[player] = rank
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def collect_matches(tour: str, years: list[int]) -> list[dict]:
    """Pull per-match records from Tennis Abstract's matchmx, filtered to `years`.

    Sackmann's match CSVs went offline in 2026; ``fetch_matchmx`` reads the same
    per-match data (full serve/return/break columns) from tennisabstract.com's
    ``leadersource`` files and returns dicts already keyed with the Sackmann field
    names the ``aggregate_*`` functions expect.
    """
    year_set = {int(y) for y in years}
    try:
        records = fetch_matchmx(tour)
    except Exception as e:  # noqa: BLE001 — network/parse; report and continue
        print(f"  ! {tour} matchmx fetch failed: {e}")
        return []
    rows = [r for r in records if _safe_int(str(r.get("tourney_date", ""))[:4]) in year_set]
    print(f"  + {tour} matchmx: {len(rows)} matches "
          f"({len(records)} total across segments, filtered to {sorted(year_set)})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed tennis model agents from Tennis Abstract matchmx")
    parser.add_argument(
        "--years",
        default=",".join(str(y) for y in DEFAULT_YEARS),
        help="Comma-separated years to pull (default: 2023,2024,2025,2026)",
    )
    parser.add_argument(
        "--tours",
        default=",".join(DEFAULT_TOURS),
        help="Comma-separated tours: atp,wta (default: both)",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Skip the winners/unforced-errors leaderboard augmentation for the advanced model",
    )
    args = parser.parse_args()

    years = sorted({int(y) for y in args.years.split(",") if y.strip()})
    year_set = set(years)
    tours = [t.strip().lower() for t in args.tours.split(",") if t.strip()]

    serve_agent = TennisServeReturnAgent()
    h2h_agent = TennisH2HAgent()
    trend_agent = TennisRankingTrendAgent()
    advanced_agent = TennisAdvancedStatsAgent()
    form_agent = TennisFormAgent()
    surface_agent = TennisModelAgent()

    combined_match_serve: dict[str, list[dict]] = {}
    combined_h2h: list[dict] = []
    combined_history: dict[str, list[dict]] = {}
    combined_advanced: dict[str, dict] = {}
    combined_form: dict[str, list[dict]] = {}

    for tour in tours:
        print(f"\n=== {tour.upper()} ===")
        matches = collect_matches(tour, years)
        if not matches:
            print(f"  ! no matches loaded for {tour}, skipping")
            continue

        # Per-match serve stats (new format)
        match_serve = aggregate_match_serve_stats(matches)
        # Match history for form agent
        match_history = aggregate_match_history(matches)
        # H2H
        h2h_records = aggregate_h2h(matches)
        # Ranking history — dated rank series straight from matchmx
        history = aggregate_ranking_history(matches)
        # Advanced stats
        advanced = aggregate_advanced_stats(matches)

        print(
            f"  → {len(match_serve)} players with serve stats, "
            f"{len(match_history)} players with form history, "
            f"{len(h2h_records)} H2H pairs, "
            f"{len(history)} players with ranking history, "
            f"{len(advanced)} players with advanced stats"
        )

        # Ranking-prior supplement: widen the surface model's 0.48-confidence
        # ranking fallback to the full recently-active matchmx population.
        # merge_ranking_priors only INSERTS players absent from the tour store,
        # so this must run AFTER seed_tennis_abstract_elo.py's weekly
        # full-replace (leaderboard entries always win) — the Monday task
        # weekly-tennis-surface-elo-refresh runs the two scripts in that order.
        recent_ranks = latest_rank_snapshots(history)
        added = surface_agent.merge_ranking_priors(recent_ranks, tour=tour)
        print(f"  → ranking priors: +{added} matchmx-only players merged into {tour}_rankings")

        # Merge across tours (ATP + WTA are disjoint player sets)
        for player, entries in match_serve.items():
            combined_match_serve.setdefault(player, []).extend(entries)
        for player, entries in match_history.items():
            combined_form.setdefault(player, []).extend(entries)
        combined_h2h.extend(h2h_records)
        combined_history.update(history)
        for player, st in advanced.items():
            if player not in combined_advanced:
                combined_advanced[player] = st
            else:
                for k in st:
                    combined_advanced[player][k] = combined_advanced[player].get(k, 0) + st[k]

        # matchmx already carries complete per-match serve stats, so no MCP serve
        # augmentation is needed (adding it would double-count). The advanced model's
        # winners/UE feature is the one thing matchmx lacks — fill it from the
        # winners/errors leaderboard (sparse; unlisted players use the RPW-reduced model).
        if not args.no_mcp:
            augment_advanced_with_winners_errors(combined_advanced, tour)

    # Sort match entries by date
    for player in combined_match_serve:
        combined_match_serve[player].sort(key=lambda e: e.get("date") or "")
    for player in combined_form:
        combined_form[player].sort(key=lambda e: e.get("date") or "")

    print(
        f"\n--- Seeding ---\n"
        f"  serve_return (match-level): {len(combined_match_serve)} players\n"
        f"  form (match history):       {len(combined_form)} players\n"
        f"  h2h:                        {len(combined_h2h)} pairs\n"
        f"  trend:                      {len(combined_history)} players\n"
        f"  advanced:                   {len(combined_advanced)} players"
    )

    if combined_match_serve:
        serve_agent.seed_match_serve_stats(combined_match_serve)
    if combined_h2h:
        h2h_agent.seed_h2h(combined_h2h)
    if combined_history:
        trend_agent.seed_history(combined_history)
    if combined_advanced:
        advanced_agent.seed_stats(combined_advanced)
    if combined_form:
        form_agent.seed_match_history(combined_form)

    print("\n[done] Agents seeded. State files:")
    print("  data/models/tennis_serve_return_state.json")
    print("  data/models/tennis_form_state.json")
    print("  data/models/tennis_h2h_state.json")
    print("  data/models/tennis_ranking_trend_state.json")
    print("  data/models/tennis_advanced_state.json")


if __name__ == "__main__":
    main()
