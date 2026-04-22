"""Seed tennis model agents from Jeff Sackmann's public CSVs.

Pulls match-level and ranking data from:
  - github.com/JeffSackmann/tennis_atp
  - github.com/JeffSackmann/tennis_wta

Computes and seeds:
  1. tennis_serve_return  — per-match SPW with date + surface (recency-weighted at prediction time)
  2. tennis_h2h           — head-to-head win records (winner, loser counts)
  3. tennis_ranking_trend  — weekly ranking snapshots from the current rankings file
  4. tennis_advanced       — BP conversion, RPW, UE rate, W/UE ratio (logistic)
  5. tennis_form           — recent match history with opponent rank, surface, minutes

Usage:
    uv run python scripts/seed_tennis_models.py
    uv run python scripts/seed_tennis_models.py --years 2023,2024,2025,2026
    uv run python scripts/seed_tennis_models.py --tours atp           # ATP only

Sackmann's repo updates on a delay; years that 404 are silently skipped so
the script remains useful as future seasons land.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict
from typing import Iterable, Optional

import httpx

from evmax.agents.models.tennis_advanced_stats_agent import TennisAdvancedStatsAgent
from evmax.agents.models.tennis_serve_return_agent import TennisServeReturnAgent
from evmax.agents.models.tennis_h2h_agent import TennisH2HAgent
from evmax.agents.models.tennis_ranking_trend_agent import TennisRankingTrendAgent
from evmax.agents.models.tennis_form_agent import TennisFormAgent

REPO_BASE = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}
MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
MCP_FILES = {
    "atp": ("charting-m-matches.csv", "charting-m-stats-Overview.csv"),
    "wta": ("charting-w-matches.csv", "charting-w-stats-Overview.csv"),
}
DEFAULT_YEARS = [2023, 2024, 2025, 2026]
DEFAULT_TOURS = ["atp", "wta"]


def _fetch_csv(url: str, timeout: float = 30.0) -> Optional[list[dict]]:
    """Fetch a CSV and return its rows as dicts. None on 404 or network error."""
    try:
        r = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as e:
        print(f"  ! fetch failed: {url} ({e})", file=sys.stderr)
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


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


def augment_advanced_with_mcp(
    stats: dict[str, dict], tour: str, years: set[int]
) -> None:
    """Fill in winners and unforced errors from MCP Overview."""
    matches_file, stats_file = MCP_FILES[tour]
    matches = _fetch_csv(f"{MCP_BASE}/{matches_file}")
    overview = _fetch_csv(f"{MCP_BASE}/{stats_file}")
    if not matches or not overview:
        print(f"  ! MCP advanced unavailable for {tour}")
        return

    match_year: dict[str, int] = {}
    for m in matches:
        mid = m.get("match_id", "")
        raw = (m.get("Date") or "").strip()
        if mid and len(raw) >= 4:
            try:
                match_year[mid] = int(raw[:4])
            except ValueError:
                pass

    count = 0
    for row in overview:
        if (row.get("set") or "").strip() != "Total":
            continue
        mid = row.get("match_id", "")
        yr = match_year.get(mid)
        if yr is None or yr not in years:
            continue
        player = _player_key(row.get("player") or "")
        if not player or player not in stats:
            continue

        stats[player]["winners"] += _safe_int(row.get("winners"))
        stats[player]["unforced"] += _safe_int(row.get("unforced"))
        count += 1

    print(f"  + MCP {tour} advanced: augmented {count} player-match records")


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

def load_ranking_history(tour: str) -> dict[str, list[dict]]:
    """Load weekly ranking snapshots from {tour}_rankings_current.csv."""
    base = REPO_BASE[tour]
    rankings = _fetch_csv(f"{base}/{tour}_rankings_current.csv")
    players = _fetch_csv(f"{base}/{tour}_players.csv")
    if rankings is None or players is None:
        print(f"  ! {tour}: ranking files unavailable")
        return {}

    pid_to_name: dict[str, str] = {}
    for p in players:
        pid = p.get("player_id")
        first = (p.get("name_first") or "").strip()
        last = (p.get("name_last") or "").strip()
        if pid and (first or last):
            pid_to_name[pid] = _player_key(f"{first} {last}".strip())

    history: dict[str, list[dict]] = defaultdict(list)
    for row in rankings:
        pid = row.get("player")
        raw_date = row.get("ranking_date")
        rank = row.get("rank")
        if not pid or not raw_date or not rank:
            continue
        name = pid_to_name.get(pid)
        if not name:
            continue
        try:
            iso_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            history[name].append({"date": iso_date, "rank": int(rank)})
        except (ValueError, IndexError):
            continue

    for snaps in history.values():
        snaps.sort(key=lambda s: s["date"])
    return dict(history)


# ---------------------------------------------------------------------------
# MCP serve stats (SPW augmentation for 2025+)
# ---------------------------------------------------------------------------

def load_mcp_serve_stats(
    tour: str, years: set[int]
) -> dict[str, list[dict]]:
    """Aggregate per-match SPW from the Match Charting Project.

    Returns match-level entries (date, surface, won, svpt) for the new
    recency-weighted format.
    """
    matches_file, stats_file = MCP_FILES[tour]
    matches = _fetch_csv(f"{MCP_BASE}/{matches_file}")
    stats = _fetch_csv(f"{MCP_BASE}/{stats_file}")
    if matches is None or stats is None:
        print(f"  ! {tour} MCP: files unavailable")
        return {}

    # Build match_id → (year, date, surface)
    match_meta: dict[str, tuple[int, Optional[str], str]] = {}
    for m in matches:
        mid = m.get("match_id")
        raw_date = (m.get("Date") or "").strip()
        surf = _normalize_surface(m.get("Surface") or "hard")
        if not mid or len(raw_date) < 4:
            continue
        try:
            year = int(raw_date[:4])
            iso_date = _parse_date(raw_date.replace("-", "")) if "-" in raw_date else _parse_date(raw_date)
            match_meta[mid] = (year, iso_date, surf)
        except ValueError:
            continue

    entries: dict[str, list[dict]] = defaultdict(list)
    for row in stats:
        if (row.get("set") or "").strip() != "Total":
            continue
        mid = row.get("match_id")
        meta = match_meta.get(mid)
        if meta is None or meta[0] not in years:
            continue
        name = row.get("player")
        if not name:
            continue
        try:
            svpt = int(row.get("serve_pts") or 0)
            first_won = int(row.get("first_won") or 0)
            second_won = int(row.get("second_won") or 0)
        except ValueError:
            continue
        if svpt <= 0:
            continue
        key = _player_key(name)
        entries[key].append({
            "date": meta[1],
            "surface": meta[2],
            "won": first_won + second_won,
            "svpt": svpt,
        })

    print(f"  + MCP {tour}: {len(entries)} players from charted matches")
    return dict(entries)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def collect_matches(tour: str, years: list[int]) -> list[dict]:
    """Pull match CSVs for every requested year. 404s are silently skipped."""
    base = REPO_BASE[tour]
    rows: list[dict] = []
    for year in years:
        url = f"{base}/{tour}_matches_{year}.csv"
        data = _fetch_csv(url)
        if data is None:
            print(f"  - {tour}_matches_{year}: not available")
            continue
        rows.extend(data)
        print(f"  + {tour}_matches_{year}: {len(data)} matches")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed tennis model agents from Sackmann data")
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
        help="Skip Match Charting Project augmentation (Sackmann-only seeding)",
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
        # Ranking history
        history = load_ranking_history(tour)
        # Advanced stats
        advanced = aggregate_advanced_stats(matches)

        print(
            f"  → {len(match_serve)} players with serve stats, "
            f"{len(match_history)} players with form history, "
            f"{len(h2h_records)} H2H pairs, "
            f"{len(history)} players with ranking history, "
            f"{len(advanced)} players with advanced stats"
        )

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

        if not args.no_mcp:
            mcp_serve = load_mcp_serve_stats(tour, year_set)
            for player, entries in mcp_serve.items():
                combined_match_serve.setdefault(player, []).extend(entries)
            augment_advanced_with_mcp(combined_advanced, tour, year_set)

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
