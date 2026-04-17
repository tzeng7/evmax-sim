"""Seed the three new tennis model agents from Jeff Sackmann's public CSVs.

Pulls match-level and ranking data from:
  - github.com/JeffSackmann/tennis_atp
  - github.com/JeffSackmann/tennis_wta

Computes and seeds:
  1. tennis_serve_return  — per-player serve-points-won % over the year window
  2. tennis_h2h            — head-to-head win records (winner, loser counts)
  3. tennis_ranking_trend  — weekly ranking snapshots from the current rankings file

Usage:
    uv run python scripts/seed_tennis_models.py
    uv run python scripts/seed_tennis_models.py --years 2023,2024
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


# ---------------------------------------------------------------------------
# Serve points won
# ---------------------------------------------------------------------------

def aggregate_serve_stats(match_rows: Iterable[dict]) -> dict[str, tuple[float, int]]:
    """For each player, compute career SPW from match rows.

    SPW = (1st serve points won + 2nd serve points won) / total serve points.
    Aggregates across both winner and loser sides of every match.
    """
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
# Advanced stats (BP conversion, RPW, UE, winners)
# ---------------------------------------------------------------------------

def aggregate_advanced_stats(match_rows: Iterable[dict]) -> dict[str, dict]:
    """Aggregate per-player advanced stats from Sackmann match rows.

    Returns {player → {bp_won, bp_faced, return_pts, return_pts_won,
    total_pts, matches, unforced, winners}} with UE/winners initialized to 0
    (filled later from MCP if available).
    """
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


def _safe_int(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


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
    """Load weekly ranking snapshots from {tour}_rankings_current.csv.

    Joins on {tour}_players.csv so the keys are normalized "first last" lowercase.
    """
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
# Main driver
# ---------------------------------------------------------------------------

def load_mcp_serve_stats(
    tour: str, years: set[int]
) -> dict[str, tuple[float, int]]:
    """Aggregate SPW per player from the Match Charting Project overview.

    MCP has community-charted match stats through late 2025, filling the gap
    where Sackmann's standard ATP/WTA match CSVs lag. We use *only* the SPW
    signal — the matches file has no winner column, so H2H stays Sackmann-only.
    """
    matches_file, stats_file = MCP_FILES[tour]
    matches = _fetch_csv(f"{MCP_BASE}/{matches_file}")
    stats = _fetch_csv(f"{MCP_BASE}/{stats_file}")
    if matches is None or stats is None:
        print(f"  ! {tour} MCP: files unavailable")
        return {}

    # Build match_id → year so we can filter by --years
    match_year: dict[str, int] = {}
    for m in matches:
        mid = m.get("match_id")
        raw_date = (m.get("Date") or "").strip()
        if not mid or len(raw_date) < 4:
            continue
        try:
            match_year[mid] = int(raw_date[:4])
        except ValueError:
            continue

    won: dict[str, int] = defaultdict(int)
    pts: dict[str, int] = defaultdict(int)
    for row in stats:
        if (row.get("set") or "").strip() != "Total":
            continue
        mid = row.get("match_id")
        year = match_year.get(mid)
        if year is None or year not in years:
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
        won[key] += first_won + second_won
        pts[key] += svpt

    print(f"  + MCP {tour}: {len(pts)} players from charted matches")
    return {p: (won[p] / pts[p], pts[p]) for p in pts if pts[p] > 0}


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

    combined_serve: dict[str, tuple[float, int]] = {}
    combined_h2h: list[dict] = []
    combined_history: dict[str, list[dict]] = {}
    combined_advanced: dict[str, dict] = {}

    for tour in tours:
        print(f"\n=== {tour.upper()} ===")
        matches = collect_matches(tour, years)
        if not matches:
            print(f"  ! no matches loaded for {tour}, skipping")
            continue

        serve_stats = aggregate_serve_stats(matches)
        h2h_records = aggregate_h2h(matches)
        history = load_ranking_history(tour)
        advanced = aggregate_advanced_stats(matches)

        print(
            f"  → {len(serve_stats)} players with serve stats, "
            f"{len(h2h_records)} H2H pairs, "
            f"{len(history)} players with ranking history, "
            f"{len(advanced)} players with advanced stats"
        )

        # Merge across tours. SPW: prefer the entry with more service points
        # (largest sample). H2H: same pair across tours is impossible. History:
        # ATP and WTA are disjoint by player, so simple update is fine.
        for player, (spw, n) in serve_stats.items():
            if player not in combined_serve or n > combined_serve[player][1]:
                combined_serve[player] = (spw, n)
        combined_h2h.extend(h2h_records)
        combined_history.update(history)
        # Advanced: merge across tours (disjoint players)
        for player, st in advanced.items():
            if player not in combined_advanced:
                combined_advanced[player] = st
            else:
                for k in st:
                    combined_advanced[player][k] = combined_advanced[player].get(k, 0) + st[k]

        if not args.no_mcp:
            mcp_stats = load_mcp_serve_stats(tour, year_set)
            for player, (spw, n) in mcp_stats.items():
                if player not in combined_serve or n > combined_serve[player][1]:
                    combined_serve[player] = (spw, n)
            augment_advanced_with_mcp(combined_advanced, tour, year_set)

    print(
        f"\n--- Seeding ---\n"
        f"  serve_return: {len(combined_serve)} players\n"
        f"  h2h:          {len(combined_h2h)} pairs\n"
        f"  trend:        {len(combined_history)} players\n"
        f"  advanced:     {len(combined_advanced)} players"
    )

    if combined_serve:
        serve_agent.seed_serve_stats(combined_serve)
    if combined_h2h:
        h2h_agent.seed_h2h(combined_h2h)
    if combined_history:
        trend_agent.seed_history(combined_history)
    if combined_advanced:
        advanced_agent.seed_stats(combined_advanced)

    print("\n[done] Agents seeded. State files:")
    print("  data/models/tennis_serve_return_state.json")
    print("  data/models/tennis_h2h_state.json")
    print("  data/models/tennis_ranking_trend_state.json")
    print("  data/models/tennis_advanced_state.json")


if __name__ == "__main__":
    main()
