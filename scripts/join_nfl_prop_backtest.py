"""Join Stage 1 Kalshi raw markets with Stage 2 nflverse features.

Produces `data/backtest/nfl_props/joined.parquet` — one row per settled
binary prop market, enriched with:
  - player identity (gsis_id) via normalized-name join to weekly rosters
  - game context (season, week, team, opponent, home flag) via schedules

Also filters out scalar/range markets (result="scalar") and prints a shape
report used to decide whether Stage 4 modeling is worthwhile.

Usage:
    python3 scripts/join_nfl_prop_backtest.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path("data/backtest/nfl_props")

# Series → canonical stat type
SERIES_TO_STAT: dict[str, str] = {
    "KXNFLPASSYDS": "passing_yards",
    "KXNFLRSHYDS": "rushing_yards",
    "KXNFLRECYDS": "receiving_yards",
    "KXNFLANYTD": "anytime_td",
    "KXNFLPASSTDS": "passing_tds",
    "KXNFLREC": "receptions",
}

# Positions that ever appear in offensive prop markets. Used to break ties
# when multiple NFL players share a name (e.g. Lamar Jackson QB vs DB).
OFFENSIVE_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}

# Kalshi occasionally lists non-player "players" in ANYTD markets —
# filter these out before the join so they don't pollute the match rate.
NON_PLAYER_KEYWORDS = ("no touchdown", "d/st", "defense/special", "defense")

# Kalshi uses a handful of nicknames the nflverse rosters don't.
# Minimal hand-maintained alias map — only the systematic misses.
NICKNAME_ALIASES: dict[str, str] = {
    "hollywood brown": "marquise brown",
    "joshua palmer": "josh palmer",
}

# YY MON DD ticker date codes → iso date
_MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}
_TICKER_DATE_RE = re.compile(
    r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})"
)


def _normalize(name: str) -> str:
    """Mirror evmax/agents/cleanup/prop_resolver.py::_normalize_for_match (line 149)."""
    name = name.lower().strip().replace("_", " ")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"\s+(jr\.?|sr\.?|iii|ii|iv|v)$", "", name, flags=re.IGNORECASE)
    # Drop punctuation that doesn't carry identity
    name = name.replace(".", "").replace("'", "")
    return re.sub(r"\s+", " ", name).strip()


def _parse_ticker_date(event_ticker: str) -> str | None:
    """KXNFLPASSYDS-25DEC01NYGNE → '2025-12-01'."""
    m = _TICKER_DATE_RE.search(event_ticker or "")
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2).upper(), m.group(3)
    return f"20{yy}-{_MONTHS[mon]}-{dd}"


def _extract_player(yes_sub_title: str) -> str | None:
    if not yes_sub_title:
        return None
    if ":" in yes_sub_title:
        return yes_sub_title.split(":", 1)[0].strip()
    return yes_sub_title.strip()


def _safe_float(x) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def load_kalshi(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text())
    rows = []
    for m in raw:
        series = m.get("_series_ticker", "")
        stat = SERIES_TO_STAT.get(series)
        if stat is None:
            continue
        result = m.get("result", "")
        if result not in ("yes", "no"):
            continue  # drop scalar / unresolved
        player = _extract_player(m.get("yes_sub_title", ""))
        if not player:
            continue
        # Drop game-level / defense ANYTD markets
        low = player.lower()
        if any(k in low for k in NON_PLAYER_KEYWORDS):
            continue
        game_date = _parse_ticker_date(m.get("event_ticker", ""))
        if not game_date:
            continue
        threshold = _safe_float(m.get("floor_strike"))
        # Anytime-TD markets have no floor_strike; the implied threshold is "≥ 1"
        if threshold is None and stat == "anytime_td":
            threshold = 0.5
        if threshold is None:
            continue
        rows.append({
            "ticker": m.get("ticker"),
            "event_ticker": m.get("event_ticker"),
            "series": series,
            "stat_type": stat,
            "player_name": player,
            "player_norm": NICKNAME_ALIASES.get(_normalize(player), _normalize(player)),
            "threshold": threshold,
            "result": 1 if result == "yes" else 0,
            "volume_fp": _safe_float(m.get("volume_fp")) or 0.0,
            "yes_ask_dollars": _safe_float(m.get("yes_ask_dollars")),
            "yes_bid_dollars": _safe_float(m.get("yes_bid_dollars")),
            "last_price_dollars": _safe_float(m.get("last_price_dollars")),
            "game_date": game_date,
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time"),
        })
    return pd.DataFrame(rows)


def build_roster_lookup(
    rosters: pd.DataFrame,
) -> tuple[dict[str, str], dict[tuple, str], dict[str, set[str]], pd.DataFrame]:
    """Build normalized-name → gsis_id lookup with offensive-position bias.

    Empty gsis_ids (which some rosters carry for unsigned/placeholder slots)
    are dropped before the ambiguity check so they can't break an otherwise
    unique player. When the same name belongs to multiple real players,
    prefer an offensive position (QB/RB/WR/TE/FB) — every Kalshi prop stat
    in scope is an offensive one.
    """
    df = rosters[["gsis_id", "full_name", "season", "week", "team", "position"]].copy()
    df = df.dropna(subset=["gsis_id", "full_name"])
    df = df[df["gsis_id"].astype(str).str.strip() != ""]
    df["norm"] = df["full_name"].map(_normalize)

    # Per (norm_name, gsis_id), record whether the player ever played offense
    offensive_ids: set[str] = set(
        df[df["position"].isin(OFFENSIVE_POSITIONS)]["gsis_id"].unique()
    )

    name_to_ids: dict[str, set[str]] = {}
    for norm, gsis in zip(df["norm"], df["gsis_id"]):
        name_to_ids.setdefault(norm, set()).add(gsis)

    unambiguous: dict[str, str] = {}
    ambiguous: dict[str, set[str]] = {}
    for name, ids in name_to_ids.items():
        if len(ids) == 1:
            unambiguous[name] = next(iter(ids))
            continue
        offensive_hits = ids & offensive_ids
        if len(offensive_hits) == 1:
            unambiguous[name] = next(iter(offensive_hits))
        else:
            ambiguous[name] = ids

    player_week_team: dict[tuple, str] = {}
    for gsis, season, week, team in zip(
        df["gsis_id"], df["season"], df["week"], df["team"]
    ):
        player_week_team[(gsis, int(season), int(week))] = team

    return unambiguous, player_week_team, ambiguous, df


def last_name_fallback(norm: str, name_index: dict[str, set[str]]) -> str | None:
    """If last name is unique across rosters, resolve to that gsis_id."""
    parts = norm.split()
    if not parts:
        return None
    last = parts[-1]
    candidates = name_index.get(last, set())
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def first_initial_fallback(
    norm: str, initial_index: dict[tuple[str, str], set[str]]
) -> str | None:
    """'joshua palmer' matches roster 'josh palmer' via (first_initial, last_name)."""
    parts = norm.split()
    if len(parts) < 2:
        return None
    key = (parts[0][0], parts[-1])
    candidates = initial_index.get(key, set())
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def build_last_name_index(roster_df: pd.DataFrame) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    offensive = roster_df[roster_df["position"].isin(OFFENSIVE_POSITIONS)]
    for norm, gsis in zip(offensive["norm"], offensive["gsis_id"]):
        parts = norm.split()
        if parts:
            idx.setdefault(parts[-1], set()).add(gsis)
    return idx


def build_initial_index(roster_df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Map (first_initial, last_name) → gsis_id set, offensive positions only."""
    idx: dict[tuple[str, str], set[str]] = {}
    offensive = roster_df[roster_df["position"].isin(OFFENSIVE_POSITIONS)]
    for norm, gsis in zip(offensive["norm"], offensive["gsis_id"]):
        parts = norm.split()
        if len(parts) >= 2 and parts[0]:
            idx.setdefault((parts[0][0], parts[-1]), set()).add(gsis)
    return idx


def build_schedule_lookup(schedules: pd.DataFrame) -> tuple[dict, dict]:
    """Return:
      - date_to_week: gameday → (season, week) (many games share this)
      - game_ctx: (gameday, team) → {opponent, home, game_id, game_type}
    """
    date_to_week: dict[str, tuple[int, int]] = {}
    game_ctx: dict[tuple[str, str], dict] = {}
    for _, g in schedules.iterrows():
        gd = str(g["gameday"])
        date_to_week[gd] = (int(g["season"]), int(g["week"]))
        game_ctx[(gd, g["home_team"])] = {
            "opponent": g["away_team"],
            "home": True,
            "game_id": g["game_id"],
            "game_type": g.get("game_type"),
        }
        game_ctx[(gd, g["away_team"])] = {
            "opponent": g["home_team"],
            "home": False,
            "game_id": g["game_id"],
            "game_type": g.get("game_type"),
        }
    return date_to_week, game_ctx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()

    print("loading Kalshi raw …")
    kalshi = load_kalshi(args.dir / "kalshi_raw.json")
    total_raw = len(kalshi)
    print(f"  {total_raw} binary markets after filtering scalar/unresolved")

    print("loading nflverse parquets …")
    rosters = pd.read_parquet(args.dir / "rosters.parquet")
    schedules = pd.read_parquet(args.dir / "schedules.parquet")

    print("building roster lookup …")
    unambiguous, player_week_team, ambiguous, roster_df = build_roster_lookup(rosters)
    last_name_index = build_last_name_index(roster_df)
    initial_index = build_initial_index(roster_df)
    print(f"  unambiguous names: {len(unambiguous)}")
    print(f"  ambiguous names: {len(ambiguous)} (will try fallbacks)")

    print("building schedule lookup …")
    date_to_week, game_ctx = build_schedule_lookup(schedules)

    # Player-name join — exact → last-name → first-initial+last fallbacks
    gsis_ids: list[str | None] = []
    fallback_last = 0
    fallback_initial = 0
    for norm in kalshi["player_norm"]:
        gid = unambiguous.get(norm)
        if gid is None:
            gid = last_name_fallback(norm, last_name_index)
            if gid is not None:
                fallback_last += 1
        if gid is None:
            gid = first_initial_fallback(norm, initial_index)
            if gid is not None:
                fallback_initial += 1
        gsis_ids.append(gid)
    kalshi["gsis_id"] = gsis_ids
    matched = kalshi["gsis_id"].notna().sum()
    match_rate = matched / max(total_raw, 1)

    # Resolve (season, week) for each row, then player's team, then opponent
    seasons: list[int | None] = []
    weeks: list[int | None] = []
    teams: list[str | None] = []
    opps: list[str | None] = []
    homes: list[bool | None] = []
    game_ids: list[str | None] = []
    game_types: list[str | None] = []
    for gsis, gd in zip(kalshi["gsis_id"], kalshi["game_date"]):
        sw = date_to_week.get(gd)
        if sw is None or gsis is None:
            seasons.append(None)
            weeks.append(None)
            teams.append(None)
            opps.append(None)
            homes.append(None)
            game_ids.append(None)
            game_types.append(None)
            continue
        s, w = sw
        seasons.append(s)
        weeks.append(w)
        team = player_week_team.get((gsis, s, w))
        if team is None:
            # Player wasn't on an active roster that week (inactive/released)
            teams.append(None)
            opps.append(None)
            homes.append(None)
            game_ids.append(None)
            game_types.append(None)
            continue
        teams.append(team)
        ctx = game_ctx.get((gd, team))
        if ctx is None:
            opps.append(None)
            homes.append(None)
            game_ids.append(None)
            game_types.append(None)
            continue
        opps.append(ctx["opponent"])
        homes.append(ctx["home"])
        game_ids.append(ctx["game_id"])
        game_types.append(ctx["game_type"])
    kalshi["season"] = seasons
    kalshi["week"] = weeks
    kalshi["team"] = teams
    kalshi["opponent"] = opps
    kalshi["is_home"] = homes
    kalshi["game_id"] = game_ids
    kalshi["game_type"] = game_types

    fully_joined = kalshi["opponent"].notna().sum()

    # Shape report
    print()
    print("=" * 60)
    print("SHAPE REPORT")
    print("=" * 60)
    print(f"total binary markets:        {total_raw:>6}")
    print(f"player name matched:         {matched:>6}  ({match_rate*100:.1f}%)")
    print(f"  via last-name fallback:    {fallback_last:>6}")
    print(f"  via first-initial fallback:{fallback_initial:>6}")
    print(f"fully joined (w/ opponent):  {fully_joined:>6}  ({fully_joined/total_raw*100:.1f}%)")
    print()
    print("per-stat coverage:")
    by_stat = kalshi.groupby("stat_type").agg(
        total=("ticker", "count"),
        matched=("gsis_id", lambda s: s.notna().sum()),
        joined=("opponent", lambda s: s.notna().sum()),
        yes_rate=("result", "mean"),
        avg_threshold=("threshold", "mean"),
    ).round(3)
    print(by_stat.to_string())
    print()
    print("date span:")
    print(f"  earliest game: {kalshi['game_date'].min()}")
    print(f"  latest game:   {kalshi['game_date'].max()}")
    print(f"  unique games:  {kalshi['event_ticker'].nunique()}")
    print(f"  unique weeks:  {kalshi.dropna(subset=['week']).groupby(['season','week']).ngroups}")
    print()
    vol = kalshi["volume_fp"]
    print("volume distribution:")
    print(
        f"  median={vol.median():.0f}  p75={vol.quantile(0.75):.0f}  "
        f"p90={vol.quantile(0.90):.0f}  max={vol.max():.0f}"
    )
    print(f"  markets with vol ≥ 100:  {(vol >= 100).sum()}")
    print(f"  markets with vol ≥ 1000: {(vol >= 1000).sum()}")
    print()
    print("unmatched sample (first 10):")
    unmatched = kalshi[kalshi["gsis_id"].isna()]["player_name"].value_counts().head(10)
    if len(unmatched):
        for name, count in unmatched.items():
            print(f"  {count:>4}x {name}")
    else:
        print("  (none)")

    out = args.dir / "joined.parquet"
    kalshi.to_parquet(out, index=False)
    print()
    print(f"wrote {out}")
    print(f"rows: {len(kalshi)}, cols: {len(kalshi.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
