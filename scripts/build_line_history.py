"""Collect closing spreads, totals and final scores for basketball leagues.

Source: ESPN's public API. The scoreboard gives each day's completed games and
scores; the competition-odds document gives the closing HOME handicap and
total for the same event. Output is one CSV per league under
data/backtest/line_history/, the training set for the alternate-line models
(scripts/fit_line_pmf.py). NFL does not need this: the nflverse schedule
already carries closing lines.

    python scripts/build_line_history.py --league nba --seasons 2023,2024,2025,2026
    python scripts/build_line_history.py --league ncaab --seasons 2025,2026
    python scripts/build_line_history.py --league ncaaw --seasons 2025,2026

A season is named by the year it ENDS (2025 = 2024-25). Every response is
cached on disk, so a re-run only fetches what is missing. Requests carry event
ids only.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/{path}/scoreboard"
CORE = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/{path}"
        "/events/{eid}/competitions/{eid}/odds")
LEAGUES = {
    # league: (ESPN path, scoreboard extra params, season start (month, day), end (month, day))
    "nba": ("nba", {}, (10, 15), (6, 25)),
    "ncaab": ("mens-college-basketball", {"groups": "50", "limit": "400"}, (11, 1), (4, 10)),
    "ncaaw": ("womens-college-basketball", {"groups": "50", "limit": "400"}, (11, 1), (4, 10)),
}
FIELDS = ["league", "season", "date", "event_id", "home", "away", "home_score", "away_score",
          "home_spread", "total", "provider", "periods", "season_type"]


def season_days(season: int, start: tuple[int, int], end: tuple[int, int]) -> list[str]:
    d, last = date(season - 1, *start), min(date(season, *end), date.today() - timedelta(days=1))
    out = []
    while d <= last:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def parse_scoreboard(doc: dict) -> list[dict]:
    """Completed games of one scoreboard day: ids, teams, scores, periods."""
    out = []
    for ev in doc.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        status = (comp.get("status") or ev.get("status") or {})
        if not (status.get("type") or {}).get("completed"):
            continue
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        h, a = sides.get("home"), sides.get("away")
        try:
            hs, as_ = float(h["score"]), float(a["score"])
        except (TypeError, KeyError, ValueError):
            continue
        out.append({
            "event_id": str(ev.get("id")),
            "home": (h.get("team") or {}).get("displayName"),
            "away": (a.get("team") or {}).get("displayName"),
            "home_score": hs, "away_score": as_,
            "periods": status.get("period"),
            "season_type": (ev.get("season") or {}).get("type"),
        })
    return out


def parse_odds(doc: dict | None) -> tuple[float | None, float | None, str | None]:
    """(closing HOME handicap, closing total, provider) from a competition-odds document.

    ESPN's ``spread`` is the home team's handicap (negative = home favored).
    Live in-game providers are skipped, and a 0 / 0 placeholder is not a line.
    """
    for item in (doc or {}).get("items") or []:
        name = str((item.get("provider") or {}).get("name") or "")
        if "live" in name.lower():
            continue
        spread, total = item.get("spread"), item.get("overUnder")
        try:
            spread = None if spread is None else float(spread)
            total = None if total is None else float(total)
        except (TypeError, ValueError):
            continue
        if total is not None and total <= 0:
            total = None
        if spread is None and total is None:
            continue
        if spread == 0 and total is None:
            continue
        return spread, total, name
    return None, None, None


async def collect(league: str, seasons: list[int], cache: Path, rps: float) -> tuple[list[dict], Counter]:
    import httpx
    from aiolimiter import AsyncLimiter

    path, extra, start, end = LEAGUES[league]
    limiter = AsyncLimiter(max(1, int(rps)), 1.0)
    sem = asyncio.Semaphore(12)
    stats: Counter = Counter()
    rows: list[dict] = []

    async def get(client, url, params, cache_file: Path):
        if cache_file.exists():
            stats["cache_hits"] += 1
            return json.loads(cache_file.read_text())
        for attempt in range(4):
            async with limiter:
                try:
                    r = await client.get(url, params=params)
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(1 + attempt)
                    continue
            if r.status_code == 404:
                doc = {}
                break
            if r.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            doc = r.json()
            break
        else:
            stats["errors"] += 1
            return None
        stats["network_calls"] += 1
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(doc, separators=(",", ":")))
        return doc

    async def one_day(client, season, day):
        async with sem:
            doc = await get(client, SITE.format(path=path), {"dates": day, **extra},
                            cache / league / "days" / f"{day}.json")
        games = parse_scoreboard(doc or {})
        # keep the day file small: only what parse_scoreboard reads is needed again
        for g in games:
            async with sem:
                odds = await get(client, CORE.format(path=path, eid=g["event_id"]), {"limit": 20},
                                 cache / league / "odds" / f"{g['event_id']}.json")
            spread, total, provider = parse_odds(odds)
            stats["games"] += 1
            if spread is None and total is None:
                stats["no_line"] += 1
                continue
            rows.append({"league": league, "season": season,
                         "date": f"{day[:4]}-{day[4:6]}-{day[6:]}", **g,
                         "home_spread": spread, "total": total, "provider": provider})

    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
        for season in seasons:
            days = season_days(season, start, end)
            await asyncio.gather(*(one_day(client, season, d) for d in days))
            print(f"  {league} {season}: {stats['games']} games so far, {len(rows)} with a line, "
                  f"{stats['network_calls']} calls, {stats['errors']} errors", flush=True)
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", required=True, choices=sorted(LEAGUES))
    ap.add_argument("--seasons", required=True, help="comma list of season END years")
    ap.add_argument("--out-dir", default=str(_REPO / "data" / "backtest" / "line_history"))
    ap.add_argument("--rps", type=float, default=8.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    rows, stats = asyncio.run(collect(args.league, seasons, out_dir / "_cache", args.rps))
    rows.sort(key=lambda r: (r["date"], r["event_id"]))
    out = out_dir / f"{args.league}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} games with a line | {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
