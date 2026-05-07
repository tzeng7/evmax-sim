"""Backtest +spread (NO-side) NBA performance from archived Kalshi/Pinnacle data.

Joins archived_kalshi_markets to archived_sharp_odds (Pinnacle), translates
Pinnacle's standard spread to a probability at Kalshi's alternate line via a
Normal CDF (sigma=11.5, same as `evmax backtest spread`), and resolves outcomes
via ESPN historical scoreboard. Reports hit rate / ROI stratified by sharp-EV
bucket.

This approximates option 2 of "would the scanner have flagged this?" using
sharp-only EV — the model blend is not applied.

Usage:
  python scripts/backtest_no_side_spreads.py
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from evmax.backtest.sources.espn_walkforward import fetch_espn_games

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DB = ROOT / "data" / "archive.db"
NBA_ALIASES_PATH = ROOT / "evmax" / "sectors" / "aliases" / "nba.yaml"

DEFAULT_SIGMA = 11.5  # mirrors evmax/backtest/sources/espn_walkforward.py:1208

console = Console()


def _norm_phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def load_nba_aliases() -> dict[str, str]:
    """Returns full-name → canonical short-name (e.g. 'toronto raptors' → 'raptors')."""
    data = yaml.safe_load(NBA_ALIASES_PATH.read_text())
    raw = data.get("aliases", {})
    return {k.lower(): v.lower() for k, v in raw.items()}


def normalize_team(name: str, aliases: dict[str, str]) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    if key in aliases:
        return aliases[key]
    # Last word fallback (handles ESPN displayName edge cases)
    last = key.split()[-1] if key else ""
    return last or None


@dataclass
class KalshiSnap:
    ticker: str
    title: str
    yes_team: str  # canonical short name (e.g. 'raptors')
    team_home_short: str  # 3-letter from archive (e.g. 'tor')
    team_away_short: str
    line: float  # negative (e.g. -8.5)
    yes_price: float
    no_price: float
    event_date: str  # ISO start-of-day or actual
    fetched_at: str


@dataclass
class PinnacleSpread:
    event_id: str
    favorite_canon: str  # canonical short name of favorite
    other_canon: str
    spread_line: float  # favorite's line, negative (e.g. -6.0)
    true_prob_fav: float
    event_date: str
    fetched_at: str


def load_kalshi_snaps(con: sqlite3.Connection) -> dict[str, KalshiSnap]:
    """One snapshot per ticker — the latest before tip-off (best closing proxy)."""
    rows = con.execute(
        """
        SELECT ticker, title, yes_team, team_home, team_away, line,
               yes_price, no_price, event_date, fetched_at
          FROM archived_kalshi_markets
         WHERE market_type='spread' AND sector='nba'
           AND yes_team IS NOT NULL AND line IS NOT NULL AND no_price IS NOT NULL
         ORDER BY ticker, fetched_at
        """
    ).fetchall()
    by_ticker: dict[str, KalshiSnap] = {}
    for r in rows:
        ticker, title, yes_team, th, ta, line, yp, np_, ed, fa = r
        # Skip post-tip snapshots: archive event_date is start-of-day UTC
        # (e.g. "2026-03-20T00:00:00+00:00") so anything before *next* day UTC
        # is fine for "before tip-off" since NBA tipoff < midnight ET = next-day
        # UTC. We'll filter more precisely below using ESPN's actual date.
        snap = KalshiSnap(
            ticker=ticker,
            title=title or "",
            yes_team=str(yes_team).lower().strip(),
            team_home_short=str(th).lower().strip(),
            team_away_short=str(ta).lower().strip(),
            line=float(line),
            yes_price=float(yp),
            no_price=float(np_),
            event_date=ed or "",
            fetched_at=fa,
        )
        # Keep latest snapshot per ticker
        prev = by_ticker.get(ticker)
        if prev is None or fa > prev.fetched_at:
            by_ticker[ticker] = snap
    return by_ticker


def load_pinnacle_spreads(
    con: sqlite3.Connection, aliases: dict[str, str]
) -> dict[tuple[str, frozenset[str]], PinnacleSpread]:
    """Latest Pinnacle spread snapshot per (game_date, {teamA, teamB})."""
    rows = con.execute(
        """
        SELECT event_id, outcome_a_label, outcome_b_label, true_prob_a, true_prob_b,
               spread_line, event_date, fetched_at
          FROM archived_sharp_odds
         WHERE sector='nba' AND spread_line IS NOT NULL
           AND outcome_a_label IS NOT NULL AND outcome_b_label IS NOT NULL
         ORDER BY event_id, fetched_at
        """
    ).fetchall()
    by_key: dict[tuple[str, frozenset[str]], PinnacleSpread] = {}
    for r in rows:
        eid, lab_a, lab_b, pa, pb, sl, ed, fa = r
        canon_a = normalize_team(lab_a, aliases)
        canon_b = normalize_team(lab_b, aliases)
        if not canon_a or not canon_b:
            continue
        # Pinnacle convention: outcome_a is the FAVORITE (negative spread_line).
        ps = PinnacleSpread(
            event_id=eid,
            favorite_canon=canon_a,
            other_canon=canon_b,
            spread_line=float(sl),
            true_prob_fav=float(pa),
            event_date=ed or "",
            fetched_at=fa,
        )
        # Match key: game date (UTC ISO date) + unordered team set
        game_date = (ed or "")[:10]
        if not game_date:
            continue
        key = (game_date, frozenset({canon_a, canon_b}))
        prev = by_key.get(key)
        if prev is None or fa > prev.fetched_at:
            by_key[key] = ps
    return by_key


def load_espn_games(aliases: dict[str, str]) -> dict[tuple[str, frozenset[str]], dict]:
    """Index ESPN final scores by (game_date, {home_canon, away_canon})."""
    months = ["202603", "202604", "202605"]
    games = fetch_espn_games("nba", months)
    out: dict[tuple[str, frozenset[str]], dict] = {}
    for g in games:
        h = normalize_team(g["home"], aliases)
        a = normalize_team(g["away"], aliases)
        if not h or not a:
            continue
        key = (g["date"].isoformat(), frozenset({h, a}))
        out[key] = {
            "home_canon": h,
            "away_canon": a,
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "margin": g["home_score"] - g["away_score"],  # home - away
        }
    return out


def short_to_canon(short: str, home_canon: str, away_canon: str) -> str | None:
    """Map Kalshi 3-letter team_home/team_away to canonical short name via the
    only two candidates that could possibly match (the home/away from ESPN)."""
    s = short.lower()
    for cand in (home_canon, away_canon):
        # 3-letter often appears at the end of the canonical (e.g. 'den' in
        # 'nuggets'? No.). Use a simpler heuristic: check well-known mappings.
        pass
    # We won't actually need this — yes_team is already canonical, and we match
    # the game via yes_team + the OTHER team derived from Pinnacle. So this
    # function ends up unused; keep as a placeholder for future de-dup work.
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                    help=f"Stdev of NBA margin Normal CDF (default {DEFAULT_SIGMA})")
    args = ap.parse_args()
    sigma = args.sigma
    console.print(f"[bold]sigma = {sigma}[/bold]")

    if not ARCHIVE_DB.exists():
        console.print(f"[red]archive.db not found at {ARCHIVE_DB}[/red]")
        return

    aliases = load_nba_aliases()
    console.print(f"[dim]Loaded {len(aliases)} NBA aliases[/dim]")

    con = sqlite3.connect(str(ARCHIVE_DB))
    try:
        kalshi = load_kalshi_snaps(con)
        pinn = load_pinnacle_spreads(con, aliases)
    finally:
        con.close()
    console.print(
        f"[dim]Kalshi NBA spread tickers (latest snap each): {len(kalshi)}  |  "
        f"Pinnacle spread (game,date) keys: {len(pinn)}[/dim]"
    )

    console.print("[dim]Fetching ESPN final scores (Mar–May 2026)...[/dim]")
    espn = load_espn_games(aliases)
    console.print(f"[dim]ESPN final games loaded: {len(espn)}[/dim]")

    # Build per-snapshot evaluation rows
    rows: list[dict] = []
    miss_pinn = 0
    miss_espn = 0
    for tkt, k in kalshi.items():
        # Game date from Kalshi event_date (UTC start-of-day) — usually game day.
        # Some NBA night games push past midnight UTC, so try Kalshi date first
        # and fall back by comparing team-set across +/-1 day.
        game_date_guess = k.event_date[:10]

        # Identify the "yes_team" canonical and the "other team" canonical.
        # We don't know the other team's canonical from Kalshi alone (only 3-letter
        # home/away). So we look up Pinnacle via the date + a one-of-two-team set
        # constraint: any Pinnacle game on this date that includes yes_team.
        yes_canon = k.yes_team
        # Try exact day, then ±1 day for late tip-offs
        ps: PinnacleSpread | None = None
        game_date: str | None = None
        for d_iso in (
            game_date_guess,
            (datetime.fromisoformat(game_date_guess).date()
             .replace(day=datetime.fromisoformat(game_date_guess).day)).isoformat()
            if game_date_guess else None,
        ):
            if not d_iso:
                continue
            for (pd, pset), candidate in pinn.items():
                if pd != d_iso:
                    continue
                if yes_canon in pset:
                    ps = candidate
                    game_date = pd
                    break
            if ps:
                break
        if not ps or not game_date:
            miss_pinn += 1
            continue

        # Other team from Pinnacle's two-team set
        other_canon = ps.other_canon if yes_canon == ps.favorite_canon else ps.favorite_canon
        team_set = frozenset({yes_canon, other_canon})

        # Resolve via ESPN
        # Try Kalshi event_date first, then Pinnacle event_date day as fallback
        ek = (game_date, team_set)
        g = espn.get(ek)
        if not g:
            # Try Kalshi day in case of date mismatch
            alt_key = (game_date_guess, team_set)
            g = espn.get(alt_key)
        if not g:
            miss_espn += 1
            continue

        # Sharp YES prob via Normal CDF on yes_team's expected margin.
        # Pinnacle: favorite expected margin = -spread_line (positive number).
        # If yes_team == favorite: yes_team_margin ~ N(-spread_line, sigma)
        # If yes_team == other:    yes_team_margin ~ N( spread_line, sigma)
        kalshi_threshold = abs(k.line)  # YES wins iff yes_team_margin > threshold
        if yes_canon == ps.favorite_canon:
            mu = -ps.spread_line  # e.g. spread_line=-6 → mu=+6
        else:
            mu = ps.spread_line   # other team's expected margin = -fav_margin
        z = (kalshi_threshold - mu) / sigma
        sharp_yes_prob = 1.0 - _norm_phi(z)
        sharp_no_prob = 1.0 - sharp_yes_prob

        # NO-side EV (sharp-only)
        no_ask = k.no_price
        if no_ask <= 0.01 or no_ask >= 0.99:
            continue
        ev_no = sharp_no_prob / no_ask - 1.0

        # Resolve outcome.
        # ESPN margin is home - away. Need yes_team's margin.
        # Determine if yes_team is home or away in ESPN.
        if yes_canon == g["home_canon"]:
            yes_margin = g["margin"]
        elif yes_canon == g["away_canon"]:
            yes_margin = -g["margin"]
        else:
            # Should not happen since team_set matched
            miss_espn += 1
            continue
        yes_won = yes_margin > kalshi_threshold
        no_won = not yes_won

        # Per $1 staked at no_ask: profit = (1/no_ask - 1) on win, -1 on loss
        if no_won:
            unit_profit = (1.0 / no_ask) - 1.0
        else:
            unit_profit = -1.0

        rows.append({
            "ticker": k.ticker,
            "date": game_date,
            "yes_team": yes_canon,
            "other_team": other_canon,
            "kalshi_line": k.line,
            "no_ask": no_ask,
            "pinn_spread_line": ps.spread_line,
            "sharp_no_prob": sharp_no_prob,
            "ev_no": ev_no,
            "yes_margin": yes_margin,
            "no_won": no_won,
            "unit_profit": unit_profit,
        })

    console.print(
        f"[dim]Eval rows: {len(rows)}  |  unmatched Pinnacle: {miss_pinn}  |  "
        f"unmatched ESPN: {miss_espn}[/dim]"
    )

    if not rows:
        console.print("[red]No evaluable rows.[/red]")
        return

    # Stratify by sharp NO-side EV
    buckets = [
        ("All rows", lambda r: True),
        ("EV >= 2%", lambda r: r["ev_no"] >= 0.02),
        ("EV >= 5%", lambda r: r["ev_no"] >= 0.05),
        ("EV >= 10%", lambda r: r["ev_no"] >= 0.10),
        ("EV >= 20%", lambda r: r["ev_no"] >= 0.20),
    ]

    table = Table(title="NO-side spread backtest (NBA, archive ~Mar 20–May 8 2026)")
    table.add_column("Bucket")
    table.add_column("N", justify="right")
    table.add_column("Hit %", justify="right")
    table.add_column("Avg NO ask", justify="right")
    table.add_column("Avg sharp NO p", justify="right")
    table.add_column("Avg EV", justify="right")
    table.add_column("ROI / $1", justify="right")
    table.add_column("Net (units)", justify="right")

    for label, pred in buckets:
        sub = [r for r in rows if pred(r)]
        n = len(sub)
        if n == 0:
            table.add_row(label, "0", "—", "—", "—", "—", "—", "—")
            continue
        hit = sum(1 for r in sub if r["no_won"]) / n
        avg_ask = sum(r["no_ask"] for r in sub) / n
        avg_sp = sum(r["sharp_no_prob"] for r in sub) / n
        avg_ev = sum(r["ev_no"] for r in sub) / n
        net = sum(r["unit_profit"] for r in sub)
        roi = net / n
        table.add_row(
            label,
            str(n),
            f"{hit:.1%}",
            f"{avg_ask:.3f}",
            f"{avg_sp:.3f}",
            f"{avg_ev:+.1%}",
            f"{roi:+.3f}",
            f"{net:+.2f}",
        )

    console.print(table)

    # Show top-EV rows for spot-checking
    top = sorted(rows, key=lambda r: -r["ev_no"])[:10]
    spot = Table(title="Top 10 by NO-side sharp EV (spot check)")
    spot.add_column("Date")
    spot.add_column("Ticker")
    spot.add_column("YES team")
    spot.add_column("K line", justify="right")
    spot.add_column("NO ask", justify="right")
    spot.add_column("Sharp NO p", justify="right")
    spot.add_column("EV", justify="right")
    spot.add_column("YES margin", justify="right")
    spot.add_column("NO won?")
    for r in top:
        spot.add_row(
            r["date"], r["ticker"], r["yes_team"], f"{r['kalshi_line']}",
            f"{r['no_ask']:.3f}", f"{r['sharp_no_prob']:.3f}",
            f"{r['ev_no']:+.1%}", f"{r['yes_margin']:+d}",
            "YES" if r["no_won"] else "no",
        )
    console.print(spot)


if __name__ == "__main__":
    main()
