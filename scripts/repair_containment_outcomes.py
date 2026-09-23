"""Repair rows mis-graded by the resolver's fuzzy name-containment bug.

Before 2026-09-22 the resolver placed a YES team on an event's two teams with
``rapidfuzz.token_set_ratio`` and broke ties toward team A. ``token_set_ratio``
scores any token-subset pair at 100 ("utah" vs "utah state"), so a bet on the
CONTAINING team was graded on the contained team's result. The close-line
aligner (``yes_aligned_close_prob``) had the same bug via substring matching
("ly" ⊂ "flyquest", "m" ⊂ "hoffenheim", "maria" ⊂ "maria sakkari").

This script re-derives the affected values with the FIXED rules and rewrites
exactly the rows that differ:

  1. ``ev_outcomes.outcome`` — ESPN-resolved rows whose event teams are nested
     (one name a token subset of the other) or whose YES label is not literally
     one of the event's teams (the old relative-fuzzy fallback). The ESPN
     scoreboard is re-fetched (read-only GETs) and graded by the fixed
     ``_match_espn``. A row is rewritten only when the game the fixed matcher
     selects is dated on the event_id's own date (a stale ``event_date`` can
     point the ±1-day series window at a different game).
  2. ``ev_outcomes.pinnacle_close_prob`` + ``ev_predictions.pinnacle_drift_pct``
     — rows whose stored close equals the OTHER side's archived close, i.e. a
     side flip by the old aligner, recomputed exactly the way ``backfill_clv``
     computes them (same archive selection, same ``clv_entry_price`` anchor).

Idempotent: once applied, every re-derived value equals the stored value and a
re-run plans nothing. ``--dry-run`` (the default) opens both databases
read-only. ``--apply`` first writes a full SQLite backup of predictions.db next
to it (``predictions.db.bak-containment-<UTC timestamp>``), then updates the
planned rows inside one transaction, each UPDATE guarded on the old value.
archive.db is only ever read.

Run manually:
    python scripts/repair_containment_outcomes.py              # dry-run (default)
    python scripts/repair_containment_outcomes.py --apply      # backup + commit
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evmax.agents.cleanup.resolver import (  # noqa: E402
    ESPN_SOCCER_LIKE_LEAGUES,
    ESPN_SPORT_MAP,
    _DRAW_LABELS,
    _ESPN_HTTP_UA,
    _fetch_espn_scores,
    _match_espn,
    _pred_sector,
    _select_espn_game,
    _sector_normalizer,
    _slug_teams,
    _to_fuzz,
    clean,
    clv_entry_price,
    yes_aligned_close_prob,
)

DEFAULT_DB = ROOT / "data" / "predictions.db"
DEFAULT_ARCHIVE = ROOT / "data" / "archive.db"

# Markets graded on the combined score — the YES side never enters the grade.
_SIDE_INDEPENDENT = ("total", "over_under")
# Market types ``backfill_clv`` gives a team-aligned Pinnacle close.
_CLOSE_MARKET_TYPES = ("spread", "moneyline", "ml", "")


@dataclass
class OutcomeFix:
    market_id: str
    pred_id: int
    sector: str
    event_title: str
    yes_team: str
    market_type: str
    mode: str
    placed: int
    stored: int
    correct: int
    game: str


@dataclass
class CloseFix:
    market_id: str
    sector: str
    event_title: str
    yes_team: str
    market_type: str
    mode: str
    placed: int
    stored_close: float
    correct_close: float
    drift_updates: list[tuple[int, Optional[float], float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def teams_nested(slug_a: str, slug_b: str, sector: str) -> bool:
    """True when one team's name is a token subset of the other's.

    Checked on the resolver's folded form (accents, punctuation, acronym
    expansion: "psg" → "paris saint germain") and on the sector-canonical form.
    """
    forms = [(_to_fuzz(slug_a), _to_fuzz(slug_b))]
    normalizer = _sector_normalizer(sector)
    if normalizer is not None:
        forms.append((clean(normalizer.normalize(slug_a)), clean(normalizer.normalize(slug_b))))
    for a, b in forms:
        ta, tb = set(a.split()), set(b.split())
        if ta and tb and (ta <= tb or tb <= ta):
            return True
    return False


def yes_is_literal_team(yes_team: str, slug_a: str, slug_b: str) -> bool:
    """True when the YES label IS one of the slug's teams (no fallback needed)."""
    y = clean(yes_team)
    return bool(y) and y in (clean(slug_a), clean(slug_b))


def is_outcome_candidate(row: dict) -> bool:
    """Rows whose stored ESPN grade depended on the old tie-to-team-A logic."""
    market_type = (row.get("market_type") or "moneyline").lower()
    if market_type in _SIDE_INDEPENDENT:
        return False
    if _to_fuzz(row.get("yes_team") or "") in _DRAW_LABELS:
        return False
    slug_a, slug_b = _slug_teams(row["event_id"])
    if not (slug_a and slug_b):
        return False
    sector = _pred_sector(row)
    return teams_nested(slug_a, slug_b, sector) or not yes_is_literal_team(
        row.get("yes_team") or "", slug_a, slug_b,
    )


def load_outcome_candidates(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT p.id, p.market_id, p.event_id, p.sector, p.yes_team, p.event_title,
                  p.event_date, p.scan_date, p.market_type, p.line, p.mode, p.placed,
                  o.outcome
           FROM ev_predictions p
           JOIN ev_outcomes o ON o.market_id = p.market_id
           WHERE o.result_source = 'espn' AND o.outcome IS NOT NULL
           ORDER BY p.scan_date DESC, p.id DESC"""
    ).fetchall()
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        if d["market_id"] in seen:
            continue
        seen.add(d["market_id"])
        if is_outcome_candidate(d):
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Outcome planning
# ---------------------------------------------------------------------------

def _espn_jobs(sector: str, event_date: str) -> list[tuple[str, str, str, dict]]:
    center = date.fromisoformat(event_date)
    days = [(center + timedelta(days=k)).isoformat().replace("-", "") for k in (-1, 0, 1)]
    if sector in ESPN_SPORT_MAP:
        sport, league, params = ESPN_SPORT_MAP[sector]
        return [(sport, league, d, params) for d in days]
    if sector in ESPN_SOCCER_LIKE_LEAGUES:
        return [("soccer", lg, d, {}) for lg in ESPN_SOCCER_LIKE_LEAGUES[sector] for d in days]
    return []


async def fetch_scores(candidates: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Re-fetch the resolver's 3-day ESPN window per (sector, event_date). Read-only."""
    import httpx

    groups = sorted({(c["sector"], c["event_date"] or c["scan_date"]) for c in candidates})
    cache: dict = {}
    out: dict[tuple[str, str], list[dict]] = {}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0), headers={"User-Agent": _ESPN_HTTP_UA},
        follow_redirects=True,
    ) as client:
        for sector, event_date in groups:
            jobs = _espn_jobs(sector, event_date)
            per = await asyncio.gather(*(
                _fetch_espn_scores(client, sport, league, d, params or None, cache=cache)
                for sport, league, d, params in jobs
            ))
            out[(sector, event_date)] = [s for day in per for s in day]
    return out


def plan_outcome_fixes(
    candidates: list[dict], scores: dict[tuple[str, str], list[dict]],
) -> tuple[list[OutcomeFix], dict[str, int]]:
    """Grade each candidate with the fixed matcher; keep rows whose grade changes."""
    fixes: list[OutcomeFix] = []
    skipped = {"unresolved_by_fixed_rules": 0, "game_date_mismatch": 0, "unchanged": 0}
    for c in candidates:
        sc = scores.get((c["sector"], c["event_date"] or c["scan_date"]), [])
        selected = _select_espn_game(c, sc)
        correct = _match_espn(c, sc)
        if selected is None or correct is None:
            skipped["unresolved_by_fixed_rules"] += 1
            continue
        game, _ = selected
        event_id_date = c["event_id"].split("::")[1] if "::" in c["event_id"] else ""
        if game.get("game_date") and event_id_date and game["game_date"] != event_id_date:
            skipped["game_date_mismatch"] += 1
            continue
        if correct == c["outcome"]:
            skipped["unchanged"] += 1
            continue
        fixes.append(OutcomeFix(
            market_id=c["market_id"], pred_id=c["id"], sector=c["sector"],
            event_title=c.get("event_title") or "", yes_team=c.get("yes_team") or "",
            market_type=c.get("market_type") or "moneyline", mode=c.get("mode") or "",
            placed=int(c.get("placed") or 0), stored=int(c["outcome"]), correct=int(correct),
            game=(f"{game.get('game_date', '')} {game['home_name']} {game['home_score']}"
                  f" - {game['away_name']} {game['away_score']}"),
        ))
    return fixes, skipped


# ---------------------------------------------------------------------------
# Close-prob planning
# ---------------------------------------------------------------------------

def _archived_close(aconn: sqlite3.Connection, event_id: str, market_type: str,
                    line: Optional[float], tolerance: float) -> Optional[sqlite3.Row]:
    """The snapshot ``DataArchiver.get_(spread_)closing_line_aligned`` would read."""
    if market_type == "spread":
        if line is None:
            return aconn.execute(
                """SELECT outcome_a_label, outcome_b_label, true_prob_a, true_prob_b, true_prob_draw
                   FROM archived_sharp_odds
                   WHERE event_id = ? AND spread_line IS NOT NULL
                     AND event_date IS NOT NULL AND fetched_at < event_date
                   ORDER BY fetched_at DESC LIMIT 1""",
                (event_id,),
            ).fetchone()
        exact = aconn.execute(
            """SELECT outcome_a_label, outcome_b_label, true_prob_a, true_prob_b, true_prob_draw
               FROM archived_sharp_odds
               WHERE event_id = ? AND spread_line IS NOT NULL
                 AND ABS(ABS(spread_line) - ABS(?)) < 0.01
                 AND event_date IS NOT NULL AND fetched_at < event_date
               ORDER BY fetched_at DESC LIMIT 1""",
            (event_id, float(line)),
        ).fetchone()
        if exact is not None:
            return exact
        return aconn.execute(
            """SELECT outcome_a_label, outcome_b_label, true_prob_a, true_prob_b, true_prob_draw
               FROM archived_sharp_odds
               WHERE event_id = ? AND spread_line IS NOT NULL
                 AND ABS(ABS(spread_line) - ABS(?)) <= ?
                 AND event_date IS NOT NULL AND fetched_at < event_date
               ORDER BY ABS(ABS(spread_line) - ABS(?)) ASC, fetched_at DESC LIMIT 1""",
            (event_id, float(line), tolerance, float(line)),
        ).fetchone()
    return aconn.execute(
        """SELECT outcome_a_label, outcome_b_label, true_prob_a, true_prob_b, true_prob_draw
           FROM archived_sharp_odds
           WHERE event_id = ? AND spread_line IS NULL
             AND event_date IS NOT NULL AND fetched_at < event_date
           ORDER BY fetched_at DESC LIMIT 1""",
        (event_id,),
    ).fetchone()


def plan_close_fixes(pconn: sqlite3.Connection, aconn: sqlite3.Connection,
                     tolerance: float) -> list[CloseFix]:
    """Stored closes that equal the OTHER side's archived close (old-aligner side flips)."""
    rows = pconn.execute(
        """SELECT p.id, p.market_id, p.event_id, p.sector, p.yes_team, p.event_title,
                  p.market_type, p.line, p.mode, p.placed, p.placed_price,
                  p.kalshi_yes_price, p.pinnacle_drift_pct, o.pinnacle_close_prob
           FROM ev_predictions p
           JOIN ev_outcomes o ON o.market_id = p.market_id
           WHERE o.outcome IS NOT NULL AND o.pinnacle_close_prob IS NOT NULL
             AND p.event_id NOT LIKE '%::prop::%'
           ORDER BY p.market_id, p.id"""
    ).fetchall()
    by_market: dict[str, list[dict]] = {}
    for r in rows:
        by_market.setdefault(r["market_id"], []).append(dict(r))

    fixes: list[CloseFix] = []
    for market_id, preds in by_market.items():
        head = preds[0]
        market_type = (head["market_type"] or "").lower()
        if market_type not in _CLOSE_MARKET_TYPES:
            continue
        snap = _archived_close(aconn, head["event_id"], market_type, head["line"], tolerance)
        if snap is None or snap["true_prob_a"] is None or snap["true_prob_b"] is None:
            continue
        correct = yes_aligned_close_prob(
            yes_team=head["yes_team"],
            outcome_a_label=snap["outcome_a_label"],
            outcome_b_label=snap["outcome_b_label"],
            true_prob_a=snap["true_prob_a"],
            true_prob_b=snap["true_prob_b"],
            true_prob_draw=snap["true_prob_draw"],
            sector=head["event_id"].split("::", 1)[0] or None,
        )
        stored = head["pinnacle_close_prob"]
        if correct is None or stored == correct:
            continue
        if correct == snap["true_prob_a"]:
            other = snap["true_prob_b"]
        elif correct == snap["true_prob_b"]:
            other = snap["true_prob_a"]
        else:  # draw leg — never a two-way side flip
            continue
        if stored != other:
            continue  # a different snapshot, not a side flip — leave it alone
        fix = CloseFix(
            market_id=market_id, sector=head["sector"], event_title=head["event_title"] or "",
            yes_team=head["yes_team"] or "", market_type=market_type, mode=head["mode"] or "",
            placed=int(head["placed"] or 0), stored_close=stored, correct_close=correct,
        )
        for p in preds:
            if p["pinnacle_drift_pct"] is None:
                continue
            entry = clv_entry_price(p["placed"], p["placed_price"], p["kalshi_yes_price"])
            if entry is None or not 0 < entry < 1:
                continue
            fix.drift_updates.append((p["id"], p["pinnacle_drift_pct"], (correct - entry) * 100))
        fixes.append(fix)
    return fixes


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def backup_database(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = db_path.with_name(f"{db_path.name}.bak-containment-{stamp}")
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def apply_fixes(conn: sqlite3.Connection, outcome_fixes: list[OutcomeFix],
                close_fixes: list[CloseFix]) -> dict[str, int]:
    """Write the planned rows in one transaction; each UPDATE guards on the old value."""
    counts = {"outcomes": 0, "closes": 0, "drifts": 0}
    with conn:
        for f in outcome_fixes:
            cur = conn.execute(
                "UPDATE ev_outcomes SET outcome = ? WHERE market_id = ? AND outcome = ?",
                (f.correct, f.market_id, f.stored),
            )
            counts["outcomes"] += cur.rowcount
        for f in close_fixes:
            cur = conn.execute(
                "UPDATE ev_outcomes SET pinnacle_close_prob = ? "
                "WHERE market_id = ? AND pinnacle_close_prob = ?",
                (f.correct_close, f.market_id, f.stored_close),
            )
            counts["closes"] += cur.rowcount
            for pred_id, old_drift, new_drift in f.drift_updates:
                cur = conn.execute(
                    "UPDATE ev_predictions SET pinnacle_drift_pct = ? "
                    "WHERE id = ? AND pinnacle_drift_pct = ?",
                    (new_drift, pred_id, old_drift),
                )
                counts["drifts"] += cur.rowcount
    return counts


def _portfolio_rows(conn: sqlite3.Connection, market_ids: list[str]) -> list[sqlite3.Row]:
    has = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolio_bets'"
    ).fetchone()
    if not has or not market_ids:
        return []
    q = ",".join("?" * len(market_ids))
    return conn.execute(
        f"SELECT portfolio_id, market_id, outcome, pnl FROM portfolio_bets "
        f"WHERE market_id IN ({q}) AND outcome IS NOT NULL",
        market_ids,
    ).fetchall()


def _print_plan(outcome_fixes: list[OutcomeFix], skipped: dict[str, int], n_candidates: int,
                close_fixes: list[CloseFix]) -> None:
    print(f"=== ev_outcomes.outcome — {n_candidates} candidate rows "
          f"(nested team names or non-literal YES label) ===")
    for f in outcome_fixes:
        print(f"  {f.market_id}  [{f.sector} {f.market_type}] {f.event_title}")
        print(f"    yes_team={f.yes_team!r} mode={f.mode} placed={f.placed}  "
              f"outcome {f.stored} -> {f.correct}   (ESPN: {f.game})")
    print(f"  planned: {len(outcome_fixes)}   skipped: {skipped}")
    print("\n=== ev_outcomes.pinnacle_close_prob (+ ev_predictions.pinnacle_drift_pct) ===")
    for f in close_fixes:
        print(f"  {f.market_id}  [{f.sector} {f.market_type}] {f.event_title}")
        print(f"    yes_team={f.yes_team!r} mode={f.mode} placed={f.placed}  "
              f"close {f.stored_close:.4f} -> {f.correct_close:.4f}")
        for pred_id, old_d, new_d in f.drift_updates:
            print(f"    ev_predictions#{pred_id} pinnacle_drift_pct {old_d:+.2f} -> {new_d:+.2f}")
    print(f"  planned: {len(close_fixes)}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="print the plan without writing (default)")
    mode.add_argument("--apply", action="store_true", help="back up predictions.db, then write")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = p.parse_args(argv)

    from evmax.archiver import _CLOSING_LINE_FALLBACK_TOLERANCE as tolerance

    ro = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    aro = sqlite3.connect(f"file:{args.archive.resolve()}?mode=ro", uri=True)
    aro.row_factory = sqlite3.Row

    candidates = load_outcome_candidates(ro)
    scores = asyncio.run(fetch_scores(candidates)) if candidates else {}
    outcome_fixes, skipped = plan_outcome_fixes(candidates, scores)
    close_fixes = plan_close_fixes(ro, aro, tolerance)
    _print_plan(outcome_fixes, skipped, len(candidates), close_fixes)

    portfolio = _portfolio_rows(ro, sorted({f.market_id for f in outcome_fixes}))
    if portfolio:
        print("\nWARNING: portfolio_bets rows copied the old outcome (not touched — "
              "re-sync portfolios after --apply):")
        for r in portfolio:
            print(f"  {r['portfolio_id']} {r['market_id']} outcome={r['outcome']} pnl={r['pnl']}")
    ro.close()
    aro.close()

    if not args.apply:
        print("\ndry-run: nothing written (pass --apply to back up and commit)")
        return 0
    if not outcome_fixes and not close_fixes:
        print("\nnothing to apply")
        return 0
    backup = backup_database(args.db)
    print(f"\nbackup: {backup}")
    conn = sqlite3.connect(str(args.db), timeout=10.0)
    try:
        counts = apply_fixes(conn, outcome_fixes, close_fixes)
    finally:
        conn.close()
    print(f"applied: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
