"""Consolidated pipeline integrity check — ONE read-only sweep, ONE alert.

Before this module the operational tripwires were spread across separate
schedules (launchd ``heartbeat`` + ``clv-monitor``, the weekly Claude tasks
``weekly-calibration-tripwire`` / ``weekly-wnba-listings-robustness-check``),
each with its own cadence, its own notifier call and its own docs row. This
module folds them into a single ``run_integrity`` sweep and adds the checks the
2026-08/09 incidents showed were missing. Every check is read-only (it never
voids, promotes, reseeds or touches model state) and returns issue dicts of the
shape ``{"check", "severity", "detail"}``; ``run_integrity`` merges them and,
with ``notify=True``, pushes ONE ``Notifier.notify_alert`` whose severity is the
worst issue found.

Cadence groups (``run_integrity(weekly=...)``):

``daily`` (every run)
  cadence        — last resolve / last scan recency            (heartbeat)
  state:<model>  — seed-maintained state stamp age in-season   (heartbeat)
  pinnacle       — one-request reachability probe, opt-in      (heartbeat)
  inplay         — LIVE rows logged at/after event start, or with an absurd
                   EV (default ≥25%) — the 2026-09-05 Brentford in-play row
                   (+35% EV logged 10 min after kickoff) and the Western
                   Michigan YES-alignment incident share this fingerprint
  model_missing  — a configured model newly MISSING on most of a sector's
                   moneyline rows vs its trailing baseline (the
                   ``ncaaf_efficiency_v2`` empty-state incident: silent for two
                   days because tests inject state and never load the file)
  match_rate     — a sector that fetched venue markets but matched ZERO Pinnacle
                   events, or whose match ratio collapsed vs its 14-day median
                   (Kalshi pagination truncation, UCL code map, tennis title
                   format — all looked like an off-season). Reads the
                   ``scan_sector_stats`` ledger written by ``logger.log_scan_stats``
  resolution     — unresolved, unvoided rows older than N days (resolver broke)
  close_capture  — resolved Kalshi rows missing a Kalshi close, and archive
                   snapshot freshness (a dead watch-closes / watch-listings
                   agent only shows up weeks later as an empty CLV column)
  clv            — LIVE-DEGRADING board groups                  (clv_monitor)
  passthrough    — a LIVE-mode board group whose blend divergence from sharp is
                   below 0.5pp: the sector is staking bankroll with no model
                   signal (tennis at 0.15–0.19pp on 2026-09-05)
  drawdown       — a live sector's flat ROI below a floor on n≥20 resolved rows
  launchd        — any ``com.evmax.*`` launchd agent whose last exit was non-zero

``weekly`` (add ``weekly=True``; Monday review)
  calibration    — value-audit ``calibration_bias`` verdicts (n≥30, one-signed,
                   ≥4pp)                                        (calibration-alert)
  gate           — a shadow group whose promotion gate has cleared
                   (PROMOTE-READY on the board) plus the explicitly watched
                   strategy sub-streams in ``GATE_WATCHES`` (WNBA spread lay
                   anchored-entry). Severity ``info`` — it is good news, but it
                   still needs a human to run ``cleanup shadow promote``

Design rules: pure ``_*_issues`` functions take already-fetched rows so they
are unit-testable without a DB; thin ``check_*`` wrappers do the SQL. A check
that cannot run (missing table, no archive, non-darwin) returns ``[]`` rather
than raising — a broken check must never mask the others.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import date, datetime, timedelta
from statistics import median
from typing import Optional

import structlog

from evmax.agents.cleanup import heartbeat as _hb

logger = structlog.get_logger(__name__)

# Severity ordering for the merged alert.
_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}

# Strategy sub-streams whose promotion gate is judged on a filtered CLV slice
# the promotion board cannot express (sources_token / side). Each entry is the
# kwargs for shadow.clv_stats plus a human label. Replaces the
# ``weekly-wnba-listings-robustness-check`` scheduled task.
GATE_WATCHES: list[dict] = [
    {
        "label": "wnba spread LAY anchored-entry (kalshi)",
        "category": "wnba", "market_type": "spread", "side": "lay",
        "venue": "kalshi", "sources_token": "anchored_entry", "max_staleness_h": 3.0,
        "promote_hint": "evmax cleanup shadow promote wnba (spread is a shadow_market_type — edit data/categories.yaml)",
    },
    # NFL spread is a shadow_market_type (no NFL betting history in this system);
    # judge promotion PER SIDE via Kalshi entry→close CLV, never pooled (the WNBA
    # spread-take lesson: laying ~breakeven while taking bled −2.2pp). These use
    # the regular scan rows, not the anchored-entry stream (that laddered lens is
    # WNBA-specific). They stay quiet until enough resolved NFL spread rows exist.
    {
        "label": "nfl spread LAY (kalshi)",
        "category": "nfl", "market_type": "spread", "side": "lay",
        "venue": "kalshi", "max_staleness_h": 3.0,
        "promote_hint": "evmax cleanup shadow clv nfl -m spread --side lay --venue kalshi; promote via data/categories.yaml shadow_market_types",
    },
    {
        "label": "nfl spread TAKE (kalshi)",
        "category": "nfl", "market_type": "spread", "side": "take",
        "venue": "kalshi", "max_staleness_h": 3.0,
        "promote_hint": "evmax cleanup shadow clv nfl -m spread --side take --venue kalshi; promote via data/categories.yaml shadow_market_types",
    },
]


def _issue(check: str, severity: str, detail: str) -> dict:
    return {"check": check, "severity": severity, "detail": detail}


# ---------------------------------------------------------------------------
# inplay — live rows logged at/after event start or with an absurd EV
# ---------------------------------------------------------------------------

def _inplay_issues(
    rows: list[dict], absurd_ev: float = 0.25, tipoff_ev: float = 0.10
) -> list[dict]:
    """``rows``: live, unvoided rows with keys sector, event_title, yes_team,
    ev_pct, minutes_to_tipoff, logged_at, market_id.

    Flags (a) ``minutes_to_tipoff == 0`` AND ev ≥ ``tipoff_ev`` — the scan ran
    at or after the start (the agent clamps negatives to 0) and the venue was
    already pricing the live game against a pre-match anchor; (b) ev ≥
    ``absurd_ev`` regardless — no genuine pre-match edge is that large; it is an
    alignment or in-play artifact.
    """
    issues: list[dict] = []
    for r in rows:
        ev = float(r.get("ev_pct") or 0.0)
        mtt = r.get("minutes_to_tipoff")
        at_tip = mtt is not None and int(mtt) <= 0
        if ev >= absurd_ev:
            why = f"EV {ev * 100:+.0f}% is implausible pre-match (alignment or in-play)"
        elif at_tip and ev >= tipoff_ev:
            why = f"logged at/after event start (minutes_to_tipoff=0) with EV {ev * 100:+.0f}%"
        else:
            continue
        issues.append(_issue(
            "inplay", "critical",
            f"{r.get('sector')} LIVE row {r.get('event_title')} / {r.get('yes_team')} "
            f"({r.get('market_id')}) — {why}; logged {r.get('logged_at')}. "
            "Void it (`evmax cleanup` void) and check the pre-kickoff gate.",
        ))
    return issues


def check_inplay(days: int = 1, absurd_ev: float = 0.25) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        rows = [dict(r) for r in conn.execute(
            """
            SELECT sector, event_title, yes_team, market_id, ev_pct,
                   minutes_to_tipoff, logged_at
            FROM ev_predictions
            WHERE mode = 'live' AND COALESCE(voided, 0) = 0
              AND logged_at >= ?
              AND (minutes_to_tipoff <= 0 OR ev_pct >= ?)
            """,
            (since, absurd_ev),
        ).fetchall()]
    return _inplay_issues(rows, absurd_ev=absurd_ev)


# ---------------------------------------------------------------------------
# model_missing — a configured model newly absent vs its trailing baseline
# ---------------------------------------------------------------------------

_NON_MODEL_SOURCE_TOKENS = {
    "sharp", "sharp(capped)", "injury", "late_news", "rest", "playoff",
    "advance_derived", "spread_dist", "total_dist", "no_side", "anchored_entry",
}


def _model_missing_issues(
    rows: list[dict],
    today: date,
    recent_days: int = 2,
    min_recent_rows: int = 5,
    recent_min_rate: float = 0.8,
    baseline_max_rate: float = 0.5,
) -> list[dict]:
    """``rows``: moneyline rows with sector, scan_date (YYYY-MM-DD), ``missing``
    (model names from model_diagnostics) and ``sources`` (model_sources tokens).

    Two signals per (sector, model), both requiring a regime CHANGE against the
    older baseline rows so structurally-absent models (tennis h2h fires on a
    minority of matches by design) stay silent:

    * diagnostics: missing-rate over the last ``recent_days`` ≥ ``recent_min_rate``
      while the baseline missing-rate ≤ ``baseline_max_rate``.
    * fire-rate: a model token present in model_sources on ≥ 50% of baseline
      rows that appears on ≤ 20% of recent rows. Independent of the diagnostics
      payload — the 2026-09-04 tennis_surface stale-stamp gating produced rows
      whose diagnostics did not mention the model at all.

    Needs ≥ ``min_recent_rows`` recent rows and a non-empty baseline.
    """
    cutoff = (today - timedelta(days=recent_days - 1)).isoformat()
    recent: dict[str, list[dict]] = {}
    base: dict[str, list[dict]] = {}
    for r in rows:
        bucket = recent if (r.get("scan_date") or "") >= cutoff else base
        bucket.setdefault(r["sector"], []).append(r)

    def _tokens(r: dict) -> set[str]:
        return {t for t in (r.get("sources") or []) if t and t not in _NON_MODEL_SOURCE_TOKENS}

    issues: list[dict] = []
    for sector, rrows in sorted(recent.items()):
        if len(rrows) < min_recent_rows:
            continue
        brows = base.get(sector, [])
        if not brows:
            continue  # no baseline → cannot tell a regime change from structure
        flagged: set[str] = set()
        models = {m for r in rrows for m in (r.get("missing") or [])}
        for model in sorted(models):
            r_rate = sum(model in (r.get("missing") or []) for r in rrows) / len(rrows)
            b_rate = sum(model in (r.get("missing") or []) for r in brows) / len(brows)
            if r_rate >= recent_min_rate and b_rate <= baseline_max_rate:
                flagged.add(model)
                issues.append(_issue(
                    "model_missing", "critical",
                    f"{sector}: `{model}` missing on {r_rate * 100:.0f}% of the last "
                    f"{recent_days}d moneyline rows (n={len(rrows)}) vs {b_rate * 100:.0f}% "
                    f"baseline (n={len(brows)}) — state file not loading? "
                    "(`evmax cleanup shadow show --why`)",
                ))
        base_tokens = {t for r in brows for t in _tokens(r)}
        for model in sorted(base_tokens - flagged):
            b_fire = sum(model in _tokens(r) for r in brows) / len(brows)
            r_fire = sum(model in _tokens(r) for r in rrows) / len(rrows)
            if b_fire >= 0.5 and r_fire <= 0.2:
                issues.append(_issue(
                    "model_missing", "critical",
                    f"{sector}: `{model}` fired on {b_fire * 100:.0f}% of baseline moneyline rows "
                    f"(n={len(brows)}) but only {r_fire * 100:.0f}% of the last {recent_days}d "
                    f"(n={len(rrows)}) — staleness guard or empty state? "
                    "(`evmax cleanup shadow show --why`)",
                ))
    return issues


def check_model_missing(
    today: Optional[date] = None, recent_days: int = 2, baseline_days: int = 14
) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    today = today or date.today()
    since = (today - timedelta(days=recent_days + baseline_days)).isoformat()
    rows: list[dict] = []
    with get_connection() as conn:
        for r in conn.execute(
            """
            SELECT sector, scan_date, model_diagnostics, model_sources
            FROM ev_predictions
            WHERE scan_date >= ? AND market_type = 'moneyline'
            """,
            (since,),
        ):
            try:
                diag = json.loads(r["model_diagnostics"]) or {} if r["model_diagnostics"] else {}
            except (TypeError, ValueError):
                diag = {}
            rows.append({
                "sector": r["sector"], "scan_date": r["scan_date"],
                "missing": list(diag.get("missing") or []),
                "sources": [t for t in (r["model_sources"] or "").split("+") if t],
            })
    return _model_missing_issues(rows, today, recent_days=recent_days)


# ---------------------------------------------------------------------------
# match_rate — fetched-but-matched-zero, or a collapsed match ratio
# ---------------------------------------------------------------------------

def _match_rate_issues(
    today_stats: dict[str, dict],
    baseline_daily: dict[str, list[dict]],
    min_fetched_zero: int = 5,
    min_fetched_ratio: int = 20,
    ratio_collapse: float = 0.25,
) -> list[dict]:
    """``today_stats``: sector → {fetched, matched} summed over today's cycles.
    ``baseline_daily``: sector → [{fetched, matched} per prior day].

    critical: today fetched ≥ ``min_fetched_zero`` markets but matched 0 while
    the baseline median matched > 0 (a first-ever-zero on a sector with history).
    warning: today's matched/fetched ratio < ``ratio_collapse`` × the baseline
    median ratio, on ≥ ``min_fetched_ratio`` fetched markets.
    """
    issues: list[dict] = []
    for sector, st in sorted(today_stats.items()):
        fetched, matched = int(st.get("fetched", 0)), int(st.get("matched", 0))
        hist = [d for d in baseline_daily.get(sector, []) if int(d.get("fetched", 0)) > 0]
        if not hist:
            continue
        base_matched = median(int(d.get("matched", 0)) for d in hist)
        base_ratio = median(int(d["matched"]) / int(d["fetched"]) for d in hist)
        if fetched >= min_fetched_zero and matched == 0 and base_matched > 0:
            issues.append(_issue(
                "match_rate", "critical",
                f"{sector}: fetched {fetched} venue markets today but matched 0 Pinnacle "
                f"events (14d median matched {base_matched:.0f}) — codes/pagination/"
                "title-format break? (`scripts/check_kalshi_series.py --probe`, "
                "`scripts/check_pinnacle_leagues.py`)",
            ))
            continue
        if fetched >= min_fetched_ratio and base_ratio > 0:
            ratio = matched / fetched
            if ratio < ratio_collapse * base_ratio:
                issues.append(_issue(
                    "match_rate", "warning",
                    f"{sector}: match ratio {ratio * 100:.0f}% today ({matched}/{fetched}) "
                    f"vs 14d median {base_ratio * 100:.0f}% — partial matching regression",
                ))
    return issues


def check_match_rate(today: Optional[date] = None, baseline_days: int = 14) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    today = today or date.today()
    since = (today - timedelta(days=baseline_days)).isoformat()
    today_stats: dict[str, dict] = {}
    baseline: dict[str, list[dict]] = {}
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT scan_date, sector,
                       SUM(markets_fetched) AS fetched, SUM(markets_matched) AS matched
                FROM scan_sector_stats
                WHERE scan_date >= ? AND error IS NULL
                GROUP BY scan_date, sector
                """,
                (since,),
            ).fetchall()
    except Exception as e:  # noqa: BLE001 — table may predate this check
        logger.info("integrity_match_rate_unavailable", error=str(e))
        return []
    for r in rows:
        d = {"fetched": r["fetched"] or 0, "matched": r["matched"] or 0}
        if r["scan_date"] == today.isoformat():
            today_stats[r["sector"]] = d
        else:
            baseline.setdefault(r["sector"], []).append(d)
    return _match_rate_issues(today_stats, baseline)


# ---------------------------------------------------------------------------
# resolution — unresolved rows past the resolver window
# ---------------------------------------------------------------------------

def _resolution_issues(counts: dict[str, int], threshold: int = 5, min_age_days: int = 3) -> list[dict]:
    return [
        _issue(
            "resolution", "warning",
            f"{sector}: {n} unresolved, unvoided rows with event_date > {min_age_days}d old "
            "— resolver may be failing for this sector (`evmax cleanup resolve --date`)",
        )
        for sector, n in sorted(counts.items()) if n >= threshold
    ]


def check_resolution(min_age_days: int = 3, lookback_days: int = 30, threshold: int = 5) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.sector, COUNT(*) AS n
            FROM ev_predictions p
            LEFT JOIN ev_outcomes o ON o.market_id = p.market_id
            WHERE o.outcome IS NULL AND COALESCE(p.voided, 0) = 0
              AND p.event_date < date('now', ?) AND p.event_date >= date('now', ?)
            GROUP BY p.sector
            """,
            (f"-{min_age_days} days", f"-{lookback_days} days"),
        ).fetchall()
    return _resolution_issues({r["sector"]: r["n"] for r in rows}, threshold, min_age_days)


# ---------------------------------------------------------------------------
# close_capture — Kalshi CLV coverage + archive snapshot freshness
# ---------------------------------------------------------------------------

def _close_capture_issues(
    coverage: dict[str, dict],
    archive_age_h: Optional[float],
    min_n: int = 10,
    max_null_frac: float = 0.30,
    max_archive_age_h: float = 24.0,
) -> list[dict]:
    """``coverage``: sector → {n, null} over resolved Kalshi rows in the window."""
    issues: list[dict] = []
    for sector, c in sorted(coverage.items()):
        n, null = int(c.get("n", 0)), int(c.get("null", 0))
        if n >= min_n and null / n > max_null_frac:
            issues.append(_issue(
                "close_capture", "warning",
                f"{sector}: {null}/{n} resolved Kalshi rows have no Kalshi close (CLV NULL) "
                "— watch-closes gap or backfill_clv not running",
            ))
    if archive_age_h is not None and archive_age_h > max_archive_age_h:
        issues.append(_issue(
            "close_capture", "warning",
            f"latest archived Kalshi snapshot is {archive_age_h:.0f}h old (> {max_archive_age_h:.0f}h) "
            "— com.evmax.watch-listings / watch-closes may be dead",
        ))
    return issues


def check_close_capture(days: int = 7) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    coverage: dict[str, dict] = {}
    with get_connection() as conn:
        for r in conn.execute(
            """
            SELECT p.sector, COUNT(*) AS n,
                   SUM(CASE WHEN p.kalshi_clv_pct IS NULL THEN 1 ELSE 0 END) AS null_n
            FROM ev_predictions p
            JOIN ev_outcomes o ON o.market_id = p.market_id
            WHERE o.outcome IS NOT NULL AND COALESCE(p.venue, 'kalshi') = 'kalshi'
              AND p.event_date >= date('now', ?)
            GROUP BY p.sector
            """,
            (f"-{days} days",),
        ):
            coverage[r["sector"]] = {"n": r["n"], "null": r["null_n"]}

    archive_age_h: Optional[float] = None
    try:
        from evmax.archiver import _get_connection as _archive_conn

        with _archive_conn() as aconn:
            last = aconn.execute(
                "SELECT MAX(fetched_at) AS m FROM archived_kalshi_markets"
            ).fetchone()["m"]
        if last:
            dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            archive_age_h = (now - dt).total_seconds() / 3600.0
    except Exception as e:  # noqa: BLE001 — no archive → skip freshness only
        logger.info("integrity_archive_unavailable", error=str(e))
    return _close_capture_issues(coverage, archive_age_h)


# ---------------------------------------------------------------------------
# board-derived: clv (LIVE-DEGRADING), passthrough (live), gate (PROMOTE-READY)
# ---------------------------------------------------------------------------

def _board_issues(board: list[dict], include_gates: bool) -> list[dict]:
    issues: list[dict] = []
    for r in board:
        verdict = r.get("verdict") or ""
        clv = r.get("clv") or {}
        tag = f"{r.get('sector')} {r.get('market_type')} [{r.get('venue')}]"
        if verdict == "LIVE-DEGRADING":
            issues.append(_issue(
                "clv", "warning",
                f"{tag} — live book bleeding CLV: mean {clv.get('mean_clv_pp', 0.0):+.2f}pp "
                f"over n={clv.get('n', 0)} ({(clv.get('frac_positive') or 0) * 100:.0f}% pos). "
                "Entry-timing issue — inspect the entry window, not the blend.",
            ))
        elif verdict == "SHARP-PASSTHROUGH" and r.get("mode") == "live":
            issues.append(_issue(
                "passthrough", "warning",
                f"{tag} — LIVE with blend divergence {(r.get('blend_divergence_pp') or 0.0):.2f}pp (<0.5): "
                "staking bankroll with no independent model signal; any EV is venue "
                "divergence. Fix seeding/models or demote.",
            ))
        elif include_gates and verdict == "PROMOTE-READY":
            issues.append(_issue(
                "gate", "info",
                f"{tag} — shadow gates cleared (clean n={r.get('n_clean_resolved', '?')}, CLV "
                f"{clv.get('mean_clv_pp', 0.0):+.2f}pp, {(clv.get('frac_positive') or 0) * 100:.0f}% pos). "
                f"Review then `evmax cleanup shadow promote {r.get('sector')}`.",
            ))
    return issues


def check_board(days: int = 30, staleness_h: Optional[float] = 3.0, include_gates: bool = False) -> list[dict]:
    from evmax.agents.cleanup.promotion_board import compute_promotion_board

    return _board_issues(compute_promotion_board(days=days, staleness_h=staleness_h), include_gates)


def _gate_watch_issues(results: list[tuple[dict, dict]]) -> list[dict]:
    """``results``: (watch spec, clv_stats dict) pairs. Only a CLEARED gate speaks."""
    issues: list[dict] = []
    for spec, st in results:
        if st.get("clears"):
            issues.append(_issue(
                "gate", "info",
                f"{spec['label']} — gate cleared: n={st.get('n')}, mean CLV "
                f"{st.get('mean_clv_pp', 0.0):+.2f}pp, {(st.get('frac_positive') or 0) * 100:.0f}% pos. "
                f"{spec.get('promote_hint', '')}".strip(),
            ))
    return issues


def check_gate_watches() -> list[dict]:
    from evmax.cli.commands.shadow import clv_stats

    results: list[tuple[dict, dict]] = []
    for spec in GATE_WATCHES:
        kwargs = {k: v for k, v in spec.items() if k not in ("label", "promote_hint")}
        try:
            results.append((spec, clv_stats(**kwargs)))
        except Exception as e:  # noqa: BLE001 — one broken watch must not mask the rest
            logger.warning("integrity_gate_watch_failed", label=spec["label"], error=str(e))
    return _gate_watch_issues(results)


# ---------------------------------------------------------------------------
# drawdown — live sector flat ROI floor
# ---------------------------------------------------------------------------

def _drawdown_issues(rows: list[dict], min_n: int = 20, roi_floor: float = -0.30, days: int = 30) -> list[dict]:
    """``rows``: sector → {n, wins, roi} where roi is flat-stake return per bet."""
    issues: list[dict] = []
    for r in rows:
        if int(r["n"]) >= min_n and float(r["roi"]) <= roi_floor:
            issues.append(_issue(
                "drawdown", "warning",
                f"{r['sector']}: live flat ROI {float(r['roi']) * 100:+.1f}% over n={r['n']} "
                f"resolved rows ({r['wins']} wins) in {days}d — below the {roi_floor * 100:.0f}% floor",
            ))
    return issues


def check_drawdown(days: int = 30, min_n: int = 20, roi_floor: float = -0.30) -> list[dict]:
    from evmax.agents.cleanup.db import get_connection

    with get_connection() as conn:
        rows = [dict(r) for r in conn.execute(
            """
            SELECT p.sector, COUNT(*) AS n, SUM(o.outcome = 1) AS wins,
                   AVG(CASE WHEN o.outcome = 1
                            THEN (1.0 - p.kalshi_yes_price) / p.kalshi_yes_price
                            ELSE -1.0 END) AS roi
            FROM ev_predictions p
            JOIN ev_outcomes o ON o.market_id = p.market_id
            WHERE p.mode = 'live' AND o.outcome IS NOT NULL
              AND COALESCE(p.voided, 0) = 0 AND p.kalshi_yes_price > 0
              AND p.event_date >= date('now', ?)
            GROUP BY p.sector
            """,
            (f"-{days} days",),
        ).fetchall()]
    return _drawdown_issues(rows, min_n=min_n, roi_floor=roi_floor, days=days)


# ---------------------------------------------------------------------------
# launchd — non-zero last exit on any com.evmax.* agent
# ---------------------------------------------------------------------------

def _launchd_issues(listing: str, prefix: str = "com.evmax.") -> list[dict]:
    """Parse ``launchctl list`` output (PID<TAB>Status<TAB>Label)."""
    issues: list[dict] = []
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[2].startswith(prefix):
            continue
        status = parts[1].strip()
        # '-' = currently running / never exited; negative = killed by a signal
        # (a KeepAlive daemon restarted with `launchctl kickstart -k` reads -15),
        # which is operator action, not a failure. Only a positive exit code is
        # a crashed run.
        if not status.isdigit() or status == "0":
            continue
        issues.append(_issue(
            "launchd", "warning",
            f"{parts[2]} last exited with status {status} — check "
            f"logs/launchd.{parts[2][len(prefix):]}.err",
        ))
    return issues


def check_launchd() -> list[dict]:
    if platform.system() != "Darwin":
        return []
    try:
        out = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except Exception as e:  # noqa: BLE001
        logger.info("integrity_launchd_unavailable", error=str(e))
        return []
    return _launchd_issues(out)


# ---------------------------------------------------------------------------
# calibration — value-audit bias verdicts (weekly)
# ---------------------------------------------------------------------------

def check_calibration(weeks: int = 8) -> list[dict]:
    from evmax.agents.cleanup.value_audit import compute_value_audit

    try:
        audit = compute_value_audit(weeks=weeks)
    except Exception as e:  # noqa: BLE001
        logger.warning("integrity_calibration_failed", error=str(e))
        return []
    issues: list[dict] = []
    for a in audit:
        v = a.get("verdict") or {}
        if v.get("tag") == "calibration_bias":
            issues.append(_issue(
                "calibration", "warning",
                f"{a.get('sector')}: consistent calibration bias ({v.get('reason', '')}) — "
                f"`evmax cleanup recalibrate --sector {a.get('sector')}` after confirming it persists",
            ))
    return issues


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

DAILY_CHECKS = (
    "cadence", "states", "inplay", "model_missing", "match_rate", "resolution",
    "close_capture", "board", "drawdown", "launchd",
)
WEEKLY_CHECKS = ("calibration", "gates")


def run_integrity(
    weekly: bool = False,
    check_pinnacle: bool = False,
    notify: bool = False,
    only: Optional[set[str]] = None,
    board_days: int = 30,
) -> dict:
    """Run every applicable check; optionally push ONE merged alert.

    Returns ``{"ok", "issues", "notified", "ran", "failed"}``. ``ok`` is True
    when no warning/critical issue was found (``info`` gate clearances do not
    make the sweep unhealthy). ``failed`` lists checks that raised — reported,
    never allowed to mask the others.
    """
    wanted = set(DAILY_CHECKS) | (set(WEEKLY_CHECKS) if weekly else set())
    if check_pinnacle:
        wanted.add("pinnacle")
    if only:
        wanted &= set(only)

    runners = {
        "cadence": lambda: _hb.check_cadence(),
        "states": lambda: _hb.check_seed_states(),
        "pinnacle": lambda: _hb.check_pinnacle(),
        "inplay": check_inplay,
        "model_missing": check_model_missing,
        "match_rate": check_match_rate,
        "resolution": check_resolution,
        "close_capture": check_close_capture,
        "board": lambda: check_board(days=board_days, include_gates=weekly),
        "drawdown": check_drawdown,
        "launchd": check_launchd,
        "calibration": check_calibration,
        "gates": check_gate_watches,
    }

    issues: list[dict] = []
    ran: list[str] = []
    failed: list[str] = []
    for name in [n for n in (*DAILY_CHECKS, "pinnacle", *WEEKLY_CHECKS) if n in wanted]:
        try:
            issues.extend(runners[name]())
            ran.append(name)
        except Exception as e:  # noqa: BLE001 — one broken check must not mask the rest
            logger.error("integrity_check_failed", check=name, error=str(e))
            failed.append(name)
            issues.append(_issue("integrity", "warning", f"check `{name}` crashed: {e}"))

    issues.sort(key=lambda i: -_SEVERITY_RANK.get(i["severity"], 1))
    actionable = [i for i in issues if i["severity"] != "info"]

    notified = False
    if issues and notify:
        from evmax.notifications import Notifier

        severity = max((i["severity"] for i in issues), key=lambda s: _SEVERITY_RANK.get(s, 1))
        title = f"integrity: {len(actionable)} issue(s)" + (
            f", {len(issues) - len(actionable)} gate clearance(s)" if len(issues) > len(actionable) else ""
        )
        message = "\n".join(f"• [{i['severity']}] {i['check']}: {i['detail']}" for i in issues)
        notified = Notifier.from_settings().notify_alert(title, message, severity=severity)
        logger.info("integrity_alert", n=len(issues), severity=severity, notified=notified)

    return {"ok": not actionable, "issues": issues, "notified": notified, "ran": ran, "failed": failed}
