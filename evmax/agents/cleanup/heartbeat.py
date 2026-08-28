"""Pipeline heartbeat — a dead-man's-switch for silent operational failures.

The evmax daily cadence fails *silently*: the Claude-desktop-app scheduled tasks
don't fire while the app is closed, ESPN 403s are swallowed, and a reseed that
403s just freezes model state (the UFC ``evmax-ufc-seed`` regression that stuck
``ufc_rating_state.json`` at 2026-08-01 for ~4 weeks). None of these page anyone
— they only show up as *absent* rows. This module turns absence into a push.

Two robust checks (no fragile per-file introspection):

1. **Cadence recency** (the #1 risk): how long since the last resolve / scan
   wrote to ``predictions.db``. A stale timestamp means the whole cadence
   stopped (app closed, or resolve erroring every run).

2. **Seed-state freshness**: for the seed-maintained states that carry a
   top-level ``last_updated`` stamp and are refreshed by the *weekly* reseed
   (not the daily resolve hook), a stamp older than ``stale_days`` means the
   reseed is silently failing. Only ``ufc_rating`` and ``tennis_surface`` carry
   a top-level stamp today; the elo/form/efficiency states use per-sector stamps
   and are fed daily by the resolve hook, so they're covered by check 1.

Uses the S2 ``Notifier.notify_alert`` primitive so a dead webhook is itself
reported (delivery result is returned), not silently swallowed.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[3] / "data" / "models"

# Seed-maintained states with a readable top-level `last_updated` stamp, mapped
# to the sector whose in-season status gates the check. Extend as more states
# grow a top-level stamp.
_SEED_STATE_CHECKS: list[dict] = [
    {"file": "ufc_rating_state.json", "sector": "ufc", "label": "ufc_rating"},
    {"file": "tennis_surface_state.json", "sector": "tennis", "label": "tennis_surface"},
]


def _age_hours(iso: Optional[str], now: datetime) -> Optional[float]:
    """Hours between an ISO timestamp/date string and ``now`` (naive compare)."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.strip().replace("T", " "))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return (now - dt).total_seconds() / 3600.0


def _cadence_issues(
    last_resolve_iso: Optional[str],
    last_scan_date: Optional[str],
    now: datetime,
    max_resolve_age_h: float,
    max_scan_age_days: int,
) -> list[dict]:
    """Pure cadence-recency logic (no DB) so it is unit-testable."""
    issues: list[dict] = []

    if not last_resolve_iso:
        issues.append({"check": "resolve", "severity": "critical",
                       "detail": "no resolved outcomes on record"})
    else:
        age = _age_hours(last_resolve_iso, now)
        if age is not None and age > max_resolve_age_h:
            issues.append({
                "check": "resolve", "severity": "critical",
                "detail": (f"last resolve {age:.0f}h ago (> {max_resolve_age_h:.0f}h) — "
                           "daily cadence may have stopped"),
            })

    if not last_scan_date:
        issues.append({"check": "scan", "severity": "warning",
                       "detail": "no scans on record"})
    else:
        try:
            scan_d = date.fromisoformat(last_scan_date[:10])
        except ValueError:
            return issues
        days_ago = (now.date() - scan_d).days
        if days_ago > max_scan_age_days:
            issues.append({
                "check": "scan", "severity": "warning",
                "detail": (f"last scan {last_scan_date} ({days_ago}d ago > "
                           f"{max_scan_age_days}d) — scans may have stopped"),
            })
    return issues


def check_cadence(
    now: Optional[datetime] = None,
    max_resolve_age_h: float = 36.0,
    max_scan_age_days: int = 1,
) -> list[dict]:
    """Read the last resolve/scan recency from predictions.db and flag staleness."""
    now = now or datetime.now()
    from evmax.agents.cleanup.db import get_connection

    with get_connection() as conn:
        last_resolve = conn.execute(
            "SELECT MAX(resolved_at) AS m FROM ev_outcomes"
        ).fetchone()["m"]
        last_scan = conn.execute(
            "SELECT MAX(scan_date) AS m FROM ev_predictions"
        ).fetchone()["m"]
    return _cadence_issues(
        last_resolve, last_scan, now, max_resolve_age_h, max_scan_age_days
    )


def _seed_state_issue(
    label: str,
    last_updated: Optional[str],
    now: datetime,
    stale_days: int,
    in_season: bool,
) -> Optional[dict]:
    """Pure seed-state freshness logic. Off-season states are never flagged."""
    if not in_season:
        return None
    age = _age_hours(last_updated, now)
    if age is None:
        if last_updated is None:
            return {"check": f"state:{label}", "severity": "warning",
                    "detail": f"{label} state has no last_updated stamp"}
        return None
    days = age / 24.0
    if days > stale_days:
        return {
            "check": f"state:{label}", "severity": "warning",
            "detail": (f"{label} state last updated {last_updated} "
                       f"({days:.0f}d ago > {stale_days}d) — weekly reseed may be "
                       "silently failing"),
        }
    return None


def check_seed_states(
    now: Optional[datetime] = None,
    stale_days: int = 10,
    today: Optional[date] = None,
) -> list[dict]:
    """Flag seed-maintained model states whose top-level stamp is stale in-season."""
    now = now or datetime.now()
    today = today or now.date()
    from evmax.categories import is_in_season

    issues: list[dict] = []
    for spec in _SEED_STATE_CHECKS:
        path = _MODELS_DIR / spec["file"]
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # missing/unreadable state — not this check's alarm
        try:
            in_season = is_in_season(spec["sector"], today)
        except Exception:  # noqa: BLE001 — unknown sector → treat as in-season
            in_season = True
        issue = _seed_state_issue(
            spec["label"], data.get("last_updated"), now, stale_days, in_season
        )
        if issue:
            issues.append(issue)
    return issues


def _pinnacle_issue(probe_result: dict) -> Optional[dict]:
    """Pure decision for a Pinnacle probe result."""
    if probe_result.get("ok"):
        return None
    reason = probe_result.get("reason", "error")
    status = probe_result.get("status")
    return {
        "check": "pinnacle", "severity": "critical",
        "detail": (f"Pinnacle unreachable (reason={reason}, status={status}) — "
                   "the sole sharp anchor is down, no new EV can be discovered"),
    }


def check_pinnacle() -> list[dict]:
    """Probe Pinnacle reachability (one request) and flag it if down.

    Network I/O — opt-in only (keeps the default heartbeat offline/fast).
    """
    import asyncio

    from evmax.clients.esports_pinnacle import PinnacleGuestClient

    async def _run() -> dict:
        async with PinnacleGuestClient() as client:
            return await client.probe()

    try:
        result = asyncio.run(_run())
    except Exception as e:  # noqa: BLE001 — a probe crash is itself the signal
        result = {"ok": False, "status": None, "reason": f"probe_error:{e}"}
    issue = _pinnacle_issue(result)
    return [issue] if issue else []


def run_heartbeat(
    now: Optional[datetime] = None,
    max_resolve_age_h: float = 36.0,
    max_scan_age_days: int = 1,
    state_stale_days: int = 10,
    check_pinnacle_reachability: bool = False,
    notify: bool = False,
) -> dict:
    """Run all checks; if ``notify`` and anything is wrong, push one alert.

    Returns ``{"ok": bool, "issues": [...], "notified": bool}``. Severity is
    ``critical`` when the resolve cadence has stopped or Pinnacle is down, else
    ``warning``. ``check_pinnacle_reachability`` adds a live network probe.
    """
    now = now or datetime.now()
    issues = check_cadence(
        now=now, max_resolve_age_h=max_resolve_age_h, max_scan_age_days=max_scan_age_days
    ) + check_seed_states(now=now, stale_days=state_stale_days)
    if check_pinnacle_reachability:
        issues = issues + check_pinnacle()

    notified = False
    if issues and notify:
        from evmax.notifications import Notifier

        severity = "critical" if any(i["severity"] == "critical" for i in issues) else "warning"
        title = f"{len(issues)} pipeline health issue(s)"
        message = "\n".join(f"• [{i['severity']}] {i['detail']}" for i in issues)
        notified = Notifier.from_settings().notify_alert(title, message, severity=severity)
        logger.info("heartbeat_alert", n=len(issues), severity=severity, notified=notified)

    return {"ok": not issues, "issues": issues, "notified": notified}
