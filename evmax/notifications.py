"""Push notification layer for +EV alerts.

Sends Slack and/or Discord webhook messages when high-EV opportunities are found.
Webhooks are configured via environment variables:
  SLACK_WEBHOOK_URL   — Slack incoming webhook URL
  DISCORD_WEBHOOK_URL — Discord webhook URL

Only bets with EV >= notification_min_ev_pct (default 5%) trigger notifications.

Example Slack message:
  🎯 *evmax* — 3 +EV plays found (soccer, nba)
  • Chelsea wins   +185 → +210  EV=+8.3%  $12.50
  • Lakers -4.5    -108 → +105  EV=+6.1%  $8.75
  • Real Madrid    +220 → +245  EV=+5.4%  $7.00
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

import structlog

from evmax.ev.odds_format import cents as _cents

if TYPE_CHECKING:
    from evmax.agents.coordinator import CycleResult

logger = structlog.get_logger(__name__)

# Delivery is retried with exponential backoff so a transient 429 / 5xx / network
# blip doesn't silently drop an alert. 4xx (other than 429) is a permanent
# failure — a bad/revoked webhook URL — and is not retried.
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 0.5  # 0.5s, 1.0s, 2.0s between attempts

_SEVERITY_EMOJI = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}


class Notifier:
    """Sends +EV alerts to configured Slack / Discord webhooks."""

    def __init__(self, slack_url: str | None = None, discord_url: str | None = None, min_ev_pct: float = 5.0) -> None:
        self._slack_url = slack_url
        self._discord_url = discord_url
        self._min_ev_pct = min_ev_pct

    @classmethod
    def from_settings(cls) -> "Notifier":
        from evmax.settings import get_settings
        s = get_settings()
        return cls(
            slack_url=s.slack_webhook_url,
            discord_url=s.discord_webhook_url,
            min_ev_pct=s.notification_min_ev_pct,
        )

    def is_configured(self) -> bool:
        return bool(self._slack_url or self._discord_url)

    def notify_cycle(self, result: "CycleResult") -> None:
        """Send notification for a completed scan cycle if high-EV bets found."""
        if not self.is_configured():
            return

        top_gaps = [g for g in result.top_gaps if g.ev_pct >= self._min_ev_pct / 100]
        if not top_gaps:
            return

        text = self._format_message(result, top_gaps)
        self._send(text)

    def _format_message(self, result, gaps: list) -> str:
        sectors = sorted({g.sector for g in gaps})
        header = f"*evmax* — {len(gaps)} +EV play{'s' if len(gaps) > 1 else ''} ({', '.join(sectors)})"

        lines = [header]
        for g in gaps[:10]:  # cap at 10 lines
            k_odds = _cents(g.kalshi_yes_price)
            true_odds = _cents(g.blended_true_prob)
            stake = result.stake_for(g)
            lines.append(
                f"• [{g.sector.upper()}] {g.display_label[:25]}  "
                f"{k_odds} → {true_odds}  EV={g.ev_pct*100:+.1f}%  ${stake:.2f}"
            )

        if len(gaps) > 10:
            lines.append(f"  _(+{len(gaps)-10} more)_")

        return "\n".join(lines)

    def send_text(self, text: str) -> None:
        """Send an arbitrary alert to all configured webhooks (no-op when
        none are configured). Used by callers with their own formatting,
        e.g. ``evmax arb scan --notify``."""
        if self.is_configured():
            self._send(text)

    def notify_alert(self, title: str, message: str, *, severity: str = "warning") -> bool:
        """Send an OPERATIONAL alert (data source down, model degrading, a
        missed scheduled run) — distinct from an EV-cycle notification.

        Not gated on ``min_ev``; prefixed with a severity marker; and returns
        whether delivery actually succeeded so an ops caller (e.g. the S4
        heartbeat) can react to a dead webhook rather than assume it got out.
        Returns ``False`` when no webhook is configured.
        """
        if not self.is_configured():
            return False
        emoji = _SEVERITY_EMOJI.get(severity, "⚠️")
        text = f"{emoji} *evmax {severity}* — {title}\n{message}"
        return self._send(text)

    def _send(self, text: str) -> bool:
        """POST text to all configured webhooks. Returns True only if every
        configured webhook accepted the message."""
        ok = True
        if self._slack_url:
            ok = self._post(self._slack_url, {"text": text}) and ok
        if self._discord_url:
            ok = self._post(self._discord_url, {"content": text}) and ok
        return ok

    def _post(self, url: str, payload: dict) -> bool:
        """POST with exponential-backoff retry. Returns True on delivery.

        Retries transient failures (network error, 429, 5xx); a non-429 4xx is
        treated as permanent (revoked/malformed webhook) and fails fast."""
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status in (200, 204):
                        return True
                    logger.warning(
                        "notification_non_ok", url=url[:40], status=resp.status,
                        attempt=attempt,
                    )
            except urllib.error.HTTPError as e:
                logger.warning(
                    "notification_http_error", url=url[:40], status=e.code,
                    attempt=attempt,
                )
                if not (e.code >= 500 or e.code == 429):
                    return False  # permanent — do not retry
            except Exception as e:  # noqa: BLE001 — network/URL error, retry
                logger.warning(
                    "notification_send_failed", url=url[:40], error=str(e),
                    attempt=attempt,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
        return False
