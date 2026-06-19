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
import urllib.request
from typing import TYPE_CHECKING

import structlog

from evmax.ev.odds_format import american_odds as _american

if TYPE_CHECKING:
    from evmax.agents.coordinator import CycleResult

logger = structlog.get_logger(__name__)


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
            k_odds = _american(g.kalshi_yes_price)
            true_odds = _american(g.blended_true_prob)
            stake = result.stake_for(g)
            lines.append(
                f"• [{g.sector.upper()}] {g.display_label[:25]}  "
                f"{k_odds} → {true_odds}  EV={g.ev_pct*100:+.1f}%  ${stake:.2f}"
            )

        if len(gaps) > 10:
            lines.append(f"  _(+{len(gaps)-10} more)_")

        return "\n".join(lines)

    def _send(self, text: str) -> None:
        """POST text to all configured webhooks."""
        if self._slack_url:
            self._post(self._slack_url, {"text": text})
        if self._discord_url:
            self._post(self._discord_url, {"content": text})

    def _post(self, url: str, payload: dict) -> None:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status not in (200, 204):
                    logger.warning("notification_non_ok", url=url[:40], status=resp.status)
        except Exception as e:
            logger.warning("notification_send_failed", url=url[:40], error=str(e))
