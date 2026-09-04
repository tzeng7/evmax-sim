"""Bot-token REST transport for Discord — post messages/embeds to a channel.

Uses only the standard library (``urllib``) so the scan feed can run from any
evmax process (CLI scan, scheduled task, dashboard) without the gateway
library. ``discord.py`` is needed only for the slash-command bot
(:mod:`evmax.discord_bot.bot`).

Delivery contract mirrors :mod:`evmax.notifications`: transient failures
(network, 5xx, 429) are retried with backoff — a 429 honours Discord's
``retry_after`` — while any other 4xx (bad token, unknown channel, missing
permission) is permanent and fails fast. Every public method returns ``True``
only when Discord accepted every message.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
USER_AGENT = "DiscordBot (https://github.com/tzeng7/evmax, 0.1)"

# Discord hard limits (https://discord.com/developers/docs/resources/message).
MESSAGE_CONTENT_MAX = 2000
MESSAGE_EMBEDS_MAX = 10
MESSAGE_EMBED_CHARS_MAX = 6000   # summed over every embed in one message
EMBED_TITLE_MAX = 256
EMBED_DESCRIPTION_MAX = 4096
EMBED_FOOTER_MAX = 2048
EMBED_FIELD_NAME_MAX = 256
EMBED_FIELD_VALUE_MAX = 1024
EMBED_FIELDS_MAX = 25

_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 0.5      # 0.5s, 1.0s, 2.0s between attempts
_RATE_LIMIT_SLEEP_CAP_S = 5.0


def embed_char_count(embed: dict[str, Any]) -> int:
    """Characters Discord counts toward the 6000/message embed budget."""
    n = len(embed.get("title") or "") + len(embed.get("description") or "")
    n += len((embed.get("footer") or {}).get("text") or "")
    n += len((embed.get("author") or {}).get("name") or "")
    for f in embed.get("fields") or []:
        n += len(f.get("name") or "") + len(f.get("value") or "")
    return n


def batch_embeds(embeds: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Greedily pack embeds into messages that respect the per-message caps
    (≤10 embeds, ≤6000 summed characters). Order is preserved. An embed that
    alone exceeds the budget is sent on its own (Discord will reject it — the
    builders in :mod:`evmax.discord_bot.embeds` never produce one)."""
    batches: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_chars = 0
    for e in embeds:
        n = embed_char_count(e)
        if cur and (len(cur) >= MESSAGE_EMBEDS_MAX or cur_chars + n > MESSAGE_EMBED_CHARS_MAX):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(e)
        cur_chars += n
    if cur:
        batches.append(cur)
    return batches


class DiscordBotClient:
    """Posts as a bot user (``Authorization: Bot``) to a channel and/or a
    user's DMs. Every message goes to EVERY configured target; a DM target is
    resolved once (``POST /users/@me/channels``) and cached."""

    def __init__(
        self,
        token: str,
        channel_id: str | int = "",
        *,
        dm_user_id: str | int = "",
        api_base: str = DISCORD_API_BASE,
    ) -> None:
        self._token = (token or "").strip()
        self._channel_id = str(channel_id or "").strip()
        self._dm_user_id = str(dm_user_id or "").strip()
        self._dm_channel_id: Optional[str] = None
        self._api_base = api_base.rstrip("/")

    @classmethod
    def from_settings(cls) -> Optional["DiscordBotClient"]:
        """A client from ``.env`` (``DISCORD_BOT_TOKEN`` + ``DISCORD_CHANNEL_ID``
        and/or ``DISCORD_DM_USER_ID``), or ``None`` when unconfigured."""
        from evmax.settings import get_settings

        s = get_settings()
        if not s.discord_bot_configured:
            return None
        return cls(s.discord_bot_token, s.discord_channel_id, dm_user_id=s.discord_dm_user_id)

    @property
    def channel_id(self) -> str:
        return self._channel_id

    @property
    def dm_user_id(self) -> str:
        return self._dm_user_id

    def describe_targets(self) -> str:
        parts = []
        if self._channel_id:
            parts.append(f"channel {self._channel_id}")
        if self._dm_user_id:
            parts.append(f"DM to user {self._dm_user_id}")
        return " + ".join(parts) or "—"

    def is_configured(self) -> bool:
        return bool(self._token and (self._channel_id or self._dm_user_id))

    def _dm_channel(self) -> Optional[str]:
        """The DM channel for ``dm_user_id`` (created/opened via the API on
        first use, then cached). None when unset or the API refuses — e.g. the
        bot shares no server with the user, or the user blocks server DMs."""
        if not self._dm_user_id:
            return None
        if self._dm_channel_id:
            return self._dm_channel_id
        status, body = self._request(
            "POST", "/users/@me/channels", {"recipient_id": self._dm_user_id},
        )
        if status is not None and 200 <= status < 300 and body and body.get("id"):
            self._dm_channel_id = str(body["id"])
            return self._dm_channel_id
        logger.warning("discord_bot_dm_channel_failed", user=self._dm_user_id, status=status)
        return None

    def _targets(self, channel_id: Optional[str]) -> list[str]:
        if channel_id:
            return [channel_id]
        out = [self._channel_id] if self._channel_id else []
        dm = self._dm_channel()
        if dm:
            out.append(dm)
        return out

    # ------------------------------------------------------------------
    # Public posting API
    # ------------------------------------------------------------------

    def post(
        self,
        content: Optional[str] = None,
        embeds: Optional[list[dict[str, Any]]] = None,
        *,
        channel_id: Optional[str] = None,
    ) -> bool:
        """Send ONE message to every configured target (or just ``channel_id``
        when given). ``content`` is truncated to the 2000-char cap;
        ``embeds`` must already fit one message (use :meth:`post_embeds` to
        batch). Returns True when Discord accepted it."""
        if not self.is_configured():
            return False
        if not content and not embeds:
            return False
        payload: dict[str, Any] = {}
        if content:
            payload["content"] = content[:MESSAGE_CONTENT_MAX]
        if embeds:
            payload["embeds"] = embeds
        targets = self._targets(channel_id)
        if not targets:
            return False
        ok = True
        for target in targets:
            status, _body = self._request("POST", f"/channels/{target}/messages", payload)
            ok = (status is not None and 200 <= status < 300) and ok
        return ok

    def post_embeds(
        self,
        embeds: list[dict[str, Any]],
        *,
        content: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> bool:
        """Send embeds as as many messages as the caps require (in order).
        ``content`` rides on the first message only. Stops at the first failed
        message so a broken channel doesn't spam retries; returns True only if
        every message was accepted."""
        if not embeds:
            return self.post(content, channel_id=channel_id) if content else False
        ok = True
        for i, batch in enumerate(batch_embeds(embeds)):
            sent = self.post(content if i == 0 else None, batch, channel_id=channel_id)
            if not sent:
                ok = False
                break
        return ok

    def post_text(self, text: str, *, channel_id: Optional[str] = None) -> bool:
        """Send plain text, split across messages at the 2000-char cap on
        line boundaries. Returns True only if every chunk was accepted."""
        chunks = _split_text(text, MESSAGE_CONTENT_MAX)
        ok = True
        for c in chunks:
            if not self.post(c, channel_id=channel_id):
                ok = False
                break
        return ok

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _request(
        self, method: str, path: str, payload: Optional[dict[str, Any]]
    ) -> tuple[Optional[int], Optional[dict[str, Any]]]:
        """One Discord REST call with retry. Returns ``(status, json_body)``;
        status is ``None`` when every attempt failed at the network layer."""
        url = f"{self._api_base}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Authorization": f"Bot {self._token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        last_status: Optional[int] = None
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = _read_json(resp)
                    if 200 <= resp.status < 300:
                        return resp.status, body
                    last_status = resp.status
                    logger.warning(
                        "discord_bot_non_ok", path=path, status=resp.status, attempt=attempt,
                    )
            except urllib.error.HTTPError as e:
                last_status = e.code
                body = _read_json(e)
                if e.code == 429:
                    wait = _retry_after(body, e.headers)
                    logger.warning(
                        "discord_bot_rate_limited", path=path, retry_after=wait, attempt=attempt,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(min(wait, _RATE_LIMIT_SLEEP_CAP_S))
                    continue
                logger.warning(
                    "discord_bot_http_error", path=path, status=e.code, attempt=attempt,
                    detail=(body or {}).get("message") if isinstance(body, dict) else None,
                )
                if e.code < 500:
                    return e.code, body  # permanent — bad token / channel / perms
            except Exception as e:  # noqa: BLE001 — network/URL error, retry
                logger.warning(
                    "discord_bot_send_failed", path=path, error=str(e), attempt=attempt,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
        return last_status, None


def _read_json(resp: Any) -> Optional[dict[str, Any]]:
    try:
        raw = resp.read()
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, AttributeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _retry_after(body: Optional[dict[str, Any]], headers: Any) -> float:
    """Seconds to wait after a 429: JSON ``retry_after`` (seconds) first, then
    the ``Retry-After`` header, else one backoff step."""
    if isinstance(body, dict):
        try:
            return max(0.0, float(body.get("retry_after", 0.0)))
        except (TypeError, ValueError):
            pass
    try:
        hdr = headers.get("Retry-After") if headers is not None else None
        if hdr:
            return max(0.0, float(hdr))
    except (TypeError, ValueError, AttributeError):
        pass
    return _RETRY_BACKOFF_S


def _split_text(text: str, limit: int) -> list[str]:
    """Split ``text`` into ≤``limit``-char chunks, preferring line boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:  # a single over-long line: hard split
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur += line
    if cur:
        chunks.append(cur)
    return chunks
