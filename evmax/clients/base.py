"""Base async HTTP client with retry logic and concurrency limiting."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Retry on network errors and HTTP 429/5xx responses."""
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class BaseAPIClient:
    """
    Async httpx client with:
      - Configurable concurrency semaphore
      - Exponential backoff retries on transient errors
      - Structured logging of requests/responses
    """

    def __init__(
        self,
        base_url: str,
        concurrency: int = 5,
        timeout: float = 10.0,
        max_retries: int = 3,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout = httpx.Timeout(timeout)
        self._max_retries = max_retries
        self._headers = headers or {}
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "BaseAPIClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            headers=self._headers,
            http2=True,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Perform a GET request with retry and semaphore.
        Returns parsed JSON response.
        """
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=2, min=2, max=30),
                reraise=True,
            ):
                with attempt:
                    if self._client is None:
                        raise RuntimeError("Client not initialised — use async context manager")
                    logger.debug("http_get", path=path, params=params)
                    response = await self._client.get(path, params=params)
                    response.raise_for_status()
                    return response.json()

        return {}  # unreachable but satisfies type checker

    async def _post(
        self,
        path: str,
        json: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Perform a POST request with retry."""
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=2, min=2, max=30),
                reraise=True,
            ):
                with attempt:
                    if self._client is None:
                        raise RuntimeError("Client not initialised — use async context manager")
                    logger.debug("http_post", path=path)
                    response = await self._client.post(path, json=json)
                    response.raise_for_status()
                    return response.json()

        return {}
