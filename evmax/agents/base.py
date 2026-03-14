"""Agent base classes and message bus for inter-agent communication.

Architecture:
  Agent       — abstract unit of work with request/response interface
  AgentBus    — async pub/sub message bus agents use to broadcast results
  AgentMessage— typed envelope for all bus traffic

Agents publish their results to topic channels so downstream agents can
react without tight coupling.  E.g. KalshiOddsAgent publishes to
"odds.kalshi.nba" and EVGapAgent subscribes to that topic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)

Handler = Callable[["AgentMessage"], Awaitable[None]]


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """Message passed between agents via the AgentBus."""

    topic: str          # dot-separated topic, e.g. "odds.kalshi.nba"
    payload: Any        # arbitrary data (list, dict, dataclass, …)
    sender: str         # agent name that published
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None  # ties request → response chain


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

@dataclass
class AgentRequest:
    """Standardized input to any agent."""

    sector: str                          # e.g. "nba", "soccer"
    params: dict = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class AgentResponse:
    """Standardized output from any agent."""

    agent_name: str
    sector: str
    data: Any
    status: str = "ok"              # "ok" | "error" | "partial"
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class AgentBus:
    """
    Async pub/sub message bus.

    Agents call `bus.subscribe(topic, handler)` at startup.
    Any agent can then call `bus.publish(message)` and all matching
    handlers are invoked concurrently via asyncio.gather.

    Topic matching:
      "odds.kalshi.nba"   — exact match only
      "*"                 — wildcard, receives every message
    """

    def __init__(self) -> None:
        import asyncio
        from collections import defaultdict
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._wildcard: list[Handler] = []
        self.log = structlog.get_logger(__name__)

    def subscribe(self, topic: str, handler: Handler) -> None:
        if topic == "*":
            self._wildcard.append(handler)
        else:
            self._subscribers[topic].append(handler)

    async def publish(self, message: AgentMessage) -> None:
        import asyncio
        handlers = list(self._subscribers.get(message.topic, [])) + self._wildcard
        if not handlers:
            return
        results = await asyncio.gather(*(h(message) for h in handlers), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                self.log.warning("bus_handler_error", topic=message.topic, error=str(r))
        self.log.debug("bus_published", topic=message.topic, sender=message.sender)

    def clear(self) -> None:
        self._subscribers.clear()
        self._wildcard.clear()


# ---------------------------------------------------------------------------
# Agent ABC
# ---------------------------------------------------------------------------

class Agent(ABC):
    """
    Base class for all evmax agents.

    Subclasses implement `run(request) -> AgentResponse`.

    The `__call__` shim handles timing, status tracking, error capture,
    and bus publishing automatically — agents just implement `run()`.
    """

    name: str           # unique agent identifier, set as class attribute
    description: str    # human-readable description

    def __init__(self) -> None:
        self.status = AgentStatus.IDLE
        self._bus: Optional[AgentBus] = None
        self.log = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Bus integration
    # ------------------------------------------------------------------

    def attach_bus(self, bus: AgentBus) -> None:
        """Attach this agent to a message bus for publishing results."""
        self._bus = bus

    async def publish(
        self,
        topic: str,
        payload: Any,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Publish a message to the attached bus (no-op if no bus attached)."""
        if self._bus is not None:
            msg = AgentMessage(
                topic=topic,
                payload=payload,
                sender=self.name,
                correlation_id=correlation_id,
            )
            await self._bus.publish(msg)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, request: AgentRequest) -> AgentResponse:
        """Execute the agent's task and return a response."""
        ...

    async def __call__(self, request: AgentRequest) -> AgentResponse:
        """Invoke the agent, wrapping run() with timing + error handling."""
        self.status = AgentStatus.RUNNING
        t0 = time.perf_counter()
        try:
            response = await self.run(request)
            response.latency_ms = (time.perf_counter() - t0) * 1000
            self.status = AgentStatus.IDLE
            self.log.info(
                "agent_done",
                agent=self.name,
                sector=request.sector,
                latency_ms=round(response.latency_ms, 1),
                status=response.status,
            )
            return response
        except Exception as exc:
            self.status = AgentStatus.ERROR
            latency = (time.perf_counter() - t0) * 1000
            self.log.error("agent_error", agent=self.name, error=str(exc))
            return AgentResponse(
                agent_name=self.name,
                sector=request.sector,
                data=None,
                status="error",
                error=str(exc),
                latency_ms=latency,
            )
