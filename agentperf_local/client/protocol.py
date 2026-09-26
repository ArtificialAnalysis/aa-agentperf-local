"""Define the shared streaming client contract."""

import asyncio
from dataclasses import dataclass
from typing import Protocol

from agentperf_local.client.request import CompletionRequest


class CompletionError(RuntimeError):
    """Report a normalized client or endpoint failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass(frozen=True, slots=True)
class RawRead:
    """Hold one timestamped network read."""

    timestamp: float
    data: bytes


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Hold raw reads from one completed or aborted request."""

    reads: tuple[RawRead, ...]
    aborted: bool


class CompletionClient(Protocol):
    """Stream completion requests and return raw reads."""

    async def complete(
        self,
        request: CompletionRequest,
        abort: asyncio.Event | None = None,
    ) -> CompletionResult:
        """Complete one request."""
        ...

    async def close(self) -> None:
        """Close network resources."""
        ...
