"""Provide the pure-Python raw streaming client."""

import asyncio
from contextlib import suppress
from time import perf_counter

import httpx
import orjson

from agentperf_local.client.endpoint import COMPLETION_PATH, MEASURED_USER_AGENT, normalize_base_url
from agentperf_local.client.protocol import CompletionError, CompletionResult, RawRead
from agentperf_local.client.request import CompletionRequest

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_CONNECTIONS = 32
ERROR_BODY_LIMIT_BYTES = 512


async def _collect_reads(response: httpx.Response, reads: list[RawRead]) -> None:
    """Collect response bytes without parsing them."""
    async for data in response.aiter_raw():
        reads.append(RawRead(timestamp=perf_counter(), data=data))


async def _collect_until_aborted(
    response: httpx.Response,
    reads: list[RawRead],
    abort: asyncio.Event | None,
) -> bool:
    """Race the raw reader against an optional abort event."""
    reader_task = asyncio.create_task(_collect_reads(response, reads))
    abort_task: asyncio.Task[bool] | None = None
    try:
        if abort is None:
            await reader_task
            return False
        abort_task = asyncio.create_task(abort.wait())
        done, _ = await asyncio.wait(
            {reader_task, abort_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        aborted = abort_task in done and not reader_task.done()
        if aborted:
            reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await reader_task
            return True
        await reader_task
        return False
    finally:
        if not reader_task.done():
            reader_task.cancel()
        if abort_task is not None:
            abort_task.cancel()
            with suppress(asyncio.CancelledError):
                await abort_task


def _status_error(response: httpx.Response) -> CompletionError:
    body = response.content[:ERROR_BODY_LIMIT_BYTES].decode("utf-8", errors="replace")
    return CompletionError(
        f"endpoint returned HTTP {response.status_code}: {body}",
        status_code=response.status_code,
        response_body=body,
    )


class PythonStreamingClient:
    """Collect timestamped response bytes with httpx."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_connections <= 0:
            raise ValueError("max_connections must be greater than zero")
        self._endpoint = f"{normalize_base_url(base_url)}{COMPLETION_PATH}"
        headers = {
            "accept": "text/event-stream",
            # aiter_raw() skips decompression, so never let the server compress.
            "accept-encoding": "identity",
            "content-type": "application/json",
            "user-agent": MEASURED_USER_AGENT,
        }
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=None,
        )
        timeout = httpx.Timeout(timeout_seconds)
        self._client = httpx.AsyncClient(
            headers=headers,
            http2=False,
            limits=limits,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        )

    async def complete(
        self,
        request: CompletionRequest,
        abort: asyncio.Event | None = None,
    ) -> CompletionResult:
        """Stream one response and retain raw reads."""
        reads: list[RawRead] = []
        try:
            async with self._client.stream(
                "POST",
                self._endpoint,
                content=orjson.dumps(request.body()),
                headers=request.headers(),
            ) as response:
                if not response.is_success:
                    await response.aread()
                    raise _status_error(response)
                aborted = await _collect_until_aborted(response, reads, abort)
        except CompletionError:
            raise
        except httpx.HTTPError as error:
            raise CompletionError(f"request failed: {error}") from error
        return CompletionResult(reads=tuple(reads), aborted=aborted)

    async def close(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()
