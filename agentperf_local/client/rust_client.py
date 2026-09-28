"""Wrap the optional Rust raw streaming extension."""

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
from importlib import import_module
from time import perf_counter
from types import ModuleType
from typing import Protocol, cast

import orjson
from pydantic import SecretStr

from agentperf_local.client.endpoint import normalize_base_url
from agentperf_local.client.protocol import CompletionError, CompletionResult, RawRead
from agentperf_local.client.python_client import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT_SECONDS
from agentperf_local.client.request import CompletionRequest

RUSTCORE_COMPATIBILITY_PREFIX = "0.2."
RUST_STATUS_PREFIX = "status "


class RustRequestHandle(Protocol):
    """Describe one request owned by the extension."""

    def wait(self) -> Awaitable[tuple[list[tuple[float, bytes]], bool]]:
        """Wait until the request closes."""
        ...

    def abort(self) -> None:
        """Stop the request."""
        ...


class RustCore(Protocol):
    """Describe one extension client instance."""

    def start(self, body: bytes, headers: dict[str, str]) -> RustRequestHandle:
        """Start one request."""
        ...

    def close(self) -> None:
        """Close the connection pool."""
        ...


class RustCoreFactory(Protocol):
    """Describe the extension client constructor."""

    def __call__(
        self,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        max_connections: int,
        perf_counter_now: float,
    ) -> RustCore:
        """Create one extension client."""
        ...


class RustCoreModule(Protocol):
    """Describe the extension module surface."""

    __version__: str
    RustCoreClient: RustCoreFactory


class _RustCoreModuleProxy:
    """Forward attributes exported dynamically by PyO3."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def __getattr__(self, name: str) -> object:
        return getattr(self._module, name)


def _load_rustcore() -> RustCoreModule:
    """Load a compatible extension or raise an actionable error."""
    try:
        loaded = import_module("agentperf_local_rustcore")
    except ImportError as exc:
        raise RuntimeError(
            "the Rust client is unavailable; install the rust extra (`uv sync --extra rust` in a source checkout, "
            "which needs a Rust toolchain) or pass --client python"
        ) from exc
    # This is the documented __getattr__ lazy-proxy carve-out. Static analysis
    # cannot see attributes exported by PyO3 at runtime.
    module = cast(RustCoreModule, _RustCoreModuleProxy(loaded))
    if not module.__version__.startswith(RUSTCORE_COMPATIBILITY_PREFIX):
        raise RuntimeError(
            f"agentperf_local_rustcore {module.__version__!r} is incompatible; "
            f"expected {RUSTCORE_COMPATIBILITY_PREFIX!r}"
        )
    return module


def validate_rustcore_available() -> None:
    """Raise an actionable error when the compatible Rust extension is unavailable."""
    _load_rustcore()


def _completion_error(error: Exception) -> CompletionError:
    """Normalize the stable Rust core error text."""
    message = str(error)
    status_text, separator, response_body = message.partition(": ")
    if status_text.startswith(RUST_STATUS_PREFIX):
        raw_status = status_text.removeprefix(RUST_STATUS_PREFIX)
        if raw_status.isdigit():
            return CompletionError(
                f"endpoint returned HTTP {raw_status}: {response_body}" if separator else message,
                status_code=int(raw_status),
                response_body=response_body if separator else None,
            )
    return CompletionError(message)


class RustStreamingClient:
    """Collect response bytes entirely inside the Rust read loop."""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_connections <= 0:
            raise ValueError("max_connections must be greater than zero")
        normalized_base_url = normalize_base_url(base_url)
        module = _load_rustcore()
        self._core = module.RustCoreClient(
            normalized_base_url,
            None if api_key is None else api_key.get_secret_value(),
            timeout_seconds,
            max_connections,
            perf_counter(),
        )

    async def complete(
        self,
        request: CompletionRequest,
        abort: asyncio.Event | None = None,
    ) -> CompletionResult:
        """Start one Rust request and materialize reads after close."""
        try:
            handle = self._core.start(orjson.dumps(request.body()), request.headers())
        except Exception as error:
            raise _completion_error(error) from error
        wait_task = asyncio.ensure_future(handle.wait())
        abort_task: asyncio.Task[bool] | None = None
        try:
            if abort is not None:
                abort_task = asyncio.create_task(abort.wait())
                done, _ = await asyncio.wait(
                    {wait_task, abort_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_task in done and not wait_task.done():
                    handle.abort()
            reads, aborted = await wait_task
        except asyncio.CancelledError:
            handle.abort()
            wait_task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await wait_task
            raise
        except Exception as error:
            raise _completion_error(error) from error
        finally:
            if abort_task is not None:
                abort_task.cancel()
                with suppress(asyncio.CancelledError):
                    await abort_task
        return CompletionResult(
            reads=tuple(RawRead(timestamp=timestamp, data=data) for timestamp, data in reads),
            aborted=aborted,
        )

    async def close(self) -> None:
        """Close the Rust connection pool."""
        self._core.close()
