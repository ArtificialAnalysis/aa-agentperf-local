"""Choose the streaming client implementation a run measures through.

- `ClientBackend`, `CLIENT_BACKENDS`: the backend names a run may select.
- `streaming_client`: build the client for one backend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import SecretStr

from agentperf_local.client.protocol import CompletionClient
from agentperf_local.client.python_client import PythonStreamingClient
from agentperf_local.client.rust_client import RustStreamingClient

type ClientBackend = Literal["python", "rust"]
CLIENT_BACKENDS: tuple[ClientBackend, ...] = ("python", "rust")


def streaming_client(
    client_backend: ClientBackend,
    *,
    base_url: str,
    api_key: SecretStr | None,
    timeout_seconds: float,
    max_connections: int,
) -> CompletionClient:
    """Build the streaming client for one backend.

    The Rust client loads its extension only when constructed, so a Python run never
    needs the optional Rust extra.
    """
    client_type = RustStreamingClient if client_backend == "rust" else PythonStreamingClient
    return client_type(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_connections=max_connections,
    )
