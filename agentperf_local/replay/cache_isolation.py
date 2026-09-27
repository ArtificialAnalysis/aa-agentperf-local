"""Create run-scoped cache namespaces and isolate request messages."""

from __future__ import annotations

import copy
import os
import secrets
from typing import Literal

from pydantic import BaseModel

from agentperf_local.common.json_types import JsonObject

CACHE_NAMESPACE_ENV = "AGENTPERF_LOCAL_CACHE_NAMESPACE"
CACHE_NAMESPACE_DIGITS = 32

type CacheIsolationMode = Literal["none", "run_namespace_prefix"]


class CacheIsolationMetadata(BaseModel, frozen=True):
    """Describe the prompt cache isolation used by one run."""

    enabled: bool
    mode: CacheIsolationMode
    namespace: str | None
    namespace_digits: int

    def to_dict(self) -> JsonObject:
        """Return report-ready JSON data."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "namespace": self.namespace,
            "namespace_digits": self.namespace_digits,
        }


def generate_cache_namespace() -> str:
    """Return a fixed-shape namespace for one replay invocation."""
    return " ".join(str(secrets.randbelow(10)) for _ in range(CACHE_NAMESPACE_DIGITS))


def resolve_cache_namespace(namespace: str | None = None) -> str:
    """Resolve an explicit, environmental, or new cache namespace."""
    return namespace or os.environ.get(CACHE_NAMESPACE_ENV) or generate_cache_namespace()


def cache_namespace_prefix(namespace: str) -> str:
    """Return the message prefix for a cache namespace."""
    return f"{namespace}\nPerformance replay cache namespace. Ignore the digits above.\n\n"


def isolate_messages(messages: list[JsonObject], namespace: str) -> list[JsonObject]:
    """Prefix the first message without changing the input messages."""
    isolated = copy.deepcopy(messages)
    prefix = cache_namespace_prefix(namespace)
    if not isolated:
        return [{"role": "system", "content": prefix.rstrip()}]

    first = isolated[0]
    content = first.get("content")
    if isinstance(content, str):
        first["content"] = prefix + content
    elif isinstance(content, list):
        first["content"] = [{"type": "text", "text": prefix}, *content]
    elif content is None:
        first["content"] = prefix.rstrip()
    else:
        first["content"] = prefix + str(content)
    return isolated


def cache_isolation_metadata(namespace: str | None) -> CacheIsolationMetadata:
    """Describe whether a run used cache isolation."""
    enabled = namespace is not None
    return CacheIsolationMetadata(
        enabled=enabled,
        mode="run_namespace_prefix" if enabled else "none",
        namespace=namespace,
        namespace_digits=CACHE_NAMESPACE_DIGITS if enabled else 0,
    )
