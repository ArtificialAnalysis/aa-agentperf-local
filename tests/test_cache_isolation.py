"""Test run-scoped prompt cache isolation."""

from agentperf_local.common.json_types import JsonObject
from agentperf_local.replay.cache_isolation import (
    CACHE_NAMESPACE_DIGITS,
    CacheIsolationMetadata,
    cache_isolation_metadata,
    cache_namespace_prefix,
    generate_cache_namespace,
    isolate_messages,
)


def test_cache_isolation_public_surface() -> None:
    namespace = generate_cache_namespace()
    assert len(namespace.split()) == CACHE_NAMESPACE_DIGITS
    assert all(len(part) == 1 and part.isdigit() for part in namespace.split())

    messages: list[JsonObject] = [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
    ]
    isolated = isolate_messages(messages, "1 2 3")
    assert messages[0]["content"] == "hello"
    assert isolated[0]["content"] == cache_namespace_prefix("1 2 3") + "hello"
    assert isolated[1] == messages[1]
    assert isolated[1] is not messages[1]

    empty = isolate_messages([], "1 2 3")
    assert empty == [{"role": "system", "content": cache_namespace_prefix("1 2 3").rstrip()}]
    assert cache_isolation_metadata(None) == CacheIsolationMetadata(
        enabled=False,
        mode="none",
        namespace=None,
        namespace_digits=0,
    )
    assert cache_isolation_metadata("1 2 3") == CacheIsolationMetadata(
        enabled=True,
        mode="run_namespace_prefix",
        namespace="1 2 3",
        namespace_digits=CACHE_NAMESPACE_DIGITS,
    )
