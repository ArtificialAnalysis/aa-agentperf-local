"""Decide which context lengths a managed deployment may use and what memory each needs.

- `resolve_context_tokens`, `require_replay_context_floor`: what a launch may request.
- `derived_minimum_memory_bytes`, `memory_fits`, `context_fit`: the memory each context needs.
- `context_ladder`, `reduced_context_rungs`, `default_context_tokens`: the contexts a picker offers.
- `smaller_offered_context_fits`, `largest_fitting_reduced_context`: what a memory refusal can suggest.
"""

from __future__ import annotations

from agentperf_local.deployment.catalog import DeploymentMemory, ModelDeployment

DEPLOYMENT_MEMORY_SLACK_BYTES = 64 * 1024 * 1024


KEY_AND_VALUE_TENSORS = 2


# A managed context must cover the model's sliding window and the shortest recorded
# agent turns; anything below this floor cannot replay a task.
MINIMUM_CONTEXT_TOKENS = 4_096


# The fixed context choices offered below the full benchmark context.
REDUCED_CONTEXT_LADDER = (32_768, 16_384, 8_192)


def resolve_context_tokens(deployment: ModelDeployment, context_tokens: int | None) -> int:
    """Resolve one requested context against the recipe's benchmark context."""
    if context_tokens is None:
        return deployment.context_tokens
    if context_tokens > deployment.context_tokens:
        raise ValueError(
            f"requested context {context_tokens} exceeds the profile's "
            f"{deployment.context_tokens}-token benchmark context"
        )
    if context_tokens < MINIMUM_CONTEXT_TOKENS:
        raise ValueError(f"requested context must be at least {MINIMUM_CONTEXT_TOKENS} tokens")
    return context_tokens


class ContextBelowReplayFloor(ValueError):
    """Report a launch context below the context the replay needs."""


def require_replay_context_floor(required_context_tokens: int | None, launch_context_tokens: int) -> None:
    """Refuse a launch whose context is below the replay's floor.

    Without this check a too-small launch fails mid-run. Nothing would then trace
    the failure back to the context choice.
    """
    if required_context_tokens is not None and launch_context_tokens < required_context_tokens:
        raise ContextBelowReplayFloor(
            f"This replay needs at least {required_context_tokens:,} tokens of context, "
            f"but the server would start with {launch_context_tokens:,}."
        )


def memory_fits(available_memory_bytes: int, minimum_memory_bytes: int) -> bool:
    """Accept cards that report slightly less usable memory than their nominal capacity."""
    return available_memory_bytes >= minimum_memory_bytes - DEPLOYMENT_MEMORY_SLACK_BYTES


def _kv_cache_bytes(memory: DeploymentMemory, context_tokens: int) -> int:
    """Estimate the KV payload one sequence of the requested length holds.

    Full and sliding layers are priced separately because they can hold different head
    shapes. Layers whose state does not grow with the context — linear attention and
    state-space layers — contribute through the recipe's constant state instead.
    """
    full_bytes = memory.full_attention_layers * memory.full_kv_heads * memory.full_kv_head_dimension * context_tokens
    windowed_tokens = min(context_tokens, memory.sliding_cached_tokens) if memory.sliding_attention_layers else 0
    sliding_bytes = (
        memory.sliding_attention_layers * memory.sliding_kv_heads * memory.sliding_kv_head_dimension * windowed_tokens
    )
    return KEY_AND_VALUE_TENSORS * memory.kv_bytes_per_scalar * (full_bytes + sliding_bytes)


def derived_minimum_memory_bytes(deployment: ModelDeployment, context_tokens: int) -> int:
    """Derive the accelerator-memory floor for one requested context length.

    The same formula at the recipe's full context must land exactly on the catalog's
    pinned minimum, so the formula and the catalog can never drift apart.
    """
    if context_tokens <= 0:
        raise ValueError("context token count must be positive to derive a memory floor")
    memory = deployment.memory

    def minimum(tokens: int) -> int:
        return (
            deployment.resident_artifact_bytes
            + _kv_cache_bytes(memory, tokens)
            + memory.constant_state_bytes
            + memory.runtime_overhead_bytes
        )

    if minimum(deployment.context_tokens) != deployment.minimum_memory_bytes:
        raise ValueError(
            "catalog minimum_memory_bytes does not equal the KV-cache formula at the full context; "
            "update the formula constants and the catalog together"
        )
    return minimum(context_tokens)


def context_memory_floor(deployment: ModelDeployment, context_tokens: int) -> int | None:
    """Return the derived memory floor, or None when the catalog and formula disagree.

    An external catalog can pin a minimum the KV formula does not reproduce. Display
    paths degrade on None instead of crashing.
    """
    try:
        return derived_minimum_memory_bytes(deployment, context_tokens)
    except ValueError:
        return None


def context_fit(deployment: ModelDeployment, context_tokens: int, available_memory_bytes: int | None) -> bool | None:
    """Report whether one context's memory floor fits the device, or None when either side is unknown."""
    if available_memory_bytes is None:
        return None
    minimum_memory_bytes = context_memory_floor(deployment, context_tokens)
    if minimum_memory_bytes is None:
        return None
    return memory_fits(available_memory_bytes, minimum_memory_bytes)


def reduced_context_rungs(deployment: ModelDeployment, replay_floor_tokens: int | None = None) -> tuple[int, ...]:
    """Return the reduced contexts offered for one recipe, honoring the replay's floor."""
    return tuple(
        rung
        for rung in REDUCED_CONTEXT_LADDER
        if MINIMUM_CONTEXT_TOKENS <= rung < deployment.context_tokens
        and (replay_floor_tokens is None or rung >= replay_floor_tokens)
    )


def context_ladder(deployment: ModelDeployment, replay_floor_tokens: int | None) -> tuple[int, ...]:
    """Return the full context first, then every reduced option this recipe and replay allow.

    Without a trustworthy per-context memory estimate, reduced rungs would advertise
    memory needs nobody can compute, so only the pinned full context is offered.
    """
    if context_memory_floor(deployment, deployment.context_tokens) is None:
        return (deployment.context_tokens,)
    return (deployment.context_tokens, *reduced_context_rungs(deployment, replay_floor_tokens))


def default_context_tokens(
    deployment: ModelDeployment,
    ladder: tuple[int, ...],
    available_memory_bytes: int | None,
) -> int:
    """Pick the full context when it fits or is unverifiable, else the largest fitting reduced option."""
    full_context_tokens = ladder[0]
    if context_fit(deployment, full_context_tokens, available_memory_bytes) is not False:
        return full_context_tokens
    for rung in ladder[1:]:
        if context_fit(deployment, rung, available_memory_bytes) is True:
            return rung
    return full_context_tokens


def smaller_offered_context_fits(
    deployment: ModelDeployment,
    requested_tokens: int,
    available_memory_bytes: int | None,
    replay_floor_tokens: int | None,
) -> bool:
    """Return whether an offered context below the requested one fits the device."""
    return any(
        rung < requested_tokens and context_fit(deployment, rung, available_memory_bytes) is True
        for rung in reduced_context_rungs(deployment, replay_floor_tokens)
    )


def largest_fitting_reduced_context(deployment: ModelDeployment, available_memory_bytes: int | None) -> int | None:
    """Return the largest reduced context that fits the device, ignoring any replay floor."""
    return max(
        (
            rung
            for rung in reduced_context_rungs(deployment)
            if context_fit(deployment, rung, available_memory_bytes) is True
        ),
        default=None,
    )
