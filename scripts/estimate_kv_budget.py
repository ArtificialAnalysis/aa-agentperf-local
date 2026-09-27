"""Estimate transparent KV-cache tensor bounds from a model config."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import orjson
from pydantic import BaseModel, model_validator

from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object
from agentperf_local.common.units import BYTES_PER_GIB

KEY_AND_VALUE_TENSORS = 2


def _required_int(data: JsonObject, key: str, source: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{source}.{key} must be a positive integer")
    return value


def _optional_int(data: JsonObject, key: str, fallback: int) -> int:
    """Read one positive integer, falling back when a config does not declare it."""
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return fallback
    return value


def _required_strings(data: JsonObject, key: str, source: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{source}.{key} must contain non-empty strings")
    return tuple(item for item in value if isinstance(item, str))


class ModelCacheShape(BaseModel, frozen=True):
    """Store attention dimensions relevant to KV tensor payload."""

    layers: int
    key_value_heads: int
    head_dimension: int
    # Full-attention layers can hold a different head shape from sliding ones. Gemma 4
    # is the case that requires this: its global layers declare their own head count
    # and dimension, and pricing every layer at the sliding shape understates the cache.
    global_key_value_heads: int
    global_head_dimension: int
    layer_types: tuple[str, ...]
    sliding_window: int

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Validate the cache shape."""
        dimensions = (
            self.layers,
            self.key_value_heads,
            self.head_dimension,
            self.global_key_value_heads,
            self.global_head_dimension,
            self.sliding_window,
        )
        if min(dimensions) <= 0:
            raise ValueError("model cache dimensions must be positive")
        if len(self.layer_types) != self.layers:
            raise ValueError("layer_types must describe every hidden layer")
        if any(layer_type not in ("full_attention", "sliding_attention") for layer_type in self.layer_types):
            raise ValueError("cache estimator supports only full_attention and sliding_attention layers")
        return self

    @classmethod
    def from_config(cls, data: JsonObject) -> ModelCacheShape:
        """Select cache dimensions from a Hugging Face model config."""
        key_value_heads = _required_int(data, "num_key_value_heads", "model_config")
        head_dimension = _required_int(data, "head_dim", "model_config")
        return cls(
            layers=_required_int(data, "num_hidden_layers", "model_config"),
            key_value_heads=key_value_heads,
            head_dimension=head_dimension,
            global_key_value_heads=_optional_int(data, "num_global_key_value_heads", key_value_heads),
            global_head_dimension=_optional_int(data, "global_head_dim", head_dimension),
            layer_types=_required_strings(data, "layer_types", "model_config"),
            sliding_window=_required_int(data, "sliding_window", "model_config"),
        )

    def to_json(self) -> JsonObject:
        """Return cache-shape facts."""
        return {
            "layers": self.layers,
            "full_attention_layers": self.layer_types.count("full_attention"),
            "sliding_attention_layers": self.layer_types.count("sliding_attention"),
            "key_value_heads": self.key_value_heads,
            "head_dimension": self.head_dimension,
            "global_key_value_heads": self.global_key_value_heads,
            "global_head_dimension": self.global_head_dimension,
            "sliding_window": self.sliding_window,
        }


class BudgetInputs(BaseModel, frozen=True):
    """Store explicit memory-planning assumptions."""

    context_tokens: int
    cache_bytes_per_scalar: int
    model_weight_gib: float
    device_memory_gib: float
    runtime_reserve_gib: float

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Validate planning assumptions."""
        if self.context_tokens <= 0 or self.cache_bytes_per_scalar <= 0:
            raise ValueError("context and cache scalar size must be positive")
        values = (self.model_weight_gib, self.device_memory_gib, self.runtime_reserve_gib)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("memory planning values must be finite and non-negative")
        if self.model_weight_gib + self.runtime_reserve_gib > self.device_memory_gib:
            raise ValueError("weights plus runtime reserve already exceed device memory")
        return self

    def to_json(self) -> JsonObject:
        """Return explicit budget assumptions."""
        return {
            "context_tokens": self.context_tokens,
            "cache_bytes_per_scalar": self.cache_bytes_per_scalar,
            "model_weight_gib": self.model_weight_gib,
            "device_memory_gib": self.device_memory_gib,
            "runtime_reserve_gib": self.runtime_reserve_gib,
        }


class CacheBudgetEstimate(BaseModel, frozen=True):
    """Store theoretical KV payload bounds and remaining headroom."""

    shape: ModelCacheShape
    inputs: BudgetInputs
    sliding_bytes_per_token_per_layer: int
    global_bytes_per_token_per_layer: int
    full_allocation_cache_gib: float
    declared_hybrid_cache_gib: float
    headroom_after_full_allocation_gib: float
    headroom_after_declared_hybrid_gib: float

    def to_json(self) -> JsonObject:
        """Return the bounded estimate and limitations."""
        limitations: list[JsonValue] = [
            "Tensor payload only; excludes allocator padding, metadata, compute buffers, graph capture, "
            "and fragmentation.",
            "The full bound assumes every layer retains every context token.",
            "The hybrid estimate assumes sliding-attention layers retain only the declared sliding window.",
            "A runtime may allocate differently; an on-device peak-memory qualification is authoritative.",
            "Parallel slots, batching, speculative decoding, and draft models require separate budgets.",
        ]
        return {
            "version": 2,
            "kind": "kv_cache_budget_estimate",
            "shape": self.shape.to_json(),
            "inputs": self.inputs.to_json(),
            "sliding_bytes_per_token_per_layer": self.sliding_bytes_per_token_per_layer,
            "global_bytes_per_token_per_layer": self.global_bytes_per_token_per_layer,
            "full_allocation_cache_gib": self.full_allocation_cache_gib,
            "declared_hybrid_cache_gib": self.declared_hybrid_cache_gib,
            "headroom_after_full_allocation_gib": self.headroom_after_full_allocation_gib,
            "headroom_after_declared_hybrid_gib": self.headroom_after_declared_hybrid_gib,
            "full_allocation_fits_assumptions": self.headroom_after_full_allocation_gib >= 0,
            "declared_hybrid_fits_assumptions": self.headroom_after_declared_hybrid_gib >= 0,
            "limitations": limitations,
        }


def estimate_cache_budget(shape: ModelCacheShape, inputs: BudgetInputs) -> CacheBudgetEstimate:
    """Calculate full-allocation and declared hybrid tensor payloads.

    Each layer class is priced at its own head shape, because a model can hold a
    different head count and dimension in its full-attention layers.
    """
    sliding_per_layer = (
        KEY_AND_VALUE_TENSORS * shape.key_value_heads * shape.head_dimension * inputs.cache_bytes_per_scalar
    )
    global_per_layer = (
        KEY_AND_VALUE_TENSORS
        * shape.global_key_value_heads
        * shape.global_head_dimension
        * inputs.cache_bytes_per_scalar
    )
    full_layers = shape.layer_types.count("full_attention")
    sliding_layers = shape.layer_types.count("sliding_attention")
    full_bytes = (global_per_layer * full_layers + sliding_per_layer * sliding_layers) * inputs.context_tokens
    windowed_tokens = min(inputs.context_tokens, shape.sliding_window)
    hybrid_bytes = (
        global_per_layer * full_layers * inputs.context_tokens + sliding_per_layer * sliding_layers * windowed_tokens
    )
    full_gib = full_bytes / BYTES_PER_GIB
    hybrid_gib = hybrid_bytes / BYTES_PER_GIB
    non_cache = inputs.model_weight_gib + inputs.runtime_reserve_gib
    return CacheBudgetEstimate(
        shape=shape,
        inputs=inputs,
        sliding_bytes_per_token_per_layer=sliding_per_layer,
        global_bytes_per_token_per_layer=global_per_layer,
        full_allocation_cache_gib=full_gib,
        declared_hybrid_cache_gib=hybrid_gib,
        headroom_after_full_allocation_gib=inputs.device_memory_gib - non_cache - full_gib,
        headroom_after_declared_hybrid_gib=inputs.device_memory_gib - non_cache - hybrid_gib,
    )


def load_cache_shape(path: Path) -> ModelCacheShape:
    """Load cache dimensions from one local model config."""
    try:
        data = normalize_json_object(orjson.loads(path.read_bytes()))
    except orjson.JSONDecodeError as error:
        raise ValueError(f"invalid model config JSON: {path}") from error
    return ModelCacheShape.from_config(data)


def build_parser() -> argparse.ArgumentParser:
    """Build the KV budget command."""
    parser = argparse.ArgumentParser(description="Estimate KV tensor memory bounds from a model config.")
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--cache-bytes-per-scalar", type=int, required=True)
    parser.add_argument("--model-weight-gib", type=float, required=True)
    parser.add_argument("--device-memory-gib", type=float, required=True)
    parser.add_argument("--runtime-reserve-gib", type=float, required=True)
    return parser


def _value(namespace: argparse.Namespace, name: str) -> object:
    return getattr(namespace, name)


def _integer(namespace: argparse.Namespace, name: str) -> int:
    value = _value(namespace, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{name} was not parsed as an integer")
    return value


def _number(namespace: argparse.Namespace, name: str) -> float:
    value = _value(namespace, name)
    if not isinstance(value, float):
        raise RuntimeError(f"{name} was not parsed as a number")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Print one transparent cache-budget estimate."""
    namespace = build_parser().parse_args(argv)
    config_path = _value(namespace, "model_config")
    if not isinstance(config_path, Path):
        raise RuntimeError("model config was not parsed as a path")
    inputs = BudgetInputs(
        context_tokens=_integer(namespace, "context_tokens"),
        cache_bytes_per_scalar=_integer(namespace, "cache_bytes_per_scalar"),
        model_weight_gib=_number(namespace, "model_weight_gib"),
        device_memory_gib=_number(namespace, "device_memory_gib"),
        runtime_reserve_gib=_number(namespace, "runtime_reserve_gib"),
    )
    estimate = estimate_cache_budget(load_cache_shape(config_path), inputs)
    print(orjson.dumps(estimate.to_json(), option=orjson.OPT_INDENT_2).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
