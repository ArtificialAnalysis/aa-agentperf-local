"""Exercise transparent KV-cache memory estimates."""

from pathlib import Path

import orjson
import pytest

from agentperf_local.common.json_types import JsonObject, JsonValue
from scripts.estimate_kv_budget import (
    BudgetInputs,
    ModelCacheShape,
    estimate_cache_budget,
    load_cache_shape,
    main,
)

CONTEXT_TOKENS = 131_072
FP16_BYTES_PER_SCALAR = 2
GPT_OSS_KEY_VALUE_HEADS = 8
GPT_OSS_HEAD_DIMENSION = 64
GPT_OSS_SLIDING_WINDOW = 128
GPT_OSS_20B_LAYERS = 24
GPT_OSS_120B_LAYERS = 36


def _alternating_layers(count: int) -> list[JsonValue]:
    return ["sliding_attention" if index % 2 == 0 else "full_attention" for index in range(count)]


def _config(layers: int = GPT_OSS_20B_LAYERS) -> JsonObject:
    return {
        "num_hidden_layers": layers,
        "num_key_value_heads": GPT_OSS_KEY_VALUE_HEADS,
        "head_dim": GPT_OSS_HEAD_DIMENSION,
        "layer_types": _alternating_layers(layers),
        "sliding_window": GPT_OSS_SLIDING_WINDOW,
    }


@pytest.mark.parametrize(
    ("layers", "expected_full_gib", "expected_hybrid_gib"),
    (
        (GPT_OSS_20B_LAYERS, 6.0, 3.0029296875),
        (GPT_OSS_120B_LAYERS, 9.0, 4.50439453125),
    ),
)
def test_estimates_gpt_oss_cache_bounds(
    layers: int,
    expected_full_gib: float,
    expected_hybrid_gib: float,
) -> None:
    shape = ModelCacheShape.from_config(_config(layers))
    inputs = BudgetInputs(
        context_tokens=CONTEXT_TOKENS,
        cache_bytes_per_scalar=FP16_BYTES_PER_SCALAR,
        model_weight_gib=12.84,
        device_memory_gib=32.0,
        runtime_reserve_gib=4.0,
    )

    result = estimate_cache_budget(shape, inputs)

    assert result.sliding_bytes_per_token_per_layer == 2_048
    # gpt-oss holds one head shape, so the global layers price the same per token.
    assert result.global_bytes_per_token_per_layer == 2_048
    assert result.full_allocation_cache_gib == expected_full_gib
    assert result.declared_hybrid_cache_gib == expected_hybrid_gib
    assert result.headroom_after_full_allocation_gib == pytest.approx(32.0 - 12.84 - 4.0 - expected_full_gib)
    assert result.headroom_after_declared_hybrid_gib == pytest.approx(32.0 - 12.84 - 4.0 - expected_hybrid_gib)


def test_loads_shape_and_cli_prints_closed_estimate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_bytes(orjson.dumps(_config()))

    shape = load_cache_shape(config_path)
    exit_code = main(
        (
            str(config_path),
            "--context-tokens",
            str(CONTEXT_TOKENS),
            "--cache-bytes-per-scalar",
            str(FP16_BYTES_PER_SCALAR),
            "--model-weight-gib",
            "12.84",
            "--device-memory-gib",
            "32",
            "--runtime-reserve-gib",
            "4",
        )
    )
    output = orjson.loads(capsys.readouterr().out)

    assert shape.layers == GPT_OSS_20B_LAYERS
    assert exit_code == 0
    assert output["kind"] == "kv_cache_budget_estimate"
    assert output["full_allocation_cache_gib"] == 6.0
    assert output["declared_hybrid_cache_gib"] == 3.0029296875
    assert output["full_allocation_fits_assumptions"] is True
    assert output["inputs"]["runtime_reserve_gib"] == 4.0


# Gemma 4 26B-A4B: 5 global layers of 2 heads at dimension 512, and 25 sliding layers
# of 8 heads at dimension 256. Pricing every layer at the sliding shape understates the
# global layers by a factor of four.
GEMMA_GLOBAL_LAYERS = 5
GEMMA_SLIDING_LAYERS = 25
GEMMA_SLIDING_WINDOW = 1_024


def _gemma_config() -> JsonObject:
    layer_types: list[JsonValue] = ["full_attention"] * GEMMA_GLOBAL_LAYERS + [
        "sliding_attention"
    ] * GEMMA_SLIDING_LAYERS
    return {
        "num_hidden_layers": GEMMA_GLOBAL_LAYERS + GEMMA_SLIDING_LAYERS,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "num_global_key_value_heads": 2,
        "global_head_dim": 512,
        "layer_types": layer_types,
        "sliding_window": GEMMA_SLIDING_WINDOW,
    }


def test_prices_full_and_sliding_layers_at_their_own_head_shapes() -> None:
    """A model whose global layers hold a different head shape is priced per class."""
    shape = ModelCacheShape.from_config(_gemma_config())
    inputs = BudgetInputs(
        context_tokens=CONTEXT_TOKENS,
        cache_bytes_per_scalar=FP16_BYTES_PER_SCALAR,
        model_weight_gib=13.45,
        device_memory_gib=64.0,
        runtime_reserve_gib=1.5,
    )

    result = estimate_cache_budget(shape, inputs)

    assert result.sliding_bytes_per_token_per_layer == 8_192
    assert result.global_bytes_per_token_per_layer == 4_096
    # The window holds every sliding layer to 1,024 tokens; only the global layers grow.
    global_bytes = 4_096 * GEMMA_GLOBAL_LAYERS * CONTEXT_TOKENS
    sliding_bytes = 8_192 * GEMMA_SLIDING_LAYERS * GEMMA_SLIDING_WINDOW
    assert result.declared_hybrid_cache_gib == (global_bytes + sliding_bytes) / 1024**3


@pytest.mark.parametrize(
    ("config", "message"),
    (
        ({"num_hidden_layers": 24}, "num_key_value_heads"),
        ({**_config(), "layer_types": ["full_attention"]}, "every hidden layer"),
        ({**_config(), "layer_types": ["unknown_attention"] * GPT_OSS_20B_LAYERS}, "supports only"),
    ),
)
def test_rejects_incomplete_or_unknown_cache_shapes(config: JsonObject, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelCacheShape.from_config(config)
