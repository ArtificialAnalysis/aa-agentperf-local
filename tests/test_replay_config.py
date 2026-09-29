"""Exercise replay configuration validation through its public models."""

import math
from typing import Any

import pytest

from agentperf_local.replay.config import (
    STANDARD_TEMPERATURE,
    STANDARD_TOP_P,
    RunConfig,
    SamplingSettings,
)

BASE_URL = "http://127.0.0.1:8000/v1"
MODEL = "local-model"
NON_FINITE_VALUES = (math.nan, math.inf, -math.inf)


def _config(**overrides: Any) -> RunConfig:
    return RunConfig(base_url=BASE_URL, model=MODEL, **overrides)


@pytest.mark.parametrize(
    "field",
    (
        "request_timeout_seconds",
        "temperature",
        "top_p",
        "min_p",
        "live_timeout_seconds",
    ),
)
@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_run_config_rejects_every_non_finite_float_field(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        _config(**{field: value})


@pytest.mark.parametrize("field", ("temperature", "top_p"))
@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_sampling_settings_reject_every_non_finite_float_field(field: str, value: float) -> None:
    values: dict[str, float | None] = {"temperature": None, "top_p": None}
    values[field] = value

    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        SamplingSettings(preset="custom", extra_body=(), **values)


def test_finite_settings_still_resolve_the_preset_defaults() -> None:
    sampling = _config(request_timeout_seconds=30.0, temperature=0.5).sampling()

    assert sampling.temperature == 0.5
    assert sampling.top_p == STANDARD_TOP_P
    assert (
        SamplingSettings(preset="standard", temperature=STANDARD_TEMPERATURE, top_p=None, extra_body=()).top_p is None
    )


def test_tool_choice_is_recorded_in_request_settings() -> None:
    """A managed framework can disable automatic tool selection."""
    sampling = _config(tool_choice="none").sampling()

    assert ("tool_choice", "none") in sampling.extra_body
