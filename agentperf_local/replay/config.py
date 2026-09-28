"""Configure one finite replay run."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, SecretStr, model_validator

from agentperf_local.client.backends import CLIENT_BACKENDS, ClientBackend
from agentperf_local.client.endpoint import normalize_base_url, url_is_cleartext_remote
from agentperf_local.client.request import DEFAULT_MAX_OUTPUT_TOKENS
from agentperf_local.common.json_types import JsonValue
from agentperf_local.tools.docker import docker_executable_from_env

STANDARD_TEMPERATURE = 0.7
STANDARD_TOP_P = 0.8
STANDARD_TOP_K = 20
STANDARD_MIN_P = 0.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_LIVE_TOOL_TIMEOUT_SECONDS = 30.0
API_KEY_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CLEARTEXT_API_KEY_MESSAGE = (
    "refusing to send an API key over http to a non-loopback host; use https:// or remove the API key"
)
DEFAULT_OUTPUT_TOKEN_MARGIN = 0
DEFAULT_TOOL_DELAY_SCALE = 1.0


def read_api_key_env(name: str) -> SecretStr | None:
    """Read one API key value, or return None when it is unset, empty, or holds whitespace."""
    value = os.environ.get(name)
    if not value or any(character.isspace() for character in value):
        return None
    return SecretStr(value)


# How each request's output length is set:
# - exact: the recorded length, and the server is told to ignore end-of-sequence, so
#   every turn generates exactly that many tokens. Verbosity differences between the
#   model under test and the recording cannot move the measurement. The default.
# - recorded: the recorded length as a cap only; a model that stops early is measured
#   on its shorter output, one that would run longer is cut.
# - fixed: one cap for every turn.
type OutputTokenPolicy = Literal["exact", "recorded", "fixed"]
OUTPUT_TOKEN_POLICIES: tuple[OutputTokenPolicy, ...] = ("exact", "recorded", "fixed")
type SamplingPreset = Literal["standard", "custom"]
type ToolReplayMode = Literal["none", "recorded", "live"]


def _require_finite(name: str, value: float | None) -> None:
    """Reject a float setting that is not a finite number."""
    # A NaN or infinite value passes the range checks it should fail and silently disables the limit it configures.
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


class SamplingSettings(BaseModel, frozen=True):
    """Store resolved sampling values for completion requests."""

    preset: SamplingPreset
    temperature: float | None
    top_p: float | None
    extra_body: tuple[tuple[str, JsonValue], ...]

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Reject non-finite sampling values."""
        _require_finite("temperature", self.temperature)
        _require_finite("top_p", self.top_p)
        return self


class RunConfig(BaseModel, frozen=True):
    """Configure one sequential manifest replay."""

    base_url: str
    model: str
    api_key: SecretStr | None = None
    client_backend: ClientBackend = "python"
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    output_token_policy: OutputTokenPolicy = "exact"
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    output_token_margin: int = DEFAULT_OUTPUT_TOKEN_MARGIN
    sampling_preset: SamplingPreset = "standard"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    tool_choice: str | None = None
    reasoning_effort: str | None = None
    cache_isolation: bool = True
    cache_namespace: str | None = None
    tool_mode: ToolReplayMode = "none"
    tool_delay_scale: float = DEFAULT_TOOL_DELAY_SCALE
    live_tool_image: str | None = None
    live_workspace_root: Path | None = None
    live_network: str | None = None
    live_timeout_seconds: float = DEFAULT_LIVE_TOOL_TIMEOUT_SECONDS
    live_docker_executable: str | None = None

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Reject invalid or contradictory settings."""
        _require_finite("request_timeout_seconds", self.request_timeout_seconds)
        _require_finite("temperature", self.temperature)
        _require_finite("top_p", self.top_p)
        _require_finite("min_p", self.min_p)
        _require_finite("tool_delay_scale", self.tool_delay_scale)
        _require_finite("live_timeout_seconds", self.live_timeout_seconds)
        normalized_base_url = normalize_base_url(self.base_url)
        if self.api_key is not None and url_is_cleartext_remote(normalized_base_url):
            raise ValueError(CLEARTEXT_API_KEY_MESSAGE)
        if not self.model:
            raise ValueError("model must not be empty")
        if self.client_backend not in CLIENT_BACKENDS:
            raise ValueError("client_backend must be python or rust")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.output_token_policy not in OUTPUT_TOKEN_POLICIES:
            raise ValueError("output_token_policy must be exact, recorded, or fixed")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.output_token_margin < 0:
            raise ValueError("output_token_margin must be non-negative")
        if self.output_token_policy == "exact" and self.output_token_margin != 0:
            raise ValueError("the exact output policy pins each turn to its recorded length; it takes no margin")
        if self.sampling_preset not in {"standard", "custom"}:
            raise ValueError("sampling_preset must be standard or custom")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be between zero and one")
        if self.tool_choice not in {None, "none"}:
            raise ValueError("tool_choice must be none or null")
        if self.reasoning_effort is not None and not self.reasoning_effort:
            raise ValueError("reasoning_effort must not be empty")
        if not self.cache_isolation and self.cache_namespace is not None:
            raise ValueError("cache_namespace requires cache_isolation")
        if self.tool_mode not in {"none", "recorded", "live"}:
            raise ValueError("tool_mode must be none, recorded, or live")
        if self.tool_delay_scale < 0:
            raise ValueError("tool_delay_scale must be non-negative")
        if self.live_timeout_seconds <= 0:
            raise ValueError("live_timeout_seconds must be positive")
        if self.live_tool_image is not None and not self.live_tool_image:
            raise ValueError("live_tool_image must not be empty")
        if self.live_network is not None and not self.live_network:
            raise ValueError("live_network must not be empty")
        if self.live_docker_executable is not None and not self.live_docker_executable:
            raise ValueError("live_docker_executable must not be empty")
        live_options_set = any(
            value is not None
            for value in (
                self.live_tool_image,
                self.live_workspace_root,
                self.live_network,
                self.live_docker_executable,
            )
        )
        if self.tool_mode != "live" and live_options_set:
            raise ValueError("live Docker options require tool_mode=live")
        return self

    def docker_executable(self) -> str:
        """Resolve the Docker-compatible executable this run drives."""
        return self.live_docker_executable or docker_executable_from_env()

    def sampling(self) -> SamplingSettings:
        """Resolve preset defaults and explicit sampling overrides."""
        temperature = STANDARD_TEMPERATURE if self.sampling_preset == "standard" else None
        top_p = STANDARD_TOP_P if self.sampling_preset == "standard" else None
        top_k = STANDARD_TOP_K if self.sampling_preset == "standard" else None
        min_p = STANDARD_MIN_P if self.sampling_preset == "standard" else None
        if self.temperature is not None:
            temperature = self.temperature
        if self.top_p is not None:
            top_p = self.top_p
        if self.top_k is not None:
            top_k = self.top_k
        if self.min_p is not None:
            min_p = self.min_p

        extra_body: list[tuple[str, JsonValue]] = []
        if top_k is not None:
            extra_body.append(("top_k", top_k))
        if min_p is not None:
            extra_body.append(("min_p", min_p))
        if self.tool_choice is not None:
            extra_body.append(("tool_choice", self.tool_choice))
        return SamplingSettings(
            preset=self.sampling_preset,
            temperature=temperature,
            top_p=top_p,
            extra_body=tuple(extra_body),
        )
