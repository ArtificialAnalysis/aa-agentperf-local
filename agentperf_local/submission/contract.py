"""Define the typed body of POST /v1/submissions, as the service's pinned spec states it.

Public surface: SubmissionRequest and its parts (Client, Benchmark, Hardware, Accelerator,
ManagedDeployment, AttachedDeployment, CappedOutputPolicy, FreeOutputPolicy, Run, Turn,
Qualification, QualificationOutcome, Power), the enum and pinned-value aliases they use,
the patterns they share, and PROBE_IDS.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, NonNegativeFloat, model_validator

from agentperf_local.common.identity import GIT_COMMIT_PATTERN
from agentperf_local.common.units import BYTES_PER_GIB

INT64_MAX = 2**63 - 1
# The service bounds each count so that run totals always fit a BigQuery INT64.
MAX_COUNT = 2**31 - 1
PERCENT = 100
# A first-to-last-token window spans one decode interval fewer than the tokens it covers.
MINIMUM_DECODE_OUTPUT_TOKENS = 2
NVIDIA_DRIVER_VERSION = re.compile(r"^[0-9]+(\.[0-9]+)*$")
RUN_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
MODEL_RELEASE_SLUG_PATTERN = r"^[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*$"
CONTAINER_REFERENCE_PATTERN = r"^[^\s@]+@sha256:[0-9a-f]{64}$"

NonEmptyString = Annotated[str, Field(min_length=1)]
GitCommit = Annotated[str, Field(pattern=GIT_COMMIT_PATTERN)]
Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
ContainerReference = Annotated[str, Field(pattern=CONTAINER_REFERENCE_PATTERN)]
Count = Annotated[int, Field(ge=0, le=MAX_COUNT)]
PositiveCount = Annotated[int, Field(ge=1, le=MAX_COUNT)]
ByteCount = Annotated[int, Field(ge=1, le=INT64_MAX)]
MemoryBytes = Annotated[int, Field(ge=BYTES_PER_GIB, le=INT64_MAX)]
Fraction = Annotated[float, Field(ge=0, le=1)]

type PlatformFamily = Literal["linux", "macos", "windows"]
type Architecture = Literal["x86_64", "arm64"]
type AcceleratorVendor = Literal["nvidia", "apple", "amd", "intel"]
type Framework = Literal["llama-cpp", "vllm", "sglang", "splash"]
type AcceleratorBackend = Literal["cuda", "metal", "rocm", "vulkan", "sycl", "xpu"]
type SourceState = Literal["release", "clean"]
type ToolReplayMode = Literal["none", "live", "fixed_delay"]
type CappedOutputTokenPolicy = Literal["exact", "recorded"]
type ProbeId = Literal["single_tool", "no_tool", "parallel_tools", "tool_history", "capped_finish"]
# The service accepts only runs made with these settings. The client reads each one from
# the run's recorded settings, so a run made with another value fails here.
type TransportPolicyId = Literal["direct-sse-no-retry-v1"]
type CacheIsolationEnabled = Literal[True]
type CacheIsolationMode = Literal["run_namespace_prefix"]
type OutputTokenFallback = Literal[16_384]
type OutputTokenMargin = Literal[0]
type SyntheticPackId = Literal["aa-runtime-synthetic-v1"]
type PowerCollectorId = Literal["aa-nvidia-smi-v1"]

PROBE_IDS: frozenset[ProbeId] = frozenset(("single_tool", "no_tool", "parallel_tools", "tool_history", "capped_finish"))


class Client(BaseModel, frozen=True, extra="forbid", strict=True):
    """Identify the client build that measured the run."""

    name: NonEmptyString
    version: NonEmptyString
    source_revision: GitCommit
    source_state: SourceState


class Benchmark(BaseModel, frozen=True, extra="forbid", strict=True):
    """Identify the workload and the context window the server ran with."""

    workload_digest: Sha256Digest
    workload_context_tokens: PositiveCount
    context_tokens: PositiveCount
    observed_context_tokens: PositiveCount

    @model_validator(mode="after")
    def require_observed_context(self) -> Self:
        """Refuse a run whose server did not serve the context it was asked for."""
        if self.observed_context_tokens < self.context_tokens:
            raise ValueError(
                f"the server served a {self.observed_context_tokens:,}-token context, "
                f"below the {self.context_tokens:,} tokens the run asked for"
            )
        return self


class Accelerator(BaseModel, frozen=True, extra="forbid", strict=True, allow_inf_nan=False):
    """Describe the one accelerator the run used, as its driver reports it."""

    vendor: AcceleratorVendor
    product: NonEmptyString
    memory_bytes: MemoryBytes | None = None
    memory_is_unified: bool
    core_count: PositiveCount | None = None
    driver_version: NonEmptyString | None = None
    max_graphics_clock_mhz: PositiveCount | None = None
    max_memory_clock_mhz: PositiveCount | None = None
    power_limit_w: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def require_memory_and_driver(self) -> Self:
        """Require a dedicated memory size, and a numeric driver version on NVIDIA."""
        if self.memory_bytes is None and not self.memory_is_unified:
            raise ValueError("the accelerator reported no memory size and does not share host memory")
        if self.vendor == "nvidia" and (
            self.driver_version is None or NVIDIA_DRIVER_VERSION.fullmatch(self.driver_version) is None
        ):
            raise ValueError("an NVIDIA accelerator needs its driver version as dotted numbers, such as 580.95.05")
        return self


class Hardware(BaseModel, frozen=True, extra="forbid", strict=True):
    """Describe the machine that ran both the server and the client."""

    platform_family: PlatformFamily
    operating_system_version: NonEmptyString | None = None
    kernel_version: NonEmptyString | None = None
    architecture: Architecture
    cpu_model: NonEmptyString | None = None
    logical_cpu_count: PositiveCount | None = None
    memory_bytes: MemoryBytes
    accelerator: Accelerator


class _DeploymentFields(BaseModel, frozen=True, extra="forbid", strict=True):
    """Hold the deployment fields that both modes send."""

    model_release_slug: Annotated[str, Field(pattern=MODEL_RELEASE_SLUG_PATTERN)]
    hf_repository: NonEmptyString
    hf_revision: GitCommit
    framework: Framework
    framework_version: NonEmptyString
    framework_commit: GitCommit | None = None
    framework_container_reference: ContainerReference | None = None
    server_launch_command: NonEmptyString
    accelerator_backend: AcceleratorBackend

    @model_validator(mode="after")
    def require_framework_source(self) -> Self:
        """Require a commit or a digest-pinned container that identifies the framework build."""
        if self.framework_commit is None and self.framework_container_reference is None:
            raise ValueError("the deployment needs framework_commit, framework_container_reference, or both")
        return self


class ManagedDeployment(_DeploymentFields):
    """Describe a server the client started from a catalog recipe."""

    deployment_mode: Literal["managed"]
    profile_id: NonEmptyString
    model_artifact_digest: Sha256Digest
    recipe: NonEmptyString
    model_size_bytes: ByteCount


class AttachedDeployment(_DeploymentFields):
    """Describe a server the user started on the same machine."""

    deployment_mode: Literal["attached"]


class _PolicyFields(BaseModel, frozen=True, extra="forbid", strict=True, allow_inf_nan=False):
    """Hold the run settings that both output-token policies send."""

    client_backend: Literal["python", "rust"]
    transport_policy_id: TransportPolicyId
    temperature: float | None = None
    top_p: float | None = None
    top_k: Annotated[int, Field(ge=-INT64_MAX - 1, le=INT64_MAX)] | None = None
    min_p: float | None = None
    reasoning_effort: str | None = None
    cache_isolation_enabled: CacheIsolationEnabled
    cache_isolation_mode: CacheIsolationMode
    tool_replay_mode: ToolReplayMode = "none"


class CappedOutputPolicy(_PolicyFields):
    """Cap each turn's output at its target length."""

    output_token_policy: CappedOutputTokenPolicy
    output_token_fallback: OutputTokenFallback
    output_token_margin: OutputTokenMargin


class FreeOutputPolicy(_PolicyFields):
    """Let each turn generate until the model stops."""

    output_token_policy: Literal["free"]


class Run(BaseModel, frozen=True, extra="forbid", strict=True, allow_inf_nan=False):
    """Hold the run durations that the turns cannot show."""

    wall_duration_ms: NonNegativeFloat
    observer_duration_ms: NonNegativeFloat


class Turn(BaseModel, frozen=True, extra="forbid", strict=True, allow_inf_nan=False):
    """Hold one successful turn's timing and token counts."""

    turn_ordinal: Count
    task_ordinal: Count
    turn_in_task: Count
    finish_reason: NonEmptyString
    response_chunks: PositiveCount
    replayed_pacing_ms: NonNegativeFloat
    e2e_latency_ms: NonNegativeFloat
    time_to_first_token_ms: NonNegativeFloat
    generation_ms: Annotated[float, Field(gt=0)]
    server_prompt_tokens: Count
    server_output_tokens: Annotated[int, Field(ge=MINIMUM_DECODE_OUTPUT_TOKENS, le=MAX_COUNT)]
    target_output_tokens: PositiveCount
    cached_input_tokens: Count | None = None
    uncached_input_tokens: Count | None = None

    @model_validator(mode="after")
    def require_consistent_turn(self) -> Self:
        """Require timings inside the request, and cache counts that explain the prompt."""
        if self.time_to_first_token_ms > self.e2e_latency_ms or self.generation_ms > self.e2e_latency_ms:
            raise ValueError("time_to_first_token_ms and generation_ms must not exceed e2e_latency_ms")
        if (self.cached_input_tokens is None) != (self.uncached_input_tokens is None):
            raise ValueError("cached_input_tokens and uncached_input_tokens must be sent together")
        if (
            self.cached_input_tokens is not None
            and self.uncached_input_tokens is not None
            and self.cached_input_tokens + self.uncached_input_tokens != self.server_prompt_tokens
        ):
            raise ValueError("cached_input_tokens + uncached_input_tokens must equal server_prompt_tokens")
        return self


class QualificationOutcome(BaseModel, frozen=True, extra="forbid", strict=True):
    """Hold one runtime qualification probe result."""

    probe_id: ProbeId
    passed: bool
    failure_codes: tuple[NonEmptyString, ...]


class Qualification(BaseModel, frozen=True, extra="forbid", strict=True):
    """Hold the synthetic probe results from before the run."""

    synthetic_pack_id: SyntheticPackId
    outcomes: tuple[QualificationOutcome, ...]

    @model_validator(mode="after")
    def require_one_outcome_per_probe(self) -> Self:
        """Require exactly one outcome for each probe."""
        probe_ids = [outcome.probe_id for outcome in self.outcomes]
        if len(probe_ids) != len(PROBE_IDS) or set(probe_ids) != PROBE_IDS:
            raise ValueError(f"qualification needs exactly one outcome for each of {sorted(PROBE_IDS)}")
        return self


class Power(BaseModel, frozen=True, extra="forbid", strict=True, allow_inf_nan=False):
    """Hold the power telemetry of the measured phase."""

    collector_id: PowerCollectorId
    sampled_power_energy_valid: bool
    power_coverage: Fraction
    power_integration_coverage: Fraction
    energy_joules: NonNegativeFloat | None = None
    power_w_mean: NonNegativeFloat | None = None
    power_w_max: NonNegativeFloat | None = None
    temperature_c_max: float | None = None
    gpu_utilization_percent_mean: Annotated[float, Field(ge=0, le=PERCENT)] | None = None
    graphics_clock_mhz_median: NonNegativeFloat | None = None
    memory_clock_mhz_median: NonNegativeFloat | None = None
    memory_used_mib_maximum: NonNegativeFloat | None = None


class SubmissionRequest(BaseModel, frozen=True, extra="forbid", strict=True):
    """Hold one complete run as the service accepts it. `run_id` is the idempotency key.

    Each model here states the service's cross-field rules as a validator, so a run the
    service would refuse fails on this computer first, with the field that is wrong.
    """

    run_id: Annotated[str, Field(pattern=RUN_ID_PATTERN)]
    privacy_notice_version: NonEmptyString
    client: Client
    benchmark: Benchmark
    hardware: Hardware
    deployment: Annotated[ManagedDeployment | AttachedDeployment, Field(discriminator="deployment_mode")]
    policy: Annotated[CappedOutputPolicy | FreeOutputPolicy, Field(discriminator="output_token_policy")]
    run: Run
    turns: Annotated[tuple[Turn, ...], Field(min_length=1)]
    qualification: Qualification
    power: Power | None = None

    @model_validator(mode="after")
    def require_ordered_turns(self) -> Self:
        """Require the turns in replay order, with each task's turns in one block."""
        previous: Turn | None = None
        for index, turn in enumerate(self.turns):
            starts_task = turn.task_ordinal == (0 if previous is None else previous.task_ordinal + 1)
            continues_task = previous is not None and (
                turn.task_ordinal == previous.task_ordinal and turn.turn_in_task == previous.turn_in_task + 1
            )
            if turn.turn_ordinal != index or not (continues_task or (starts_task and turn.turn_in_task == 0)):
                raise ValueError(f"turns[{index}] is out of order")
            previous = turn
        return self

    @model_validator(mode="after")
    def require_mode_evidence(self) -> Self:
        """Require cache counts from managed runs and power telemetry from NVIDIA."""
        if isinstance(self.deployment, ManagedDeployment) and any(
            turn.cached_input_tokens is None for turn in self.turns
        ):
            raise ValueError("a managed run needs cached and uncached input tokens on every turn")
        if self.hardware.accelerator.vendor == "nvidia" and self.power is None:
            raise ValueError("a run on an NVIDIA accelerator needs power telemetry")
        return self
