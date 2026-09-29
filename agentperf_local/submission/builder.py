"""Build the submission body for one finished run from the files in its results folder.

Public surface: build_submission_request, encode_submission, write_prepared_submission,
read_prepared_submission, and PreparedSubmission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agentperf_local.client.backends import ClientBackend
from agentperf_local.client.endpoint import url_names_loopback_host
from agentperf_local.common.durable_files import (
    NewFile,
    read_bounded_file,
    validate_public_output_path,
    write_new_file,
)
from agentperf_local.common.identity import sha256_bytes
from agentperf_local.common.json_types import normalize_json, pretty_json_bytes
from agentperf_local.common.models import read_record
from agentperf_local.deployment.attached_server import (
    ATTACHED_SERVER_FILENAME,
    MAX_ATTACHED_SERVER_BYTES,
    AttachedServer,
    read_attached_server,
)
from agentperf_local.deployment.managed import (
    DEPLOYMENT_RECORD_FILENAME,
    MAX_DEPLOYMENT_RECORD_BYTES,
    read_deployment_record,
)
from agentperf_local.deployment.qualification import QUALIFICATION_FILENAME, load_qualification_file
from agentperf_local.provenance.benchmark import (
    MEASUREMENT_BINDING_FILENAME,
    PRODUCER_CLIENT_NAME,
    MeasurementBinding,
    SourceProvenance,
    load_measurement_binding,
    workload_digest,
)
from agentperf_local.provenance.hardware import HardwareSnapshot, contract_vendor, selected_accelerator
from agentperf_local.replay.config import OutputTokenPolicy, ToolReplayMode
from agentperf_local.reports.reporting import SUMMARY_FILENAME, TURNS_FILENAME
from agentperf_local.submission.client import MAX_REQUEST_BYTES
from agentperf_local.submission.contract import (
    Accelerator,
    Architecture,
    AttachedDeployment,
    Benchmark,
    CacheIsolationEnabled,
    CacheIsolationMode,
    CappedOutputPolicy,
    Client,
    Hardware,
    ManagedDeployment,
    OutputTokenFallback,
    OutputTokenMargin,
    PlatformFamily,
    Power,
    Qualification,
    QualificationOutcome,
    Run,
    SourceState,
    SubmissionRequest,
    TransportPolicyId,
    Turn,
)
from agentperf_local.submission.framework_commit import (
    commit_ref,
    framework_ref,
    resolve_framework_commit,
)
from agentperf_local.submission.notice import PRIVACY_NOTICE_VERSION
from agentperf_local.submission.spec import validate_against_spec
from agentperf_local.telemetry.power import POWER_SUMMARY_FILENAME, load_power_summary
from agentperf_local.workload.schema import load_manifest, load_trace

# Observers run between turns, and their time is left out of the measured window. A run
# stays submittable while that time is a small share of what was measured.
OBSERVER_OVERHEAD_TOLERANCE_FRACTION = 0.01
MAX_SUMMARY_BYTES = 16 * 1024 * 1024

_PLATFORM_FAMILIES: dict[str, PlatformFamily] = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}
# platform.machine() spells the same architecture differently on each operating system.
_ARCHITECTURES: dict[str, Architecture] = {
    "x86_64": "x86_64",
    "AMD64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}


class _RecordedContext(BaseModel, frozen=True):
    full_benchmark_tokens: int


class _RecordedOutputTokens(BaseModel, frozen=True):
    policy: OutputTokenPolicy
    fallback: OutputTokenFallback
    margin: OutputTokenMargin


class _RecordedExtraBody(BaseModel, frozen=True):
    top_k: int | None = None
    min_p: float | None = None


class _RecordedSampling(BaseModel, frozen=True):
    temperature: float | None
    top_p: float | None
    extra_body: _RecordedExtraBody


class _RecordedCacheIsolation(BaseModel, frozen=True):
    enabled: CacheIsolationEnabled
    mode: CacheIsolationMode


class _RecordedToolReplay(BaseModel, frozen=True):
    mode: ToolReplayMode


class _RecordedConfig(BaseModel, frozen=True):
    base_url: str
    model: str
    client_backend: ClientBackend
    transport_policy_id: TransportPolicyId
    context: _RecordedContext
    output_tokens: _RecordedOutputTokens
    sampling: _RecordedSampling
    reasoning_effort: str | None
    cache_isolation: _RecordedCacheIsolation
    tool_replay: _RecordedToolReplay


class _RecordedObserver(BaseModel, frozen=True):
    duration_ms: float


class _RecordedSummary(BaseModel, frozen=True):
    """Hold the summary.json fields a submission reads; the reader skips the rest."""

    version: Literal[1]
    kind: Literal["run_summary"]
    run_id: str
    success: bool
    wall_duration_ms: float
    measured_duration_ms: float
    observer: _RecordedObserver
    config: _RecordedConfig


class _RecordedTiming(BaseModel, frozen=True):
    e2e_latency_ms: float | None
    time_to_first_token_ms: float | None
    generation_ms: float | None


class _RecordedTokens(BaseModel, frozen=True):
    server_prompt_tokens: int | None
    server_output_tokens: int | None
    server_cached_prompt_tokens: int | None = None
    server_uncached_prompt_tokens: int | None = None


class _RecordedNormalization(BaseModel, frozen=True):
    target_output_tokens: int | None


class _RecordedToolCall(BaseModel, frozen=True):
    exception_info: str
    returncode_matches_recorded: bool | None


class _RecordedTurn(BaseModel, frozen=True):
    """Hold the turns.jsonl fields a submission reads; the reader skips prompts and every other field."""

    version: Literal[1]
    kind: Literal["turn"]
    turn_id: str
    task_id: str
    success: bool
    aborted: bool
    error: str | None
    timing: _RecordedTiming
    tokens: _RecordedTokens
    normalization: _RecordedNormalization
    replayed_tool_delay_ms: float
    tool_calls: tuple[_RecordedToolCall, ...]
    response_chunks: int
    finish_reason: str | None


class _TurnPosition(BaseModel, frozen=True):
    turn_ordinal: int
    task_ordinal: int
    turn_in_task: int


def _read_summary(results_dir: Path) -> _RecordedSummary:
    path = results_dir / SUMMARY_FILENAME
    encoded = read_bounded_file(path, MAX_SUMMARY_BYTES, label="run summary")
    try:
        return read_record(_RecordedSummary, encoded, SUMMARY_FILENAME, unknown_keys="skip")
    except ValueError as error:
        raise ValueError(f"this run's recorded settings cannot be submitted:\n{error}") from error


def _read_turns(results_dir: Path) -> tuple[_RecordedTurn, ...]:
    path = results_dir / TURNS_FILENAME
    turns: list[_RecordedTurn] = []
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                turns.append(read_record(_RecordedTurn, line, f"{TURNS_FILENAME}:{line_number}", unknown_keys="skip"))
    return tuple(turns)


def _require_bound_run(binding: MeasurementBinding, summary: _RecordedSummary) -> None:
    """Refuse a results folder whose files no longer describe the run the binding names."""
    if summary.run_id != binding.run_id:
        raise ValueError("summary.json names a different run than measurement.json")
    if not summary.success:
        raise ValueError("the run had failed turns; only a run where every turn succeeded can be submitted")
    if workload_digest(binding.manifest_path) != binding.manifest_digest:
        raise ValueError("the workload files changed after the run started")
    if sha256_bytes(summary.config.model.encode("utf-8")) != binding.endpoint_model_digest:
        raise ValueError("summary.json names a different model than measurement.json")
    observer_ms = summary.observer.duration_ms
    if observer_ms > summary.measured_duration_ms * OBSERVER_OVERHEAD_TOLERANCE_FRACTION:
        raise ValueError(
            f"progress display time exceeds {OBSERVER_OVERHEAD_TOLERANCE_FRACTION:.0%} of the measured time; "
            "run again without --progress"
        )


def _turn_positions(binding: MeasurementBinding, turns: tuple[_RecordedTurn, ...]) -> tuple[_TurnPosition, ...]:
    """Number the turns by the workload's task order, and refuse turns that are missing or out of order."""
    manifest = load_manifest(binding.manifest_path)
    root = binding.manifest_path.parent
    expected = tuple(
        (row.turn_id, task_ordinal, turn_in_task)
        for task_ordinal, task in enumerate(manifest.tasks)
        for turn_in_task, row in enumerate(load_trace(root / task.trace))
    )
    if tuple(turn.turn_id for turn in turns) != tuple(turn_id for turn_id, _, _ in expected):
        raise ValueError("turns.jsonl is missing turns, has them out of order, or belongs to another workload")
    return tuple(
        _TurnPosition(turn_ordinal=turn_ordinal, task_ordinal=task_ordinal, turn_in_task=turn_in_task)
        for turn_ordinal, (_, task_ordinal, turn_in_task) in enumerate(expected)
    )


def _required[Value](value: Value | None, turn: _RecordedTurn, field: str) -> Value:
    if value is None:
        raise ValueError(f"turn {turn.turn_id} has no {field}")
    return value


def _contract_turn(turn: _RecordedTurn, position: _TurnPosition) -> Turn:
    """Copy one successful turn's raw timing and token counts."""
    if not turn.success or turn.aborted or turn.error is not None:
        raise ValueError(f"turn {turn.turn_id} did not succeed")
    if any(call.exception_info or call.returncode_matches_recorded is False for call in turn.tool_calls):
        raise ValueError(f"turn {turn.turn_id} has a failed tool call")
    return Turn(
        turn_ordinal=position.turn_ordinal,
        task_ordinal=position.task_ordinal,
        turn_in_task=position.turn_in_task,
        finish_reason=_required(turn.finish_reason, turn, "finish reason"),
        response_chunks=turn.response_chunks,
        replayed_pacing_ms=turn.replayed_tool_delay_ms,
        e2e_latency_ms=_required(turn.timing.e2e_latency_ms, turn, "end-to-end latency"),
        time_to_first_token_ms=_required(turn.timing.time_to_first_token_ms, turn, "time to first token"),
        generation_ms=_required(turn.timing.generation_ms, turn, "generation time"),
        server_prompt_tokens=_required(turn.tokens.server_prompt_tokens, turn, "server prompt token count"),
        server_output_tokens=_required(turn.tokens.server_output_tokens, turn, "server output token count"),
        target_output_tokens=_required(turn.normalization.target_output_tokens, turn, "target output token count"),
        cached_input_tokens=turn.tokens.server_cached_prompt_tokens,
        uncached_input_tokens=turn.tokens.server_uncached_prompt_tokens,
    )


def _client(producer: SourceProvenance) -> Client:
    """Name the client build, refusing one whose source commit is unknown or has uncommitted changes."""
    if producer.source_state == "dirty":
        raise ValueError("the client source had uncommitted changes when the run started; commit them and run again")
    if producer.source_revision is None or producer.source_state not in ("release", "clean"):
        raise ValueError(
            "this client build cannot name its source commit; install a release or run from a clean git checkout"
        )
    source_state: SourceState = "release" if producer.source_state == "release" else "clean"
    return Client(
        name=PRODUCER_CLIENT_NAME,
        version=producer.client_version,
        source_revision=producer.source_revision,
        source_state=source_state,
    )


def _hardware(snapshot: HardwareSnapshot) -> Hardware:
    """Describe the host and its one accelerator with the raw values their probes read."""
    platform_family = _PLATFORM_FAMILIES.get(snapshot.operating_system)
    architecture = _ARCHITECTURES.get(snapshot.architecture)
    if platform_family is None or architecture is None:
        raise ValueError(f"{snapshot.operating_system} on {snapshot.architecture} cannot be submitted")
    if snapshot.memory_bytes is None:
        raise ValueError("the host did not report its memory size")
    accelerator = selected_accelerator(snapshot)
    vendor = contract_vendor(accelerator.vendor)
    return Hardware(
        platform_family=platform_family,
        operating_system_version=snapshot.operating_system_version,
        kernel_version=snapshot.kernel_version,
        architecture=architecture,
        cpu_model=snapshot.cpu_model,
        logical_cpu_count=snapshot.logical_cpu_count,
        memory_bytes=snapshot.memory_bytes,
        accelerator=Accelerator(
            vendor=vendor,
            product=accelerator.name,
            memory_bytes=accelerator.memory_bytes,
            memory_is_unified=accelerator.memory_is_unified,
            core_count=accelerator.core_count,
            driver_version=accelerator.driver_version,
            max_graphics_clock_mhz=accelerator.max_graphics_clock_mhz,
            max_memory_clock_mhz=accelerator.max_memory_clock_mhz,
            power_limit_w=accelerator.power_limit_w,
        ),
    )


def _bound_file(results_dir: Path, filename: str, max_bytes: int, binding: MeasurementBinding) -> bytes:
    """Read one deployment description, refusing bytes the measurement binding does not name."""
    encoded = read_bounded_file(results_dir / filename, max_bytes, label=filename)
    if sha256_bytes(encoded) != binding.deployment_digest:
        raise ValueError(f"{filename} changed after the run started")
    return encoded


def _managed_deployment(results_dir: Path, binding: MeasurementBinding) -> ManagedDeployment:
    source = DEPLOYMENT_RECORD_FILENAME
    record = read_deployment_record(
        _bound_file(results_dir, source, MAX_DEPLOYMENT_RECORD_BYTES, binding), str(results_dir / source)
    )
    record.require_benchmark(binding.benchmark)
    plan = record.deployment
    return ManagedDeployment(
        deployment_mode="managed",
        model_release_slug=plan.model_release_slug,
        hf_repository=plan.hf_repository,
        hf_revision=plan.hf_revision,
        framework=plan.framework,
        framework_version=plan.runtime.version,
        framework_commit=resolve_framework_commit(framework_ref(plan.framework, plan.runtime.version)),
        server_launch_command=plan.server_launch_command,
        accelerator_backend=plan.accelerator_backend,
        profile_id=plan.profile_id,
        model_artifact_digest=plan.artifact_manifest_sha256,
        recipe=record.recipe,
        model_size_bytes=plan.artifact_size_bytes,
    )


def _attached_framework_commit(server: AttachedServer) -> str | None:
    """Return the full framework commit, or None when a pinned container identifies the build."""
    if server.framework_commit is not None:
        return resolve_framework_commit(commit_ref(server.framework, server.framework_commit))
    if server.framework_container_reference is not None:
        return None
    return resolve_framework_commit(framework_ref(server.framework, server.framework_version))


def _attached_deployment(results_dir: Path, binding: MeasurementBinding, base_url: str) -> AttachedDeployment:
    if not url_names_loopback_host(base_url):
        raise ValueError("an attached run can be submitted only when the server ran on this computer")
    source = ATTACHED_SERVER_FILENAME
    server = read_attached_server(
        _bound_file(results_dir, source, MAX_ATTACHED_SERVER_BYTES, binding), str(results_dir / source)
    )
    return AttachedDeployment(
        deployment_mode="attached",
        model_release_slug=server.model_release_slug,
        hf_repository=server.hf_repository,
        hf_revision=server.hf_revision,
        framework=server.framework,
        framework_version=server.framework_version,
        framework_commit=_attached_framework_commit(server),
        framework_container_reference=server.framework_container_reference,
        server_launch_command=server.server_launch_command,
        accelerator_backend=server.accelerator_backend,
    )


def _deployment(
    results_dir: Path, binding: MeasurementBinding, base_url: str
) -> ManagedDeployment | AttachedDeployment:
    if binding.deployment_digest is None:
        raise ValueError(
            "this run recorded no server description; for your own server, run again with --attached-server FILE"
        )
    if (results_dir / DEPLOYMENT_RECORD_FILENAME).exists():
        return _managed_deployment(results_dir, binding)
    return _attached_deployment(results_dir, binding, base_url)


def _qualification(results_dir: Path, binding: MeasurementBinding) -> Qualification:
    path = results_dir / QUALIFICATION_FILENAME
    if not path.exists():
        raise ValueError(f"the run has no {QUALIFICATION_FILENAME}; run again with a current client")
    report = load_qualification_file(path)
    if report.run_id != binding.run_id or report.endpoint_model_digest != binding.endpoint_model_digest:
        raise ValueError(f"{QUALIFICATION_FILENAME} belongs to a different run")
    return Qualification(
        synthetic_pack_id=report.synthetic_pack_id,
        outcomes=tuple(
            QualificationOutcome(probe_id=outcome.probe_id, passed=outcome.passed, failure_codes=outcome.failure_codes)
            for outcome in report.outcomes
        ),
    )


def _power(results_dir: Path, binding: MeasurementBinding) -> Power | None:
    path = results_dir / POWER_SUMMARY_FILENAME
    if not path.exists():
        return None
    summary = load_power_summary(path)
    if summary.run_id != binding.run_id:
        raise ValueError(f"{POWER_SUMMARY_FILENAME} belongs to a different run")
    measured = summary.measured
    return Power(
        collector_id=summary.collector_id,
        sampled_power_energy_valid=measured.sampled_power_energy_valid,
        power_coverage=measured.power_coverage,
        power_integration_coverage=measured.power_integration_coverage,
        energy_joules=measured.sampled_power_energy_joules,
        power_w_mean=measured.power_w_time_weighted_mean,
        power_w_max=measured.power_w_maximum,
        temperature_c_max=measured.temperature_c_maximum,
        gpu_utilization_percent_mean=measured.gpu_utilization_percent_time_weighted_mean,
        graphics_clock_mhz_median=measured.graphics_clock_mhz_median,
        memory_clock_mhz_median=measured.memory_clock_mhz_median,
        memory_used_mib_maximum=measured.memory_used_mib_maximum,
    )


def _policy(config: _RecordedConfig) -> CappedOutputPolicy:
    output = config.output_tokens
    if output.policy == "fixed":
        raise ValueError("a run with the fixed output token policy cannot be submitted; use exact or recorded")
    return CappedOutputPolicy(
        client_backend=config.client_backend,
        transport_policy_id=config.transport_policy_id,
        temperature=config.sampling.temperature,
        top_p=config.sampling.top_p,
        top_k=config.sampling.extra_body.top_k,
        min_p=config.sampling.extra_body.min_p,
        reasoning_effort=config.reasoning_effort,
        cache_isolation_enabled=config.cache_isolation.enabled,
        cache_isolation_mode=config.cache_isolation.mode,
        tool_replay_mode=config.tool_replay.mode,
        output_token_policy=output.policy,
        output_token_fallback=output.fallback,
        output_token_margin=output.margin,
    )


def build_submission_request(results_dir: Path) -> SubmissionRequest:
    """Build the body of one finished run from its results folder.

    The body holds only what the service's contract names. The service derives every
    total, distribution, speed, and cache value from the turns, so the body carries raw
    values. A run the service would refuse fails here, with the reason, before any upload.
    The only network request asks GitHub for the full commit of the framework build.
    """
    binding = load_measurement_binding(results_dir / MEASUREMENT_BINDING_FILENAME)
    summary = _read_summary(results_dir)
    _require_bound_run(binding, summary)
    if binding.observed_context_tokens is None:
        raise ValueError("the server did not report its context window, so the run cannot be submitted")
    recorded_turns = _read_turns(results_dir)
    positions = _turn_positions(binding, recorded_turns)
    return SubmissionRequest(
        run_id=binding.run_id,
        privacy_notice_version=PRIVACY_NOTICE_VERSION,
        client=_client(binding.producer),
        benchmark=Benchmark(
            workload_digest=binding.benchmark.suite_digest,
            workload_context_tokens=summary.config.context.full_benchmark_tokens,
            context_tokens=binding.benchmark.context_tokens,
            observed_context_tokens=binding.observed_context_tokens,
        ),
        hardware=_hardware(binding.hardware),
        deployment=_deployment(results_dir, binding, summary.config.base_url),
        policy=_policy(summary.config),
        run=Run(wall_duration_ms=summary.wall_duration_ms, observer_duration_ms=summary.observer.duration_ms),
        turns=tuple(_contract_turn(turn, position) for turn, position in zip(recorded_turns, positions, strict=True)),
        qualification=_qualification(results_dir, binding),
        power=_power(results_dir, binding),
    )


def encode_submission(request: SubmissionRequest) -> bytes:
    """Encode one body as readable JSON and check it against the service's pinned spec."""
    encoded = pretty_json_bytes(normalize_json(request.model_dump(mode="json")))
    validate_against_spec(encoded)
    return encoded


class PreparedSubmission(BaseModel, frozen=True):
    """Hold one prepared body: the exact bytes to send and the request they parse to."""

    request: SubmissionRequest
    encoded: bytes


def write_prepared_submission(results_dir: Path, output_path: Path, encoded: bytes) -> None:
    """Write the body to a new file outside the results folder, so a person can read what will be sent."""
    validate_public_output_path(results_dir, output_path, "submission file")
    write_new_file(NewFile(path=output_path, data=encoded))


def read_prepared_submission(path: Path) -> PreparedSubmission:
    """Read a prepared body and check it against the contract, the pinned spec, and the current notice.

    A retry must send the same content, so the bytes are sent exactly as they were written.
    """
    encoded = read_bounded_file(path, MAX_REQUEST_BYTES, label="submission file")
    request = read_record(SubmissionRequest, encoded, str(path))
    validate_against_spec(encoded)
    if request.privacy_notice_version != PRIVACY_NOTICE_VERSION:
        raise ValueError(
            f"{path} accepts privacy notice {request.privacy_notice_version}, but this client shows notice "
            f"{PRIVACY_NOTICE_VERSION}; prepare the submission again. If the service already holds this run, "
            "it keeps that copy and refuses the new one with idempotency_conflict"
        )
    return PreparedSubmission(request=request, encoded=encoded)
