"""State what an interface asks a replay controller for, and what it may answer.

A request is validated before anything is launched, so a refusal names a
block code the interface can explain rather than an exception it must guess at.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from agentperf_local.client.backends import ClientBackend
from agentperf_local.client.endpoint import normalize_base_url, url_names_loopback_host
from agentperf_local.client.rust_client import validate_rustcore_available
from agentperf_local.common.durable_files import nearest_existing_ancestor, validate_new_file_paths
from agentperf_local.common.identity import validate_digest
from agentperf_local.deployment.catalog import DeploymentFramework, ModelCandidate
from agentperf_local.deployment.context_policy import (
    ContextBelowReplayFloor,
    require_replay_context_floor,
    resolve_context_tokens,
)
from agentperf_local.deployment.endpoint_probes import ContextProbeResult
from agentperf_local.deployment.frameworks import FrameworkOffer
from agentperf_local.deployment.managed import (
    DEFAULT_DEPLOYMENT_PORT,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    DEPLOYMENT_LOG_FILENAME,
    DEPLOYMENT_RECORD_FILENAME,
    MAXIMUM_PORT,
    MINIMUM_USER_PORT,
)
from agentperf_local.deployment.managed_run import ManagedRunObserver
from agentperf_local.deployment.model_cache import (
    default_model_cache_root,
)
from agentperf_local.deployment.qualification import (
    QUALIFICATION_FILENAME,
)
from agentperf_local.provenance.benchmark import (
    MEASUREMENT_BINDING_FILENAME,
)
from agentperf_local.provenance.context import below_benchmark_context
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.replay.config import API_KEY_ENV_PATTERN, OutputTokenPolicy, read_api_key_env
from agentperf_local.reports.reporting import (
    ArtifactPaths,
    validate_run_artifact_output,
)
from agentperf_local.telemetry.power import (
    POWER_SUMMARY_FILENAME,
    TELEMETRY_FILENAME,
)
from agentperf_local.tui.evidence import (
    EndpointScope,
    SelectionKind,
)
from agentperf_local.workload.schema import load_manifest

DEVICE_SELECTION_REQUIRED_MESSAGE = "Choose a device to run the model on."


RUN_DIRECTORY_PREFIX = "run-"


RUN_DIRECTORY_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


SERVER_UNREACHABLE_MESSAGE = (
    "The server at your URL did not answer. Check that it is running and that the URL is right."
)


SERVER_REFUSED_MESSAGE = "The server answered with an HTTP error. Check the URL path and the API key."


class EndpointProblem(RuntimeError):
    """Report an attached server that cannot take the replay, in words safe to show."""


def next_run_directory(results_root: Path, *, now: Callable[[], datetime] = datetime.now) -> Path:
    """Name one unclaimed run subdirectory under the chosen results folder.

    The name is only reserved by the run's own O_EXCL file writes, so preflight
    still rejects a directory that gains files after this choice.
    """
    stamp = now().strftime(RUN_DIRECTORY_TIMESTAMP_FORMAT)
    candidate = results_root / f"{RUN_DIRECTORY_PREFIX}{stamp}"
    # A second run inside the same clock second takes an ordinal suffix, starting at 2
    # so the plain timestamped name keeps naming that second's first run.
    ordinal = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = results_root / f"{RUN_DIRECTORY_PREFIX}{stamp}-{ordinal}"
        ordinal += 1
    return candidate


class PreflightBlockCode(StrEnum):
    """Name a safe preflight failure category."""

    INPUTS_INVALID = "inputs-invalid"
    CLIENT_UNAVAILABLE = "client-unavailable"
    ENDPOINT_NEEDS_HTTPS = "endpoint-needs-https"
    URL_INVALID = "url-invalid"
    ENDPOINT_MODEL_EMPTY = "endpoint-model-empty"
    OUTPUT_DIR_MISSING = "output-dir-missing"
    OUTPUT_DIR_USED = "output-dir-used"
    MANIFEST_UNREADABLE = "manifest-unreadable"
    API_KEY_ENV_UNSET = "api-key-env-unset"
    DEPLOYMENT_UNAVAILABLE = "deployment-unavailable"
    CONTEXT_BELOW_REPLAY_FLOOR = "context-below-replay-floor"


class SetupProblem(ValueError):
    """Report one rejected setup with a safe cause code."""

    block_code: PreflightBlockCode

    def __init__(self, message: str, *, block_code: PreflightBlockCode) -> None:
        super().__init__(message)
        self.block_code = block_code


class ManagedLaunchSettingsProblem(ValueError):
    """Report managed launch parameters that no setup field can correct."""


class ManagedDeviceSelectionRequired(ValueError):
    """Report that this computer detected several accelerators and none was chosen."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ManagedDeviceOption:
    """Describe one detected accelerator a managed launch can be pinned to."""

    index: int
    name: str
    memory_bytes: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class SafeHardwareSummary:
    """Store identifier-free host facts for preflight display."""

    operating_system: str
    architecture: str
    accelerator_count: int
    accelerator_name: str | None
    accelerator_memory_bytes: int | None
    warning_count: int

    @classmethod
    def from_snapshot(cls, snapshot: HardwareSnapshot) -> SafeHardwareSummary:
        """Reduce one private-safe probe snapshot for the TUI."""
        accelerator = snapshot.accelerators[0] if len(snapshot.accelerators) == 1 else None
        accelerator_memory_bytes = accelerator.memory_bytes if accelerator is not None else None
        if accelerator is not None and accelerator.api == "Metal" and accelerator_memory_bytes is None:
            # Apple GPUs use unified memory, so the host total is the capacity relevant to deployment checks.
            accelerator_memory_bytes = snapshot.memory_bytes
        return cls(
            operating_system=snapshot.operating_system,
            architecture=snapshot.architecture,
            accelerator_count=len(snapshot.accelerators),
            accelerator_name=accelerator.name if accelerator is not None else None,
            accelerator_memory_bytes=accelerator_memory_bytes,
            warning_count=len(snapshot.warnings),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ManagedModelAvailability:
    """Describe managed deployment choices for one model on this computer."""

    hardware: SafeHardwareSummary
    offers: tuple[FrameworkOffer, ...]
    reason: str | None
    device_selection_required: bool = False

    @property
    def deployable_offers(self) -> tuple[FrameworkOffer, ...]:
        """Return declared-compatible installed launch candidates with sufficient memory."""
        return tuple(offer for offer in self.offers if offer.installed and offer.memory_fit is True)

    @property
    def can_deploy(self) -> bool:
        """Return whether at least one framework is ready for a verified launch attempt."""
        return bool(self.deployable_offers)


@dataclass(frozen=True, slots=True, kw_only=True)
class ManagedDeploymentChoice:
    """Bind one catalog model to a local framework choice."""

    candidate: ModelCandidate
    catalog_as_of: str
    # The catalog file this candidate came from. It travels into the deployment record
    # so a submission names the release its recipe belongs to.
    catalog_digest: str
    framework: DeploymentFramework
    # An index is only checked against the detected accelerators at bind time, so a stale
    # choice fails where the launch snapshot is read rather than where the form was filled.
    device_index: int | None = None
    # None launches the recipe's full benchmark context; a smaller value records a reduced run.
    context_tokens: int | None = None
    cache_root: Path = field(default_factory=default_model_cache_root)
    port: int = DEFAULT_DEPLOYMENT_PORT
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """Reject choices that are not present in the model recipe."""
        validate_digest(self.catalog_digest, "catalog_digest")
        deployment = self.candidate.deployment
        if self.framework not in deployment.frameworks:
            raise ValueError("managed framework is not available for the selected model")
        resolve_context_tokens(deployment, self.context_tokens)
        if self.port < MINIMUM_USER_PORT or self.port > MAXIMUM_PORT:
            raise ManagedLaunchSettingsProblem(
                f"the --port value must be between {MINIMUM_USER_PORT} and {MAXIMUM_PORT}"
            )
        if self.startup_timeout_seconds <= 0:
            raise ManagedLaunchSettingsProblem("the --startup-timeout-seconds value must be positive")

    @property
    def resolved_context_tokens(self) -> int:
        """Return the context this choice launches at, defaulting to the full benchmark context."""
        return resolve_context_tokens(self.candidate.deployment, self.context_tokens)

    @property
    def reduced_context(self) -> bool:
        """Return whether this choice launches below the full benchmark context."""
        return below_benchmark_context(self.resolved_context_tokens)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayRequest:
    """Describe one Explorer replay without storing an API key."""

    manifest_path: Path
    output_dir: Path
    base_url: str
    endpoint_model: str
    api_key_env: str | None = None
    client_backend: ClientBackend = "python"
    selection_kind: SelectionKind = SelectionKind.CUSTOM_ENDPOINT
    catalog_profile_id: str | None = None
    catalog_digest: str | None = None
    candidate_revision: str | None = None
    managed_deployment: ManagedDeploymentChoice | None = None

    def __post_init__(self) -> None:
        """Reject invalid input before execution starts."""
        if self.api_key_env is not None and not API_KEY_ENV_PATTERN.fullmatch(self.api_key_env):
            raise ValueError("API key environment variable name is invalid")
        if not self.endpoint_model:
            raise SetupProblem(
                "endpoint model must not be empty",
                block_code=PreflightBlockCode.ENDPOINT_MODEL_EMPTY,
            )
        scheme = urlsplit(self.normalized_base_url).scheme
        # A plain-http endpoint on the local network stays allowed while no key travels
        # with the request. Sending a key over cleartext would expose it, so that needs HTTPS.
        if (
            self.api_key_env is not None
            and self.endpoint_scope is EndpointScope.NON_LOOPBACK_NAME
            and scheme != "https"
        ):
            raise SetupProblem(
                "a non-loopback endpoint with an API key must use HTTPS",
                block_code=PreflightBlockCode.ENDPOINT_NEEDS_HTTPS,
            )
        if self.managed_deployment is not None:
            candidate = self.managed_deployment.candidate
            if self.api_key_env is not None:
                raise ValueError("managed localhost deployments do not accept an API key")
            if self.catalog_profile_id != candidate.profile_id or self.candidate_revision != candidate.hf_revision:
                raise ValueError("managed deployment choice must match the selected catalog candidate")
            expected_base_url = f"http://127.0.0.1:{self.managed_deployment.port}/v1"
            if self.normalized_base_url != expected_base_url:
                raise ValueError("managed deployment URL must match its owned localhost port")
            if self.endpoint_model != candidate.profile_id:
                raise ValueError("managed deployment model must use its recipe profile_id before launch")

    @property
    def endpoint_is_loopback(self) -> bool:
        """Return whether the endpoint URL names a loopback host."""
        return self.endpoint_scope is EndpointScope.LOOPBACK_NAME

    @property
    def normalized_base_url(self) -> str:
        """Return the same canonical base URL used by the measured client."""
        try:
            return normalize_base_url(self.base_url)
        except ValueError as error:
            raise SetupProblem("server URL is not valid", block_code=PreflightBlockCode.URL_INVALID) from error

    @property
    def endpoint_scope(self) -> EndpointScope:
        """Classify the URL name without making a serving-locality claim."""
        if url_names_loopback_host(self.normalized_base_url):
            return EndpointScope.LOOPBACK_NAME
        return EndpointScope.NON_LOOPBACK_NAME


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayPreflight:
    """Store safe checks that do not contact the endpoint."""

    ready: bool
    manifest_tasks: int
    manifest_turns: int
    endpoint_scope: EndpointScope
    reason: str | None
    block_code: PreflightBlockCode | None = None
    hardware: SafeHardwareSummary | None = None

    @property
    def endpoint_is_loopback(self) -> bool:
        """Return whether the configured URL names a loopback host."""
        return self.endpoint_scope is EndpointScope.LOOPBACK_NAME


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedReplayInputs:
    """Store replay counts and context demand after local freshness validation."""

    manifest_tasks: int
    manifest_turns: int
    required_context_tokens: int | None


def require_api_key_value(name: str) -> str:
    """Return the named API key value, or reject the setup when it cannot be sent."""
    value = read_api_key_env(name)
    if value is None:
        raise SetupProblem(
            f"environment variable {name} is not set",
            block_code=PreflightBlockCode.API_KEY_ENV_UNSET,
        )
    return value


def _validate_output_dir_writable(output_dir: Path) -> None:
    """Reject a read-only destination here so it cannot resurface as an endpoint failure."""
    try:
        root = nearest_existing_ancestor(output_dir)
    except ValueError:
        # A missing or symlinked ancestor leaves no real directory this run could write into.
        root = None
    if root is None or not os.access(root, os.W_OK | os.X_OK):
        raise SetupProblem("Results folder is not writable.", block_code=PreflightBlockCode.INPUTS_INVALID)


def validate_replay_inputs(request: ReplayRequest) -> ValidatedReplayInputs:
    """Validate replay inputs without probing hardware or contacting the endpoint."""
    if request.client_backend == "rust":
        try:
            validate_rustcore_available()
        except RuntimeError as error:
            raise SetupProblem(
                "Rust client unavailable; install the rust extra or choose Python",
                block_code=PreflightBlockCode.CLIENT_UNAVAILABLE,
            ) from error
    if request.api_key_env is not None:
        require_api_key_value(request.api_key_env)
    try:
        manifest = load_manifest(request.manifest_path)
    except (OSError, ValueError) as error:
        raise SetupProblem(
            "replay manifest could not be read",
            block_code=PreflightBlockCode.MANIFEST_UNREADABLE,
        ) from error
    _validate_output_dir_writable(request.output_dir)
    try:
        validate_run_artifact_output(request.output_dir)
        validate_new_file_paths((request.output_dir / MEASUREMENT_BINDING_FILENAME,))
    except FileExistsError as error:
        raise SetupProblem(
            "output directory already holds run files",
            block_code=PreflightBlockCode.OUTPUT_DIR_USED,
        ) from error
    return ValidatedReplayInputs(
        manifest_tasks=len(manifest.tasks),
        manifest_turns=sum(task.model_calls for task in manifest.tasks),
        required_context_tokens=manifest.required_context_tokens,
    )


def validate_managed_replay_inputs(request: ReplayRequest) -> ValidatedReplayInputs:
    """Validate the additional private files used by an owned deployment."""
    if request.managed_deployment is None:
        raise ValueError("managed replay request is missing its deployment choice")
    inputs = validate_replay_inputs(request)
    try:
        require_replay_context_floor(inputs.required_context_tokens, request.managed_deployment.resolved_context_tokens)
    except ContextBelowReplayFloor as error:
        raise SetupProblem(str(error), block_code=PreflightBlockCode.CONTEXT_BELOW_REPLAY_FLOOR) from error
    validate_new_file_paths(
        (
            request.output_dir / DEPLOYMENT_RECORD_FILENAME,
            request.output_dir / DEPLOYMENT_LOG_FILENAME,
            request.output_dir / TELEMETRY_FILENAME,
            request.output_dir / POWER_SUMMARY_FILENAME,
            request.output_dir / QUALIFICATION_FILENAME,
        )
    )
    return inputs


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayExecution:
    """Store the local artifacts and headline timings produced by one replay."""

    artifacts: ArtifactPaths
    output_dir: Path
    output_tokens_per_second: float | None
    ttft_p50_ms: float | None
    e2e_p50_ms: float | None
    failed_turns: int
    total_turns: int
    deployment_record: Path | None = None
    gpu_startup_verified: bool = False
    power_summary: Path | None = None
    qualification: Path | None = None
    output_token_policy: OutputTokenPolicy = "exact"

    @property
    def comparable_policy(self) -> bool:
        """Return whether the run measured under the exact policy, the one results are compared on."""
        return self.output_token_policy == "exact"

    @property
    def success(self) -> bool:
        """Return whether every turn succeeded."""
        return self.failed_turns == 0


class ReplayController(Protocol):
    """Execute one TUI replay through a replaceable public boundary."""

    def preflight(self, request: ReplayRequest) -> ReplayPreflight:
        """Validate local inputs without contacting the endpoint."""
        ...

    def probe_endpoint(self, request: ReplayRequest) -> ContextProbeResult:
        """Send one GET /models to the configured server and report what it answered."""
        ...

    def detects_ollama(self, request: ReplayRequest) -> bool:
        """Return whether the configured server is Ollama, which drops ignore_eos."""
        ...

    async def execute(self, request: ReplayRequest, observer: TuiReplayObserver) -> ReplayExecution:
        """Run one replay and write its private artifacts."""
        ...


class ManagedReplayController(Protocol):
    """Detect and execute an owned model deployment for the TUI."""

    def hardware_summary(self, device_index: int | None = None) -> SafeHardwareSummary:
        """Return the host facts detected once at launch, narrowed to one chosen device."""
        ...

    def device_options(self) -> tuple[ManagedDeviceOption, ...]:
        """Return every detected accelerator a managed launch can be pinned to."""
        ...

    def availability(
        self,
        candidate: ModelCandidate,
        device_index: int | None = None,
        *,
        context_tokens: int | None = None,
        replay_floor_tokens: int | None = None,
    ) -> ManagedModelAvailability:
        """Return compatible local framework choices for one model on one device and context."""
        ...

    def preflight(self, request: ReplayRequest) -> ReplayPreflight:
        """Validate managed local inputs without downloading or launching."""
        ...

    async def execute(self, request: ReplayRequest, observer: TuiReplayObserver) -> ReplayExecution:
        """Own the model server for one replay and stop it afterward."""
        ...


class TuiReplayObserver(ManagedRunObserver, Protocol):
    """Receive replay boundaries and steps, disable cancellation before report commits, and show server logs."""

    def on_server_log(self, lines: tuple[str, ...]) -> None:
        """Report new lines from the owned server's log on the event loop."""
        ...
