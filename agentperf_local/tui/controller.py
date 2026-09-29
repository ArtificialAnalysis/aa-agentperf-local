"""Provide the typed replay boundary used by the full-screen TUI."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr

from agentperf_local.client.backends import ClientBackend
from agentperf_local.common.models import error_text
from agentperf_local.deployment.catalog import ModelCandidate, ModelDeployment
from agentperf_local.deployment.context_policy import (
    largest_fitting_reduced_context,
    largest_offered_fitting_context,
    resolve_context_tokens,
    smaller_offered_context_fits,
)
from agentperf_local.deployment.endpoint_probes import (
    ContextProbeResult,
    IgnoreEosProbeResult,
    is_ollama_endpoint,
    probe_ignore_eos,
    probe_served_context_tokens,
)
from agentperf_local.deployment.frameworks import (
    FrameworkOffer,
    framework_offers,
)
from agentperf_local.deployment.managed import (
    DEPLOYMENT_LOG_FILENAME,
    BoundDeploymentDevice,
    bind_snapshot_to_device,
)
from agentperf_local.deployment.managed_run import (
    ManagedRunInputs,
    RunActivity,
    RunActivityKind,
    discard_unfinished_output,
    run_managed_replay,
    write_run_artifacts_uncancellable,
)
from agentperf_local.provenance.benchmark import (
    BENCHMARK_CONTEXT_TOKENS,
    MEASUREMENT_BINDING_FILENAME,
    create_attached_submission_context,
    create_measurement_binding,
    write_measurement_binding,
)
from agentperf_local.provenance.context import ContextObservationReason, RunContextFacts
from agentperf_local.provenance.hardware import HardwareSnapshot, collect_hardware_snapshot
from agentperf_local.replay.config import OutputTokenPolicy, RunConfig
from agentperf_local.replay.runner import run_manifest
from agentperf_local.reports.progress import ANSI_SEQUENCE_PATTERN
from agentperf_local.reports.reporting import (
    run_output_tokens_per_second,
    run_timing_summary,
)
from agentperf_local.tui.replay_contract import (
    DEVICE_SELECTION_REQUIRED_MESSAGE,
    PLATFORM_MISMATCH_REASON,
    SERVER_REFUSED_MESSAGE,
    SERVER_UNREACHABLE_MESSAGE,
    EndpointProblem,
    ManagedDeploymentChoice,
    ManagedDeviceOption,
    ManagedDeviceSelectionRequired,
    ManagedModelAvailability,
    PreflightBlockCode,
    ReplayExecution,
    ReplayPreflight,
    ReplayRequest,
    SafeHardwareSummary,
    SetupProblem,
    TuiReplayObserver,
    require_api_key_value,
    validate_managed_replay_inputs,
    validate_replay_inputs,
)

# The owned server's log is tailed from disk on a modest cadence, off the event loop; a
# poll hands the UI at most one bounded batch so a chatty server cannot flood it.
SERVER_LOG_POLL_SECONDS = 0.25
SERVER_LOG_MAX_LINES_PER_POLL = 200
SERVER_LOG_MAX_LINE_CHARS = 400


async def _execute_validated_replay(
    request: ReplayRequest,
    observer: TuiReplayObserver,
    *,
    output_token_policy: OutputTokenPolicy,
    context_probe: ContextProbeResult,
    run_id: str,
) -> ReplayExecution:
    """Run one attached replay under the run identifier its measurement binding already carries."""
    api_key = None if request.api_key_env is None else require_api_key_value(request.api_key_env)
    config = RunConfig(
        base_url=request.normalized_base_url,
        model=request.endpoint_model,
        api_key=api_key,
        client_backend=request.client_backend,
        output_token_policy=output_token_policy,
        tool_choice=request.tool_choice,
    )
    # Preflight stays offline; the only pre-replay contact is the /models probe whose
    # outcome arrives here, so the observation is bound before the run.
    run_context = RunContextFacts(
        requested_tokens=BENCHMARK_CONTEXT_TOKENS,
        observed_tokens=context_probe.observed_tokens,
        observed_reason=context_probe.reason,
    )
    observer.on_activity(RunActivity(kind=RunActivityKind.REPLAY_STARTING))
    result = await run_manifest(request.manifest_path, config, observer=observer)
    await observer.on_finalizing()
    artifacts = await write_run_artifacts_uncancellable(result, request.output_dir, config, run_context, run_id)
    timings = run_timing_summary(result)
    return ReplayExecution(
        artifacts=artifacts,
        output_dir=request.output_dir,
        output_tokens_per_second=run_output_tokens_per_second(result),
        ttft_p50_ms=timings.ttft_p50_ms,
        e2e_p50_ms=timings.e2e_p50_ms,
        failed_turns=timings.failed_turns,
        total_turns=timings.total_turns,
        output_token_policy=output_token_policy,
    )


def _display_log_line(raw: bytes) -> str:
    """Make one server log line safe to draw: decoded, colour codes and control characters dropped, length bounded."""
    plain = ANSI_SEQUENCE_PATTERN.sub("", raw.decode("utf-8", errors="replace"))
    text = "".join(character for character in plain if character.isprintable() or character == "\t").rstrip()
    if len(text) <= SERVER_LOG_MAX_LINE_CHARS:
        return text
    return f"{text[: SERVER_LOG_MAX_LINE_CHARS - 1]}…"


def _read_new_log_lines(path: Path, position: int, *, flush_partial: bool = False) -> tuple[tuple[str, ...], int]:
    """Return the newest lines written after position, and the position just past what was read.

    A partial trailing line waits for its newline unless flush_partial is set, which
    the final drain uses because a stopped server never finishes its last line. A
    burst larger than one batch keeps only its newest lines: the log view caps what
    it shows anyway, and carrying a growing backlog would re-read it on every poll
    of the machine being measured.
    """
    try:
        with path.open("rb") as source:
            source.seek(position)
            data = source.read()
    except OSError:
        return (), position
    complete, _, partial = data.rpartition(b"\n")
    raw_lines = complete.split(b"\n") if complete or data.endswith(b"\n") else []
    consumed = len(data) if flush_partial else len(data) - len(partial)
    if flush_partial and partial:
        raw_lines.append(partial)
    lines = tuple(text for raw in raw_lines[-SERVER_LOG_MAX_LINES_PER_POLL:] if (text := _display_log_line(raw)))
    return lines, position + consumed


def _tail_server_log(path: Path, deliver: Callable[[tuple[str, ...]], None], stop: threading.Event) -> None:
    """Follow the owned server's log until stopped, then drain what it wrote while shutting down."""
    position = 0
    while True:
        lines, position = _read_new_log_lines(path, position)
        if lines:
            deliver(lines)
        if stop.wait(SERVER_LOG_POLL_SECONDS):
            lines, position = _read_new_log_lines(path, position, flush_partial=True)
            while lines:
                deliver(lines)
                lines, position = _read_new_log_lines(path, position, flush_partial=True)
            return


async def _stop_log_tail(tail: asyncio.Task[None], stop: threading.Event) -> None:
    """Let the tail drain the final server lines even when the replay worker is cancelled."""
    stop.set()
    try:
        await asyncio.shield(tail)
    except asyncio.CancelledError:
        await tail
        raise


def _require_reachable(probe: ContextProbeResult) -> None:
    """Refuse to start a replay against a server that did not answer the probe."""
    if probe.endpoint_answered:
        return
    if probe.reason is ContextObservationReason.ENDPOINT_UNREACHABLE:
        raise EndpointProblem(SERVER_UNREACHABLE_MESSAGE)
    raise EndpointProblem(SERVER_REFUSED_MESSAGE)


@dataclass(slots=True, kw_only=True)
class LocalReplayController:
    """Run one attached endpoint replay with the existing measured client."""

    hardware_collector: Callable[[], HardwareSnapshot] = collect_hardware_snapshot
    context_prober: Callable[[str, str, SecretStr | None], ContextProbeResult] = probe_served_context_tokens
    ignore_eos_prober: Callable[[str, str, ClientBackend, SecretStr | None], Awaitable[IgnoreEosProbeResult]] = (
        probe_ignore_eos
    )
    ollama_detector: Callable[[str, SecretStr | None], bool] = is_ollama_endpoint

    def preflight(self, request: ReplayRequest) -> ReplayPreflight:
        """Validate the manifest and fresh artifact destinations."""

        def blocked(reason: str, code: PreflightBlockCode) -> ReplayPreflight:
            """Refuse this setup before any hardware is probed."""
            return ReplayPreflight(
                ready=False,
                manifest_tasks=0,
                manifest_turns=0,
                endpoint_scope=request.endpoint_scope,
                reason=reason,
                block_code=code,
                hardware=None,
            )

        try:
            inputs = validate_replay_inputs(request)
        except SetupProblem as error:
            return blocked(str(error), error.block_code)
        except (OSError, ValueError) as error:
            return blocked(error_text(error), PreflightBlockCode.INPUTS_INVALID)
        hardware = SafeHardwareSummary.from_snapshot(self.hardware_collector())
        return ReplayPreflight(
            ready=True,
            manifest_tasks=inputs.manifest_tasks,
            manifest_turns=inputs.manifest_turns,
            endpoint_scope=request.endpoint_scope,
            reason=None,
            block_code=None,
            hardware=hardware,
        )

    def probe_endpoint(self, request: ReplayRequest) -> ContextProbeResult:
        """Send one GET /models with the run's key, the same call the replay repeats before it starts."""
        api_key = None if request.api_key_env is None else require_api_key_value(request.api_key_env)
        return self.context_prober(request.normalized_base_url, request.endpoint_model, api_key)

    def detects_ollama(self, request: ReplayRequest) -> bool:
        """Send one GET /api/version, cheap enough for the consent step and again before the run."""
        api_key = None if request.api_key_env is None else require_api_key_value(request.api_key_env)
        return self.ollama_detector(request.normalized_base_url, api_key)

    async def execute(self, request: ReplayRequest, observer: TuiReplayObserver) -> ReplayExecution:
        """Check the server once more, then run the measured replay and commit private local reports."""
        await asyncio.to_thread(validate_replay_inputs, request)
        observer.on_activity(RunActivity(kind=RunActivityKind.SETUP_CHECKED))
        # The consent-time check may be minutes old, so the observation bound into the
        # run is taken here, immediately before the replay, as the CLI does.
        probe = await asyncio.to_thread(self.probe_endpoint, request)
        observer.on_activity(RunActivity(kind=RunActivityKind.SERVER_CHECKED, context_probe=probe))
        _require_reachable(probe)
        context = await asyncio.to_thread(
            create_attached_submission_context,
            request.manifest_path,
            request.endpoint_model,
            model_semantics_id=request.catalog_profile_id,
        )
        snapshot = await asyncio.to_thread(self.hardware_collector)
        binding = await asyncio.to_thread(
            create_measurement_binding,
            context,
            request.manifest_path,
            request.endpoint_model,
            snapshot,
            observed_context_tokens=probe.observed_tokens,
        )
        # The exact policy needs an endpoint that generates past end-of-sequence. Asking
        # here, rather than reading N short turns as N failures, names the cause once,
        # and asking before the binding write leaves nothing on disk if the wait is cancelled.
        # Ollama is known to drop ignore_eos, so its cheap identity check stands in for the
        # generation probe, and the consent step has already warned the user.
        output_token_policy: OutputTokenPolicy
        if await asyncio.to_thread(self.detects_ollama, request):
            output_token_policy = "recorded"
            observer.on_activity(RunActivity(kind=RunActivityKind.OUTPUT_POLICY_CHOSEN, ollama_endpoint=True))
        else:
            capability = await self.ignore_eos_prober(
                request.normalized_base_url,
                request.endpoint_model,
                request.client_backend,
                None if request.api_key_env is None else require_api_key_value(request.api_key_env),
            )
            output_token_policy = capability.output_token_policy
            observer.on_activity(RunActivity(kind=RunActivityKind.OUTPUT_POLICY_CHOSEN, ignore_eos_probe=capability))
        measurement_path = request.output_dir / MEASUREMENT_BINDING_FILENAME
        created_output_dir = not await asyncio.to_thread(request.output_dir.exists)
        await asyncio.to_thread(write_measurement_binding, measurement_path, binding)
        try:
            return await _execute_validated_replay(
                request,
                observer,
                output_token_policy=output_token_policy,
                context_probe=probe,
                run_id=binding.run_id,
            )
        except BaseException:
            discard_unfinished_output(measurement_path, request.output_dir, created_output_dir=created_output_dir)
            raise


def _insufficient_memory_reason(
    deployment: ModelDeployment,
    context_tokens: int | None,
    available_memory_bytes: int | None,
    replay_floor_tokens: int | None = None,
) -> str:
    """Explain a memory refusal, pointing at the context picker only when it holds a fix."""
    requested = resolve_context_tokens(deployment, context_tokens)
    if smaller_offered_context_fits(deployment, requested, available_memory_bytes, replay_floor_tokens):
        if requested < deployment.context_tokens:
            return f"Not enough accelerator memory for a {requested:,}-token context. Choose a smaller context."
        return (
            f"Not enough accelerator memory for the full {deployment.context_tokens:,}-token context. "
            "Choose a reduced context — reduced runs are recorded separately from full-context results."
        )
    # No offered smaller rung fits, so pointing at the picker would only mislead. When the
    # replay's floor is what emptied the picker, name both numbers so the dead end is honest.
    largest_fitting = largest_fitting_reduced_context(deployment, available_memory_bytes)
    if replay_floor_tokens is not None and largest_fitting is not None:
        return (
            f"This replay can not run on this computer: it needs at least {replay_floor_tokens:,} tokens "
            f"of context and this device fits at most {largest_fitting:,}."
        )
    return "The detected accelerator does not have enough memory for this model."


@dataclass(slots=True, kw_only=True)
class LocalManagedReplayController:
    """Own one catalog-driven local model server for a TUI replay."""

    catalog_as_of: str
    hardware: HardwareSnapshot
    offer_collector: Callable[[HardwareSnapshot, ModelCandidate, int | None], tuple[FrameworkOffer, ...]] = (
        framework_offers
    )

    @classmethod
    def detect(cls, catalog_as_of: str) -> LocalManagedReplayController:
        """Collect hardware once before presenting deployment choices."""
        return cls(catalog_as_of=catalog_as_of, hardware=collect_hardware_snapshot())

    def hardware_summary(self, device_index: int | None = None) -> SafeHardwareSummary:
        """Return the launch-time host facts, narrowed to one accelerator once a device is chosen."""
        return SafeHardwareSummary.from_snapshot(self._bound_device(device_index).snapshot)

    def device_options(self) -> tuple[ManagedDeviceOption, ...]:
        """List every detected accelerator in the order the launch snapshot reports them."""
        return tuple(
            ManagedDeviceOption(index=index, name=accelerator.name, memory_bytes=accelerator.memory_bytes)
            for index, accelerator in enumerate(self.hardware.accelerators)
        )

    def _bound_device(self, device_index: int | None) -> BoundDeploymentDevice:
        """Reduce the launch snapshot to the chosen accelerator and its pinning environment."""
        return bind_snapshot_to_device(self.hardware, device_index)

    def _device_selection_required(self, device_index: int | None) -> bool:
        """Report whether this computer offers a choice that nobody has made yet."""
        # Binding without an index keeps every accelerator, so the count is the only signal here.
        return device_index is None and len(self.hardware.accelerators) > 1

    def availability(
        self,
        candidate: ModelCandidate,
        device_index: int | None = None,
        *,
        context_tokens: int | None = None,
        replay_floor_tokens: int | None = None,
    ) -> ManagedModelAvailability:
        """Return safe compatibility facts for one catalog candidate on one device and context."""
        return self._bound_availability(
            candidate,
            self._bound_device(device_index),
            device_index,
            context_tokens,
            replay_floor_tokens,
        )

    def _bound_availability(
        self,
        candidate: ModelCandidate,
        bound: BoundDeploymentDevice,
        device_index: int | None,
        context_tokens: int | None = None,
        replay_floor_tokens: int | None = None,
    ) -> ManagedModelAvailability:
        """Return the same compatibility facts for a device that is already bound."""
        safe_hardware = SafeHardwareSummary.from_snapshot(bound.snapshot)
        if self._device_selection_required(device_index):
            # The offer collector rejects a many-accelerator snapshot with command-line advice,
            # so the device question is answered here before any compatibility question is asked.
            return ManagedModelAvailability(
                hardware=safe_hardware,
                offers=(),
                reason=DEVICE_SELECTION_REQUIRED_MESSAGE,
                device_selection_required=True,
            )
        try:
            offers = self.offer_collector(bound.snapshot, candidate, context_tokens)
        except ValueError as error:
            return ManagedModelAvailability(hardware=safe_hardware, offers=(), reason=error_text(error))
        deployable = tuple(offer for offer in offers if offer.installed and offer.memory_fit is True)
        reduced_context_tokens = None
        if deployable:
            reason = None
        elif not offers:
            reason = PLATFORM_MISMATCH_REASON
        elif all(offer.memory_fit is False for offer in offers):
            reason = _insufficient_memory_reason(
                candidate.deployment,
                context_tokens,
                offers[0].available_memory_bytes,
                replay_floor_tokens,
            )
            reduced_context_tokens = largest_offered_fitting_context(
                candidate.deployment,
                resolve_context_tokens(candidate.deployment, context_tokens),
                offers[0].available_memory_bytes,
                replay_floor_tokens,
            )
        elif all(offer.memory_fit is None for offer in offers):
            reason = "Accelerator memory could not be verified for this model."
        else:
            installation_hints = tuple(offer.installation_hint for offer in offers if not offer.installed)
            reason = installation_hints[0] if installation_hints else "No compatible framework is ready to launch."
        return ManagedModelAvailability(
            hardware=safe_hardware, offers=offers, reason=reason, reduced_context_tokens=reduced_context_tokens
        )

    def _selected_offer(self, choice: ManagedDeploymentChoice, bound: BoundDeploymentDevice) -> FrameworkOffer:
        """Return the installed framework offer one choice names on its already-bound device."""
        if choice.catalog_as_of != self.catalog_as_of:
            raise ValueError("managed deployment choice does not match the loaded catalog")
        availability = self._bound_availability(choice.candidate, bound, choice.device_index, choice.context_tokens)
        if availability.device_selection_required:
            raise ManagedDeviceSelectionRequired(DEVICE_SELECTION_REQUIRED_MESSAGE)
        offer = next((item for item in availability.offers if item.framework == choice.framework), None)
        if offer is None:
            raise ValueError("selected framework is not compatible with the detected accelerator")
        if not offer.installed:
            raise ValueError(offer.installation_hint)
        if offer.memory_fit is not True:
            raise ValueError("detected accelerator memory is unknown or below the model requirement")
        return offer

    def preflight(self, request: ReplayRequest) -> ReplayPreflight:
        """Validate managed inputs without downloading or starting a process."""
        bound: BoundDeploymentDevice | None = None

        def blocked(reason: str, code: PreflightBlockCode) -> ReplayPreflight:
            """Refuse this setup, describing the bound device once there is one."""
            # An unusable device index binds nothing, so the whole computer is described instead
            # and the index itself is reported as the block reason.
            hardware = self.hardware_summary() if bound is None else SafeHardwareSummary.from_snapshot(bound.snapshot)
            return ReplayPreflight(
                ready=False,
                manifest_tasks=0,
                manifest_turns=0,
                endpoint_scope=request.endpoint_scope,
                reason=reason,
                block_code=code,
                hardware=hardware,
            )

        choice = request.managed_deployment
        if choice is None:
            return blocked("managed deployment choice is missing", PreflightBlockCode.DEPLOYMENT_UNAVAILABLE)
        try:
            # One binding serves the reported hardware and the framework check, so both
            # describe the same device.
            bound = self._bound_device(choice.device_index)
            inputs = validate_managed_replay_inputs(request)
            self._selected_offer(choice, bound)
        except SetupProblem as error:
            return blocked(str(error), error.block_code)
        except (OSError, ValueError) as error:
            return blocked(error_text(error), PreflightBlockCode.DEPLOYMENT_UNAVAILABLE)
        return ReplayPreflight(
            ready=True,
            manifest_tasks=inputs.manifest_tasks,
            manifest_turns=inputs.manifest_turns,
            endpoint_scope=request.endpoint_scope,
            reason=None,
            block_code=None,
            hardware=SafeHardwareSummary.from_snapshot(bound.snapshot),
        )

    async def execute(self, request: ReplayRequest, observer: TuiReplayObserver) -> ReplayExecution:
        """Launch, verify, benchmark, and stop one owned local model server."""
        choice = request.managed_deployment
        if choice is None:
            raise ValueError("managed deployment choice is missing")
        await asyncio.to_thread(validate_managed_replay_inputs, request)
        observer.on_activity(RunActivity(kind=RunActivityKind.SETUP_CHECKED))
        # One binding serves the framework check, the launch, and every record, so they all
        # describe the same device. Checking the offer walks PATH, so it stays off the event loop.
        bound = self._bound_device(choice.device_index)
        await asyncio.to_thread(self._selected_offer, choice, bound)
        inputs = ManagedRunInputs(
            manifest_path=request.manifest_path,
            output_dir=request.output_dir,
            candidate=choice.candidate,
            framework=choice.framework,
            device=bound,
            catalog_as_of=choice.catalog_as_of,
            catalog_digest=choice.catalog_digest,
            cache_root=choice.cache_root,
            port=choice.port,
            context_tokens=choice.context_tokens,
            startup_timeout_seconds=choice.startup_timeout_seconds,
            client_backend=request.client_backend,
        )
        log_stop = threading.Event()
        loop = asyncio.get_running_loop()

        def deliver_server_log(lines: tuple[str, ...]) -> None:
            # The tail runs in a worker thread, so the observer is only ever touched on the loop.
            loop.call_soon_threadsafe(observer.on_server_log, lines)

        # The tail reads nothing until the server creates its log. It is stopped only
        # after the pipeline has stopped the server, so the shutdown lines reach the view too.
        log_tail = asyncio.create_task(
            asyncio.to_thread(
                _tail_server_log, request.output_dir / DEPLOYMENT_LOG_FILENAME, deliver_server_log, log_stop
            )
        )
        try:
            outcome = await run_managed_replay(inputs, observer)
        finally:
            await _stop_log_tail(log_tail, log_stop)
        timings = run_timing_summary(outcome.result)
        return ReplayExecution(
            artifacts=outcome.artifacts,
            output_dir=request.output_dir,
            output_tokens_per_second=run_output_tokens_per_second(outcome.result),
            ttft_p50_ms=timings.ttft_p50_ms,
            e2e_p50_ms=timings.e2e_p50_ms,
            failed_turns=timings.failed_turns,
            total_turns=timings.total_turns,
            deployment_record=outcome.deployment_record,
            gpu_startup_verified=True,
            power_summary=outcome.power_summary,
            qualification=outcome.qualification,
        )
