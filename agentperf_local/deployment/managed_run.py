"""Run one managed replay: own a local model server, measure it, and write its evidence.

- `run_managed_replay`: the one pipeline the CLI and the TUI both call.
- `measure_endpoint` / `EndpointMeasurement`: qualify a ready server, replay the manifest under power
  telemetry, and write the evidence; managed runs and attached `run` share it.
- `ManagedRunInputs` / `ManagedRunOutcome`: what one run takes and leaves behind.
- `ManagedRunObserver`, `RunActivity`, `RunActivityKind`: what a run reports as it goes.
- `write_run_artifacts_uncancellable`, `discard_unfinished_output`: report commit and cleanup.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from agentperf_local.client.backends import ClientBackend
from agentperf_local.common.models import replace_fields
from agentperf_local.deployment.catalog import DeploymentFramework, ModelCandidate
from agentperf_local.deployment.endpoint_probes import (
    ContextProbeResult,
    IgnoreEosProbeResult,
    probe_ignore_eos,
    require_measurable_exact_policy,
)
from agentperf_local.deployment.managed import (
    DEPLOYMENT_LOG_FILENAME,
    DEPLOYMENT_RECORD_FILENAME,
    BoundDeploymentDevice,
    DeploymentPlan,
    ManagedDeployment,
    create_deployment_plan,
    start_managed_deployment,
    verify_gpu_startup,
    wait_for_deployment,
    write_deployment_record,
)
from agentperf_local.deployment.model_cache import (
    MODEL_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
    VerifiedDeployment,
    ensure_model_artifacts,
)
from agentperf_local.deployment.qualification import (
    QUALIFICATION_FILENAME,
    RuntimeQualification,
    qualify_endpoint,
    write_runtime_qualification,
)
from agentperf_local.provenance.benchmark import (
    MEASUREMENT_BINDING_FILENAME,
    SubmissionContext,
    create_measurement_binding,
    workload_digest,
    write_measurement_binding,
)
from agentperf_local.provenance.context import ContextObservationReason, RunContextFacts
from agentperf_local.provenance.hardware import AcceleratorPlatform
from agentperf_local.replay.config import DEFAULT_REQUEST_TIMEOUT_SECONDS, OutputTokenPolicy, RunConfig
from agentperf_local.replay.runner import CompositeRunObserver, RunObserver, RunResult, run_manifest
from agentperf_local.reports.reporting import ArtifactPaths, write_run_artifacts
from agentperf_local.telemetry.power import (
    POWER_SUMMARY_FILENAME,
    TELEMETRY_FILENAME,
    NvidiaPowerCollector,
    PhaseClockObserver,
    nvidia_power_collector,
    write_power_summary,
)

MANAGED_SUITE_ID = "agentperf-local-managed"


class RunActivityKind(StrEnum):
    """Name one step of a run as the activity log reports it."""

    SETUP_CHECKED = "setup-checked"
    SERVER_CHECKED = "server-checked"
    OUTPUT_POLICY_CHOSEN = "output-policy-chosen"
    MODEL_CHECKING = "model-checking"
    MODEL_READY = "model-ready"
    SERVER_STARTING = "server-starting"
    SERVER_READY = "server-ready"
    GPU_VERIFIED = "gpu-verified"
    QUALIFYING = "qualifying"
    QUALIFIED = "qualified"
    POWER_UNAVAILABLE = "power-unavailable"
    REPLAY_STARTING = "replay-starting"
    POWER_STARTED = "power-started"
    POWER_RECORDED = "power-recorded"
    SERVER_STOPPING = "server-stopping"
    SERVER_STOPPED = "server-stopped"


class RunActivity(BaseModel, frozen=True):
    """Describe one run step with display-safe facts: counts, sizes, and names, never content or URLs."""

    kind: RunActivityKind
    elapsed_seconds: float | None = None
    artifact_bytes: int | None = None
    downloaded: bool | None = None
    framework: DeploymentFramework | None = None
    accelerator_platform: AcceleratorPlatform | None = None
    context_tokens: int | None = None
    context_probe: ContextProbeResult | None = None
    energy_joules: float | None = None
    power_valid: bool | None = None
    power_first_sample: bool | None = None
    probes_passed: int | None = None
    probes_total: int | None = None
    ignore_eos_probe: IgnoreEosProbeResult | None = None
    # True when the policy was chosen because the server is Ollama, with no probe sent.
    ollama_endpoint: bool = False


class ManagedRunObserver(RunObserver, Protocol):
    """Receive the steps of one managed run on the event loop."""

    async def on_finalizing(self) -> None:
        """Acknowledge that durable report commits are about to begin."""
        ...

    def on_artifact_progress(self, downloaded_bytes: int, total_bytes: int) -> None:
        """Report model download progress."""
        ...

    def on_activity(self, activity: RunActivity) -> None:
        """Report one completed or started run step."""
        ...


class ManagedRunInputs(BaseModel, frozen=True):
    """Describe one managed run after its caller has validated the choice."""

    manifest_path: Path
    output_dir: Path
    candidate: ModelCandidate
    # The recipe file's exact text; the deployment record keeps it for the submission.
    recipe_text: str
    framework: DeploymentFramework
    device: BoundDeploymentDevice
    catalog_as_of: str
    catalog_digest: str
    cache_root: Path
    port: int
    # None launches the recipe's full benchmark context.
    context_tokens: int | None
    startup_timeout_seconds: float
    client_backend: ClientBackend
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    download_timeout_seconds: float = MODEL_DOWNLOAD_TOTAL_TIMEOUT_SECONDS
    # "exact" refuses a server that drops ignore_eos, and "recorded" never asks.
    # None asks the server and runs under the strongest policy it supports.
    output_token_policy: OutputTokenPolicy | None = None
    power: bool = True
    # False keeps replay boundaries away from the observer. The CLI clears it when no
    # progress view is shown, so a plain run reports an unobserved replay.
    observe_replay: bool = True


class ManagedRunOutcome(BaseModel, frozen=True):
    """Store the result and every evidence file one managed run wrote."""

    result: RunResult
    run_id: str
    run_context: RunContextFacts
    output_token_policy: OutputTokenPolicy
    artifacts: ArtifactPaths
    deployment_record: Path
    deployment_log: Path
    measurement: Path
    qualification: Path
    qualification_passed: bool
    power_summary: Path | None


def _managed_run_config(
    plan: DeploymentPlan,
    client_backend: ClientBackend,
    request_timeout_seconds: float,
    output_token_policy: OutputTokenPolicy,
) -> RunConfig:
    """Return the replay settings for one owned server."""
    return RunConfig(
        base_url=plan.base_url,
        model=plan.model_alias,
        api_key=None,
        client_backend=client_backend,
        request_timeout_seconds=request_timeout_seconds,
        output_token_policy=output_token_policy,
        # vLLM otherwise defaults exact requests to automatic tool selection and
        # stops at a parsed call. Some llama.cpp grammars reject this option.
        tool_choice="none" if plan.framework == "vllm" else None,
    )


def discard_unfinished_output(binding_path: Path, output_dir: Path, *, created_output_dir: bool) -> None:
    """Remove an unfinished run binding so it does not block the next run.

    The folder goes too when it is empty and this run created it. A folder the user
    made before the run stays.
    """
    # This cleanup runs while a failure or cancellation is in flight, so a filesystem
    # race here must not replace the original exception.
    with suppress(OSError):
        binding_path.unlink(missing_ok=True)
        if created_output_dir and not any(output_dir.iterdir()):
            output_dir.rmdir()


async def write_run_artifacts_uncancellable(
    result: RunResult,
    output_dir: Path,
    config: RunConfig,
    run_context: RunContextFacts,
    run_id: str,
) -> ArtifactPaths:
    """Finish the durable report commit even if the worker is canceled."""
    write_task = asyncio.create_task(
        asyncio.to_thread(write_run_artifacts, result, output_dir, config, run_context=run_context, run_id=run_id)
    )
    try:
        return await asyncio.shield(write_task)
    except asyncio.CancelledError:
        return await write_task


class _OwnedArtifact(BaseModel, frozen=True):
    """Store the verified model files and whether this run had to download any."""

    artifact: VerifiedDeployment
    downloaded: bool


async def _ensure_owned_model_artifact(inputs: ManagedRunInputs, observer: ManagedRunObserver) -> _OwnedArtifact:
    """Stop a cache verification or download before acknowledging cancellation."""
    cancellation = threading.Event()
    loop = asyncio.get_running_loop()
    downloaded = threading.Event()

    def report_progress(downloaded_bytes: int, total_bytes: int) -> None:
        # The download runs in a worker thread, so the observer is only ever touched on the loop.
        downloaded.set()
        loop.call_soon_threadsafe(observer.on_artifact_progress, downloaded_bytes, total_bytes)

    artifact_task = asyncio.create_task(
        asyncio.to_thread(
            ensure_model_artifacts,
            inputs.cache_root,
            inputs.candidate,
            cancellation_requested=cancellation.is_set,
            download_timeout_seconds=inputs.download_timeout_seconds,
            progress=report_progress,
        )
    )
    try:
        artifact = await asyncio.shield(artifact_task)
    except asyncio.CancelledError:
        cancellation.set()
        with suppress(Exception):
            await artifact_task
        raise
    return _OwnedArtifact(artifact=artifact, downloaded=downloaded.is_set())


async def _start_owned_deployment(plan: DeploymentPlan, log_path: Path) -> ManagedDeployment:
    """Close a child that starts concurrently with cancellation."""
    start_task = asyncio.create_task(asyncio.to_thread(start_managed_deployment, plan, log_path))
    try:
        return await asyncio.shield(start_task)
    except asyncio.CancelledError:
        try:
            deployment = await start_task
        except Exception:
            pass
        else:
            await asyncio.to_thread(deployment.close)
        raise


async def _close_owned_deployment(deployment: ManagedDeployment) -> None:
    """Finish process-group cleanup even when the worker is canceled."""
    close_task = asyncio.create_task(asyncio.to_thread(deployment.close))
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        await close_task
        raise


async def _measurable_output_token_policy(
    plan: DeploymentPlan, inputs: ManagedRunInputs, observer: ManagedRunObserver
) -> OutputTokenPolicy:
    """Return the policy this run measures under, asking the ready server when the policy depends on it.

    A server that drops ignore_eos, as Splash before 1.2.1 does, fails every exact-policy turn. An
    explicit exact policy is refused on such a server; an unset one falls back to the
    recorded policy, as an attached run does.
    """
    requested = inputs.output_token_policy
    if requested is not None and requested != "exact":
        return requested
    capability = await probe_ignore_eos(plan.base_url, plan.model_alias, inputs.client_backend)
    if requested == "exact":
        require_measurable_exact_policy(capability)
    observer.on_activity(RunActivity(kind=RunActivityKind.OUTPUT_POLICY_CHOSEN, ignore_eos_probe=capability))
    return capability.output_token_policy


def _start_power_collector(collector: NvidiaPowerCollector) -> bool:
    """Start the collector child and give it its head start before the first request."""
    collector.start()
    return collector.wait_for_first_sample()


async def _stop_power_collector(collector: NvidiaPowerCollector) -> None:
    """Stop the collector even if the worker is being canceled; the child must not outlive the run."""
    stop_task = asyncio.create_task(asyncio.to_thread(collector.stop))
    try:
        await asyncio.shield(stop_task)
    except asyncio.CancelledError:
        await stop_task
        raise


def _replay_observer(clock: RunObserver | None, observer: RunObserver | None) -> RunObserver | None:
    """Join the power clock and the caller's observer, or return None when neither listens."""
    if clock is not None and observer is not None:
        return CompositeRunObserver(observers=(clock, observer))
    return clock if clock is not None else observer


class EndpointMeasurement(BaseModel, frozen=True):
    """Describe one measurement of a ready server: where it writes, what it replays, and what it samples."""

    manifest_path: Path
    output_dir: Path
    config: RunConfig
    run_id: str
    run_context: RunContextFacts
    # The qualification report names the recipe it probed, or "attached" for a user's server.
    # None skips the probes, for a run that cannot be submitted anyway.
    qualification_profile_id: str | None
    # None when the accelerator is not one that power telemetry can sample.
    accelerator_platform: AcceleratorPlatform | None
    device_index: int
    power: bool
    observe_replay: bool


class MeasuredEndpoint(BaseModel, frozen=True):
    """Store the replay result and the evidence files one measurement wrote."""

    result: RunResult
    artifacts: ArtifactPaths
    qualification: RuntimeQualification | None
    power_summary: Path | None


async def _qualify(
    measurement: EndpointMeasurement, profile_id: str, observer: ManagedRunObserver
) -> RuntimeQualification:
    """Run the synthetic protocol probes against the ready server and write their report."""
    config = measurement.config
    observer.on_activity(RunActivity(kind=RunActivityKind.QUALIFYING))
    qualification = await qualify_endpoint(
        config.base_url, config.model, profile_id, config.client_backend, measurement.run_id, api_key=config.api_key
    )
    await asyncio.to_thread(write_runtime_qualification, measurement.output_dir / QUALIFICATION_FILENAME, qualification)
    observer.on_activity(
        RunActivity(
            kind=RunActivityKind.QUALIFIED,
            probes_passed=sum(1 for outcome in qualification.required_outcomes if outcome.passed),
            probes_total=len(qualification.required_outcomes),
        )
    )
    return qualification


async def measure_endpoint(measurement: EndpointMeasurement, observer: ManagedRunObserver) -> MeasuredEndpoint:
    """Probe the ready server, replay the manifest under power telemetry, and write every evidence file.

    The protocol probes run before any measured request, and their report shares the
    run identifier with the binding. Power is sampled only on an NVIDIA accelerator.
    """
    output_dir = measurement.output_dir
    config = measurement.config
    qualification = (
        None
        if measurement.qualification_profile_id is None
        else await _qualify(measurement, measurement.qualification_profile_id, observer)
    )
    power_collector = (
        nvidia_power_collector(
            measurement.accelerator_platform,
            measurement.device_index,
            measurement.run_id,
            output_dir / TELEMETRY_FILENAME,
        )
        if measurement.power and measurement.accelerator_platform is not None
        else None
    )
    if measurement.power and power_collector is None:
        observer.on_activity(
            RunActivity(kind=RunActivityKind.POWER_UNAVAILABLE, accelerator_platform=measurement.accelerator_platform)
        )
    observer.on_activity(RunActivity(kind=RunActivityKind.REPLAY_STARTING))
    clock = PhaseClockObserver() if power_collector is not None else None
    run_observer = _replay_observer(clock, observer if measurement.observe_replay else None)
    if power_collector is not None:
        first_sample = await asyncio.to_thread(_start_power_collector, power_collector)
        observer.on_activity(RunActivity(kind=RunActivityKind.POWER_STARTED, power_first_sample=first_sample))
    try:
        result = await run_manifest(measurement.manifest_path, config, observer=run_observer)
    finally:
        if power_collector is not None:
            await _stop_power_collector(power_collector)
    await observer.on_finalizing()
    artifacts = await write_run_artifacts_uncancellable(
        result, output_dir, config, measurement.run_context, measurement.run_id
    )
    power_path: Path | None = None
    if power_collector is not None and clock is not None:
        power_path = output_dir / POWER_SUMMARY_FILENAME
        power_summary = await asyncio.to_thread(power_collector.summarize, clock.measured_phase())
        await asyncio.to_thread(write_power_summary, power_path, power_summary)
        observer.on_activity(
            RunActivity(
                kind=RunActivityKind.POWER_RECORDED,
                energy_joules=power_summary.measured.sampled_power_energy_joules,
                power_valid=power_summary.measured.sampled_power_energy_valid,
            )
        )
    return MeasuredEndpoint(result=result, artifacts=artifacts, qualification=qualification, power_summary=power_path)


async def run_managed_replay(inputs: ManagedRunInputs, observer: ManagedRunObserver) -> ManagedRunOutcome:
    """Fetch the model, own its server, qualify it, replay the manifest, and write every evidence file.

    The deployment record lands before the measurement binding, because the binding
    carries the record's digest. The server stops in all cases. A run that does not
    finish leaves no binding behind.
    """
    recipe = inputs.candidate.deployment
    observer.on_activity(
        RunActivity(
            kind=RunActivityKind.MODEL_CHECKING,
            artifact_bytes=None if recipe is None else recipe.artifact_size_bytes,
        )
    )
    owned_artifact = await _ensure_owned_model_artifact(inputs, observer)
    artifact = owned_artifact.artifact
    observer.on_activity(
        RunActivity(
            kind=RunActivityKind.MODEL_READY,
            artifact_bytes=artifact.size_bytes,
            downloaded=owned_artifact.downloaded,
        )
    )
    snapshot = inputs.device.snapshot
    plan = await asyncio.to_thread(
        create_deployment_plan,
        snapshot,
        inputs.candidate,
        inputs.framework,
        artifact,
        catalog_digest=inputs.catalog_digest,
        port=inputs.port,
        device_environment=inputs.device.environment,
        context_tokens=inputs.context_tokens,
    )
    suite_digest = await asyncio.to_thread(workload_digest, inputs.manifest_path)
    context = SubmissionContext(
        suite_id=MANAGED_SUITE_ID,
        suite_epoch=inputs.catalog_as_of,
        suite_digest=suite_digest,
        model_semantics_id=inputs.candidate.profile_id,
        model_artifact_digest=plan.artifact_manifest_sha256,
        runtime_id=plan.runtime_id,
        context_tokens=plan.context_tokens,
    )
    binding = await asyncio.to_thread(
        create_measurement_binding, context, inputs.manifest_path, plan.model_alias, snapshot
    )
    output_dir = inputs.output_dir
    created_output_dir = not await asyncio.to_thread(output_dir.exists)
    deployment_path = output_dir / DEPLOYMENT_RECORD_FILENAME
    log_path = output_dir / DEPLOYMENT_LOG_FILENAME
    measurement_path = output_dir / MEASUREMENT_BINDING_FILENAME
    deployment: ManagedDeployment | None = None
    try:
        observer.on_activity(
            RunActivity(
                kind=RunActivityKind.SERVER_STARTING,
                framework=plan.framework,
                accelerator_platform=plan.accelerator_platform,
                context_tokens=plan.context_tokens,
            )
        )
        startup_started = time.monotonic()
        deployment = await _start_owned_deployment(plan, log_path)
        served_context_tokens = await asyncio.to_thread(
            wait_for_deployment, deployment, timeout_seconds=inputs.startup_timeout_seconds
        )
        observer.on_activity(
            RunActivity(
                kind=RunActivityKind.SERVER_READY,
                framework=plan.framework,
                context_tokens=served_context_tokens,
                elapsed_seconds=time.monotonic() - startup_started,
            )
        )
        await asyncio.to_thread(verify_gpu_startup, deployment)
        observer.on_activity(
            RunActivity(kind=RunActivityKind.GPU_VERIFIED, accelerator_platform=plan.accelerator_platform)
        )
        # Settled before the records land, so a refused policy leaves no deployment record behind.
        output_token_policy = await _measurable_output_token_policy(plan, inputs, observer)
        config = _managed_run_config(plan, inputs.client_backend, inputs.request_timeout_seconds, output_token_policy)
        written_deployment = await asyncio.to_thread(
            write_deployment_record,
            deployment_path,
            plan,
            snapshot,
            recipe_text=inputs.recipe_text,
            gpu_startup_verified=True,
        )
        # The readiness check proved the served context, so it joins the binding
        # before the binding lands on disk, and the deployment record's exact
        # bytes join it the same way.
        await asyncio.to_thread(
            write_measurement_binding,
            measurement_path,
            replace_fields(
                binding,
                observed_context_tokens=served_context_tokens,
                deployment_digest=written_deployment.file_digest,
            ),
        )
        run_context = RunContextFacts(
            requested_tokens=plan.context_tokens,
            observed_tokens=served_context_tokens,
            observed_reason=ContextObservationReason.REPORTED,
        )
        measurement = await measure_endpoint(
            EndpointMeasurement(
                manifest_path=inputs.manifest_path,
                output_dir=output_dir,
                config=config,
                run_id=binding.run_id,
                run_context=run_context,
                qualification_profile_id=inputs.candidate.profile_id,
                accelerator_platform=plan.accelerator_platform,
                device_index=inputs.device.device_index,
                power=inputs.power,
                observe_replay=inputs.observe_replay,
            ),
            observer,
        )
        return ManagedRunOutcome(
            result=measurement.result,
            run_id=binding.run_id,
            run_context=run_context,
            output_token_policy=output_token_policy,
            artifacts=measurement.artifacts,
            deployment_record=deployment_path,
            deployment_log=log_path,
            measurement=measurement_path,
            qualification=output_dir / QUALIFICATION_FILENAME,
            qualification_passed=measurement.qualification is not None and measurement.qualification.passed,
            power_summary=measurement.power_summary,
        )
    except BaseException:
        discard_unfinished_output(measurement_path, output_dir, created_output_dir=created_output_dir)
        raise
    finally:
        if deployment is not None:
            # A failing observer must not leave the server running.
            try:
                observer.on_activity(RunActivity(kind=RunActivityKind.SERVER_STOPPING))
            finally:
                await _close_owned_deployment(deployment)
            observer.on_activity(RunActivity(kind=RunActivityKind.SERVER_STOPPED))
