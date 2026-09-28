"""Run one replay: attached endpoint, owned deployment, or guided interface."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from agentperf_local.cli.options import (
    print_json,
    read_api_key,
    read_bound_device,
    read_client_backend,
    read_deployment_framework,
    read_live_workspace_root,
    read_managed_target,
    read_output_token_margin,
    read_replay_manifest_path,
    read_requested_output_token_policy,
    read_sampling_preset,
    read_tool_delay_scale,
    read_tool_mode,
    require_fresh_output_dir,
    resolve_submit_token,
)
from agentperf_local.client.rust_client import validate_rustcore_available
from agentperf_local.common.argparse_fields import (
    read_boolean,
    read_integer,
    read_number,
    read_optional_integer,
    read_optional_number,
    read_optional_path,
    read_optional_string,
    read_path,
    read_string,
)
from agentperf_local.common.durable_files import (
    validate_new_file_paths,
)
from agentperf_local.common.identity import mint_run_id
from agentperf_local.common.json_types import JsonObject
from agentperf_local.common.models import replace_fields
from agentperf_local.common.units import BYTES_PER_GIB
from agentperf_local.deployment.catalog import (
    ModelCatalog,
    load_model_catalog,
)
from agentperf_local.deployment.context_policy import (
    require_replay_context_floor,
    resolve_context_tokens,
)
from agentperf_local.deployment.endpoint_probes import (
    OLLAMA_RECORDED_POLICY_WARNING,
    IgnoreEosSupport,
    is_ollama_endpoint,
    probe_ignore_eos,
    probe_served_context_tokens,
)
from agentperf_local.deployment.frameworks import framework_offers
from agentperf_local.deployment.managed import (
    DEPLOYMENT_LOG_FILENAME,
    DEPLOYMENT_RECORD_FILENAME,
)
from agentperf_local.deployment.managed_run import (
    ManagedRunInputs,
    RunActivity,
    RunActivityKind,
    run_managed_replay,
)
from agentperf_local.deployment.qualification import (
    QUALIFICATION_FILENAME,
)
from agentperf_local.provenance.benchmark import (
    BENCHMARK_CONTEXT_TOKENS,
    MEASUREMENT_BINDING_FILENAME,
    SourceProvenance,
    collect_source_provenance,
    create_attached_submission_context,
    create_measurement_binding,
    write_measurement_binding,
)
from agentperf_local.provenance.context import RunContextFacts
from agentperf_local.provenance.hardware import HardwareSnapshot, collect_hardware_snapshot
from agentperf_local.replay.config import (
    OutputTokenPolicy,
    RunConfig,
)
from agentperf_local.replay.runner import RunBoundaryEvent, RunObserver, RunResult, TurnResult, run_manifest
from agentperf_local.reports.progress import TerminalRunObserver
from agentperf_local.reports.reporting import (
    ArtifactPaths,
    write_run_artifacts,
)
from agentperf_local.submission.client import (
    check_revision_allowlist,
)
from agentperf_local.telemetry.power import (
    POWER_SUMMARY_FILENAME,
    TELEMETRY_FILENAME,
)
from agentperf_local.tui.app import AgentPerfLocalApp, TuiDefaults, TuiOutcome
from agentperf_local.workload.schema import load_manifest

DOWNLOAD_PROGRESS_STEPS = 20


@contextmanager
def _discarded_binding_on_failure(measurement_path: Path) -> Iterator[None]:
    """Remove a measurement binding whose artifact set never landed.

    A binding without its artifact set proves nothing and permanently blocks the
    directory.
    """
    try:
        yield
    except BaseException:
        measurement_path.unlink(missing_ok=True)
        raise


def _run_status(result: RunResult, failures_path: Path) -> int:
    """Map one replay result to a process status, explaining failures on stderr."""
    if result.success:
        return 0
    _report_run_failures(result, failures_path)
    return 1


def _run_config(namespace: argparse.Namespace, output_token_policy: OutputTokenPolicy) -> RunConfig:
    tool_mode = read_tool_mode(namespace)
    return RunConfig(
        base_url=read_string(namespace, "base_url"),
        model=read_string(namespace, "model"),
        api_key=read_api_key(namespace),
        client_backend=read_client_backend(namespace),
        request_timeout_seconds=read_number(namespace, "timeout_seconds"),
        output_token_policy=output_token_policy,
        max_output_tokens=read_integer(namespace, "max_output_tokens"),
        output_token_margin=read_output_token_margin(namespace, output_token_policy),
        sampling_preset=read_sampling_preset(namespace),
        temperature=read_optional_number(namespace, "temperature"),
        top_p=read_optional_number(namespace, "top_p"),
        top_k=read_optional_integer(namespace, "top_k"),
        min_p=read_optional_number(namespace, "min_p"),
        tool_choice=read_optional_string(namespace, "tool_choice"),
        reasoning_effort=read_optional_string(namespace, "reasoning_effort"),
        cache_isolation=read_boolean(namespace, "cache_isolation"),
        cache_namespace=read_optional_string(namespace, "cache_namespace"),
        tool_mode=tool_mode,
        tool_delay_scale=read_tool_delay_scale(namespace, tool_mode),
        live_tool_image=read_optional_string(namespace, "live_tool_image"),
        live_workspace_root=read_live_workspace_root(namespace, tool_mode),
        live_network=read_optional_string(namespace, "live_network"),
        live_timeout_seconds=read_number(namespace, "live_timeout_seconds"),
        live_docker_executable=read_optional_string(namespace, "live_docker_executable"),
    )


def _artifact_json(paths: ArtifactPaths) -> JsonObject:
    return {
        "turns": str(paths.turns),
        "tasks": str(paths.tasks),
        "tools": str(paths.tools),
        "failures": str(paths.failures),
        "summary": str(paths.summary),
    }


@dataclass(slots=True)
class _CliManagedObserver:
    """Print managed-run notes on stderr and forward replay boundaries to the optional progress view."""

    progress: TerminalRunObserver | None
    download_progress: Callable[[int, int], None] | None

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        if self.progress is not None:
            self.progress.on_boundary(event)

    async def on_finalizing(self) -> None:
        return

    def on_artifact_progress(self, downloaded_bytes: int, total_bytes: int) -> None:
        if self.download_progress is not None:
            self.download_progress(downloaded_bytes, total_bytes)

    def on_activity(self, activity: RunActivity) -> None:
        kind = activity.kind
        passed, total = activity.probes_passed, activity.probes_total
        if kind is RunActivityKind.QUALIFIED and passed is not None and total is not None and passed < total:
            print(
                f"warning: {total - passed}/{total} runtime probes failed; "
                "the run continues but cannot reach the verified tier",
                file=sys.stderr,
            )
        elif kind is RunActivityKind.POWER_UNAVAILABLE:
            if activity.accelerator_platform != "nvidia-cuda":
                print("note: GPU power sampling is NVIDIA-only; this run records no power evidence", file=sys.stderr)
            else:
                print("warning: nvidia-smi was not found; this run records no power evidence", file=sys.stderr)
        elif kind is RunActivityKind.POWER_STARTED and activity.power_first_sample is False:
            print("warning: the power collector produced no sample before the replay started", file=sys.stderr)


def _managed_observer(
    namespace: argparse.Namespace, snapshot: HardwareSnapshot, catalog: ModelCatalog
) -> _CliManagedObserver:
    if not read_boolean(namespace, "progress"):
        return _CliManagedObserver(progress=None, download_progress=None)
    accelerator = snapshot.accelerators[0]
    return _CliManagedObserver(
        progress=TerminalRunObserver(
            device_label=accelerator.name,
            suite_label=f"managed local · {catalog.as_of}",
            stream=sys.stderr,
        ),
        download_progress=_download_progress_printer(),
    )


def _download_progress_printer() -> Callable[[int, int], None]:
    last_step = -1

    def report(downloaded_bytes: int, total_bytes: int) -> None:
        nonlocal last_step
        if total_bytes <= 0:
            return
        step = min(downloaded_bytes * DOWNLOAD_PROGRESS_STEPS // total_bytes, DOWNLOAD_PROGRESS_STEPS)
        if step > last_step:
            last_step = step
            done_gib = downloaded_bytes / BYTES_PER_GIB
            total_gib = total_bytes / BYTES_PER_GIB
            print(f"download {done_gib:.2f} / {total_gib:.2f} GiB", file=sys.stderr)

    return report


def _advise_on_revision(namespace: argparse.Namespace, provenance: SourceProvenance) -> None:
    """Warn before a run, not after, when this build cannot reach verified.

    Only a user who has a submit token configured intends to submit, so only they
    pay for the network round trip. The service check stays authoritative; a
    service that cannot be reached leaves the run untouched.
    """
    if resolve_submit_token(namespace, "submit_token_env") is None:
        return
    check = check_revision_allowlist(provenance, base_url=read_string(namespace, "submit_base_url"))
    if not check.reachable:
        print("note: the submission service could not be reached to check the commit allowlist", file=sys.stderr)
        return
    if check.advice is not None:
        print(f"note: {check.advice}; this run can be submitted as self-reported only", file=sys.stderr)


def managed_run_command(namespace: argparse.Namespace) -> int:
    manifest_path = read_replay_manifest_path(namespace)
    output_dir = read_path(namespace, "output_dir")
    require_fresh_output_dir(output_dir, "managed run")
    validate_new_file_paths(
        tuple(
            output_dir / name
            for name in (
                DEPLOYMENT_RECORD_FILENAME,
                DEPLOYMENT_LOG_FILENAME,
                MEASUREMENT_BINDING_FILENAME,
                TELEMETRY_FILENAME,
                POWER_SUMMARY_FILENAME,
                QUALIFICATION_FILENAME,
            )
        )
    )
    client_backend = read_client_backend(namespace)
    if client_backend == "rust":
        validate_rustcore_available()

    target = read_managed_target(namespace)
    catalog = target.catalog
    candidate = target.candidate
    requested_context_tokens = read_optional_integer(namespace, "context_tokens")
    require_replay_context_floor(
        load_manifest(manifest_path).required_context_tokens,
        resolve_context_tokens(target.candidate.deployment, requested_context_tokens),
    )
    bound = read_bound_device(namespace)
    snapshot = bound.snapshot
    framework = read_deployment_framework(namespace)
    offers = framework_offers(snapshot, candidate, context_tokens=requested_context_tokens)
    selected_offer = next((offer for offer in offers if offer.framework == framework), None)
    if selected_offer is None:
        raise ValueError(f"{framework} is not compatible with the detected accelerator")
    if not selected_offer.installed:
        raise ValueError(selected_offer.installation_hint)
    if selected_offer.memory_fit is not True:
        raise ValueError("detected accelerator memory is unknown or below the requested-context requirement")
    _advise_on_revision(namespace, collect_source_provenance())
    inputs = ManagedRunInputs(
        manifest_path=manifest_path,
        output_dir=output_dir,
        candidate=candidate,
        framework=framework,
        device=bound,
        catalog_as_of=catalog.as_of,
        catalog_digest=catalog.digest,
        cache_root=read_path(namespace, "cache_root"),
        port=read_integer(namespace, "port"),
        context_tokens=requested_context_tokens,
        startup_timeout_seconds=read_number(namespace, "startup_timeout_seconds"),
        client_backend=client_backend,
        request_timeout_seconds=read_number(namespace, "request_timeout_seconds"),
        download_timeout_seconds=read_number(namespace, "download_timeout_seconds"),
        power=read_boolean(namespace, "power"),
        observe_replay=read_boolean(namespace, "progress"),
    )
    outcome = asyncio.run(run_managed_replay(inputs, _managed_observer(namespace, snapshot, catalog)))
    print_json(
        {
            "success": outcome.result.success,
            "run_id": outcome.run_id,
            "context": outcome.run_context.to_json(),
            "deployment": str(outcome.deployment_record),
            "deployment_log": str(outcome.deployment_log),
            "measurement": str(outcome.measurement),
            "qualification": str(outcome.qualification),
            "qualification_passed": outcome.qualification_passed,
            "power": None if outcome.power_summary is None else str(outcome.power_summary),
            "artifacts": _artifact_json(outcome.artifacts),
        }
    )
    return _run_status(outcome.result, outcome.artifacts.failures)


def _first_failure_message(turns: tuple[TurnResult, ...]) -> str:
    """Return the first recorded request or tool failure message."""
    for turn in turns:
        if turn.error is not None:
            return turn.error
    for turn in turns:
        for replay in turn.tool_replays:
            if replay.exception_info:
                return replay.exception_info
    return "see failures file"


def _report_run_failures(result: RunResult, failures_path: Path) -> None:
    """Explain a failed run on stderr while stdout keeps the machine-readable result."""
    turns = result.turns
    failed = sum(1 for turn in turns if not turn.success)
    print(f"{failed}/{len(turns)} turns failed. First error: {_first_failure_message(turns)}", file=sys.stderr)
    print(f"Details: {failures_path}", file=sys.stderr)


def _warn_non_comparable_context(run_context: RunContextFacts) -> None:
    """Warn when this result will be separated from full-context results."""
    if not run_context.reduced:
        return
    if run_context.observed_tokens is None:
        print(
            f"warning: the endpoint did not prove its context window ({run_context.observed_reason.value}); "
            "this run will be recorded as non-comparable — attached servers other than llama.cpp "
            "may not report their context (meta.n_ctx)",
            file=sys.stderr,
        )
        return
    print(
        f"warning: the endpoint serves a {run_context.observed_tokens:,}-token context, below the "
        f"{BENCHMARK_CONTEXT_TOKENS:,}-token benchmark; this run will be recorded as non-comparable",
        file=sys.stderr,
    )


def _resolve_output_token_policy(namespace: argparse.Namespace) -> OutputTokenPolicy:
    """Return the policy the run uses, asking the server whether it is Ollama when the user named none.

    Two cheap GETs name Ollama before any long probe or replay. Only a policy the user
    left to the command switches; an explicit exact still meets the probe's refusal.
    The policy is settled before the config is built, because options such as the
    output token margin are valid only under the recorded policy.
    """
    requested_policy = read_requested_output_token_policy(namespace)
    if requested_policy is not None:
        return requested_policy
    if is_ollama_endpoint(read_string(namespace, "base_url"), api_key=read_api_key(namespace)):
        print(f"warning: {OLLAMA_RECORDED_POLICY_WARNING}", file=sys.stderr)
        return "recorded"
    return "exact"


def run_command(namespace: argparse.Namespace) -> int:
    if read_client_backend(namespace) == "rust":
        validate_rustcore_available()

    manifest_path = read_replay_manifest_path(namespace)
    output_dir = read_path(namespace, "output_dir")
    require_fresh_output_dir(output_dir, "run")
    config = _run_config(namespace, _resolve_output_token_policy(namespace))
    measurement_path = output_dir / MEASUREMENT_BINDING_FILENAME
    # Minted before any endpoint contact so the binding and the summary share it.
    run_id = mint_run_id()
    snapshot = collect_hardware_snapshot()
    context = create_attached_submission_context(manifest_path, config.model)
    binding = create_measurement_binding(
        context,
        manifest_path,
        config.model,
        snapshot,
        run_id=run_id,
    )
    # First endpoint contact. Probe before the binding write so the observed context
    # lands in measurement.json and a later summary-only edit cannot upgrade it.
    probe = probe_served_context_tokens(
        config.base_url,
        config.model,
        api_key=config.api_key,
    )
    run_context = RunContextFacts(
        requested_tokens=BENCHMARK_CONTEXT_TOKENS,
        observed_tokens=probe.observed_tokens,
        observed_reason=probe.reason,
    )
    _warn_non_comparable_context(run_context)
    # A server that drops ignore_eos fails every exact-policy turn with a message that
    # blames the model, so refuse before the binding exists. Past the Ollama check above,
    # the command never switches the policy on the user's behalf.
    if config.output_token_policy == "exact":
        capability = asyncio.run(
            probe_ignore_eos(config.base_url, config.model, config.client_backend, api_key=config.api_key)
        )
        if capability.output_token_policy != config.output_token_policy:
            raise RuntimeError(
                f"{capability.summary}; pass --output-token-policy {capability.output_token_policy} "
                "(its results are normalized, not exact, and are not comparable)"
            )
        if capability.support is IgnoreEosSupport.UNDETERMINED:
            print(f"warning: {capability.summary}; keeping the exact policy", file=sys.stderr)
    write_measurement_binding(
        measurement_path,
        replace_fields(binding, observed_context_tokens=probe.observed_tokens),
    )
    with _discarded_binding_on_failure(measurement_path):
        observer: RunObserver | None = None
        if read_boolean(namespace, "progress"):
            device_label = snapshot.accelerators[0].name if snapshot.accelerators else "local endpoint"
            suite_label = f"{context.suite_id} · {context.suite_epoch}"
            observer = TerminalRunObserver(device_label=device_label, suite_label=suite_label, stream=sys.stderr)
        result = asyncio.run(run_manifest(manifest_path, config, observer=observer))
        artifacts = write_run_artifacts(result, output_dir, config, run_context=run_context, run_id=run_id)
    print_json(
        {
            "success": result.success,
            "run_id": run_id,
            "context": run_context.to_json(),
            "artifacts": _artifact_json(artifacts),
        }
    )
    return _run_status(result, artifacts.failures)


def _tui_status(outcome: TuiOutcome | None, return_code: int | None) -> int:
    """Map one Textual session to a process status."""
    # A crashed app leaves no outcome, so its own return code is the only signal left.
    if return_code is not None and return_code != 0:
        return 1
    return 1 if outcome in {TuiOutcome.FAILED, TuiOutcome.CANCELLED} else 0


def tui_command(namespace: argparse.Namespace) -> int:
    catalog = load_model_catalog(read_path(namespace, "recipes"))
    app = AgentPerfLocalApp(
        catalog,
        defaults=TuiDefaults(
            replay_id=read_string(namespace, "replay"),
            manifest_path=read_optional_path(namespace, "manifest"),
            output_dir=read_optional_path(namespace, "output_dir"),
            base_url=read_string(namespace, "base_url"),
            endpoint_model=read_optional_string(namespace, "model"),
            api_key_env=read_optional_string(namespace, "api_key_env"),
            client_backend=read_client_backend(namespace),
            model_cache_root=read_path(namespace, "cache_root"),
            deployment_port=read_integer(namespace, "port"),
            submit_base_url=read_string(namespace, "submit_base_url"),
            submit_token_env=read_string(namespace, "submit_token_env"),
            deployment_startup_timeout_seconds=read_number(namespace, "startup_timeout_seconds"),
            device_index=read_optional_integer(namespace, "device"),
        ),
    )
    outcome = app.run()
    return _tui_status(outcome, app.return_code)
