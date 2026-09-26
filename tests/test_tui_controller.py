"""Exercise the typed replay controller used by the Textual app."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import pytest

from agentperf_local.client.backends import ClientBackend
from agentperf_local.deployment.catalog import (
    BUNDLED_MODEL_CATALOG_DIGEST,
    BUNDLED_MODEL_CATALOG_PATH,
    ModelCandidate,
    ModelCatalog,
    load_model_catalog,
)
from agentperf_local.deployment.context_policy import derived_minimum_memory_bytes
from agentperf_local.deployment.endpoint_probes import (
    IGNORE_EOS_PROBE_OUTPUT_TOKENS,
    ContextProbeResult,
    IgnoreEosProbeResult,
    IgnoreEosSupport,
    ProbeAnswer,
)
from agentperf_local.deployment.frameworks import FrameworkOffer
from agentperf_local.deployment.managed import DEFAULT_STARTUP_TIMEOUT_SECONDS
from agentperf_local.deployment.managed_run import RunActivity, RunActivityKind
from agentperf_local.provenance.benchmark import MEASUREMENT_BINDING_FILENAME
from agentperf_local.provenance.context import ContextObservationReason
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from agentperf_local.replay.config import RunConfig
from agentperf_local.replay.runner import RunBoundaryEvent, RunObserver, RunResult
from agentperf_local.tui.controller import LocalManagedReplayController, LocalReplayController
from agentperf_local.tui.evidence import SelectionKind
from agentperf_local.tui.replay_contract import (
    DEVICE_SELECTION_REQUIRED_MESSAGE,
    EndpointProblem,
    ManagedDeploymentChoice,
    ManagedDeviceOption,
    ManagedDeviceSelectionRequired,
    ManagedLaunchSettingsProblem,
    PreflightBlockCode,
    ReplayRequest,
    SetupProblem,
    TuiReplayObserver,
    next_run_directory,
)
from tests.replay_workload import write_replay_workload

REPLAY_TIMEOUT_SECONDS = 2.0


def _named(catalog: ModelCatalog, profile_id: str) -> ModelCandidate:
    """Return one catalog model by name, so tests never depend on catalog order."""
    return next(model for model in catalog.models if model.profile_id == profile_id)


def _manifest(tmp_path: Path) -> Path:
    """Write one manifest with the trace bytes needed for its submission binding."""
    return write_replay_workload(tmp_path / "workload", name="TUI test")


def _runnable_manifest(tmp_path: Path) -> Path:
    """Write one manifest a real replay can run end to end."""
    return write_replay_workload(tmp_path, name="Managed controller test")


@dataclass(slots=True)
class _NoopObserver(TuiReplayObserver):
    artifact_progress: list[tuple[int, int]] = field(default_factory=list)
    activities: list[RunActivity] = field(default_factory=list)
    server_log: list[str] = field(default_factory=list)

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        del event

    async def on_finalizing(self) -> None:
        return

    def on_artifact_progress(self, downloaded_bytes: int, total_bytes: int) -> None:
        self.artifact_progress.append((downloaded_bytes, total_bytes))

    def on_activity(self, activity: RunActivity) -> None:
        self.activities.append(activity)

    def on_server_log(self, lines: tuple[str, ...]) -> None:
        self.server_log.extend(lines)


def _hardware(accelerator_memory_bytes: int = 32 * 1024**3) -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system="Linux",
        operating_system_version="test",
        kernel_version="private-kernel",
        architecture="x86_64",
        cpu_model="test-cpu",
        logical_cpu_count=8,
        memory_bytes=64 * 1024**3,
        accelerators=(
            AcceleratorSnapshot(
                vendor="NVIDIA",
                name="NVIDIA GeForce RTX 5090",
                memory_bytes=accelerator_memory_bytes,
                core_count=None,
                driver_version="private-driver",
                api="CUDA",
            ),
        ),
        warnings=(),
    )


def _apple_hardware() -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system="Darwin",
        operating_system_version="test",
        kernel_version="private-kernel",
        architecture="arm64",
        cpu_model="Apple M5 Pro",
        logical_cpu_count=18,
        memory_bytes=64 * 1024**3,
        accelerators=(
            AcceleratorSnapshot(
                vendor="Apple",
                name="Apple M5 Pro",
                memory_bytes=None,
                core_count=20,
                driver_version=None,
                api="Metal",
            ),
        ),
        warnings=(),
    )


def _multi_accelerator_hardware() -> HardwareSnapshot:
    return replace(
        _hardware(),
        accelerators=(
            *_hardware().accelerators,
            AcceleratorSnapshot(
                vendor="NVIDIA",
                name="NVIDIA RTX PRO 6000",
                memory_bytes=96 * 1024**3,
                core_count=None,
                driver_version="private-driver",
                api="CUDA",
            ),
            AcceleratorSnapshot(
                vendor="NVIDIA",
                name="NVIDIA GeForce RTX 4090",
                memory_bytes=None,
                core_count=None,
                driver_version="private-driver",
                api="CUDA",
            ),
        ),
    )


def _unexpected_hardware() -> HardwareSnapshot:
    raise AssertionError("hardware collection must follow cheap input validation")


class _SilentObserver:
    """Accept replay boundaries without rendering them."""

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        """Ignore one coarse boundary."""

    async def on_finalizing(self) -> None:
        """Allow report commits without a user interface handshake."""

    def on_artifact_progress(self, downloaded_bytes: int, total_bytes: int) -> None:
        """Ignore one managed download measurement."""

    def on_activity(self, activity: RunActivity) -> None:
        """Ignore one run step report."""

    def on_server_log(self, lines: tuple[str, ...]) -> None:
        """Ignore one batch of server log lines."""


def _listed_model_probe(base_url: str, model: str, api_key: str | None = None) -> ContextProbeResult:
    """Answer as a reachable server that lists the model above the full benchmark context."""
    del base_url, model, api_key
    return ContextProbeResult(observed_tokens=131_072, reason=ContextObservationReason.REPORTED)


async def _honouring_probe(
    base_url: str, model: str, client_backend: ClientBackend, api_key: str | None = None
) -> IgnoreEosProbeResult:
    """Answer the capability probe without touching a socket."""
    del base_url, model, client_backend, api_key
    return IgnoreEosProbeResult(
        support=IgnoreEosSupport.HONOURED,
        observed=ProbeAnswer(finish_reason="length", completion_tokens=IGNORE_EOS_PROBE_OUTPUT_TOKENS),
    )


def _not_ollama(base_url: str, api_key: str | None = None) -> bool:
    """Answer the Ollama identity check without touching a socket."""
    del base_url, api_key
    return False


def _installed_llama_offer(
    hardware: HardwareSnapshot,
    candidate: ModelCandidate,
    context_tokens: int | None = None,
) -> tuple[FrameworkOffer, ...]:
    assert candidate.deployment is not None
    deployment = candidate.deployment
    minimum_memory_bytes = derived_minimum_memory_bytes(
        deployment, deployment.context_tokens if context_tokens is None else context_tokens
    )
    available_memory_bytes = hardware.accelerators[0].memory_bytes
    return (
        FrameworkOffer(
            framework="llama-cpp",
            display_name="llama.cpp",
            accelerator_platform="nvidia-cuda",
            installed=True,
            available_memory_bytes=available_memory_bytes,
            minimum_memory_bytes=minimum_memory_bytes,
            memory_fit=None if available_memory_bytes is None else available_memory_bytes >= minimum_memory_bytes,
            installation_hint="Install llama.cpp.",
            support_note="Native GGUF path.",
        ),
    )


@pytest.mark.parametrize(
    ("base_url", "expected_loopback"),
    (
        ("http://127.0.0.1:30000/v1", True),
        ("http://localhost:8000/v1", True),
        ("http://LOCALHOST:8000/v1", True),
        ("http://[::1]:8000/v1", True),
        ("http://0.0.0.0:8000/v1", True),
        ("http://192.168.1.5:8000/v1", False),
        ("https://inference.example/v1", False),
    ),
)
def test_replay_preflight_classifies_endpoint_without_contacting_it(
    tmp_path: Path,
    base_url: str,
    expected_loopback: bool,
) -> None:
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=tmp_path / "results",
        base_url=base_url,
        endpoint_model="served-model",
        client_backend="python",
    )

    preflight = LocalReplayController(hardware_collector=_hardware).preflight(request)

    assert preflight.ready
    assert preflight.manifest_tasks == 1
    assert preflight.manifest_turns == 1
    assert preflight.endpoint_is_loopback is expected_loopback
    assert preflight.reason is None
    assert preflight.hardware is not None
    assert preflight.hardware.accelerator_name == "NVIDIA GeForce RTX 5090"
    assert preflight.hardware.accelerator_memory_bytes == 32 * 1024**3


def test_replay_preflight_reports_unified_memory_for_apple_silicon(tmp_path: Path) -> None:
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:30000/v1",
        endpoint_model="served-model",
        client_backend="python",
    )

    preflight = LocalReplayController(hardware_collector=_apple_hardware).preflight(request)

    assert preflight.hardware is not None
    assert preflight.hardware.accelerator_memory_bytes == 64 * 1024**3


@pytest.mark.parametrize(
    "existing_filename",
    ("summary.json", MEASUREMENT_BINDING_FILENAME),
)
def test_replay_request_rejects_secret_values_and_stale_outputs(tmp_path: Path, existing_filename: str) -> None:
    manifest_path = _manifest(tmp_path)
    with pytest.raises(ValueError, match="environment variable name"):
        ReplayRequest(
            manifest_path=manifest_path,
            output_dir=tmp_path / "results",
            base_url="http://127.0.0.1:30000/v1",
            endpoint_model="served-model",
            api_key_env="secret value",
            client_backend="python",
        )

    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / existing_filename).write_text("already here")
    request = ReplayRequest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        base_url="http://127.0.0.1:30000/v1",
        endpoint_model="served-model",
        client_backend="python",
    )

    preflight = LocalReplayController(hardware_collector=_unexpected_hardware).preflight(request)

    assert not preflight.ready
    assert preflight.block_code is PreflightBlockCode.OUTPUT_DIR_USED
    assert preflight.hardware is None


def test_preflight_rejects_a_results_folder_no_run_could_write(tmp_path: Path) -> None:
    read_only_parent = tmp_path / "read-only"
    read_only_parent.mkdir()
    read_only_parent.chmod(0o500)
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=read_only_parent / "results",
        base_url="http://127.0.0.1:8000/v1",
        endpoint_model="served-model",
        client_backend="python",
    )

    try:
        preflight = LocalReplayController(hardware_collector=_unexpected_hardware).preflight(request)
    finally:
        read_only_parent.chmod(0o700)

    assert not preflight.ready
    assert preflight.block_code is PreflightBlockCode.INPUTS_INVALID
    assert preflight.reason == "Results folder is not writable."


@pytest.mark.parametrize(
    ("port", "startup_timeout_seconds"),
    ((80, DEFAULT_STARTUP_TIMEOUT_SECONDS), (99_999, DEFAULT_STARTUP_TIMEOUT_SECONDS), (8080, 0.0)),
)
def test_managed_choice_names_launch_settings_no_setup_field_can_fix(
    port: int,
    startup_timeout_seconds: float,
) -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)

    with pytest.raises(ManagedLaunchSettingsProblem):
        ManagedDeploymentChoice(
            candidate=_named(catalog, "gemma4-12b-it-q4-0"),
            catalog_as_of=catalog.as_of,
            catalog_digest=BUNDLED_MODEL_CATALOG_DIGEST,
            framework="llama-cpp",
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
        )


def test_replay_request_rejects_a_cleartext_remote_endpoint_that_sends_a_key(tmp_path: Path) -> None:
    with pytest.raises(SetupProblem) as problem:
        ReplayRequest(
            manifest_path=_manifest(tmp_path),
            output_dir=tmp_path / "results",
            base_url="http://inference.example/v1",
            endpoint_model="served-model",
            api_key_env="PRIVATE_TOKEN",
            client_backend="python",
        )

    assert problem.value.block_code is PreflightBlockCode.ENDPOINT_NEEDS_HTTPS


@pytest.mark.parametrize(
    ("base_url", "endpoint_model", "expected_block_code"),
    (
        ("http://127.0.0.1:8000/v1", "", PreflightBlockCode.ENDPOINT_MODEL_EMPTY),
        ("not-a-url", "served-model", PreflightBlockCode.URL_INVALID),
        ("http://user:key@127.0.0.1:8000/v1", "served-model", PreflightBlockCode.URL_INVALID),
    ),
)
def test_replay_request_names_the_setup_problem_it_rejects(
    tmp_path: Path,
    base_url: str,
    endpoint_model: str,
    expected_block_code: PreflightBlockCode,
) -> None:
    with pytest.raises(SetupProblem) as problem:
        ReplayRequest(
            manifest_path=_manifest(tmp_path),
            output_dir=tmp_path / "results",
            base_url=base_url,
            endpoint_model=endpoint_model,
            client_backend="python",
        )

    assert problem.value.block_code is expected_block_code


def test_preflight_reports_an_unreadable_replay_file(tmp_path: Path) -> None:
    request = ReplayRequest(
        manifest_path=tmp_path / "missing" / "manifest.json",
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:8000/v1",
        endpoint_model="served-model",
        client_backend="python",
    )

    preflight = LocalReplayController(hardware_collector=_unexpected_hardware).preflight(request)

    assert not preflight.ready
    assert preflight.block_code is PreflightBlockCode.MANIFEST_UNREADABLE


@pytest.mark.parametrize("api_key", (None, "", "not a key"))
def test_preflight_reports_an_unusable_api_key_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
) -> None:
    api_key_env = "TUI_CONTROLLER_TEST_TOKEN"
    if api_key is None:
        monkeypatch.delenv(api_key_env, raising=False)
    else:
        monkeypatch.setenv(api_key_env, api_key)
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:8000/v1",
        endpoint_model="served-model",
        api_key_env=api_key_env,
        client_backend="python",
    )

    preflight = LocalReplayController(hardware_collector=_unexpected_hardware).preflight(request)

    assert not preflight.ready
    assert preflight.block_code is PreflightBlockCode.API_KEY_ENV_UNSET
    assert preflight.hardware is None

    monkeypatch.setenv(api_key_env, "planted-secret-value")
    assert LocalReplayController(hardware_collector=_hardware).preflight(request).ready


def test_replay_request_exposes_the_measured_clients_normalized_base_url(tmp_path: Path) -> None:
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=tmp_path / "results",
        base_url="http://LOCALHOST:8000/v1/",
        endpoint_model="served-model",
        client_backend="python",
    )

    assert request.normalized_base_url == "http://LOCALHOST:8000/v1"
    assert request.endpoint_is_loopback


def test_preflight_offers_portable_fallback_when_rustcore_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        raise RuntimeError("private import detail")

    monkeypatch.setattr("agentperf_local.tui.replay_contract.validate_rustcore_available", unavailable)
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=tmp_path / "results",
        base_url="http://localhost:8000/v1",
        endpoint_model="served-model",
        client_backend="rust",
    )

    preflight = LocalReplayController(hardware_collector=_unexpected_hardware).preflight(request)

    assert not preflight.ready
    assert preflight.block_code is PreflightBlockCode.CLIENT_UNAVAILABLE
    assert preflight.reason == "Rust client unavailable; install the rust extra or choose Python"
    assert preflight.hardware is None


@pytest.mark.parametrize("cancel_run", (True, False))
async def test_a_stopped_replay_leaves_a_reusable_output_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_run: bool,
) -> None:
    started = asyncio.Event()

    async def stopped_run_manifest(manifest_path: Path, config: RunConfig, *, observer: RunObserver) -> RunResult:
        started.set()
        if not cancel_run:
            raise RuntimeError("private endpoint failure detail")
        await asyncio.Event().wait()
        raise AssertionError("a cancelled replay never finishes")

    monkeypatch.setattr("agentperf_local.tui.controller.run_manifest", stopped_run_manifest)
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:8000/v1",
        endpoint_model="served-model",
        client_backend="python",
    )
    controller = LocalReplayController(
        hardware_collector=_hardware,
        context_prober=_listed_model_probe,
        ignore_eos_prober=_honouring_probe,
        ollama_detector=_not_ollama,
    )

    replay = asyncio.create_task(controller.execute(request, _SilentObserver()))
    async with asyncio.timeout(REPLAY_TIMEOUT_SECONDS):
        await started.wait()
    if cancel_run:
        replay.cancel()
    with pytest.raises(asyncio.CancelledError if cancel_run else RuntimeError):
        await replay

    assert not request.output_dir.exists()
    assert controller.preflight(request).ready


def test_next_run_directory_names_collision_safe_run_folders(tmp_path: Path) -> None:
    moment = datetime(2026, 9, 1, 14, 25, 30)

    first = next_run_directory(tmp_path, now=lambda: moment)

    assert first == tmp_path / "run-20260901-142530"
    assert not first.exists()
    first.mkdir()
    second = next_run_directory(tmp_path, now=lambda: moment)
    assert second == tmp_path / "run-20260901-142530-2"
    second.mkdir()
    assert next_run_directory(tmp_path, now=lambda: moment) == tmp_path / "run-20260901-142530-3"
    missing_root = tmp_path / "missing"
    assert next_run_directory(missing_root, now=lambda: moment) == missing_root / "run-20260901-142530"
    assert not missing_root.exists()


def test_managed_availability_distinguishes_missing_recipes_from_deployable_models() -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)
    controller = LocalManagedReplayController(
        catalog_as_of=catalog.as_of,
        hardware=_hardware(),
        offer_collector=_installed_llama_offer,
    )

    gemma = _named(catalog, "gemma4-12b-it-q4-0")
    # Every bundled model now ships a recipe, so the no-recipe case is built here.
    without_recipe = replace(
        gemma,
        deployment=None,
        artifact_manifest_status="pending-complete-file-manifest",
        device_evidence=tuple(
            replace(evidence, device_id=device_id, architecture=architecture)
            for evidence, (device_id, architecture) in zip(
                gemma.device_evidence,
                (("rtx-5090", "sm120"), ("rtx-pro-6000", "sm120"), ("dgx-spark", "sm121")),
                strict=True,
            )
        ),
    )
    unavailable = controller.availability(without_recipe)
    available = controller.availability(gemma)

    assert not unavailable.can_deploy
    assert unavailable.reason is not None
    assert "cannot start this model" in unavailable.reason
    assert available.can_deploy
    assert tuple(offer.framework for offer in available.deployable_offers) == ("llama-cpp",)


@pytest.mark.parametrize(
    ("accelerator_memory_gib", "context_tokens", "replay_floor_tokens", "can_deploy", "expected_reason"),
    (
        (
            9,
            None,
            None,
            False,
            "Not enough accelerator memory for the full 65,536-token context. "
            "Choose a reduced context — reduced runs are recorded separately from full-context results.",
        ),
        (9, 32_768, None, True, None),
        (
            8.7,
            32_768,
            None,
            False,
            "Not enough accelerator memory for a 32,768-token context. Choose a smaller context.",
        ),
        # 8 GiB fits no offered reduced context, so pointing at the picker would mislead.
        (8, None, None, False, "The detected accelerator does not have enough memory for this model."),
        # Even the smallest offered rung is the one refused, so there is no smaller pick.
        (8, 8_192, None, False, "The detected accelerator does not have enough memory for this model."),
        # The replay floor empties the picker of fitting rungs, so the dead end names both numbers.
        (
            9,
            None,
            65_536,
            False,
            "This replay can not run on this computer: it needs at least 65,536 tokens "
            "of context and this device fits at most 32,768.",
        ),
        (
            9,
            65_536,
            65_536,
            False,
            "This replay can not run on this computer: it needs at least 65,536 tokens "
            "of context and this device fits at most 32,768.",
        ),
    ),
)
def test_managed_availability_points_small_devices_at_a_context_that_fits(
    accelerator_memory_gib: float,
    context_tokens: int | None,
    replay_floor_tokens: int | None,
    can_deploy: bool,
    expected_reason: str | None,
) -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)
    controller = LocalManagedReplayController(
        catalog_as_of=catalog.as_of,
        hardware=_hardware(round(accelerator_memory_gib * 1024**3)),
        offer_collector=_installed_llama_offer,
    )

    availability = controller.availability(
        _named(catalog, "gemma4-12b-it-q4-0"),
        context_tokens=context_tokens,
        replay_floor_tokens=replay_floor_tokens,
    )

    assert availability.can_deploy is can_deploy
    assert availability.reason == expected_reason


@pytest.mark.parametrize(
    ("context_tokens", "resolved", "reduced"),
    ((None, 65_536, False), (65_536, 65_536, False), (32_768, 32_768, True)),
)
def test_managed_choice_resolves_its_context_and_binds_reduced_evidence(
    tmp_path: Path,
    context_tokens: int | None,
    resolved: int,
    reduced: bool,
) -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)
    choice = ManagedDeploymentChoice(
        candidate=_named(catalog, "gemma4-12b-it-q4-0"),
        catalog_as_of=catalog.as_of,
        catalog_digest=BUNDLED_MODEL_CATALOG_DIGEST,
        framework="llama-cpp",
        context_tokens=context_tokens,
        cache_root=tmp_path / "cache",
    )

    assert choice.resolved_context_tokens == resolved
    assert choice.reduced_context is reduced
    # The request derives its unforgeable context evidence from the same choice.
    request = _managed_request(tmp_path, catalog, choice)
    assert request.managed_deployment is choice


@pytest.mark.parametrize("context_tokens", (2_048, 262_144))
def test_managed_choice_rejects_a_context_outside_the_recipe(context_tokens: int) -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)

    with pytest.raises(ValueError, match="context"):
        ManagedDeploymentChoice(
            candidate=_named(catalog, "gemma4-12b-it-q4-0"),
            catalog_as_of=catalog.as_of,
            catalog_digest=BUNDLED_MODEL_CATALOG_DIGEST,
            framework="llama-cpp",
            context_tokens=context_tokens,
        )


def test_managed_preflight_binds_the_selected_catalog_recipe(tmp_path: Path) -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)
    candidate = _named(catalog, "gemma4-12b-it-q4-0")
    deployment = candidate.deployment
    assert deployment is not None
    choice = ManagedDeploymentChoice(
        candidate=candidate,
        catalog_as_of=catalog.as_of,
        catalog_digest=BUNDLED_MODEL_CATALOG_DIGEST,
        framework="llama-cpp",
        cache_root=tmp_path / "cache",
    )
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:8080/v1",
        endpoint_model=deployment.model_alias,
        client_backend="python",
        selection_kind=SelectionKind.BUNDLED_CATALOG_CANDIDATE,
        catalog_profile_id=candidate.profile_id,
        catalog_digest=catalog.file_digest,
        candidate_revision=candidate.hf_revision,
        managed_deployment=choice,
    )
    controller = LocalManagedReplayController(
        catalog_as_of=catalog.as_of,
        hardware=_hardware(),
        offer_collector=_installed_llama_offer,
    )

    preflight = controller.preflight(request)

    assert preflight.ready
    assert preflight.block_code is None
    assert preflight.manifest_tasks == 1
    assert preflight.hardware is not None


async def test_managed_preflight_blocks_a_context_below_the_replay_floor(tmp_path: Path) -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)
    candidate = _named(catalog, "gemma4-12b-it-q4-0")
    deployment = candidate.deployment
    assert deployment is not None
    choice = ManagedDeploymentChoice(
        candidate=candidate,
        catalog_as_of=catalog.as_of,
        catalog_digest=BUNDLED_MODEL_CATALOG_DIGEST,
        framework="llama-cpp",
        context_tokens=32_768,
        cache_root=tmp_path / "cache",
    )
    request = ReplayRequest(
        manifest_path=write_replay_workload(
            tmp_path / "workload",
            name="floored replay",
            write_traces=False,
            required_context_tokens=65_536,
        ),
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:8080/v1",
        endpoint_model=deployment.model_alias,
        client_backend="python",
        selection_kind=SelectionKind.BUNDLED_CATALOG_CANDIDATE,
        catalog_profile_id=candidate.profile_id,
        catalog_digest=catalog.file_digest,
        candidate_revision=candidate.hf_revision,
        managed_deployment=choice,
    )
    controller = LocalManagedReplayController(
        catalog_as_of=catalog.as_of,
        hardware=_hardware(),
        offer_collector=_installed_llama_offer,
    )

    preflight = controller.preflight(request)

    # The same floor the CLI enforces blocks before launch, naming both numbers.
    assert not preflight.ready
    assert preflight.block_code is PreflightBlockCode.CONTEXT_BELOW_REPLAY_FLOOR
    assert preflight.reason == (
        "This replay needs at least 65,536 tokens of context, but the server would start with 32,768."
    )
    with pytest.raises(SetupProblem):
        await controller.execute(request, _NoopObserver())
    assert not (request.output_dir / "deployment.json").exists()


def _managed_request(tmp_path: Path, catalog: ModelCatalog, choice: ManagedDeploymentChoice) -> ReplayRequest:
    """Build the managed replay request the TUI would submit for one deployment choice."""
    deployment = choice.candidate.deployment
    assert deployment is not None
    return ReplayRequest(
        manifest_path=_runnable_manifest(tmp_path / "workload"),
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:8080/v1",
        endpoint_model=deployment.model_alias,
        client_backend="python",
        selection_kind=SelectionKind.BUNDLED_CATALOG_CANDIDATE,
        catalog_profile_id=choice.candidate.profile_id,
        catalog_digest=catalog.file_digest,
        candidate_revision=choice.candidate.hf_revision,
        managed_deployment=choice,
    )


def test_managed_availability_asks_for_a_device_before_reporting_frameworks() -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)
    candidate = _named(catalog, "gemma4-12b-it-q4-0")
    controller = LocalManagedReplayController(
        catalog_as_of=catalog.as_of,
        hardware=_multi_accelerator_hardware(),
        offer_collector=_installed_llama_offer,
    )

    unchosen = controller.availability(candidate)
    chosen = controller.availability(candidate, device_index=1)

    assert controller.device_options() == (
        ManagedDeviceOption(index=0, name="NVIDIA GeForce RTX 5090", memory_bytes=32 * 1024**3),
        ManagedDeviceOption(index=1, name="NVIDIA RTX PRO 6000", memory_bytes=96 * 1024**3),
        ManagedDeviceOption(index=2, name="NVIDIA GeForce RTX 4090", memory_bytes=None),
    )
    assert unchosen.device_selection_required
    assert not unchosen.can_deploy
    assert unchosen.reason == DEVICE_SELECTION_REQUIRED_MESSAGE
    assert not chosen.device_selection_required
    assert chosen.can_deploy
    assert controller.hardware_summary().accelerator_count == 3
    assert controller.hardware_summary(1).accelerator_name == "NVIDIA RTX PRO 6000"


async def test_managed_launch_stops_before_planning_while_no_device_is_chosen(tmp_path: Path) -> None:
    catalog = load_model_catalog(BUNDLED_MODEL_CATALOG_PATH)
    choice = ManagedDeploymentChoice(
        candidate=_named(catalog, "gemma4-12b-it-q4-0"),
        catalog_as_of=catalog.as_of,
        catalog_digest=BUNDLED_MODEL_CATALOG_DIGEST,
        framework="llama-cpp",
        cache_root=tmp_path / "cache",
    )
    request = _managed_request(tmp_path, catalog, choice)
    controller = LocalManagedReplayController(
        catalog_as_of=catalog.as_of,
        hardware=_multi_accelerator_hardware(),
        offer_collector=_installed_llama_offer,
    )

    preflight = controller.preflight(request)

    assert not preflight.ready
    assert preflight.reason == DEVICE_SELECTION_REQUIRED_MESSAGE
    assert preflight.block_code is PreflightBlockCode.DEPLOYMENT_UNAVAILABLE
    with pytest.raises(ManagedDeviceSelectionRequired):
        await controller.execute(request, _NoopObserver())
    assert not (request.output_dir / "deployment.json").exists()


@pytest.mark.parametrize(
    ("reason", "starts_replay"),
    [
        (ContextObservationReason.REPORTED, True),
        (ContextObservationReason.CONTEXT_NOT_REPORTED, True),
        (ContextObservationReason.MODEL_NOT_LISTED, True),
        (ContextObservationReason.MALFORMED_RESPONSE, True),
        (ContextObservationReason.ENDPOINT_UNREACHABLE, False),
        (ContextObservationReason.HTTP_ERROR, False),
    ],
)
async def test_attached_execute_checks_the_server_with_the_key_and_refuses_one_that_does_not_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: ContextObservationReason,
    starts_replay: bool,
) -> None:
    api_key_env = "CONTROLLER_PROBE_TOKEN"
    monkeypatch.setenv(api_key_env, "planted-probe-secret")
    probes: list[tuple[str, str, str | None]] = []
    replays: list[Path] = []

    def probe(base_url: str, model: str, api_key: str | None = None) -> ContextProbeResult:
        probes.append((base_url, model, api_key))
        observed = 131_072 if reason is ContextObservationReason.REPORTED else None
        return ContextProbeResult(observed_tokens=observed, reason=reason)

    async def stopped_run_manifest(manifest_path: Path, config: RunConfig, *, observer: RunObserver) -> RunResult:
        replays.append(manifest_path)
        raise RuntimeError("replay stopped by the test")

    monkeypatch.setattr("agentperf_local.tui.controller.run_manifest", stopped_run_manifest)
    request = ReplayRequest(
        manifest_path=_manifest(tmp_path),
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:8000/v1/",
        endpoint_model="served-model",
        api_key_env=api_key_env,
        client_backend="python",
    )
    controller = LocalReplayController(
        hardware_collector=_hardware,
        context_prober=probe,
        ignore_eos_prober=_honouring_probe,
        ollama_detector=_not_ollama,
    )
    observer = _NoopObserver()

    assert controller.probe_endpoint(request).reason is reason
    with pytest.raises(EndpointProblem if not starts_replay else RuntimeError) as raised:
        await controller.execute(request, observer)

    # Both the consent-time check and the pre-replay check send the run's own key to the
    # normalized URL; the URL itself never reaches the observer.
    assert probes == [("http://127.0.0.1:8000/v1", "served-model", "planted-probe-secret")] * 2
    assert [activity.kind for activity in observer.activities][:2] == [
        RunActivityKind.SETUP_CHECKED,
        RunActivityKind.SERVER_CHECKED,
    ]
    checked = observer.activities[1]
    assert checked.context_probe is not None and checked.context_probe.reason is reason
    assert bool(replays) is starts_replay
    if not starts_replay:
        assert "127.0.0.1" not in str(raised.value)
        assert not request.output_dir.exists()
