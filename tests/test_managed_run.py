"""Exercise the one managed-run pipeline through the CLI and the TUI against a fake local runtime."""

import asyncio
import hashlib
import os
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel

from agentperf_local.cli import main
from agentperf_local.deployment.catalog import (
    BUNDLED_RECIPES_DIGEST,
    BUNDLED_RECIPES_ROOT,
    DeploymentFramework,
    ModelCandidate,
    ModelCatalog,
    RecipeSource,
    load_model_catalog,
)
from agentperf_local.deployment.frameworks import FrameworkIdentity, FrameworkOffer
from agentperf_local.deployment.managed import DeploymentPlan
from agentperf_local.deployment.managed_run import RunActivity, RunActivityKind
from agentperf_local.deployment.model_cache import VerifiedArtifact, VerifiedDeployment
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from agentperf_local.replay.config import RunConfig
from agentperf_local.replay.runner import RunBoundaryEvent, RunObserver, RunResult
from agentperf_local.submission.bundle import validate_submission_bundle
from agentperf_local.tui.controller import LocalManagedReplayController
from agentperf_local.tui.evidence import SelectionKind
from agentperf_local.tui.replay_contract import ManagedDeploymentChoice, ReplayRequest
from tests.fake_nvidia_smi import write_looping_nvidia_smi
from tests.replay_workload import write_replay_workload
from tests.submission_server import LocalSubmissionServer

CHOSEN_DEVICE_INDEX = 1
CHOSEN_ACCELERATOR = "NVIDIA RTX PRO 6000"
SUBMIT_TOKEN_ENV = "AGENTPERF_TEST_SUBMIT_TOKEN"
REPLAY_TIMEOUT_SECONDS = 30.0


def _closed_loopback_port() -> int:
    """Return a loopback port that no server holds."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        address = probe.getsockname()
    if not isinstance(address, tuple) or len(address) < 2 or not isinstance(address[1], int):
        raise AssertionError("loopback probe did not report a port")
    return address[1]


def _accelerator(name: str) -> AcceleratorSnapshot:
    return AcceleratorSnapshot(
        vendor="NVIDIA", name=name, memory_bytes=96 * 1024**3, core_count=None, driver_version=None, api="CUDA"
    )


def _two_gpu_hardware() -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system="Linux",
        operating_system_version="test",
        kernel_version="test",
        architecture="x86_64",
        cpu_model="test-cpu",
        logical_cpu_count=8,
        memory_bytes=64 * 1024**3,
        accelerators=(_accelerator("NVIDIA GeForce RTX 5090"), _accelerator(CHOSEN_ACCELERATOR)),
        warnings=(),
    )


class _FakeRuntime(BaseModel, frozen=True):
    """Name the catalog model the fake runtime serves and the port it listens on."""

    catalog: ModelCatalog
    candidate: ModelCandidate
    framework: DeploymentFramework
    port: int
    offer_collector: Callable[[HardwareSnapshot, ModelCandidate, int | None], tuple[FrameworkOffer, ...]]


def _install_fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework: DeploymentFramework,
    *,
    drops_ignore_eos: bool = False,
) -> _FakeRuntime:
    """Serve a catalog model from tests.managed_server with a cached artifact and a fake nvidia-smi.

    A Splash server runs on Metal and reports the GGUF file it selected, as Splash does.
    """
    fake_smi = write_looping_nvidia_smi(tmp_path / "bin" / "nvidia-smi")
    monkeypatch.setenv("PATH", f"{fake_smi.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr("agentperf_local.cli.options.collect_hardware_snapshot", _two_gpu_hardware)
    catalog = load_model_catalog(BUNDLED_RECIPES_ROOT)
    candidate = next(model for model in catalog.models if framework in model.deployment.frameworks)
    recipe = candidate.deployment
    assert recipe is not None
    artifact_path = tmp_path / "model.gguf"
    artifact_path.write_bytes(b"verified fixture")
    metadata = artifact_path.stat()
    pinned = recipe.artifacts[0]
    artifact = VerifiedDeployment(
        model_path=artifact_path,
        artifacts=(
            VerifiedArtifact(
                filename=pinned.filename,
                path=artifact_path,
                sha256=f"sha256:{pinned.sha256}",
                size_bytes=pinned.size_bytes,
                device_id=metadata.st_dev,
                inode=metadata.st_ino,
                modified_ns=metadata.st_mtime_ns,
            ),
        ),
    )
    port = _closed_loopback_port()
    runtime_digest = f"sha256:{'1' * 64}"

    def provide_artifact(cache_root: Path, selected: ModelCandidate, **_: object) -> VerifiedDeployment:
        del cache_root, selected
        return artifact

    def offer(
        snapshot: HardwareSnapshot, selected: ModelCandidate, context_tokens: int | None = None
    ) -> tuple[FrameworkOffer, ...]:
        del selected, context_tokens
        return (
            FrameworkOffer(
                framework=framework,
                display_name=framework,
                accelerator_platform="nvidia-cuda",
                installed=True,
                available_memory_bytes=snapshot.accelerators[0].memory_bytes,
                minimum_memory_bytes=1,
                memory_fit=True,
                installation_hint="",
                support_note="",
            ),
        )

    def make_plan(
        snapshot: HardwareSnapshot,
        selected: ModelCandidate,
        planned_framework: DeploymentFramework,
        verified: VerifiedDeployment,
        *,
        catalog_digest: str,
        recipe: RecipeSource,
        port: int,
        device_environment: tuple[tuple[str, str], ...] = (),
        context_tokens: int | None = None,
    ) -> DeploymentPlan:
        del snapshot, context_tokens
        splash = selected.deployment.splash
        server_flags: tuple[str, ...] = ("--platform", "cuda", "--backend", planned_framework)
        if splash is not None:
            server_flags = (
                "--platform",
                "metal",
                "--backend",
                "splash",
                "--model",
                f"{selected.hf_repository}:{splash.gguf_variant}",
                "--selected",
                verified.model_path.name,
                *(("--drop-ignore-eos",) if drops_ignore_eos else ()),
            )
        return DeploymentPlan(
            profile_id=selected.profile_id,
            hf_repository=selected.hf_repository,
            hf_revision=selected.hf_revision,
            catalog_digest=catalog_digest,
            recipe=recipe,
            framework=planned_framework,
            accelerator_platform="nvidia-cuda" if splash is None else "apple-metal",
            model_path=verified.model_path,
            artifact_manifest_sha256=verified.manifest_sha256,
            artifact_size_bytes=verified.size_bytes,
            context_tokens=selected.deployment.context_tokens,
            model_alias=candidate.profile_id,
            host="127.0.0.1",
            port=port,
            command=(
                sys.executable,
                "-m",
                "tests.managed_server",
                "--port",
                str(port),
                "--alias",
                candidate.profile_id,
                *server_flags,
            ),
            runtime=FrameworkIdentity(version="fixture", executable_sha256=runtime_digest, fingerprint=runtime_digest),
            device_environment=device_environment,
        )

    monkeypatch.setattr("agentperf_local.deployment.managed_run.ensure_model_artifacts", provide_artifact)
    monkeypatch.setattr("agentperf_local.deployment.managed_run.create_deployment_plan", make_plan)
    monkeypatch.setattr("agentperf_local.cli.replay.framework_offers", offer)
    return _FakeRuntime(catalog=catalog, candidate=candidate, framework=framework, port=port, offer_collector=offer)


@dataclass(slots=True)
class _RecordingObserver:
    """Keep what the TUI would show: each run step and the server's log lines."""

    activities: list[RunActivity] = field(default_factory=list)
    server_log: list[str] = field(default_factory=list)

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        del event

    async def on_finalizing(self) -> None:
        return

    def on_artifact_progress(self, downloaded_bytes: int, total_bytes: int) -> None:
        del downloaded_bytes, total_bytes

    def on_activity(self, activity: RunActivity) -> None:
        self.activities.append(activity)

    def on_server_log(self, lines: tuple[str, ...]) -> None:
        self.server_log.extend(lines)


def _tui_launch(
    runtime: _FakeRuntime, manifest_path: Path, output_dir: Path
) -> tuple[LocalManagedReplayController, ReplayRequest]:
    """Build the controller and request the TUI submits for the fake runtime's model on the chosen device."""
    recipe = runtime.candidate.deployment
    assert recipe is not None
    choice = ManagedDeploymentChoice(
        candidate=runtime.candidate,
        recipe=runtime.catalog.source(runtime.candidate.profile_id),
        catalog_as_of=runtime.catalog.as_of,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        framework=runtime.framework,
        device_index=CHOSEN_DEVICE_INDEX,
        port=runtime.port,
    )
    request = ReplayRequest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        base_url=f"http://127.0.0.1:{runtime.port}/v1",
        endpoint_model=runtime.candidate.profile_id,
        client_backend="python",
        selection_kind=SelectionKind.BUNDLED_CATALOG_CANDIDATE,
        catalog_profile_id=runtime.candidate.profile_id,
        catalog_digest=runtime.catalog.digest,
        candidate_revision=runtime.candidate.hf_revision,
        managed_deployment=choice,
    )
    controller = LocalManagedReplayController(
        catalog_as_of=runtime.catalog.as_of, hardware=_two_gpu_hardware(), offer_collector=runtime.offer_collector
    )
    return controller, request


def _port_is_free(port: int) -> bool:
    """Report whether nothing listens on the port; recent client connections may still linger in TIME_WAIT."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


@pytest.mark.parametrize("framework", ("llama-cpp", "vllm"))
@pytest.mark.parametrize("entry_point", ("cli", "tui"))
def test_managed_run_binds_every_record_to_one_run_and_the_chosen_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entry_point: str,
    framework: DeploymentFramework,
) -> None:
    runtime = _install_fake_runtime(tmp_path, monkeypatch, framework)
    manifest_path = write_replay_workload(tmp_path / "workload", name="managed-e2e", task_count=2)
    output_dir = tmp_path / "results"
    if entry_point == "cli":
        monkeypatch.setenv(SUBMIT_TOKEN_ENV, "aa-test-token")
        with LocalSubmissionServer() as submission_server:
            status = main(
                [
                    "managed-run",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--profile-id",
                    runtime.candidate.profile_id,
                    "--framework",
                    framework,
                    "--device",
                    str(CHOSEN_DEVICE_INDEX),
                    "--client",
                    "python",
                    "--port",
                    str(runtime.port),
                    "--submit-base-url",
                    submission_server.base_url,
                    "--submit-token-env",
                    SUBMIT_TOKEN_ENV,
                ]
            )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        # A configured submit token turns on the advisory allowlist check; this checkout is never listed.
        assert "self-reported only" in captured.err
    else:
        controller, request = _tui_launch(runtime, manifest_path, output_dir)
        assert asyncio.run(controller.execute(request, _RecordingObserver())).success

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "deployment.json",
        "deployment.log",
        "failures.json",
        "measurement.json",
        "power.json",
        "qualification.json",
        "summary.json",
        "tasks.json",
        "telemetry.jsonl",
        "tools.json",
        "turns.jsonl",
    ]
    deployment_bytes = (output_dir / "deployment.json").read_bytes()
    deployment = orjson.loads(deployment_bytes)
    measurement = orjson.loads((output_dir / "measurement.json").read_bytes())
    summary = orjson.loads((output_dir / "summary.json").read_bytes())
    power = orjson.loads((output_dir / "power.json").read_bytes())
    qualification = orjson.loads((output_dir / "qualification.json").read_bytes())
    telemetry_header = orjson.loads((output_dir / "telemetry.jsonl").read_bytes().splitlines()[0])
    run_id = measurement["run_id"]
    assert summary["run_id"] == power["run_id"] == qualification["run_id"] == telemetry_header["run_id"] == run_id
    assert measurement["deployment_digest"] == f"sha256:{hashlib.sha256(deployment_bytes).hexdigest()}"
    assert deployment["deployment"]["catalog_digest"] == BUNDLED_RECIPES_DIGEST
    assert deployment["deployment"]["device_environment"] == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(CHOSEN_DEVICE_INDEX),
    }
    for record in (deployment, measurement):
        assert [accelerator["name"] for accelerator in record["hardware"]["accelerators"]] == [CHOSEN_ACCELERATOR]
    # vLLM otherwise picks tools automatically and stops at the first parsed call; the rule lives in one place.
    expected_tool_choice = "none" if framework == "vllm" else None
    assert summary["config"]["sampling"]["extra_body"].get("tool_choice") == expected_tool_choice
    assert power["phases"][0]["sampled_power_energy_valid"] is True, power["phases"][0]
    assert _port_is_free(runtime.port)

    bundle_dir = tmp_path / "bundle"
    assert main(["prepare-submission", str(output_dir), "--output-dir", str(bundle_dir)]) == 0
    prepared = orjson.loads(capsys.readouterr().out)
    audit = orjson.loads((bundle_dir / "private-audit.json").read_bytes())
    audit_schema = orjson.loads(
        (Path(__file__).parents[1] / "docs" / "schemas" / "private-audit-v2.schema.json").read_bytes()
    )
    Draft202012Validator(audit_schema).validate(audit)
    assert prepared["private_audit"] == {
        "deployment_record": True,
        "runtime_qualification": True,
        "power_summary": True,
    }
    assert audit["deployment"]["record_digest"] == measurement["deployment_digest"]
    assert audit["deployment"]["profile_id"] == runtime.candidate.profile_id
    # The audit carries the exact recipe file, which matches its line in the catalog digest listing.
    recipe = audit["deployment"]["recipe"]
    recipe_bytes = (BUNDLED_RECIPES_ROOT / recipe["path"]).read_bytes()
    assert recipe["text"].encode() == recipe_bytes
    assert recipe["sha256"] == f"sha256:{hashlib.sha256(recipe_bytes).hexdigest()}"
    assert recipe["path"].endswith(f"/{runtime.candidate.profile_id}.yaml")
    assert audit["runtime_qualification"]["run_id"] == run_id
    assert b"model.gguf" not in (bundle_dir / "private-audit.json").read_bytes()
    assert validate_submission_bundle(bundle_dir).aggregate_payload_digest == prepared["aggregate_payload_digest"]


@pytest.mark.parametrize("cancel_run", (True, False))
async def test_a_stopped_managed_replay_stops_the_server_and_drops_its_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_run: bool,
) -> None:
    runtime = _install_fake_runtime(tmp_path, monkeypatch, "llama-cpp")
    started = asyncio.Event()

    async def stopped_run_manifest(manifest_path: Path, config: RunConfig, *, observer: RunObserver) -> RunResult:
        del manifest_path, config, observer
        started.set()
        if not cancel_run:
            raise RuntimeError("fixture replay failure")
        await asyncio.Event().wait()
        raise AssertionError("a cancelled replay never finishes")

    monkeypatch.setattr("agentperf_local.deployment.managed_run.run_manifest", stopped_run_manifest)
    controller, request = _tui_launch(
        runtime, write_replay_workload(tmp_path / "workload", name="managed-stop"), tmp_path / "results"
    )
    observer = _RecordingObserver()

    replay = asyncio.create_task(controller.execute(request, observer))
    async with asyncio.timeout(REPLAY_TIMEOUT_SECONDS):
        await started.wait()
    if cancel_run:
        replay.cancel()
    with pytest.raises(asyncio.CancelledError if cancel_run else RuntimeError):
        await replay

    assert [activity.kind for activity in observer.activities] == [
        RunActivityKind.SETUP_CHECKED,
        RunActivityKind.MODEL_CHECKING,
        RunActivityKind.MODEL_READY,
        RunActivityKind.SERVER_STARTING,
        RunActivityKind.SERVER_READY,
        RunActivityKind.GPU_VERIFIED,
        RunActivityKind.QUALIFYING,
        RunActivityKind.QUALIFIED,
        RunActivityKind.REPLAY_STARTING,
        RunActivityKind.POWER_STARTED,
        RunActivityKind.SERVER_STOPPING,
        RunActivityKind.SERVER_STOPPED,
    ]
    # The log tail outlives the server, so its startup line reaches the view.
    assert any("offloaded 1/1 layers to GPU" in line for line in observer.server_log)
    assert (request.output_dir / "deployment.json").is_file()
    assert not (request.output_dir / "measurement.json").exists()
    assert _port_is_free(runtime.port)


def test_a_cli_managed_run_without_progress_or_power_runs_unobserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _install_fake_runtime(tmp_path, monkeypatch, "llama-cpp")
    manifest_path = write_replay_workload(tmp_path / "workload", name="managed-unobserved")
    output_dir = tmp_path / "results"

    status = main(
        [
            "managed-run",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--profile-id",
            runtime.candidate.profile_id,
            "--framework",
            "llama-cpp",
            "--device",
            str(CHOSEN_DEVICE_INDEX),
            "--client",
            "python",
            "--port",
            str(runtime.port),
            "--no-power",
        ]
    )

    assert status == 0, capsys.readouterr().err
    summary = orjson.loads((output_dir / "summary.json").read_bytes())
    assert summary["observer"]["enabled"] is False
    assert summary["observer"]["duration_ms"] == 0


@pytest.mark.parametrize("precreated", (True, False), ids=("user-folder", "pipeline-folder"))
def test_a_managed_run_that_fails_to_start_removes_only_a_folder_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], precreated: bool
) -> None:
    runtime = _install_fake_runtime(tmp_path, monkeypatch, "llama-cpp")

    def refuse_to_start(plan: DeploymentPlan, log_path: Path) -> None:
        del plan
        log_path.parent.mkdir(parents=True, exist_ok=True)
        raise RuntimeError("fixture start failure")

    monkeypatch.setattr("agentperf_local.deployment.managed_run.start_managed_deployment", refuse_to_start)
    output_dir = tmp_path / "results"
    if precreated:
        output_dir.mkdir()

    status = main(
        [
            "managed-run",
            str(write_replay_workload(tmp_path / "workload", name="managed-start-failure")),
            "--output-dir",
            str(output_dir),
            "--profile-id",
            runtime.candidate.profile_id,
            "--framework",
            "llama-cpp",
            "--client",
            "python",
            "--port",
            str(runtime.port),
        ]
    )

    assert status == 1
    assert "fixture start failure" in capsys.readouterr().err
    assert output_dir.is_dir() is precreated


@dataclass(slots=True)
class _FailingStopObserver(_RecordingObserver):
    """Raise when the server-stopping step arrives, as a broken view might."""

    def on_activity(self, activity: RunActivity) -> None:
        if activity.kind is RunActivityKind.SERVER_STOPPING:
            raise RuntimeError("fixture observer failure")
        self.activities.append(activity)


async def test_an_observer_failure_while_stopping_still_stops_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _install_fake_runtime(tmp_path, monkeypatch, "llama-cpp")
    controller, request = _tui_launch(
        runtime, write_replay_workload(tmp_path / "workload", name="managed-observer-failure"), tmp_path / "results"
    )

    with pytest.raises(RuntimeError, match="fixture observer failure"):
        async with asyncio.timeout(REPLAY_TIMEOUT_SECONDS):
            await controller.execute(request, _FailingStopObserver())

    assert _port_is_free(runtime.port)


@pytest.mark.parametrize("drops_ignore_eos", (False, True), ids=("patched-build", "released-build"))
def test_a_managed_splash_run_needs_a_build_that_honours_ignore_eos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], drops_ignore_eos: bool
) -> None:
    """A Splash build that drops ignore_eos would end every long exact turn early, so the run refuses it."""
    runtime = _install_fake_runtime(tmp_path, monkeypatch, "splash", drops_ignore_eos=drops_ignore_eos)
    output_dir = tmp_path / "results"

    status = main(
        [
            "managed-run",
            str(write_replay_workload(tmp_path / "workload", name="managed-splash")),
            "--output-dir",
            str(output_dir),
            "--profile-id",
            runtime.candidate.profile_id,
            "--framework",
            "splash",
            "--device",
            str(CHOSEN_DEVICE_INDEX),
            "--client",
            "python",
            "--port",
            str(runtime.port),
            "--no-power",
        ]
    )

    captured = capsys.readouterr()
    if drops_ignore_eos:
        assert status != 0
        assert "the exact policy needs a Splash build that honours ignore_eos" in captured.err
        assert not (output_dir / "measurement.json").exists()
    else:
        assert status == 0, captured.err
        summary = orjson.loads((output_dir / "summary.json").read_bytes())
        assert summary["config"]["output_tokens"]["policy"] == "exact"
    assert _port_is_free(runtime.port)
