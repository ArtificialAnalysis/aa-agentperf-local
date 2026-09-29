"""Exercise managed deployment planning through its public surface."""

from __future__ import annotations

import errno
import gzip
import hashlib
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import orjson
import pytest
from huggingface_hub import scan_cache_dir
from huggingface_hub.file_download import repo_folder_name

from agentperf_local.common.durable_files import PRIVATE_FILE_PERMISSIONS
from agentperf_local.common.models import replace_fields
from agentperf_local.deployment.catalog import (
    BUNDLED_RECIPES_DIGEST,
    BUNDLED_RECIPES_ROOT,
    DeploymentArtifact,
    DeploymentFramework,
    DeploymentMemory,
    LlamaCppLaunch,
    ModelCandidate,
    ModelDeployment,
    SpeculationPolicy,
    VllmLaunch,
    load_model_catalog,
)
from agentperf_local.deployment.context_policy import (
    MINIMUM_CONTEXT_TOKENS,
    derived_minimum_memory_bytes,
)
from agentperf_local.deployment.endpoint_probes import probe_served_context_tokens
from agentperf_local.deployment.frameworks import (
    CommandFinder,
    FrameworkIdentity,
    framework_offers,
)
from agentperf_local.deployment.managed import (
    PROCESS_STOP_TIMEOUT_SECONDS,
    DeploymentPlan,
    ManagedDeployment,
    bind_snapshot_to_device,
    create_deployment_plan,
    start_managed_deployment,
    verify_gpu_startup,
    wait_for_deployment,
)
from agentperf_local.deployment.model_cache import default_model_cache_root, ensure_model_artifacts
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from tests.fake_executable import write_python_executable
from tests.file_modes import has_mode

MODEL_BYTES = b"small deterministic GGUF fixture"
MODEL_DIGEST = hashlib.sha256(MODEL_BYTES).hexdigest()
HF_REPOSITORY_DIRECTORY = "models--example--model-gguf"
MANAGED_TEST_STARTUP_TIMEOUT_SECONDS = 5.0
CANCELLATION_TEST_TIMEOUT_SECONDS = 2.0
POSIX_PROCESS_GROUPS = pytest.mark.skipif(sys.platform == "win32", reason="patches os.killpg, which Windows lacks")
RETRY_TEST_BACKOFF_SECONDS = 0.2
IDLE_CHILD_SECONDS = 60.0
SHORT_CHILD_SECONDS = 0.2
RESUMED_PREFIX_BYTES = 10
PROFILE_CONTEXT_TOKENS = 65536
REDUCED_CONTEXT_TOKENS = 32768
SERVED_CONTEXT_TOKENS_MISMATCH = 8192
# The fixture uses the attention shape of the bundled Gemma recipe: 12 full-attention
# and 36 sliding-attention layers of 8 KV heads, so its KV payload is 6.28 GiB at the
# full context, and 4.47 GiB of runtime overhead sits on top of the artifact bytes.
FIXTURE_RUNTIME_OVERHEAD_BYTES = 4_801_726_336
FIXTURE_MEMORY = DeploymentMemory(
    full_attention_layers=12,
    full_kv_heads=8,
    full_kv_head_dimension=256,
    sliding_attention_layers=36,
    sliding_kv_heads=8,
    sliding_kv_head_dimension=256,
    sliding_cached_tokens=1024,
    kv_bytes_per_scalar=2,
    recurrent_state_slots=0,
    constant_state_bytes=0,
    runtime_overhead_bytes=FIXTURE_RUNTIME_OVERHEAD_BYTES,
)

FIXTURE_MINIMUM_MEMORY_BYTES = len(MODEL_BYTES) + 6_744_440_832 + FIXTURE_RUNTIME_OVERHEAD_BYTES
# A card can report a few dozen MiB under its listed capacity; 48 MiB short stays inside the 64 MiB slack.
MEMORY_WITHIN_SLACK_BYTES = FIXTURE_MINIMUM_MEMORY_BYTES - 48 * 1024 * 1024
# A shortfall past the 64 MiB slack must fail the memory gate.
MEMORY_BELOW_SLACK_BYTES = FIXTURE_MINIMUM_MEMORY_BYTES - 128 * 1024 * 1024
# The reduced-context minimum drops by the KV savings: 32768 tokens keep about 3.3 GiB
# of KV instead of 6.28 GiB, so 9 GiB hardware fits between the two floors.
REDUCED_MINIMUM_MEMORY_BYTES = len(MODEL_BYTES) + 3_523_215_360 + FIXTURE_RUNTIME_OVERHEAD_BYTES
SMALL_HARDWARE_MEMORY_BYTES = 9 * 1024**3


def _candidate() -> ModelCandidate:
    deployment = ModelDeployment(
        artifact_kind="gguf-single-file",
        artifacts=(DeploymentArtifact(filename="model.gguf", sha256=MODEL_DIGEST, size_bytes=len(MODEL_BYTES)),),
        context_tokens=PROFILE_CONTEXT_TOKENS,
        frameworks=("llama-cpp",),
        memory=FIXTURE_MEMORY,
        runtime_versions=(),
        moe_runner_backend=None,
    )
    return ModelCandidate(
        profile_id="fixture-q4-0",
        as_of="2026-09-21",
        display_name="Fixture Q4_0",
        model_release_slug="fixture",
        hf_repository="example/model-gguf",
        hf_revision="a" * 40,
        devices=("nvidia-cuda", "amd-rocm", "apple-silicon"),
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        thinking_policy="disabled",
        speculation_policy="disabled-target-only-baseline",
        deployment=deployment,
    )


def _hardware(vendor: str, api: str, memory_bytes: int = 32 * 1024**3) -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system="Darwin" if vendor == "Apple" else "Linux",
        operating_system_version="test",
        kernel_version="test",
        architecture="arm64" if vendor == "Apple" else "x86_64",
        cpu_model="test-cpu",
        logical_cpu_count=8,
        memory_bytes=memory_bytes,
        accelerators=(
            AcceleratorSnapshot(
                vendor=vendor,
                name=f"{vendor} test GPU",
                memory_bytes=None if vendor == "Apple" else memory_bytes,
                core_count=None,
                driver_version=None,
                api=api,
                memory_is_unified=vendor == "Apple",
            ),
        ),
        warnings=(),
    )


def _installed_command(command: str) -> str:
    del command
    return sys.executable


def _cancel_requested() -> bool:
    return True


def _blob_path(cache_root: Path) -> Path:
    return cache_root / HF_REPOSITORY_DIRECTORY / "blobs" / MODEL_DIGEST


def _snapshot_path(cache_root: Path, candidate: ModelCandidate) -> Path:
    revision_dir = cache_root / HF_REPOSITORY_DIRECTORY / "snapshots" / candidate.hf_revision
    return revision_dir / candidate.deployment.artifacts[0].filename


def _cached_artifact(
    cache_root: Path,
    candidate: ModelCandidate,
    *,
    content: bytes = MODEL_BYTES,
    symlinked: bool = True,
) -> Path:
    """Lay one artifact out exactly as huggingface_hub caches it."""
    blob = cache_root / HF_REPOSITORY_DIRECTORY / "blobs" / hashlib.sha256(content).hexdigest()
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(content)
    snapshot = _snapshot_path(cache_root, candidate)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if symlinked:
        snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
    else:
        # huggingface_hub copies the file instead on filesystems without symlinks.
        snapshot.write_bytes(content)
    return snapshot


def _refuse_download(candidate: ModelCandidate, artifact: DeploymentArtifact) -> str:
    raise AssertionError("this test expects the artifact to be served without a download")


def _incomplete_path(cache_root: Path) -> Path:
    return cache_root / HF_REPOSITORY_DIRECTORY / "blobs" / f"{MODEL_DIGEST}.incomplete"


def _local_port(server: ThreadingHTTPServer) -> int:
    address = server.server_address
    assert isinstance(address, tuple)
    port = address[1]
    assert isinstance(port, int)
    return port


@dataclass(slots=True)
class _ArtifactServer:
    """Record what one fixture artifact server was asked for."""

    url: str
    requested_ranges: list[str | None]
    requested_encodings: list[str | None]


@contextmanager
def _artifact_server(
    *,
    body: bytes = MODEL_BYTES,
    failures: int = 0,
    compress_unless_identity: bool = False,
) -> Iterator[_ArtifactServer]:
    """Serve the fixture artifact with range resumption after a scripted number of failures.

    A hosting front end compresses small text responses unless identity encoding is asked
    for, which the compress_unless_identity server reproduces.
    """
    state = _ArtifactServer(url="", requested_ranges=[], requested_encodings=[])

    class ArtifactHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requested_range = self.headers.get("Range")
            state.requested_ranges.append(requested_range)
            encoding = self.headers.get("Accept-Encoding")
            state.requested_encodings.append(encoding)
            if len(state.requested_ranges) <= failures:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            start = 0 if requested_range is None else int(requested_range.removeprefix("bytes=").removesuffix("-"))
            if requested_range is None:
                self.send_response(HTTPStatus.OK)
            else:
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            payload = body[start:]
            if compress_unless_identity and encoding != "identity":
                payload = gzip.compress(payload)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ArtifactHandler)
    state.url = f"http://127.0.0.1:{_local_port(server)}/model.gguf"
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(CANCELLATION_TEST_TIMEOUT_SECONDS)


@contextmanager
def _models_endpoint(port: int, model_alias: str, served_context_tokens: int | None) -> Iterator[None]:
    """Serve one llama.cpp style model list; None hides the served context length."""
    model: dict[str, object] = {"id": model_alias, "object": "model"}
    if served_context_tokens is not None:
        model["meta"] = {"n_ctx": served_context_tokens}
    body = orjson.dumps({"object": "list", "data": [model]})

    class ModelsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), ModelsHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(CANCELLATION_TEST_TIMEOUT_SECONDS)


def _child_deployment(plan: DeploymentPlan, log_path: Path, lifetime_seconds: float) -> ManagedDeployment:
    """Own a placeholder child so lifecycle checks can inspect a real process group."""
    log_stream = log_path.open("wb")
    process = subprocess.Popen(
        (sys.executable, "-c", f"import time; time.sleep({lifetime_seconds})"),
        stdin=subprocess.DEVNULL,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    return ManagedDeployment(
        plan=plan,
        process=process,
        process_group_id=process.pid if sys.platform == "win32" else os.getpgid(process.pid),
        log_stream=log_stream,
        log_path=log_path,
    )


@pytest.mark.parametrize(
    ("hardware", "expected"),
    (
        (_hardware("NVIDIA", "CUDA"), ("llama-cpp",)),
        (_hardware("AMD", "ROCm"), ("llama-cpp",)),
        (_hardware("Apple", "Metal"), ("llama-cpp",)),
    ),
)
def test_offers_only_compatible_frameworks_for_the_detected_gpu(
    hardware: HardwareSnapshot,
    expected: tuple[str, ...],
) -> None:
    offers = framework_offers(hardware, _candidate(), command_finder=_installed_command)

    assert tuple(offer.framework for offer in offers) == expected
    assert all(offer.installed for offer in offers)
    assert all(offer.memory_fit is True for offer in offers)


def test_offers_reject_a_gpu_family_absent_from_the_model_recipe() -> None:
    candidate = _candidate()
    nvidia_only = replace_fields(candidate, devices=("nvidia-cuda",))

    offers = framework_offers(_hardware("Apple", "Metal"), nvidia_only, command_finder=_installed_command)

    assert offers == ()


@pytest.mark.parametrize("vendor,api", (("NVIDIA", "CUDA"), ("Apple", "Metal")))
def test_offers_report_when_detected_memory_is_below_the_canonical_requirement(vendor: str, api: str) -> None:
    offers = framework_offers(_hardware(vendor, api, 8 * 1024**3), _candidate(), command_finder=_installed_command)

    assert offers
    assert all(offer.memory_fit is False for offer in offers)


@pytest.mark.parametrize("symlinked", (True, False))
def test_cached_artifact_builds_a_pinned_framework_launch_plan(
    tmp_path: Path,
    symlinked: bool,
) -> None:
    candidate = _candidate()
    _cached_artifact(tmp_path, candidate, symlinked=symlinked)

    cached = ensure_model_artifacts(tmp_path, candidate)
    plan = create_deployment_plan(
        _hardware("NVIDIA", "CUDA"),
        candidate,
        "llama-cpp",
        cached,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_installed_command,
        alias_nonce="test",
    )

    assert plan.artifact_manifest_sha256 == cached.manifest_sha256
    assert plan.model_alias == "fixture-q4-0-test"
    assert plan.base_url == "http://127.0.0.1:8080/v1"
    assert "--ctx-size" in plan.command
    assert str(cached.model_path) in plan.command
    assert plan.command[-2:] == ("--verbosity", "4")
    # The shareable command names the executable and the model by placeholder, never by local path.
    assert plan.server_launch_command.startswith('"$PYTHON" --model "$MODEL_DIR"/model.gguf --alias fixture-q4-0-test')
    assert str(tmp_path) not in plan.server_launch_command
    assert (plan.model_release_slug, plan.accelerator_backend) == ("fixture", "cuda")


def test_split_gguf_recipe_launches_the_first_part_by_its_snapshot_name(tmp_path: Path) -> None:
    """llama.cpp finds later split parts by the first part's name, so a blob path can not load them."""
    parts = tuple(
        (f"UD-Q4_K_M/model-0000{index}-of-00002.gguf", f"split GGUF part {index}".encode()) for index in (1, 2)
    )
    artifacts = tuple(
        DeploymentArtifact(filename=name, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
        for name, content in parts
    )
    base = _candidate()
    candidate = replace_fields(
        base,
        deployment=replace_fields(
            base.deployment,
            artifact_kind="gguf-file-set",
            artifacts=artifacts,
            model_filename=artifacts[0].filename,
        ),
    )
    repository_root = tmp_path / repo_folder_name(repo_id=candidate.hf_repository, repo_type="model")
    snapshot_root = repository_root / "snapshots" / candidate.hf_revision
    for (name, content), artifact in zip(parts, artifacts, strict=True):
        blob = repository_root / "blobs" / artifact.sha256
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        snapshot = snapshot_root / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))

    plan = create_deployment_plan(
        _hardware("NVIDIA", "CUDA"),
        candidate,
        "llama-cpp",
        ensure_model_artifacts(tmp_path, candidate),
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_installed_command,
        alias_nonce="test",
    )

    assert plan.command[plan.command.index("--model") + 1] == str(snapshot_root / artifacts[0].filename)


@pytest.mark.parametrize(
    ("speculation_policy", "spec_type"),
    [("enabled-dflash-external-draft", "draft-dflash"), ("enabled-dspark-external-draft", "draft-dspark")],
)
def test_external_draft_recipe_launches_both_pinned_gguf_files(
    tmp_path: Path, speculation_policy: SpeculationPolicy, spec_type: str
) -> None:
    """A multi-file recipe must serve its target and pass its verified draft to llama.cpp."""
    target_bytes = b"target GGUF fixture"
    draft_bytes = b"draft GGUF fixture"
    target = DeploymentArtifact(
        filename="target.gguf",
        sha256=hashlib.sha256(target_bytes).hexdigest(),
        size_bytes=len(target_bytes),
    )
    draft = DeploymentArtifact(
        filename="draft.gguf",
        sha256=hashlib.sha256(draft_bytes).hexdigest(),
        size_bytes=len(draft_bytes),
        source_repository="example/draft-gguf",
        source_revision="b" * 40,
    )
    base = _candidate()
    deployment = replace_fields(
        base.deployment,
        artifact_kind="gguf-file-set",
        artifacts=(draft, target),
        model_filename=target.filename,
        llama_cpp=LlamaCppLaunch(
            batch_size=2048,
            ubatch_size=1024,
            speculative_tokens=7,
            draft_model_filename=draft.filename,
            target_backend_sampling=True,
            draft_backend_sampling=True,
            context_checkpoints=2,
        ),
    )
    candidate = replace_fields(
        base,
        speculation_policy=speculation_policy,
        deployment=deployment,
    )

    def cache_artifact(repository: str, revision: str, artifact: DeploymentArtifact, content: bytes) -> Path:
        repository_root = tmp_path / repo_folder_name(repo_id=repository, repo_type="model")
        blob = repository_root / "blobs" / artifact.sha256
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        snapshot = repository_root / "snapshots" / revision / artifact.filename
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
        return blob

    target_path = cache_artifact(candidate.hf_repository, candidate.hf_revision, target, target_bytes)
    assert draft.source_repository is not None
    assert draft.source_revision is not None
    draft_path = cache_artifact(draft.source_repository, draft.source_revision, draft, draft_bytes)
    verified = ensure_model_artifacts(tmp_path, candidate)

    plan = create_deployment_plan(
        _hardware("NVIDIA", "CUDA"),
        candidate,
        "llama-cpp",
        verified,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_installed_command,
        alias_nonce="test",
    )

    assert plan.command[plan.command.index("--model") + 1] == str(target_path)
    assert plan.command[plan.command.index("--model-draft") + 1] == str(draft_path)
    assert plan.command[plan.command.index("--spec-type") + 1] == spec_type
    assert plan.command[plan.command.index("--spec-draft-n-max") + 1] == "7"
    assert plan.command[plan.command.index("--ctx-checkpoints") + 1] == "2"
    assert plan.command[plan.command.index("--batch-size") + 1] == "2048"
    assert plan.command[plan.command.index("--ubatch-size") + 1] == "1024"
    assert "--backend-sampling" in plan.command
    assert "--spec-draft-backend-sampling" in plan.command
    assert '--model "$MODEL_DIR"/target.gguf' in plan.server_launch_command
    assert '--model-draft "$MODEL_DIR"/draft.gguf' in plan.server_launch_command
    assert str(tmp_path) not in plan.server_launch_command


@pytest.mark.parametrize(
    ("profile_id", "vendor", "api", "device", "load_mode", "threads", "depth", "ubatch"),
    (
        ("qwen35-9b-q4-k-m-mtp-strix-halo", "AMD", "ROCm", "ROCm0", "none", "16", "2", "2048"),
        ("qwen36-35b-a3b-q4-k-m-mtp-strix-halo", "AMD", "ROCm", "Vulkan0", "none", "16", "2", "512"),
        ("qwen35-9b-q4-k-m-mtp-m5-pro", "Apple", "Metal", "MTL0", "mmap", "6", "2", "512"),
        ("qwen38-27b-q4-k-m-mtp-m5-pro", "Apple", "Metal", "MTL0", "mmap", "6", "4", "512"),
    ),
)
def test_screened_recipes_keep_their_measured_launch_settings(
    tmp_path: Path,
    profile_id: str,
    vendor: str,
    api: str,
    device: str,
    load_mode: str,
    threads: str,
    depth: str,
    ubatch: str,
) -> None:
    catalog = load_model_catalog(BUNDLED_RECIPES_ROOT)
    recipe = next(model for model in catalog.models if model.profile_id == profile_id)
    base = _candidate()
    candidate = replace_fields(
        base,
        speculation_policy=recipe.speculation_policy,
        devices=recipe.devices,
        deployment=replace_fields(base.deployment, llama_cpp=recipe.deployment.llama_cpp),
    )
    _cached_artifact(tmp_path, candidate)
    verified = ensure_model_artifacts(tmp_path, candidate)
    hardware = _hardware(vendor, api)
    plan = create_deployment_plan(
        hardware,
        candidate,
        "llama-cpp",
        verified,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_installed_command,
        alias_nonce="test",
    )
    for flag, expected in (
        ("--device", device),
        ("--load-mode", load_mode),
        ("--threads", threads),
        ("--spec-draft-n-max", depth),
        ("--ubatch-size", ubatch),
        ("--batch-size", "2048"),
        ("--flash-attn", "on"),
        ("--fit", "off"),
        ("--cache-ram", "0"),
    ):
        assert plan.command[plan.command.index(flag) + 1] == expected
    assert "--spec-draft-backend-sampling" in plan.command
    if device == "Vulkan0":
        with pytest.raises(ValueError, match="unpinned single-accelerator"):
            create_deployment_plan(
                hardware,
                candidate,
                "llama-cpp",
                verified,
                catalog_digest=BUNDLED_RECIPES_DIGEST,
                command_finder=_installed_command,
                device_environment=(("HIP_VISIBLE_DEVICES", "1"),),
            )


def test_lazy_mode_recipe_passes_its_lazy_read_flag(tmp_path: Path) -> None:
    """A recipe that reads tensors on demand launches llama.cpp with that lazy mode."""
    catalog = load_model_catalog(BUNDLED_RECIPES_ROOT)
    recipe = next(model for model in catalog.models if model.profile_id == "qwen35-9b-q4-k-m-mtp-strix-halo")
    base = _candidate()
    assert recipe.deployment.llama_cpp is not None
    candidate = replace_fields(
        base,
        speculation_policy=recipe.speculation_policy,
        devices=recipe.devices,
        deployment=replace_fields(
            base.deployment, llama_cpp=replace_fields(recipe.deployment.llama_cpp, lazy_mode="on-direct")
        ),
    )
    _cached_artifact(tmp_path, candidate)
    plan = create_deployment_plan(
        _hardware("AMD", "ROCm"),
        candidate,
        "llama-cpp",
        ensure_model_artifacts(tmp_path, candidate),
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_installed_command,
        alias_nonce="test",
    )

    assert plan.command[plan.command.index("--lazy-mode") + 1] == "on-direct"


def test_launch_plan_rejects_a_framework_the_recipe_does_not_name(tmp_path: Path) -> None:
    """A GGUF recipe names llama.cpp alone, so asking for the weights runtime is refused."""
    candidate = _candidate()
    _cached_artifact(tmp_path, candidate)

    with pytest.raises(ValueError, match="does not support"):
        create_deployment_plan(
            _hardware("NVIDIA", "CUDA"),
            candidate,
            "sglang",
            ensure_model_artifacts(tmp_path, candidate),
            catalog_digest=BUNDLED_RECIPES_DIGEST,
            command_finder=_installed_command,
            alias_nonce="test",
        )


def test_rejects_a_cached_artifact_with_different_bytes(tmp_path: Path) -> None:
    candidate = _candidate()
    _cached_artifact(tmp_path, candidate, content=b"wrong")

    with pytest.raises(ValueError, match="size does not match"):
        ensure_model_artifacts(tmp_path, candidate)


def test_cached_artifact_verification_honors_cooperative_cancellation(tmp_path: Path) -> None:
    candidate = _candidate()
    artifact = _cached_artifact(tmp_path, candidate)

    with pytest.raises(InterruptedError, match="canceled"):
        ensure_model_artifacts(tmp_path, candidate, cancellation_requested=_cancel_requested)

    assert artifact.read_bytes() == MODEL_BYTES


def test_stalled_download_honors_cancellation_with_a_real_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_started = threading.Event()
    cancellation = threading.Event()
    release_response = threading.Event()
    errors: list[BaseException] = []

    class StalledHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", str(len(MODEL_BYTES)))
            self.end_headers()
            download_started.set()
            release_response.wait(CANCELLATION_TEST_TIMEOUT_SECONDS)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), StalledHandler)
    address = server.server_address
    assert isinstance(address, tuple)
    port = address[1]
    assert isinstance(port, int)
    model_url = f"http://127.0.0.1:{port}/model.gguf"
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    def download() -> None:
        try:
            ensure_model_artifacts(tmp_path, _candidate(), cancellation_requested=cancellation.is_set)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr("agentperf_local.deployment.model_cache._artifact_url", lambda candidate, deployment: model_url)
    download_thread = threading.Thread(target=download)
    try:
        download_thread.start()
        assert download_started.wait(CANCELLATION_TEST_TIMEOUT_SECONDS)
        cancellation.set()
        release_response.set()
        download_thread.join(CANCELLATION_TEST_TIMEOUT_SECONDS)
    finally:
        cancellation.set()
        release_response.set()
        server.shutdown()
        server.server_close()
        server_thread.join(CANCELLATION_TEST_TIMEOUT_SECONDS)

    assert not download_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], InterruptedError)
    assert tuple(tmp_path.rglob("*.incomplete")) == (_incomplete_path(tmp_path),)
    assert not _blob_path(tmp_path).exists()
    # The abandoned attempt must leave a shape the hub's own scanner accepts.
    assert (tmp_path / HF_REPOSITORY_DIRECTORY / "snapshots").is_dir()
    assert scan_cache_dir(tmp_path).warnings == []


def _free_port() -> int:
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        address = port_probe.getsockname()
        assert isinstance(address, tuple)
        port = address[1]
        assert isinstance(port, int)
    return port


def _owned_plan(tmp_path: Path, *, offloaded: str = "1/1", port: int | None = None) -> DeploymentPlan:
    port = _free_port() if port is None else port
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(MODEL_BYTES)
    runtime_digest = f"sha256:{'1' * 64}"
    plan = DeploymentPlan(
        profile_id="fixture-q4-0",
        model_release_slug="fixture",
        hf_repository="example/model-gguf",
        hf_revision="a" * 40,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        framework="llama-cpp",
        accelerator_platform="apple-metal",
        accelerator_backend="metal",
        model_path=artifact,
        artifact_manifest_sha256=f"sha256:{MODEL_DIGEST}",
        artifact_size_bytes=len(MODEL_BYTES),
        context_tokens=PROFILE_CONTEXT_TOKENS,
        model_alias="fixture-q4-0",
        host="127.0.0.1",
        port=port,
        command=(
            sys.executable,
            "-m",
            "tests.managed_server",
            "--port",
            str(port),
            "--alias",
            "fixture-q4-0",
            "--offloaded",
            offloaded,
        ),
        server_launch_command="python -m tests.managed_server",
        runtime=FrameworkIdentity(
            version="fixture",
            executable_sha256=runtime_digest,
            fingerprint=runtime_digest,
        ),
    )
    return plan


def test_owned_deployment_reaches_readiness_and_is_stopped(tmp_path: Path) -> None:
    plan = _owned_plan(tmp_path)
    log_path = tmp_path / "deployment.log"

    with start_managed_deployment(plan, log_path) as deployment:
        wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)
        verify_gpu_startup(deployment)
        assert deployment.process.poll() is None

    assert deployment.process.poll() is not None
    assert has_mode(log_path, PRIVATE_FILE_PERMISSIONS)


@pytest.mark.parametrize(
    ("framework", "hint"),
    (
        ("llama-cpp", "older llama.cpp build"),
        ("sglang", "killed by the operating system"),
        ("vllm", "ran out of memory"),
    ),
)
def test_a_server_that_exits_before_readiness_names_its_own_runtime(
    tmp_path: Path,
    framework: DeploymentFramework,
    hint: str,
) -> None:
    """A failure hint must name the runtime that failed, not whichever one is bundled."""
    plan = replace_fields(
        _owned_plan(tmp_path), framework=framework, command=(sys.executable, "-c", "raise SystemExit(1)")
    )
    log_path = tmp_path / "deployment.log"

    with start_managed_deployment(plan, log_path) as deployment:
        with pytest.raises(RuntimeError, match=hint) as failure:
            wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)

    assert f"managed {framework} server exited" in str(failure.value)


def test_owned_deployment_rejects_partial_gpu_offload(tmp_path: Path) -> None:
    plan = _owned_plan(tmp_path, offloaded="1/2")
    log_path = tmp_path / "deployment.log"

    with start_managed_deployment(plan, log_path) as deployment:
        wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)
        with pytest.raises(RuntimeError, match="did not report full apple-metal offload"):
            verify_gpu_startup(deployment)


@pytest.mark.parametrize(
    ("requested_device", "reported_backend", "offloaded", "passes"),
    (
        ("Vulkan0", "vulkan", "1/1", True),
        ("Vulkan0", "vulkan", "0/1", False),
        ("Vulkan0", "rocm", "1/1", False),
        ("ROCm0", "vulkan", "1/1", False),
        ("ROCm0", "rocm", "1/1", True),
    ),
)
def test_amd_startup_requires_the_requested_backend_and_full_offload(
    tmp_path: Path, requested_device: str, reported_backend: str, offloaded: str, passes: bool
) -> None:
    base = _owned_plan(tmp_path, offloaded=offloaded)
    plan = replace_fields(
        base,
        accelerator_platform="amd-rocm",
        command=(*base.command, "--device", requested_device, "--platform", reported_backend),
    )
    with start_managed_deployment(plan, tmp_path / "deployment.log") as deployment:
        wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)
        if passes:
            verify_gpu_startup(deployment)
        else:
            with pytest.raises(RuntimeError, match="did not report"):
                verify_gpu_startup(deployment)


def test_owned_deployment_rejects_an_endpoint_port_owned_by_another_process(tmp_path: Path) -> None:
    plan = _owned_plan(tmp_path)
    log_path = tmp_path / "deployment.log"

    with socket.socket() as port_owner:
        port_owner.bind((plan.host, plan.port))
        port_owner.listen(1)
        with pytest.raises(ValueError, match="is already in use; pass --port to choose another"):
            start_managed_deployment(plan, log_path)

    assert not log_path.exists()


def _binds_without_address_reuse(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def test_owned_deployment_accepts_a_port_left_behind_by_a_stopped_server(tmp_path: Path) -> None:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    address = listener.getsockname()
    assert isinstance(address, tuple)
    port = address[1]
    assert isinstance(port, int)
    listener.listen(1)
    client = socket.create_connection(("127.0.0.1", port))
    accepted, _ = listener.accept()
    accepted.close()
    client.close()
    listener.close()
    if _binds_without_address_reuse(port):
        pytest.skip("this platform did not leave the stopped server's port in TIME_WAIT")

    plan = _owned_plan(tmp_path, port=port)
    log_path = tmp_path / "deployment.log"

    with start_managed_deployment(plan, log_path) as deployment:
        wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)


def test_failed_launch_leaves_the_output_directory_fresh(tmp_path: Path) -> None:
    plan = replace_fields(_owned_plan(tmp_path), command=(str(tmp_path / "missing-llama-server"),))
    log_path = tmp_path / "deployment.log"

    with pytest.raises(OSError):
        start_managed_deployment(plan, log_path)

    assert not log_path.exists()


@pytest.mark.parametrize(
    ("served_context_tokens", "expected_error"),
    (
        (PROFILE_CONTEXT_TOKENS, None),
        (
            SERVED_CONTEXT_TOKENS_MISMATCH,
            f"managed server reports a {SERVED_CONTEXT_TOKENS_MISMATCH}-token context "
            f"but the profile requires {PROFILE_CONTEXT_TOKENS}",
        ),
        (None, "managed server did not report its served context length (meta.n_ctx)"),
    ),
)
def test_readiness_requires_the_served_context_to_match_the_profile(
    tmp_path: Path,
    served_context_tokens: int | None,
    expected_error: str | None,
) -> None:
    plan = _owned_plan(tmp_path)
    log_path = tmp_path / "deployment.log"

    with _models_endpoint(plan.port, plan.model_alias, served_context_tokens):
        with _child_deployment(plan, log_path, IDLE_CHILD_SECONDS) as deployment:
            if expected_error is None:
                served = wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)
                assert served == PROFILE_CONTEXT_TOKENS
                return
            with pytest.raises(RuntimeError, match=re.escape(f"{expected_error}; inspect {log_path}")):
                wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)


@POSIX_PROCESS_GROUPS
def test_teardown_treats_a_denied_signal_as_a_finished_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _child_deployment(_owned_plan(tmp_path), tmp_path / "deployment.log", SHORT_CHILD_SECONDS)

    def deny_signal(process_group_id: int, signal_number: int) -> None:
        raise PermissionError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(os, "killpg", deny_signal)
    started_at = time.monotonic()
    deployment.close()

    assert time.monotonic() - started_at < PROCESS_STOP_TIMEOUT_SECONDS
    assert deployment.process.returncode is not None


@POSIX_PROCESS_GROUPS
def test_teardown_failure_keeps_the_error_that_is_already_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deployment = _child_deployment(_owned_plan(tmp_path), tmp_path / "deployment.log", SHORT_CHILD_SECONDS)

    def refuse_signal(process_group_id: int, signal_number: int) -> None:
        raise OSError(errno.EINVAL, "invalid signal")

    monkeypatch.setattr(os, "killpg", refuse_signal)
    with pytest.raises(RuntimeError, match="fixture benchmark failure"):
        with deployment:
            raise RuntimeError("fixture benchmark failure")

    assert "warning: managed server teardown failed: " in capsys.readouterr().err
    deployment.process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)


@pytest.mark.parametrize(
    ("available_memory_bytes", "fits"),
    ((MEMORY_WITHIN_SLACK_BYTES, True), (MEMORY_BELOW_SLACK_BYTES, False)),
)
def test_memory_gate_allows_only_a_small_reporting_shortfall(
    tmp_path: Path,
    available_memory_bytes: int,
    fits: bool,
) -> None:
    candidate = _candidate()
    hardware = _hardware("NVIDIA", "CUDA", available_memory_bytes)
    _cached_artifact(tmp_path, candidate)
    cached = ensure_model_artifacts(tmp_path, candidate)

    offers = framework_offers(hardware, candidate, command_finder=_installed_command)

    assert offers
    assert all(offer.memory_fit is fits for offer in offers)
    if not fits:
        with pytest.raises(ValueError, match="requires at least"):
            create_deployment_plan(
                hardware,
                candidate,
                "llama-cpp",
                cached,
                catalog_digest=BUNDLED_RECIPES_DIGEST,
                command_finder=_installed_command,
                alias_nonce="test",
            )
        return
    plan = create_deployment_plan(
        hardware,
        candidate,
        "llama-cpp",
        cached,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_installed_command,
        alias_nonce="test",
    )
    assert plan.context_tokens == PROFILE_CONTEXT_TOKENS
    assert plan.reduced_context is False


def test_reduced_context_launch_fits_small_hardware_and_lands_ctx_size(tmp_path: Path) -> None:
    candidate = _candidate()
    hardware = _hardware("NVIDIA", "CUDA", SMALL_HARDWARE_MEMORY_BYTES)
    _cached_artifact(tmp_path, candidate)
    cached = ensure_model_artifacts(tmp_path, candidate)

    full_offers = framework_offers(hardware, candidate, command_finder=_installed_command)
    reduced_offers = framework_offers(
        hardware,
        candidate,
        command_finder=_installed_command,
        context_tokens=REDUCED_CONTEXT_TOKENS,
    )

    assert all(offer.memory_fit is False for offer in full_offers)
    assert all(offer.memory_fit is True for offer in reduced_offers)
    assert all(offer.minimum_memory_bytes == REDUCED_MINIMUM_MEMORY_BYTES for offer in reduced_offers)
    with pytest.raises(ValueError, match="requires at least"):
        create_deployment_plan(
            hardware,
            candidate,
            "llama-cpp",
            cached,
            catalog_digest=BUNDLED_RECIPES_DIGEST,
            command_finder=_installed_command,
            alias_nonce="test",
        )
    plan = create_deployment_plan(
        hardware,
        candidate,
        "llama-cpp",
        cached,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_installed_command,
        alias_nonce="test",
        context_tokens=REDUCED_CONTEXT_TOKENS,
    )
    assert plan.context_tokens == REDUCED_CONTEXT_TOKENS
    assert plan.reduced_context is True
    assert plan.command[plan.command.index("--ctx-size") + 1] == str(REDUCED_CONTEXT_TOKENS)
    assert plan.to_json()["reduced_context"] is True


@pytest.mark.parametrize(
    ("context_tokens", "message"),
    (
        (PROFILE_CONTEXT_TOKENS + 1, "exceeds the profile"),
        (MINIMUM_CONTEXT_TOKENS - 1, f"at least {MINIMUM_CONTEXT_TOKENS} tokens"),
    ),
)
def test_plan_rejects_a_context_outside_the_allowed_range(
    tmp_path: Path,
    context_tokens: int,
    message: str,
) -> None:
    candidate = _candidate()
    _cached_artifact(tmp_path, candidate)
    cached = ensure_model_artifacts(tmp_path, candidate)

    with pytest.raises(ValueError, match=message):
        create_deployment_plan(
            _hardware("NVIDIA", "CUDA"),
            candidate,
            "llama-cpp",
            cached,
            catalog_digest=BUNDLED_RECIPES_DIGEST,
            command_finder=_installed_command,
            alias_nonce="test",
            context_tokens=context_tokens,
        )


def test_derived_minimum_prices_artifacts_kv_cache_and_overhead() -> None:
    deployment = _candidate().deployment

    assert derived_minimum_memory_bytes(deployment, PROFILE_CONTEXT_TOKENS) == FIXTURE_MINIMUM_MEMORY_BYTES
    # A non-positive token count would produce a negative KV term; refuse it outright.
    for degenerate_tokens in (0, -100):
        with pytest.raises(ValueError, match="must be positive"):
            derived_minimum_memory_bytes(deployment, degenerate_tokens)


def test_download_resumes_a_partial_file_left_by_an_earlier_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    incomplete_path = _incomplete_path(tmp_path)
    incomplete_path.parent.mkdir(parents=True)
    incomplete_path.write_bytes(MODEL_BYTES[:RESUMED_PREFIX_BYTES])
    reports: list[tuple[int, int]] = []

    def record(downloaded_bytes: int, total_bytes: int) -> None:
        reports.append((downloaded_bytes, total_bytes))

    with _artifact_server() as server:
        monkeypatch.setattr(
            "agentperf_local.deployment.model_cache._artifact_url", lambda candidate, deployment: server.url
        )
        artifact = ensure_model_artifacts(tmp_path, candidate, progress=record)

    assert server.requested_ranges == [f"bytes={RESUMED_PREFIX_BYTES}-"]
    assert artifact.model_path == _blob_path(tmp_path)
    assert artifact.model_path.read_bytes() == MODEL_BYTES
    assert artifact.artifacts[0].sha256 == f"sha256:{MODEL_DIGEST}"
    assert not incomplete_path.exists()
    assert reports[-1] == (len(MODEL_BYTES), len(MODEL_BYTES))
    snapshot = _snapshot_path(tmp_path, candidate)
    assert snapshot.is_symlink()
    assert snapshot.resolve() == _blob_path(tmp_path)


def test_download_retries_a_transient_failure_after_backing_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agentperf_local.deployment.model_cache.MODEL_DOWNLOAD_INITIAL_BACKOFF_SECONDS",
        RETRY_TEST_BACKOFF_SECONDS,
    )

    with _artifact_server(failures=1) as server:
        monkeypatch.setattr(
            "agentperf_local.deployment.model_cache._artifact_url", lambda candidate, deployment: server.url
        )
        started_at = time.monotonic()
        artifact = ensure_model_artifacts(tmp_path, _candidate())
        elapsed_seconds = time.monotonic() - started_at

    assert len(server.requested_ranges) == 2
    assert elapsed_seconds >= RETRY_TEST_BACKOFF_SECONDS
    assert artifact.artifacts[0].sha256 == f"sha256:{MODEL_DIGEST}"


def test_download_discards_a_partial_file_that_can_never_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()

    with _artifact_server(body=b"x" * len(MODEL_BYTES)) as server:
        monkeypatch.setattr(
            "agentperf_local.deployment.model_cache._artifact_url", lambda candidate, deployment: server.url
        )
        with pytest.raises(ValueError, match="model.gguf does not match its canonical digest"):
            ensure_model_artifacts(tmp_path, candidate)

    assert not _incomplete_path(tmp_path).exists()
    assert not _blob_path(tmp_path).exists()
    assert not _snapshot_path(tmp_path, candidate).is_symlink()


def test_download_refuses_to_start_without_room_in_the_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agentperf_local.deployment.model_cache._free_disk_bytes", lambda path: 0)

    with pytest.raises(ValueError, match=re.escape(f"more free bytes in {tmp_path}")):
        ensure_model_artifacts(tmp_path, _candidate())


def test_download_is_published_when_the_cache_filesystem_rejects_hard_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()

    def refuse_link(source: object, target: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", refuse_link)
    with _artifact_server() as server:
        monkeypatch.setattr(
            "agentperf_local.deployment.model_cache._artifact_url", lambda candidate, deployment: server.url
        )
        artifact = ensure_model_artifacts(tmp_path, candidate)

    assert artifact.model_path.read_bytes() == MODEL_BYTES
    assert artifact.artifacts[0].sha256 == f"sha256:{MODEL_DIGEST}"
    assert not _incomplete_path(tmp_path).exists()


def test_download_survives_a_filesystem_without_symlink_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed verified blob must serve runs even when the snapshot link cannot exist."""
    candidate = _candidate()

    def refuse_symlink(source: object, target: object) -> None:
        raise OSError(errno.EPERM, "symbolic links are not supported here")

    monkeypatch.setattr(os, "symlink", refuse_symlink)
    with _artifact_server() as server:
        monkeypatch.setattr(
            "agentperf_local.deployment.model_cache._artifact_url", lambda candidate, deployment: server.url
        )
        artifact = ensure_model_artifacts(tmp_path, candidate)

    assert artifact.model_path == _blob_path(tmp_path)
    assert artifact.model_path.read_bytes() == MODEL_BYTES
    assert not _snapshot_path(tmp_path, candidate).exists()
    assert scan_cache_dir(tmp_path).warnings == []
    # The next run reuses the committed blob without downloading again.
    monkeypatch.setattr("agentperf_local.deployment.model_cache._artifact_url", _refuse_download)
    assert ensure_model_artifacts(tmp_path, candidate).model_path == _blob_path(tmp_path)


def test_cache_reuse_requires_the_pinned_commit_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-named file cached under another commit must not shadow the pinned revision."""
    candidate = _candidate()
    foreign = _cached_artifact(
        tmp_path,
        replace_fields(candidate, hf_revision="b" * 40),
        content=b"bytes from an unpinned commit",
    )

    with _artifact_server() as server:
        monkeypatch.setattr(
            "agentperf_local.deployment.model_cache._artifact_url", lambda candidate, deployment: server.url
        )
        artifact = ensure_model_artifacts(tmp_path, candidate)

    assert server.requested_ranges == [None]
    assert artifact.model_path == _blob_path(tmp_path)
    assert artifact.model_path.read_bytes() == MODEL_BYTES
    assert foreign.read_bytes() == b"bytes from an unpinned commit"


def test_matching_blob_without_a_snapshot_link_is_relinked_without_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    blob = _blob_path(tmp_path)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(MODEL_BYTES)
    monkeypatch.setattr("agentperf_local.deployment.model_cache._artifact_url", _refuse_download)

    artifact = ensure_model_artifacts(tmp_path, candidate)

    assert artifact.model_path == blob
    assert artifact.artifacts[0].sha256 == f"sha256:{MODEL_DIGEST}"
    assert _snapshot_path(tmp_path, candidate).resolve() == blob


@pytest.mark.parametrize(
    ("environment", "expected"),
    (
        ({"HF_HUB_CACHE": "{tmp}/hub-cache", "HF_HOME": "{tmp}/hf-home"}, "{tmp}/hub-cache"),
        ({"HF_HOME": "{tmp}/hf-home", "XDG_CACHE_HOME": "{tmp}/xdg"}, "{tmp}/hf-home/hub"),
        ({"XDG_CACHE_HOME": "{tmp}/xdg"}, "{tmp}/xdg/huggingface/hub"),
        ({}, str(Path.home() / ".cache" / "huggingface" / "hub")),
    ),
)
def test_default_model_cache_root_honors_the_hugging_face_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected: str,
) -> None:
    for name in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value.format(tmp=tmp_path))

    assert default_model_cache_root() == Path(expected.format(tmp=tmp_path))


def _dual_accelerator_snapshot(vendor: str, api: str) -> HardwareSnapshot:
    base = _hardware(vendor, api, memory_bytes=24 * 1024**3)
    second = replace_fields(base.accelerators[0], name=f"{vendor} second test GPU")
    return replace_fields(base, accelerators=(base.accelerators[0], second))


def test_binding_without_an_index_keeps_a_single_accelerator_unpinned() -> None:
    snapshot = _hardware("NVIDIA", "CUDA")

    bound = bind_snapshot_to_device(snapshot, None)

    assert bound.snapshot is snapshot
    assert bound.device_index == 0
    assert bound.environment == ()


def test_multi_accelerator_hosts_must_name_a_device() -> None:
    snapshot = _dual_accelerator_snapshot("NVIDIA", "CUDA")

    with pytest.raises(ValueError, match=r"found 2 — choose one with --device"):
        framework_offers(snapshot, _candidate())


@pytest.mark.parametrize(
    ("vendor", "api", "device_index", "expected_environment"),
    [
        ("NVIDIA", "CUDA", 1, (("CUDA_DEVICE_ORDER", "PCI_BUS_ID"), ("CUDA_VISIBLE_DEVICES", "1"))),
        ("AMD", "ROCm", 0, (("ROCR_VISIBLE_DEVICES", "0"), ("HIP_VISIBLE_DEVICES", "0"))),
    ],
)
def test_binding_selects_and_pins_one_device_by_index(
    vendor: str, api: str, device_index: int, expected_environment: tuple[tuple[str, str], ...]
) -> None:
    snapshot = _dual_accelerator_snapshot(vendor, api)

    bound = bind_snapshot_to_device(snapshot, device_index)

    assert bound.device_index == device_index
    assert bound.snapshot.accelerators == (snapshot.accelerators[device_index],)
    assert bound.environment == expected_environment


def test_binding_rejects_an_out_of_range_device_index() -> None:
    snapshot = _dual_accelerator_snapshot("NVIDIA", "CUDA")

    with pytest.raises(ValueError, match=r"device index 5 is out of range; detected accelerators run from 0 to 1"):
        bind_snapshot_to_device(snapshot, 5)


def test_device_environment_reaches_the_owned_child_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = replace_fields(
        _owned_plan(tmp_path),
        device_environment=(("CUDA_DEVICE_ORDER", "PCI_BUS_ID"), ("CUDA_VISIBLE_DEVICES", "1")),
    )
    captured: dict[str, object] = {}

    def refuse_spawn(*args: object, **kwargs: object) -> object:
        captured["env"] = kwargs.get("env")
        raise OSError("spawn refused by test")

    monkeypatch.setattr(subprocess, "Popen", refuse_spawn)
    with pytest.raises(OSError, match="spawn refused by test"):
        start_managed_deployment(plan, tmp_path / "deployment.log")

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["PATH"] == os.environ["PATH"]
    assert not (tmp_path / "deployment.log").exists()


def test_deployment_record_json_carries_the_pinning_environment(tmp_path: Path) -> None:
    plan = replace_fields(_owned_plan(tmp_path), device_environment=(("CUDA_VISIBLE_DEVICES", "1"),))

    data = plan.to_json()

    assert data["device_environment"] == {"CUDA_VISIBLE_DEVICES": "1"}


WEIGHTS_FILES: tuple[tuple[str, bytes], ...] = (
    ("config.json", b'{"architectures": ["FixtureForCausalLM"]}'),
    ("model-00001-of-00001.safetensors", b"fixture weights payload"),
    ("tokenizer.json", b'{"version": "fixture"}'),
)
# The weights fixture has no sliding attention and no recurrent layers, so its floor is
# artifact bytes plus one full-context KV cache plus the runtime reserve.
WEIGHTS_MEMORY = DeploymentMemory(
    full_attention_layers=2,
    full_kv_heads=2,
    full_kv_head_dimension=64,
    sliding_attention_layers=0,
    sliding_kv_heads=0,
    sliding_kv_head_dimension=0,
    sliding_cached_tokens=0,
    kv_bytes_per_scalar=1,
    recurrent_state_slots=0,
    constant_state_bytes=0,
    runtime_overhead_bytes=FIXTURE_RUNTIME_OVERHEAD_BYTES,
)
WEIGHTS_ARTIFACT_BYTES = sum(len(content) for _, content in WEIGHTS_FILES)


def _weights_candidate() -> ModelCandidate:
    """Build one SGLang weights-repository recipe with the same shape as the NVFP4 profiles."""
    artifacts = tuple(
        DeploymentArtifact(
            filename=filename,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        for filename, content in WEIGHTS_FILES
    )
    deployment = ModelDeployment(
        artifact_kind="safetensors-repository",
        artifacts=artifacts,
        context_tokens=PROFILE_CONTEXT_TOKENS,
        frameworks=("sglang",),
        memory=WEIGHTS_MEMORY,
        runtime_versions=(),
        moe_runner_backend=None,
    )
    return replace_fields(
        _candidate(),
        profile_id="fixture-nvfp4",
        display_name="Fixture NVFP4",
        hf_repository="example/model-nvfp4",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        thinking_policy="enabled",
        deployment=deployment,
        devices=("nvidia-cuda",),
    )


def _cached_weights(cache_root: Path, candidate: ModelCandidate) -> Path:
    """Lay every pinned file of a weights recipe out as huggingface_hub caches it."""
    snapshot_root = cache_root / "models--example--model-nvfp4" / "snapshots" / candidate.hf_revision
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for filename, content in WEIGHTS_FILES:
        blob = cache_root / "models--example--model-nvfp4" / "blobs" / hashlib.sha256(content).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        entry = snapshot_root / filename
        entry.symlink_to(os.path.relpath(blob, snapshot_root))
    return snapshot_root


def test_cached_weights_repository_is_served_from_its_snapshot_directory(tmp_path: Path) -> None:
    """A weights recipe is verified file by file and served from its snapshot directory."""
    candidate = _weights_candidate()
    snapshot_root = _cached_weights(tmp_path, candidate)

    cached = ensure_model_artifacts(tmp_path, candidate)
    plan = create_deployment_plan(
        _hardware("NVIDIA", "CUDA"),
        candidate,
        "sglang",
        cached,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_installed_command,
        alias_nonce="test",
    )

    assert cached.model_path == snapshot_root
    assert cached.size_bytes == WEIGHTS_ARTIFACT_BYTES
    assert plan.artifact_manifest_sha256 == cached.manifest_sha256
    assert plan.model_alias == "fixture-nvfp4-test"
    assert str(snapshot_root) in plan.command


def test_a_gguf_recipe_can_not_name_a_fused_expert_kernel() -> None:
    """llama.cpp has no such flag, so naming one there is a catalog mistake."""
    catalog = load_model_catalog(BUNDLED_RECIPES_ROOT)
    gguf = next(model for model in catalog.models if model.profile_id == "gemma4-12b-it-q4-0")

    with pytest.raises(ValueError, match="only an SGLang recipe can name a fused-expert kernel"):
        replace_fields(gguf.deployment, moe_runner_backend="flashinfer_cutlass")


@pytest.mark.parametrize(
    ("token_pool", "expected_error"),
    (
        (PROFILE_CONTEXT_TOKENS, None),
        (PROFILE_CONTEXT_TOKENS // 2, "below the 65536-token context"),
    ),
)
def test_owned_sglang_deployment_requires_a_pool_that_covers_the_context(
    tmp_path: Path,
    token_pool: int,
    expected_error: str | None,
) -> None:
    """A server can advertise the full context while holding a pool too small to reach it."""
    plan = replace_fields(
        _owned_plan(tmp_path),
        framework="sglang",
        accelerator_platform="nvidia-cuda",
    )
    plan = replace_fields(
        plan,
        command=(*plan.command, "--backend", "sglang", "--token-pool", str(token_pool)),
    )
    log_path = tmp_path / "deployment.log"

    with start_managed_deployment(plan, log_path) as deployment:
        wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)
        if expected_error is None:
            verify_gpu_startup(deployment)
        else:
            with pytest.raises(RuntimeError, match=expected_error):
                verify_gpu_startup(deployment)


def test_download_asks_for_identity_encoding_so_the_bytes_are_the_pinned_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compressed response would hash as sent, so the request refuses compression.

    The Hugging Face front end compresses small text files, which every weights recipe
    pins beside its shards, and iter_raw yields whatever encoding the server chose.
    """
    candidate = _candidate()

    with _artifact_server(compress_unless_identity=True) as server:
        monkeypatch.setattr(
            "agentperf_local.deployment.model_cache._artifact_url", lambda candidate, artifact: server.url
        )
        artifact = ensure_model_artifacts(tmp_path, candidate)

    assert server.requested_encodings == ["identity"]
    assert artifact.artifacts[0].sha256 == f"sha256:{MODEL_DIGEST}"
    assert artifact.model_path.read_bytes() == MODEL_BYTES


def _sglang_reporting(tmp_path: Path, version: str) -> CommandFinder:
    """Return a finder for a stand-in SGLang that reports the given version."""
    executable = write_python_executable(tmp_path / "sglang", f"print('sglang version: {version}')\n")
    return lambda command: str(executable)


def _vllm_reporting(tmp_path: Path, version: str) -> CommandFinder:
    """Return a finder for a stand-in vLLM that reports the given version on `--version`."""
    executable = write_python_executable(tmp_path / "vllm", f"print({version!r})\n")
    return lambda command: str(executable)


def _vllm_candidate() -> ModelCandidate:
    """Return the weights candidate with the vLLM recipe enabled alongside SGLang."""
    candidate = _weights_candidate()
    deployment = replace_fields(
        candidate.deployment,
        frameworks=("sglang", "vllm"),
        runtime_versions=(("sglang", "0.5.18"), ("vllm", "0.29.0")),
    )
    return replace_fields(candidate, deployment=deployment)


def _mtp_candidate() -> ModelCandidate:
    """Return the GeForce MTP recipe, which drafts with the checkpoint's own nextn layers."""
    return replace_fields(_candidate(), speculation_policy="enabled-mtp-self-draft")


def _recurrent_weights_candidate() -> ModelCandidate:
    """Return a hybrid weights recipe, which pins its state pool so free memory cannot size it."""
    candidate = _weights_candidate()
    memory = replace_fields(WEIGHTS_MEMORY, recurrent_state_slots=10, constant_state_bytes=1024)
    deployment = replace_fields(candidate.deployment, memory=memory)
    return replace_fields(candidate, deployment=deployment)


def _fused_expert_weights_candidate() -> ModelCandidate:
    """Return a weights recipe that names the NVFP4 kernel SGLang may not pick on its own."""
    candidate = _weights_candidate()
    return replace_fields(
        candidate, deployment=replace_fields(candidate.deployment, moe_runner_backend="flashinfer_cutlass")
    )


def _tuned_vllm_candidate() -> ModelCandidate:
    """Return the vLLM recipe with its own serve arguments and environment."""
    candidate = _vllm_candidate()
    vllm = VllmLaunch(
        arguments=("--gpu-memory-utilization", "0.8", "--enable-prefix-caching"),
        environment=(("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),),
    )
    return replace_fields(candidate, deployment=replace_fields(candidate.deployment, vllm=vllm))


def _vllm_dflash_candidate() -> ModelCandidate:
    """Return the DGX Spark recipe, which drafts with a DFlash model vLLM fetches at launch."""
    return replace_fields(_vllm_candidate(), speculation_policy="enabled-dflash-draft")


# llama.cpp proves its GPU path by the offload count; the weights runtimes by the backend they log.
OFFLOAD_VERIFICATION = "startup-log-reports-full-gpu-offload"
BACKEND_VERIFICATION = "startup-log-reports-gpu-backend-and-ready-endpoint"


# A flag mapped to None must appear without a value.
@pytest.mark.parametrize(
    ("build_candidate", "framework", "expected_flags", "absent_flags", "expected_environment", "verification"),
    (
        pytest.param(
            _candidate,
            "llama-cpp",
            (("--ctx-size", str(PROFILE_CONTEXT_TOKENS)),),
            ("--spec-type",),
            (),
            OFFLOAD_VERIFICATION,
            id="llama-cpp-baseline",
        ),
        pytest.param(
            _mtp_candidate,
            "llama-cpp",
            (
                ("--spec-type", "draft-mtp"),
                ("--spec-draft-n-max", "2"),
                ("--spec-draft-backend-sampling", None),
                ("--backend-sampling", None),
            ),
            (),
            (),
            OFFLOAD_VERIFICATION,
            id="llama-cpp-mtp",
        ),
        pytest.param(
            _weights_candidate,
            "sglang",
            (
                ("--context-length", str(PROFILE_CONTEXT_TOKENS)),
                ("--max-total-tokens", str(PROFILE_CONTEXT_TOKENS)),
                ("--tool-call-parser", "qwen3_coder"),
                ("--reasoning-parser", "qwen3"),
                ("--enable-cache-report", None),
            ),
            ("--max-mamba-cache-size", "--moe-runner-backend"),
            (),
            BACKEND_VERIFICATION,
            id="sglang-dense-baseline",
        ),
        pytest.param(
            _recurrent_weights_candidate,
            "sglang",
            (("--max-mamba-cache-size", "10"),),
            (),
            (),
            BACKEND_VERIFICATION,
            id="sglang-state-pool",
        ),
        pytest.param(
            _fused_expert_weights_candidate,
            "sglang",
            (("--moe-runner-backend", "flashinfer_cutlass"),),
            (),
            (),
            BACKEND_VERIFICATION,
            id="sglang-fused-expert",
        ),
        pytest.param(
            _tuned_vllm_candidate,
            "vllm",
            (
                ("--served-model-name", "fixture-nvfp4-test"),
                ("--max-model-len", str(PROFILE_CONTEXT_TOKENS)),
                ("--tool-call-parser", "qwen3_coder"),
                ("--reasoning-parser", "qwen3"),
                ("--enable-auto-tool-choice", None),
                ("--enable-prompt-tokens-details", None),
                ("--gpu-memory-utilization", "0.8"),
                ("--enable-prefix-caching", None),
            ),
            ("--speculative-config",),
            (("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),),
            BACKEND_VERIFICATION,
            id="vllm-baseline",
        ),
        pytest.param(
            _vllm_dflash_candidate,
            "vllm",
            (
                (
                    "--speculative-config",
                    '{"method": "dflash", "model": "z-lab/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 8}',
                ),
            ),
            (),
            (),
            BACKEND_VERIFICATION,
            id="vllm-dflash",
        ),
    ),
)
def test_recipe_launch_command_carries_its_pinned_flags(
    tmp_path: Path,
    build_candidate: Callable[[], ModelCandidate],
    framework: DeploymentFramework,
    expected_flags: tuple[tuple[str, str | None], ...],
    absent_flags: tuple[str, ...],
    expected_environment: tuple[tuple[str, str], ...],
    verification: str,
) -> None:
    candidate = build_candidate()
    if framework == "llama-cpp":
        _cached_artifact(tmp_path, candidate)
    else:
        _cached_weights(tmp_path, candidate)
    finder = _vllm_reporting(tmp_path, "0.29.0") if framework == "vllm" else _installed_command
    cached = ensure_model_artifacts(tmp_path, candidate)

    plan = create_deployment_plan(
        _hardware("NVIDIA", "CUDA"),
        candidate,
        framework,
        cached,
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=finder,
        alias_nonce="test",
    )

    assert plan.framework == framework
    assert str(cached.model_path) in plan.command
    for flag, value in expected_flags:
        assert flag in plan.command
        if value is not None:
            assert plan.command[plan.command.index(flag) + 1] == value
    for flag in absent_flags:
        assert flag not in plan.command
    assert plan.device_environment == expected_environment
    assert plan.gpu_verification_policy == verification


DEVELOPMENT_BUILD = "0.1.dev20073+g8e685d198"


@pytest.mark.parametrize(
    ("pinned", "reported", "expected_error"),
    [
        ("0.29.0", "0.29.0", None),
        ("0.29.0", "0.28.0", "verified against vllm 0.29.0, but 0.28.0 is installed"),
        ("0.29.0", "0.30.0", "verified against vllm 0.29.0, but 0.30.0 is installed"),
        (DEVELOPMENT_BUILD, DEVELOPMENT_BUILD, None),
        (DEVELOPMENT_BUILD, "0.1.dev20074+g1234567", "development build 0.1.dev20073"),
        (DEVELOPMENT_BUILD, "0.30.0", "development build 0.1.dev20073"),
    ],
)
def test_vllm_launch_serves_only_the_pinned_version(
    tmp_path: Path,
    pinned: str,
    reported: str,
    expected_error: str | None,
) -> None:
    """vLLM can pass every startup check and still emit nonsense, so the version is pinned too."""
    base = _vllm_candidate()
    candidate = replace_fields(base, deployment=replace_fields(base.deployment, runtime_versions=(("vllm", pinned),)))
    _cached_weights(tmp_path, candidate)
    finder = _vllm_reporting(tmp_path, reported)

    def plan() -> DeploymentPlan:
        return create_deployment_plan(
            _hardware("NVIDIA", "CUDA"),
            candidate,
            "vllm",
            ensure_model_artifacts(tmp_path, candidate),
            catalog_digest=BUNDLED_RECIPES_DIGEST,
            command_finder=finder,
            alias_nonce="test",
        )

    if expected_error is None:
        assert plan().framework == "vllm"
    else:
        with pytest.raises(ValueError, match=expected_error):
            plan()


@pytest.mark.parametrize(
    ("kv_cache", "expected_error"),
    (
        (PROFILE_CONTEXT_TOKENS, None),
        (PROFILE_CONTEXT_TOKENS // 2, "below the 65536-token context"),
    ),
)
def test_owned_vllm_deployment_requires_a_kv_cache_that_covers_the_context(
    tmp_path: Path,
    kv_cache: int,
    expected_error: str | None,
) -> None:
    """vLLM can advertise the full context while holding a KV cache too small to reach it."""
    plan = replace_fields(_owned_plan(tmp_path), framework="vllm", accelerator_platform="nvidia-cuda")
    plan = replace_fields(plan, command=(*plan.command, "--backend", "vllm", "--token-pool", str(kv_cache)))
    log_path = tmp_path / "deployment.log"

    with start_managed_deployment(plan, log_path) as deployment:
        wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)
        if expected_error is None:
            verify_gpu_startup(deployment)
        else:
            with pytest.raises(RuntimeError, match=expected_error):
                verify_gpu_startup(deployment)


@pytest.mark.parametrize(
    ("reported", "expected_error"),
    [
        ("0.5.18", None),
        ("0.5.17", "verified against sglang 0.5.18, but 0.5.17 is installed"),
        ("0.6.0", "verified against sglang 0.5.18, but 0.6.0 is installed"),
        ("dev1+gc0b6474b4", "did not report a version it can be compared against"),
    ],
)
def test_launch_serves_only_the_runtime_version_the_recipe_names(
    tmp_path: Path,
    reported: str,
    expected_error: str | None,
) -> None:
    """A runtime can pass every startup check and still emit nonsense, so the version is checked.

    An older release is refused, and so is a newer one: the recipe names the release it was
    verified against, not a floor.
    """
    candidate = _weights_candidate()
    deployment = replace_fields(candidate.deployment, runtime_versions=(("sglang", "0.5.18"),))
    candidate = replace_fields(candidate, deployment=deployment)
    _cached_weights(tmp_path, candidate)
    finder = _sglang_reporting(tmp_path, reported)

    def plan() -> DeploymentPlan:
        return create_deployment_plan(
            _hardware("NVIDIA", "CUDA"),
            candidate,
            "sglang",
            ensure_model_artifacts(tmp_path, candidate),
            catalog_digest=BUNDLED_RECIPES_DIGEST,
            command_finder=finder,
            alias_nonce="test",
        )

    if expected_error is None:
        assert plan().framework == "sglang"
    else:
        with pytest.raises(ValueError, match=expected_error):
            plan()


def test_launch_names_a_version_check_that_timed_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow version command is not a missing version; the refusal must say which it was."""
    candidate = _weights_candidate()
    candidate = replace_fields(
        candidate, deployment=replace_fields(candidate.deployment, runtime_versions=(("sglang", "0.5.18"),))
    )
    _cached_weights(tmp_path, candidate)
    executable = write_python_executable(
        tmp_path / "sglang", "import time\ntime.sleep(2)\nprint('sglang version: 0.5.18')\n"
    )
    monkeypatch.setattr("agentperf_local.deployment.frameworks.FRAMEWORK_VERSION_TIMEOUT_SECONDS", 0.2)

    with pytest.raises(ValueError, match=r"did not report a version .* \(unreported \(version check timed out\)\)"):
        create_deployment_plan(
            _hardware("NVIDIA", "CUDA"),
            candidate,
            "sglang",
            ensure_model_artifacts(tmp_path, candidate),
            catalog_digest=BUNDLED_RECIPES_DIGEST,
            command_finder=lambda command: str(executable),
            alias_nonce="test",
        )


SPLASH_FILES: tuple[tuple[str, bytes], ...] = (
    ("draft/model.bin", b"fixture draft"),
    ("manifest.json", b'{"schema_version": 3}'),
    ("target/layer-0.bin", b"fixture target layer"),
    ("tokenizer/tokenizer.json", b'{"version": "fixture"}'),
)
SPLASH_REPOSITORY = "example/model-splash"
SPLASH_VERSION = "1.0.2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# A stand-in for the server script inside a Splash install. It reads the flags the
# launcher passes and serves them through the shared fixture server.
SPLASH_SERVER_SCRIPT = """\
import argparse
import sys

sys.path.insert(0, {root!r})
from tests.managed_server import main

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
for positional in ("target", "draft"):
    parser.add_argument(positional)
parser.add_argument("--port", required=True)
parser.add_argument("--served-model-name", required=True)
parser.add_argument("--max-context", required=True)
known, _ = parser.parse_known_args()
arguments = ["--port", known.port, "--alias", known.served_model_name, "--ctx-size", known.max_context]
raise SystemExit(main([*arguments, "--backend", "splash"]))
"""
# The packaged interpreter stand-in hands its arguments to this test's Python.
SPLASH_PYTHON_SCRIPT = "import os\nimport sys\n\nos.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"


def _splash_candidate() -> ModelCandidate:
    """Build one Splash package recipe with the directory layout of the bundled Qwen3.8 package."""
    artifacts = tuple(
        DeploymentArtifact(filename=filename, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
        for filename, content in SPLASH_FILES
    )
    deployment = ModelDeployment(
        artifact_kind="splash-package",
        artifacts=artifacts,
        context_tokens=PROFILE_CONTEXT_TOKENS,
        frameworks=("splash",),
        memory=WEIGHTS_MEMORY,
        runtime_versions=(("splash", SPLASH_VERSION),),
    )
    return replace_fields(
        _candidate(),
        profile_id="fixture-splash",
        display_name="Fixture Splash",
        hf_repository=SPLASH_REPOSITORY,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        thinking_policy="enabled",
        speculation_policy="enabled-dflash-draft",
        deployment=deployment,
        devices=("apple-silicon",),
    )


def _cached_splash_package(cache_root: Path, candidate: ModelCandidate) -> Path:
    """Lay every pinned file of a Splash package out as huggingface_hub caches it."""
    repository_root = cache_root / repo_folder_name(repo_id=SPLASH_REPOSITORY, repo_type="model")
    snapshot_root = repository_root / "snapshots" / candidate.hf_revision
    for filename, content in SPLASH_FILES:
        blob = repository_root / "blobs" / hashlib.sha256(content).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        entry = snapshot_root / filename
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.symlink_to(os.path.relpath(blob, entry.parent))
    return snapshot_root


def _splash_install(tmp_path: Path) -> CommandFinder:
    """Lay out a packaged Splash install whose server script is the fixture server."""
    cellar = tmp_path / "Cellar" / "splash"
    libexec = cellar / "libexec"
    launcher = write_python_executable(cellar / "bin" / "splash", f"print('Splash {SPLASH_VERSION}')\n")
    write_python_executable(libexec / "python" / "bin" / "python3", SPLASH_PYTHON_SCRIPT)
    (libexec / "release.json").write_text(f'{{"version": "{SPLASH_VERSION}"}}')
    (libexec / "server").mkdir()
    (libexec / "server" / "server.py").write_text(SPLASH_SERVER_SCRIPT.format(root=str(REPOSITORY_ROOT)))
    (libexec / "engine").mkdir()
    (libexec / "engine" / "splash").write_bytes(b"fixture engine")
    linked = tmp_path / "bin" / "splash"
    linked.parent.mkdir()
    linked.symlink_to(launcher)
    return lambda command: str(linked) if command == "splash" else None


@pytest.mark.skipif(sys.platform == "win32", reason="Splash runs on macOS only; the fixture install uses shebangs")
@pytest.mark.parametrize(
    ("served_context", "expected_error"),
    (
        (PROFILE_CONTEXT_TOKENS, None),
        (SERVED_CONTEXT_TOKENS_MISMATCH, "reports a 8192-token context but the profile requires 65536"),
    ),
)
def test_splash_serves_the_verified_snapshot_and_proves_its_context(
    tmp_path: Path, served_context: int, expected_error: str | None
) -> None:
    """Splash starts on the pinned snapshot, not its own download, and reports its context on /status."""
    candidate = _splash_candidate()
    snapshot_root = _cached_splash_package(tmp_path, candidate)

    plan = create_deployment_plan(
        _hardware("Apple", "Metal", memory_bytes=64 * 1024**3),
        candidate,
        "splash",
        ensure_model_artifacts(tmp_path, candidate),
        catalog_digest=BUNDLED_RECIPES_DIGEST,
        command_finder=_splash_install(tmp_path),
        alias_nonce="test",
        port=_free_port(),
    )

    engine = tmp_path / "Cellar" / "splash" / "libexec" / "engine" / "splash"
    assert plan.command[plan.command.index("--binary") + 1] == str(engine)
    assert plan.runtime.executable_sha256 == f"sha256:{hashlib.sha256(b'fixture engine').hexdigest()}"
    assert plan.runtime.version == f"Splash {SPLASH_VERSION}"
    assert str(snapshot_root / "target") in plan.command
    assert str(snapshot_root / "draft") in plan.command
    assert plan.command[plan.command.index("--tokenizer") + 1] == str(snapshot_root / "tokenizer")
    assert plan.command[plan.command.index("--model") + 1] == SPLASH_REPOSITORY
    assert plan.command[plan.command.index("--served-model-name") + 1] == "fixture-splash-test"
    assert plan.command[plan.command.index("--max-context") + 1] == str(PROFILE_CONTEXT_TOKENS)
    assert plan.server_launch_command.startswith(
        '"$SPLASH_HOME"/python/bin/python3 -u "$SPLASH_HOME"/server/server.py --binary "$SPLASH_HOME"/engine/splash '
        '"$MODEL_DIR"/target "$MODEL_DIR"/draft --tokenizer "$MODEL_DIR"/tokenizer'
    )
    assert plan.accelerator_backend == "metal"
    plan = replace_fields(
        plan,
        command=tuple(
            str(served_context) if previous == "--max-context" else argument
            for previous, argument in zip(("", *plan.command), plan.command, strict=False)
        ),
    )

    with start_managed_deployment(plan, tmp_path / "deployment.log") as deployment:
        if expected_error is None:
            served = wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)
            assert served == served_context
            verify_gpu_startup(deployment)
            # An attached run against the same server reads the context the same way.
            assert probe_served_context_tokens(plan.base_url, plan.model_alias).observed_tokens == served_context
        else:
            with pytest.raises(RuntimeError, match=expected_error):
                wait_for_deployment(deployment, timeout_seconds=MANAGED_TEST_STARTUP_TIMEOUT_SECONDS)
