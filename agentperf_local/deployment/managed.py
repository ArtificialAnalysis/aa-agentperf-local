"""Plan and own exact local model deployments."""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal

import httpx
import orjson
from pydantic import BaseModel

from agentperf_local.common.durable_files import (
    NEW_FILE_OPEN_FLAGS,
    PRIVATE_FILE_PERMISSIONS,
    WrittenFile,
    validate_new_file_paths,
    write_digest_file,
)
from agentperf_local.common.identity import mint_run_id, sha256_bytes, validate_digest
from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object
from agentperf_local.common.models import read_record
from agentperf_local.deployment.catalog import (
    DeploymentFramework,
    LlamaCppLaunch,
    ModelCandidate,
    ModelDeployment,
)
from agentperf_local.deployment.context_policy import (
    derived_minimum_memory_bytes,
    memory_fits,
    resolve_context_tokens,
)
from agentperf_local.deployment.endpoint_probes import model_entry, reported_context_tokens
from agentperf_local.deployment.frameworks import (
    CommandFinder,
    FrameworkExecutable,
    FrameworkIdentity,
    available_accelerator_memory,
    find_executable,
    framework_identity,
    framework_supported,
    installation_hint,
    platform_device_id,
    require_supported_runtime,
    resolve_framework_executable,
)
from agentperf_local.deployment.launch_command import LocalPath, render_launch_command
from agentperf_local.deployment.model_cache import VerifiedDeployment
from agentperf_local.provenance.benchmark import SubmissionContext
from agentperf_local.provenance.context import below_benchmark_context
from agentperf_local.provenance.hardware import (
    AcceleratorPlatform,
    HardwareSnapshot,
    accelerator_platform,
    select_accelerator,
)
from agentperf_local.submission.contract import AcceleratorBackend

# Version 2 added the deployment identifier, creation time, and launch configuration
# digest. Version 3 replaced the single artifact_sha256 field with the artifact-manifest
# digest, because a managed recipe can now pin a whole weights repository. Version 4
# added the recipe identity and the digest of the catalog it came from, which is what
# anchors a submission to an allowlisted release.
# Version 5 added the model release, the accelerator backend, the redacted launch command,
# and the recipe text, which a submission sends.
type DeploymentRecordVersion = Literal[5]
DEPLOYMENT_RECORD_VERSION: DeploymentRecordVersion = 5
type DeploymentRecordKind = Literal["managed_model_deployment"]
type DeploymentRecordStatus = Literal["ready-at-benchmark-start"]
DEPLOYMENT_RECORD_KIND: DeploymentRecordKind = "managed_model_deployment"
DEPLOYMENT_RECORD_STATUS: DeploymentRecordStatus = "ready-at-benchmark-start"
MAX_DEPLOYMENT_RECORD_BYTES = 1024 * 1024
DEPLOYMENT_RECORD_FILENAME = "deployment.json"
DEPLOYMENT_LOG_FILENAME = "deployment.log"
DEFAULT_DEPLOYMENT_PORT = 8080
DEFAULT_STARTUP_TIMEOUT_SECONDS = 1_800.0
LOOPBACK_HOST = "127.0.0.1"
MINIMUM_USER_PORT = 1_024
MAXIMUM_PORT = 65_535
LOCAL_READINESS_TIMEOUT_SECONDS = 2.0
READINESS_POLL_SECONDS = 0.25
PROCESS_STOP_TIMEOUT_SECONDS = 10.0
PROCESS_STOP_POLL_SECONDS = 0.05
PROCESS_GROUP_PROBE_SIGNAL = 0
# /T ends every process the server started, and /F does not wait for them to agree.
WINDOWS_TREE_KILL_COMMAND = ("taskkill", "/T", "/F", "/PID")
MAX_GPU_VERIFICATION_LOG_BYTES = 8 * 1024 * 1024
DEPLOYMENT_ALIAS_NONCE_BYTES = 8
FULL_OFFLOAD_PATTERN = re.compile(r"offloaded\s+([0-9]+)/([0-9]+)\s+layers\s+to\s+gpu", re.IGNORECASE)
TOKEN_POOL_PATTERN = re.compile(r"max_total_num_tokens=([0-9]+)")
# vLLM logs the KV cache it settled on as e.g. "GPU KV cache size: 1,177,344 tokens".
VLLM_KV_CACHE_PATTERN = re.compile(r"kv cache size[:\s]+([0-9,]+)\s*tokens")
# Splash's engine runs only on Metal and names the Apple GPU family it chose kernels
# for, e.g. "Kernel policy for GPU family 10 with 20 cores".
SPLASH_METAL_MARKER = "kernel policy for gpu family"

# What most often ends a managed server before it ever answers, named per runtime so a
# failure points at the right thing to check.
_EARLY_EXIT_HINTS: dict[DeploymentFramework, str] = {
    "llama-cpp": "an older llama.cpp build may not support the launch flags; upgrade to a current build",
    "sglang": (
        "a server killed with status -9 was killed by the operating system for using too much memory, "
        "which on a device that shares host memory can mean another process holds it"
    ),
    "vllm": ("a server killed with status -9 ran out of memory; lower --gpu-memory-utilization or free the device"),
    "splash": (
        "Splash exits when its engine cannot hold the requested context in memory; "
        "close other applications that hold unified memory"
    ),
}


def _log_hint(message: str, log_path: Path) -> str:
    """Point a managed-deployment failure at the log that explains it."""
    return f"{message}; inspect {log_path}"


def strip_log_hint(message: str, log_path: Path) -> str:
    """Remove the pointer _log_hint appended, for a surface that names the log itself."""
    return message.replace(_log_hint("", log_path), "")


class DeploymentPlan(BaseModel, frozen=True):
    """Bind an exact model file, runtime command, and local endpoint."""

    profile_id: str
    model_release_slug: str
    hf_repository: str
    hf_revision: str
    # The catalog file these three came from. A submission is anchored on it: the
    # service knows the catalog shipped at an allowlisted commit and can check that
    # this recipe is one of its entries, so no separate signature is needed.
    catalog_digest: str
    framework: DeploymentFramework
    accelerator_platform: AcceleratorPlatform
    # The compute backend the framework serves on, as a submission names it.
    accelerator_backend: AcceleratorBackend
    model_path: Path
    artifact_manifest_sha256: str
    artifact_size_bytes: int
    context_tokens: int
    reduced_context: bool = False
    model_alias: str
    host: str
    port: int
    command: tuple[str, ...]
    # The command and its environment as one shell line, with local paths and secrets replaced.
    server_launch_command: str
    runtime: FrameworkIdentity
    device_environment: tuple[tuple[str, str], ...] = ()

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible API base."""
        return f"http://{self.host}:{self.port}/v1"

    @property
    def runtime_id(self) -> str:
        """Return a short identifier bound to the runtime fingerprint."""
        return f"{self.framework}-managed-{self.runtime.fingerprint.removeprefix('sha256:')[:16]}"

    @property
    def gpu_verification_policy(self) -> str:
        """Return how this recipe proves that the selected GPU path started."""
        if self.framework == "llama-cpp":
            return "startup-log-reports-full-gpu-offload"
        return "startup-log-reports-gpu-backend-and-ready-endpoint"

    def to_json(self) -> JsonObject:
        """Return private deployment facts for the benchmark result directory."""
        command: list[JsonValue] = []
        command.extend(self.command)
        return {
            "profile_id": self.profile_id,
            "model_release_slug": self.model_release_slug,
            "hf_repository": self.hf_repository,
            "hf_revision": self.hf_revision,
            "catalog_digest": self.catalog_digest,
            "framework": self.framework,
            "accelerator_platform": self.accelerator_platform,
            "accelerator_backend": self.accelerator_backend,
            "model_path": str(self.model_path),
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "context_tokens": self.context_tokens,
            "reduced_context": self.reduced_context,
            "model_alias": self.model_alias,
            "base_url": self.base_url,
            "gpu_execution_requested": True,
            "gpu_verification_policy": self.gpu_verification_policy,
            "command": command,
            "server_launch_command": self.server_launch_command,
            "device_environment": dict(self.device_environment),
            "runtime": {
                "version": self.runtime.version,
                "executable_sha256": self.runtime.executable_sha256,
                "fingerprint": self.runtime.fingerprint,
                "runtime_id": self.runtime_id,
            },
        }


@dataclass(slots=True, kw_only=True)
class ManagedDeployment:
    """Own one child server and its private log stream."""

    plan: DeploymentPlan
    process: subprocess.Popen[bytes]
    process_group_id: int
    log_stream: IO[bytes]
    log_path: Path

    def close(self) -> None:
        """Stop the complete process group created for this deployment."""
        try:
            _stop_process_group(self.process, self.process_group_id)
        finally:
            self.log_stream.close()

    def __enter__(self) -> ManagedDeployment:
        """Return the running deployment."""
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        """Stop the owned process without hiding the diagnostic that is already propagating."""
        try:
            self.close()
        except Exception as error:
            if exception is None:
                raise
            print(f"warning: managed server teardown failed: {error}", file=sys.stderr)


class BoundDeploymentDevice(BaseModel, frozen=True):
    """Bind one detected accelerator plus the environment that pins a child server to it."""

    snapshot: HardwareSnapshot
    device_index: int
    environment: tuple[tuple[str, str], ...]


def _device_pinning_environment(platform: AcceleratorPlatform, device_index: int) -> tuple[tuple[str, str], ...]:
    # CUDA_DEVICE_ORDER aligns CUDA enumeration with the PCI order nvidia-smi reports,
    # so the pinned index names the same card the snapshot describes.
    if platform == "nvidia-cuda":
        return (("CUDA_DEVICE_ORDER", "PCI_BUS_ID"), ("CUDA_VISIBLE_DEVICES", str(device_index)))
    if platform == "amd-rocm":
        return (("ROCR_VISIBLE_DEVICES", str(device_index)), ("HIP_VISIBLE_DEVICES", str(device_index)))
    return ()


def bind_snapshot_to_device(snapshot: HardwareSnapshot, device_index: int | None) -> BoundDeploymentDevice:
    """Reduce a snapshot to one accelerator and derive the pinning environment for it.

    Without an explicit index a single-accelerator machine binds unpinned, keeping the
    launch environment unchanged; a multi-accelerator machine must name a device.
    """
    if device_index is None:
        return BoundDeploymentDevice(snapshot=snapshot, device_index=0, environment=())
    bound = select_accelerator(snapshot, device_index)
    platform = accelerator_platform(bound)
    return BoundDeploymentDevice(
        snapshot=bound,
        device_index=device_index,
        environment=_device_pinning_environment(platform, device_index),
    )


def _signal_process_group(process_group_id: int, signal_number: int) -> bool:
    """Signal the group and report whether it still belongs to this process."""
    assert sys.platform != "win32", "Windows stops the tree in _stop_windows_process_group"
    try:
        os.killpg(process_group_id, signal_number)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _leader_exited(process: subprocess.Popen[bytes]) -> bool:
    """Report whether the group leader has exited, without reaping it.

    Darwin refuses to signal a group whose only member is the unreaped leader, so the
    caller already sees that group as stopped and no peek is needed here.
    """
    if process.returncode is not None:
        return True
    assert sys.platform != "win32", "only the POSIX process-group stop peeks at the leader"
    if sys.platform == "darwin":
        return False
    try:
        report = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return True
    return report is not None


def _process_group_stopped(process: subprocess.Popen[bytes], process_group_id: int) -> bool:
    """Report whether the group holds nothing this process still has to stop.

    An unreaped leader keeps its group signalable on Linux, so the leader is
    probed separately instead of being reaped by the wait loop.
    """
    if not _signal_process_group(process_group_id, PROCESS_GROUP_PROBE_SIGNAL):
        return True
    return _leader_exited(process)


def _wait_for_process_group_exit(process: subprocess.Popen[bytes], process_group_id: int, deadline: float) -> bool:
    while True:
        if _process_group_stopped(process, process_group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_STOP_POLL_SECONDS)


def _stop_windows_process_group(process: subprocess.Popen[bytes], break_event: int) -> None:
    """Ask the owned Windows process group to exit, then end whatever is left of its tree.

    The break event reaches every process in the group that shares this console. It
    fails when this process has no console, and taskkill then does all the work.
    """
    if process.poll() is None:
        with contextlib.suppress(OSError):
            process.send_signal(break_event)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
    if process.poll() is None:
        subprocess.run(
            (*WINDOWS_TREE_KILL_COMMAND, str(process.pid)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=PROCESS_STOP_TIMEOUT_SECONDS,
            check=False,
        )
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("managed deployment process group did not stop") from error


def _stop_process_group(process: subprocess.Popen[bytes], process_group_id: int) -> None:
    """Stop the owned group and reap the leader only after the last signal it may need.

    The unreaped leader holds the group identifier, so no signal here can reach a
    recycled group.
    """
    if sys.platform == "win32":
        _stop_windows_process_group(process, signal.CTRL_BREAK_EVENT)
        return
    if _signal_process_group(process_group_id, signal.SIGTERM):
        deadline = time.monotonic() + PROCESS_STOP_TIMEOUT_SECONDS
        if not _wait_for_process_group_exit(process, process_group_id, deadline):
            _signal_process_group(process_group_id, signal.SIGKILL)
            kill_deadline = time.monotonic() + PROCESS_STOP_TIMEOUT_SECONDS
            if not _wait_for_process_group_exit(process, process_group_id, kill_deadline):
                raise RuntimeError("managed deployment process group did not stop")
    process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)


def _llama_cpp_argv(
    candidate: ModelCandidate,
    deployment: ModelDeployment,
    context_tokens: int,
    model_path: Path,
    draft_model_path: Path | None,
    model_alias: str,
    endpoint: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the llama.cpp server flags for one recipe, after the executable prefix."""
    thinking_enabled = candidate.thinking_policy != "disabled"
    llama_reasoning = () if thinking_enabled else ("--reasoning", "off")
    launch = deployment.llama_cpp
    tuning: list[str] = []
    if launch is not None:
        if launch.backend_device is not None:
            tuning.extend(("--device", launch.backend_device))
        if launch.load_mode is not None:
            tuning.extend(("--load-mode", launch.load_mode))
        if launch.lazy_mode is not None:
            tuning.extend(("--lazy-mode", launch.lazy_mode))
        if launch.threads is not None:
            tuning.extend(("--threads", str(launch.threads)))
        if launch.flash_attention:
            tuning.extend(("--flash-attn", "on"))
        if launch.disable_fit:
            tuning.extend(("--fit", "off"))
        if launch.cache_ram_mib is not None:
            tuning.extend(("--cache-ram", str(launch.cache_ram_mib)))
        if launch.context_checkpoints is not None:
            tuning.extend(("--ctx-checkpoints", str(launch.context_checkpoints)))
    batching = (
        ()
        if launch is None
        else ("--batch-size", str(launch.batch_size), "--ubatch-size", str(launch.ubatch_size), "--no-mmproj")
    )
    if launch is None:
        speculation = (
            (
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "2",
                "--spec-draft-backend-sampling",
                "--backend-sampling",
            )
            if candidate.speculation_policy == "enabled-mtp-self-draft"
            else ()
        )
    elif candidate.speculation_policy == "disabled-target-only-baseline":
        speculation = ()
    else:
        if launch.draft_model_filename is not None and draft_model_path is None:
            raise ValueError("llama.cpp external-draft recipe is missing its verified draft model path")
        if launch.speculative_tokens is None:
            raise ValueError("speculative llama.cpp recipe is missing its speculative_tokens")
        if candidate.speculation_policy in ("enabled-dflash-external-draft", "enabled-dflash-draft"):
            spec_type = "draft-dflash"
        elif candidate.speculation_policy == "enabled-dspark-external-draft":
            spec_type = "draft-dspark"
        else:
            spec_type = "draft-mtp"
        draft_model = (
            (
                "--model-draft",
                str(draft_model_path),
                *(("--gpu-layers-draft", "all") if launch.backend is not None else ()),
            )
            if launch.draft_model_filename is not None and draft_model_path is not None
            else ()
        )
        target_sampling = ("--backend-sampling",) if launch.target_backend_sampling else ()
        draft_sampling = ("--spec-draft-backend-sampling",) if launch.draft_backend_sampling else ()
        speculation = (
            *draft_model,
            "--spec-type",
            spec_type,
            "--spec-draft-n-max",
            str(launch.speculative_tokens),
            *target_sampling,
            *draft_sampling,
        )
    return (
        "--model",
        str(model_path),
        "--alias",
        model_alias,
        *endpoint,
        "--ctx-size",
        str(context_tokens),
        "--parallel",
        "1",
        "--gpu-layers",
        "all",
        "--jinja",
        *batching,
        *tuning,
        *speculation,
        *llama_reasoning,
        "--no-webui",
        "--verbosity",
        "4",
    )


def _sglang_argv(
    candidate: ModelCandidate,
    deployment: ModelDeployment,
    context_tokens: int,
    model_path: Path,
    model_alias: str,
    endpoint: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the SGLang server flags for one recipe, after the executable prefix."""
    # SGLang sizes its caches from whatever memory it finds, which would make one
    # recipe behave differently on every card. The token pool and, for a model with
    # recurrent layers, the state pool are pinned to the recipe instead, so the
    # served configuration is the same everywhere and the memory floor is exact.
    recurrent_slots = deployment.memory.recurrent_state_slots
    state_pool = () if recurrent_slots == 0 else ("--max-mamba-cache-size", str(recurrent_slots))
    # SGLang also picks a fused-expert kernel from the device, and its pick is not
    # always implemented for the recipe's quantization: on an SM121 part it selects
    # the TensorRT-LLM kernel for an NVFP4 mixture of experts and then refuses the
    # first forward pass. A recipe that needs a particular kernel names it.
    moe_backend = (
        () if deployment.moe_runner_backend is None else ("--moe-runner-backend", deployment.moe_runner_backend)
    )
    # The replay is sequential, so the server holds one running request and one
    # tensor-parallel rank on the bound device. Speculative decoding stays off
    # because the catalog pins a target-only baseline.
    return (
        "--model-path",
        str(model_path),
        "--served-model-name",
        model_alias,
        *endpoint,
        "--context-length",
        str(context_tokens),
        "--tp-size",
        "1",
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        str(context_tokens),
        *state_pool,
        *moe_backend,
        # A managed submission needs the cached prompt tokens of every turn, and SGLang
        # leaves them out of usage unless asked.
        "--enable-cache-report",
        "--tool-call-parser",
        candidate.tool_call_parser,
        "--reasoning-parser",
        candidate.reasoning_parser,
        "--log-level",
        "info",
    )


def _vllm_argv(
    candidate: ModelCandidate,
    deployment: ModelDeployment,
    context_tokens: int,
    model_path: Path,
    model_alias: str,
    endpoint: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the vLLM server flags for one recipe, after the executable prefix."""
    # The replay is sequential: one running request and one tensor-parallel rank on the
    # bound device. The context is pinned to the recipe so the served window is identical
    # on every card, and the KV cache is fp8 to match the recipe's one-byte-per-scalar
    # memory model.
    # DFlash speculative decoding (the DGX Spark recipe): a small draft model proposes
    # tokens the target verifies. vLLM fetches the draft model named here at launch; it is
    # not one of the recipe's sha256-pinned artifacts. Otherwise speculation stays off.
    fallback_speculation = (
        (
            "--speculative-config",
            '{"method": "dflash", "model": "z-lab/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 8}',
        )
        if candidate.speculation_policy == "enabled-dflash-draft"
        else ()
    )
    recipe_arguments = deployment.vllm.arguments if deployment.vllm is not None else ()
    fallback_arguments = (
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--gpu-memory-utilization",
        "0.9",
        *fallback_speculation,
    )
    runtime_arguments = recipe_arguments if deployment.vllm is not None else fallback_arguments
    return (
        str(model_path),
        "--served-model-name",
        model_alias,
        *endpoint,
        "--max-model-len",
        str(context_tokens),
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "1",
        # A managed submission needs the cached prompt tokens of every turn, and vLLM
        # leaves them out of usage unless asked.
        "--enable-prompt-tokens-details",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        candidate.tool_call_parser,
        "--reasoning-parser",
        candidate.reasoning_parser,
        "--seed",
        "0",
        *runtime_arguments,
    )


def _splash_argv(
    candidate: ModelCandidate,
    context_tokens: int,
    model_path: Path,
    model_alias: str,
    endpoint: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the Splash server flags for one recipe, after the executable prefix.

    The package directories come from the verified snapshot, never from Splash's own
    model directory, so the served weights are the recipe's pinned revision. Splash
    exits unless its engine grants exactly the requested context, and it parses tool
    calls and reasoning itself, so no parser is named. The engine sizes its memory
    budget from the context.
    """
    return (
        str(model_path / "target"),
        str(model_path / "draft"),
        "--tokenizer",
        str(model_path / "tokenizer"),
        "--model",
        candidate.hf_repository,
        "--served-model-name",
        model_alias,
        *endpoint,
        "--max-context",
        str(context_tokens),
        "--no-webui",
    )


def _launch_command(
    executable: FrameworkExecutable,
    candidate: ModelCandidate,
    deployment: ModelDeployment,
    context_tokens: int,
    model_path: Path,
    draft_model_path: Path | None,
    model_alias: str,
    host: str,
    port: int,
) -> tuple[str, ...]:
    """Return the full argv for one framework: its executable prefix, then its server flags."""
    endpoint = ("--host", host, "--port", str(port))
    if executable.framework == "llama-cpp":
        arguments = _llama_cpp_argv(
            candidate, deployment, context_tokens, model_path, draft_model_path, model_alias, endpoint
        )
    elif executable.framework == "sglang":
        arguments = _sglang_argv(candidate, deployment, context_tokens, model_path, model_alias, endpoint)
    elif executable.framework == "vllm":
        arguments = _vllm_argv(candidate, deployment, context_tokens, model_path, model_alias, endpoint)
    else:
        arguments = _splash_argv(candidate, context_tokens, model_path, model_alias, endpoint)
    return (*executable.command_prefix, *arguments)


# The launch command names the recipe's files under this placeholder, by their repository paths.
MODEL_DIRECTORY_PLACEHOLDER = "$MODEL_DIR"
PYTHON_PLACEHOLDER = "$PYTHON"


def _accelerator_backend(platform: AcceleratorPlatform, launch: LlamaCppLaunch | None) -> AcceleratorBackend:
    """Name the compute backend one managed launch serves on."""
    if platform == "nvidia-cuda":
        return "cuda"
    if platform == "apple-metal":
        return "metal"
    return "vulkan" if launch is not None and launch.backend == "vulkan" else "rocm"


def _known_launch_paths(
    executable: FrameworkExecutable, deployment: ModelDeployment, artifacts: VerifiedDeployment
) -> tuple[LocalPath, ...]:
    """Name each local path a managed launch command holds, with the placeholder that replaces it."""
    target = deployment.target_model_filename
    paths = [
        *(
            LocalPath(path=str(artifact.path), placeholder=f"{MODEL_DIRECTORY_PLACEHOLDER}/{artifact.filename}")
            for artifact in artifacts.artifacts
        ),
        LocalPath(
            path=str(artifacts.model_path),
            placeholder=MODEL_DIRECTORY_PLACEHOLDER if target is None else f"{MODEL_DIRECTORY_PLACEHOLDER}/{target}",
        ),
        LocalPath(path=sys.executable, placeholder=PYTHON_PLACEHOLDER),
    ]
    if executable.install_root is not None:
        home = executable.framework.upper().replace("-", "_")
        paths.append(LocalPath(path=str(executable.install_root), placeholder=f"${home}_HOME"))
    return tuple(paths)


def _require_unchanged_artifacts(artifacts: VerifiedDeployment, deployment: ModelDeployment) -> None:
    """Reject a verified artifact set that no longer matches the catalog or the disk.

    Verification and launch are separate steps, so every pinned file is re-checked
    here by identity: a file swapped between the two would otherwise be served.
    """
    pinned = {artifact.filename: artifact for artifact in deployment.artifacts}
    verified = {artifact.filename: artifact for artifact in artifacts.artifacts}
    if pinned.keys() != verified.keys():
        raise ValueError("verified model artifacts do not match the selected catalog deployment")
    for filename, expected in pinned.items():
        artifact = verified[filename]
        if artifact.sha256 != f"sha256:{expected.sha256}" or artifact.size_bytes != expected.size_bytes:
            raise ValueError("verified model artifacts do not match the selected catalog deployment")
        if artifact.path.is_symlink() or not artifact.path.is_file():
            raise ValueError(f"verified model artifact {filename} is no longer a regular file")
        metadata = artifact.path.stat()
        if (
            metadata.st_dev != artifact.device_id
            or metadata.st_ino != artifact.inode
            or metadata.st_size != artifact.size_bytes
            or metadata.st_mtime_ns != artifact.modified_ns
        ):
            raise ValueError(f"verified model artifact {filename} changed before deployment planning")


def create_deployment_plan(
    snapshot: HardwareSnapshot,
    candidate: ModelCandidate,
    framework: DeploymentFramework,
    artifacts: VerifiedDeployment,
    *,
    catalog_digest: str,
    port: int = DEFAULT_DEPLOYMENT_PORT,
    command_finder: CommandFinder = find_executable,
    alias_nonce: str | None = None,
    device_environment: tuple[tuple[str, str], ...] = (),
    context_tokens: int | None = None,
) -> DeploymentPlan:
    """Create one launch plan after compatibility and artifact checks.

    A context below the recipe's benchmark context launches an exploratory
    reduced-context deployment with a proportionally smaller memory floor.
    """
    validate_digest(catalog_digest, "catalog_digest")
    if port < MINIMUM_USER_PORT or port > MAXIMUM_PORT:
        raise ValueError(f"deployment port must be between {MINIMUM_USER_PORT} and {MAXIMUM_PORT}")
    deployment = candidate.deployment
    if framework not in deployment.frameworks:
        raise ValueError("selected model does not support the requested managed framework")
    resolved_context_tokens = resolve_context_tokens(deployment, context_tokens)
    minimum_memory_bytes = derived_minimum_memory_bytes(deployment, resolved_context_tokens)
    platform = accelerator_platform(snapshot)
    launch = deployment.llama_cpp
    if launch is not None and launch.backend == "vulkan" and device_environment:
        # ROCm's visible-device variables do not bind Vulkan enumeration. Keep this
        # single-iGPU recipe from silently benchmarking a different selected device.
        raise ValueError("the Vulkan recipe requires an unpinned single-accelerator host")
    candidate_devices = frozenset(candidate.devices)
    if platform_device_id(platform) not in candidate_devices or not framework_supported(framework, platform):
        raise ValueError(f"{framework} is not supported on {platform}")
    available_memory_bytes = available_accelerator_memory(snapshot, platform)
    if available_memory_bytes is None:
        raise ValueError("managed deployment requires a known accelerator memory capacity")
    if not memory_fits(available_memory_bytes, minimum_memory_bytes):
        raise ValueError(f"managed deployment requires at least {minimum_memory_bytes} bytes of accelerator memory")
    _require_unchanged_artifacts(artifacts, deployment)
    executable = resolve_framework_executable(framework, command_finder=command_finder)
    if executable is None:
        raise ValueError(installation_hint(framework))
    runtime = framework_identity(executable)
    require_supported_runtime(deployment, runtime, framework)
    nonce = alias_nonce if alias_nonce is not None else secrets.token_hex(DEPLOYMENT_ALIAS_NONCE_BYTES)
    if not nonce or not nonce.isascii() or not nonce.isalnum():
        raise ValueError("managed deployment alias nonce must be non-empty ASCII letters and digits")
    model_alias = f"{candidate.profile_id}-{nonce}"
    recipe_environment = deployment.vllm.environment if framework == "vllm" and deployment.vllm is not None else ()
    combined_environment = (*recipe_environment, *device_environment)
    environment_names = tuple(name for name, _ in combined_environment)
    if len(set(environment_names)) != len(environment_names):
        raise ValueError("managed deployment environment names must be unique")
    command = _launch_command(
        executable,
        candidate,
        deployment,
        resolved_context_tokens,
        artifacts.model_path,
        artifacts.draft_model_path,
        model_alias,
        LOOPBACK_HOST,
        port,
    )
    return DeploymentPlan(
        profile_id=candidate.profile_id,
        model_release_slug=candidate.model_release_slug,
        hf_repository=candidate.hf_repository,
        hf_revision=candidate.hf_revision,
        catalog_digest=catalog_digest,
        framework=framework,
        accelerator_platform=platform,
        accelerator_backend=_accelerator_backend(platform, launch),
        model_path=artifacts.model_path,
        artifact_manifest_sha256=artifacts.manifest_sha256,
        artifact_size_bytes=artifacts.size_bytes,
        context_tokens=resolved_context_tokens,
        reduced_context=below_benchmark_context(resolved_context_tokens),
        model_alias=model_alias,
        host=LOOPBACK_HOST,
        port=port,
        command=command,
        server_launch_command=render_launch_command(
            command, combined_environment, _known_launch_paths(executable, deployment, artifacts)
        ),
        runtime=runtime,
        device_environment=combined_environment,
    )


def _discard_unwritten_log(log_stream: IO[bytes], log_path: Path) -> None:
    """Remove a log file the child never reached, so the output directory stays fresh."""
    log_stream.close()
    log_path.unlink(missing_ok=True)


def start_managed_deployment(plan: DeploymentPlan, log_path: Path) -> ManagedDeployment:
    """Start one owned child server with a new private log file."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
        port_probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            port_probe.bind((plan.host, plan.port))
        except OSError as error:
            raise ValueError(
                f"managed deployment port {plan.port} is already in use; pass --port to choose another "
                "(a recently stopped server can hold a port for up to a minute)"
            ) from error
    validate_new_file_paths((log_path,))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    validate_new_file_paths((log_path,))
    descriptor = os.open(log_path, NEW_FILE_OPEN_FLAGS, PRIVATE_FILE_PERMISSIONS)
    log_stream = os.fdopen(descriptor, "wb")
    environment = {**os.environ, **dict(plan.device_environment)}
    try:
        process = subprocess.Popen(
            plan.command,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            env=environment,
        )
    except OSError:
        _discard_unwritten_log(log_stream, log_path)
        raise
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP makes the child the leader of a group named by its pid.
        return ManagedDeployment(
            plan=plan,
            process=process,
            process_group_id=process.pid,
            log_stream=log_stream,
            log_path=log_path,
        )
    try:
        process_group_id = os.getpgid(process.pid)
    except ProcessLookupError as exception:
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        _discard_unwritten_log(log_stream, log_path)
        raise RuntimeError("managed deployment exited before process-group ownership was established") from exception
    if process_group_id != process.pid:
        process.terminate()
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        _discard_unwritten_log(log_stream, log_path)
        raise RuntimeError("managed deployment did not start in an isolated process group")
    return ManagedDeployment(
        plan=plan,
        process=process,
        process_group_id=process_group_id,
        log_stream=log_stream,
        log_path=log_path,
    )


def wait_for_deployment(deployment: ManagedDeployment, *, timeout_seconds: float) -> int:
    """Wait for the owned endpoint to serve the selected alias at the planned context length.

    Returns the served context length. A managed server that reports no context
    (meta.n_ctx, max_model_len, or Splash's status endpoint) fails readiness: a run
    cannot prove its context without it.
    """
    if timeout_seconds <= 0:
        raise ValueError("startup timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    models_url = f"{deployment.plan.base_url}/models"
    log_path = deployment.log_path
    with httpx.Client(timeout=LOCAL_READINESS_TIMEOUT_SECONDS) as client:
        while time.monotonic() < deadline:
            returncode = deployment.process.poll()
            if returncode is not None:
                raise RuntimeError(
                    _log_hint(
                        f"managed {deployment.plan.framework} server exited with status {returncode}",
                        log_path,
                    )
                    + f" ({_EARLY_EXIT_HINTS[deployment.plan.framework]})"
                )
            try:
                response = client.get(models_url)
                response.raise_for_status()
                payload = normalize_json_object(orjson.loads(response.content))
            except (httpx.HTTPError, orjson.JSONDecodeError, ValueError):
                time.sleep(READINESS_POLL_SECONDS)
                continue
            raw_models = payload.get("data")
            entry = model_entry(raw_models, deployment.plan.model_alias) if isinstance(raw_models, list) else None
            if entry is not None:
                served_tokens = reported_context_tokens(client, deployment.plan.base_url, entry)
                if served_tokens is None:
                    raise RuntimeError(
                        _log_hint(
                            "managed server did not report its served context length (meta.n_ctx)",
                            log_path,
                        )
                    )
                if served_tokens != deployment.plan.context_tokens:
                    raise RuntimeError(
                        _log_hint(
                            f"managed server reports a {served_tokens}-token context but the profile "
                            f"requires {deployment.plan.context_tokens}",
                            log_path,
                        )
                    )
                if deployment.process.poll() is not None:
                    raise RuntimeError(
                        _log_hint(f"managed {deployment.plan.framework} server exited during readiness", log_path)
                    )
                return served_tokens
            time.sleep(READINESS_POLL_SECONDS)
    raise TimeoutError(_log_hint(f"managed {deployment.plan.framework} server did not become ready", log_path))


def verify_gpu_startup(deployment: ManagedDeployment) -> None:
    """Reject a ready endpoint without positive startup evidence for its GPU backend."""
    log_path = deployment.log_path
    if deployment.process.poll() is not None:
        raise RuntimeError(_log_hint("managed server exited before GPU verification", log_path))
    with log_path.open("rb") as source:
        encoded = source.read(MAX_GPU_VERIFICATION_LOG_BYTES + 1)
    if len(encoded) > MAX_GPU_VERIFICATION_LOG_BYTES:
        raise RuntimeError(_log_hint("managed startup log exceeded the verification bound", log_path))
    log = encoded.decode("utf-8", errors="replace").lower()
    platform_markers: tuple[str, ...]
    failure_markers: tuple[str, ...]
    if deployment.plan.accelerator_platform == "nvidia-cuda":
        failure_markers = (
            "cuda unavailable",
            "cuda is not available",
            "cuda platform unavailable",
            "no cuda device",
            "failed to initialize cuda",
        )
        if deployment.plan.framework == "llama-cpp":
            # llama.cpp renamed its CUDA init log across versions: older builds print
            # "ggml_cuda_init", current builds report the backend as "using device CUDA0"
            # and a per-device "CUDA0 model buffer size". Accept any of them.
            platform_markers = ("ggml_cuda_init", "using device cuda", "cuda0 model buffer")
        else:
            platform_markers = ("cuda platform", "cudaplatform", "detected platform cuda", "using cuda", "cuda graph")
    elif deployment.plan.accelerator_platform == "amd-rocm":
        vulkan_requested = any(
            flag == "--device" and value == "Vulkan0"
            for flag, value in zip(deployment.plan.command, deployment.plan.command[1:])
        )
        if deployment.plan.framework == "llama-cpp" and vulkan_requested:
            failure_markers = ("vulkan unavailable", "failed to initialize vulkan", "no vulkan devices")
            platform_markers = ("using device vulkan0", "vulkan0 model buffer")
        else:
            failure_markers = (
                "rocm unavailable",
                "rocm is not available",
                "hip unavailable",
                "failed to initialize hip",
            )
            platform_markers = ("rocm", "hipblas", "rocblas", "hip platform", "rocmplatform")
    else:
        failure_markers = ("metal unavailable", "metal is not available", "failed to initialize metal")
        platform_markers = (
            (SPLASH_METAL_MARKER,) if deployment.plan.framework == "splash" else ("ggml_metal_init", "metal backend")
        )
    if any(marker in log for marker in failure_markers):
        raise RuntimeError(
            _log_hint(
                f"managed {deployment.plan.framework} reported a failed {deployment.plan.accelerator_platform} backend",
                log_path,
            )
        )
    reported_backend = any(marker in log for marker in platform_markers)
    if not reported_backend:
        raise RuntimeError(
            _log_hint(
                f"managed {deployment.plan.framework} did not report its "
                f"{deployment.plan.accelerator_platform} backend",
                log_path,
            )
        )
    if deployment.plan.framework == "llama-cpp":
        offload_counts = tuple(
            (int(match.group(1)), int(match.group(2))) for match in FULL_OFFLOAD_PATTERN.finditer(log)
        )
        reported_full_offload = any(offloaded == total and total > 0 for offloaded, total in offload_counts)
        if not reported_full_offload:
            raise RuntimeError(
                _log_hint(
                    f"managed llama.cpp did not report full {deployment.plan.accelerator_platform} offload", log_path
                )
            )
        return
    if deployment.plan.framework == "sglang":
        # SGLang sizes its token pool from free device memory, so a server can advertise the
        # full context while holding a pool too small to ever reach it. The pool it reports
        # must cover one full-context sequence, or long turns fail mid-replay.
        pool_sizes = tuple(int(match.group(1)) for match in TOKEN_POOL_PATTERN.finditer(log))
        if not pool_sizes:
            raise RuntimeError(_log_hint("managed SGLang did not report its token pool size", log_path))
        if max(pool_sizes) < deployment.plan.context_tokens:
            raise RuntimeError(
                _log_hint(
                    f"managed SGLang allocated a {max(pool_sizes)}-token pool, below the "
                    f"{deployment.plan.context_tokens}-token context",
                    log_path,
                )
            )
        return
    if deployment.plan.framework == "splash":
        # Splash exits before readiness unless its engine grants the exact requested
        # context, and readiness has already read that context back from /status.
        return
    # vLLM likewise sizes its KV cache from free device memory and reports the token capacity
    # it settled on. That capacity must cover one full-context sequence, or long turns fail
    # mid-replay.
    cache_sizes = tuple(int(match.group(1).replace(",", "")) for match in VLLM_KV_CACHE_PATTERN.finditer(log))
    if not cache_sizes:
        raise RuntimeError(_log_hint("managed vLLM did not report its GPU KV cache size", log_path))
    if max(cache_sizes) < deployment.plan.context_tokens:
        raise RuntimeError(
            _log_hint(
                f"managed vLLM sized a {max(cache_sizes)}-token KV cache, below the "
                f"{deployment.plan.context_tokens}-token context",
                log_path,
            )
        )


class RecordedRuntime(BaseModel, frozen=True):
    """Hold the runtime facts a deployment record states."""

    version: str
    runtime_id: str


class RecordedPlan(BaseModel, frozen=True):
    """Hold the launch facts a submission reads back from a deployment record."""

    profile_id: str
    model_release_slug: str
    hf_repository: str
    hf_revision: str
    framework: DeploymentFramework
    accelerator_backend: AcceleratorBackend
    artifact_manifest_sha256: str
    artifact_size_bytes: int
    context_tokens: int
    server_launch_command: str
    runtime: RecordedRuntime


class DeploymentRecord(BaseModel, frozen=True):
    """Hold the deployment.json facts a submission sends.

    The file also holds the local model path, the raw launch command, and the hardware
    snapshot. The reader skips them, so none of them can reach a submission.
    """

    version: DeploymentRecordVersion
    kind: DeploymentRecordKind
    status: DeploymentRecordStatus
    gpu_startup_verified: Literal[True]
    recipe: str
    deployment: RecordedPlan

    def require_benchmark(self, context: SubmissionContext) -> None:
        """Refuse a record that describes a different model, runtime, or context than the benchmark identity."""
        plan = self.deployment
        if (
            plan.artifact_manifest_sha256 != context.model_artifact_digest
            or plan.runtime.runtime_id != context.runtime_id
        ):
            raise ValueError("deployment record does not match the bound benchmark identity")
        if plan.context_tokens != context.context_tokens:
            raise ValueError("deployment record context does not match the bound benchmark identity")


def read_deployment_record(encoded: bytes, source: str) -> DeploymentRecord:
    """Parse one deployment record's exact bytes, keeping only the facts a submission sends."""
    return read_record(DeploymentRecord, encoded, source, unknown_keys="skip")


def launch_configuration_digest(plan: DeploymentPlan) -> str:
    """Hash the exact launch the plan describes, so two records can be compared."""
    command: list[JsonValue] = []
    command.extend(plan.command)
    configuration: JsonObject = {
        "framework": plan.framework,
        "accelerator_platform": plan.accelerator_platform,
        "context_tokens": plan.context_tokens,
        "reduced_context": plan.reduced_context,
        "model_alias": plan.model_alias,
        "command": command,
        "device_environment": {name: value for name, value in plan.device_environment},
        "runtime_fingerprint": plan.runtime.fingerprint,
    }
    return sha256_bytes(orjson.dumps(configuration, option=orjson.OPT_SORT_KEYS))


def write_deployment_record(
    path: Path,
    plan: DeploymentPlan,
    snapshot: HardwareSnapshot,
    *,
    recipe_text: str,
    gpu_startup_verified: bool,
) -> WrittenFile:
    """Write the exact managed deployment facts, with the recipe file's text, before inference starts."""
    if not gpu_startup_verified:
        raise ValueError("managed deployment records require verified GPU startup evidence")
    deployment_id = mint_run_id()
    data: JsonObject = {
        "version": DEPLOYMENT_RECORD_VERSION,
        "kind": DEPLOYMENT_RECORD_KIND,
        "status": DEPLOYMENT_RECORD_STATUS,
        "deployment_id": deployment_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "launch_configuration_digest": launch_configuration_digest(plan),
        "deployment": plan.to_json(),
        "recipe": recipe_text,
        "gpu_startup_verified": True,
        "hardware": snapshot.to_json(),
        "log_file": DEPLOYMENT_LOG_FILENAME,
        "upload_performed": False,
    }
    return write_digest_file(path, orjson.dumps(data, option=orjson.OPT_INDENT_2))
