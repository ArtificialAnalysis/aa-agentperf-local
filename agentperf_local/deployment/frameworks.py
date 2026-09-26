"""Decide which installed runtime can serve one recipe on this host.

A framework is offered only when the host's accelerator, the recipe's declared
support, an installed executable, and the memory a served context needs all
agree. Every refusal carries the local reason that produced it.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import orjson

from agentperf_local.common.identity import sha256_bytes, sha256_file
from agentperf_local.common.json_records import json_record
from agentperf_local.common.json_types import JsonObject
from agentperf_local.deployment.catalog import (
    DeploymentFramework,
    ModelCandidate,
    ModelDeployment,
    PortableDeviceId,
    is_development_build,
    release_version,
)
from agentperf_local.deployment.context_policy import (
    derived_minimum_memory_bytes,
    memory_fits,
    resolve_context_tokens,
)
from agentperf_local.provenance.hardware import (
    AcceleratorPlatform,
    HardwareSnapshot,
    accelerator_platform,
    selected_accelerator,
)

# `sglang version` imports torch and initialises CUDA before it can answer: measured at
# 8.5 s from a cold page cache on a DGX Spark, 2.9 s warm. A bound that a cold first run
# trips reports a real 0.5.18 as unreported and refuses the recipe.
FRAMEWORK_VERSION_TIMEOUT_SECONDS = 60.0


MAX_FRAMEWORK_VERSION_CHARACTERS = 160


type CommandFinder = Callable[[str], str | None]


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameworkExecutable:
    """Store one installed framework command without a shell."""

    framework: DeploymentFramework
    command_prefix: tuple[str, ...]
    version_command: tuple[str, ...]
    executable_path: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameworkOffer:
    """Describe one framework compatible with the detected accelerator."""

    framework: DeploymentFramework
    display_name: str
    accelerator_platform: AcceleratorPlatform
    installed: bool
    available_memory_bytes: int | None
    minimum_memory_bytes: int
    memory_fit: bool | None
    installation_hint: str
    support_note: str

    def to_json(self) -> JsonObject:
        """Return the offer as JSON data."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameworkIdentity:
    """Identify the executable selected for one managed deployment."""

    version: str
    executable_sha256: str
    fingerprint: str


def framework_supported(framework: DeploymentFramework, platform: AcceleratorPlatform) -> bool:
    if framework == "llama-cpp":
        return True
    return platform == "nvidia-cuda"


def platform_device_id(platform: AcceleratorPlatform) -> PortableDeviceId:
    if platform == "nvidia-cuda":
        return "nvidia-cuda"
    if platform == "amd-rocm":
        return "amd-rocm"
    return "apple-silicon"


def framework_display_name(framework: DeploymentFramework) -> str:
    """Name one deployment framework the way its project spells it."""
    if framework == "llama-cpp":
        return "llama.cpp"
    if framework == "vllm":
        return "vLLM"
    return "SGLang"


def installation_hint(framework: DeploymentFramework) -> str:
    if framework == "llama-cpp":
        return "Install a backend-enabled llama.cpp build that provides llama-server or llama."
    if framework == "vllm":
        return "Install vLLM in the selected CUDA environment."
    return "Install SGLang in the selected CUDA environment."


def _support_note(framework: DeploymentFramework, platform: AcceleratorPlatform) -> str:
    if framework == "llama-cpp":
        return "Native GGUF path using the framework's CUDA, HIP, or Metal backend."
    if platform == "nvidia-cuda":
        return "Native weights path using the framework's CUDA backend; this path is not offered on ROCm."
    raise ValueError(f"{framework_display_name(framework)} is not supported on this platform")


def available_accelerator_memory(snapshot: HardwareSnapshot, platform: AcceleratorPlatform) -> int | None:
    """Return the memory one managed deployment may use on the selected accelerator.

    A unified-memory accelerator draws on host memory and reports no capacity of its
    own, so the host total is the honest figure. That covers Apple Silicon and NVIDIA's
    coherent-memory parts alike; the platform itself does not decide it.
    """
    del platform
    accelerator = selected_accelerator(snapshot)
    if accelerator.memory_is_unified:
        # An AMD APU can expose a GPU pool smaller than the host's shared memory.
        if accelerator.memory_bytes is not None:
            if snapshot.memory_bytes is None:
                return accelerator.memory_bytes
            return min(accelerator.memory_bytes, snapshot.memory_bytes)
        return snapshot.memory_bytes
    return accelerator.memory_bytes


def find_executable(command: str) -> str | None:
    """Look up shutil.which at call time so tests can patch it after import."""
    return shutil.which(command)


def resolve_framework_executable(
    framework: DeploymentFramework,
    *,
    command_finder: CommandFinder = find_executable,
) -> FrameworkExecutable | None:
    """Resolve one installed framework without invoking a shell."""
    if framework == "llama-cpp":
        server = command_finder("llama-server")
        if server is not None:
            return FrameworkExecutable(
                framework=framework,
                command_prefix=(server,),
                version_command=(server, "--version"),
                executable_path=Path(server),
            )
        unified = command_finder("llama")
        if unified is not None:
            return FrameworkExecutable(
                framework=framework,
                command_prefix=(unified, "serve"),
                version_command=(unified, "--version"),
                executable_path=Path(unified),
            )
        return None
    if framework == "vllm":
        # vLLM serves through `vllm serve` and answers `vllm --version` on a zero exit.
        vllm = command_finder("vllm")
        if vllm is not None:
            return FrameworkExecutable(
                framework=framework,
                command_prefix=(vllm, "serve"),
                version_command=(vllm, "--version"),
                executable_path=Path(vllm),
            )
        # Fall back to the module the managed interpreter imports, the way SGLang does.
        if importlib.util.find_spec("vllm") is None:
            return None
        return FrameworkExecutable(
            framework=framework,
            command_prefix=(sys.executable, "-m", "vllm", "serve"),
            version_command=(sys.executable, "-c", "import vllm; print(vllm.__version__)"),
            executable_path=Path(sys.executable),
        )
    executable = command_finder("sglang")
    if executable is not None:
        return FrameworkExecutable(
            framework=framework,
            # The SGLang CLI reports its version through a subcommand; --version exits
            # non-zero, which would record every SGLang run as an unreported version.
            command_prefix=(executable, "serve"),
            version_command=(executable, "version"),
            executable_path=Path(executable),
        )
    if importlib.util.find_spec("sglang") is None:
        return None
    return FrameworkExecutable(
        framework=framework,
        command_prefix=(sys.executable, "-m", "sglang.launch_server"),
        # `python -m sglang.version` prints only a RuntimeWarning, so the attribute is read.
        version_command=(sys.executable, "-c", "import sglang; print(sglang.__version__)"),
        executable_path=Path(sys.executable),
    )


def framework_offers(
    snapshot: HardwareSnapshot,
    candidate: ModelCandidate,
    context_tokens: int | None = None,
    *,
    command_finder: CommandFinder = find_executable,
) -> tuple[FrameworkOffer, ...]:
    """Return compatible frameworks in the model's canonical order.

    The memory requirement scales with the requested context; the default is the
    recipe's full benchmark context and reproduces the catalog minimum exactly.
    """
    deployment = candidate.deployment
    if deployment is None:
        return ()
    platform = accelerator_platform(snapshot)
    minimum_memory_bytes = derived_minimum_memory_bytes(deployment, resolve_context_tokens(deployment, context_tokens))
    platform_device = platform_device_id(platform)
    available_memory_bytes = available_accelerator_memory(snapshot, platform)
    memory_fit = None if available_memory_bytes is None else memory_fits(available_memory_bytes, minimum_memory_bytes)
    offers: list[FrameworkOffer] = []
    candidate_devices = frozenset(evidence.device_id for evidence in candidate.device_evidence)
    for framework in deployment.frameworks:
        if platform_device not in candidate_devices:
            continue
        if not framework_supported(framework, platform):
            continue
        offers.append(
            FrameworkOffer(
                framework=framework,
                display_name=framework_display_name(framework),
                accelerator_platform=platform,
                installed=resolve_framework_executable(framework, command_finder=command_finder) is not None,
                available_memory_bytes=available_memory_bytes,
                minimum_memory_bytes=minimum_memory_bytes,
                memory_fit=memory_fit,
                installation_hint=installation_hint(framework),
                support_note=_support_note(framework, platform),
            )
        )
    return tuple(offers)


def _safe_version(encoded: bytes) -> str:
    decoded = encoded.decode("utf-8", errors="replace").splitlines()
    if not decoded:
        return "unreported"
    version = decoded[0].strip()
    if (
        not version
        or len(version) > MAX_FRAMEWORK_VERSION_CHARACTERS
        or not version.isascii()
        or not version.isprintable()
    ):
        return "unreported"
    return version


def framework_identity(executable: FrameworkExecutable) -> FrameworkIdentity:
    try:
        completed = subprocess.run(
            executable.version_command,
            check=False,
            capture_output=True,
            timeout=FRAMEWORK_VERSION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # Distinct from "unreported": the runtime exists and may be the pinned release,
        # it just did not answer in time. The gate names this so the fix is obvious.
        version = "unreported (version check timed out)"
    except OSError:
        version = "unreported"
    else:
        version = _safe_version(completed.stdout or completed.stderr) if completed.returncode == 0 else "unreported"
    try:
        executable_sha256 = sha256_file(executable.executable_path)
    except OSError as error:
        raise ValueError("managed framework executable could not be fingerprinted") from error
    fingerprint = sha256_bytes(orjson.dumps([executable.framework, version, executable_sha256]))
    return FrameworkIdentity(version=version, executable_sha256=executable_sha256, fingerprint=fingerprint)


RUNTIME_VERSION_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def require_supported_runtime(
    deployment: ModelDeployment, runtime: FrameworkIdentity, framework: DeploymentFramework
) -> None:
    """Refuse any runtime other than the one release or build this recipe is verified against.

    A runtime can load the pinned weights, report the right context, and report its GPU
    backend while generating nonsense. No later check reads generated text, so the only
    place to catch it is here, before a replay produces a plausible-looking number. Each
    profile names the runtime version that produced its verified behavior.
    """
    required = deployment.runtime_version_for(framework)
    if required is None:
        return
    if is_development_build(required):
        # A development build has no release to compare against, so only the exact build is served.
        if runtime.version != required:
            raise ValueError(
                f"this recipe is verified against the {framework} development build {required}, "
                f"but the installed {framework} reports {runtime.version}"
            )
        return
    match = RUNTIME_VERSION_PATTERN.search(runtime.version)
    if match is None:
        raise ValueError(
            f"this recipe is verified against {framework} {required}, and the installed "
            f"{framework} did not report a version it can be compared against ({runtime.version})"
        )
    found = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if found != release_version(required, "runtime_version"):
        installed = ".".join(str(part) for part in found)
        raise ValueError(f"this recipe is verified against {framework} {required}, but {installed} is installed")
