"""Load the closed pilot model catalog into typed records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from agentperf_local.common.identity import sha256_bytes, validate_identifier
from agentperf_local.common.json_fields import (
    decode_json_object,
    one_of,
    optional_integer,
    optional_object,
    optional_string,
    require_exact_keys,
    required_boolean,
    required_integer,
    required_list,
    required_non_negative_integer,
    required_object,
    required_string,
)
from agentperf_local.common.json_records import json_field_names
from agentperf_local.common.json_types import JsonObject
from agentperf_local.common.package_paths import PACKAGE_DATA_ROOT
from agentperf_local.provenance.benchmark import BENCHMARK_CONTEXT_TOKENS

MODEL_CATALOG_VERSION = 2
MODEL_CATALOG_KIND = "consumer_gpu_model_candidates"
MODEL_CATALOG_STATUS = "pilot-candidates-not-release"
EXPECTED_MODEL_COUNT = 29
LEGACY_DEVICE_IDS: tuple[DeviceId, ...] = ("rtx-5090", "rtx-pro-6000", "dgx-spark")
PORTABLE_DEVICE_IDS: tuple[PortableDeviceId, ...] = ("nvidia-cuda", "amd-rocm", "apple-silicon")
MAX_DISPLAY_NAME_CHARACTERS = 160
MAX_URL_CHARACTERS = 2_048
MAX_MODEL_CATALOG_BYTES = 1_048_576
MAX_DEPLOYMENT_ARTIFACTS = 512
MAX_ARTIFACT_PATH_CHARACTERS = 255
HF_REVISION_HEX_DIGITS = 40
SHA256_HEX_DIGITS = 64
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
RELEASE_VERSION_PARTS = 3
# A development build is pinned by its exact version string, such as 0.1.dev20073+g8e685d198:
# a base version, a commit count, and the abbreviated commit it was built from.
DEVELOPMENT_BUILD_PATTERN = re.compile(r"^\d+\.\d+\.dev\d+\+g[0-9a-f]{7,40}$")
ARTIFACT_PATH_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9._+-]*[A-Za-z0-9_])?$")
BUNDLED_MODEL_CATALOG_PATH = PACKAGE_DATA_ROOT / "model-candidates-v2.json"
BUNDLED_MODEL_CATALOG_DIGEST = "sha256:3eb44c124946317fd05b285bb44427dfd43ce302574d8e28bf420e69e2122afd"

type CatalogKind = Literal["consumer_gpu_model_candidates"]
type CatalogStatus = Literal["pilot-candidates-not-release"]
type ModelLicense = Literal["Apache-2.0", "OpenMDW-1.1", "NVIDIA-Nemotron-Open-Model-License", "Qwen-Community-1.0"]
type RuntimeFamily = Literal["llama-cpp", "sglang", "vllm"]
type ToolCallParser = Literal["gemma4", "gpt-oss", "qwen3_coder", "qwen3_xml"]
type ReasoningParser = Literal["gemma4", "gpt-oss", "nemotron_v3", "qwen3"]
type ThinkingPolicy = Literal["disabled", "enabled", "enabled-medium-candidate"]
type SpeculationPolicy = Literal[
    "disabled-target-only-baseline",
    "enabled-mtp-self-draft",
    "enabled-mtp-external-draft",
    "enabled-dflash-external-draft",
    "enabled-dflash-draft",
    "enabled-dspark-draft",
    "enabled-vllm-external-draft",
]
type StoredCheckpointKind = Literal["gguf-q4-0", "gguf-q4-k-m", "gguf-iq4-nl", "modelopt-nvfp4-mixed", "mxfp4-native"]
type ComputePathPolicy = Literal["record-resolved-path"]
type ArtifactManifestStatus = Literal["pending-complete-file-manifest", "complete-file-sha256-pinned"]
type DeviceId = Literal[
    "rtx-5090",
    "rtx-pro-6000",
    "dgx-spark",
    "nvidia-cuda",
    "amd-rocm",
    "apple-silicon",
]
type PortableDeviceId = Literal["nvidia-cuda", "amd-rocm", "apple-silicon"]
type DeviceArchitecture = Literal["sm120", "sm121", "cuda", "rocm", "metal"]
type DeviceEvidenceLevel = Literal[
    "official-gguf-backend",
    "upstream-gguf-conversion",
    "vendor-hardware-listed-recipe-confirm",
    "vendor-local-surface-aa-recipe-needed",
    "vendor-sglang-recipe",
    "upstream-end-to-end",
    "upstream-boot-and-serve",
    "local-end-to-end",
    "local-inference-screen",
]
type AaAdmissionState = Literal["unqualified"]
type DeploymentFramework = Literal["llama-cpp", "sglang", "vllm"]
type ArtifactKind = Literal["gguf-single-file", "gguf-file-set", "safetensors-repository"]
type Quantization = Literal["Q4_0", "Q4_K_M", "IQ4_NL", "NVFP4", "MXFP4"]
type LlamaCppBackend = Literal["rocm", "vulkan", "metal"]
type LlamaCppLoadMode = Literal["none", "mmap"]
# How llama.cpp serves tensors it reads on demand. "on-direct" serves the rows of a
# per-layer-embedding table with explicit reads from disk, so the table is never resident.
type LlamaCppLazyMode = Literal["on-direct"]
LLAMA_CPP_BACKENDS: tuple[LlamaCppBackend, ...] = ("rocm", "vulkan", "metal")
LLAMA_CPP_LOAD_MODES: tuple[LlamaCppLoadMode, ...] = ("none", "mmap")
LLAMA_CPP_LAZY_MODES: tuple[LlamaCppLazyMode, ...] = ("on-direct",)
# Which fused-expert kernel serves a mixture-of-experts recipe. SGLang picks one from
# the device when this is unset, and its choice is not always implemented for the
# recipe's quantization, so a recipe that needs a particular kernel names it.
type MoeRunnerBackend = Literal["flashinfer_cutlass"]

DEPLOYMENT_FRAMEWORK_ORDER: tuple[DeploymentFramework, ...] = ("llama-cpp", "sglang", "vllm")
DEVICE_IDS: tuple[DeviceId, ...] = (*LEGACY_DEVICE_IDS, *PORTABLE_DEVICE_IDS)
DEVICE_ARCHITECTURES: tuple[DeviceArchitecture, ...] = ("sm120", "sm121", "cuda", "rocm", "metal")
DEVICE_EVIDENCE_LEVELS: tuple[DeviceEvidenceLevel, ...] = (
    "official-gguf-backend",
    "upstream-gguf-conversion",
    "vendor-hardware-listed-recipe-confirm",
    "vendor-local-surface-aa-recipe-needed",
    "vendor-sglang-recipe",
    "upstream-end-to-end",
    "upstream-boot-and-serve",
    "local-end-to-end",
    "local-inference-screen",
)
MODEL_LICENSES: tuple[ModelLicense, ...] = (
    "Apache-2.0",
    "OpenMDW-1.1",
    "NVIDIA-Nemotron-Open-Model-License",
    "Qwen-Community-1.0",
)
MOE_RUNNER_BACKENDS: tuple[MoeRunnerBackend, ...] = ("flashinfer_cutlass",)
REASONING_PARSERS: tuple[ReasoningParser, ...] = ("gemma4", "gpt-oss", "nemotron_v3", "qwen3")
THINKING_POLICIES: tuple[ThinkingPolicy, ...] = ("disabled", "enabled", "enabled-medium-candidate")
SPECULATION_POLICIES: tuple[SpeculationPolicy, ...] = (
    "disabled-target-only-baseline",
    "enabled-mtp-self-draft",
    "enabled-mtp-external-draft",
    "enabled-dflash-external-draft",
    "enabled-dflash-draft",
    "enabled-dspark-draft",
    "enabled-vllm-external-draft",
)
RUNTIME_FAMILIES: tuple[RuntimeFamily, ...] = ("llama-cpp", "sglang", "vllm")
TOOL_CALL_PARSERS: tuple[ToolCallParser, ...] = ("gemma4", "gpt-oss", "qwen3_coder", "qwen3_xml")
STORED_CHECKPOINT_KINDS: tuple[StoredCheckpointKind, ...] = (
    "gguf-q4-0",
    "gguf-q4-k-m",
    "gguf-iq4-nl",
    "modelopt-nvfp4-mixed",
    "mxfp4-native",
)
ARTIFACT_KINDS: tuple[ArtifactKind, ...] = ("gguf-single-file", "gguf-file-set", "safetensors-repository")
QUANTIZATIONS: tuple[Quantization, ...] = ("Q4_0", "Q4_K_M", "IQ4_NL", "NVFP4", "MXFP4")
ARTIFACT_MANIFEST_STATUSES: tuple[ArtifactManifestStatus, ...] = (
    "pending-complete-file-manifest",
    "complete-file-sha256-pinned",
)
# Each stored checkpoint kind can only be served from the artifact layout it was
# published in, so the catalog cannot describe a GGUF checkpoint as a weights
# repository or the reverse.
_CHECKPOINT_ARTIFACT_KINDS: dict[StoredCheckpointKind, tuple[ArtifactKind, ...]] = {
    "gguf-q4-0": ("gguf-single-file",),
    "gguf-q4-k-m": ("gguf-single-file", "gguf-file-set"),
    "gguf-iq4-nl": ("gguf-single-file", "gguf-file-set"),
    "modelopt-nvfp4-mixed": ("safetensors-repository",),
    "mxfp4-native": ("safetensors-repository",),
}
_CHECKPOINT_QUANTIZATIONS: dict[StoredCheckpointKind, Quantization] = {
    "gguf-q4-0": "Q4_0",
    "gguf-q4-k-m": "Q4_K_M",
    "gguf-iq4-nl": "IQ4_NL",
    "modelopt-nvfp4-mixed": "NVFP4",
    "mxfp4-native": "MXFP4",
}
# A weights repository is only servable when the runtime can read the model shape
# and the tokenizer beside the tensors.
REQUIRED_REPOSITORY_FILES: tuple[str, ...] = ("config.json", "tokenizer.json")

_CATALOG_FIELDS = frozenset({"version", "kind", "status", "as_of", "models"})
_DEVICE_ARCHITECTURES: dict[DeviceId, DeviceArchitecture] = {
    "rtx-5090": "sm120",
    "rtx-pro-6000": "sm120",
    "dgx-spark": "sm121",
    "nvidia-cuda": "cuda",
    "amd-rocm": "rocm",
    "apple-silicon": "metal",
}


def _optional_positive_integer(data: JsonObject, key: str, source: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{source}.{key} must be a positive integer or null")
    return value


def _literal[LiteralValue: str](value: str, expected: LiteralValue, field: str) -> LiteralValue:
    if value != expected:
        raise ValueError(f"{field} must be {expected}")
    return expected


def _revision(value: str, field: str) -> str:
    if len(value) != HF_REVISION_HEX_DIGITS or value != value.lower():
        raise ValueError(f"{field} must be 40 lowercase hexadecimal digits")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be 40 lowercase hexadecimal digits")
    return value


def _sha256(value: str, field: str) -> str:
    if len(value) != SHA256_HEX_DIGITS or value != value.lower():
        raise ValueError(f"{field} must be 64 lowercase hexadecimal digits")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def _https_url(value: str, field: str) -> str:
    if len(value) > MAX_URL_CHARACTERS or not value.startswith("https://"):
        raise ValueError(f"{field} must be a short HTTPS URL")
    return value


def release_version(value: str, field: str) -> tuple[int, int, int]:
    """Read one three-part release version as comparable numbers."""
    parts = value.split(".")
    if len(parts) != RELEASE_VERSION_PARTS or not all(part.isdigit() for part in parts):
        raise ValueError(f"{field} must be a three-part release version such as 0.5.18")
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch


def is_development_build(value: str) -> bool:
    """Return whether a runtime pin names one exact development build instead of a release."""
    return DEVELOPMENT_BUILD_PATTERN.fullmatch(value) is not None


def validate_artifact_path(value: str, field: str) -> str:
    """Require one relative repository path that can never escape the model cache."""
    if not value or len(value) > MAX_ARTIFACT_PATH_CHARACTERS:
        raise ValueError(f"{field} must be a short repository path")
    segments = value.split("/")
    if any(ARTIFACT_PATH_SEGMENT_PATTERN.fullmatch(segment) is None for segment in segments):
        raise ValueError(f"{field} must be a relative path of plain name segments")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DeploymentArtifact:
    """Pin one file of a managed deployment by path, digest, and size."""

    filename: str
    sha256: str
    size_bytes: int
    source_repository: str | None = None
    source_revision: str | None = None

    def __post_init__(self) -> None:
        """Require a relative path with an exact digest and a non-negative size."""
        validate_artifact_path(self.filename, "artifact.filename")
        _sha256(self.sha256, "artifact.sha256")
        if self.size_bytes < 0:
            raise ValueError("artifact.size_bytes must not be negative")
        if (self.source_repository is None) != (self.source_revision is None):
            raise ValueError("artifact source_repository and source_revision must be set together")
        if self.source_repository is not None:
            if REPOSITORY_PATTERN.fullmatch(self.source_repository) is None:
                raise ValueError("artifact.source_repository must contain one owner and repository name")
            if self.source_revision is None:
                raise ValueError("artifact.source_revision must be set with source_repository")
            _revision(self.source_revision, "artifact.source_revision")

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> DeploymentArtifact:
        """Read one pinned artifact record."""
        require_exact_keys(data, json_field_names(cls), source)
        return cls(
            filename=validate_artifact_path(required_string(data, "filename", source), f"{source}.filename"),
            sha256=_sha256(required_string(data, "sha256", source), f"{source}.sha256"),
            size_bytes=required_integer(data, "size_bytes", source),
            source_repository=optional_string(data, "source_repository", source),
            source_revision=optional_string(data, "source_revision", source),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LlamaCppLaunch:
    """Pin llama.cpp loading, batching, and speculative decoding."""

    batch_size: int
    ubatch_size: int
    speculative_tokens: int
    draft_model_filename: str | None
    target_backend_sampling: bool
    draft_backend_sampling: bool
    backend: LlamaCppBackend | None = None
    load_mode: LlamaCppLoadMode | None = None
    lazy_mode: LlamaCppLazyMode | None = None
    threads: int | None = None
    flash_attention: bool = False
    disable_fit: bool = False
    cache_ram_mib: int | None = None

    def __post_init__(self) -> None:
        """Require usable batch sizes and a positive draft depth."""
        if min(self.batch_size, self.ubatch_size, self.speculative_tokens) <= 0:
            raise ValueError("llama.cpp batch sizes and speculative token count must be positive")
        if self.ubatch_size > self.batch_size:
            raise ValueError("llama.cpp ubatch_size must not exceed batch_size")
        if self.draft_model_filename is not None:
            validate_artifact_path(self.draft_model_filename, "llama_cpp.draft_model_filename")
        if self.backend is not None and self.backend not in LLAMA_CPP_BACKENDS:
            raise ValueError("llama.cpp backend must be rocm, vulkan, or metal")
        if self.load_mode is not None and self.load_mode not in LLAMA_CPP_LOAD_MODES:
            raise ValueError("llama.cpp load_mode must be none or mmap")
        if self.lazy_mode is not None and self.lazy_mode not in LLAMA_CPP_LAZY_MODES:
            raise ValueError("llama.cpp lazy_mode must be on-direct")
        if self.threads is not None and self.threads <= 0:
            raise ValueError("llama.cpp threads must be positive")
        if self.cache_ram_mib is not None and self.cache_ram_mib < 0:
            raise ValueError("llama.cpp cache_ram_mib must be non-negative")

    @property
    def backend_device(self) -> str | None:
        """Name the first device in the requested backend."""
        if self.backend == "rocm":
            return "ROCm0"
        if self.backend == "vulkan":
            return "Vulkan0"
        if self.backend == "metal":
            return "MTL0"
        return None

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> LlamaCppLaunch:
        """Read one pinned llama.cpp launch configuration."""
        require_exact_keys(data, json_field_names(cls), source)
        draft_model_filename = optional_string(data, "draft_model_filename", source)
        backend = optional_string(data, "backend", source)
        load_mode = optional_string(data, "load_mode", source)
        lazy_mode = optional_string(data, "lazy_mode", source)
        return cls(
            batch_size=required_integer(data, "batch_size", source),
            ubatch_size=required_integer(data, "ubatch_size", source),
            speculative_tokens=required_integer(data, "speculative_tokens", source),
            draft_model_filename=(
                validate_artifact_path(draft_model_filename, f"{source}.draft_model_filename")
                if draft_model_filename is not None
                else None
            ),
            target_backend_sampling=required_boolean(data, "target_backend_sampling", source),
            draft_backend_sampling=required_boolean(data, "draft_backend_sampling", source),
            backend=one_of(backend, LLAMA_CPP_BACKENDS, f"{source}.backend") if backend is not None else None,
            load_mode=one_of(load_mode, LLAMA_CPP_LOAD_MODES, f"{source}.load_mode") if load_mode is not None else None,
            lazy_mode=one_of(lazy_mode, LLAMA_CPP_LAZY_MODES, f"{source}.lazy_mode") if lazy_mode is not None else None,
            threads=_optional_positive_integer(data, "threads", source),
            flash_attention=required_boolean(data, "flash_attention", source),
            disable_fit=required_boolean(data, "disable_fit", source),
            cache_ram_mib=optional_integer(data, "cache_ram_mib", source),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VllmLaunch:
    """Pin extra vLLM arguments and environment variables for one recipe."""

    arguments: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Require stable arguments and a unique sorted environment."""
        if not self.arguments or any(not argument or not argument.isprintable() for argument in self.arguments):
            raise ValueError("vLLM arguments must be non-empty printable text")
        names = tuple(name for name, _ in self.environment)
        if len(set(names)) != len(names) or names != tuple(sorted(names)):
            raise ValueError("vLLM environment names must be unique and sorted")
        for name, value in self.environment:
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None or not value or not value.isprintable():
                raise ValueError("vLLM environment must use uppercase names and printable values")

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> VllmLaunch:
        """Read one pinned vLLM launch configuration."""
        require_exact_keys(data, json_field_names(cls), source)
        raw_arguments = required_list(data, "arguments", source)
        arguments: list[str] = []
        for index, value in enumerate(raw_arguments):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{source}.arguments[{index}] must be non-empty text")
            arguments.append(value)
        raw_environment = data.get("environment")
        if not isinstance(raw_environment, dict):
            raise ValueError(f"{source}.environment must be an object")
        environment: list[tuple[str, str]] = []
        for name, value in raw_environment.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{source}.environment.{name} must be non-empty text")
            environment.append((name, value))
        environment.sort()
        return cls(arguments=tuple(arguments), environment=tuple(environment))


@dataclass(frozen=True, slots=True, kw_only=True)
class DeploymentMemory:
    """Describe how one model's resident memory grows with the served context.

    The catalog stores the attention shape rather than a single opaque number so the
    memory floor can be recomputed for a reduced context and checked against the
    pinned full-context minimum on every use.
    """

    # Full and sliding layers can hold different head shapes: a Gemma 4 global layer
    # keeps one or two wide heads while its sliding layers keep eight narrow ones, so
    # one shape for both classes would misprice the cache by several gigabytes.
    full_attention_layers: int
    full_kv_heads: int
    full_kv_head_dimension: int
    sliding_attention_layers: int
    sliding_kv_heads: int
    sliding_kv_head_dimension: int
    # Tokens each sliding-attention layer keeps. This is the runtime's choice, not
    # only the model's window: llama.cpp pads the window, while SGLang sizes a share
    # of the token pool, so the pinned value is read from the runtime.
    sliding_cached_tokens: int
    kv_bytes_per_scalar: int
    recurrent_state_slots: int
    constant_state_bytes: int
    runtime_overhead_bytes: int
    # Artifact bytes the runtime reads from disk on demand and never holds resident,
    # such as a per-layer-embedding table served by llama.cpp lazy reads. The memory
    # floor leaves them out.
    lazy_read_bytes: int = 0

    def __post_init__(self) -> None:
        """Require an attention shape that can hold a cache."""
        if self.full_attention_layers < 0 or self.sliding_attention_layers < 0:
            raise ValueError("attention layer counts must not be negative")
        if self.full_attention_layers + self.sliding_attention_layers <= 0:
            raise ValueError("a managed recipe must have at least one attention layer")
        if self.kv_bytes_per_scalar <= 0:
            raise ValueError("KV scalar width must be positive")
        for layers, heads, dimension, name in (
            (self.full_attention_layers, self.full_kv_heads, self.full_kv_head_dimension, "full"),
            (self.sliding_attention_layers, self.sliding_kv_heads, self.sliding_kv_head_dimension, "sliding"),
        ):
            if layers > 0 and min(heads, dimension) <= 0:
                raise ValueError(f"{name}-attention layers need a positive KV head shape")
            if layers == 0 and (heads or dimension):
                raise ValueError(f"a recipe without {name} attention must not declare its head shape")
        if self.sliding_attention_layers > 0 and self.sliding_cached_tokens <= 0:
            raise ValueError("sliding attention layers must cache a positive number of tokens")
        if self.sliding_attention_layers == 0 and self.sliding_cached_tokens != 0:
            raise ValueError("a recipe without sliding attention must not cache sliding tokens")
        if self.constant_state_bytes < 0 or self.runtime_overhead_bytes <= 0:
            raise ValueError("memory reserves must be non-negative and the runtime overhead positive")
        if (self.recurrent_state_slots > 0) != (self.constant_state_bytes > 0):
            raise ValueError("recurrent state slots and constant state bytes must both be set or both be zero")
        if self.recurrent_state_slots < 0:
            raise ValueError("recurrent state slots must not be negative")
        if self.lazy_read_bytes < 0:
            raise ValueError("lazy_read_bytes must not be negative")

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> DeploymentMemory:
        """Read one memory-shape record."""
        require_exact_keys(data, json_field_names(cls), source)
        return cls(
            full_attention_layers=required_non_negative_integer(data, "full_attention_layers", source),
            full_kv_heads=required_non_negative_integer(data, "full_kv_heads", source),
            full_kv_head_dimension=required_non_negative_integer(data, "full_kv_head_dimension", source),
            sliding_attention_layers=required_non_negative_integer(data, "sliding_attention_layers", source),
            sliding_kv_heads=required_non_negative_integer(data, "sliding_kv_heads", source),
            sliding_kv_head_dimension=required_non_negative_integer(data, "sliding_kv_head_dimension", source),
            sliding_cached_tokens=required_non_negative_integer(data, "sliding_cached_tokens", source),
            kv_bytes_per_scalar=required_integer(data, "kv_bytes_per_scalar", source),
            recurrent_state_slots=required_non_negative_integer(data, "recurrent_state_slots", source),
            constant_state_bytes=required_non_negative_integer(data, "constant_state_bytes", source),
            runtime_overhead_bytes=required_integer(data, "runtime_overhead_bytes", source),
            lazy_read_bytes=required_non_negative_integer(data, "lazy_read_bytes", source),
        )


def _optional_moe_runner_backend(data: JsonObject, source: str) -> MoeRunnerBackend | None:
    """Read the fused-expert kernel a recipe names, or None when it names none."""
    value = optional_string(data, "moe_runner_backend", source)
    if value is None:
        return None
    return one_of(value, MOE_RUNNER_BACKENDS, f"{source}.moe_runner_backend")


def _runtime_versions_from_json(data: JsonObject, source: str) -> tuple[tuple[DeploymentFramework, str], ...]:
    """Read the per-framework runtime pins as canonical-ordered (framework, version) pairs."""
    raw = data.get("runtime_versions")
    if not isinstance(raw, dict):
        raise ValueError(f"{source}.runtime_versions must be an object")
    pairs: list[tuple[DeploymentFramework, str]] = []
    for key, value in raw.items():
        framework = one_of(key, DEPLOYMENT_FRAMEWORK_ORDER, f"{source}.runtime_versions key '{key}'")
        if not isinstance(value, str):
            raise ValueError(f"{source}.runtime_versions.{key} must be text")
        pairs.append((framework, value))
    pairs.sort(key=lambda pair: DEPLOYMENT_FRAMEWORK_ORDER.index(pair[0]))
    return tuple(pairs)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelDeployment:
    """Describe one exact managed model artifact set and its runtimes."""

    artifact_kind: ArtifactKind
    artifacts: tuple[DeploymentArtifact, ...]
    context_tokens: int
    frameworks: tuple[DeploymentFramework, ...]
    memory: DeploymentMemory
    minimum_memory_bytes: int
    # The release each framework is verified against, keyed by framework in canonical
    # order. A runtime can load a checkpoint, pass every startup check, and still emit
    # nonsense, and nothing downstream reads generated text, so the launcher serves the
    # named version and refuses every other. A framework with no comparable release
    # (llama.cpp) is simply absent from the map.
    runtime_versions: tuple[tuple[DeploymentFramework, str], ...]
    # The fused-expert kernel this recipe is served with, or None when it has no
    # mixture-of-experts layers or the runtime's own choice is known to serve it.
    moe_runner_backend: MoeRunnerBackend | None
    model_alias: str
    quantization: Quantization
    model_filename: str | None = None
    llama_cpp: LlamaCppLaunch | None = None
    vllm: VllmLaunch | None = None

    def __post_init__(self) -> None:
        """Require one complete pinned recipe the managed launcher can serve."""
        if not self.artifacts or len(self.artifacts) > MAX_DEPLOYMENT_ARTIFACTS:
            raise ValueError("a managed deployment must pin between one and 512 files")
        filenames = tuple(artifact.filename for artifact in self.artifacts)
        if len(set(filenames)) != len(filenames) or filenames != tuple(sorted(filenames)):
            raise ValueError("managed artifact filenames must be unique and sorted")
        if self.artifact_kind in ("gguf-single-file", "gguf-file-set"):
            if self.artifact_kind == "gguf-single-file" and len(self.artifacts) != 1:
                raise ValueError("a GGUF recipe must pin exactly one .gguf file")
            if not all(filename.endswith(".gguf") for filename in filenames):
                raise ValueError("a GGUF recipe must pin only .gguf files")
            if self.frameworks != ("llama-cpp",):
                raise ValueError("a GGUF recipe is served by llama.cpp alone")
            if self.artifact_kind == "gguf-file-set" and self.model_filename is None:
                raise ValueError("a multi-file GGUF recipe must name its pinned target model file")
            if self.model_filename is not None and self.model_filename not in filenames:
                raise ValueError("a GGUF recipe must name its pinned target model file")
            if self.vllm is not None:
                raise ValueError("a GGUF recipe must not carry vLLM launch settings")
        else:
            missing = tuple(name for name in REQUIRED_REPOSITORY_FILES if name not in filenames)
            if missing:
                raise ValueError(f"a weights recipe must pin {', '.join(missing)}")
            if not any(name.endswith(".safetensors") for name in filenames):
                raise ValueError("a weights recipe must pin at least one safetensors file")
            if "llama-cpp" in self.frameworks:
                raise ValueError("llama.cpp does not serve a safetensors weights repository")
            if self.model_filename is not None or self.llama_cpp is not None:
                raise ValueError("a weights recipe must not carry llama.cpp file or launch settings")
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if self.memory.lazy_read_bytes > self.artifact_size_bytes:
            raise ValueError("lazy_read_bytes must not exceed the model artifacts")
        if self.memory.lazy_read_bytes and (self.llama_cpp is None or self.llama_cpp.lazy_mode is None):
            raise ValueError("only a llama.cpp recipe with a lazy_mode can read artifact bytes on demand")
        if self.minimum_memory_bytes < self.resident_artifact_bytes:
            raise ValueError("minimum_memory_bytes must cover the resident model artifacts")
        if not self.frameworks:
            raise ValueError("a managed recipe must name at least one framework")
        expected_order = tuple(framework for framework in DEPLOYMENT_FRAMEWORK_ORDER if framework in self.frameworks)
        if len(set(self.frameworks)) != len(self.frameworks) or self.frameworks != expected_order:
            raise ValueError("managed frameworks must be unique and use canonical order")
        pinned_frameworks = tuple(name for name, _ in self.runtime_versions)
        if pinned_frameworks != tuple(name for name in DEPLOYMENT_FRAMEWORK_ORDER if name in pinned_frameworks):
            raise ValueError("runtime_versions must name each framework once, in canonical order")
        for name, version in self.runtime_versions:
            if name not in self.frameworks:
                raise ValueError(f"runtime_versions names {name}, which this recipe does not serve")
            if name == "llama-cpp":
                raise ValueError("llama.cpp publishes no comparable release version to pin")
            if not is_development_build(version):
                release_version(version, "runtime_versions")
        if self.moe_runner_backend is not None and "sglang" not in self.frameworks:
            raise ValueError("only an SGLang recipe can name a fused-expert kernel")
        if self.llama_cpp is not None:
            if "llama-cpp" not in self.frameworks:
                raise ValueError("only a llama.cpp recipe can name llama.cpp launch settings")
            draft_filename = self.llama_cpp.draft_model_filename
            if draft_filename is not None and draft_filename not in filenames:
                raise ValueError("llama.cpp draft_model_filename must name a pinned artifact")
        if self.vllm is not None and "vllm" not in self.frameworks:
            raise ValueError("only a vLLM recipe can name vLLM launch settings")
        validate_identifier(self.model_alias, "model_alias")

    @property
    def artifact_size_bytes(self) -> int:
        """Return the bytes every pinned file of this recipe occupies together."""
        return sum(artifact.size_bytes for artifact in self.artifacts)

    @property
    def resident_artifact_bytes(self) -> int:
        """Return the artifact bytes the served model holds in memory."""
        return self.artifact_size_bytes - self.memory.lazy_read_bytes

    @property
    def target_model_filename(self) -> str | None:
        """Return the GGUF target filename, including the single-file default."""
        if self.model_filename is not None:
            return self.model_filename
        if self.artifact_kind == "gguf-single-file":
            return self.artifacts[0].filename
        return None

    def runtime_version_for(self, framework: DeploymentFramework) -> str | None:
        """Return the release this recipe pins for one framework, or None when unpinned."""
        for name, version in self.runtime_versions:
            if name == framework:
                return version
        return None

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> ModelDeployment:
        """Read one exact managed-deployment record."""
        require_exact_keys(data, json_field_names(cls), source)
        raw_frameworks = data.get("frameworks")
        if not isinstance(raw_frameworks, list):
            raise ValueError(f"{source}.frameworks must be an array")
        frameworks: list[DeploymentFramework] = []
        for index, value in enumerate(raw_frameworks):
            if not isinstance(value, str):
                raise ValueError(f"{source}.frameworks[{index}] must be text")
            frameworks.append(one_of(value, DEPLOYMENT_FRAMEWORK_ORDER, f"{source}.frameworks[{index}]"))
        raw_artifacts = data.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ValueError(f"{source}.artifacts must be an array")
        artifacts: list[DeploymentArtifact] = []
        for index, raw_artifact in enumerate(raw_artifacts):
            if not isinstance(raw_artifact, dict):
                raise ValueError(f"{source}.artifacts[{index}] must be an object")
            artifacts.append(DeploymentArtifact.from_json(raw_artifact, f"{source}.artifacts[{index}]"))
        raw_llama_cpp = optional_object(data, "llama_cpp", source)
        raw_vllm = optional_object(data, "vllm", source)
        return cls(
            artifact_kind=one_of(
                required_string(data, "artifact_kind", source),
                ARTIFACT_KINDS,
                f"{source}.artifact_kind",
            ),
            artifacts=tuple(artifacts),
            context_tokens=required_integer(data, "context_tokens", source),
            frameworks=tuple(frameworks),
            memory=DeploymentMemory.from_json(required_object(data, "memory", source), f"{source}.memory"),
            minimum_memory_bytes=required_integer(data, "minimum_memory_bytes", source),
            runtime_versions=_runtime_versions_from_json(data, source),
            moe_runner_backend=_optional_moe_runner_backend(data, source),
            model_alias=required_string(data, "model_alias", source),
            quantization=one_of(
                required_string(data, "quantization", source),
                QUANTIZATIONS,
                f"{source}.quantization",
            ),
            model_filename=optional_string(data, "model_filename", source),
            llama_cpp=(
                LlamaCppLaunch.from_json(raw_llama_cpp, f"{source}.llama_cpp") if raw_llama_cpp is not None else None
            ),
            vllm=VllmLaunch.from_json(raw_vllm, f"{source}.vllm") if raw_vllm is not None else None,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceEvidence:
    """Describe upstream evidence for one candidate device."""

    device_id: DeviceId
    architecture: DeviceArchitecture
    evidence_level: DeviceEvidenceLevel
    source_url: str
    tested_input_tokens: int | None
    tested_output_tokens: int | None
    tested_concurrency: int | None
    aa_admission_state: AaAdmissionState

    def __post_init__(self) -> None:
        """Require consistent device and evidence fields."""
        if self.architecture != _DEVICE_ARCHITECTURES[self.device_id]:
            raise ValueError("device architecture does not match the catalog device")
        _https_url(self.source_url, "source_url")
        tested = (self.tested_input_tokens, self.tested_output_tokens, self.tested_concurrency)
        if any(value is not None and value <= 0 for value in tested):
            raise ValueError("tested token and concurrency values must be positive or null")
        if any(value is None for value in tested) and any(value is not None for value in tested):
            raise ValueError("tested token and concurrency values must be all present or all null")
        if self.aa_admission_state != "unqualified":
            raise ValueError("pilot device evidence must remain unqualified")

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> DeviceEvidence:
        """Read one closed device-evidence record."""
        require_exact_keys(data, json_field_names(cls), source)
        return cls(
            device_id=one_of(required_string(data, "device_id", source), DEVICE_IDS, f"{source}.device_id"),
            architecture=one_of(
                required_string(data, "architecture", source),
                DEVICE_ARCHITECTURES,
                f"{source}.architecture",
            ),
            evidence_level=one_of(
                required_string(data, "evidence_level", source),
                DEVICE_EVIDENCE_LEVELS,
                f"{source}.evidence_level",
            ),
            source_url=_https_url(required_string(data, "source_url", source), f"{source}.source_url"),
            tested_input_tokens=_optional_positive_integer(data, "tested_input_tokens", source),
            tested_output_tokens=_optional_positive_integer(data, "tested_output_tokens", source),
            tested_concurrency=_optional_positive_integer(data, "tested_concurrency", source),
            aa_admission_state=_literal(
                required_string(data, "aa_admission_state", source),
                "unqualified",
                f"{source}.aa_admission_state",
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCandidate:
    """Describe one pinned but unqualified pilot model."""

    profile_id: str
    display_name: str
    hf_repository: str
    hf_revision: str
    source_url: str
    license_id: ModelLicense
    native_context_tokens: int
    pilot_context_tokens: int
    runtime_family: RuntimeFamily
    tool_call_parser: ToolCallParser
    reasoning_parser: ReasoningParser
    thinking_policy: ThinkingPolicy
    speculation_policy: SpeculationPolicy
    stored_checkpoint_kind: StoredCheckpointKind
    compute_path_policy: ComputePathPolicy
    artifact_manifest_status: ArtifactManifestStatus
    deployment: ModelDeployment | None
    device_evidence: tuple[DeviceEvidence, ...]

    def __post_init__(self) -> None:
        """Require portable identity and fixed pilot semantics."""
        validate_identifier(self.profile_id, "profile_id")
        if len(self.display_name) > MAX_DISPLAY_NAME_CHARACTERS or not self.display_name.isprintable():
            raise ValueError("display_name must be short printable text")
        if REPOSITORY_PATTERN.fullmatch(self.hf_repository) is None:
            raise ValueError("hf_repository must contain one owner and repository name")
        _revision(self.hf_revision, "hf_revision")
        expected_source_url = f"https://huggingface.co/{self.hf_repository}"
        if self.source_url != expected_source_url:
            raise ValueError("source_url must match hf_repository")
        if self.pilot_context_tokens != BENCHMARK_CONTEXT_TOKENS:
            raise ValueError(f"pilot_context_tokens must be {BENCHMARK_CONTEXT_TOKENS}")
        if self.native_context_tokens < self.pilot_context_tokens:
            raise ValueError("native_context_tokens must cover the benchmark context")
        device_ids = tuple(evidence.device_id for evidence in self.device_evidence)
        if self.deployment is None:
            if device_ids != LEGACY_DEVICE_IDS:
                raise ValueError("device_evidence must cover the pilot devices in catalog order")
            if self.artifact_manifest_status != "pending-complete-file-manifest":
                raise ValueError("unmanaged candidates must retain a pending artifact manifest")
            return
        if self.pilot_context_tokens != self.deployment.context_tokens:
            raise ValueError("pilot_context_tokens must match the managed deployment context")
        expected_device_order = tuple(device_id for device_id in PORTABLE_DEVICE_IDS if device_id in device_ids)
        if not device_ids or len(set(device_ids)) != len(device_ids) or device_ids != expected_device_order:
            raise ValueError("managed device_evidence must be a canonical subset of CUDA, ROCm, and Apple Silicon")
        if self.artifact_manifest_status != "complete-file-sha256-pinned":
            raise ValueError("managed candidates require a complete pinned file manifest")
        if self.deployment.artifact_kind not in _CHECKPOINT_ARTIFACT_KINDS[self.stored_checkpoint_kind]:
            raise ValueError("managed artifact_kind does not match the stored checkpoint kind")
        if _CHECKPOINT_QUANTIZATIONS[self.stored_checkpoint_kind] != self.deployment.quantization:
            raise ValueError("managed quantization does not match the stored checkpoint kind")
        llama_cpp = self.deployment.llama_cpp
        if llama_cpp is not None and llama_cpp.backend is not None:
            expected_devices = ("apple-silicon",) if llama_cpp.backend == "metal" else ("amd-rocm",)
            if device_ids != expected_devices:
                raise ValueError("llama.cpp backend must match the recipe's device evidence")
        if self.speculation_policy in ("enabled-mtp-external-draft", "enabled-dflash-external-draft"):
            if llama_cpp is None or llama_cpp.draft_model_filename is None:
                raise ValueError("an external-draft policy requires a pinned llama.cpp draft model")
        if self.speculation_policy == "enabled-mtp-self-draft" and llama_cpp is not None:
            if llama_cpp.draft_model_filename is not None:
                raise ValueError("an MTP self-draft policy must not name an external draft model")

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> ModelCandidate:
        """Read one closed pilot model record."""
        require_exact_keys(data, json_field_names(cls), source)
        raw_evidence = data.get("device_evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError(f"{source}.device_evidence must be an array")
        evidence: list[DeviceEvidence] = []
        for index, raw_entry in enumerate(raw_evidence):
            if not isinstance(raw_entry, dict):
                raise ValueError(f"{source}.device_evidence[{index}] must be an object")
            evidence.append(DeviceEvidence.from_json(raw_entry, f"{source}.device_evidence[{index}]"))
        raw_deployment = data.get("deployment")
        if raw_deployment is not None and not isinstance(raw_deployment, dict):
            raise ValueError(f"{source}.deployment must be an object or null")
        deployment = (
            ModelDeployment.from_json(raw_deployment, f"{source}.deployment")
            if isinstance(raw_deployment, dict)
            else None
        )
        return cls(
            profile_id=required_string(data, "profile_id", source),
            display_name=required_string(data, "display_name", source),
            hf_repository=required_string(data, "hf_repository", source),
            hf_revision=_revision(required_string(data, "hf_revision", source), f"{source}.hf_revision"),
            source_url=_https_url(required_string(data, "source_url", source), f"{source}.source_url"),
            license_id=one_of(required_string(data, "license_id", source), MODEL_LICENSES, f"{source}.license_id"),
            native_context_tokens=required_integer(data, "native_context_tokens", source),
            pilot_context_tokens=required_integer(data, "pilot_context_tokens", source),
            runtime_family=one_of(
                required_string(data, "runtime_family", source),
                RUNTIME_FAMILIES,
                f"{source}.runtime_family",
            ),
            tool_call_parser=one_of(
                required_string(data, "tool_call_parser", source),
                TOOL_CALL_PARSERS,
                f"{source}.tool_call_parser",
            ),
            reasoning_parser=one_of(
                required_string(data, "reasoning_parser", source),
                REASONING_PARSERS,
                f"{source}.reasoning_parser",
            ),
            thinking_policy=one_of(
                required_string(data, "thinking_policy", source),
                THINKING_POLICIES,
                f"{source}.thinking_policy",
            ),
            speculation_policy=one_of(
                required_string(data, "speculation_policy", source),
                SPECULATION_POLICIES,
                f"{source}.speculation_policy",
            ),
            stored_checkpoint_kind=one_of(
                required_string(data, "stored_checkpoint_kind", source),
                STORED_CHECKPOINT_KINDS,
                f"{source}.stored_checkpoint_kind",
            ),
            compute_path_policy=_literal(
                required_string(data, "compute_path_policy", source),
                "record-resolved-path",
                f"{source}.compute_path_policy",
            ),
            artifact_manifest_status=one_of(
                required_string(data, "artifact_manifest_status", source),
                ARTIFACT_MANIFEST_STATUSES,
                f"{source}.artifact_manifest_status",
            ),
            deployment=deployment,
            device_evidence=tuple(evidence),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCatalog:
    """Store one closed pilot catalog and its source identity."""

    kind: CatalogKind
    status: CatalogStatus
    as_of: str
    models: tuple[ModelCandidate, ...]
    file_digest: str
    byte_size: int
    version: int = MODEL_CATALOG_VERSION

    def __post_init__(self) -> None:
        """Require the exact pilot envelope and unique ordered models."""
        if self.version != MODEL_CATALOG_VERSION:
            raise ValueError(f"catalog version must be {MODEL_CATALOG_VERSION}")
        if self.kind != MODEL_CATALOG_KIND:
            raise ValueError(f"catalog kind must be {MODEL_CATALOG_KIND}")
        if self.status != MODEL_CATALOG_STATUS:
            raise ValueError(f"catalog status must be {MODEL_CATALOG_STATUS}")
        try:
            parsed_date = date.fromisoformat(self.as_of)
        except ValueError as error:
            raise ValueError("catalog as_of must be an ISO date") from error
        if parsed_date.isoformat() != self.as_of:
            raise ValueError("catalog as_of must be an ISO date")
        if len(self.models) != EXPECTED_MODEL_COUNT:
            raise ValueError(f"catalog must contain exactly {EXPECTED_MODEL_COUNT} models")
        profile_ids = tuple(model.profile_id for model in self.models)
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("catalog profile_id values must be unique")
        aliases = tuple(model.deployment.model_alias for model in self.models if model.deployment is not None)
        if len(set(aliases)) != len(aliases):
            raise ValueError("catalog model_alias values must be unique")
        if self.byte_size <= 0:
            raise ValueError("catalog byte_size must be positive")

    @property
    def is_bundled_snapshot(self) -> bool:
        """Return whether the bytes match the catalog embedded in this release."""
        return self.file_digest == BUNDLED_MODEL_CATALOG_DIGEST


def load_model_catalog(path: Path) -> ModelCatalog:
    """Load a regular pilot catalog file and preserve model order."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("model catalog must be a regular file, not a symbolic link")
    with path.open("rb") as source:
        encoded = source.read(MAX_MODEL_CATALOG_BYTES + 1)
    if len(encoded) > MAX_MODEL_CATALOG_BYTES:
        raise ValueError(f"model catalog must not exceed {MAX_MODEL_CATALOG_BYTES} bytes")
    data = decode_json_object(encoded, f"invalid model catalog JSON: {path}")
    require_exact_keys(data, _CATALOG_FIELDS, "catalog")
    version = required_integer(data, "version", "catalog")
    if version != MODEL_CATALOG_VERSION:
        raise ValueError(f"catalog.version must be {MODEL_CATALOG_VERSION}")
    kind = _literal(
        required_string(data, "kind", "catalog"),
        "consumer_gpu_model_candidates",
        "catalog.kind",
    )
    status = _literal(
        required_string(data, "status", "catalog"),
        "pilot-candidates-not-release",
        "catalog.status",
    )
    raw_models = data.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("catalog.models must be an array")
    models: list[ModelCandidate] = []
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ValueError(f"catalog.models[{index}] must be an object")
        models.append(ModelCandidate.from_json(raw_model, f"catalog.models[{index}]"))
    return ModelCatalog(
        kind=kind,
        status=status,
        as_of=required_string(data, "as_of", "catalog"),
        models=tuple(models),
        file_digest=sha256_bytes(encoded),
        byte_size=len(encoded),
        version=version,
    )
