"""Load the recipe folder into typed catalog records."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, ValidationInfo, field_validator, model_validator

from agentperf_local.common.durable_files import read_bounded_file
from agentperf_local.common.identity import sha256_bytes, validate_identifier
from agentperf_local.common.json_types import JsonObject, normalize_json_object
from agentperf_local.common.models import read_object
from agentperf_local.common.package_paths import PACKAGE_DATA_ROOT, PACKAGE_ROOT
from agentperf_local.provenance.benchmark import BENCHMARK_CONTEXT_TOKENS

MAX_DISPLAY_NAME_CHARACTERS = 160
MAX_RECIPE_BYTES = 262_144
MAX_RECIPES = 512
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
# Recipes live at recipes/<model>/<hardware>/<profile_id>.yaml.
# libyaml's loader parses the recipes about four times faster; the pure-Python one is the fallback.
_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
RECIPE_SUFFIX = ".yaml"
# A recipe path has three parts: <model>/<hardware>/<profile_id>.yaml.
RECIPE_PATH_PARTS = 3
RECIPES_README = "README.md"
# A wheel carries a copy of the repository's recipes/ folder inside the package. A
# source checkout has no copy, so it reads the folder at the repository root.
_PACKAGED_RECIPES_ROOT = PACKAGE_DATA_ROOT / "recipes"
BUNDLED_RECIPES_ROOT = _PACKAGED_RECIPES_ROOT if _PACKAGED_RECIPES_ROOT.is_dir() else PACKAGE_ROOT.parent / "recipes"
BUNDLED_RECIPES_DIGEST = "sha256:3814f36b629873beb148e8a5fb3c385c4d2fcee4c174bc3adafb428ae07b86b2"

type ToolCallParser = Literal["gemma4", "glm45", "gpt-oss", "qwen3_coder", "qwen3_xml"]
type ReasoningParser = Literal["gemma4", "gpt-oss", "nemotron_v3", "qwen3"]
type ThinkingPolicy = Literal["disabled", "enabled", "enabled-medium-candidate"]
type SpeculationPolicy = Literal[
    "disabled-target-only-baseline",
    "enabled-mtp-self-draft",
    "enabled-mtp-external-draft",
    "enabled-dflash-external-draft",
    "enabled-dflash-draft",
    "enabled-dspark-external-draft",
    "enabled-dspark-draft",
    "enabled-vllm-external-draft",
]
type DeviceId = Literal["nvidia-cuda", "amd-rocm", "apple-silicon"]
type DeploymentFramework = Literal["llama-cpp", "sglang", "vllm"]
type ArtifactKind = Literal["gguf-single-file", "gguf-file-set", "safetensors-repository"]
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
DEVICE_IDS: tuple[DeviceId, ...] = ("nvidia-cuda", "amd-rocm", "apple-silicon")
MOE_RUNNER_BACKENDS: tuple[MoeRunnerBackend, ...] = ("flashinfer_cutlass",)
REASONING_PARSERS: tuple[ReasoningParser, ...] = ("gemma4", "gpt-oss", "nemotron_v3", "qwen3")
THINKING_POLICIES: tuple[ThinkingPolicy, ...] = ("disabled", "enabled", "enabled-medium-candidate")
SPECULATION_POLICIES: tuple[SpeculationPolicy, ...] = (
    "disabled-target-only-baseline",
    "enabled-mtp-self-draft",
    "enabled-mtp-external-draft",
    "enabled-dflash-external-draft",
    "enabled-dflash-draft",
    "enabled-dspark-external-draft",
    "enabled-dspark-draft",
    "enabled-vllm-external-draft",
)
TOOL_CALL_PARSERS: tuple[ToolCallParser, ...] = ("gemma4", "glm45", "gpt-oss", "qwen3_coder", "qwen3_xml")
ARTIFACT_KINDS: tuple[ArtifactKind, ...] = ("gguf-single-file", "gguf-file-set", "safetensors-repository")
# A weights repository is only servable when the runtime can read the model shape
# and the tokenizer beside the tensors.
REQUIRED_REPOSITORY_FILES: tuple[str, ...] = ("config.json", "tokenizer.json")


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


def _pairs_without_object(value: object, info: ValidationInfo, field: str) -> object:
    """Handle a pairs field whose input is not a JSON object.

    A recipe writes the field as an object, or leaves it out or null for no pairs.
    Any other JSON value is an error. A Python caller passes the pairs itself.
    """
    if value is None:
        return ()
    if info.mode == "json":
        raise ValueError(f"{field} must be an object or null")
    return value


def _framework_rank(name: object) -> int:
    """Return a framework's canonical position; an unknown name sorts last and fails validation."""
    for rank, framework in enumerate(DEPLOYMENT_FRAMEWORK_ORDER):
        if name == framework:
            return rank
    return len(DEPLOYMENT_FRAMEWORK_ORDER)


def validate_artifact_path(value: str, field: str) -> str:
    """Require one relative repository path that can never escape the model cache."""
    if not value or len(value) > MAX_ARTIFACT_PATH_CHARACTERS:
        raise ValueError(f"{field} must be a short repository path")
    segments = value.split("/")
    if any(ARTIFACT_PATH_SEGMENT_PATTERN.fullmatch(segment) is None for segment in segments):
        raise ValueError(f"{field} must be a relative path of plain name segments")
    return value


class DeploymentArtifact(BaseModel, frozen=True):
    """Pin one file of a managed deployment by path, digest, and size."""

    filename: str
    sha256: str
    size_bytes: NonNegativeInt
    source_repository: str | None = None
    source_revision: str | None = None

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require a relative path with an exact digest and a non-negative size."""
        validate_artifact_path(self.filename, "artifact.filename")
        _sha256(self.sha256, "artifact.sha256")
        if (self.source_repository is None) != (self.source_revision is None):
            raise ValueError("artifact source_repository and source_revision must be set together")
        if self.source_repository is not None:
            if REPOSITORY_PATTERN.fullmatch(self.source_repository) is None:
                raise ValueError("artifact.source_repository must contain one owner and repository name")
            if self.source_revision is None:
                raise ValueError("artifact.source_revision must be set with source_repository")
            _revision(self.source_revision, "artifact.source_revision")
        return self


class LlamaCppLaunch(BaseModel, frozen=True):
    """Pin llama.cpp loading, batching, and speculative decoding."""

    batch_size: int
    ubatch_size: int
    # A target-only recipe drafts nothing, so it leaves the draft depth out.
    speculative_tokens: PositiveInt | None = None
    draft_model_filename: str | None = None
    target_backend_sampling: bool
    draft_backend_sampling: bool
    backend: LlamaCppBackend | None = None
    load_mode: LlamaCppLoadMode | None = None
    lazy_mode: LlamaCppLazyMode | None = None
    threads: PositiveInt | None = None
    flash_attention: bool = False
    disable_fit: bool = False
    cache_ram_mib: int | None = None
    # How many recurrent-state checkpoints llama.cpp keeps per slot. A checkpoint also
    # copies an external draft's KV cache, so a draft recipe caps them to bound memory.
    context_checkpoints: PositiveInt | None = None

    @field_validator("flash_attention", "disable_fit", mode="before")
    @classmethod
    def _null_is_off(cls, value: object) -> object:
        """Read a null switch as off, as a recipe may write it."""
        return False if value is None else value

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require usable batch sizes."""
        if min(self.batch_size, self.ubatch_size) <= 0:
            raise ValueError("llama.cpp batch sizes must be positive")
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
        if self.cache_ram_mib is not None and self.cache_ram_mib < 0:
            raise ValueError("llama.cpp cache_ram_mib must be non-negative")
        return self

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


class VllmLaunch(BaseModel, frozen=True):
    """Pin extra vLLM arguments and environment variables for one recipe."""

    arguments: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()

    @field_validator("environment", mode="before")
    @classmethod
    def _environment_pairs(cls, value: object, info: ValidationInfo) -> object:
        """Read the recipe's name-to-value object as pairs sorted by name."""
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        return _pairs_without_object(value, info, "deployment.vllm.environment")

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require stable arguments and a unique sorted environment."""
        if not self.arguments or any(not argument or not argument.isprintable() for argument in self.arguments):
            raise ValueError("vLLM arguments must be non-empty printable text")
        names = tuple(name for name, _ in self.environment)
        if len(set(names)) != len(names) or names != tuple(sorted(names)):
            raise ValueError("vLLM environment names must be unique and sorted")
        for name, value in self.environment:
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None or not value or not value.isprintable():
                raise ValueError("vLLM environment must use uppercase names and printable values")
        return self


class DeploymentMemory(BaseModel, frozen=True):
    """Describe how one model's resident memory grows with the served context.

    A recipe stores the attention shape rather than a single opaque number, so the
    memory floor can be computed for any context.
    """

    # Full and sliding layers can hold different head shapes: a Gemma 4 global layer
    # keeps one or two wide heads while its sliding layers keep eight narrow ones, so
    # one shape for both classes would misprice the cache by several gigabytes.
    full_attention_layers: NonNegativeInt
    full_kv_heads: NonNegativeInt
    full_kv_head_dimension: NonNegativeInt
    sliding_attention_layers: NonNegativeInt
    sliding_kv_heads: NonNegativeInt
    sliding_kv_head_dimension: NonNegativeInt
    # Tokens each sliding-attention layer keeps. This is the runtime's choice, not
    # only the model's window: llama.cpp pads the window, while SGLang sizes a share
    # of the token pool, so the pinned value is read from the runtime.
    sliding_cached_tokens: NonNegativeInt
    kv_bytes_per_scalar: PositiveInt
    recurrent_state_slots: NonNegativeInt
    constant_state_bytes: NonNegativeInt
    runtime_overhead_bytes: int
    # Artifact bytes the runtime reads from disk on demand and never holds resident,
    # such as a per-layer-embedding table served by llama.cpp lazy reads. The memory
    # floor leaves them out.
    lazy_read_bytes: NonNegativeInt = 0

    @field_validator("lazy_read_bytes", mode="before")
    @classmethod
    def _null_is_zero(cls, value: object) -> object:
        """Read a null lazy byte count as zero, as a recipe may write it."""
        return 0 if value is None else value

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require an attention shape that can hold a cache."""
        if self.full_attention_layers < 0 or self.sliding_attention_layers < 0:
            raise ValueError("attention layer counts must not be negative")
        if self.full_attention_layers + self.sliding_attention_layers <= 0:
            raise ValueError("a managed recipe must have at least one attention layer")
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
        return self


class ModelDeployment(BaseModel, frozen=True):
    """Describe one exact managed model artifact set and its runtimes."""

    artifact_kind: ArtifactKind
    artifacts: tuple[DeploymentArtifact, ...]
    context_tokens: int
    frameworks: tuple[DeploymentFramework, ...]
    memory: DeploymentMemory
    # The release each framework is verified against, keyed by framework in canonical
    # order. A runtime can load a checkpoint, pass every startup check, and still emit
    # nonsense, and nothing downstream reads generated text, so the launcher serves the
    # named version and refuses every other. A framework with no comparable release
    # (llama.cpp) is simply absent from the map.
    runtime_versions: tuple[tuple[DeploymentFramework, str], ...] = ()
    # The fused-expert kernel this recipe is served with, or None when it has no
    # mixture-of-experts layers or the runtime's own choice is known to serve it.
    moe_runner_backend: MoeRunnerBackend | None = None
    model_filename: str | None = None
    llama_cpp: LlamaCppLaunch | None = None
    vllm: VllmLaunch | None = None

    @field_validator("runtime_versions", mode="before")
    @classmethod
    def _runtime_version_pairs(cls, value: object, info: ValidationInfo) -> object:
        """Read the recipe's framework-to-version object as pairs in canonical framework order."""
        if isinstance(value, dict):
            return tuple(sorted(value.items(), key=lambda pair: _framework_rank(pair[0])))
        return _pairs_without_object(value, info, "deployment.runtime_versions")

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
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
        if self.context_tokens != BENCHMARK_CONTEXT_TOKENS:
            raise ValueError(f"context_tokens must be {BENCHMARK_CONTEXT_TOKENS}")
        if self.memory.lazy_read_bytes > self.artifact_size_bytes:
            raise ValueError("lazy_read_bytes must not exceed the model artifacts")
        if self.memory.lazy_read_bytes and (self.llama_cpp is None or self.llama_cpp.lazy_mode is None):
            raise ValueError("only a llama.cpp recipe with a lazy_mode can read artifact bytes on demand")
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
        return self

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


class ModelCandidate(BaseModel, frozen=True):
    """Describe one recipe: what to download, where it runs, and how to launch it."""

    profile_id: str
    as_of: str
    display_name: str
    hf_repository: str
    hf_revision: str
    # The accelerator platforms this recipe may launch on, in canonical order.
    devices: tuple[DeviceId, ...]
    tool_call_parser: ToolCallParser
    reasoning_parser: ReasoningParser
    thinking_policy: ThinkingPolicy
    speculation_policy: SpeculationPolicy
    deployment: ModelDeployment

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require a portable identity and launch settings that agree with each other."""
        validate_identifier(self.profile_id, "profile_id")
        _iso_date(self.as_of, "as_of")
        if not self.display_name or len(self.display_name) > MAX_DISPLAY_NAME_CHARACTERS:
            raise ValueError("display_name must be short printable text")
        if not self.display_name.isprintable():
            raise ValueError("display_name must be short printable text")
        if REPOSITORY_PATTERN.fullmatch(self.hf_repository) is None:
            raise ValueError("hf_repository must contain one owner and repository name")
        _revision(self.hf_revision, "hf_revision")
        expected_device_order = tuple(device_id for device_id in DEVICE_IDS if device_id in self.devices)
        if not self.devices or len(set(self.devices)) != len(self.devices) or self.devices != expected_device_order:
            raise ValueError("devices must be a canonical subset of nvidia-cuda, amd-rocm, and apple-silicon")
        llama_cpp = self.deployment.llama_cpp
        if llama_cpp is not None and llama_cpp.backend is not None:
            expected_devices = ("apple-silicon",) if llama_cpp.backend == "metal" else ("amd-rocm",)
            if self.devices != expected_devices:
                raise ValueError("llama.cpp backend must match the recipe's devices")
        if self.speculation_policy in (
            "enabled-mtp-external-draft",
            "enabled-dflash-external-draft",
            "enabled-dspark-external-draft",
        ):
            if llama_cpp is None or llama_cpp.draft_model_filename is None:
                raise ValueError("an external-draft policy requires a pinned llama.cpp draft model")
        if llama_cpp is not None:
            drafts = self.speculation_policy != "disabled-target-only-baseline"
            if drafts and llama_cpp.speculative_tokens is None:
                raise ValueError("a speculative llama.cpp recipe must set speculative_tokens")
            if not drafts and llama_cpp.speculative_tokens is not None:
                raise ValueError("a target-only llama.cpp recipe must not set speculative_tokens")
        if self.speculation_policy == "enabled-mtp-self-draft" and llama_cpp is not None:
            if llama_cpp.draft_model_filename is not None:
                raise ValueError("an MTP self-draft policy must not name an external draft model")
        return self


class RecipeSource(BaseModel, frozen=True):
    """Hold one recipe file's exact text, its path under the recipe folder, and its digest.

    The path and digest form one line of the catalog digest listing, so a reader who
    has the catalog digest can check that this text is one of its recipes.
    """

    path: str
    sha256: str
    text: str

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require a <model>/<hardware>/<profile_id>.yaml path and a digest of the exact text."""
        parts = self.path.split("/")
        if len(parts) != RECIPE_PATH_PARTS or not parts[-1].endswith(RECIPE_SUFFIX):
            raise ValueError("recipe path must be <model>/<hardware>/<profile_id>.yaml")
        for part in (*parts[:-1], self.profile_id):
            validate_identifier(part, "recipe path")
        if len(self.text.encode()) > MAX_RECIPE_BYTES:
            raise ValueError(f"recipe text must not exceed {MAX_RECIPE_BYTES} bytes")
        if self.sha256 != sha256_bytes(self.text.encode()):
            raise ValueError("recipe sha256 does not match the recipe text")
        return self

    @property
    def profile_id(self) -> str:
        """Return the profile_id the file name carries."""
        return self.path.rsplit("/", 1)[-1].removesuffix(RECIPE_SUFFIX)

    def to_json(self) -> JsonObject:
        """Return the recipe file as a JSON object."""
        return {"path": self.path, "sha256": self.sha256, "text": self.text}


class ModelCatalog(BaseModel, frozen=True):
    """Store every recipe of one folder, the file each came from, and the folder's identity."""

    models: Annotated[tuple[ModelCandidate, ...], Field(min_length=1)]
    # One source per model, in the same order.
    sources: tuple[RecipeSource, ...]
    digest: str

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require at least one recipe, unique names, and one matching source per recipe."""
        profile_ids = tuple(model.profile_id for model in self.models)
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("recipe profile_id values must be unique")
        if tuple(source.profile_id for source in self.sources) != profile_ids:
            raise ValueError("every recipe needs the one source file it was read from")
        return self

    def source(self, profile_id: str) -> RecipeSource:
        """Return the file one recipe was read from."""
        for source in self.sources:
            if source.profile_id == profile_id:
                return source
        raise ValueError(f"unknown recipe profile_id: {profile_id}")

    @property
    def as_of(self) -> str:
        """Return the newest recipe date, which names this catalog's epoch."""
        return max(model.as_of for model in self.models)

    @property
    def is_bundled_snapshot(self) -> bool:
        """Return whether the recipes match the ones shipped with this release."""
        return self.digest == BUNDLED_RECIPES_DIGEST


def _iso_date(value: str, field: str) -> None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must be an ISO date")


def _parse_recipe(encoded: bytes, name: str) -> JsonObject:
    try:
        decoded = yaml.load(encoded, Loader=_YAML_SAFE_LOADER)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid recipe YAML: {name}") from error
    try:
        return normalize_json_object(decoded)
    except ValueError as error:
        raise ValueError(f"recipe {name} must be a mapping of JSON values; quote dates") from error


def _subfolders(folder: Path, skip: str | None = None) -> tuple[Path, ...]:
    """Return the named subfolders of one recipe folder level, in name order, ignoring one named file."""
    folders: list[Path] = []
    # Sort on the name: WindowsPath ignores case when it sorts, and the digest needs byte order.
    for path in sorted(folder.iterdir(), key=lambda entry: entry.name):
        if path.name == skip:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"recipe folder {folder.name} may only hold folders, but holds {path.name}")
        validate_identifier(path.name, f"recipe folder {path.name}")
        folders.append(path)
    return tuple(folders)


def _recipe_paths(root: Path) -> tuple[Path, ...]:
    """Return every recipe file under root/<model>/<hardware>/, in path order."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("recipe folder must be a directory, not a symbolic link")
    paths: list[Path] = []
    for model_folder in _subfolders(root, skip=RECIPES_README):
        for hardware_folder in _subfolders(model_folder):
            for path in sorted(hardware_folder.iterdir(), key=lambda entry: entry.name):
                if path.suffix != RECIPE_SUFFIX:
                    raise ValueError(f"recipe folder holds {path.name}; recipes must be {RECIPE_SUFFIX} files")
                paths.append(path)
    if len(paths) > MAX_RECIPES:
        raise ValueError(f"recipe folder must not hold more than {MAX_RECIPES} recipes")
    return tuple(paths)


def load_model_catalog(root: Path) -> ModelCatalog:
    """Load every recipe under root/<model>/<hardware>/<profile_id>.yaml, in path order.

    The digest is the SHA-256 of the `sha256sum */*/*.yaml` listing, so running
    `LC_ALL=C sha256sum */*/*.yaml | sha256sum` in the folder reproduces it.
    """
    models: list[ModelCandidate] = []
    sources: list[RecipeSource] = []
    listing = bytearray()
    for path in _recipe_paths(root):
        relative = path.relative_to(root).as_posix()
        encoded = read_bounded_file(path, MAX_RECIPE_BYTES, label="recipe")
        model = read_object(ModelCandidate, _parse_recipe(encoded, relative), relative)
        if model.profile_id != path.stem:
            raise ValueError(f"recipe {relative} must be named after its profile_id {model.profile_id}")
        try:
            text = encoded.decode()
        except UnicodeDecodeError as error:
            raise ValueError(f"recipe {relative} must be UTF-8 text") from error
        models.append(model)
        sources.append(RecipeSource(path=relative, sha256=sha256_bytes(encoded), text=text))
        listing += f"{hashlib.sha256(encoded).hexdigest()}  {relative}\n".encode()
    return ModelCatalog(models=tuple(models), sources=tuple(sources), digest=sha256_bytes(bytes(listing)))
