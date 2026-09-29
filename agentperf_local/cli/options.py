"""Read and validate one parsed command line, one value at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson
from pydantic import BaseModel, SecretStr

from agentperf_local.client.backends import CLIENT_BACKENDS, ClientBackend
from agentperf_local.common.argparse_fields import (
    read_optional_integer,
    read_optional_number,
    read_optional_path,
    read_optional_string,
    read_path,
    read_string,
)
from agentperf_local.common.json_fields import one_of
from agentperf_local.common.json_types import JsonObject
from agentperf_local.deployment.catalog import (
    BUNDLED_RECIPES_ROOT,
    DEPLOYMENT_FRAMEWORK_ORDER,
    DeploymentFramework,
    ModelCandidate,
    ModelCatalog,
    load_model_catalog,
)
from agentperf_local.deployment.managed import (
    DEPLOYMENT_LOG_FILENAME,
    DEPLOYMENT_RECORD_FILENAME,
    BoundDeploymentDevice,
    bind_snapshot_to_device,
)
from agentperf_local.deployment.qualification import (
    QUALIFICATION_FILENAME,
)
from agentperf_local.provenance.benchmark import (
    MEASUREMENT_BINDING_FILENAME,
)
from agentperf_local.provenance.hardware import (
    collect_hardware_snapshot,
)
from agentperf_local.replay.config import (
    API_KEY_ENV_PATTERN,
    DEFAULT_OUTPUT_TOKEN_MARGIN,
    DEFAULT_TOOL_DELAY_SCALE,
    OUTPUT_TOKEN_POLICIES,
    TOOL_CHOICES,
    OutputTokenPolicy,
    SamplingPreset,
    ToolChoice,
    ToolReplayMode,
    read_api_key_env,
)
from agentperf_local.reports.reporting import (
    FAILURES_FILENAME,
    SUMMARY_FILENAME,
    TASKS_FILENAME,
    TOOLS_FILENAME,
    TURNS_FILENAME,
    validate_run_artifact_output,
)
from agentperf_local.submission.client import (
    read_submit_token,
)
from agentperf_local.telemetry.power import (
    POWER_SUMMARY_FILENAME,
    TELEMETRY_FILENAME,
)
from agentperf_local.workload.bundled import find_bundled_replay
from agentperf_local.workload.schema import MessageSource

MESSAGE_SOURCES: tuple[MessageSource, ...] = ("provider-request", "request-messages")


SAMPLING_PRESETS: tuple[SamplingPreset, ...] = ("standard", "custom")


TOOL_REPLAY_MODES: tuple[ToolReplayMode, ...] = ("none", "recorded", "live")


DEFAULT_MANAGED_PROFILE_ID = "gemma4-12b-it-q4-0"


BOUND_RESULT_FILENAMES = (
    MEASUREMENT_BINDING_FILENAME,
    SUMMARY_FILENAME,
    TURNS_FILENAME,
    TASKS_FILENAME,
    TOOLS_FILENAME,
    FAILURES_FILENAME,
    DEPLOYMENT_RECORD_FILENAME,
    TELEMETRY_FILENAME,
    POWER_SUMMARY_FILENAME,
    QUALIFICATION_FILENAME,
    DEPLOYMENT_LOG_FILENAME,
)


DEFAULT_RECIPES_ROOT = BUNDLED_RECIPES_ROOT


# Shells report an interrupted process as 128 plus the signal number, and SIGINT is signal 2.
INTERRUPTED_STATUS = 130


def nonempty_path(value: str) -> Path:
    """Parse one path argument, refusing the empty value.

    An empty string becomes the current directory, which silently scatters new files
    into the working tree; argparse prefixes the offending argument's name itself.
    """
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return Path(value)


def read_api_key(namespace: argparse.Namespace) -> SecretStr | None:
    """Read one explicitly named endpoint secret."""
    name = read_optional_string(namespace, "api_key_env")
    if name is None:
        return None
    if API_KEY_ENV_PATTERN.fullmatch(name) is None:
        raise ValueError("--api-key-env must name a portable environment variable")
    value = read_api_key_env(name)
    if value is None:
        raise ValueError(f"API key environment variable {name} is not set, is empty, or contains whitespace")
    return value


def read_message_source(namespace: argparse.Namespace) -> MessageSource:
    return one_of(read_string(namespace, "message_source"), MESSAGE_SOURCES, "--message-source")


def read_client_backend(namespace: argparse.Namespace) -> ClientBackend:
    return one_of(read_string(namespace, "client"), CLIENT_BACKENDS, "--client")


def read_tool_choice(namespace: argparse.Namespace) -> ToolChoice | None:
    """Return the tool_choice the user passed, or None to leave the server default."""
    tool_choice = read_optional_string(namespace, "tool_choice")
    return None if tool_choice is None else one_of(tool_choice, TOOL_CHOICES, "--tool-choice")


def read_requested_output_token_policy(namespace: argparse.Namespace) -> OutputTokenPolicy | None:
    """Return the policy the user passed, or None when they left the choice to the command."""
    policy = read_optional_string(namespace, "output_token_policy")
    return None if policy is None else one_of(policy, OUTPUT_TOKEN_POLICIES, "--output-token-policy")


# A managed run has no fixed-cap flag, so it offers the two recorded-length policies.
MANAGED_OUTPUT_TOKEN_POLICIES: tuple[OutputTokenPolicy, ...] = ("exact", "recorded")


def read_managed_output_token_policy(namespace: argparse.Namespace) -> OutputTokenPolicy:
    return one_of(read_string(namespace, "output_token_policy"), MANAGED_OUTPUT_TOKEN_POLICIES, "--output-token-policy")


def read_replay_manifest_path(namespace: argparse.Namespace) -> Path:
    """Resolve the custom manifest, or the bundled replay's manifest for this host."""
    manifest = read_optional_path(namespace, "manifest")
    if manifest is not None:
        return manifest
    replay_id = read_string(namespace, "replay")
    replay = find_bundled_replay(replay_id)
    if replay is None:
        raise ValueError(f"unknown bundled replay {replay_id}")
    return replay.manifest_path


def read_sampling_preset(namespace: argparse.Namespace) -> SamplingPreset:
    return one_of(read_string(namespace, "sampling"), SAMPLING_PRESETS, "--sampling")


def read_tool_mode(namespace: argparse.Namespace) -> ToolReplayMode:
    return one_of(read_string(namespace, "tool_mode"), TOOL_REPLAY_MODES, "--tool-mode")


def read_output_token_margin(namespace: argparse.Namespace, policy: OutputTokenPolicy) -> int:
    """Resolve the recorded-target margin and reject it under any other policy."""
    margin = read_optional_integer(namespace, "output_token_margin")
    if margin is None:
        return DEFAULT_OUTPUT_TOKEN_MARGIN
    if policy != "recorded":
        raise ValueError("output token margin applies only to the recorded policy")
    return margin


def read_tool_delay_scale(namespace: argparse.Namespace, tool_mode: ToolReplayMode) -> float:
    """Resolve the recorded delay scale and reject it without tool replay."""
    scale = read_optional_number(namespace, "tool_delay_scale")
    if scale is None:
        return DEFAULT_TOOL_DELAY_SCALE
    if tool_mode == "none":
        raise ValueError("tool delay scale requires a tool replay mode")
    return scale


def read_live_workspace_root(namespace: argparse.Namespace, tool_mode: ToolReplayMode) -> Path | None:
    """Resolve the live workspace root, defaulting live runs to a folder inside the output directory."""
    root = read_optional_path(namespace, "live_workspace_root")
    if root is not None or tool_mode != "live":
        return root
    return read_path(namespace, "output_dir") / "workspaces"


def read_deployment_framework(namespace: argparse.Namespace) -> DeploymentFramework:
    return one_of(read_string(namespace, "framework"), DEPLOYMENT_FRAMEWORK_ORDER, "--framework")


class ManagedTarget(BaseModel, frozen=True):
    """Name the catalog and candidate one managed command targets."""

    catalog: ModelCatalog
    candidate: ModelCandidate


def read_managed_target(namespace: argparse.Namespace) -> ManagedTarget:
    catalog = load_model_catalog(read_path(namespace, "recipes"))
    candidate = catalog_candidate(catalog, read_string(namespace, "profile_id"))
    return ManagedTarget(catalog=catalog, candidate=candidate)


def read_bound_device(namespace: argparse.Namespace) -> BoundDeploymentDevice:
    return bind_snapshot_to_device(collect_hardware_snapshot(), read_optional_integer(namespace, "device"))


def require_fresh_output_dir(output_dir: Path, label: str) -> None:
    """Refuse an output directory that already holds any bound result artifact."""
    validate_run_artifact_output(output_dir)
    existing = tuple(
        name for name in BOUND_RESULT_FILENAMES if (output_dir / name).exists() or (output_dir / name).is_symlink()
    )
    if existing:
        raise ValueError(f"{label} output must be fresh; existing artifacts: {', '.join(existing)}")


def catalog_candidate(catalog: ModelCatalog, profile_id: str) -> ModelCandidate:
    candidate = next((model for model in catalog.models if model.profile_id == profile_id), None)
    if candidate is None:
        available = ", ".join(sorted(model.profile_id for model in catalog.models))
        raise ValueError(f"model catalog does not contain profile {profile_id}; available: {available}")
    return candidate


def print_json(value: JsonObject) -> None:
    print(orjson.dumps(value, option=orjson.OPT_INDENT_2).decode("utf-8"))


def resolve_submit_token(namespace: argparse.Namespace, attribute: str = "token_env") -> str | None:
    """Read the submit token from the named variable; an unset variable means anonymous."""
    return read_submit_token(read_string(namespace, attribute))
