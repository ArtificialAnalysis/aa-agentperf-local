"""Word one measured fact for the screen.

Every function here turns data into display text and nothing else, so the app
module holds no formatting rules and the wording can be reviewed in one place.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from rich.markup import escape

from agentperf_local.common.units import BYTES_PER_GIB, MILLISECONDS_PER_SECOND
from agentperf_local.deployment.catalog import ArtifactKind, DeploymentFramework
from agentperf_local.deployment.endpoint_probes import (
    ContextProbeResult,
    IgnoreEosProbeResult,
    IgnoreEosSupport,
)
from agentperf_local.deployment.frameworks import framework_display_name
from agentperf_local.provenance.context import ContextObservationReason
from agentperf_local.provenance.hardware import AcceleratorPlatform
from agentperf_local.replay.runner import TurnCompletedBoundary
from agentperf_local.reports.progress import RunTurnSample, rate_text
from agentperf_local.tui.replay_contract import PreflightBlockCode
from agentperf_local.tui.widgets import DONE_MARK, FAILED_MARK, WARNING_MARK

RUN_SECOND_CHART_MIN_HEIGHT = 27
RECORDED_POLICY_OUTCOME = "recorded policy · e2e reported as a normalized estimate, not comparable to exact runs"
PLATFORM_DISPLAY_NAMES: dict[AcceleratorPlatform, str] = {
    "nvidia-cuda": "CUDA",
    "amd-rocm": "ROCm",
    "apple-metal": "Metal",
}
# Textual wraps text only at whitespace and folds any overlong word mid-token. U+001F is
# zero cells wide yet counts as whitespace to the wrapper, so a path carrying it after
# every "/" wraps only at directory boundaries and never mid-name.
PATH_WRAP_BREAK = "\x1f"
BLOCK_CODE_MESSAGES = {
    PreflightBlockCode.ENDPOINT_NEEDS_HTTPS: (
        "[b]This server is not local and an API key is set, so HTTPS is required.[/b]\n"
        "Use an https:// URL, or clear the API key variable."
    ),
    PreflightBlockCode.URL_INVALID: (
        "[b]The server URL is not valid.[/b]\nUse a plain address like http://127.0.0.1:8000/v1 with no credentials."
    ),
    PreflightBlockCode.ENDPOINT_MODEL_EMPTY: "[b]Enter the model name your server uses.[/b]",
    PreflightBlockCode.OUTPUT_DIR_MISSING: "[b]Enter a results folder.[/b]",
    PreflightBlockCode.OUTPUT_DIR_USED: (
        "[b]The new run folder already holds files.[/b]\nGo back and continue again to get a fresh run folder."
    ),
    PreflightBlockCode.MANIFEST_UNREADABLE: "[b]The replay file could not be read.[/b]\nCheck the path.",
    PreflightBlockCode.CLIENT_UNAVAILABLE: (
        "The Rust client is not installed. Go back and set Client to Python, or run `uv sync --extra rust`."
    ),
    PreflightBlockCode.DEPLOYMENT_UNAVAILABLE: (
        "[b]Can't start this model on this computer.[/b]\nGo back and read the message under Framework."
    ),
}


def count(quantity: int, noun: str) -> str:
    """Count one noun with a plain plural s."""
    return f"{quantity} {noun}" if quantity == 1 else f"{quantity} {noun}s"


def result_path_text(path: Path) -> str:
    """Render one result path: home shortens to ~, and wrapping breaks only after a separator."""
    return home_relative_path_text(path.absolute()).replace(os.sep, f"{os.sep}{PATH_WRAP_BREAK}")


def home_relative_path_text(path: Path) -> str:
    """Name a cache location without spelling out the home directory.

    Windows keeps the full path: neither cmd nor PowerShell expands ~ for a native
    command, so a shortened path would break the commands the app tells users to run.
    """
    home = Path.home()
    if sys.platform == "win32" or not path.is_relative_to(home):
        return escape(str(path))
    # The home directory itself is relative to itself as ".", which joins back to a plain "~".
    return escape(str(Path("~") / path.relative_to(home)))


def block_code_message(block_code: PreflightBlockCode | None, api_key_env: str | None) -> str | None:
    """Explain one blocked setup cause, or return None when only a generic message fits."""
    if block_code is PreflightBlockCode.API_KEY_ENV_UNSET and api_key_env is not None:
        # The name already matched the environment variable pattern, so it carries no markup.
        return f"[b]The environment variable {api_key_env} is not set.[/b]\nSet it before starting, or clear the field."
    if block_code is None:
        return None
    return BLOCK_CODE_MESSAGES.get(block_code)


def metric_text(value: float | None, spec: str) -> str:
    """Format one optional metric, or an em dash when it is missing."""
    return "—" if value is None else format(value, spec)


def unit_text(value: float | None, spec: str, unit: str) -> str:
    """Format one optional metric with its unit, or an em dash when it is missing."""
    return "—" if value is None else f"{format(value, spec)} {unit}"


def integer_label(value: float) -> str:
    return f"{value:,.0f}"


def seconds_label(value: float) -> str:
    return f"{value:,.1f}"


def gib_suffix(size_bytes: int | None) -> str:
    """Name a file size in GiB as a separator-led suffix, or nothing when unknown."""
    return "" if size_bytes is None else f" · {size_bytes / BYTES_PER_GIB:.1f} GiB"


def memory_need_gib(size_bytes: float) -> str:
    """Name a memory requirement in GiB, rounded up so the figure never undersells it."""
    return f"{math.ceil(size_bytes / BYTES_PER_GIB * 10) / 10:.1f}"


def artifact_kind_text(artifact_kind: ArtifactKind) -> str:
    """Name what a recipe downloads, so a weights repository is not called a GGUF."""
    return "GGUF" if artifact_kind in ("gguf-single-file", "gguf-file-set") else "weights"


def framework_text(framework: DeploymentFramework | None) -> str:
    return "the model server" if framework is None else framework_display_name(framework)


def platform_suffix(platform: AcceleratorPlatform | None) -> str:
    return "" if platform is None else f" · {PLATFORM_DISPLAY_NAMES[platform]}"


def probe_activity_text(probe: ContextProbeResult | None) -> tuple[str, str]:
    """Word one server check as an outcome glyph and a plain sentence that names no URL."""
    if probe is None or probe.reason is ContextObservationReason.NOT_PROBED:
        return WARNING_MARK, "Server not checked"
    reason = probe.reason
    if reason is ContextObservationReason.REPORTED and probe.observed_tokens is not None:
        return DONE_MARK, f"Server answered · model listed · {probe.observed_tokens:,}-token context"
    if reason in {ContextObservationReason.CONTEXT_NOT_REPORTED, ContextObservationReason.NON_POSITIVE_CONTEXT}:
        return DONE_MARK, "Server answered · model listed · context length not reported"
    if reason is ContextObservationReason.MODEL_NOT_LISTED:
        return WARNING_MARK, "Server answered · it does not list this model name"
    if reason is ContextObservationReason.MALFORMED_RESPONSE:
        return WARNING_MARK, "Server answered · the reply was not a model list"
    if reason is ContextObservationReason.ENDPOINT_UNREACHABLE:
        return FAILED_MARK, "Server did not answer"
    return FAILED_MARK, "Server answered with an HTTP error"


def output_policy_activity_text(probe: IgnoreEosProbeResult | None, *, ollama_endpoint: bool) -> tuple[str, str]:
    """Word the chosen output policy as an outcome glyph and a plain sentence.

    Only a dropped ignore_eos costs the run the exact policy, so that case says what the
    endpoint did and how the run reports end-to-end latency instead.
    """
    support = None if probe is None else probe.support
    if ollama_endpoint:
        return WARNING_MARK, f"Ollama drops ignore_eos · {RECORDED_POLICY_OUTCOME}"
    if support is IgnoreEosSupport.IGNORED:
        return WARNING_MARK, f"Server ignores ignore_eos · {RECORDED_POLICY_OUTCOME}"
    if support is IgnoreEosSupport.UNDETERMINED:
        return WARNING_MARK, "Could not check ignore_eos · keeping the exact policy"
    return DONE_MARK, "Server generates past end-of-sequence · exact policy"


def turn_live_text(turn: int, turns: int) -> str:
    return f"Turn {turn}/{turns} · waiting for the first token"


def turn_record_text(sample: RunTurnSample, event: TurnCompletedBoundary) -> str:
    """Word one closed turn for the activity log from its chart sample."""
    if not event.success:
        return f"Turn {event.turn}/{event.turns} · task {event.task} · failed · details go to failures.json"
    total = "—" if sample.e2e_ms is None else f"{sample.e2e_ms / MILLISECONDS_PER_SECOND:,.1f} s"
    return (
        f"Turn {event.turn}/{event.turns} · task {event.task} · {rate_text(sample.decode_tokens_per_second)} · {total}"
    )
