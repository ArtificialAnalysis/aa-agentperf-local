"""Define every subcommand and its flags."""

from __future__ import annotations

import argparse

from agentperf_local import __version__
from agentperf_local.cli.options import (
    DEFAULT_MANAGED_PROFILE_ID,
    DEFAULT_RECIPES_ROOT,
    MESSAGE_SOURCES,
    OUTPUT_TOKEN_POLICIES,
    SAMPLING_PRESETS,
    TOOL_REPLAY_MODES,
    nonempty_path,
)
from agentperf_local.client.backends import CLIENT_BACKENDS
from agentperf_local.client.request import DEFAULT_MAX_OUTPUT_TOKENS
from agentperf_local.deployment.catalog import (
    DEPLOYMENT_FRAMEWORK_ORDER,
)
from agentperf_local.deployment.managed import (
    DEFAULT_DEPLOYMENT_PORT,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
)
from agentperf_local.deployment.model_cache import (
    MODEL_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
    default_model_cache_root,
)
from agentperf_local.provenance.benchmark import (
    BENCHMARK_CONTEXT_TOKENS,
)
from agentperf_local.replay.cache_isolation import CACHE_NAMESPACE_ENV
from agentperf_local.replay.config import DEFAULT_LIVE_TOOL_TIMEOUT_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS
from agentperf_local.submission.client import (
    SUBMIT_BASE_URL,
    SUBMIT_TOKEN_ENV,
)
from agentperf_local.workload.bundled import BUNDLED_REPLAYS, DEFAULT_BUNDLED_REPLAY

PROGRAM_NAME = "agentperf-local"
VERSION_TEXT = f"{PROGRAM_NAME} {__version__}"
CLIENT_HELP = "measured streaming client; rust is an experimental client for high-concurrency benchmarking"


PROGRAM_DESCRIPTION = (
    "Replay recorded agent tasks against an OpenAI-compatible endpoint. "
    "With no command, open the guided full-screen benchmark app."
)
# Run this command when the user types no command at all.
DEFAULT_COMMAND = "tui"


def _add_api_key_env_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="read the endpoint API key from this environment variable",
    )


def _add_streaming_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--client",
        choices=CLIENT_BACKENDS,
        default="python",
        help=CLIENT_HELP,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="render AA progress frames to stderr after response streams close",
    )


def _add_cache_root_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-root",
        type=nonempty_path,
        default=default_model_cache_root(),
        help="model cache root (default: the Hugging Face hub cache, honoring HF_HUB_CACHE and HF_HOME)",
    )


def _add_managed_target_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recipes",
        type=nonempty_path,
        default=DEFAULT_RECIPES_ROOT,
        help="folder of recipe YAML files (default: the bundled recipes)",
    )
    parser.add_argument("--profile-id", default=DEFAULT_MANAGED_PROFILE_ID, help="managed catalog model profile")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="index of the detected accelerator to use; required when more than one is detected",
    )


def _add_replay_selection(parser: argparse.ArgumentParser) -> None:
    """Pick the workload by bundled identifier, or by a custom manifest path in its place."""
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument(
        "manifest",
        nargs="?",
        type=nonempty_path,
        default=None,
        help="custom converted replay manifest JSON, in place of --replay",
    )
    replay.add_argument(
        "--replay",
        choices=tuple(item.replay_id for item in BUNDLED_REPLAYS),
        default=DEFAULT_BUNDLED_REPLAY.replay_id,
        help=f"bundled replay identifier (default: {DEFAULT_BUNDLED_REPLAY.replay_id})",
    )


def _add_convert_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("convert", help="convert agent recording data into a replay manifest")
    parser.add_argument("input", type=nonempty_path, help="agent recording or manifest JSON")
    parser.add_argument("--output-dir", required=True, type=nonempty_path)
    parser.add_argument("--kind", choices=("recording", "manifest"), default="recording")
    parser.add_argument("--family")
    parser.add_argument("--adapter")
    parser.add_argument("--message-source", choices=MESSAGE_SOURCES, default="provider-request")
    parser.add_argument("--include-tool-outputs", action="store_true")
    parser.add_argument("--manifest-name")


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run", help="replay one workload against a server you run")
    _add_replay_selection(parser)
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible API base before chat/completions; stored in private results",
    )
    parser.add_argument("--model", required=True, help="endpoint model identifier; stored in private results")
    parser.add_argument("--output-dir", required=True, type=nonempty_path, help="new private result directory")
    _add_api_key_env_flag(parser)
    _add_streaming_flags(parser)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="per-request timeout",
    )
    parser.add_argument(
        "--output-token-policy",
        choices=OUTPUT_TOKEN_POLICIES,
        default=None,
        help=(
            "exact generates each turn's recorded length with end-of-sequence ignored (default; "
            "an Ollama server switches the default to recorded); recorded caps at the recorded length; "
            "fixed uses one cap for every turn"
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="fixed limit or upper bound/fallback for recorded targets",
    )
    parser.add_argument(
        "--output-token-margin",
        type=int,
        default=None,
        help="tokens added to a recorded target before applying the upper bound",
    )
    parser.add_argument(
        "--sampling",
        choices=SAMPLING_PRESETS,
        default="standard",
        help="Standard sampling defaults or only explicitly supplied sampling values",
    )
    parser.add_argument("--temperature", type=float, help="override sampling temperature")
    parser.add_argument("--top-p", type=float, help="override nucleus sampling probability")
    parser.add_argument("--top-k", type=int, help="override top-k sampling")
    parser.add_argument("--min-p", type=float, help="override min-p sampling")
    parser.add_argument("--reasoning-effort", help="optional endpoint reasoning-effort label")
    parser.add_argument(
        "--tool-choice",
        choices=("none",),
        help=(
            "send tool_choice none so the server applies no tool-call grammar; llama.cpp can fail "
            "an exact-length request when the grammar ends before the recorded length"
        ),
    )
    parser.add_argument(
        "--cache-isolation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prefix prompts with one private run namespace",
    )
    parser.add_argument(
        "--cache-namespace",
        help=f"explicit private namespace; generated by default (env {CACHE_NAMESPACE_ENV} overrides)",
    )
    parser.add_argument(
        "--tool-mode",
        choices=TOOL_REPLAY_MODES,
        default="none",
        help="omit, sleep for, or execute recorded tool activity",
    )
    parser.add_argument("--tool-delay-scale", type=float, default=None, help="multiply recorded tool delays")
    parser.add_argument("--live-tool-image", help="override the recorded container image in live mode")
    parser.add_argument(
        "--live-workspace-root",
        type=nonempty_path,
        help="private workspace root for live tools; defaults to OUTPUT_DIR/workspaces",
    )
    parser.add_argument("--live-network", help="Docker network mode for live tools")
    parser.add_argument(
        "--live-timeout-seconds",
        type=float,
        default=DEFAULT_LIVE_TOOL_TIMEOUT_SECONDS,
        help="per-command live-tool timeout",
    )
    parser.add_argument("--live-docker-executable", help="Docker-compatible executable for live tools")


def _add_tui_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(DEFAULT_COMMAND, help="open the guided full-screen benchmark app")
    parser.add_argument(
        "--recipes",
        type=nonempty_path,
        default=DEFAULT_RECIPES_ROOT,
        help="folder of recipe YAML files (default: the bundled recipes)",
    )
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument(
        "--replay",
        choices=tuple(item.replay_id for item in BUNDLED_REPLAYS),
        default=DEFAULT_BUNDLED_REPLAY.replay_id,
        help=f"bundled replay identifier (default: {DEFAULT_BUNDLED_REPLAY.replay_id})",
    )
    replay.add_argument(
        "--manifest",
        type=nonempty_path,
        help="replay a custom converted manifest instead of a bundled replay",
    )
    parser.add_argument(
        "--output-dir",
        type=nonempty_path,
        help="prefill the private results folder; each run writes one fresh run-<timestamp> subdirectory",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000/v1",
        help="prefill the API base URL; a non-loopback endpoint with an API key requires HTTPS",
    )
    parser.add_argument("--model", help="prefill the model name your server reports")
    parser.add_argument("--api-key-env", metavar="NAME", help="prefill the API key environment-variable name")
    parser.add_argument(
        "--client",
        choices=CLIENT_BACKENDS,
        default="python",
        help=CLIENT_HELP,
    )
    _add_cache_root_flag(parser)
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_DEPLOYMENT_PORT,
        help="localhost port used when this app starts the model server",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="maximum seconds to wait for a started model server to become ready",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="index of the detected accelerator to use when this app starts the model",
    )
    parser.add_argument("--submit-base-url", default=SUBMIT_BASE_URL, help="submission service base URL")
    parser.add_argument(
        "--submit-token-env",
        metavar="NAME",
        default=SUBMIT_TOKEN_ENV,
        help=f"optional environment variable holding the submit token ({SUBMIT_TOKEN_ENV})",
    )


def _add_doctor_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("doctor", help="inspect identifier-free local hardware facts")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")


def _add_deployment_options_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("deployment-options", help="detect local GPU deployment choices")
    _add_managed_target_flags(parser)


def _add_managed_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("managed-run", help="deploy a canonical local model and benchmark it")
    _add_replay_selection(parser)
    parser.add_argument("--output-dir", required=True, type=nonempty_path, help="new private result directory")
    _add_managed_target_flags(parser)
    parser.add_argument("--framework", required=True, choices=DEPLOYMENT_FRAMEWORK_ORDER)
    _add_cache_root_flag(parser)
    parser.add_argument("--port", type=int, default=DEFAULT_DEPLOYMENT_PORT, help="owned localhost server port")
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="maximum time for the managed endpoint to become ready",
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=float,
        default=MODEL_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
        help="maximum total time for the model download; an interrupted download resumes on the next run",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help=(
            f"serve a smaller context than the {BENCHMARK_CONTEXT_TOKENS}-token benchmark for smaller devices; "
            "reduced-context runs are recorded separately from full-context results"
        ),
    )
    _add_streaming_flags(parser)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="maximum time for each replay model request",
    )
    parser.add_argument(
        "--power",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample GPU power with nvidia-smi in a child process; NVIDIA only, skipped when nvidia-smi is absent",
    )
    parser.add_argument("--submit-base-url", default=SUBMIT_BASE_URL, help="submission service base URL")
    parser.add_argument(
        "--submit-token-env",
        metavar="NAME",
        default=SUBMIT_TOKEN_ENV,
        help="environment variable holding the submit token; when set, the commit allowlist is checked",
    )


def _add_prepare_submission_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("prepare-submission", help="build the four-file submission bundle without uploading")
    parser.add_argument("results_dir", type=nonempty_path, help="bound benchmark results containing measurement.json")
    parser.add_argument("--output-dir", required=True, type=nonempty_path)


def _add_submit_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("submit", help="send one prepared bundle to Artificial Analysis")
    parser.add_argument("bundle_dir", type=nonempty_path, help="directory written by prepare-submission")
    parser.add_argument("--base-url", default=SUBMIT_BASE_URL, help="submission service base URL")
    parser.add_argument(
        "--token-env",
        metavar="NAME",
        default=SUBMIT_TOKEN_ENV,
        help=f"environment variable holding the submit token (default: {SUBMIT_TOKEN_ENV}); unset sends anonymously",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="acknowledge the private-audit notice without an interactive prompt",
    )


def _add_submission_status_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("submission-status", help="read one submission's status and reason codes")
    parser.add_argument("submission_id", help="identifier returned by submit")
    parser.add_argument("--base-url", default=SUBMIT_BASE_URL, help="submission service base URL")
    parser.add_argument("--token-env", metavar="NAME", default=SUBMIT_TOKEN_ENV)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
    parser.add_argument("--version", action="version", version=VERSION_TEXT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_convert_parser(subparsers)
    _add_run_parser(subparsers)
    _add_tui_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_deployment_options_parser(subparsers)
    _add_managed_run_parser(subparsers)
    _add_prepare_submission_parser(subparsers)
    _add_submit_parser(subparsers)
    _add_submission_status_parser(subparsers)
    return parser
