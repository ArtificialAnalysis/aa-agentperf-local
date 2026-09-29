"""Run the full-screen AgentPerf Local terminal interface."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel, Field
from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    Footer,
    Input,
    Label,
    OptionList,
    ProgressBar,
    RichLog,
    Select,
    Sparkline,
    Static,
)

from agentperf_local.client.backends import ClientBackend
from agentperf_local.common.json_fields import one_of
from agentperf_local.common.models import error_text, raising_validator_errors
from agentperf_local.common.statistics import P50_PERCENTILE, P90_PERCENTILE, percentile
from agentperf_local.common.units import BYTES_PER_GIB, MILLISECONDS_PER_SECOND
from agentperf_local.deployment.catalog import (
    DEPLOYMENT_FRAMEWORK_ORDER,
    ModelCandidate,
    ModelCatalog,
    ModelDeployment,
)
from agentperf_local.deployment.context_policy import (
    MINIMUM_CONTEXT_TOKENS,
    context_fit,
    context_ladder,
    default_context_tokens,
    derived_minimum_memory_bytes,
)
from agentperf_local.deployment.endpoint_probes import (
    OLLAMA_RECORDED_POLICY_WARNING,
    ContextProbeResult,
)
from agentperf_local.deployment.frameworks import framework_display_name
from agentperf_local.deployment.managed import (
    DEFAULT_DEPLOYMENT_PORT,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    DEPLOYMENT_LOG_FILENAME,
    strip_log_hint,
)
from agentperf_local.deployment.managed_run import RunActivity, RunActivityKind
from agentperf_local.deployment.model_cache import default_model_cache_root
from agentperf_local.provenance.benchmark import BENCHMARK_CONTEXT_TOKENS, collect_source_provenance
from agentperf_local.provenance.context import below_benchmark_context, context_is_reduced
from agentperf_local.replay.config import ToolChoice
from agentperf_local.replay.runner import RunStartedBoundary, TurnCompletedBoundary, TurnStartedBoundary
from agentperf_local.reports.progress import (
    RunProgress,
    RunProgressState,
    RunTurnSample,
    cumulative_decode_tokens_per_second,
    rate_text,
    reduce_run_boundary,
)
from agentperf_local.submission.bundle import (
    build_submission_bundle,
    validate_bundle_output_path,
    write_submission_bundle,
)
from agentperf_local.submission.client import (
    SUBMIT_BASE_URL,
    SUBMIT_TOKEN_ENV,
    SubmissionError,
    SubmissionReceipt,
    check_revision_allowlist,
    read_submit_token,
    submit_bundle_async,
)
from agentperf_local.submission.private_audit import PRIVATE_AUDIT_RETENTION_DAYS
from agentperf_local.tui.branding import (
    AA_NEUTRAL_500,
    AA_PURPLE,
    COMPACT_LAYOUT_WIDTH,
    RUN_COMPACT_LAYOUT_WIDTH,
    PixelLogo,
    key_value_block,
    spinner_frame,
)
from agentperf_local.tui.controller import (
    LocalManagedReplayController,
    LocalReplayController,
)
from agentperf_local.tui.evidence import EndpointScope, EvidenceSummary, ResultPartition, SelectionKind
from agentperf_local.tui.inputs import (
    CleanSelectStatic,
    ClientBackendSelect,
    ConsentCheckbox,
    DisclosureButton,
    FieldInput,
    FieldSelect,
    FormPage,
    ManagedContextSelect,
    ManagedDeviceSelect,
    ManagedFrameworkSelect,
    ModelDetailPane,
    ModelList,
    ReadingPage,
    ReplayWorkloadSelect,
    RunPage,
    SubmitCheckbox,
    ToolChoiceSelect,
    WelcomeChoice,
)
from agentperf_local.tui.labels import (
    BLOCK_CODE_MESSAGES,
    accelerator_summary_text,
    artifact_kind_text,
    block_code_message,
    count,
    framework_text,
    gib_suffix,
    hardware_target_text,
    home_relative_path_text,
    integer_label,
    memory_need_gib,
    metric_text,
    output_policy_activity_text,
    platform_suffix,
    probe_activity_text,
    recipe_build_text,
    recipe_title_text,
    result_path_text,
    seconds_label,
    standing_headline_text,
    standing_reason_text,
    turn_live_text,
    turn_record_text,
    unit_text,
)
from agentperf_local.tui.messages import (
    ArtifactProgressMessage,
    EndpointProbeMessage,
    PreflightCompletedMessage,
    PreflightFailedMessage,
    ReplayCancelledMessage,
    ReplayCompletedMessage,
    ReplayFailedMessage,
    ReplayFinalizingMessage,
    RevisionAdviceMessage,
    RunActivityMessage,
    RunBoundaryMessage,
    RunPhaseMessage,
    ServerLogMessage,
    TextualRunObserver,
    UploadCompletedMessage,
    UploadFailedMessage,
    UploadProgressMessage,
)
from agentperf_local.tui.model_list import (
    ListedRecipe,
    model_list_options,
    ordered_recipes,
    standing_mark,
    table_width,
)
from agentperf_local.tui.replay_contract import (
    DEVICE_SELECTION_REQUIRED_MESSAGE,
    RECIPE_STANDING_ORDER,
    EndpointProblem,
    ManagedDeploymentChoice,
    ManagedDeviceOption,
    ManagedDeviceSelectionRequired,
    ManagedLaunchSettingsProblem,
    ManagedModelAvailability,
    ManagedReplayController,
    PreflightBlockCode,
    ReplayController,
    ReplayExecution,
    ReplayPreflight,
    ReplayRequest,
    SafeHardwareSummary,
    SetupProblem,
    TuiReplayObserver,
    next_run_directory,
)
from agentperf_local.tui.steps import ESCAPE_ACTION_STEPS, TuiOutcome, TuiStep
from agentperf_local.tui.styles import APP_CSS, MODEL_LIST_CHROME_WIDTH
from agentperf_local.tui.widgets import (
    DONE_MARK,
    FAILED_MARK,
    LATENCY_PLOT_EDGE_RATIO,
    SPEED_PLOT_EDGE_RATIO,
    WARNING_MARK,
    ActivityLog,
    ContextGauge,
    HeadlineDigits,
    Kitty,
    RangeChart,
    SpinnerLine,
    scroll_log_to_end,
)
from agentperf_local.workload.bundled import (
    BUNDLED_REPLAYS,
    CUSTOM_REPLAY_ID,
    DEFAULT_BUNDLED_REPLAY,
    find_bundled_replay,
)
from agentperf_local.workload.schema import load_manifest

DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_OUTPUT_DIR = "agentperf-results"
MINIMUM_SUPPORTED_WIDTH = 48
MINIMUM_SUPPORTED_HEIGHT = 16
SHORT_LAYOUT_HEIGHT = 24
UNAVAILABLE_FRAMEWORK_VALUE = "unavailable"
# A Select needs a string for every option, so the server default gets a value that is not a real tool_choice.
SERVER_DEFAULT_TOOL_CHOICE_VALUE = "server-default"
SERVER_DEFAULT_TOOL_CHOICE_LABEL = "Server default"
NO_TOOL_CHOICE_LABEL = "none · recommended for vLLM"
SETUP_UNUSABLE_MESSAGE = (
    "[b]Can't use this setup.[/b]\n"
    "Check the replay, results folder, server URL, model name, and API key environment variable."
)
SETUP_UNRUNNABLE_MESSAGE = "[b]Can't run this setup.[/b]\nGo back and review the fields."
RUN_STOPPED_MESSAGE = "Check that the server at your URL is running and that the model name matches what it serves."
MANAGED_RUN_STOPPED_MESSAGE = "The model server this app started stopped before the replay finished."
MANAGED_LAUNCH_INVALID_TEMPLATE = (
    "Launch settings are not valid: {reason}. Start the app again with a different --port or --startup-timeout-seconds."
)
FORCE_QUIT_MESSAGE = "Still cancelling. Press q again to force quit. The model server may be left running."
CANCEL_CONFIRM_MESSAGE = "Press esc again to cancel the run"
# One stray Esc must not abort a possibly hours-long run, so the first Esc only arms
# cancellation and the confirming Esc must land within this window.
CANCEL_ARMING_WINDOW_SECONDS = 3.0
BLOCKED_ACTION_MESSAGE = "Press esc to go back and fix this."
RUN_METRICS_PLACEHOLDER = "Last turn · first token — · decode — · total —"
RUN_COUNTERS_PLACEHOLDER = "Starting…"
RUN_SAVING_MESSAGE = "Saving results…"
RUN_CLEANUP_MESSAGE = "Waiting for cleanup…"
# The hero names the phase in plain words; the activity log below it carries the detail.
RUN_HERO_PREPARING = "Preparing the model"
RUN_HERO_CHECKING_SERVER = "Checking the server"
RUN_HERO_STARTING = "Starting the server"
RUN_HERO_RUNNING = "Running the benchmark"
RUN_HERO_SAVING = "Saving results"
RUN_HERO_STOPPING = "Stopping the server"
RUN_HERO_CANCELLING = "Cancelling the run"
RUN_LIVE_CHECKING_SETUP = "Checking the setup"
RUN_LIVE_DOWNLOADING = "Downloading the model"
RUN_ACTIVITY_TITLE = "ACTIVITY"
RUN_SERVER_LOG_TITLE = "SERVER LOG"
RUN_THROUGHPUT_CAPTION = "tokens per second · so far"
RUN_THROUGHPUT_WAITING = "tokens per second · waiting for the first turn"
SERVER_LOG_MAX_LINES = 2_000
# Below these terminal heights the metrics column drops its lowest-value rows so the
# decode-speed chart, the one a benchmark is about, always stays whole.
RUN_TREND_MIN_HEIGHT = 28
RUN_SECOND_CHART_MIN_HEIGHT = 21
PREFLIGHT_CHECKING_MESSAGE = "Checking files, client, and hardware…"
PREFLIGHT_READY_HERO = "Ready to run?"
PREFLIGHT_BLOCKED_HERO = "Check the setup"
ATTACHED_CONTEXT_UNVERIFIED_MESSAGE = (
    "Context not verified for a server you run yourself · submitted as self-reported context evidence"
)
ATTACHED_CONTEXT_FULL_TEMPLATE = "Server reports a {tokens:,}-token context · full benchmark context"
ATTACHED_CONTEXT_REDUCED_TEMPLATE = (
    "Server reports a {tokens:,}-token context · reduced · submitted separately from full-context results"
)
CONSENT_ATTACHED_LABEL = "Check the server, then send the replay to it."
# One line per popular local server: the command that starts it and the URL to enter.
SERVER_START_HINTS = (
    (
        "llama.cpp",
        f"llama-server -m model.gguf -c {BENCHMARK_CONTEXT_TOKENS} --port 8080 · URL http://127.0.0.1:8080/v1",
    ),
    ("Ollama", "ollama serve · URL http://127.0.0.1:11434/v1 · model name is the tag you pulled"),
    ("LM Studio", "turn on the local server under Developer · URL http://127.0.0.1:1234/v1"),
    ("vLLM", f"vllm serve MODEL --max-model-len {BENCHMARK_CONTEXT_TOKENS} · URL http://127.0.0.1:8000/v1"),
)
SERVER_CHECK_RUNNING_MESSAGE = "Asking the server which models it serves"
SERVER_CHECK_FAILED_MESSAGE = "Could not check the server"
SERVER_CHECK_RETRY_HINT = "Fix the URL or start the server, then tick the box again."
PREFLIGHT_SPINNER_INTERVAL_SECONDS = 0.12
RUN_STEP_EYEBROW = "STEP 4 OF 4 · RUN"
RUN_STEP_PREPARING_EYEBROW = "STEP 4 OF 4 · RUN · PREPARING"
RUN_REDUCED_EYEBROW_SUFFIX = " · REDUCED CONTEXT"
RESULT_REDUCED_MESSAGE = "Reduced-context run · submitted separately from full-context results"
PREFLIGHT_OLLAMA_MESSAGE = f"Warning: {OLLAMA_RECORDED_POLICY_WARNING}."
RESULT_RECORDED_POLICY_MESSAGE = (
    "Recorded output policy · turns stop where the model stops · "
    "e2e is reported as a normalized estimate · not directly comparable to exact-policy results"
)
RESULT_EYEBROW = "RESULT"
EXTERNAL_CATALOG_PROVENANCE = "External catalog · not from Artificial Analysis"
CUSTOM_ENDPOINT_DETAIL = (
    "[b]Other model or server[/b]\n\n"
    "Use a server you already run: llama.cpp, Ollama, LM Studio, vLLM, or any OpenAI-compatible API.\n"
    "Enter its URL and model name on the next screen. Press ? for the commands that start each server."
)
SUBMIT_CHECKBOX_LABEL = "Submit results to Artificial Analysis (optional)"
SUBMIT_NOTICE = (
    "May be published: aggregate results and sanitized turn timings.\n"
    f"Hardware and verification evidence stays private for {PRIVATE_AUDIT_RETENTION_DAYS} days.\n"
    "Never sent: prompts, responses, credentials, local paths, URLs or hostnames.\n"
    "Failed checks are still submitted as self-reported."
)
SUBMIT_SELF_REPORTED_NOTE = "\n" + key_value_block(
    (("Eligibility", "This client version can submit, but cannot reach verified"),)
)
SUBMIT_ALLOWLIST_UNREACHABLE_NOTE = "\n" + key_value_block(
    (("Eligibility", "Could not check; submission is still available"),)
)
UPLOAD_PREPARING_MESSAGE = "Preparing the submission bundle…"
UPLOAD_SENDING_TEMPLATE = "Uploading to Artificial Analysis · {sent:,} / {total:,} bytes"
UPLOAD_WAITING_MESSAGE = "Upload sent · waiting for Artificial Analysis to confirm the submission…"
UPLOAD_BUSY_MESSAGE = "Submitting…"
UPLOAD_CONFIRMING_MESSAGE = "Waiting for confirmation…"
UPLOAD_QUIT_BLOCKED_MESSAGE = "Submission in progress · wait for confirmation · Ctrl+C cancels and quits"
UPLOAD_DONE_TEMPLATE = (
    "Submitted · {submission_id} · {status}{duplicate}\nCheck later: agentperf-local submission-status {submission_id}"
)
UPLOAD_DUPLICATE_NOTE = " · this bundle was already on file"
UPLOAD_FAILED_TEMPLATE = (
    "Upload failed · {reason}\nBundle kept at {bundle}\nRetry: agentperf-local submit {bundle} --yes"
)
UPLOAD_CANCELLED_REASON = "cancelled before the service answered"
UPLOAD_NOT_PREPARED_TEMPLATE = "Upload failed · {reason}\nNo bundle was written; the run folder is unchanged."
# Every back-style button leaves its own page, so all of them share the escape handler.
BACK_BUTTON_IDS = frozenset(
    {"model-back", "config-back", "preflight-back", "privacy-back", "methodology-back", "result-new"}
)


class ClosedTurnSeries(BaseModel, frozen=True):
    """Hold the chartable numbers of every successful closed turn, one series per metric."""

    ttft_ms: tuple[float, ...]
    e2e_ms: tuple[float, ...]
    decode_tokens_per_second: tuple[float, ...]


def _closed_turn_series(samples: Sequence[RunTurnSample]) -> ClosedTurnSeries:
    """Filter the closed turns to the successful ones and split them into chart series."""
    closed = tuple(sample for sample in samples if sample.success)
    return ClosedTurnSeries(
        ttft_ms=tuple(sample.ttft_ms for sample in closed if sample.ttft_ms is not None),
        e2e_ms=tuple(sample.e2e_ms for sample in closed if sample.e2e_ms is not None),
        decode_tokens_per_second=tuple(
            sample.decode_tokens_per_second for sample in closed if sample.decode_tokens_per_second is not None
        ),
    )


def _context_option_label(deployment: ModelDeployment, context_tokens: int) -> str:
    """Label one context option with its honest memory need and comparability."""
    minimum_memory_bytes = derived_minimum_memory_bytes(deployment, context_tokens)
    if context_tokens == deployment.context_tokens:
        return f"{context_tokens:,} tokens · full benchmark · needs {memory_need_gib(minimum_memory_bytes)} GiB"
    return f"{context_tokens:,} tokens · needs {memory_need_gib(minimum_memory_bytes)} GiB · not comparable"


def _requested_context_tokens(request: ReplayRequest) -> int:
    """Return the context this run asks for: the managed launch context, or the full benchmark."""
    choice = request.managed_deployment
    return BENCHMARK_CONTEXT_TOKENS if choice is None else choice.resolved_context_tokens


def _run_context_window(request: ReplayRequest, probe: ContextProbeResult | None) -> int | None:
    """Return the context length this run is served at, or None when nothing observed one.

    A managed request states the context it launches at. A user-run server only has the
    length its own model list reported, and an unreported one is not a denominator.
    """
    choice = request.managed_deployment
    if choice is None:
        return None if probe is None else probe.observed_tokens
    return choice.resolved_context_tokens


def _reduced_context_facts(request: ReplayRequest, probe: ContextProbeResult | None) -> tuple[int, int] | None:
    """Return (served, full) context tokens for a request served below the full context, else None.

    An unreported attached context is unproven, not reduced, and is worded separately.
    """
    served = _run_context_window(request, probe)
    if served is None or not below_benchmark_context(served):
        return None
    return served, BENCHMARK_CONTEXT_TOKENS


def _request_context_reduced(request: ReplayRequest, probe: ContextProbeResult | None) -> bool:
    """Apply the run summary's reduced-context rule to one request, so the screen agrees with the disk."""
    return context_is_reduced(_requested_context_tokens(request), _run_context_window(request, probe))


def _attached_context_text(probe: ContextProbeResult | None) -> str:
    """Word what is known about a user-run server's context from its /models answer."""
    if probe is None or probe.observed_tokens is None:
        return ATTACHED_CONTEXT_UNVERIFIED_MESSAGE
    if below_benchmark_context(probe.observed_tokens):
        return ATTACHED_CONTEXT_REDUCED_TEMPLATE.format(tokens=probe.observed_tokens)
    return ATTACHED_CONTEXT_FULL_TEMPLATE.format(tokens=probe.observed_tokens)


def _failure_cause(request: ReplayRequest, error: BaseException) -> str | None:
    """Return display-safe cause text for a failed run, or None to stay generic.

    An attached run shows only an EndpointProblem, whose text names no URL. A managed
    run shows deployment ValueError, RuntimeError, and TimeoutError text: those messages
    name frameworks, exit statuses, token counts, and catalog filenames — never URLs or
    log contents. ValueError carries the refusals raised before the server starts, such
    as a port already in use, which name the flag that resolves them. The log-path hint
    is dropped because the result page names the deployment log when one exists.
    """
    if isinstance(error, EndpointProblem):
        return str(error)
    if request.managed_deployment is None or not isinstance(error, ValueError | RuntimeError | TimeoutError):
        return None
    cause = strip_log_hint(error_text(error), request.output_dir / DEPLOYMENT_LOG_FILENAME).strip()
    return cause or None


def _device_option_label(option: ManagedDeviceOption) -> str:
    """Name one accelerator by its stable launch-time index."""
    if option.memory_bytes is None:
        return f"{option.index} · {escape(option.name)} · memory not reported"
    return f"{option.index} · {escape(option.name)} · {option.memory_bytes / BYTES_PER_GIB:.0f} GiB"


class BenchmarkSelection(BaseModel, frozen=True):
    """Store model intent separately from execution evidence."""

    kind: SelectionKind
    display_name: str
    endpoint_model: str
    profile_id: str | None
    catalog_digest: str | None
    candidate_revision: str | None

    @classmethod
    def from_candidate(cls, candidate: ModelCandidate, catalog: ModelCatalog) -> BenchmarkSelection:
        """Create selection intent from one unsigned pilot candidate."""
        return cls(
            kind=(
                SelectionKind.BUNDLED_CATALOG_CANDIDATE
                if catalog.is_bundled_snapshot
                else SelectionKind.EXTERNAL_CATALOG_ENTRY
            ),
            display_name=recipe_title_text(candidate, catalog.hardware_of(candidate)),
            endpoint_model=candidate.hf_repository,
            profile_id=candidate.profile_id,
            catalog_digest=catalog.digest,
            candidate_revision=candidate.hf_revision,
        )

    @classmethod
    def custom(cls, endpoint_model: str) -> BenchmarkSelection:
        """Create an Explorer-only custom endpoint selection."""
        return cls(
            kind=SelectionKind.CUSTOM_ENDPOINT,
            display_name="Custom OpenAI-compatible endpoint",
            endpoint_model=endpoint_model,
            profile_id=None,
            catalog_digest=None,
            candidate_revision=None,
        )


class TuiDefaults(BaseModel, frozen=True):
    """Seed editable fields without starting work."""

    replay_id: str = DEFAULT_BUNDLED_REPLAY.replay_id
    manifest_path: Path | None = None
    output_dir: Path | None = None
    base_url: str = DEFAULT_BASE_URL
    endpoint_model: str | None = None
    api_key_env: str | None = None
    client_backend: ClientBackend = "python"
    tool_choice: ToolChoice | None = None
    model_cache_root: Path = Field(default_factory=default_model_cache_root)
    deployment_port: int = DEFAULT_DEPLOYMENT_PORT
    deployment_startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS
    device_index: int | None = None
    submit_base_url: str = SUBMIT_BASE_URL
    submit_token_env: str = SUBMIT_TOKEN_ENV


@dataclass(frozen=True, slots=True, kw_only=True)
class _NavigationBookmark:
    """Keep the page and editing position to restore after Help or Privacy."""

    step: TuiStep
    focused: Widget | None = None
    cursor: int | None = None
    scroll_y: float = 0.0


class AgentPerfLocalApp(App[TuiOutcome]):
    """Guide model setup, benchmark progress, and local results."""

    TITLE = "AgentPerf Local"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = (
        Binding("escape", "back_quit", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("escape", "back_cancel", "Cancel"),
        Binding("escape", "back_new_run", "New run"),
        Binding("l", "toggle_server_log", "Server log"),
        Binding("?", "help", "Help"),
        Binding("f1", "help", "Help", show=False),
        Binding("up,left", "focus_previous", "Previous control", show=False),
        Binding("down,right", "focus_next", "Next control", show=False),
        Binding("q", "quit_or_type_q", "Quit"),
        Binding("ctrl+p", "privacy", "Privacy", show=False),
        Binding("ctrl+c", "quit", "Cancel", show=False, priority=True),
    )
    CSS = APP_CSS

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        controller: ReplayController | None = None,
        managed_controller: ManagedReplayController | None = None,
        defaults: TuiDefaults | None = None,
    ) -> None:
        super().__init__()
        if not catalog.models:
            raise ValueError("model catalog must not be empty")
        self.catalog = catalog
        self.controller = LocalReplayController() if controller is None else controller
        self.managed_controller = (
            LocalManagedReplayController.detect(catalog.as_of)
            if controller is None and managed_controller is None
            else managed_controller
        )
        self.defaults = TuiDefaults() if defaults is None else defaults
        self.device_options = () if self.managed_controller is None else self.managed_controller.device_options()
        self.step = TuiStep.WELCOME
        self.setup_return_step = TuiStep.MODEL
        self._information_bookmark: _NavigationBookmark | None = None
        self.listed_recipes = self._listed_recipes()
        self.selection = (
            BenchmarkSelection.custom(self.defaults.endpoint_model)
            if self.defaults.endpoint_model is not None
            else BenchmarkSelection.from_candidate(self._default_candidate(), catalog)
        )
        self.request: ReplayRequest | None = None
        self.pending_preflight: ReplayRequest | None = None
        self.progress_state = RunProgressState()
        self.execution: ReplayExecution | None = None
        self.evidence = EvidenceSummary.derive(self.selection.kind, EndpointScope.LOOPBACK_NAME)
        self.replay_active = False
        self.replay_finalizing = False
        self.replay_cancelling = False
        self.force_quit_armed = False
        # Captured when the run starts, so a later click on the checkbox cannot change a run in flight.
        self.submit_requested = False
        self.upload_active = False
        self.upload_bundle_dir: Path | None = None
        self.submission_receipt: SubmissionReceipt | None = None
        self.cancel_armed = False
        self.cancel_arming_timer: Timer | None = None
        self.run_generation = 0
        self.outcome = TuiOutcome.NO_RUN
        self.terminal_too_small = False
        self.attached_base_url = self.defaults.base_url
        self.attached_api_key_env = self.defaults.api_key_env or ""
        self.run_metrics_text = RUN_METRICS_PLACEHOLDER
        # The consent-time server check and the check the run itself made; the run's wins once it exists.
        self.preflight_probe: ContextProbeResult | None = None
        self.run_context_probe: ContextProbeResult | None = None
        self.preflight_spinner_timer: Timer | None = None
        self.preflight_spinner_tick = 0
        # Bundled replay manifests are tiny package files; their declared context
        # demand is read once and reused by every context-ladder rebuild.
        self.replay_floor_cache: dict[str, int | None] = {}

    def compose(self) -> ComposeResult:
        """Compose the permanent shell and all stateful pages."""
        with Vertical(id="shell"):
            yield Static(
                f"This terminal is too small. "
                f"Resize it to at least {MINIMUM_SUPPORTED_WIDTH}×{MINIMUM_SUPPORTED_HEIGHT}.",
                id="small-terminal-warning",
            )
            with ContentSwitcher(initial=TuiStep.WELCOME.value, id="content"):
                yield from self._welcome_page()
                yield from self._model_page()
                yield from self._config_page()
                yield from self._preflight_page()
                yield from self._run_page()
                yield from self._result_page()
                yield from self._privacy_page()
                yield from self._methodology_page()
        yield Footer(show_command_palette=False, compact=True)

    def _welcome_page(self) -> ComposeResult:
        with FormPage(id=TuiStep.WELCOME.value, classes="page"):
            with Horizontal(id="welcome-brand"):
                yield PixelLogo(id="welcome-logo")
                with Vertical(id="welcome-heading"):
                    yield Static("AgentPerf Local", classes="hero")
                    yield Static("Artificial Analysis", classes="lede")
            yield Static(
                "Benchmark a model on your computer. Replay a recorded agent session "
                "and measure response speed and latency.",
                classes="lede",
            )
            yield WelcomeChoice(
                "Choose a model & config", "Download and start a supported model here.", id="welcome-start"
            )
            yield WelcomeChoice(
                "Use existing server", "Connect to Ollama, LM Studio, llama.cpp, or another API.", id="welcome-existing"
            )
            yield Static("Arrow keys to move · Enter or click to choose", classes="key-hint")

    def _model_page(self) -> ComposeResult:
        # Text prompts are not parsed as markup; the list's CSS truncates long rows.
        options = model_list_options(self.listed_recipes, self._computer_text())
        with FormPage(id=TuiStep.MODEL.value, classes="page"):
            yield Static("STEP 1 OF 4 · MODEL", id="model-intro", classes="eyebrow")
            yield Static("Choose a model & config", classes="hero")
            yield Static(
                "↑ ↓ browse · ← → move between controls · Enter or click to select",
                id="model-list-hint",
                classes="lede",
            )
            with Horizontal(id="model-layout"):
                yield ModelList(*options, id="model-list")
                with ModelDetailPane(id="model-detail-pane"):
                    yield Static(self._candidate_detail(self._default_candidate()), id="model-detail")
            with Horizontal(classes="actions"):
                yield Button("Select", id="model-continue", flat=True, compact=True)
                yield Button("Back", id="model-back", flat=True, compact=True)

    def _config_page(self) -> ComposeResult:
        manifest = "" if self.defaults.manifest_path is None else str(self.defaults.manifest_path)
        selected_replay = CUSTOM_REPLAY_ID if self.defaults.manifest_path is not None else self.defaults.replay_id
        replay_options = tuple(
            (f"{replay.label} · {replay.summary}", replay.replay_id) for replay in BUNDLED_REPLAYS
        ) + (("Custom replay file", CUSTOM_REPLAY_ID),)
        output = str(self.defaults.output_dir or Path(DEFAULT_OUTPUT_DIR))
        endpoint_model = self.defaults.endpoint_model or self.selection.endpoint_model
        api_key_env = self.defaults.api_key_env or ""
        with FormPage(id=TuiStep.CONFIG.value, classes="page"):
            yield Static("STEP 2 OF 4 · SETUP", id="config-intro", classes="eyebrow")
            yield Static("Set up the run", classes="hero")
            yield Static("", id="config-selection", classes="lede")
            with Horizontal(classes="field-row"):
                yield Label("Replay")
                yield ReplayWorkloadSelect(
                    replay_options,
                    value=selected_replay,
                    allow_blank=False,
                    compact=True,
                    id="replay-workload-select",
                )
            with Horizontal(id="manifest-row", classes="field-row"):
                yield Label("Replay file")
                yield FieldInput(manifest, placeholder="/path/to/replay.json", id="manifest-input", compact=True)
            yield Static("YOUR SERVER", id="section-server", classes="section-label")
            with Horizontal(id="managed-device-row", classes="field-row"):
                yield Label("Device")
                yield ManagedDeviceSelect(
                    tuple((_device_option_label(option), str(option.index)) for option in self.device_options),
                    prompt="Choose a device",
                    allow_blank=True,
                    compact=True,
                    id="managed-device-select",
                )
            with Horizontal(id="managed-framework-row", classes="field-row"):
                yield Label("Framework")
                yield ManagedFrameworkSelect(
                    (("No compatible framework", UNAVAILABLE_FRAMEWORK_VALUE),),
                    value=UNAVAILABLE_FRAMEWORK_VALUE,
                    allow_blank=False,
                    compact=True,
                    disabled=True,
                    id="managed-framework-select",
                )
            with Horizontal(id="managed-context-row", classes="field-row"):
                yield Label("Context")
                yield ManagedContextSelect(
                    # A placeholder only; selecting a managed model rebuilds the real ladder.
                    (("Full benchmark context", "full"),),
                    value="full",
                    allow_blank=False,
                    compact=True,
                    id="managed-context-select",
                )
            yield Static("", id="managed-deployment-status", classes="lede")
            with Horizontal(id="base-url-row", classes="field-row"):
                yield Label("Server URL")
                yield FieldInput(
                    self.defaults.base_url, placeholder=DEFAULT_BASE_URL, id="base-url-input", compact=True
                )
            with Horizontal(id="endpoint-model-row", classes="field-row"):
                yield Label("Model name")
                yield FieldInput(
                    endpoint_model,
                    placeholder="name used by your server",
                    id="endpoint-model-input",
                    compact=True,
                )
            with Horizontal(id="api-key-row", classes="field-row"):
                yield Label("API key variable")
                yield FieldInput(
                    api_key_env,
                    placeholder="Optional · e.g. OPENAI_API_KEY",
                    id="api-key-env-input",
                    compact=True,
                )
            with Horizontal(id="output-row", classes="field-row"):
                yield Label("Results folder")
                yield FieldInput(output, placeholder=DEFAULT_OUTPUT_DIR, id="output-input", compact=True)
            # The measured client is a detail most people never change, so it sits last.
            yield DisclosureButton("Advanced options", id="advanced-toggle")
            with Horizontal(id="client-row", classes="field-row disclosed"):
                yield Label("Client")
                yield ClientBackendSelect(
                    (("Python", "python"), ("Rust (experimental)", "rust")),
                    value=self.defaults.client_backend,
                    allow_blank=False,
                    compact=True,
                    id="client-backend-select",
                )
            with Horizontal(id="tool-choice-row", classes="field-row disclosed"):
                yield Label("Tool choice")
                yield ToolChoiceSelect(
                    (
                        (SERVER_DEFAULT_TOOL_CHOICE_LABEL, SERVER_DEFAULT_TOOL_CHOICE_VALUE),
                        (NO_TOOL_CHOICE_LABEL, "none"),
                    ),
                    value=self.defaults.tool_choice or SERVER_DEFAULT_TOOL_CHOICE_VALUE,
                    allow_blank=False,
                    compact=True,
                    id="tool-choice-select",
                )
            with Horizontal(classes="actions"):
                yield Button("Review setup", id="config-continue", flat=True, compact=True)
                yield Button("Back", id="config-back", flat=True, compact=True)

    def _preflight_page(self) -> ComposeResult:
        with FormPage(id=TuiStep.PREFLIGHT.value, classes="page"):
            yield Static("STEP 3 OF 4 · CONFIRM", classes="eyebrow")
            yield Static(PREFLIGHT_READY_HERO, id="preflight-hero", classes="hero")
            yield Static("Checking setup…", id="preflight-status", classes="card")
            yield Static("", id="preflight-reduced", classes="cancel-card")
            yield Static(
                "Checking the server location…",
                id="preflight-evidence",
                classes="lede",
            )
            yield ConsentCheckbox(
                CONSENT_ATTACHED_LABEL,
                id="endpoint-consent-checkbox",
                compact=True,
                disabled=True,
            )
            yield SpinnerLine(id="preflight-server-check")
            yield Static(PREFLIGHT_OLLAMA_MESSAGE, id="preflight-ollama", classes="cancel-card")
            with Vertical(id="submit-panel"):
                yield SubmitCheckbox(SUBMIT_CHECKBOX_LABEL, id="submit-checkbox", compact=True, disabled=True)
                yield Static("", id="submit-notice")
            with Horizontal(classes="actions"):
                yield Button("Run benchmark", id="run-start", flat=True, compact=True, disabled=True)
                yield Button("Back", id="preflight-back", flat=True, compact=True)

    def _run_page(self) -> ComposeResult:
        with RunPage(id=TuiStep.RUN.value, classes="page"):
            with Horizontal(classes="page-heading"):
                with Vertical(classes="page-heading-copy"):
                    yield Static(RUN_STEP_EYEBROW, id="run-eyebrow", classes="eyebrow")
                    yield Static(RUN_HERO_PREPARING, id="run-hero", classes="hero")
                yield Kitty(id="run-kitty")
            with Horizontal(id="run-progress-row"):
                yield ProgressBar(total=1, show_eta=False, id="run-progress")
                yield Static(RUN_COUNTERS_PLACEHOLDER, id="run-counters")
            # The status-row class gives it the status line's inset, so the two align.
            yield ContextGauge(id="run-context", classes="status-row")
            yield Static(RUN_METRICS_PLACEHOLDER, id="run-metrics", classes="status-row")
            with Horizontal(id="run-body"):
                with Vertical(id="run-left"):
                    yield Static(RUN_ACTIVITY_TITLE, id="run-left-title", classes="section-label")
                    with ContentSwitcher(initial="run-activity", id="run-left-switcher"):
                        yield ActivityLog(id="run-activity")
                        yield RichLog(
                            id="run-server-log", markup=False, wrap=True, min_width=20, max_lines=SERVER_LOG_MAX_LINES
                        )
                with VerticalScroll(id="run-metrics-column"):
                    yield HeadlineDigits(id="run-throughput")
                    yield Static(RUN_THROUGHPUT_WAITING, id="run-throughput-caption")
                    yield RangeChart(
                        "DECODE SPEED",
                        "tok/s",
                        format_value=integer_label,
                        edge_ratio=SPEED_PLOT_EDGE_RATIO,
                        id="run-decode-chart",
                    )
                    yield RangeChart(
                        "FIRST TOKEN",
                        "ms",
                        format_value=integer_label,
                        edge_ratio=LATENCY_PLOT_EDGE_RATIO,
                        id="run-ttft-chart",
                    )
                    yield Static("TREND · tok/s per turn", id="run-trend-title", classes="section-label")
                    yield Sparkline([], id="run-trend")
            yield Button("Cancel", id="run-cancel", flat=True, compact=True)

    def _result_page(self) -> ComposeResult:
        with ReadingPage(id=TuiStep.RESULT.value, classes="page"):
            with Horizontal(classes="page-heading"):
                with Vertical(classes="page-heading-copy"):
                    yield Static(RESULT_EYEBROW, id="result-eyebrow", classes="eyebrow")
                    yield Static("Run finished.", id="result-title", classes="hero")
                yield Kitty(id="result-kitty")
            yield CleanSelectStatic("Waiting for results…", id="result-status", classes="card")
            yield HeadlineDigits(id="result-throughput")
            yield Static("output tokens per second", id="result-throughput-caption", classes="status-row")
            yield Static("No metrics yet.", id="result-metrics", classes="status-row")
            yield Static(RESULT_REDUCED_MESSAGE, id="result-reduced", classes="cancel-card")
            yield Static(RESULT_RECORDED_POLICY_MESSAGE, id="result-policy", classes="cancel-card")
            yield CleanSelectStatic(
                "Saved on this computer · model and GPU not verified",
                id="result-evidence",
                classes="lede",
            )
            yield DisclosureButton("Result details", id="result-details-toggle")
            with Vertical(id="result-details", classes="disclosed"):
                yield Static("", id="result-metrics-detail", classes="status-row")
                with Vertical(id="result-charts"):
                    yield RangeChart(
                        "FIRST TOKEN",
                        "ms",
                        format_value=integer_label,
                        edge_ratio=LATENCY_PLOT_EDGE_RATIO,
                        id="result-ttft-chart",
                    )
                    yield RangeChart(
                        "DECODE SPEED",
                        "tok/s",
                        format_value=integer_label,
                        edge_ratio=SPEED_PLOT_EDGE_RATIO,
                        id="result-decode-chart",
                    )
                    yield RangeChart(
                        "TURN TIME",
                        "s",
                        format_value=seconds_label,
                        edge_ratio=LATENCY_PLOT_EDGE_RATIO,
                        id="result-e2e-chart",
                    )
            yield CleanSelectStatic("", id="result-upload", classes="card")
            yield ProgressBar(total=1, show_eta=False, id="result-upload-progress")
            with Horizontal(classes="actions"):
                yield Button("New run", id="result-new", flat=True, compact=True)
                yield SpinnerLine(id="result-upload-busy")
            yield Static("", id="result-next-hint", classes="key-hint")

    def _privacy_page(self) -> ComposeResult:
        with ReadingPage(id=TuiStep.PRIVACY.value, classes="page"):
            yield Static("PRIVACY", id="privacy-title", classes="eyebrow")
            yield Static("Local data", classes="hero")
            yield Static(
                key_value_block(
                    (
                        ("prompts go to", "only the server URL you set"),
                        ("model downloads", "only the exact pinned file from Hugging Face"),
                        ("results", "a new run folder inside your chosen results folder"),
                        ("uploads", "only when you tick Submit on the confirm screen · four files, one request"),
                    )
                ),
                classes="card",
            )
            with Horizontal(classes="actions"):
                yield Button("Back", id="privacy-back", flat=True, compact=True)

    def _methodology_page(self) -> ComposeResult:
        with ReadingPage(id=TuiStep.METHODOLOGY.value, classes="page"):
            yield Static("HELP", id="methodology-title", classes="eyebrow")
            yield Static("Keys, servers, and evidence", classes="hero")
            yield Static("KEYS", classes="section-label")
            yield Static(
                key_value_block(
                    (
                        ("↑ ↓", "preview models, move between fields, or scroll a log"),
                        ("← →", "previous / next control · move the cursor while typing"),
                        ("Enter", "open or choose"),
                        ("Click", "choose a model, open a control, or press a button"),
                        ("Space", "turn a checkbox on or off"),
                        ("Esc", "go back one screen · press it twice during a run to cancel"),
                        ("l", "show the server log during a run this app started"),
                        ("d", "show or hide extra run charts"),
                        ("q", "quit the app"),
                        ("?", "open or close help"),
                        ("Ctrl+P", "open the privacy page"),
                    )
                ),
                classes="card",
            )
            yield Static("START YOUR OWN SERVER", classes="section-label")
            yield Static(
                key_value_block(SERVER_START_HINTS)
                + f"\n\nThe full benchmark needs a {BENCHMARK_CONTEXT_TOKENS:,}-token context. "
                "A smaller context still runs, but the result is marked as not comparable.",
                classes="card",
            )
            yield Static("EVIDENCE", classes="section-label")
            yield Static(
                "A server you run yourself is not verified: this app can't check which model file or GPU it uses. "
                "When this app starts the server, it saves the exact model file digest and GPU startup evidence "
                "with the results.",
                classes="card",
            )
            with Horizontal(classes="actions"):
                yield Button("Back", id="methodology-back", flat=True, compact=True)
                yield Button("Privacy and sharing", id="help-privacy", action="app.privacy", flat=True, compact=True)

    def on_mount(self) -> None:
        """Seed the model list and initial detail panel."""
        self._show_result_throughput(None)
        self.query_one("#result-charts", Vertical).add_class("-empty")
        self.query_one("#manifest-row", Horizontal).display = self._uses_custom_manifest()
        self.query_one("#preflight-reduced", Static).display = False
        self.query_one("#preflight-ollama", Static).display = False
        self.query_one("#result-reduced", Static).display = False
        self.query_one("#result-policy", Static).display = False
        device_sync_pending = self._apply_default_device()
        initial_option_id = (
            SelectionKind.CUSTOM_ENDPOINT.value
            if self.defaults.endpoint_model is not None or self._first_recipe_here() is None
            else self._default_candidate().profile_id
        )
        self.query_one("#model-list", OptionList).highlighted = self._model_option_index(initial_option_id)
        self._render_option_detail(initial_option_id)
        if not device_sync_pending:
            # Syncing twice would run the whole framework-availability probe twice at startup.
            self._sync_managed_deployment_controls()
        self._apply_terminal_size(self.size.width, self.size.height)
        self.call_after_refresh(self._focus_step)

    def on_resize(self, event: events.Resize) -> None:
        """Apply the explicit minimum and compact layout contracts."""
        self._apply_terminal_size(event.size.width, event.size.height)

    def _apply_terminal_size(self, width: int, height: int) -> None:
        too_small = width < MINIMUM_SUPPORTED_WIDTH or height < MINIMUM_SUPPORTED_HEIGHT
        restored_supported_size = self.terminal_too_small and not too_small
        self.terminal_too_small = too_small
        self.query_one("#small-terminal-warning", Static).display = too_small
        self.query_one("#content", ContentSwitcher).display = not too_small
        compact = width < COMPACT_LAYOUT_WIDTH
        self.query_one("#model-layout", Horizontal).set_class(compact, "compact")
        # Beside the detail, the list never narrows below its table; stacked, it spans the page anyway.
        self.query_one("#model-list", OptionList).styles.min_width = (
            None if compact else table_width(self.listed_recipes) + MODEL_LIST_CHROME_WIDTH
        )
        run_page = self.query_one(f"#{TuiStep.RUN.value}", Vertical)
        run_page.set_class(height < RUN_TREND_MIN_HEIGHT, "short-run")
        run_page.set_class(height < RUN_SECOND_CHART_MIN_HEIGHT, "shorter-run")
        for page in self.query(".page"):
            page.set_class(compact, "compact")
            page.set_class(height < SHORT_LAYOUT_HEIGHT, "short")
        run_compact = width < RUN_COMPACT_LAYOUT_WIDTH
        run_page.set_class(run_compact, "compact")
        if run_compact and self._showing_server_log():
            # A compact page shows either the log or the details; keep the log the user opened.
            run_page.remove_class("show-details")
        # The gauge draws a narrower bar once the run page carries the compact class.
        self._render_context_gauge()
        self._sync_kitty()
        if restored_supported_size:
            self.call_after_refresh(self._focus_step)

    def _candidate(self, profile_id: str) -> ModelCandidate | None:
        return next((candidate for candidate in self.catalog.models if candidate.profile_id == profile_id), None)

    def _selected_candidate(self) -> ModelCandidate | None:
        """Return the catalog candidate represented by current selection intent."""
        if self.selection.profile_id is None:
            return None
        return self._candidate(self.selection.profile_id)

    def _listed_recipes(self) -> tuple[ListedRecipe, ...]:
        return ordered_recipes(self.catalog, self._listing_availability)

    def _first_recipe_here(self) -> ModelCandidate | None:
        """Return the first listed recipe for this computer, or None when every recipe needs other hardware."""
        return next((recipe.candidate for recipe in self.listed_recipes if not recipe.for_other_hardware), None)

    def _default_candidate(self) -> ModelCandidate:
        """Return the recipe the model screen opens on: the first for this computer, else the first listed."""
        return self._first_recipe_here() or self.listed_recipes[0].candidate

    def _model_option_index(self, option_id: str) -> int:
        return self.query_one("#model-list", OptionList).get_option_index(option_id)

    def _computer_text(self) -> str | None:
        """Name this computer's accelerator for the model list, or None without hardware detection."""
        if self.managed_controller is None:
            return None
        return accelerator_summary_text(self.managed_controller.hardware_summary())

    def _listing_availability(self, candidate: ModelCandidate) -> ManagedModelAvailability | None:
        """Return what the model screen shows for one recipe at the chosen context.

        With several accelerators and none chosen yet, the best device's result stands
        for the computer, because the device is picked only on the next screen.
        """
        controller = self.managed_controller
        if controller is None:
            return None
        context_tokens = self._chosen_context_tokens(candidate.deployment)
        device_index = self._selected_device_index()
        if device_index is not None or len(self.device_options) <= 1:
            device_indexes: tuple[int | None, ...] = (device_index,)
        else:
            device_indexes = tuple(option.index for option in self.device_options)
        results = tuple(
            self._device_availability(controller, candidate, index, context_tokens) for index in device_indexes
        )
        return min(results, key=lambda availability: RECIPE_STANDING_ORDER.index(availability.standing))

    def _device_availability(
        self,
        controller: ManagedReplayController,
        candidate: ModelCandidate,
        device_index: int | None,
        context_tokens: int,
    ) -> ManagedModelAvailability:
        """Return one device's availability, counting a device no framework can serve as other hardware.

        Binding an accelerator the app cannot classify, such as integrated graphics,
        raises; the model screen must still open so an existing server stays reachable.
        """
        try:
            return controller.availability(
                candidate,
                device_index=device_index,
                context_tokens=context_tokens,
                replay_floor_tokens=self._replay_context_floor(),
            )
        except ValueError as error:
            return ManagedModelAvailability(hardware=controller.hardware_summary(), offers=(), reason=error_text(error))

    def _managed_availability(
        self,
        candidate: ModelCandidate,
        device_index: int | None = None,
        context_tokens: int | None = None,
    ) -> ManagedModelAvailability | None:
        """Return machine compatibility for one recipe, or None when no managed controller exists."""
        if self.managed_controller is None:
            return None
        return self.managed_controller.availability(
            candidate,
            device_index=device_index,
            context_tokens=context_tokens,
            replay_floor_tokens=self._replay_context_floor(),
        )

    def _apply_default_device(self) -> bool:
        """Preselect the device named at launch, ignoring an index this computer does not have.

        Report whether the picker changed, because that raises its own Changed message and
        the handler for it syncs the deployment controls.
        """
        index = self.defaults.device_index
        if index is None or not any(option.index == index for option in self.device_options):
            return False
        self.query_one("#managed-device-select", ManagedDeviceSelect).value = str(index)
        return True

    def _selected_device_index(self) -> int | None:
        """Return the accelerator the managed launch is pinned to, or None while none is chosen."""
        if len(self.device_options) <= 1:
            # A computer with one accelerator never offers a choice, so it stays unpinned.
            return None
        value = self._select_text("#managed-device-select", ManagedDeviceSelect)
        return None if value is None else int(value)

    def _select_text(self, selector: str, kind: type[FieldSelect]) -> str | None:
        """Return one setup dropdown's chosen text, or None while it is unmounted or blank."""
        try:
            value = self.query_one(selector, kind).value
        except (NoMatches, ScreenStackError):
            # The model list is ordered at construction and its detail renders during
            # compose, both before the setup form mounts.
            return None
        return value if isinstance(value, str) else None

    def _sync_managed_deployment_controls(self, *, rebuild_context: bool = True) -> None:
        """Render the setup fields for an attached endpoint or the selected owned deployment.

        Setting select values here posts Changed messages that re-enter this method once
        each; those re-renders keep the already-settled values, so the chain stops there.
        """
        candidate = self._selected_candidate()
        device_row = self.query_one("#managed-device-row", Horizontal)
        row = self.query_one("#managed-framework-row", Horizontal)
        context_row = self.query_one("#managed-context-row", Horizontal)
        status = self.query_one("#managed-deployment-status", Static)
        framework_select = self.query_one("#managed-framework-select", ManagedFrameworkSelect)
        base_url = self.query_one("#base-url-input", Input)
        endpoint_model = self.query_one("#endpoint-model-input", Input)
        api_key_env = self.query_one("#api-key-env-input", Input)
        self.query_one("#section-server", Static).update(
            "MODEL SERVER · started for you" if candidate is not None else "YOUR SERVER"
        )
        self.query_one("#config-selection", Static).update(
            escape(recipe_title_text(candidate, self.catalog.hardware_of(candidate)))
            if candidate is not None
            else "Use the URL and model name shown by your server."
        )
        self.query_one("#base-url-row", Horizontal).display = candidate is None
        self.query_one("#endpoint-model-row", Horizontal).display = candidate is None
        # One accelerator leaves nothing to choose, so that computer never sees the picker.
        device_row.display = candidate is not None and len(self.device_options) > 1
        row.display = candidate is not None
        # An attached server's context is observed, not chosen, so only managed models see this picker.
        context_row.display = candidate is not None
        status.display = candidate is not None
        # A managed server never takes a key, so the always-empty disabled row only adds noise.
        self.query_one("#api-key-row", Horizontal).display = candidate is None
        # A managed run sets tool_choice from its framework, so only an attached server offers the choice.
        self.query_one("#tool-choice-row", Horizontal).set_class(candidate is not None, "-inapplicable")
        for endpoint_input in (base_url, endpoint_model, api_key_env):
            endpoint_input.disabled = candidate is not None
        if candidate is None:
            base_url.value = self.attached_base_url
            endpoint_model.value = self.selection.endpoint_model
            api_key_env.value = self.attached_api_key_env
            status.update("")
            return
        deployment = candidate.deployment
        base_url.value = f"http://127.0.0.1:{self.defaults.deployment_port}/v1"
        endpoint_model.value = candidate.profile_id
        api_key_env.value = ""
        if rebuild_context:
            self._rebuild_context_options(deployment)
        availability = self._managed_availability(
            candidate,
            self._selected_device_index(),
            self._selected_context_tokens(),
        )
        deployable = () if availability is None else availability.deployable_offers
        if availability is not None and deployable:
            options = tuple(
                (f"{offer.display_name} · {offer.accelerator_platform}", offer.framework) for offer in deployable
            )
            framework_select.set_options(options)
            framework_select.value = deployable[0].framework
            framework_select.disabled = False
            hardware_name = availability.hardware.accelerator_name or "detected accelerator"
            status.update(
                f"This app will start the model server on this computer ({escape(hardware_name)}). "
                "The model downloads only after you confirm."
            )
            return
        framework_select.set_options((("No compatible framework", UNAVAILABLE_FRAMEWORK_VALUE),))
        framework_select.value = UNAVAILABLE_FRAMEWORK_VALUE
        framework_select.disabled = True
        if availability is not None and availability.device_selection_required:
            # Nothing is wrong with this computer yet, so the picker asks rather than refuses.
            status.update(DEVICE_SELECTION_REQUIRED_MESSAGE)
            return
        reason = (
            "This app cannot start models on this computer."
            if availability is None or availability.reason is None
            else availability.reason
        )
        status.update(f"[b]Can't start this model on this computer.[/b] {escape(reason)}")

    def _rebuild_context_options(self, deployment: ModelDeployment) -> None:
        """Rebuild the context ladder for the selected device, keeping a still-fitting choice.

        A choice that no longer fits the device is replaced by the computed default, so
        changing the device never strands the form on a dead context.
        """
        select = self.query_one("#managed-context-select", ManagedContextSelect)
        ladder = context_ladder(deployment, self._replay_context_floor())
        hardware = self._launch_hardware()
        available_memory_bytes = None if hardware is None else hardware.accelerator_memory_bytes
        previous = select.value
        select.set_options((_context_option_label(deployment, tokens), str(tokens)) for tokens in ladder)
        if (
            isinstance(previous, str)
            and any(str(tokens) == previous for tokens in ladder)
            and context_fit(deployment, int(previous), available_memory_bytes) is not False
        ):
            select.value = previous
        else:
            select.value = str(default_context_tokens(deployment, ladder, available_memory_bytes))

    def _replay_context_floor(self) -> int | None:
        """Return the bundled replay's declared context demand, when it declares one.

        A custom manifest is read off-thread at preflight; this render path stays free
        of user file I/O, so its demand is unknown here.
        """
        replay_id = self._select_text("#replay-workload-select", ReplayWorkloadSelect)
        if replay_id is None or replay_id == CUSTOM_REPLAY_ID:
            return None
        replay = find_bundled_replay(replay_id)
        if replay is None:
            return None
        if replay_id not in self.replay_floor_cache:
            try:
                floor = load_manifest(replay.manifest_path).required_context_tokens
            except (OSError, ValueError):
                floor = None
            self.replay_floor_cache[replay_id] = floor
        return self.replay_floor_cache[replay_id]

    def _selected_context_tokens(self) -> int | None:
        """Return the picked managed context, or None while the picker holds no numeric choice."""
        value = self._select_text("#managed-context-select", ManagedContextSelect)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _chosen_context_tokens(self, deployment: ModelDeployment) -> int:
        """Return the context the form has chosen for this recipe, or its full benchmark context."""
        selected = self._selected_context_tokens()
        if selected is None or not (MINIMUM_CONTEXT_TOKENS <= selected <= deployment.context_tokens):
            return deployment.context_tokens
        return selected

    def _managed_deployment_choice(self) -> ManagedDeploymentChoice | None:
        """Return the validated managed choice for the selected model."""
        candidate = self._selected_candidate()
        if candidate is None:
            return None
        if self.managed_controller is None:
            raise ValueError("managed deployment support is unavailable")
        device_index = self._selected_device_index()
        context_tokens = self._selected_context_tokens()
        availability = self.managed_controller.availability(
            candidate, device_index=device_index, context_tokens=context_tokens
        )
        if availability.device_selection_required:
            raise ManagedDeviceSelectionRequired(DEVICE_SELECTION_REQUIRED_MESSAGE)
        value = self.query_one("#managed-framework-select", ManagedFrameworkSelect).value
        if not isinstance(value, str) or value not in DEPLOYMENT_FRAMEWORK_ORDER:
            raise ValueError("no compatible installed framework is selected")
        framework = one_of(value, DEPLOYMENT_FRAMEWORK_ORDER, "framework")
        if not any(offer.framework == framework for offer in availability.deployable_offers):
            raise ValueError("selected framework is not deployable on this computer")
        with raising_validator_errors():
            return ManagedDeploymentChoice(
                candidate=candidate,
                catalog_as_of=self.catalog.as_of,
                catalog_digest=self.catalog.digest,
                framework=framework,
                device_index=device_index,
                context_tokens=context_tokens,
                cache_root=self.defaults.model_cache_root,
                port=self.defaults.deployment_port,
                startup_timeout_seconds=self.defaults.deployment_startup_timeout_seconds,
            )

    def _candidate_detail(self, candidate: ModelCandidate) -> str:
        # Short keys keep each value on one line beside its key at the full-size panel
        # width; the standing's reason gets its own full-width line above them.
        artifact_gib = candidate.deployment.artifact_size_bytes / BYTES_PER_GIB
        selected_context_tokens = self._chosen_context_tokens(candidate.deployment)
        minimum_memory_need = memory_need_gib(
            derived_minimum_memory_bytes(candidate.deployment, selected_context_tokens)
        )
        context_value = (
            f"{selected_context_tokens:,} of {candidate.deployment.context_tokens:,} tokens · reduced"
            if selected_context_tokens < candidate.deployment.context_tokens
            else f"{candidate.deployment.context_tokens:,} tokens"
        )
        frameworks = " / ".join(framework_display_name(framework) for framework in candidate.deployment.frameworks)
        rows = (
            ("Built for", hardware_target_text(candidate, self.catalog.hardware_of(candidate))),
            ("Runs with", f"{frameworks} · needs {minimum_memory_need} GiB"),
            ("Context", context_value),
            (
                "Download",
                f"{artifact_kind_text(candidate.deployment.artifact_kind)} · {artifact_gib:.1f} GiB · SHA-256 checked",
            ),
            ("HF repo", candidate.hf_repository),
        )
        external_note = (
            "" if self.catalog.is_bundled_snapshot else f"\n\n[{AA_NEUTRAL_500}]{EXTERNAL_CATALOG_PROVENANCE}[/]"
        )
        return (
            f"[b]{escape(candidate.model_name)}[/b] · {escape(recipe_build_text(candidate))}\n"
            f"{self._standing_detail(candidate)}\n\n"
            f"{key_value_block(rows)}"
            f"{external_note}"
        )

    def _standing_detail(self, candidate: ModelCandidate) -> str:
        """Lead the detail pane with a marked standing and, when the recipe cannot start, the reason."""
        availability = self._listing_availability(candidate)
        if availability is None:
            return f"[{AA_NEUTRAL_500}]This computer is checked when the app starts.[/]"
        headline = f"{standing_mark(availability.standing).markup} {escape(standing_headline_text(availability))}"
        reason = standing_reason_text(candidate, availability)
        return headline if reason is None else f"{headline}\n{escape(reason)}"

    def _show(self, step: TuiStep) -> None:
        information_steps = {TuiStep.PRIVACY, TuiStep.METHODOLOGY}
        if step in information_steps and self.step not in information_steps:
            self._information_bookmark = _NavigationBookmark(
                step=self.step,
                focused=self.focused,
                cursor=self.focused.cursor_position if isinstance(self.focused, Input) else None,
                scroll_y=self.query_one(f"#{self.step.value}").scroll_y,
            )
        if self.step is TuiStep.RUN and step is not TuiStep.RUN and self.cancel_armed:
            # An armed cancellation must not outlive the page it was armed on, or a
            # later return would show a stale prompt for a disarmed state.
            self._disarm_cancel()
            self._restore_run_metrics()
        self.step = step
        if step is TuiStep.MODEL:
            # A context or device picked on the setup screen changes what fits, so
            # returning to the model page re-sorts the list and renders the detail again.
            self._refresh_model_list()
            self._refresh_model_detail()
        self.query_one("#content", ContentSwitcher).current = step.value
        self._sync_kitty()
        self.refresh_bindings()
        self.call_after_refresh(self._focus_step)

    def _sync_kitty(self) -> None:
        """Animate only while a run is making progress on a visible, supported-size run page."""
        kitty = self.query_one("#run-kitty", Kitty)
        run_in_progress = self.replay_active and not self.replay_cancelling
        if self.step is TuiStep.RUN and not self.terminal_too_small and kitty.display and run_in_progress:
            kitty.start()
        else:
            kitty.settle()

    def _refresh_model_list(self) -> None:
        """Rebuild the list when a setup change moved a recipe, keeping the highlighted option."""
        listed_recipes = self._listed_recipes()
        if listed_recipes == self.listed_recipes:
            return
        self.listed_recipes = listed_recipes
        model_list = self.query_one("#model-list", OptionList)
        highlighted = None if model_list.highlighted is None else model_list.get_option_at_index(model_list.highlighted)
        model_list.set_options(model_list_options(listed_recipes, self._computer_text()))
        if highlighted is not None and highlighted.id is not None:
            model_list.highlighted = model_list.get_option_index(highlighted.id)

    def _refresh_model_detail(self) -> None:
        """Render the highlighted option's detail from the current form state."""
        model_list = self.query_one("#model-list", OptionList)
        if model_list.highlighted is None:
            return
        option = model_list.get_option_at_index(model_list.highlighted)
        if option.id is not None:
            self._render_option_detail(option.id)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Enable navigation and run shortcuts for the visible page."""
        if action == "toggle_server_log":
            return self._server_log_available()
        steps = ESCAPE_ACTION_STEPS.get(action)
        if steps is None:
            return True
        return self.step in steps

    def _focus_step(self) -> None:
        """Put keyboard focus on the first useful control for the visible step."""
        if self.step is TuiStep.WELCOME:
            self.query_one("#welcome-start", Button).focus()
        elif self.step is TuiStep.MODEL:
            self.query_one("#model-list", OptionList).focus()
        elif self.step is TuiStep.CONFIG:
            self.query_one("#config", VerticalScroll).scroll_home(animate=False)
            manifest = self.query_one("#manifest-input", FieldInput)
            endpoint_model = self.query_one("#endpoint-model-input", Input)
            if self._uses_custom_manifest() and not manifest.value.strip():
                manifest.focus()
            elif self.selection.kind is SelectionKind.CUSTOM_ENDPOINT and not endpoint_model.value.strip():
                endpoint_model.focus()
            else:
                self.query_one("#config-continue", Button).focus()
        elif self.step is TuiStep.PREFLIGHT:
            consent = self.query_one("#endpoint-consent-checkbox", Checkbox)
            if self.request is not None and not consent.disabled:
                consent.focus()
            else:
                self.query_one("#preflight-back", Button).focus()
        elif self.step is TuiStep.RUN:
            # Focus readable content so arrows scroll it and Enter cannot abort a run.
            run_page = self.query_one("#run", Vertical)
            if run_page.has_class("compact", "show-details"):
                self.query_one("#run-metrics-column", VerticalScroll).focus()
            else:
                self.query_one("#run-server-log" if self._showing_server_log() else "#activity-lines", RichLog).focus()
        elif self.step is TuiStep.RESULT:
            new_run = self.query_one("#result-new", Button)
            if new_run.disabled:
                # A submission in flight disables New run, so focus lands on the details
                # toggle rather than staying on whatever page was just hidden.
                self.query_one("#result-details-toggle", Button).focus()
            else:
                new_run.focus()
        elif self.step is TuiStep.PRIVACY:
            self.query_one("#privacy-back", Button).focus()
        elif self.step is TuiStep.METHODOLOGY:
            self.query_one("#methodology-back", Button).focus()

    def action_privacy(self) -> None:
        """Open the privacy page, or close it."""
        self._toggle_information_page(TuiStep.PRIVACY)

    def action_help(self) -> None:
        """Open help, even while a text field has focus, or close it."""
        self._toggle_information_page(TuiStep.METHODOLOGY)

    def _toggle_information_page(self, step: TuiStep) -> None:
        if self.step is step:
            self._go_back()
        else:
            self._show(step)

    async def action_back(self) -> None:
        """Move back one screen or arm cancellation of the active run."""
        self._go_back()

    def _go_back(self) -> None:
        """Apply the visible step's escape meaning: quit, one screen back, arm cancel, or a new run."""
        if self.step is TuiStep.WELCOME:
            self.exit(self.outcome)
        elif self.step is TuiStep.MODEL:
            self._show(TuiStep.WELCOME)
        elif self.step is TuiStep.CONFIG:
            self._show(self.setup_return_step)
        elif self.step is TuiStep.PREFLIGHT:
            self._leave_preflight()
            self._show(TuiStep.CONFIG)
        elif self.step is TuiStep.RUN:
            self._arm_or_cancel_replay()
        elif self.step is TuiStep.RESULT:
            self._prepare_next_run()
        else:
            bookmark = self._information_bookmark
            self._information_bookmark = None
            self._show(TuiStep.WELCOME if bookmark is None else bookmark.step)
            if bookmark is not None:
                self.call_after_refresh(self._restore_information_focus, bookmark)

    def _restore_information_focus(self, bookmark: _NavigationBookmark) -> None:
        """Return to the control the user left, or the one their page chose while they were away."""
        focused = bookmark.focused
        if focused is not None and focused in self.screen.focus_chain:
            focused.focus(scroll_visible=False)
            self.query_one(f"#{self.step.value}").scroll_to(y=bookmark.scroll_y, animate=False)
            if isinstance(focused, Input) and bookmark.cursor is not None:
                focused.call_after_refresh(setattr, focused, "cursor_position", bookmark.cursor)

    def _focus_on_page(self, step: TuiStep, selector: str) -> None:
        """Focus a control on its page now, or make it the return target while Help or Privacy is open."""
        bookmark = self._information_bookmark
        if self.step is step:
            self.query_one(selector).focus()
        elif bookmark is not None and bookmark.step is step:
            self._information_bookmark = replace(bookmark, focused=self.query_one(selector), cursor=None)

    def _page_focus(self) -> Widget | None:
        """Return the focused control of the working page, looking past an open Help or Privacy page."""
        bookmark = self._information_bookmark
        return self.focused if bookmark is None else bookmark.focused

    async def action_back_quit(self) -> None:
        """Apply escape on the welcome screen, where it quits."""
        await self.action_back()

    async def action_back_cancel(self) -> None:
        """Apply escape on the run screen, where it arms cancellation."""
        await self.action_back()

    async def action_back_new_run(self) -> None:
        """Apply escape on the result screen, where it starts the next run's setup."""
        await self.action_back()

    def _arm_or_cancel_replay(self) -> None:
        """Cancel only on the second escape inside the arming window."""
        if self.replay_finalizing or self.replay_cancelling:
            self._cancel_replay()
            return
        if self.cancel_armed:
            self._cancel_replay()
            return
        self.cancel_armed = True
        self.query_one("#run-metrics", Static).update(CANCEL_CONFIRM_MESSAGE)
        if self.cancel_arming_timer is not None:
            self.cancel_arming_timer.stop()
        self.cancel_arming_timer = self.set_timer(CANCEL_ARMING_WINDOW_SECONDS, self._expire_cancel_arming)

    def _disarm_cancel(self) -> None:
        """Drop the armed cancellation without touching the run page."""
        self.cancel_armed = False
        if self.cancel_arming_timer is not None:
            self.cancel_arming_timer.stop()
            self.cancel_arming_timer = None

    def _expire_cancel_arming(self) -> None:
        """Let an unconfirmed cancellation lapse and restore the metrics line."""
        self.cancel_arming_timer = None
        if not self.cancel_armed:
            return
        self.cancel_armed = False
        # The repaint is unconditional: a prompt left behind on a hidden run page would
        # otherwise reappear stale when the user returns.
        self._restore_run_metrics()

    def _restore_run_metrics(self) -> None:
        """Repaint the status line the armed-cancel prompt replaced, matching run state."""
        if self.replay_finalizing:
            text = RUN_SAVING_MESSAGE
        elif self.replay_cancelling:
            text = RUN_CLEANUP_MESSAGE
        else:
            text = self.run_metrics_text
        self.query_one("#run-metrics", Static).update(text)

    async def action_quit_or_type_q(self) -> None:
        """Type into an input or apply the plain-key quit shortcut."""
        focused = self.focused
        if isinstance(focused, Input):
            focused.insert_text_at_cursor("q")
            return
        if self.replay_active and not self.replay_cancelling and not self.replay_finalizing:
            # One stray q must not abort a run any more than one stray Esc, so q joins
            # the same two-press arming; ctrl+c stays the immediate cancel.
            self._arm_or_cancel_replay()
            return
        if self.upload_active:
            # A plain q waits for the upload; ctrl+c is the deliberate interrupt.
            self._show_upload_quit_hint()
            return
        await self.action_quit()

    async def action_quit(self) -> None:
        """Cancel an active replay, arm a force quit, or leave from an idle screen."""
        if self.upload_active:
            # The bundle stays on disk, so an interrupted upload costs one retry command.
            self.workers.cancel_group(self, "upload")
            self._finish_upload()
            self.exit(self.outcome)
            return
        if self.replay_cancelling and not self.replay_finalizing:
            # Owned teardown can wedge on an unresponsive child, so quitting must stay reachable.
            if self.force_quit_armed:
                self.exit(self.outcome)
                return
            self.force_quit_armed = True
            self.query_one("#run-metrics", Static).update(FORCE_QUIT_MESSAGE)
            return
        if self.replay_active:
            self._cancel_replay()
            return
        self.exit(self.outcome)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Preview model intent while keyboard focus moves."""
        option_id = event.option_id
        if option_id is None:
            return
        try:
            self._render_option_detail(option_id)
        except NoMatches:
            return

    def _render_option_detail(self, option_id: str) -> None:
        """Render one option without committing selection intent."""
        candidate = self._candidate(option_id)
        detail = self.query_one("#model-detail", Static)
        if candidate is None:
            detail.update(CUSTOM_ENDPOINT_DETAIL)
        else:
            detail.update(self._candidate_detail(candidate))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Select the model and advance to setup."""
        option_id = event.option_id
        if option_id is None:
            return
        self._select_option(option_id, return_step=TuiStep.MODEL)
        self._show(TuiStep.CONFIG)

    def _select_option(self, option_id: str, *, return_step: TuiStep) -> None:
        """Commit one option as selection intent; Back from setup leads to return_step."""
        self.setup_return_step = return_step
        # The list follows the selection, so a later return to the model page shows the chosen option.
        model_list = self.query_one("#model-list", OptionList)
        model_list.highlighted = model_list.get_option_index(option_id)
        self._remember_attached_endpoint_fields()
        candidate = self._candidate(option_id)
        if candidate is None:
            self.selection = BenchmarkSelection.custom(self._custom_endpoint_model())
        else:
            self.selection = BenchmarkSelection.from_candidate(candidate, self.catalog)
        self.query_one("#endpoint-model-input", Input).value = self.selection.endpoint_model
        self._sync_managed_deployment_controls()

    def _remember_attached_endpoint_fields(self) -> None:
        """Keep the user's own server URL and API-key variable across a model change."""
        base_url = self.query_one("#base-url-input", Input)
        api_key_env = self.query_one("#api-key-env-input", Input)
        if base_url.disabled:
            # A managed selection owns these fields, so their values are not the user's.
            return
        self.attached_base_url = base_url.value
        self.attached_api_key_env = api_key_env.value

    def _custom_endpoint_model(self) -> str:
        """Return the model name that follows the user into the custom endpoint field."""
        previous_candidate = self._selected_candidate()
        if previous_candidate is not None:
            # The managed alias only names the server this app launches, so no attached server serves it.
            return self.defaults.endpoint_model or ""
        configured_model = self.query_one("#endpoint-model-input", Input).value.strip()
        preserve_configured_model = (
            self.selection.kind is SelectionKind.CUSTOM_ENDPOINT or configured_model != self.selection.endpoint_model
        )
        return configured_model if preserve_configured_model else (self.defaults.endpoint_model or "")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Offer Run once the user accepts the network action and, for a user-run server, it has answered."""
        if event.checkbox.id == "submit-checkbox":
            self.submit_requested = event.checkbox.value
            return
        if event.checkbox.id != "endpoint-consent-checkbox":
            return
        run = self.query_one("#run-start", Button)
        server_check = self.query_one("#preflight-server-check", SpinnerLine)
        if not event.checkbox.value or self.request is None:
            run.disabled = True
            self.workers.cancel_group(self, "probe")
            server_check.clear()
            self.query_one("#preflight-ollama", Static).display = False
            return
        if self.request.managed_deployment is not None:
            # This app starts that server itself, so there is nothing to ask yet.
            run.disabled = False
            return
        run.disabled = True
        self.preflight_probe = None
        server_check.start(SERVER_CHECK_RUNNING_MESSAGE)
        self.execute_endpoint_probe(self.request)

    @work(thread=True, exclusive=True, group="probe", exit_on_error=False)
    def execute_endpoint_probe(self, request: ReplayRequest) -> None:
        """Ask the user's server which models it serves, and whether it is Ollama, off the message loop."""
        try:
            probe = self.controller.probe_endpoint(request)
            # The identity check is one short GET, so Ollama users learn before Run that
            # their result will not be directly comparable.
            ollama = probe.endpoint_answered and self.controller.detects_ollama(request)
        except Exception:
            self.post_message(EndpointProbeMessage(request, None))
            return
        self.post_message(EndpointProbeMessage(request, probe, ollama))

    def on_endpoint_probe_message(self, message: EndpointProbeMessage) -> None:
        """Show what the server answered and offer Run only when it answered at all."""
        consent = self.query_one("#endpoint-consent-checkbox", Checkbox)
        if message.request is not self.request or not consent.value:
            return
        server_check = self.query_one("#preflight-server-check", SpinnerLine)
        probe = message.probe
        self.preflight_probe = probe
        if probe is None:
            server_check.finish(FAILED_MARK, f"{SERVER_CHECK_FAILED_MESSAGE} · {SERVER_CHECK_RETRY_HINT}")
            return
        mark, text = probe_activity_text(probe)
        reachable = probe.endpoint_answered
        server_check.finish(mark, text if reachable else f"{text} · {SERVER_CHECK_RETRY_HINT}")
        self.query_one("#preflight-ollama", Static).display = message.ollama
        self._derive_evidence(message.request, probe)
        self._update_preflight_evidence(message.request)
        self._update_reduced_context_notice(message.request)
        self.query_one("#run-start", Button).disabled = not reachable
        if reachable and self._page_focus() is consent:
            # Move on to the optional submit box, not past it to Run; a control the user
            # moved to during the check keeps focus.
            self._focus_on_page(TuiStep.PREFLIGHT, "#submit-checkbox")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Reveal the custom path, or rebind the managed launch to another device."""
        if event.select.id == "managed-device-select":
            self._sync_managed_deployment_controls()
            return
        if event.select.id == "managed-context-select":
            # The ladder itself stays; only the fit messaging follows the new choice.
            self._sync_managed_deployment_controls(rebuild_context=False)
            return
        if event.select.id != "replay-workload-select":
            return
        uses_custom_manifest = self._uses_custom_manifest()
        self.query_one("#manifest-row", Horizontal).display = uses_custom_manifest
        if uses_custom_manifest and self.step is TuiStep.CONFIG:
            self.query_one("#manifest-input", Input).focus()
        candidate = self._selected_candidate()
        if candidate is not None:
            # The replay can raise the context floor, so its ladder is rebuilt with the replay.
            self._sync_managed_deployment_controls()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Check setup when Enter is pressed in a setup field."""
        if self.step is TuiStep.CONFIG:
            self._review_preflight()

    def _uses_custom_manifest(self) -> bool:
        """Return whether setup currently targets a user-supplied manifest."""
        return self.query_one("#replay-workload-select", ReplayWorkloadSelect).value == CUSTOM_REPLAY_ID

    def _selected_manifest_path(self) -> Path:
        """Resolve the selected replay without exposing bundled package paths."""
        replay_id = self.query_one("#replay-workload-select", ReplayWorkloadSelect).value
        if replay_id == CUSTOM_REPLAY_ID:
            manifest_value = self.query_one("#manifest-input", Input).value.strip()
            if not manifest_value:
                raise ValueError("custom manifest path must not be empty")
            return Path(manifest_value).expanduser()
        if not isinstance(replay_id, str):
            raise ValueError("replay workload selection is invalid")
        replay = find_bundled_replay(replay_id)
        if replay is None:
            raise ValueError("replay workload selection is invalid")
        return replay.manifest_path

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle navigation and one explicitly confirmed replay start."""
        button_id = event.button.id
        if button_id in BACK_BUTTON_IDS:
            self._go_back()
        elif button_id == "welcome-start":
            self._show(TuiStep.MODEL)
        elif button_id == "welcome-existing":
            self._select_option(SelectionKind.CUSTOM_ENDPOINT.value, return_step=TuiStep.WELCOME)
            self._show(TuiStep.CONFIG)
        elif button_id == "model-continue":
            model_list = self.query_one("#model-list", OptionList)
            if model_list.highlighted is not None:
                option = model_list.get_option_at_index(model_list.highlighted)
                if option.id is not None:
                    self._select_option(option.id, return_step=TuiStep.MODEL)
            self._show(TuiStep.CONFIG)
        elif button_id == "config-continue":
            self._review_preflight()
        elif button_id == "run-start":
            self._start_replay()
        elif button_id == "run-cancel":
            self._cancel_replay()

    def _review_preflight(self) -> None:
        output_value = self.query_one("#output-input", Input).value.strip()
        base_url = self.query_one("#base-url-input", Input).value.strip()
        endpoint_model = self.query_one("#endpoint-model-input", Input).value.strip()
        api_key_env_value = self.query_one("#api-key-env-input", Input).value.strip()
        if not output_value:
            self._block_setup(BLOCK_CODE_MESSAGES[PreflightBlockCode.OUTPUT_DIR_MISSING])
            return
        # Selection intent is only committed once the whole setup is accepted, so a rejected
        # setup can never leave a stale selection behind for the next run.
        downgrades_to_custom = False
        try:
            managed_deployment = self._managed_deployment_choice()
            downgrades_to_custom = managed_deployment is None and (
                self.selection.kind is SelectionKind.CUSTOM_ENDPOINT or endpoint_model != self.selection.endpoint_model
            )
            selection = BenchmarkSelection.custom(endpoint_model) if downgrades_to_custom else self.selection
            client_backend = self._selected_client_backend()
            with raising_validator_errors():
                request = ReplayRequest(
                    manifest_path=self._selected_manifest_path(),
                    # The typed folder is a stable parent; every attempt writes one fresh
                    # run subdirectory, so running again never needs a new folder choice.
                    output_dir=next_run_directory(Path(output_value).expanduser()),
                    base_url=base_url,
                    endpoint_model=endpoint_model,
                    api_key_env=api_key_env_value or None,
                    client_backend=client_backend,
                    tool_choice=None if managed_deployment is not None else self._selected_tool_choice(),
                    selection_kind=selection.kind,
                    catalog_profile_id=selection.profile_id,
                    catalog_digest=selection.catalog_digest,
                    candidate_revision=selection.candidate_revision,
                    managed_deployment=managed_deployment,
                )
        except SetupProblem as problem:
            message = block_code_message(problem.block_code, api_key_env_value or None)
            self._block_setup(SETUP_UNUSABLE_MESSAGE if message is None else message)
            return
        except ManagedLaunchSettingsProblem as problem:
            self._block_setup(MANAGED_LAUNCH_INVALID_TEMPLATE.format(reason=escape(str(problem))))
            return
        except ManagedDeviceSelectionRequired:
            self._block_setup(DEVICE_SELECTION_REQUIRED_MESSAGE)
            return
        except (ValueError, OSError):
            candidate = self._selected_candidate()
            availability = (
                None if candidate is None else self._managed_availability(candidate, self._selected_device_index())
            )
            if candidate is not None and (availability is None or not availability.can_deploy):
                reason = (
                    "This app cannot start models on this computer."
                    if availability is None or availability.reason is None
                    else availability.reason
                )
                status = f"[b]Can't start this model on this computer.[/b]\n{escape(reason)}"
            else:
                status = SETUP_UNUSABLE_MESSAGE
            self._block_setup(status)
            return
        self.selection = selection
        if downgrades_to_custom:
            self.query_one("#model-list", OptionList).highlighted = self._model_option_index(
                SelectionKind.CUSTOM_ENDPOINT.value
            )
        self.request = None
        self.pending_preflight = request
        self.preflight_probe = None
        self.run_context_probe = None
        self._derive_evidence(request, None)
        self._update_preflight_evidence(request)
        self._update_reduced_context_notice(request)
        self._set_preflight_blocked(False)
        self._start_preflight_spinner()
        self.query_one("#endpoint-consent-checkbox", Checkbox).label = self._consent_label(request)
        self._set_preflight_consent_enabled(False)
        self._show(TuiStep.PREFLIGHT)
        self.execute_preflight(request)

    def _start_preflight_spinner(self) -> None:
        """Animate the checking line so a slow probe still reads as alive."""
        self._stop_preflight_spinner()
        self.preflight_spinner_tick = 0
        self.query_one("#preflight-status", Static).update(self._preflight_spinner_text())
        self.preflight_spinner_timer = self.set_interval(
            PREFLIGHT_SPINNER_INTERVAL_SECONDS, self._tick_preflight_spinner
        )

    def _stop_preflight_spinner(self) -> None:
        """Release the spinner once the check resolves or the page is left."""
        if self.preflight_spinner_timer is None:
            return
        self.preflight_spinner_timer.stop()
        self.preflight_spinner_timer = None

    def _tick_preflight_spinner(self) -> None:
        """Swap one equal-width frame; the region never changes, so no layout pass."""
        self.preflight_spinner_tick += 1
        self.query_one("#preflight-status", Static).update(self._preflight_spinner_text(), layout=False)

    def _preflight_spinner_text(self) -> str:
        return f"[{AA_PURPLE}]{spinner_frame(self.preflight_spinner_tick)}[/] {PREFLIGHT_CHECKING_MESSAGE}"

    def _consent_label(self, request: ReplayRequest) -> str:
        """State the action the user is about to authorize, including its cost and destination."""
        managed_deployment = request.managed_deployment
        if managed_deployment is None:
            return CONSENT_ATTACHED_LABEL
        deployment = managed_deployment.candidate.deployment
        artifact_gib = deployment.artifact_size_bytes / BYTES_PER_GIB
        cache_root = home_relative_path_text(managed_deployment.cache_root)
        return (
            f"Download {artifact_gib:.1f} GiB to the Hugging Face cache at {cache_root} "
            "if needed and run the model locally."
        )

    def _display_probe(self) -> ContextProbeResult | None:
        """Return the server check that describes the current flow: the run's own once it exists."""
        return self.preflight_probe if self.run_context_probe is None else self.run_context_probe

    def _update_reduced_context_notice(self, request: ReplayRequest | None) -> None:
        """Show the persistent reduced-context warning exactly when the setup asks for one."""
        notice = self.query_one("#preflight-reduced", Static)
        facts = None if request is None else _reduced_context_facts(request, self._display_probe())
        notice.display = facts is not None
        if facts is not None:
            selected_tokens, full_tokens = facts
            served = "the server reports " if request is not None and request.managed_deployment is None else ""
            notice.update(
                f"Reduced context: {served}{selected_tokens:,} of {full_tokens:,} tokens. "
                "It will be submitted separately from full-context results."
            )

    def _leave_preflight(self) -> None:
        """Drop the pending check and its request so the setup form can be edited again."""
        self.pending_preflight = None
        self.request = None
        self.workers.cancel_group(self, "preflight")
        self._stop_preflight_spinner()
        self._set_preflight_consent_enabled(False)

    def _block_setup(self, message: str) -> None:
        """Show one rejected setup on the readiness page without starting a run."""
        self._leave_preflight()
        self._set_preflight_blocked(True)
        self._update_reduced_context_notice(None)
        self.query_one("#preflight-status", Static).update(f"{message}\n{BLOCKED_ACTION_MESSAGE}")
        self.query_one("#preflight-evidence", Static).update("Nothing was sent.")
        self._show(TuiStep.PREFLIGHT)

    def _set_preflight_blocked(self, blocked: bool) -> None:
        """Style the readiness page for a blocked setup, or restore its ready framing."""
        self.query_one("#preflight-status", Static).set_classes("error-card" if blocked else "card")
        self.query_one("#endpoint-consent-checkbox", Checkbox).display = not blocked
        # A cheerful hero over an error card would contradict it, and the reduced-context
        # notice describes a run this blocked setup can no longer start.
        self.query_one("#preflight-hero", Static).update(PREFLIGHT_BLOCKED_HERO if blocked else PREFLIGHT_READY_HERO)
        if blocked:
            self.query_one("#preflight-reduced", Static).display = False

    def _selected_client_backend(self) -> ClientBackend:
        """Return the closed measured-client selection."""
        value = self.query_one("#client-backend-select", ClientBackendSelect).value
        if value == "python":
            return "python"
        if value == "rust":
            return "rust"
        raise ValueError("measured client selection is invalid")

    def _selected_tool_choice(self) -> ToolChoice | None:
        """Return the closed tool_choice selection; the server default sends no field."""
        value = self.query_one("#tool-choice-select", ToolChoiceSelect).value
        if value == SERVER_DEFAULT_TOOL_CHOICE_VALUE:
            return None
        if value == "none":
            return "none"
        raise ValueError("tool choice selection is invalid")

    def _controller_for(self, request: ReplayRequest) -> ReplayController | ManagedReplayController:
        """Return the controller that owns this request's server: attached, or the app's own."""
        if request.managed_deployment is None:
            return self.controller
        if self.managed_controller is None:
            raise ValueError("managed deployment support is unavailable")
        return self.managed_controller

    @work(thread=True, exclusive=True, group="preflight", exit_on_error=False)
    def execute_preflight(self, request: ReplayRequest) -> None:
        """Run local probes without blocking the Textual message loop."""
        try:
            preflight = self._controller_for(request).preflight(request)
        except Exception:
            self.post_message(PreflightFailedMessage(request))
            return
        self.post_message(PreflightCompletedMessage(request, preflight))

    def on_preflight_completed_message(self, message: PreflightCompletedMessage) -> None:
        """Display only the latest completed preflight."""
        if message.request is not self.pending_preflight:
            return
        self.pending_preflight = None
        self._apply_preflight(message.request, message.preflight)

    def on_preflight_failed_message(self, message: PreflightFailedMessage) -> None:
        """Fail closed without rendering private probe errors."""
        if message.request is not self.pending_preflight:
            return
        self._leave_preflight()
        self._set_preflight_blocked(True)
        self.query_one("#preflight-status", Static).update(
            f"Setup check failed.\nGo back and review the fields.\n{BLOCKED_ACTION_MESSAGE}"
        )
        if self.step is TuiStep.PREFLIGHT:
            self.query_one("#preflight-back", Button).focus()

    def _apply_preflight(self, request: ReplayRequest, preflight: ReplayPreflight) -> None:
        """Apply one completed local preflight to visible evidence state."""
        self._stop_preflight_spinner()
        if preflight.endpoint_scope is not request.endpoint_scope:
            self.request = None
            self._derive_evidence(request, None)
            self._set_preflight_blocked(True)
            self.query_one("#preflight-status", Static).update(
                f"Setup changed while it was being checked.\nGo back and try again.\n{BLOCKED_ACTION_MESSAGE}"
            )
            self._update_preflight_evidence(request)
            self._set_preflight_consent_enabled(False)
            self._focus_on_page(TuiStep.PREFLIGHT, "#preflight-back")
            return
        self.request = request if preflight.ready else None
        blocked_message = block_code_message(preflight.block_code, request.api_key_env)
        if (
            blocked_message is None
            and preflight.block_code is PreflightBlockCode.CONTEXT_BELOW_REPLAY_FLOOR
            and preflight.reason is not None
        ):
            # This reason names only the two token counts, both read from local files.
            blocked_message = f"[b]{escape(preflight.reason)}[/b]\nGo back and pick a larger context."
        input_status = (
            f"{count(preflight.manifest_tasks, 'task')} · {count(preflight.manifest_turns, 'turn')}"
            if preflight.ready
            else (SETUP_UNRUNNABLE_MESSAGE if blocked_message is None else blocked_message)
        )
        self._derive_evidence(request, None)
        self._update_preflight_evidence(request)
        host_status = self._hardware_status(preflight.hardware)
        if host_status is not None and preflight.block_code is not PreflightBlockCode.CLIENT_UNAVAILABLE:
            # Naming a client this run cannot use would contradict the block message above.
            client_status = "Rust client" if request.client_backend == "rust" else "Python client"
            host_status = f"{host_status} · {client_status}"
        status_lines = [input_status]
        if host_status is not None:
            status_lines.append(host_status)
        if not preflight.ready:
            status_lines.append(BLOCKED_ACTION_MESSAGE)
        self._set_preflight_blocked(not preflight.ready)
        self.query_one("#preflight-status", Static).update("\n".join(status_lines))
        self._set_preflight_consent_enabled(preflight.ready)
        self._focus_on_page(TuiStep.PREFLIGHT, "#endpoint-consent-checkbox" if preflight.ready else "#preflight-back")

    def _set_preflight_consent_enabled(self, enabled: bool) -> None:
        """Reset the network acknowledgement and any server check after each setup check."""
        checkbox = self.query_one("#endpoint-consent-checkbox", Checkbox)
        checkbox.value = False
        checkbox.disabled = not enabled
        self.workers.cancel_group(self, "probe")
        self.query_one("#preflight-server-check", SpinnerLine).clear()
        self.query_one("#preflight-ollama", Static).display = False
        self.query_one("#run-start", Button).disabled = True
        self._reset_submit_controls(enabled)

    def _submit_token(self) -> str | None:
        """Return the submit token from the configured variable, or None when it is unset or blank."""
        return read_submit_token(self.defaults.submit_token_env)

    def _reset_submit_controls(self, enabled: bool) -> None:
        """Offer the submit checkbox for every setup that is ready to run."""
        panel = self.query_one("#submit-panel", Vertical)
        checkbox = self.query_one("#submit-checkbox", Checkbox)
        notice = self.query_one("#submit-notice", Static)
        checkbox.value = False
        self.submit_requested = False
        checkbox.disabled = not enabled
        panel.display = enabled
        notice.update(SUBMIT_NOTICE)
        self.workers.cancel_group(self, "revision")
        if enabled and self.request is not None and self._submit_token() is not None:
            self.check_revision_allowlist(self.request)

    @work(thread=True, exclusive=True, group="revision", exit_on_error=False)
    def check_revision_allowlist(self, request: ReplayRequest) -> None:
        """Ask the service, off the message loop, whether this build can reach verified; advisory only."""
        result = check_revision_allowlist(
            collect_source_provenance(),
            base_url=self.defaults.submit_base_url,
        )
        self.post_message(RevisionAdviceMessage(request, result.advice, result.reachable))

    def on_revision_advice_message(self, message: RevisionAdviceMessage) -> None:
        """Append the allowlist advice under the submit checkbox for the setup it describes."""
        if message.request is not self.request:
            return
        notice = self.query_one("#submit-notice", Static)
        if not message.reachable:
            notice.update(SUBMIT_NOTICE + SUBMIT_ALLOWLIST_UNREACHABLE_NOTE)
        elif message.advice is not None:
            notice.update(SUBMIT_NOTICE + SUBMIT_SELF_REPORTED_NOTE)

    def _prepare_next_run(self) -> None:
        """Return to setup with every field kept; the next run names its own fresh folder."""
        if self.upload_active:
            self._show_upload_quit_hint()
            return
        self.request = None
        self.preflight_probe = None
        self.run_context_probe = None
        self.progress_state = RunProgressState()
        self._set_preflight_consent_enabled(False)
        self._show(TuiStep.CONFIG)

    def _derive_evidence(self, request: ReplayRequest, probe: ContextProbeResult | None) -> None:
        """Bind the displayed evidence to one request and the server check that describes it."""
        self.evidence = EvidenceSummary.derive(
            self.selection.kind,
            request.endpoint_scope,
            managed_deployment=request.managed_deployment is not None,
            reduced_context=_request_context_reduced(request, probe),
        )

    def _update_preflight_evidence(self, request: ReplayRequest) -> None:
        """Explain the server scope in plain language."""
        if request.managed_deployment is not None:
            framework = framework_display_name(request.managed_deployment.framework)
            text = (
                f"This app starts the server with {framework} · "
                "the exact model file and GPU startup are checked before the replay"
            )
        elif request.endpoint_scope is EndpointScope.LOOPBACK_NAME:
            text = f"Local server URL · model and GPU not verified\n{_attached_context_text(self._display_probe())}"
        else:
            text = f"Remote server URL · timing includes network delay\n{_attached_context_text(self._display_probe())}"
        self.query_one("#preflight-evidence", Static).update(text)

    def _hardware_status(self, hardware: SafeHardwareSummary | None) -> str | None:
        """Describe this computer, or return None when no probe or launch snapshot exists."""
        summary = self._launch_hardware() if hardware is None else hardware
        if summary is None:
            return None
        warning_suffix = f" · {summary.warning_count} warnings" if summary.warning_count else ""
        if summary.accelerator_name is None:
            return f"This computer: {summary.accelerator_count} accelerators{warning_suffix}"
        if summary.accelerator_memory_bytes is None:
            memory = "memory not reported"
        else:
            memory = f"{summary.accelerator_memory_bytes / BYTES_PER_GIB:.0f} GiB"
        accelerator_name = escape(summary.accelerator_name)
        return f"This computer: {accelerator_name} · {memory}{warning_suffix}"

    def _launch_hardware(self) -> SafeHardwareSummary | None:
        """Reuse the hardware detected at launch so every flow describes the same computer."""
        if self.managed_controller is None:
            return None
        return self.managed_controller.hardware_summary(device_index=self._selected_device_index())

    def _start_replay(self) -> None:
        if self.request is None:
            return
        self.progress_state = RunProgressState()
        self.execution = None
        self.submission_receipt = None
        self.upload_bundle_dir = None
        self.submit_requested = self.query_one("#submit-checkbox", Checkbox).value
        self._reset_result_controls()
        self.replay_active = True
        self.replay_finalizing = False
        self.replay_cancelling = False
        self.force_quit_armed = False
        self._disarm_cancel()
        cancel = self.query_one("#run-cancel", Button)
        cancel.disabled = False
        cancel.label = "Cancel"
        self._reset_run_page(managed=self.request.managed_deployment is not None)
        self.run_generation += 1
        generation = self.run_generation
        self._show(TuiStep.RUN)
        self.execute_replay(generation)

    def _run_step_eyebrow(self, *, preparing: bool = False) -> str:
        """Name the run step, carrying the reduced-context warning through the whole run."""
        base = RUN_STEP_PREPARING_EYEBROW if preparing else RUN_STEP_EYEBROW
        if self._active_run_reduced():
            return f"{base}{RUN_REDUCED_EYEBROW_SUFFIX}"
        return base

    def _active_run_reduced(self) -> bool:
        """Return whether the request driving the current flow runs below the full context."""
        return self.request is not None and _reduced_context_facts(self.request, self._display_probe()) is not None

    def _reset_run_page(self, *, managed: bool) -> None:
        """Clear the previous run from the run page and name the first step of the next one."""
        self.query_one("#run-eyebrow", Static).update(self._run_step_eyebrow(preparing=managed))
        self.query_one("#run-hero", Static).update(RUN_HERO_PREPARING if managed else RUN_HERO_CHECKING_SERVER)
        self.query_one("#run-progress", ProgressBar).update(total=1, progress=0)
        self.query_one("#run-progress-row", Horizontal).display = False
        self.query_one("#run-counters", Static).update(RUN_COUNTERS_PLACEHOLDER)
        self._set_run_metrics(RUN_METRICS_PLACEHOLDER)
        activity = self.query_one("#run-activity", ActivityLog)
        activity.clear()
        activity.set_live(RUN_LIVE_CHECKING_SETUP)
        self.query_one("#run-server-log", RichLog).clear()
        self.query_one("#run", Vertical).remove_class("show-details")
        self._show_server_log(False)
        self.query_one("#run-throughput", HeadlineDigits).show_value(None)
        self._render_context_gauge()
        self._render_live_metrics()
        # The server-log key exists only for a run that owns a server.
        self.refresh_bindings()

    def _set_run_metrics(self, text: str) -> None:
        """Paint the status line and remember it, so an armed-cancel prompt can hand it back."""
        self.run_metrics_text = text
        self.query_one("#run-metrics", Static).update(text)

    def _accept_run_message(self, message: RunPhaseMessage) -> bool:
        """Report whether one run message still describes the active replay."""
        return message.generation == self.run_generation and self.replay_active

    def action_toggle_server_log(self) -> None:
        """Swap the activity log for the owned server's log and back."""
        self._show_server_log(not self._showing_server_log())

    def action_toggle_run_details(self) -> None:
        """Show or hide the secondary charts during a run."""
        page = self.query_one("#run", Vertical)
        page.toggle_class("show-details")
        if page.has_class("compact", "show-details"):
            # A compact page shows either the details or the log; this also refocuses the page.
            self._show_server_log(False)
        else:
            self._focus_step()

    def _showing_server_log(self) -> bool:
        """Return whether the left column shows the owned server's log rather than the activity log."""
        return self.query_one("#run-left-switcher", ContentSwitcher).current == "run-server-log"

    def _show_server_log(self, show: bool) -> None:
        """Show either the owned server's raw log or the activity log in the left column."""
        self.query_one("#run-left-switcher", ContentSwitcher).current = "run-server-log" if show else "run-activity"
        run_page = self.query_one("#run", Vertical)
        if show and run_page.has_class("compact"):
            # A compact page shows either the log or the details; a wide one keeps both.
            run_page.remove_class("show-details")
        framework = (
            None
            if self.request is None or self.request.managed_deployment is None
            else (framework_display_name(self.request.managed_deployment.framework))
        )
        title = RUN_ACTIVITY_TITLE if not show or framework is None else f"{RUN_SERVER_LOG_TITLE} · {framework}"
        self.query_one("#run-left-title", Static).update(title)
        if self.step is TuiStep.RUN:
            self._focus_step()

    def _server_log_available(self) -> bool:
        """Return whether the visible run owns a server whose log can be shown."""
        return self.step is TuiStep.RUN and self.request is not None and self.request.managed_deployment is not None

    def on_run_activity_message(self, message: RunActivityMessage) -> None:
        """Write one run step of the active replay into the activity log."""
        if not self._accept_run_message(message):
            return
        self._render_activity(message.activity)

    def _render_activity(self, activity: RunActivity) -> None:
        """Turn one typed run step into a finished line, a live line, or a new page heading."""
        log = self.query_one("#run-activity", ActivityLog)
        hero = self.query_one("#run-hero", Static)
        kind = activity.kind
        attached = self.request is not None and self.request.managed_deployment is None
        if kind is RunActivityKind.SETUP_CHECKED:
            log.record(DONE_MARK, "Setup checked")
            if attached:
                hero.update(RUN_HERO_CHECKING_SERVER)
                log.set_live("Asking the server which models it serves")
        elif kind is RunActivityKind.SERVER_CHECKED:
            mark, text = probe_activity_text(activity.context_probe)
            log.record(mark, text)
            log.set_live("Preparing the replay")
            if activity.context_probe is not None and self.request is not None:
                # The run bound this observation into its evidence, so the screens follow it.
                self.run_context_probe = activity.context_probe
                self._derive_evidence(self.request, activity.context_probe)
                self.query_one("#run-eyebrow", Static).update(self._run_step_eyebrow())
        elif kind is RunActivityKind.OUTPUT_POLICY_CHOSEN:
            log.record(
                *output_policy_activity_text(activity.ignore_eos_probe, ollama_endpoint=activity.ollama_endpoint)
            )
            log.set_live("Preparing the replay")
        elif kind is RunActivityKind.MODEL_CHECKING:
            hero.update(RUN_HERO_PREPARING)
            log.set_live(f"Checking the model{gib_suffix(activity.artifact_bytes)}")
        elif kind is RunActivityKind.MODEL_READY:
            log.set_live(None)
            source = "downloaded" if activity.downloaded else "from the Hugging Face cache"
            log.record(DONE_MARK, f"Model ready{gib_suffix(activity.artifact_bytes)} · {source} · SHA-256 checked")
        elif kind is RunActivityKind.SERVER_STARTING:
            hero.update(RUN_HERO_STARTING)
            log.set_live(
                f"Starting {framework_text(activity.framework)}{platform_suffix(activity.accelerator_platform)}"
            )
        elif kind is RunActivityKind.SERVER_READY:
            context = "" if activity.context_tokens is None else f" · {activity.context_tokens:,}-token context"
            elapsed = "" if activity.elapsed_seconds is None else f" · {activity.elapsed_seconds:,.1f} s"
            log.record(DONE_MARK, f"Server ready · {framework_text(activity.framework)}{context}{elapsed}")
        elif kind is RunActivityKind.GPU_VERIFIED:
            log.record(DONE_MARK, f"GPU startup verified{platform_suffix(activity.accelerator_platform)}")
        elif kind is RunActivityKind.QUALIFYING:
            log.set_live("Checking the server's agent protocol · synthetic probes")
        elif kind is RunActivityKind.QUALIFIED:
            passed = activity.probes_passed or 0
            total = activity.probes_total or 0
            if passed == total:
                log.record(DONE_MARK, f"Protocol probes passed · {passed}/{total}")
            else:
                log.record(
                    WARNING_MARK,
                    f"Protocol checks recorded · {passed}/{total} passed · submission stays self-reported",
                )
        elif kind is RunActivityKind.POWER_STARTED:
            log.record(DONE_MARK, "GPU power sampling started · nvidia-smi")
        elif kind is RunActivityKind.POWER_RECORDED:
            energy = "" if activity.energy_joules is None else f" · {activity.energy_joules:,.0f} J"
            coverage = "coverage valid" if activity.power_valid else "coverage below threshold"
            log.record(DONE_MARK if activity.power_valid else WARNING_MARK, f"GPU power recorded{energy} · {coverage}")
        elif kind is RunActivityKind.REPLAY_STARTING:
            # The preparing phase is over, so the stepper drops its phase suffix.
            self.query_one("#run-eyebrow", Static).update(self._run_step_eyebrow())
            hero.update(RUN_HERO_RUNNING)
            log.set_live("Starting the replay")
        elif kind is RunActivityKind.SERVER_STOPPING:
            hero.update(RUN_HERO_STOPPING)
            log.set_live("Stopping the server")
        elif kind is RunActivityKind.SERVER_STOPPED:
            log.record(DONE_MARK, "Server stopped")
            log.set_live(None)

    def on_server_log_message(self, message: ServerLogMessage) -> None:
        """Append new owned-server log lines of the active replay."""
        if not self._accept_run_message(message):
            return
        server_log = self.query_one("#run-server-log", RichLog)
        for line in message.lines:
            server_log.write(Text(line, style=AA_NEUTRAL_500))
        self.call_after_refresh(scroll_log_to_end, server_log)

    def on_artifact_progress_message(self, message: ArtifactProgressMessage) -> None:
        """Render managed download progress while the replay is still waiting."""
        if not self._accept_run_message(message) or self.replay_cancelling:
            return
        if message.total_bytes <= 0:
            return
        downloaded_bytes = max(0, min(message.downloaded_bytes, message.total_bytes))
        done_gib = downloaded_bytes / BYTES_PER_GIB
        total_gib = message.total_bytes / BYTES_PER_GIB
        self.query_one("#run-activity", ActivityLog).set_progress(
            f"{RUN_LIVE_DOWNLOADING} · {done_gib:.1f} / {total_gib:.1f} GiB",
            total=message.total_bytes,
            progress=downloaded_bytes,
        )

    def _cancel_replay(self) -> None:
        self._disarm_cancel()
        if self.replay_finalizing:
            self._set_run_metrics(RUN_SAVING_MESSAGE)
            return
        if self.replay_cancelling:
            return
        self.workers.cancel_group(self, "replay")
        self.replay_cancelling = True
        # The run stops making progress the moment cancellation starts, so the
        # kitty settles now rather than blinking through the cleanup wait.
        self._sync_kitty()
        cancel = self.query_one("#run-cancel", Button)
        cancel.disabled = True
        cancel.label = "Stopping…"
        managed = self.request is not None and self.request.managed_deployment is not None
        self.query_one("#run-hero", Static).update(RUN_HERO_STOPPING if managed else RUN_HERO_CANCELLING)
        log = self.query_one("#run-activity", ActivityLog)
        log.record(WARNING_MARK, "Cancel requested")
        log.set_live("Stopping the server" if managed else "Cancelling the replay")
        self._set_run_metrics(RUN_CLEANUP_MESSAGE)

    @work(exclusive=True, group="replay", exit_on_error=False)
    async def execute_replay(self, generation: int) -> None:
        """Await the controller without nesting another event loop."""
        request = self.request
        if request is None:
            self.post_message(ReplayFailedMessage(generation))
            return
        observer: TuiReplayObserver = TextualRunObserver(
            app=self,
            generation=generation,
            finalization_allowed=asyncio.Event(),
        )
        try:
            result = await self._controller_for(request).execute(request, observer)
        except asyncio.CancelledError:
            self.post_message(ReplayCancelledMessage(generation))
            return
        except Exception as error:
            self.post_message(ReplayFailedMessage(generation, cause=_failure_cause(request, error)))
            return
        self.post_message(ReplayCompletedMessage(result, generation))

    def on_run_boundary_message(self, message: RunBoundaryMessage) -> None:
        """Render only reducer output from a coarse runner boundary."""
        if not self._accept_run_message(message):
            return
        event = message.event
        self.progress_state = reduce_run_boundary(self.progress_state, event)
        progress = self.progress_state.progress
        if progress is None:
            return
        log = self.query_one("#run-activity", ActivityLog)
        hero = self.query_one("#run-hero", Static)
        bar = self.query_one("#run-progress", ProgressBar)
        self.query_one("#run-progress-row", Horizontal).display = True
        bar.update(total=max(1, progress.turns), progress=progress.turn)
        self.query_one("#run-counters", Static).update(self._progress_counters(progress))
        if isinstance(event, RunStartedBoundary):
            # The replay has begun, so the stepper drops its preparing suffix here as well.
            self.query_one("#run-eyebrow", Static).update(self._run_step_eyebrow())
            hero.update(RUN_HERO_RUNNING)
            log.record(DONE_MARK, f"Replay loaded · {count(event.tasks, 'task')} · {count(event.turns, 'turn')}")
        elif isinstance(event, TurnStartedBoundary):
            # The request is announced before it is sent, so its size leads the turn.
            log.set_live(turn_live_text(event.turn, event.turns))
        elif isinstance(event, TurnCompletedBoundary):
            sample = self.progress_state.samples[-1]
            log.record(DONE_MARK if event.success else FAILED_MARK, turn_record_text(sample, event))
            if event.turn >= event.turns:
                log.set_live("Finishing the replay")
            self._render_live_metrics()
        else:
            hero.update(RUN_HERO_SAVING)
            mark = DONE_MARK if event.success else WARNING_MARK
            elapsed_seconds = event.elapsed_ms / MILLISECONDS_PER_SECOND
            log.record(
                mark, f"Replay finished · {event.completed_turns}/{event.turns} turns · {elapsed_seconds:,.1f} s"
            )
            log.set_live("Saving results")
        self._render_context_gauge()
        if self.cancel_armed:
            # The armed-cancel prompt owns the status line until it is confirmed or lapses,
            # so only remember the text for the repaint that follows.
            self.run_metrics_text = self._progress_metrics(progress)
        else:
            self._set_run_metrics(self._progress_metrics(progress))

    def _render_context_gauge(self) -> None:
        """Draw the size of the request the reducer says is in flight, at the current width."""
        progress = self.progress_state.progress
        request = self.request
        self.query_one("#run-context", ContextGauge).show_request(
            None if progress is None else progress.request_prompt_tokens,
            context_tokens=None if request is None else _run_context_window(request, self._display_probe()),
            compact=self.query_one(f"#{TuiStep.RUN.value}", Vertical).has_class("compact"),
        )

    def _render_live_metrics(self) -> None:
        """Repaint the charts and the running headline from every closed turn so far."""
        samples = self.progress_state.samples
        series = _closed_turn_series(samples)
        self.query_one("#run-ttft-chart", RangeChart).update_samples(series.ttft_ms)
        self.query_one("#run-decode-chart", RangeChart).update_samples(series.decode_tokens_per_second)
        self.query_one("#run-trend", Sparkline).data = list(series.decode_tokens_per_second)
        rate = cumulative_decode_tokens_per_second(samples)
        throughput = self.query_one("#run-throughput", HeadlineDigits)
        throughput.show_value(rate)
        self.query_one("#run-throughput-caption", Static).update(
            RUN_THROUGHPUT_CAPTION if rate is not None else RUN_THROUGHPUT_WAITING
        )

    def on_replay_finalizing_message(self, message: ReplayFinalizingMessage) -> None:
        """Disable cancellation before the controller commits durable reports."""
        observer = message.observer
        if observer.generation == self.run_generation and self.replay_active:
            self.replay_finalizing = True
            cancel = self.query_one("#run-cancel", Button)
            cancel.disabled = True
            cancel.label = "Saving…"
            self.query_one("#run-hero", Static).update(RUN_HERO_SAVING)
            if self.step is TuiStep.RUN:
                self._focus_step()
            self._set_run_metrics(RUN_SAVING_MESSAGE)
        observer.allow_finalization()

    def _progress_metrics(self, progress: RunProgress) -> str:
        """Word the last turn in one line that fits the narrowest supported terminal."""
        ttft = unit_text(progress.latest_ttft_ms, ",.0f", "ms")
        e2e = unit_text(progress.latest_e2e_ms, ",.0f", "ms")
        rate = rate_text(progress.latest_decode_tokens_per_second)
        return f"Last turn · first token {ttft} · decode {rate} · total {e2e}"

    def _progress_counters(self, progress: RunProgress) -> str:
        """Count tasks and turns beside the bar, with the replay's running time."""
        return (
            f"Task {progress.task}/{progress.tasks} · turn {progress.turn}/{progress.turns} · "
            f"{progress.elapsed_seconds:,.1f}s"
        )

    def on_replay_completed_message(self, message: ReplayCompletedMessage) -> None:
        """Show artifact completion after the controller commits reports."""
        if not self._accept_run_message(message):
            return
        self._settle_replay()
        execution = message.result
        self._show_result_throughput(execution.output_tokens_per_second, successful=execution.success)
        self.execution = execution
        self.outcome = TuiOutcome.SUCCESS if execution.success else TuiOutcome.FAILED
        result_title = self.query_one("#result-title", Static)
        result_title.set_classes("hero success-title" if execution.success else "hero")
        status = self.query_one("#result-status", Static)
        status.set_classes("success-card" if execution.success else "error-card")
        if execution.success:
            result_title.update("Run complete.")
            saved_description = "Results and server startup evidence" if execution.gpu_startup_verified else "Results"
            status.update(f"{saved_description} saved to {result_path_text(execution.output_dir)}.")
        else:
            result_title.update(f"Run finished: {execution.failed_turns} of {execution.total_turns} turns failed.")
            failures_path = result_path_text(execution.artifacts.failures)
            evidence_note = " Server startup evidence was also saved." if execution.gpu_startup_verified else ""
            status.update(f"The server's error messages are in {failures_path}.{evidence_note}")
        self.query_one("#result-metrics", Static).update(self._execution_metrics(execution))
        self.query_one("#result-metrics-detail", Static).update(self._execution_metrics_detail())
        self._render_result_charts()
        self._update_result_evidence()
        self._reveal_result()
        request = self.request
        if execution.success and self.submit_requested and request is not None:
            self._start_upload(execution, message.generation)

    def _reset_result_controls(self) -> None:
        """Hide the previous submission and collapse result details before a run.

        The upload indicators need no reset here: a run can only start once
        _finish_upload has cleared them.
        """
        self.query_one("#result-upload", Static).display = False
        self.query_one("#result-upload-progress", ProgressBar).display = False
        self.query_one("#result-details-toggle", DisclosureButton).expanded = False

    def _start_upload(self, execution: ReplayExecution, generation: int) -> None:
        """Begin the awaited upload and show its stage on the result page."""
        self.upload_active = True
        self.query_one("#result-upload-busy", SpinnerLine).start(UPLOAD_BUSY_MESSAGE)
        self.query_one("#result-new", Button).disabled = True
        card = self.query_one("#result-upload", Static)
        card.set_classes("card")
        card.update(UPLOAD_PREPARING_MESSAGE)
        card.display = True
        bar = self.query_one("#result-upload-progress", ProgressBar)
        bar.update(total=1, progress=0)
        bar.display = True
        self.query_one("#result-next-hint", Static).update(UPLOAD_QUIT_BLOCKED_MESSAGE)
        self.execute_upload(generation, execution.output_dir)

    def _show_upload_quit_hint(self) -> None:
        """Explain why the result page stays put while the upload is in flight."""
        self.query_one("#result-next-hint", Static).update(UPLOAD_QUIT_BLOCKED_MESSAGE)

    def _upload_progress(self, generation: int) -> Callable[[int, int], None]:
        def report(sent_bytes: int, total_bytes: int) -> None:
            self.post_message(UploadProgressMessage(generation, sent_bytes, total_bytes))

        return report

    @work(exclusive=True, group="upload", exit_on_error=False)
    async def execute_upload(self, generation: int, output_dir: Path) -> None:
        """Prepare the bundle beside the run folder, then send it in one request."""
        bundle_dir = output_dir.with_name(f"{output_dir.name}-submission")
        prepared = False
        try:
            await asyncio.to_thread(self._prepare_bundle, output_dir, bundle_dir)
            prepared = True
            receipt = await submit_bundle_async(
                bundle_dir,
                base_url=self.defaults.submit_base_url,
                token=self._submit_token(),
                progress=self._upload_progress(generation),
            )
        except asyncio.CancelledError:
            self.post_message(
                UploadFailedMessage(generation, UPLOAD_CANCELLED_REASON, bundle_dir if prepared else None)
            )
            raise
        except (SubmissionError, ValueError, OSError) as error:
            self.post_message(UploadFailedMessage(generation, error_text(error), bundle_dir if prepared else None))
            return
        self.post_message(UploadCompletedMessage(generation, receipt, bundle_dir))

    @staticmethod
    def _prepare_bundle(output_dir: Path, bundle_dir: Path) -> None:
        validate_bundle_output_path(output_dir, bundle_dir)
        write_submission_bundle(bundle_dir, build_submission_bundle(output_dir))

    def on_upload_progress_message(self, message: UploadProgressMessage) -> None:
        """Advance the upload bar for the active run."""
        if message.generation != self.run_generation or not self.upload_active:
            return
        self.query_one("#result-upload-progress", ProgressBar).update(
            total=max(1, message.total_bytes), progress=message.sent_bytes
        )
        sent = message.total_bytes > 0 and message.sent_bytes >= message.total_bytes
        if sent:
            self.query_one("#result-upload-busy", SpinnerLine).start(UPLOAD_CONFIRMING_MESSAGE)
        self.query_one("#result-upload", Static).update(
            UPLOAD_WAITING_MESSAGE
            if sent
            else UPLOAD_SENDING_TEMPLATE.format(sent=message.sent_bytes, total=message.total_bytes)
        )

    def _finish_upload(self) -> None:
        """Stop the submission spinner and make the next run available."""
        self.upload_active = False
        self.query_one("#result-next-hint", Static).update("")
        self.query_one("#result-upload-busy", SpinnerLine).clear()
        self.query_one("#result-new", Button).disabled = False
        if self.step is TuiStep.RESULT:
            self._focus_step()

    def on_upload_completed_message(self, message: UploadCompletedMessage) -> None:
        """Show the receipt and hand the result page back to the user."""
        if message.generation != self.run_generation:
            return
        self._finish_upload()
        self.submission_receipt = message.receipt
        self.upload_bundle_dir = message.bundle_dir
        receipt = message.receipt
        card = self.query_one("#result-upload", Static)
        card.set_classes("success-card")
        card.update(
            UPLOAD_DONE_TEMPLATE.format(
                submission_id=receipt.submission_id,
                status=receipt.status,
                duplicate="" if receipt.created else UPLOAD_DUPLICATE_NOTE,
            )
        )
        self.query_one("#result-upload-progress", ProgressBar).update(total=1, progress=1)

    def on_upload_failed_message(self, message: UploadFailedMessage) -> None:
        """Keep the bundle on disk and show the exact command that retries the upload."""
        if message.generation != self.run_generation:
            return
        self._finish_upload()
        self.upload_bundle_dir = message.bundle_dir
        card = self.query_one("#result-upload", Static)
        card.set_classes("error-card")
        if message.bundle_dir is None:
            card.update(UPLOAD_NOT_PREPARED_TEMPLATE.format(reason=escape(message.reason)))
        else:
            bundle = result_path_text(message.bundle_dir)
            card.update(UPLOAD_FAILED_TEMPLATE.format(reason=escape(message.reason), bundle=bundle))
        self.query_one("#result-upload-progress", ProgressBar).display = False

    def _render_result_charts(self) -> None:
        """Draw the final distributions from every closed turn; an empty run hides the row."""
        series = _closed_turn_series(self.progress_state.samples)
        self.query_one("#result-details-toggle", Button).display = True
        turn_seconds = tuple(e2e_ms / MILLISECONDS_PER_SECOND for e2e_ms in series.e2e_ms)
        self.query_one("#result-charts", Vertical).set_class(
            not (series.ttft_ms or series.decode_tokens_per_second or turn_seconds), "-empty"
        )
        self.query_one("#result-ttft-chart", RangeChart).update_samples(series.ttft_ms)
        self.query_one("#result-decode-chart", RangeChart).update_samples(series.decode_tokens_per_second)
        self.query_one("#result-e2e-chart", RangeChart).update_samples(turn_seconds)

    def _reveal_result(self) -> None:
        """Show the result page unless the user is reading an information page."""
        self.query_one("#result-kitty", Kitty).settle(happy=self.outcome is TuiOutcome.SUCCESS)
        self.query_one("#result-reduced", Static).display = self._active_run_reduced()
        self.query_one("#result-policy", Static).display = (
            self.execution is not None and not self.execution.comparable_policy
        )
        if self.step in {TuiStep.PRIVACY, TuiStep.METHODOLOGY}:
            self._information_bookmark = _NavigationBookmark(step=TuiStep.RESULT)
            return
        self._show(TuiStep.RESULT)

    def _update_result_evidence(self) -> None:
        if self.execution is not None and self.execution.gpu_startup_verified:
            text = "This app started the server · exact model file and GPU startup evidence saved"
        elif self.request is not None and self.request.managed_deployment is not None:
            # A refusal raised before the child starts writes no log, and the empty run
            # folder is discarded, so pointing at that path would name a missing file.
            log_path = self.request.output_dir / DEPLOYMENT_LOG_FILENAME
            failure = (
                f"The model server run failed · check {result_path_text(log_path)}"
                if log_path.exists()
                else "The model server run failed"
            )
            text = (
                "Model server stopped and cleaned up · no benchmark result saved"
                if self.outcome is TuiOutcome.CANCELLED
                else failure
            )
        elif self.outcome is TuiOutcome.CANCELLED:
            text = "Run cancelled · model and GPU not verified"
        elif self.evidence.partition is ResultPartition.SERVICE_LATENCY_ONLY:
            text = (
                "Remote server · timing includes network delay · saved on this computer · "
                f"model and GPU not verified\n{_attached_context_text(self._display_probe())}"
            )
        else:
            context = _attached_context_text(self._display_probe())
            text = f"Saved on this computer · model and GPU not verified\n{context}"
        self.query_one("#result-evidence", Static).update(text)

    def _settle_replay(self) -> None:
        """Clear the live line and reset the active run flags."""
        self.replay_active = False
        self.replay_finalizing = False
        self.replay_cancelling = False
        self.query_one("#run-activity", ActivityLog).set_live(None)

    def _end_run_without_report(self, outcome: TuiOutcome, mark: str, log_text: str) -> None:
        """Close the run page and blank the result page for a run that saved nothing."""
        self.query_one("#run-activity", ActivityLog).record(mark, log_text)
        self._settle_replay()
        self._show_result_throughput(None)
        self.execution = None
        self.outcome = outcome
        self.query_one("#result-title", Static).set_classes("hero")
        self.query_one("#result-metrics-detail", Static).update("")
        self.query_one("#result-charts", Vertical).add_class("-empty")
        self.query_one("#result-details-toggle", Button).display = False

    def on_replay_failed_message(self, message: ReplayFailedMessage) -> None:
        """Fail safely without rendering private exception data."""
        if not self._accept_run_message(message):
            return
        failed_during_finalization = self.replay_finalizing
        self._end_run_without_report(
            TuiOutcome.FAILED,
            FAILED_MARK,
            "Run stopped early" if message.cause is None else message.cause,
        )
        title = self.query_one("#result-title", Static)
        status = self.query_one("#result-status", Static)
        status.set_classes("error-card")
        result_metrics = self.query_one("#result-metrics", Static)
        if failed_during_finalization:
            title.update("Could not save results.")
            status.update("The run failed.")
            result_metrics.update(self._observed_progress_metrics())
            self._update_result_evidence()
        else:
            title.update("The run stopped early.")
            result_metrics.update("No metrics available.")
            if self.request is not None and self.request.managed_deployment is not None:
                # This server was started by the app itself, so attached-endpoint advice
                # would mislead; the evidence line below keeps the deployment-log hint.
                status.update(MANAGED_RUN_STOPPED_MESSAGE if message.cause is None else escape(message.cause))
                self._update_result_evidence()
            else:
                status.update(RUN_STOPPED_MESSAGE if message.cause is None else escape(message.cause))
                self.query_one("#result-evidence", Static).update("No files were written.")
        self._reveal_result()

    def _show_result_throughput(self, output_tokens_per_second: float | None, *, successful: bool = True) -> None:
        """Show the headline throughput figure only when a committed run measured it.

        The success-green lime is reserved for fully successful runs; a partly failed
        report keeps its number in the neutral tier so it cannot contradict the error card.
        """
        throughput = self.query_one("#result-throughput", HeadlineDigits)
        caption = self.query_one("#result-throughput-caption", Static)
        throughput.display = output_tokens_per_second is not None
        caption.display = output_tokens_per_second is not None
        throughput.set_class(not successful, "-muted")
        throughput.show_value(output_tokens_per_second)

    def _execution_metrics(self, execution: ReplayExecution) -> str:
        """Render the median turn timings of the committed report; the Digits above carry throughput."""
        progress = self.progress_state.progress
        elapsed_seconds = 0.0 if progress is None else progress.elapsed_seconds
        ttft = metric_text(execution.ttft_p50_ms, ",.1f")
        e2e = metric_text(execution.e2e_p50_ms, ",.1f")
        completed_turns = execution.total_turns - execution.failed_turns
        return (
            f"{completed_turns}/{execution.total_turns} turns · {elapsed_seconds:,.1f}s elapsed\n"
            f"median first token {ttft} ms · median turn {e2e} ms"
        )

    def _execution_metrics_detail(self) -> str:
        """Render the p90 and per-turn decode lines from the closed turns the run page charted.

        These apply the same successful-turn rule as the committed report's medians.
        """
        series = _closed_turn_series(self.progress_state.samples)
        ttft_p90 = percentile(series.ttft_ms, P90_PERCENTILE)
        e2e_p90 = percentile(series.e2e_ms, P90_PERCENTILE)
        decode_p50 = percentile(series.decode_tokens_per_second, P50_PERCENTILE)
        decode_p90 = percentile(series.decode_tokens_per_second, P90_PERCENTILE)
        return (
            f"p90 first token {metric_text(ttft_p90, ',.1f')} ms · p90 turn {metric_text(e2e_p90, ',.1f')} ms\n"
            f"decode per turn · p50 {metric_text(decode_p50, ',.0f')} · p90 {metric_text(decode_p90, ',.0f')} tok/s"
        )

    def _observed_progress_metrics(self) -> str:
        """Render the last observed timings when no committed run report exists."""
        progress = self.progress_state.progress
        if progress is None:
            return "Run complete · timing unavailable"
        ttft = unit_text(progress.latest_ttft_ms, ",.1f", "ms")
        e2e = unit_text(progress.latest_e2e_ms, ",.1f", "ms")
        return (
            f"{progress.turn}/{progress.turns} turns · {progress.elapsed_seconds:,.1f}s elapsed\n"
            f"Last turn · first token {ttft} · total {e2e}"
        )

    def on_replay_cancelled_message(self, message: ReplayCancelledMessage) -> None:
        """Show cancellation only after controller cleanup has returned."""
        if not self.replay_cancelling or not self._accept_run_message(message):
            return
        self.run_generation += 1
        self.force_quit_armed = False
        self._end_run_without_report(TuiOutcome.CANCELLED, WARNING_MARK, "Run cancelled")
        title = self.query_one("#result-title", Static)
        title.update("Run cancelled.")
        status = self.query_one("#result-status", Static)
        status.set_classes("cancel-card")
        status.update(
            "No results were saved. Server logs may remain in the run folder."
            if self.request is not None and self.request.managed_deployment is not None
            else "No results were saved."
        )
        self.query_one("#result-metrics", Static).update("No metrics available.")
        self._update_result_evidence()
        self._reveal_result()
