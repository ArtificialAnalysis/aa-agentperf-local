"""Render deterministic Textual pilot screens as SVG files."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from textual.widgets import Button, Checkbox, OptionList

from agentperf_local.common.units import BYTES_PER_GIB
from agentperf_local.deployment.catalog import BUNDLED_MODEL_CATALOG_PATH, load_model_catalog
from agentperf_local.deployment.endpoint_probes import ContextProbeResult
from agentperf_local.deployment.managed_run import RunActivity, RunActivityKind
from agentperf_local.provenance.context import ContextObservationReason
from agentperf_local.replay.runner import (
    RunFinishedBoundary,
    RunStartedBoundary,
    TurnCompletedBoundary,
    TurnStartedBoundary,
)
from agentperf_local.reports.progress import MAXIMUM_WIDTH
from agentperf_local.reports.reporting import ArtifactPaths
from agentperf_local.tui.app import (
    MINIMUM_SUPPORTED_HEIGHT,
    MINIMUM_SUPPORTED_WIDTH,
    AgentPerfLocalApp,
    TuiDefaults,
)
from agentperf_local.tui.evidence import EndpointScope
from agentperf_local.tui.labels import PATH_WRAP_BREAK
from agentperf_local.tui.replay_contract import (
    ReplayExecution,
    ReplayPreflight,
    ReplayRequest,
    SafeHardwareSummary,
    TuiReplayObserver,
)

DEFAULT_WIDTH = MAXIMUM_WIDTH
DEFAULT_HEIGHT = 36
PREVIEW_TASKS = 8
PREVIEW_TURNS = 32
PREVIEW_TASK_TURNS = 4
PREVIEW_TASK_TURN = 2
PREVIEW_ACCELERATOR_MEMORY_GIB = 32
PREVIEW_TOOL_DELAY_MS = 220.0
HEADLINE_SETTLE_SECONDS = 1.0
PREVIEW_TTFT_MS = 193.4
PREVIEW_E2E_MS = 1_624.8
PREVIEW_RUN_ELAPSED_MS = 86_210.0
PREVIEW_OUTPUT_TOKENS_PER_SECOND = 1_284.0
PREVIEW_SERVED_CONTEXT_TOKENS = 131_072
# The request in flight when the screenshot is taken: the first of a new task.
PREVIEW_PENDING_TASK = 4
PREVIEW_PENDING_PROMPT_TOKENS = 12_900
# Ten closed turns with a plausible spread, so the charts and the log have shape in the screenshot.
# Columns: task, first-token ms, end-to-end ms, output tokens, decode window ms, recorded prompt
# size, success. The prompt grows through a task and starts small again at the next one.
PREVIEW_TURN_TABLE = (
    (1, 188.0, 1_410.0, 1_602, 1_205.0, 14_100, True),
    (1, 196.0, 2_230.0, 2_640, 2_015.0, 38_600, True),
    (1, 204.0, 980.0, 990, 770.0, 61_200, True),
    (1, 191.0, 1_760.0, 2_010, 1_552.0, 84_900, True),
    (2, 240.0, 3_120.0, 3_680, 2_860.0, 15_800, True),
    (2, 187.0, 1_190.0, 1_270, 995.0, 44_300, True),
    (2, 199.0, 1_520.0, 1_690, 1_308.0, 72_600, True),
    (3, 312.0, 2_040.0, 2_210, 1_712.0, 16_400, True),
    (3, 193.4, 1_624.8, 1_838, 1_431.4, 47_900, True),
    (3, 205.0, 1_480.0, 1_610, 1_262.0, 96_200, True),
)


def _announcement(*, task: int, turn: int, prompt_tokens: int) -> TurnStartedBoundary:
    """Announce one preview request at the run's fixed preview shape."""
    return TurnStartedBoundary(
        task=task,
        tasks=PREVIEW_TASKS,
        task_turn=PREVIEW_TASK_TURN,
        task_turns=PREVIEW_TASK_TURNS,
        turn=turn,
        turns=PREVIEW_TURNS,
        recorded_prompt_tokens=prompt_tokens,
    )


@dataclass(slots=True, kw_only=True)
class PreviewReplayController:
    """Render safe example states without reading files or contacting a server."""

    release_run: asyncio.Event = field(default_factory=asyncio.Event)
    release_finalization: asyncio.Event = field(default_factory=asyncio.Event)

    def preflight(self, request: ReplayRequest) -> ReplayPreflight:
        """Return a deterministic local preflight summary."""
        return ReplayPreflight(
            ready=True,
            manifest_tasks=PREVIEW_TASKS,
            manifest_turns=PREVIEW_TURNS,
            endpoint_scope=EndpointScope.LOOPBACK_NAME,
            reason=None,
            hardware=SafeHardwareSummary(
                operating_system="Linux",
                architecture="x86_64",
                accelerator_count=1,
                accelerator_name="NVIDIA GeForce RTX 5090",
                accelerator_memory_bytes=PREVIEW_ACCELERATOR_MEMORY_GIB * BYTES_PER_GIB,
                warning_count=0,
            ),
        )

    def probe_endpoint(self, request: ReplayRequest) -> ContextProbeResult:
        """Answer as a local server that lists the model at the full benchmark context."""
        del request
        return ContextProbeResult(
            observed_tokens=PREVIEW_SERVED_CONTEXT_TOKENS, reason=ContextObservationReason.REPORTED
        )

    def detects_ollama(self, request: ReplayRequest) -> bool:
        """Answer as a server that is not Ollama."""
        del request
        return False

    async def execute(self, request: ReplayRequest, observer: TuiReplayObserver) -> ReplayExecution:
        """Hold a representative run until its screenshot is captured."""
        observer.on_activity(RunActivity(kind=RunActivityKind.SETUP_CHECKED))
        observer.on_activity(
            RunActivity(kind=RunActivityKind.SERVER_CHECKED, context_probe=self.probe_endpoint(request))
        )
        observer.on_activity(RunActivity(kind=RunActivityKind.REPLAY_STARTING))
        observer.on_boundary(RunStartedBoundary(tasks=PREVIEW_TASKS, turns=PREVIEW_TURNS))
        elapsed_ms = 0.0
        for turn, (task, ttft_ms, e2e_ms, output_tokens, generation_ms, prompt_tokens, success) in enumerate(
            PREVIEW_TURN_TABLE, start=1
        ):
            elapsed_ms += e2e_ms + PREVIEW_TOOL_DELAY_MS
            observer.on_boundary(_announcement(task=task, turn=turn, prompt_tokens=prompt_tokens))
            observer.on_boundary(
                TurnCompletedBoundary(
                    task=task,
                    tasks=PREVIEW_TASKS,
                    task_turn=PREVIEW_TASK_TURN,
                    task_turns=PREVIEW_TASK_TURNS,
                    turn=turn,
                    turns=PREVIEW_TURNS,
                    elapsed_ms=elapsed_ms,
                    time_to_first_token_ms=ttft_ms,
                    e2e_latency_ms=e2e_ms,
                    output_tokens=output_tokens,
                    generation_time_ms=generation_ms,
                    success=success,
                )
            )
        # The screenshot is taken with a request in flight, as a running benchmark is.
        observer.on_boundary(
            _announcement(
                task=PREVIEW_PENDING_TASK,
                turn=len(PREVIEW_TURN_TABLE) + 1,
                prompt_tokens=PREVIEW_PENDING_PROMPT_TOKENS,
            )
        )
        await self.release_run.wait()
        observer.on_boundary(
            RunFinishedBoundary(
                completed_tasks=PREVIEW_TASKS,
                tasks=PREVIEW_TASKS,
                completed_turns=PREVIEW_TURNS,
                turns=PREVIEW_TURNS,
                elapsed_ms=PREVIEW_RUN_ELAPSED_MS,
                success=True,
            )
        )
        await observer.on_finalizing()
        await self.release_finalization.wait()
        output_dir = request.output_dir
        return ReplayExecution(
            artifacts=ArtifactPaths(
                turns=output_dir / "turns.jsonl",
                tasks=output_dir / "tasks.json",
                tools=output_dir / "tools.json",
                failures=output_dir / "failures.json",
                summary=output_dir / "summary.json",
            ),
            output_dir=output_dir,
            output_tokens_per_second=PREVIEW_OUTPUT_TOKENS_PER_SECOND,
            ttft_p50_ms=PREVIEW_TTFT_MS,
            e2e_p50_ms=PREVIEW_E2E_MS,
            failed_turns=0,
            total_turns=PREVIEW_TURNS,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render AgentPerf Local Textual screens as SVG files.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser


def _save_screen(app: AgentPerfLocalApp, output_dir: Path, filename: str) -> Path:
    """Save a preview with valid XML and a locally available monospace font."""
    path = output_dir / filename
    # The terminal's path wrap marker is an XML control character, so export a zero-width space.
    screenshot = app.export_screenshot().replace(PATH_WRAP_BREAK, "\u200b").replace("Fira Code", "Menlo")
    path.write_text(screenshot)
    return path


async def _render(output_dir: Path, size: tuple[int, int]) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = PreviewReplayController()
    app = AgentPerfLocalApp(
        load_model_catalog(BUNDLED_MODEL_CATALOG_PATH),
        controller=controller,
        defaults=TuiDefaults(
            output_dir=Path("results/aa-agentic-pilot"),
            endpoint_model="preview-model",
        ),
    )
    rendered: list[Path] = []
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "welcome.svg"))
        app.query_one("#welcome-start", Button).focus()
        await pilot.press("enter")
        # The prefilled served model highlights the custom endpoint; the model page previews
        # a catalog card first, then the flow continues with the endpoint.
        model_list = app.query_one("#model-list", OptionList)
        model_list.highlighted = 0
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "model.svg"))
        model_list.highlighted = len(app.catalog.models)
        app.query_one("#model-continue", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "config.svg"))
        app.query_one("#config-continue", Button).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "preflight.svg"))
        app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
        # Ticking the box asks the server which models it serves; Run appears once it answers.
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "preflight-checked.svg"))
        app.query_one("#run-start", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause(HEADLINE_SETTLE_SECONDS)
        rendered.append(_save_screen(app, output_dir, "run.svg"))
        await pilot.press("d")
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "run-details.svg"))
        await pilot.press("d")
        controller.release_run.set()
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "finalize.svg"))
        controller.release_finalization.set()
        await app.workers.wait_for_complete()
        # The headline counts up over a short ease; the screenshot shows its final value.
        await pilot.pause(HEADLINE_SETTLE_SECONDS)
        rendered.append(_save_screen(app, output_dir, "result.svg"))
        app.query_one("#result-details-toggle", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "result-details.svg"))
        await pilot.press("ctrl+p")
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "privacy.svg"))
        await pilot.press("?")
        await pilot.pause()
        rendered.append(_save_screen(app, output_dir, "methodology.svg"))
    return tuple(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the static pilot screens and print their paths."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.width < MINIMUM_SUPPORTED_WIDTH or arguments.height < MINIMUM_SUPPORTED_HEIGHT:
        parser.error(f"preview size must be at least {MINIMUM_SUPPORTED_WIDTH}x{MINIMUM_SUPPORTED_HEIGHT}")
    rendered = asyncio.run(_render(arguments.output_dir, (arguments.width, arguments.height)))
    for path in rendered:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
