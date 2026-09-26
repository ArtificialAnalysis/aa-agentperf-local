"""Track run progress from phase boundaries and render it on a plain terminal.

- `RunProgressState`, `reduce_run_boundary`: the progress state the CLI and the TUI share.
- `TerminalRunObserver`, `render_run_progress`: the CLI's plain-terminal progress view.
- `render_hardware`: the CLI's hardware summary box.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TextIO

from agentperf_local.common.units import BYTES_PER_GIB, MILLISECONDS_PER_SECOND
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import HARDWARE_WARNING_MESSAGES
from agentperf_local.replay.runner import (
    RunBoundaryEvent,
    RunStartedBoundary,
    TurnCompletedBoundary,
    TurnStartedBoundary,
)
from agentperf_local.reports.reporting import pooled_decode_tokens_per_second, turn_decode_tokens_per_second

MINIMUM_WIDTH = 72
MAXIMUM_WIDTH = 118
PERCENT_SCALE = 100.0
PROGRESS_BAR_WIDTH = 28
AA_PURPLE = "\x1b[38;2;136;66;253m"
AA_LILAC = "\x1b[38;2;195;148;255m"
AA_GREEN = "\x1b[38;2;206;255;115m"
AA_AMBER = "\x1b[38;2;200;174;125m"
MUTED = "\x1b[38;2;170;169;178m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"
ANSI_SEQUENCE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


class RunPhase(StrEnum):
    """Name a coarse controller phase shown by the TUI."""

    PREFLIGHT = "preflight"
    MEASURE = "measure"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProgress:
    """Store one phase-boundary UI snapshot."""

    phase: RunPhase
    task: int
    tasks: int
    turn: int
    turns: int
    elapsed_ms: float
    estimated_remaining_ms: float | None
    latest_ttft_ms: float | None
    latest_e2e_ms: float | None
    latest_decode_tokens_per_second: float | None
    peak_accelerator_memory_gib: float | None
    # The recorded prompt size of the request the run announced last: the one now in
    # flight. The recording counted it, so a differently tokenized server may disagree.
    request_prompt_tokens: int | None
    note: str

    def __post_init__(self) -> None:
        """Validate one display snapshot."""
        pairs = (
            (self.task, self.tasks, "task"),
            (self.turn, self.turns, "turn"),
        )
        for current, total, label in pairs:
            if total < 0 or current < 0 or current > total:
                raise ValueError(f"{label} progress must be between zero and its total")
        durations = (self.elapsed_ms, self.estimated_remaining_ms, self.latest_ttft_ms, self.latest_e2e_ms)
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in durations):
            raise ValueError("TUI durations must be finite and non-negative")
        if self.peak_accelerator_memory_gib is not None and (
            not math.isfinite(self.peak_accelerator_memory_gib) or self.peak_accelerator_memory_gib < 0
        ):
            raise ValueError("TUI peak memory must be finite and non-negative")
        if self.request_prompt_tokens is not None and self.request_prompt_tokens < 0:
            raise ValueError("TUI prompt sizes must be non-negative")
        if self.latest_decode_tokens_per_second is not None and (
            not math.isfinite(self.latest_decode_tokens_per_second) or self.latest_decode_tokens_per_second < 0
        ):
            raise ValueError("TUI decode speed must be finite and non-negative")
        if "\n" in self.note or "\r" in self.note:
            raise ValueError("TUI notes must fit on one line")

    @property
    def elapsed_seconds(self) -> float:
        """Return the replay's running time in seconds."""
        return self.elapsed_ms / MILLISECONDS_PER_SECOND


@dataclass(frozen=True, slots=True, kw_only=True)
class RunTurnSample:
    """Store the chart-safe numbers of one closed turn: timings and counts, never content."""

    turn: int
    task: int
    success: bool
    ttft_ms: float | None
    e2e_ms: float | None
    output_tokens: int | None
    generation_time_ms: float | None

    @property
    def decode_tokens_per_second(self) -> float | None:
        """Return this turn's decode speed over its first-to-last-token window, or None when unmeasurable."""
        return turn_decode_tokens_per_second(self.output_tokens, self.generation_time_ms)

    @classmethod
    def from_boundary(cls, event: TurnCompletedBoundary) -> RunTurnSample:
        """Reduce one closed turn to its chart sample."""
        return cls(
            turn=event.turn,
            task=event.task,
            success=event.success,
            ttft_ms=event.time_to_first_token_ms,
            e2e_ms=event.e2e_latency_ms,
            output_tokens=event.output_tokens,
            generation_time_ms=event.generation_time_ms,
        )


def request_size_text(tokens: int | None) -> str:
    """Word one recorded request size, kept approximate because the server tokenizes its own way."""
    return "unrecorded size" if tokens is None else f"about {tokens:,} tokens"


def cumulative_decode_tokens_per_second(samples: Sequence[RunTurnSample]) -> float | None:
    """Return the run-so-far decode speed, pooling successful measurable turns as the final summary does."""
    return pooled_decode_tokens_per_second(
        (sample.output_tokens, sample.generation_time_ms / MILLISECONDS_PER_SECOND)
        for sample in samples
        if sample.success
        and sample.output_tokens is not None
        and sample.generation_time_ms is not None
        and sample.decode_tokens_per_second is not None
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProgressState:
    """Store the latest boundary-derived replay progress and every closed turn's sample."""

    progress: RunProgress | None = None
    samples: tuple[RunTurnSample, ...] = ()


def reduce_run_boundary(state: RunProgressState, event: RunBoundaryEvent) -> RunProgressState:
    """Reduce one privacy-contained runner boundary into display state."""
    latest_ttft_ms = state.progress.latest_ttft_ms if state.progress is not None else None
    latest_e2e_ms = state.progress.latest_e2e_ms if state.progress is not None else None
    latest_decode = state.progress.latest_decode_tokens_per_second if state.progress is not None else None
    # An announced request stays the current one until the next announcement, so a
    # closed turn keeps the size on screen instead of blanking it between requests.
    request_prompt_tokens = state.progress.request_prompt_tokens if state.progress is not None else None
    elapsed_ms = state.progress.elapsed_ms if state.progress is not None else 0.0
    samples = state.samples
    if isinstance(event, RunStartedBoundary):
        progress = RunProgress(
            phase=RunPhase.PREFLIGHT,
            task=0,
            tasks=event.tasks,
            turn=0,
            turns=event.turns,
            elapsed_ms=0.0,
            estimated_remaining_ms=None,
            latest_ttft_ms=None,
            latest_e2e_ms=None,
            latest_decode_tokens_per_second=None,
            peak_accelerator_memory_gib=None,
            request_prompt_tokens=None,
            note="Replay plan loaded; starting the first request",
        )
    elif isinstance(event, TurnStartedBoundary):
        progress = RunProgress(
            phase=RunPhase.MEASURE,
            # The turn is not complete, so the counters still name the work behind it.
            task=event.task,
            tasks=event.tasks,
            turn=event.turn - 1,
            turns=event.turns,
            elapsed_ms=elapsed_ms,
            estimated_remaining_ms=None,
            latest_ttft_ms=latest_ttft_ms,
            latest_e2e_ms=latest_e2e_ms,
            latest_decode_tokens_per_second=latest_decode,
            peak_accelerator_memory_gib=None,
            request_prompt_tokens=event.recorded_prompt_tokens,
            note=(
                f"Turn {event.turn}/{event.turns} sending a request of "
                f"{request_size_text(event.recorded_prompt_tokens)}"
            ),
        )
    elif isinstance(event, TurnCompletedBoundary):
        estimated_remaining_ms = event.elapsed_ms * (event.turns - event.turn) / event.turn
        outcome = "completed" if event.success else "failed"
        sample = RunTurnSample.from_boundary(event)
        samples = (*samples, sample)
        progress = RunProgress(
            phase=RunPhase.MEASURE,
            task=event.task,
            tasks=event.tasks,
            turn=event.turn,
            turns=event.turns,
            elapsed_ms=event.elapsed_ms,
            estimated_remaining_ms=estimated_remaining_ms,
            latest_ttft_ms=event.time_to_first_token_ms,
            latest_e2e_ms=event.e2e_latency_ms,
            latest_decode_tokens_per_second=sample.decode_tokens_per_second,
            peak_accelerator_memory_gib=None,
            request_prompt_tokens=request_prompt_tokens,
            note=f"Task turn {event.task_turn}/{event.task_turns} {outcome}; response metrics derived post-close",
        )
    else:
        progress = RunProgress(
            phase=RunPhase.COMPLETE if event.success else RunPhase.FAILED,
            task=event.completed_tasks,
            tasks=event.tasks,
            turn=event.completed_turns,
            turns=event.turns,
            elapsed_ms=event.elapsed_ms,
            estimated_remaining_ms=0.0,
            latest_ttft_ms=latest_ttft_ms,
            latest_e2e_ms=latest_e2e_ms,
            latest_decode_tokens_per_second=latest_decode,
            peak_accelerator_memory_gib=None,
            request_prompt_tokens=None,
            note="Replay closed; writing local artifacts",
        )
    return RunProgressState(progress=progress, samples=samples)


@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalStyle:
    """Store terminal rendering capabilities."""

    width: int
    color: bool

    def __post_init__(self) -> None:
        """Clamp responsibility stays with the constructor helper."""
        if self.width < MINIMUM_WIDTH or self.width > MAXIMUM_WIDTH:
            raise ValueError("terminal width is outside the supported range")


def terminal_style(*, color: bool | None = None, width: int | None = None) -> TerminalStyle:
    """Detect conservative terminal capabilities."""
    detected_width = shutil.get_terminal_size(fallback=(96, 24)).columns if width is None else width
    resolved_width = min(MAXIMUM_WIDTH, max(MINIMUM_WIDTH, detected_width))
    detected_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    resolved_color = detected_color if color is None else color
    return TerminalStyle(width=resolved_width, color=resolved_color)


def _styled(text: str, style: str, enabled: bool) -> str:
    return f"{style}{text}{RESET}" if enabled else text


def _visible_length(text: str) -> int:
    return len(ANSI_SEQUENCE_PATTERN.sub("", text))


def _truncate(text: str, width: int) -> str:
    """Cut text to a visible width and keep the escape sequences that remain."""
    if _visible_length(text) <= width:
        return text
    if width <= 0:
        return ""
    kept: list[str] = []
    visible = 0
    index = 0
    while index < len(text) and visible < width - 1:
        sequence = ANSI_SEQUENCE_PATTERN.match(text, index)
        if sequence is not None:
            kept.append(sequence.group())
            index = sequence.end()
            continue
        kept.append(text[index])
        visible += 1
        index += 1
    kept.append("…")
    clipped = "".join(kept)
    # A cut can drop the closing reset, so styling must not bleed into the frame border.
    return f"{clipped}{RESET}" if ANSI_SEQUENCE_PATTERN.search(clipped) is not None else clipped


def _box_line(content: str, style: TerminalStyle) -> str:
    inner_width = style.width - 4
    clipped = _truncate(content, inner_width)
    padding = " " * (inner_width - _visible_length(clipped))
    return f"│ {clipped}{padding} │"


def _top(style: TerminalStyle) -> str:
    return f"╭{'─' * (style.width - 2)}╮"


def _divider(style: TerminalStyle) -> str:
    return f"├{'─' * (style.width - 2)}┤"


def _bottom(style: TerminalStyle) -> str:
    return f"╰{'─' * (style.width - 2)}╯"


def _gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / BYTES_PER_GIB:.0f} GiB"


def _duration(value: float | None) -> str:
    if value is None:
        return "—"
    seconds = value / MILLISECONDS_PER_SECOND
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _metric(value: float | None) -> str:
    return "—" if value is None else f"{value:,.1f} ms"


def _memory_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} GiB"


def rate_text(value: float | None) -> str:
    """Format one decode speed with its unit, or an em dash when there was no window."""
    return "—" if value is None else f"{value:,.0f} tok/s"


def _progress_bar(current: int, total: int, style: TerminalStyle) -> str:
    fraction = 0.0 if total == 0 else current / total
    filled = round(PROGRESS_BAR_WIDTH * fraction)
    bar = f"{'━' * filled}{'─' * (PROGRESS_BAR_WIDTH - filled)}"
    percent = fraction * PERCENT_SCALE
    return f"{_styled(bar, AA_PURPLE, style.color)} {percent:5.1f}%"


def render_hardware(snapshot: HardwareSnapshot, *, style: TerminalStyle | None = None) -> str:
    """Render a privacy-safe local hardware preflight card."""
    resolved = terminal_style() if style is None else style
    brand = _styled("ARTIFICIAL ANALYSIS", AA_LILAC + BOLD, resolved.color)
    heading = f"{brand}  /  AGENTPERF LOCAL"
    lines = [_top(resolved), _box_line(heading, resolved)]
    preflight = _styled("PREFLIGHT", AA_GREEN + BOLD, resolved.color)
    lines.append(_box_line(f"{preflight}  Local device inspection", resolved))
    lines.append(_divider(resolved))
    system = f"System       {snapshot.operating_system} {snapshot.operating_system_version} · {snapshot.architecture}"
    processor = f"Processor    {snapshot.cpu_model} · {snapshot.logical_cpu_count or 'unknown'} threads"
    lines.extend((_box_line(system, resolved), _box_line(processor, resolved)))
    lines.append(_box_line(f"Host memory  {_gib(snapshot.memory_bytes)}", resolved))
    if snapshot.accelerators:
        for index, accelerator in enumerate(snapshot.accelerators, start=1):
            memory = "unified / unreported" if accelerator.memory_bytes is None else _gib(accelerator.memory_bytes)
            cores = "unknown cores" if accelerator.core_count is None else f"{accelerator.core_count} cores"
            value = f"GPU {index}        {accelerator.name} · {memory} · {cores} · {accelerator.api or 'unknown API'}"
            lines.append(_box_line(value, resolved))
        ready = len(snapshot.accelerators) == 1
        readiness = "READY" if ready else "CHECK"
        detail = "one accelerator detected" if ready else "benchmark requires one accelerator"
        badge = _styled(readiness, (AA_GREEN if ready else AA_AMBER) + BOLD, resolved.color)
        lines.append(_divider(resolved))
        lines.append(_box_line(f"{badge}  {detail}", resolved))
    else:
        blocked = _styled("CHECK", AA_AMBER + BOLD, resolved.color)
        lines.append(_divider(resolved))
        lines.append(_box_line(f"{blocked}  no supported accelerator detected", resolved))
    for warning in snapshot.warnings:
        # An unmapped code must degrade to its own text rather than break the preflight card.
        message = HARDWARE_WARNING_MESSAGES.get(warning, warning)
        lines.append(_box_line(f"Warning      {message}", resolved))
    lines.append(
        _box_line(
            _styled("PRIVACY", MUTED + BOLD, resolved.color)
            + "  hostnames · serials · UUIDs · PCI addresses · raw probe output omitted",
            resolved,
        )
    )
    lines.append(_bottom(resolved))
    return "\n".join(lines)


def render_run_progress(
    progress: RunProgress,
    *,
    device_label: str,
    suite_label: str,
    style: TerminalStyle | None = None,
) -> str:
    """Render one phase-boundary progress frame."""
    if "\n" in device_label or "\n" in suite_label:
        raise ValueError("TUI labels must fit on one line")
    resolved = terminal_style() if style is None else style
    phase_style = AA_GREEN + BOLD if progress.phase == RunPhase.COMPLETE else AA_LILAC + BOLD
    phase = _styled(progress.phase.value.upper(), phase_style, resolved.color)
    heading = _styled("ARTIFICIAL ANALYSIS", AA_LILAC + BOLD, resolved.color) + "  /  AGENTPERF LOCAL"
    counters = f"task {progress.task}/{progress.tasks}  ·  turn {progress.turn}/{progress.turns}"
    metrics = (
        f"latest TTFT {_metric(progress.latest_ttft_ms)}  ·  E2E {_metric(progress.latest_e2e_ms)}  ·  "
        f"decode {rate_text(progress.latest_decode_tokens_per_second)}  ·  "
        f"peak GPU memory {_memory_metric(progress.peak_accelerator_memory_gib)}"
    )
    timing = (
        f"elapsed {_duration(progress.elapsed_ms)}  ·  "
        f"remaining {_duration(progress.estimated_remaining_ms)}  ·  "
        f"request {request_size_text(progress.request_prompt_tokens)}"
    )
    lines = [
        _top(resolved),
        _box_line(heading, resolved),
        _box_line(f"{device_label}  ·  {suite_label}", resolved),
        _divider(resolved),
        _box_line(f"{phase}  {counters}", resolved),
        _box_line(_progress_bar(progress.turn, progress.turns, resolved), resolved),
        _box_line(metrics, resolved),
        _box_line(timing, resolved),
        _box_line(progress.note or "Waiting for the next phase boundary", resolved),
        _divider(resolved),
        _box_line("LOCAL  no upload · response parsing and evaluation happen after stream close", resolved),
        _bottom(resolved),
    ]
    return "\n".join(lines)


@dataclass(slots=True, kw_only=True)
class TerminalRunObserver:
    """Render privacy-contained frames from post-close run boundaries."""

    device_label: str
    suite_label: str
    stream: TextIO
    style: TerminalStyle | None = None
    state: RunProgressState | None = None

    def __post_init__(self) -> None:
        """Validate stable labels and repetition counters."""
        if "\n" in self.device_label or "\r" in self.device_label:
            raise ValueError("device_label must fit on one line")
        if "\n" in self.suite_label or "\r" in self.suite_label:
            raise ValueError("suite_label must fit on one line")
        self.state = RunProgressState()

    def _style(self) -> TerminalStyle:
        if self.style is not None:
            return self.style
        return terminal_style(color=self.stream.isatty())

    def _write(self, progress: RunProgress) -> None:
        frame = render_run_progress(
            progress,
            device_label=self.device_label,
            suite_label=self.suite_label,
            style=self._style(),
        )
        prefix = "\x1b[H\x1b[J" if self.stream.isatty() else ""
        self.stream.write(f"{prefix}{frame}\n")
        self.stream.flush()

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        """Render one runner boundary event."""
        if self.state is None:
            raise RuntimeError("terminal observer state was not initialized")
        self.state = reduce_run_boundary(self.state, event)
        if self.state.progress is None:
            raise RuntimeError("runner boundary did not produce display progress")
        self._write(self.state.progress)
