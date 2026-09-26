"""Exercise the phase-boundary terminal renderer."""

from io import StringIO

import pytest

from agentperf_local.cli import main
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot, HardwareWarningCode
from agentperf_local.replay.runner import (
    RunFinishedBoundary,
    RunStartedBoundary,
    TurnCompletedBoundary,
    TurnStartedBoundary,
)
from agentperf_local.reports.progress import (
    AA_AMBER,
    ANSI_SEQUENCE_PATTERN,
    RunPhase,
    RunProgress,
    RunProgressState,
    TerminalRunObserver,
    TerminalStyle,
    cumulative_decode_tokens_per_second,
    reduce_run_boundary,
    render_hardware,
    render_run_progress,
)


def _accelerators(count: int) -> tuple[AcceleratorSnapshot, ...]:
    return tuple(
        AcceleratorSnapshot(
            vendor="Apple",
            name="Apple M5 Pro",
            memory_bytes=None,
            core_count=20,
            driver_version=None,
            api="Metal",
        )
        for _ in range(count)
    )


def _hardware(
    accelerator_count: int = 1,
    warnings: tuple[HardwareWarningCode, ...] | None = None,
) -> HardwareSnapshot:
    default_warnings: tuple[HardwareWarningCode, ...] = () if accelerator_count else ("no_supported_accelerator",)
    return HardwareSnapshot(
        operating_system="macOS",
        operating_system_version="26",
        kernel_version="private-kernel",
        architecture="arm64",
        cpu_model="Apple M5 Pro",
        logical_cpu_count=20,
        memory_bytes=64 * 1024**3,
        accelerators=_accelerators(accelerator_count),
        warnings=default_warnings if warnings is None else warnings,
    )


def _progress() -> RunProgress:
    return RunProgress(
        phase=RunPhase.MEASURE,
        task=17,
        tasks=32,
        turn=614,
        turns=1204,
        elapsed_ms=12 * 60 * 1000,
        estimated_remaining_ms=11 * 60 * 1000,
        latest_ttft_ms=232.4,
        latest_e2e_ms=1353.1,
        latest_decode_tokens_per_second=1_284.0,
        peak_accelerator_memory_gib=13.9,
        request_prompt_tokens=47_318,
        note="Turn closed; deriving metrics outside the stream loop",
    )


def test_hardware_card_is_fixed_width_and_privacy_safe() -> None:
    style = TerminalStyle(width=92, color=False)

    rendered = render_hardware(_hardware(), style=style)

    assert "ARTIFICIAL ANALYSIS  /  AGENTPERF LOCAL" in rendered
    assert "PREFLIGHT  Local device inspection" in rendered
    assert "Apple M5 Pro · unified / unreported · 20 cores · Metal" in rendered
    assert "READY  one accelerator detected" in rendered
    assert "hostnames · serials · UUIDs" in rendered
    assert "private-kernel" not in rendered
    assert "\x1b[" not in rendered
    assert all(len(line) == style.width for line in rendered.splitlines())


def test_progress_frame_renders_boundary_metrics_without_private_content() -> None:
    style = TerminalStyle(width=118, color=False)

    rendered = render_run_progress(
        _progress(),
        device_label="M5 Pro · 20 GPU cores · 64 GiB unified",
        suite_label="Core / 2026-Q4",
        style=style,
    )

    assert "MEASURE  task 17/32  ·  turn 614/1204" in rendered
    assert "51.0%" in rendered
    assert "latest TTFT 232.4 ms" in rendered
    assert "peak GPU memory 13.9 GiB" in rendered
    assert "request about 47,318 tokens" in rendered
    assert "response parsing and evaluation happen after stream close" in rendered
    assert all(len(line) == style.width for line in rendered.splitlines())


def test_color_is_opt_in_and_progress_values_are_validated() -> None:
    rendered = render_hardware(_hardware(), style=TerminalStyle(width=92, color=True))
    assert "\x1b[38;2;195;148;255m" in rendered

    with pytest.raises(ValueError, match="turn progress"):
        RunProgress(
            phase=RunPhase.MEASURE,
            task=1,
            tasks=1,
            turn=2,
            turns=1,
            elapsed_ms=0,
            estimated_remaining_ms=None,
            latest_ttft_ms=None,
            latest_e2e_ms=None,
            latest_decode_tokens_per_second=None,
            peak_accelerator_memory_gib=None,
            request_prompt_tokens=None,
            note="",
        )


def test_terminal_observer_renders_only_post_close_boundary_fields() -> None:
    stream = StringIO()
    observer = TerminalRunObserver(
        device_label="RTX 5090 · 32 GiB",
        suite_label="Core · 2026-Q4",
        stream=stream,
        style=TerminalStyle(width=118, color=False),
    )

    observer.on_boundary(RunStartedBoundary(tasks=2, turns=4))
    observer.on_boundary(
        TurnStartedBoundary(
            task=1,
            tasks=2,
            task_turn=1,
            task_turns=2,
            turn=1,
            turns=4,
            recorded_prompt_tokens=47_318,
        )
    )
    observer.on_boundary(
        TurnCompletedBoundary(
            task=1,
            tasks=2,
            task_turn=1,
            task_turns=2,
            turn=1,
            turns=4,
            elapsed_ms=1_000.0,
            time_to_first_token_ms=100.0,
            e2e_latency_ms=300.0,
            output_tokens=101,
            generation_time_ms=200.0,
            success=True,
        )
    )
    observer.on_boundary(
        RunFinishedBoundary(
            completed_tasks=2,
            tasks=2,
            completed_turns=4,
            turns=4,
            elapsed_ms=4_000.0,
            success=True,
        )
    )

    rendered = stream.getvalue()
    assert rendered.count("ARTIFICIAL ANALYSIS  /  AGENTPERF LOCAL") == 4
    assert "PREFLIGHT" in rendered
    # The announced request is drawn before its turn closes, and the size stays with the closed turn.
    assert "MEASURE  task 1/2  ·  turn 0/4" in rendered
    assert rendered.count("request about 47,318 tokens") == 2
    assert "MEASURE  task 1/2  ·  turn 1/4" in rendered
    assert "remaining 3.0s" in rendered
    assert "COMPLETE  task 2/2  ·  turn 4/4" in rendered
    assert rendered.count("latest TTFT 100.0 ms  ·  E2E 300.0 ms  ·  decode 500 tok/s") == 2
    assert "prompt" not in rendered
    assert "\x1b[" not in rendered


def test_doctor_uses_terminal_card_and_preserves_json_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("agentperf_local.cli.inspect.collect_hardware_snapshot", _hardware)

    assert main(["doctor"]) == 0
    assert "PREFLIGHT  Local device inspection" in capsys.readouterr().out

    assert main(["doctor", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"kind": "hardware_snapshot"' in output
    assert "PREFLIGHT" not in output


@pytest.mark.parametrize(
    ("accelerator_count", "expected_badge", "expected_style"),
    ((1, "READY", "\x1b[38;2;206;255;115m"), (2, "CHECK", AA_AMBER), (0, "CHECK", AA_AMBER)),
)
def test_blocking_hardware_findings_are_amber(
    accelerator_count: int,
    expected_badge: str,
    expected_style: str,
) -> None:
    rendered = render_hardware(_hardware(accelerator_count), style=TerminalStyle(width=92, color=True))

    assert f"{expected_style}\x1b[1m{expected_badge}\x1b[0m" in rendered


@pytest.mark.parametrize(
    ("warning", "expected_message"),
    (
        ("amd_smi_failed", "amd-smi or rocm-smi failed"),
        ("amd_smi_unparseable", "amd-smi or rocm-smi returned no parseable accelerators"),
    ),
)
def test_amd_probe_warnings_render_as_readable_text(warning: HardwareWarningCode, expected_message: str) -> None:
    snapshot = _hardware(0, warnings=(warning, "no_supported_accelerator"))

    rendered = render_hardware(snapshot, style=TerminalStyle(width=92, color=False))

    assert f"Warning      {expected_message}" in rendered
    assert "Warning      No supported accelerator was detected" in rendered


def test_colored_frames_keep_the_visible_text_the_plain_frames_keep() -> None:
    colored = render_run_progress(
        _progress(),
        device_label="M5 Pro · 20 GPU cores",
        suite_label="Core / 2026-Q4",
        style=TerminalStyle(width=72, color=True),
    )
    plain = render_run_progress(
        _progress(),
        device_label="M5 Pro · 20 GPU cores",
        suite_label="Core / 2026-Q4",
        style=TerminalStyle(width=72, color=False),
    )

    stripped = ANSI_SEQUENCE_PATTERN.sub("", colored)
    assert "MEASURE  task 17/32  ·  turn 614/1204" in plain
    assert stripped == plain
    assert all(len(line) == 72 for line in stripped.splitlines())


def _turn(turn: int, *, output_tokens: int | None, generation_time_ms: float | None, success: bool = True):
    return TurnCompletedBoundary(
        task=1,
        tasks=1,
        task_turn=turn,
        task_turns=3,
        turn=turn,
        turns=3,
        elapsed_ms=turn * 1_000.0,
        time_to_first_token_ms=100.0 * turn,
        e2e_latency_ms=900.0,
        output_tokens=output_tokens,
        generation_time_ms=generation_time_ms,
        success=success,
    )


def test_reducer_keeps_one_chart_sample_per_closed_turn_with_the_shared_decode_rule() -> None:
    state = reduce_run_boundary(RunProgressState(), RunStartedBoundary(tasks=1, turns=3))
    state = reduce_run_boundary(state, _turn(1, output_tokens=201, generation_time_ms=400.0))
    state = reduce_run_boundary(state, _turn(2, output_tokens=1, generation_time_ms=400.0))
    state = reduce_run_boundary(state, _turn(3, output_tokens=None, generation_time_ms=None, success=False))
    finished = reduce_run_boundary(
        state,
        RunFinishedBoundary(completed_tasks=1, tasks=1, completed_turns=3, turns=3, elapsed_ms=3_000.0, success=False),
    )

    assert [sample.turn for sample in finished.samples] == [1, 2, 3]
    assert cumulative_decode_tokens_per_second(finished.samples) == 500.0
    assert cumulative_decode_tokens_per_second(()) is None
    assert [sample.decode_tokens_per_second for sample in finished.samples] == [500.0, None, None]
    assert [sample.ttft_ms for sample in finished.samples] == [100.0, 200.0, 300.0]
    assert [sample.success for sample in finished.samples] == [True, True, False]
    assert finished.progress is not None
    assert finished.progress.latest_decode_tokens_per_second is None
    assert state.progress is not None and state.progress.phase is RunPhase.MEASURE
