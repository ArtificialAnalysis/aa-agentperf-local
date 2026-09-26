"""Provide the synthetic replay workload shared by runner, controller, and TUI tests."""

from pathlib import Path

from agentperf_local.workload.schema import (
    ManifestTask,
    ReplayManifest,
    RequestMessage,
    TraceRow,
    write_manifest,
    write_trace,
)


def write_replay_workload(
    root: Path,
    *,
    name: str,
    task_count: int = 1,
    write_traces: bool = True,
    required_context_tokens: int | None = None,
) -> Path:
    """Write one replay manifest under root and return its path.

    Task ``index`` is named ``task-{index}`` and holds a single turn prompted with ``prompt {index}``.
    Pass ``write_traces=False`` when a test only reads manifest metadata, so no trace bytes reach the disk.
    Pass ``required_context_tokens`` to declare a context floor on every task.
    """
    tasks: list[ManifestTask] = []
    for index in range(task_count):
        task_id = f"task-{index}"
        trace = Path("traces") / f"{task_id}.jsonl"
        if write_traces:
            write_trace(
                root / trace,
                [
                    TraceRow(
                        turn_id=f"{task_id}:0000",
                        task_id=task_id,
                        conversation_id=task_id,
                        conversation_idx=0,
                        messages=[RequestMessage(role="user", content=f"prompt {index}")],
                        target_output_tokens=2,
                        recorded_prompt_tokens=3,
                        recorded_completion_tokens=2,
                        recorded_total_tokens=5,
                    )
                ],
            )
        tasks.append(
            ManifestTask(
                task_id=task_id,
                trace=trace,
                source_recording=Path(f"{task_id}.json"),
                model_calls=1,
                tool_calls=0,
                total_recorded_tool_delay_ms=0.0,
                required_context_tokens=required_context_tokens,
            )
        )
    manifest_path = root / "manifest.json"
    write_manifest(manifest_path, ReplayManifest(name=name, source="synthetic", tasks=tasks))
    return manifest_path
