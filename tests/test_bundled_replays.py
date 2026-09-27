"""Exercise replay workloads shipped with the package."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentperf_local.cli.options import DEFAULT_MANAGED_PROFILE_ID
from agentperf_local.client.request import DEFAULT_MAX_OUTPUT_TOKENS
from agentperf_local.deployment.catalog import BUNDLED_RECIPES_ROOT, load_model_catalog
from agentperf_local.deployment.context_policy import REDUCED_CONTEXT_LADDER, reduced_context_rungs
from agentperf_local.provenance.benchmark import (
    BENCHMARK_CONTEXT_TOKENS,
    MEASUREMENT_BINDING_FILENAME,
    SourceProvenance,
    create_attached_submission_context,
    create_measurement_binding,
    write_measurement_binding,
)
from agentperf_local.provenance.context import ContextObservationReason, RunContextFacts
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from agentperf_local.replay.cache_isolation import CACHE_NAMESPACE_DIGITS, cache_namespace_prefix
from agentperf_local.replay.config import OutputTokenPolicy, RunConfig
from agentperf_local.replay.runner import run_manifest
from agentperf_local.reports.reporting import write_run_artifacts
from agentperf_local.submission.bundle import (
    build_submission_bundle,
    validate_submission_bundle,
    write_submission_bundle,
)
from agentperf_local.workload.bundled import BUNDLED_REPLAYS, DEFAULT_BUNDLED_REPLAY, find_bundled_replay
from agentperf_local.workload.schema import load_manifest, load_trace, parse_json_object
from tests.localhost_sse import LocalSseServer
from tests.token_counter import CharacterCounter

MINI_CONTEXT_TOKENS = 8_192
# Count each wire byte as a token and still leave room for server chat templates.
CHAT_TEMPLATE_RESERVE_TOKENS = 1_024
MINI_MAX_OUTPUT_TOKENS = 256
# Flush output in separate reads so the test measures a real decode window.
EVENT_DELAY_SECONDS = 0.01
SHORT_TOOL_RESPONSE = (
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",'
    b'"function":{"name":"search_catalog","arguments":"{\\"topic\\":\\"astronomy\\","}}]},'
    b'"finish_reason":null}]}\n\n',
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
    b'{"arguments":"\\"branch\\":\\"North\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":529,"completion_tokens":23,"total_tokens":552}}\n\n',
    b"data: [DONE]\n\n",
)
SUMMARY_RESPONSE = (
    b'data: {"choices":[{"delta":{"content":"Reserved Pocket Stargazing at North for Alex. "},'
    b'"finish_reason":null}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"Pickup: 2030-03-16; confirmation: RSV204. "},"finish_reason":null}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"Orion is reference only."},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":763,"completion_tokens":62,"total_tokens":825}}\n\n',
    b"data: [DONE]\n\n",
)
TEST_HARDWARE = HardwareSnapshot(
    operating_system="Linux",
    operating_system_version="test-os",
    kernel_version="test-kernel",
    architecture="x86_64",
    cpu_model="test-cpu",
    logical_cpu_count=None,
    memory_bytes=None,
    accelerators=(
        AcceleratorSnapshot(
            vendor="NVIDIA", name="Synthetic GPU", memory_bytes=None, core_count=None, driver_version=None, api="CUDA"
        ),
    ),
    warnings=(),
)


def _exact_response(output_tokens: int) -> tuple[bytes, ...]:
    """Serve a measurable response that fills the requested output budget."""
    return (
        b'data: {"choices":[{"delta":{"content":"Reserved Pocket Stargazing. "},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"Pickup confirmed."},"finish_reason":"length"}]}\n\n',
        f'data: {{"choices":[],"usage":{{"prompt_tokens":763,"completion_tokens":{output_tokens}}}}}\n\n'.encode(),
        b"data: [DONE]\n\n",
    )


def test_mini_bundled_replay_loads_through_the_public_schema() -> None:
    replay = find_bundled_replay("aa-mini-v1")
    assert replay is not None

    manifest = load_manifest(replay.manifest_path)
    rows = load_trace(replay.manifest_path.parent / manifest.tasks[0].trace)

    assert find_bundled_replay("missing") is None
    assert manifest.name == "AA bundled mini replay v1"
    assert len(manifest.tasks) == 1
    assert len(rows) == 6
    assert rows[0].recorded_tool_calls_after[0].tool_name == "search_catalog"


def test_default_bundled_replay_is_the_full_replay_and_loads_through_the_public_schema() -> None:
    replay = DEFAULT_BUNDLED_REPLAY

    manifest = load_manifest(replay.manifest_path)
    rows = [row for task in manifest.tasks for row in load_trace(replay.manifest_path.parent / task.trace)]

    assert BUNDLED_REPLAYS[0] is replay
    assert [item.replay_id for item in BUNDLED_REPLAYS] == ["agentperf-default-v1", "aa-mini-v1"]
    assert find_bundled_replay("agentperf-default-v1") is replay
    assert manifest.name == "AgentPerf default replay v1"
    assert len(manifest.tasks) == 8
    assert sum(task.model_calls for task in manifest.tasks) == 168
    assert sum(task.tool_calls for task in manifest.tasks) == 172
    assert len(rows) == 168
    assert sum(len(row.recorded_tool_calls_after) for row in rows) == 172
    pinchbench_tasks = [task for task in manifest.tasks if task.adapter == "pinchbench"]
    assert len(pinchbench_tasks) == 5
    for task in pinchbench_tasks:
        environment = task.tool_environment
        assert environment is not None
        assert environment.workspace_path is not None
        assert (replay.manifest_path.parent / environment.workspace_path).is_dir()
    instance_path = replay.manifest_path.parent / "swebench-instances.jsonl"
    instance_ids = {json.loads(line)["instance_id"] for line in instance_path.read_text().splitlines() if line.strip()}
    assert instance_ids == {task.task_id for task in manifest.tasks if task.adapter == "swebench"}


def test_default_replay_declares_the_smallest_rung_that_holds_its_largest_turn() -> None:
    """Pin each task's declared floor to its bundled traces so the two cannot drift apart.

    The margin is the cache-isolation prefix, counted one token per character.
    """
    replay = DEFAULT_BUNDLED_REPLAY
    manifest = load_manifest(replay.manifest_path)
    prefix_margin = len(cache_namespace_prefix(" ".join("0" * CACHE_NAMESPACE_DIGITS)))
    context_rungs = (BENCHMARK_CONTEXT_TOKENS, *REDUCED_CONTEXT_LADDER)

    for task in manifest.tasks:
        rows = load_trace(replay.manifest_path.parent / task.trace)
        demand = prefix_margin + max(
            (row.recorded_prompt_tokens or 0) + (row.target_output_tokens or 0) for row in rows
        )
        assert task.required_context_tokens == min(rung for rung in context_rungs if rung >= demand)

    deployment = next(
        model.deployment
        for model in load_model_catalog(BUNDLED_RECIPES_ROOT).models
        if model.profile_id == DEFAULT_MANAGED_PROFILE_ID
    )
    assert deployment is not None
    assert manifest.required_context_tokens == BENCHMARK_CONTEXT_TOKENS
    assert reduced_context_rungs(deployment, manifest.required_context_tokens) == ()


@pytest.mark.parametrize(
    ("policy", "output_limit", "expected_limit"),
    (
        ("exact", DEFAULT_MAX_OUTPUT_TOKENS, MINI_MAX_OUTPUT_TOKENS),
        ("exact", MINI_MAX_OUTPUT_TOKENS // 2, MINI_MAX_OUTPUT_TOKENS // 2),
        ("recorded", DEFAULT_MAX_OUTPUT_TOKENS, MINI_MAX_OUTPUT_TOKENS),
        ("fixed", DEFAULT_MAX_OUTPUT_TOKENS, MINI_MAX_OUTPUT_TOKENS),
        ("recorded", MINI_MAX_OUTPUT_TOKENS // 2, MINI_MAX_OUTPUT_TOKENS // 2),
    ),
)
async def test_mini_replay_fits_8k_and_validates_submission_outputs(
    tmp_path: Path, policy: OutputTokenPolicy, output_limit: int, expected_limit: int
) -> None:
    replay = find_bundled_replay("aa-mini-v1")
    assert replay is not None
    manifest = load_manifest(replay.manifest_path)
    task = manifest.tasks[0]
    rows = load_trace(replay.manifest_path.parent / task.trace)
    assert task.model_calls == len(rows) == 6
    assert task.tool_calls == 5
    assert manifest.required_context_tokens == MINI_CONTEXT_TOKENS

    for previous, current in zip(rows, rows[1:]):
        assert current.messages[: len(previous.messages)] == previous.messages
        assistant, tool, user = current.messages[len(previous.messages) :]
        recorded = previous.recorded_tool_calls_after[0]
        assert assistant.role == "assistant"
        assert assistant.tool_calls is not None
        assert assistant.tool_calls[0]["id"] == recorded.tool_call_id
        assert tool.role == "tool" and tool.tool_call_id == recorded.tool_call_id
        assert user.role == "user"
    assert not rows[-1].tools
    assert not rows[-1].recorded_tool_calls_after

    responses = (SHORT_TOOL_RESPONSE,) * task.tool_calls + (SUMMARY_RESPONSE,)
    if policy == "exact":
        responses = (_exact_response(expected_limit),) * task.model_calls
    async with LocalSseServer(
        SHORT_TOOL_RESPONSE, responses=responses, inter_chunk_delay_seconds=EVENT_DELAY_SECONDS
    ) as server:
        config = RunConfig(
            base_url=server.base_url,
            model="mini-test",
            client_backend="python",
            max_output_tokens=output_limit,
            output_token_policy=policy,
        )
        context = replace(
            create_attached_submission_context(replay.manifest_path, config.model), context_tokens=MINI_CONTEXT_TOKENS
        )
        binding = create_measurement_binding(
            context,
            replay.manifest_path,
            config.model,
            TEST_HARDWARE,
            SourceProvenance(client_version="test", source_revision=None, source_state="unavailable"),
            observed_context_tokens=MINI_CONTEXT_TOKENS,
        )
        result = await run_manifest(
            replay.manifest_path,
            config,
            token_counter=CharacterCounter(),
        )
        assert result.success
        assert len(server.requests) == task.model_calls
        for captured, row in zip(server.requests, rows, strict=True):
            request = parse_json_object(captured.body, "mini request")
            assert row.target_output_tokens == MINI_MAX_OUTPUT_TOKENS
            assert row.recorded_completion_tokens is None
            assert row.max_output_tokens == MINI_MAX_OUTPUT_TOKENS
            assert request["max_tokens"] == expected_limit
            assert request.get("ignore_eos", False) is (policy == "exact")
            assert request.get("tool_choice") is None
            assert len(captured.body) + row.max_output_tokens + CHAT_TEMPLATE_RESERVE_TOKENS <= MINI_CONTEXT_TOKENS

    results_dir = tmp_path / "results"
    run_context = RunContextFacts(
        requested_tokens=MINI_CONTEXT_TOKENS,
        observed_tokens=MINI_CONTEXT_TOKENS,
        observed_reason=ContextObservationReason.REPORTED,
    )
    write_run_artifacts(result, results_dir, config, run_context=run_context, run_id=binding.run_id)
    write_measurement_binding(results_dir / MEASUREMENT_BINDING_FILENAME, binding)
    if policy != "exact":
        with pytest.raises(ValueError, match="short or unmeasurable output"):
            build_submission_bundle(results_dir)
        return

    bundle = build_submission_bundle(results_dir)
    assert len(bundle.evidence.turns) == task.model_calls
    assert bundle.aggregate.run.totals.short_output_warnings == 0
    for turn in bundle.evidence.turns:
        assert turn.tokens.target_output_tokens == MINI_MAX_OUTPUT_TOKENS
        assert turn.tokens.observed_output_tokens == expected_limit
    bundle_dir = tmp_path / "bundle"
    write_submission_bundle(bundle_dir, bundle)
    validated = validate_submission_bundle(bundle_dir)
    assert validated.manifest.run_id == binding.run_id
