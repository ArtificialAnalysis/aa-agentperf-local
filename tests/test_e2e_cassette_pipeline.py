"""Exercise the command-line replay pipeline with committed SSE responses."""

import asyncio
from pathlib import Path

import orjson
import pytest

from agentperf_local.cli import main
from tests.localhost_sse import LocalSseServer

RECORDING_FILE = Path(__file__).parent / "fixtures" / "recording" / "recordings" / "demo.json"
CASSETTE_DIR = Path(__file__).parent / "fixtures" / "cassette"
EXPECTED_TURNS = 2
EXPECTED_TOOL_CALLS = 1
QUALIFICATION_PROBES = 5


def _cassette_chunks(path: Path) -> tuple[bytes, ...]:
    """Split one raw SSE cassette into network reads at event boundaries."""
    return tuple(event + b"\n\n" for event in path.read_bytes().split(b"\n\n") if event)


async def test_cassette_survives_the_full_cli_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Replay canned model outputs through conversion, HTTP, metrics, and reports."""
    converted_dir = tmp_path / "converted"
    assert main(["convert", str(RECORDING_FILE), "--output-dir", str(converted_dir)]) == 0
    capsys.readouterr()

    output_dir = tmp_path / "results"
    # The ignore_eos probe and its control stream first, then the qualification probes, like
    # any seven turns would; the replay's two turns take the streams after them.
    filenames = (
        "demo-content.sse",
        "demo-content.sse",
        *("demo-content.sse",) * QUALIFICATION_PROBES,
        "demo-tool-and-content.sse",
        "demo-content.sse",
    )
    responses = tuple(_cassette_chunks(CASSETTE_DIR / filename) for filename in filenames)
    async with LocalSseServer((), responses=responses) as server:
        status = await asyncio.to_thread(
            main,
            [
                "run",
                str(converted_dir / "manifest.json"),
                "--base-url",
                server.base_url,
                "--model",
                "cassette-model",
                "--output-dir",
                str(output_dir),
                "--client",
                "python",
                "--no-cache-isolation",
            ],
        )

    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert orjson.loads(captured.out)["success"] is True
    assert [request.method for request in server.requests] == ["GET", "GET", *("POST",) * (4 + QUALIFICATION_PROBES)]
    # Probe with the field, control without it, the qualification probes, then the two exact-policy turns.
    assert [request.asks_ignore_eos for request in server.requests if request.method == "POST"] == [
        True,
        False,
        *(False,) * QUALIFICATION_PROBES,
        True,
        True,
    ]

    summary = orjson.loads((output_dir / "summary.json").read_bytes())
    turns = [orjson.loads(line) for line in (output_dir / "turns.jsonl").read_bytes().splitlines()]
    tools = orjson.loads((output_dir / "tools.json").read_bytes())

    assert summary["success"] is True
    assert summary["totals"]["turns"] == EXPECTED_TURNS
    assert summary["totals"]["successful_turns"] == EXPECTED_TURNS
    assert summary["totals"]["tool_calls"] == EXPECTED_TOOL_CALLS
    assert [turn["response_chunks"] for turn in turns] == [4, 3]
    assert [turn["response_tool_calls"] for turn in turns] == [0, 0]
    # The exact policy tells the server to ignore end-of-sequence, so every turn ends on length.
    assert [turn["finish_reason"] for turn in turns] == ["length", "length"]
    assert all(turn["tokens"]["local_output_tokens"] > 0 for turn in turns)
    assert tools["overall"]["calls"] == EXPECTED_TOOL_CALLS
