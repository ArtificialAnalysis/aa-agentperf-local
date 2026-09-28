"""Exercise the public synthetic runtime conformance pack."""

import asyncio
import re
from pathlib import Path

import orjson
import pytest
from jsonschema import Draft202012Validator

from agentperf_local.client.backends import ClientBackend
from agentperf_local.client.protocol import CompletionError, CompletionResult, RawRead
from agentperf_local.client.request import CompletionRequest
from agentperf_local.common.json_types import JsonObject, JsonValue
from agentperf_local.common.models import read_object
from agentperf_local.deployment.qualification import (
    CAPPED_PROBE_OUTPUT_TOKENS,
    MAX_FINISH_REASON_CHARACTERS,
    MAX_TOOL_NAME_CHARACTERS,
    PROFILE_ID_PATTERN,
    QUALIFICATION_FAILURE_CODES,
    SYNTHETIC_PACK_ID,
    UNNAMED_TOOL_NAME,
    QualificationFile,
    RuntimeQualification,
    load_runtime_qualification,
    probe_pack_digest,
    qualify_runtime,
    synthetic_probe_pack,
    write_runtime_qualification,
)
from tests.file_modes import has_mode

MODEL = "synthetic-model"
PROFILE = "synthetic-profile"
RUN_ID = "8f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"
SCHEMA = Path(__file__).parents[1] / "docs" / "schemas" / "runtime-qualification-v2.schema.json"
SCHEMA_DOC = orjson.loads(SCHEMA.read_bytes())
# The checked-in MLX evidence predates the run identifier and stays on its own schema.
LEGACY_SCHEMA = Path(__file__).parents[1] / "docs" / "schemas" / "runtime-qualification-v1.schema.json"
LEGACY_SCHEMA_DOC = orjson.loads(LEGACY_SCHEMA.read_bytes())
EVIDENCE_ROOT = Path(__file__).parents[1] / "docs" / "evidence"
MLX_EVIDENCE = EVIDENCE_ROOT / "mlx-gpt-oss-20b-runtime-qualification.json"
GEMMA_EVIDENCE = EVIDENCE_ROOT / "gemma4-26b-a4b-q4-0-runtime-qualification.json"
PROMPT_TOKENS = 40
COMPLETION_TOKENS = 8
HOSTILE_TOOL_NAME = "lookup_issue" + "x" * 300
HOSTILE_FINISH_REASON = "length" + "y" * 300


def test_checked_mlx_failure_evidence_matches_the_current_public_pack() -> None:
    evidence = orjson.loads(MLX_EVIDENCE.read_bytes())

    Draft202012Validator(LEGACY_SCHEMA_DOC).validate(evidence)
    assert evidence["pack_digest"] == probe_pack_digest(synthetic_probe_pack("digest-does-not-bind-model"))
    assert evidence["passed"] is False
    assert sum(outcome["passed"] for outcome in evidence["outcomes"]) == 2
    assert all(not outcome["tool_names"] for outcome in evidence["outcomes"] if not outcome["passed"])


def test_checked_gemma_pass_evidence_matches_the_current_public_pack() -> None:
    """The managed llama.cpp path is checked in as a pass, opposite the MLX failure."""
    evidence = orjson.loads(GEMMA_EVIDENCE.read_bytes())

    Draft202012Validator(SCHEMA_DOC).validate(evidence)
    assert evidence["pack_digest"] == probe_pack_digest(synthetic_probe_pack("digest-does-not-bind-model"))
    assert evidence["profile_id"] == "gemma4-26b-a4b-q4-0"
    assert evidence["passed"] is True
    assert all(outcome["passed"] and not outcome["failure_codes"] for outcome in evidence["outcomes"])


class ScriptedClient:
    """Return one scripted response per request."""

    def __init__(self, responses: tuple[CompletionResult | CompletionError, ...]) -> None:
        self._responses = responses
        self.requests: list[CompletionRequest] = []

    async def complete(
        self,
        request: CompletionRequest,
        abort: asyncio.Event | None = None,
    ) -> CompletionResult:
        """Return the next scripted response."""
        del abort
        index = len(self.requests)
        self.requests.append(request)
        response = self._responses[index]
        if isinstance(response, CompletionError):
            raise response
        return response

    async def close(self) -> None:
        """Close the no-op client."""


def _choice(delta: JsonObject, finish_reason: str | None = None) -> JsonObject:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def _tool_call(index: int, identifier: str, name: str, arguments: str) -> JsonObject:
    return {
        "index": index,
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _result(*events: JsonObject) -> CompletionResult:
    usage: JsonObject = {
        "choices": [],
        "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS},
    }
    payloads = (*events, usage)
    encoded = b"".join(b"data: " + orjson.dumps(event) + b"\n\n" for event in payloads) + b"data: [DONE]\n\n"
    return CompletionResult(reads=(RawRead(timestamp=1.0, data=encoded),), aborted=False)


def _tool_result(*calls: JsonObject) -> CompletionResult:
    tool_calls: list[JsonValue] = list(calls)
    return _result(_choice({"tool_calls": tool_calls}), _choice({}, "tool_calls"))


async def _qualify(
    responses: tuple[CompletionResult | CompletionError, ...],
    *,
    backend: ClientBackend = "python",
) -> RuntimeQualification:
    return await qualify_runtime(ScriptedClient(responses), profile_id=PROFILE, model=MODEL, client_backend=backend)


def _passing_responses() -> tuple[CompletionResult, ...]:
    return (
        _tool_result(_tool_call(0, "call-single", "lookup_issue", '{"issue_id":"AA-17"}')),
        _result(_choice({"content": "READY"}), _choice({}, "stop")),
        _tool_result(
            _tool_call(0, "call-read", "read_file", '{"path":"README.md"}'),
            _tool_call(1, "call-search", "search_text", '{"query":"runtime qualification"}'),
        ),
        _tool_result(_tool_call(0, "call-result", "record_result", '{"status":"pass"}')),
        _result(_choice({"reasoning": "Synthetic"}), _choice({}, "length")),
    )


async def test_qualifies_structured_streams_without_storing_generated_values(tmp_path: Path) -> None:
    client = ScriptedClient(_passing_responses())

    report = await qualify_runtime(client, profile_id=PROFILE, model=MODEL, client_backend="python", run_id=RUN_ID)
    output_path = tmp_path / "qualification.json"
    written = write_runtime_qualification(output_path, report)
    encoded = output_path.read_bytes()
    written_report = orjson.loads(encoded)

    Draft202012Validator.check_schema(SCHEMA_DOC)
    Draft202012Validator(SCHEMA_DOC).validate(written_report)
    assert written_report["run_id"] == RUN_ID
    assert load_runtime_qualification(output_path) == report
    with pytest.raises(ValueError, match=re.escape("qualification: endpoint_url: Extra inputs are not permitted")):
        read_object(QualificationFile, {**written_report, "endpoint_url": "http://secret"}, "qualification").report()
    with pytest.raises(
        ValueError, match=re.escape("qualification: qualification pass state does not match its outcomes")
    ):
        read_object(QualificationFile, {**written_report, "passed": False}, "qualification").report()
    schema_failure_codes = SCHEMA_DOC["$defs"]["outcome"]["properties"]["failure_codes"]["items"]["enum"]
    assert tuple(schema_failure_codes) == QUALIFICATION_FAILURE_CODES
    assert SCHEMA_DOC["properties"]["profile_id"]["pattern"] == PROFILE_ID_PATTERN.pattern
    assert list(written_report) == sorted(written_report)
    assert report.passed
    assert len(report.outcomes) == len(client.requests) == 5
    assert report.outcomes[2].tool_names == ("read_file", "search_text")
    assert report.outcomes[4].finish_reason == "length"
    assert client.requests[4].max_tokens == CAPPED_PROBE_OUTPUT_TOKENS
    assert report.pack_digest == "sha256:95690cb1f6037c919eaf7780381436e8c0957922333972a67e239b5d0458033e"
    assert report.pack_digest == probe_pack_digest(synthetic_probe_pack("different-model"))
    assert written.byte_size == len(encoded)
    assert has_mode(output_path, 0o600)
    assert b"AA-17" not in encoded
    assert b"READY" not in encoded
    assert b"runtime qualification" not in encoded
    assert MODEL.encode() not in encoded
    assert report.endpoint_model_digest.startswith("sha256:")
    assert SYNTHETIC_PACK_ID.encode() in encoded

    with pytest.raises(FileExistsError):
        write_runtime_qualification(output_path, report)


async def test_reports_structural_failures_and_sanitizes_request_errors() -> None:
    responses: list[CompletionResult | CompletionError] = list(_passing_responses())
    responses[0] = _tool_result(_tool_call(0, "call-wrong", "lookup_issue", '{"issue_id":"WRONG"}'))
    responses[1] = CompletionError("secret endpoint failure", status_code=503, response_body="secret response")

    report = await _qualify(tuple(responses), backend="rust")
    encoded = orjson.dumps(report.to_json())

    assert not report.passed
    assert report.outcomes[0].failure_codes == ("tool_arguments_mismatch",)
    assert report.outcomes[1].failure_codes == ("request_error",)
    assert report.outcomes[1].status_code == 503
    assert b"secret" not in encoded


async def test_bounds_server_chosen_tool_names_and_finish_reasons(tmp_path: Path) -> None:
    responses = list(_passing_responses())
    responses[0] = _tool_result(_tool_call(0, "call-single", HOSTILE_TOOL_NAME, '{"issue_id":"AA-17"}'))
    responses[2] = _tool_result(
        _tool_call(0, "call-read", "", '{"path":"README.md"}'),
        _tool_call(1, "call-search", "search_text", '{"query":"runtime qualification"}'),
    )
    responses[4] = _result(_choice({"reasoning": "Synthetic"}), _choice({}, HOSTILE_FINISH_REASON))

    report = await qualify_runtime(
        ScriptedClient(tuple(responses)),
        profile_id=PROFILE,
        model=MODEL,
        client_backend="python",
    )
    output_path = tmp_path / "qualification.json"
    write_runtime_qualification(output_path, report)
    encoded = output_path.read_bytes()
    schema = orjson.loads(SCHEMA.read_bytes())

    Draft202012Validator(schema).validate(orjson.loads(encoded))
    assert report.outcomes[0].tool_names == (HOSTILE_TOOL_NAME[:MAX_TOOL_NAME_CHARACTERS],)
    assert report.outcomes[2].tool_names == (UNNAMED_TOOL_NAME, "search_text")
    assert report.outcomes[4].finish_reason == HOSTILE_FINISH_REASON[:MAX_FINISH_REASON_CHARACTERS]
    assert report.passed is False
    assert HOSTILE_TOOL_NAME.encode() not in encoded
    assert HOSTILE_FINISH_REASON.encode() not in encoded


async def test_reports_malformed_sse_without_storing_response_bytes() -> None:
    malformed = CompletionResult(
        reads=(RawRead(timestamp=1.0, data=b"data: private-invalid-response\n\n"),),
        aborted=False,
    )
    responses = (malformed, *_passing_responses()[1:])

    report = await _qualify(responses)
    encoded = orjson.dumps(report.to_json())

    assert report.passed is False
    assert report.outcomes[0].failure_codes == ("decode_error",)
    assert report.outcomes[0].raw_read_count == 1
    assert b"private-invalid-response" not in encoded
    Draft202012Validator(SCHEMA_DOC).validate(orjson.loads(encoded))


@pytest.mark.parametrize(
    ("probe_index", "malformed"),
    [
        (0, _tool_result(_tool_call(-1, "call-single", "lookup_issue", '{"issue_id":"AA-17"}'))),
        (1, _result(_choice({"content": "READY"}), _choice({}, "tool_calls"))),
    ],
    ids=("negative-tool-index", "no-tool-finish-mismatch"),
)
async def test_rejects_malformed_tool_transport(probe_index: int, malformed: CompletionResult) -> None:
    responses = list(_passing_responses())
    responses[probe_index] = malformed

    report = await _qualify(tuple(responses))

    assert report.passed is False
    assert "tool_transport_invalid" in report.outcomes[probe_index].failure_codes


@pytest.mark.parametrize(
    ("probe_index", "stored_passes"),
    [(2, (True, False)), (0, (False,))],
    ids=("advisory-parallel-tools", "required-single-tool"),
)
async def test_only_required_probes_decide_the_pass(
    tmp_path: Path, probe_index: int, stored_passes: tuple[bool, ...]
) -> None:
    """A lone parallel-call failure is recorded but still passes; older files that stored a fail still load."""
    responses = list(_passing_responses())
    responses[probe_index] = _tool_result(_tool_call(0, "call-read", "read_file", '{"path":"README.md"}'))

    report = await _qualify(tuple(responses))
    output_path = tmp_path / "qualification.json"
    write_runtime_qualification(output_path, report)
    written = orjson.loads(output_path.read_bytes())

    assert report.outcomes[probe_index].passed is False
    assert report.passed is written["passed"] is stored_passes[0]
    for stored in (True, False):
        record = {**written, "passed": stored}
        if stored in stored_passes:
            assert read_object(QualificationFile, record, "qualification").report() == report
        else:
            with pytest.raises(ValueError, match="qualification pass state does not match its outcomes"):
                read_object(QualificationFile, record, "qualification")
