"""Build, run, write, and load public synthetic endpoint qualification probes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import orjson

from agentperf_local.client.backends import CLIENT_BACKENDS, ClientBackend, streaming_client
from agentperf_local.client.protocol import CompletionClient, CompletionError, CompletionResult
from agentperf_local.client.request import CompletionRequest
from agentperf_local.common.durable_files import WrittenFile, read_bounded_file, write_digest_file
from agentperf_local.common.identity import sha256_bytes, validate_digest, validate_run_id
from agentperf_local.common.json_fields import (
    decode_json_object,
    one_of,
    optional_integer,
    optional_string,
    require_exact_keys,
    required_boolean,
    required_integer,
    required_string,
    required_strings,
)
from agentperf_local.common.json_records import json_field_names, json_record
from agentperf_local.common.json_types import JsonObject, normalize_json_object, pretty_json_bytes
from agentperf_local.deployment.endpoint_probes import PROBE_MAX_CONNECTIONS, probe_request, stream_usage
from agentperf_local.metrics.decode import decode_sse_reads
from agentperf_local.metrics.response import ToolCall, parse_response_channels
from agentperf_local.replay.config import DEFAULT_REQUEST_TIMEOUT_SECONDS
from agentperf_local.replay.fidelity import LENGTH_FINISH_REASON, ExpectedToolCall, evaluate_tool_fidelity

# Version 2 added the run identifier that ties a report to one benchmark attempt.
QUALIFICATION_VERSION = 2
QUALIFICATION_KIND = "runtime_qualification"
QUALIFICATION_FILENAME = "qualification.json"
MAX_QUALIFICATION_BYTES = 256 * 1024
SYNTHETIC_PACK_ID = "aa-runtime-synthetic-v1"
CAPPED_PROBE_OUTPUT_TOKENS = 1
CAPPED_FINISH_REASONS = (LENGTH_FINISH_REASON, "max_tokens")
PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
MAX_QUALIFICATION_TOOL_CALLS = 8
MAX_TOOL_NAME_CHARACTERS = 128
MAX_FINISH_REASON_CHARACTERS = 64
UNNAMED_TOOL_NAME = "unnamed"

type QualificationProbeId = Literal["single_tool", "no_tool", "parallel_tools", "tool_history", "capped_finish"]
type QualificationFailureCode = Literal[
    "empty_stream",
    "aborted",
    "decode_error",
    "tool_transport_invalid",
    "tool_sequence_mismatch",
    "tool_arguments_mismatch",
    "tool_call_identifier_missing",
    "content_missing",
    "finish_reason_missing",
    "capped_finish_not_reported",
    "unexpected_generation_cap",
    "usage_missing",
    "request_error",
]

QUALIFICATION_PROBE_IDS: tuple[QualificationProbeId, ...] = (
    "single_tool",
    "no_tool",
    "parallel_tools",
    "tool_history",
    "capped_finish",
)
QUALIFICATION_FAILURE_CODES: tuple[QualificationFailureCode, ...] = (
    "empty_stream",
    "aborted",
    "decode_error",
    "tool_transport_invalid",
    "tool_sequence_mismatch",
    "tool_arguments_mismatch",
    "tool_call_identifier_missing",
    "content_missing",
    "finish_reason_missing",
    "capped_finish_not_reported",
    "unexpected_generation_cap",
    "usage_missing",
    "request_error",
)
_REPORT_KEYS = frozenset(
    (
        "version",
        "kind",
        "run_id",
        "profile_id",
        "endpoint_model_digest",
        "client_backend",
        "synthetic_pack_id",
        "pack_digest",
        "passed",
        "outcomes",
        "privacy",
    )
)
_PRIVACY_BLOCK: JsonObject = {
    "generated_text_included": False,
    "tool_arguments_included": False,
    "endpoint_url_included": False,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedCall:
    """Describe one exact synthetic tool call."""

    name: str
    arguments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Reject empty call expectations."""
        if not self.name:
            raise ValueError("expected call name must not be empty")
        if any(not key or not value for key, value in self.arguments):
            raise ValueError("expected call arguments must not be empty")

    def argument_object(self) -> JsonObject:
        """Return the expected JSON argument object."""
        return dict(self.arguments)

    def to_json(self) -> JsonObject:
        """Return the expected call contract."""
        return {"name": self.name, "arguments": self.argument_object()}


@dataclass(frozen=True, slots=True, kw_only=True)
class QualificationProbe:
    """Bind one synthetic request to structural expectations."""

    probe_id: QualificationProbeId
    description: str
    request: CompletionRequest
    expected_calls: tuple[ExpectedCall, ...]
    require_content: bool = False
    require_capped_finish: bool = False

    def __post_init__(self) -> None:
        """Validate one probe contract."""
        if not self.probe_id or not self.description:
            raise ValueError("probe identity and description must not be empty")
        if self.require_content and self.expected_calls:
            raise ValueError("a probe cannot require content and exact tool calls")
        if self.require_capped_finish and self.request.max_tokens != CAPPED_PROBE_OUTPUT_TOKENS:
            raise ValueError("a capped probe must use the one-token limit")

    def digest_json(self) -> JsonObject:
        """Return the public request and expectation used for pack identity."""
        request = self.request.body()
        del request["model"]
        return {
            "probe_id": self.probe_id,
            "description": self.description,
            "request": request,
            "expected_calls": [call.to_json() for call in self.expected_calls],
            "require_content": self.require_content,
            "require_capped_finish": self.require_capped_finish,
        }


def _bounded_tool_name(name: str) -> str:
    """Return one tool name inside the published report limits."""
    return name[:MAX_TOOL_NAME_CHARACTERS] if name else UNNAMED_TOOL_NAME


def _bounded_finish_reason(finish_reason: str) -> str | None:
    """Return one finish reason inside the published report limits."""
    return finish_reason[:MAX_FINISH_REASON_CHARACTERS] if finish_reason else None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeOutcome:
    """Store allowlisted structural evidence from one probe."""

    probe_id: QualificationProbeId
    passed: bool
    failure_codes: tuple[QualificationFailureCode, ...]
    tool_names: tuple[str, ...]
    tool_call_count: int
    arguments_valid: bool
    call_identifiers_present: bool
    content_present: bool
    reasoning_present: bool
    finish_reason: str | None
    usage_present: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_read_count: int
    decoded_chunk_count: int
    status_code: int | None = None

    def __post_init__(self) -> None:
        """Bound server-chosen text and reject outcomes outside the closed public contract.

        The endpoint chooses tool names and finish reasons, so the report caps them to its schema limits.
        """
        object.__setattr__(self, "tool_names", tuple(_bounded_tool_name(name) for name in self.tool_names))
        if self.finish_reason is not None:
            object.__setattr__(self, "finish_reason", _bounded_finish_reason(self.finish_reason))
        if self.probe_id not in QUALIFICATION_PROBE_IDS:
            raise ValueError("qualification probe ID is not supported")
        if self.passed != (not self.failure_codes):
            raise ValueError("qualification pass state and failure codes are inconsistent")
        if len(self.failure_codes) != len(set(self.failure_codes)):
            raise ValueError("qualification failure codes must be unique")
        if any(code not in QUALIFICATION_FAILURE_CODES for code in self.failure_codes):
            raise ValueError("qualification failure code is not supported")
        if len(self.tool_names) > MAX_QUALIFICATION_TOOL_CALLS:
            raise ValueError("qualification tool-name count exceeds the public limit")
        counts = (self.tool_call_count, self.raw_read_count, self.decoded_chunk_count)
        if any(value < 0 for value in counts) or self.tool_call_count > MAX_QUALIFICATION_TOOL_CALLS:
            raise ValueError("qualification counts are outside the public limits")
        for value in (self.prompt_tokens, self.completion_tokens):
            if value is not None and value < 0:
                raise ValueError("qualification token counts must be non-negative")
        if self.status_code is not None and not 100 <= self.status_code <= 599:
            raise ValueError("qualification status code is outside the HTTP range")

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> ProbeOutcome:
        """Parse one strict probe outcome."""
        require_exact_keys(data, json_field_names(cls), source)
        return cls(
            probe_id=one_of(required_string(data, "probe_id", source), QUALIFICATION_PROBE_IDS, f"{source}.probe_id"),
            passed=required_boolean(data, "passed", source),
            failure_codes=tuple(
                one_of(name, QUALIFICATION_FAILURE_CODES, f"{source}.failure_codes")
                for name in required_strings(data, "failure_codes", source)
            ),
            tool_names=required_strings(data, "tool_names", source),
            tool_call_count=required_integer(data, "tool_call_count", source),
            arguments_valid=required_boolean(data, "arguments_valid", source),
            call_identifiers_present=required_boolean(data, "call_identifiers_present", source),
            content_present=required_boolean(data, "content_present", source),
            reasoning_present=required_boolean(data, "reasoning_present", source),
            finish_reason=optional_string(data, "finish_reason", source),
            usage_present=required_boolean(data, "usage_present", source),
            prompt_tokens=optional_integer(data, "prompt_tokens", source),
            completion_tokens=optional_integer(data, "completion_tokens", source),
            raw_read_count=required_integer(data, "raw_read_count", source),
            decoded_chunk_count=required_integer(data, "decoded_chunk_count", source),
            status_code=optional_integer(data, "status_code", source),
        )

    def to_json(self) -> JsonObject:
        """Return structural evidence without generated text or arguments."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeQualification:
    """Store one endpoint qualification result."""

    profile_id: str
    endpoint_model_digest: str
    client_backend: ClientBackend
    pack_digest: str
    outcomes: tuple[ProbeOutcome, ...]
    # The benchmark attempt this report belongs to; None for a standalone probe of an endpoint.
    run_id: str | None = None

    def __post_init__(self) -> None:
        """Validate report identity and coverage."""
        if self.run_id is not None:
            validate_run_id(self.run_id, "run_id")
        if PROFILE_ID_PATTERN.fullmatch(self.profile_id) is None:
            raise ValueError("profile_id must be a portable lowercase label")
        validate_digest(self.endpoint_model_digest, "endpoint_model_digest")
        validate_digest(self.pack_digest, "pack_digest")
        if self.client_backend not in {"python", "rust"}:
            raise ValueError("qualification client backend is not supported")
        if tuple(outcome.probe_id for outcome in self.outcomes) != QUALIFICATION_PROBE_IDS:
            raise ValueError("qualification outcomes must cover the synthetic probes in order")

    @property
    def passed(self) -> bool:
        """Return whether every synthetic probe passed."""
        return all(outcome.passed for outcome in self.outcomes)

    def to_json(self) -> JsonObject:
        """Return the closed local qualification report."""
        return {
            "version": QUALIFICATION_VERSION,
            "kind": QUALIFICATION_KIND,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "endpoint_model_digest": self.endpoint_model_digest,
            "client_backend": self.client_backend,
            "synthetic_pack_id": SYNTHETIC_PACK_ID,
            "pack_digest": self.pack_digest,
            "passed": self.passed,
            "outcomes": [outcome.to_json() for outcome in self.outcomes],
            "privacy": dict(_PRIVACY_BLOCK),
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> RuntimeQualification:
        """Parse one strict qualification report."""
        require_exact_keys(data, _REPORT_KEYS, "qualification")
        if required_integer(data, "version", "qualification") != QUALIFICATION_VERSION:
            raise ValueError("qualification version is not supported")
        if data.get("kind") != QUALIFICATION_KIND or data.get("synthetic_pack_id") != SYNTHETIC_PACK_ID:
            raise ValueError("qualification kind or synthetic pack is not supported")
        if data.get("privacy") != _PRIVACY_BLOCK:
            raise ValueError("qualification privacy declaration is not supported")
        client_backend = one_of(
            required_string(data, "client_backend", "qualification"), CLIENT_BACKENDS, "qualification.client_backend"
        )
        raw_outcomes = data.get("outcomes")
        if not isinstance(raw_outcomes, list):
            raise ValueError("qualification.outcomes must be an array")
        outcomes: list[ProbeOutcome] = []
        for index, value in enumerate(raw_outcomes):
            if not isinstance(value, dict):
                raise ValueError(f"qualification.outcomes[{index}] must be an object")
            outcomes.append(ProbeOutcome.from_json(value, f"qualification.outcomes[{index}]"))
        report = cls(
            profile_id=required_string(data, "profile_id", "qualification"),
            endpoint_model_digest=required_string(data, "endpoint_model_digest", "qualification"),
            client_backend=client_backend,
            pack_digest=required_string(data, "pack_digest", "qualification"),
            outcomes=tuple(outcomes),
            run_id=optional_string(data, "run_id", "qualification"),
        )
        if required_boolean(data, "passed", "qualification") != report.passed:
            raise ValueError("qualification pass state does not match its outcomes")
        return report


def _tool(name: str, description: str, properties: JsonObject, required: tuple[str, ...]) -> JsonObject:
    """Build one strict public synthetic function schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def _text_property(description: str) -> JsonObject:
    """Build one required string property."""
    return {"type": "string", "description": description}


def synthetic_probe_pack(model: str) -> tuple[QualificationProbe, ...]:
    """Build the public synthetic runtime conformance pack."""
    issue_tool = _tool(
        "lookup_issue",
        "Look up one synthetic issue.",
        {"issue_id": _text_property("The exact synthetic issue identifier.")},
        ("issue_id",),
    )
    read_tool = _tool(
        "read_file",
        "Read one synthetic repository path.",
        {"path": _text_property("The exact repository path.")},
        ("path",),
    )
    search_tool = _tool(
        "search_text",
        "Search a synthetic repository.",
        {"query": _text_property("The exact search phrase.")},
        ("query",),
    )
    result_tool = _tool(
        "record_result",
        "Record one synthetic qualification result.",
        {"status": _text_property("The exact qualification status.")},
        ("status",),
    )
    history_messages: tuple[JsonObject, ...] = (
        {"role": "user", "content": "Look up synthetic issue AA-17."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "synthetic-history-call",
                    "type": "function",
                    "function": {"name": "lookup_issue", "arguments": '{"issue_id":"AA-17"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "synthetic-history-call",
            "content": '{"state":"resolved"}',
        },
        {"role": "user", "content": 'Call record_result once with status exactly "pass".'},
    )
    return (
        QualificationProbe(
            probe_id="single_tool",
            description="Emit one exact structured call.",
            request=probe_request(
                model=model,
                messages=(
                    {
                        "role": "user",
                        "content": 'Call lookup_issue once with issue_id exactly "AA-17". Do not call another tool.',
                    },
                ),
                tools=(issue_tool,),
                tool_choice="required",
            ),
            expected_calls=(ExpectedCall(name="lookup_issue", arguments=(("issue_id", "AA-17"),)),),
        ),
        QualificationProbe(
            probe_id="no_tool",
            description="Honor disabled tool choice and return content.",
            request=probe_request(
                model=model,
                messages=({"role": "user", "content": "Reply with the word READY and do not call a tool."},),
                tools=(issue_tool,),
                tool_choice="none",
            ),
            expected_calls=(),
            require_content=True,
        ),
        QualificationProbe(
            probe_id="parallel_tools",
            description="Preserve two ordered calls in one assistant response.",
            request=probe_request(
                model=model,
                messages=(
                    {
                        "role": "user",
                        "content": (
                            'Call read_file with path exactly "README.md", then call search_text with query exactly '
                            '"runtime qualification". Emit both calls in this response and in that order.'
                        ),
                    },
                ),
                tools=(read_tool, search_tool),
                tool_choice="required",
                parallel_tool_calls=True,
            ),
            expected_calls=(
                ExpectedCall(name="read_file", arguments=(("path", "README.md"),)),
                ExpectedCall(name="search_text", arguments=(("query", "runtime qualification"),)),
            ),
        ),
        QualificationProbe(
            probe_id="tool_history",
            description="Round-trip prior call identifiers and tool returns.",
            request=probe_request(
                model=model,
                messages=history_messages,
                tools=(issue_tool, result_tool),
                tool_choice="required",
            ),
            expected_calls=(ExpectedCall(name="record_result", arguments=(("status", "pass"),)),),
        ),
        QualificationProbe(
            probe_id="capped_finish",
            description="Expose a one-token generation cap through the finish reason.",
            request=probe_request(
                model=model,
                messages=(
                    {
                        "role": "user",
                        "content": "Write at least two hundred words about synthetic benchmark protocol conformance.",
                    },
                ),
                max_tokens=CAPPED_PROBE_OUTPUT_TOKENS,
            ),
            expected_calls=(),
            require_capped_finish=True,
        ),
    )


def probe_pack_digest(probes: tuple[QualificationProbe, ...]) -> str:
    """Return the digest of public requests and exact expectations."""
    encoded = orjson.dumps([probe.digest_json() for probe in probes], option=orjson.OPT_SORT_KEYS)
    return sha256_bytes(encoded)


def _arguments_match(call: ToolCall, expected: ExpectedCall) -> bool:
    try:
        arguments = normalize_json_object(orjson.loads(call.arguments))
    except (orjson.JSONDecodeError, ValueError):
        return False
    return arguments == expected.argument_object()


def _evaluate_probe(
    probe: QualificationProbe,
    result: CompletionResult,
) -> ProbeOutcome:
    chunks = decode_sse_reads(result.reads)
    channels = parse_response_channels(chunks)
    prompt_tokens, completion_tokens = stream_usage(chunks)
    expected_names = tuple(call.name for call in probe.expected_calls)
    observed_names = tuple(call.name for call in channels.tool_calls)
    arguments_valid = len(channels.tool_calls) == len(probe.expected_calls) and all(
        _arguments_match(call, expected)
        for call, expected in zip(channels.tool_calls, probe.expected_calls, strict=True)
    )
    identifiers_present = not channels.tool_calls or all(call.identifier for call in channels.tool_calls)
    content_present = bool(channels.content)
    usage_present = prompt_tokens is not None and completion_tokens is not None
    failures: list[QualificationFailureCode] = []
    if not result.reads or not chunks:
        failures.append("empty_stream")
    if result.aborted:
        failures.append("aborted")
    if not probe.require_capped_finish:
        transport = evaluate_tool_fidelity(
            channels.tool_calls,
            tuple(
                ExpectedToolCall(name=expected.name, arguments=expected.argument_object())
                for expected in probe.expected_calls
            ),
            finish_reason=channels.finish_reason,
        )
        if not transport.transport_valid:
            failures.append("tool_transport_invalid")
    if observed_names != expected_names:
        failures.append("tool_sequence_mismatch")
    if not arguments_valid:
        failures.append("tool_arguments_mismatch")
    if not identifiers_present:
        failures.append("tool_call_identifier_missing")
    if probe.require_content and not content_present:
        failures.append("content_missing")
    if channels.finish_reason is None:
        failures.append("finish_reason_missing")
    if probe.require_capped_finish:
        if channels.finish_reason not in CAPPED_FINISH_REASONS:
            failures.append("capped_finish_not_reported")
    elif channels.finish_reason in CAPPED_FINISH_REASONS:
        failures.append("unexpected_generation_cap")
    if not usage_present:
        failures.append("usage_missing")
    return ProbeOutcome(
        probe_id=probe.probe_id,
        passed=not failures,
        failure_codes=tuple(failures),
        tool_names=observed_names,
        tool_call_count=len(channels.tool_calls),
        arguments_valid=arguments_valid,
        call_identifiers_present=bool(identifiers_present),
        content_present=content_present,
        reasoning_present=bool(channels.reasoning),
        finish_reason=channels.finish_reason,
        usage_present=usage_present,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        raw_read_count=len(result.reads),
        decoded_chunk_count=len(chunks),
    )


def _request_error(probe: QualificationProbe, error: CompletionError) -> ProbeOutcome:
    """Return a content-free request error outcome."""
    return ProbeOutcome(
        probe_id=probe.probe_id,
        passed=False,
        failure_codes=("request_error",),
        tool_names=(),
        tool_call_count=0,
        arguments_valid=False,
        call_identifiers_present=False,
        content_present=False,
        reasoning_present=False,
        finish_reason=None,
        usage_present=False,
        prompt_tokens=None,
        completion_tokens=None,
        raw_read_count=0,
        decoded_chunk_count=0,
        status_code=error.status_code,
    )


def _decode_error(probe: QualificationProbe, result: CompletionResult) -> ProbeOutcome:
    """Return a content-free decoder failure outcome."""
    return ProbeOutcome(
        probe_id=probe.probe_id,
        passed=False,
        failure_codes=("decode_error",),
        tool_names=(),
        tool_call_count=0,
        arguments_valid=False,
        call_identifiers_present=False,
        content_present=False,
        reasoning_present=False,
        finish_reason=None,
        usage_present=False,
        prompt_tokens=None,
        completion_tokens=None,
        raw_read_count=len(result.reads),
        decoded_chunk_count=0,
    )


async def qualify_runtime(
    client: CompletionClient,
    *,
    profile_id: str,
    model: str,
    client_backend: ClientBackend,
    run_id: str | None = None,
) -> RuntimeQualification:
    """Run the public probes sequentially through one streaming client."""
    probes = synthetic_probe_pack(model)
    outcomes: list[ProbeOutcome] = []
    for probe in probes:
        try:
            result = await client.complete(probe.request)
        except CompletionError as error:
            outcomes.append(_request_error(probe, error))
            continue
        try:
            outcomes.append(_evaluate_probe(probe, result))
        except ValueError:
            outcomes.append(_decode_error(probe, result))
    return RuntimeQualification(
        profile_id=profile_id,
        endpoint_model_digest=sha256_bytes(model.encode("utf-8")),
        client_backend=client_backend,
        pack_digest=probe_pack_digest(probes),
        outcomes=tuple(outcomes),
        run_id=run_id,
    )


def load_runtime_qualification(path: Path) -> RuntimeQualification:
    """Read and validate one qualification report."""
    encoded = read_bounded_file(path, MAX_QUALIFICATION_BYTES, label="qualification report")
    return RuntimeQualification.from_json(decode_json_object(encoded, f"invalid qualification JSON: {path}"))


async def qualify_managed_endpoint(
    base_url: str,
    model: str,
    profile_id: str,
    client_backend: ClientBackend,
    run_id: str,
) -> RuntimeQualification:
    """Probe an owned localhost server with the public synthetic pack and bind the report to the run."""
    client = streaming_client(
        client_backend,
        base_url=base_url,
        api_key=None,
        timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_connections=PROBE_MAX_CONNECTIONS,
    )
    try:
        return await qualify_runtime(
            client,
            profile_id=profile_id,
            model=model,
            client_backend=client_backend,
            run_id=run_id,
        )
    finally:
        await client.close()


def write_runtime_qualification(path: Path, report: RuntimeQualification) -> WrittenFile:
    """Write one exact report without replacing an existing path."""
    return write_digest_file(path, pretty_json_bytes(report.to_json()))
