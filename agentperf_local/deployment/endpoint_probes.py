"""Ask an OpenAI-compatible endpoint what it serves and how it behaves.

- `probe_served_context_tokens`: one GET /models that reads the served context length.
- `is_ollama_endpoint`: two cheap GETs that name an Ollama server before any long probe.
- `probe_ignore_eos`: the behavioural probe the exact policy depends on.
- `probe_request`, `stream_usage`, `PROBE_MAX_CONNECTIONS`: the request, usage reader, and
  connection cap every endpoint probe shares, runtime qualification included.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import httpx
import orjson

from agentperf_local.client.backends import ClientBackend, streaming_client
from agentperf_local.client.protocol import CompletionClient, CompletionError
from agentperf_local.client.request import CompletionRequest
from agentperf_local.common.json_fields import lenient_integer
from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object
from agentperf_local.metrics.decode import StreamChunk, decode_sse_reads
from agentperf_local.metrics.response import parse_response_channels
from agentperf_local.provenance.context import ContextObservationReason
from agentperf_local.replay.config import OutputTokenPolicy
from agentperf_local.replay.fidelity import generated_whole_budget

DEFAULT_PROBE_OUTPUT_TOKENS = 128
# Probes send one request at a time.
PROBE_MAX_CONNECTIONS = 1
CONTEXT_PROBE_TIMEOUT_SECONDS = 5.0

# The budget must exceed what a thinking model spends on a one-word answer, or both probe
# requests fill it and the verdict is undetermined on exactly the endpoints the probe is
# for. 512 leaves room for reasoning at low effort; an honouring server generates it all
# once per run, which also warms the model before the first measured turn.
IGNORE_EOS_PROBE_OUTPUT_TOKENS = 512
IGNORE_EOS_PROBE_TIMEOUT_SECONDS = 180.0
IGNORE_EOS_PROBE_PROMPT = "Reply with exactly the word: hi"

OLLAMA_IDENTITY_TIMEOUT_SECONDS = 5.0
OPENAI_COMPATIBLE_PATH_SUFFIX = "/v1"
OLLAMA_VERSION_PATH = "/api/version"
OLLAMA_TAGS_PATH = "/api/tags"
# The CLI prints this and the TUI shows it, so both say the same thing about Ollama.
OLLAMA_RECORDED_POLICY_WARNING = (
    "this server is Ollama, which cannot honour ignore_eos, so the run uses the recorded output policy; "
    "end-to-end latency is reported as a normalized estimate and is not directly comparable with exact-policy runs"
)


def probe_request(
    *,
    model: str,
    messages: tuple[JsonObject, ...],
    tools: tuple[JsonObject, ...] = (),
    max_tokens: int = DEFAULT_PROBE_OUTPUT_TOKENS,
    tool_choice: str | None = None,
    parallel_tool_calls: bool = False,
    ignore_eos: bool = False,
) -> CompletionRequest:
    """Build one deterministic synthetic request; every endpoint probe samples the same way."""
    extra_body: list[tuple[str, JsonValue]] = []
    if tool_choice is not None:
        extra_body.append(("tool_choice", tool_choice))
    if parallel_tool_calls:
        extra_body.append(("parallel_tool_calls", True))
    return CompletionRequest(
        messages=messages,
        model=model,
        tools=tools,
        max_tokens=max_tokens,
        reasoning_effort="low",
        temperature=0.0,
        top_p=1.0,
        ignore_eos=ignore_eos,
        extra_body=tuple(extra_body),
    )


def stream_usage(chunks: tuple[StreamChunk, ...]) -> tuple[int | None, int | None]:
    """Read final server token counts (prompt, completion) from decoded events."""
    for chunk in reversed(chunks):
        value = chunk.data.get("usage")
        if not isinstance(value, dict):
            continue
        return lenient_integer(value, "prompt_tokens"), lenient_integer(value, "completion_tokens")
    return None, None


def model_entry(raw_models: Sequence[object], model_id: str) -> JsonObject | None:
    """Return the model-list entry served under this id, when the list names it."""
    for raw_model in raw_models:
        if isinstance(raw_model, dict) and raw_model.get("id") == model_id:
            return raw_model
    return None


def served_context_tokens(raw_model: JsonObject) -> int | None:
    """Return the context length one model entry reports, when it reports one.

    llama.cpp reports it as meta.n_ctx; SGLang reports max_model_len on the entry.
    """
    meta = raw_model.get("meta")
    if isinstance(meta, dict):
        reported = meta.get("n_ctx")
        if isinstance(reported, int) and not isinstance(reported, bool):
            return reported
    reported = raw_model.get("max_model_len")
    if isinstance(reported, int) and not isinstance(reported, bool):
        return reported
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextProbeResult:
    """Store one served-context probe outcome and why an observation is missing."""

    observed_tokens: int | None
    reason: ContextObservationReason

    @property
    def endpoint_answered(self) -> bool:
        """Return whether the server answered the probe at all, whatever it said."""
        return self.reason not in (
            ContextObservationReason.ENDPOINT_UNREACHABLE,
            ContextObservationReason.HTTP_ERROR,
        )


def _unobserved(reason: ContextObservationReason) -> ContextProbeResult:
    return ContextProbeResult(observed_tokens=None, reason=reason)


def probe_served_context_tokens(base_url: str, model: str, api_key: str | None = None) -> ContextProbeResult:
    """Ask an OpenAI-compatible endpoint what context length it serves one model at.

    Endpoint misbehavior never raises: any failure to observe returns None with the
    reason, because an unknown context is itself non-comparable evidence.
    """
    headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=CONTEXT_PROBE_TIMEOUT_SECONDS, headers=headers) as client:
            response = client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError:
        return _unobserved(ContextObservationReason.ENDPOINT_UNREACHABLE)
    if not response.is_success:
        return _unobserved(ContextObservationReason.HTTP_ERROR)
    try:
        payload = normalize_json_object(orjson.loads(response.content))
    except (orjson.JSONDecodeError, ValueError):
        return _unobserved(ContextObservationReason.MALFORMED_RESPONSE)
    raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        return _unobserved(ContextObservationReason.MALFORMED_RESPONSE)
    entry = model_entry(raw_models, model)
    if entry is None:
        return _unobserved(ContextObservationReason.MODEL_NOT_LISTED)
    served = served_context_tokens(entry)
    if served is None:
        return _unobserved(ContextObservationReason.CONTEXT_NOT_REPORTED)
    if served <= 0:
        # A nonsense report is recorded as unobserved so it cannot crash the run.
        return _unobserved(ContextObservationReason.NON_POSITIVE_CONTEXT)
    return ContextProbeResult(observed_tokens=served, reason=ContextObservationReason.REPORTED)


class IgnoreEosSupport(StrEnum):
    """Say what one probe proved about an endpoint's handling of ignore_eos."""

    HONOURED = "honoured"
    IGNORED = "ignored"
    UNDETERMINED = "undetermined"


class ProbeFailure(StrEnum):
    """Say why one probe request produced no answer to judge."""

    # The endpoint answered with an error status, or with a stream that carried no verdict.
    REFUSED = "refused"
    # The request never reached an answer: connection, timeout, or an aborted stream.
    UNREACHABLE = "unreachable"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeAnswer:
    """Hold the two facts one probe completion is judged on, read the way a replay turn is."""

    finish_reason: str | None
    completion_tokens: int | None

    @property
    def spent_whole_budget(self) -> bool:
        """Report whether the response ran to the cap the probe set."""
        return generated_whole_budget(self.finish_reason, self.completion_tokens, IGNORE_EOS_PROBE_OUTPUT_TOKENS)


type ProbeOutcome = ProbeAnswer | ProbeFailure


@dataclass(frozen=True, slots=True, kw_only=True)
class IgnoreEosProbeResult:
    """Record what the probe observed and the policy that observation supports."""

    support: IgnoreEosSupport
    # The answer to the request that carried ignore_eos, or None when it produced none.
    observed: ProbeAnswer | None

    @property
    def summary(self) -> str:
        """State the observation in one sentence for a log line or an error."""
        budget = f"{IGNORE_EOS_PROBE_OUTPUT_TOKENS}-token probe"
        if self.support is IgnoreEosSupport.HONOURED:
            return f"server honours ignore_eos: the {budget} generated its whole budget"
        if self.support is IgnoreEosSupport.UNDETERMINED:
            return "could not determine whether the server honours ignore_eos"
        if self.observed is None:
            return f"server ignores ignore_eos: it refused a {budget}"
        tokens = "" if self.observed.completion_tokens is None else f" after {self.observed.completion_tokens} tokens"
        return f"server ignores ignore_eos: a {budget} finished with {self.observed.finish_reason!r}{tokens}"

    @property
    def output_token_policy(self) -> OutputTokenPolicy:
        """Return the policy this endpoint can actually measure under.

        Only a proven-ignored endpoint gives up the exact policy. An undetermined probe
        keeps the stronger default, because a run under exact stays comparable and a
        guess that silently weakened it would not.
        """
        return "recorded" if self.support is IgnoreEosSupport.IGNORED else "exact"


def is_ollama_endpoint(
    base_url: str, api_key: str | None = None, *, deadline_seconds: float = OLLAMA_IDENTITY_TIMEOUT_SECONDS
) -> bool:
    """Return whether the server behind this OpenAI-compatible base URL is Ollama.

    Call it off the event loop; it runs its own. Any failure to get both answers means not Ollama.
    """
    return asyncio.run(_answers_as_ollama(base_url, api_key, deadline_seconds))


async def _answers_as_ollama(base_url: str, api_key: str | None, deadline_seconds: float) -> bool:
    """Ask both Ollama identity paths under one overall deadline.

    Ollama serves its native API next to the OpenAI one. GET /api/version answers with a
    version string, and GET /api/tags lists models. Other servers can answer the first
    path, so both answers are required. The port alone proves nothing, because any
    server can listen on 11434. httpx timeouts apply per read, so a server that trickles
    bytes would outlast them; the deadline bounds the whole check.
    """
    root = base_url.rstrip("/").removesuffix(OPENAI_COMPATIBLE_PATH_SUFFIX)
    headers = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
    try:
        async with (
            asyncio.timeout(deadline_seconds),
            httpx.AsyncClient(timeout=deadline_seconds, headers=headers) as client,
        ):
            version = _json_object_answer(await client.get(f"{root}{OLLAMA_VERSION_PATH}"))
            if version is None or not isinstance(version.get("version"), str):
                return False
            tags = _json_object_answer(await client.get(f"{root}{OLLAMA_TAGS_PATH}"))
    except (httpx.HTTPError, TimeoutError):
        return False
    return tags is not None and isinstance(tags.get("models"), list)


def _json_object_answer(response: httpx.Response) -> JsonObject | None:
    """Return a successful response's JSON object body, or None for anything else."""
    if not response.is_success:
        return None
    try:
        return normalize_json_object(orjson.loads(response.content))
    except (orjson.JSONDecodeError, ValueError):
        return None


def ignore_eos_probe_request(model: str, *, ignore_eos: bool) -> CompletionRequest:
    """Build the one-word probe, with or without the field under test."""
    return probe_request(
        model=model,
        messages=({"role": "user", "content": IGNORE_EOS_PROBE_PROMPT},),
        max_tokens=IGNORE_EOS_PROBE_OUTPUT_TOKENS,
        ignore_eos=ignore_eos,
    )


async def _answer(client: CompletionClient, request: CompletionRequest) -> ProbeOutcome:
    """Send one probe request and read it the way the replay reads a turn."""
    try:
        result = await client.complete(request)
    except CompletionError as error:
        return ProbeFailure.UNREACHABLE if error.status_code is None else ProbeFailure.REFUSED
    chunks = decode_sse_reads(result.reads)
    if result.aborted or not chunks:
        return ProbeFailure.REFUSED
    _, completion_tokens = stream_usage(chunks)
    return ProbeAnswer(finish_reason=parse_response_channels(chunks).finish_reason, completion_tokens=completion_tokens)


def _needs_control(asked: ProbeOutcome) -> bool:
    """A short answer already proves the field was dropped; a full budget or a refusal needs the control."""
    if asked is ProbeFailure.UNREACHABLE:
        return False
    return not isinstance(asked, ProbeAnswer) or asked.spent_whole_budget


def _support(asked: ProbeOutcome, control: ProbeOutcome | None) -> IgnoreEosSupport:
    """Judge the field from the answer with it and, when needed, the answer without it.

    A server that honours the field always returns the whole budget, so a shorter answer
    proves it was dropped. A full budget proves nothing on its own: only a control that
    stops early shows the field made the difference, because a model long-winded enough
    to reach the cap by itself fills it either way. A refusal is read the same way: a
    strict endpoint rejects the field outright, and a control it does answer shows the
    refusal was the field, not the model.
    """
    if asked is ProbeFailure.UNREACHABLE:
        return IgnoreEosSupport.UNDETERMINED
    if asked is ProbeFailure.REFUSED:
        return IgnoreEosSupport.UNDETERMINED if not isinstance(control, ProbeAnswer) else IgnoreEosSupport.IGNORED
    if not asked.spent_whole_budget:
        return IgnoreEosSupport.IGNORED
    if isinstance(control, ProbeAnswer) and not control.spent_whole_budget:
        return IgnoreEosSupport.HONOURED
    return IgnoreEosSupport.UNDETERMINED


async def probe_ignore_eos(
    base_url: str,
    model: str,
    client_backend: ClientBackend,
    api_key: str | None = None,
) -> IgnoreEosProbeResult:
    """Return whether this endpoint generates past end-of-sequence when asked to.

    The probe travels the client the run will use and is judged by the predicate the
    replay applies to every exact-policy turn, so what it proves is what the run needs.
    Endpoint misbehavior never raises; it leaves the question undetermined.
    """
    client = streaming_client(
        client_backend,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=IGNORE_EOS_PROBE_TIMEOUT_SECONDS,
        max_connections=PROBE_MAX_CONNECTIONS,
    )
    try:
        asked = await _answer(client, ignore_eos_probe_request(model, ignore_eos=True))
        control = (
            await _answer(client, ignore_eos_probe_request(model, ignore_eos=False)) if _needs_control(asked) else None
        )
    finally:
        await client.close()
    return IgnoreEosProbeResult(
        support=_support(asked, control),
        observed=asked if isinstance(asked, ProbeAnswer) else None,
    )
