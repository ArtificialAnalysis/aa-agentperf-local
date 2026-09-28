"""Exercise the public-minimal submission boundary."""

import hashlib
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import orjson
import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from pydantic import BaseModel
from pydantic import ValidationError as ModelValidationError

from agentperf_local.cli import main
from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object, pretty_json_bytes
from agentperf_local.common.models import error_text, replace_fields
from agentperf_local.deployment.qualification import (
    QUALIFICATION_FILENAME,
    QUALIFICATION_PROBE_IDS,
    ProbeOutcome,
    RuntimeQualification,
    write_runtime_qualification,
)
from agentperf_local.provenance.benchmark import (
    SourceProvenance,
    SubmissionContext,
    create_measurement_binding,
    workload_digest,
    write_measurement_binding,
)
from agentperf_local.provenance.hardware import HardwareSnapshot, public_hardware_profile
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from agentperf_local.reports.reporting import normalize_output_length
from agentperf_local.submission.aggregate import build_public_submission
from agentperf_local.submission.bundle import (
    AGGREGATE_FILENAME,
    BUNDLE_MANIFEST_FILENAME,
    MAX_BUNDLE_MANIFEST_BYTES,
    SANITIZED_EVIDENCE_FILENAME,
    build_submission_bundle,
    validate_submission_bundle,
    write_submission_bundle,
)
from agentperf_local.submission.evidence import (
    build_sanitized_evidence,
    recompute_sanitized_metrics,
    validate_sanitized_evidence,
)
from agentperf_local.submission.private_audit import PRIVATE_AUDIT_FILENAME, PrivateAudit
from agentperf_local.workload.recording import convert_one_recording_to_dir
from agentperf_local.workload.schema import load_manifest, load_trace
from tests.file_modes import lacks_mode_bits

RECORDING = Path(__file__).parent / "fixtures" / "recording" / "recordings" / "demo.json"
PUBLIC_SUBMISSION_SCHEMA = Path(__file__).parents[1] / "docs" / "schemas" / "public-submission-v2.schema.json"
SANITIZED_EVIDENCE_SCHEMA = Path(__file__).parents[1] / "docs" / "schemas" / "sanitized-turn-evidence-v1.schema.json"
BUNDLE_MANIFEST_SCHEMA = Path(__file__).parents[1] / "docs" / "schemas" / "submission-bundle-v2.schema.json"
PRIVATE_AUDIT_SCHEMA = Path(__file__).parents[1] / "docs" / "schemas" / "private-audit-v1.schema.json"
OTHER_RUN_ID = "1f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"

RUN_ID = "8f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"
TURN_PROMPT_TOKENS = 500
TURN_CACHED_PROMPT_TOKENS = 400
TURN_OUTPUT_TOKENS = 50
TURN_GENERATION_MS = 200.0
EXTRA_MODEL_CALL_TIMESTAMP = 10.0
NON_ROUND_E2E_MS = 800.1 / 3
NON_ROUND_TTFT_MS = 60.1 / 7
NON_ROUND_GENERATION_MS = 200.1 / 7
# Per-turn tool call delays whose flat sum differs from the sum of per-turn subtotals at 2, 3, and 5 turns.
TOOL_CALL_DELAYS_MS = (
    (165.92, 276.03),
    (179.85, 283.91),
    (54.34, 292.63),
    (21.5, 127.79),
    (11.89, 243.82),
)

PRIVATE_MARKERS = (
    b"alice",
    b"secret-endpoint",
    b"secret-model-alias",
    b"secret-cache-namespace",
    b"secret-tool-profile",
    b"secret-container",
    b"secret-future-field",
    b"secret-task-artifact",
    b"secret-tool-artifact",
    b"secret-failure-artifact",
    b"secret-turn-artifact",
    b"secret-measurement-artifact",
    b"wire prompt",
    b"wire follow-up",
)


def test_public_submission_is_deterministic_and_excludes_private_fields(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    first = build_public_submission(results_dir)
    second = build_public_submission(results_dir)
    encoded = pretty_json_bytes(first.to_json())

    assert first.payload_digest == second.payload_digest
    assert first.to_json() == second.to_json()
    assert all(marker not in encoded for marker in PRIVATE_MARKERS)
    assert b"namespace_digits" in encoded
    assert b"community-self-reported" in encoded
    assert b"test-cpu" not in encoded
    assert b"test-kernel" not in encoded
    assert b"590.42" not in encoded
    assert b'"hardware": null' in encoded
    assert orjson.loads(encoded)["payload_digest"] == first.payload_digest


def test_public_submission_schema_is_strict(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    submission = build_public_submission(results_dir).to_json()
    schema = normalize_json_object(orjson.loads(PUBLIC_SUBMISSION_SCHEMA.read_bytes()))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    validator.validate(submission)
    contradictory_cache = orjson.loads(orjson.dumps(submission))
    contradictory_cache["payload"]["run"]["policy"]["cache_isolation"]["mode"] = "none"
    with pytest.raises(ValidationError):
        validator.validate(contradictory_cache)
    unsupported_tool_mode = orjson.loads(orjson.dumps(submission))
    unsupported_tool_mode["payload"]["run"]["policy"]["tool_replay"]["mode"] = "profiled"
    with pytest.raises(ValidationError):
        validator.validate(unsupported_tool_mode)
    payload = submission["payload"]
    assert isinstance(payload, dict)
    payload["endpoint"] = "https://private.example/v1"

    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        validator.validate(submission)


def test_sanitized_evidence_is_ordinal_only_and_excludes_private_fields(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    submission, evidence = build_sanitized_evidence(results_dir)
    encoded = pretty_json_bytes(evidence.to_json())
    data = orjson.loads(encoded)
    canonical_rows = orjson.dumps(data["turns"], option=orjson.OPT_SORT_KEYS)
    schema = normalize_json_object(orjson.loads(SANITIZED_EVIDENCE_SCHEMA.read_bytes()))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    assert evidence.aggregate_payload_digest == submission.payload_digest
    assert evidence.source_turns_digest == submission.artifacts.turns
    assert evidence.rows_digest == f"sha256:{hashlib.sha256(canonical_rows).hexdigest()}"
    assert "normalized_generation_ms" not in encoded.decode()
    assert "normalized_e2e_latency_ms" not in encoded.decode()
    recomputed = recompute_sanitized_metrics(evidence)
    assert recomputed.totals == submission.run.totals
    assert recomputed.latency_distributions_ms == submission.run.latency_distributions_ms
    validator.validate(data)
    assert [turn.turn_ordinal for turn in evidence.turns] == [0, 1]
    assert [turn.turn_in_task for turn in evidence.turns] == [0, 1]
    assert evidence.turns[0].tokens.server_cached_prompt_tokens == TURN_CACHED_PROMPT_TOKENS
    assert evidence.turns[0].tokens.server_uncached_prompt_tokens == (TURN_PROMPT_TOKENS - TURN_CACHED_PROMPT_TOKENS)
    assert all(marker not in encoded for marker in PRIVATE_MARKERS)
    assert b"recording_task" not in encoded
    assert b"experimental-local-preview-no-upload" in encoded
    data["turns"][0]["prompt"] = "private prompt"
    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        validator.validate(data)


def test_sanitized_evidence_round_trips_through_semantic_validator(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    _, evidence = build_sanitized_evidence(results_dir)
    loaded = validate_sanitized_evidence(normalize_json_object(orjson.loads(pretty_json_bytes(evidence.to_json()))))

    assert loaded == evidence

    legacy = evidence.to_json()
    legacy_turns = legacy.get("turns")
    assert isinstance(legacy_turns, list)
    for turn in legacy_turns:
        assert isinstance(turn, dict)
        tokens = turn.get("tokens")
        assert isinstance(tokens, dict)
        del tokens["server_cached_prompt_tokens"]
        del tokens["server_uncached_prompt_tokens"]
    _refresh_rows_digest(legacy)

    loaded_legacy = validate_sanitized_evidence(legacy)

    assert all(turn.tokens.server_cached_prompt_tokens is None for turn in loaded_legacy.turns)
    assert all(turn.tokens.server_uncached_prompt_tokens is None for turn in loaded_legacy.turns)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("privacy_profile", "different-profile"),
        ("normalization_policy_id", "different-policy"),
        ("status", "upload-ready"),
    ],
)
def test_sanitized_semantic_validator_rejects_envelope_relabeling(
    tmp_path: Path,
    field: str,
    value: int | str,
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    _, evidence = build_sanitized_evidence(results_dir)
    data = evidence.to_json()
    data[field] = value

    with pytest.raises(ValueError):
        validate_sanitized_evidence(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("turn_ordinal", 1, "turn ordinals"),
        ("task_ordinal", 1, "task layout"),
        ("turn_in_task", 1, "task layout"),
        ("response_chunks", 10_000_001, "chunk count"),
        ("replayed_pacing_ms", -1.0, "pacing values"),
    ],
)
def test_sanitized_semantic_validator_rejects_impossible_turns(
    tmp_path: Path,
    field: str,
    value: int | float,
    message: str,
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    _, evidence = build_sanitized_evidence(results_dir)
    data = evidence.to_json()
    turns = data.get("turns")
    assert isinstance(turns, list)
    first = turns[0]
    assert isinstance(first, dict)
    first[field] = value
    _refresh_rows_digest(data)

    with pytest.raises(ValueError, match=message):
        validate_sanitized_evidence(data)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("timing", "e2e_latency_ms", -1.0),
        ("timing", "generation_ms", 86_400_001.0),
        ("tokens", "target_output_tokens", 0),
        ("tokens", "server_prompt_tokens", 10_000_001),
        ("tokens", "observed_output_tokens", 49),
    ],
)
def test_sanitized_semantic_validator_rejects_invalid_nested_metrics(
    tmp_path: Path,
    section: str,
    field: str,
    value: int | float,
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    _, evidence = build_sanitized_evidence(results_dir)
    data = evidence.to_json()
    turns = data.get("turns")
    assert isinstance(turns, list)
    first = turns[0]
    assert isinstance(first, dict)
    nested = first.get(section)
    assert isinstance(nested, dict)
    nested[field] = value
    _refresh_rows_digest(data)

    with pytest.raises(ValueError):
        validate_sanitized_evidence(data)


def test_sanitized_semantic_validator_rejects_rows_digest_mismatch(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    _, evidence = build_sanitized_evidence(results_dir)
    data = evidence.to_json()
    data["rows_digest"] = f"sha256:{'f' * 64}"

    with pytest.raises(ValueError, match="rows digest"):
        validate_sanitized_evidence(data)


def test_public_packaging_rejects_precomputed_normalization_tampering(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    def relabel_normalized_generation(turns: list[JsonObject]) -> None:
        _nested_object(turns[0], ("normalization",))["normalized_generation_ms"] = 199.0

    _rewrite_turns(results_dir, relabel_normalized_generation)

    with pytest.raises(ValueError, match="normalized generation timing"):
        build_sanitized_evidence(results_dir)


@pytest.mark.parametrize("turn_count", [2, 3, 5])
def test_sanitized_rows_reproduce_totals_for_non_round_durations(tmp_path: Path, turn_count: int) -> None:
    results_dir = _write_non_round_results(tmp_path, turn_count)
    tools = normalize_json_object(orjson.loads((results_dir / "tools.json").read_bytes()))
    summary = normalize_json_object(orjson.loads((results_dir / "summary.json").read_bytes()))
    flat_recorded_ms = _nested_object(tools, ("overall",)).get("recorded_duration_ms")
    per_turn_recorded_ms = _nested_object(summary, ("totals",)).get("total_recorded_tool_delay_ms")
    assert isinstance(flat_recorded_ms, float)
    assert isinstance(per_turn_recorded_ms, float)
    assert flat_recorded_ms != per_turn_recorded_ms
    assert math.isclose(flat_recorded_ms, per_turn_recorded_ms, rel_tol=1e-9, abs_tol=1e-6)

    submission, evidence = build_sanitized_evidence(results_dir)
    recomputed = recompute_sanitized_metrics(evidence)

    assert len(evidence.turns) == turn_count
    assert recomputed.totals == submission.run.totals
    assert recomputed.latency_distributions_ms == submission.run.latency_distributions_ms
    assert submission.run.totals.total_recorded_tool_delay_ms == per_turn_recorded_ms
    assert submission.run.totals.tool_calls == sum(len(delays) for delays in TOOL_CALL_DELAYS_MS[:turn_count])


def test_sanitized_recompute_still_rejects_an_altered_row(tmp_path: Path) -> None:
    results_dir = _write_non_round_results(tmp_path, 3)
    submission, evidence = build_sanitized_evidence(results_dir)
    data = evidence.to_json()
    turns = data.get("turns")
    assert isinstance(turns, list)
    first = turns[0]
    assert isinstance(first, dict)
    timing = _nested_object(first, ("timing",))
    latency = timing.get("e2e_latency_ms")
    assert isinstance(latency, float)
    timing["e2e_latency_ms"] = latency + 1.0
    _refresh_rows_digest(data)

    altered = validate_sanitized_evidence(data)

    assert recompute_sanitized_metrics(altered).totals != submission.run.totals


def _qualification(run_id: str, *, endpoint_model: str = "secret-model-alias") -> RuntimeQualification:
    """Return a passing report bound to one run and one endpoint model."""
    outcomes = tuple(
        ProbeOutcome(
            probe_id=probe_id,
            passed=True,
            failure_codes=(),
            tool_names=(),
            tool_call_count=0,
            arguments_valid=True,
            call_identifiers_present=True,
            content_present=True,
            reasoning_present=False,
            finish_reason="stop",
            usage_present=True,
            prompt_tokens=1,
            completion_tokens=1,
            raw_read_count=1,
            decoded_chunk_count=1,
        )
        for probe_id in QUALIFICATION_PROBE_IDS
    )
    return RuntimeQualification(
        profile_id="fixture",
        endpoint_model_digest=f"sha256:{hashlib.sha256(endpoint_model.encode()).hexdigest()}",
        client_backend="python",
        pack_digest=f"sha256:{'3' * 64}",
        outcomes=outcomes,
        run_id=run_id,
    )


def test_submission_bundle_binds_exact_four_file_bytes(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    bundle = build_submission_bundle(results_dir)
    output_dir = tmp_path / "upload-preview"
    written = write_submission_bundle(output_dir, bundle)
    aggregate_bytes = (output_dir / AGGREGATE_FILENAME).read_bytes()
    evidence_bytes = (output_dir / SANITIZED_EVIDENCE_FILENAME).read_bytes()
    audit_bytes = (output_dir / PRIVATE_AUDIT_FILENAME).read_bytes()
    manifest_bytes = (output_dir / BUNDLE_MANIFEST_FILENAME).read_bytes()
    manifest = normalize_json_object(orjson.loads(manifest_bytes))
    audit = normalize_json_object(orjson.loads(audit_bytes))
    schema = normalize_json_object(orjson.loads(BUNDLE_MANIFEST_SCHEMA.read_bytes()))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    audit_schema = normalize_json_object(orjson.loads(PRIVATE_AUDIT_SCHEMA.read_bytes()))
    Draft202012Validator.check_schema(audit_schema)
    Draft202012Validator(audit_schema).validate(audit)
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    aggregate_record = artifacts[0]
    evidence_record = artifacts[1]
    audit_record = artifacts[2]
    assert isinstance(aggregate_record, dict)
    assert isinstance(evidence_record, dict)
    assert isinstance(audit_record, dict)

    assert aggregate_bytes == bundle.aggregate_bytes
    assert evidence_bytes == bundle.evidence_bytes
    assert audit_bytes == bundle.audit_bytes
    assert manifest_bytes == bundle.manifest_bytes
    assert manifest["run_id"] == audit["run_id"] == RUN_ID
    assert aggregate_record["file_digest"] == f"sha256:{hashlib.sha256(aggregate_bytes).hexdigest()}"
    assert aggregate_record["byte_size"] == len(aggregate_bytes)
    assert evidence_record["file_digest"] == f"sha256:{hashlib.sha256(evidence_bytes).hexdigest()}"
    assert evidence_record["byte_size"] == len(evidence_bytes)
    assert audit_record["file_digest"] == f"sha256:{hashlib.sha256(audit_bytes).hexdigest()}"
    assert audit_record["privacy_profile"] == "aa-private-audit-v1"
    assert written.manifest_digest == f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"
    assert all(lacks_mode_bits(path, 0o133) for path in output_dir.iterdir())
    with pytest.raises(FileExistsError):
        write_submission_bundle(output_dir, bundle)
    assert (output_dir / AGGREGATE_FILENAME).read_bytes() == aggregate_bytes
    assert (output_dir / SANITIZED_EVIDENCE_FILENAME).read_bytes() == evidence_bytes
    assert (output_dir / BUNDLE_MANIFEST_FILENAME).read_bytes() == manifest_bytes
    assert all(marker not in aggregate_bytes + evidence_bytes + manifest_bytes for marker in PRIVATE_MARKERS)
    # The private audit keeps the full snapshot but still no prompts, paths, endpoints, or secrets.
    assert b"test-cpu" in audit_bytes
    assert b"test-kernel" in audit_bytes
    assert b"590.42" in audit_bytes
    assert all(marker not in audit_bytes for marker in PRIVATE_MARKERS)
    assert audit["deployment"] is None
    assert audit["runtime_qualification"] is None
    assert audit["power"] is None
    assert audit["aggregate_payload_digest"] == bundle.aggregate.payload_digest
    validated = validate_submission_bundle(output_dir)
    assert validated.aggregate_payload_digest == bundle.aggregate.payload_digest
    assert validated.evidence == bundle.evidence
    assert validated.audit == bundle.audit


def test_submission_bundle_validator_rejects_exact_byte_tampering(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    bundle = build_submission_bundle(results_dir)
    output_dir = tmp_path / "upload-preview"
    write_submission_bundle(output_dir, bundle)
    aggregate_path = output_dir / AGGREGATE_FILENAME
    aggregate_path.write_bytes(aggregate_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="bytes do not match"):
        validate_submission_bundle(output_dir)


def test_submission_bundle_validator_rejects_self_consistent_aggregate_tampering(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    output_dir = tmp_path / "upload-preview"
    write_submission_bundle(output_dir, build_submission_bundle(results_dir))
    totals = _payload_object(output_dir, ("run", "totals"))
    inference_latency = totals.get("total_inference_latency_ms")
    assert isinstance(inference_latency, int | float)
    _rewrite_bundle(output_dir, ("run", "totals"), {"total_inference_latency_ms": inference_latency + 1})

    with pytest.raises(ValueError, match="do not reproduce aggregate totals"):
        validate_submission_bundle(output_dir)


@pytest.mark.parametrize(
    ("path", "updates", "message"),
    [
        (("benchmark",), {"suite_id": "https://secret.example/suite"}, "must use letters"),
        (("benchmark",), {"model_artifact_digest": "sha256:private"}, "64 lowercase hex digits"),
        (("benchmark",), {"hostname": "gpu-node-7.internal"}, "unexpected hostname"),
        (("producer",), {"client_name": "different-client"}, "client_name must be agentperf-local"),
        (("producer",), {"operator_email": "alice@example.com"}, "unexpected operator_email"),
        (("hardware",), {"kind": "private_hardware_profile"}, "closed contract"),
        (("hardware",), {"version": 2}, "hardware profile version is not supported"),
        (("hardware",), {"privacy_notice": "Collected from gpu-node-7."}, "closed contract"),
        (("hardware",), {"serial_number": "GPU-84c2"}, "unexpected serial_number"),
        (
            ("hardware",),
            {"platform_family": "gpu-node-7.corp.internal @ /Users/alice/model.gguf"},
            "path, address, or markup character",
        ),
        (
            ("hardware", "accelerator"),
            {"product": "NVIDIA RTX 5090 @ gpu-node-7.corp.internal"},
            "path, address, or markup character",
        ),
        (("hardware", "accelerator"), {"driver_branch": "5" * 200}, "short printable ASCII text"),
        (("hardware", "accelerator"), {"product": 5}, "must be non-empty text"),
        (("hardware", "accelerator"), {"uuid": "GPU-0d3f"}, "unexpected uuid"),
        (("private_evidence_digests",), {"turns": "/Users/alice/private/turns.jsonl"}, "64 lowercase hex digits"),
        (("private_evidence_digests",), {"results_path": "/Users/alice/private"}, "unexpected results_path"),
        (("run", "policy"), {"client_backend": "secret-backend"}, "client_backend must be python or rust"),
        (("run", "policy"), {"api_key": "secret-endpoint-token"}, "unexpected api_key"),
        (("run", "policy", "sampling"), {"seed": 7}, "closed contract"),
        (("run", "policy", "tool_replay"), {"mode": "live"}, "only none or recorded"),
        (
            ("run", "policy", "context"),
            {"requested_tokens": 32768, "observed_tokens": 32768, "reduced": True},
            "benchmark context does not match",
        ),
        (("run", "policy", "context"), {"observed_tokens": 32768}, "context.reduced must match"),
        (("benchmark",), {"context_tokens": 8192}, "benchmark context does not match the run policy"),
    ],
)
def test_submission_bundle_validator_rejects_replaced_payload_values(
    tmp_path: Path,
    path: tuple[str, ...],
    updates: JsonObject,
    message: str,
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    output_dir = tmp_path / "upload-preview"
    write_submission_bundle(output_dir, build_submission_bundle(results_dir))
    _rewrite_bundle(output_dir, path, updates)

    with pytest.raises(ValueError, match=message):
        validate_submission_bundle(output_dir)


def test_submission_bundle_validator_rejects_extra_linked_and_oversized_members(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    output_dir = tmp_path / "upload-preview"
    write_submission_bundle(output_dir, build_submission_bundle(results_dir))
    extra_path = output_dir / "extra.json"
    extra_path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="exactly the four contract files"):
        validate_submission_bundle(output_dir)

    extra_path.unlink()
    aggregate_path = output_dir / AGGREGATE_FILENAME
    outside_path = tmp_path / "outside.json"
    outside_path.write_bytes(aggregate_path.read_bytes())
    aggregate_path.unlink()
    aggregate_path.symlink_to(outside_path)
    with pytest.raises(ValueError, match="regular file"):
        validate_submission_bundle(output_dir)

    aggregate_path.unlink()
    aggregate_path.write_bytes(outside_path.read_bytes())
    manifest_path = output_dir / BUNDLE_MANIFEST_FILENAME
    manifest_path.write_bytes(b"x" * (MAX_BUNDLE_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="outside the accepted range"):
        validate_submission_bundle(output_dir)


def test_prepare_submission_cli_is_local_and_no_clobber(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    output_dir = tmp_path / "prepared"

    assert main(["prepare-submission", str(results_dir), "--output-dir", str(output_dir)]) == 0
    response = orjson.loads(capsys.readouterr().out)
    assert response["upload_performed"] is False
    assert response["status"] == "prepared-not-uploaded"
    assert response["run_id"] == RUN_ID
    assert response["private_audit"] == {
        "deployment_record": False,
        "runtime_qualification": False,
        "power_summary": False,
    }
    manifest_bytes = (output_dir / BUNDLE_MANIFEST_FILENAME).read_bytes()

    assert main(["prepare-submission", str(results_dir), "--output-dir", str(output_dir)]) == 1
    assert (output_dir / BUNDLE_MANIFEST_FILENAME).read_bytes() == manifest_bytes
    assert "already exists" in capsys.readouterr().err

    private_output = results_dir / "prepared"
    assert main(["prepare-submission", str(results_dir), "--output-dir", str(private_output)]) == 1
    assert not private_output.exists()
    assert "outside the private results directory" in capsys.readouterr().err


class _SyntheticTurn(BaseModel, frozen=True):
    """Hold one synthetic turn's duration inputs."""

    e2e_ms: float
    ttft_ms: float
    generation_ms: float
    tool_call_delays_ms: tuple[float, ...]

    @property
    def tool_delay_ms(self) -> float:
        """Return the turn subtotal the runner writes for its own calls."""
        return sum(self.tool_call_delays_ms, 0.0)


def _synthetic_turns(turn_count: int) -> tuple[_SyntheticTurn, ...]:
    return tuple(
        _SyntheticTurn(
            e2e_ms=NON_ROUND_E2E_MS + index,
            ttft_ms=NON_ROUND_TTFT_MS + index,
            generation_ms=NON_ROUND_GENERATION_MS + index,
            tool_call_delays_ms=TOOL_CALL_DELAYS_MS[index],
        )
        for index in range(turn_count)
    )


def _normalized_ms(turn: _SyntheticTurn) -> tuple[float, float]:
    report = normalize_output_length(
        e2e_latency_ms=turn.e2e_ms,
        generation_ms=turn.generation_ms,
        observed_output_tokens=TURN_OUTPUT_TOKENS,
        target_output_tokens=TURN_OUTPUT_TOKENS,
    )
    generation_ms = report.normalized_generation_ms
    e2e_ms = report.normalized_e2e_latency_ms
    assert generation_ms is not None and e2e_ms is not None
    return generation_ms, e2e_ms


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: tuple[float, ...]) -> JsonObject:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _synthetic_summary(turns: tuple[_SyntheticTurn, ...]) -> JsonObject:
    """Return a private summary whose totals associate durations per turn."""
    normalized = tuple(_normalized_ms(turn) for turn in turns)
    normalized_e2e = tuple(e2e_ms for _, e2e_ms in normalized)
    inference_ms = sum(turn.e2e_ms for turn in turns)
    tool_delay_ms = sum(turn.tool_delay_ms for turn in turns)
    summary = _private_summary()
    summary["totals"] = {
        "turns": len(turns),
        "successful_turns": len(turns),
        "failed_turns": 0,
        "short_output_warnings": 0,
        "tool_calls": sum(len(turn.tool_call_delays_ms) for turn in turns),
        "total_inference_latency_ms": inference_ms,
        "total_recorded_tool_delay_ms": tool_delay_ms,
        "total_replayed_tool_delay_ms": tool_delay_ms,
        "total_agentic_replay_ms": inference_ms + tool_delay_ms,
        "total_normalized_generation_ms": sum(generation_ms for generation_ms, _ in normalized),
        "total_normalized_inference_latency_ms": sum(normalized_e2e),
        "total_normalized_agentic_replay_ms": sum(
            e2e_ms + turn.tool_delay_ms for e2e_ms, turn in zip(normalized_e2e, turns, strict=True)
        ),
        "total_server_prompt_tokens": TURN_PROMPT_TOKENS * len(turns),
        "total_server_output_tokens": TURN_OUTPUT_TOKENS * len(turns),
        "total_local_output_tokens": TURN_OUTPUT_TOKENS * len(turns),
        "total_recorded_prompt_tokens": TURN_PROMPT_TOKENS * len(turns),
        "total_recorded_completion_tokens": TURN_OUTPUT_TOKENS * len(turns),
    }
    summary["latency_distributions_ms"] = {
        "e2e": _distribution(tuple(turn.e2e_ms for turn in turns)),
        "time_to_first_token": _distribution(tuple(turn.ttft_ms for turn in turns)),
        "normalized_e2e": _distribution(normalized_e2e),
    }
    return summary


def _recording_with_turns(turn_count: int) -> JsonObject:
    recording = normalize_json_object(orjson.loads(RECORDING.read_bytes()))
    events = recording.get("events")
    assert isinstance(events, list)
    model_calls = [event for event in events if isinstance(event, dict) and event.get("type") == "model_call"]
    extra: list[JsonValue] = []
    for index in range(turn_count - len(model_calls)):
        clone = normalize_json_object(orjson.loads(orjson.dumps(model_calls[-1])))
        clone["timestamp"] = EXTRA_MODEL_CALL_TIMESTAMP + index
        extra.append(clone)
    recording["events"] = [*events, *extra]
    return recording


def _write_non_round_results(root: Path, turn_count: int) -> Path:
    """Write bound results whose per-turn durations are non-round floats."""
    recording_path = root / "non-round-recording.json"
    recording_path.write_bytes(orjson.dumps(_recording_with_turns(turn_count)))
    manifest_path, turn_ids = _write_workload(root, recording_path)
    assert len(turn_ids) == turn_count
    synthetic = _synthetic_turns(turn_count)
    turns = tuple(
        _private_turn(
            turn_id,
            index,
            e2e_ms=turn.e2e_ms,
            ttft_ms=turn.ttft_ms,
            tool_call_delays_ms=turn.tool_call_delays_ms,
            generation_ms=turn.generation_ms,
        )
        for index, (turn_id, turn) in enumerate(zip(turn_ids, synthetic, strict=True))
    )
    return _write_results_dir(root, _synthetic_summary(synthetic), turns, manifest_path, None)


def _nested_object(data: JsonObject, path: tuple[str, ...]) -> JsonObject:
    current = data
    for key in path:
        nested = current.get(key)
        assert isinstance(nested, dict)
        current = nested
    return current


def _payload_object(output_dir: Path, path: tuple[str, ...]) -> JsonObject:
    """Return one nested payload object from a written bundle."""
    aggregate = normalize_json_object(orjson.loads((output_dir / AGGREGATE_FILENAME).read_bytes()))
    return _nested_object(_nested_object(aggregate, ("payload",)), path)


def _rewrite_bundle(output_dir: Path, path: tuple[str, ...], updates: JsonObject) -> None:
    """Apply one payload edit and rebind every digest the bundle carries."""
    aggregate_path = output_dir / AGGREGATE_FILENAME
    evidence_path = output_dir / SANITIZED_EVIDENCE_FILENAME
    audit_path = output_dir / PRIVATE_AUDIT_FILENAME
    manifest_path = output_dir / BUNDLE_MANIFEST_FILENAME
    aggregate = normalize_json_object(orjson.loads(aggregate_path.read_bytes()))
    evidence = normalize_json_object(orjson.loads(evidence_path.read_bytes()))
    audit = normalize_json_object(orjson.loads(audit_path.read_bytes()))
    manifest = normalize_json_object(orjson.loads(manifest_path.read_bytes()))
    payload = _nested_object(aggregate, ("payload",))
    if path and path[0] == "hardware" and payload.get("hardware") is None:
        payload["hardware"] = public_hardware_profile(_hardware()).to_json()
    _nested_object(payload, path).update(updates)
    payload_digest = f"sha256:{hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()}"
    aggregate["payload_digest"] = payload_digest
    evidence["aggregate_payload_digest"] = payload_digest
    audit["aggregate_payload_digest"] = payload_digest
    manifest["aggregate_payload_digest"] = payload_digest
    _rebind_bundle_files(output_dir, aggregate, evidence, audit, manifest)


def _rebind_bundle_files(
    output_dir: Path,
    aggregate: JsonObject,
    evidence: JsonObject,
    audit: JsonObject,
    manifest: JsonObject,
) -> None:
    """Write every bundle file and refresh the manifest's exact-byte records."""
    options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
    aggregate_bytes = orjson.dumps(aggregate, option=options)
    evidence_bytes = orjson.dumps(evidence, option=options)
    audit_bytes = orjson.dumps(audit, option=options)
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list)
    for artifact, encoded in zip(artifacts, (aggregate_bytes, evidence_bytes, audit_bytes), strict=True):
        assert isinstance(artifact, dict)
        artifact["byte_size"] = len(encoded)
        artifact["file_digest"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    (output_dir / AGGREGATE_FILENAME).write_bytes(aggregate_bytes)
    (output_dir / SANITIZED_EVIDENCE_FILENAME).write_bytes(evidence_bytes)
    (output_dir / PRIVATE_AUDIT_FILENAME).write_bytes(audit_bytes)
    (output_dir / BUNDLE_MANIFEST_FILENAME).write_bytes(orjson.dumps(manifest, option=options))


def _rewrite_audit(output_dir: Path, path: tuple[str, ...], updates: JsonObject) -> None:
    """Apply one private audit edit and rebind the manifest's exact-byte record."""
    aggregate = normalize_json_object(orjson.loads((output_dir / AGGREGATE_FILENAME).read_bytes()))
    evidence = normalize_json_object(orjson.loads((output_dir / SANITIZED_EVIDENCE_FILENAME).read_bytes()))
    audit = normalize_json_object(orjson.loads((output_dir / PRIVATE_AUDIT_FILENAME).read_bytes()))
    manifest = normalize_json_object(orjson.loads((output_dir / BUNDLE_MANIFEST_FILENAME).read_bytes()))
    _nested_object(audit, path).update(updates)
    _rebind_bundle_files(output_dir, aggregate, evidence, audit, manifest)


@pytest.mark.parametrize(
    ("path", "updates", "message"),
    [
        ((), {"run_id": OTHER_RUN_ID}, "run identifiers do not match"),
        (("hardware",), {"hostname": "gpu-node-7"}, "closed contract"),
        (
            (),
            {"retention": {"recipient": "artificial-analysis", "published": True, "bounded_days": 180}},
            "not supported",
        ),
        ((), {"operator_email": "alice@example.com"}, "unexpected operator_email"),
    ],
    ids=("run-id", "hardware-extra-key", "retention", "extra-key"),
)
def test_bundle_validator_rejects_a_private_audit_that_drifts_from_the_aggregate(
    tmp_path: Path,
    path: tuple[str, ...],
    updates: JsonObject,
    message: str,
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    output_dir = tmp_path / "upload-preview"
    write_submission_bundle(output_dir, build_submission_bundle(results_dir))
    _rewrite_audit(output_dir, path, updates)

    with pytest.raises(ValueError, match=message):
        validate_submission_bundle(output_dir)


def test_private_audit_carries_only_a_qualification_that_names_this_run_and_endpoint(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    qualification_path = results_dir / QUALIFICATION_FILENAME
    write_runtime_qualification(qualification_path, _qualification(RUN_ID))

    bundle = build_submission_bundle(results_dir)
    output_dir = tmp_path / "upload-preview"
    write_submission_bundle(output_dir, bundle)
    validated = validate_submission_bundle(output_dir)
    audit = normalize_json_object(orjson.loads((output_dir / PRIVATE_AUDIT_FILENAME).read_bytes()))
    Draft202012Validator(normalize_json_object(orjson.loads(PRIVATE_AUDIT_SCHEMA.read_bytes()))).validate(audit)

    assert bundle.audit.runtime_qualification == _qualification(RUN_ID)
    assert validated.audit.runtime_qualification is not None
    assert validated.audit.runtime_qualification.passed
    qualification_path.unlink()
    write_runtime_qualification(qualification_path, _qualification(OTHER_RUN_ID))
    with pytest.raises(ValueError, match="belongs to a different run"):
        build_submission_bundle(results_dir)
    qualification_path.unlink()
    write_runtime_qualification(qualification_path, _qualification(RUN_ID, endpoint_model="another-model"))
    with pytest.raises(ValueError, match="probed a different endpoint model"):
        build_submission_bundle(results_dir)
    with pytest.raises(ValueError, match="belongs to a different run"):
        PrivateAudit(
            run_id=RUN_ID,
            aggregate_payload_digest=bundle.aggregate.payload_digest,
            hardware=bundle.audit.hardware,
            deployment=None,
            runtime_qualification=_qualification(OTHER_RUN_ID),
            power=None,
        )


def _rewrite_turns(results_dir: Path, mutate: Callable[[list[JsonObject]], None]) -> None:
    """Apply one edit to the private turn rows and rewrite the evidence file."""
    turns_path = results_dir / "turns.jsonl"
    turns = [normalize_json_object(orjson.loads(line)) for line in turns_path.read_bytes().splitlines()]
    mutate(turns)
    turns_path.write_bytes(b"".join(orjson.dumps(turn, option=orjson.OPT_APPEND_NEWLINE) for turn in turns))


def _refresh_rows_digest(data: JsonObject) -> None:
    turns = data.get("turns")
    assert isinstance(turns, list)
    encoded = orjson.dumps(turns, option=orjson.OPT_SORT_KEYS)
    data["rows_digest"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _hardware() -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system="Linux",
        operating_system_version="test-os",
        kernel_version="test-kernel",
        architecture="x86_64",
        cpu_model="test-cpu",
        logical_cpu_count=24,
        memory_bytes=128 * 1024**3,
        accelerators=(
            AcceleratorSnapshot(
                vendor="NVIDIA",
                name="NVIDIA RTX 5090",
                memory_bytes=32 * 1024**3,
                core_count=None,
                driver_version="590.42",
                api="CUDA",
            ),
        ),
        warnings=(),
    )


def _write_workload(root: Path, recording_path: Path) -> tuple[Path, tuple[str, ...]]:
    """Convert one recording and return its manifest path and turn IDs."""
    workload_dir = root / "workload"
    convert_one_recording_to_dir(recording_path, workload_dir)
    manifest_path = workload_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    turn_ids = tuple(row.turn_id for task in manifest.tasks for row in load_trace(workload_dir / task.trace))
    return manifest_path, turn_ids


def _turn_tool_calls(turn: JsonObject) -> tuple[JsonObject, ...]:
    """Return one turn's recorded tool calls."""
    calls = turn.get("tool_calls")
    assert isinstance(calls, list)
    return tuple(call for call in calls if isinstance(call, dict))


def _numbers(rows: tuple[JsonObject, ...], key: str) -> tuple[float, ...]:
    """Return one float field from every row."""
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        assert isinstance(value, float)
        values.append(value)
    return tuple(values)


def _integers(rows: tuple[JsonObject, ...], key: str) -> tuple[int, ...]:
    """Return one integer field from every row."""
    values: list[int] = []
    for row in rows:
        value = row.get(key)
        assert isinstance(value, int)
        values.append(value)
    return tuple(values)


def _sections(turns: tuple[JsonObject, ...], name: str) -> tuple[JsonObject, ...]:
    """Return one nested object from every turn."""
    return tuple(_nested_object(turn, (name,)) for turn in turns)


def _tool_overall(turns: tuple[JsonObject, ...]) -> JsonObject:
    """Add every tool call once, exactly as the runner's tool accumulator does."""
    calls = tuple(call for turn in turns for call in _turn_tool_calls(turn))
    return {
        "calls": len(calls),
        "failures": 0,
        "recorded_duration_ms": sum(_numbers(calls, "recorded_duration_ms"), 0.0),
        "replayed_duration_ms": sum(_numbers(calls, "replayed_duration_ms"), 0.0),
        "returncode_mismatches": 0,
    }


def _task_totals(turns: tuple[JsonObject, ...]) -> JsonObject:
    """Recompute task totals from the turn rows with an exact sum."""
    timings = _sections(turns, "timing")
    normalizations = _sections(turns, "normalization")
    tokens = _sections(turns, "tokens")
    e2e_ms = _numbers(timings, "e2e_latency_ms")
    normalized_e2e_ms = _numbers(normalizations, "normalized_e2e_latency_ms")
    replayed_ms = _numbers(turns, "replayed_tool_delay_ms")
    return {
        "turns": len(turns),
        "successful_turns": len(turns),
        "failed_turns": 0,
        "short_output_warnings": 0,
        "tool_calls": sum(len(_turn_tool_calls(turn)) for turn in turns),
        "total_inference_latency_ms": math.fsum(e2e_ms),
        "total_recorded_tool_delay_ms": math.fsum(_numbers(turns, "recorded_tool_delay_ms")),
        "total_replayed_tool_delay_ms": math.fsum(replayed_ms),
        "total_agentic_replay_ms": math.fsum((*e2e_ms, *replayed_ms)),
        "total_normalized_generation_ms": math.fsum(_numbers(normalizations, "normalized_generation_ms")),
        "total_normalized_inference_latency_ms": math.fsum(normalized_e2e_ms),
        "total_normalized_agentic_replay_ms": math.fsum((*normalized_e2e_ms, *replayed_ms)),
        "total_server_prompt_tokens": sum(_integers(tokens, "server_prompt_tokens")),
        "total_server_output_tokens": sum(_integers(tokens, "server_output_tokens")),
        "total_local_output_tokens": sum(_integers(tokens, "local_output_tokens")),
        "total_recorded_prompt_tokens": sum(_integers(tokens, "recorded_prompt_tokens")),
        "total_recorded_completion_tokens": sum(_integers(tokens, "recorded_completion_tokens")),
    }


def _write_bound_results(
    root: Path,
    summary: JsonObject,
    hardware: HardwareSnapshot | None = None,
) -> Path:
    manifest_path, turn_ids = _write_workload(root, RECORDING)
    assert len(turn_ids) == 2
    turns = (
        _private_turn(turn_ids[0], 0, e2e_ms=800.0, ttft_ms=60.0, tool_call_delays_ms=TOOL_CALL_DELAYS_MS[0]),
        _private_turn(turn_ids[1], 1, e2e_ms=1000.0, ttft_ms=100.0, tool_call_delays_ms=()),
    )
    return _write_results_dir(root, summary, turns, manifest_path, hardware)


def _summary_observed_context(summary: JsonObject) -> int | None:
    """Mirror the run path: the binding records the same observation the summary reports."""
    config = summary.get("config")
    context = config.get("context") if isinstance(config, dict) else None
    observed = context.get("observed_tokens") if isinstance(context, dict) else None
    if isinstance(observed, int) and not isinstance(observed, bool) and observed > 0:
        return observed
    return None


def _summary_requested_context(summary: JsonObject) -> int:
    """Return the requested context the binding records as benchmark identity."""
    config = summary.get("config")
    context = config.get("context") if isinstance(config, dict) else None
    requested = context.get("requested_tokens") if isinstance(context, dict) else None
    if isinstance(requested, int) and not isinstance(requested, bool) and requested > 0:
        return requested
    raise ValueError("fixture summary has no requested context")


def _write_results_dir(
    root: Path,
    summary: JsonObject,
    turns: tuple[JsonObject, ...],
    manifest_path: Path,
    hardware: HardwareSnapshot | None,
) -> Path:
    results_dir = root / "private"
    results_dir.mkdir()
    (results_dir / "summary.json").write_bytes(orjson.dumps(summary))
    (results_dir / "turns.jsonl").write_bytes(
        b"".join(orjson.dumps(turn, option=orjson.OPT_APPEND_NEWLINE) for turn in turns)
    )
    task_artifact: JsonObject = {
        "version": 1,
        "kind": "task_summaries",
        "private": "secret-task-artifact",
        "tasks": [
            {
                "task_id": "recording_task",
                "failed_turn_ids": [],
                "totals": _task_totals(turns),
            }
        ],
    }
    tool_artifact: JsonObject = {
        "version": 1,
        "kind": "tool_summary",
        "private": "secret-tool-artifact",
        "overall": _tool_overall(turns),
    }
    failures_artifact: JsonObject = {
        "version": 1,
        "kind": "failures",
        "failures": [],
        "private": "secret-failure-artifact",
    }
    (results_dir / "tasks.json").write_bytes(orjson.dumps(task_artifact))
    (results_dir / "tools.json").write_bytes(orjson.dumps(tool_artifact))
    (results_dir / "failures.json").write_bytes(orjson.dumps(failures_artifact))
    context = SubmissionContext(
        suite_id="aa-agentic-gpu-core",
        suite_epoch="2026-q4",
        suite_digest=workload_digest(manifest_path),
        model_semantics_id="gpt-oss-20b-mxfp4",
        model_artifact_digest=f"sha256:{'1' * 64}",
        runtime_id="llama-cpp-b1234",
        context_tokens=_summary_requested_context(summary),
    )
    binding = create_measurement_binding(
        context,
        manifest_path,
        "secret-model-alias",
        hardware if hardware is not None else _hardware(),
        SourceProvenance(
            client_version="0.1.0",
            source_revision="2" * 40,
            source_state="clean",
        ),
        observed_context_tokens=_summary_observed_context(summary),
        run_id=RUN_ID,
    )
    write_measurement_binding(results_dir / "measurement.json", binding)
    measurement_path = results_dir / "measurement.json"
    measurement = orjson.loads(measurement_path.read_bytes())
    measurement["private_future_field"] = "secret-measurement-artifact"
    measurement_path.write_bytes(orjson.dumps(measurement))
    return results_dir


def _private_turn(
    turn_id: str,
    index: int,
    *,
    e2e_ms: float,
    ttft_ms: float,
    tool_call_delays_ms: tuple[float, ...],
    generation_ms: float = TURN_GENERATION_MS,
) -> JsonObject:
    """Return one private turn whose pacing is the subtotal of its own tool calls."""
    normalization = normalize_output_length(
        e2e_latency_ms=e2e_ms,
        generation_ms=generation_ms,
        observed_output_tokens=TURN_OUTPUT_TOKENS,
        target_output_tokens=TURN_OUTPUT_TOKENS,
    )
    tool_calls = len(tool_call_delays_ms)
    tool_delay_ms = sum(tool_call_delays_ms, 0.0)
    return {
        "version": 1,
        "kind": "turn",
        "turn_id": turn_id,
        "task_id": "recording_task",
        "conversation_idx": index,
        "success": True,
        "aborted": False,
        "error": None,
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "response_chunks": 4,
        "response_tool_calls": tool_calls,
        "timing": {
            "e2e_latency_ms": e2e_ms,
            "time_to_first_byte_ms": ttft_ms,
            "time_to_first_token_ms": ttft_ms,
            "generation_ms": generation_ms,
        },
        "tokens": {
            "server_prompt_tokens": TURN_PROMPT_TOKENS,
            "server_output_tokens": TURN_OUTPUT_TOKENS,
            "local_output_tokens": TURN_OUTPUT_TOKENS,
            "server_cached_prompt_tokens": TURN_CACHED_PROMPT_TOKENS,
            "server_uncached_prompt_tokens": TURN_PROMPT_TOKENS - TURN_CACHED_PROMPT_TOKENS,
            "recorded_prompt_tokens": TURN_PROMPT_TOKENS,
            "recorded_completion_tokens": TURN_OUTPUT_TOKENS,
            "target_output_tokens": TURN_OUTPUT_TOKENS,
        },
        "normalization": normalization.to_json(),
        "recorded_tool_delay_ms": tool_delay_ms,
        "replayed_tool_delay_ms": tool_delay_ms,
        "private_future_field": "secret-turn-artifact",
        "tool_calls": [
            {
                "private": "tool",
                "exception_info": "",
                "returncode_matches_recorded": None,
                "recorded_duration_ms": delay_ms,
                "replayed_duration_ms": delay_ms,
            }
            for delay_ms in tool_call_delays_ms
        ],
    }


@pytest.mark.parametrize(
    "value",
    [
        "https://secret.example/v1",
        "alice@example.com",
        "C:/Users/alice/model",
        "/Users/alice/model",
        "model?token=secret",
    ],
)
def test_submission_context_rejects_private_identifier_shapes(value: str) -> None:
    with pytest.raises(ValueError, match="must use letters"):
        SubmissionContext(
            suite_id=value,
            suite_epoch="2026-q4",
            suite_digest=f"sha256:{'0' * 64}",
            model_semantics_id="gpt-oss-20b-mxfp4",
            model_artifact_digest=f"sha256:{'1' * 64}",
            runtime_id="llama-cpp-b1234",
        )


@pytest.mark.parametrize(
    ("path", "updates", "message"),
    [
        ((), {"measured_duration_ms": 1999.0}, "durations are inconsistent"),
        ((), {"run_id": "1f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"}, "run_id does not match"),
        (
            (),
            # Consistent durations, so the observer rule itself is what refuses the run.
            {
                "observer": {"enabled": True, "duration_ms": 1000.0, "ranked_eligible": False},
                "wall_duration_ms": 3000.0,
            },
            "observer overhead exceeds",
        ),
        (
            (),
            {"observer": {"enabled": False, "duration_ms": 1.0, "ranked_eligible": False}},
            "cannot report observer time",
        ),
        (("config",), {"model": "different-model"}, "model does not match"),
        (("config",), {"transport_policy_id": "ambient-proxy-and-retry"}, "transport_policy_id is not supported"),
        (("totals",), {"turns": -1}, "counters must be non-negative"),
        (("totals",), {"successful_turns": 3}, "must add up to turns"),
        (("totals",), {"total_inference_latency_ms": -1.0}, "durations must be finite"),
        (("totals",), {"total_server_output_tokens": 99}, "totals do not match"),
        (("config", "tool_replay"), {"mode": "profiled", "profile_statistic": "p50"}, "supports only none or recorded"),
        (("config", "tool_replay"), {"mode": "live", "profile_statistic": None}, "supports only none or recorded"),
    ],
)
def test_public_submission_rejects_tampered_summary(
    tmp_path: Path,
    path: tuple[str, ...],
    updates: JsonObject,
    message: str,
) -> None:
    summary = _private_summary()
    _nested_object(summary, path).update(updates)
    results_dir = _write_bound_results(tmp_path, summary)

    with pytest.raises(ValueError, match=message):
        build_public_submission(results_dir)


def test_public_submission_rejects_workload_changed_after_binding(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    trace_path = next((tmp_path / "workload" / "traces").glob("*.jsonl"))
    trace_path.write_bytes(trace_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="changed after"):
        build_public_submission(results_dir)


@pytest.mark.parametrize("accelerator_count", (0, 2))
def test_attached_submission_keeps_unattested_hardware_private(tmp_path: Path, accelerator_count: int) -> None:
    base_hardware = _hardware()
    hardware = replace_fields(
        base_hardware,
        accelerators=base_hardware.accelerators * accelerator_count,
        warnings=("no_supported_accelerator",) if accelerator_count == 0 else (),
    )
    results_dir = _write_bound_results(tmp_path, _private_summary(), hardware)

    bundle = build_submission_bundle(results_dir)

    assert bundle.aggregate.hardware is None
    assert len(bundle.audit.hardware.accelerators) == accelerator_count
    Draft202012Validator(normalize_json_object(orjson.loads(PRIVATE_AUDIT_SCHEMA.read_bytes()))).validate(
        bundle.audit.to_json()
    )


def test_bundle_rejects_publishing_attached_client_hardware(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    output_dir = tmp_path / "upload-preview"
    write_submission_bundle(output_dir, build_submission_bundle(results_dir))
    _rewrite_bundle(output_dir, (), {"hardware": public_hardware_profile(_hardware()).to_json()})

    with pytest.raises(ValueError, match="must not publish client hardware"):
        validate_submission_bundle(output_dir)


@pytest.mark.parametrize(
    ("reorder", "message"),
    (
        pytest.param(lambda lines: [lines[0], lines[0], *lines[2:]], "duplicate turn IDs", id="duplicated"),
        pytest.param(lambda lines: list(reversed(lines)), "missing, reordered", id="reordered"),
    ),
)
def test_public_submission_rejects_rewritten_turn_order(
    tmp_path: Path, reorder: Callable[[list[bytes]], list[bytes]], message: str
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    turns_path = results_dir / "turns.jsonl"
    turns_path.write_bytes(b"\n".join(reorder(turns_path.read_bytes().splitlines())) + b"\n")

    with pytest.raises(ValueError, match=message):
        build_public_submission(results_dir)


@pytest.mark.parametrize(
    "context",
    [
        {"requested_tokens": 32768, "observed_tokens": 32768, "full_benchmark_tokens": 65536, "reduced": True},
        {"requested_tokens": 65536, "observed_tokens": None, "full_benchmark_tokens": 65536, "reduced": True},
        {"requested_tokens": 65536, "observed_tokens": 8192, "full_benchmark_tokens": 65536, "reduced": True},
    ],
    ids=("reduced-launch", "unobserved-attached", "observed-below-benchmark"),
)
def test_public_submission_accepts_reduced_or_unproven_context(tmp_path: Path, context: JsonObject) -> None:
    summary = _private_summary()
    _nested_object(summary, ("config",))["context"] = context
    results_dir = _write_bound_results(tmp_path, summary)

    submission = build_public_submission(results_dir)

    assert submission.run.policy.context_reduced
    assert submission.run.policy.context_requested_tokens == context["requested_tokens"]
    assert submission.run.policy.context_observed_tokens == context["observed_tokens"]


def test_public_submission_rejects_a_forged_clean_context_flag(tmp_path: Path) -> None:
    summary = _private_summary()
    _nested_object(summary, ("config", "context"))["observed_tokens"] = 32768
    results_dir = _write_bound_results(tmp_path, summary)

    with pytest.raises(ValueError, match="context.reduced must match"):
        build_public_submission(results_dir)


def test_summary_only_context_edit_cannot_upgrade_an_attached_reduced_run(tmp_path: Path) -> None:
    """Mirror red-team attack A: the bound observation catches a two-scalar summary edit."""
    summary = _private_summary()
    _nested_object(summary, ("config",))["context"] = {
        "requested_tokens": 65536,
        "observed_tokens": 8192,
        "full_benchmark_tokens": 65536,
        "reduced": True,
    }
    results_dir = _write_bound_results(tmp_path, summary)

    summary_path = results_dir / "summary.json"
    edited = normalize_json_object(orjson.loads(summary_path.read_bytes()))
    context = _nested_object(edited, ("config", "context"))
    context["observed_tokens"] = 65536
    context["reduced"] = False
    summary_path.write_bytes(orjson.dumps(edited))

    with pytest.raises(ValueError, match="observed context does not match"):
        build_public_submission(results_dir)
    with pytest.raises(ValueError, match="observed context does not match"):
        build_submission_bundle(results_dir)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda measurement: _nested_object(measurement, ("benchmark",)).pop("context_tokens"), "predates"),
        (lambda measurement: measurement.pop("observed_context_tokens"), "observed_context_tokens is missing"),
        (lambda measurement: measurement.update({"version": 1}), "version is not supported"),
    ],
    ids=("deleted-context", "deleted-observation", "downgraded-version"),
)
def test_absent_binding_context_cannot_upgrade_to_the_full_benchmark(
    tmp_path: Path,
    mutate: Callable[[JsonObject], object],
    message: str,
) -> None:
    """Mirror red-team attack B: absence in measurement.json fails closed, never defaults."""
    results_dir = _write_bound_results(tmp_path, _private_summary())
    measurement_path = results_dir / "measurement.json"
    measurement = normalize_json_object(orjson.loads(measurement_path.read_bytes()))
    mutate(measurement)
    measurement_path.write_bytes(orjson.dumps(measurement))

    with pytest.raises(ValueError, match=message):
        build_public_submission(results_dir)


def test_reduced_context_changes_the_benchmark_identity_and_absence_fails_closed() -> None:
    base = SubmissionContext(
        suite_id="aa-agentic-gpu-core",
        suite_epoch="2026-q4",
        suite_digest=f"sha256:{'0' * 64}",
        model_semantics_id="gpt-oss-20b-mxfp4",
        model_artifact_digest=f"sha256:{'1' * 64}",
        runtime_id="llama-cpp-b1234",
    )
    reduced = SubmissionContext(
        suite_id=base.suite_id,
        suite_epoch=base.suite_epoch,
        suite_digest=base.suite_digest,
        model_semantics_id=base.model_semantics_id,
        model_artifact_digest=base.model_artifact_digest,
        runtime_id=base.runtime_id,
        context_tokens=32768,
    )
    legacy = dict(base.to_json())
    legacy.pop("context_tokens")

    assert base.context_tokens == 65536
    assert base.to_json() != reduced.to_json()
    assert SubmissionContext.from_json(base.to_json()) == base
    # A missing context never defaults to full: deleting the field must not upgrade a run.
    with pytest.raises(ValueError, match="predates"):
        SubmissionContext.from_json(legacy)
    with pytest.raises(ValueError, match="context_tokens"):
        SubmissionContext.from_json({**base.to_json(), "context_tokens": 262144})


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "function_call"])
def test_submission_keeps_nonstandard_finish_reason(tmp_path: Path, finish_reason: str) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    def relabel_finish_reason(turns: list[JsonObject]) -> None:
        turns[0]["finish_reason"] = finish_reason

    _rewrite_turns(results_dir, relabel_finish_reason)

    bundle = build_submission_bundle(results_dir)
    schema = normalize_json_object(orjson.loads(SANITIZED_EVIDENCE_SCHEMA.read_bytes()))
    Draft202012Validator(schema).validate(bundle.evidence.to_json())

    assert bundle.evidence.turns[0].finish_reason == finish_reason


def test_submission_keeps_action_count_mismatch(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    def add_unrecorded_tool_call(turns: list[JsonObject]) -> None:
        turns[1]["finish_reason"] = "tool_calls"
        turns[1]["response_tool_calls"] = 1

    _rewrite_turns(results_dir, add_unrecorded_tool_call)

    bundle = build_submission_bundle(results_dir)

    assert bundle.evidence.turns[1].response_action_count == 1
    assert bundle.evidence.turns[1].recorded_action_count == 0


@pytest.mark.parametrize(
    ("section", "field", "value", "reason"),
    [
        (
            "normalization",
            "warning",
            {"observed_output_tokens": 5, "target_output_tokens": 50, "observed_to_target_ratio": 0.1},
            "short or unmeasurable output (short_output_warning)",
        ),
        ("timing", "time_to_first_byte_ms", 5000.0, "time to first byte must not exceed end-to-end latency"),
        ("timing", "time_to_first_token_ms", 5000.0, "time to first token must not exceed end-to-end latency"),
        ("timing", "generation_ms", 5000.0, "generation time must not exceed end-to-end latency"),
    ],
    ids=("short-output-warning", "late-first-byte", "late-first-token", "long-generation"),
)
def test_public_submission_names_the_turn_it_rejects(
    tmp_path: Path,
    section: str,
    field: str,
    value: JsonValue,
    reason: str,
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    turns_path = results_dir / "turns.jsonl"
    turns = [normalize_json_object(orjson.loads(line)) for line in turns_path.read_bytes().splitlines()]
    _nested_object(turns[0], (section,))[field] = value
    turns_path.write_bytes(b"".join(orjson.dumps(turn, option=orjson.OPT_APPEND_NEWLINE) for turn in turns))
    turn_id = turns[0]["turn_id"]

    with pytest.raises(ModelValidationError) as rejection:
        build_public_submission(results_dir)

    assert error_text(rejection.value) == f"turn {turn_id} cannot join a public submission: {reason}"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("aborted", True, "successful, not aborted, and error-free"),
        ("error", "private failure", "successful, not aborted, and error-free"),
        ("finish_reason", None, "must contain a finish reason"),
    ],
)
def test_public_submission_rejects_a_turn_it_cannot_publish(
    tmp_path: Path, field: str, value: bool | str | None, message: str
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    def rewrite_first_turn(turns: list[JsonObject]) -> None:
        turns[0][field] = value

    _rewrite_turns(results_dir, rewrite_first_turn)

    with pytest.raises(ValueError, match=message):
        build_public_submission(results_dir)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("timing", "e2e_latency_ms"),
        ("timing", "time_to_first_byte_ms"),
        ("timing", "time_to_first_token_ms"),
        ("timing", "generation_ms"),
        ("normalization", "normalized_generation_ms"),
        ("normalization", "normalized_e2e_latency_ms"),
        ("tokens", "server_prompt_tokens"),
        ("tokens", "server_output_tokens"),
        ("tokens", "local_output_tokens"),
    ],
)
def test_public_submission_requires_complete_primary_evidence(tmp_path: Path, section: str, field: str) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    def drop_primary_evidence(turns: list[JsonObject]) -> None:
        _nested_object(turns[0], (section,))[field] = None

    _rewrite_turns(results_dir, drop_primary_evidence)

    with pytest.raises(ValueError):
        build_public_submission(results_dir)


def test_submission_excludes_optional_source_token_baselines(tmp_path: Path) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    def drop_source_baselines(turns: list[JsonObject]) -> None:
        for turn in turns:
            tokens = _nested_object(turn, ("tokens",))
            tokens["recorded_prompt_tokens"] = None
            tokens["recorded_completion_tokens"] = None
            tokens["target_output_tokens"] = None

    _rewrite_turns(results_dir, drop_source_baselines)

    bundle = build_submission_bundle(results_dir)
    totals = bundle.aggregate.run.totals.to_json()
    first_tokens = bundle.evidence.turns[0].tokens.to_json()
    public_schema = normalize_json_object(orjson.loads(PUBLIC_SUBMISSION_SCHEMA.read_bytes()))
    evidence_schema = normalize_json_object(orjson.loads(SANITIZED_EVIDENCE_SCHEMA.read_bytes()))
    Draft202012Validator(public_schema).validate(bundle.aggregate.to_json())
    Draft202012Validator(evidence_schema).validate(bundle.evidence.to_json())

    assert "total_recorded_prompt_tokens" not in totals
    assert "total_recorded_completion_tokens" not in totals
    assert "recorded_prompt_tokens" not in first_tokens
    assert "recorded_completion_tokens" not in first_tokens
    assert first_tokens["target_output_tokens"] == TURN_OUTPUT_TOKENS


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exception_info", "private tool failure"),
        ("returncode_matches_recorded", False),
    ],
)
def test_public_submission_rejects_hidden_tool_failure(tmp_path: Path, field: str, value: bool | str) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())

    def hide_tool_failure(turns: list[JsonObject]) -> None:
        tool_calls = turns[0]["tool_calls"]
        assert isinstance(tool_calls, list)
        first_tool = tool_calls[0]
        assert isinstance(first_tool, dict)
        first_tool[field] = value

    _rewrite_turns(results_dir, hide_tool_failure)

    with pytest.raises(ValueError, match="must not contain tool failures"):
        build_public_submission(results_dir)


@pytest.mark.parametrize(
    ("filename", "path", "field", "value", "message"),
    [
        ("failures.json", (), "failures", [{"request_error": "private failure"}], "must not contain failures"),
        ("tools.json", ("overall",), "failures", 1, "must not contain tool failures"),
        ("measurement.json", ("producer",), "client_name", "different-client", "client_name must be agentperf-local"),
    ],
)
def test_public_submission_rejects_tampered_result_artifacts(
    tmp_path: Path,
    filename: str,
    path: tuple[str, ...],
    field: str,
    value: JsonValue,
    message: str,
) -> None:
    results_dir = _write_bound_results(tmp_path, _private_summary())
    artifact_path = results_dir / filename
    artifact = normalize_json_object(orjson.loads(artifact_path.read_bytes()))
    _nested_object(artifact, path)[field] = value
    artifact_path.write_bytes(orjson.dumps(artifact))

    with pytest.raises(ValueError, match=message):
        build_public_submission(results_dir)


def _private_summary() -> JsonObject:
    """Return the private summary for the two bound fixture turns."""
    # The runner adds one turn's calls first, then adds the turn subtotals together.
    tool_delay_ms = sum(TOOL_CALL_DELAYS_MS[0], 0.0)
    return {
        "version": 1,
        "kind": "run_summary",
        "run_id": RUN_ID,
        "manifest": "/Users/alice/private/manifest.json",
        "started_at": 10.0,
        "ended_at": 12.0,
        "wall_duration_ms": 2000.0,
        "measured_duration_ms": 2000.0,
        "observer": {"enabled": False, "duration_ms": 0.0, "ranked_eligible": True},
        "success": True,
        "config": {
            "base_url": "https://secret-endpoint.example/v1",
            "model": "secret-model-alias",
            "client_backend": "rust",
            "transport_policy_id": "direct-sse-no-retry-v1",
            "context": {
                "requested_tokens": 65536,
                "observed_tokens": 65536,
                "full_benchmark_tokens": 65536,
                "reduced": False,
            },
            "request_timeout_seconds": 300.0,
            "output_tokens": {"policy": "recorded", "fallback": 4096, "margin": 0},
            "sampling": {
                "preset": "standard",
                "temperature": 0.7,
                "top_p": 0.8,
                "extra_body": {"top_k": 20, "min_p": 0.0, "future_secret": "secret-future-field"},
            },
            "reasoning_effort": "high",
            "cache_isolation": {
                "enabled": True,
                "mode": "run_namespace_prefix",
                "namespace": "secret-cache-namespace",
                "namespace_digits": 32,
            },
            "tool_replay": {
                "mode": "recorded",
                "delay_scale": 1.0,
                "profile": "/Users/alice/secret-tool-profile.json",
                "profile_statistic": None,
                "live_image_override": "secret-container",
                "live_network_override": "private-network",
            },
        },
        "totals": {
            "turns": 2,
            "successful_turns": 2,
            "failed_turns": 0,
            "short_output_warnings": 0,
            "tool_calls": len(TOOL_CALL_DELAYS_MS[0]),
            "total_inference_latency_ms": 1800.0,
            "total_recorded_tool_delay_ms": tool_delay_ms,
            "total_replayed_tool_delay_ms": tool_delay_ms,
            "total_agentic_replay_ms": 1800.0 + tool_delay_ms,
            "total_normalized_generation_ms": 400.0,
            "total_normalized_inference_latency_ms": 1800.0,
            "total_normalized_agentic_replay_ms": (800.0 + tool_delay_ms) + 1000.0,
            "total_server_prompt_tokens": 1000,
            "total_server_output_tokens": 100,
            "total_local_output_tokens": 100,
            "total_recorded_prompt_tokens": 1000,
            "total_recorded_completion_tokens": 100,
        },
        "latency_distributions_ms": {
            "e2e": {"count": 2, "mean": 900.0, "p50": 900.0, "p95": 990.0},
            "time_to_first_token": {"count": 2, "mean": 80.0, "p50": 80.0, "p95": 98.0},
            "normalized_e2e": {"count": 2, "mean": 900.0, "p50": 900.0, "p95": 990.0},
        },
        "short_output_warning_turn_ids": [],
        "failed_turn_ids": [],
        "artifacts": {
            "turns": "turns.jsonl",
            "tasks": "tasks.json",
            "tools": "tools.json",
            "failures": "failures.json",
        },
        "future_private_field": "secret-future-field",
    }
