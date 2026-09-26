"""Build and validate the exact-byte four-file submission bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import orjson

from agentperf_local.common.durable_files import (
    PUBLIC_FILE_PERMISSIONS,
    NewFile,
    commit_new_file_set,
    create_new_directory,
    read_bounded_file,
    validate_public_output_directory,
)
from agentperf_local.common.identity import sha256_bytes, validate_digest, validate_run_id
from agentperf_local.common.json_fields import (
    decode_json_object,
    optional_integer,
    optional_number,
    optional_string,
    require_exact_keys,
    required_boolean,
    required_integer,
    required_number,
    required_object,
    required_string,
)
from agentperf_local.common.json_records import json_field_names, json_record
from agentperf_local.common.json_types import JsonObject, JsonValue, pretty_json_bytes
from agentperf_local.provenance.benchmark import (
    MEASUREMENT_BINDING_FILENAME,
    SourceProvenance,
    SubmissionContext,
    load_measurement_binding,
)
from agentperf_local.provenance.hardware import (
    PUBLIC_HARDWARE_PROFILE_VERSION,
    PublicAcceleratorProfile,
    PublicHardwareProfile,
    public_hardware_profile,
)
from agentperf_local.submission.aggregate import (
    PRIVATE_FIELD_CATEGORIES,
    PUBLIC_PRIVACY_PROFILE,
    PUBLIC_SUBMISSION_VERSION,
    SELF_REPORTED_TRUST_TIER,
    ArtifactDigests,
    PublicDistribution,
    PublicLatencyDistributions,
    PublicRunPolicy,
    PublicRunResult,
    PublicSubmission,
    PublicTotals,
)
from agentperf_local.submission.evidence import (
    SANITIZED_EVIDENCE_PROFILE,
    SanitizedEvidence,
    build_sanitized_evidence,
    recompute_sanitized_metrics,
    validate_sanitized_evidence,
)
from agentperf_local.submission.private_audit import (
    PRIVATE_AUDIT_FILENAME,
    PRIVATE_AUDIT_PROFILE,
    PRIVATE_AUDIT_SCHEMA_ID,
    PrivateAudit,
    build_private_audit,
)

# Version 2 added the run identifier and the private audit artifact.
BUNDLE_MANIFEST_VERSION = 2
BUNDLE_MANIFEST_KIND = "agentperf_local_submission_bundle"
# The bundle is complete on disk; sending it is a separate, explicit step.
BUNDLE_STATUS = "prepared-not-uploaded"
BUNDLE_MANIFEST_FILENAME = "bundle-manifest.json"
AGGREGATE_FILENAME = "aggregate.json"
SANITIZED_EVIDENCE_FILENAME = "sanitized-turn-evidence.json"
JSON_MEDIA_TYPE = "application/json"
PUBLIC_SUBMISSION_SCHEMA_ID = "https://artificialanalysis.ai/schemas/agentperf-local/public-submission-v2.schema.json"
SANITIZED_EVIDENCE_SCHEMA_ID = (
    "https://artificialanalysis.ai/schemas/agentperf-local/sanitized-turn-evidence-v1.schema.json"
)
AGGREGATE_ROLE = "aggregate"
SANITIZED_EVIDENCE_ROLE = "sanitized_turn_evidence"
PRIVATE_AUDIT_ROLE = "private_audit"


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactContract:
    """Name one bundle file's role, filename, schema, and privacy profile."""

    role: str
    filename: str
    schema_id: str
    privacy_profile: str


# The three payload files, in manifest order. The manifest itself is written last.
ARTIFACT_CONTRACTS = (
    ArtifactContract(
        role=AGGREGATE_ROLE,
        filename=AGGREGATE_FILENAME,
        schema_id=PUBLIC_SUBMISSION_SCHEMA_ID,
        privacy_profile=PUBLIC_PRIVACY_PROFILE,
    ),
    ArtifactContract(
        role=SANITIZED_EVIDENCE_ROLE,
        filename=SANITIZED_EVIDENCE_FILENAME,
        schema_id=SANITIZED_EVIDENCE_SCHEMA_ID,
        privacy_profile=SANITIZED_EVIDENCE_PROFILE,
    ),
    ArtifactContract(
        role=PRIVATE_AUDIT_ROLE,
        filename=PRIVATE_AUDIT_FILENAME,
        schema_id=PRIVATE_AUDIT_SCHEMA_ID,
        privacy_profile=PRIVATE_AUDIT_PROFILE,
    ),
)
BUNDLE_FILENAMES = (*(contract.filename for contract in ARTIFACT_CONTRACTS), BUNDLE_MANIFEST_FILENAME)
MAX_BUNDLE_FILE_BYTES = 256 * 1024**2
MAX_BUNDLE_MANIFEST_BYTES = 64 * 1024
_MANIFEST_KEYS = frozenset(
    ("version", "kind", "status", "run_id", "aggregate_payload_digest", "sanitized_rows_digest", "artifacts")
)
_PUBLIC_ENVELOPE_KEYS = frozenset(
    ("version", "kind", "privacy_profile", "payload_digest", "payload", "excluded_private_categories")
)
# These objects are not a writer dataclass's field list: the envelope and payload
# add constants, the producer adds its client name, the hardware profile adds a kind
# and a notice, and the policy nests its fields. Each flat object is checked
# against `json_field_names` of its writer instead.
_PUBLIC_PAYLOAD_KEYS = frozenset(
    ("run_id", "benchmark", "producer", "hardware", "private_evidence_digests", "run", "trust_tier")
)
_PUBLIC_PRODUCER_KEYS = frozenset(("client_name", "client_version", "source_revision", "source_state"))
_PUBLIC_HARDWARE_KEYS = frozenset(
    (
        "version",
        "kind",
        "platform_family",
        "platform_major",
        "architecture",
        "host_memory_gib",
        "accelerator",
        "privacy_notice",
    )
)
_PUBLIC_POLICY_KEYS = frozenset(
    (
        "client_backend",
        "transport_policy_id",
        "context",
        "output_tokens",
        "sampling",
        "reasoning_effort",
        "cache_isolation",
        "tool_replay",
    )
)
_PUBLIC_POLICY_CONTEXT_KEYS = frozenset(("requested_tokens", "observed_tokens", "full_benchmark_tokens", "reduced"))


@dataclass(frozen=True, slots=True, kw_only=True)
class BundleArtifact:
    """Bind one named bundle file to its exact bytes and schema."""

    role: str
    filename: str
    media_type: str
    schema_id: str
    privacy_profile: str
    byte_size: int
    file_digest: str

    def __post_init__(self) -> None:
        """Validate one fixed bundle artifact record."""
        validate_digest(self.file_digest, "file_digest")
        if self.byte_size <= 0:
            raise ValueError("bundle artifact byte size must be positive")
        if self.media_type != JSON_MEDIA_TYPE:
            raise ValueError("bundle artifacts must use the JSON media type")

    def to_json(self) -> JsonObject:
        """Return one exact-byte artifact record."""
        return json_record(self)

    @classmethod
    def from_json(cls, data: JsonObject, source: str) -> BundleArtifact:
        """Parse one strict artifact record."""
        require_exact_keys(data, json_field_names(cls), source)
        return cls(
            role=required_string(data, "role", source),
            filename=required_string(data, "filename", source),
            media_type=required_string(data, "media_type", source),
            schema_id=required_string(data, "schema_id", source),
            privacy_profile=required_string(data, "privacy_profile", source),
            byte_size=required_integer(data, "byte_size", source),
            file_digest=required_string(data, "file_digest", source),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmissionBundleManifest:
    """Bind the aggregate, sanitized evidence, and private audit into one upload unit."""

    run_id: str
    aggregate_payload_digest: str
    sanitized_rows_digest: str
    artifacts: tuple[BundleArtifact, BundleArtifact, BundleArtifact]
    version: int = BUNDLE_MANIFEST_VERSION

    def __post_init__(self) -> None:
        """Validate the closed three-file bundle layout."""
        validate_run_id(self.run_id, "run_id")
        validate_digest(self.aggregate_payload_digest, "aggregate_payload_digest")
        validate_digest(self.sanitized_rows_digest, "sanitized_rows_digest")
        if self.version != BUNDLE_MANIFEST_VERSION:
            raise ValueError("submission bundle manifest version is not supported")
        for artifact, contract in zip(self.artifacts, ARTIFACT_CONTRACTS, strict=True):
            actual = (artifact.role, artifact.filename, artifact.schema_id, artifact.privacy_profile)
            expected = (contract.role, contract.filename, contract.schema_id, contract.privacy_profile)
            if actual != expected:
                raise ValueError(f"submission bundle {contract.role} artifact contract does not match")

    def to_json(self) -> JsonObject:
        """Return the exact-byte bundle manifest."""
        artifact_values: list[JsonValue] = [artifact.to_json() for artifact in self.artifacts]
        return {
            "version": self.version,
            "kind": BUNDLE_MANIFEST_KIND,
            "status": BUNDLE_STATUS,
            "run_id": self.run_id,
            "aggregate_payload_digest": self.aggregate_payload_digest,
            "sanitized_rows_digest": self.sanitized_rows_digest,
            "artifacts": artifact_values,
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> SubmissionBundleManifest:
        """Parse one strict bundle manifest."""
        require_exact_keys(data, _MANIFEST_KEYS, "bundle_manifest")
        if required_string(data, "kind", "bundle_manifest") != BUNDLE_MANIFEST_KIND:
            raise ValueError("bundle manifest kind is not supported")
        if required_string(data, "status", "bundle_manifest") != BUNDLE_STATUS:
            raise ValueError("bundle manifest status is not supported")
        values = data.get("artifacts")
        if not isinstance(values, list) or len(values) != len(ARTIFACT_CONTRACTS):
            raise ValueError(f"bundle manifest must contain exactly {len(ARTIFACT_CONTRACTS)} artifacts")
        parsed: list[BundleArtifact] = []
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"bundle_manifest.artifacts[{index}] must be an object")
            parsed.append(BundleArtifact.from_json(value, f"bundle_manifest.artifacts[{index}]"))
        return cls(
            version=required_integer(data, "version", "bundle_manifest"),
            run_id=required_string(data, "run_id", "bundle_manifest"),
            aggregate_payload_digest=required_string(data, "aggregate_payload_digest", "bundle_manifest"),
            sanitized_rows_digest=required_string(data, "sanitized_rows_digest", "bundle_manifest"),
            artifacts=(parsed[0], parsed[1], parsed[2]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedSubmissionBundle:
    """Hold validated objects and the exact bytes ready for local review."""

    aggregate: PublicSubmission
    evidence: SanitizedEvidence
    audit: PrivateAudit
    manifest: SubmissionBundleManifest
    aggregate_bytes: bytes
    evidence_bytes: bytes
    audit_bytes: bytes
    manifest_bytes: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class WrittenSubmissionBundle:
    """Describe a complete local bundle directory."""

    path: Path
    manifest_digest: str
    total_byte_size: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedSubmissionBundle:
    """Describe a bundle whose exact bytes and cross-file bindings match."""

    path: Path
    manifest: SubmissionBundleManifest
    evidence: SanitizedEvidence
    audit: PrivateAudit
    producer: SourceProvenance
    aggregate_payload_digest: str
    # The exact bytes the manifest digests, in manifest order, so a sender need not reread them.
    artifact_bytes: tuple[bytes, bytes, bytes]


def _artifact(contract: ArtifactContract, encoded: bytes) -> BundleArtifact:
    return BundleArtifact(
        role=contract.role,
        filename=contract.filename,
        media_type=JSON_MEDIA_TYPE,
        schema_id=contract.schema_id,
        privacy_profile=contract.privacy_profile,
        byte_size=len(encoded),
        file_digest=sha256_bytes(encoded),
    )


def build_submission_bundle(results_dir: Path) -> PreparedSubmissionBundle:
    """Build a validated aggregate, evidence file, private audit, and exact-byte manifest.

    The qualification report is read from the results directory when the run wrote one there.
    """
    aggregate, evidence = build_sanitized_evidence(results_dir)
    binding = load_measurement_binding(results_dir / MEASUREMENT_BINDING_FILENAME)
    audit = build_private_audit(results_dir, aggregate, binding)
    aggregate_encoded = pretty_json_bytes(aggregate.to_json())
    evidence_encoded = pretty_json_bytes(evidence.to_json())
    audit_encoded = pretty_json_bytes(audit.to_json())
    encoded_files = (aggregate_encoded, evidence_encoded, audit_encoded)
    artifacts = tuple(
        _artifact(contract, encoded) for contract, encoded in zip(ARTIFACT_CONTRACTS, encoded_files, strict=True)
    )
    manifest = SubmissionBundleManifest(
        run_id=aggregate.run_id,
        aggregate_payload_digest=aggregate.payload_digest,
        sanitized_rows_digest=evidence.rows_digest,
        artifacts=(artifacts[0], artifacts[1], artifacts[2]),
    )
    manifest_encoded = pretty_json_bytes(manifest.to_json())
    return PreparedSubmissionBundle(
        aggregate=aggregate,
        evidence=evidence,
        audit=audit,
        manifest=manifest,
        aggregate_bytes=aggregate_encoded,
        evidence_bytes=evidence_encoded,
        audit_bytes=audit_encoded,
        manifest_bytes=manifest_encoded,
    )


def validate_bundle_output_path(results_dir: Path, output_dir: Path) -> None:
    """Keep bundle output separate from private results and existing paths."""
    validate_public_output_directory(results_dir, output_dir, "submission bundle output")


def write_submission_bundle(output_dir: Path, bundle: PreparedSubmissionBundle) -> WrittenSubmissionBundle:
    """Create and sync a no-clobber bundle with its manifest last."""
    create_new_directory(output_dir)
    commit_new_file_set(
        (
            NewFile(path=output_dir / AGGREGATE_FILENAME, data=bundle.aggregate_bytes),
            NewFile(path=output_dir / SANITIZED_EVIDENCE_FILENAME, data=bundle.evidence_bytes),
            NewFile(path=output_dir / PRIVATE_AUDIT_FILENAME, data=bundle.audit_bytes),
        ),
        NewFile(path=output_dir / BUNDLE_MANIFEST_FILENAME, data=bundle.manifest_bytes),
        permissions=PUBLIC_FILE_PERMISSIONS,
    )
    return WrittenSubmissionBundle(
        path=output_dir,
        manifest_digest=sha256_bytes(bundle.manifest_bytes),
        total_byte_size=(
            len(bundle.aggregate_bytes)
            + len(bundle.evidence_bytes)
            + len(bundle.audit_bytes)
            + len(bundle.manifest_bytes)
        ),
    )


def _validate_public_hardware(payload: JsonObject) -> PublicHardwareProfile | None:
    """Check the hardware profile against the public hardware writer.

    Reconstructing the profile classes validates every label in their __post_init__.
    """
    source = "aggregate.payload.hardware"
    raw_hardware = payload.get("hardware")
    if raw_hardware is None:
        return None
    if not isinstance(raw_hardware, dict):
        raise ValueError(f"{source} must be an object or null")
    data = raw_hardware
    require_exact_keys(data, _PUBLIC_HARDWARE_KEYS, source)
    accelerator_source = f"{source}.accelerator"
    accelerator_data = required_object(data, "accelerator", source)
    require_exact_keys(accelerator_data, json_field_names(PublicAcceleratorProfile), accelerator_source)
    if required_integer(data, "version", source) != PUBLIC_HARDWARE_PROFILE_VERSION:
        raise ValueError("aggregate hardware profile version is not supported")
    profile = PublicHardwareProfile(
        version=PUBLIC_HARDWARE_PROFILE_VERSION,
        platform_family=required_string(data, "platform_family", source),
        platform_major=optional_string(data, "platform_major", source),
        architecture=required_string(data, "architecture", source),
        host_memory_gib=optional_integer(data, "host_memory_gib", source),
        accelerator=PublicAcceleratorProfile(
            vendor=required_string(accelerator_data, "vendor", accelerator_source),
            product=required_string(accelerator_data, "product", accelerator_source),
            memory_gib=optional_integer(accelerator_data, "memory_gib", accelerator_source),
            core_count=optional_integer(accelerator_data, "core_count", accelerator_source),
            driver_branch=optional_string(accelerator_data, "driver_branch", accelerator_source),
            api=optional_string(accelerator_data, "api", accelerator_source),
        ),
    )
    if profile.to_json() != data:
        raise ValueError(f"{source} fields do not match the closed contract")
    return profile


@dataclass(frozen=True, slots=True, kw_only=True)
class _PublicIdentity:
    """Hold the parsed identity blocks of one aggregate payload."""

    run_id: str
    context: SubmissionContext
    producer: SourceProvenance
    hardware: PublicHardwareProfile | None


def _validate_public_payload_objects(payload: JsonObject) -> _PublicIdentity:
    """Check every payload object against the public submission writers."""
    run_id = required_string(payload, "run_id", "aggregate.payload")
    validate_run_id(run_id, "aggregate.payload.run_id")
    benchmark = required_object(payload, "benchmark", "aggregate.payload")
    require_exact_keys(benchmark, json_field_names(SubmissionContext), "aggregate.payload.benchmark")
    context = SubmissionContext.from_json(benchmark, "aggregate.payload.benchmark")
    producer = required_object(payload, "producer", "aggregate.payload")
    require_exact_keys(producer, _PUBLIC_PRODUCER_KEYS, "aggregate.payload.producer")
    provenance = SourceProvenance.from_json(producer)
    hardware = _validate_public_hardware(payload)
    digest_source = "aggregate.payload.private_evidence_digests"
    digests = required_object(payload, "private_evidence_digests", "aggregate.payload")
    require_exact_keys(digests, json_field_names(ArtifactDigests), digest_source)
    for key in sorted(json_field_names(ArtifactDigests)):
        validate_digest(required_string(digests, key, digest_source), f"{digest_source}.{key}")
    return _PublicIdentity(run_id=run_id, context=context, producer=provenance, hardware=hardware)


def _validate_public_policy(run: JsonObject) -> PublicRunPolicy:
    """Check the run policy against the public policy writer."""
    source = "aggregate.payload.run.policy"
    data = required_object(run, "policy", "aggregate.payload.run")
    require_exact_keys(data, _PUBLIC_POLICY_KEYS, source)
    output_tokens = required_object(data, "output_tokens", source)
    sampling = required_object(data, "sampling", source)
    cache = required_object(data, "cache_isolation", source)
    tools = required_object(data, "tool_replay", source)
    context = required_object(data, "context", source)
    require_exact_keys(context, _PUBLIC_POLICY_CONTEXT_KEYS, f"{source}.context")
    policy = PublicRunPolicy(
        client_backend=required_string(data, "client_backend", source),
        transport_policy_id=required_string(data, "transport_policy_id", source),
        context_requested_tokens=required_integer(context, "requested_tokens", f"{source}.context"),
        context_observed_tokens=optional_integer(context, "observed_tokens", f"{source}.context"),
        context_full_benchmark_tokens=required_integer(context, "full_benchmark_tokens", f"{source}.context"),
        context_reduced=required_boolean(context, "reduced", f"{source}.context"),
        output_token_policy=required_string(output_tokens, "policy", f"{source}.output_tokens"),
        max_output_tokens=required_integer(output_tokens, "fallback", f"{source}.output_tokens"),
        output_token_margin=required_integer(output_tokens, "margin", f"{source}.output_tokens"),
        sampling_preset=required_string(sampling, "preset", f"{source}.sampling"),
        temperature=optional_number(sampling, "temperature", f"{source}.sampling"),
        top_p=optional_number(sampling, "top_p", f"{source}.sampling"),
        top_k=optional_integer(sampling, "top_k", f"{source}.sampling"),
        min_p=optional_number(sampling, "min_p", f"{source}.sampling"),
        reasoning_effort=optional_string(data, "reasoning_effort", source),
        cache_isolation_enabled=required_boolean(cache, "enabled", f"{source}.cache_isolation"),
        cache_isolation_mode=required_string(cache, "mode", f"{source}.cache_isolation"),
        cache_namespace_digits=required_integer(cache, "namespace_digits", f"{source}.cache_isolation"),
        tool_replay_mode=required_string(tools, "mode", f"{source}.tool_replay"),
        tool_delay_scale=required_number(tools, "delay_scale", f"{source}.tool_replay"),
        tool_profile_statistic=optional_string(tools, "profile_statistic", f"{source}.tool_replay"),
    )
    # Round-tripping through the writer closes the nested objects and catches value drift.
    if policy.to_json() != data:
        raise ValueError(f"{source} fields do not match the closed contract")
    return policy


def validate_submission_bundle(bundle_dir: Path) -> ValidatedSubmissionBundle:
    """Verify an exact local bundle and its cross-file semantic bindings."""
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise ValueError("submission bundle path must be a regular directory")
    if frozenset(path.name for path in bundle_dir.iterdir()) != frozenset(BUNDLE_FILENAMES):
        raise ValueError("submission bundle directory must contain exactly the four contract files")
    manifest_path = bundle_dir / BUNDLE_MANIFEST_FILENAME
    manifest_encoded = read_bounded_file(manifest_path, MAX_BUNDLE_MANIFEST_BYTES, label="bundle manifest")
    manifest_data = decode_json_object(manifest_encoded, "invalid bundle manifest JSON")
    manifest = SubmissionBundleManifest.from_json(manifest_data)
    artifact_bytes: list[bytes] = []
    for artifact in manifest.artifacts:
        encoded = read_bounded_file(bundle_dir / artifact.filename, MAX_BUNDLE_FILE_BYTES, label="bundle artifact")
        if len(encoded) != artifact.byte_size or sha256_bytes(encoded) != artifact.file_digest:
            raise ValueError(f"bundle artifact bytes do not match the manifest: {artifact.filename}")
        artifact_bytes.append(encoded)

    aggregate_data = decode_json_object(artifact_bytes[0], "invalid aggregate JSON")
    require_exact_keys(aggregate_data, _PUBLIC_ENVELOPE_KEYS, "aggregate")
    if required_integer(aggregate_data, "version", "aggregate") != PUBLIC_SUBMISSION_VERSION:
        raise ValueError("aggregate version is not supported")
    if required_string(aggregate_data, "kind", "aggregate") != "agentperf_local_public_submission":
        raise ValueError("aggregate kind is not supported")
    if required_string(aggregate_data, "privacy_profile", "aggregate") != PUBLIC_PRIVACY_PROFILE:
        raise ValueError("aggregate privacy profile is not supported")
    if aggregate_data.get("excluded_private_categories") != list(PRIVATE_FIELD_CATEGORIES):
        raise ValueError("aggregate excluded private categories are not supported")
    payload = aggregate_data.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("aggregate.payload must be an object")
    require_exact_keys(payload, _PUBLIC_PAYLOAD_KEYS, "aggregate.payload")
    if required_string(payload, "trust_tier", "aggregate.payload") != SELF_REPORTED_TRUST_TIER:
        raise ValueError("aggregate trust tier is not supported")
    identity = _validate_public_payload_objects(payload)
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("aggregate.payload.run must be an object")
    require_exact_keys(run, json_field_names(PublicRunResult), "aggregate.payload.run")
    if run.get("success") is not True:
        raise ValueError("aggregate run must be successful")
    raw_policy = run.get("policy")
    raw_context = raw_policy.get("context") if isinstance(raw_policy, dict) else None
    if not isinstance(raw_context, dict):
        raise ValueError("aggregate run policy context must be an object")
    # The benchmark identity and the run policy state the served context independently;
    # a bundle whose two statements disagree is self-contradictory.
    benchmark_context = required_object(payload, "benchmark", "aggregate.payload").get("context_tokens")
    if raw_context.get("requested_tokens") != benchmark_context:
        raise ValueError("aggregate benchmark context does not match the run policy context")
    totals_data = run.get("totals")
    latency_data = run.get("latency_distributions_ms")
    if not isinstance(totals_data, dict) or not isinstance(latency_data, dict):
        raise ValueError("aggregate run totals and distributions must be objects")
    policy = _validate_public_policy(run)
    require_exact_keys(totals_data, json_field_names(PublicTotals), "aggregate.payload.run.totals")
    require_exact_keys(
        latency_data, json_field_names(PublicLatencyDistributions), "aggregate.payload.run.latency_distributions_ms"
    )
    for name in json_field_names(PublicLatencyDistributions):
        distribution = latency_data.get(name)
        if not isinstance(distribution, dict):
            raise ValueError(f"aggregate latency distribution {name} must be an object")
        require_exact_keys(distribution, json_field_names(PublicDistribution), f"aggregate.latency.{name}")
    # The run result carries the public invariants, including the observer rule, so the
    # packaging gate and this validator apply exactly one rule set.
    aggregate_run = PublicRunResult(
        success=True,
        wall_duration_ms=required_number(run, "wall_duration_ms", "aggregate.payload.run"),
        observer_duration_ms=required_number(run, "observer_duration_ms", "aggregate.payload.run"),
        totals=PublicTotals.from_json(totals_data),
        latency_distributions_ms=PublicLatencyDistributions.from_json(latency_data),
        policy=policy,
    )
    payload_digest = required_string(aggregate_data, "payload_digest", "aggregate")
    validate_digest(payload_digest, "aggregate.payload_digest")
    if sha256_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)) != payload_digest:
        raise ValueError("aggregate payload digest does not match its payload")

    evidence_data = decode_json_object(artifact_bytes[1], "invalid sanitized evidence JSON")
    evidence = validate_sanitized_evidence(evidence_data)
    if manifest.aggregate_payload_digest != payload_digest or evidence.aggregate_payload_digest != payload_digest:
        raise ValueError("bundle aggregate payload bindings do not match")
    if manifest.sanitized_rows_digest != evidence.rows_digest:
        raise ValueError("bundle sanitized rows bindings do not match")

    audit = PrivateAudit.from_json(decode_json_object(artifact_bytes[2], "invalid private audit JSON"))
    if manifest.run_id != identity.run_id or audit.run_id != identity.run_id:
        raise ValueError("bundle run identifiers do not match")
    if audit.aggregate_payload_digest != payload_digest:
        raise ValueError("private audit is bound to a different aggregate")
    if audit.deployment is None:
        if identity.hardware is not None:
            raise ValueError("an attached submission must not publish client hardware as server hardware")
    else:
        if identity.hardware is None or public_hardware_profile(audit.hardware) != identity.hardware:
            raise ValueError("private audit hardware does not reproduce the public hardware profile")
        audit.deployment.require_benchmark(identity.context)
    recomputed = recompute_sanitized_metrics(evidence)
    if recomputed.totals != aggregate_run.totals:
        raise ValueError("sanitized rows do not reproduce aggregate totals")
    if recomputed.latency_distributions_ms != aggregate_run.latency_distributions_ms:
        raise ValueError("sanitized rows do not reproduce aggregate latency distributions")
    return ValidatedSubmissionBundle(
        path=bundle_dir,
        manifest=manifest,
        evidence=evidence,
        audit=audit,
        producer=identity.producer,
        aggregate_payload_digest=payload_digest,
        artifact_bytes=(artifact_bytes[0], artifact_bytes[1], artifact_bytes[2]),
    )
