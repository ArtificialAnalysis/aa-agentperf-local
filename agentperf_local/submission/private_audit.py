"""Build and validate the AA-private audit file that travels with a submission.

Public surface: PrivateAudit, build_private_audit, and the file, profile,
and schema constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator

from agentperf_local.common.identity import validate_digest, validate_run_id
from agentperf_local.common.json_fields import optional_object, require_exact_keys, required_object, required_string
from agentperf_local.common.json_types import JsonObject
from agentperf_local.common.models import read_object
from agentperf_local.deployment.managed import DEPLOYMENT_RECORD_FILENAME, DeploymentRecord, load_deployment_record
from agentperf_local.deployment.qualification import (
    QUALIFICATION_FILENAME,
    QualificationFile,
    RuntimeQualification,
    load_runtime_qualification,
)
from agentperf_local.provenance.benchmark import MeasurementBinding
from agentperf_local.provenance.hardware import HardwareSnapshot, hardware_snapshot_from_json
from agentperf_local.submission.aggregate import PRIVATE_FIELD_CATEGORIES, PublicSubmission
from agentperf_local.telemetry.power import POWER_SUMMARY_FILENAME, PowerSummary, load_power_summary

PRIVATE_AUDIT_VERSION = 1
PRIVATE_AUDIT_KIND = "agentperf_local_private_audit"
PRIVATE_AUDIT_PROFILE = "aa-private-audit-v1"
PRIVATE_AUDIT_FILENAME = "private-audit.json"
PRIVATE_AUDIT_SCHEMA_ID = "https://artificialanalysis.ai/schemas/agentperf-local/private-audit-v1.schema.json"
PRIVATE_AUDIT_RECIPIENT = "artificial-analysis"
# The service keeps this file for a bounded period after acceptance, then deletes it.
PRIVATE_AUDIT_RETENTION_DAYS = 180
_AUDIT_KEYS = frozenset(
    (
        "version",
        "kind",
        "privacy_profile",
        "run_id",
        "aggregate_payload_digest",
        "hardware",
        "deployment",
        "runtime_qualification",
        "power",
        "retention",
        "excluded_private_categories",
    )
)
_RETENTION_KEYS = frozenset(("recipient", "published", "bounded_days"))


class PrivateAudit(BaseModel, frozen=True):
    """Hold the evidence the service needs for classification but never publishes."""

    run_id: str
    aggregate_payload_digest: str
    hardware: HardwareSnapshot
    deployment: DeploymentRecord | None
    runtime_qualification: RuntimeQualification | None
    power: PowerSummary | None
    version: int = PRIVATE_AUDIT_VERSION

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Require every companion record to name this run."""
        validate_run_id(self.run_id, "run_id")
        validate_digest(self.aggregate_payload_digest, "aggregate_payload_digest")
        if self.runtime_qualification is not None and self.runtime_qualification.run_id != self.run_id:
            raise ValueError("the runtime qualification report belongs to a different run")
        if self.power is not None and self.power.run_id != self.run_id:
            raise ValueError("the power summary belongs to a different run")
        return self

    def to_json(self) -> JsonObject:
        """Return the closed private audit envelope."""
        return {
            "version": self.version,
            "kind": PRIVATE_AUDIT_KIND,
            "privacy_profile": PRIVATE_AUDIT_PROFILE,
            "run_id": self.run_id,
            "aggregate_payload_digest": self.aggregate_payload_digest,
            "hardware": self.hardware.to_json(),
            "deployment": None if self.deployment is None else self.deployment.to_audit_json(),
            "runtime_qualification": (
                None if self.runtime_qualification is None else self.runtime_qualification.to_json()
            ),
            "power": None if self.power is None else self.power.to_json(),
            "retention": {
                "recipient": PRIVATE_AUDIT_RECIPIENT,
                "published": False,
                "bounded_days": PRIVATE_AUDIT_RETENTION_DAYS,
            },
            "excluded_private_categories": list(PRIVATE_FIELD_CATEGORIES),
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> PrivateAudit:
        """Parse one strict private audit envelope."""
        require_exact_keys(data, _AUDIT_KEYS, "private_audit")
        if data.get("version") != PRIVATE_AUDIT_VERSION:
            raise ValueError("private audit version is not supported")
        if data.get("kind") != PRIVATE_AUDIT_KIND or data.get("privacy_profile") != PRIVATE_AUDIT_PROFILE:
            raise ValueError("private audit kind or privacy profile is not supported")
        if data.get("excluded_private_categories") != list(PRIVATE_FIELD_CATEGORIES):
            raise ValueError("private audit excluded categories are not supported")
        retention = required_object(data, "retention", "private_audit")
        require_exact_keys(retention, _RETENTION_KEYS, "private_audit.retention")
        if retention.get("recipient") != PRIVATE_AUDIT_RECIPIENT or retention.get("published") is not False:
            raise ValueError("private audit retention block is not supported")
        if retention.get("bounded_days") != PRIVATE_AUDIT_RETENTION_DAYS:
            raise ValueError("private audit retention period is not supported")
        hardware_data = required_object(data, "hardware", "private_audit")
        hardware = hardware_snapshot_from_json(hardware_data)
        # Round-tripping through the writer closes the snapshot's key set.
        if hardware.to_json() != hardware_data:
            raise ValueError("private audit hardware fields do not match the closed contract")
        deployment_data = optional_object(data, "deployment", "private_audit")
        qualification_data = optional_object(data, "runtime_qualification", "private_audit")
        power_data = optional_object(data, "power", "private_audit")
        return cls(
            run_id=required_string(data, "run_id", "private_audit"),
            aggregate_payload_digest=required_string(data, "aggregate_payload_digest", "private_audit"),
            hardware=hardware,
            deployment=(
                None
                if deployment_data is None
                else DeploymentRecord.from_audit_json(deployment_data, "private_audit.deployment")
            ),
            runtime_qualification=(
                None
                if qualification_data is None
                else read_object(QualificationFile, qualification_data, "private_audit.runtime_qualification").report()
            ),
            power=None if power_data is None else read_object(PowerSummary, power_data, "private_audit.power"),
        )


def _load_deployment(results_dir: Path, binding: MeasurementBinding) -> DeploymentRecord | None:
    """Load the deployment record the binding names, and refuse one it does not."""
    path = results_dir / DEPLOYMENT_RECORD_FILENAME
    if binding.deployment_digest is None:
        if path.exists() or path.is_symlink():
            raise ValueError("deployment.json is present but the measurement binding does not name it")
        return None
    record = load_deployment_record(path)
    if record.record_digest != binding.deployment_digest:
        raise ValueError("deployment.json changed after the measurement binding was captured")
    record.require_benchmark(binding.benchmark)
    return record


def _load_qualification(results_dir: Path, binding: MeasurementBinding) -> RuntimeQualification | None:
    """Load the report a managed run wrote into its results directory, if any."""
    path = results_dir / QUALIFICATION_FILENAME
    if not path.exists():
        return None
    report = load_runtime_qualification(path)
    # The audit invariant checks the run identifier; only the endpoint is checked here.
    if report.endpoint_model_digest != binding.endpoint_model_digest:
        raise ValueError("the runtime qualification report probed a different endpoint model")
    return report


def _load_power(results_dir: Path) -> PowerSummary | None:
    path = results_dir / POWER_SUMMARY_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    return load_power_summary(path)


def build_private_audit(
    results_dir: Path,
    aggregate: PublicSubmission,
    binding: MeasurementBinding,
) -> PrivateAudit:
    """Collect the private companions of one bound run and bind them to its aggregate."""
    if aggregate.run_id != binding.run_id:
        raise ValueError("aggregate and measurement binding name different runs")
    return PrivateAudit(
        run_id=binding.run_id,
        aggregate_payload_digest=aggregate.payload_digest,
        hardware=binding.hardware,
        deployment=_load_deployment(results_dir, binding),
        runtime_qualification=_load_qualification(results_dir, binding),
        power=_load_power(results_dir),
    )
