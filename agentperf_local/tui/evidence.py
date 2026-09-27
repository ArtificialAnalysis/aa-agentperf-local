"""Derive closed evidence labels for the local TUI."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class SelectionKind(StrEnum):
    """Name how endpoint model intent was selected."""

    BUNDLED_CATALOG_CANDIDATE = "bundled-catalog-candidate"
    EXTERNAL_CATALOG_ENTRY = "external-catalog-entry"
    CUSTOM_ENDPOINT = "custom-endpoint"


class EndpointScope(StrEnum):
    """Classify the endpoint URL without making a locality claim."""

    LOOPBACK_NAME = "loopback-name"
    NON_LOOPBACK_NAME = "non-loopback-name"


class ArtifactEvidence(StrEnum):
    """Name the strongest model-artifact fact available to the pilot."""

    BUNDLED_UNSIGNED_CANDIDATE_METADATA = "bundled-unsigned-candidate-metadata"
    EXTERNAL_UNSIGNED_CATALOG_METADATA = "external-unsigned-catalog-metadata"
    SELF_REPORTED_ENDPOINT_NAME = "self-reported-endpoint-name"
    MANAGED_PINNED_ARTIFACT_VERIFIED = "managed-pinned-artifact-verified"


class ExecutionEvidence(StrEnum):
    """Name what the pilot knows about serving execution."""

    ATTACHED_ENDPOINT_UNVERIFIED = "attached-endpoint-unverified"
    REMOTE_BACKEND_OPAQUE = "remote-backend-opaque"
    MANAGED_GPU_STARTUP_VERIFIED = "managed-gpu-startup-verified"


class ResultPartition(StrEnum):
    """Name the strongest comparison partition the pilot may display."""

    ATTACHED_ENDPOINT_EXPLORER = "attached-endpoint-explorer"
    SERVICE_LATENCY_ONLY = "service-latency-only"
    MANAGED_LOCAL_EXPLORER = "managed-local-explorer"


class EligibilityReason(StrEnum):
    """Name why this implementation cannot produce a ranked result."""

    USER_CHOSEN_ENDPOINT = "user-chosen-endpoint"
    OBSERVER_ENABLED = "observer-enabled"
    SINGLE_REPLAY = "single-replay"
    UNSIGNED_CATALOG = "unsigned-catalog"
    REDUCED_CONTEXT = "reduced-context"


class EvidenceSummary(BaseModel, frozen=True):
    """Store evidence derived from closed app state."""

    selection: SelectionKind
    endpoint_scope: EndpointScope
    artifact: ArtifactEvidence
    execution: ExecutionEvidence
    partition: ResultPartition
    ineligibility_reasons: tuple[EligibilityReason, ...]

    @classmethod
    def derive(
        cls,
        selection: SelectionKind,
        endpoint_scope: EndpointScope,
        *,
        managed_deployment: bool = False,
        reduced_context: bool = False,
    ) -> EvidenceSummary:
        """Derive evidence without accepting trust labels from a user."""
        if managed_deployment:
            if selection is SelectionKind.CUSTOM_ENDPOINT or endpoint_scope is not EndpointScope.LOOPBACK_NAME:
                raise ValueError("managed evidence requires a catalog model on a loopback endpoint")
            artifact = ArtifactEvidence.MANAGED_PINNED_ARTIFACT_VERIFIED
            execution = ExecutionEvidence.MANAGED_GPU_STARTUP_VERIFIED
            partition = ResultPartition.MANAGED_LOCAL_EXPLORER
        elif selection is SelectionKind.BUNDLED_CATALOG_CANDIDATE:
            artifact = ArtifactEvidence.BUNDLED_UNSIGNED_CANDIDATE_METADATA
        elif selection is SelectionKind.EXTERNAL_CATALOG_ENTRY:
            artifact = ArtifactEvidence.EXTERNAL_UNSIGNED_CATALOG_METADATA
        else:
            artifact = ArtifactEvidence.SELF_REPORTED_ENDPOINT_NAME
        if not managed_deployment:
            execution = (
                ExecutionEvidence.ATTACHED_ENDPOINT_UNVERIFIED
                if endpoint_scope is EndpointScope.LOOPBACK_NAME
                else ExecutionEvidence.REMOTE_BACKEND_OPAQUE
            )
            partition = (
                ResultPartition.ATTACHED_ENDPOINT_EXPLORER
                if endpoint_scope is EndpointScope.LOOPBACK_NAME
                else ResultPartition.SERVICE_LATENCY_ONLY
            )
        reasons = [EligibilityReason.OBSERVER_ENABLED, EligibilityReason.SINGLE_REPLAY]
        if not managed_deployment:
            reasons.insert(0, EligibilityReason.USER_CHOSEN_ENDPOINT)
        if selection is not SelectionKind.CUSTOM_ENDPOINT:
            reasons.append(EligibilityReason.UNSIGNED_CATALOG)
        if reduced_context:
            reasons.append(EligibilityReason.REDUCED_CONTEXT)
        return cls(
            selection=selection,
            endpoint_scope=endpoint_scope,
            artifact=artifact,
            execution=execution,
            partition=partition,
            ineligibility_reasons=tuple(reasons),
        )
