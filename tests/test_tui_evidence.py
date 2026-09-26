"""Exercise closed evidence derivation for the Textual pilot."""

import pytest

from agentperf_local.tui.evidence import (
    ArtifactEvidence,
    EligibilityReason,
    EndpointScope,
    EvidenceSummary,
    ExecutionEvidence,
    ResultPartition,
    SelectionKind,
)


@pytest.mark.parametrize(
    ("selection", "scope", "artifact", "execution", "partition"),
    (
        (
            SelectionKind.BUNDLED_CATALOG_CANDIDATE,
            EndpointScope.LOOPBACK_NAME,
            ArtifactEvidence.BUNDLED_UNSIGNED_CANDIDATE_METADATA,
            ExecutionEvidence.ATTACHED_ENDPOINT_UNVERIFIED,
            ResultPartition.ATTACHED_ENDPOINT_EXPLORER,
        ),
        (
            SelectionKind.CUSTOM_ENDPOINT,
            EndpointScope.LOOPBACK_NAME,
            ArtifactEvidence.SELF_REPORTED_ENDPOINT_NAME,
            ExecutionEvidence.ATTACHED_ENDPOINT_UNVERIFIED,
            ResultPartition.ATTACHED_ENDPOINT_EXPLORER,
        ),
        (
            SelectionKind.BUNDLED_CATALOG_CANDIDATE,
            EndpointScope.NON_LOOPBACK_NAME,
            ArtifactEvidence.BUNDLED_UNSIGNED_CANDIDATE_METADATA,
            ExecutionEvidence.REMOTE_BACKEND_OPAQUE,
            ResultPartition.SERVICE_LATENCY_ONLY,
        ),
        (
            SelectionKind.CUSTOM_ENDPOINT,
            EndpointScope.NON_LOOPBACK_NAME,
            ArtifactEvidence.SELF_REPORTED_ENDPOINT_NAME,
            ExecutionEvidence.REMOTE_BACKEND_OPAQUE,
            ResultPartition.SERVICE_LATENCY_ONLY,
        ),
    ),
)
def test_evidence_is_derived_without_a_reference_state(
    selection: SelectionKind,
    scope: EndpointScope,
    artifact: ArtifactEvidence,
    execution: ExecutionEvidence,
    partition: ResultPartition,
) -> None:
    evidence = EvidenceSummary.derive(selection, scope)

    assert evidence.artifact is artifact
    assert evidence.execution is execution
    assert evidence.partition is partition
    assert EligibilityReason.USER_CHOSEN_ENDPOINT in evidence.ineligibility_reasons
    assert EligibilityReason.OBSERVER_ENABLED in evidence.ineligibility_reasons
    assert EligibilityReason.SINGLE_REPLAY in evidence.ineligibility_reasons
    assert all("reference" not in value.value for value in ResultPartition)
