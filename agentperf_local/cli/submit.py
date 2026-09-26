"""Build, validate, and send one public submission."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from agentperf_local.cli.options import print_json, resolve_submit_token
from agentperf_local.common.argparse_fields import (
    read_boolean,
    read_path,
    read_string,
)
from agentperf_local.provenance.benchmark import (
    SourceProvenance,
)
from agentperf_local.submission.bundle import (
    BUNDLE_STATUS,
    build_submission_bundle,
    validate_bundle_output_path,
    validate_submission_bundle,
    write_submission_bundle,
)
from agentperf_local.submission.client import (
    PRIVATE_AUDIT_NOTICE,
    PRIVATE_AUDIT_NOTICE_VERSION,
    SubmissionError,
    check_revision_allowlist,
    fetch_submission_status,
    submit_bundle,
)


def prepare_submission_command(namespace: argparse.Namespace) -> int:
    results_dir = read_path(namespace, "results_dir")
    output_dir = read_path(namespace, "output_dir")
    validate_bundle_output_path(results_dir, output_dir)
    bundle = build_submission_bundle(results_dir)
    written = write_submission_bundle(output_dir, bundle)
    print_json(
        {
            "bundle": str(output_dir),
            "run_id": bundle.aggregate.run_id,
            "aggregate_payload_digest": bundle.aggregate.payload_digest,
            "sanitized_rows_digest": bundle.evidence.rows_digest,
            "manifest_digest": written.manifest_digest,
            "total_byte_size": written.total_byte_size,
            "private_audit": {
                "deployment_record": bundle.audit.deployment is not None,
                "runtime_qualification": bundle.audit.runtime_qualification is not None,
                "power_summary": bundle.audit.power is not None,
            },
            "status": BUNDLE_STATUS,
            "upload_performed": False,
        }
    )
    return 0


def _confirm_private_audit_notice(namespace: argparse.Namespace) -> None:
    """Show the notice and require an explicit yes before any byte leaves the machine."""
    print(PRIVATE_AUDIT_NOTICE, file=sys.stderr)
    print(f"(notice version {PRIVATE_AUDIT_NOTICE_VERSION})", file=sys.stderr)
    if read_boolean(namespace, "yes"):
        return
    if not sys.stdin.isatty():
        raise ValueError("submit needs --yes when it cannot ask on a terminal")
    answer = input("Send this bundle to Artificial Analysis? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        raise ValueError("submission cancelled; nothing was sent")


def _upload_progress_printer() -> Callable[[int, int], None]:
    last_step = -1

    def report(sent_bytes: int, total_bytes: int) -> None:
        nonlocal last_step
        if total_bytes <= 0:
            return
        step = min(sent_bytes * 10 // total_bytes, 10)
        if step > last_step:
            last_step = step
            print(f"upload {sent_bytes:,} / {total_bytes:,} bytes", file=sys.stderr)

    return report


def _warn_about_revision(base_url: str, provenance: SourceProvenance) -> None:
    """Advisory only: say now if this build cannot reach verified, but never block the upload."""
    advice = check_revision_allowlist(provenance, base_url=base_url).advice
    if advice is not None:
        print(f"note: {advice}", file=sys.stderr)


def submit_command(namespace: argparse.Namespace) -> int:
    bundle_dir = read_path(namespace, "bundle_dir")
    base_url = read_string(namespace, "base_url")
    token = resolve_submit_token(namespace)
    if token is None:
        print(f"note: {read_string(namespace, 'token_env')} is unset or blank; submitting anonymously", file=sys.stderr)
    _confirm_private_audit_notice(namespace)
    try:
        bundle = validate_submission_bundle(bundle_dir)
        _warn_about_revision(base_url, bundle.producer)
        receipt = submit_bundle(bundle, base_url=base_url, token=token, progress=_upload_progress_printer())
    except SubmissionError as error:
        detail = f" ({', '.join(error.reasons)})" if error.reasons else ""
        retry = f"; retry after {error.retry_after_seconds} s" if error.retry_after_seconds is not None else ""
        raise ValueError(f"submission refused: {error}{detail}{retry}") from error
    print_json(
        {
            "submission_id": receipt.submission_id,
            "status": receipt.status,
            "created": receipt.created,
            "bytes_sent": receipt.bytes_sent,
            "bundle": str(bundle_dir),
            "authenticated": token is not None,
            "private_audit_acknowledged": True,
            "notice_version": PRIVATE_AUDIT_NOTICE_VERSION,
            "upload_performed": True,
        }
    )
    return 0


def submission_status_command(namespace: argparse.Namespace) -> int:
    try:
        status = fetch_submission_status(
            read_string(namespace, "submission_id"),
            base_url=read_string(namespace, "base_url"),
            token=resolve_submit_token(namespace),
        )
    except SubmissionError as error:
        raise ValueError(f"status request refused: {error}") from error
    print_json(status.to_json())
    return 0
