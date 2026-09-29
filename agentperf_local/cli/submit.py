"""Build, check, and send one submission."""

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
from agentperf_local.submission.builder import (
    build_submission_request,
    encode_submission,
    read_prepared_submission,
    write_prepared_submission,
)
from agentperf_local.submission.client import (
    SubmissionError,
    fetch_submission_status,
    submit_body,
)
from agentperf_local.submission.notice import PRIVACY_NOTICE, PRIVACY_NOTICE_VERSION


def prepare_submission_command(namespace: argparse.Namespace) -> int:
    results_dir = read_path(namespace, "results_dir")
    output = read_path(namespace, "output")
    request = build_submission_request(results_dir)
    encoded = encode_submission(request)
    write_prepared_submission(results_dir, output, encoded)
    print_json(
        {
            "submission": str(output),
            "run_id": request.run_id,
            "deployment_mode": request.deployment.deployment_mode,
            "turns": len(request.turns),
            "byte_size": len(encoded),
            "privacy_notice_version": request.privacy_notice_version,
            "upload_performed": False,
        }
    )
    return 0


def _confirm_privacy_notice(namespace: argparse.Namespace) -> None:
    """Show the notice and require an explicit yes before any byte leaves the machine."""
    print(PRIVACY_NOTICE, file=sys.stderr)
    print(f"(notice version {PRIVACY_NOTICE_VERSION})", file=sys.stderr)
    if read_boolean(namespace, "yes"):
        return
    if not sys.stdin.isatty():
        raise ValueError("submit needs --yes when it cannot ask on a terminal")
    answer = input("Send this submission to Artificial Analysis? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        raise ValueError("submission canceled; nothing was sent")


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


def submit_command(namespace: argparse.Namespace) -> int:
    submission_path = read_path(namespace, "submission")
    base_url = read_string(namespace, "base_url")
    token = resolve_submit_token(namespace)
    if token is None:
        print(f"note: {read_string(namespace, 'token_env')} is unset or blank; submitting anonymously", file=sys.stderr)
    prepared = read_prepared_submission(submission_path)
    _confirm_privacy_notice(namespace)
    try:
        receipt = submit_body(prepared.encoded, base_url=base_url, token=token, progress=_upload_progress_printer())
    except SubmissionError as error:
        detail = f" ({', '.join(error.reasons)})" if error.reasons else ""
        raise ValueError(f"submission refused: {error}{detail}") from error
    print_json(
        {
            "submission_id": receipt.submission_id,
            "status": receipt.status,
            "created": receipt.created,
            "bytes_sent": receipt.bytes_sent,
            "submission": str(submission_path),
            "run_id": prepared.request.run_id,
            "authenticated": token is not None,
            "privacy_notice_version": PRIVACY_NOTICE_VERSION,
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
