"""Report local facts: hardware and managed deployment choices."""

from __future__ import annotations

import argparse
import sys

from agentperf_local.cli.options import (
    print_json,
    read_bound_device,
    read_managed_target,
)
from agentperf_local.common.argparse_fields import (
    read_boolean,
)
from agentperf_local.deployment.frameworks import framework_offers
from agentperf_local.provenance.hardware import (
    collect_hardware_snapshot,
)
from agentperf_local.reports.progress import render_hardware


def doctor_command(namespace: argparse.Namespace) -> int:
    snapshot = collect_hardware_snapshot()
    if read_boolean(namespace, "json"):
        print_json(snapshot.to_json())
    else:
        print(render_hardware(snapshot))
    # A benchmark run binds exactly one accelerator, so any other count is a blocking finding.
    accelerator_count = len(snapshot.accelerators)
    if accelerator_count == 1:
        return 0
    print(f"doctor requires exactly one detected accelerator; found {accelerator_count}", file=sys.stderr)
    return 1


def deployment_options_command(namespace: argparse.Namespace) -> int:
    candidate = read_managed_target(namespace).candidate
    bound = read_bound_device(namespace)
    snapshot = bound.snapshot
    offers = framework_offers(snapshot, candidate)
    can_deploy_now = any(offer.installed and offer.memory_fit is True for offer in offers)
    print_json(
        {
            "kind": "managed_deployment_options",
            "profile_id": candidate.profile_id,
            "model": candidate.hf_repository,
            "device_index": bound.device_index,
            "hardware": snapshot.to_json(),
            "offers": [offer.to_json() for offer in offers],
            "can_deploy_now": can_deploy_now,
            "upload_performed": False,
        }
    )
    return 0 if can_deploy_now else 1
