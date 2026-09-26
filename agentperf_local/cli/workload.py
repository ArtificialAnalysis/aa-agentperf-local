"""Convert agent recordings into replay manifests."""

from __future__ import annotations

import argparse

from agentperf_local.cli.options import print_json, read_message_source
from agentperf_local.common.argparse_fields import (
    read_boolean,
    read_optional_string,
    read_path,
    read_string,
)
from agentperf_local.workload.recording import convert_manifest_to_dir, convert_one_recording_to_dir


def convert_command(namespace: argparse.Namespace) -> int:
    input_path = read_path(namespace, "input")
    output_dir = read_path(namespace, "output_dir")
    message_source = read_message_source(namespace)
    include_outputs = read_boolean(namespace, "include_tool_outputs")
    manifest_name = read_optional_string(namespace, "manifest_name")
    if read_string(namespace, "kind") == "manifest":
        if (
            read_optional_string(namespace, "family") is not None
            or read_optional_string(namespace, "adapter") is not None
        ):
            raise ValueError("--family and --adapter apply only to recording conversion")
        manifest = convert_manifest_to_dir(
            input_path,
            output_dir,
            message_source=message_source,
            include_tool_outputs=include_outputs,
            manifest_name=manifest_name,
        )
        trace_path = None
    else:
        manifest, trace_path = convert_one_recording_to_dir(
            input_path,
            output_dir,
            family=read_optional_string(namespace, "family"),
            adapter=read_optional_string(namespace, "adapter"),
            message_source=message_source,
            include_tool_outputs=include_outputs,
            manifest_name=manifest_name,
        )
    print_json(
        {
            "manifest": str(output_dir / "manifest.json"),
            "trace": str(trace_path) if trace_path is not None else None,
            "tasks": len(manifest.tasks),
            "model_calls": sum(task.model_calls for task in manifest.tasks),
            "tool_calls": sum(task.tool_calls for task in manifest.tasks),
        }
    )
    return 0
