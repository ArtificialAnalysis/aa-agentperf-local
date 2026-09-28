"""Run one agentperf-local command.

Every subcommand is defined in a sibling module and named here once.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from agentperf_local.cli.inspect import deployment_options_command, doctor_command
from agentperf_local.cli.options import INTERRUPTED_STATUS
from agentperf_local.cli.parsers import DEFAULT_COMMAND, build_parser
from agentperf_local.cli.replay import managed_run_command, run_command, tui_command
from agentperf_local.cli.submit import (
    prepare_submission_command,
    submission_status_command,
    submit_command,
)
from agentperf_local.cli.workload import convert_command
from agentperf_local.common.argparse_fields import read_string
from agentperf_local.common.models import error_text

COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "convert": convert_command,
    "run": run_command,
    DEFAULT_COMMAND: tui_command,
    "doctor": doctor_command,
    "deployment-options": deployment_options_command,
    "managed-run": managed_run_command,
    "prepare-submission": prepare_submission_command,
    "submit": submit_command,
    "submission-status": submission_status_command,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its process status.

    An empty command line opens the TUI with its default settings.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    namespace = build_parser().parse_args(arguments or [DEFAULT_COMMAND])
    try:
        return COMMANDS[read_string(namespace, "command")](namespace)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return INTERRUPTED_STATUS
    except Exception as error:
        print(f"error: {error_text(error)}", file=sys.stderr)
        return 1
