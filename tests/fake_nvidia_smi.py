"""Write a fake nvidia-smi that streams one fixed row per interval until it is stopped."""

from __future__ import annotations

from pathlib import Path

from tests.fake_executable import write_python_executable

FAKE_ROW = "91, 14322, 67, 400.0, 2745, 14001"
FAKE_POWER_W = 400.0
FAKE_INTERVAL_SECONDS = 0.02


def write_looping_nvidia_smi(path: Path) -> Path:
    """Write an executable that prints the fixed row at the configured cadence."""
    return write_python_executable(
        path,
        "import sys, time\n"
        "while True:\n"
        f"    sys.stdout.write({FAKE_ROW!r} + '\\n')\n"
        "    sys.stdout.flush()\n"
        f"    time.sleep({FAKE_INTERVAL_SECONDS})\n",
    )
