"""Exercise the release version check through its command line."""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_release_version.py"
PROJECT_VERSION = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


@pytest.mark.parametrize(
    ("tag", "exit_code"),
    [
        (f"v{PROJECT_VERSION}", 0),
        (PROJECT_VERSION, 1),
        ("v999.0.0", 1),
    ],
)
def test_release_tag_must_match_every_version_string(tag: str, exit_code: int) -> None:
    completed = subprocess.run((sys.executable, str(SCRIPT), tag), capture_output=True, text=True, check=False)

    assert completed.returncode == exit_code, completed.stderr
