"""Exercise container build options without running Docker."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from stat import S_IXUSR

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the container build script is a bash script")


@pytest.mark.parametrize(
    ("initial_arch", "requested_arch", "docker_arch", "platform_override"),
    [
        ("arm64", "x86_64", "amd64", None),
        ("x86_64", "arm64", "arm64", None),
        ("x86_64", "arm64", "arm64", "linux/arm64/v8"),
    ],
)
def test_arch_override_controls_build_and_image_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_arch: str,
    requested_arch: str,
    docker_arch: str,
    platform_override: str | None,
) -> None:
    docker = tmp_path / "docker"
    shutil.copyfile(REPOSITORY_ROOT / "tests/fixtures/container_build/docker", docker)
    docker.chmod(docker.stat().st_mode | S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setenv("AGENTPERF_LOCAL_IMAGE_ARCH", initial_arch)
    monkeypatch.setenv("TEST_DOCKER_ARCH", docker_arch)
    monkeypatch.setenv("TEST_DOCKER_PLATFORM", platform_override or f"linux/{docker_arch}")
    monkeypatch.setenv("PINCHBENCH_IMAGE", "test-pinchbench:latest")
    monkeypatch.delenv("PINCHBENCH_PLATFORM", raising=False)
    if platform_override is not None:
        monkeypatch.setenv("PINCHBENCH_PLATFORM", platform_override)

    result = subprocess.run(
        [
            "bash",
            str(REPOSITORY_ROOT / "scripts/build-default-containers.sh"),
            "--arch",
            requested_arch,
            "--skip-swebench",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"Verified test-pinchbench:latest architecture: {docker_arch}" in result.stdout
