"""Check how a framework's version text becomes the full commit a submission names."""

from pathlib import Path

import httpx
import pytest

from agentperf_local.deployment.frameworks import FrameworkExecutable, framework_identity
from agentperf_local.provenance.benchmark import collect_source_provenance
from agentperf_local.submission.contract import Framework
from agentperf_local.submission.framework_commit import (
    FrameworkRef,
    RefKind,
    framework_ref,
    resolve_framework_commit,
)
from tests.fake_executable import write_python_executable

FULL_COMMIT = "c1d0e7a004015f23bc0233470b747b596f29b264"


@pytest.mark.parametrize(
    ("framework", "version", "kind", "value"),
    [
        ("llama-cpp", "version: 0.3.0 (build 10621, commit c1d0e7a00)", "commit", "c1d0e7a00"),
        ("llama-cpp", "version: 6890 (a1b2c3d4)", "commit", "a1b2c3d4"),
        ("vllm", "0.11.0", "tag", "v0.11.0"),
        ("vllm", "0.1.dev20073+g8e685d198", "commit", "8e685d198"),
        ("sglang", "sglang version: 0.5.18", "tag", "v0.5.18"),
        ("splash", "Splash 1.0.2", "tag", "1.0.2"),
        ("sglang", "0.5.3.post1", "tag", "v0.5.3.post1"),
        ("vllm", "0.10.1.1", "tag", "v0.10.1.1"),
        ("vllm", "0.11.0+cu128", "tag", "v0.11.0"),
    ],
)
def test_a_version_names_the_git_reference_of_its_build(
    framework: Framework, version: str, kind: RefKind, value: str
) -> None:
    ref = framework_ref(framework, version)

    assert (ref.kind, ref.value) == (kind, value)


@pytest.mark.parametrize(
    ("framework", "version"),
    [("llama-cpp", "version: 6890"), ("vllm", "unreported"), ("vllm", "0.5.3rc1"), ("sglang", "0.5.3.dev1")],
)
def test_a_version_that_names_no_build_is_refused(framework: Framework, version: str) -> None:
    with pytest.raises(ValueError, match="names no commit or release"):
        framework_ref(framework, version)


@pytest.mark.parametrize(
    ("ref", "answer", "expected"),
    [
        (FrameworkRef(repository="ggml-org/llama.cpp", kind="commit", value="c1d0e7a00"), FULL_COMMIT, FULL_COMMIT),
        (FrameworkRef(repository="ggml-org/llama.cpp", kind="commit", value="0000000"), FULL_COMMIT, "does not match"),
        (FrameworkRef(repository="vllm-project/vllm", kind="tag", value="v0.11.0"), "not-a-commit", "full commit"),
        (FrameworkRef(repository="vllm-project/vllm", kind="tag", value="v9.9.9"), None, "answered 422"),
    ],
    ids=["short-commit", "other-commit", "not-a-commit", "unknown-tag"],
)
def test_github_must_answer_with_the_commit_the_reference_names(
    ref: FrameworkRef, answer: str | None, expected: str
) -> None:
    def github(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/repos/{ref.repository}/commits/{ref.value}"
        if answer is None:
            return httpx.Response(422, json={"message": "No commit found"})
        return httpx.Response(200, json={"sha": answer})

    transport = httpx.MockTransport(github)
    if expected == FULL_COMMIT:
        assert resolve_framework_commit(ref, transport=transport) == FULL_COMMIT
    else:
        with pytest.raises(ValueError, match=expected):
            resolve_framework_commit(ref, transport=transport)


def test_a_full_commit_needs_no_lookup() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    ref = FrameworkRef(repository="incoai/splash", kind="commit", value=FULL_COMMIT)

    assert resolve_framework_commit(ref, transport=httpx.MockTransport(refuse)) == FULL_COMMIT


@pytest.mark.parametrize(
    ("stdout", "stderr", "version"),
    [
        (
            "",
            "ggml_cuda_init: found 1 CUDA devices:\n  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0\n"
            "version: 6890 (a1b2c3d4)\nbuilt with cc for x86_64-linux-gnu\n",
            "version: 6890 (a1b2c3d4)",
        ),
        ("0.11.0\n", "INFO 09-29 12:00:00 [__init__.py:241] Automatically detected platform cuda.\n", "0.11.0"),
    ],
    ids=["llama-cpp-cuda-device-lines", "vllm-log-line"],
)
def test_the_recorded_version_is_the_line_that_states_it(
    tmp_path: Path, stdout: str, stderr: str, version: str
) -> None:
    executable = write_python_executable(
        tmp_path / "server", f"import sys\nsys.stdout.write({stdout!r})\nsys.stderr.write({stderr!r})\n"
    )
    identity = framework_identity(
        FrameworkExecutable(
            framework="llama-cpp",
            command_prefix=(str(executable),),
            version_command=(str(executable),),
            executable_path=executable,
        )
    )

    assert identity.version == version


def test_a_release_build_names_its_embedded_commit_before_asking_git(tmp_path: Path) -> None:
    revision_file = tmp_path / "release-revision.txt"
    revision_file.write_text(f"{FULL_COMMIT}\n", encoding="ascii")

    provenance = collect_source_provenance(tmp_path, release_revision_path=revision_file)

    assert (provenance.source_state, provenance.source_revision) == ("release", FULL_COMMIT)
