"""Exercise the llama.cpp draft converter without the network or a real conversion."""

import io
import re
import subprocess
import tarfile
from pathlib import Path

import httpx
import pytest

from agentperf_local.deployment.draft_conversion import (
    CONVERTER_REQUIREMENTS,
    UV_INSTALL_HINT,
    ConversionJob,
    convert_with_llama_cpp,
)

COMMIT = "f" * 40
ARCHIVE_FILES = {
    "convert_hf_to_gguf.py": b"# converter",
    "conversion/qwen.py": b"# model classes",
    "gguf-py/gguf/__init__.py": b"# gguf",
    CONVERTER_REQUIREMENTS: b"torch==2.11.0",
    # Everything else in the source tree stays out of the cache.
    "README.md": b"# llama.cpp",
    "ggml/src/ggml.c": b"/* kernels */",
}


def _source_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in ARCHIVE_FILES.items():
            member = tarfile.TarInfo(f"llama.cpp-{COMMIT}/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def _job(tmp_path: Path) -> ConversionJob:
    return ConversionJob(
        converter_commit=COMMIT,
        outtype="bf16",
        draft_dir=tmp_path / "draft",
        tokenizer_dir=tmp_path / "tokenizer",
        output_path=tmp_path / "draft-bf16.gguf",
    )


def test_converter_runs_the_commits_script_in_its_own_uv_environment(tmp_path: Path) -> None:
    requests: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=_source_archive())

    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    convert = convert_with_llama_cpp(
        tmp_path / "cache", command_finder=lambda _: "/usr/bin/uv", run=run, transport=httpx.MockTransport(serve)
    )
    job = _job(tmp_path)
    convert(job)
    # The converter is downloaded once per commit.
    convert(job)

    converter = tmp_path / "cache" / "agentperf-llama-cpp-converters" / COMMIT
    assert requests == [f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{COMMIT}"]
    assert sorted(path.relative_to(converter).as_posix() for path in converter.rglob("*") if path.is_file()) == [
        "conversion/qwen.py",
        "convert_hf_to_gguf.py",
        "gguf-py/gguf/__init__.py",
        CONVERTER_REQUIREMENTS,
    ]
    assert commands[0] == (
        "/usr/bin/uv",
        "run",
        "--no-project",
        "--python",
        "3.12",
        "--with-requirements",
        str(converter / CONVERTER_REQUIREMENTS),
        "python",
        str(converter / "convert_hf_to_gguf.py"),
        str(job.draft_dir),
        "--target-model-dir",
        str(job.tokenizer_dir),
        "--outtype",
        "bf16",
        "--outfile",
        str(job.output_path),
    )


@pytest.mark.parametrize(
    ("uv", "returncode", "message"),
    [
        (None, 0, UV_INSTALL_HINT),
        ("/usr/bin/uv", 1, "the draft conversion failed with exit status 1:\nTraceback\nKeyError: 'hidden_size'"),
    ],
)
def test_converter_reports_why_it_could_not_convert(
    tmp_path: Path, uv: str | None, returncode: int, message: str
) -> None:
    def run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="Traceback\nKeyError: 'hidden_size'")

    convert = convert_with_llama_cpp(
        tmp_path / "cache",
        command_finder=lambda _: uv,
        run=run,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=_source_archive())),
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        convert(_job(tmp_path))
