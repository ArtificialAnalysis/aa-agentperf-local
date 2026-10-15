"""Convert a recipe's safetensors draft model into the GGUF llama.cpp loads.

- ConvertedDraft: one converted draft file and the commit that converted it.
- ConversionJob: the inputs and output path of one conversion.
- DraftConverter: the callable that runs one conversion.
- ensure_converted_draft: reuse a cached conversion, or run one.
- convert_with_llama_cpp: run llama.cpp's convert_hf_to_gguf.py at a pinned commit with uv.
- fetch_converter: put one llama.cpp commit's converter files in the cache.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import httpx
import orjson
from pydantic import BaseModel

from agentperf_local.common.models import read_record
from agentperf_local.deployment.catalog import DeploymentArtifact, LlamaCppConversionType, LlamaCppDraftConversion

# Cache folders beside the Hugging Face repositories in the same cache root.
CONVERSION_CACHE_DIRECTORY = "agentperf-draft-conversions"
CONVERTER_CACHE_DIRECTORY = "agentperf-llama-cpp-converters"
CONVERSION_RECORD_FILENAME = "conversion.json"
LLAMA_CPP_ARCHIVE_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{commit}"
# The parts of a llama.cpp source tree the converter needs: the script, its model
# classes, the gguf package it imports from the tree, and its pinned requirements.
CONVERTER_PATHS = ("convert_hf_to_gguf.py", "conversion", "gguf-py", "requirements")
CONVERTER_REQUIREMENTS = "requirements/requirements-convert_hf_to_gguf.txt"
# torch, which the converter needs, publishes wheels for this Python on every platform we serve.
CONVERTER_PYTHON_VERSION = "3.12"
CONVERTER_DOWNLOAD_TIMEOUT_SECONDS = 300.0
# Installing torch on first use dominates; a conversion of a few gigabytes takes minutes.
CONVERSION_TIMEOUT_SECONDS = 3_600.0
CONVERSION_ERROR_LINES = 12
HASH_CHUNK_BYTES = 4 * 1024 * 1024
UV_INSTALL_HINT = "converting this recipe's draft model needs uv on PATH: https://docs.astral.sh/uv/"


class ConvertedDraft(BaseModel, frozen=True):
    """Store one converted draft file, its digest, and the llama.cpp commit that wrote it."""

    path: Path
    sha256: str
    size_bytes: int
    converter_commit: str


class ConversionJob(BaseModel, frozen=True):
    """Name the inputs and the output of one draft conversion."""

    converter_commit: str
    outtype: LlamaCppConversionType
    draft_dir: Path
    tokenizer_dir: Path
    output_path: Path


type DraftConverter = Callable[[ConversionJob], None]
type CommandFinder = Callable[[str], str | None]


class _ConversionRecord(BaseModel, frozen=True):
    """Record what one cached conversion produced, so a later run can verify it."""

    converter_commit: str
    outtype: LlamaCppConversionType
    sha256: str
    size_bytes: int


def _conversion_key(conversion: LlamaCppDraftConversion) -> str:
    """Name one conversion by everything that decides its output."""
    sources = [
        [artifact.source_repository, artifact.source_revision, artifact.filename, artifact.sha256]
        for artifact in conversion.source_files
    ]
    identity = [conversion.converter_commit, conversion.outtype, conversion.output_filename, sources]
    return hashlib.sha256(orjson.dumps(identity)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _reuse_cached(output_path: Path, record_path: Path, converter_commit: str) -> ConvertedDraft | None:
    """Return a cached conversion whose file still matches its record, or None."""
    if not output_path.is_file() or not record_path.is_file():
        return None
    try:
        record = read_record(_ConversionRecord, record_path.read_bytes(), str(record_path))
    except ValueError:
        return None
    if output_path.stat().st_size != record.size_bytes or _file_sha256(output_path) != record.sha256:
        return None
    return ConvertedDraft(
        path=output_path, sha256=record.sha256, size_bytes=record.size_bytes, converter_commit=converter_commit
    )


def _link_inputs(directory: Path, files: tuple[tuple[DeploymentArtifact, Path], ...]) -> None:
    """Lay out verified files under their repository names, as the converter expects.

    A hard link costs no space; a copy covers a file system that refuses links.
    """
    for artifact, path in files:
        entry = directory / PurePosixPath(artifact.filename)
        entry.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, entry)
        except OSError:
            shutil.copyfile(path, entry)


def ensure_converted_draft(
    cache_root: Path,
    conversion: LlamaCppDraftConversion,
    verified_sources: tuple[tuple[DeploymentArtifact, Path], ...],
    *,
    convert: DraftConverter,
) -> ConvertedDraft:
    """Return the draft GGUF for one conversion, converting it once per set of pinned inputs.

    The output is cached under a key over the converter commit, the tensor type, and
    every source file's digest, and reused only while it still matches the size and
    digest recorded when it was written.
    """
    key = _conversion_key(conversion)
    final_dir = cache_root / CONVERSION_CACHE_DIRECTORY / key
    record_path = final_dir / CONVERSION_RECORD_FILENAME
    cached = _reuse_cached(final_dir / conversion.output_filename, record_path, conversion.converter_commit)
    if cached is not None:
        return cached
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{key[:16]}-", dir=final_dir.parent))
    try:
        draft_dir = work_dir / "draft"
        tokenizer_dir = work_dir / "tokenizer"
        output_dir = work_dir / "output"
        output_dir.mkdir()
        # Both groups pin a config.json, so a file is placed by its whole pinned identity.
        _link_inputs(draft_dir, tuple(pair for pair in verified_sources if pair[0] in conversion.model_files))
        _link_inputs(tokenizer_dir, tuple(pair for pair in verified_sources if pair[0] in conversion.tokenizer_files))
        output_path = output_dir / conversion.output_filename
        convert(
            ConversionJob(
                converter_commit=conversion.converter_commit,
                outtype=conversion.outtype,
                draft_dir=draft_dir,
                tokenizer_dir=tokenizer_dir,
                output_path=output_path,
            )
        )
        if not output_path.is_file():
            raise ValueError(f"the draft converter wrote no {conversion.output_filename}")
        record = _ConversionRecord(
            converter_commit=conversion.converter_commit,
            outtype=conversion.outtype,
            sha256=_file_sha256(output_path),
            size_bytes=output_path.stat().st_size,
        )
        (output_dir / CONVERSION_RECORD_FILENAME).write_bytes(orjson.dumps(record.model_dump(mode="json")))
        # A folder left by an interrupted run never got a matching record; replace it.
        shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(output_dir, final_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return ConvertedDraft(
        path=final_dir / conversion.output_filename,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        converter_commit=conversion.converter_commit,
    )


def fetch_converter(cache_root: Path, commit: str, *, transport: httpx.BaseTransport | None = None) -> Path:
    """Return a folder with one llama.cpp commit's converter, downloading it once.

    Only the converter's own files are kept from the source archive, and an archive
    member that would land outside the folder is refused.
    """
    final_dir = cache_root / CONVERTER_CACHE_DIRECTORY / commit
    if (final_dir / CONVERTER_PATHS[0]).is_file():
        return final_dir
    url = LLAMA_CPP_ARCHIVE_URL.format(commit=commit)
    with httpx.Client(timeout=CONVERTER_DOWNLOAD_TIMEOUT_SECONDS, transport=transport, follow_redirects=True) as client:
        try:
            response = client.get(url)
        except httpx.HTTPError as error:
            reason = type(error).__name__
            raise ValueError(f"could not download the llama.cpp converter at {commit}: {reason}") from error
    if response.status_code != httpx.codes.OK:
        raise ValueError(f"could not download the llama.cpp converter at {commit}: HTTP {response.status_code}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{commit[:12]}-", dir=final_dir.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
            members = []
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts
                # The archive holds one top-level folder named after the repository and commit.
                if len(parts) >= 2 and parts[1] in CONVERTER_PATHS:
                    member.name = PurePosixPath(*parts[1:]).as_posix()
                    members.append(member)
            archive.extractall(work_dir, members=members, filter="data")
        if not (work_dir / CONVERTER_PATHS[0]).is_file() or not (work_dir / CONVERTER_REQUIREMENTS).is_file():
            raise ValueError(f"the llama.cpp source at {commit} has no draft converter")
        shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(work_dir, final_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return final_dir


def convert_with_llama_cpp(
    cache_root: Path,
    *,
    command_finder: CommandFinder = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    transport: httpx.BaseTransport | None = None,
) -> DraftConverter:
    """Return a converter that runs llama.cpp's convert_hf_to_gguf.py in its own uv environment.

    The environment comes from the commit's own requirements file, so the client never
    installs torch itself.
    """

    def convert(job: ConversionJob) -> None:
        uv = command_finder("uv")
        if uv is None:
            raise ValueError(UV_INSTALL_HINT)
        converter = fetch_converter(cache_root, job.converter_commit, transport=transport)
        command = (
            uv,
            "run",
            "--no-project",
            "--python",
            CONVERTER_PYTHON_VERSION,
            "--with-requirements",
            str(converter / CONVERTER_REQUIREMENTS),
            "python",
            str(converter / CONVERTER_PATHS[0]),
            str(job.draft_dir),
            "--target-model-dir",
            str(job.tokenizer_dir),
            "--outtype",
            job.outtype,
            "--outfile",
            str(job.output_path),
        )
        try:
            result = run(command, capture_output=True, text=True, timeout=CONVERSION_TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired as error:
            raise ValueError("the draft conversion did not finish in time") from error
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-CONVERSION_ERROR_LINES:])
            raise ValueError(f"the draft conversion failed with exit status {result.returncode}:\n{tail}")

    return convert
