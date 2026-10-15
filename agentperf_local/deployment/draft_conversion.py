"""Convert a recipe's safetensors draft model into the GGUF llama.cpp loads.

- ConvertedDraft: one converted draft file and its digest.
- ConversionJob: the inputs and output path of one conversion.
- DraftConverter: the callable that runs one conversion.
- cached_converted_draft: return a cached conversion that still matches its record.
- convert_draft: run one conversion and cache its output.
- convert_with_llama_cpp: run llama.cpp's convert_hf_to_gguf.py at a pinned commit with uv.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import httpx
import orjson
from pydantic import BaseModel

from agentperf_local.common.durable_files import NewFile, write_new_file
from agentperf_local.common.identity import sha256_file
from agentperf_local.common.json_records import json_record
from agentperf_local.common.models import read_record
from agentperf_local.deployment.catalog import DeploymentArtifact, LlamaCppConversionType, LlamaCppDraftConversion
from agentperf_local.deployment.frameworks import CommandFinder, find_executable

# Cache folders beside the Hugging Face repositories in the same cache root.
CONVERSION_CACHE_DIRECTORY = "agentperf-draft-conversions"
CONVERTER_CACHE_DIRECTORY = "agentperf-llama-cpp-converters"
CONVERSION_RECORD_FILENAME = "conversion.json"
LLAMA_CPP_ARCHIVE_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{commit}"
CONVERTER_SCRIPT = "convert_hf_to_gguf.py"
CONVERTER_REQUIREMENTS = "requirements/requirements-convert_hf_to_gguf.txt"
# The parts of a llama.cpp source tree the converter needs: the script, its model
# classes, the gguf package it imports from the tree, and its pinned requirements.
CONVERTER_PATHS = (CONVERTER_SCRIPT, "conversion", "gguf-py", "requirements")
# torch, which the converter needs, publishes wheels for this Python on every platform we serve.
CONVERTER_PYTHON_VERSION = "3.12"
CONVERTER_DOWNLOAD_TIMEOUT_SECONDS = 300.0
# Installing torch on first use dominates; a conversion of a few gigabytes takes minutes.
CONVERSION_TIMEOUT_SECONDS = 3_600.0
CONVERSION_ERROR_LINES = 12
UV_INSTALL_HINT = "converting this recipe's draft model needs uv on PATH: https://docs.astral.sh/uv/"


class ConvertedDraft(BaseModel, frozen=True):
    """Store one converted draft file and its digest."""

    path: Path
    sha256: str
    size_bytes: int


class ConversionJob(BaseModel, frozen=True):
    """Name the inputs and the output of one draft conversion."""

    converter_commit: str
    outtype: LlamaCppConversionType
    draft_dir: Path
    tokenizer_dir: Path
    output_path: Path


type DraftConverter = Callable[[ConversionJob], None]


class _ConversionRecord(BaseModel, frozen=True):
    """Record one cached conversion's digest and the file state it was taken from."""

    sha256: str
    size_bytes: int
    modified_ns: int


def _conversion_dir(cache_root: Path, conversion: LlamaCppDraftConversion) -> Path:
    """Name one conversion's cache folder by everything that decides its output."""
    sources = [
        [artifact.source_repository, artifact.source_revision, artifact.filename, artifact.sha256]
        for artifact in conversion.source_files
    ]
    identity = [conversion.converter_commit, conversion.outtype, conversion.output_filename, sources]
    return cache_root / CONVERSION_CACHE_DIRECTORY / hashlib.sha256(orjson.dumps(identity)).hexdigest()


@contextmanager
def _staged_folder(final_dir: Path) -> Iterator[Path]:
    """Yield an empty folder beside final_dir that replaces it when the block succeeds.

    A folder left by an interrupted run is replaced whole, and a failed build leaves
    the previous folder alone.
    """
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{final_dir.name[:16]}-", dir=final_dir.parent))
    try:
        yield work_dir
        shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(work_dir, final_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def cached_converted_draft(cache_root: Path, conversion: LlamaCppDraftConversion) -> ConvertedDraft | None:
    """Return the cached conversion while its file is unchanged since it was recorded, or None.

    The converted bytes are not pinned, so the size and modification time recorded when
    the file was written stand in for re-hashing it on every run.
    """
    folder = _conversion_dir(cache_root, conversion)
    output_path = folder / conversion.output_filename
    record_path = folder / CONVERSION_RECORD_FILENAME
    if output_path.is_symlink() or not output_path.is_file() or not record_path.is_file():
        return None
    try:
        record = read_record(_ConversionRecord, record_path.read_bytes(), str(record_path))
    except ValueError:
        return None
    metadata = output_path.stat()
    if metadata.st_size != record.size_bytes or metadata.st_mtime_ns != record.modified_ns:
        return None
    return ConvertedDraft(path=output_path, sha256=record.sha256, size_bytes=record.size_bytes)


def _link_inputs(
    directory: Path, files: tuple[DeploymentArtifact, ...], paths: Mapping[DeploymentArtifact, Path]
) -> None:
    """Lay out verified files under their repository names, as the converter expects.

    A hard link costs no space; a copy covers a file system that refuses links.
    """
    for artifact in files:
        entry = directory / PurePosixPath(artifact.filename)
        entry.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(paths[artifact], entry)
        except OSError:
            shutil.copyfile(paths[artifact], entry)


def convert_draft(
    cache_root: Path,
    conversion: LlamaCppDraftConversion,
    source_paths: Mapping[DeploymentArtifact, Path],
    *,
    convert: DraftConverter,
) -> ConvertedDraft:
    """Convert one draft from its verified sources into the conversion's cache folder."""
    folder = _conversion_dir(cache_root, conversion)
    with _staged_folder(folder) as output_dir, tempfile.TemporaryDirectory(dir=folder.parent) as inputs:
        draft_dir = Path(inputs) / "draft"
        tokenizer_dir = Path(inputs) / "tokenizer"
        # Each group gets its own folder, because both pin a config.json.
        _link_inputs(draft_dir, conversion.model_files, source_paths)
        _link_inputs(tokenizer_dir, conversion.tokenizer_files, source_paths)
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
        metadata = output_path.stat()
        record = _ConversionRecord(
            sha256=sha256_file(output_path), size_bytes=metadata.st_size, modified_ns=metadata.st_mtime_ns
        )
        write_new_file(NewFile(path=output_dir / CONVERSION_RECORD_FILENAME, data=orjson.dumps(json_record(record))))
    return ConvertedDraft(path=folder / conversion.output_filename, sha256=record.sha256, size_bytes=record.size_bytes)


def _fetch_converter(cache_root: Path, commit: str, transport: httpx.BaseTransport | None) -> Path:
    """Return a folder with one llama.cpp commit's converter, downloading it once.

    The source archive is streamed to disk and read in one pass. Only the converter's
    own files are kept, and a member that would land outside the folder is refused.
    """
    final_dir = cache_root / CONVERTER_CACHE_DIRECTORY / commit
    if (final_dir / CONVERTER_SCRIPT).is_file():
        return final_dir
    with _staged_folder(final_dir) as work_dir, tempfile.TemporaryDirectory(dir=final_dir.parent) as scratch:
        archive_path = Path(scratch) / "llama.cpp.tar.gz"
        url = LLAMA_CPP_ARCHIVE_URL.format(commit=commit)
        try:
            with (
                httpx.Client(
                    timeout=CONVERTER_DOWNLOAD_TIMEOUT_SECONDS, transport=transport, follow_redirects=True
                ) as client,
                client.stream("GET", url) as response,
            ):
                if response.status_code != httpx.codes.OK:
                    status = response.status_code
                    raise ValueError(f"could not download the llama.cpp converter at {commit}: HTTP {status}")
                with archive_path.open("wb") as archive_file:
                    for chunk in response.iter_bytes():
                        archive_file.write(chunk)
        except httpx.HTTPError as error:
            reason = type(error).__name__
            raise ValueError(f"could not download the llama.cpp converter at {commit}: {reason}") from error
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                # The archive holds one top-level folder named after the repository and commit.
                if len(parts) >= 2 and parts[1] in CONVERTER_PATHS:
                    member.name = PurePosixPath(*parts[1:]).as_posix()
                    archive.extract(member, work_dir, filter="data")
        if not (work_dir / CONVERTER_SCRIPT).is_file() or not (work_dir / CONVERTER_REQUIREMENTS).is_file():
            raise ValueError(f"the llama.cpp source at {commit} has no draft converter")
    return final_dir


def convert_with_llama_cpp(
    cache_root: Path,
    *,
    command_finder: CommandFinder = find_executable,
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
        converter = _fetch_converter(cache_root, job.converter_commit, transport)
        command = (
            uv,
            "run",
            "--no-project",
            "--python",
            CONVERTER_PYTHON_VERSION,
            "--with-requirements",
            str(converter / CONVERTER_REQUIREMENTS),
            "python",
            str(converter / CONVERTER_SCRIPT),
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
