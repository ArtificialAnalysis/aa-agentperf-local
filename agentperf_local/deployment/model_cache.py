"""Put one recipe's exact pinned bytes into the local model cache.

Every published file is verified by size and SHA-256 before it can be served.
A download resumes only onto bytes that still hash correctly, and bytes that can
never verify are discarded rather than kept.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Annotated

import httpx
import orjson
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.file_download import repo_folder_name
from pydantic import BaseModel, Field

from agentperf_local.common.durable_files import (
    PRIVATE_FILE_PERMISSIONS,
    PRIVATE_OPEN_FLAGS,
    nearest_existing_ancestor,
)
from agentperf_local.common.identity import sha256_bytes
from agentperf_local.common.json_types import JsonValue
from agentperf_local.common.units import BYTES_PER_GIB
from agentperf_local.deployment.catalog import DeploymentArtifact, ModelCandidate, ModelDeployment

MODEL_DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024


MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 10.0


MODEL_DOWNLOAD_READ_TIMEOUT_SECONDS = 30.0


MODEL_DOWNLOAD_WRITE_TIMEOUT_SECONDS = 30.0


MODEL_DOWNLOAD_POOL_TIMEOUT_SECONDS = 10.0


MODEL_DOWNLOAD_TOTAL_TIMEOUT_SECONDS = 7_200.0


# The suffix huggingface_hub gives its own unfinished blob downloads.
MODEL_DOWNLOAD_INCOMPLETE_SUFFIX = ".incomplete"


MODEL_DOWNLOAD_INITIAL_BACKOFF_SECONDS = 1.0


MODEL_DOWNLOAD_MAXIMUM_BACKOFF_SECONDS = 60.0


MODEL_DOWNLOAD_BACKOFF_MULTIPLIER = 2.0


MODEL_DOWNLOAD_BACKOFF_POLL_SECONDS = 0.05


MODEL_DOWNLOAD_PROGRESS_INTERVAL_SECONDS = 0.5


MODEL_DOWNLOAD_FREE_SPACE_MARGIN_BYTES = BYTES_PER_GIB


# llama.cpp names the parts of a split GGUF "<prefix>-00001-of-00003.gguf" and finds the other
# parts from the first part's file name.
SPLIT_GGUF_FIRST_PART = re.compile(r"^(?P<prefix>.+)-00001-of-(?P<count>\d{5})\.gguf$")

type CancellationCheck = Callable[[], bool]


type ProgressReport = Callable[[int, int], None]


type Sha256Hash = hashlib._Hash


def _never_cancelled() -> bool:
    return False


class _CorruptDownloadError(ValueError):
    """Report downloaded bytes that can never verify, so the partial file must be discarded."""


class VerifiedArtifact(BaseModel, frozen=True):
    """Store one verified model artifact without hashing it again."""

    filename: str
    path: Path
    sha256: str
    size_bytes: int
    device_id: int
    inode: int
    modified_ns: int


class VerifiedDeployment(BaseModel, frozen=True):
    """Store every verified file of one recipe and the path its runtime is given."""

    model_path: Path
    artifacts: Annotated[tuple[VerifiedArtifact, ...], Field(min_length=1)]
    draft_model_path: Path | None = None
    # The hub cache that holds the verified snapshots, for a runtime that reads them by
    # repository and revision itself (Splash) instead of by path.
    hub_cache: Path | None = None

    @property
    def size_bytes(self) -> int:
        """Return the bytes every verified file occupies together."""
        return sum(artifact.size_bytes for artifact in self.artifacts)

    @property
    def manifest_sha256(self) -> str:
        """Return one digest over every pinned file, so a whole repository has one identity.

        A single-file recipe still gets a digest of its manifest rather than of its
        bytes, so every managed run records the same kind of identifier.
        """
        entries: list[JsonValue] = [
            [artifact.filename, artifact.sha256] for artifact in sorted(self.artifacts, key=lambda one: one.filename)
        ]
        return sha256_bytes(orjson.dumps(entries))


def default_model_cache_root() -> Path:
    """Return the standard Hugging Face hub cache directory.

    The environment contract mirrors huggingface_hub: HF_HUB_CACHE wins, then
    HF_HOME/hub, then XDG_CACHE_HOME/huggingface/hub, then ~/.cache/huggingface/hub.
    huggingface_hub freezes these variables into constants at import time, so this
    function reads the environment itself to honor changes made after import.
    """
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _artifact_url(candidate: ModelCandidate, artifact: DeploymentArtifact) -> str:
    repository = artifact.source_repository or candidate.hf_repository
    revision = artifact.source_revision or candidate.hf_revision
    return f"https://huggingface.co/{repository}/resolve/{revision}/{artifact.filename}?download=true"


def _artifact_repository(candidate: ModelCandidate, artifact: DeploymentArtifact) -> str:
    """Return the repository that owns one artifact."""
    return artifact.source_repository or candidate.hf_repository


def _artifact_revision(candidate: ModelCandidate, artifact: DeploymentArtifact) -> str:
    """Return the commit that owns one artifact."""
    return artifact.source_revision or candidate.hf_revision


def _hf_repository_cache(cache_root: Path, repository: str) -> Path:
    return cache_root / repo_folder_name(repo_id=repository, repo_type="model")


def _hf_snapshot_root(cache_root: Path, repository: str, revision: str) -> Path:
    """Return the hub cache directory that holds the pinned commit revision."""
    return _hf_repository_cache(cache_root, repository) / "snapshots" / revision


def _hf_snapshot_path(cache_root: Path, candidate: ModelCandidate, artifact: DeploymentArtifact) -> Path:
    """Return the hub cache path a file of the pinned commit revision must occupy."""
    return (
        _hf_snapshot_root(
            cache_root,
            _artifact_repository(candidate, artifact),
            _artifact_revision(candidate, artifact),
        )
        / artifact.filename
    )


def _hf_blob_path(cache_root: Path, candidate: ModelCandidate, artifact: DeploymentArtifact) -> Path:
    # The hub names a Git LFS blob by the sha256 of its content, so huggingface_hub
    # downloads of a weights file land on this same blob. A small plain file is named
    # by its Git SHA-1 there instead; naming it by sha256 here only means the hub may
    # later keep its own copy of a few kilobytes, and the snapshot link still resolves.
    return _hf_repository_cache(cache_root, _artifact_repository(candidate, artifact)) / "blobs" / artifact.sha256


def _model_artifact_digest(path: Path, cancellation_requested: CancellationCheck) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(MODEL_DOWNLOAD_CHUNK_BYTES):
            if cancellation_requested():
                raise InterruptedError("model artifact operation was cancelled")
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def verify_model_artifact(
    path: Path,
    artifact: DeploymentArtifact,
    *,
    cancellation_requested: CancellationCheck = _never_cancelled,
) -> VerifiedArtifact:
    """Verify one regular artifact against its pinned size and digest."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("model artifact must be a regular file, not a symbolic link")
    initial_metadata = path.stat()
    size_bytes = initial_metadata.st_size
    if size_bytes != artifact.size_bytes:
        raise ValueError(f"model artifact {artifact.filename} size does not match the canonical catalog")
    expected = f"sha256:{artifact.sha256}"
    artifact_sha256 = _model_artifact_digest(path, cancellation_requested)
    if artifact_sha256 != expected:
        raise ValueError(f"model artifact {artifact.filename} digest does not match the canonical catalog")
    final_metadata = path.stat()
    stable_identity = (
        initial_metadata.st_dev == final_metadata.st_dev
        and initial_metadata.st_ino == final_metadata.st_ino
        and initial_metadata.st_size == final_metadata.st_size
        and initial_metadata.st_mtime_ns == final_metadata.st_mtime_ns
    )
    if not stable_identity:
        raise ValueError("model artifact changed while its digest was being verified")
    return VerifiedArtifact(
        filename=artifact.filename,
        path=path,
        sha256=artifact_sha256,
        size_bytes=size_bytes,
        device_id=final_metadata.st_dev,
        inode=final_metadata.st_ino,
        modified_ns=final_metadata.st_mtime_ns,
    )


def _prepare_cache_parent(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("model cache path must be a directory, not a symbolic link")


def _validate_download_response(
    response: httpx.Response,
    *,
    downloaded_bytes: int,
    artifact_size_bytes: int,
) -> None:
    """Validate an initial or resumed artifact response before writing bytes."""
    response.raise_for_status()
    expected_response_bytes = artifact_size_bytes - downloaded_bytes
    if downloaded_bytes == 0:
        if response.status_code != httpx.codes.OK:
            raise ValueError("initial model download did not return a complete response")
    else:
        if response.status_code != httpx.codes.PARTIAL_CONTENT:
            raise ValueError("model download server does not support safe range resumption")
        content_range = response.headers.get("Content-Range")
        expected_prefix = f"bytes {downloaded_bytes}-"
        expected_suffix = f"/{artifact_size_bytes}"
        if (
            content_range is None
            or not content_range.startswith(expected_prefix)
            or not content_range.endswith(expected_suffix)
        ):
            raise ValueError("resumed model download returned an invalid content range")
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return
    try:
        reported_size = int(content_length)
    except ValueError as error:
        raise ValueError("model download reported an invalid content length") from error
    if reported_size != expected_response_bytes:
        raise ValueError("model download size does not match the canonical artifact remainder")


def _free_disk_bytes(path: Path) -> int:
    """Return the bytes this user may still write to the filesystem holding one path."""
    return shutil.disk_usage(nearest_existing_ancestor(path)).free


def _require_free_disk_space(cache_root: Path, remaining_bytes: int) -> None:
    """Refuse a download that cannot fit in the cache with room left over."""
    required_bytes = remaining_bytes + MODEL_DOWNLOAD_FREE_SPACE_MARGIN_BYTES
    free_bytes = _free_disk_bytes(cache_root)
    if free_bytes < required_bytes:
        raise ValueError(f"model download needs {required_bytes - free_bytes} more free bytes in {cache_root}")


def _resumable_partial_bytes(partial_path: Path, artifact_size_bytes: int) -> int:
    """Return how many bytes of an earlier attempt can be resumed."""
    try:
        metadata = os.lstat(partial_path)
    except FileNotFoundError:
        return 0
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("partial model download must be a regular file, not a symbolic link")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("partial model download must be a regular file")
    if metadata.st_size <= artifact_size_bytes:
        return metadata.st_size
    partial_path.unlink()
    return 0


def _hash_partial_download(
    partial_path: Path,
    digest: Sha256Hash,
    cancellation_requested: CancellationCheck,
) -> int:
    """Fold the bytes already on disk into the running digest and return their count."""
    hashed_bytes = 0
    with partial_path.open("rb") as source:
        while chunk := source.read(MODEL_DOWNLOAD_CHUNK_BYTES):
            if cancellation_requested():
                raise InterruptedError("model artifact operation was cancelled")
            digest.update(chunk)
            hashed_bytes += len(chunk)
    return hashed_bytes


def _is_retryable_download_error(error: httpx.HTTPError) -> bool:
    """Report whether one transport or status failure is worth another attempt."""
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code == httpx.codes.TOO_MANY_REQUESTS or status_code >= httpx.codes.INTERNAL_SERVER_ERROR
    return isinstance(
        error,
        httpx.ReadTimeout | httpx.ConnectTimeout | httpx.ConnectError | httpx.ReadError | httpx.RemoteProtocolError,
    )


def _wait_before_retry(backoff_seconds: float, deadline: float, cancellation_requested: CancellationCheck) -> None:
    """Back off between attempts without outliving the deadline or a cancellation."""
    wake_time = min(time.monotonic() + backoff_seconds, deadline)
    while time.monotonic() < wake_time:
        if cancellation_requested():
            raise InterruptedError("model artifact operation was cancelled")
        time.sleep(MODEL_DOWNLOAD_BACKOFF_POLL_SECONDS)


def _report_progress(
    progress: ProgressReport | None,
    downloaded_bytes: int,
    total_bytes: int,
    reported_at: float,
) -> float:
    """Report download progress no more often than the reporting interval."""
    if progress is None:
        return reported_at
    now = time.monotonic()
    if now - reported_at < MODEL_DOWNLOAD_PROGRESS_INTERVAL_SECONDS:
        return reported_at
    progress(downloaded_bytes, total_bytes)
    return now


def _download_model_artifact(
    cache_root: Path,
    partial_path: Path,
    url: str,
    artifact: DeploymentArtifact,
    *,
    cancellation_requested: CancellationCheck,
    download_timeout_seconds: float,
    progress: ProgressReport | None,
) -> None:
    """Fill the partial file with the pinned artifact and verify what was written."""
    resumable_bytes = _resumable_partial_bytes(partial_path, artifact.size_bytes)
    _require_free_disk_space(cache_root, artifact.size_bytes - resumable_bytes)
    digest = hashlib.sha256()
    downloaded_bytes = 0
    if resumable_bytes > 0:
        downloaded_bytes = _hash_partial_download(partial_path, digest, cancellation_requested)
    deadline = time.monotonic() + download_timeout_seconds
    backoff_seconds = MODEL_DOWNLOAD_INITIAL_BACKOFF_SECONDS
    reported_at = time.monotonic()
    timeout = httpx.Timeout(
        connect=MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECONDS,
        read=MODEL_DOWNLOAD_READ_TIMEOUT_SECONDS,
        write=MODEL_DOWNLOAD_WRITE_TIMEOUT_SECONDS,
        pool=MODEL_DOWNLOAD_POOL_TIMEOUT_SECONDS,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | PRIVATE_OPEN_FLAGS
    descriptor = os.open(partial_path, flags, PRIVATE_FILE_PERMISSIONS)
    with os.fdopen(descriptor, "ab") as destination:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            while downloaded_bytes < artifact.size_bytes:
                if cancellation_requested():
                    raise InterruptedError("model artifact operation was cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("model download exceeded the total time limit")
                # Identity encoding keeps the stream byte-exact: iter_raw yields whatever
                # the server encoded, and a compressed response would be hashed as sent,
                # its Content-Length would describe the compressed size, and a Range
                # offset would address the compressed stream. Small text files in a
                # weights repository are exactly the ones a server chooses to compress.
                headers = {"Accept-Encoding": "identity"}
                if downloaded_bytes > 0:
                    headers["Range"] = f"bytes={downloaded_bytes}-"
                try:
                    with client.stream("GET", url, headers=headers) as response:
                        _validate_download_response(
                            response,
                            downloaded_bytes=downloaded_bytes,
                            artifact_size_bytes=artifact.size_bytes,
                        )
                        for chunk in response.iter_raw():
                            if cancellation_requested():
                                raise InterruptedError("model artifact operation was cancelled")
                            if time.monotonic() >= deadline:
                                raise TimeoutError("model download exceeded the total time limit")
                            if downloaded_bytes + len(chunk) > artifact.size_bytes:
                                raise _CorruptDownloadError("model download exceeded the canonical artifact size")
                            destination.write(chunk)
                            digest.update(chunk)
                            downloaded_bytes += len(chunk)
                            reported_at = _report_progress(
                                progress,
                                downloaded_bytes,
                                artifact.size_bytes,
                                reported_at,
                            )
                except httpx.HTTPError as error:
                    if cancellation_requested():
                        raise InterruptedError("model artifact operation was cancelled") from error
                    if not _is_retryable_download_error(error):
                        raise
                    destination.flush()
                    _wait_before_retry(backoff_seconds, deadline, cancellation_requested)
                    backoff_seconds = min(
                        backoff_seconds * MODEL_DOWNLOAD_BACKOFF_MULTIPLIER,
                        MODEL_DOWNLOAD_MAXIMUM_BACKOFF_SECONDS,
                    )
                    continue
                backoff_seconds = MODEL_DOWNLOAD_INITIAL_BACKOFF_SECONDS
        destination.flush()
        os.fsync(destination.fileno())
    if progress is not None:
        progress(downloaded_bytes, artifact.size_bytes)
    if digest.hexdigest() != artifact.sha256:
        raise _CorruptDownloadError(f"downloaded {artifact.filename} does not match its canonical digest")


def _publish_downloaded_artifact(
    partial_path: Path,
    blob_path: Path,
    artifact: DeploymentArtifact,
    *,
    cancellation_requested: CancellationCheck,
) -> VerifiedArtifact:
    """Move one verified download into its cache path across filesystems that reject hard links."""
    try:
        os.link(partial_path, blob_path)
    except FileExistsError:
        existing = verify_model_artifact(
            blob_path,
            artifact,
            cancellation_requested=cancellation_requested,
        )
        partial_path.unlink(missing_ok=True)
        return existing
    except OSError:
        os.replace(partial_path, blob_path)
    else:
        partial_path.unlink(missing_ok=True)
    metadata = blob_path.stat()
    return VerifiedArtifact(
        filename=artifact.filename,
        path=blob_path,
        sha256=f"sha256:{artifact.sha256}",
        size_bytes=artifact.size_bytes,
        device_id=metadata.st_dev,
        inode=metadata.st_ino,
        modified_ns=metadata.st_mtime_ns,
    )


def _resolve_hf_snapshot_file(snapshot_path: Path) -> Path:
    """Follow one cached snapshot entry to the regular file that holds its bytes.

    huggingface_hub stores a snapshot entry as a symlink into the blob store, or as a
    plain file on filesystems without symlinks; both resolve here. The resolved file
    still passes through full digest verification before it is served.
    """
    try:
        return snapshot_path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cached model artifact {snapshot_path} does not resolve to a file") from error


def _publish_snapshot_link(blob_path: Path, snapshot_path: Path) -> None:
    """Expose one verified blob at its pinned snapshot path, replacing any stale entry.

    The link is relative, exactly as huggingface_hub writes it, so the cache stays
    valid when the hub directory moves. Linking is best-effort: a filesystem without
    symlink support must not fail the run whose verified blob is already committed,
    because every later call still serves that blob through the content-addressed
    fast path. No refs entry is written: refs name branches and tags, and
    hf_hub_download skips them too when given a commit revision.
    """
    staging_path = snapshot_path.with_name(f"{snapshot_path.name}{MODEL_DOWNLOAD_INCOMPLETE_SUFFIX}")
    try:
        _prepare_cache_parent(snapshot_path.parent)
        staging_path.unlink(missing_ok=True)
        os.symlink(os.path.relpath(blob_path, snapshot_path.parent), staging_path)
        os.replace(staging_path, snapshot_path)
    except OSError:
        staging_path.unlink(missing_ok=True)


def _ensure_artifact(
    cache_root: Path,
    candidate: ModelCandidate,
    artifact: DeploymentArtifact,
    *,
    cancellation_requested: CancellationCheck,
    download_timeout_seconds: float,
    progress: ProgressReport | None,
) -> VerifiedArtifact:
    """Serve one pinned file from the shared Hugging Face cache, downloading it there if missing.

    One gate admits every reused file: its size and sha256 digest must match the
    canonical catalog. The pinned commit revision scopes only the snapshot lookup,
    so a same-named file cached under another commit is never consulted; the blob
    store is content-addressed by the pinned digest, so a matching blob is reused
    directly. A failed download keeps its incomplete
    file so the next call resumes it, unless the bytes on disk can never verify.
    """
    repository = _artifact_repository(candidate, artifact)
    revision = _artifact_revision(candidate, artifact)
    cached = try_to_load_from_cache(
        repo_id=repository,
        filename=artifact.filename,
        cache_dir=cache_root,
        revision=revision,
    )
    if isinstance(cached, str):
        return verify_model_artifact(
            _resolve_hf_snapshot_file(Path(cached)),
            artifact,
            cancellation_requested=cancellation_requested,
        )
    if cancellation_requested():
        raise InterruptedError("model artifact operation was cancelled")
    blob_path = _hf_blob_path(cache_root, candidate, artifact)
    snapshot_path = _hf_snapshot_path(cache_root, candidate, artifact)
    if blob_path.exists() or blob_path.is_symlink():
        # The blob is content-addressed, so a hit left by another tool or an earlier
        # run that lost its snapshot link only needs verification and a fresh link.
        verified = verify_model_artifact(blob_path, artifact, cancellation_requested=cancellation_requested)
        _publish_snapshot_link(blob_path, snapshot_path)
        return verified
    _prepare_cache_parent(blob_path.parent)
    # huggingface_hub's cache scanner reports a repository directory without snapshots/
    # as corrupted, so the directory must exist before a cancelled attempt can leave a
    # resumable incomplete file behind in blobs/.
    _prepare_cache_parent(_hf_repository_cache(cache_root, repository) / "snapshots")
    partial_path = blob_path.with_name(f"{blob_path.name}{MODEL_DOWNLOAD_INCOMPLETE_SUFFIX}")
    try:
        _download_model_artifact(
            cache_root,
            partial_path,
            _artifact_url(candidate, artifact),
            artifact,
            cancellation_requested=cancellation_requested,
            download_timeout_seconds=download_timeout_seconds,
            progress=progress,
        )
    except _CorruptDownloadError:
        partial_path.unlink(missing_ok=True)
        raise
    if cancellation_requested():
        raise InterruptedError("model artifact operation was cancelled")
    # The hub's .locks protocol is deliberately not taken here: huggingface_hub holds a
    # blob lock for its entire multi-gigabyte download, and waiting on it would need
    # cancellation-aware polling. The worst unlocked race with a concurrent hub download
    # of the same blob is one run failing its identity check and succeeding on retry.
    verified = _publish_downloaded_artifact(
        partial_path,
        blob_path,
        artifact,
        cancellation_requested=cancellation_requested,
    )
    _publish_snapshot_link(blob_path, snapshot_path)
    return verified


def _recipe_progress(progress: ProgressReport | None, completed_bytes: int, total_bytes: int) -> ProgressReport | None:
    """Report one file's progress as progress through the whole recipe."""
    if progress is None:
        return None

    def report(file_bytes: int, _file_total_bytes: int) -> None:
        progress(completed_bytes + file_bytes, total_bytes)

    return report


def _require_snapshot_directory(snapshot_root: Path, deployment: ModelDeployment) -> None:
    """Reject a snapshot directory that does not hold every pinned file by name.

    A weights runtime opens shards and tokenizer files by the names the index lists,
    so verifying the blobs is not enough: the names must resolve inside this directory.
    """
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise ValueError(f"model snapshot directory {snapshot_root} is missing")
    for artifact in deployment.artifacts:
        entry = snapshot_root / artifact.filename
        if not entry.exists():
            raise ValueError(f"model snapshot directory does not expose {artifact.filename}")


def _gguf_model_path(
    cache_root: Path,
    candidate: ModelCandidate,
    model_artifact: VerifiedArtifact,
    verified: tuple[VerifiedArtifact, ...],
) -> Path:
    """Return the path llama.cpp loads the target from.

    A single GGUF file is served from wherever its verified bytes sit. A split GGUF is
    served from the pinned snapshot directory instead, because llama.cpp finds the later
    parts by the first part's file name and a content-addressed blob has no such name.
    Every part's snapshot entry must be the same file as its verified blob.
    """
    # Catalog file names are repository paths, which use "/" on every platform.
    split = SPLIT_GGUF_FIRST_PART.match(PurePosixPath(model_artifact.filename).name)
    if split is None:
        return model_artifact.path
    snapshot_root = _hf_snapshot_root(cache_root, candidate.hf_repository, candidate.hf_revision)
    folder = PurePosixPath(model_artifact.filename).parent
    part_count = int(split.group("count"))
    for index in range(1, part_count + 1):
        filename = (folder / f"{split.group('prefix')}-{index:05d}-of-{split.group('count')}.gguf").as_posix()
        part = next((artifact for artifact in verified if artifact.filename == filename), None)
        if part is None:
            raise ValueError(f"split GGUF recipe does not pin part {filename}")
        entry = snapshot_root / filename
        if not entry.exists() or not os.path.samefile(entry, part.path):
            raise ValueError(f"model snapshot directory does not expose verified part {filename}")
    return snapshot_root / model_artifact.filename


def ensure_model_artifacts(
    cache_root: Path,
    candidate: ModelCandidate,
    *,
    cancellation_requested: CancellationCheck = _never_cancelled,
    download_timeout_seconds: float = MODEL_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
    progress: ProgressReport | None = None,
) -> VerifiedDeployment:
    """Verify every pinned file of one recipe and return the path its runtime is given.

    A GGUF recipe serves its verified target file; a split target is served by its
    snapshot name so llama.cpp can find the other parts. A weights
    repository is served from the pinned revision's snapshot directory, so every file
    the runtime opens by name sits beside the tensors under one path.
    """
    deployment = candidate.deployment
    total_bytes = deployment.artifact_size_bytes
    completed_bytes = 0
    verified: list[VerifiedArtifact] = []
    for artifact in deployment.artifacts:
        verified.append(
            _ensure_artifact(
                cache_root,
                candidate,
                artifact,
                cancellation_requested=cancellation_requested,
                download_timeout_seconds=download_timeout_seconds,
                progress=_recipe_progress(progress, completed_bytes, total_bytes),
            )
        )
        completed_bytes += artifact.size_bytes
    if deployment.artifact_kind in ("gguf-single-file", "gguf-file-set"):
        model_artifact = next(
            artifact for artifact in verified if artifact.filename == deployment.target_model_filename
        )
        draft_filename = deployment.llama_cpp.draft_model_filename if deployment.llama_cpp is not None else None
        draft_artifact = next((artifact for artifact in verified if artifact.filename == draft_filename), None)
        return VerifiedDeployment(
            model_path=_gguf_model_path(cache_root, candidate, model_artifact, tuple(verified)),
            artifacts=tuple(verified),
            draft_model_path=draft_artifact.path if draft_artifact is not None else None,
        )
    if deployment.artifact_kind == "gguf-target-safetensors-draft":
        return _splash_deployment(cache_root, candidate, tuple(verified))
    snapshot_root = _hf_snapshot_root(cache_root, candidate.hf_repository, candidate.hf_revision)
    _require_snapshot_directory(snapshot_root, deployment)
    return VerifiedDeployment(model_path=snapshot_root, artifacts=tuple(verified))


def _splash_deployment(
    cache_root: Path, candidate: ModelCandidate, verified: tuple[VerifiedArtifact, ...]
) -> VerifiedDeployment:
    """Return the snapshot paths Splash reads a GGUF target and its draft from.

    Splash opens the target by repository, variant, and revision from the hub cache,
    and the draft as a directory. Each snapshot entry must be the verified file itself.
    """
    deployment = candidate.deployment
    draft = deployment.draft_artifacts[0]
    draft_root = _hf_snapshot_root(
        cache_root, _artifact_repository(candidate, draft), _artifact_revision(candidate, draft)
    )
    target_name = deployment.target_model_filename
    target = next(artifact for artifact in deployment.artifacts if artifact.filename == target_name)
    target_path = _hf_snapshot_path(cache_root, candidate, target)
    for pinned, artifact in zip(deployment.artifacts, verified, strict=True):
        entry = _hf_snapshot_path(cache_root, candidate, pinned)
        if not entry.exists() or not os.path.samefile(entry, artifact.path):
            raise ValueError(f"model snapshot directory does not expose verified file {pinned.filename}")
    return VerifiedDeployment(
        model_path=target_path, artifacts=verified, draft_model_path=draft_root, hub_cache=cache_root
    )
