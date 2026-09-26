"""Create durable benchmark files and read bounded private inputs."""

from __future__ import annotations

import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from agentperf_local.common.identity import sha256_bytes

PRIVATE_FILE_PERMISSIONS = 0o600
PUBLIC_FILE_PERMISSIONS = 0o644
PUBLIC_DIRECTORY_PERMISSIONS = 0o755
COMMIT_MARKER_TEMP_NONCE_BYTES = 8


@dataclass(frozen=True, slots=True, kw_only=True)
class NewFile:
    """Hold one final path and its complete encoded bytes."""

    path: Path
    data: bytes

    def __post_init__(self) -> None:
        """Require a concrete file name."""
        if self.path.name in {"", ".", ".."}:
            raise ValueError("new file path must name a file")


def _exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def validate_new_file_paths(paths: tuple[Path, ...]) -> None:
    """Reject duplicate, existing, or symlink-parent output paths."""
    if not paths:
        raise ValueError("at least one new file path is required")
    if len(frozenset(paths)) != len(paths):
        raise ValueError("new file paths must be unique")
    parent = paths[0].parent
    if any(path.parent != parent for path in paths):
        raise ValueError("new files must share one parent directory")
    if _exists_without_following(parent):
        metadata = parent.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("new file parent must be a directory, not a symbolic link")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("new file parent must be a directory")
    existing = tuple(path.name for path in paths if _exists_without_following(path))
    if existing:
        raise FileExistsError(
            f"new files must not replace existing paths: {', '.join(existing)}; "
            "remove them or choose a new output location"
        )


def write_all(descriptor: int, data: bytes) -> None:
    """Write every byte to an open descriptor, refusing a stalled write."""
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written < 1:
            raise OSError("file write made no progress")
        offset += written


def _write_new_file(file: NewFile, permissions: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(file.path, flags, permissions)
    try:
        write_all(descriptor, file.data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_marker_temp_path(path: Path) -> Path:
    """Return one hidden same-directory staging name for a commit marker."""
    return path.with_name(f".{path.name}.{secrets.token_hex(COMMIT_MARKER_TEMP_NONCE_BYTES)}.tmp")


def _write_commit_marker(file: NewFile, permissions: int) -> None:
    """Publish a fully written commit marker with one atomic rename."""
    # A marker written in place can survive a crash half-written, which both lies and blocks the directory.
    temp_path = _commit_marker_temp_path(file.path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(temp_path, flags, PRIVATE_FILE_PERMISSIONS)
    try:
        try:
            write_all(descriptor, file.data)
            os.fchmod(descriptor, permissions)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(temp_path, file.path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def fsync_directory(path: Path) -> None:
    """Sync a directory so its entries survive a crash."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_new_directory(path: Path, *, permissions: int = PUBLIC_DIRECTORY_PERMISSIONS) -> None:
    """Create one new directory and sync its parent entry."""
    if path.name in {"", ".", ".."}:
        raise ValueError("new directory path must name a directory")
    if _exists_without_following(path):
        raise FileExistsError(f"new directory must not replace an existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_metadata = path.parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("new directory parent must be a directory, not a symbolic link")
    os.mkdir(path, permissions)
    fsync_directory(path.parent)


def write_new_file(file: NewFile, *, permissions: int = PRIVATE_FILE_PERMISSIONS) -> None:
    """Write and sync one new file and its directory entry."""
    validate_new_file_paths((file.path,))
    file.path.parent.mkdir(parents=True, exist_ok=True)
    validate_new_file_paths((file.path,))
    _write_new_file(file, permissions)
    fsync_directory(file.path.parent)


def commit_new_file_set(
    data_files: tuple[NewFile, ...],
    commit_file: NewFile,
    *,
    permissions: int = PRIVATE_FILE_PERMISSIONS,
) -> None:
    """Sync data files before renaming one final commit marker into place."""
    if not data_files:
        raise ValueError("a committed file set requires at least one data file")
    files = (*data_files, commit_file)
    validate_new_file_paths(tuple(file.path for file in files))
    commit_file.path.parent.mkdir(parents=True, exist_ok=True)
    validate_new_file_paths(tuple(file.path for file in files))
    for file in data_files:
        _write_new_file(file, permissions)
    fsync_directory(commit_file.path.parent)
    _write_commit_marker(commit_file, permissions)
    fsync_directory(commit_file.path.parent)


def _require_outside(private_root: Path, output_path: Path, label: str) -> None:
    if output_path.resolve(strict=False).is_relative_to(private_root.resolve()):
        raise ValueError(f"{label} must be outside the private results directory")


def validate_public_output_directory(private_root: Path, output_dir: Path, label: str) -> None:
    """Reject a new public directory inside the private results tree or over an existing path."""
    _require_outside(private_root, output_dir, label)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"{label} already exists: {output_dir}")


@dataclass(frozen=True, slots=True, kw_only=True)
class WrittenFile:
    """Describe one exact file written to disk."""

    path: Path
    byte_size: int
    file_digest: str


def write_digest_file(path: Path, encoded: bytes, *, permissions: int = PRIVATE_FILE_PERMISSIONS) -> WrittenFile:
    """Write new bytes without replacing a file, and describe them by size and digest."""
    write_new_file(NewFile(path=path, data=encoded), permissions=permissions)
    return WrittenFile(path=path, byte_size=len(encoded), file_digest=sha256_bytes(encoded))


def nearest_existing_ancestor(path: Path) -> Path:
    """Return the closest existing ancestor a new path would be created under.

    A symbolic-link ancestor is rejected so free-space and writability checks
    describe the real filesystem location.
    """
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ValueError(f"{path} has no existing filesystem ancestor")
        candidate = parent
    if candidate.is_symlink():
        raise ValueError("filesystem ancestor must not be a symbolic link")
    return candidate


def read_bounded_file(path: Path, maximum_bytes: int, *, label: str = "file") -> bytes:
    """Read one regular file of at most maximum_bytes, refusing symlinks and size changes underfoot."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path.name}")
    byte_size = path.stat().st_size
    if byte_size <= 0 or byte_size > maximum_bytes:
        raise ValueError(f"{label} size is outside the accepted range: {path.name}")
    with path.open("rb") as source:
        encoded = source.read(maximum_bytes + 1)
    if not encoded or len(encoded) != byte_size:
        raise ValueError(f"{label} size changed while it was read: {path.name}")
    return encoded
