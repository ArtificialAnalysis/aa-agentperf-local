"""Hash bytes and validate the identifiers that name one measured run."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

HASH_CHUNK_BYTES = 1024 * 1024
SHA256_PREFIX = "sha256:"
SHA256_HEX_DIGITS = 64
MAX_IDENTIFIER_CHARACTERS = 128
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._+-]*[A-Za-z0-9])?$")
# A full git commit, and one shortened as git and version strings print it.
GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
SHORT_GIT_COMMIT_PATTERN = r"^[0-9a-f]{7,40}$"
_GIT_COMMIT = re.compile(GIT_COMMIT_PATTERN)


def validate_identifier(value: str, field: str) -> None:
    """Reject identifiers that could contain private paths or addresses."""
    if len(value) > MAX_IDENTIFIER_CHARACTERS or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must use letters, numbers, dots, underscores, plus signs, and hyphens")


def validate_digest(value: str, field: str) -> None:
    """Require one lowercase SHA-256 identifier."""
    prefix, separator, hexadecimal = value.partition(":")
    if separator != ":" or prefix != "sha256" or len(hexadecimal) != SHA256_HEX_DIGITS:
        raise ValueError(f"{field} must use sha256:<64 lowercase hex digits>")
    if hexadecimal != hexadecimal.lower() or any(character not in "0123456789abcdef" for character in hexadecimal):
        raise ValueError(f"{field} must use sha256:<64 lowercase hex digits>")


def is_git_commit(value: str) -> bool:
    """Return whether the value is one full, lowercase git commit."""
    return _GIT_COMMIT.fullmatch(value) is not None


def validate_run_id(value: str, field: str) -> None:
    """Require one lowercase, hyphenated, version-4 UUID."""
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a UUID") from error
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field} must be a lowercase hyphenated version-4 UUID")


def mint_run_id() -> str:
    """Mint one random run identifier."""
    return str(uuid.uuid4())


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 digest."""
    return f"{SHA256_PREFIX}{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return f"{SHA256_PREFIX}{digest.hexdigest()}"
