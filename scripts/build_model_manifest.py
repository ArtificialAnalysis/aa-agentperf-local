"""Build the pinned artifact manifest for one catalog deployment recipe.

The catalog pins every file a managed deployment fetches by size and SHA-256.
The Hub reports the SHA-256 of a Git LFS file in its pointer, but a small plain
file only carries a Git blob SHA-1, so this tool downloads those files and
hashes them itself.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import orjson

from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json_object

HUB_ORIGIN = "https://huggingface.co"
PATHS_INFO_BATCH = 200
# The tree endpoint rejects a larger page; no catalog repository comes close to it.
TREE_PAGE_LIMIT = 1000
REQUEST_TIMEOUT_SECONDS = 60.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
# A plain file must be downloaded to be hashed, so the tool refuses to pull a
# large one; every large artifact on the Hub is stored in LFS and needs no download.
MAX_PLAIN_FILE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestEntry:
    """Store one pinned file of a managed deployment."""

    filename: str
    sha256: str
    size_bytes: int

    def to_json(self) -> JsonObject:
        """Return the catalog representation of one pinned file."""
        return {"filename": self.filename, "sha256": self.sha256, "size_bytes": self.size_bytes}


def _get(client: httpx.Client, url: str) -> JsonValue:
    response = client.get(url)
    response.raise_for_status()
    return orjson.loads(response.content)


def _repository_files(client: httpx.Client, repository: str, revision: str) -> tuple[str, ...]:
    """List every file of one repository revision, ignoring directories."""
    listing = _get(
        client, f"{HUB_ORIGIN}/api/models/{repository}/tree/{revision}?recursive=true&limit={TREE_PAGE_LIMIT}"
    )
    if not isinstance(listing, list):
        raise ValueError("repository tree response is not an array")
    if len(listing) == TREE_PAGE_LIMIT:
        raise ValueError("repository tree is larger than one page; this tool does not paginate")
    names: list[str] = []
    for raw_entry in listing:
        entry = normalize_json_object(raw_entry)
        if entry.get("type") != "file":
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("repository tree entry has no path")
        names.append(path)
    return tuple(names)


def _paths_info(client: httpx.Client, repository: str, revision: str, paths: Sequence[str]) -> tuple[JsonObject, ...]:
    """Read Hub metadata for the named paths, in batches the API accepts."""
    records: list[JsonObject] = []
    for start in range(0, len(paths), PATHS_INFO_BATCH):
        response = client.post(
            f"{HUB_ORIGIN}/api/models/{repository}/paths-info/{revision}",
            json={"paths": list(paths[start : start + PATHS_INFO_BATCH])},
        )
        response.raise_for_status()
        decoded = orjson.loads(response.content)
        if not isinstance(decoded, list):
            raise ValueError("paths-info response is not an array")
        records.extend(normalize_json_object(raw_record) for raw_record in decoded)
    return tuple(records)


def _plain_file_digest(client: httpx.Client, repository: str, revision: str, path: str) -> tuple[str, int]:
    """Download one plain file and return its digest and byte count."""
    digest = hashlib.sha256()
    size_bytes = 0
    with client.stream("GET", f"{HUB_ORIGIN}/{repository}/resolve/{revision}/{path}") as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(DOWNLOAD_CHUNK_BYTES):
            size_bytes += len(chunk)
            if size_bytes > MAX_PLAIN_FILE_BYTES:
                raise ValueError(f"{path} is too large to hash as a plain file")
            digest.update(chunk)
    return digest.hexdigest(), size_bytes


def manifest_entry(client: httpx.Client, repository: str, revision: str, record: JsonObject) -> ManifestEntry:
    """Turn one Hub path record into a pinned manifest entry."""
    path = record.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("paths-info record has no path")
    lfs = record.get("lfs")
    if isinstance(lfs, dict):
        oid = lfs.get("oid")
        size_bytes = lfs.get("size")
        if not isinstance(oid, str) or not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
            raise ValueError(f"{path} has an unusable LFS pointer")
        return ManifestEntry(filename=path, sha256=oid, size_bytes=size_bytes)
    sha256, size_bytes = _plain_file_digest(client, repository, revision, path)
    return ManifestEntry(filename=path, sha256=sha256, size_bytes=size_bytes)


def select_files(available: Sequence[str], include: Sequence[str]) -> tuple[str, ...]:
    """Return the requested files in catalog order, rejecting names the revision lacks."""
    known = frozenset(available)
    missing = sorted(name for name in include if name not in known)
    if missing:
        raise ValueError(f"revision does not contain {', '.join(missing)}")
    return tuple(sorted(set(include)))


def build_manifest(repository: str, revision: str, include: Sequence[str]) -> tuple[ManifestEntry, ...]:
    """Build the pinned manifest for the named files of one repository revision."""
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
        selected = select_files(_repository_files(client, repository, revision), include)
        records = _paths_info(client, repository, revision, selected)
        by_path = {record.get("path"): record for record in records}
        return tuple(manifest_entry(client, repository, revision, by_path[name]) for name in selected)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the pinned artifact manifest for one deployment recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="Hugging Face repository, as owner/name")
    parser.add_argument("--revision", required=True, help="40-character commit revision to pin")
    parser.add_argument("--include", required=True, action="append", help="repository file to pin; repeatable")
    parser.add_argument("--output", type=Path, default=None, help="write JSON here instead of standard output")
    namespace = parser.parse_args(argv)
    entries = build_manifest(namespace.repository, namespace.revision, namespace.include)
    payload: JsonObject = {
        "artifacts": [entry.to_json() for entry in entries],
        "artifact_total_bytes": sum(entry.size_bytes for entry in entries),
    }
    encoded = orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    if namespace.output is None:
        print(encoded.decode())
    else:
        namespace.output.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
