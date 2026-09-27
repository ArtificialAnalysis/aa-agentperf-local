"""Exercise durable file creation and its crash-visible commit marker."""

import os
from pathlib import Path

import pytest

from agentperf_local.common import durable_files
from agentperf_local.common.durable_files import (
    PRIVATE_FILE_PERMISSIONS,
    PUBLIC_FILE_PERMISSIONS,
    NewFile,
    commit_new_file_set,
    validate_new_file_paths,
)
from tests.file_modes import has_mode

DATA_PAYLOAD = b'{"kind": "turns"}\n'
MARKER_PAYLOAD = b'{"kind": "run_summary"}\n'


def _file_set(directory: Path) -> tuple[NewFile, NewFile]:
    return (
        NewFile(path=directory / "turns.jsonl", data=DATA_PAYLOAD),
        NewFile(path=directory / "summary.json", data=MARKER_PAYLOAD),
    )


def _names(directory: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in directory.iterdir()))


@pytest.mark.parametrize("permissions", (PRIVATE_FILE_PERMISSIONS, PUBLIC_FILE_PERMISSIONS))
def test_commit_marker_only_appears_as_one_complete_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    permissions: int,
) -> None:
    data_file, marker_file = _file_set(tmp_path)
    renames: list[tuple[Path, bytes]] = []
    real_rename = os.rename

    def record_rename(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        renames.append((Path(target), Path(source).read_bytes()))
        real_rename(source, target)

    monkeypatch.setattr(durable_files.os, "rename", record_rename)

    commit_new_file_set((data_file,), marker_file, permissions=permissions)

    assert renames == [(marker_file.path, MARKER_PAYLOAD)]
    assert _names(tmp_path) == ("summary.json", "turns.jsonl")
    assert marker_file.path.read_bytes() == MARKER_PAYLOAD
    assert has_mode(marker_file.path, permissions)
    assert has_mode(data_file.path, permissions)


def test_a_failed_marker_write_leaves_no_marker_and_no_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_file, marker_file = _file_set(tmp_path)
    real_write_all = durable_files.write_all

    def fail_on_marker(descriptor: int, data: bytes) -> None:
        if data == MARKER_PAYLOAD:
            raise OSError("simulated power loss")
        real_write_all(descriptor, data)

    monkeypatch.setattr(durable_files, "write_all", fail_on_marker)

    with pytest.raises(OSError, match="simulated power loss"):
        commit_new_file_set((data_file,), marker_file)

    assert _names(tmp_path) == ("turns.jsonl",)


def test_validation_names_the_real_reason_a_new_file_set_is_refused(tmp_path: Path) -> None:
    plain_parent = tmp_path / "plain"
    plain_parent.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="^new file parent must be a directory$"):
        validate_new_file_paths((plain_parent / "summary.json",))

    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(tmp_path / "results", target_is_directory=True)
    with pytest.raises(ValueError, match="new file parent must be a directory, not a symbolic link"):
        validate_new_file_paths((linked_parent / "summary.json",))

    leftover = tmp_path / "summary.json"
    leftover.write_bytes(MARKER_PAYLOAD)
    with pytest.raises(FileExistsError) as refusal:
        validate_new_file_paths((leftover,))
    assert str(refusal.value) == (
        "new files must not replace existing paths: summary.json; remove them or choose a new output location"
    )
