"""Describe replay workloads shipped with the client."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from agentperf_local.common.package_paths import PACKAGE_DATA_ROOT

CUSTOM_REPLAY_ID = "custom-manifest"


@dataclass(frozen=True, slots=True, kw_only=True)
class BundledReplay:
    """Describe one replay workload included in the Python package."""

    replay_id: str
    label: str
    summary: str
    manifest_path: Path


_DATA_ROOT = PACKAGE_DATA_ROOT / "replays"


def _native_swebench_manifest(root: Path) -> Path:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return root / "manifest-arm64.json"
    return root / "manifest-x86_64.json"


# The full recorded replay comes first because it is the one results are compared on.
# The mini replay only proves an install works end to end in a few seconds.
BUNDLED_REPLAYS = (
    BundledReplay(
        replay_id="agentperf-default-v1",
        label="AgentPerf default replay v1",
        summary="8 tasks · 168 turns",
        manifest_path=_native_swebench_manifest(_DATA_ROOT / "agentperf-default-v1"),
    ),
    BundledReplay(
        replay_id="aa-mini-v1",
        label="AgentPerf-Local mini (quick check, not comparable)",
        summary="1 task · 6 turns · 8k context",
        manifest_path=_DATA_ROOT / "aa-mini-v1" / "manifest.json",
    ),
)
DEFAULT_BUNDLED_REPLAY = BUNDLED_REPLAYS[0]


def find_bundled_replay(replay_id: str) -> BundledReplay | None:
    """Return the bundled replay with the requested identifier."""
    return next((replay for replay in BUNDLED_REPLAYS if replay.replay_id == replay_id), None)
