"""Find the full git commit of the framework build that served a run.

Public surface: FrameworkRef, RefKind, GITHUB_API_URL_ENV, framework_ref, commit_ref, and
resolve_framework_commit.
"""

from __future__ import annotations

import os
import re
from typing import Literal

import httpx
from pydantic import BaseModel

from agentperf_local import __version__
from agentperf_local.common.json_fields import decode_json_object, lenient_string
from agentperf_local.submission.contract import GIT_COMMIT_PATTERN, Framework

GITHUB_API_BASE_URL = "https://api.github.com"
# Points the lookup at another GitHub API, such as a mirror or a local test server.
GITHUB_API_URL_ENV = "AGENTPERF_GITHUB_API_URL"
GITHUB_TIMEOUT_SECONDS = 20.0
GITHUB_ACCEPT = "application/vnd.github+json"
FULL_COMMIT = re.compile(GIT_COMMIT_PATTERN)
# llama.cpp prints "version: 0.3.0 (build 10621, commit c1d0e7a00)" or "version: 6890 (c1d0e7a00)".
# A development build appends "+g<commit>", as in 0.1.dev20073+g8e685d198.
SHORT_COMMIT_PATTERNS = (
    re.compile(r"\bcommit ([0-9a-f]{7,40})\b"),
    re.compile(r"\(([0-9a-f]{7,40})\)"),
    re.compile(r"\+g([0-9a-f]{7,40})\b"),
)
# A release is x.y.z, x.y.z.w, or x.y.z.postN, and its tag spells the whole version. A version
# with any other suffix, such as 0.5.3rc1, names no tag and is refused rather than shortened.
RELEASE_VERSION = re.compile(r"(?<![\w.])(\d+\.\d+\.\d+(?:\.post\d+|\.\d+)?)(?![\w.])")
# A local build label, such as +cu128, is not part of the release tag.
LOCAL_VERSION_LABEL = re.compile(r"\+[0-9A-Za-z.]+")

type RefKind = Literal["commit", "tag"]


class _FrameworkSource(BaseModel, frozen=True):
    """Name the GitHub repository of one framework and how its release tags are spelled."""

    repository: str
    tag_prefix: str | None


# Tags: vLLM and SGLang write v0.11.0, Splash writes 1.0.2. llama.cpp tags builds, not
# releases, and its version text always names the commit.
_FRAMEWORK_SOURCES: dict[Framework, _FrameworkSource] = {
    "llama-cpp": _FrameworkSource(repository="ggml-org/llama.cpp", tag_prefix=None),
    "vllm": _FrameworkSource(repository="vllm-project/vllm", tag_prefix="v"),
    "sglang": _FrameworkSource(repository="sgl-project/sglang", tag_prefix="v"),
    "splash": _FrameworkSource(repository="incoai/splash", tag_prefix=""),
}


class FrameworkRef(BaseModel, frozen=True):
    """Name the git reference one framework version points at."""

    repository: str
    kind: RefKind
    value: str


def framework_ref(framework: Framework, version: str) -> FrameworkRef:
    """Read the commit or release tag that one framework's version text names.

    llama.cpp and development builds name a short commit. A release names a tag, and
    GitHub turns either into the full commit.
    """
    source = _FRAMEWORK_SOURCES[framework]
    for pattern in SHORT_COMMIT_PATTERNS:
        match = pattern.search(version)
        if match is not None:
            return FrameworkRef(repository=source.repository, kind="commit", value=match.group(1))
    release = RELEASE_VERSION.search(LOCAL_VERSION_LABEL.sub("", version))
    if release is None or source.tag_prefix is None:
        raise ValueError(f"the {framework} version {version!r} names no commit or release to look up")
    return FrameworkRef(repository=source.repository, kind="tag", value=f"{source.tag_prefix}{release.group(1)}")


def commit_ref(framework: Framework, commit: str) -> FrameworkRef:
    """Name one full or short commit in a framework's repository."""
    return FrameworkRef(repository=_FRAMEWORK_SOURCES[framework].repository, kind="commit", value=commit)


def resolve_framework_commit(
    ref: FrameworkRef,
    *,
    timeout_seconds: float = GITHUB_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Ask GitHub for the full commit a short commit or a tag names."""
    if ref.kind == "commit" and FULL_COMMIT.fullmatch(ref.value) is not None:
        return ref.value
    base_url = os.environ.get(GITHUB_API_URL_ENV) or GITHUB_API_BASE_URL
    url = f"{base_url.rstrip('/')}/repos/{ref.repository}/commits/{ref.value}"
    headers = {"Accept": GITHUB_ACCEPT, "User-Agent": f"agentperf-local/{__version__}"}
    try:
        with httpx.Client(timeout=timeout_seconds, transport=transport, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as error:
        raise ValueError(f"GitHub could not be reached to look up {ref.repository} {ref.value}") from error
    if response.status_code != httpx.codes.OK:
        raise ValueError(f"GitHub answered {response.status_code} for {ref.repository} {ref.kind} {ref.value}")
    sha = lenient_string(decode_json_object(response.content, "GitHub answered without a JSON object"), "sha")
    if sha is None or FULL_COMMIT.fullmatch(sha) is None:
        raise ValueError(f"GitHub answered without a full commit for {ref.repository} {ref.value}")
    if ref.kind == "commit" and not sha.startswith(ref.value):
        raise ValueError(f"GitHub resolved {ref.repository} commit {ref.value} to {sha}, which does not match")
    return sha
