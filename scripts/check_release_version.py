"""Check that a release tag matches every version string in the repository."""

import argparse
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TAG_PREFIX = "v"


class VersionSite(BaseModel, frozen=True):
    """Name one file and the exact line it must hold for a version."""

    path: str
    line_template: str

    def expected_line(self, version: str) -> str:
        """Return the line this file must contain for the given version.

        `{series}` is the version's major and minor parts with a trailing dot, such as "0.2.".
        """
        major, minor, _ = version.split(".", 2)
        return self.line_template.format(version=version, series=f"{major}.{minor}.")


# pyproject.toml is the source of truth; every other copy must repeat its version exactly.
VERSION_SITES = (
    VersionSite(path="pyproject.toml", line_template='rust = ["agentperf-local-rustcore=={version}"]'),
    VersionSite(path="agentperf_local/__init__.py", line_template='__version__ = "{version}"'),
    VersionSite(
        path="agentperf_local/client/endpoint.py", line_template='MEASURED_USER_AGENT = "agentperf-local/{version}"'
    ),
    VersionSite(path="rustcore/pyproject.toml", line_template='version = "{version}"'),
    VersionSite(path="rustcore/Cargo.toml", line_template='version = "{version}"'),
    VersionSite(
        path="rustcore/src/engine.rs", line_template='const MEASURED_USER_AGENT: &str = "agentperf-local/{version}";'
    ),
    # The Python client accepts any Rust extension from the same major and minor series.
    VersionSite(
        path="agentperf_local/client/rust_client.py", line_template='RUSTCORE_COMPATIBILITY_PREFIX = "{series}"'
    ),
)


def project_version(root: Path) -> str:
    """Read the package version from pyproject.toml."""
    with (root / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def version_errors(tag: str, root: Path) -> list[str]:
    """List every way the tag and the version strings disagree."""
    version = project_version(root)
    errors: list[str] = []
    if tag != f"{TAG_PREFIX}{version}":
        errors.append(f"tag {tag!r} does not match pyproject.toml version {version!r}")
    for site in VERSION_SITES:
        expected = site.expected_line(version)
        lines = (root / site.path).read_text(encoding="utf-8").splitlines()
        if expected not in (line.strip() for line in lines):
            errors.append(f"{site.path} does not contain {expected!r}")
    return errors


def main() -> int:
    """Print each mismatch and return 1, or return 0 when the tag and versions agree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="release tag, such as v0.1.0")
    arguments = parser.parse_args()
    errors = version_errors(arguments.tag, REPOSITORY_ROOT)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
