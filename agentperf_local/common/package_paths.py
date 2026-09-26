"""Locate the installed package and its bundled data files."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DATA_ROOT = PACKAGE_ROOT / "data"
