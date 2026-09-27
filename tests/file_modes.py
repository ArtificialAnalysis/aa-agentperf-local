"""Check POSIX permission bits on platforms that keep them."""

from __future__ import annotations

import stat
import sys
from pathlib import Path


def has_mode(path: Path, mode: int) -> bool:
    """Report whether path has exactly this permission mode.

    Windows keeps only a read-only bit, so every mode passes there.
    """
    return sys.platform == "win32" or stat.S_IMODE(path.stat().st_mode) == mode


def lacks_mode_bits(path: Path, bits: int) -> bool:
    """Report whether path has none of these permission bits; always true on Windows."""
    return sys.platform == "win32" or stat.S_IMODE(path.stat().st_mode) & bits == 0
