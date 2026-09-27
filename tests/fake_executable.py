"""Write a fake command-line tool, as Python source, that subprocess can launch on any platform."""

from __future__ import annotations

import sys
from pathlib import Path


def write_python_executable(path: Path, source: str) -> Path:
    """Write source as a runnable command and return the path to launch.

    POSIX runs the script through its shebang line. Windows ignores shebangs, so it gets
    a .cmd launcher beside the script, which PATH lookups find through PATHEXT.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        script = path.with_suffix(".py")
        script.write_text(source, encoding="utf-8")
        launcher = path.with_suffix(".cmd")
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return launcher
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o700)
    return path
