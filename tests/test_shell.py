"""Test how a recorded shell command names its executable."""

from __future__ import annotations

import pytest

from agentperf_local.tools.shell import shell_executable


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cd /workspace && rg -n needle file.txt", "rg"),
        ("echo 'one;still one'; cat file.txt", "cat"),
        ("python3 -c \"import yaml; print('ok')\"", "python3"),
        ("echo 'unterminated", None),
        (None, None),
        # A heredoc body is data. Punctuation in the text must not end the segment.
        ("cat > report.md << 'EOF'\nbudget; wrap up\nEOF", "cat"),
        ("python3 - <<'PY'\nprint(1); print(2)\nPY", "python3"),
        ("cat <<-EOF > f\n\tindented; body\n\tEOF", "cat"),
        ("cat << 'A' > one && cat << 'B' > two\nbody a; x\nA\nbody b; y\nB", "cat"),
        ("cat << 'EOF' > f\nbody; x\nEOF\ncd /t && rg -n needle", "rg"),
        # A <<< here-string has no body, so the segment continues on the same line.
        ('grep -c foo <<< "a; b"', "grep"),
        # An unterminated marker keeps its lines, so a quoted << cannot discard a command.
        ("cat << 'NEVER_CLOSED' > f\nbody; x", "x"),
    ],
)
def test_shell_executable_preserves_quoted_segments(command: str | None, expected: str | None) -> None:
    assert shell_executable(command) == expected
