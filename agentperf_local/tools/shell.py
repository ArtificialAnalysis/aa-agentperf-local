"""Name the executable a recorded shell command runs."""

from __future__ import annotations

import re
import shlex

# `<<` or `<<-` followed by the delimiter word, quoted or bare.
_HEREDOC_MARKER = re.compile(r"<<-?[ \t]*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_]\w*))")


def _heredoc_body_end(lines: list[str], start: int, delimiter: str) -> int | None:
    """Return the index of the line closing a heredoc, or None when it never closes."""
    for index in range(start, len(lines)):
        # A <<- heredoc allows the closing delimiter to be indented with tabs.
        if lines[index].strip() == delimiter:
            return index
    return None


def _without_heredoc_bodies(command: str) -> str:
    """Drop heredoc bodies so document text is never lexed as shell tokens.

    A heredoc body is data, not shell. Left in place, a `;` or `&` anywhere in the
    text ends the segment and the executable is read from the document instead of
    the command. An unterminated marker keeps its lines, so a quoted `<<` that is
    not a heredoc cannot discard the rest of the command.
    """
    lines = command.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        index += 1
        for match in _HEREDOC_MARKER.finditer(line):
            if line[match.start() - 1 : match.start()] == "<":
                continue  # A <<< here-string carries no body.
            delimiter = next(group for group in match.groups() if group is not None)
            end = _heredoc_body_end(lines, index, delimiter)
            if end is not None:
                index = end + 1
    return "\n".join(kept)


def shell_executable(command: str | None) -> str | None:
    """Return the executable in the final quoted shell segment."""
    if not command:
        return None
    try:
        lexer = shlex.shlex(_without_heredoc_bodies(command), posix=True, punctuation_chars=";&")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None

    segment: list[str] = []
    for token in tokens:
        if token in {";", "&&"}:
            segment = []
            continue
        segment.append(token)
    parts = segment or tokens
    return parts[0] if parts else None
