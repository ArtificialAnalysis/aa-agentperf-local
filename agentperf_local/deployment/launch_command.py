"""Render a server launch command that a person can paste into a shell.

Public surface: LocalPath, render_launch_command, and redact_launch_command.

The rendered command holds no local path and no secret. A local path becomes a named
placeholder, such as "$MODEL_DIR"/model.gguf. A secret value becomes a placeholder named
after its flag, such as --api-key "$API_KEY". A secret environment variable is dropped.
Every other flag and value stays as it was run.
"""

from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel

# A placeholder the command already holds, such as $MODEL_DIR/model.gguf, stays expandable.
PLACEHOLDER_PATTERN = re.compile(r"^\$([A-Z][A-Z0-9_]*)(.*)$", re.DOTALL)
ENVIRONMENT_ASSIGNMENT_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
# A flag or variable holds a secret when one word of its name is one of these.
SECRET_NAME_WORDS = frozenset(("key", "token", "secret", "password"))
NAME_WORD_SEPARATOR = re.compile(r"[-_]")
# An unknown local directory becomes $LOCAL_DIR_1, $LOCAL_DIR_2, and so on, in order of use.
LOCAL_DIRECTORY_PLACEHOLDER = "LOCAL_DIR"
HOME_PREFIX = "~"


class LocalPath(BaseModel, frozen=True):
    """Name one local path a command uses and the placeholder that replaces it."""

    path: str
    placeholder: str


def _is_secret_name(name: str) -> bool:
    """Return whether a flag or variable name says it holds a secret, such as --api-key or HF_TOKEN."""
    words = NAME_WORD_SEPARATOR.split(name.lstrip("-").lower())
    return any(word in SECRET_NAME_WORDS for word in words)


def _secret_placeholder(flag: str) -> str:
    """Name the placeholder for one secret flag's value: --api-key becomes $API_KEY."""
    return "$" + NAME_WORD_SEPARATOR.sub("_", flag.lstrip("-")).upper()


def _is_local_path(value: str) -> bool:
    return value.startswith(HOME_PREFIX) or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _path_parts(value: str) -> tuple[str, str]:
    """Split one absolute path into its directory and its final name."""
    path = PureWindowsPath(value) if PureWindowsPath(value).drive else PurePosixPath(value)
    return str(path.parent), path.name


class _Redactor:
    """Replace local paths in one command, numbering each unknown directory once."""

    def __init__(self, known_paths: tuple[LocalPath, ...]) -> None:
        # The longest path wins, so a file inside a known directory keeps its own name.
        self._known_paths = sorted(known_paths, key=lambda known: len(known.path), reverse=True)
        self._directories: dict[str, str] = {}

    def _known(self, value: str) -> str | None:
        """Return the placeholder for a known path, or for a file inside a known directory."""
        for known in self._known_paths:
            if value == known.path:
                return known.placeholder
            for separator in ("/", "\\"):
                if value.startswith(known.path + separator):
                    return known.placeholder + "/" + value.removeprefix(known.path + separator).replace("\\", "/")
        return None

    def path(self, value: str) -> str:
        """Return one value with a known or unknown local path replaced by a placeholder."""
        known = self._known(value)
        if known is not None:
            return known
        if not _is_local_path(value):
            return value
        directory, name = _path_parts(value)
        placeholder = self._directories.setdefault(
            directory, f"${LOCAL_DIRECTORY_PLACEHOLDER}_{len(self._directories) + 1}"
        )
        return f"{placeholder}/{name}" if name else placeholder

    def executable(self, value: str) -> str:
        """Return the command name: a known placeholder, or else the executable's own file name."""
        known = self._known(value)
        if known is not None:
            return known
        return _path_parts(value)[1] if _is_local_path(value) else value

    def argument_words(self, arguments: tuple[str, ...]) -> list[str]:
        """Return the arguments as shell words, with local paths replaced and secret values named by their flag."""
        words: list[str] = []
        pending_secret: str | None = None
        for argument in arguments:
            if pending_secret is not None:
                words.append(_shell_word(pending_secret))
                pending_secret = None
                continue
            flag, separator, value = argument.partition("=")
            is_flag = argument.startswith("-")
            if is_flag and _is_secret_name(flag):
                if separator:
                    words.append(shlex.quote(f"{flag}=") + _shell_word(_secret_placeholder(flag)))
                else:
                    words.append(shlex.quote(argument))
                    pending_secret = _secret_placeholder(flag)
            elif is_flag and separator:
                words.append(shlex.quote(f"{flag}=") + _shell_word(self.path(value)))
            else:
                words.append(_shell_word(self.path(argument)))
        return words


def _shell_word(value: str) -> str:
    """Quote one word for a POSIX shell, leaving a leading placeholder expandable."""
    match = PLACEHOLDER_PATTERN.fullmatch(value)
    if match is None:
        return shlex.quote(value)
    name, rest = match.groups()
    return f'"${name}"' + (shlex.quote(rest) if rest else "")


def render_launch_command(
    argv: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
    known_paths: tuple[LocalPath, ...] = (),
) -> str:
    """Return argv, after the environment it adds, as one shell-quoted, path-free, secret-free line."""
    if not argv:
        raise ValueError("a launch command needs an executable")
    redactor = _Redactor(known_paths)
    assignments = [
        f"{name}={_shell_word(redactor.path(value))}" for name, value in environment if not _is_secret_name(name)
    ]
    words = [_shell_word(redactor.executable(argv[0])), *redactor.argument_words(argv[1:])]
    return " ".join((*assignments, *words))


def redact_launch_command(command: str) -> str:
    """Return one command a person typed, split as a POSIX shell would, with paths and secrets removed."""
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise ValueError(f"server_launch_command is not a valid shell command: {error}") from error
    environment: list[tuple[str, str]] = []
    while tokens and (match := ENVIRONMENT_ASSIGNMENT_PATTERN.fullmatch(tokens[0])) is not None:
        environment.append((match.group(1), match.group(2)))
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("server_launch_command must name the server executable")
    return render_launch_command(tuple(tokens), tuple(environment))
