"""Render a server launch command that a person can paste into a shell.

Public surface: LocalPath, render_launch_command, and redact_launch_command.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel

# A placeholder the command already holds, such as $MODEL_DIR/model.gguf, stays expandable.
PLACEHOLDER_PATTERN = re.compile(r"^\$([A-Z][A-Z0-9_]*)(.*)$", re.DOTALL)
ENVIRONMENT_ASSIGNMENT_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
# A flag or variable holds a secret when one word of its name is one of these.
SECRET_NAME_WORDS = frozenset(("key", "token", "secret", "password"))
NAME_WORD_SEPARATOR = re.compile(r"[-_]")
# Short flags whose name does not say it holds a secret: llama.cpp spells --hf-token as -hft.
SECRET_SHORT_FLAGS = {"-hft": "$HF_TOKEN"}
HOST_FLAGS = frozenset(("--host", "--hostname"))
LOOPBACK_OR_ANY_HOSTS = frozenset(("127.0.0.1", "localhost", "0.0.0.0", "::1", "::"))
HOST_PLACEHOLDER = "$HOST"
# An unknown local directory becomes $LOCAL_DIR_1, $LOCAL_DIR_2, and so on, in order of use.
LOCAL_DIRECTORY_PLACEHOLDER = "LOCAL_DIR"
HOME_PREFIX = "~"
# Token shapes that must never leave the machine, whichever flag or variable carries them.
TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")
# An absolute path that starts inside a word, such as a path in a JSON flag value.
EMBEDDED_PATH_PATTERN = re.compile(r"(?:^|[\s\"'=:,\[{(])(?:/[^\s\"'/]|~/|[A-Za-z]:[\\/])")
WINDOWS_BACKSLASH_PATH_PATTERN = re.compile(r"[A-Za-z]:\\")


class LocalPath(BaseModel, frozen=True):
    """Name one local path a command uses and the placeholder that replaces it."""

    path: str
    placeholder: str


def _is_secret_name(name: str) -> bool:
    """Return whether a flag or variable name says it holds a secret, such as --api-key or HF_TOKEN."""
    words = NAME_WORD_SEPARATOR.split(name.lstrip("-").lower())
    return any(word in SECRET_NAME_WORDS for word in words)


def _is_secret_flag(flag: str) -> bool:
    return flag.startswith("-") and (_is_secret_name(flag) or flag in SECRET_SHORT_FLAGS)


def _secret_placeholder(flag: str) -> str:
    """Name the placeholder for one secret flag's value: --api-key becomes $API_KEY."""
    return SECRET_SHORT_FLAGS.get(flag) or "$" + NAME_WORD_SEPARATOR.sub("_", flag.lstrip("-")).upper()


def _is_local_path(value: str) -> bool:
    return value.startswith(HOME_PREFIX) or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _path_parts(value: str) -> tuple[str, str]:
    """Split one absolute path into its directory and its final name."""
    path = PureWindowsPath(value) if PureWindowsPath(value).drive else PurePosixPath(value)
    return str(path.parent), path.name


def _shell_word(value: str) -> str:
    """Quote one word for a POSIX shell, leaving a leading placeholder expandable."""
    match = PLACEHOLDER_PATTERN.fullmatch(value)
    if match is None:
        return shlex.quote(value)
    name, rest = match.groups()
    return f'"${name}"' + (shlex.quote(rest) if rest else "")


def _require_nothing_private(value: str) -> None:
    """Refuse a value that still holds a token or a local path after redaction.

    The rules name secrets and paths by flag name and by position. A value they miss,
    such as a path inside a JSON flag, fails here rather than leave the machine. The
    message never repeats the value.
    """
    if TOKEN_PATTERN.search(value) is not None:
        raise ValueError("server_launch_command holds what looks like an access token; remove it")
    if EMBEDDED_PATH_PATTERN.search(value) is not None:
        raise ValueError(
            "server_launch_command holds a local path inside an argument; "
            "replace it with a placeholder such as $MODEL_DIR"
        )


def _host_placeholders(arguments: tuple[str, ...]) -> tuple[str, ...]:
    """Replace the value after --host with $HOST unless it names this machine."""
    replaced = list(arguments)
    for index, argument in enumerate(arguments[:-1]):
        if argument in HOST_FLAGS and arguments[index + 1] not in LOOPBACK_OR_ANY_HOSTS:
            replaced[index + 1] = HOST_PLACEHOLDER
    return tuple(replaced)


@dataclass(slots=True)
class _Redactor:
    """Replace local paths in one command, numbering each unknown directory once."""

    # Longest first, so a file inside a known directory keeps its own placeholder.
    known_paths: tuple[LocalPath, ...]
    directories: dict[str, str] = field(default_factory=dict)

    def _known(self, value: str) -> str | None:
        """Return the placeholder for a known path, or for a file inside a known directory."""
        for known in self.known_paths:
            if value == known.path:
                return known.placeholder
            for separator in ("/", "\\"):
                if value.startswith(known.path + separator):
                    return known.placeholder + "/" + value.removeprefix(known.path + separator).replace("\\", "/")
        return None

    def _path(self, value: str) -> str:
        """Return one value with a known or unknown local path replaced by a placeholder."""
        known = self._known(value)
        if known is not None:
            return known
        if not _is_local_path(value):
            return value
        directory, name = _path_parts(value)
        placeholder = self.directories.setdefault(
            directory, f"${LOCAL_DIRECTORY_PLACEHOLDER}_{len(self.directories) + 1}"
        )
        return f"{placeholder}/{name}" if name else placeholder

    def executable(self, value: str) -> str:
        """Return the command name: a known placeholder, or else the executable's own file name."""
        known = self._known(value)
        if known is not None:
            return known
        return _path_parts(value)[1] if _is_local_path(value) else value

    def value_word(self, value: str) -> str:
        """Return one value as a shell word; a secret NAME=value, as in `-e HF_TOKEN=...`, keeps only its name."""
        assignment = ENVIRONMENT_ASSIGNMENT_PATTERN.fullmatch(value)
        if assignment is not None and _is_secret_name(assignment.group(1)):
            name = assignment.group(1)
            return f"{name}={_shell_word('$' + name)}"
        if assignment is not None:
            name, assigned = assignment.groups()
            redacted = self._path(assigned)
            _require_nothing_private(redacted)
            return f"{name}={_shell_word(redacted)}"
        redacted = self._path(value)
        _require_nothing_private(redacted)
        return _shell_word(redacted)

    def argument_words(self, arguments: tuple[str, ...]) -> list[str]:
        """Return the arguments as shell words, with local paths, secrets, and hostnames replaced."""
        words: list[str] = []
        pending_secret: str | None = None
        for argument in arguments:
            if pending_secret is not None:
                words.append(_shell_word(pending_secret))
                pending_secret = None
                continue
            flag, separator, value = argument.partition("=")
            if _is_secret_flag(flag) and separator:
                words.append(shlex.quote(f"{flag}=") + _shell_word(_secret_placeholder(flag)))
            elif _is_secret_flag(flag):
                words.append(shlex.quote(flag))
                pending_secret = _secret_placeholder(flag)
            elif flag in HOST_FLAGS and separator and value not in LOOPBACK_OR_ANY_HOSTS:
                words.append(shlex.quote(f"{flag}=") + _shell_word(HOST_PLACEHOLDER))
            elif argument.startswith("-") and separator:
                words.append(shlex.quote(f"{flag}=") + self.value_word(value))
            else:
                words.append(self.value_word(argument))
        return words


def render_launch_command(
    argv: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
    known_paths: tuple[LocalPath, ...] = (),
) -> str:
    """Return argv, after the environment it adds, as one shell-quoted line with no local path or secret.

    A local path becomes a named placeholder, such as "$MODEL_DIR"/model.gguf. A secret
    value becomes a placeholder named after its flag or variable, such as --api-key
    "$API_KEY". A secret variable set before the command is dropped, and a --host that
    is not this machine becomes "$HOST". Every other flag and value stays as it was run.
    """
    if not argv:
        raise ValueError("server_launch_command must name the server executable")
    redactor = _Redactor(tuple(sorted(known_paths, key=lambda known: len(known.path), reverse=True)))
    assignments = [f"{name}={redactor.value_word(value)}" for name, value in environment if not _is_secret_name(name)]
    words = [_shell_word(redactor.executable(argv[0])), *redactor.argument_words(_host_placeholders(argv[1:]))]
    return " ".join((*assignments, *words))


def redact_launch_command(command: str) -> str:
    """Return one command a person typed, split as a POSIX shell would, with paths and secrets removed.

    POSIX splitting drops backslashes, so a Windows path must use forward slashes.
    """
    if WINDOWS_BACKSLASH_PATH_PATTERN.search(command) is not None:
        raise ValueError("server_launch_command must write Windows paths with forward slashes, such as C:/models")
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise ValueError(f"server_launch_command is not a valid shell command: {error}") from error
    environment: list[tuple[str, str]] = []
    while tokens and (match := ENVIRONMENT_ASSIGNMENT_PATTERN.fullmatch(tokens[0])) is not None:
        environment.append((match.group(1), match.group(2)))
        tokens = tokens[1:]
    return render_launch_command(tuple(tokens), tuple(environment))
