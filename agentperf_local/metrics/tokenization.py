"""Count output tokens after response close."""

from typing import Protocol

import tiktoken

DEFAULT_ENCODING = "o200k_base"


class TokenCounter(Protocol):
    """Count tokens in model output text."""

    def count(self, text: str) -> int:
        """Return the token count."""
        ...


class TiktokenTokenCounter:
    """Count tokens with one configured tiktoken encoding."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Return the encoded token count."""
        return len(self._encoding.encode(text, disallowed_special=()))
