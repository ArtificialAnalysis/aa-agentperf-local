"""Provide the deterministic token counter shared by streaming tests."""


class CharacterCounter:
    """Count characters so expected token totals stay transparent."""

    def count(self, text: str) -> int:
        """Return the number of characters."""
        return len(text)
