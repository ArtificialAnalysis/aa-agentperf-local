"""Summarize measured samples with one percentile convention."""

from collections.abc import Sequence

P50_PERCENTILE = 0.50
P90_PERCENTILE = 0.90
P95_PERCENTILE = 0.95


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return the linearly interpolated percentile of values, or None when there are none.

    Every reported percentile in this package uses this one definition, so a
    private artifact, a public aggregate, and a chart label cannot disagree.
    """
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
