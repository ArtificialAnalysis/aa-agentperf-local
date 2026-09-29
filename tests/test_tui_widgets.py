"""Pin the range chart's one-line box plot and its labels."""

import pytest

from agentperf_local.tui.labels import integer_label
from agentperf_local.tui.widgets import (
    LATENCY_PLOT_EDGE_RATIO,
    SPEED_PLOT_EDGE_RATIO,
    range_labels,
    range_plot,
    summarize_range,
)


@pytest.mark.parametrize(
    ("values", "edge_ratio", "plot", "labels"),
    [
        # A tight run still draws a box a cell either side of the median.
        (
            [1275, 1283, 1286, 1288, 1290, 1291, 1294, 1310, 1329],
            SPEED_PLOT_EDGE_RATIO,
            "··················├███┤··················",
            "min 1,275     median 1,290      max 1,329",
        ),
        (
            [12, 14, 15, 16, 17, 17, 18, 19, 20, 22, 24],
            SPEED_PLOT_EDGE_RATIO,
            "··········├──────████████─────┤··········",
            "min 12          median 17          max 24",
        ),
        (
            [500, 600, 800, 1200, 2000, 3500, 5000, 7500, 12000, 20000],
            LATENCY_PLOT_EDGE_RATIO,
            "·····├────███████████████████────────┤···",
            "min 500       median 2,750     max 20,000",
        ),
        # A whisker past the edge ends in an arrow, and the box leaves it room.
        (
            [200, 210, 220, 230, 240, 250, 2600, 9000, 15000],
            LATENCY_PLOT_EDGE_RATIO,
            "··················├█████████████████████›",
            "min 200        median 240      max 15,000",
        ),
    ],
)
def test_range_chart_sketches_spread_and_labels_min_median_max(
    values: list[float], edge_ratio: float, plot: str, labels: str
) -> None:
    summary = summarize_range(values)
    assert summary is not None
    assert range_plot(summary, edge_ratio).plain == plot
    assert range_labels(summary, integer_label).plain.rstrip() == labels
