"""Draw the activity log, charts, headline metrics, busy indicators, and animated kitty."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import cache

from pydantic import BaseModel
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import Digits, ProgressBar, RichLog, Static

from agentperf_local.common.statistics import P50_PERCENTILE, P90_PERCENTILE, percentile
from agentperf_local.reports.progress import PERCENT_SCALE, request_size_text
from agentperf_local.tui.branding import (
    AA_LIME,
    AA_NEUTRAL_50,
    AA_NEUTRAL_500,
    AA_ORANGE,
    AA_PURPLE,
    AA_PURPLE_LIGHT,
    AA_RED,
    KITTY_CURIOUS,
    KITTY_FRAMES,
    KITTY_HAPPY,
    spinner_frame,
)

# One glyph per outcome, so a log line's state reads at a glance.
DONE_MARK = "✓"
FAILED_MARK = "✗"
WARNING_MARK = "⚠"
MARK_COLORS = {DONE_MARK: AA_LIME, FAILED_MARK: AA_RED, WARNING_MARK: AA_ORANGE}
# Eighth-block steps from empty to full; index n draws a bar n/8 of a cell tall.
BAR_STEPS = (" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")
EIGHTHS_PER_ROW = len(BAR_STEPS) - 1
DEFAULT_BIN_COUNT = 8
DEFAULT_CHART_ROWS = 3
BAR_WIDTH = 3
BAR_GAP = 1
EMPTY_CHART_MESSAGE = "waiting for the first turn…"
LIVE_TICK_SECONDS = 0.1
COUNT_UP_SECONDS = 0.6
COUNT_UP_STEPS = 12
ACTIVITY_LOG_MAX_LINES = 500
# The context gauge draws one filled bar between two thin end caps. A compact
# layout loses the bar's tail and the window size to keep the line on one row.
CONTEXT_BAR_WIDTH = 20
CONTEXT_BAR_COMPACT_WIDTH = 10
CONTEXT_BAR_FILLED = "█"
CONTEXT_BAR_EMPTY = "░"
CONTEXT_BAR_LEFT_CAP = "▕"
CONTEXT_BAR_RIGHT_CAP = "▏"
CONTEXT_WINDOW_UNKNOWN = "context length not reported"


@cache
def _colored_kitty(art: str) -> Text:
    """Color one pose's face once; every later frame change is a lookup and a repaint."""
    ears, face, paws = art.splitlines()
    return Text.assemble(ears, "\n", face[0], (face[1:-1], AA_PURPLE_LIGHT), face[-1], "\n", paws)


class Kitty(Static):
    """Blink and glance within a fixed ASCII silhouette."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(_colored_kitty(KITTY_CURIOUS), id=id, classes="kitty")
        self._art = KITTY_CURIOUS
        self._frame_index = 0
        self._timer: Timer | None = None

    @property
    def is_animating(self) -> bool:
        """Report whether the kitty is waiting for its next expression."""
        return self._timer is not None

    def start(self) -> None:
        """Animate while visible, unless terminal animations are disabled."""
        if self.is_animating or self.app.animation_level == "none":
            return
        self._frame_index = 0
        self._show_frame()

    def settle(self, *, happy: bool = False) -> None:
        """Stop the animation and hold the requested expression."""
        self._stop_timer()
        self._show_art(KITTY_HAPPY if happy else KITTY_CURIOUS)

    def on_unmount(self) -> None:
        self._stop_timer()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _show_art(self, art: str) -> None:
        """Repaint only these cells, and only when the pose changes: resizes and page switches settle repeatedly."""
        if art == self._art:
            return
        self._art = art
        self.update(_colored_kitty(art), layout=False)

    def _show_frame(self) -> None:
        frame = KITTY_FRAMES[self._frame_index]
        # Schedule the next change instead of polling while the pose holds.
        self._show_art(frame.art)
        timer: Timer | None = None

        def advance() -> None:
            # Textual queues a fired timer's callback as a message, so a tick can arrive after
            # settle() stopped its timer, or after start() replaced it. Only the current timer advances.
            if self._timer is timer:
                self._advance()

        timer = self.set_timer(frame.hold_seconds, advance)
        self._timer = timer

    def _advance(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(KITTY_FRAMES)
        self._show_frame()


def chart_width(bins: int = DEFAULT_BIN_COUNT) -> int:
    """Return the cell width of a histogram with this many bins."""
    return bins * (BAR_WIDTH + BAR_GAP) - BAR_GAP


class Histogram(BaseModel, frozen=True):
    """Store binned counts of one metric with the bounds and percentiles the chart labels."""

    bins: tuple[int, ...]
    low: float
    high: float
    p50: float
    p90: float
    count: int

    @property
    def median_bin(self) -> int:
        """Return the index of the bin that holds the median."""
        return _bin_index(self.p50, self.low, self.high, len(self.bins))


def _bin_index(value: float, low: float, high: float, bins: int) -> int:
    if high <= low:
        return bins // 2
    return min(int((value - low) / (high - low) * bins), bins - 1)


def build_histogram(values: Sequence[float], bins: int = DEFAULT_BIN_COUNT) -> Histogram | None:
    """Bin values evenly between their minimum and maximum, or return None without values."""
    ordered = tuple(sorted(values))
    if not ordered:
        return None
    low, high = ordered[0], ordered[-1]
    counts = [0] * bins
    for value in ordered:
        counts[_bin_index(value, low, high, bins)] += 1
    p50 = percentile(ordered, P50_PERCENTILE)
    p90 = percentile(ordered, P90_PERCENTILE)
    if p50 is None or p90 is None:
        return None
    return Histogram(bins=tuple(counts), low=low, high=high, p50=p50, p90=p90, count=len(ordered))


def histogram_rows(histogram: Histogram, rows: int = DEFAULT_CHART_ROWS) -> tuple[str, ...]:
    """Render the bars top row first, each bin BAR_WIDTH cells wide, scaled so the tallest bin fills the rows."""
    tallest = max(histogram.bins)
    heights = tuple(round(count / tallest * rows * EIGHTHS_PER_ROW) if tallest else 0 for count in histogram.bins)
    lines: list[str] = []
    for row in range(rows - 1, -1, -1):
        cells: list[str] = []
        for eighths in heights:
            visible = min(max(eighths - row * EIGHTHS_PER_ROW, 0), EIGHTHS_PER_ROW)
            cells.append(BAR_STEPS[visible] * BAR_WIDTH)
        lines.append((" " * BAR_GAP).join(cells))
    return tuple(lines)


class DistributionChart(Static):
    """Draw a small block-character histogram of one per-turn metric with its p50 and p90.

    The bin holding the median is lit in the success color so the center of the
    distribution reads without a legend.
    """

    def __init__(
        self,
        title: str,
        unit: str,
        *,
        format_value: Callable[[float], str],
        rows: int = DEFAULT_CHART_ROWS,
        bins: int = DEFAULT_BIN_COUNT,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.title = title
        self.unit = unit
        self.format_value = format_value
        self.rows = rows
        self.bins = bins
        self.histogram: Histogram | None = None
        self.update(self._render_text())

    def update_samples(self, values: Sequence[float]) -> None:
        """Rebuild the chart from every value seen so far."""
        self.histogram = build_histogram(values, self.bins)
        # Every rendition has the same row count and width, so a repaint never needs a layout pass.
        self.update(self._render_text(), layout=False)

    def _render_text(self) -> Text:
        width = chart_width(self.bins)
        text = Text()
        text.append(self.title, style=f"bold {AA_PURPLE_LIGHT}")
        text.append("\n")
        histogram = self.histogram
        if histogram is None:
            text.append(EMPTY_CHART_MESSAGE, style=AA_NEUTRAL_500)
            text.append("\n")
            for _ in range(self.rows):
                text.append(" " * width + "\n")
            text.append(" " * width)
            return text
        text.append(
            f"p50 {self.format_value(histogram.p50)} · p90 {self.format_value(histogram.p90)} {self.unit}",
            style=AA_NEUTRAL_50,
        )
        text.append("\n")
        median_start = histogram.median_bin * (BAR_WIDTH + BAR_GAP)
        for row in histogram_rows(histogram, self.rows):
            text.append(row[:median_start], style=AA_PURPLE)
            text.append(row[median_start : median_start + BAR_WIDTH], style=AA_LIME)
            text.append(row[median_start + BAR_WIDTH :], style=AA_PURPLE)
            text.append("\n")
        low = self.format_value(histogram.low)
        high = self.format_value(histogram.high)
        gap = max(width - len(low) - len(high), 1)
        text.append(f"{low}{' ' * gap}{high}", style=AA_NEUTRAL_500)
        return text


class ContextGauge(Static):
    """Show the size of the request about to be sent and how much of the context window it fills.

    The size is the one the recording counted, because nothing has tokenized the prompt
    yet, so it is worded as approximate. Without a served context length there is no
    honest denominator, and the bar is left out.
    """

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__("", id=id, classes=classes)
        self.plain_text = ""
        self.display = False

    def show_request(self, tokens: int | None, *, context_tokens: int | None, compact: bool = False) -> None:
        """Show one request size, or hide the line when nothing counted the prompt."""
        if tokens is None:
            self.clear()
            return
        width = CONTEXT_BAR_COMPACT_WIDTH if compact else CONTEXT_BAR_WIDTH
        line = Text()
        line.append("Next · " if compact else "Next request · ", style=AA_NEUTRAL_500)
        line.append(request_size_text(tokens), style=AA_NEUTRAL_50)
        if context_tokens is None or context_tokens <= 0:
            line.append(f" · {CONTEXT_WINDOW_UNKNOWN}", style=AA_NEUTRAL_500)
        else:
            fraction = tokens / context_tokens
            filled = min(round(fraction * width), width)
            # A request larger than the window will not fit, so the bar says so in the warning color.
            fill_style = AA_ORANGE if fraction > 1 else AA_PURPLE
            line.append(f" {CONTEXT_BAR_LEFT_CAP}", style=AA_NEUTRAL_500)
            line.append(CONTEXT_BAR_FILLED * filled, style=fill_style)
            line.append(CONTEXT_BAR_EMPTY * (width - filled), style=AA_NEUTRAL_500)
            line.append(CONTEXT_BAR_RIGHT_CAP, style=AA_NEUTRAL_500)
            window = "" if compact else f" of {context_tokens:,}"
            line.append(f" {fraction * PERCENT_SCALE:.0f}%{window}", style=AA_NEUTRAL_500)
        self.plain_text = line.plain
        self.display = True
        self.update(line)

    def clear(self) -> None:
        """Hide the line before a run and whenever no request size is known."""
        self.plain_text = ""
        self.update("")
        self.display = False


class ActivityLog(Vertical):
    """Scroll finished run steps above one live line that spins while a step is in progress."""

    DEFAULT_CSS = """
    ActivityLog {
        height: 1fr;
    }

    ActivityLog > RichLog {
        height: 1fr;
        background: transparent;
        scrollbar-size: 1 1;
        padding: 0;
    }

    ActivityLog > #activity-live {
        height: auto;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._live_text: str | None = None
        self._live_started = 0.0
        self._live_tick = 0
        self._live_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Compose the scrolling history and the single live line."""
        yield RichLog(id="activity-lines", markup=False, wrap=True, min_width=20, max_lines=ACTIVITY_LOG_MAX_LINES)
        yield Static("", id="activity-live")
        yield ProgressBar(total=1, show_eta=False, id="activity-progress")

    def on_mount(self) -> None:
        """Keep the download bar hidden until progress arrives."""
        self.query_one("#activity-progress", ProgressBar).display = False

    def clear(self) -> None:
        """Drop every line and stop the live spinner before a new run."""
        self.query_one("#activity-lines", RichLog).clear()
        self.set_live(None)

    def record(self, mark: str, text: str) -> None:
        """Append one finished step with its outcome glyph."""
        line = Text()
        line.append(mark, style=f"bold {MARK_COLORS.get(mark, AA_NEUTRAL_500)}")
        line.append(" ")
        line.append(text, style=AA_NEUTRAL_50)
        lines = self.query_one("#activity-lines", RichLog)
        lines.write(line)
        # A burst written before the page has its size defers wrapping, and the log's own
        # auto-scroll runs before that; scrolling after the refresh keeps the newest line in view.
        self.call_after_refresh(scroll_log_to_end, lines)

    def set_live(self, text: str | None) -> None:
        """Show the step now in progress with a spinner and its running time, or clear the line.

        Re-posting the step already shown changes nothing, so a continuing step keeps
        its running time instead of restarting the clock.
        """
        if text == self._live_text and self._live_timer is not None:
            return
        self._stop_clock()
        self._live_text = text
        self._live_started = time.monotonic()
        self._live_tick = 0
        if text is None:
            self.query_one("#activity-live", Static).update("")
            return
        self._paint_live()
        self._live_timer = self.set_interval(LIVE_TICK_SECONDS, self._advance_live)

    def set_progress(self, text: str, *, total: int, progress: int) -> None:
        """Replace the live spinner with a download label and a measured bar."""
        self._stop_clock()
        self._live_text = text
        self.query_one("#activity-live", Static).update(Text(text, style=AA_NEUTRAL_50))
        bar = self.query_one("#activity-progress", ProgressBar)
        bar.display = True
        bar.update(total=total, progress=progress)

    def _stop_clock(self) -> None:
        """Drop whatever is measuring the live line: the spinner's timer or the bar."""
        if self._live_timer is not None:
            self._live_timer.stop()
            self._live_timer = None
        self.query_one("#activity-progress", ProgressBar).display = False

    def _advance_live(self) -> None:
        self._live_tick += 1
        self._paint_live()

    def _paint_live(self) -> None:
        if self._live_text is None:
            return
        elapsed = time.monotonic() - self._live_started
        line = Text()
        line.append(spinner_frame(self._live_tick), style=AA_PURPLE_LIGHT)
        line.append(" ")
        line.append(self._live_text, style=AA_NEUTRAL_50)
        line.append(f" · {elapsed:,.1f} s", style=AA_NEUTRAL_500)
        self.query_one("#activity-live", Static).update(line)


class SpinnerLine(Static):
    """Show one in-progress line with a spinner, then settle on its outcome glyph and text."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.plain_text = ""
        self._pending: str | None = None
        self._tick = 0
        self._timer: Timer | None = None
        self.display = False

    def start(self, text: str) -> None:
        """Spin beside text until finish or clear is called."""
        self._pending = text
        self._tick = 0
        self.display = True
        self._paint()
        if self._timer is None:
            self._timer = self.set_interval(LIVE_TICK_SECONDS, self._advance)

    def finish(self, mark: str, text: str) -> None:
        """Stop spinning and show the outcome."""
        self._stop_timer()
        self._pending = None
        self.plain_text = f"{mark} {text}"
        line = Text()
        line.append(mark, style=f"bold {MARK_COLORS.get(mark, AA_NEUTRAL_500)}")
        line.append(" ")
        line.append(text, style=AA_NEUTRAL_50)
        self.display = True
        self.update(line)

    def clear(self) -> None:
        """Stop spinning and hide the line."""
        self._stop_timer()
        self._pending = None
        self.plain_text = ""
        self.update("")
        self.display = False

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _advance(self) -> None:
        self._tick += 1
        self._paint()

    def _paint(self) -> None:
        if self._pending is None:
            return
        self.plain_text = self._pending
        line = Text()
        line.append(spinner_frame(self._tick), style=AA_PURPLE_LIGHT)
        line.append(" ")
        line.append(self._pending, style=AA_NEUTRAL_50)
        self.update(line, layout=False)


def scroll_log_to_end(log: RichLog) -> None:
    """Show the newest line of a log whose size may have arrived after its content."""
    log.scroll_end(animate=False)


def _ease_out(fraction: float) -> float:
    return 1.0 - (1.0 - fraction) ** 3


class HeadlineDigits(Digits):
    """Count up to a new headline value in a short ease so a result lands instead of popping."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self._shown = 0.0
        self._target = 0.0
        self._step = 0
        self._timer: Timer | None = None

    def show_value(self, value: float | None, *, animate: bool = True) -> None:
        """Show one value, easing from the value on screen; None clears the digits."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if value is None:
            self._shown = 0.0
            self.update("")
            return
        self._target = value
        if not animate:
            self._shown = value
            self.update(self._format(value))
            return
        self._step = 0
        self._timer = self.set_interval(COUNT_UP_SECONDS / COUNT_UP_STEPS, self._advance)

    def _advance(self) -> None:
        self._step += 1
        fraction = min(self._step / COUNT_UP_STEPS, 1.0)
        current = self._shown + (self._target - self._shown) * _ease_out(fraction)
        self.update(self._format(current))
        if fraction >= 1.0:
            self._shown = self._target
            if self._timer is not None:
                self._timer.stop()
                self._timer = None

    @staticmethod
    def _format(value: float) -> str:
        return f"{value:,.0f}"
