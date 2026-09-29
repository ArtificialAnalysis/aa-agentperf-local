"""Draw the activity log, charts, headline metrics, busy indicators, and animated kitty."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from functools import cache

from pydantic import BaseModel
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import Digits, ProgressBar, RichLog, Static

from agentperf_local.common.statistics import P25_PERCENTILE, P50_PERCENTILE, P75_PERCENTILE, percentile
from agentperf_local.reports.progress import PERCENT_SCALE, request_size_text
from agentperf_local.tui.branding import (
    AA_LIME,
    AA_NEUTRAL_50,
    AA_NEUTRAL_500,
    AA_NEUTRAL_700,
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
# A range chart is a title, a one-line box plot, and a line of labels under it.
RANGE_PLOT_CELLS = 41
RANGE_PLOT_CENTER = RANGE_PLOT_CELLS // 2
CHART_WIDTH = RANGE_PLOT_CELLS
# The plot centers on the median and reaches a fixed ratio either side of it on a log
# scale, so its width shows spread relative to the median, not to the unit. Decode speed
# varies far less than latency, which grows with each task's context.
SPEED_PLOT_EDGE_RATIO = 2.0
LATENCY_PLOT_EDGE_RATIO = 10.0
PLOT_TRACK = "·"
PLOT_LOW_CAP = "├"
PLOT_HIGH_CAP = "┤"
PLOT_LOW_CLIPPED = "‹"
PLOT_HIGH_CLIPPED = "›"
PLOT_WHISKER = "─"
PLOT_BOX = "█"
EMPTY_CHART_MESSAGE = "waiting for the first turn…"
LIVE_TICK_SECONDS = 0.1
COUNT_UP_SECONDS = 0.6
COUNT_UP_STEPS = 12
ACTIVITY_LOG_MAX_LINES = 500
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


class RangeSummary(BaseModel, frozen=True):
    """Store the extremes and quartiles one range chart draws, from the package's one percentile definition."""

    minimum: float
    p25: float
    median: float
    p75: float
    maximum: float


def summarize_range(values: Sequence[float]) -> RangeSummary | None:
    """Return the extremes and quartiles of values, or None without values."""
    p25 = percentile(values, P25_PERCENTILE)
    median = percentile(values, P50_PERCENTILE)
    p75 = percentile(values, P75_PERCENTILE)
    if p25 is None or median is None or p75 is None:
        return None
    return RangeSummary(minimum=min(values), p25=p25, median=median, p75=p75, maximum=max(values))


def _plot_cell(value: float, median: float, edge_ratio: float) -> int:
    """Place one value on the median-centered log scale, clamped to the plot."""
    if median <= 0:
        return RANGE_PLOT_CENTER
    if value <= 0:
        return 0
    offset = round(math.log(value / median, edge_ratio) * RANGE_PLOT_CENTER)
    return min(max(RANGE_PLOT_CENTER + offset, 0), RANGE_PLOT_CELLS - 1)


def range_plot(summary: RangeSummary, edge_ratio: float) -> Text:
    """Draw a one-line box plot: whiskers at the extremes, the box from p25 to p75, the median lit.

    It is a sketch, not to scale: the box always shows a cell either side of the
    median and each whisker at least one cell beyond the box, so a tight run still
    reads as a box. A whisker past the plot's edge ends in an arrow.
    """
    median = summary.median
    # The box stops a cell short of each edge, so a clipped whisker always has room for its arrow.
    box_low = max(min(_plot_cell(summary.p25, median, edge_ratio), RANGE_PLOT_CENTER - 1), 1)
    box_high = min(max(_plot_cell(summary.p75, median, edge_ratio), RANGE_PLOT_CENTER + 1), RANGE_PLOT_CELLS - 2)
    whisker_low = max(min(_plot_cell(summary.minimum, median, edge_ratio), box_low - 1), 0)
    whisker_high = min(max(_plot_cell(summary.maximum, median, edge_ratio), box_high + 1), RANGE_PLOT_CELLS - 1)
    plot = Text()
    for cell in range(RANGE_PLOT_CELLS):
        if cell == RANGE_PLOT_CENTER:
            plot.append(PLOT_BOX, style=AA_LIME)
        elif box_low <= cell <= box_high:
            plot.append(PLOT_BOX, style=AA_PURPLE)
        elif cell == whisker_low:
            clipped = summary.minimum < median / edge_ratio
            plot.append(PLOT_LOW_CLIPPED if clipped else PLOT_LOW_CAP, style=AA_NEUTRAL_500)
        elif cell == whisker_high:
            clipped = summary.maximum > median * edge_ratio
            plot.append(PLOT_HIGH_CLIPPED if clipped else PLOT_HIGH_CAP, style=AA_NEUTRAL_500)
        elif whisker_low < cell < whisker_high:
            plot.append(PLOT_WHISKER, style=AA_NEUTRAL_500)
        else:
            plot.append(PLOT_TRACK, style=AA_NEUTRAL_700)
    return plot


def range_labels(summary: RangeSummary, format_value: Callable[[float], str]) -> Text:
    """Label the plot: min at its left, the median under its center, where the median sits, and max at its right.

    When the numbers are too long to spread out, they fall back to one run separated by gaps.
    """
    parts = (
        ("min ", format_value(summary.minimum)),
        ("median ", format_value(summary.median)),
        ("max ", format_value(summary.maximum)),
    )
    low, middle, high = (len(word) + len(number) for word, number in parts)
    middle_start = min(max(RANGE_PLOT_CENTER - middle // 2, low + 1), RANGE_PLOT_CELLS - high - middle - 1)
    gaps = (" " * max(middle_start - low, 1), " " * max(RANGE_PLOT_CELLS - high - middle_start - middle, 1), "")
    labels = Text()
    for (word, number), gap in zip(parts, gaps, strict=True):
        labels.append(word, style=AA_NEUTRAL_500)
        labels.append(number, style=AA_NEUTRAL_50)
        labels.append(gap)
    return labels


class RangeChart(Static):
    """Show one per-turn metric as a one-line box plot labeled with its min, median, and max.

    Every rendition is three lines, so a repaint never needs a layout pass.
    """

    def __init__(
        self,
        title: str,
        unit: str,
        *,
        format_value: Callable[[float], str],
        edge_ratio: float,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.title = title
        self.unit = unit
        self.format_value = format_value
        self.edge_ratio = edge_ratio
        self.summary: RangeSummary | None = None
        self.update(self._render_text())

    def update_samples(self, values: Sequence[float]) -> None:
        """Rebuild the chart from every value seen so far."""
        self.summary = summarize_range(values)
        self.update(self._render_text(), layout=False)

    def _render_text(self) -> Text:
        summary = self.summary
        text = Text()
        text.append(self.title, style=f"bold {AA_PURPLE_LIGHT}")
        text.append(f" · {self.unit}".ljust(CHART_WIDTH - len(self.title)), style=AA_NEUTRAL_500)
        text.append("\n")
        if summary is None:
            text.append(EMPTY_CHART_MESSAGE.ljust(CHART_WIDTH), style=AA_NEUTRAL_500)
            text.append("\n" + " " * CHART_WIDTH)
            return text
        text.append_text(range_plot(summary, self.edge_ratio))
        text.append("\n")
        text.append_text(range_labels(summary, self.format_value))
        return text


class ContextGauge(Static):
    """Say in words how much of the context window the request about to be sent fills.

    It is words rather than a bar, so it never reads as a second progress bar. The size
    is the one the recording counted, because nothing has tokenized the prompt yet, so
    it is worded as approximate. Without a served context length there is no honest
    denominator, and the share is left out.
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
        line = Text()
        line.append("Next request · " if compact else "Context for next request · ", style=AA_NEUTRAL_500)
        if context_tokens is None or context_tokens <= 0:
            line.append(request_size_text(tokens), style=AA_NEUTRAL_50)
            line.append(f" · {CONTEXT_WINDOW_UNKNOWN}", style=AA_NEUTRAL_500)
        else:
            fraction = tokens / context_tokens
            # A request larger than the window will not fit, so its size says so in the warning color.
            size_style = AA_ORANGE if fraction > 1 else AA_NEUTRAL_50
            unit = "" if compact else " tokens"
            line.append(f"about {tokens:,} of {context_tokens:,}{unit}", style=size_style)
            line.append(f" ({fraction * PERCENT_SCALE:.0f}%)", style=AA_NEUTRAL_500)
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
