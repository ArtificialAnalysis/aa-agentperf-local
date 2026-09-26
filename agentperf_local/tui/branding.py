"""Provide the TUI palette, pixel logo, ASCII kitty, spinner frames, and aligned labels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from textual.widgets import Static

# Palette sourced from https://artificialanalysis.ai on 2026-09-01.
# Brand hues are the site's published CSS design tokens in
# /_next/static/css/f919fb80f750d365.css (HSL values converted to hex here). The primary
# purple is the fill of the logo mark (/img/logo-icon.svg), which is also the dominant
# accent hex in the homepage HTML.
AA_PURPLE = "#7F4BF3"  # fill of https://artificialanalysis.ai/img/logo-icon.svg
AA_PURPLE_LIGHT = "#C394FF"  # site token --brand-purple-light: 266.36 100% 79.02%
AA_LIME = "#CEFF73"  # site token --brand-lime: 81 100% 72.55%
AA_ORANGE = "#FF7734"  # site token --brand-orange: 19.8 100% 60.2%
AA_RED = "#EF4444"  # site token --destructive: 0 84.2% 60.2%
AA_NEUTRAL_50 = "#FAFAFA"  # site token --neutral-50: 0 0% 98%
AA_NEUTRAL_100 = "#E7E7E7"  # site token --neutral-100: 0 0% 90.59%
AA_NEUTRAL_500 = "#949494"  # site token --neutral-500: 0 0% 58%
AA_NEUTRAL_700 = "#4C4C4C"  # site token --neutral-700: 0 0% 30%
AA_NEUTRAL_900 = "#1F1F1F"  # site token --neutral-900: 0 0% 12%
# Terminal-only derivations: the site's dark sections sit on #000000, so the canvas drops
# below --neutral-900, and the purple-tinted tones blend AA_PURPLE onto --neutral-900.
AA_PLACEHOLDER = "#767676"  # between --neutral-500 and --neutral-700: a hint, dimmer than real muted text
AA_CANVAS = "#0F0F0F"  # midpoint of the site's #000000 dark sections and --neutral-900
AA_PANEL = "#171717"  # midpoint of AA_CANVAS and --neutral-900
AA_BORDER = "#353535"  # midpoint of --neutral-900 and --neutral-700
AA_FOCUS = "#322849"  # 20% AA_PURPLE blended over --neutral-900
AA_PURPLE_DIM = "#412E69"  # 35% AA_PURPLE blended over --neutral-900

AA_BRAND_CSS_VARIABLES = f"""
$aa-purple: {AA_PURPLE};
$aa-purple-light: {AA_PURPLE_LIGHT};
$aa-purple-dim: {AA_PURPLE_DIM};
$aa-lime: {AA_LIME};
$aa-orange: {AA_ORANGE};
$aa-red: {AA_RED};
$aa-bg: {AA_CANVAS};
$aa-panel: {AA_PANEL};
$aa-surface: {AA_NEUTRAL_900};
$aa-focus: {AA_FOCUS};
$aa-border: {AA_BORDER};
$aa-disabled: {AA_NEUTRAL_700};
$aa-placeholder: {AA_PLACEHOLDER};
$aa-muted: {AA_NEUTRAL_500};
$aa-text: {AA_NEUTRAL_100};
$aa-text-bright: {AA_NEUTRAL_50};
"""

_KEY_VALUE_GAP = "  "

# Keep every pose the same size so changing expression never moves nearby text.
KITTY_CURIOUS = ' /\\_/\\ \n( o.o )\n(")_(")'
KITTY_HAPPY = KITTY_CURIOUS.replace("o.o", "^.^")
KITTY_BLINK = KITTY_CURIOUS.replace("o.o", "-.-")
KITTY_GLANCE = KITTY_CURIOUS.replace(" o.o ", "o.o  ")


@dataclass(frozen=True, slots=True, kw_only=True)
class KittyFrame:
    """Hold one expression before showing the next."""

    art: str
    hold_seconds: float


KITTY_FRAMES = (
    KittyFrame(art=KITTY_CURIOUS, hold_seconds=4.0),
    KittyFrame(art=KITTY_BLINK, hold_seconds=0.18),
    KittyFrame(art=KITTY_CURIOUS, hold_seconds=4.0),
    KittyFrame(art=KITTY_GLANCE, hold_seconds=0.7),
)


def key_value_block(rows: Iterable[tuple[str, str]]) -> str:
    """Render key/value rows: muted keys, bright values, key column sized to the longest key."""
    materialized = tuple(rows)
    key_width = max(len(key) for key, _ in materialized)
    return "\n".join(
        f"[{AA_NEUTRAL_500}]{key.ljust(key_width)}[/]{_KEY_VALUE_GAP}[{AA_NEUTRAL_50}]{value}[/]"
        for key, value in materialized
    )


_FILLED_PIXEL = "X"


def _render_half_blocks(grid: tuple[str, ...]) -> str:
    """Collapse an X-marked grid into half-block rows, two pixels per character."""
    width = max(len(row) for row in grid)
    rows = tuple(row.ljust(width, ".") for row in grid)
    lines: list[str] = []
    for top in range(0, len(rows), 2):
        bottom = top + 1
        cells: list[str] = []
        for column in range(width):
            upper = rows[top][column] == _FILLED_PIXEL
            lower = bottom < len(rows) and rows[bottom][column] == _FILLED_PIXEL
            cells.append("█" if upper and lower else "▀" if upper else "▄" if lower else " ")
        lines.append("".join(cells))
    return "\n".join(lines)


# The Artificial Analysis mark, rasterized from the 53x53 logo-icon.svg: two chevron bands
# descending to the left on a 4x4 cell grid, plus two squares in the right-hand column.
# Sampled at pixel centers on an 8x8 grid.
_PIXEL_LOGO_SMALL_GRID = (
    "...XXX..",
    "..XXXX..",
    "XXXX..XX",
    "XXX...XX",
    "...XXX..",
    "..XXXX..",
    "XXXX..XX",
    "XXX...XX",
)
PIXEL_LOGO_SMALL = _render_half_blocks(_PIXEL_LOGO_SMALL_GRID)
# Below this width the app switches to its compact layout.
COMPACT_LAYOUT_WIDTH = 90


class PixelLogo(Static):
    """Show the small Artificial Analysis mark as half-block pixel art beside the welcome heading."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(PIXEL_LOGO_SMALL, id=id)


# A braille spinner for indeterminate waits. Every frame is one cell wide, so a tick
# repaints the same region without a layout pass.
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def spinner_frame(tick: int) -> str:
    """Return the spinner frame for one timer tick."""
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
