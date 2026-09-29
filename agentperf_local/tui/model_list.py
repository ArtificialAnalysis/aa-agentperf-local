"""Order recipes for the model screen and word each row.

- ListedRecipe: one recipe and what this computer can do with it.
- ordered_recipes: this computer's recipes, then other hardware, each grouped by model.
- model_list_options: the list rows, with part, column, and model headings.
- table_width: the width the widest recipe row needs.
- standing_mark: the colored mark each standing shows.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

from pydantic import BaseModel
from rich.style import Style
from rich.text import Text
from textual.widgets.option_list import Option

from agentperf_local.deployment.catalog import ModelCandidate, ModelCatalog
from agentperf_local.tui.branding import AA_LIME, AA_NEUTRAL_500, AA_ORANGE, AA_PLACEHOLDER, AA_RED
from agentperf_local.tui.evidence import SelectionKind
from agentperf_local.tui.labels import hardware_target_text, quantization_text, speedup_text
from agentperf_local.tui.replay_contract import RECIPE_STANDING_ORDER, ManagedModelAvailability, RecipeStanding

CUSTOM_ENDPOINT_LABEL = "Other model or server"
THIS_COMPUTER_HEADING = "THIS COMPUTER"
OTHER_HARDWARE_HEADING = "OTHER HARDWARE"
NOTHING_RUNS_HERE = "No recipe runs on this computer."
STANDING_MARKS: dict[RecipeStanding, tuple[str, str | None]] = {
    RecipeStanding.READY: ("●", AA_LIME),
    RecipeStanding.NEEDS_SETUP: ("▲", AA_ORANGE),
    RecipeStanding.REDUCED_ONLY: ("▲", AA_ORANGE),
    RecipeStanding.TOO_LARGE: ("✗", AA_RED),
    RecipeStanding.OTHER_HARDWARE: ("·", None),
}
# Without hardware detection no recipe is assessed, so its mark column stays blank.
UNASSESSED_MARK = " "
RECIPE_INDENT = "  "
MARK_GAP = " "
# Model names sit at the left edge; a recipe's quantization starts after its indent and mark.
QUANTIZATION_OFFSET = len(RECIPE_INDENT + UNASSESSED_MARK + MARK_GAP)
COLUMN_GAP = "  "
MODEL_COLUMN_HEADING = "Model/Quant"
SPEEDUP_COLUMN_HEADING = "Spec decode"
HARDWARE_COLUMN_HEADING = "Built for"
NO_SPEEDUP = "—"
COLUMN_HEADING = Style(bold=True)
MUTED = Style(color=AA_NEUTRAL_500)
# Other hardware is greyed out, marks included, so it reads as out of reach at a glance.
GREYED = Style(color=AA_PLACEHOLDER)
MODEL_HEADING = Style(bold=True)
DIGIT_RUN = re.compile(r"(\d+)")
# Device recipes sort first by folder name, then these portable ones, narrowest first.
PORTABLE_HARDWARE = ("nvidia-cuda", "any")


class ListedRecipe(BaseModel, frozen=True):
    """Pair one recipe and its hardware folder with this computer's availability, None without detection."""

    candidate: ModelCandidate
    hardware: str
    availability: ManagedModelAvailability | None

    @property
    def for_other_hardware(self) -> bool:
        """Return whether the recipe belongs under other hardware rather than this computer."""
        return self.availability is not None and self.availability.standing is RecipeStanding.OTHER_HARDWARE


def standing_mark(standing: RecipeStanding) -> Text:
    """Return the mark for one standing, colored when the standing has a color."""
    mark, color = STANDING_MARKS[standing]
    return Text(mark, style=Style(color=color))


def _natural_key(name: str) -> tuple[tuple[int, int | str], ...]:
    # Digit runs compare as numbers, so Qwen3.5 9B sorts before Qwen3.5 122B.
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in DIGIT_RUN.split(name))


def _sort_key(recipe: ListedRecipe) -> tuple[bool, tuple[tuple[int, int | str], ...], int, int, str, str]:
    standing_rank = 0 if recipe.availability is None else RECIPE_STANDING_ORDER.index(recipe.availability.standing)
    hardware = recipe.hardware
    portable_rank = PORTABLE_HARDWARE.index(hardware) + 1 if hardware in PORTABLE_HARDWARE else 0
    return (
        recipe.for_other_hardware,
        _natural_key(recipe.candidate.model_name),
        standing_rank,
        portable_rank,
        hardware,
        recipe.candidate.profile_id,
    )


def ordered_recipes(
    catalog: ModelCatalog,
    availability_of: Callable[[ModelCandidate], ManagedModelAvailability | None],
) -> tuple[ListedRecipe, ...]:
    """List recipes in screen order.

    This computer's recipes come before other hardware. Within each part, models
    follow in natural name order, and one model's recipes go best standing first,
    then device recipes before portable ones.
    """
    listed = (
        ListedRecipe(
            candidate=candidate,
            hardware=catalog.hardware_of(candidate),
            availability=availability_of(candidate),
        )
        for candidate in catalog.models
    )
    return tuple(sorted(listed, key=_sort_key))


def model_list_options(recipes: tuple[ListedRecipe, ...], computer: str | None) -> tuple[Option | None, ...]:
    """Build the list rows: this computer, the custom entry, a gap, then other hardware.

    Each part opens with its heading, a divider (the None rows), a gap, then column
    headings over its rows. Headings, gaps, and model names are disabled rows, so
    the cursor skips them.
    `computer` names this computer's accelerator; None means hardware was not
    detected, so the list is one part without a part heading.
    """
    here = tuple(recipe for recipe in recipes if not recipe.for_other_hardware)
    elsewhere = tuple(recipe for recipe in recipes if recipe.for_other_hardware)
    widths = _column_widths(recipes)
    rows: list[Option | None] = []
    if computer is not None:
        rows.extend((_heading(f"{THIS_COMPUTER_HEADING}: {computer}"), None, _gap()))
    if here:
        rows.append(_column_headings(widths, greyed=False))
        rows.extend(_part(here, widths, greyed=False))
    else:
        rows.append(Option(Text(NOTHING_RUNS_HERE), disabled=True))
    rows.append(_gap())
    rows.append(Option(CUSTOM_ENDPOINT_LABEL, id=SelectionKind.CUSTOM_ENDPOINT.value))
    if elsewhere:
        rows.extend((_gap(), _heading(OTHER_HARDWARE_HEADING), None, _gap()))
        rows.append(_column_headings(widths, greyed=True))
        rows.extend(_part(elsewhere, widths, greyed=True))
    return tuple(rows)


def _heading(text: str) -> Option:
    return Option(Text(text, style=MUTED), disabled=True)


def _gap() -> Option:
    return Option("", disabled=True)


def _column_headings(widths: tuple[int, int], *, greyed: bool) -> Option:
    model_width, speedup_width = widths
    headings = Text(
        f"{MODEL_COLUMN_HEADING.ljust(model_width)}{COLUMN_GAP}"
        f"{SPEEDUP_COLUMN_HEADING.ljust(speedup_width)}{COLUMN_GAP}{HARDWARE_COLUMN_HEADING}",
        style=COLUMN_HEADING,
    )
    if greyed:
        headings.stylize(GREYED)
    return Option(headings, disabled=True)


def table_width(recipes: tuple[ListedRecipe, ...]) -> int:
    """Return the width the widest recipe row needs, so the list can keep its columns whole."""
    model_width, speedup_width = _column_widths(recipes)
    hardware_width = max(
        (len(hardware_target_text(recipe.candidate, recipe.hardware)) for recipe in recipes), default=0
    )
    return (
        model_width
        + len(COLUMN_GAP)
        + speedup_width
        + len(COLUMN_GAP)
        + max(hardware_width, len(HARDWARE_COLUMN_HEADING))
    )


def _speedup_cell(candidate: ModelCandidate) -> str:
    return speedup_text(candidate) or NO_SPEEDUP


def _column_widths(recipes: tuple[ListedRecipe, ...]) -> tuple[int, int]:
    """Return the first column's width, counted from the left edge, and the spec decode column's width.

    Both parts share these widths, so their columns line up with each other.
    """
    quantization_width = max((len(quantization_text(recipe.candidate)) for recipe in recipes), default=0)
    speedup_width = max((len(_speedup_cell(recipe.candidate)) for recipe in recipes), default=0)
    return (
        max(QUANTIZATION_OFFSET + quantization_width, len(MODEL_COLUMN_HEADING)),
        max(speedup_width, len(SPEEDUP_COLUMN_HEADING)),
    )


def _part(recipes: tuple[ListedRecipe, ...], widths: tuple[int, int], *, greyed: bool) -> Iterator[Option]:
    """Yield one part's rows, a model heading before each model's recipes."""
    model_name: str | None = None
    for recipe in recipes:
        if recipe.candidate.model_name != model_name:
            model_name = recipe.candidate.model_name
            heading = Text(model_name, style=MODEL_HEADING)
            if greyed:
                heading.stylize(GREYED)
            yield Option(heading, disabled=True)
        yield Option(_recipe_row(recipe, widths, greyed=greyed), id=recipe.candidate.profile_id)


def _recipe_row(recipe: ListedRecipe, widths: tuple[int, int], *, greyed: bool) -> Text:
    """Word one recipe as a mark, then its quantization, spec decode method, and target hardware."""
    model_width, speedup_width = widths
    availability = recipe.availability
    candidate = recipe.candidate
    row = Text(RECIPE_INDENT)
    row.append_text(Text(UNASSESSED_MARK) if availability is None else standing_mark(availability.standing))
    row.append(
        f"{MARK_GAP}{quantization_text(candidate).ljust(model_width - QUANTIZATION_OFFSET)}{COLUMN_GAP}"
        f"{_speedup_cell(candidate).ljust(speedup_width)}{COLUMN_GAP}{hardware_target_text(candidate, recipe.hardware)}"
    )
    if greyed:
        row.stylize(GREYED)
    return row
