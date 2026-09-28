"""Subclass the Textual widgets the setup form uses, one keyboard rule each."""

from __future__ import annotations

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.geometry import Size
from textual.reactive import reactive
from textual.selection import Selection
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static

from agentperf_local.tui.labels import PATH_WRAP_BREAK


class CleanSelectStatic(Static):
    """Show directory-boundary wrap breaks without letting them into copied text.

    The zero-width PATH_WRAP_BREAK steers the wrapper, but a copy that carries it
    would paste a path no shell can resolve.
    """

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Return the selected text with every wrap-break character removed."""
        selected = super().get_selection(selection)
        if selected is None:
            return None
        text, ending = selected
        return text.replace(PATH_WRAP_BREAK, ""), ending


class ReadingPage(VerticalScroll, can_focus=False):
    """Scroll a page of text with the arrow keys from whichever control has focus.

    The page claims only the vertical arrows. It never scrolls sideways, so left and
    right fall through to the app's focus bindings.
    """


class FormPage(ReadingPage, inherit_bindings=False):
    """Scroll a form page without claiming the arrow keys.

    A stock VerticalScroll binds Up and Down to scrolling and only yields them once
    it cannot scroll, so on a page that overflows the arrows would stop moving
    between fields. Here they always reach the app's focus bindings, and the focused
    field scrolls itself into view. The mouse and the Page keys still scroll.
    """

    BINDINGS = (
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
    )


class ModelDetailPane(ReadingPage, can_focus=True):
    """Scroll model details vertically and leave them with left or right."""


class RunPage(Vertical):
    """Hold the arrow keys and carry the details key while a run is on screen.

    The focused log scrolls with the arrows and yields them only when it cannot
    scroll yet; left to the app, they would then walk focus onto Cancel, where
    Enter aborts the run. The details key lives here so the footer offers it only
    while the run page has focus.
    """

    BINDINGS = (
        Binding("up", "hold", "Scroll", show=False),
        Binding("down", "hold", "Scroll", show=False),
        Binding("d", "app.toggle_run_details", "Details"),
    )

    def action_hold(self) -> None:
        """Keep an arrow the focused log declined on this page."""


class DisclosureButton(Button):
    """Open or close the enclosing page's `.disclosed` sections; the label carries the marker."""

    expanded = reactive(False, init=False)

    def __init__(self, title: str, *, id: str | None = None) -> None:
        self._title = title
        super().__init__(f"▸ {title}", id=id, classes="disclosure", flat=True, compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.expanded = not self.expanded

    def watch_expanded(self, expanded: bool) -> None:
        self.label = f"{'▾' if expanded else '▸'} {self._title}"
        self.query_ancestor(".page").set_class(expanded, "-expanded")


class FieldInput(Input):
    """Let up and down leave a setup field instead of stopping inside it.

    The stock Input only moves its cursor horizontally, so the vertical arrows
    are free to traverse the form, mirroring FieldSelect. Enter submits the form,
    so the footer names it Review setup while a field has focus; the stock Input hides
    its enter binding entirely.
    """

    BINDINGS = (
        Binding("up", "app.focus_previous", "Previous field", show=False),
        Binding("down", "app.focus_next", "Next field", show=False),
        Binding("enter", "submit", "Review setup"),
    )

    def __init__(
        self,
        value: str = "",
        *,
        placeholder: str = "",
        id: str | None = None,
        compact: bool = False,
    ) -> None:
        # Selecting the whole value on focus paints a heavy band over the field;
        # the focus handler below parks the cursor at the end instead.
        super().__init__(value, placeholder=placeholder, id=id, compact=compact, select_on_focus=False)

    def on_focus(self) -> None:
        """Put the cursor at the end of the value without selecting anything."""
        self.cursor_position = len(self.value)

    def on_blur(self) -> None:
        """Rewind the view so a long value shows its start while the field is idle."""
        self.cursor_position = 0


class ModelList(OptionList):
    """Name the list's enter action Select in the footer while it has focus."""

    BINDINGS = (Binding("enter", "select", "Select"),)


class WelcomeChoice(Button):
    """Make a welcome description one clickable, wrapping choice."""

    def __init__(self, title: str, description: str, *, id: str) -> None:
        label = Text.assemble((f"› {title}", "bold"), f"\n  {description}")
        super().__init__(label, id=id, classes="welcome-choice", flat=True, compact=True)


class ConsentCheckbox(Checkbox):
    """Wrap the consent label in full and name its space action Agree in the footer."""

    BINDINGS = (Binding("space", "toggle_button", "Agree"),)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        """Report the wrapped label height; the stock toggle pins itself to one row.

        A consent the user cannot read in full is not a consent, so the label
        must never truncate.
        """
        del container, viewport
        return self.render().get_height(self.styles.get_rules(), width)


class SubmitCheckbox(ConsentCheckbox):
    """Opt one finished run into the upload; the footer names its space action Submit."""

    BINDINGS = (Binding("space", "toggle_button", "Submit"),)


class FieldSelect(Select[str]):
    """Keep a closed dropdown from trapping arrow-key focus traversal.

    The stock Select opens its overlay on up and down, so arrow keys can never
    move past it. Enter and space still open the overlay, which keeps its own
    arrow, enter, and escape keys.
    """

    BINDINGS = (
        Binding("up", "app.focus_previous", "Previous field", show=False),
        Binding("down", "app.focus_next", "Next field", show=False),
    )


class ClientBackendSelect(FieldSelect):
    """Select one measured streaming client with a concrete runtime type."""


class ToolChoiceSelect(FieldSelect):
    """Select whether an attached run sends tool_choice none or leaves the server default."""


class ReplayWorkloadSelect(FieldSelect):
    """Select one bundled replay or a custom manifest."""


class ManagedFrameworkSelect(FieldSelect):
    """Select one compatible installed framework for an owned deployment."""


class ManagedDeviceSelect(FieldSelect):
    """Select which detected accelerator an owned deployment runs on."""


class ManagedContextSelect(FieldSelect):
    """Select the context length an owned deployment serves."""
