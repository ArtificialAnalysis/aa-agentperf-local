"""Exercise the full-screen TUI through its public controller boundary."""

import asyncio
import base64
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.pilot import Pilot
from textual.selection import Selection
from textual.widget import Widget
from textual.widgets import Button, Checkbox, ContentSwitcher, Digits, Input, OptionList, ProgressBar, RichLog, Static

from agentperf_local.deployment.catalog import (
    BUNDLED_MODEL_CATALOG_PATH,
    ModelCandidate,
    load_model_catalog,
)
from agentperf_local.deployment.context_policy import derived_minimum_memory_bytes
from agentperf_local.deployment.endpoint_probes import ContextProbeResult
from agentperf_local.deployment.frameworks import FrameworkOffer
from agentperf_local.deployment.managed import (
    MAXIMUM_PORT,
    MINIMUM_USER_PORT,
)
from agentperf_local.deployment.managed_run import RunActivity, RunActivityKind
from agentperf_local.deployment.qualification import QUALIFICATION_FILENAME, write_runtime_qualification
from agentperf_local.provenance.context import ContextObservationReason
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from agentperf_local.replay.runner import (
    RunFinishedBoundary,
    RunStartedBoundary,
    TurnCompletedBoundary,
    TurnStartedBoundary,
)
from agentperf_local.reports.reporting import ArtifactPaths
from agentperf_local.submission.private_audit import PRIVATE_AUDIT_FILENAME
from agentperf_local.tui import app as textual_app
from agentperf_local.tui.app import (
    AA_CATALOG_RUNTIME_PROVENANCE,
    ATTACHED_CONTEXT_UNVERIFIED_MESSAGE,
    BLOCKED_ACTION_MESSAGE,
    CANCEL_CONFIRM_MESSAGE,
    CONSENT_ATTACHED_LABEL,
    MANAGED_RUN_STOPPED_MESSAGE,
    MINIMUM_SUPPORTED_HEIGHT,
    MINIMUM_SUPPORTED_WIDTH,
    PREFLIGHT_BLOCKED_HERO,
    PREFLIGHT_READY_HERO,
    RESULT_REDUCED_MESSAGE,
    RUN_HERO_CHECKING_SERVER,
    RUN_HERO_RUNNING,
    RUN_METRICS_PLACEHOLDER,
    RUN_STEP_EYEBROW,
    RUN_STEP_PREPARING_EYEBROW,
    SERVER_CHECK_RETRY_HINT,
    UPLOAD_CONFIRMING_MESSAGE,
    UPLOAD_WAITING_MESSAGE,
    AgentPerfLocalApp,
    TuiDefaults,
)
from agentperf_local.tui.branding import (
    KITTY_BLINK,
    KITTY_CURIOUS,
    KITTY_FRAMES,
    KITTY_GLANCE,
    KITTY_HAPPY,
    PIXEL_LOGO_SMALL,
    KittyFrame,
    PixelLogo,
)
from agentperf_local.tui.controller import LocalManagedReplayController
from agentperf_local.tui.evidence import (
    ArtifactEvidence,
    EligibilityReason,
    EndpointScope,
    ResultPartition,
    SelectionKind,
)
from agentperf_local.tui.inputs import (
    ClientBackendSelect,
    FormPage,
    ManagedContextSelect,
    ManagedDeviceSelect,
    ManagedFrameworkSelect,
    ReplayWorkloadSelect,
)
from agentperf_local.tui.labels import PATH_WRAP_BREAK
from agentperf_local.tui.messages import ReplayCompletedMessage
from agentperf_local.tui.replay_contract import (
    DEVICE_SELECTION_REQUIRED_MESSAGE,
    ManagedDeviceOption,
    ManagedModelAvailability,
    PreflightBlockCode,
    ReplayExecution,
    ReplayPreflight,
    ReplayRequest,
    SafeHardwareSummary,
    TuiReplayObserver,
)
from agentperf_local.tui.steps import TuiOutcome, TuiStep
from agentperf_local.tui.widgets import ContextGauge, DistributionChart, Kitty, SpinnerLine, chart_width
from agentperf_local.workload.bundled import CUSTOM_REPLAY_ID, DEFAULT_BUNDLED_REPLAY
from agentperf_local.workload.schema import parse_json_object
from tests.localhost_sse import SSE_OK_RESPONSE, LocalSseServer
from tests.replay_workload import write_replay_workload

CATALOG_PATH = BUNDLED_MODEL_CATALOG_PATH
WORKER_EVENT_TIMEOUT_SECONDS = 2.0
UI_SETTLE_TIMEOUT_SECONDS = 10.0
# Naming a served model opens the model page on the custom-endpoint entry.
ATTACHED_ENDPOINT_MODEL = "served-model"


async def _settle_until(pilot: Pilot[TuiOutcome], condition: Callable[[], bool]) -> None:
    """Pump the app until the condition holds; loaded CI runners need more than one pause."""
    deadline = time.monotonic() + UI_SETTLE_TIMEOUT_SECONDS
    while not condition() and time.monotonic() < deadline:
        await pilot.pause()
    assert condition()


async def _check_setup(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome]) -> None:
    """Press Continue and wait until its setup check has answered.

    The check runs on a worker thread that Continue starts only once its press is handled,
    so waiting for workers alone can return first. The answer resets the consent checkbox,
    so a test that ticks it earlier loses the tick.
    """
    app.query_one("#config-continue", Button).press()
    await _settle_until(pilot, lambda: app.step is TuiStep.PREFLIGHT and app.pending_preflight is None)
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _start_run(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome]) -> None:
    """Tick consent, wait until Run is offered, press it, and wait until the run page shows.

    An attached run offers Run only after its server check answers on a worker thread.
    """
    app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
    await _settle_until(pilot, lambda: not app.query_one("#run-start", Button).disabled)
    app.query_one("#run-start", Button).press()
    await _settle_until(pilot, lambda: app.step is not TuiStep.PREFLIGHT)


async def test_prefilled_model_stays_custom_after_mount() -> None:
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        defaults=TuiDefaults(endpoint_model="private-served-alias"),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.pause()

        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        assert app.selection.endpoint_model == "private-served-alias"
        assert app.query_one("#endpoint-model-input", Input).value == "private-served-alias"
        assert app.query_one("#model-list", OptionList).highlighted == len(app.catalog.models)


async def test_bundled_replay_is_the_default_without_showing_its_package_path(tmp_path: Path) -> None:
    controller = FakeReplayController()
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=controller,
        defaults=TuiDefaults(output_dir=tmp_path / "results", endpoint_model=ATTACHED_ENDPOINT_MODEL),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()

        rendered = app.export_screenshot()
        assert "AgentPerf&#160;default&#160;replay&#160;v1" in rendered
        assert str(DEFAULT_BUNDLED_REPLAY.manifest_path) not in rendered
        assert not app.query_one("#manifest-row").display
        assert "Replay" in rendered
        assert "YOUR&#160;SERVER" in rendered
        assert "Results&#160;folder" in rendered
        assert "Advanced&#160;options" in rendered

        app.query_one("#config-continue", Button).press()
        await pilot.pause()

        assert app.pending_preflight is None
        assert app.request is not None
        assert app.request.manifest_path == DEFAULT_BUNDLED_REPLAY.manifest_path


async def test_primary_flow_is_keyboard_first(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        assert app.step is TuiStep.MODEL
        assert app.focused is app.query_one("#model-list", OptionList)

        await _highlight_profile(app, pilot, "qwen38-27b-q4-k-m")
        await pilot.press("enter")
        assert app.step is TuiStep.CONFIG
        assert app.selection.profile_id == "qwen38-27b-q4-k-m"

        await pilot.press("escape")
        assert app.step is TuiStep.MODEL
        app.query_one("#model-list", OptionList).focus()
        await pilot.press("end", "enter")
        assert app.step is TuiStep.CONFIG

        app.query_one("#endpoint-model-input", Input).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert app.step is TuiStep.PREFLIGHT
        assert app.focused is app.query_one("#endpoint-consent-checkbox", Checkbox)

        await pilot.press("space")
        await _settle_until(pilot, lambda: app.focused is app.query_one("#run-start", Button))
        await pilot.press("enter")
        await pilot.pause()
        assert app.step is TuiStep.RESULT
        assert app.outcome is TuiOutcome.SUCCESS

        await pilot.press("escape")
        assert app.step is TuiStep.CONFIG
        assert app.focused is app.query_one("#config-continue", Button)
        assert app.query_one("#output-input", Input).value == str(tmp_path / "private" / "results")


@pytest.mark.parametrize("size", ((48, 16), (60, 20), (96, 16)))
async def test_small_terminal_completes_mini_run_with_arrow_navigation(size: tuple[int, int], tmp_path: Path) -> None:
    release = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=release))

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        welcome = app.query_one("#welcome-start", Button)
        existing = app.query_one("#welcome-existing", Button)
        assert app.focused is welcome
        assert welcome.region.bottom <= existing.region.y
        assert existing.region.bottom < size[1]
        assert app.screen.active_bindings["question_mark"].binding.show
        assert not app.screen.active_bindings["f1"].binding.show
        await pilot.click("#welcome-existing", offset=(3, 1))
        assert app.step is TuiStep.CONFIG
        await pilot.press("escape")
        assert app.step is TuiStep.WELCOME
        await pilot.press("right")
        assert app.focused is existing
        await pilot.press("left")
        assert app.focused is welcome
        await pilot.press("down")
        assert app.focused is existing
        await pilot.press("up", "enter")
        assert app.step is TuiStep.MODEL
        assert app.query_one("#model-intro").region.y >= 0
        await pilot.press("right")
        assert app.focused is app.query_one("#model-detail-pane", VerticalScroll)
        await pilot.press("right", "enter")
        assert app.step is TuiStep.CONFIG

        await pilot.press("down", "down")
        replay = app.query_one("#replay-workload-select", ReplayWorkloadSelect)
        assert app.focused is replay
        # The full replay is first, so the quick mini check sits one below it.
        await pilot.press("enter", "home", "down", "enter")
        assert replay.value == "aa-mini-v1"
        await pilot.press("down")
        assert app.focused is app.query_one("#base-url-input", Input)
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.step is TuiStep.PREFLIGHT
        consent = app.query_one("#endpoint-consent-checkbox", Checkbox)
        assert app.focused is consent
        assert consent.region.bottom < size[1]
        await pilot.press("space")
        await _settle_until(pilot, lambda: app.focused is app.query_one("#run-start", Button))
        await pilot.press("enter")
        await _settle_until(pilot, lambda: app.progress_state.progress is not None)
        assert app.step is TuiStep.RUN
        assert app.query_one("#activity-live").region.bottom < size[1]
        assert app.query_one("#run-cancel").region.bottom < size[1]

        await pilot.press("?")
        assert app.step is TuiStep.METHODOLOGY
        await pilot.press("right")
        assert app.focused is app.query_one("#help-privacy", Button)
        await pilot.press("?")
        assert app.step is TuiStep.RUN
        release.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)
        assert app.outcome is TuiOutcome.SUCCESS
        assert app.query_one("#result-new").region.bottom < size[1]
        await pilot.press("escape")
        assert app.step is TuiStep.CONFIG
        assert replay.value == "aa-mini-v1"


async def test_arrow_keys_move_focus_across_closed_dropdowns_and_inputs(tmp_path: Path) -> None:
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        defaults=TuiDefaults(output_dir=tmp_path / "results", endpoint_model=ATTACHED_ENDPOINT_MODEL),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "enter")
        await pilot.pause()
        assert app.step is TuiStep.CONFIG
        assert app.focused is app.query_one("#config-continue", Button)

        replay_select = app.query_one("#replay-workload-select", ReplayWorkloadSelect)
        replay_select.focus()
        await pilot.pause()
        await pilot.press("down")
        assert not replay_select.expanded
        assert app.focused is app.query_one("#base-url-input", Input)

        await pilot.press("down")
        assert app.focused is app.query_one("#endpoint-model-input", Input)
        await pilot.press("down")
        assert app.focused is app.query_one("#api-key-env-input", Input)
        await pilot.press("up")
        assert app.focused is app.query_one("#endpoint-model-input", Input)

        app.query_one("#advanced-toggle", Button).focus()
        await pilot.press("enter", "down")
        client_select = app.query_one("#client-backend-select", ClientBackendSelect)
        assert app.focused is client_select
        client_select.focus()
        await pilot.pause()
        await pilot.press("down")
        assert not client_select.expanded
        assert app.focused is app.query_one("#config-continue", Button)

        await pilot.press("shift+tab", "up", "up")
        assert not client_select.expanded
        assert app.focused is app.query_one("#output-input", Input)


async def test_setup_field_focus_parks_the_cursor_and_blur_rewinds_the_view(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(72, 24)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "enter")
        await pilot.pause()
        output_input = app.query_one("#output-input", Input)
        assert len(output_input.value) > 40

        output_input.focus()
        await pilot.pause()
        # Focus must not paint the whole value as a selection band.
        assert output_input.cursor_position == len(output_input.value)
        assert output_input.selection.start == output_input.selection.end

        await pilot.press("tab")
        await pilot.pause()
        # Leaving the field rewinds it so the start of a long path stays readable.
        assert output_input.cursor_position == 0


@pytest.mark.parametrize("size", ((72, 24), (96, 30)))
async def test_setup_actions_stay_visible_and_help_returns_to_the_edit(size: tuple[int, int], tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("enter", "enter")
        await pilot.pause()
        page = app.query_one("#config", VerticalScroll)
        review = app.query_one("#config-continue", Button)
        assert not app.query_one("#client-row").display
        assert review.region.height == 1
        assert review.region.bottom < size[1]
        review_region = review.region

        app.query_one("#advanced-toggle", Button).focus()
        await pilot.press("enter")
        assert app.query_one("#client-row").display
        output = app.query_one("#output-input", Input)
        output.focus()
        await pilot.press("left", "left")
        cursor = output.cursor_position
        scroll_y = page.scroll_y
        await pilot.press("f1", "ctrl+p", "escape")
        await pilot.pause()
        assert app.step is TuiStep.CONFIG
        assert app.focused is output
        assert output.cursor_position == cursor
        assert page.scroll_y == scroll_y
        assert review.region == review_region

        await pilot.press("f1", "f1")
        assert app.focused is output
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.step is TuiStep.PREFLIGHT
        run = app.query_one("#run-start", Button)
        assert run.region.height == 1
        assert run.region.bottom < size[1]


@pytest.mark.parametrize("size", ((72, 24), (96, 32), (118, 40)))
async def test_existing_server_shortcut_and_detail_views_keep_the_run_flow_clear(
    size: tuple[int, int], tmp_path: Path
) -> None:
    release = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=release))

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("tab", "enter")
        assert app.step is TuiStep.CONFIG
        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        assert not app.query_one("#client-row").display
        output = app.query_one("#output-input", Input)
        assert output.region.y == app.query_one("#output-row Label").region.y
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await _confirm_and_run(app, pilot)
        assert not app.submit_requested
        await _settle_until(pilot, lambda: app.progress_state.progress is not None)
        kitty = app.query_one("#run-kitty", Static)
        assert str(kitty.content) == KITTY_CURIOUS
        assert kitty.region.right <= size[0]
        assert kitty.region.x >= app.query_one("#run-hero").region.right
        assert kitty.region.bottom <= app.query_one("#run-progress").region.y
        await pilot.press("d")
        assert app.query_one("#run-metrics-column").display
        assert app.query_one("#run-decode-chart").region.right <= size[0]
        # A scrollbar on the column must not squeeze a histogram row into a wrap.
        assert app.query_one("#run-decode-chart").content_region.width >= chart_width()
        assert app.query_one("#activity-live").region.height == 1
        assert app.query_one("#activity-live").region.bottom < size[1]
        await pilot.press("enter")
        assert app.replay_active
        assert not app.replay_cancelling
        await pilot.press("d")
        assert app.focused is app.query_one("#activity-lines", RichLog)

        release.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)
        await pilot.pause()
        kitty = app.query_one("#result-kitty", Static)
        assert str(kitty.content) == KITTY_HAPPY
        assert kitty.region.right <= size[0]
        assert kitty.region.x >= app.query_one("#result-title").region.right
        assert not app.query_one("#result-details").display
        details = app.query_one("#result-details-toggle", Button)
        details.focus()
        await pilot.press("enter")
        assert app.query_one("#result-details").display
        for chart in app.query_one("#result-charts").query(DistributionChart):
            assert chart.region.right <= size[0]
        assert app.query_one("#result-new", Button).region.bottom < size[1]
        await pilot.press("escape")
        assert app.step is TuiStep.CONFIG
        assert output.value == str(tmp_path / "private" / "results")
        await pilot.press("escape")
        assert app.step is TuiStep.WELCOME


@pytest.mark.parametrize("size", ((72, 24), (96, 30)))
async def test_arrow_keys_only_scroll_the_run_page(size: tuple[int, int], tmp_path: Path) -> None:
    release = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=release))

    async with app.run_test(size=size) as pilot:
        await _start_existing_server_run(app, pilot)
        await _settle_until(pilot, lambda: app.progress_state.progress is not None)
        # A short log cannot scroll yet, so a stray arrow must not walk focus onto Cancel.
        await pilot.press("down", "down", "up", "enter")
        await pilot.pause()
        assert app.step is TuiStep.RUN
        assert app.focused is app.query_one("#activity-lines", RichLog)
        assert not app.replay_cancelling
        assert not app.cancel_armed

        release.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)
        assert app.outcome is TuiOutcome.SUCCESS


async def test_a_second_run_starts_with_the_default_run_view(tmp_path: Path) -> None:
    release = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=release))

    async with app.run_test(size=(72, 24)) as pilot:
        await _start_existing_server_run(app, pilot)
        await _settle_until(pilot, lambda: app.progress_state.progress is not None)
        await pilot.press("d")
        assert app.query_one("#run").has_class("show-details")
        release.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)

        release.clear()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await _confirm_and_run(app, pilot)
        await _settle_until(pilot, lambda: app.step is TuiStep.RUN)
        await pilot.pause()
        # The compact run page opens on the activity log; the details view belongs to one run.
        assert not app.query_one("#run").has_class("show-details")
        assert app.query_one("#activity-lines", RichLog).display
        assert app.focused is app.query_one("#activity-lines", RichLog)
        release.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)


async def test_help_during_the_server_check_hands_focus_to_run_on_return(tmp_path: Path) -> None:
    probe_release = threading.Event()
    app = _app(tmp_path, FakeReplayController(probe_gate=probe_release))

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-existing")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        run = app.query_one("#run-start", Button)
        assert app.focused is app.query_one("#endpoint-consent-checkbox", Checkbox)
        assert run.disabled
        await pilot.press("f1")
        assert app.step is TuiStep.METHODOLOGY
        probe_release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not run.disabled
        # The server answered while Help was open, so Run gets the focus it would have taken then.
        await pilot.press("escape")
        await pilot.pause()
        assert app.step is TuiStep.PREFLIGHT
        assert app.focused is run

        # With nothing left to become ready, a detour returns to the control the user left.
        await pilot.press("tab")
        back = app.query_one("#preflight-back", Button)
        assert app.focused is back
        await pilot.press("f1", "escape")
        await pilot.pause()
        assert app.focused is back

        # An answer that arrives after the user moved on leaves their control focused.
        probe_release.clear()
        consent = app.query_one("#endpoint-consent-checkbox", Checkbox)
        consent.value = False
        await pilot.pause()
        assert run.disabled
        consent.focus()
        await pilot.pause()
        await pilot.press("space", "tab")
        submit = app.query_one("#submit-checkbox", Checkbox)
        assert app.focused is submit
        probe_release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not run.disabled
        assert app.focused is submit


async def test_form_pages_leave_the_arrow_keys_to_focus_movement(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(72, 24)) as pilot:
        await pilot.click("#welcome-existing")
        await pilot.pause()
        assert app.focused is app.query_one("#config-continue", Button)
        assert isinstance(app.query_one("#config"), FormPage)
        # A form page never claims the arrows, so even an overflowing form still moves between fields.
        assert app.screen.active_bindings["down"].binding.action == "focus_next"
        assert app.screen.active_bindings["up"].binding.action == "focus_previous"
        assert app.screen.active_bindings["pagedown"].binding.action == "page_down"
        await pilot.press("f1")
        await pilot.pause()
        back = app.query_one("#methodology-back", Button)
        assert app.focused is back
        # Reading pages keep the arrows for scrolling, and only real controls are focus stops.
        assert app.screen.active_bindings["down"].binding.action == "scroll_down"
        assert app.query_one("#help-privacy", Button).region.bottom <= 24
        await pilot.press("shift+tab")
        assert app.focused is app.query_one("#help-privacy", Button)
        await pilot.press("tab")
        assert app.focused is back


@pytest.mark.parametrize("animated", (True, False))
async def test_kitty_animation_keeps_its_shape_and_stops_off_screen(
    tmp_path: Path, animated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real poses hold for seconds; short holds keep every expression pollable and the test quick.
    monkeypatch.setattr(
        "agentperf_local.tui.widgets.KITTY_FRAMES",
        tuple(KittyFrame(art=frame.art, hold_seconds=0.1) for frame in KITTY_FRAMES),
    )
    release = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=release))
    app.animation_level = "full" if animated else "none"

    async with app.run_test(size=(72, 24)) as pilot:
        await _start_existing_server_run(app, pilot)
        await pilot.pause()
        kitty = app.query_one("#run-kitty", Kitty)
        region = kitty.region
        assert kitty.is_animating is animated
        if animated:
            for expression in (KITTY_BLINK, KITTY_GLANCE):
                await _settle_until(pilot, lambda: str(kitty.content) == expression)
                assert kitty.region == region

        await pilot.press("f1")
        assert app.step is TuiStep.METHODOLOGY
        assert not kitty.is_animating
        assert str(kitty.content) == KITTY_CURIOUS
        await pilot.press("escape")
        assert kitty.is_animating is animated
        # resize_terminal posts the event and pauses once, which a loaded runner can
        # outrun, so wait for on_resize to land rather than reading straight after it.
        # The assertion names every input _sync_kitty reads, to say which one was not
        # what the resize set.
        await pilot.resize_terminal(60, 20)
        deadline = time.monotonic() + UI_SETTLE_TIMEOUT_SECONDS
        while kitty.is_animating and time.monotonic() < deadline:
            await pilot.pause()
        pages = [node for node in kitty.ancestors_with_self if node.has_class("page")]
        assert not kitty.is_animating, (
            f"size={tuple(app.size)} display={kitty.display} too_small={app.terminal_too_small} "
            f"step={app.step} active={app.replay_active} short={[node.has_class('short') for node in pages]}"
        )
        await pilot.resize_terminal(72, 24)
        await _settle_until(pilot, lambda: kitty.is_animating is animated)

        release.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)
        assert not kitty.is_animating
        result_kitty = app.query_one("#result-kitty", Kitty)
        assert not result_kitty.is_animating
        assert str(result_kitty.content) == KITTY_HAPPY


class HeldTimer:
    """Stand in for a Textual timer whose tick the test delivers by hand."""

    def stop(self) -> None:
        pass


class KittyOnlyApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Kitty()


async def test_a_stale_kitty_tick_neither_restarts_nor_doubles_the_animation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Textual queues a fired timer's callback as a message, so a tick can land after the timer stopped.
    ticks: list[Callable[[], None]] = []

    def hold_tick(_kitty: Kitty, _delay: float, callback: Callable[[], None]) -> HeldTimer:
        ticks.append(callback)
        return HeldTimer()

    monkeypatch.setattr(Kitty, "set_timer", hold_tick)
    async with KittyOnlyApp().run_test() as pilot:
        kitty = pilot.app.query_one(Kitty)
        kitty.start()
        stale = ticks[-1]
        kitty.settle()
        stale()
        assert not kitty.is_animating

        kitty.start()
        scheduled = len(ticks)
        stale()
        assert len(ticks) == scheduled
        ticks[-1]()
        assert len(ticks) == scheduled + 1


async def test_dropdown_opens_with_enter_and_escape_closes_it_on_the_same_screen(tmp_path: Path) -> None:
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        defaults=TuiDefaults(output_dir=tmp_path / "results", endpoint_model=ATTACHED_ENDPOINT_MODEL),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "enter")
        await pilot.pause()
        replay_select = app.query_one("#replay-workload-select", ReplayWorkloadSelect)
        replay_select.focus()
        await pilot.pause()
        initial_value = replay_select.value

        await pilot.press("enter")
        assert replay_select.expanded

        await pilot.press("escape")
        assert not replay_select.expanded
        assert app.step is TuiStep.CONFIG
        assert app.focused is replay_select
        assert replay_select.value == initial_value

        # The custom entry is the last option, whatever the bundled replay count.
        await pilot.press("space", "end", "enter")
        assert not replay_select.expanded
        assert app.step is TuiStep.CONFIG
        assert replay_select.value == CUSTOM_REPLAY_ID
        await _settle_until(pilot, lambda: app.focused is app.query_one("#manifest-input", Input))


@pytest.mark.parametrize("key", ("q", "ctrl+c"))
async def test_idle_quit_shortcuts_exit_cleanly(tmp_path: Path, key: str) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(72, 24)) as pilot:
        await pilot.press(key)

    assert app.return_value is TuiOutcome.NO_RUN
    assert app.return_code == 0


@pytest.mark.parametrize("key", ("q", "?"))
async def test_plain_shortcut_keys_remain_editable_text_when_an_input_has_focus(tmp_path: Path, key: str) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        endpoint_model = app.query_one("#endpoint-model-input", Input)
        endpoint_model.value = ""
        endpoint_model.focus()
        await pilot.press(key)

        assert app.step is TuiStep.CONFIG
        assert app.is_running
        assert endpoint_model.value == key


@dataclass(slots=True, kw_only=True)
class FakeReplayController:
    """Drive deterministic public boundary events without a network."""

    endpoint_is_loopback: bool = True
    accelerator_name: str = "NVIDIA GeForce RTX 5090"
    start_gate: asyncio.Event | None = None
    started_event: asyncio.Event | None = None
    turn_event: asyncio.Event | None = None
    turn_gate: asyncio.Event | None = None
    gate: asyncio.Event | None = None
    execution_success: bool = True
    execution_error: str | None = None
    hardware_warning_count: int = 0
    # The size the recording carries for the turn the fake replays.
    recorded_prompt_tokens: int | None = 48_120
    finalization_gate: asyncio.Event | None = None
    finalization_error: bool = False
    results_writer: Callable[[Path], None] | None = None
    # A server that answers and lists the model but reports no context length is the
    # common case outside llama.cpp, so it is the fake's default.
    probe_result: ContextProbeResult = ContextProbeResult(
        observed_tokens=None, reason=ContextObservationReason.CONTEXT_NOT_REPORTED
    )
    probe_gate: threading.Event | None = None
    ollama: bool = False
    # Holds preflight in its worker thread until the test releases it.
    preflight_release: threading.Event | None = None
    # A named block carries no hardware, as a refused production preflight does not.
    block_code: PreflightBlockCode | None = None
    requests: list[ReplayRequest] = field(default_factory=list)
    probes: list[ReplayRequest] = field(default_factory=list)

    def probe_endpoint(self, request: ReplayRequest) -> ContextProbeResult:
        """Return the configured server answer without touching the network."""
        if self.probe_gate is not None:
            self.probe_gate.wait(UI_SETTLE_TIMEOUT_SECONDS)
        self.probes.append(request)
        return self.probe_result

    def detects_ollama(self, request: ReplayRequest) -> bool:
        """Return the configured server identity without touching the network."""
        del request
        return self.ollama

    def preflight(self, request: ReplayRequest) -> ReplayPreflight:
        """Return a stable local input check, blocked only when a block code is set."""
        if self.preflight_release is not None:
            self.preflight_release.wait(UI_SETTLE_TIMEOUT_SECONDS)
        if self.block_code is not None:
            return ReplayPreflight(
                ready=False,
                manifest_tasks=0,
                manifest_turns=0,
                endpoint_scope=request.endpoint_scope,
                reason="private probe detail",
                block_code=self.block_code,
                hardware=None,
            )
        return ReplayPreflight(
            ready=True,
            manifest_tasks=2,
            manifest_turns=4,
            endpoint_scope=(
                EndpointScope.LOOPBACK_NAME if self.endpoint_is_loopback else EndpointScope.NON_LOOPBACK_NAME
            ),
            reason=None,
            hardware=SafeHardwareSummary(
                operating_system="Linux",
                architecture="x86_64",
                accelerator_count=1,
                accelerator_name=self.accelerator_name,
                accelerator_memory_bytes=32 * 1024**3,
                warning_count=self.hardware_warning_count,
            ),
        )

    async def execute(self, request: ReplayRequest, observer: TuiReplayObserver) -> ReplayExecution:
        """Emit only the same boundaries as the production runner."""
        self.requests.append(request)
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.execution_error is not None:
            raise RuntimeError(self.execution_error)
        observer.on_activity(RunActivity(kind=RunActivityKind.SETUP_CHECKED))
        observer.on_activity(RunActivity(kind=RunActivityKind.SERVER_CHECKED, context_probe=self.probe_result))
        observer.on_activity(RunActivity(kind=RunActivityKind.REPLAY_STARTING))
        observer.on_boundary(RunStartedBoundary(tasks=2, turns=4))
        if self.started_event is not None:
            self.started_event.set()
        observer.on_boundary(
            TurnStartedBoundary(
                task=1,
                tasks=2,
                task_turn=1,
                task_turns=2,
                turn=1,
                turns=4,
                recorded_prompt_tokens=self.recorded_prompt_tokens,
            )
        )
        if self.turn_gate is not None:
            await self.turn_gate.wait()
        observer.on_boundary(
            TurnCompletedBoundary(
                task=1,
                tasks=2,
                task_turn=1,
                task_turns=2,
                turn=1,
                turns=4,
                elapsed_ms=1_000.0,
                time_to_first_token_ms=100.0,
                e2e_latency_ms=300.0,
                output_tokens=101,
                generation_time_ms=200.0,
                success=self.execution_success,
            )
        )
        if self.turn_event is not None:
            self.turn_event.set()
        if self.gate is not None:
            await self.gate.wait()
        observer.on_boundary(
            RunFinishedBoundary(
                completed_tasks=2,
                tasks=2,
                completed_turns=4,
                turns=4,
                elapsed_ms=4_000.0,
                success=self.execution_success,
            )
        )
        await observer.on_finalizing()
        if self.finalization_gate is not None:
            await self.finalization_gate.wait()
        if self.finalization_error:
            raise OSError("private report finalization detail")
        if self.results_writer is not None:
            self.results_writer(request.output_dir)
        return _fake_execution(request.output_dir, success=self.execution_success)


def _fake_execution(output_dir: Path, *, success: bool) -> ReplayExecution:
    """Build one committed replay result with deterministic headline timings."""
    return ReplayExecution(
        artifacts=ArtifactPaths(
            turns=output_dir / "turns.jsonl",
            tasks=output_dir / "tasks.json",
            tools=output_dir / "tools.json",
            failures=output_dir / "failures.json",
            summary=output_dir / "summary.json",
        ),
        output_dir=output_dir,
        output_tokens_per_second=1_234.0,
        ttft_p50_ms=100.0,
        e2e_p50_ms=300.0,
        failed_turns=0 if success else 1,
        total_turns=4,
    )


@dataclass(slots=True, kw_only=True)
class FakeManagedReplayController:
    """Drive the owned-deployment TUI path without starting a model server."""

    availability_result: ManagedModelAvailability
    started_event: asyncio.Event | None = None
    gate: asyncio.Event | None = None
    cleanup_event: asyncio.Event | None = None
    cleanup_error: bool = False
    artifact_progress: tuple[tuple[int, int], ...] = ()
    download_gate: asyncio.Event | None = None
    preflight_error: bool = False
    # A "{log}" placeholder receives the run's deployment log path, mirroring the
    # production _log_hint suffix on deployment RuntimeErrors.
    execution_error: str | None = None
    execution_error_type: type[Exception] = RuntimeError
    # A refusal raised before the child starts leaves no log; a server that started and
    # then failed leaves one. The result page names the log only in the second case.
    execution_error_writes_log: bool = True
    server_log_lines: tuple[str, ...] = ()
    # Writes real bound result files into the run folder, so the upload stage has a bundle to build.
    results_writer: Callable[[Path], None] | None = None
    requests: list[ReplayRequest] = field(default_factory=list)

    def hardware_summary(self, device_index: int | None = None) -> SafeHardwareSummary:
        """Return the hardware detected once at launch."""
        del device_index
        return self.availability_result.hardware

    def device_options(self) -> tuple[ManagedDeviceOption, ...]:
        """Offer no device choice; device picking is covered against the real controller."""
        return ()

    def availability(
        self,
        candidate: object,
        device_index: int | None = None,
        *,
        context_tokens: int | None = None,
        replay_floor_tokens: int | None = None,
    ) -> ManagedModelAvailability:
        """Return the configured machine compatibility result."""
        del candidate, device_index, context_tokens, replay_floor_tokens
        return self.availability_result

    def preflight(self, request: ReplayRequest) -> ReplayPreflight:
        """Allow only a compatible managed choice to reach execution."""
        if self.preflight_error:
            raise RuntimeError("private managed probe detail")
        return ReplayPreflight(
            ready=self.availability_result.can_deploy,
            manifest_tasks=1,
            manifest_turns=1,
            endpoint_scope=request.endpoint_scope,
            reason=self.availability_result.reason,
            block_code=None if self.availability_result.can_deploy else PreflightBlockCode.DEPLOYMENT_UNAVAILABLE,
            hardware=self.availability_result.hardware,
        )

    async def execute(self, request: ReplayRequest, observer: TuiReplayObserver) -> ReplayExecution:
        """Emit one replay and always signal owned-server cleanup."""
        self.requests.append(request)
        try:
            if self.execution_error is not None:
                log_path = request.output_dir / "deployment.log"
                if self.execution_error_writes_log:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text("server output\n", encoding="utf-8")
                raise self.execution_error_type(self.execution_error.format(log=log_path))
            observer.on_activity(RunActivity(kind=RunActivityKind.SETUP_CHECKED))
            observer.on_activity(RunActivity(kind=RunActivityKind.MODEL_CHECKING, artifact_bytes=4 * 1024**3))
            for downloaded_bytes, total_bytes in self.artifact_progress:
                observer.on_artifact_progress(downloaded_bytes, total_bytes)
            if self.download_gate is not None:
                await self.download_gate.wait()
            observer.on_activity(
                RunActivity(
                    kind=RunActivityKind.MODEL_READY,
                    artifact_bytes=4 * 1024**3,
                    downloaded=bool(self.artifact_progress),
                )
            )
            observer.on_activity(
                RunActivity(
                    kind=RunActivityKind.SERVER_STARTING,
                    framework="llama-cpp",
                    accelerator_platform="nvidia-cuda",
                    context_tokens=65_536,
                )
            )
            # Production opens the log as the child starts, so everything failing from
            # here on has one for the result page to name.
            log_path = request.output_dir / "deployment.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("server output\n", encoding="utf-8")
            if self.server_log_lines:
                observer.on_server_log(self.server_log_lines)
            observer.on_activity(
                RunActivity(
                    kind=RunActivityKind.SERVER_READY,
                    framework="llama-cpp",
                    context_tokens=65_536,
                    elapsed_seconds=14.2,
                )
            )
            observer.on_activity(RunActivity(kind=RunActivityKind.GPU_VERIFIED, accelerator_platform="nvidia-cuda"))
            observer.on_activity(RunActivity(kind=RunActivityKind.REPLAY_STARTING))
            observer.on_boundary(RunStartedBoundary(tasks=1, turns=1))
            observer.on_boundary(
                TurnStartedBoundary(
                    task=1,
                    tasks=1,
                    task_turn=1,
                    task_turns=1,
                    turn=1,
                    turns=1,
                    recorded_prompt_tokens=48_120,
                )
            )
            if self.started_event is not None:
                self.started_event.set()
            if self.gate is not None:
                await self.gate.wait()
            observer.on_boundary(
                TurnCompletedBoundary(
                    task=1,
                    tasks=1,
                    task_turn=1,
                    task_turns=1,
                    turn=1,
                    turns=1,
                    elapsed_ms=1_000.0,
                    time_to_first_token_ms=100.0,
                    e2e_latency_ms=300.0,
                    output_tokens=101,
                    generation_time_ms=200.0,
                    success=True,
                )
            )
            observer.on_boundary(
                RunFinishedBoundary(
                    completed_tasks=1,
                    tasks=1,
                    completed_turns=1,
                    turns=1,
                    elapsed_ms=1_000.0,
                    success=True,
                )
            )
            await observer.on_finalizing()
            output_dir = request.output_dir
            if self.results_writer is not None:
                self.results_writer(output_dir)
            return ReplayExecution(
                artifacts=ArtifactPaths(
                    turns=output_dir / "turns.jsonl",
                    tasks=output_dir / "tasks.json",
                    tools=output_dir / "tools.json",
                    failures=output_dir / "failures.json",
                    summary=output_dir / "summary.json",
                ),
                output_dir=output_dir,
                output_tokens_per_second=1_234.0,
                ttft_p50_ms=100.0,
                e2e_p50_ms=300.0,
                failed_turns=0,
                total_turns=1,
                deployment_record=output_dir / "deployment.json",
                gpu_startup_verified=True,
            )
        finally:
            observer.on_activity(RunActivity(kind=RunActivityKind.SERVER_STOPPING))
            observer.on_activity(RunActivity(kind=RunActivityKind.SERVER_STOPPED))
            if self.cleanup_event is not None:
                self.cleanup_event.set()
            if self.cleanup_error:
                raise OSError("private managed cleanup failure")


@dataclass(slots=True, kw_only=True)
class WedgedCleanupController:
    """Hold owned cleanup open so a cancelled run cannot finish on its own."""

    started: asyncio.Event
    cancelling: asyncio.Event
    release: asyncio.Event

    def probe_endpoint(self, request: ReplayRequest) -> ContextProbeResult:
        """Answer as a reachable server that lists the model without a context length."""
        del request
        return ContextProbeResult(observed_tokens=None, reason=ContextObservationReason.CONTEXT_NOT_REPORTED)

    def detects_ollama(self, request: ReplayRequest) -> bool:
        """Answer as a server that is not Ollama."""
        del request
        return False

    def preflight(self, request: ReplayRequest) -> ReplayPreflight:
        """Allow the run to start so its cancellation can wedge."""
        return ReplayPreflight(
            ready=True,
            manifest_tasks=1,
            manifest_turns=1,
            endpoint_scope=request.endpoint_scope,
            reason=None,
        )

    async def execute(self, request: ReplayRequest, observer: TuiReplayObserver) -> ReplayExecution:
        """Never acknowledge cancellation until the test releases cleanup."""
        del request, observer
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelling.set()
            await self.release.wait()
            raise
        raise AssertionError("a wedged replay never finishes on its own")


async def _highlight_profile(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome], profile_id: str) -> None:
    """Move the model list onto one named catalog model, whatever its catalog position."""
    index = next(position for position, model in enumerate(app.catalog.models) if model.profile_id == profile_id)
    model_list = app.query_one("#model-list", OptionList)
    model_list.focus()
    await pilot.press("home")
    for _ in range(index):
        await pilot.press("down")
    await pilot.pause()
    assert model_list.highlighted == index


def _app(tmp_path: Path, controller: FakeReplayController) -> AgentPerfLocalApp:
    return AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=controller,
        defaults=TuiDefaults(
            manifest_path=tmp_path / "private" / "manifest.json",
            output_dir=tmp_path / "private" / "results",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )


def _managed_availability(*, installed: bool = True, memory_fit: bool | None = True) -> ManagedModelAvailability:
    hardware = SafeHardwareSummary(
        operating_system="Darwin",
        architecture="arm64",
        accelerator_count=1,
        accelerator_name="Apple M5 Pro",
        accelerator_memory_bytes=24 * 1024**3,
        warning_count=0,
    )
    offer = FrameworkOffer(
        framework="llama-cpp",
        display_name="llama.cpp",
        accelerator_platform="apple-metal",
        installed=installed,
        available_memory_bytes=hardware.accelerator_memory_bytes,
        # The bundled Gemma 12B recipe minimum at the 65,536-token benchmark context.
        minimum_memory_bytes=10_020_944_000,
        memory_fit=memory_fit,
        installation_hint="Install a Metal-enabled llama.cpp build.",
        support_note="Native GGUF path using Metal.",
    )
    if installed and memory_fit is True:
        reason = None
    elif memory_fit is False:
        reason = "The detected accelerator does not have enough memory for this model."
    else:
        reason = offer.installation_hint
    return ManagedModelAvailability(hardware=hardware, offers=(offer,), reason=reason)


def _managed_artifact_gib(app: AgentPerfLocalApp) -> float:
    """Read the canonical artifact size the managed consent line must disclose."""
    deployment = next(model.deployment for model in app.catalog.models if model.deployment is not None)
    return deployment.artifact_size_bytes / 1024**3


async def _advance_to_preflight(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome]) -> None:
    """Walk from welcome to a settled preflight page, clicking the two continue buttons."""
    await pilot.click("#welcome-start")
    await pilot.click("#model-continue")
    await pilot.pause()
    app.query_one("#config-continue", Button).focus()
    await pilot.press("enter")
    await app.workers.wait_for_complete()


async def _start_run(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome]) -> None:
    """Walk the keyboard-only path from welcome to a started replay."""
    await pilot.press("enter")
    app.query_one("#model-continue", Button).focus()
    await pilot.press("enter")
    app.query_one("#config-continue", Button).focus()
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
    await pilot.pause()
    app.query_one("#run-start", Button).focus()
    await pilot.press("enter")


async def _confirm_and_run(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome]) -> None:
    """Tick the consent box on the confirm page and press Run once it takes focus."""
    await pilot.press("space")
    await _settle_until(pilot, lambda: app.focused is app.query_one("#run-start", Button))
    await pilot.press("enter")


async def _start_run_by_click(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome]) -> None:
    """Walk from welcome to a started replay through the focused controls."""
    await pilot.click("#welcome-start")
    await pilot.click("#model-continue")
    await pilot.pause()
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await _confirm_and_run(app, pilot)


async def _start_existing_server_run(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome]) -> None:
    """Take the welcome shortcut for an existing server through setup to a started replay."""
    await pilot.click("#welcome-existing")
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    await _confirm_and_run(app, pilot)


def _assert_render_omits(app: AgentPerfLocalApp, private_values: tuple[str, ...]) -> None:
    """Require the visible terminal frame to omit planted private values."""
    rendered = app.export_screenshot()
    for value in private_values:
        assert value not in rendered


async def test_keyboard_driven_candidate_run_preserves_honest_evidence(tmp_path: Path) -> None:
    controller = FakeReplayController(hardware_warning_count=2)
    app = _app(tmp_path, controller)

    async with app.run_test(size=(96, 30)) as pilot:
        # Every catalog model carries a managed recipe, so a run against a server the
        # user already operates goes through the custom-endpoint entry.
        await pilot.click("#welcome-start")
        assert app.query_one("#model-list", OptionList).highlighted == len(app.catalog.models)
        assert "Other model or server" in str(app.query_one("#model-detail", Static).content)
        await pilot.click("#model-continue")
        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        await pilot.pause()
        await pilot.press("enter")

        preflight = app.query_one("#preflight-status", Static)
        assert app.request is not None
        assert "This computer: NVIDIA GeForce RTX 5090 · 32 GiB · 2 warnings · Python client" in str(preflight.content)
        preflight_evidence = str(app.query_one("#preflight-evidence", Static).content)
        assert "Local server URL · model and GPU not verified" in preflight_evidence
        # An attached server's context is never observed before launch, so the page
        # says so in a muted line.
        assert ATTACHED_CONTEXT_UNVERIFIED_MESSAGE in preflight_evidence
        assert EligibilityReason.REDUCED_CONTEXT in app.evidence.ineligibility_reasons
        rendered = app.export_screenshot()
        assert "Ready" in rendered
        assert "NVIDIA&#160;GeForce&#160;RTX&#160;5090" in rendered
        consent = app.query_one("#endpoint-consent-checkbox", Checkbox)
        assert not consent.value
        assert str(consent.label) == CONSENT_ATTACHED_LABEL
        await pilot.press("space")
        await _settle_until(pilot, lambda: app.focused is app.query_one("#run-start", Button))
        assert not app.query_one("#run-start", Button).disabled
        await pilot.press("enter")
        await pilot.pause()

        assert app.step is TuiStep.RESULT
        assert app.selection.profile_id is None
        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        assert app.evidence.partition is ResultPartition.ATTACHED_ENDPOINT_EXPLORER
        assert len(controller.requests) == 1
        assert controller.requests[0].endpoint_model == ATTACHED_ENDPOINT_MODEL
        result_evidence = str(app.query_one("#result-evidence", Static).content)
        assert "model and GPU not verified" in result_evidence
        # The honest context state follows the run onto the result page, muted, while
        # the orange reduced-context card stays reserved for managed reduced launches.
        assert ATTACHED_CONTEXT_UNVERIFIED_MESSAGE in result_evidence
        assert not app.query_one("#result-reduced", Static).display
        assert "Run complete." in str(app.query_one("#result-title", Static).content)
        result_status = str(app.query_one("#result-status", Static).content).replace(PATH_WRAP_BREAK, "")
        assert str(tmp_path / "private" / "results") in result_status
        result_metrics = str(app.query_one("#result-metrics", Static).content)
        assert "4/4 turns · 4.0s elapsed" in result_metrics
        # The Digits headline is the only place throughput is printed.
        assert "median first token 100.0 ms · median turn 300.0 ms" in result_metrics
        assert "tokens/s" not in result_metrics
        # The headline counts up to its value over a short ease.
        await _settle_until(pilot, lambda: app.query_one("#result-throughput", Digits).value == "1,234")


async def test_external_catalog_never_receives_aa_candidate_provenance(tmp_path: Path) -> None:
    external_path = tmp_path / "external-catalog.json"
    external_path.write_bytes(
        CATALOG_PATH.read_bytes().replace(b"Google Gemma 4 12B IT QAT", b"[red]User catalog spoof[/red]")
    )
    catalog = load_model_catalog(external_path)
    app = AgentPerfLocalApp(catalog, controller=FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.pause()

        assert not catalog.is_bundled_snapshot
        assert app.selection.kind is SelectionKind.EXTERNAL_CATALOG_ENTRY
        detail = str(app.query_one("#model-detail", Static).content)
        assert "\\[red]User catalog spoof\\[/red]" in detail
        assert "Hugging Face" in detail
        assert "ARTIFICIAL ANALYSIS CATALOG" not in detail
        assert "QUALIFIED" not in detail


async def test_external_catalog_with_a_drifted_memory_pin_degrades_instead_of_crashing(tmp_path: Path) -> None:
    # An external catalog may pin a minimum the KV formula does not reproduce; every
    # render path must degrade to the pinned figure rather than crash the app.
    external_path = tmp_path / "external-catalog.json"
    external_path.write_bytes(
        CATALOG_PATH.read_bytes().replace(
            b'"minimum_memory_bytes": 10020944000', b'"minimum_memory_bytes": 11020944000'
        )
    )
    catalog = load_model_catalog(external_path)
    app = AgentPerfLocalApp(
        catalog,
        controller=FakeReplayController(),
        managed_controller=LocalManagedReplayController(
            catalog_as_of=catalog.as_of,
            hardware=_single_device_hardware(),
            offer_collector=_installed_llama_offer,
        ),
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")

        detail = str(app.query_one("#model-detail", Static).content)
        # The detail card survives, states the pinned figure, and admits the estimate gap.
        assert "needs 10.3 GiB" in detail
        assert "Memory estimate unavailable at reduced contexts." in detail

        await pilot.press("enter")
        await pilot.pause()
        context = app.query_one("#managed-context-select", ManagedContextSelect)
        # Without a per-context estimate no reduced rung is offered, only the pinned full context.
        assert context.value == "65536"
        assert len(context._options) == 1
        assert "full benchmark" in str(context.query_one("#label", Static).content)


async def test_gemma_profile_exposes_its_canonical_managed_artifact(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")

        detail = str(app.query_one("#model-detail", Static).content)
        assert "Google Gemma 4 12B IT QAT Q4_0 GGUF" in detail
        assert AA_CATALOG_RUNTIME_PROVENANCE in detail
        assert "Benchmark context" in detail
        assert "65,536 tokens" in detail
        assert "Runs with" in detail
        assert "llama-cpp" in detail
        assert "needs 9.4 GiB" in detail
        assert "Q4_0 GGUF · 6.5 GiB · SHA-256 checked" in detail


async def test_managed_candidate_selects_a_compatible_framework_and_saves_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    attached_controller = FakeReplayController()
    managed_controller = FakeManagedReplayController(availability_result=_managed_availability())
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=attached_controller,
        managed_controller=managed_controller,
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        await pilot.pause()

        framework = app.query_one("#managed-framework-select", ManagedFrameworkSelect)
        assert framework.value == "llama-cpp"
        assert not framework.disabled
        assert app.query_one("#base-url-input", Input).disabled
        assert app.query_one("#endpoint-model-input", Input).disabled
        assert app.query_one("#api-key-env-input", Input).disabled
        assert str(app.query_one("#section-server", Static).content) == "MODEL SERVER · started for you"
        assert not app.query_one("#base-url-row").display
        assert not app.query_one("#endpoint-model-row").display
        assert not app.query_one("#api-key-row").display
        assert app.query_one("#managed-context-row").display
        assert app.query_one("#managed-context-select", ManagedContextSelect).value == "65536"
        assert "This app will start the model server on this computer (Apple M5 Pro)" in str(
            app.query_one("#managed-deployment-status", Static).content
        )

        await _check_setup(app, pilot)
        assert "1 task · 1 turn" in str(app.query_one("#preflight-status", Static).content)
        assert str(app.query_one("#preflight-hero", Static).content) == PREFLIGHT_READY_HERO
        assert not app.query_one("#preflight-reduced", Static).display
        preflight_evidence = str(app.query_one("#preflight-evidence", Static).content)
        assert "exact model file and GPU startup are checked" in preflight_evidence
        assert "starts the server with llama.cpp" in preflight_evidence
        consent = app.query_one("#endpoint-consent-checkbox", Checkbox)
        assert not consent.value
        assert str(consent.label) == (
            f"Download {_managed_artifact_gib(app):.1f} GiB to the Hugging Face cache "
            f"at {tmp_path / 'hf-cache'} if needed and run the model locally."
        )
        # The download consent is never truncated: the long managed label wraps.
        assert consent.region.height >= 2
        consent.value = True
        await pilot.pause()
        assert app.focused is app.query_one("#run-start", Button)
        app.query_one("#run-start", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert attached_controller.requests == []
        assert len(managed_controller.requests) == 1
        request = managed_controller.requests[0]
        assert request.managed_deployment is not None
        assert request.managed_deployment.candidate.profile_id == "gemma4-12b-it-q4-0"
        assert request.managed_deployment.framework == "llama-cpp"
        assert app.outcome is TuiOutcome.SUCCESS
        assert app.execution is not None
        assert app.execution.gpu_startup_verified
        assert app.evidence.artifact is ArtifactEvidence.MANAGED_PINNED_ARTIFACT_VERIFIED
        assert app.evidence.partition is ResultPartition.MANAGED_LOCAL_EXPLORER
        assert EligibilityReason.REDUCED_CONTEXT not in app.evidence.ineligibility_reasons
        assert not app.query_one("#result-reduced", Static).display
        assert "exact model file and GPU startup evidence saved" in str(
            app.query_one("#result-evidence", Static).content
        )


async def test_managed_candidate_flags_incompatible_hardware_before_preflight(tmp_path: Path) -> None:
    availability = _managed_availability(memory_fit=False)
    managed_controller = FakeManagedReplayController(availability_result=availability)
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=managed_controller,
        defaults=TuiDefaults(output_dir=tmp_path / "results", endpoint_model=ATTACHED_ENDPOINT_MODEL),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        app.query_one("#model-list", OptionList).focus()
        await pilot.press("down", "down")
        detail = str(app.query_one("#model-detail", Static).content)
        assert "This computer" in detail
        assert "cannot start this model" in detail
        assert "does not have enough memory" in detail
        await pilot.press("enter")
        await pilot.pause()

        framework = app.query_one("#managed-framework-select", ManagedFrameworkSelect)
        assert framework.disabled
        assert "Can't start this model on this computer" in str(
            app.query_one("#managed-deployment-status", Static).content
        )
        assert "does not have enough memory" in str(app.query_one("#managed-deployment-status", Static).content)
        app.query_one("#config-continue", Button).press()
        await pilot.pause()

        assert app.step is TuiStep.PREFLIGHT
        assert app.request is None
        assert managed_controller.requests == []
        assert "does not have enough memory" in str(app.query_one("#preflight-status", Static).content)


@pytest.mark.parametrize(
    ("cleanup_error", "expected_outcome"),
    ((False, TuiOutcome.CANCELLED), (True, TuiOutcome.FAILED)),
)
async def test_cancelling_a_managed_run_waits_for_owned_server_cleanup(
    tmp_path: Path,
    cleanup_error: bool,
    expected_outcome: TuiOutcome,
) -> None:
    started_event = asyncio.Event()
    gate = asyncio.Event()
    cleanup_event = asyncio.Event()
    managed_controller = FakeManagedReplayController(
        availability_result=_managed_availability(),
        started_event=started_event,
        gate=gate,
        cleanup_event=cleanup_event,
        cleanup_error=cleanup_error,
    )
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=managed_controller,
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        await _check_setup(app, pilot)
        app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
        await pilot.pause()
        app.query_one("#run-start", Button).press()
        async with asyncio.timeout(WORKER_EVENT_TIMEOUT_SECONDS):
            await started_event.wait()

        await pilot.press("ctrl+c")
        async with asyncio.timeout(WORKER_EVENT_TIMEOUT_SECONDS):
            await cleanup_event.wait()
        await pilot.pause()

        run_dir = managed_controller.requests[0].output_dir
        assert run_dir.parent == tmp_path / "results"
        expected_evidence = (
            "Model server stopped and cleaned up · no benchmark result saved"
            if expected_outcome is TuiOutcome.CANCELLED
            else f"The model server run failed · check {run_dir / 'deployment.log'}"
        )
        assert app.outcome is expected_outcome
        assert app.execution is None
        evidence = str(app.query_one("#result-evidence", Static).content).replace(PATH_WRAP_BREAK, "")
        assert evidence == expected_evidence


@pytest.mark.parametrize(
    ("execution_error", "execution_error_type", "expected_status", "writes_log"),
    (
        (
            # The production readiness failure carries a log hint; the card keeps the
            # cause and drops the raw path, which the evidence line already names.
            "managed server did not report its served context length (meta.n_ctx); inspect {log}",
            RuntimeError,
            "managed server did not report its served context length (meta.n_ctx)",
            True,
        ),
        (
            "managed server reports a 8192-token context but the profile requires 65536; inspect {log}",
            RuntimeError,
            "managed server reports a 8192-token context but the profile requires 65536",
            True,
        ),
        (
            # A refusal raised before the child starts names the flag that resolves it,
            # and writes no log for the result page to send the reader to.
            "managed deployment port 8080 is already in use; pass --port to choose another",
            ValueError,
            "managed deployment port 8080 is already in use; pass --port to choose another",
            False,
        ),
        # A non-deployment exception type stays generic rather than leaking its text.
        ("private cleanup detail", OSError, MANAGED_RUN_STOPPED_MESSAGE, True),
    ),
)
async def test_managed_failure_names_its_safe_cause_instead_of_attached_advice(
    tmp_path: Path,
    execution_error: str,
    execution_error_type: type[Exception],
    expected_status: str,
    writes_log: bool,
) -> None:
    managed_controller = FakeManagedReplayController(
        availability_result=_managed_availability(),
        execution_error=execution_error,
        execution_error_type=execution_error_type,
        execution_error_writes_log=writes_log,
    )
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=managed_controller,
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        await _check_setup(app, pilot)
        app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
        await pilot.pause()
        app.query_one("#run-start", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.step is TuiStep.RESULT
        assert app.outcome is TuiOutcome.FAILED
        assert "The run stopped early." in str(app.query_one("#result-title", Static).content)
        status = str(app.query_one("#result-status", Static).content)
        assert status == expected_status
        # Attached-server advice would point at a server this app started itself.
        assert "Check that the server at your URL" not in status
        assert "inspect" not in status
        run_dir = managed_controller.requests[0].output_dir
        evidence = str(app.query_one("#result-evidence", Static).content).replace(PATH_WRAP_BREAK, "")
        expected_evidence = (
            f"The model server run failed · check {run_dir / 'deployment.log'}"
            if writes_log
            else "The model server run failed"
        )
        assert evidence == expected_evidence
        assert "private cleanup detail" not in app.export_screenshot()


async def test_candidate_alias_edit_downgrades_to_custom_intent(tmp_path: Path) -> None:
    controller = FakeReplayController()
    app = _app(tmp_path, controller)

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        app.query_one("#endpoint-model-input", Input).value = "user-chosen-serving-alias"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        assert app.request is not None
        assert app.request.selection_kind is SelectionKind.CUSTOM_ENDPOINT
        assert app.request.catalog_profile_id is None
        app.query_one("#preflight-back", Button).focus()
        await pilot.press("enter")
        app.query_one("#config-back", Button).focus()
        await pilot.press("enter")
        assert app.query_one("#model-list", OptionList).highlighted == len(app.catalog.models)
        assert "Other model or server" in str(app.query_one("#model-detail", Static).content)


async def test_cleartext_remote_endpoint_only_blocks_runs_that_send_a_key(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController(endpoint_is_loopback=False))

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.press("end", "enter")
        app.query_one("#base-url-input", Input).value = "http://192.168.1.5:8000/v1"
        app.query_one("#endpoint-model-input", Input).value = "private-model"
        app.query_one("#api-key-env-input", Input).value = "PRIVATE_TOKEN"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert app.step is TuiStep.PREFLIGHT
        assert app.request is None
        assert app.query_one("#run-start", Button).disabled
        assert str(app.query_one("#preflight-status", Static).content) == (
            "[b]This server is not local and an API key is set, so HTTPS is required.[/b]\n"
            "Use an https:// URL, or clear the API key variable.\n" + BLOCKED_ACTION_MESSAGE
        )
        assert "Nothing was sent" in str(app.query_one("#preflight-evidence", Static).content)

        app.query_one("#api-key-env-input", Input).value = ""
        await _check_setup(app, pilot)

        assert app.request is not None
        assert not app.request.endpoint_is_loopback


@pytest.mark.parametrize(
    ("base_url", "endpoint_model", "expected_status"),
    (
        (
            "http://127.0.0.1:8000/v1 with spaces",
            "served-model",
            "[b]The server URL is not valid.[/b]\n"
            "Use a plain address like http://127.0.0.1:8000/v1 with no credentials.",
        ),
        ("http://127.0.0.1:8000/v1", "", "[b]Enter the model name your server uses.[/b]"),
    ),
)
async def test_rejected_setup_names_its_cause_before_any_probe(
    tmp_path: Path,
    base_url: str,
    endpoint_model: str,
    expected_status: str,
) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        app.query_one("#base-url-input", Input).value = base_url
        app.query_one("#endpoint-model-input", Input).value = endpoint_model
        app.query_one("#config-continue", Button).press()
        await pilot.pause()

        assert app.step is TuiStep.PREFLIGHT
        assert app.request is None
        assert app.query_one("#run-start", Button).disabled
        status = str(app.query_one("#preflight-status", Static).content)
        assert status == f"{expected_status}\n{BLOCKED_ACTION_MESSAGE}"
        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        assert app.selection.profile_id is None
        assert app.query_one("#model-list", OptionList).highlighted == len(app.catalog.models)


@pytest.mark.parametrize(
    ("block_code", "expected_status"),
    (
        (
            PreflightBlockCode.OUTPUT_DIR_USED,
            "[b]The new run folder already holds files.[/b]\nGo back and continue again to get a fresh run folder.",
        ),
        (PreflightBlockCode.MANIFEST_UNREADABLE, "[b]The replay file could not be read.[/b]\nCheck the path."),
        (
            PreflightBlockCode.API_KEY_ENV_UNSET,
            "[b]The environment variable PRIVATE_TOKEN is not set.[/b]\nSet it before starting, or clear the field.",
        ),
        (PreflightBlockCode.INPUTS_INVALID, "[b]Can't run this setup.[/b]\nGo back and review the fields."),
        (
            PreflightBlockCode.CLIENT_UNAVAILABLE,
            "The Rust client is not installed. Go back and set Client to Python, or run `uv sync --extra rust`.",
        ),
    ),
)
async def test_blocked_preflight_names_its_cause(
    tmp_path: Path,
    block_code: PreflightBlockCode,
    expected_status: str,
) -> None:
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(block_code=block_code),
        defaults=TuiDefaults(
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "results",
            api_key_env="PRIVATE_TOKEN",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.request is None
        assert app.query_one("#run-start", Button).disabled
        consent = app.query_one("#endpoint-consent-checkbox", Checkbox)
        assert consent.disabled
        assert not consent.display
        preflight_status = app.query_one("#preflight-status", Static)
        assert preflight_status.has_class("error-card")
        assert str(app.query_one("#preflight-hero", Static).content) == PREFLIGHT_BLOCKED_HERO
        assert app.focused is app.query_one("#preflight-back", Button)
        status = str(preflight_status.content)
        assert status.startswith(expected_status)
        assert status.endswith(BLOCKED_ACTION_MESSAGE)
        assert "private probe detail" not in status


@pytest.mark.parametrize(
    ("selector", "expected_status"),
    (
        (
            "#manifest-input",
            "[b]Can't use this setup.[/b]\n"
            "Check the replay, results folder, server URL, model name, and API key environment variable.",
        ),
        ("#output-input", "[b]Enter a results folder.[/b]"),
    ),
)
async def test_blank_required_path_never_resolves_to_the_working_directory(
    tmp_path: Path,
    selector: str,
    expected_status: str,
) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        app.query_one(selector, Input).value = "   "
        app.query_one("#config-continue", Button).focus()
        await pilot.press("enter")
        await pilot.pause()

        assert app.step is TuiStep.PREFLIGHT
        assert app.request is None
        assert app.query_one("#run-start", Button).disabled
        status = str(app.query_one("#preflight-status", Static).content)
        assert status == f"{expected_status}\n{BLOCKED_ACTION_MESSAGE}"


async def test_preflight_worker_does_not_block_navigation_or_apply_a_stale_result(tmp_path: Path) -> None:
    release = threading.Event()
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(preflight_release=release),
        defaults=TuiDefaults(
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "results",
            base_url="https://remote.example/v1",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        await pilot.press("enter")

        assert app.step is TuiStep.PREFLIGHT
        assert "Checking files" in str(app.query_one("#preflight-status", Static).content)
        # The wait is animated so a slow probe still reads as alive.
        assert app.preflight_spinner_timer is not None
        assert app.focused is app.query_one("#preflight-back", Button)
        assert app.query_one("#endpoint-consent-checkbox", Checkbox).disabled
        assert "Remote server URL" in str(app.query_one("#preflight-evidence", Static).content)
        await pilot.click("#preflight-back")
        release.set()
        await pilot.pause()

        assert app.step is TuiStep.CONFIG
        assert app.request is None
        assert app.preflight_spinner_timer is None


async def test_preflight_completion_while_privacy_is_open_preserves_visible_focus(tmp_path: Path) -> None:
    release = threading.Event()
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(preflight_release=release),
        defaults=TuiDefaults(
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "results",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(72, 24)) as pilot:
        await pilot.press("enter")
        app.query_one("#model-continue", Button).focus()
        await pilot.press("enter")
        app.query_one("#config-continue", Button).focus()
        await pilot.press("enter")
        await pilot.press("ctrl+p")
        await pilot.pause()

        privacy_back = app.query_one("#privacy-back", Button)
        assert app.step is TuiStep.PRIVACY
        assert app.focused is privacy_back

        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.step is TuiStep.PRIVACY
        assert app.focused is privacy_back

        app.query_one("#privacy-back", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.step is TuiStep.PREFLIGHT
        assert app.focused is app.query_one("#endpoint-consent-checkbox", Checkbox)
        assert app.request is not None


async def test_preflight_blocks_when_controller_scope_disagrees_with_request(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController(endpoint_is_loopback=False))

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert app.request is None
        assert app.query_one("#run-start", Button).disabled
        assert "Setup changed while it was being checked" in str(app.query_one("#preflight-status", Static).content)


async def test_preflight_back_revokes_the_ready_request_and_consent(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await _advance_to_preflight(app, pilot)
        app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
        await pilot.pause()
        assert app.request is not None
        assert not app.query_one("#run-start", Button).disabled

        app.query_one("#preflight-back", Button).focus()
        await pilot.press("enter")
        await pilot.pause()

        assert app.step is TuiStep.CONFIG
        assert app.request is None
        assert app.query_one("#endpoint-consent-checkbox", Checkbox).disabled
        assert app.query_one("#run-start", Button).disabled


async def test_minimum_supported_terminal_keeps_primary_keyboard_actions_reachable(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(72, 24)) as pilot:
        await pilot.pause()
        assert app.focused is app.query_one("#welcome-start", Button)
        assert app.focused.region.height == 2
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused is app.query_one("#model-list", OptionList)
        rendered = app.export_screenshot()
        assert "STEP&#160;1&#160;OF&#160;4" in rendered
        assert "Choose&#160;a&#160;model" in rendered
        model_detail = app.query_one("#model-detail", Static)
        model_continue = app.query_one("#model-continue", Button)
        assert model_detail.region.intersection(model_continue.region).area == 0
        assert model_detail.region.height >= 5
        await pilot.press("right")
        assert app.focused is app.query_one("#model-detail-pane", VerticalScroll)
        await pilot.press("right")
        assert app.focused is model_continue
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused is app.query_one("#config-continue", Button)
        assert app.focused.region.bottom <= 24
        rendered = app.export_screenshot()
        assert "STEP&#160;2&#160;OF&#160;4" in rendered
        assert "Set&#160;up&#160;the&#160;run" in rendered
        await pilot.press("up")
        assert app.focused is app.query_one("#advanced-toggle", Button)
        await pilot.press("down")
        assert app.focused is app.query_one("#config-continue", Button)
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert app.step is TuiStep.PREFLIGHT
        endpoint_consent = app.query_one("#endpoint-consent-checkbox", Checkbox)
        assert app.focused is endpoint_consent
        assert endpoint_consent.region.right <= 72
        assert endpoint_consent.region.height == 1
        await pilot.press("space")
        await _settle_until(pilot, lambda: app.focused is app.query_one("#run-start", Button))
        submit = app.query_one("#submit-checkbox", Checkbox)
        notice = app.query_one("#submit-notice", Static)
        panel = app.query_one("#submit-panel", Vertical)
        assert submit.region.intersection(notice.region).area == 0
        assert notice.region.height >= 4
        assert submit.region.x >= panel.region.x
        assert notice.region.right <= panel.region.right

        await pilot.press("enter")
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)
        assert app.outcome is TuiOutcome.SUCCESS


async def test_remote_custom_run_is_service_latency_only_and_hides_private_run_fields(tmp_path: Path) -> None:
    gate = asyncio.Event()
    controller = FakeReplayController(endpoint_is_loopback=False, gate=gate)
    app = _app(tmp_path, controller)

    async with app.run_test(size=(118, 36)) as pilot:
        await pilot.click("#welcome-start")
        model_list = app.query_one("#model-list", OptionList)
        model_list.focus()
        await pilot.press("end", "enter")
        app.query_one("#base-url-input", Input).value = "https://secret-endpoint.example/v1"
        app.query_one("#endpoint-model-input", Input).value = "private-model-alias"
        app.query_one("#api-key-env-input", Input).value = "VERY_PRIVATE_TOKEN"
        planted_private_values = ("secret-endpoint", "private-model-alias", "VERY_PRIVATE_TOKEN")
        await pilot.pause()
        await pilot.press("enter")

        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        preflight_text = str(app.query_one("#preflight-status", Static).content)
        assert app.request is not None
        assert "This computer: NVIDIA GeForce RTX 5090 · 32 GiB" in preflight_text
        assert "Remote server URL · timing includes network delay" in str(
            app.query_one("#preflight-evidence", Static).content
        )
        assert app.evidence.partition is ResultPartition.SERVICE_LATENCY_ONLY
        _assert_render_omits(app, planted_private_values)
        app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
        await _settle_until(pilot, lambda: not app.query_one("#run-start", Button).disabled)
        await pilot.press("enter")
        assert app.step is TuiStep.RUN
        await pilot.pause(0.2)
        assert len(controller.requests) == 1

        run_text = " ".join(
            str(app.query_one(selector, Static).content) for selector in ("#run-counters", "#run-metrics", "#run-hero")
        )
        assert "private-model-alias" not in run_text
        assert "secret-endpoint" not in run_text
        assert "VERY_PRIVATE_TOKEN" not in run_text
        assert "first token 100 ms" in run_text
        _assert_render_omits(app, planted_private_values)
        gate.set()
        await pilot.pause()

        _assert_render_omits(app, planted_private_values)

        result_evidence = str(app.query_one("#result-evidence", Static).content)
        assert "Remote server · timing includes network delay" in result_evidence
        assert ATTACHED_CONTEXT_UNVERIFIED_MESSAGE in result_evidence


async def test_context_gauge_words_every_request_size_it_can_be_given(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(118, 36)):
        gauge = app.query_one("#run-context", ContextGauge)

        gauge.show_request(48_120, context_tokens=131_072)
        assert gauge.plain_text == "Next request · about 48,120 tokens ▕███████░░░░░░░░░░░░░▏ 37% of 131,072"
        assert gauge.display is True

        # A request larger than the window cannot fit, and the bar says so by filling.
        gauge.show_request(151_000, context_tokens=131_072)
        assert gauge.plain_text == "Next request · about 151,000 tokens ▕████████████████████▏ 115% of 131,072"

        # Without a served context length there is no honest denominator, so the size stands alone.
        gauge.show_request(48_120, context_tokens=None)
        assert gauge.plain_text == "Next request · about 48,120 tokens · context length not reported"

        # Nothing counted this prompt, so the line goes away rather than inventing a number.
        gauge.show_request(None, context_tokens=131_072)
        assert gauge.plain_text == ""
        assert gauge.display is False

        # A compact layout keeps the size and the share, and gives up the bar's tail.
        gauge.show_request(48_120, context_tokens=131_072, compact=True)
        assert gauge.plain_text == "Next · about 48,120 tokens ▕████░░░░░░▏ 37%"
        # The narrowest supported terminal still fits the line beside the status-row inset.
        assert len(gauge.plain_text) <= MINIMUM_SUPPORTED_WIDTH - 2


async def test_run_timeline_matches_the_latest_closed_boundary(tmp_path: Path) -> None:
    started_event = asyncio.Event()
    turn_event = asyncio.Event()
    turn_gate = asyncio.Event()
    finish_gate = asyncio.Event()
    app = _app(
        tmp_path,
        FakeReplayController(
            started_event=started_event,
            turn_event=turn_event,
            turn_gate=turn_gate,
            gate=finish_gate,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await _advance_to_preflight(app, pilot)
        await pilot.pause()
        app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
        await pilot.pause()
        run_start = app.query_one("#run-start", Button)
        assert not run_start.disabled
        run_start.focus()
        await pilot.press("enter")
        assert app.step is TuiStep.RUN
        async with asyncio.timeout(WORKER_EVENT_TIMEOUT_SECONDS):
            await started_event.wait()
        await pilot.pause()

        assert str(app.query_one("#run-hero", Static).content) == RUN_HERO_RUNNING
        # The announced request opens its task before any turn has closed.
        assert str(app.query_one("#run-counters", Static).content).startswith("Task 1/2 · turn 0/4")
        gauge = app.query_one("#run-context", ContextGauge)
        # This server never reported a context length, so the size stands without a bar.
        assert gauge.plain_text == "Next request · about 48,120 tokens · context length not reported"

        turn_gate.set()
        async with asyncio.timeout(WORKER_EVENT_TIMEOUT_SECONDS):
            await turn_event.wait()
        await pilot.pause()
        assert str(app.query_one("#run-hero", Static).content) == RUN_HERO_RUNNING
        assert str(app.query_one("#run-counters", Static).content).startswith("Task 1/2 · turn 1/4")
        # The size stays on screen while the closed turn is written up.
        assert gauge.plain_text == "Next request · about 48,120 tokens · context length not reported"

        await pilot.press("ctrl+c")
        assert app.outcome is TuiOutcome.CANCELLED


async def test_hardware_labels_cannot_inject_rich_evidence_markup(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController(accelerator_name="[b]AA VERIFIED[/b]"))

    async with app.run_test(size=(96, 30)) as pilot:
        await _advance_to_preflight(app, pilot)
        await pilot.pause()

        status = str(app.query_one("#preflight-status", Static).content)
        assert r"\[b]AA VERIFIED\[/b]" in status
        assert "[b]AA&#160;VERIFIED[/b]" in app.export_screenshot()


async def test_execution_failure_hides_private_error_and_disables_upload_action(tmp_path: Path) -> None:
    private_error = "SECRET_ENDPOINT_EXCEPTION private-model-alias"
    app = _app(tmp_path, FakeReplayController(execution_error=private_error))

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        await pilot.pause()

        assert app.step is TuiStep.RESULT
        assert app.outcome is TuiOutcome.FAILED
        assert app.execution is None
        assert app.focused is app.query_one("#result-new", Button)
        visible_text = " ".join(str(widget.content) for widget in app.query(Static))
        assert private_error not in visible_text
        assert private_error not in app.export_screenshot()
        assert "The run stopped early." in str(app.query_one("#result-title", Static).content)
        assert "Check that the server at your URL is running" in str(app.query_one("#result-status", Static).content)
        assert str(app.query_one("#result-evidence", Static).content) == "No files were written."
        assert str(app.query_one("#result-metrics", Static).content) == "No metrics available."


async def test_unsuccessful_report_set_uses_failure_outcome_and_style(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController(execution_success=False))

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        await pilot.pause()

        status = app.query_one("#result-status", Static)
        assert app.outcome is TuiOutcome.FAILED
        assert app.execution is not None
        assert status.has_class("error-card")
        assert "Run finished: 1 of 4 turns failed." in str(app.query_one("#result-title", Static).content)
        assert app.execution.artifacts.failures.parent.parent == tmp_path / "private" / "results"
        assert str(app.execution.artifacts.failures) in str(status.content).replace(PATH_WRAP_BREAK, "")
        assert "median first token 100.0 ms" in str(app.query_one("#result-metrics", Static).content)
        await _settle_until(pilot, lambda: app.query_one("#result-throughput", Digits).value == "1,234")
        assert app.query_one("#result-throughput", Digits).has_class("-muted")


async def test_privacy_and_methodology_shortcuts_restore_the_previous_step(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.press("ctrl+p")
        assert app.step is TuiStep.PRIVACY
        assert app.focused is app.query_one("#privacy-back", Button)
        await pilot.press("?")
        assert app.step is TuiStep.METHODOLOGY
        await pilot.press("ctrl+p")
        assert app.step is TuiStep.PRIVACY
        await pilot.press("enter")
        assert app.step is TuiStep.WELCOME

        await pilot.click("#welcome-start")
        await pilot.press("?")
        assert app.step is TuiStep.METHODOLOGY
        await _settle_until(pilot, lambda: app.focused is app.query_one("#methodology-back", Button))
        await pilot.click("#methodology-back")
        assert app.step is TuiStep.MODEL


@pytest.mark.parametrize(
    ("honours_ignore_eos", "answers_as_ollama", "expected_policy", "expected_posts"),
    [
        # The ignore_eos probe posts once and its control once; the replay's one turn posts last.
        (True, False, "exact", 3),
        # A server that drops ignore_eos still runs, under the recorded policy, and the
        # result page says so. The short probe answer settles it without a control.
        (False, False, "recorded", 2),
        # Ollama is named at the consent step, so no generation probe is sent at all.
        (False, True, "recorded", 1),
    ],
)
async def test_textual_worker_executes_a_real_localhost_sse_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    honours_ignore_eos: bool,
    answers_as_ollama: bool,
    expected_policy: str,
    expected_posts: int,
) -> None:
    manifest_path = write_replay_workload(tmp_path / "workload", name="Textual localhost test")
    output_dir = tmp_path / "results"
    api_key_env = "TUI_PRIVATE_TOKEN"
    api_key = "planted-secret-value-7c07a54b"
    monkeypatch.setenv(api_key_env, api_key)

    async with LocalSseServer(
        (SSE_OK_RESPONSE,),
        models=("local-test-model",),
        served_context_tokens=131_072,
        honours_ignore_eos=honours_ignore_eos,
        answers_as_ollama=answers_as_ollama,
    ) as server:
        app = AgentPerfLocalApp(
            load_model_catalog(CATALOG_PATH),
            defaults=TuiDefaults(
                manifest_path=manifest_path,
                output_dir=output_dir,
                base_url=f"{server.base_url}/",
                endpoint_model="local-test-model",
                api_key_env=api_key_env,
                client_backend="python",
            ),
        )
        async with app.run_test(size=(96, 30)) as pilot:
            await pilot.click("#welcome-start")
            await pilot.click("#model-continue")
            await _settle_until(pilot, lambda: app.focused is app.query_one("#config-continue", Button))
            assert app.step is TuiStep.CONFIG
            await pilot.press("enter")
            await pilot.pause()
            assert app.step is TuiStep.PREFLIGHT
            await app.workers.wait_for_complete()
            app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
            await _settle_until(pilot, lambda: not app.query_one("#run-start", Button).disabled)
            assert app.query_one("#preflight-ollama", Static).display is answers_as_ollama
            await pilot.press("enter")
            await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)

            assert app.execution is not None
            assert app.execution.success
            assert app.execution.output_dir.parent == output_dir
            assert app.execution.output_token_policy == expected_policy
            assert app.query_one("#result-policy", Static).display is (expected_policy != "exact")
            assert not app.query_one("#result-reduced", Static).display

    # The server checks are GET /models and the Ollama identity GETs, at consent and again before the run.
    # Only a server that answers /api/version like Ollama is asked for /api/tags.
    posts = [request for request in server.requests if request.method == "POST"]
    assert len(posts) == expected_posts
    replay_post = posts[-1]
    probes = [request for request in server.requests if request.method == "GET"]
    ollama_paths = {"/api/version", "/api/tags"} if answers_as_ollama else {"/api/version"}
    assert {probe.path for probe in probes} == {"/v1/models", *ollama_paths}
    assert all(request.headers["authorization"] == f"Bearer {api_key}" for request in server.requests)
    request_json = parse_json_object(replay_post.body, "captured TUI request")
    assert request_json.get("model") == "local-test-model"
    assert request_json.get("ignore_eos", False) is (expected_policy == "exact")
    # The chosen folder holds exactly one fresh run folder with the flat report
    # layout the packaging and bundle commands consume.
    (run_dir,) = tuple(output_dir.iterdir())
    assert run_dir.name.startswith("run-")
    assert {artifact.name for artifact in run_dir.iterdir()} == {
        "failures.json",
        "measurement.json",
        "summary.json",
        "tasks.json",
        "tools.json",
        "turns.jsonl",
    }
    assert (run_dir / "summary.json").is_file()
    assert b"prompt 0" not in (run_dir / "summary.json").read_bytes()
    summary = parse_json_object((run_dir / "summary.json").read_bytes(), "TUI summary")
    summary_config = summary.get("config")
    assert isinstance(summary_config, dict)
    assert summary_config.get("base_url") == server.base_url
    assert summary_config.get("model") == request_json.get("model")
    assert summary_config.get("client_backend") == "python"
    output_tokens = summary_config.get("output_tokens")
    assert isinstance(output_tokens, dict)
    assert output_tokens.get("policy") == expected_policy
    for artifact in run_dir.iterdir():
        if artifact.is_file():
            encoded = artifact.read_bytes()
            assert api_key.encode() not in encoded
            assert api_key_env.encode() not in encoded


async def test_cancel_discards_the_active_replay_without_resume(tmp_path: Path) -> None:
    gate = asyncio.Event()
    controller = FakeReplayController(gate=gate)
    app = _app(tmp_path, controller)

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        await pilot.pause()
        assert app.step is TuiStep.RUN
        assert app.focused is app.query_one("#activity-lines", RichLog)

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app.step is TuiStep.RESULT
        assert app.outcome is TuiOutcome.CANCELLED
        assert app.execution is None
        result_status = str(app.query_one("#result-status", Static).content)
        assert "No results were saved" in result_status
        assert "Run cancelled" in str(app.query_one("#result-evidence", Static).content)

        stale_generation = app.run_generation - 1
        output_dir = tmp_path / "private" / "results"
        app.post_message(ReplayCompletedMessage(_fake_execution(output_dir, success=True), stale_generation))
        await pilot.pause()

        assert app.execution is None
        assert "No results were saved" in str(app.query_one("#result-status", Static).content)


async def test_final_report_commit_cannot_be_cancelled(tmp_path: Path) -> None:
    finalization_gate = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(finalization_gate=finalization_gate))

    async with app.run_test(size=(72, 24)) as pilot:
        await _start_run(app, pilot)
        await pilot.pause()

        cancel = app.query_one("#run-cancel", Button)
        timeline = app.query_one("#activity-lines", RichLog)
        assert app.step is TuiStep.RUN
        assert app.replay_finalizing
        assert cancel.disabled
        assert "Saving" in str(cancel.label)
        assert app.focused is timeline
        await pilot.press("q")
        assert app.step is TuiStep.RUN
        assert app.replay_active
        assert app.focused is timeline
        await pilot.press("ctrl+c")
        assert app.step is TuiStep.RUN
        assert app.replay_active
        assert app.focused is timeline

        finalization_gate.set()
        await pilot.pause()

        assert app.step is TuiStep.RESULT
        assert app.outcome is TuiOutcome.SUCCESS


async def test_finalization_while_privacy_is_open_preserves_visible_keyboard_focus(tmp_path: Path) -> None:
    replay_gate = asyncio.Event()
    finalization_gate = asyncio.Event()
    app = _app(
        tmp_path,
        FakeReplayController(gate=replay_gate, finalization_gate=finalization_gate),
    )

    async with app.run_test(size=(72, 24)) as pilot:
        await _start_run(app, pilot)
        await pilot.press("ctrl+p")
        await pilot.pause()

        privacy_back = app.query_one("#privacy-back", Button)
        assert app.step is TuiStep.PRIVACY
        assert app.focused is privacy_back

        replay_gate.set()
        await pilot.pause()
        assert app.replay_finalizing
        assert app.step is TuiStep.PRIVACY
        assert app.focused is privacy_back

        app.query_one("#privacy-back", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.step is TuiStep.RUN
        assert app.focused is app.query_one("#activity-lines", RichLog)

        finalization_gate.set()
        await pilot.pause()
        assert app.step is TuiStep.RESULT
        assert app.outcome is TuiOutcome.SUCCESS


async def test_report_finalization_failure_is_uncommitted_and_requires_fresh_output(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController(finalization_error=True))

    async with app.run_test(size=(72, 24)) as pilot:
        await _start_run(app, pilot)
        await pilot.pause()

        assert app.step is TuiStep.RESULT
        assert app.outcome is TuiOutcome.FAILED
        assert "Could not save results" in str(app.query_one("#result-title", Static).content)
        assert "The run failed" in str(app.query_one("#result-status", Static).content)
        metrics = str(app.query_one("#result-metrics", Static).content)
        assert "4/4 turns · 4.0s elapsed" in metrics
        assert "first token 100.0 ms · total 300.0 ms" in metrics
        rendered = app.export_screenshot()
        assert "private report finalization detail" not in rendered
        assert "New&#160;run" in rendered
        assert "Quit" in rendered
        new_run = app.query_one("#result-new", Button)
        assert new_run.region.right <= 72 and new_run.region.bottom <= 24


@pytest.mark.parametrize(
    ("scenario", "expected_outcome", "rendered_status"),
    (
        ("success", TuiOutcome.SUCCESS, "Results&#160;saved"),
        ("failure", TuiOutcome.FAILED, "The&#160;run&#160;stopped&#160;early."),
        ("cancel", TuiOutcome.CANCELLED, "No&#160;results&#160;were&#160;saved"),
    ),
)
async def test_compact_result_keeps_outcome_and_actions_visible(
    tmp_path: Path,
    scenario: str,
    expected_outcome: TuiOutcome,
    rendered_status: str,
) -> None:
    gate = asyncio.Event() if scenario == "cancel" else None
    controller = FakeReplayController(
        gate=gate,
        execution_error="private failure detail" if scenario == "failure" else None,
    )
    app = _app(tmp_path, controller)

    async with app.run_test(size=(72, 24)) as pilot:
        await _start_run(app, pilot)
        await pilot.pause()
        if scenario == "cancel":
            await pilot.press("escape", "escape")
            await pilot.pause()

        assert app.step is TuiStep.RESULT
        assert app.outcome is expected_outcome
        rendered = app.export_screenshot()
        assert rendered_status in rendered
        if scenario == "success":
            assert "Run&#160;complete." in rendered
        assert "New&#160;run" in rendered
        new_run = app.query_one("#result-new", Button)
        assert new_run.region.right <= 72 and new_run.region.bottom <= 24


@pytest.mark.parametrize(
    ("size", "warning_visible"),
    (
        ((47, 16), True),
        ((48, 15), True),
        ((48, 16), False),
        ((60, 20), False),
        ((71, 23), False),
        ((72, 24), False),
        ((96, 30), False),
        ((118, 36), False),
    ),
)
async def test_terminal_size_contract(size: tuple[int, int], warning_visible: bool, tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=size) as pilot:
        warning = app.query_one("#small-terminal-warning", Static)
        content = app.query_one("#content", ContentSwitcher)
        assert warning.display is warning_visible
        assert content.display is not warning_visible

        if not warning_visible:
            # Shrinking below the minimum and growing back restores the welcome page,
            # its focus, and the compact brand header every supported size shows.
            await pilot.resize_terminal(MINIMUM_SUPPORTED_WIDTH - 1, MINIMUM_SUPPORTED_HEIGHT - 1)
            await pilot.pause()
            assert warning.display
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert not warning.display
            assert content.display
            assert str(app.query_one("#welcome-logo", PixelLogo).content) == PIXEL_LOGO_SMALL
            assert app.focused is app.query_one("#welcome-start", Button)


async def test_a_second_run_with_unchanged_settings_starts_clean_in_a_fresh_run_folder(tmp_path: Path) -> None:
    controller = FakeReplayController()
    app = _app(tmp_path, controller)

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        await pilot.pause()
        assert app.step is TuiStep.RESULT
        first_run_dir = controller.requests[0].output_dir
        # Stand in for the first run's committed reports so the next name must move on.
        first_run_dir.mkdir(parents=True)

        controller.start_gate = asyncio.Event()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#output-input", Input).value == str(tmp_path / "private" / "results")
        await _check_setup(app, pilot)
        await _start_run(app, pilot)

        progress = app.query_one("#run-progress", ProgressBar)
        assert app.step is TuiStep.RUN
        assert str(app.query_one("#run-hero", Static).content) == RUN_HERO_CHECKING_SERVER
        assert str(app.query_one("#run-counters", Static).content) == "Starting…"
        assert str(app.query_one("#run-metrics", Static).content) == RUN_METRICS_PLACEHOLDER
        assert progress.total == 1
        assert progress.progress == 0

        controller.start_gate.set()
        await pilot.pause()
        assert app.step is TuiStep.RESULT
        assert app.outcome is TuiOutcome.SUCCESS
        second_run_dir = controller.requests[1].output_dir
        assert second_run_dir != first_run_dir
        assert second_run_dir.name.startswith("run-")
        assert second_run_dir.parent == first_run_dir.parent == tmp_path / "private" / "results"


async def test_enter_pressed_twice_from_ready_never_cancels_the_run_it_started(tmp_path: Path) -> None:
    gate = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=gate))

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        await pilot.press("enter")
        await pilot.pause()

        assert app.step is TuiStep.RUN
        assert app.replay_active
        assert app.outcome is not TuiOutcome.CANCELLED

        gate.set()
        await pilot.pause()
        assert app.outcome is TuiOutcome.SUCCESS


@pytest.mark.parametrize(
    ("ending", "expected_outcome", "result_title"),
    (
        pytest.param("complete", TuiOutcome.SUCCESS, "Run complete.", id="completion"),
        pytest.param("cancel", TuiOutcome.CANCELLED, "Run cancelled.", id="cancellation"),
    ),
)
async def test_a_run_ending_while_privacy_is_open_keeps_the_page_and_returns_to_result(
    tmp_path: Path, ending: str, expected_outcome: TuiOutcome, result_title: str
) -> None:
    gate = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=gate))

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        await pilot.press("ctrl+p")
        await pilot.pause()
        privacy_back = app.query_one("#privacy-back", Button)
        assert app.step is TuiStep.PRIVACY

        if ending == "complete":
            gate.set()
        else:
            await pilot.press("ctrl+c")
        await pilot.pause()

        assert app.step is TuiStep.PRIVACY
        assert app.focused is privacy_back
        assert app.outcome is expected_outcome

        await pilot.press("escape")
        await pilot.pause()

        assert app.step is TuiStep.RESULT
        assert result_title in str(app.query_one("#result-title", Static).content)


async def test_quit_during_a_wedged_cancellation_can_still_force_an_exit(tmp_path: Path) -> None:
    controller = WedgedCleanupController(
        started=asyncio.Event(),
        cancelling=asyncio.Event(),
        release=asyncio.Event(),
    )
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=controller,
        defaults=TuiDefaults(
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "results",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        async with asyncio.timeout(WORKER_EVENT_TIMEOUT_SECONDS):
            await controller.started.wait()

        # q joins the same two-press arming as Esc, so one stray q cannot abort a run.
        await pilot.press("q")
        await pilot.pause()
        assert app.cancel_armed
        assert not app.replay_cancelling
        assert str(app.query_one("#run-metrics", Static).content) == CANCEL_CONFIRM_MESSAGE

        await pilot.press("q")
        async with asyncio.timeout(WORKER_EVENT_TIMEOUT_SECONDS):
            await controller.cancelling.wait()
        await pilot.pause()
        assert app.replay_cancelling
        assert app.step is TuiStep.RUN
        # The kitty settles the moment cancellation starts, not only after cleanup returns.
        assert not app.query_one("#run-kitty", Kitty).is_animating
        await pilot.press("q")
        await pilot.pause()
        assert app.is_running
        assert str(app.query_one("#run-metrics", Static).content) == (
            "Still cancelling. Press q again to force quit. The model server may be left running."
        )

        await pilot.press("q")
        controller.release.set()

    assert app.return_value is TuiOutcome.NO_RUN


async def test_user_endpoint_fields_survive_a_model_screen_round_trip(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        app.query_one("#endpoint-model-input", Input).value = "user-chosen-serving-alias"
        app.query_one("#config-back", Button).focus()
        await pilot.press("enter")
        app.query_one("#model-list", OptionList).focus()
        await pilot.press("end", "enter")
        await pilot.pause()

        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        assert app.query_one("#endpoint-model-input", Input).value == "user-chosen-serving-alias"

        app.query_one("#base-url-input", Input).value = "http://192.168.1.5:8000/v1"
        app.query_one("#api-key-env-input", Input).value = "PRIVATE_TOKEN"
        app.query_one("#config-back", Button).focus()
        await pilot.press("enter")
        # A managed candidate takes over the server fields, so the round trip goes
        # through one and back to the custom entry the typed values belong to.
        await _highlight_profile(app, pilot, "qwen38-27b-q4-k-m")
        await pilot.press("enter")
        await pilot.pause()
        assert app.selection.profile_id == "qwen38-27b-q4-k-m"
        assert app.query_one("#endpoint-model-input", Input).value == "qwen38-27b-q4-k-m"

        app.query_one("#config-back", Button).focus()
        await pilot.press("enter")
        app.query_one("#model-list", OptionList).focus()
        await pilot.press("end", "enter")
        await pilot.pause()

        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        assert app.query_one("#base-url-input", Input).value == "http://192.168.1.5:8000/v1"
        assert app.query_one("#api-key-env-input", Input).value == "PRIVATE_TOKEN"
        # The managed detour hands the model name back to the launch default.
        assert app.query_one("#endpoint-model-input", Input).value == ATTACHED_ENDPOINT_MODEL


@pytest.mark.parametrize("cli_endpoint_model", (None, "cli-endpoint-model"))
async def test_leaving_the_managed_model_never_carries_its_alias_into_a_custom_run(
    tmp_path: Path,
    cli_endpoint_model: str | None,
) -> None:
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=FakeManagedReplayController(availability_result=_managed_availability()),
        defaults=TuiDefaults(output_dir=tmp_path / "results", endpoint_model=cli_endpoint_model),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        model_list = app.query_one("#model-list", OptionList)
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#endpoint-model-input", Input).value == "gemma4-12b-it-q4-0"

        app.query_one("#config-back", Button).focus()
        await pilot.press("enter")
        model_list.focus()
        await pilot.press("end", "enter")
        await pilot.pause()

        assert app.selection.kind is SelectionKind.CUSTOM_ENDPOINT
        assert app.query_one("#endpoint-model-input", Input).value == (cli_endpoint_model or "")
        assert not app.query_one("#endpoint-model-input", Input).disabled
        assert str(app.query_one("#section-server", Static).content) == "YOUR SERVER"
        assert app.query_one("#api-key-row").display
        assert not app.query_one("#managed-context-row").display


@pytest.mark.parametrize("typed_output", ("~/agentperf-local-results", "agentperf-local-results"))
async def test_typed_results_folder_expands_home_and_names_the_saved_run_folder(
    tmp_path: Path, typed_output: str
) -> None:
    controller = FakeReplayController()
    app = _app(tmp_path, controller)
    expected_output_dir = Path(typed_output).expanduser()

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        app.query_one("#output-input", Input).value = typed_output
        await _check_setup(app, pilot)

        assert app.request is not None
        run_dir = app.request.output_dir
        assert run_dir.parent == expected_output_dir
        assert run_dir.name.startswith("run-")
        assert not Path("~").exists()

        await _start_run(app, pilot)

        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)
        assert app.execution is not None
        assert app.execution.output_dir.parent == expected_output_dir
        # The saved-to path shortens home to ~, except on Windows, and wraps only at directory boundaries.
        absolute_run_dir = app.execution.output_dir.absolute()
        shortened = sys.platform != "win32" and absolute_run_dir.is_relative_to(Path.home())
        expected_display = (
            str(Path("~") / absolute_run_dir.relative_to(Path.home())) if shortened else str(absolute_run_dir)
        )
        displayed = str(app.query_one("#result-status", Static).content).replace(PATH_WRAP_BREAK, "")
        assert expected_display in displayed
        assert not shortened or str(Path.home()) not in displayed


async def test_copied_result_path_contains_no_wrap_break_characters(tmp_path: Path) -> None:
    controller = FakeReplayController()
    app = _app(tmp_path, controller)

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        await pilot.pause()
        assert app.step is TuiStep.RESULT

        status = app.query_one("#result-status", Static)
        # The rendered text steers wrapping with the zero-width break character…
        assert PATH_WRAP_BREAK in str(status.content)
        selections: dict[Widget, Selection] = {status: Selection(None, None)}
        app.screen.selections = selections
        selected = app.screen.get_selected_text()

        # …but a copy must paste a path a shell can resolve.
        assert selected is not None
        assert PATH_WRAP_BREAK not in selected
        assert str(controller.requests[0].output_dir) in selected


async def test_out_of_range_managed_port_names_the_launch_settings(tmp_path: Path) -> None:
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=FakeManagedReplayController(availability_result=_managed_availability()),
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            deployment_port=80,
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        app.query_one("#config-continue", Button).press()
        await pilot.pause()

        assert app.step is TuiStep.PREFLIGHT
        assert app.request is None
        assert str(app.query_one("#preflight-status", Static).content) == (
            f"Launch settings are not valid: the --port value must be between "
            f"{MINIMUM_USER_PORT} and {MAXIMUM_PORT}. "
            "Start the app again with a different --port or --startup-timeout-seconds.\n" + BLOCKED_ACTION_MESSAGE
        )


@pytest.mark.parametrize(
    ("downloaded_bytes", "expected_progress"),
    (
        (3 * 1024**3 // 2, "Downloading the model · 1.5 / 4.0 GiB"),
        (4 * 1024**3, "Downloading the model · 4.0 / 4.0 GiB"),
    ),
)
@pytest.mark.parametrize("size", ((48, 16), (96, 30)))
async def test_managed_download_progress_replaces_the_activity_spinner(
    tmp_path: Path,
    downloaded_bytes: int,
    expected_progress: str,
    size: tuple[int, int],
) -> None:
    total_bytes = 4 * 1024**3
    download_gate = asyncio.Event()
    managed_controller = FakeManagedReplayController(
        availability_result=_managed_availability(),
        artifact_progress=((downloaded_bytes, total_bytes),),
        download_gate=download_gate,
    )
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=managed_controller,
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=size) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        await _check_setup(app, pilot)
        await _start_run(app, pilot)

        progress = app.query_one("#activity-progress", ProgressBar)
        # The stepper names the deploy phase so it survives compact scrolling.
        assert str(app.query_one("#run-eyebrow", Static).content) == "STEP 4 OF 4 · RUN · PREPARING"
        assert not app.query_one("#run-progress-row").display
        live = app.query_one("#activity-live", Static)
        assert str(live.content) == expected_progress
        assert progress.display
        await _settle_until(pilot, lambda: live.region.bottom == progress.region.y)
        assert progress.region.bottom < size[1]
        assert progress.total == total_bytes
        assert progress.progress == downloaded_bytes

        download_gate.set()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert str(app.query_one("#run-eyebrow", Static).content) == "STEP 4 OF 4 · RUN"
        assert not progress.display
        assert app.outcome is TuiOutcome.SUCCESS


async def test_attached_preflight_reuses_the_launch_hardware_snapshot(tmp_path: Path) -> None:
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(block_code=PreflightBlockCode.OUTPUT_DIR_USED),
        managed_controller=FakeManagedReplayController(availability_result=_managed_availability()),
        defaults=TuiDefaults(
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "results",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        status = str(app.query_one("#preflight-status", Static).content)
        assert "This computer: Apple M5 Pro · 24 GiB · Python client" in status
        assert "hardware unavailable" not in status


async def test_help_page_and_footer_name_the_overlay_shortcuts(tmp_path: Path) -> None:
    app = _app(tmp_path, FakeReplayController())

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.press("?")
        await pilot.pause()

        assert app.step is TuiStep.METHODOLOGY
        help_text = "\n".join(str(widget.content) for widget in app.query_one("#methodology").query(Static))
        assert "Ctrl+P" in help_text
        assert "open the privacy page" in help_text
        app.query_one("#help-privacy", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.step is TuiStep.PRIVACY


def _accelerator(name: str, memory_bytes: int | None) -> AcceleratorSnapshot:
    return AcceleratorSnapshot(
        vendor="NVIDIA",
        name=name,
        memory_bytes=memory_bytes,
        core_count=None,
        driver_version="private-driver",
        api="CUDA",
    )


def _hardware_with(*accelerators: AcceleratorSnapshot) -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system="Linux",
        operating_system_version="test",
        kernel_version="private-kernel",
        architecture="x86_64",
        cpu_model="test-cpu",
        logical_cpu_count=8,
        memory_bytes=64 * 1024**3,
        accelerators=accelerators,
        warnings=(),
    )


def _single_device_hardware() -> HardwareSnapshot:
    return _hardware_with(_accelerator("NVIDIA GeForce RTX 5090", 32 * 1024**3))


def _multi_device_hardware() -> HardwareSnapshot:
    return _hardware_with(
        _accelerator("NVIDIA GeForce RTX 5090", 32 * 1024**3),
        _accelerator("NVIDIA RTX PRO 6000", 96 * 1024**3),
        _accelerator("NVIDIA GeForce RTX 4090", None),
    )


def _installed_llama_offer(
    hardware: HardwareSnapshot,
    candidate: ModelCandidate,
    context_tokens: int | None = None,
) -> tuple[FrameworkOffer, ...]:
    """Offer an installed llama.cpp build sized against the single bound accelerator."""
    deployment = candidate.deployment
    assert deployment is not None
    minimum_memory_bytes = derived_minimum_memory_bytes(
        deployment, deployment.context_tokens if context_tokens is None else context_tokens
    )
    available_memory_bytes = hardware.accelerators[0].memory_bytes
    return (
        FrameworkOffer(
            framework="llama-cpp",
            display_name="llama.cpp",
            accelerator_platform="nvidia-cuda",
            installed=True,
            available_memory_bytes=available_memory_bytes,
            minimum_memory_bytes=minimum_memory_bytes,
            memory_fit=(None if available_memory_bytes is None else available_memory_bytes >= minimum_memory_bytes),
            installation_hint="Install llama.cpp.",
            support_note="Native GGUF path.",
        ),
    )


def _installed_sglang_offer(
    hardware: HardwareSnapshot,
    candidate: ModelCandidate,
    context_tokens: int | None = None,
) -> tuple[FrameworkOffer, ...]:
    """Offer an installed SGLang build for a weights recipe on the bound accelerator."""
    deployment = candidate.deployment
    assert deployment is not None
    minimum_memory_bytes = derived_minimum_memory_bytes(
        deployment, deployment.context_tokens if context_tokens is None else context_tokens
    )
    available_memory_bytes = hardware.accelerators[0].memory_bytes
    return (
        FrameworkOffer(
            framework="sglang",
            display_name="SGLang",
            accelerator_platform="nvidia-cuda",
            installed=True,
            available_memory_bytes=available_memory_bytes,
            minimum_memory_bytes=minimum_memory_bytes,
            memory_fit=(None if available_memory_bytes is None else available_memory_bytes >= minimum_memory_bytes),
            installation_hint="Install SGLang.",
            support_note="Native weights path.",
        ),
    )


def _device_app(
    tmp_path: Path,
    hardware: HardwareSnapshot,
    *,
    device_index: int | None = None,
) -> AgentPerfLocalApp:
    """Build the app against a real managed controller with an injected computer."""
    catalog = load_model_catalog(CATALOG_PATH)
    return AgentPerfLocalApp(
        catalog,
        controller=FakeReplayController(),
        managed_controller=LocalManagedReplayController(
            catalog_as_of=catalog.as_of,
            hardware=hardware,
            offer_collector=_installed_llama_offer,
        ),
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            device_index=device_index,
        ),
    )


async def _select_managed_model(app: AgentPerfLocalApp, pilot: Pilot[TuiOutcome]) -> None:
    """Walk the guided flow to the setup screen with the managed Gemma candidate selected."""
    await pilot.click("#welcome-start")
    await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.parametrize(
    ("hardware", "device_index", "picked", "expected_status", "expected_launch_device", "expected_computer"),
    (
        pytest.param(
            _single_device_hardware(),
            None,
            None,
            "This app will start the model server on this computer (NVIDIA GeForce RTX 5090)",
            None,
            "This computer: NVIDIA GeForce RTX 5090 · 32 GiB",
            id="single-device-needs-no-picker",
        ),
        pytest.param(
            _multi_device_hardware(),
            None,
            None,
            DEVICE_SELECTION_REQUIRED_MESSAGE,
            None,
            None,
            id="multi-device-waits-for-a-choice",
        ),
        pytest.param(
            _multi_device_hardware(),
            1,
            None,
            "This app will start the model server on this computer (NVIDIA RTX PRO 6000)",
            1,
            "This computer: NVIDIA RTX PRO 6000 · 96 GiB",
            id="device-named-at-launch",
        ),
        pytest.param(
            _multi_device_hardware(),
            None,
            "1",
            "This app will start the model server on this computer (NVIDIA RTX PRO 6000)",
            1,
            "This computer: NVIDIA RTX PRO 6000 · 96 GiB",
            id="device-picked-on-setup",
        ),
    ),
)
async def test_managed_launch_is_pinned_to_the_device_this_computer_offers(
    tmp_path: Path,
    hardware: HardwareSnapshot,
    device_index: int | None,
    picked: str | None,
    expected_status: str,
    expected_launch_device: int | None,
    expected_computer: str | None,
) -> None:
    app = _device_app(tmp_path, hardware, device_index=device_index)

    async with app.run_test(size=(96, 30)) as pilot:
        await _select_managed_model(app, pilot)
        if picked is not None:
            app.query_one("#managed-device-select", ManagedDeviceSelect).value = picked
            await pilot.pause()

        status = str(app.query_one("#managed-deployment-status", Static).content)
        assert app.query_one("#managed-device-row").display is (len(hardware.accelerators) > 1)
        assert app.query_one("#managed-framework-select", ManagedFrameworkSelect).disabled is (
            expected_computer is None
        )
        assert status.startswith(expected_status)

        await _check_setup(app, pilot)

        assert app.step is TuiStep.PREFLIGHT
        preflight_status = str(app.query_one("#preflight-status", Static).content)
        if expected_computer is None:
            assert app.request is None
            assert preflight_status == f"{DEVICE_SELECTION_REQUIRED_MESSAGE}\n{BLOCKED_ACTION_MESSAGE}"
        else:
            assert app.request is not None
            assert app.request.managed_deployment is not None
            assert app.request.managed_deployment.device_index == expected_launch_device
            assert expected_computer in preflight_status


async def test_device_picker_labels_every_detected_accelerator(tmp_path: Path) -> None:
    app = _device_app(tmp_path, _multi_device_hardware())

    async with app.run_test(size=(96, 30)) as pilot:
        await _select_managed_model(app, pilot)
        device = app.query_one("#managed-device-select", ManagedDeviceSelect)
        labels: list[str] = []
        for value in ("0", "1", "2"):
            device.value = value
            await pilot.pause()
            labels.append(str(device.query_one("#label", Static).content))

        assert labels == [
            "0 · NVIDIA GeForce RTX 5090 · 32 GiB",
            "1 · NVIDIA RTX PRO 6000 · 96 GiB",
            "2 · NVIDIA GeForce RTX 4090 · memory not reported",
        ]


@pytest.mark.parametrize(
    ("memory_gib", "replay_id", "expected_value", "expected_label"),
    (
        (32, "agentperf-default-v1", "65536", "65,536 tokens · full benchmark · needs 9.4 GiB"),
        # 9 GiB misses the full context (9.33 GiB) but fits 32,768 tokens (8.9 GiB),
        # which the mini replay's floor allows, so the default is that reduced option.
        (9, "aa-mini-v1", "32768", "32,768 tokens · needs 8.9 GiB · not comparable"),
    ),
)
async def test_context_picker_defaults_to_the_largest_context_the_device_fits(
    tmp_path: Path,
    memory_gib: int,
    replay_id: str,
    expected_value: str,
    expected_label: str,
) -> None:
    app = _device_app(tmp_path, _hardware_with(_accelerator("NVIDIA test GPU", memory_gib * 1024**3)))

    async with app.run_test(size=(96, 30)) as pilot:
        await _select_managed_model(app, pilot)
        app.query_one("#replay-workload-select", ReplayWorkloadSelect).value = replay_id
        await pilot.pause()

        context = app.query_one("#managed-context-select", ManagedContextSelect)
        assert app.query_one("#managed-context-row").display
        assert context.value == expected_value
        assert str(context.query_one("#label", Static).content) == expected_label
        # The default always leaves a working launch, never a memory dead end.
        assert not app.query_one("#managed-framework-select", ManagedFrameworkSelect).disabled
        status = str(app.query_one("#managed-deployment-status", Static).content)
        assert "This app will start the model server on this computer" in status


async def test_custom_replay_floor_blocks_a_smaller_managed_context_before_launch(tmp_path: Path) -> None:
    manifest_path = write_replay_workload(
        tmp_path / "workload",
        name="floored custom replay",
        write_traces=False,
        required_context_tokens=65_536,
    )
    app = _device_app(tmp_path, _single_device_hardware())

    async with app.run_test(size=(96, 30)) as pilot:
        await _select_managed_model(app, pilot)
        app.query_one("#replay-workload-select", ReplayWorkloadSelect).value = CUSTOM_REPLAY_ID
        await _settle_until(pilot, lambda: app.focused is app.query_one("#manifest-input", Input))
        app.query_one("#manifest-input", Input).value = str(manifest_path)
        # A custom manifest is only read at preflight, so the picker cannot filter for it.
        app.query_one("#managed-context-select", ManagedContextSelect).value = "32768"
        await pilot.pause()
        await _check_setup(app, pilot)

        assert app.step is TuiStep.PREFLIGHT
        assert app.request is None
        status = str(app.query_one("#preflight-status", Static).content)
        # The block names both numbers instead of letting the launch fail mid-run.
        assert "needs at least 65,536 tokens of context" in status
        assert "would start with 32,768" in status
        assert "Go back and pick a larger context." in status
        # A blocked page drops the cheerful hero and the reduced-context notice.
        assert str(app.query_one("#preflight-hero", Static).content) == PREFLIGHT_BLOCKED_HERO
        assert not app.query_one("#preflight-reduced", Static).display


async def test_selecting_the_full_context_on_a_small_device_names_the_reduced_escape_hatch(tmp_path: Path) -> None:
    app = _device_app(tmp_path, _hardware_with(_accelerator("NVIDIA test GPU", 9 * 1024**3)))

    async with app.run_test(size=(96, 30)) as pilot:
        await _select_managed_model(app, pilot)
        app.query_one("#replay-workload-select", ReplayWorkloadSelect).value = "aa-mini-v1"
        await pilot.pause()
        app.query_one("#managed-context-select", ManagedContextSelect).value = "65536"
        await pilot.pause()

        status = str(app.query_one("#managed-deployment-status", Static).content)
        assert "Not enough accelerator memory for the full 65,536-token context." in status
        assert "reduced runs are recorded separately from full-context results" in status
        assert app.query_one("#managed-framework-select", ManagedFrameworkSelect).disabled
        # The explicit full choice sticks so its refusal stays readable.
        assert app.query_one("#managed-context-select", ManagedContextSelect).value == "65536"


@pytest.mark.parametrize(
    ("hardware", "device_index"),
    (
        (_single_device_hardware(), None),
        # Multiple accelerators add the device row; a chosen device also shows the
        # full launch status, which is the tallest managed form this page renders.
        (_multi_device_hardware(), 0),
    ),
)
async def test_managed_setup_with_the_context_row_fits_the_minimum_terminal(
    tmp_path: Path,
    hardware: HardwareSnapshot,
    device_index: int | None,
) -> None:
    app = _device_app(tmp_path, hardware, device_index=device_index)

    async with app.run_test(size=(72, 24)) as pilot:
        await _select_managed_model(app, pilot)

        assert app.query_one("#managed-context-row").display
        assert app.query_one("#managed-device-row").display is (device_index is not None)
        # The whole managed form fits without scrolling at the supported minimum size,
        # keeping the stepper eyebrow on screen.
        assert app.query_one("#config", VerticalScroll).max_scroll_y == 0
        assert "STEP&#160;2&#160;OF&#160;4" in app.export_screenshot()
        continue_button = app.query_one("#config-continue", Button)
        assert continue_button.region.bottom <= 24
        assert continue_button.region.height == 1


async def test_reduced_context_run_carries_the_warning_from_preflight_to_result(tmp_path: Path) -> None:
    download_gate = asyncio.Event()
    gate = asyncio.Event()
    managed_controller = FakeManagedReplayController(
        availability_result=_managed_availability(),
        download_gate=download_gate,
        gate=gate,
    )
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=managed_controller,
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(110, 34)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        await pilot.pause()
        app.query_one("#replay-workload-select", ReplayWorkloadSelect).value = "aa-mini-v1"
        await pilot.pause()
        app.query_one("#managed-context-select", ManagedContextSelect).value = "32768"
        await pilot.pause()

        # The model detail card reflects the chosen context after a round trip.
        app.query_one("#config-back", Button).focus()
        await pilot.press("enter")
        detail = str(app.query_one("#model-detail", Static).content)
        assert "32,768 of 65,536 tokens · reduced" in detail
        assert "needs 8.9 GiB" in detail
        app.query_one("#model-continue", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#managed-context-select", ManagedContextSelect).value == "32768"

        await _check_setup(app, pilot)
        reduced_notice = app.query_one("#preflight-reduced", Static)
        assert reduced_notice.display
        assert str(reduced_notice.content) == (
            "Reduced context: 32,768 of 65,536 tokens. It will be submitted separately from full-context results."
        )
        assert EligibilityReason.REDUCED_CONTEXT in app.evidence.ineligibility_reasons

        await _start_run(app, pilot)
        run_eyebrow = app.query_one("#run-eyebrow", Static)
        assert str(run_eyebrow.content) == f"{RUN_STEP_PREPARING_EYEBROW} · REDUCED CONTEXT"

        download_gate.set()
        await _settle_until(pilot, lambda: str(run_eyebrow.content) == f"{RUN_STEP_EYEBROW} · REDUCED CONTEXT")

        gate.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)

        assert app.outcome is TuiOutcome.SUCCESS
        result_reduced = app.query_one("#result-reduced", Static)
        assert result_reduced.display
        assert str(result_reduced.content) == RESULT_REDUCED_MESSAGE
        request = managed_controller.requests[0]
        assert request.managed_deployment is not None
        assert request.managed_deployment.context_tokens == 32_768
        assert request.managed_deployment.reduced_context


async def test_failed_preflight_hides_the_reduced_context_notice_under_its_error(tmp_path: Path) -> None:
    managed_controller = FakeManagedReplayController(
        availability_result=_managed_availability(),
        preflight_error=True,
    )
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=managed_controller,
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        await pilot.pause()
        app.query_one("#replay-workload-select", ReplayWorkloadSelect).value = "aa-mini-v1"
        await pilot.pause()
        app.query_one("#managed-context-select", ManagedContextSelect).value = "32768"
        await pilot.pause()
        await _check_setup(app, pilot)

        assert app.request is None
        assert "Setup check failed." in str(app.query_one("#preflight-status", Static).content)
        # The reduced-context notice describes a run this failed setup can no longer
        # start, so it never lingers under the error card.
        assert not app.query_one("#preflight-reduced", Static).display
        assert str(app.query_one("#preflight-hero", Static).content) == PREFLIGHT_BLOCKED_HERO
        assert "private managed probe detail" not in app.export_screenshot()


# Short enough to wait out, long enough to outlast one turn boundary on a loaded runner.
LAPSING_CANCEL_WINDOW_SECONDS = 0.5


@pytest.mark.parametrize("leave_page", (False, True), ids=("prompt-lapses", "page-left"))
async def test_turn_completed_under_an_armed_cancel_shows_once_the_prompt_resolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leave_page: bool,
) -> None:
    if not leave_page:
        monkeypatch.setattr(textual_app, "CANCEL_ARMING_WINDOW_SECONDS", LAPSING_CANCEL_WINDOW_SECONDS)
    started_event = asyncio.Event()
    turn_event = asyncio.Event()
    turn_gate = asyncio.Event()
    app = _app(
        tmp_path,
        FakeReplayController(
            started_event=started_event, turn_event=turn_event, turn_gate=turn_gate, gate=asyncio.Event()
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run_by_click(app, pilot)
        async with asyncio.timeout(WORKER_EVENT_TIMEOUT_SECONDS):
            await started_event.wait()
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        assert app.cancel_armed
        metrics = app.query_one("#run-metrics", Static)
        assert str(metrics.content) == CANCEL_CONFIRM_MESSAGE

        # A turn completes while armed: progress updates, but the prompt keeps the status line.
        turn_gate.set()
        async with asyncio.timeout(WORKER_EVENT_TIMEOUT_SECONDS):
            await turn_event.wait()
        await pilot.pause()
        assert str(app.query_one("#run-counters", Static).content).startswith("Task 1/2")
        assert str(metrics.content) == CANCEL_CONFIRM_MESSAGE

        if leave_page:
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert app.step is TuiStep.PRIVACY
            assert not app.cancel_armed
            await pilot.press("escape")
            await pilot.pause()
            assert app.step is TuiStep.RUN
        else:
            await _settle_until(pilot, lambda: not app.cancel_armed)

        # The status line comes back with the turn that completed under the prompt.
        assert "first token 100 ms" in str(metrics.content)
        assert app.replay_active

        # Escape arms afresh instead of confirming a lapsed or abandoned prompt.
        await pilot.press("escape")
        await pilot.pause()
        assert app.outcome is not TuiOutcome.CANCELLED
        assert str(metrics.content) == CANCEL_CONFIRM_MESSAGE

        await pilot.press("escape")
        await pilot.pause()
        assert app.outcome is TuiOutcome.CANCELLED


async def test_setup_focuses_continue_unless_a_custom_manifest_needs_a_path(tmp_path: Path) -> None:
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        defaults=TuiDefaults(output_dir=tmp_path / "results", endpoint_model=ATTACHED_ENDPOINT_MODEL),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter", "enter")
        await pilot.pause()
        assert app.step is TuiStep.CONFIG
        assert app.focused is app.query_one("#config-continue", Button)

        app.query_one("#replay-workload-select", ReplayWorkloadSelect).value = CUSTOM_REPLAY_ID
        await _settle_until(pilot, lambda: app.focused is app.query_one("#manifest-input", Input))
        assert app.query_one("#manifest-row").display
        await pilot.press("escape")
        await pilot.press("enter")
        await pilot.pause()
        assert app.step is TuiStep.CONFIG
        assert app.focused is app.query_one("#manifest-input", Input)

        app.query_one("#manifest-input", Input).value = str(tmp_path / "replay.json")
        await pilot.press("escape")
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused is app.query_one("#config-continue", Button)


async def test_footer_names_the_keys_of_the_visible_screen(tmp_path: Path) -> None:
    gate = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=gate))

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.pause()
        welcome_bindings = app.screen.active_bindings
        assert welcome_bindings["escape"].binding.description == "Quit"

        await pilot.press("enter", "enter")
        await pilot.pause()
        assert app.step is TuiStep.CONFIG
        setup_bindings = app.screen.active_bindings
        assert setup_bindings["escape"].binding.description == "Back"
        assert not setup_bindings["enter"].binding.show

        # Enter in a text field reviews the form, so its less familiar action stays visible.
        app.query_one("#output-input", Input).focus()
        await pilot.pause()
        field_bindings = app.screen.active_bindings
        assert field_bindings["enter"].binding.description == "Review setup"
        assert field_bindings["enter"].binding.show
        app.query_one("#config-continue", Button).focus()
        await pilot.pause()

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.step is TuiStep.PREFLIGHT
        confirm_bindings = app.screen.active_bindings
        assert confirm_bindings["space"].binding.description == "Agree"
        # Run takes focus only after the server check answers from its worker thread.
        await pilot.press("space")
        await _settle_until(pilot, lambda: app.focused is app.query_one("#run-start", Button))
        await pilot.press("shift+tab")
        assert app.focused is app.query_one("#submit-checkbox", Checkbox)
        assert app.screen.active_bindings["space"].binding.description == "Submit"
        await pilot.press("tab")
        assert not app.screen.active_bindings["enter"].binding.show

        await pilot.press("enter")
        await pilot.pause()
        assert app.step is TuiStep.RUN
        run_bindings = app.screen.active_bindings
        assert run_bindings["escape"].binding.description == "Cancel"
        assert "enter" not in run_bindings or not run_bindings["enter"].binding.show
        assert "Details" in app.export_screenshot()

        gate.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)
        result_bindings = app.screen.active_bindings
        assert result_bindings["escape"].binding.description == "New run"
        assert result_bindings["escape"].binding.show
        # Enter still starts a new run from the focused button, but the footer names
        # New run only once; the button label already names its action.
        assert app.focused is app.query_one("#result-new", Button)
        assert not result_bindings["enter"].binding.show


def _activity_text(app: AgentPerfLocalApp) -> str:
    """Return the finished activity lines as one whitespace-normalized string; the log wraps long lines."""
    joined = " ".join(strip.text for strip in app.query_one("#activity-lines", RichLog).lines)
    return " ".join(joined.split())


async def test_run_page_logs_each_step_and_toggles_the_owned_server_log(tmp_path: Path) -> None:
    gate = asyncio.Event()
    managed_controller = FakeManagedReplayController(
        availability_result=_managed_availability(),
        gate=gate,
        server_log_lines=("llama_model_loader: loaded meta data with 40 key-value pairs", "srv listening on port 8080"),
    )
    app = AgentPerfLocalApp(
        load_model_catalog(CATALOG_PATH),
        controller=FakeReplayController(),
        managed_controller=managed_controller,
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(118, 36)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-12b-it-q4-0")
        await pilot.press("enter")
        await _check_setup(app, pilot)
        app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
        await pilot.pause()
        app.query_one("#run-start", Button).press()
        await _settle_until(pilot, lambda: str(app.query_one("#run-hero", Static).content) == RUN_HERO_RUNNING)

        activity = _activity_text(app)
        assert "Setup checked" in activity
        assert "Model ready · 4.0 GiB · from the Hugging Face cache · SHA-256 checked" in activity
        assert "Server ready · llama.cpp · 65,536-token context · 14.2 s" in activity
        assert "GPU startup verified · CUDA" in activity
        assert "Replay loaded · 1 task · 1 turn" in activity
        assert "Turn 1/1 · waiting for the first token" in str(app.query_one("#activity-live", Static).content)
        # This run owns its server, so the context it launched at gives the bar its scale.
        gauge = app.query_one("#run-context", ContextGauge)
        assert gauge.plain_text == "Next request · about 48,120 tokens ▕███████████████░░░░░▏ 73% of 65,536"
        await _settle_until(pilot, lambda: app.focused is app.query_one("#activity-lines", RichLog))
        assert "Server&#160;log" in app.export_screenshot()

        # The owned server's own log is one key away and swaps back the same way.
        switcher = app.query_one("#run-left-switcher", ContentSwitcher)
        assert switcher.current == "run-activity"
        # A wide page keeps the extra charts beside the server log; a compact one shows one or the other.
        run_page = app.query_one("#run", Vertical)
        await pilot.press("d")
        assert run_page.has_class("show-details")
        assert app.query_one("#run-ttft-chart").display
        await pilot.press("l")
        await pilot.pause()
        assert switcher.current == "run-server-log"
        assert run_page.has_class("show-details")
        assert app.query_one("#run-ttft-chart").display
        await pilot.resize_terminal(72, 24)
        await pilot.pause()
        assert not run_page.has_class("show-details")
        assert switcher.current == "run-server-log"
        assert app.query_one("#run-server-log", RichLog).region.height > 1
        await pilot.resize_terminal(118, 36)
        await pilot.pause()
        assert str(app.query_one("#run-left-title", Static).content) == "SERVER LOG · llama.cpp"
        server_log = "\n".join(strip.text for strip in app.query_one("#run-server-log", RichLog).lines)
        assert "llama_model_loader" in server_log
        assert app.focused is app.query_one("#run-server-log", RichLog)
        await pilot.press("l")
        await pilot.pause()
        assert switcher.current == "run-activity"

        gate.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)

        activity = _activity_text(app)
        assert "Turn 1/1 · task 1 · 500 tok/s · 0.3 s" in activity
        assert "Replay finished · 1/1 turns · 1.0 s" in activity
        # No request is in flight once the replay closes, so the line goes away with it.
        assert gauge.plain_text == ""
        assert "Server stopped" in activity
        detail = str(app.query_one("#result-metrics-detail", Static).content)
        assert "p90 first token 100.0 ms · p90 turn 300.0 ms" in detail
        assert "decode per turn · p50 500 · p90 500 tok/s" in detail
        assert not app.query_one("#result-charts", Horizontal).has_class("-empty")
        assert app.query_one("#result-ttft-chart", DistributionChart).histogram is not None
        assert app.query_one("#result-decode-chart", DistributionChart).histogram is not None


async def test_server_log_key_is_inert_for_a_server_you_run_yourself(tmp_path: Path) -> None:
    gate = asyncio.Event()
    app = _app(tmp_path, FakeReplayController(gate=gate))

    async with app.run_test(size=(96, 30)) as pilot:
        await _start_run(app, pilot)
        await pilot.pause()
        assert app.step is TuiStep.RUN
        activity = _activity_text(app)
        assert "Setup checked" in activity
        assert "Server answered · model listed · context length not reported" in activity

        await pilot.press("l")
        await pilot.pause()
        assert app.query_one("#run-left-switcher", ContentSwitcher).current == "run-activity"
        assert "Server&#160;log" not in app.export_screenshot()

        gate.set()
        await _settle_until(pilot, lambda: app.step is TuiStep.RESULT)


@pytest.mark.parametrize(
    ("probe", "run_offered", "check_text", "evidence_text", "reduced_shown"),
    [
        (
            ContextProbeResult(observed_tokens=131_072, reason=ContextObservationReason.REPORTED),
            True,
            "✓ Server answered · model listed · 131,072-token context",
            "Server reports a 131,072-token context · full benchmark context",
            False,
        ),
        (
            ContextProbeResult(observed_tokens=32_768, reason=ContextObservationReason.REPORTED),
            True,
            "✓ Server answered · model listed · 32,768-token context",
            "Server reports a 32,768-token context · reduced",
            True,
        ),
        (
            ContextProbeResult(observed_tokens=None, reason=ContextObservationReason.MODEL_NOT_LISTED),
            True,
            "⚠ Server answered · it does not list this model name",
            ATTACHED_CONTEXT_UNVERIFIED_MESSAGE,
            False,
        ),
        (
            ContextProbeResult(observed_tokens=None, reason=ContextObservationReason.ENDPOINT_UNREACHABLE),
            False,
            f"✗ Server did not answer · {SERVER_CHECK_RETRY_HINT}",
            ATTACHED_CONTEXT_UNVERIFIED_MESSAGE,
            False,
        ),
    ],
)
async def test_consent_checks_the_server_before_run_is_offered(
    tmp_path: Path,
    probe: ContextProbeResult,
    run_offered: bool,
    check_text: str,
    evidence_text: str,
    reduced_shown: bool,
) -> None:
    controller = FakeReplayController(probe_result=probe)
    app = _app(tmp_path, controller)

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await pilot.click("#model-continue")
        await _settle_until(pilot, lambda: app.focused is app.query_one("#config-continue", Button))
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.step is TuiStep.PREFLIGHT
        run = app.query_one("#run-start", Button)
        server_check = app.query_one("#preflight-server-check", SpinnerLine)
        assert run.disabled
        assert not server_check.display

        app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
        await _settle_until(pilot, lambda: server_check.plain_text == check_text)

        assert len(controller.probes) == 1
        assert run.disabled is not run_offered
        assert evidence_text in str(app.query_one("#preflight-evidence", Static).content)
        assert app.query_one("#preflight-reduced", Static).display is reduced_shown
        reduced_recorded = EligibilityReason.REDUCED_CONTEXT in app.evidence.ineligibility_reasons
        assert reduced_recorded is (probe.observed_tokens is None or probe.observed_tokens < 65_536)
        if run_offered:
            assert app.focused is app.query_one("#run-start", Button)

        app.query_one("#endpoint-consent-checkbox", Checkbox).value = False
        await pilot.pause()
        assert run.disabled
        assert not server_check.display


def _bound_results_writer(tmp_path: Path, *, failed_qualification: bool = False) -> Callable[[Path], None]:
    """Return a writer that fills the run folder with a bound, packageable result set."""
    from tests.test_submission import RUN_ID, _private_summary, _qualification, _write_bound_results

    def write(output_dir: Path) -> None:
        source = _write_bound_results(tmp_path / f"seed-{output_dir.name}", _private_summary())
        output_dir.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            (output_dir / child.name).write_bytes(child.read_bytes())
        if failed_qualification:
            passing = _qualification(RUN_ID)
            failed_outcome = replace(passing.outcomes[0], passed=False, failure_codes=("request_error",))
            failed = replace(passing, outcomes=(failed_outcome, *passing.outcomes[1:]))
            write_runtime_qualification(output_dir / QUALIFICATION_FILENAME, failed)

    return write


async def _start_managed_run_with_submit(
    app: AgentPerfLocalApp,
    pilot: Pilot[TuiOutcome],
    *,
    submit: bool,
) -> None:
    await pilot.click("#welcome-start")
    app.query_one("#model-list", OptionList).focus()
    await pilot.press("down", "down", "enter")
    await pilot.pause()
    await _check_setup(app, pilot)
    app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
    if submit:
        app.query_one("#submit-checkbox", Checkbox).value = True
    await pilot.pause()
    app.query_one("#run-start", Button).press()
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()
    if submit:
        # Replay completion posts a message that starts a second worker. Waiting for
        # the replay worker alone can return before Textual has registered the upload.
        await _settle_until(pilot, lambda: app.upload_bundle_dir is not None)


@pytest.mark.parametrize("outcome", ["accepted", "refused"])
async def test_ticking_submit_uploads_the_finished_managed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    from tests.submission_server import SUBMISSION_ID, VALID_TOKEN, FakeSubmissionService, LocalSubmissionServer

    token_env = "AGENTPERF_TUI_TEST_TOKEN"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.setenv(token_env, VALID_TOKEN)
    service = FakeSubmissionService(require_token=True, rate_limited=outcome == "refused")
    managed_controller = FakeManagedReplayController(
        availability_result=_managed_availability(),
        results_writer=_bound_results_writer(tmp_path, failed_qualification=outcome == "accepted"),
    )
    with LocalSubmissionServer(service) as server:
        app = AgentPerfLocalApp(
            load_model_catalog(CATALOG_PATH),
            controller=FakeReplayController(),
            managed_controller=managed_controller,
            defaults=TuiDefaults(
                output_dir=tmp_path / "results",
                client_backend="python",
                submit_base_url=server.base_url,
                submit_token_env=token_env,
            ),
        )
        async with app.run_test(size=(96, 32)) as pilot:
            await _start_managed_run_with_submit(app, pilot, submit=True)
            submit_box = app.query_one("#submit-checkbox", Checkbox)
            notice = str(app.query_one("#submit-notice", Static).content)
            assert not submit_box.disabled
            assert "May be published" in notice
            assert "private for 180 days" in notice
            assert "Never sent" in notice
            assert "Failed checks" in notice
            # The fake allowlist never contains this checkout, so the advisory check has spoken.
            assert "This client version can submit, but cannot reach verified" in notice
            assert app.submit_requested
            await app.workers.wait_for_complete()
            await pilot.pause()
            # The result page inserts soft break points into long paths; the command itself is plain.
            upload = str(app.query_one("#result-upload", Static).content).replace("\x1f", "")
            request = managed_controller.requests[0]
            bundle_dir = request.output_dir.with_name(f"{request.output_dir.name}-submission")

    assert app.outcome is TuiOutcome.SUCCESS
    assert not app.upload_active
    assert (bundle_dir / "private-audit.json").is_file()
    assert app.upload_bundle_dir == bundle_dir
    if outcome == "accepted":
        assert app.submission_receipt is not None
        assert app.submission_receipt.submission_id == SUBMISSION_ID
        assert f"Submitted · {SUBMISSION_ID} · queued" in upload
        assert f"submission-status {SUBMISSION_ID}" in upload
        assert service.captured[0].status == 202
        assert service.captured[0].headers["authorization"] == f"Bearer {VALID_TOKEN}"
        files = service.captured[0].body["files"]
        assert isinstance(files, dict)
        encoded_audit = files[PRIVATE_AUDIT_FILENAME]
        assert isinstance(encoded_audit, str)
        audit = parse_json_object(base64.b64decode(encoded_audit), PRIVATE_AUDIT_FILENAME)
        qualification = audit["runtime_qualification"]
        assert isinstance(qualification, dict)
        assert qualification["passed"] is False
    else:
        assert app.submission_receipt is None
        assert "Upload failed · rate_limited" in upload
        assert f"agentperf-local submit {bundle_dir} --yes" in upload
    assert VALID_TOKEN not in upload


async def test_submit_checkbox_uploads_anonymously_without_a_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.submission_server import LocalSubmissionServer

    token_env = "AGENTPERF_TUI_TEST_TOKEN"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.delenv(token_env, raising=False)
    managed_controller = FakeManagedReplayController(
        availability_result=_managed_availability(),
        results_writer=_bound_results_writer(tmp_path),
    )
    with LocalSubmissionServer() as server:
        app = AgentPerfLocalApp(
            load_model_catalog(CATALOG_PATH),
            controller=FakeReplayController(),
            managed_controller=managed_controller,
            defaults=TuiDefaults(
                output_dir=tmp_path / "results",
                client_backend="python",
                submit_base_url=server.base_url,
                submit_token_env=token_env,
            ),
        )
        async with app.run_test(size=(96, 32)) as pilot:
            await _start_managed_run_with_submit(app, pilot, submit=True)
            submit_box = app.query_one("#submit-checkbox", Checkbox)
            notice = str(app.query_one("#submit-notice", Static).content)
            await app.workers.wait_for_complete()
            await pilot.pause()

    assert not submit_box.disabled
    assert "Failed checks" in notice
    assert "still submitted as self-reported" in notice
    assert app.submit_requested
    assert app.submission_receipt is not None
    assert server.service.captured[0].status == 202
    assert "authorization" not in server.service.captured[0].headers
    assert app.outcome is TuiOutcome.SUCCESS


@pytest.mark.parametrize(("size", "accepted"), (((72, 24), True), ((96, 32), False)))
async def test_submission_stays_busy_until_server_responds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
    accepted: bool,
) -> None:
    from tests.submission_server import FakeSubmissionService, LocalSubmissionServer

    token_env = "AGENTPERF_TUI_TEST_TOKEN"
    monkeypatch.delenv(token_env, raising=False)
    controller = FakeReplayController(results_writer=_bound_results_writer(tmp_path))
    release = threading.Event()
    service = FakeSubmissionService(response_release=release, rate_limited=not accepted)
    with LocalSubmissionServer(service) as server:
        app = AgentPerfLocalApp(
            load_model_catalog(CATALOG_PATH),
            controller=controller,
            defaults=TuiDefaults(
                output_dir=tmp_path / "results",
                client_backend="python",
                # Every catalog entry is a managed recipe, so an attached run is reached
                # through the endpoint entry, which a configured endpoint model selects.
                endpoint_model=ATTACHED_ENDPOINT_MODEL,
                submit_base_url=server.base_url,
                submit_token_env=token_env,
            ),
        )
        async with app.run_test(size=size) as pilot:
            await pilot.click("#welcome-start")
            await pilot.click("#model-continue")
            await pilot.pause()
            await _check_setup(app, pilot)
            assert app.step is TuiStep.PREFLIGHT
            app.query_one("#endpoint-consent-checkbox", Checkbox).value = True
            await _settle_until(pilot, lambda: not app.query_one("#run-start", Button).disabled)
            app.query_one("#submit-checkbox", Checkbox).value = True
            await pilot.pause()
            app.query_one("#run-start", Button).press()
            busy = app.query_one("#result-upload-busy", SpinnerLine)
            new_run = app.query_one("#result-new", Button)
            try:
                await _settle_until(pilot, lambda: busy.plain_text == UPLOAD_CONFIRMING_MESSAGE)
                assert app.step is TuiStep.RESULT
                assert app.upload_active
                assert app.query_one("#result-upload-progress", ProgressBar).percentage == 1
                assert str(app.query_one("#result-upload", Static).content) == UPLOAD_WAITING_MESSAGE
                assert busy.display
                assert new_run.disabled
                # New run is disabled, so focus takes the visible details toggle and survives Help.
                details = app.query_one("#result-details-toggle", Button)
                assert app.focused is details
                await pilot.press("f1")
                assert app.step is TuiStep.METHODOLOGY
                await pilot.press("escape")
                await pilot.pause()
                assert app.step is TuiStep.RESULT
                assert app.focused is details
                assert busy.region.height == 1
                assert busy.region.right <= size[0]
                assert busy.region.bottom < size[1]
                frame = str(busy.content)
                await _settle_until(pilot, lambda: str(busy.content) != frame)
                await pilot.press("escape", "q")
                assert app.step is TuiStep.RESULT
                assert app.upload_active
                assert "Waiting&#160;for&#160;confirmation" in app.export_screenshot()
            finally:
                release.set()
            await _settle_until(pilot, lambda: app.upload_bundle_dir is not None)
            assert not app.upload_active
            assert not busy.display
            assert not new_run.disabled
            assert app.focused is new_run
            await pilot.press("enter")
            assert app.step is TuiStep.CONFIG
            await pilot.press("q")

    assert controller.requests[0].managed_deployment is None
    assert (app.submission_receipt is not None) is accepted
    assert server.service.captured[0].status == (202 if accepted else 429)
    assert "authorization" not in server.service.captured[0].headers


async def test_weights_profile_is_selectable_and_launchable_from_the_model_screen(tmp_path: Path) -> None:
    """A weights recipe reaches a ready run through the same guided flow as a GGUF one."""
    catalog = load_model_catalog(CATALOG_PATH)
    app = AgentPerfLocalApp(
        catalog,
        controller=FakeReplayController(),
        managed_controller=LocalManagedReplayController(
            catalog_as_of=catalog.as_of,
            hardware=_hardware_with(_accelerator("NVIDIA test GPU", 96 * 1024**3)),
            offer_collector=_installed_sglang_offer,
        ),
        defaults=TuiDefaults(
            output_dir=tmp_path / "results",
            client_backend="python",
            endpoint_model=ATTACHED_ENDPOINT_MODEL,
        ),
    )

    async with app.run_test(size=(96, 30)) as pilot:
        await pilot.click("#welcome-start")
        await _highlight_profile(app, pilot, "gemma4-26b-a4b-nvfp4")

        detail = str(app.query_one("#model-detail", Static).content)
        assert "sglang" in detail
        assert "NVFP4 weights" in detail

        await pilot.press("enter")
        await pilot.pause()
        framework = app.query_one("#managed-framework-select", ManagedFrameworkSelect)
        assert framework.value == "sglang"
        assert app.query_one("#managed-context-select", ManagedContextSelect).value == "65536"

        await _check_setup(app, pilot)

        assert str(app.query_one("#preflight-hero", Static).content) == PREFLIGHT_READY_HERO
        assert "starts the server with SGLang" in str(app.query_one("#preflight-evidence", Static).content)
        assert app.request is not None
        assert app.request.managed_deployment is not None
        assert app.request.managed_deployment.framework == "sglang"
