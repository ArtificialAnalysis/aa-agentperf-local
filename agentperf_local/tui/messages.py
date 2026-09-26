"""Carry worker results into the Textual message queue.

A worker thread never touches a widget. It posts one of these messages and the
app renders on its own event loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from textual.app import App
from textual.message import Message

from agentperf_local.deployment.endpoint_probes import ContextProbeResult
from agentperf_local.deployment.managed_run import RunActivity
from agentperf_local.replay.runner import RunBoundaryEvent
from agentperf_local.submission.client import SubmissionReceipt
from agentperf_local.tui.replay_contract import ReplayExecution, ReplayPreflight, ReplayRequest
from agentperf_local.tui.steps import TuiOutcome


class UploadProgressMessage(Message):
    """Report bytes sent so far for the active run's upload."""

    def __init__(self, generation: int, sent_bytes: int, total_bytes: int) -> None:
        super().__init__()
        self.generation = generation
        self.sent_bytes = sent_bytes
        self.total_bytes = total_bytes


class UploadCompletedMessage(Message):
    """Carry the service's receipt for the active run's upload."""

    def __init__(self, generation: int, receipt: SubmissionReceipt, bundle_dir: Path) -> None:
        super().__init__()
        self.generation = generation
        self.receipt = receipt
        self.bundle_dir = bundle_dir


class UploadFailedMessage(Message):
    """Report a failed or cancelled upload with a display-safe reason and the bundle left behind."""

    def __init__(self, generation: int, reason: str, bundle_dir: Path | None) -> None:
        super().__init__()
        self.generation = generation
        self.reason = reason
        self.bundle_dir = bundle_dir


class RevisionAdviceMessage(Message):
    """Carry the advisory allowlist check for one setup back from its worker thread."""

    def __init__(self, request: ReplayRequest, advice: str | None, reachable: bool) -> None:
        super().__init__()
        self.request = request
        self.advice = advice
        self.reachable = reachable


@dataclass
class RunBoundaryMessage(Message):
    """Carry one post-close runner boundary into Textual."""

    event: RunBoundaryEvent
    generation: int


@dataclass
class ArtifactProgressMessage(Message):
    """Carry one managed artifact download measurement into Textual."""

    downloaded_bytes: int
    total_bytes: int
    generation: int


@dataclass
class RunActivityMessage(Message):
    """Carry one run step report into Textual."""

    activity: RunActivity
    generation: int


@dataclass
class ServerLogMessage(Message):
    """Carry one batch of owned-server log lines into Textual."""

    lines: tuple[str, ...]
    generation: int


@dataclass
class ReplayCompletedMessage(Message):
    """Carry local artifact completion into Textual."""

    result: ReplayExecution
    generation: int


@dataclass
class ReplayFailedMessage(Message):
    """Report a safe execution failure, optionally naming a display-safe managed cause."""

    generation: int
    cause: str | None = field(default=None, kw_only=True)


@dataclass
class ReplayCancelledMessage(Message):
    """Report that replay cancellation and owned cleanup have finished."""

    generation: int


type RunPhaseMessage = (
    RunBoundaryMessage
    | ArtifactProgressMessage
    | RunActivityMessage
    | ServerLogMessage
    | ReplayCompletedMessage
    | ReplayFailedMessage
    | ReplayCancelledMessage
)
"""Name the run messages whose relevance is bound to one replay generation."""


@dataclass
class ReplayFinalizingMessage(Message):
    """Request a cancellation-safe transition before report commits."""

    observer: TextualRunObserver


@dataclass
class PreflightCompletedMessage(Message):
    """Carry a local preflight result back from its worker thread."""

    request: ReplayRequest
    preflight: ReplayPreflight


@dataclass
class EndpointProbeMessage(Message):
    """Carry the consent-time server check back from its worker thread; None means the check itself failed."""

    request: ReplayRequest
    probe: ContextProbeResult | None
    # True when the server answered as Ollama, whose runs fall back to the recorded policy.
    ollama: bool = False


@dataclass
class PreflightFailedMessage(Message):
    """Report a safe preflight failure without private exception text."""

    request: ReplayRequest


@dataclass(slots=True, kw_only=True)
class TextualRunObserver:
    """Enqueue coarse runner boundaries without rendering in the callback."""

    app: App[TuiOutcome]
    generation: int
    finalization_allowed: asyncio.Event

    def on_boundary(self, event: RunBoundaryEvent) -> None:
        """Post one immutable boundary to the Textual message queue."""
        self.app.post_message(RunBoundaryMessage(event, self.generation))

    def on_artifact_progress(self, downloaded_bytes: int, total_bytes: int) -> None:
        """Post one managed download measurement to the Textual message queue."""
        self.app.post_message(ArtifactProgressMessage(downloaded_bytes, total_bytes, self.generation))

    def on_activity(self, activity: RunActivity) -> None:
        """Post one run step report to the Textual message queue."""
        self.app.post_message(RunActivityMessage(activity, self.generation))

    def on_server_log(self, lines: tuple[str, ...]) -> None:
        """Post one batch of server log lines to the Textual message queue."""
        self.app.post_message(ServerLogMessage(lines, self.generation))

    async def on_finalizing(self) -> None:
        """Wait until the UI disables cancellation before report commits."""
        self.app.post_message(ReplayFinalizingMessage(self))
        await self.finalization_allowed.wait()

    def allow_finalization(self) -> None:
        """Release the controller after the UI enters finalizing state."""
        self.finalization_allowed.set()
