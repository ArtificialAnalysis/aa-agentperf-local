"""Name the app's visible steps and its exit outcome."""

from enum import StrEnum


class TuiStep(StrEnum):
    """Name one visible app step."""

    WELCOME = "welcome"
    MODEL = "model"
    CONFIG = "config"
    PREFLIGHT = "preflight"
    RUN = "run"
    RESULT = "result"
    PRIVACY = "privacy"
    METHODOLOGY = "methodology"


class TuiOutcome(StrEnum):
    """Name the process outcome of the latest guided attempt."""

    NO_RUN = "no-run"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Escape means something different per screen, so each meaning is its own action and the
# footer shows only the one whose step is visible (via App.check_action).
ESCAPE_ACTION_STEPS: dict[str, frozenset[TuiStep]] = {
    "back_quit": frozenset({TuiStep.WELCOME}),
    "back": frozenset({TuiStep.MODEL, TuiStep.CONFIG, TuiStep.PREFLIGHT, TuiStep.PRIVACY, TuiStep.METHODOLOGY}),
    "back_cancel": frozenset({TuiStep.RUN}),
    "back_new_run": frozenset({TuiStep.RESULT}),
}
