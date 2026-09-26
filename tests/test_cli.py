"""Exercise file-producing CLI commands through the public entry point."""

import socket
from pathlib import Path

import orjson
import pytest

from agentperf_local.cli import main
from agentperf_local.cli.options import DEFAULT_MODEL_CATALOG_PATH
from agentperf_local.deployment.catalog import load_model_catalog
from agentperf_local.provenance.benchmark import (
    MEASUREMENT_BINDING_FILENAME,
)
from agentperf_local.provenance.hardware import HardwareSnapshot
from agentperf_local.provenance.hardware_facts import AcceleratorSnapshot
from agentperf_local.replay.config import CLEARTEXT_API_KEY_MESSAGE
from agentperf_local.tui.app import AgentPerfLocalApp
from agentperf_local.tui.evidence import SelectionKind
from agentperf_local.tui.steps import TuiOutcome
from agentperf_local.workload.schema import load_manifest

RECORDING_FIXTURES = Path(__file__).parent / "fixtures" / "recording"
RECORDING = RECORDING_FIXTURES / "recordings" / "demo.json"
API_KEY_ENV = "PRIVATE_TOKEN"
API_KEY = "private-endpoint-token"
RECORDING_TURNS = 2
UNREACHABLE_BASE_URL = "http://127.0.0.1:1/v1"
INTERRUPTED_STATUS = 130
ARGPARSE_ERROR_STATUS = 2


def _closed_loopback_port() -> int:
    """Return a loopback port that no server holds."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        address = probe.getsockname()
    if not isinstance(address, tuple) or len(address) < 2 or not isinstance(address[1], int):
        raise AssertionError("loopback probe did not report a port")
    return address[1]


def _converted_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Convert the bundled recording and discard its command output."""
    converted = tmp_path / "converted"
    assert main(["convert", str(RECORDING), "--output-dir", str(converted)]) == 0
    capsys.readouterr()
    return converted / "manifest.json"


def _deployment_hardware(vendor: str, api: str) -> HardwareSnapshot:
    return HardwareSnapshot(
        operating_system="Darwin" if vendor == "Apple" else "Linux",
        operating_system_version="test",
        kernel_version="test",
        architecture="arm64" if vendor == "Apple" else "x86_64",
        cpu_model="test-cpu",
        logical_cpu_count=8,
        memory_bytes=32 * 1024**3,
        accelerators=(
            AcceleratorSnapshot(
                vendor=vendor,
                name=f"{vendor} test GPU",
                memory_bytes=None if vendor == "Apple" else 32 * 1024**3,
                core_count=None,
                driver_version=None,
                api=api,
            ),
        ),
        warnings=(),
    )


def test_cli_detects_hardware_and_offers_only_compatible_frameworks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agentperf_local.cli.options.collect_hardware_snapshot", lambda: _deployment_hardware("NVIDIA", "CUDA")
    )
    monkeypatch.setattr("agentperf_local.deployment.frameworks.shutil.which", lambda command: f"/test/{command}")

    status = main(["deployment-options"])

    output = orjson.loads(capsys.readouterr().out)
    assert status == 0
    assert output["profile_id"] == "gemma4-12b-it-q4-0"
    assert [offer["framework"] for offer in output["offers"]] == ["llama-cpp"]
    assert output["can_deploy_now"] is True


def test_cli_launches_textual_pilot_with_prefilled_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[AgentPerfLocalApp] = []

    def record_launch(app: AgentPerfLocalApp) -> None:
        launched.append(app)

    monkeypatch.setattr(AgentPerfLocalApp, "run", record_launch)

    status = main(
        [
            "tui",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "results"),
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "served-model",
            "--api-key-env",
            "PRIVATE_TOKEN",
            "--client",
            "python",
            "--cache-root",
            str(tmp_path / "model-cache"),
            "--port",
            "18080",
            "--startup-timeout-seconds",
            "45",
        ]
    )

    assert status == 0
    assert len(launched) == 1
    assert launched[0].defaults.replay_id == "agentperf-default-v1"
    assert launched[0].defaults.manifest_path == tmp_path / "manifest.json"
    assert launched[0].defaults.output_dir == tmp_path / "results"
    assert launched[0].defaults.base_url == "http://localhost:8000/v1"
    assert launched[0].defaults.endpoint_model == "served-model"
    assert launched[0].defaults.api_key_env == "PRIVATE_TOKEN"
    assert launched[0].defaults.client_backend == "python"
    assert launched[0].defaults.model_cache_root == tmp_path / "model-cache"
    assert launched[0].defaults.deployment_port == 18080
    assert launched[0].defaults.deployment_startup_timeout_seconds == 45.0
    assert launched[0].selection.kind is SelectionKind.CUSTOM_ENDPOINT


@pytest.mark.parametrize("arguments", ([], ["tui"]))
@pytest.mark.parametrize(
    ("outcome", "return_code", "expected_status"),
    (
        (TuiOutcome.NO_RUN, None, 0),
        (TuiOutcome.SUCCESS, 0, 0),
        (TuiOutcome.FAILED, 0, 1),
        (TuiOutcome.CANCELLED, None, 1),
        (None, 1, 1),
        (TuiOutcome.SUCCESS, 1, 1),
    ),
)
def test_cli_returns_the_textual_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    outcome: TuiOutcome | None,
    return_code: int | None,
    expected_status: int,
) -> None:
    def finish_with_outcome(app: AgentPerfLocalApp) -> TuiOutcome | None:
        return outcome

    def finish_with_return_code(app: AgentPerfLocalApp) -> int | None:
        return return_code

    monkeypatch.setattr(AgentPerfLocalApp, "run", finish_with_outcome)
    monkeypatch.setattr(AgentPerfLocalApp, "return_code", property(finish_with_return_code))

    assert main(arguments) == expected_status


def test_cli_converts_recording(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    converted = tmp_path / "converted"
    conversion_status = main(
        [
            "convert",
            str(RECORDING),
            "--output-dir",
            str(converted),
            "--family",
            "pinchbench-sanitized",
            "--adapter",
            "pinchbench",
        ]
    )
    conversion_output = capsys.readouterr()

    manifest = load_manifest(converted / "manifest.json")
    assert conversion_status == 0
    assert len(manifest.tasks) == 1
    assert orjson.loads(conversion_output.out)["manifest"] == str(converted / "manifest.json")


def test_run_explains_failed_turns_on_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = _converted_manifest(tmp_path, capsys)
    output_dir = tmp_path / "results"

    status = main(
        [
            "run",
            str(manifest_path),
            "--base-url",
            f"http://127.0.0.1:{_closed_loopback_port()}/v1",
            "--model",
            "unreachable-model",
            "--output-dir",
            str(output_dir),
            "--client",
            "python",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert orjson.loads(captured.out)["success"] is False
    assert f"{RECORDING_TURNS}/{RECORDING_TURNS} turns failed. First error: " in captured.err
    assert f"Details: {output_dir / 'failures.json'}" in captured.err


@pytest.mark.parametrize(("host", "refused"), (("endpoint.example", True), ("127.0.0.1", False), ("localhost", False)))
def test_run_refuses_a_cleartext_api_key_outside_loopback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    refused: bool,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    manifest_path = _converted_manifest(tmp_path, capsys)
    output_dir = tmp_path / "results"

    status = main(
        [
            "run",
            str(manifest_path),
            "--base-url",
            f"http://{host}:{_closed_loopback_port()}/v1",
            "--model",
            "unreachable-model",
            "--output-dir",
            str(output_dir),
            "--client",
            "python",
            "--api-key-env",
            API_KEY_ENV,
        ]
    )

    captured = capsys.readouterr()
    refusal = CLEARTEXT_API_KEY_MESSAGE
    assert status == 1
    assert (refusal in captured.err) is refused
    assert output_dir.exists() is not refused
    assert API_KEY not in captured.out + captured.err


def test_prepare_submission_explains_an_unbound_results_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "summary.json").write_bytes(b"{}")

    status = main(["prepare-submission", str(results_dir), "--output-dir", str(tmp_path / "public")])

    captured = capsys.readouterr()
    assert status == 1
    assert f"{results_dir} has no {MEASUREMENT_BINDING_FILENAME}" in captured.err
    assert "choose a results directory written by run, managed-run, or tui" in captured.err


def test_run_refuses_a_prior_attempt_before_contacting_the_endpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _converted_manifest(tmp_path, capsys)
    output_dir = tmp_path / "prior-attempt"
    output_dir.mkdir()
    existing = output_dir / MEASUREMENT_BINDING_FILENAME
    existing.write_text("prior attempt")

    status = main(_run_arguments(manifest_path, output_dir))

    captured = capsys.readouterr()
    assert status == 1
    assert "run output must be fresh" in captured.err
    assert existing.read_text() == "prior attempt"
    assert tuple(path.name for path in output_dir.iterdir()) == (MEASUREMENT_BINDING_FILENAME,)


def _run_arguments(manifest_path: Path, output_dir: Path, *extra: str) -> list[str]:
    """Return one offline replay invocation that never reaches the endpoint."""
    return [
        "run",
        str(manifest_path),
        "--base-url",
        UNREACHABLE_BASE_URL,
        "--model",
        "must-not-be-contacted",
        "--output-dir",
        str(output_dir),
        "--client",
        "python",
        *extra,
    ]


def test_main_reports_an_interrupt_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt() -> HardwareSnapshot:
        raise KeyboardInterrupt

    monkeypatch.setattr("agentperf_local.cli.inspect.collect_hardware_snapshot", interrupt)

    status = main(["doctor"])

    captured = capsys.readouterr()
    assert status == INTERRUPTED_STATUS
    assert captured.err.strip() == "error: interrupted"
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("extra_arguments", "message"),
    (
        (
            ("--output-token-policy", "fixed", "--output-token-margin", "8"),
            "output token margin applies only to the recorded policy",
        ),
        (("--tool-delay-scale", "2.0"), "tool delay scale requires a tool replay mode"),
    ),
)
def test_run_rejects_a_flag_combination_it_would_otherwise_ignore(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_arguments: tuple[str, ...],
    message: str,
) -> None:
    manifest_path = _converted_manifest(tmp_path, capsys)
    output_dir = tmp_path / "results"

    status = main(_run_arguments(manifest_path, output_dir, *extra_arguments))

    captured = capsys.readouterr()
    assert status == 1
    assert f"error: {message}" in captured.err
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("convert", str(RECORDING), "--output-dir", ""), "argument --output-dir: must not be empty"),
        (("prepare-submission", "", "--output-dir", "bundle"), "argument results_dir: must not be empty"),
        (
            ("managed-run", "manifest.json", "--output-dir", "out", "--cache-root", ""),
            "argument --cache-root: must not be empty",
        ),
        (
            ("run", "manifest.json", "--base-url", UNREACHABLE_BASE_URL, "--model", "unused", "--output-dir", ""),
            "argument --output-dir: must not be empty",
        ),
    ),
)
def test_path_arguments_reject_an_empty_value(
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(SystemExit) as exit_status:
        main(list(arguments))

    captured = capsys.readouterr()
    assert exit_status.value.code == ARGPARSE_ERROR_STATUS
    assert message in captured.err
    # The parser refuses before any command runs, so nothing is created or printed.
    assert not captured.out


@pytest.mark.parametrize("accelerator_count", (0, 2))
@pytest.mark.parametrize("json_output", (True, False))
def test_doctor_explains_an_unusable_accelerator_count_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    accelerator_count: int,
    json_output: bool,
) -> None:
    snapshot = _deployment_hardware("NVIDIA", "CUDA")
    monkeypatch.setattr(
        "agentperf_local.cli.inspect.collect_hardware_snapshot",
        lambda: HardwareSnapshot(
            operating_system=snapshot.operating_system,
            operating_system_version=snapshot.operating_system_version,
            kernel_version=snapshot.kernel_version,
            architecture=snapshot.architecture,
            cpu_model=snapshot.cpu_model,
            logical_cpu_count=snapshot.logical_cpu_count,
            memory_bytes=snapshot.memory_bytes,
            accelerators=snapshot.accelerators * accelerator_count,
            warnings=(),
        ),
    )

    status = main(["doctor", "--json"] if json_output else ["doctor"])

    captured = capsys.readouterr()
    assert status == 1
    assert f"doctor requires exactly one detected accelerator; found {accelerator_count}" in captured.err
    assert captured.out
    if json_output:
        assert len(orjson.loads(captured.out)["accelerators"]) == accelerator_count


def test_deployment_options_fails_when_no_offer_can_deploy_now(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agentperf_local.cli.options.collect_hardware_snapshot", lambda: _deployment_hardware("NVIDIA", "CUDA")
    )
    monkeypatch.setattr("agentperf_local.deployment.frameworks.shutil.which", lambda command: None)

    status = main(["deployment-options"])

    output = orjson.loads(capsys.readouterr().out)
    assert status == 1
    assert output["offers"]
    assert output["can_deploy_now"] is False


def test_unknown_catalog_profile_lists_the_available_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    available = ", ".join(sorted(model.profile_id for model in load_model_catalog(DEFAULT_MODEL_CATALOG_PATH).models))

    status = main(["deployment-options", "--profile-id", "no-such-profile"])

    captured = capsys.readouterr()
    assert status == 1
    assert f"error: model catalog does not contain profile no-such-profile; available: {available}" in captured.err


@pytest.mark.parametrize(
    ("context_arguments", "message"),
    (
        (
            ("--context-tokens", "8192"),
            "error: This replay needs at least 1,000,000 tokens of context, but the server would start with 8,192.",
        ),
        (
            (),
            "error: This replay needs at least 1,000,000 tokens of context, but the server would start with 65,536.",
        ),
    ),
    ids=("explicit-flag", "recipe-default"),
)
def test_managed_run_rejects_a_context_below_the_replay_requirement(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    context_arguments: tuple[str, ...],
    message: str,
) -> None:
    manifest_path = _converted_manifest(tmp_path, capsys)
    manifest_data = orjson.loads(manifest_path.read_bytes())
    manifest_data["tasks"][0]["required_context_tokens"] = 1_000_000
    manifest_path.write_bytes(orjson.dumps(manifest_data))
    output_dir = tmp_path / "results"

    status = main(
        [
            "managed-run",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--framework",
            "llama-cpp",
            "--client",
            "python",
            *context_arguments,
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert message in captured.err
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("replay_id", "context_tokens", "floor"),
    (("aa-mini-v1", "4096", "8,192"), ("agentperf-default-v1", "32768", "65,536")),
)
def test_managed_run_resolves_a_bundled_replay_by_identifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], replay_id: str, context_tokens: str, floor: str
) -> None:
    """--replay reaches the bundled manifest, whose own context floor then applies."""
    output_dir = tmp_path / "results"

    status = main(
        [
            "managed-run",
            "--replay",
            replay_id,
            "--output-dir",
            str(output_dir),
            "--framework",
            "llama-cpp",
            "--context-tokens",
            context_tokens,
        ]
    )

    assert status == 1
    assert f"This replay needs at least {floor} tokens of context" in capsys.readouterr().err
    assert not output_dir.exists()


def test_run_requires_the_rust_client_before_touching_the_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable() -> None:
        raise ValueError("the Rust client is unavailable; install the rust extra")

    monkeypatch.setattr("agentperf_local.cli.replay.validate_rustcore_available", unavailable)
    manifest_path = _converted_manifest(tmp_path, capsys)
    output_dir = tmp_path / "results"

    status = main(
        [
            "run",
            str(manifest_path),
            "--base-url",
            UNREACHABLE_BASE_URL,
            "--model",
            "must-not-be-contacted",
            "--output-dir",
            str(output_dir),
            "--client",
            "rust",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "error: the Rust client is unavailable" in captured.err
    assert not output_dir.exists()


def test_run_removes_its_measurement_binding_when_the_run_never_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _converted_manifest(tmp_path, capsys)
    monkeypatch.setattr(
        "agentperf_local.cli.replay.collect_hardware_snapshot", lambda: _deployment_hardware("NVIDIA", "CUDA")
    )

    def interrupt(*arguments: object, **keywords: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("agentperf_local.cli.replay.run_manifest", interrupt)
    output_dir = tmp_path / "results"

    status = main(_run_arguments(manifest_path, output_dir))

    captured = capsys.readouterr()
    assert status == INTERRUPTED_STATUS
    # The unreachable endpoint yields a non-comparable warning before the interrupt lands.
    assert "will be recorded as non-comparable" in captured.err
    assert captured.err.strip().endswith("error: interrupted")
    assert tuple(output_dir.iterdir()) == ()


@pytest.mark.parametrize(
    ("requested", "expected_client"),
    [
        (None, "python"),
        ("python", "python"),
        ("rust", "rust"),
    ],
)
def test_tui_uses_the_requested_client_and_defaults_to_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str | None,
    expected_client: str,
) -> None:
    launched: list[AgentPerfLocalApp] = []

    def record_launch(app: AgentPerfLocalApp) -> None:
        launched.append(app)

    monkeypatch.setattr(AgentPerfLocalApp, "run", record_launch)
    arguments = ["tui", "--output-dir", str(tmp_path / "results")]
    if requested is not None:
        arguments.extend(["--client", requested])

    assert main(arguments) == 0

    assert launched[0].defaults.client_backend == expected_client
