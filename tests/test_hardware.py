"""Exercise privacy-safe hardware collection through its public surface."""

import os
import platform
import sys
from pathlib import Path

import orjson
import pytest
from pydantic import BaseModel

from agentperf_local.common.json_types import JsonValue
from agentperf_local.common.models import replace_fields
from agentperf_local.deployment.frameworks import available_accelerator_memory
from agentperf_local.provenance.accelerator_probes import NVIDIA_SMI_COMMAND, ROCMINFO_COMMAND
from agentperf_local.provenance.hardware import (
    HardwareSnapshot,
    collect_hardware_snapshot,
    hardware_snapshot_from_json,
    public_hardware_profile,
)
from agentperf_local.provenance.hardware_facts import (
    HARDWARE_WARNING_MESSAGES,
    AcceleratorSnapshot,
    CommandResult,
    HardwareWarningCode,
    LocalSystemProbe,
)


class FakeSystemProbe(BaseModel, frozen=True):
    system: str
    cpu: str
    command_name: str
    result: CommandResult
    # Answers the base nvidia-smi query when the extended one is refused.
    fallback_result: CommandResult | None = None
    base_frequency_mhz: int | None = None
    # nvidia-smi is also asked for the addressing mode when a row reports no memory.
    query_result: CommandResult | None = None
    rocminfo_result: CommandResult | None = None

    def system_name(self) -> str:
        return self.system

    def system_version(self) -> str:
        return "test-os-version"

    def kernel_version(self) -> str:
        return "test-kernel"

    def machine(self) -> str:
        return "test-architecture"

    def cpu_model(self) -> str:
        return self.cpu

    def logical_cpu_count(self) -> int:
        return 16

    def total_memory_bytes(self) -> int:
        return 64 * 1024**3

    def cpu_base_frequency_mhz(self) -> int | None:
        return self.base_frequency_mhz

    def command_available(self, command: str) -> bool:
        return command == self.command_name or (command == ROCMINFO_COMMAND[0] and self.rocminfo_result is not None)

    def run(self, command: tuple[str, ...]) -> CommandResult:
        if command == ROCMINFO_COMMAND and self.rocminfo_result is not None:
            return self.rocminfo_result
        assert command[0] == self.command_name
        # The addressing query is matched first; it is neither the extended nor the base row query.
        if "-q" in command and self.query_result is not None:
            return self.query_result
        if self.fallback_result is not None and command != NVIDIA_SMI_COMMAND:
            return self.fallback_result
        return self.result


def _macos_probe(metal_key: str) -> FakeSystemProbe:
    return FakeSystemProbe(
        system="Darwin",
        cpu="arm64",
        command_name="system_profiler",
        result=CommandResult(
            returncode=0,
            stdout=orjson.dumps(
                {
                    "SPHardwareDataType": [
                        {
                            "chip_type": "Apple M5 Pro",
                            "serial_number": "must-never-leave-this-machine",
                            "platform_UUID": "private-hardware-uuid",
                        }
                    ],
                    "SPDisplaysDataType": [
                        {
                            "sppci_model": "Apple M5 Pro",
                            "sppci_cores": "20",
                            metal_key: "spdisplays_metal4",
                        }
                    ],
                }
            ),
            stderr=b"",
        ),
    )


def _nvidia_probe(stdout: bytes, addressing: bytes | None = None) -> FakeSystemProbe:
    return FakeSystemProbe(
        system="Linux",
        cpu="AMD Ryzen test CPU",
        command_name="nvidia-smi",
        result=CommandResult(returncode=0, stdout=stdout, stderr=b""),
        query_result=None if addressing is None else CommandResult(returncode=0, stdout=addressing, stderr=b""),
    )


# A GB10 reports no framebuffer at all and names its coherent addressing mode instead.
GB10_ROW = b"NVIDIA GB10, [N/A], 580.126.09\n"
POSIX_MEMORY_PROBE = pytest.mark.skipif(sys.platform == "win32", reason="Windows reads memory without os.sysconf")
DISCRETE_ROW = b"NVIDIA RTX PRO 6000 Blackwell, 97887, 590.42\n"
COHERENT_ADDRESSING = b"    Product Name : NVIDIA GB10\n    Addressing Mode : ATS\n"
DISCRETE_ADDRESSING = b"    Product Name : NVIDIA RTX PRO 6000\n    Addressing Mode : None\n"


def _amd_probe(command_name: str, payload: JsonValue) -> FakeSystemProbe:
    return FakeSystemProbe(
        system="Linux",
        cpu="AMD EPYC test CPU",
        command_name=command_name,
        result=CommandResult(returncode=0, stdout=orjson.dumps(payload), stderr=b""),
    )


@pytest.mark.parametrize("memory_kind", ["APU", "DISCRETE"])
def test_amd_gpu_pool_capacity_requires_explicit_apu_evidence(memory_kind: str) -> None:
    probe = _amd_probe(
        "rocm-smi",
        {"card0": {"Card series": "AMD Radeon 8060S Graphics", "VRAM Total Memory (B)": str(512 * 1024**2)}},
    )
    pool_kib = 48 * 1024**2
    info = f"""Agent 1
  Marketing Name: AMD Radeon 8060S Graphics
  Device Type: GPU
  Memory Properties: {memory_kind}
  Pool Info:
    Pool 1
      Segment: GLOBAL; FLAGS: COARSE GRAINED
      Size: {pool_kib}(0x3000000) KB
      Allocatable: TRUE
    Pool 2
      Segment: GLOBAL; FLAGS: EXTENDED FINE GRAINED
      Size: {pool_kib}(0x3000000) KB
      Allocatable: TRUE
"""
    snapshot = collect_hardware_snapshot(
        replace_fields(probe, rocminfo_result=CommandResult(returncode=0, stdout=info.encode(), stderr=b""))
    )
    expected = pool_kib * 1024 if memory_kind == "APU" else 512 * 1024**2
    assert snapshot.accelerators[0].memory_bytes == expected
    assert snapshot.accelerators[0].memory_is_unified == (memory_kind == "APU")
    assert available_accelerator_memory(snapshot, "amd-rocm") == expected


@pytest.mark.parametrize(
    ("probe", "expected_vendor", "expected_name", "expected_memory", "expected_api"),
    [
        (_macos_probe("spdisplays_mtlgpufamilysupport"), "Apple", "Apple M5 Pro", None, "Metal"),
        (_macos_probe("spdisplays_metal"), "Apple", "Apple M5 Pro", None, "Metal"),
        (
            _nvidia_probe(b"NVIDIA RTX PRO 6000 Blackwell, 97887, 590.42\n"),
            "NVIDIA",
            "NVIDIA RTX PRO 6000 Blackwell",
            97_887 * 1024**2,
            "CUDA",
        ),
        (
            _amd_probe(
                "rocm-smi",
                {
                    "card0": {
                        "Card series": "AMD Radeon PRO W7900",
                        "VRAM Total Memory (B)": "51527024640",
                        "Driver version": "6.4.1",
                    }
                },
            ),
            "AMD",
            "AMD Radeon PRO W7900",
            51_527_024_640,
            "ROCm",
        ),
        (
            _amd_probe(
                "amd-smi",
                [
                    {
                        "gpu": 0,
                        "asic": {"market_name": "AMD Instinct MI210", "asic_serial": "must-never-leave-this-machine"},
                        "vram": {"type": "HBM2e", "size": {"value": 65536, "unit": "MB"}},
                        "driver": {"driver_version": "6.4.1"},
                    }
                ],
            ),
            "AMD",
            "AMD Instinct MI210",
            64 * 1024**3,
            "ROCm",
        ),
    ],
)
def test_collect_hardware_snapshot_uses_only_allowlisted_facts(
    probe: FakeSystemProbe,
    expected_vendor: str,
    expected_name: str,
    expected_memory: int | None,
    expected_api: str,
) -> None:
    snapshot: HardwareSnapshot = collect_hardware_snapshot(probe)
    encoded = orjson.dumps(snapshot.to_json())

    assert snapshot.accelerators[0].vendor == expected_vendor
    assert snapshot.accelerators[0].name == expected_name
    assert snapshot.accelerators[0].memory_bytes == expected_memory
    assert snapshot.accelerators[0].api == expected_api
    assert b"must-never-leave-this-machine" not in encoded
    assert b"private-hardware-uuid" not in encoded
    assert b"serial" not in encoded.lower()
    assert b"uuid" not in encoded.lower()


@pytest.mark.parametrize(
    ("stdout", "expected_names", "expected_driver_versions", "expected_warnings"),
    [
        (
            b"NVIDIA RTX PRO 6000 Blackwell, [N/A], [N/A]\n",
            ("NVIDIA RTX PRO 6000 Blackwell",),
            (None,),
            (),
        ),
        (
            b"NVIDIA RTX PRO 6000 Blackwell, 97887, 590.42\nNVIDIA RTX PRO 6000 Blackwell, [N/A], [N/A]\n",
            ("NVIDIA RTX PRO 6000 Blackwell", "NVIDIA RTX PRO 6000 Blackwell"),
            ("590.42", None),
            (),
        ),
        (
            b"NVIDIA GPU @ rack-3, 97887, 590.42\n",
            (),
            (),
            ("nvidia_smi_unparseable", "no_supported_accelerator"),
        ),
        (
            b"NVIDIA RTX PRO 6000 Blackwell, 97887, 590.42\nNVIDIA GPU @ rack-3, 97887, 590.42\n",
            ("NVIDIA RTX PRO 6000 Blackwell",),
            ("590.42",),
            ("nvidia_smi_unparseable",),
        ),
    ],
)
def test_nvidia_placeholders_null_fields_and_unreadable_rows_drop(
    stdout: bytes,
    expected_names: tuple[str, ...],
    expected_driver_versions: tuple[str | None, ...],
    expected_warnings: tuple[str, ...],
) -> None:
    snapshot = collect_hardware_snapshot(_nvidia_probe(stdout))

    assert tuple(accelerator.name for accelerator in snapshot.accelerators) == expected_names
    assert tuple(accelerator.driver_version for accelerator in snapshot.accelerators) == expected_driver_versions
    assert snapshot.warnings == expected_warnings


@pytest.mark.parametrize(
    ("payload", "expected_accelerators"),
    [
        (
            [
                {
                    "gpu": 0,
                    "board": {"Card Series": "AMD Radeon RX 7900 XTX", "product_serial": "must-never-leave-this"},
                    "vram": {"size": {"value": 24560, "unit": "MB"}},
                    "driver": {"driver_version": "6.4.1"},
                }
            ],
            (("AMD Radeon RX 7900 XTX", 24560 * 1024**2, "6.4.1"),),
        ),
        (
            {
                "card0": {"Card Series": "AMD Radeon PRO W7900", "VRAM Total Memory (B)": "51527024640"},
                "system": {"Driver version": "6.8.5"},
            },
            (("AMD Radeon PRO W7900", 51_527_024_640, "6.8.5"),),
        ),
        (
            {
                "card0": {
                    "Card series": "N/A",
                    "Card vendor": "Advanced Micro Devices, Inc.",
                    "VRAM Total Memory (B)": "536870912",
                    "Driver version": "6.8.5",
                },
                "card1": {
                    "Card series": "AMD Radeon RX 7900 XTX",
                    "VRAM Total Memory (B)": "25757614080",
                    "Driver version": "6.8.5",
                },
            },
            (("AMD Radeon RX 7900 XTX", 25_757_614_080, "6.8.5"),),
        ),
    ],
)
def test_amd_probe_shapes_keep_memory_driver_and_readable_cards(
    payload: JsonValue,
    expected_accelerators: tuple[tuple[str, int | None, str | None], ...],
) -> None:
    snapshot = collect_hardware_snapshot(_amd_probe("amd-smi", payload))

    assert (
        tuple(
            (accelerator.name, accelerator.memory_bytes, accelerator.driver_version)
            for accelerator in snapshot.accelerators
        )
        == expected_accelerators
    )
    assert b"must-never-leave-this" not in orjson.dumps(snapshot.to_json())


def test_macos_reports_discrete_vendors_memory_and_drops_unpublishable_rows() -> None:
    probe = FakeSystemProbe(
        system="Darwin",
        cpu="i386",
        command_name="system_profiler",
        result=CommandResult(
            returncode=0,
            stdout=orjson.dumps(
                {
                    "SPHardwareDataType": [{"chip_type": "Intel Core i9"}],
                    "SPDisplaysDataType": [
                        {
                            "sppci_model": "AMD Radeon Pro 5500M",
                            "sppci_vendor": "sppci_vendor_Amd",
                            "spdisplays_vram": "8 GB",
                            "spdisplays_metal": "spdisplays_metal3",
                        },
                        {
                            "sppci_model": "Intel UHD Graphics 630",
                            "sppci_vendor": "sppci_vendor_intel",
                            "spdisplays_vram_shared": "1536 MB",
                        },
                        {"sppci_model": "N/A", "sppci_vendor": "sppci_vendor_Amd", "spdisplays_vram": "N/A"},
                        {"sppci_model": "/Library/GPUBundles/private-path"},
                    ],
                }
            ),
            stderr=b"",
        ),
    )

    snapshot = collect_hardware_snapshot(probe)

    assert tuple(
        (accelerator.vendor, accelerator.name, accelerator.memory_bytes) for accelerator in snapshot.accelerators
    ) == (
        ("AMD", "AMD Radeon Pro 5500M", 8 * 1024**3),
        ("Intel", "Intel UHD Graphics 630", 1536 * 1024**2),
        ("AMD", "AMD GPU", None),
    )
    assert b"private-path" not in orjson.dumps(snapshot.to_json())


@pytest.mark.parametrize(
    ("host_memory_bytes", "accelerator_memory_bytes", "expected_host_gib", "expected_accelerator_gib"),
    [
        (4 * 1024**3, 12 * 1024**3, 4, 12),
        (63 * 1024**3, 24560 * 1024**2, 56, 23),
        (64 * 1024**3, 97_887 * 1024**2, 64, 95),
    ],
)
def test_public_profile_floors_memory_to_its_bucket(
    host_memory_bytes: int,
    accelerator_memory_bytes: int,
    expected_host_gib: int,
    expected_accelerator_gib: int,
) -> None:
    snapshot = HardwareSnapshot(
        operating_system="Linux",
        operating_system_version="24.04",
        kernel_version="test-kernel",
        architecture="x86_64",
        cpu_model="test CPU",
        logical_cpu_count=16,
        memory_bytes=host_memory_bytes,
        accelerators=(
            AcceleratorSnapshot(
                vendor="NVIDIA",
                name="NVIDIA GeForce RTX 5090",
                memory_bytes=accelerator_memory_bytes,
                core_count=None,
                driver_version="590.42",
                api="CUDA",
            ),
        ),
        warnings=(),
    )

    profile = public_hardware_profile(snapshot)

    assert profile.host_memory_gib == expected_host_gib
    assert profile.accelerator.memory_gib == expected_accelerator_gib


@pytest.mark.parametrize(
    ("stdout", "fallback_stdout", "expected_envelope", "expected_warnings"),
    [
        (
            b"NVIDIA RTX PRO 6000 Blackwell, 97887, 590.42, 600.00, 2617, 14001\n",
            None,
            (600.0, 2617, 14001),
            (),
        ),
        (
            b"NVIDIA RTX PRO 6000 Blackwell, 97887, 590.42, [N/A], [Not Supported], 14001\n",
            None,
            (None, None, 14001),
            (),
        ),
        (
            b'Field "power.limit" is not a valid field to query.\n',
            b"NVIDIA RTX PRO 6000 Blackwell, 97887, 590.42\n",
            (None, None, None),
            (),
        ),
    ],
    ids=("extended", "placeholders", "legacy-driver-fallback"),
)
def test_nvidia_envelope_fields_stay_private_and_survive_a_legacy_driver(
    stdout: bytes,
    fallback_stdout: bytes | None,
    expected_envelope: tuple[float | None, int | None, int | None],
    expected_warnings: tuple[str, ...],
) -> None:
    probe = FakeSystemProbe(
        system="Linux",
        cpu="AMD Ryzen test CPU",
        command_name="nvidia-smi",
        result=CommandResult(returncode=0 if fallback_stdout is None else 1, stdout=stdout, stderr=b""),
        fallback_result=None
        if fallback_stdout is None
        else CommandResult(returncode=0, stdout=fallback_stdout, stderr=b""),
        base_frequency_mhz=3600,
    )

    snapshot = collect_hardware_snapshot(probe)
    accelerator = snapshot.accelerators[0]
    profile_json = orjson.dumps(public_hardware_profile(snapshot).to_json())
    round_tripped = hardware_snapshot_from_json(snapshot.to_json())

    assert (accelerator.power_limit_w, accelerator.max_graphics_clock_mhz, accelerator.max_memory_clock_mhz) == (
        expected_envelope
    )
    assert snapshot.warnings == expected_warnings
    assert snapshot.cpu_base_frequency_mhz == 3600
    assert round_tripped == snapshot
    assert b"power_limit" not in profile_json
    assert b"clock" not in profile_json
    assert b"base_frequency" not in profile_json
    assert b"3600" not in profile_json


@pytest.mark.parametrize(
    ("sysfs_text", "model_name", "expected_mhz"),
    [
        ("3600000\n", "AMD Ryzen 9 7950X 16-Core Processor", 3600),
        (None, "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz", 3600),
        (None, "AMD Ryzen 9 7950X 16-Core Processor", None),
    ],
    ids=("sysfs", "model-name", "unknown"),
)
def test_local_probe_reads_the_linux_cpu_base_frequency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sysfs_text: str | None,
    model_name: str,
    expected_mhz: int | None,
) -> None:
    cpu_info = tmp_path / "cpuinfo"
    cpu_info.write_text(f"processor\t: 0\nmodel name\t: {model_name}\n", encoding="utf-8")
    sysfs_path = tmp_path / "base_frequency"
    if sysfs_text is not None:
        sysfs_path.write_text(sysfs_text, encoding="ascii")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("agentperf_local.provenance.hardware_facts.LINUX_CPU_INFO_PATH", cpu_info)
    monkeypatch.setattr("agentperf_local.provenance.hardware_facts.LINUX_CPU_BASE_FREQUENCY_PATH", sysfs_path)

    assert LocalSystemProbe().cpu_base_frequency_mhz() == expected_mhz


def test_local_probe_reads_the_linux_cpu_model_without_its_frequency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cpu_info = tmp_path / "cpuinfo"
    cpu_info.write_text(
        "processor\t: 0\nmodel name\t: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz\nflags\t\t: fpu vme\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("agentperf_local.provenance.hardware_facts.LINUX_CPU_INFO_PATH", cpu_info)

    assert LocalSystemProbe().cpu_model() == "Intel(R) Core(TM) i7-9700K CPU"


@POSIX_MEMORY_PROBE
def test_local_probe_handles_platform_without_sysconf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "sysconf")

    assert LocalSystemProbe().total_memory_bytes() is None


@POSIX_MEMORY_PROBE
def test_local_probe_ignores_the_posix_sysconf_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sysconf", lambda name: -1)

    assert LocalSystemProbe().total_memory_bytes() is None


@pytest.mark.parametrize(
    "value",
    ["GPU\nprivate-host", "https://secret.example", "C:\\Users\\alice\\GPU", "GPU\u0085private", "GPU™"],
)
def test_hardware_snapshot_rejects_private_or_control_characters(value: str) -> None:
    with pytest.raises(ValueError):
        AcceleratorSnapshot(
            vendor="NVIDIA",
            name=value,
            memory_bytes=None,
            core_count=None,
            driver_version=None,
            api="CUDA",
        )


def test_failed_nvidia_probe_emits_only_safe_warning_codes() -> None:
    probe = FakeSystemProbe(
        system="Linux",
        cpu="test-cpu",
        command_name="nvidia-smi",
        result=CommandResult(returncode=1, stdout=b"private-host", stderr=b"secret-path"),
    )

    snapshot = collect_hardware_snapshot(probe)

    assert snapshot.accelerators == ()
    assert snapshot.warnings == ("nvidia_smi_failed", "no_supported_accelerator")
    assert b"private-host" not in orjson.dumps(snapshot.to_json())


def test_every_warning_code_has_a_rendered_message() -> None:
    from typing import get_args

    assert set(HARDWARE_WARNING_MESSAGES) == set(get_args(HardwareWarningCode.__value__))


@pytest.mark.parametrize(
    ("addressing", "expected_unified"),
    [
        # The driver names coherent addressing, so host memory is this device's capacity.
        (COHERENT_ADDRESSING, True),
        # A driver that answers but reports no coherent mode leaves the capacity unknown.
        (DISCRETE_ADDRESSING, False),
        # A driver that says nothing must not be read as unified memory.
        (b"", False),
    ],
)
def test_accelerator_without_a_framebuffer_is_unified_only_when_the_driver_says_so(
    addressing: bytes,
    expected_unified: bool,
) -> None:
    """An unreported memory size alone never becomes a claim about host memory."""
    snapshot = collect_hardware_snapshot(_nvidia_probe(GB10_ROW, addressing))

    accelerator = snapshot.accelerators[0]
    assert accelerator.name == "NVIDIA GB10"
    assert accelerator.memory_bytes is None
    assert accelerator.memory_is_unified is expected_unified


def test_a_reported_framebuffer_is_never_unified_memory() -> None:
    """A card with its own memory keeps it, whatever the addressing mode says."""
    snapshot = collect_hardware_snapshot(_nvidia_probe(DISCRETE_ROW, COHERENT_ADDRESSING))

    accelerator = snapshot.accelerators[0]
    assert accelerator.memory_bytes == 97_887 * 1024**2
    assert accelerator.memory_is_unified is False


class RecordingProbe:
    """Delegate to one fake probe while recording every command it is asked to run."""

    def __init__(self, inner: FakeSystemProbe) -> None:
        self._inner = inner
        self.commands: list[tuple[str, ...]] = []

    def system_name(self) -> str:
        return self._inner.system_name()

    def system_version(self) -> str:
        return self._inner.system_version()

    def kernel_version(self) -> str:
        return self._inner.kernel_version()

    def machine(self) -> str:
        return self._inner.machine()

    def cpu_model(self) -> str:
        return self._inner.cpu_model()

    def logical_cpu_count(self) -> int:
        return self._inner.logical_cpu_count()

    def cpu_base_frequency_mhz(self) -> int | None:
        return self._inner.cpu_base_frequency_mhz()

    def total_memory_bytes(self) -> int:
        return self._inner.total_memory_bytes()

    def command_available(self, command: str) -> bool:
        return self._inner.command_available(command)

    def run(self, command: tuple[str, ...]) -> CommandResult:
        self.commands.append(command)
        return self._inner.run(command)


def test_the_addressing_query_is_skipped_when_every_card_reports_its_memory() -> None:
    """`nvidia-smi -q` is slow, so only a card without a reported size is worth asking about."""
    probe = RecordingProbe(_nvidia_probe(DISCRETE_ROW, COHERENT_ADDRESSING))

    collect_hardware_snapshot(probe)

    assert probe.commands
    assert all("-q" not in command for command in probe.commands)
