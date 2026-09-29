"""Define what a hardware probe may report and the rules those facts must satisfy.

A public label carries no serial, address, or path. A probe that cannot run
reports a warning code, never a guess.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, model_validator

from agentperf_local.common.json_fields import one_of
from agentperf_local.common.json_records import json_record
from agentperf_local.common.json_types import JsonObject, JsonValue
from agentperf_local.common.units import KILOHERTZ_PER_MEGAHERTZ, MEGAHERTZ_PER_GIGAHERTZ

INSPECTION_TIMEOUT_SECONDS = 15.0


MAX_PUBLIC_LABEL_CHARACTERS = 160


# amd-smi, rocm-smi, and system_profiler print these instead of leaving a field out.
PLACEHOLDER_TEXT_VALUES = frozenset(("", "n/a"))


type HardwareWarningCode = Literal[
    "no_supported_accelerator",
    "amd_smi_failed",
    "amd_smi_unparseable",
    "clinfo_failed",
    "clinfo_unparseable",
    "nvidia_smi_failed",
    "nvidia_smi_unparseable",
    "system_profiler_failed",
    "system_profiler_invalid_json",
    "system_profiler_no_accelerator",
    "system_profiler_unavailable",
]


# Rendered beside the codes so a new code cannot ship without its message.
HARDWARE_WARNING_MESSAGES: dict[HardwareWarningCode, str] = {
    "no_supported_accelerator": "No supported accelerator was detected",
    "amd_smi_failed": "amd-smi or rocm-smi failed",
    "amd_smi_unparseable": "amd-smi or rocm-smi returned no parseable accelerators",
    "clinfo_failed": "clinfo failed",
    "clinfo_unparseable": "clinfo returned Intel GPU facts that could not be parsed",
    "nvidia_smi_failed": "nvidia-smi failed",
    "nvidia_smi_unparseable": "nvidia-smi returned no parseable accelerators",
    "system_profiler_failed": "system_profiler failed",
    "system_profiler_invalid_json": "system_profiler returned invalid JSON",
    "system_profiler_no_accelerator": "system_profiler did not report an Apple accelerator",
    "system_profiler_unavailable": "system_profiler is unavailable",
}


HARDWARE_WARNING_CODES: tuple[HardwareWarningCode, ...] = tuple(HARDWARE_WARNING_MESSAGES)


def warning_code(value: JsonValue, source: str) -> HardwareWarningCode:
    if not isinstance(value, str):
        raise ValueError(f"{source} is not a supported hardware warning code")
    return one_of(value, HARDWARE_WARNING_CODES, source)


def validate_public_label(value: str, field: str) -> None:
    if not value or len(value) > MAX_PUBLIC_LABEL_CHARACTERS or not value.isascii() or not value.isprintable():
        raise ValueError(f"{field} must be short printable ASCII text")
    if any(marker in value for marker in ("/", "\\", "@", "<", ">")):
        raise ValueError(f"{field} contains a path, address, or markup character")


def public_text(value: str | None) -> str | None:
    """Return real text, treating probe placeholders such as N/A as missing."""
    if value is None:
        return None
    trimmed = value.strip()
    return None if trimmed.lower() in PLACEHOLDER_TEXT_VALUES else trimmed


LINUX_CPU_INFO_PATH = Path("/proc/cpuinfo")


LINUX_CPU_MODEL_KEY = "model name"


# intel_pstate publishes the base frequency here in kHz; other drivers leave it out.
LINUX_CPU_BASE_FREQUENCY_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency")


# The public label rules reject "@", so the trailing " @ 3.60GHz" frequency must go.
CPU_FREQUENCY_SUFFIX_PATTERN = re.compile(r"\s*@.*$")


# Intel model names carry the base frequency; the private snapshot keeps it as a number.
CPU_MODEL_FREQUENCY_PATTERN = re.compile(r"@\s*([0-9]+(?:\.[0-9]+)?)\s*GHz", re.IGNORECASE)


def _public_cpu_model(value: str) -> str | None:
    """Return one CPU model that the public label rules accept."""
    collapsed = " ".join(CPU_FREQUENCY_SUFFIX_PATTERN.sub("", value).split())
    try:
        validate_public_label(collapsed, "cpu_model")
    except ValueError:
        return None
    return collapsed


def _linux_raw_cpu_model() -> str | None:
    """Read the raw CPU model line, frequency suffix included."""
    try:
        with LINUX_CPU_INFO_PATH.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                key, separator, value = line.partition(":")
                if separator and key.strip() == LINUX_CPU_MODEL_KEY:
                    return value
    except OSError:
        return None
    return None


def _linux_cpu_model() -> str | None:
    """Read the CPU model that platform.processor() leaves empty on Linux."""
    raw = _linux_raw_cpu_model()
    return None if raw is None else _public_cpu_model(raw)


WINDOWS_CPU_REGISTRY_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"


WINDOWS_CPU_MODEL_VALUE = "ProcessorNameString"


def _windows_cpu_model() -> str | None:
    """Read the CPU model from the registry; platform.processor() gives only a family code on Windows."""
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, WINDOWS_CPU_REGISTRY_KEY) as key:
            value, _ = winreg.QueryValueEx(key, WINDOWS_CPU_MODEL_VALUE)
    except OSError:
        return None
    return _public_cpu_model(value) if isinstance(value, str) else None


class _WindowsMemoryStatus(ctypes.Structure):
    """Mirror MEMORYSTATUSEX, the only argument GlobalMemoryStatusEx takes."""

    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _windows_total_memory_bytes() -> int | None:
    """Read installed memory from GlobalMemoryStatusEx, since Windows has no os.sysconf."""
    if sys.platform != "win32":
        return None
    status = _WindowsMemoryStatus()
    # The call reads dwLength to learn which structure version the caller passed.
    status.dwLength = ctypes.sizeof(_WindowsMemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    total = status.ullTotalPhys
    return total if isinstance(total, int) and total > 0 else None


def _linux_cpu_base_frequency_mhz() -> int | None:
    """Read the CPU base frequency from sysfs, or from the model name when sysfs has none."""
    try:
        kilohertz = int(LINUX_CPU_BASE_FREQUENCY_PATH.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        kilohertz = None
    if kilohertz is not None and kilohertz > 0:
        return kilohertz // KILOHERTZ_PER_MEGAHERTZ
    raw = _linux_raw_cpu_model()
    match = CPU_MODEL_FREQUENCY_PATTERN.search(raw) if raw is not None else None
    if match is None:
        return None
    megahertz = round(float(match.group(1)) * MEGAHERTZ_PER_GIGAHERTZ)
    return megahertz if megahertz > 0 else None


class CommandResult(BaseModel, frozen=True):
    """Store one command result."""

    returncode: int
    stdout: bytes
    stderr: bytes


class SystemProbe(Protocol):
    """Provide the host facts needed by the collector."""

    def system_name(self) -> str:
        """Return the operating system name."""
        ...

    def system_version(self) -> str:
        """Return the operating system version."""
        ...

    def kernel_version(self) -> str:
        """Return the kernel version."""
        ...

    def machine(self) -> str:
        """Return the machine architecture."""
        ...

    def cpu_model(self) -> str:
        """Return the best available CPU model."""
        ...

    def logical_cpu_count(self) -> int | None:
        """Return the logical CPU count."""
        ...

    def total_memory_bytes(self) -> int | None:
        """Return installed memory in bytes."""
        ...

    def cpu_base_frequency_mhz(self) -> int | None:
        """Return the CPU base frequency in MHz when the platform reports one."""
        ...

    def command_available(self, command: str) -> bool:
        """Return whether a command can be executed."""
        ...

    def run(self, command: tuple[str, ...]) -> CommandResult:
        """Run one narrow inspection command."""
        ...


class LocalSystemProbe:
    """Read facts from the local operating system."""

    def system_name(self) -> str:
        """Return the operating system name."""
        return platform.system()

    def system_version(self) -> str:
        """Return the operating system version."""
        system_name = platform.system()
        if system_name == "Darwin":
            return platform.mac_ver()[0]
        if system_name == "Linux":
            try:
                release = platform.freedesktop_os_release()
            except OSError:
                return platform.release()
            return release.get("VERSION_ID") or platform.release()
        if system_name == "Windows":
            # release() is only "10" or "11"; version() carries the build, as "10.0.26100".
            return platform.version()
        return platform.release()

    def kernel_version(self) -> str:
        """Return the kernel version."""
        return platform.release()

    def machine(self) -> str:
        """Return the machine architecture."""
        return platform.machine()

    def cpu_model(self) -> str:
        """Return the best available CPU model."""
        if platform.system() == "Linux":
            model = _linux_cpu_model()
            if model is not None:
                return model
        if platform.system() == "Windows":
            model = _windows_cpu_model()
            if model is not None:
                return model
        return platform.processor() or platform.machine()

    def logical_cpu_count(self) -> int | None:
        """Return the logical CPU count."""
        return os.cpu_count()

    def total_memory_bytes(self) -> int | None:
        """Return installed memory in bytes."""
        if sys.platform == "win32":
            return _windows_total_memory_bytes()
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            physical_pages = os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            return None
        if not isinstance(page_size, int) or not isinstance(physical_pages, int):
            return None
        # POSIX answers an unsupported name with -1, which is not a memory size.
        if page_size <= 0 or physical_pages <= 0:
            return None
        return page_size * physical_pages

    def cpu_base_frequency_mhz(self) -> int | None:
        """Return the base frequency on Linux; other platforms do not publish one."""
        if platform.system() == "Linux":
            return _linux_cpu_base_frequency_mhz()
        return None

    def command_available(self, command: str) -> bool:
        """Return whether a command can be executed."""
        return shutil.which(command) is not None

    def run(self, command: tuple[str, ...]) -> CommandResult:
        """Run one narrow inspection command."""
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=INSPECTION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return CommandResult(returncode=1, stdout=b"", stderr=b"")
        return CommandResult(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


class AcceleratorSnapshot(BaseModel, frozen=True):
    """Describe one accelerator without stable device identifiers."""

    vendor: str
    name: str
    memory_bytes: int | None
    core_count: int | None
    driver_version: str | None
    api: str | None
    # True when the accelerator has no memory of its own and draws on host memory, as
    # Apple Silicon and NVIDIA's coherent-memory parts do. memory_bytes is None unless
    # the driver reports a GPU-accessible pool limit, as ROCm does for AMD APUs.
    memory_is_unified: bool = False
    # Private-snapshot envelope facts. The public profile never copies them.
    power_limit_w: float | None = None
    max_graphics_clock_mhz: int | None = None
    max_memory_clock_mhz: int | None = None

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Reject unsafe or impossible public values."""
        for field, value in (("vendor", self.vendor), ("name", self.name)):
            validate_public_label(value, field)
        for field, value in (("driver_version", self.driver_version), ("api", self.api)):
            if value is not None:
                validate_public_label(value, field)
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        if self.core_count is not None and self.core_count <= 0:
            raise ValueError("core_count must be positive")
        if self.power_limit_w is not None and (not math.isfinite(self.power_limit_w) or self.power_limit_w <= 0):
            raise ValueError("power_limit_w must be positive")
        for field, clock in (
            ("max_graphics_clock_mhz", self.max_graphics_clock_mhz),
            ("max_memory_clock_mhz", self.max_memory_clock_mhz),
        ):
            if clock is not None and clock <= 0:
                raise ValueError(f"{field} must be positive")
        return self

    def to_json(self) -> JsonObject:
        """Return the accelerator as JSON data."""
        return json_record(self)
