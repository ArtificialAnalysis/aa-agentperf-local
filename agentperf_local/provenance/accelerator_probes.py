"""Read one accelerator's facts from the vendor tool that reports them.

Each probe runs one read-only command and parses its structured output into an
AcceleratorSnapshot. A field the tool omits stays None; a tool that cannot run
yields no accelerator and a warning code, never a guess.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import orjson
from pydantic import BaseModel

from agentperf_local.common.json_fields import lenient_integer, lenient_string
from agentperf_local.common.json_types import JsonObject, JsonValue, normalize_json, normalize_json_object
from agentperf_local.common.models import replace_fields
from agentperf_local.common.units import BYTES_PER_MIB
from agentperf_local.provenance.hardware_facts import (
    AcceleratorSnapshot,
    HardwareWarningCode,
    SystemProbe,
    public_text,
)

NVIDIA_SMI_BASE_FIELDS = ("name", "memory.total", "driver_version")


# The private snapshot also records the board power limit and clock caps so a
# plausibility check can compare measured draw against what the card allows.
NVIDIA_SMI_FIELDS = (*NVIDIA_SMI_BASE_FIELDS, "power.limit", "clocks.max.graphics", "clocks.max.memory")


def _nvidia_smi_command(fields: tuple[str, ...]) -> tuple[str, ...]:
    return ("nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")


NVIDIA_SMI_COMMAND = _nvidia_smi_command(NVIDIA_SMI_FIELDS)


# A driver that rejects one of the extended fields still answers the base query.
NVIDIA_SMI_BASE_COMMAND = _nvidia_smi_command(NVIDIA_SMI_BASE_FIELDS)


# An accelerator that addresses host memory coherently reports no framebuffer of its
# own; the driver names that mode, so it is read rather than guessed from a missing size.
NVIDIA_SMI_ADDRESSING_COMMAND = ("nvidia-smi", "-q")


NVIDIA_COHERENT_ADDRESSING_MODES = ("ATS", "HMM")


AMD_SMI_COMMAND = ("amd-smi", "static", "--gpu", "all", "--json")


ROCM_SMI_COMMAND = (
    "rocm-smi",
    "--showproductname",
    "--showmeminfo",
    "vram",
    "--showdriverversion",
    "--json",
)


# rocminfo is the only tool that says whether an AMD GPU is an APU and how large its GPU pool is.
ROCMINFO_COMMAND = ("rocminfo",)


# clinfo prints what Intel's compute runtime reports for each GPU, as JSON. Intel's GPU setup
# guides use clinfo to check that runtime, which llama.cpp SYCL and vLLM XPU both run on.
CLINFO_COMMAND = ("clinfo", "--json")


# The Intel compute runtime names itself this platform vendor. Another OpenCL platform, such
# as Mesa's rusticl, can list the same card a second time, so only this platform is read.
INTEL_PLATFORM_VENDOR_PREFIX = "Intel"


INTEL_PCI_VENDOR_ID = 0x8086


# CL_DEVICE_TYPE_GPU in the OpenCL headers. An Intel CPU runtime lists the processor too.
OPENCL_DEVICE_TYPE_GPU = 1 << 2


INTEL_VENDOR = "Intel"


INTEL_API = "OpenCL"


MACOS_PROFILE_COMMAND = ("system_profiler", "-json", "SPHardwareDataType", "SPDisplaysDataType")


# Modern macOS reports Metal support under the family key; the plain key survives on older releases.
MACOS_METAL_KEYS = ("spdisplays_mtlgpufamilysupport", "spdisplays_metal")


MACOS_VENDOR_KEY = "sppci_vendor"


MACOS_VENDOR_PREFIX = "sppci_vendor_"


MACOS_VENDOR_NAMES = {"amd": "AMD", "ati": "AMD", "intel": "Intel"}


MACOS_DEFAULT_VENDOR = "Apple"


# Intel Macs report discrete card memory; Apple silicon leaves both keys out because memory is unified.
MACOS_VRAM_KEYS = ("spdisplays_vram", "spdisplays_vram_shared")


MACOS_VRAM_DEFAULT_UNIT = "MB"


NVIDIA_SMI_MISSING_VALUES = frozenset(
    ("", "n/a", "[n/a]", "not supported", "[not supported]", "[unknown error]", "[insufficient permissions]")
)


def _optional_smi_text(value: str) -> str | None:
    """Return one nvidia-smi field value, or None for a missing-value placeholder."""
    return None if value.lower() in NVIDIA_SMI_MISSING_VALUES else value


def _optional_smi_number(value: str) -> float | None:
    """Return one numeric nvidia-smi field, or None when the driver reports no value."""
    text = _optional_smi_text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


def _optional_smi_integer(value: str) -> int | None:
    number = _optional_smi_number(value)
    return None if number is None else round(number)


class _MacInspection(BaseModel, frozen=True):
    cpu_model: str | None
    accelerators: tuple[AcceleratorSnapshot, ...]
    warning: HardwareWarningCode | None


def _macos_integer(data: JsonObject, key: str) -> int | None:
    """Read a count that system_profiler may report as a number or as digits."""
    value = data.get(key)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return lenient_integer(data, key)


def _object_rows(value: JsonValue | None) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


_MEMORY_VALUE_PATTERN = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)$")


# amd-smi, rocm-smi, and system_profiler label binary units as MB and GB, so both map to binary factors.
_MEMORY_UNIT_FACTORS = (
    ("b", 1),
    ("byte", 1),
    ("bytes", 1),
    ("kb", 1_000),
    ("kib", 1024),
    ("mb", 1024**2),
    ("mib", 1024**2),
    ("gb", 1024**3),
    ("gib", 1024**3),
)


def _memory_unit_factor(unit: str) -> int | None:
    normalized = unit.strip().lower()
    return next((factor for label, factor in _MEMORY_UNIT_FACTORS if label == normalized), None)


def _memory_bytes(value: JsonValue, default_unit: str) -> int | None:
    if isinstance(value, dict):
        amount = value.get("value", value.get("Value"))
        unit = value.get("unit", value.get("Unit", default_unit))
        if not isinstance(unit, str):
            return None
        return _memory_bytes(amount, unit)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        factor = _memory_unit_factor(default_unit)
        parsed = round(value * factor) if factor is not None else 0
        return parsed if parsed > 0 else None
    if not isinstance(value, str):
        return None
    match = _MEMORY_VALUE_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    factor = _memory_unit_factor(match.group(2) or default_unit)
    if factor is None:
        return None
    parsed = round(float(match.group(1)) * factor)
    return parsed if parsed > 0 else None


def _macos_graphics_api(row: JsonObject) -> str | None:
    if any(lenient_string(row, key) is not None for key in MACOS_METAL_KEYS):
        return "Metal"
    return None


def _macos_vendor(row: JsonObject) -> str:
    """Map the system_profiler vendor tag to a public vendor name."""
    tag = public_text(lenient_string(row, MACOS_VENDOR_KEY))
    if tag is None:
        return MACOS_DEFAULT_VENDOR
    return MACOS_VENDOR_NAMES.get(tag.removeprefix(MACOS_VENDOR_PREFIX).lower(), MACOS_DEFAULT_VENDOR)


def _macos_memory_bytes(row: JsonObject) -> int | None:
    """Read the card memory that Intel Macs report beside a discrete GPU."""
    for key in MACOS_VRAM_KEYS:
        value = public_text(lenient_string(row, key))
        if value is not None and (memory_bytes := _memory_bytes(value, MACOS_VRAM_DEFAULT_UNIT)) is not None:
            return memory_bytes
    return None


def _accelerator_or_none(
    *,
    vendor: str,
    name: str,
    memory_bytes: int | None,
    core_count: int | None,
    driver_version: str | None,
    api: str | None,
    memory_is_unified: bool = False,
) -> AcceleratorSnapshot | None:
    """Drop one device whose probe placeholders fail the public label rules."""
    try:
        return AcceleratorSnapshot(
            vendor=vendor,
            name=name,
            memory_bytes=memory_bytes,
            core_count=core_count,
            driver_version=driver_version,
            api=api,
            memory_is_unified=memory_is_unified,
        )
    except ValueError:
        return None


def _macos_accelerator(row: JsonObject) -> AcceleratorSnapshot | None:
    vendor = _macos_vendor(row)
    name = public_text(lenient_string(row, "sppci_model") or lenient_string(row, "_name"))
    return _accelerator_or_none(
        vendor=vendor,
        name=name if name is not None else f"{vendor} GPU",
        memory_bytes=_macos_memory_bytes(row),
        core_count=_macos_integer(row, "sppci_cores"),
        driver_version=None,
        api=_macos_graphics_api(row),
        memory_is_unified=_macos_memory_bytes(row) is None,
    )


def inspect_macos(probe: SystemProbe) -> _MacInspection:
    if not probe.command_available(MACOS_PROFILE_COMMAND[0]):
        return _MacInspection(cpu_model=None, accelerators=(), warning="system_profiler_unavailable")
    result = probe.run(MACOS_PROFILE_COMMAND)
    if result.returncode != 0:
        return _MacInspection(cpu_model=None, accelerators=(), warning="system_profiler_failed")
    try:
        profile = normalize_json_object(orjson.loads(result.stdout))
    except (orjson.JSONDecodeError, ValueError):
        return _MacInspection(cpu_model=None, accelerators=(), warning="system_profiler_invalid_json")

    hardware_rows = _object_rows(profile.get("SPHardwareDataType"))
    cpu_model = lenient_string(hardware_rows[0], "chip_type") if hardware_rows else None
    accelerators = tuple(
        accelerator
        for row in _object_rows(profile.get("SPDisplaysDataType"))
        if (accelerator := _macos_accelerator(row)) is not None
    )
    warning = None if accelerators else "system_profiler_no_accelerator"
    return _MacInspection(cpu_model=cpu_model, accelerators=accelerators, warning=warning)


def _nvidia_row_fields(row: str) -> tuple[str, ...] | None:
    """Split one base or extended nvidia-smi row, padding the optional fields."""
    fields = tuple(field.strip() for field in row.split(","))
    if len(fields) not in (len(NVIDIA_SMI_BASE_FIELDS), len(NVIDIA_SMI_FIELDS)):
        return None
    # A base row carries no envelope fields; empty placeholders read back as None.
    return fields + ("",) * (len(NVIDIA_SMI_FIELDS) - len(fields))


def _nvidia_row_memory_bytes(row: str) -> int | None:
    """Read one nvidia-smi row's reported memory, or None when it names no size."""
    fields = _nvidia_row_fields(row)
    if fields is None:
        return None
    try:
        return round(float(fields[1]) * BYTES_PER_MIB)
    except ValueError:
        return None


def _parse_nvidia_row(row: str, *, memory_is_unified: bool) -> AcceleratorSnapshot | None:
    """Parse one base or extended nvidia-smi row; the extended fields are optional."""
    fields = _nvidia_row_fields(row)
    if fields is None:
        return None
    name, _, driver_version, power_limit, graphics_clock, memory_clock = fields
    memory_bytes = _nvidia_row_memory_bytes(row)
    try:
        return AcceleratorSnapshot(
            vendor="NVIDIA",
            name=name,
            memory_bytes=memory_bytes,
            core_count=None,
            driver_version=_optional_smi_text(driver_version),
            api="CUDA",
            memory_is_unified=memory_is_unified and memory_bytes is None,
            power_limit_w=_optional_smi_number(power_limit),
            max_graphics_clock_mhz=_optional_smi_integer(graphics_clock),
            max_memory_clock_mhz=_optional_smi_integer(memory_clock),
        )
    except ValueError:
        # A row whose remaining labels still fail the public rules hides no usable device.
        return None


def _nvidia_addressing_is_coherent(probe: SystemProbe) -> bool:
    """Report whether the driver says this GPU addresses host memory coherently.

    A part such as the GB10 has no framebuffer of its own, so nvidia-smi reports its
    memory as unavailable. Reading the addressing mode tells that case apart from a
    driver that simply failed to answer, which must stay an unknown capacity.
    """
    result = probe.run(NVIDIA_SMI_ADDRESSING_COMMAND)
    if result.returncode != 0:
        return False
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        label, separator, value = line.partition(":")
        if separator and label.strip() == "Addressing Mode":
            return value.strip() in NVIDIA_COHERENT_ADDRESSING_MODES
    return False


def inspect_nvidia(probe: SystemProbe) -> tuple[tuple[AcceleratorSnapshot, ...], HardwareWarningCode | None]:
    if not probe.command_available(NVIDIA_SMI_COMMAND[0]):
        return (), None
    result = probe.run(NVIDIA_SMI_COMMAND)
    if result.returncode != 0:
        # An older driver may reject an extended field; the base facts still matter.
        result = probe.run(NVIDIA_SMI_BASE_COMMAND)
    if result.returncode != 0:
        return (), "nvidia_smi_failed"
    rows = tuple(row for row in result.stdout.decode("utf-8", errors="replace").splitlines() if row.strip())
    # Only a row without a reported size can be a coherent-memory part, so the second and
    # much slower query runs only when one appears.
    coherent = any(_nvidia_row_memory_bytes(row) is None for row in rows) and _nvidia_addressing_is_coherent(probe)
    accelerators = tuple(
        accelerator for row in rows if (accelerator := _parse_nvidia_row(row, memory_is_unified=coherent)) is not None
    )
    # A dropped row hides a real device, so warn even when the other rows parsed.
    if not accelerators or len(accelerators) != len(rows):
        return accelerators, "nvidia_smi_unparseable"
    return accelerators, None


# amd-smi and rocm-smi disagree on capitalization across releases, so keys are matched in lower case.
_AMD_NAME_KEYS = frozenset(("card series", "market name", "market_name", "product_name"))


_AMD_DRIVER_KEYS = frozenset(("driver version", "driver_version"))


# One of these blocks holds the device name while the vram and driver blocks sit beside it.
_AMD_DEVICE_BLOCK_KEYS = frozenset(("asic", "board", "product"))


# ROCm 6 reports the driver once for the host instead of once per card.
_AMD_SYSTEM_KEY = "system"


_AMD_MEMORY_FIELDS = (
    ("VRAM Total Memory (B)", "B"),
    ("vram_size_bytes", "B"),
    ("vram_size", "B"),
)


_AMD_VRAM_MEMORY_FIELDS = ("size", "total", "total_memory")


def _nested_texts(data: JsonObject, keys: frozenset[str], *, recurse: bool = True) -> Iterator[str]:
    """Yield the text stored under one of the lower-case keys, outermost first."""
    for key, value in data.items():
        if key.lower() in keys and isinstance(value, str):
            yield value
    if not recurse:
        return
    for value in data.values():
        if isinstance(value, dict):
            yield from _nested_texts(value, keys)


def _nested_text(data: JsonObject, keys: frozenset[str], *, recurse: bool = True) -> str | None:
    """Return the first real text stored under one of the lower-case keys."""
    return next(
        (text for value in _nested_texts(data, keys, recurse=recurse) if (text := public_text(value)) is not None),
        None,
    )


def _nested_memory_bytes(value: JsonValue) -> int | None:
    if isinstance(value, list):
        for item in value:
            if memory_bytes := _nested_memory_bytes(item):
                return memory_bytes
        return None
    if not isinstance(value, dict):
        return None
    for key, default_unit in _AMD_MEMORY_FIELDS:
        if key in value and (memory_bytes := _memory_bytes(value[key], default_unit)):
            return memory_bytes
    vram = value.get("vram")
    if isinstance(vram, dict):
        for key in _AMD_VRAM_MEMORY_FIELDS:
            if key in vram and (memory_bytes := _memory_bytes(vram[key], "B")):
                return memory_bytes
    for nested in value.values():
        if memory_bytes := _nested_memory_bytes(nested):
            return memory_bytes
    return None


def _names_a_device(value: JsonObject) -> bool:
    """Report whether this object is one device rather than a container of devices."""
    if _nested_text(value, _AMD_NAME_KEYS, recurse=False) is not None:
        return True
    blocks = (item for key, item in value.items() if key.lower() in _AMD_DEVICE_BLOCK_KEYS and isinstance(item, dict))
    return any(_nested_text(block, _AMD_NAME_KEYS) is not None for block in blocks)


def _amd_device_objects(value: JsonValue) -> tuple[JsonObject, ...]:
    """Return the outermost object per device so vram and driver blocks stay reachable."""
    if isinstance(value, list):
        devices: list[JsonObject] = []
        for item in value:
            devices.extend(_amd_device_objects(item))
        return tuple(devices)
    if not isinstance(value, dict):
        return ()
    if _names_a_device(value):
        return (value,)
    devices = []
    for item in value.values():
        devices.extend(_amd_device_objects(item))
    return tuple(devices)


def _amd_system_driver_version(root: JsonValue) -> str | None:
    if not isinstance(root, dict):
        return None
    system = root.get(_AMD_SYSTEM_KEY)
    return _nested_text(system, _AMD_DRIVER_KEYS) if isinstance(system, dict) else None


def _parse_amd_output(encoded: bytes) -> tuple[AcceleratorSnapshot, ...]:
    try:
        # amd-smi prints a top-level array of per-GPU objects; rocm-smi prints an object keyed by card.
        root = normalize_json(orjson.loads(encoded))
    except (orjson.JSONDecodeError, ValueError):
        return ()
    system_driver_version = _amd_system_driver_version(root)
    accelerators: list[AcceleratorSnapshot] = []
    for device in _amd_device_objects(root):
        name = _nested_text(device, _AMD_NAME_KEYS)
        if name is None:
            continue
        accelerator = _accelerator_or_none(
            vendor="AMD",
            name=name,
            memory_bytes=_nested_memory_bytes(device),
            core_count=None,
            driver_version=_nested_text(device, _AMD_DRIVER_KEYS) or system_driver_version,
            api="ROCm",
        )
        if accelerator is not None:
            accelerators.append(accelerator)
    return tuple(accelerators)


def _amd_apu_memory(
    probe: SystemProbe, accelerators: tuple[AcceleratorSnapshot, ...]
) -> tuple[AcceleratorSnapshot, ...]:
    """Use the allocatable GPU pool for APUs explicitly identified by ROCm."""
    if not probe.command_available(ROCMINFO_COMMAND[0]):
        return accelerators
    result = probe.run(ROCMINFO_COMMAND)
    if result.returncode != 0:
        return accelerators
    updated = list(accelerators)
    for agent in re.split(r"(?m)^Agent \d+\s*$", result.stdout.decode("utf-8", errors="replace")):
        if not re.search(r"(?m)^\s*Device Type:\s*GPU\s*$", agent):
            continue
        if not re.search(r"(?m)^\s*Memory Properties:\s*APU\s*$", agent):
            continue
        name = re.search(r"(?m)^\s*Marketing Name:\s*(.*?)\s*$", agent)
        if name is None:
            continue
        capacities: list[int] = []
        for pool in re.split(r"(?m)^\s*Pool \d+\s*$", agent)[1:]:
            if not re.search(r"(?m)^\s*Segment:\s*GLOBAL;", pool):
                continue
            if not re.search(r"(?m)^\s*Allocatable:\s*TRUE\s*$", pool):
                continue
            size = re.search(r"(?m)^\s*Size:\s*(\d+)(?:\(0x[0-9a-f]+\))?\s*KB\s*$", pool)
            if size is not None:
                capacities.append(int(size.group(1)) * 1024)
        matches = [index for index, accelerator in enumerate(updated) if accelerator.name == name.group(1)]
        if capacities and len(matches) == 1 and max(capacities) > 0:
            index = matches[0]
            updated[index] = replace_fields(updated[index], memory_bytes=max(capacities), memory_is_unified=True)
    return tuple(updated)


def inspect_amd(probe: SystemProbe) -> tuple[tuple[AcceleratorSnapshot, ...], HardwareWarningCode | None]:
    available_commands = tuple(
        command for command in (AMD_SMI_COMMAND, ROCM_SMI_COMMAND) if probe.command_available(command[0])
    )
    if not available_commands:
        return (), None
    saw_success = False
    for command in available_commands:
        result = probe.run(command)
        if result.returncode != 0:
            continue
        saw_success = True
        if accelerators := _parse_amd_output(result.stdout):
            return _amd_apu_memory(probe, accelerators), None
    return (), "amd_smi_unparseable" if saw_success else "amd_smi_failed"


def _is_intel_gpu(device: JsonObject) -> bool:
    """Return whether one clinfo device row is an Intel GPU rather than a CPU or another vendor."""
    device_type = device.get("CL_DEVICE_TYPE")
    type_bits = lenient_integer(device_type, "raw") if isinstance(device_type, dict) else None
    return (
        lenient_integer(device, "CL_DEVICE_VENDOR_ID") == INTEL_PCI_VENDOR_ID
        and type_bits is not None
        and type_bits & OPENCL_DEVICE_TYPE_GPU != 0
    )


def _intel_gpu_rows(root: JsonObject) -> tuple[JsonObject, ...] | None:
    """Return the Intel runtime's GPU rows, or None when the platform and device lists are malformed."""
    platforms = root.get("platforms")
    if not isinstance(platforms, list):
        return None
    if not platforms:
        # clinfo leaves out the device list when no OpenCL platform is installed.
        return ()
    device_groups = root.get("devices")
    # clinfo lists each platform's devices at the same position as the platform.
    if not isinstance(device_groups, list) or len(device_groups) != len(platforms):
        return None
    rows: list[JsonObject] = []
    for platform, group in zip(platforms, device_groups, strict=True):
        if not isinstance(platform, dict) or not isinstance(group, dict):
            return None
        vendor = lenient_string(platform, "CL_PLATFORM_VENDOR")
        if vendor is None or not vendor.startswith(INTEL_PLATFORM_VENDOR_PREFIX):
            continue
        rows.extend(device for device in _object_rows(group.get("online")) if _is_intel_gpu(device))
    return tuple(rows)


def _intel_accelerator(device: JsonObject) -> AcceleratorSnapshot | None:
    """Parse one Intel GPU row, or return None when its name is missing or fails the public label rules.

    The runtime reports unified memory for a GPU without local memory of its own. Its
    global memory size is then a share of host memory the runtime picks, so it is not
    recorded as the GPU's memory.
    """
    name = public_text(lenient_string(device, "CL_DEVICE_NAME"))
    if name is None:
        return None
    unified = device.get("CL_DEVICE_HOST_UNIFIED_MEMORY") is True
    global_memory_bytes = lenient_integer(device, "CL_DEVICE_GLOBAL_MEM_SIZE")
    return _accelerator_or_none(
        vendor=INTEL_VENDOR,
        name=name,
        memory_bytes=None
        if unified or global_memory_bytes is None or global_memory_bytes <= 0
        else global_memory_bytes,
        core_count=None,
        driver_version=public_text(lenient_string(device, "CL_DRIVER_VERSION")),
        api=INTEL_API,
        memory_is_unified=unified,
    )


def inspect_intel(probe: SystemProbe) -> tuple[tuple[AcceleratorSnapshot, ...], HardwareWarningCode | None]:
    """Read Intel GPUs from clinfo; a host without clinfo or without the Intel runtime reports none."""
    if not probe.command_available(CLINFO_COMMAND[0]):
        return (), None
    result = probe.run(CLINFO_COMMAND)
    if result.returncode != 0:
        return (), "clinfo_failed"
    try:
        root = normalize_json_object(orjson.loads(result.stdout))
    except (orjson.JSONDecodeError, ValueError):
        return (), "clinfo_unparseable"
    rows = _intel_gpu_rows(root)
    if rows is None:
        return (), "clinfo_unparseable"
    accelerators = tuple(accelerator for row in rows if (accelerator := _intel_accelerator(row)) is not None)
    # A dropped row hides a real device, so warn even when the other rows parsed.
    return accelerators, None if len(accelerators) == len(rows) else "clinfo_unparseable"
