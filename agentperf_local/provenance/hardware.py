"""Collect an identifier-free local hardware snapshot."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, model_validator

from agentperf_local.common.json_fields import (
    optional_integer,
    optional_number,
    optional_string,
    required_boolean,
    required_string,
)
from agentperf_local.common.json_types import JsonObject, JsonValue
from agentperf_local.common.models import replace_fields
from agentperf_local.provenance.accelerator_probes import (
    AMD_VENDOR,
    INTEL_VENDOR,
    MACOS_DEFAULT_VENDOR,
    NVIDIA_VENDOR,
    inspect_amd,
    inspect_intel,
    inspect_macos,
    inspect_nvidia,
)
from agentperf_local.provenance.hardware_facts import (
    AcceleratorSnapshot,
    HardwareWarningCode,
    LocalSystemProbe,
    SystemProbe,
    validate_public_label,
    warning_code,
)
from agentperf_local.submission.contract import AcceleratorVendor

# Version 2 added memory_is_unified, without which a coherent-memory accelerator such as
# the GB10 reports no capacity at all and every managed recipe is refused on it.
HARDWARE_SNAPSHOT_VERSION = 2


class HardwareSnapshot(BaseModel, frozen=True):
    """Describe public hardware and operating system facts."""

    operating_system: str
    operating_system_version: str
    kernel_version: str
    architecture: str
    cpu_model: str
    logical_cpu_count: int | None
    memory_bytes: int | None
    accelerators: tuple[AcceleratorSnapshot, ...]
    warnings: tuple[HardwareWarningCode, ...]
    cpu_base_frequency_mhz: int | None = None
    version: int = HARDWARE_SNAPSHOT_VERSION

    @model_validator(mode="after")
    def check_invariants(self) -> Self:
        """Reject unsafe or impossible public values."""
        for field, value in (
            ("operating_system", self.operating_system),
            ("operating_system_version", self.operating_system_version),
            ("kernel_version", self.kernel_version),
            ("architecture", self.architecture),
            ("cpu_model", self.cpu_model),
        ):
            validate_public_label(value, field)
        if self.logical_cpu_count is not None and self.logical_cpu_count <= 0:
            raise ValueError("logical_cpu_count must be positive")
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        if self.cpu_base_frequency_mhz is not None and self.cpu_base_frequency_mhz <= 0:
            raise ValueError("cpu_base_frequency_mhz must be positive")
        return self

    def to_json(self) -> JsonObject:
        """Return the snapshot as JSON data."""
        accelerators: list[JsonValue] = [accelerator.to_json() for accelerator in self.accelerators]
        return {
            "version": self.version,
            "kind": "hardware_snapshot",
            "operating_system": self.operating_system,
            "operating_system_version": self.operating_system_version,
            "kernel_version": self.kernel_version,
            "architecture": self.architecture,
            "cpu_model": self.cpu_model,
            "logical_cpu_count": self.logical_cpu_count,
            "cpu_base_frequency_mhz": self.cpu_base_frequency_mhz,
            "memory_bytes": self.memory_bytes,
            "accelerators": accelerators,
            "warnings": list(self.warnings),
        }


type AcceleratorPlatform = Literal["nvidia-cuda", "amd-rocm", "apple-metal"]


# The submission contract's name for each vendor label the probes write.
_CONTRACT_VENDORS: dict[str, AcceleratorVendor] = {
    NVIDIA_VENDOR: "nvidia",
    MACOS_DEFAULT_VENDOR: "apple",
    AMD_VENDOR: "amd",
    INTEL_VENDOR: "intel",
}


def contract_vendor(label: str) -> AcceleratorVendor:
    """Return the submission contract's name for one accelerator vendor, as a probe labels it."""
    vendor = _CONTRACT_VENDORS.get(label)
    if vendor is None:
        raise ValueError(f"a {label} accelerator cannot be submitted")
    return vendor


def selected_accelerator(snapshot: HardwareSnapshot) -> AcceleratorSnapshot:
    """Return the one accelerator a bound run measures, refusing an ambiguous host."""
    if len(snapshot.accelerators) != 1:
        raise ValueError(
            "a bound run requires exactly one detected accelerator; "
            f"found {len(snapshot.accelerators)} — choose one with --device"
        )
    return snapshot.accelerators[0]


def select_accelerator(snapshot: HardwareSnapshot, device_index: int | None) -> HardwareSnapshot:
    """Keep only the accelerator at device_index; with no index, keep the snapshot as it is."""
    if device_index is None:
        return snapshot
    count = len(snapshot.accelerators)
    if count == 0:
        raise ValueError("no accelerator was detected, so no device can be selected")
    if not 0 <= device_index < count:
        raise ValueError(
            f"device index {device_index} is out of range; detected accelerators run from 0 to {count - 1}"
        )
    return replace_fields(snapshot, accelerators=(snapshot.accelerators[device_index],))


def accelerator_platform(snapshot: HardwareSnapshot) -> AcceleratorPlatform:
    """Classify one detected accelerator for managed serving."""
    accelerator = selected_accelerator(snapshot)
    if accelerator.vendor == "NVIDIA" and accelerator.api == "CUDA":
        return "nvidia-cuda"
    if accelerator.vendor == "AMD" and accelerator.api == "ROCm":
        return "amd-rocm"
    if accelerator.vendor == "Apple" and accelerator.api == "Metal":
        return "apple-metal"
    if accelerator.vendor == INTEL_VENDOR:
        raise ValueError(
            "managed serving does not support Intel GPUs yet; start your own server and measure it with run"
        )
    raise ValueError("detected accelerator does not expose CUDA, ROCm, or Apple Metal")


def collect_hardware_snapshot(probe: SystemProbe | None = None) -> HardwareSnapshot:
    """Collect a snapshot that omits host and device identifiers."""
    selected_probe = probe if probe is not None else LocalSystemProbe()
    system_name = selected_probe.system_name()
    cpu_model = selected_probe.cpu_model()
    warnings: list[HardwareWarningCode] = []

    if system_name == "Darwin":
        mac = inspect_macos(selected_probe)
        accelerators = mac.accelerators
        cpu_model = mac.cpu_model or cpu_model
        if mac.warning is not None:
            warnings.append(mac.warning)
    else:
        accelerators, warning = inspect_nvidia(selected_probe)
        if warning is not None:
            warnings.append(warning)
        if not accelerators:
            accelerators, warning = inspect_amd(selected_probe)
            if warning is not None:
                warnings.append(warning)
        if not accelerators:
            accelerators, warning = inspect_intel(selected_probe)
            if warning is not None:
                warnings.append(warning)

    if not accelerators:
        warnings.append("no_supported_accelerator")
    return HardwareSnapshot(
        operating_system=system_name,
        operating_system_version=selected_probe.system_version(),
        kernel_version=selected_probe.kernel_version(),
        architecture=selected_probe.machine(),
        cpu_model=cpu_model,
        logical_cpu_count=selected_probe.logical_cpu_count(),
        memory_bytes=selected_probe.total_memory_bytes(),
        accelerators=accelerators,
        warnings=tuple(warnings),
        cpu_base_frequency_mhz=selected_probe.cpu_base_frequency_mhz(),
    )


def _accelerator_from_json(data: JsonObject, source: str) -> AcceleratorSnapshot:
    return AcceleratorSnapshot(
        vendor=required_string(data, "vendor", source),
        name=required_string(data, "name", source),
        memory_bytes=optional_integer(data, "memory_bytes", source),
        core_count=optional_integer(data, "core_count", source),
        driver_version=optional_string(data, "driver_version", source),
        api=optional_string(data, "api", source),
        memory_is_unified=required_boolean(data, "memory_is_unified", source),
        power_limit_w=optional_number(data, "power_limit_w", source),
        max_graphics_clock_mhz=optional_integer(data, "max_graphics_clock_mhz", source),
        max_memory_clock_mhz=optional_integer(data, "max_memory_clock_mhz", source),
    )


def hardware_snapshot_from_json(data: JsonObject) -> HardwareSnapshot:
    """Read and validate one public hardware snapshot."""
    version = data.get("version")
    if version != HARDWARE_SNAPSHOT_VERSION:
        raise ValueError("hardware.version is not supported")
    if data.get("kind") != "hardware_snapshot":
        raise ValueError("hardware.kind must be hardware_snapshot")

    accelerator_values = data.get("accelerators")
    if not isinstance(accelerator_values, list):
        raise ValueError("hardware.accelerators must be an array")
    accelerators: list[AcceleratorSnapshot] = []
    for index, value in enumerate(accelerator_values):
        if not isinstance(value, dict):
            raise ValueError(f"hardware.accelerators[{index}] must be an object")
        accelerators.append(_accelerator_from_json(value, f"hardware.accelerators[{index}]"))

    warning_values = data.get("warnings")
    if not isinstance(warning_values, list):
        raise ValueError("hardware.warnings must be an array")
    warnings = tuple(warning_code(value, f"hardware.warnings[{index}]") for index, value in enumerate(warning_values))
    return HardwareSnapshot(
        operating_system=required_string(data, "operating_system", "hardware"),
        operating_system_version=required_string(data, "operating_system_version", "hardware"),
        kernel_version=required_string(data, "kernel_version", "hardware"),
        architecture=required_string(data, "architecture", "hardware"),
        cpu_model=required_string(data, "cpu_model", "hardware"),
        logical_cpu_count=optional_integer(data, "logical_cpu_count", "hardware"),
        memory_bytes=optional_integer(data, "memory_bytes", "hardware"),
        accelerators=tuple(accelerators),
        warnings=warnings,
        cpu_base_frequency_mhz=optional_integer(data, "cpu_base_frequency_mhz", "hardware"),
    )
