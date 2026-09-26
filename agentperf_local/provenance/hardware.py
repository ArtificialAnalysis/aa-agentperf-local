"""Collect an identifier-free local hardware snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentperf_local.common.json_fields import (
    optional_integer,
    optional_number,
    optional_string,
    required_string,
)
from agentperf_local.common.json_records import json_record
from agentperf_local.common.json_types import JsonObject, JsonValue
from agentperf_local.common.units import (
    BYTES_PER_GIB,
)
from agentperf_local.provenance.accelerator_probes import inspect_amd, inspect_macos, inspect_nvidia
from agentperf_local.provenance.hardware_facts import (
    AcceleratorSnapshot,
    HardwareWarningCode,
    LocalSystemProbe,
    SystemProbe,
    validate_public_label,
    warning_code,
)

# Version 2 added memory_is_unified, without which a coherent-memory accelerator such as
# the GB10 reports no capacity at all and every managed recipe is refused on it.
HARDWARE_SNAPSHOT_VERSION = 2
PUBLIC_HARDWARE_PROFILE_VERSION = 1
HOST_MEMORY_BUCKET_GIB = 8


@dataclass(frozen=True, slots=True, kw_only=True)
class HardwareSnapshot:
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

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicAcceleratorProfile:
    """Describe the accelerator fields needed for public comparison."""

    vendor: str
    product: str
    memory_gib: int | None
    core_count: int | None
    driver_branch: str | None
    api: str | None

    def __post_init__(self) -> None:
        """Hold every label to the public contract, wherever the profile was built."""
        validate_public_label(self.vendor, "vendor")
        validate_public_label(self.product, "product")
        for field_name, value in (("driver_branch", self.driver_branch), ("api", self.api)):
            if value is not None:
                validate_public_label(value, field_name)

    def to_json(self) -> JsonObject:
        """Return the public accelerator profile."""
        return json_record(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicHardwareProfile:
    """Store normalized hardware facts approved for public sharing."""

    platform_family: str
    platform_major: str | None
    architecture: str
    host_memory_gib: int | None
    accelerator: PublicAcceleratorProfile
    version: int = PUBLIC_HARDWARE_PROFILE_VERSION

    def __post_init__(self) -> None:
        """Hold every label to the public contract, wherever the profile was built."""
        validate_public_label(self.platform_family, "platform_family")
        validate_public_label(self.architecture, "architecture")
        if self.platform_major is not None:
            validate_public_label(self.platform_major, "platform_major")

    def to_json(self) -> JsonObject:
        """Return the normalized public profile."""
        return {
            "version": self.version,
            "kind": "public_hardware_profile",
            "platform_family": self.platform_family,
            "platform_major": self.platform_major,
            "architecture": self.architecture,
            "host_memory_gib": self.host_memory_gib,
            "accelerator": self.accelerator.to_json(),
            "privacy_notice": "This combination can still fingerprint a device class.",
        }


type AcceleratorPlatform = Literal["nvidia-cuda", "amd-rocm", "apple-metal"]
ACCELERATOR_PLATFORMS: tuple[AcceleratorPlatform, ...] = ("nvidia-cuda", "amd-rocm", "apple-metal")


def selected_accelerator(snapshot: HardwareSnapshot) -> AcceleratorSnapshot:
    """Return the one accelerator a bound run measures, refusing an ambiguous host."""
    if len(snapshot.accelerators) != 1:
        raise ValueError(
            "a bound run requires exactly one detected accelerator; "
            f"found {len(snapshot.accelerators)} — choose one with --device"
        )
    return snapshot.accelerators[0]


def accelerator_platform(snapshot: HardwareSnapshot) -> AcceleratorPlatform:
    """Classify one detected accelerator for managed serving."""
    accelerator = selected_accelerator(snapshot)
    if accelerator.vendor == "NVIDIA" and accelerator.api == "CUDA":
        return "nvidia-cuda"
    if accelerator.vendor == "AMD" and accelerator.api == "ROCm":
        return "amd-rocm"
    if accelerator.vendor == "Apple" and accelerator.api == "Metal":
        return "apple-metal"
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


def _major_version(value: str) -> str | None:
    first = value.split(".", maxsplit=1)[0]
    return first if first.isdigit() else None


def _memory_gib(memory_bytes: int | None, bucket_gib: int) -> int | None:
    """Floor memory to the bucket so a public profile never claims more than the device has."""
    if memory_bytes is None:
        return None
    whole_gib = memory_bytes // BYTES_PER_GIB
    bucketed_gib = whole_gib // bucket_gib * bucket_gib
    if bucketed_gib > 0:
        return bucketed_gib
    # A device smaller than one bucket would floor to zero, so publish its whole count instead.
    return whole_gib if whole_gib > 0 else None


def _product_name(accelerator: AcceleratorSnapshot) -> str:
    name = accelerator.name
    if accelerator.vendor == "NVIDIA":
        for prefix in ("NVIDIA GeForce ", "NVIDIA "):
            if name.startswith(prefix):
                return name.removeprefix(prefix)
    return name


def public_hardware_profile(snapshot: HardwareSnapshot) -> PublicHardwareProfile:
    """Normalize one local snapshot for explicit public sharing."""
    if len(snapshot.accelerators) != 1:
        raise ValueError("public submissions require exactly one detected accelerator")
    accelerator = snapshot.accelerators[0]
    driver_branch = None
    if accelerator.driver_version is not None:
        driver_branch = _major_version(accelerator.driver_version)
    platform_major = _major_version(snapshot.operating_system_version)
    return PublicHardwareProfile(
        platform_family=snapshot.operating_system,
        platform_major=platform_major,
        architecture=snapshot.architecture,
        host_memory_gib=_memory_gib(snapshot.memory_bytes, HOST_MEMORY_BUCKET_GIB),
        accelerator=PublicAcceleratorProfile(
            vendor=accelerator.vendor,
            product=_product_name(accelerator),
            memory_gib=_memory_gib(accelerator.memory_bytes, 1),
            core_count=accelerator.core_count,
            driver_branch=driver_branch,
            api=accelerator.api,
        ),
    )
