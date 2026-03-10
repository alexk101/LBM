"""GPU profiling utilities with a unified interface for AMD (ROCm) and NVIDIA."""

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

# -----------------------------------------------------------------------------
# Abstract base class
# -----------------------------------------------------------------------------


def _bytes_to_gib(b: int) -> float:
    return b / (1024**3)


class GPUProfiler(ABC):
    """Abstract base class for GPU profilers (ROCm, NVIDIA, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging, e.g. 'nvidia' or 'amd'."""
        ...

    @abstractmethod
    def getPower(self, device: int) -> Dict[str, Any]:
        """Get power usage for a GPU.

        Returns:
            Dict with at least "power" (numeric or string), "unit" (e.g. "W" or "mW"),
            and implementation-specific keys.
        """
        ...

    @abstractmethod
    def listDevices(self) -> List[int]:
        """Return list of available device indices."""
        ...

    @abstractmethod
    def getMemInfo(self, device: int) -> Tuple[int, int]:
        """Get memory info (used, total) in bytes for a GPU."""
        ...

    @abstractmethod
    def getUtilization(self, device: int) -> float:
        """Get GPU utilization as a value in [0, 100] (percentage)."""
        ...

    @abstractmethod
    def getTemp(self, device: int, sensor: str | None = None) -> Tuple[str, float]:
        """Get temperature for a GPU.

        Returns:
            (sensor_name, temperature_celsius)
        """
        ...

    def log(self, device: int | None = None) -> str:
        """Return a compact one-line string with power, temp, utilization, and memory for GPU(s).

        Args:
            device: GPU index to log, or None to log all devices.

        Returns:
            String like "nvidia gpu0: 120W, 45°C, 80% util, 8.2/16.0 GiB" or
            multiple devices joined by "; ".
        """
        devices = [device] if device is not None else self.listDevices()
        if not devices:
            return f"{self.name}: no devices"
        parts: List[str] = []
        for d in devices:
            try:
                power_info = self.getPower(d)
                power = power_info.get("power", 0)
                try:
                    power = float(power)
                except (TypeError, ValueError):
                    power = 0.0
                unit = power_info.get("unit", "W")
                used_b, total_b = self.getMemInfo(d)
                used_gib = _bytes_to_gib(used_b)
                total_gib = _bytes_to_gib(total_b)
                util = self.getUtilization(d)
                _, temp = self.getTemp(d)
                parts.append(
                    f"{self.name} gpu{d}: {power:.1f}{unit}, {temp:.0f}°C, "
                    f"{util:.0f}% util, {used_gib:.1f}/{total_gib:.1f} GiB"
                )
            except Exception:
                parts.append(f"{self.name} gpu{d}: (unavailable)")
        return "; ".join(parts)


# -----------------------------------------------------------------------------
# ROCm implementation
# -----------------------------------------------------------------------------


class RocmProfiler(GPUProfiler):
    """Profiler for AMD GPUs using ROCm SMI."""

    @property
    def name(self) -> str:
        return "amd"

    def __init__(self) -> None:
        rocm_smi = _import_rocm_smi()
        rocm_smi.initializeRsmi()

    def getPower(self, device: int) -> Dict[str, Any]:
        """Gets the power usage of a given GPU.

        Args:
            device: The device index to get the power usage for.

        Returns:
            A dictionary containing the power usage with keys such as
            "power", "power_type", "unit", "ret".
        """
        import rocm_smi as rs  # noqa: F811

        return rs.getPower(device)

    def listDevices(self) -> List[int]:
        import rocm_smi as rs  # noqa: F811

        return rs.listDevices()

    def getMemInfo(self, device: int) -> Tuple[int, int]:
        import rocm_smi as rs  # noqa: F811

        mem_used, mem_total = rs.getMemInfo(device, "vram")
        return mem_used, mem_total

    def getUtilization(self, device: int) -> float:
        import rocm_smi as rs  # noqa: F811

        return rs.getGpuUse(device)

    def getTemp(self, device: int, sensor: str | None = None) -> Tuple[str, float]:
        """Gets the temperature of a given GPU.

        Args:
            device: The device index.
            sensor: Optional sensor name. If None, uses first available.

        Returns:
            (sensor_name, temperature_celsius)
        """
        import rocm_smi as rs  # noqa: F811

        if sensor is None:
            return rs.findFirstAvailableTemp(device)
        return rs.getTemp(device, sensor)


def _import_rocm_smi():
    """Import rocm_smi, adding ROCM_SMI_PATH to sys.path if set."""
    rocm_smi_path = os.environ.get("ROCM_SMI_PATH", None)
    if rocm_smi_path is not None:
        if rocm_smi_path not in sys.path:
            sys.path.append(rocm_smi_path)
    else:
        raise ValueError("ROCM_SMI_PATH not set; required for ROCm profiling")
    import rocm_smi  # noqa: PLC0415

    return rocm_smi


# -----------------------------------------------------------------------------
# NVIDIA implementation
# -----------------------------------------------------------------------------


class NvidiaProfiler(GPUProfiler):
    """Profiler for NVIDIA GPUs using NVML (nvidia-ml-py / pynvml)."""

    @property
    def name(self) -> str:
        return "nvidia"

    def __init__(self) -> None:
        pynvml = _import_pynvml()
        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handles: List[Any] = []

    def _handle(self, device: int):
        """Get or cache NVML handle for device index."""
        pynvml = self._pynvml
        while len(self._handles) <= device:
            self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(len(self._handles)))
        return self._handles[device]

    def getPower(self, device: int) -> Dict[str, Any]:
        """Get power usage in milliwatts. Normalized to dict with 'power' and 'unit'."""
        pynvml = self._pynvml
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(self._handle(device))
            return {
                "power": mw / 1000.0 if mw is not None else 0.0,
                "power_mw": mw,
                "unit": "W",
                "power_type": "draw",
            }
        except pynvml.NVMLError:
            return {"power": 0.0, "power_mw": 0, "unit": "W", "power_type": "draw"}

    def listDevices(self) -> List[int]:
        pynvml = self._pynvml
        count = pynvml.nvmlDeviceGetCount()
        return list(range(count))

    def getMemInfo(self, device: int) -> Tuple[int, int]:
        pynvml = self._pynvml
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle(device))
        return info.used, info.total

    def getUtilization(self, device: int) -> float:
        pynvml = self._pynvml
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(self._handle(device))
            return float(rates.gpu)
        except pynvml.NVMLError:
            return 0.0

    def getTemp(self, device: int, sensor: str | None = None) -> Tuple[str, float]:
        """Get GPU temperature. sensor ignored; NVML reports GPU core temp."""
        pynvml = self._pynvml
        # NVML_TEMPERATURE_GPU = 0
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle(device), pynvml.NVML_TEMPERATURE_GPU
            )
            return ("GPU", float(temp))
        except pynvml.NVMLError:
            return ("GPU", 0.0)


def _import_pynvml():
    """Import pynvml (nvidia-ml-py)."""
    try:
        import pynvml  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "nvidia-ml-py is required for NVIDIA profiling. "
            "Install with: pip install nvidia-ml-py"
        ) from e
    return pynvml


# -----------------------------------------------------------------------------
# General interface: pick profiler for current system
# -----------------------------------------------------------------------------


def get_profiler() -> GPUProfiler:
    """Return a GPUProfiler for the current system (NVIDIA or AMD ROCm).

    Tries NVIDIA first, then ROCm. Use this as the single entry point when
    you want profiling to work on any supported system.

    Returns:
        An instance of NvidiaProfiler or RocmProfiler.

    Raises:
        RuntimeError: If neither NVIDIA nor ROCm is available.
    """
    try:
        return NvidiaProfiler()
    except (ImportError, Exception):
        pass
    try:
        return RocmProfiler()
    except (ValueError, ImportError, Exception):
        pass
    raise RuntimeError(
        "No GPU profiler available. "
        "For NVIDIA: install nvidia-ml-py and ensure NVML is available. "
        "For AMD: set ROCM_SMI_PATH and ensure rocm_smi is available."
    )
