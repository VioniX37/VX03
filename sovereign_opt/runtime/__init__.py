"""
Heterogeneous CPU-GPU execution runtime.
"""
from sovereign_opt.runtime.device import DeviceDetector, DeviceInfo
from sovereign_opt.runtime.backend import ExecutionBackend, CPUBackend, GPUBackend

__all__ = ["DeviceDetector", "DeviceInfo", "ExecutionBackend", "CPUBackend", "GPUBackend"]
