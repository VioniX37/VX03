"""
Device detection and runtime execution capabilities.
"""
from dataclasses import dataclass
import os


@dataclass
class DeviceInfo:
    cpu_cores: int
    has_cuda: bool
    cuda_device_name: str
    preferred_device: str


class DeviceDetector:
    @classmethod
    def get_info(cls) -> DeviceInfo:
        cpu_cores = os.cpu_count() or 4
        has_cuda = False
        cuda_name = "None"

        try:
            import torch
            if torch.cuda.is_available():
                has_cuda = True
                cuda_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        pref = "cuda" if has_cuda else "cpu"
        return DeviceInfo(
            cpu_cores=cpu_cores,
            has_cuda=has_cuda,
            cuda_device_name=cuda_name,
            preferred_device=pref,
        )
