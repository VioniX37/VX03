"""
Heterogeneous execution backend abstraction (CPU ↔ GPU) with automatic fallback.
"""
from typing import Optional
import numpy as np
import scipy.sparse as sp
from sovereign_opt.runtime.device import DeviceDetector, DeviceInfo


class ExecutionBackend:
    """Abstract execution backend for numerical linear algebra primitives."""

    def spmv(self, A_csr: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def dot(self, x: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError()


class CPUBackend(ExecutionBackend):
    """Vectorized CPU execution using NumPy BLAS."""

    def spmv(self, A_csr: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        return A_csr.dot(x)

    def dot(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.dot(x, y))


class GPUBackend(ExecutionBackend):
    """
    GPU-accelerated backend using PyTorch CUDA tensors.
    Includes automated fallback to CPUBackend if GPU is unavailable or throws errors.
    """

    def __init__(self):
        self.cpu_fallback = CPUBackend()
        self.device_info = DeviceDetector.get_info()

    def spmv(self, A_csr: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
        if not self.device_info.has_cuda:
            return self.cpu_fallback.spmv(A_csr, x)

        try:
            import torch
            # Convert CSR to PyTorch sparse tensor
            crow = torch.from_numpy(A_csr.indptr).to("cuda", dtype=torch.int64)
            col = torch.from_numpy(A_csr.indices).to("cuda", dtype=torch.int64)
            values = torch.from_numpy(A_csr.data).to("cuda", dtype=torch.float64)
            tensor_A = torch.sparse_csr_tensor(crow, col, values, size=A_csr.shape, device="cuda")

            tensor_x = torch.from_numpy(x).to("cuda", dtype=torch.float64)
            res = torch.mv(tensor_A, tensor_x)
            return res.cpu().numpy()
        except Exception:
            # Graceful automated CPU fallback
            return self.cpu_fallback.spmv(A_csr, x)

    def dot(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.device_info.has_cuda:
            return self.cpu_fallback.dot(x, y)

        try:
            import torch
            tx = torch.from_numpy(x).to("cuda", dtype=torch.float64)
            ty = torch.from_numpy(y).to("cuda", dtype=torch.float64)
            return float(torch.dot(tx, ty).item())
        except Exception:
            return self.cpu_fallback.dot(x, y)
