"""
Benchmark problems for Sovereign Optimizer.
Includes standard Netlib LP instances and realistic industrial formulations.
"""
from benchmarks.netlib.afiro import build_netlib_afiro
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.power_dispatch import build_unit_commitment_model

__all__ = ["build_netlib_afiro", "build_refinery_blending_model", "build_unit_commitment_model"]
