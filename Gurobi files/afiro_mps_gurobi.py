"""
Industrial MPS Benchmark Instance: afiro.mps using Gurobi.
Reads and constructs the Gurobi optimization model for afiro.mps.
"""
import os
import time
import numpy as np

# Compatibility workaround for scipy/numpy version mismatch in environment
if not hasattr(np, "long"):
    setattr(np, "long", np.int64)
if not hasattr(np, "ulong"):
    setattr(np, "ulong", np.uint64)

try:
    np.array([1], copy=None)
except ValueError:
    _orig_array = np.array
    def _compat_array(object, dtype=None, copy=None, order='K', subok=False, ndmin=0, **kwargs):
        if copy is None:
            copy = False
        return _orig_array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, **kwargs)
    np.array = _compat_array

import gurobipy as gp
from gurobipy import GRB


def build_afiro_mps_model(mps_path: str = None) -> gp.Model:
    """
    Constructs the Gurobi model for afiro.mps.
    """
    if mps_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mps_path = os.path.join(base_dir, "afiro.mps")

    model = gp.read(mps_path)
    model.ModelName = "afiro_mps"
    return model


if __name__ == "__main__":
    print("Building afiro.mps model (Gurobi)...")
    t0 = time.perf_counter()
    m = build_afiro_mps_model()
    m.optimize()
    elapsed = time.perf_counter() - t0

    if m.Status == GRB.OPTIMAL:
        print("\nOptimization Status: OPTIMAL")
        print(f"Optimal Objective Value: {m.ObjVal:.6f}")
        print(f"Gurobi Solver Runtime:   {m.Runtime:.6f} seconds")
        print(f"Wall-Clock Execution:   {elapsed:.6f} seconds")
    else:
        print(f"\nOptimization ended with status: {m.Status}")
        print(f"Gurobi Solver Runtime:   {m.Runtime:.6f} seconds")
        print(f"Wall-Clock Execution:   {elapsed:.6f} seconds")
