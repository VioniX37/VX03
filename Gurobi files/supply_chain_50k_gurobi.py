"""
Scalable Industrial Benchmark: Supply Chain Production-Distribution Network (50k variables) using Gurobi.
Constructs the Gurobi optimization model corresponding to supply_chain_50k_variables.csv.
"""
import os
import time
import pandas as pd
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

from benchmarks.industrial.supply_chain_gurobi import (
    build_supply_chain_arrays,
    arrays_to_model,
    sizes_for,
)


def build_supply_chain_50k_model(csv_path: str = None) -> gp.Model:
    """
    Constructs the 50,000 variable Supply Chain LP Gurobi model.
    Captures all variables from supply_chain_50k_variables.csv and attaches complete constraints.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "supply_chain_50k_variables.csv")

    # Read variable definitions from CSV
    df = pd.read_csv(csv_path)

    # Determine dimensions matching 50k variables (9 plants, 49 warehouses, 2403 customers, 4 products)
    P, W, C = sizes_for(50_000, products=4, neighbours=5)
    lp = build_supply_chain_arrays(P, W, C, products=4, neighbours=5, seed=7)

    # Create Gurobi Model
    model = arrays_to_model(lp)
    model.ModelName = "Supply_Chain_50k_LP"

    # Override objective cost coefficients if available from CSV
    if len(df) == len(lp.c):
        costs = df["cost_per_unit"].values
        for j, var in enumerate(model.getVars()):
            var.Obj = float(costs[j])
            ub_val = df["upper_bound"].iloc[j]
            if ub_val != float("inf") and not np.isinf(ub_val):
                var.Ub = float(ub_val)
            var.Lb = float(df["lower_bound"].iloc[j])

    model.update()
    return model


if __name__ == "__main__":
    print("Building Supply Chain 50k LP model (Gurobi)...")
    t0 = time.perf_counter()
    m = build_supply_chain_50k_model()
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
