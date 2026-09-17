"""
Industrial Benchmark Replica: Cardinality-Constrained Mean-Variance Portfolio Selection (MIQP) using Gurobi.

    min   lambda * w^T Sigma w  -  mu^T w
    s.t.  sum_i w_i = 1
          min_weight * z_i <= w_i <= z_i        (w_i can only be held if asset i is selected)
          sum_i z_i <= max_assets               (cardinality limit)
          z_i binary

Deterministic data: a fixed 3-factor covariance model plus idiosyncratic variance.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

ASSETS = [
    "Tech_Growth", "Healthcare", "Energy", "Utilities", "Financials",
    "Consumer_Staples", "Industrials", "Real_Estate", "Gov_Bonds", "Gold",
]


def build_portfolio_selection_model(num_assets: int = 8, max_assets: int = 4, min_weight: float = 0.05,
                                    risk_aversion: float = 3.0) -> gp.Model:
    num_assets = max(2, min(num_assets, len(ASSETS)))
    names = ASSETS[:num_assets]
    rng = np.random.default_rng(2026)
    loadings = rng.normal(0.0, 0.12, size=(len(ASSETS), 3))[:num_assets]
    idio = np.linspace(0.02, 0.06, len(ASSETS))[:num_assets] ** 2
    sigma = loadings @ loadings.T + np.diag(idio)
    mu = np.array([0.14, 0.10, 0.09, 0.05, 0.08, 0.06, 0.085, 0.07, 0.03, 0.04])[:num_assets]

    model = gp.Model("Portfolio_Cardinality_MIQP")

    w = {}
    z = {}
    for a in names:
        w[a] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"w_{a}")
        z[a] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"z_{a}")
        model.addConstr(w[a] - z[a] <= 0.0, name=f"Hold_Only_If_Selected_{a}")
        model.addConstr(w[a] - min_weight * z[a] >= 0.0, name=f"Min_Position_{a}")

    model.addConstr(gp.quicksum(w[a] for a in names) == 1.0, name="Budget")
    model.addConstr(gp.quicksum(z[a] for a in names) <= float(max_assets), name="Cardinality")

    quad_expr = risk_aversion * gp.quicksum(
        sigma[i, j] * w[names[i]] * w[names[j]]
        for i in range(num_assets) for j in range(num_assets)
    )
    lin_expr = gp.quicksum(mu[i] * w[names[i]] for i in range(num_assets))

    model.setObjective(quad_expr - lin_expr, GRB.MINIMIZE)
    return model


import time


if __name__ == "__main__":
    print("Building Portfolio Selection model (Gurobi)...")
    t0 = time.perf_counter()
    m = build_portfolio_selection_model()
    m.optimize()
    elapsed = time.perf_counter() - t0
    if m.Status == GRB.OPTIMAL:
        print(f"\nOptimization Status: OPTIMAL")
        print(f"Optimal Objective Value: {m.ObjVal:.6f}")
        print(f"Gurobi Solver Runtime:   {m.Runtime:.6f} seconds")
        print(f"Wall-Clock Execution:   {elapsed:.6f} seconds")
        print("\nSelected Asset Weights:")
        for var in m.getVars():
            if var.VarName.startswith("w_") and abs(var.X) > 1e-6:
                print(f"  {var.VarName}: {var.X:.4f}")
    else:
        print(f"Optimization ended with status: {m.Status}")

