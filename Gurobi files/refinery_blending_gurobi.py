"""
Industrial Benchmark Replica: Refinery Crude & Blendstock Blending Problem using Gurobi.
Formulates the blending of crudes and high-octane blendstocks (Alkylate)
to produce market-grade fuels (Regular Gas, Premium Gas, Diesel)
meeting strict octane ratings and sulfur limits while maximizing refinery profit.
"""
import gurobipy as gp
from gurobipy import GRB


def build_refinery_blending_model(as_qp: bool = False) -> gp.Model:
    """
    Creates a production refinery blending optimization model using Gurobi.
    If as_qp is True, adds quadratic penalties for deviating from target crude mixes.
    """
    model = gp.Model("Refinery_Crude_Blending")

    # Feedstocks available (barrels/day available, cost $/bbl, octane rating, sulfur fraction)
    feedstocks = {
        "Crude_Brent": {"max_supply": 40000, "cost": 72.0, "octane": 91.0, "sulfur": 0.003},
        "Crude_WTI": {"max_supply": 50000, "cost": 68.0, "octane": 88.0, "sulfur": 0.002},
        "Crude_Dubai": {"max_supply": 60000, "cost": 62.0, "octane": 82.0, "sulfur": 0.015},
        "Crude_Maya": {"max_supply": 30000, "cost": 55.0, "octane": 78.0, "sulfur": 0.030},
        "Alkylate_HighOctane": {"max_supply": 25000, "cost": 85.0, "octane": 99.0, "sulfur": 0.0005},
    }

    # Finished products (selling price $/bbl, min demand bbl/day, min octane, max sulfur)
    products = {
        "Regular_Gas": {"price": 95.0, "min_demand": 40000, "min_octane": 87.0, "max_sulfur": 0.010},
        "Premium_Gas": {"price": 112.0, "min_demand": 25000, "min_octane": 93.0, "max_sulfur": 0.005},
        "Diesel": {"price": 88.0, "min_demand": 35000, "min_octane": 80.0, "max_sulfur": 0.015},
    }

    x_vars = {}
    lin_obj = gp.LinExpr()

    for f_name, f_data in feedstocks.items():
        for p_name, p_data in products.items():
            v_name = f"x_{f_name}_{p_name}"
            # Profit margin = selling price - feedstock purchase cost
            margin = p_data["price"] - f_data["cost"]
            var = model.addVar(lb=0.0, ub=float(f_data["max_supply"]), vtype=GRB.CONTINUOUS, name=v_name)
            x_vars[(f_name, p_name)] = var
            # Maximize profit -> Minimize negative profit
            lin_obj += -margin * var

    # Constraint 1: Feedstock supply availability: sum_{prod} x_{feed, prod} <= max_supply
    for f_name, f_data in feedstocks.items():
        model.addConstr(
            gp.quicksum(x_vars[(f_name, p_name)] for p_name in products.keys()) <= float(f_data["max_supply"]),
            name=f"Supply_{f_name}",
        )

    # Constraint 2: Product minimum demand satisfaction: sum_{feed} x_{feed, prod} >= min_demand
    for p_name, p_data in products.items():
        model.addConstr(
            gp.quicksum(x_vars[(f_name, p_name)] for f_name in feedstocks.keys()) >= float(p_data["min_demand"]),
            name=f"Demand_{p_name}",
        )

    # Constraint 3: Octane quality specifications
    for p_name, p_data in products.items():
        model.addConstr(
            gp.quicksum(
                (f_data["octane"] - p_data["min_octane"]) * x_vars[(f_name, p_name)]
                for f_name, f_data in feedstocks.items()
            ) >= 0.0,
            name=f"Octane_{p_name}",
        )

    # Constraint 4: Sulfur limits
    for p_name, p_data in products.items():
        model.addConstr(
            gp.quicksum(
                (f_data["sulfur"] - p_data["max_sulfur"]) * x_vars[(f_name, p_name)]
                for f_name, f_data in feedstocks.items()
            ) <= 0.0,
            name=f"Sulfur_{p_name}",
        )

    # Optional Quadratic formulation: smoothing penalty for heavy crude usage
    quad_obj = gp.QuadExpr()
    if as_qp:
        for f_name in ("Crude_Dubai", "Crude_Maya"):
            for p_name in products.keys():
                var = x_vars[(f_name, p_name)]
                quad_obj += 0.0001 * var * var

    model.setObjective(lin_obj + quad_obj, GRB.MINIMIZE)
    return model


import time


if __name__ == "__main__":
    print("Building Refinery Blending LP model (Gurobi)...")
    t0 = time.perf_counter()
    mlp = build_refinery_blending_model(as_qp=False)
    mlp.optimize()
    elapsed = time.perf_counter() - t0
    if mlp.Status == GRB.OPTIMAL:
        print(f"Refinery Blending LP Optimal Objective: {mlp.ObjVal:.6f}")
        print(f"Refinery Blending LP Gurobi Runtime:   {mlp.Runtime:.6f} seconds")
        print(f"Refinery Blending LP Wall-Clock Time:   {elapsed:.6f} seconds")
        print(f"Refinery Max Profit: {-mlp.ObjVal:.2f} $/day")

    print("\nBuilding Refinery Blending QP model (Gurobi)...")
    t0 = time.perf_counter()
    mqp = build_refinery_blending_model(as_qp=True)
    mqp.optimize()
    elapsed = time.perf_counter() - t0
    if mqp.Status == GRB.OPTIMAL:
        print(f"Refinery Blending QP Optimal Objective: {mqp.ObjVal:.6f}")
        print(f"Refinery Blending QP Gurobi Runtime:   {mqp.Runtime:.6f} seconds")
        print(f"Refinery Blending QP Wall-Clock Time:   {elapsed:.6f} seconds")

