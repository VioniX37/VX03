"""
Industrial Benchmark Replica: Power Generation Unit Commitment (MILP) using Gurobi.
Schedules thermal generators with on/off binary commitment states, minimum generation,
and maximum capacity to meet hourly electricity demand at minimum total cost.
"""
import math
import gurobipy as gp
from gurobipy import GRB


def build_unit_commitment_model(time_periods: int = 4) -> gp.Model:
    model = gp.Model("Power_Grid_Unit_Commitment")

    # Generators: min_mw, max_mw, cost_per_mwh, startup_cost
    generators = {
        "Gen_Coal_Base": {"min_mw": 100.0, "max_mw": 500.0, "cost": 25.0, "startup": 1200.0},
        "Gen_Gas_CCGT": {"min_mw": 50.0, "max_mw": 350.0, "cost": 42.0, "startup": 600.0},
        "Gen_Gas_Peaker": {"min_mw": 20.0, "max_mw": 150.0, "cost": 75.0, "startup": 200.0},
        "Gen_Hydro_Flex": {"min_mw": 10.0, "max_mw": 200.0, "cost": 15.0, "startup": 100.0},
    }

    # Demand across time periods (MW)
    base_profile = [450.0, 680.0, 850.0, 520.0]
    demand = [
        base_profile[t] if t < len(base_profile)
        else round(450.0 + 350.0 * (0.5 - 0.5 * math.cos(2.0 * math.pi * t / 24.0)), 1)
        for t in range(time_periods)
    ]

    obj_expr = gp.LinExpr()

    for t in range(time_periods):
        demand_vars = []

        for g_name, g_data in generators.items():
            u_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"u_{g_name}_t{t}")
            p_var = model.addVar(lb=0.0, ub=g_data["max_mw"], vtype=GRB.CONTINUOUS, name=f"p_{g_name}_t{t}")

            obj_expr += g_data["cost"] * p_var + (g_data["startup"] * 0.2) * u_var
            demand_vars.append(p_var)

            # Max capacity constraint: p - max_mw * u <= 0
            model.addConstr(p_var - g_data["max_mw"] * u_var <= 0.0, name=f"MaxCap_{g_name}_t{t}")

            # Min capacity constraint: p - min_mw * u >= 0
            model.addConstr(p_var - g_data["min_mw"] * u_var >= 0.0, name=f"MinCap_{g_name}_t{t}")

        # Demand balance constraint: sum(p) >= demand[t]
        model.addConstr(gp.quicksum(demand_vars) >= demand[t], name=f"Demand_Balance_t{t}")

    model.setObjective(obj_expr, GRB.MINIMIZE)
    return model


def build_unit_commitment_fleet(num_generators: int = 20, time_periods: int = 48, seed: int = 5) -> gp.Model:
    """
    Scalable unit commitment (MILP) for a regional fleet using Gurobi.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    G, T = num_generators, time_periods
    kind = rng.choice(["coal", "ccgt", "peaker", "hydro"], size=G, p=[0.35, 0.3, 0.25, 0.1])
    spec = {"coal": (150, 500, 22, 3000, 0.35), "ccgt": (60, 350, 38, 900, 0.6),
            "peaker": (15, 120, 80, 250, 1.0), "hydro": (10, 200, 12, 150, 0.9)}
    pmin = np.array([spec[k][0] for k in kind]) * rng.uniform(0.8, 1.2, G)
    pmax = np.array([spec[k][1] for k in kind]) * rng.uniform(0.8, 1.2, G)
    cost = np.array([spec[k][2] for k in kind]) * rng.uniform(0.9, 1.1, G)
    start = np.array([spec[k][3] for k in kind]) * rng.uniform(0.8, 1.2, G)
    ramp = np.array([spec[k][4] for k in kind]) * pmax
    base = 0.62 * pmax.sum()
    demand = [round(base * (0.55 + 0.45 * (0.5 - 0.5 * math.cos(2 * math.pi * ((t % 24) - 5) / 24))), 1) for t in range(T)]

    model = gp.Model(f"Unit_Commitment_Fleet_G{G}_T{T}")
    u_vars = {}
    v_vars = {}
    p_vars = {}
    obj_expr = gp.LinExpr()

    for g in range(G):
        for t in range(T):
            u_name, v_name, p_name = f"u_{g}_{t}", f"v_{g}_{t}", f"p_{g}_{t}"
            u = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=u_name)
            v = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=v_name)
            p = model.addVar(lb=0.0, ub=float(pmax[g]), vtype=GRB.CONTINUOUS, name=p_name)
            u_vars[(g, t)] = u
            v_vars[(g, t)] = v
            p_vars[(g, t)] = p

            obj_expr += float(cost[g]) * p + float(0.05 * cost[g] * pmin[g]) * u + float(start[g]) * v

            model.addConstr(p - float(pmax[g]) * u <= 0.0, name=f"max_{g}_{t}")
            model.addConstr(p - float(pmin[g]) * u >= 0.0, name=f"min_{g}_{t}")

            if t == 0:
                model.addConstr(v - u >= 0.0, name=f"start_{g}_{t}")
            else:
                model.addConstr(v - u + u_vars[(g, t - 1)] >= 0.0, name=f"start_{g}_{t}")
                model.addConstr(p - p_vars[(g, t - 1)] - float(pmax[g]) * v <= float(ramp[g]), name=f"rampup_{g}_{t}")
                model.addConstr(p_vars[(g, t - 1)] - p <= float(max(ramp[g], pmax[g])), name=f"rampdn_{g}_{t}")

    for t in range(T):
        model.addConstr(gp.quicksum(p_vars[(g, t)] for g in range(G)) >= demand[t], name=f"demand_{t}")
        model.addConstr(gp.quicksum(float(pmax[g]) * u_vars[(g, t)] for g in range(G)) >= 1.1 * demand[t], name=f"reserve_{t}")

    model.setObjective(obj_expr, GRB.MINIMIZE)
    return model


import time


if __name__ == "__main__":
    print("Building Unit Commitment 4h model (Gurobi)...")
    t0 = time.perf_counter()
    m4 = build_unit_commitment_model(4)
    m4.optimize()
    elapsed = time.perf_counter() - t0
    if m4.Status == GRB.OPTIMAL:
        print(f"4h Unit Commitment Optimal Objective: {m4.ObjVal:.6f}")
        print(f"4h Unit Commitment Gurobi Runtime:   {m4.Runtime:.6f} seconds")
        print(f"4h Unit Commitment Wall-Clock Time:   {elapsed:.6f} seconds")

    print("\nBuilding Unit Commitment 24h Fleet model (Gurobi)...")
    t0 = time.perf_counter()
    mfleet = build_unit_commitment_fleet(10, 24)
    mfleet.optimize()
    elapsed = time.perf_counter() - t0
    if mfleet.Status == GRB.OPTIMAL:
        print(f"Fleet (10, 24) Optimal Objective: {mfleet.ObjVal:.6f}")
        print(f"Fleet (10, 24) Gurobi Runtime:   {mfleet.Runtime:.6f} seconds")
        print(f"Fleet (10, 24) Wall-Clock Time:   {elapsed:.6f} seconds")

