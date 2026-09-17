"""
Industrial Benchmark: Power Generation Unit Commitment (MILP).
Schedules thermal generators with on/off binary commitment states, minimum generation,
and maximum capacity to meet hourly electricity demand at minimum total cost.
"""
import math

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense


def build_unit_commitment_model(time_periods: int = 4) -> OptimizationModel:
    model = OptimizationModel(name="Power_Grid_Unit_Commitment")

    # Generators: min_mw, max_mw, cost_per_mwh, startup_cost
    generators = {
        "Gen_Coal_Base": {"min_mw": 100.0, "max_mw": 500.0, "cost": 25.0, "startup": 1200.0},
        "Gen_Gas_CCGT": {"min_mw": 50.0, "max_mw": 350.0, "cost": 42.0, "startup": 600.0},
        "Gen_Gas_Peaker": {"min_mw": 20.0, "max_mw": 150.0, "cost": 75.0, "startup": 200.0},
        "Gen_Hydro_Flex": {"min_mw": 10.0, "max_mw": 200.0, "cost": 15.0, "startup": 100.0},
    }

    # Demand across time periods (MW): the original 4-hour profile, extended by a
    # deterministic daily load curve (450-800 MW) for longer horizons.
    base_profile = [450.0, 680.0, 850.0, 520.0]
    demand = [
        base_profile[t] if t < len(base_profile)
        else round(450.0 + 350.0 * (0.5 - 0.5 * math.cos(2.0 * math.pi * t / 24.0)), 1)
        for t in range(time_periods)
    ]

    obj_coeffs = {}

    for t in range(time_periods):
        # Demand balance constraint: sum(p_{g, t}) >= demand[t]
        demand_coeffs = {}

        for g_name, g_data in generators.items():
            u_var = f"u_{g_name}_t{t}"  # Binary: 1 if generator is on
            p_var = f"p_{g_name}_t{t}"  # Continuous: MW power generated

            model.add_variable(name=u_var, lower_bound=0.0, upper_bound=1.0, var_type=VariableType.BINARY)
            model.add_variable(name=p_var, lower_bound=0.0, upper_bound=g_data["max_mw"], var_type=VariableType.CONTINUOUS)

            obj_coeffs[p_var] = g_data["cost"]
            obj_coeffs[u_var] = g_data["startup"] * 0.2  # amortized commitment cost

            demand_coeffs[p_var] = 1.0

            # Constraint: p_{g, t} <= max_mw * u_{g, t}  --> p_{g, t} - max_mw * u_{g, t} <= 0
            model.add_constraint(
                name=f"MaxCap_{g_name}_t{t}",
                coefficients={p_var: 1.0, u_var: -g_data["max_mw"]},
                sense=ConstraintSense.LE,
                rhs=0.0,
            )

            # Constraint: p_{g, t} >= min_mw * u_{g, t}  --> p_{g, t} - min_mw * u_{g, t} >= 0
            model.add_constraint(
                name=f"MinCap_{g_name}_t{t}",
                coefficients={p_var: 1.0, u_var: -g_data["min_mw"]},
                sense=ConstraintSense.GE,
                rhs=0.0,
            )

        model.add_constraint(
            name=f"Demand_Balance_t{t}",
            coefficients=demand_coeffs,
            sense=ConstraintSense.GE,
            rhs=demand[t],
        )

    model.set_objective(linear_coefficients=obj_coeffs, sense=ObjectiveSense.MINIMIZE)
    return model


def build_unit_commitment_fleet(num_generators: int = 20, time_periods: int = 48, seed: int = 5) -> OptimizationModel:
    """
    Scalable unit commitment (MILP) for a regional fleet: on/off (u), start-up (v) and output (p) per
    generator and hour, with min / max output, ramp limits, start-up costs and a spinning-reserve
    requirement. Synthetic fleet with a fixed seed; 3 * G * T variables (2 * G * T binaries).
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

    model = OptimizationModel(name=f"Unit_Commitment_Fleet_G{G}_T{T}")
    obj = {}
    for g in range(G):
        for t in range(T):
            u, v, p = f"u_{g}_{t}", f"v_{g}_{t}", f"p_{g}_{t}"
            model.add_variable(u, 0.0, 1.0, VariableType.BINARY)
            model.add_variable(v, 0.0, 1.0, VariableType.BINARY)
            model.add_variable(p, 0.0, float(pmax[g]))
            obj[p] = float(cost[g])
            obj[u] = float(0.05 * cost[g] * pmin[g])  # no-load cost
            obj[v] = float(start[g])
            model.add_constraint(f"max_{g}_{t}", {p: 1.0, u: -float(pmax[g])}, ConstraintSense.LE, rhs=0.0)
            model.add_constraint(f"min_{g}_{t}", {p: 1.0, u: -float(pmin[g])}, ConstraintSense.GE, rhs=0.0)
            if t == 0:
                model.add_constraint(f"start_{g}_{t}", {v: 1.0, u: -1.0}, ConstraintSense.GE, rhs=0.0)
            else:
                model.add_constraint(f"start_{g}_{t}", {v: 1.0, u: -1.0, f"u_{g}_{t-1}": 1.0}, ConstraintSense.GE, rhs=0.0)
                model.add_constraint(f"rampup_{g}_{t}", {p: 1.0, f"p_{g}_{t-1}": -1.0, v: -float(pmax[g])},
                                     ConstraintSense.LE, rhs=float(ramp[g]))
                model.add_constraint(f"rampdn_{g}_{t}", {f"p_{g}_{t-1}": 1.0, p: -1.0}, ConstraintSense.LE,
                                     rhs=float(max(ramp[g], pmax[g])))
    for t in range(T):
        model.add_constraint(f"demand_{t}", {f"p_{g}_{t}": 1.0 for g in range(G)}, ConstraintSense.GE, rhs=demand[t])
        model.add_constraint(f"reserve_{t}", {f"u_{g}_{t}": float(pmax[g]) for g in range(G)}, ConstraintSense.GE,
                             rhs=1.1 * demand[t])
    model.set_objective(obj, ObjectiveSense.MINIMIZE)
    return model
