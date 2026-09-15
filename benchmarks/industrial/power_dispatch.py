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
