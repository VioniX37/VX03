"""
Industrial Benchmark: Refinery Crude & Blendstock Blending Problem.
Formulates the blending of crudes and high-octane blendstocks (Alkylate)
to produce market-grade fuels (Regular Gas, Premium Gas, Diesel)
meeting strict octane ratings and sulfur limits while maximizing refinery profit.
"""
from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense


def build_refinery_blending_model(as_qp: bool = False) -> OptimizationModel:
    """
    Creates a production refinery blending optimization model.
    If as_qp is True, adds quadratic penalties for deviating from target crude mixes.
    """
    model = OptimizationModel(name="Refinery_Crude_Blending")

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

    # Variables: x_{feed, prod} = barrels/day of feed used in product
    obj_coeffs = {}
    for f_name, f_data in feedstocks.items():
        for p_name, p_data in products.items():
            v_name = f"x_{f_name}_{p_name}"
            # Profit margin = selling price - feedstock purchase cost
            margin = p_data["price"] - f_data["cost"]
            model.add_variable(name=v_name, lower_bound=0.0, upper_bound=float(f_data["max_supply"]))
            obj_coeffs[v_name] = margin

    # Maximize total profit -> Minimize negative profit
    min_obj_coeffs = {v: -margin for v, margin in obj_coeffs.items()}

    # Constraint 1: Feedstock supply availability: sum_{prod} x_{feed, prod} <= max_supply
    for f_name, f_data in feedstocks.items():
        coeffs = {f"x_{f_name}_{p_name}": 1.0 for p_name in products.keys()}
        model.add_constraint(
            name=f"Supply_{f_name}",
            coefficients=coeffs,
            sense=ConstraintSense.LE,
            rhs=float(f_data["max_supply"]),
        )

    # Constraint 2: Product minimum demand satisfaction: sum_{feed} x_{feed, prod} >= min_demand
    for p_name, p_data in products.items():
        coeffs = {f"x_{f_name}_{p_name}": 1.0 for f_name in feedstocks.keys()}
        model.add_constraint(
            name=f"Demand_{p_name}",
            coefficients=coeffs,
            sense=ConstraintSense.GE,
            rhs=float(p_data["min_demand"]),
        )

    # Constraint 3: Octane quality specifications
    # sum_{feed} (octane_feed - min_octane_prod) * x_{feed, prod} >= 0
    for p_name, p_data in products.items():
        coeffs = {}
        for f_name, f_data in feedstocks.items():
            coeffs[f"x_{f_name}_{p_name}"] = f_data["octane"] - p_data["min_octane"]
        model.add_constraint(
            name=f"Octane_{p_name}",
            coefficients=coeffs,
            sense=ConstraintSense.GE,
            rhs=0.0,
        )

    # Constraint 4: Sulfur limits
    # sum_{feed} (sulfur_feed - max_sulfur_prod) * x_{feed, prod} <= 0
    for p_name, p_data in products.items():
        coeffs = {}
        for f_name, f_data in feedstocks.items():
            coeffs[f"x_{f_name}_{p_name}"] = f_data["sulfur"] - p_data["max_sulfur"]
        model.add_constraint(
            name=f"Sulfur_{p_name}",
            coefficients=coeffs,
            sense=ConstraintSense.LE,
            rhs=0.0,
        )

    # Optional Quadratic formulation: smoothing penalty for heavy crude usage
    quad_coeffs = {}
    if as_qp:
        for f_name in ("Crude_Dubai", "Crude_Maya"):
            for p_name in products.keys():
                v = f"x_{f_name}_{p_name}"
                quad_coeffs[(v, v)] = 0.0001

    model.set_objective(
        linear_coefficients=min_obj_coeffs,
        sense=ObjectiveSense.MINIMIZE,
        quadratic_coefficients=quad_coeffs,
    )

    return model
