"""
Industrial Benchmark: Cardinality-Constrained Mean-Variance Portfolio Selection (MIQP).

    min   lambda * w^T Sigma w  -  mu^T w
    s.t.  sum_i w_i = 1
          min_weight * z_i <= w_i <= z_i        (w_i can only be held if asset i is selected)
          sum_i z_i <= max_assets               (cardinality limit)
          z_i binary

Deterministic data: a fixed 3-factor covariance model plus idiosyncratic variance.
"""
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense

ASSETS = [
    "Tech_Growth", "Healthcare", "Energy", "Utilities", "Financials",
    "Consumer_Staples", "Industrials", "Real_Estate", "Gov_Bonds", "Gold",
]


def build_portfolio_selection_model(num_assets: int = 8, max_assets: int = 4, min_weight: float = 0.05,
                                    risk_aversion: float = 3.0) -> OptimizationModel:
    num_assets = max(2, min(num_assets, len(ASSETS)))
    names = ASSETS[:num_assets]
    rng = np.random.default_rng(2026)
    loadings = rng.normal(0.0, 0.12, size=(len(ASSETS), 3))[:num_assets]
    idio = np.linspace(0.02, 0.06, len(ASSETS))[:num_assets] ** 2
    sigma = loadings @ loadings.T + np.diag(idio)
    mu = np.array([0.14, 0.10, 0.09, 0.05, 0.08, 0.06, 0.085, 0.07, 0.03, 0.04])[:num_assets]

    model = OptimizationModel(name="Portfolio_Cardinality_MIQP")
    for a in names:
        model.add_variable(f"w_{a}", 0.0, 1.0)
        model.add_variable(f"z_{a}", 0.0, 1.0, VariableType.BINARY)
        model.add_constraint(f"Hold_Only_If_Selected_{a}", {f"w_{a}": 1.0, f"z_{a}": -1.0}, ConstraintSense.LE, rhs=0.0)
        model.add_constraint(f"Min_Position_{a}", {f"w_{a}": 1.0, f"z_{a}": -min_weight}, ConstraintSense.GE, rhs=0.0)
    model.add_constraint("Budget", {f"w_{a}": 1.0 for a in names}, ConstraintSense.EQ, rhs=1.0)
    model.add_constraint("Cardinality", {f"z_{a}": 1.0 for a in names}, ConstraintSense.LE, rhs=float(max_assets))

    quad = {}
    for i, a in enumerate(names):
        for j in range(i, num_assets):
            b = names[j]
            # model convention: coefficient q on (x_i, x_j) contributes 0.5 * q * x_i * x_j
            q = 2.0 * risk_aversion * sigma[i, i] if i == j else 4.0 * risk_aversion * sigma[i, j]
            if abs(q) > 1e-14:
                quad[(f"w_{a}", f"w_{b}")] = float(q)
    model.set_objective({f"w_{a}": -float(mu[i]) for i, a in enumerate(names)},
                        sense=ObjectiveSense.MINIMIZE, quadratic_coefficients=quad)
    return model
