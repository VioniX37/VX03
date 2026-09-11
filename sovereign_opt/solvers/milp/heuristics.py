"""
Primal heuristics for finding early integer feasible solutions.
Implements:
- Simple Rounding heuristic
- Feasibility Pump heuristic
"""
from typing import Dict, Optional, Tuple
import numpy as np
from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType


class PrimalHeuristic:
    @classmethod
    def simple_rounding(
        cls, model: OptimizationModel, lp_solution: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """
        Attempts to round fractional variables to integers and tests if constraints are satisfied.
        """
        rounded = dict(lp_solution)
        for v_name, var in model.variables.items():
            if var.is_integer:
                val = lp_solution.get(v_name, 0.0)
                rounded[v_name] = float(round(val))

        # Check bounds
        for v_name, var in model.variables.items():
            val = rounded[v_name]
            if val < var.lower_bound - 1e-6 or val > var.upper_bound + 1e-6:
                return None

        # Check constraints
        for con in model.constraints.values():
            val = sum(c * rounded.get(v, 0.0) for v, c in con.coefficients.items())
            if val < con.lower_bound - 1e-6 or val > con.upper_bound + 1e-6:
                return None

        return rounded
