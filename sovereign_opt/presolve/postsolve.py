"""
Postsolve mapper for reconstructing original problem solutions from presolved results.

Each presolve reduction pushes a step onto a stack; postsolve unwinds the stack in reverse
order so that every step sees variable values expressed exactly as they were when it was recorded.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List
import numpy as np


class ReductionType(str, Enum):
    FIXED_VARIABLE = "fixed_variable"
    EMPTY_ROW = "empty_row"
    EMPTY_COL = "empty_col"
    SINGLETON_ROW = "singleton_row"
    REDUNDANT_ROW = "redundant_row"
    FORCING_ROW = "forcing_row"
    DOUBLETON_EQUATION = "doubleton_equation"
    DOMINATED_COLUMN = "dominated_column"
    PARALLEL_ROW = "parallel_row"
    DUPLICATE_COLUMN = "duplicate_column"
    BOUND_TIGHTENING = "bound_tightening"
    COEFFICIENT_TIGHTENING = "coefficient_tightening"


FIXING_REDUCTIONS = (ReductionType.FIXED_VARIABLE, ReductionType.EMPTY_COL, ReductionType.DOMINATED_COLUMN)


@dataclass
class PresolveStep:
    reduction_type: ReductionType
    details: Dict[str, Any]


class PostsolveMapper:
    """
    Maintains the stack of presolve reductions to reconstruct
    the original problem's solution vector.
    """

    def __init__(self, original_var_names: List[str], original_con_names: List[str]):
        self.original_var_names = list(original_var_names)
        self.original_con_names = list(original_con_names)
        self.steps: List[PresolveStep] = []
        self.fixed_vars: Dict[str, float] = {}
        self.eliminated_vars: Dict[str, str] = {}

    def record_step(self, step: PresolveStep):
        self.steps.append(step)
        d = step.details
        if step.reduction_type in FIXING_REDUCTIONS:
            value = d.get("value", d.get("fixed_value"))
            self.fixed_vars[d["var_name"]] = value
            self.eliminated_vars[d["var_name"]] = step.reduction_type.value
        elif step.reduction_type in (ReductionType.DOUBLETON_EQUATION, ReductionType.DUPLICATE_COLUMN):
            self.eliminated_vars[d["var_name"]] = step.reduction_type.value

    def restore_dict(self, presolved_solution: Dict[str, float]) -> Dict[str, float]:
        values = dict(presolved_solution)
        for step in reversed(self.steps):
            t, d = step.reduction_type, step.details
            if t in FIXING_REDUCTIONS:
                if d["var_name"] not in presolved_solution:
                    values[d["var_name"]] = float(d.get("value", d.get("fixed_value")))
            elif t == ReductionType.DOUBLETON_EQUATION:
                # x_k = intercept + slope * x_j
                values[d["var_name"]] = d["intercept"] + d["slope"] * values.get(d["base_var"], 0.0)
            elif t == ReductionType.DUPLICATE_COLUMN:
                # merged y = x_j + nu * x_k; split y back into bounded x_j and x_k
                j, k, nu = d["base_var"], d["var_name"], d["nu"]
                y = values.get(j, 0.0)
                lj, uj, lk, uk = d["lb_base"], d["ub_base"], d["lb_var"], d["ub_var"]
                lo_k, hi_k = lk, uk
                if nu > 0:
                    lo_k, hi_k = max(lo_k, (y - uj) / nu), min(hi_k, (y - lj) / nu)
                else:
                    lo_k, hi_k = max(lo_k, (y - lj) / nu), min(hi_k, (y - uj) / nu)
                xk = 0.0 if lo_k <= 0.0 <= hi_k else (lo_k if abs(lo_k) < abs(hi_k) else hi_k)
                if not np.isfinite(xk):
                    xk = lo_k if np.isfinite(lo_k) else hi_k
                values[k] = float(xk)
                values[j] = float(y - nu * xk)
        return values

    def restore_primal(self, presolved_solution: Dict[str, float]) -> np.ndarray:
        """
        Reconstruct original primal solution vector x in original variable order.
        """
        values = self.restore_dict(presolved_solution)
        return np.array([float(values.get(v, 0.0)) for v in self.original_var_names], dtype=np.float64)
