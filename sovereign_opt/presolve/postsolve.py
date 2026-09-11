"""
Postsolve mapper for reconstructing original problem solutions from presolved results.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any
import numpy as np


class ReductionType(str, Enum):
    FIXED_VARIABLE = "fixed_variable"
    EMPTY_ROW = "empty_row"
    EMPTY_COL = "empty_col"
    SINGLETON_ROW = "singleton_row"
    REDUNDANT_ROW = "redundant_row"


@dataclass
class PresolveStep:
    reduction_type: ReductionType
    details: Dict[str, Any]


class PostsolveMapper:
    """
    Maintains the stack of presolve reductions to reconstruct
    the original problem's solution vector and dual multipliers.
    """

    def __init__(self, original_var_names: List[str], original_con_names: List[str]):
        self.original_var_names = list(original_var_names)
        self.original_con_names = list(original_con_names)
        self.steps: List[PresolveStep] = []
        self.fixed_vars: Dict[str, float] = {}

    def record_step(self, step: PresolveStep):
        self.steps.append(step)
        if step.reduction_type == ReductionType.FIXED_VARIABLE:
            self.fixed_vars[step.details["var_name"]] = step.details["value"]

    def restore_primal(self, presolved_solution: Dict[str, float]) -> np.ndarray:
        """
        Reconstruct original primal solution vector x in original variable order.
        """
        x_full = np.zeros(len(self.original_var_names), dtype=np.float64)
        
        # Start with presolved values and fixed values
        for i, v_name in enumerate(self.original_var_names):
            if v_name in presolved_solution:
                x_full[i] = presolved_solution[v_name]
            elif v_name in self.fixed_vars:
                x_full[i] = self.fixed_vars[v_name]
            else:
                # Default to 0.0 or lower bound if empty column
                x_full[i] = 0.0

        # Unwind reduction steps in reverse order if needed
        for step in reversed(self.steps):
            if step.reduction_type == ReductionType.EMPTY_COL:
                v_name = step.details["var_name"]
                val = step.details["fixed_value"]
                idx = self.original_var_names.index(v_name)
                x_full[idx] = val

        return x_full
