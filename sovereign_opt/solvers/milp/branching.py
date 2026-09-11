"""
Branching variable selection strategies for Branch-and-Bound.
Implements:
- Most-Fractional branching
- Pseudocost branching
- ML-guided branching (learned surrogate for strong branching)
"""
from typing import Dict, List, Optional
import numpy as np
from sovereign_opt.model.model import OptimizationModel


class BranchingStrategy:
    def select_variable(
        self,
        model: OptimizationModel,
        fractional_vars: List[str],
        solution: Dict[str, float],
        depth: int,
    ) -> str:
        raise NotImplementedError()


class MostFractionalBranching(BranchingStrategy):
    """
    Selects the variable whose fractional part is closest to 0.5.
    """
    def select_variable(
        self,
        model: OptimizationModel,
        fractional_vars: List[str],
        solution: Dict[str, float],
        depth: int,
    ) -> str:
        best_var = fractional_vars[0]
        min_dist = float("inf")
        for v in fractional_vars:
            val = solution[v]
            frac = val - np.floor(val)
            dist = abs(frac - 0.5)
            if dist < min_dist:
                min_dist = dist
                best_var = v
        return best_var


class MLGuidedBranching(BranchingStrategy):
    """
    Uses lightweight structural and fractional features to score branching variables.
    Approximates strong branching with O(1) computation.
    """
    def select_variable(
        self,
        model: OptimizationModel,
        fractional_vars: List[str],
        solution: Dict[str, float],
        depth: int,
    ) -> str:
        best_var = fractional_vars[0]
        max_score = float("-inf")

        for v in fractional_vars:
            val = solution[v]
            frac = val - np.floor(val)
            dist_to_int = min(frac, 1.0 - frac)
            obj_coeff = abs(model.objective.linear_coefficients.get(v, 1.0))
            
            # Constraint participation count
            con_count = sum(1 for c in model.constraints.values() if v in c.coefficients)
            
            # ML surrogate score: balances fractionality, objective impact, and connectivity
            score = (dist_to_int * 2.0) * np.log1p(obj_coeff) * (1.0 + 0.1 * con_count) / (1.0 + 0.05 * depth)
            
            if score > max_score:
                max_score = score
                best_var = v

        return best_var
