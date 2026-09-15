"""
Branching variable selection strategies for Branch-and-Bound.
Implements:
- Most-Fractional branching
- ML-guided branching (learned surrogate score, now used as a tie-breaker)
- Pseudocost tracking with product scoring
- Reliability branching support (pseudocosts initialized by strong branching in the tree search)
"""
from typing import Dict, List, Tuple
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
    Constraint participation counts are cached per model.
    """
    def __init__(self):
        self._cache_key = None
        self._con_count: Dict[str, int] = {}

    def _counts(self, model: OptimizationModel) -> Dict[str, int]:
        key = (id(model), model.num_constraints)
        if key != self._cache_key:
            counts: Dict[str, int] = {}
            for con in model.constraints.values():
                for v in con.coefficients:
                    counts[v] = counts.get(v, 0) + 1
            self._con_count = counts
            self._cache_key = key
        return self._con_count

    def select_variable(
        self,
        model: OptimizationModel,
        fractional_vars: List[str],
        solution: Dict[str, float],
        depth: int,
    ) -> str:
        counts = self._counts(model)
        best_var = fractional_vars[0]
        max_score = float("-inf")
        for v in fractional_vars:
            val = solution[v]
            frac = val - np.floor(val)
            dist_to_int = min(frac, 1.0 - frac)
            obj_coeff = abs(model.objective.linear_coefficients.get(v, 1.0))
            score = (dist_to_int * 2.0) * np.log1p(obj_coeff) * (1.0 + 0.1 * counts.get(v, 0)) / (1.0 + 0.05 * depth)
            if score > max_score:
                max_score = score
                best_var = v
        return best_var


class ReliabilityBranching(BranchingStrategy):
    """
    Marker / configuration object for reliability pseudocost branching, which is executed
    inside the branch-and-cut tree (it needs warm-started LP solves for strong branching).
    """
    def __init__(self, reliability: int = 4, strong_candidates: int = 8, strong_iterations: int = 100):
        self.reliability = reliability
        self.strong_candidates = strong_candidates
        self.strong_iterations = strong_iterations


class PseudocostTracker:
    """Per-variable average objective degradation per unit change, for down and up branches."""

    def __init__(self, n: int):
        self.down_sum = np.zeros(n)
        self.down_cnt = np.zeros(n)
        self.up_sum = np.zeros(n)
        self.up_cnt = np.zeros(n)

    def update(self, j: int, direction: int, frac: float, gain: float):
        if not np.isfinite(gain):
            return
        unit = frac if direction < 0 else 1.0 - frac
        if unit <= 1e-9:
            return
        if direction < 0:
            self.down_sum[j] += gain / unit
            self.down_cnt[j] += 1
        else:
            self.up_sum[j] += gain / unit
            self.up_cnt[j] += 1

    def estimates(self, cols: np.ndarray, fracs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        def averaged(sums, cnts):
            known = cnts > 0
            default = float(np.mean(sums[known] / cnts[known])) if known.any() else 1.0
            vals = np.full(len(cols), default)
            k = cnts[cols] > 0
            vals[k] = sums[cols][k] / cnts[cols][k]
            return vals

        down = averaged(self.down_sum, self.down_cnt) * fracs
        up = averaged(self.up_sum, self.up_cnt) * (1.0 - fracs)
        return down, up

    def reliable(self, j: int, eta: int) -> bool:
        return min(self.down_cnt[j], self.up_cnt[j]) >= eta


def product_score(down: np.ndarray, up: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(down, eps) * np.maximum(up, eps)


def static_branching_scores(form) -> np.ndarray:
    """ML surrogate prior per structural column: objective impact x constraint connectivity."""
    col_nnz = np.diff(form.A_orig.tocsc().indptr)
    return np.log1p(np.abs(form.c_orig)) * (1.0 + 0.1 * col_nnz)
