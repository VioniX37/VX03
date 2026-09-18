"""
Structural feature extraction for the ML Strategy Engine.
Computes a normalized 28-dimensional feature vector from raw and presolved models.
"""
from typing import Dict, List, Optional
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.presolve.presolver import PresolveStats


class FeatureExtractor:
    """
    Extracts numerical features describing the mathematical structure of an optimization instance.
    """

    FEATURE_NAMES = [
        "log_num_vars",
        "log_num_cons",
        "log_num_nnz",
        "density",
        "aspect_ratio",
        "coeff_range_log10",
        "frac_continuous",
        "frac_integer",
        "frac_binary",
        "frac_bounded_below",
        "frac_bounded_above",
        "frac_boxed",
        "frac_le_cons",
        "frac_ge_cons",
        "frac_eq_cons",
        "var_reduction_pct",
        "con_reduction_pct",
        "nnz_reduction_pct",
        "fixed_vars_ratio",
        "singleton_rows_ratio",
        "is_quadratic",
        "has_integers",
        "est_condition_log10",
    ]

    @classmethod
    def extract(
        cls,
        model: OptimizationModel,
        presolve_stats: Optional[PresolveStats] = None,
    ) -> Dict[str, float]:
        n = max(model.num_variables, 1)
        m = max(model.num_constraints, 1)
        meta = model.get_metadata()

        nnz = meta.num_nonzeros
        density = meta.density
        aspect = float(m) / float(n)

        min_c = meta.min_coefficient if meta.min_coefficient > 0 else 1e-6
        max_c = meta.max_coefficient if meta.max_coefficient > 0 else 1.0
        coeff_range = float(np.log10(max(max_c / min_c, 1.0)))

        # Variable bound distributions (plain comparisons: a NumPy ufunc per scalar is ~50x slower)
        lo = [v.lower_bound != -np.inf for v in model.variables.values()]
        hi = [v.upper_bound != np.inf for v in model.variables.values()]
        bounded_below = sum(lo)
        bounded_above = sum(hi)
        boxed = sum(a and b for a, b in zip(lo, hi))

        # Constraint senses
        le_count = sum(1 for c in model.constraints.values() if c.sense == ConstraintSense.LE)
        ge_count = sum(1 for c in model.constraints.values() if c.sense == ConstraintSense.GE)
        eq_count = sum(1 for c in model.constraints.values() if c.is_equality)

        # Presolve statistics
        if presolve_stats is not None:
            v_red = presolve_stats.var_reduction_pct
            c_red = presolve_stats.con_reduction_pct
            nnz_red = presolve_stats.nnz_reduction_pct
            fixed_ratio = presolve_stats.fixed_vars_count / n
            sing_ratio = presolve_stats.singleton_rows_count / m
        else:
            v_red, c_red, nnz_red = 0.0, 0.0, 0.0
            fixed_ratio, sing_ratio = 0.0, 0.0

        features = {
            "log_num_vars": float(np.log10(n)),
            "log_num_cons": float(np.log10(m)),
            "log_num_nnz": float(np.log10(max(nnz, 1))),
            "density": float(density),
            "aspect_ratio": float(np.clip(aspect, 0.01, 100.0)),
            "coeff_range_log10": coeff_range,
            "frac_continuous": meta.num_continuous / n,
            "frac_integer": meta.num_integer / n,
            "frac_binary": meta.num_binary / n,
            "frac_bounded_below": bounded_below / n,
            "frac_bounded_above": bounded_above / n,
            "frac_boxed": boxed / n,
            "frac_le_cons": le_count / m,
            "frac_ge_cons": ge_count / m,
            "frac_eq_cons": eq_count / m,
            "var_reduction_pct": v_red / 100.0,
            "con_reduction_pct": c_red / 100.0,
            "nnz_reduction_pct": nnz_red / 100.0,
            "fixed_vars_ratio": fixed_ratio,
            "singleton_rows_ratio": sing_ratio,
            "is_quadratic": 1.0 if meta.num_quadratic_terms > 0 else 0.0,
            "has_integers": 1.0 if (meta.num_integer + meta.num_binary) > 0 else 0.0,
            "est_condition_log10": min(coeff_range, 8.0),
        }

        return features

    @classmethod
    def to_vector(cls, features: Dict[str, float]) -> np.ndarray:
        return np.array([features.get(name, 0.0) for name in cls.FEATURE_NAMES], dtype=np.float64)
