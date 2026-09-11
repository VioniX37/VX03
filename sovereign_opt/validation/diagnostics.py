"""
Numerical and structural diagnostics for optimization problems.
"""
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import scipy.sparse as sp
from sovereign_opt.model.model import OptimizationModel


@dataclass
class ProblemDiagnostics:
    matrix_condition_estimate: float
    coefficient_magnitude_ratio: float
    degeneracy_risk: str
    sparsity_structure: str
    numerical_warnings: List[str]


class DiagnosticAnalyzer:
    """
    Analyzes numerical conditioning and risk factors before solving.
    """

    @classmethod
    def analyze(cls, model: OptimizationModel) -> ProblemDiagnostics:
        warnings = []
        meta = model.get_metadata()

        ratio = (meta.max_coefficient / meta.min_coefficient) if meta.min_coefficient > 0 else 1.0
        if ratio > 1e6:
            warnings.append(f"High coefficient dynamic range ({ratio:.1e}): numerical scaling strongly advised.")

        # Estimate condition
        cond_est = min(ratio, 1e8)

        # Degeneracy risk assessment
        degeneracy_risk = "LOW"
        if meta.num_constraints > 0 and (meta.num_variables / meta.num_constraints) > 5.0:
            degeneracy_risk = "MODERATE"
        if ratio > 1e7:
            degeneracy_risk = "HIGH"

        sparsity_str = f"Sparse ({meta.density*100:.2f}% density)" if meta.density < 0.2 else "Dense"

        return ProblemDiagnostics(
            matrix_condition_estimate=float(cond_est),
            coefficient_magnitude_ratio=float(ratio),
            degeneracy_risk=degeneracy_risk,
            sparsity_structure=sparsity_str,
            numerical_warnings=warnings,
        )
