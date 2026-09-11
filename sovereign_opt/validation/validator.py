"""
Independent mathematical solution validator.
Enforces the Sovereign Safety Tenet:
"ML decides HOW the problem should be attacked.
 The mathematical solver determines and verifies WHAT the solution is."
Uses scaled relative-plus-absolute tolerances for industrial-scale models.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.solvers.base import SolverResult


@dataclass
class ValidationCertificate:
    """
    Mathematical trust certificate verifying solution feasibility and objective accuracy.
    """
    is_valid: bool
    status: str  # "PASSED" or "FAILED"
    max_primal_violation: float
    max_bound_violation: float
    max_integrality_violation: float
    reported_objective: Optional[float]
    recomputed_objective: Optional[float]
    objective_difference: float
    checks: Dict[str, bool] = field(default_factory=dict)
    violation_details: List[str] = field(default_factory=list)


class IndependentValidator:
    """
    Independently verifies any candidate solution against raw problem equations.
    Does not use solver internal state; recomputes everything directly.
    """

    def __init__(
        self,
        feasibility_tolerance: float = 1e-4,
        integrality_tolerance: float = 1e-4,
        objective_tolerance: float = 1e-3,
    ):
        self.feas_tol = feasibility_tolerance
        self.int_tol = integrality_tolerance
        self.obj_tol = objective_tolerance

    def verify(self, model: OptimizationModel, result: SolverResult) -> ValidationCertificate:
        if not result.primal_solution or not result.is_feasible:
            return ValidationCertificate(
                is_valid=False,
                status="NO_SOLUTION_TO_VERIFY",
                max_primal_violation=0.0,
                max_bound_violation=0.0,
                max_integrality_violation=0.0,
                reported_objective=result.objective_value,
                recomputed_objective=None,
                objective_difference=0.0,
                checks={"has_solution": False},
                violation_details=["Result does not contain a feasible primal solution."],
            )

        sol = result.primal_solution
        violations = []
        checks = {}

        # 1. Variable bounds verification
        max_bound_viol = 0.0
        for v_name, var in model.variables.items():
            val = sol.get(v_name, 0.0)
            lb = var.lower_bound
            ub = var.upper_bound

            # Relative-scaled tolerance
            lb_tol = self.feas_tol * max(1.0, abs(lb) if not np.isneginf(lb) else 1.0)
            ub_tol = self.feas_tol * max(1.0, abs(ub) if not np.isposinf(ub) else 1.0)

            if val < lb - lb_tol:
                viol = lb - val
                max_bound_viol = max(max_bound_viol, viol)
                violations.append(f"Var '{v_name}' = {val:.6g} < lower bound {lb:.6g} (viol: {viol:.2e})")

            if val > ub + ub_tol:
                viol = val - ub
                max_bound_viol = max(max_bound_viol, viol)
                violations.append(f"Var '{v_name}' = {val:.6g} > upper bound {ub:.6g} (viol: {viol:.2e})")

        checks["bounds_satisfied"] = max_bound_viol <= self.feas_tol

        # 2. Constraints verification
        max_primal_viol = 0.0
        for c_name, con in model.constraints.items():
            row_val = sum(coeff * sol.get(v_name, 0.0) for v_name, coeff in con.coefficients.items())
            lb = con.lower_bound
            ub = con.upper_bound

            lb_tol = self.feas_tol * max(1.0, abs(lb) if not np.isneginf(lb) else 1.0)
            ub_tol = self.feas_tol * max(1.0, abs(ub) if not np.isposinf(ub) else 1.0)

            if row_val < lb - lb_tol:
                viol = lb - row_val
                max_primal_viol = max(max_primal_viol, viol)
                violations.append(f"Con '{c_name}' = {row_val:.6g} < lower bound {lb:.6g} (viol: {viol:.2e})")

            if row_val > ub + ub_tol:
                viol = row_val - ub
                max_primal_viol = max(max_primal_viol, viol)
                violations.append(f"Con '{c_name}' = {row_val:.6g} > upper bound {ub:.6g} (viol: {viol:.2e})")

        checks["constraints_satisfied"] = max_primal_viol <= self.feas_tol

        # 3. Integrality verification
        max_int_viol = 0.0
        for v_name, var in model.variables.items():
            if var.is_integer:
                val = sol.get(v_name, 0.0)
                int_viol = abs(val - round(val))
                max_int_viol = max(max_int_viol, int_viol)
                if int_viol > self.int_tol:
                    violations.append(f"Integer Var '{v_name}' = {val:.6g} is fractional (viol: {int_viol:.2e})")

        checks["integrality_satisfied"] = max_int_viol <= self.int_tol

        # 4. Objective independent recomputation
        recomputed_obj = model.objective.offset
        multiplier = 1.0 if model.objective.sense == "minimize" else -1.0

        for v_name, coeff in model.objective.linear_coefficients.items():
            recomputed_obj += coeff * sol.get(v_name, 0.0)

        for (v1, v2), coeff in model.objective.quadratic_coefficients.items():
            recomputed_obj += 0.5 * coeff * sol.get(v1, 0.0) * sol.get(v2, 0.0)

        obj_diff = 0.0
        if result.objective_value is not None:
            obj_diff = abs(recomputed_obj - result.objective_value)
            rel_obj_diff = obj_diff / max(1.0, abs(recomputed_obj))
            checks["objective_matches"] = rel_obj_diff <= self.obj_tol
        else:
            checks["objective_matches"] = True

        all_passed = (
            checks["bounds_satisfied"]
            and checks["constraints_satisfied"]
            and checks["integrality_satisfied"]
            and checks["objective_matches"]
        )

        return ValidationCertificate(
            is_valid=all_passed,
            status="PASSED" if all_passed else "FAILED",
            max_primal_violation=float(max_primal_viol),
            max_bound_violation=float(max_bound_viol),
            max_integrality_violation=float(max_int_viol),
            reported_objective=result.objective_value,
            recomputed_objective=float(recomputed_obj),
            objective_difference=float(obj_diff),
            checks=checks,
            violation_details=violations[:20],
        )
