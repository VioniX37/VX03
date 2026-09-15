"""
Independent mathematical solution validator (v2).
Enforces the Sovereign Safety Tenet:
"ML decides HOW the problem should be attacked.
 The mathematical solver determines and verifies WHAT the solution is."

v2 certifies optimality, not only feasibility:
- Feasibility: row activities, variable bounds, integrality (scaled relative tolerances).
- Objective: independent recomputation of c^T x + 1/2 x^T Q x + offset.
- LP / QP optimality: KKT recomputation from the reported row duals y only. Reduced costs are
  recomputed as z = grad f(x) - A^T y, then dual feasibility, complementary slackness, and the
  primal-dual (strong duality) gap are checked.
- MILP / MIQP: incumbent versus reported best bound (branch-and-bound dual bound).

Everything is recomputed from the raw model; solver-internal state is never trusted.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.base import SolverResult, SolverStatus


@dataclass
class ValidationCertificate:
    """
    Mathematical trust certificate.
    status: OPTIMAL_CERTIFIED | FEASIBLE_CERTIFIED | FAILED | NO_SOLUTION_TO_VERIFY
    is_valid: the primal point is feasible and the objective value is reproduced.
    """
    is_valid: bool
    status: str
    max_primal_violation: float
    max_bound_violation: float
    max_integrality_violation: float
    reported_objective: Optional[float]
    recomputed_objective: Optional[float]
    objective_difference: float
    checks: Dict[str, bool] = field(default_factory=dict)
    violation_details: List[str] = field(default_factory=list)
    optimality_certified: bool = False
    optimality_method: str = "none"
    max_dual_violation: Optional[float] = None
    duality_gap: Optional[float] = None


class IndependentValidator:
    """
    Independently verifies any candidate solution against raw problem equations.
    """

    def __init__(
        self,
        feasibility_tolerance: float = 1e-6,
        integrality_tolerance: float = 1e-5,
        objective_tolerance: float = 1e-6,
        optimality_tolerance: float = 1e-6,
        mip_gap_tolerance: float = 1e-4,
    ):
        self.feas_tol = feasibility_tolerance
        self.int_tol = integrality_tolerance
        self.obj_tol = objective_tolerance
        self.opt_tol = optimality_tolerance
        self.gap_tol = mip_gap_tolerance

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
                violation_details=[f"Result status '{result.status.value}' carries no feasible primal solution."],
            )

        names = model.variable_names
        con_names = model.constraint_names
        sol = result.primal_solution
        violations: List[str] = []
        checks: Dict[str, bool] = {}

        missing = [v for v in names if v not in sol]
        if missing:
            violations.append(f"{len(missing)} variable(s) missing from solution (treated as 0), e.g. '{missing[0]}'")
        x = np.array([float(sol.get(v, 0.0)) for v in names])
        checks["values_finite"] = bool(np.all(np.isfinite(x)))
        x = np.nan_to_num(x, nan=0.0, posinf=1e300, neginf=-1e300)

        A, rl, ru, cl, cu = model.to_matrix_form()

        # 1. Variable bounds
        lo_v = np.where(np.isfinite(cl), cl - x, 0.0)
        hi_v = np.where(np.isfinite(cu), x - cu, 0.0)
        bviol = np.maximum(np.maximum(lo_v, hi_v), 0.0)
        btol = self.feas_tol * np.maximum(1.0, np.maximum(np.where(np.isfinite(cl), np.abs(cl), 0.0),
                                                            np.where(np.isfinite(cu), np.abs(cu), 0.0)))
        bad = np.flatnonzero(bviol > btol)
        for j in bad[:10]:
            violations.append(f"Var '{names[j]}' = {x[j]:.6g} outside [{cl[j]:.6g}, {cu[j]:.6g}] (viol: {bviol[j]:.2e})")
        checks["bounds_satisfied"] = bad.size == 0
        max_bound_viol = float(bviol.max()) if bviol.size else 0.0

        # 2. Constraints
        act = A @ x if A.shape[0] else np.zeros(0)
        lo_r = np.where(np.isfinite(rl), rl - act, 0.0)
        hi_r = np.where(np.isfinite(ru), act - ru, 0.0)
        rviol = np.maximum(np.maximum(lo_r, hi_r), 0.0)
        rtol = self.feas_tol * np.maximum(1.0, np.maximum(np.where(np.isfinite(rl), np.abs(rl), 0.0),
                                                            np.where(np.isfinite(ru), np.abs(ru), 0.0)))
        bad = np.flatnonzero(rviol > rtol)
        for i in bad[:10]:
            violations.append(f"Con '{con_names[i]}' = {act[i]:.6g} outside [{rl[i]:.6g}, {ru[i]:.6g}] (viol: {rviol[i]:.2e})")
        checks["constraints_satisfied"] = bad.size == 0
        max_primal_viol = float(rviol.max()) if rviol.size else 0.0

        # 3. Integrality
        int_mask = np.array([model.variables[v].is_integer for v in names], dtype=bool)
        iviol = np.abs(x - np.round(x)) * int_mask
        bad = np.flatnonzero(iviol > self.int_tol)
        for j in bad[:10]:
            violations.append(f"Integer Var '{names[j]}' = {x[j]:.6g} is fractional (viol: {iviol[j]:.2e})")
        checks["integrality_satisfied"] = bad.size == 0
        max_int_viol = float(iviol.max()) if iviol.size else 0.0

        # 4. Objective recomputation (user sense)
        c_min = model.get_objective_vector()
        Q_min = model.get_quadratic_matrix()
        sign = 1.0 if model.objective.sense == "minimize" else -1.0
        f_min = float(c_min @ x) + (0.5 * float(x @ (Q_min @ x)) if Q_min.nnz else 0.0)
        recomputed_obj = sign * f_min + model.objective.offset
        obj_diff = 0.0
        if result.objective_value is not None:
            obj_diff = abs(recomputed_obj - result.objective_value)
            checks["objective_matches"] = obj_diff <= self.obj_tol * max(1.0, abs(recomputed_obj))
            if not checks["objective_matches"]:
                violations.append(f"Reported objective {result.objective_value:.10g} != recomputed {recomputed_obj:.10g}")
        else:
            checks["objective_matches"] = True

        is_valid = all(checks.get(k, True) for k in
                       ("values_finite", "bounds_satisfied", "constraints_satisfied", "integrality_satisfied", "objective_matches"))

        # 5. Optimality certification
        optimality_certified = False
        method = "none"
        max_dual_viol = None
        gap_value = None
        p_class = model.classify()

        if is_valid and p_class in ("LP", "QP") and result.dual_solution:
            method = "KKT recomputation (dual feasibility, complementary slackness, strong duality)"
            y = np.array([float(result.dual_solution.get(cn, 0.0)) for cn in con_names])
            grad = c_min + (Q_min @ x if Q_min.nnz else 0.0)
            z = grad - (A.T @ y if A.shape[0] else 0.0)
            gscale = 1.0 + float(np.max(np.abs(grad), initial=0.0)) + float(np.max(np.abs(y), initial=0.0))
            act_tol_v = 1e-6 * np.maximum(1.0, np.abs(x))
            at_l = np.isfinite(cl) & (x - cl <= act_tol_v)
            at_u = np.isfinite(cu) & (cu - x <= act_tol_v)
            dv_var = np.where(at_l & at_u, 0.0, np.where(at_l, np.maximum(-z, 0.0), np.where(at_u, np.maximum(z, 0.0), np.abs(z))))
            act_tol_r = 1e-6 * np.maximum(1.0, np.abs(act))
            r_l = np.isfinite(rl) & (act - rl <= act_tol_r)
            r_u = np.isfinite(ru) & (ru - act <= act_tol_r)
            dv_row = np.where(r_l & r_u, 0.0, np.where(r_l, np.maximum(-y, 0.0), np.where(r_u, np.maximum(y, 0.0), np.abs(y))))
            max_dual_viol = float(max(dv_var.max(initial=0.0), dv_row.max(initial=0.0)) / gscale)
            checks["dual_feasible"] = max_dual_viol <= self.opt_tol

            # Strong duality: primal objective vs Lagrangian dual objective built from bounds
            yb = np.where(y > 0, np.where(np.isfinite(rl), rl, 0.0), np.where(np.isfinite(ru), ru, 0.0))
            zb = np.where(z > 0, np.where(np.isfinite(cl), cl, 0.0), np.where(np.isfinite(cu), cu, 0.0))
            quad = 0.5 * float(x @ (Q_min @ x)) if Q_min.nnz else 0.0
            dual_obj = float(y @ yb) + float(z @ zb) - quad
            gap_value = abs(f_min - dual_obj) / max(1.0, abs(f_min))
            checks["duality_gap_closed"] = gap_value <= 10 * self.opt_tol
            optimality_certified = checks["dual_feasible"] and checks["duality_gap_closed"]
            if not optimality_certified:
                violations.append(f"Optimality not certified: dual violation {max_dual_viol:.2e}, duality gap {gap_value:.2e}")

        elif is_valid and p_class in ("MILP", "MIQP"):
            bb = result.best_bound if result.best_bound is not None else result.diagnostics.get("best_bound")
            if bb is not None and np.isfinite(bb):
                method = "branch-and-bound dual bound vs verified incumbent"
                inc = recomputed_obj
                gap_value = abs(inc - bb) / max(1.0, abs(inc))
                consistent = (bb <= inc + 1e-6 * max(1.0, abs(inc))) if sign > 0 else (bb >= inc - 1e-6 * max(1.0, abs(inc)))
                checks["bound_consistent"] = bool(consistent)
                checks["mip_gap_closed"] = gap_value <= self.gap_tol
                optimality_certified = checks["bound_consistent"] and checks["mip_gap_closed"]

        if not is_valid:
            status = "FAILED"
        elif optimality_certified:
            status = "OPTIMAL_CERTIFIED"
        else:
            status = "FEASIBLE_CERTIFIED"

        return ValidationCertificate(
            is_valid=is_valid,
            status=status,
            max_primal_violation=max_primal_viol,
            max_bound_violation=max_bound_viol,
            max_integrality_violation=max_int_viol,
            reported_objective=result.objective_value,
            recomputed_objective=float(recomputed_obj),
            objective_difference=float(obj_diff),
            checks=checks,
            violation_details=violations[:20],
            optimality_certified=optimality_certified,
            optimality_method=method,
            max_dual_violation=max_dual_viol,
            duality_gap=gap_value,
        )
