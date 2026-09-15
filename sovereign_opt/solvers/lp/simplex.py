"""
Revised Simplex solver for Linear Programming (v2).

Built on the sovereign bounded simplex engine (`simplex_engine.py`):
- Bounded slack form (no free-variable shifting, no upper-bound rows, no artificials)
- Power-of-two geometric scaling
- Primal simplex with composite Phase 1 and Devex pricing
- Dual simplex with dual steepest-edge pricing (used automatically when the start basis is dual feasible)
- Harris two-pass ratio tests, bound perturbation, Bland anti-cycling fallback
- Dense / sparse LU basis factorization with eta updates
- Returns primal values, row duals, and reduced costs
"""
import time
from typing import Optional
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus
from sovereign_opt.solvers.lp.standard_form import BoundedForm
from sovereign_opt.solvers.lp.simplex_engine import SimplexEngine, LPStatus

LP_TO_SOLVER_STATUS = {
    LPStatus.OPTIMAL: SolverStatus.OPTIMAL,
    LPStatus.INFEASIBLE: SolverStatus.INFEASIBLE,
    LPStatus.UNBOUNDED: SolverStatus.UNBOUNDED,
    LPStatus.ITERATION_LIMIT: SolverStatus.ITERATION_LIMIT,
    LPStatus.TIME_LIMIT: SolverStatus.TIME_LIMIT,
    LPStatus.NUMERICAL_ERROR: SolverStatus.NUMERICAL_ERROR,
}


def downsample_trace(trace, limit: int = 60):
    if len(trace) <= limit:
        return list(trace)
    stride = int(np.ceil(len(trace) / limit))
    sampled = trace[::stride]
    if sampled[-1] is not trace[-1]:
        sampled.append(trace[-1])
    return sampled


def lp_result_from_engine(
    form: BoundedForm,
    engine: SimplexEngine,
    lp_status: str,
    start_time: float,
    extra_diagnostics: Optional[dict] = None,
) -> SolverResult:
    status = LP_TO_SOLVER_STATUS.get(lp_status, SolverStatus.NUMERICAL_ERROR)
    diagnostics = {
        "basis_size": int(engine.m),
        "pricing_method": "devex (primal) / dual steepest edge (dual)",
        "phase1_iterations": int(engine.phase1_iterations),
        "refactorizations": int(engine.factor.num_factorizations),
        "dual_sign_convention": "minimize-normalized: y_i > 0 row at lower bound, y_i < 0 at upper bound",
    }
    trace = []
    for t in engine.trace:
        entry = dict(t)
        entry["objective"] = form.obj_sign * entry.pop("objective_internal") + form.offset
        trace.append(entry)
    diagnostics["iteration_trace"] = downsample_trace(trace)
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)

    result = SolverResult(status=status, iterations=int(engine.total_iterations),
                          runtime_seconds=time.time() - start_time, diagnostics=diagnostics)

    x = form.unscale_x(engine.x)
    xs = x[: form.n]
    primal_ok = engine.primal_infeasibility() <= 1e-6

    if status == SolverStatus.OPTIMAL or (status in (SolverStatus.ITERATION_LIMIT, SolverStatus.TIME_LIMIT) and primal_ok):
        result.primal_solution = {name: float(v) for name, v in zip(form.var_names, xs)}
        result.objective_value = form.user_objective(xs)
        diagnostics["has_feasible_point"] = bool(primal_ok)
        if status == SolverStatus.OPTIMAL:
            y_s, d_s = engine.duals()
            y = form.unscale_y(y_s)
            d = form.unscale_d(d_s)
            result.dual_solution = {name: float(v) for name, v in zip(form.con_names, y)}
            result.reduced_costs = {name: float(v) for name, v in zip(form.var_names, d[: form.n])}
            diagnostics["iteration_trace"].append({
                "iteration": int(engine.total_iterations), "status": "OPTIMAL", "objective": result.objective_value,
            })
    elif status == SolverStatus.UNBOUNDED and engine.ray is not None:
        q, direction, delta = engine.ray
        ray = np.zeros(engine.N)
        ray[q] = direction
        ray[engine.head] = delta
        ray = form.unscale_x(ray)[: form.n]
        diagnostics["unbounded_ray"] = {n: float(v) for n, v in zip(form.var_names, ray) if abs(v) > 1e-12}
    elif status == SolverStatus.INFEASIBLE and engine.farkas_row is not None:
        diagnostics["infeasibility_certificate"] = "dual simplex ray (row combination proving infeasibility)"
    return result


class RevisedSimplexSolver(SolverBase):
    """
    Sovereign Bounded Revised Simplex LP Solver (primal + dual).
    """

    def __init__(
        self,
        max_iterations: int = 200000,
        tolerance: float = 1e-9,
        pricing: str = "devex",
        method: str = "auto",
        scaling: bool = True,
    ):
        super().__init__(name="RevisedSimplex")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.pricing = pricing
        self.method = method
        self.scaling = scaling

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 60.0, **kwargs) -> SolverResult:
        start_time = time.time()
        if model.objective.is_quadratic:
            return SolverResult(
                status=SolverStatus.NUMERICAL_ERROR,
                runtime_seconds=0.0,
                diagnostics={"error": "Quadratic objective: use a QP solver (qp_interior_point / active_set)."},
            )
        form = BoundedForm(model, scale=self.scaling)
        if np.any(form.lb > form.ub + 1e-9):
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                runtime_seconds=time.time() - start_time,
                diagnostics={"infeasibility_certificate": "a variable or row has lower bound > upper bound"},
            )
        engine = SimplexEngine.from_form(form, primal_tol=self.tolerance, dual_tol=self.tolerance)
        warm = kwargs.get("warm_start")
        if warm is not None:
            engine.set_basis(*warm)
        lp_status = engine.solve(max_iterations=self.max_iterations, time_limit=time_limit_seconds, method=self.method)
        return lp_result_from_engine(form, engine, lp_status, start_time, {"method": self.method})
