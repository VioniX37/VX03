"""
Unified solve pipeline shared by the REST API and the CLI.

    validate -> classify -> presolve -> ML strategy -> compatible solver -> core solve
             -> postsolve -> dual recovery on the original model -> independent certificate

The ML engine only chooses HOW to solve; the certificate is always computed on the
original, un-presolved model.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.presolve.presolver import Presolver, PresolveStats, PresolveInfeasibleError, PresolveUnboundedError
from sovereign_opt.presolve.postsolve import PostsolveMapper
from sovereign_opt.ml.strategy import MLStrategyEngine, StrategyRecommendation
from sovereign_opt.solvers.base import SolverResult, SolverStatus
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver, lp_result_from_engine
from sovereign_opt.solvers.lp.interior_point import InteriorPointSolver, crossover
from sovereign_opt.solvers.lp.standard_form import BoundedForm
from sovereign_opt.solvers.lp.simplex_engine import LPStatus
from sovereign_opt.solvers.milp.branch_bound import BranchAndBoundSolver
from sovereign_opt.solvers.qp.active_set import ActiveSetQPSolver
from sovereign_opt.solvers.qp.interior_point_qp import QPInteriorPointSolver
from sovereign_opt.solvers.qp.polish import recover_duals
from sovereign_opt.validation.validator import IndependentValidator, ValidationCertificate

LP_ALGORITHMS = ("simplex", "dual_simplex", "interior_point")
QP_ALGORITHMS = ("qp_interior_point", "active_set")
ALL_ALGORITHMS = ("auto",) + LP_ALGORITHMS + ("branch_and_bound",) + QP_ALGORITHMS


@dataclass
class SolveOutcome:
    result: SolverResult
    core_result: SolverResult
    solver_name: str
    algorithm: str
    problem_class: str
    recommendation: StrategyRecommendation
    certificate: ValidationCertificate
    presolved_model: Optional[OptimizationModel] = None
    mapper: Optional[PostsolveMapper] = None
    presolve_stats: Optional[PresolveStats] = None
    notes: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


def make_solver(algorithm: str, time_limit_seconds: float):
    if algorithm == "simplex":
        return RevisedSimplexSolver()
    if algorithm == "dual_simplex":
        solver = RevisedSimplexSolver(method="dual")
        solver.name = "DualSimplex"
        return solver
    if algorithm == "interior_point":
        return InteriorPointSolver()
    if algorithm == "branch_and_bound":
        return BranchAndBoundSolver(time_limit_seconds=time_limit_seconds)
    if algorithm == "active_set":
        return ActiveSetQPSolver()
    if algorithm == "qp_interior_point":
        return QPInteriorPointSolver()
    raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {ALL_ALGORITHMS}.")


def compatible_algorithm(algorithm: str, problem_class: str, notes: List[str]) -> str:
    if problem_class in ("MILP", "MIQP"):
        if algorithm != "branch_and_bound":
            notes.append(f"'{algorithm}' cannot enforce integrality on a {problem_class}; using branch_and_bound.")
        return "branch_and_bound"
    if problem_class == "QP":
        if algorithm not in QP_ALGORITHMS:
            notes.append(f"'{algorithm}' cannot handle a quadratic objective; using qp_interior_point.")
            return "qp_interior_point"
        return algorithm
    return algorithm


def _recompute_objective(model: OptimizationModel, primal: Dict[str, float]) -> float:
    names = model.variable_names
    x = np.array([primal.get(v, 0.0) for v in names])
    c = model.get_objective_vector()
    Q = model.get_quadratic_matrix()
    sign = 1.0 if model.objective.sense == "minimize" else -1.0
    f = float(c @ x) + (0.5 * float(x @ (Q @ x)) if Q.nnz else 0.0)
    return sign * f + model.objective.offset


def _recover_original_duals(model: OptimizationModel, final: SolverResult, problem_class: str, notes: List[str]):
    """After postsolve, rebuild duals for the ORIGINAL model (LP: crossover basis; QP: signed NNLS multipliers)."""
    form = BoundedForm(model, scale=True)
    names = model.variable_names
    x = np.array([final.primal_solution.get(v, 0.0) for v in names])
    if problem_class == "LP":
        A = form.A_orig
        x_full = np.concatenate([x, A @ x]) / form.col_scale
        engine, st = crossover(form, x_full, time_limit=30.0)
        if engine is not None and st == LPStatus.OPTIMAL:
            polished = lp_result_from_engine(form, engine, st, time.time())
            if polished.objective_value is not None and final.objective_value is not None:
                if polished.objective_value * form.obj_sign < final.objective_value * form.obj_sign - 1e-7 * max(1.0, abs(final.objective_value)):
                    notes.append("Dual recovery found a strictly better vertex than the presolved solution; using it.")
            final.primal_solution = polished.primal_solution
            final.objective_value = polished.objective_value
            final.dual_solution = polished.dual_solution
            final.reduced_costs = polished.reduced_costs
            final.diagnostics["dual_recovery"] = f"crossover on original model ({engine.total_iterations} pivots)"
            return
    y = recover_duals(form, x)
    if y is not None:
        final.dual_solution = {cn: float(v) for cn, v in zip(model.constraint_names, y)}
        grad = form.c_orig + (form.Q_orig @ x if form.Q_orig is not None else 0.0)
        d = grad - form.A_orig.T @ y
        final.reduced_costs = {nm: float(v) for nm, v in zip(names, d)}
        final.diagnostics["dual_recovery"] = "signed NNLS multipliers on original model"
    else:
        final.diagnostics["dual_recovery"] = "not available for this size / point"


def solve_model(
    model: OptimizationModel,
    algorithm: str = "auto",
    enable_presolve: bool = True,
    time_limit_seconds: float = 60.0,
    presolve_result: Optional[Tuple[OptimizationModel, PostsolveMapper, PresolveStats]] = None,
) -> SolveOutcome:
    t_start = time.time()
    timings: Dict[str, float] = {}
    notes: List[str] = list(model.validate())
    problem_class = model.classify()

    # ---------------------------------------------------------------- presolve
    t0 = time.time()
    pmodel, mapper, stats = None, None, None
    model_to_solve = model
    presolve_failure: Optional[SolverResult] = None
    if enable_presolve:
        try:
            pmodel, mapper, stats = presolve_result if presolve_result is not None else Presolver().presolve(model)
            model_to_solve = pmodel
        except PresolveInfeasibleError as exc:
            presolve_failure = SolverResult(status=SolverStatus.INFEASIBLE,
                                            diagnostics={"infeasibility_certificate": f"presolve: {exc}"})
        except PresolveUnboundedError as exc:
            notes.append(f"Presolve found an unbounded direction ({exc}); solving the original model to classify it.")
    timings["presolve"] = time.time() - t0

    # ---------------------------------------------------------------- strategy
    t0 = time.time()
    recommendation = MLStrategyEngine().recommend(model, stats)
    requested = recommendation.recommended_algorithm if algorithm in (None, "auto") else algorithm
    chosen = compatible_algorithm(requested, problem_class, notes)
    solver = make_solver(chosen, time_limit_seconds)
    timings["strategy"] = time.time() - t0

    # -------------------------------------------------------------------- solve
    t0 = time.time()
    if presolve_failure is not None:
        core = presolve_failure
    elif model_to_solve.num_variables == 0:
        core = SolverResult(status=SolverStatus.OPTIMAL, objective_value=model_to_solve.objective.offset,
                            diagnostics={"note": "presolve eliminated every variable"})
    else:
        remaining = max(0.1, time_limit_seconds - (time.time() - t_start))
        core = solver.solve(model_to_solve, time_limit_seconds=remaining)
    core.runtime_seconds = time.time() - t0
    timings["solve"] = core.runtime_seconds

    # ----------------------------------------------------------------- postsolve
    t0 = time.time()
    final = SolverResult(
        status=core.status,
        objective_value=core.objective_value,
        primal_solution=dict(core.primal_solution),
        dual_solution=dict(core.dual_solution),
        reduced_costs=dict(core.reduced_costs),
        iterations=core.iterations,
        runtime_seconds=core.runtime_seconds,
        nodes_explored=core.nodes_explored,
        mip_gap=core.mip_gap,
        best_bound=core.best_bound,
        diagnostics=dict(core.diagnostics),
    )
    has_point = bool(core.primal_solution) or (core.status == SolverStatus.OPTIMAL and model_to_solve.num_variables == 0)
    if mapper is not None and has_point and presolve_failure is None:
        restored = mapper.restore_dict(core.primal_solution)
        final.primal_solution = {v: float(restored.get(v, 0.0)) for v in model.variable_names}
        final.objective_value = _recompute_objective(model, final.primal_solution)
        final.dual_solution, final.reduced_costs = {}, {}
        if core.status == SolverStatus.OPTIMAL and problem_class in ("LP", "QP"):
            _recover_original_duals(model, final, problem_class, notes)
    timings["postsolve"] = time.time() - t0

    # ---------------------------------------------------------------- certificate
    t0 = time.time()
    certificate = IndependentValidator().verify(model, final)
    timings["certificate"] = time.time() - t0
    timings["total"] = time.time() - t_start

    return SolveOutcome(
        result=final,
        core_result=core,
        solver_name=solver.name,
        algorithm=chosen,
        problem_class=problem_class,
        recommendation=recommendation,
        certificate=certificate,
        presolved_model=pmodel,
        mapper=mapper,
        presolve_stats=stats,
        notes=notes,
        timings=timings,
    )
