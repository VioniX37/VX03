"""
Regression tests for every accuracy defect found in v1, plus v2 capability checks.
"""
import itertools

import numpy as np
import pytest

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.model.variable import VariableType
from sovereign_opt.solvers.base import SolverResult, SolverStatus
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver
from sovereign_opt.solvers.lp.interior_point import InteriorPointSolver
from sovereign_opt.solvers.milp.branch_bound import BranchAndBoundSolver
from sovereign_opt.solvers.qp.active_set import ActiveSetQPSolver
from sovereign_opt.solvers.qp.interior_point_qp import QPInteriorPointSolver
from sovereign_opt.solvers.dispatch import solve_model
from sovereign_opt.validation.validator import IndependentValidator
from benchmarks.netlib.afiro import build_netlib_afiro, AFIRO_OPTIMAL_OBJECTIVE
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.power_dispatch import build_unit_commitment_model
from benchmarks.industrial.portfolio_selection import build_portfolio_selection_model

V = IndependentValidator()
REFINERY_QP_OPTIMUM = -5903161.066392752  # certified by KKT recomputation


def _free_variable_model():
    m = OptimizationModel("FreeVar")
    m.add_variable("x", float("-inf"), float("inf"))
    m.add_constraint("c", {"x": 1.0}, ConstraintSense.GE, rhs=-50000.0)
    m.set_objective({"x": 1.0})
    return m


@pytest.mark.parametrize("solver", [RevisedSimplexSolver(), InteriorPointSolver(), QPInteriorPointSolver()])
def test_free_variable_is_not_truncated(solver):
    # v1: simplex returned -10000 and IPM returned 0, both labelled optimal.
    m = _free_variable_model()
    r = solver.solve(m)
    assert r.is_optimal
    assert abs(r.objective_value + 50000.0) < 1e-6
    assert V.verify(m, r).status == "OPTIMAL_CERTIFIED"


def test_afiro_model_matches_netlib_dimensions():
    m = build_netlib_afiro()
    meta = m.get_metadata()
    assert (meta.num_variables, meta.num_constraints, meta.num_nonzeros) == (32, 27, 83)


@pytest.mark.parametrize("solver", [
    RevisedSimplexSolver(method="primal"),
    RevisedSimplexSolver(method="dual"),
    InteriorPointSolver(),
    InteriorPointSolver(crossover=False),
    QPInteriorPointSolver(),
])
def test_afiro_matches_netlib_optimum(solver):
    m = build_netlib_afiro()
    r = solver.solve(m)
    assert r.is_optimal
    assert abs(r.objective_value - AFIRO_OPTIMAL_OBJECTIVE) <= 1e-6 * abs(AFIRO_OPTIMAL_OBJECTIVE)
    cert = V.verify(m, r)
    if isinstance(solver, InteriorPointSolver) and not solver.do_crossover:
        # An interior solution on AFIRO's degenerate optimal face is optimal to ~1e-10 in objective, but its
        # reduced costs are not exactly complementary; exact KKT certificates require crossover (the default).
        assert cert.is_valid and cert.duality_gap is not None and cert.duality_gap <= 1e-6
    else:
        assert cert.status == "OPTIMAL_CERTIFIED"


def test_iteration_limit_is_not_reported_as_feasible():
    # v1 relabelled ITERATION_LIMIT as FEASIBLE.
    r = InteriorPointSolver(max_iterations=2, crossover=False).solve(build_netlib_afiro())
    assert r.status == SolverStatus.ITERATION_LIMIT
    assert not r.is_feasible


def test_ipm_certifies_infeasibility_and_unboundedness():
    inf = OptimizationModel()
    inf.add_variable("x", 0, 10)
    inf.add_variable("y", 0, 10)
    inf.add_constraint("c", {"x": 1, "y": 1}, ConstraintSense.GE, rhs=30)
    inf.set_objective({"x": 1})
    unb = OptimizationModel()
    unb.add_variable("x", 0)
    unb.add_variable("y", 0)
    unb.add_constraint("c", {"x": 1, "y": -1}, ConstraintSense.LE, rhs=3)
    unb.set_objective({"x": 1, "y": 1}, sense=ObjectiveSense.MAXIMIZE)
    for solver in (RevisedSimplexSolver(), InteriorPointSolver(), QPInteriorPointSolver(), ActiveSetQPSolver()):
        assert solver.solve(inf).status == SolverStatus.INFEASIBLE
        assert solver.solve(unb).status == SolverStatus.UNBOUNDED


def test_miqp_is_routed_to_branch_and_bound():
    # v1 sent MIQP to the simplex, silently dropping Q and integrality.
    m = build_portfolio_selection_model()
    out = solve_model(m)
    assert out.problem_class == "MIQP"
    assert out.algorithm == "branch_and_bound"
    assert out.result.is_optimal
    assert out.certificate.status == "OPTIMAL_CERTIFIED"
    selected = [v for v, val in out.result.primal_solution.items() if v.startswith("z_") and val > 0.5]
    assert 1 <= len(selected) <= 4


@pytest.mark.parametrize("solver", [QPInteriorPointSolver(), ActiveSetQPSolver()])
def test_refinery_qp_reaches_certified_optimum(solver):
    # v1 active set stopped at its iteration cap around -5.889e6.
    m = build_refinery_blending_model(as_qp=True)
    r = solver.solve(m)
    assert r.is_optimal
    assert abs(r.objective_value - REFINERY_QP_OPTIMUM) <= 1e-7 * abs(REFINERY_QP_OPTIMUM)
    assert V.verify(m, r).status == "OPTIMAL_CERTIFIED"


@pytest.mark.parametrize("periods,expected", [(4, 58210.0), (8, 111562.5), (24, 363372.9)])
def test_unit_commitment_is_solved_to_proven_optimality(periods, expected):
    # Reference optima cross-checked against the HiGHS MILP solver during development.
    m = build_unit_commitment_model(time_periods=periods)
    r = BranchAndBoundSolver().solve(m)
    assert r.is_optimal
    assert abs(r.objective_value - expected) <= 1e-6 * expected
    assert r.mip_gap is not None and r.mip_gap <= 1e-4
    assert V.verify(m, r).status == "OPTIMAL_CERTIFIED"


def test_unit_commitment_long_horizon_builds():
    # v1 crashed with IndexError for more than 4 periods.
    m = build_unit_commitment_model(time_periods=48)
    assert m.num_constraints == 48 * 9


@pytest.mark.parametrize("sense", [ObjectiveSense.MAXIMIZE, ObjectiveSense.MINIMIZE])
def test_knapsack_both_senses_match_brute_force(sense):
    w = [5, 4, 3, 7, 6, 2]
    v = [10, 40, 30, 50, 35, 8]
    m = OptimizationModel("Knapsack")
    for i in range(len(w)):
        m.add_variable(f"b{i}", 0, 1, VariableType.BINARY)
    if sense == ObjectiveSense.MAXIMIZE:
        m.add_constraint("cap", {f"b{i}": w[i] for i in range(len(w))}, ConstraintSense.LE, rhs=14)
    else:
        m.add_constraint("need", {f"b{i}": w[i] for i in range(len(w))}, ConstraintSense.GE, rhs=14)
    m.set_objective({f"b{i}": v[i] for i in range(len(w))}, sense=sense)

    best = None
    for bits in itertools.product((0, 1), repeat=len(w)):
        weight = sum(a * b for a, b in zip(w, bits))
        if (sense == ObjectiveSense.MAXIMIZE and weight <= 14) or (sense == ObjectiveSense.MINIMIZE and weight >= 14):
            val = sum(a * b for a, b in zip(v, bits))
            if best is None or (val > best if sense == ObjectiveSense.MAXIMIZE else val < best):
                best = val
    r = BranchAndBoundSolver().solve(m)
    assert r.is_optimal
    assert abs(r.objective_value - best) < 1e-9
    assert V.verify(m, r).status == "OPTIMAL_CERTIFIED"


def test_validator_detects_wrong_objective_and_unproven_optimality():
    m = OptimizationModel("Simple")
    m.add_variable("x1", 0)
    m.add_variable("x2", 0)
    m.add_constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LE, rhs=100.0)
    m.add_constraint("c2", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LE, rhs=80.0)
    m.set_objective({"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE)
    good = RevisedSimplexSolver().solve(m)
    assert V.verify(m, good).status == "OPTIMAL_CERTIFIED"

    wrong_obj = SolverResult(status=SolverStatus.OPTIMAL, objective_value=999.0, primal_solution=good.primal_solution,
                             dual_solution=good.dual_solution)
    assert V.verify(m, wrong_obj).status == "FAILED"

    # Feasible but suboptimal point with made-up duals must not be certified optimal.
    sub = SolverResult(status=SolverStatus.OPTIMAL, objective_value=30.0, primal_solution={"x1": 10.0, "x2": 0.0},
                       dual_solution={"c1": 0.0, "c2": 0.0})
    cert = V.verify(m, sub)
    assert cert.is_valid and cert.status == "FEASIBLE_CERTIFIED"
