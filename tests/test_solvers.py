"""
Comprehensive solver tests for Simplex, IPM, Branch & Bound, and Active Set QP.
"""
import pytest
from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver
from sovereign_opt.solvers.lp.interior_point import InteriorPointSolver
from sovereign_opt.solvers.milp.branch_bound import BranchAndBoundSolver
from sovereign_opt.solvers.qp.active_set import ActiveSetQPSolver
from sovereign_opt.validation.validator import IndependentValidator
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.power_dispatch import build_unit_commitment_model


def create_simple_lp():
    m = OptimizationModel("SimpleLP")
    m.add_variable("x1", lower_bound=0)
    m.add_variable("x2", lower_bound=0)
    m.add_constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LE, rhs=100.0)
    m.add_constraint("c2", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LE, rhs=80.0)
    m.add_constraint("c3", {"x1": 1.0}, ConstraintSense.LE, rhs=40.0)
    m.set_objective({"x1": 3.0, "x2": 2.0}, sense=ObjectiveSense.MAXIMIZE)
    return m


def test_revised_simplex_lp():
    m = create_simple_lp()
    solver = RevisedSimplexSolver()
    res = solver.solve(m)

    assert res.is_optimal
    assert abs(res.objective_value - 180.0) < 1e-4
    assert abs(res.primal_solution["x1"] - 20.0) < 1e-4
    assert abs(res.primal_solution["x2"] - 60.0) < 1e-4

    cert = IndependentValidator().verify(m, res)
    assert cert.is_valid


def test_interior_point_lp():
    m = create_simple_lp()
    solver = InteriorPointSolver()
    res = solver.solve(m)

    assert res.is_optimal
    assert abs(res.objective_value - 180.0) < 1e-3
    assert abs(res.primal_solution["x1"] - 20.0) < 1e-3
    assert abs(res.primal_solution["x2"] - 60.0) < 1e-3

    cert = IndependentValidator().verify(m, res)
    assert cert.is_valid


def test_branch_and_bound_milp():
    m = build_unit_commitment_model(time_periods=2)
    solver = BranchAndBoundSolver(max_nodes=100)
    res = solver.solve(m)

    assert res.is_feasible
    cert = IndependentValidator().verify(m, res)
    assert cert.is_valid
    assert cert.max_integrality_violation < 1e-4


def test_active_set_qp():
    m = build_refinery_blending_model(as_qp=True)
    solver = ActiveSetQPSolver()
    res = solver.solve(m)

    assert res.is_feasible
    cert = IndependentValidator().verify(m, res)
    assert cert.is_valid
