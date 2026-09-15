"""
Unit tests for Presolve reductions and Postsolve reconstruction.
"""
import pytest

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.presolve.presolver import Presolver, PresolveInfeasibleError
from sovereign_opt.solvers.dispatch import solve_model


def _base_model():
    m = OptimizationModel("PresolveTest")
    m.add_variable("x_fixed", lower_bound=5.0, upper_bound=5.0)
    m.add_variable("x1", lower_bound=0.0, upper_bound=100.0)
    m.add_variable("x2", lower_bound=0.0, upper_bound=100.0)
    m.add_variable("x_empty_col", lower_bound=2.0, upper_bound=10.0)
    # 2 * x_fixed + 3 * x1 + 4 * x2 <= 100
    m.add_constraint("c1", {"x_fixed": 2.0, "x1": 3.0, "x2": 4.0}, ConstraintSense.LE, rhs=100.0)
    # Singleton row on x1: 2 * x1 <= 20 -> x1 <= 10
    m.add_constraint("c_singleton", {"x1": 2.0}, ConstraintSense.LE, rhs=20.0)
    m.set_objective({"x_fixed": 1.0, "x1": 2.0, "x2": 5.0, "x_empty_col": 3.0}, sense=ObjectiveSense.MINIMIZE)
    return m


def test_presolve_fixed_variables_and_singletons():
    # Dual fixing disabled so the singleton bound tightening on x1 stays observable.
    p_model, mapper, stats = Presolver(enable_dual_fixing=False).presolve(_base_model())

    assert stats.fixed_vars_count >= 1
    assert stats.singleton_rows_count >= 1
    assert stats.empty_cols_count >= 1
    assert p_model.variables["x1"].upper_bound == 10.0

    x_full = mapper.restore_primal({"x1": 4.0, "x2": 10.0})
    names = mapper.original_var_names
    assert x_full[names.index("x_fixed")] == 5.0
    assert x_full[names.index("x1")] == 4.0
    assert x_full[names.index("x2")] == 10.0
    assert x_full[names.index("x_empty_col")] == 2.0


def test_presolve_v2_dual_fixing_solves_model_completely():
    p_model, mapper, stats = Presolver().presolve(_base_model())
    # x1 and x2 have positive cost and only appear in a <= row with positive coefficients: fixed at lower bound.
    assert stats.dominated_cols_count >= 2
    assert p_model.num_variables == 0
    values = mapper.restore_dict({})
    assert values == {"x_fixed": 5.0, "x1": 0.0, "x2": 0.0, "x_empty_col": 2.0}


def test_presolve_maximize_empty_column_regression():
    # v1 fixed both variables at 0 because it ignored the objective sense.
    m = OptimizationModel()
    m.add_variable("x", 0, 10)
    m.add_variable("y", 0, 5)
    m.add_constraint("c", {"y": 1}, ConstraintSense.LE, rhs=5)
    m.set_objective({"x": 1, "y": 1}, sense=ObjectiveSense.MAXIMIZE)
    _, mapper, _ = Presolver().presolve(m)
    assert mapper.restore_dict({}) == {"x": 10.0, "y": 5.0}


def test_presolve_doubleton_aggregation_round_trip():
    m = OptimizationModel("Doubleton")
    for v in ("x", "y", "z"):
        m.add_variable(v, 0.0, 10.0)
    m.add_constraint("link", {"x": 1.0, "y": 2.0}, ConstraintSense.EQ, rhs=4.0)
    m.add_constraint("cover", {"y": 1.0, "z": 1.0}, ConstraintSense.GE, rhs=1.0)
    m.set_objective({"x": 1.0, "y": 2.0, "z": 1.0})
    _, _, stats = Presolver().presolve(m)
    assert stats.doubleton_eliminations >= 1

    with_presolve = solve_model(m, enable_presolve=True)
    without = solve_model(m, enable_presolve=False)
    assert with_presolve.result.is_optimal and without.result.is_optimal
    assert abs(with_presolve.result.objective_value - 4.0) < 1e-9
    assert abs(without.result.objective_value - 4.0) < 1e-9
    assert with_presolve.certificate.status == "OPTIMAL_CERTIFIED"


def test_presolve_parallel_rows_and_duplicate_columns():
    m = OptimizationModel("ParallelDuplicate")
    m.add_variable("a", 0.0, 5.0)
    m.add_variable("b", 0.0, 5.0)
    m.add_variable("b_copy", 0.0, 3.0)
    m.add_constraint("r1", {"a": 1.0, "b": 2.0, "b_copy": 4.0}, ConstraintSense.GE, rhs=3.0)
    m.add_constraint("r1_scaled", {"a": 2.0, "b": 4.0, "b_copy": 8.0}, ConstraintSense.LE, rhs=20.0)
    m.set_objective({"a": 3.0, "b": 1.0, "b_copy": 2.0})
    _, _, stats = Presolver().presolve(m)
    assert stats.parallel_rows_count >= 1
    assert stats.duplicate_cols_count >= 1

    direct = solve_model(m, enable_presolve=False).result
    reduced = solve_model(m, enable_presolve=True)
    assert abs(reduced.result.objective_value - direct.objective_value) < 1e-9
    assert reduced.certificate.status == "OPTIMAL_CERTIFIED"


def test_presolve_detects_infeasible_activity():
    m = OptimizationModel("Infeasible")
    m.add_variable("x", 0.0, 1.0)
    m.add_variable("y", 0.0, 1.0)
    m.add_constraint("too_big", {"x": 1.0, "y": 1.0}, ConstraintSense.GE, rhs=5.0)
    m.set_objective({"x": 1.0})
    with pytest.raises(PresolveInfeasibleError):
        Presolver().presolve(m)
