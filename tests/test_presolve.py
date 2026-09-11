"""
Unit tests for Presolve reductions and Postsolve reconstruction.
"""
from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.presolve.presolver import Presolver


def test_presolve_fixed_variables_and_singletons():
    m = OptimizationModel("PresolveTest")
    # Fixed variable: lb == ub
    m.add_variable("x_fixed", lower_bound=5.0, upper_bound=5.0)
    m.add_variable("x1", lower_bound=0.0, upper_bound=100.0)
    m.add_variable("x2", lower_bound=0.0, upper_bound=100.0)
    # Empty column variable
    m.add_variable("x_empty_col", lower_bound=2.0, upper_bound=10.0)

    # Multi-variable constraint: 2 * x_fixed + 3 * x1 + 4 * x2 <= 100
    m.add_constraint("c1", {"x_fixed": 2.0, "x1": 3.0, "x2": 4.0}, ConstraintSense.LE, rhs=100.0)
    
    # Singleton row on x1: 2 * x1 <= 20 -> x1 <= 10
    m.add_constraint("c_singleton", {"x1": 2.0}, ConstraintSense.LE, rhs=20.0)

    m.set_objective({"x_fixed": 1.0, "x1": 2.0, "x2": 5.0, "x_empty_col": 3.0}, sense=ObjectiveSense.MINIMIZE)

    presolver = Presolver()
    p_model, mapper, stats = presolver.presolve(m)

    assert stats.fixed_vars_count >= 1
    assert stats.singleton_rows_count >= 1
    assert stats.empty_cols_count >= 1

    # In p_model, x1 upper bound should be tightened to 10.0
    assert p_model.variables["x1"].upper_bound == 10.0

    # Test postsolve reconstruction
    presolved_sol = {"x1": 4.0, "x2": 10.0}
    x_full = mapper.restore_primal(presolved_sol)

    # Check restored values
    var_names = mapper.original_var_names
    assert x_full[var_names.index("x_fixed")] == 5.0
    assert x_full[var_names.index("x1")] == 4.0
    assert x_full[var_names.index("x2")] == 10.0
    assert x_full[var_names.index("x_empty_col")] == 2.0
