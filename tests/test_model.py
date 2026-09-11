"""
Unit tests for OptimizationModel, Variable, Constraint, and Objective.
"""
import pytest
import numpy as np
from sovereign_opt.model.model import OptimizationModel, ModelValidationError
from sovereign_opt.model.variable import Variable, VariableType
from sovereign_opt.model.constraint import Constraint, ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense


def test_model_construction():
    m = OptimizationModel("TestModel")
    v1 = m.add_variable("x1", lower_bound=0.0, upper_bound=10.0)
    v2 = m.add_variable("x2", lower_bound=-5.0, upper_bound=5.0, var_type=VariableType.INTEGER)
    v3 = m.add_variable("y", var_type=VariableType.BINARY)

    assert m.num_variables == 3
    assert v1.is_integer is False
    assert v2.is_integer is True
    assert v3.is_binary is True
    assert v3.lower_bound == 0.0
    assert v3.upper_bound == 1.0


def test_constraints_and_matrix_export():
    m = OptimizationModel("MatrixModel")
    m.add_variable("x1", lower_bound=0.0, upper_bound=10.0)
    m.add_variable("x2", lower_bound=0.0, upper_bound=10.0)

    m.add_constraint("c1", {"x1": 2.0, "x2": 1.0}, ConstraintSense.LE, rhs=20.0)
    m.add_constraint("c2", {"x1": 1.0, "x2": 3.0}, ConstraintSense.GE, rhs=15.0)

    assert m.num_constraints == 2
    A, row_lb, row_ub, col_lb, col_ub = m.to_matrix_form()

    assert A.shape == (2, 2)
    assert A[0, 0] == 2.0
    assert A[0, 1] == 1.0
    assert row_ub[0] == 20.0
    assert row_lb[1] == 15.0


def test_model_validation_errors():
    m = OptimizationModel("InvalidModel")
    # Contradicting bounds: lb > ub
    m.add_variable("x", lower_bound=10.0, upper_bound=5.0)
    with pytest.raises(ModelValidationError):
        m.validate()


def test_problem_classification():
    lp = OptimizationModel("LP")
    lp.add_variable("x1")
    assert lp.classify() == "LP"

    milp = OptimizationModel("MILP")
    milp.add_variable("x1", var_type=VariableType.INTEGER)
    assert milp.classify() == "MILP"

    qp = OptimizationModel("QP")
    qp.add_variable("x1")
    qp.set_objective(linear_coefficients={"x1": 1.0}, quadratic_coefficients={("x1", "x1"): 2.0})
    assert qp.classify() == "QP"
