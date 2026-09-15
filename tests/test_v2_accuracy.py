"""
Randomized accuracy tests for the v2 solvers.

Independent references: brute-force enumeration, closed-form KKT solutions, cross-checks
between different sovereign algorithms, and (test-only) the HiGHS solver shipped with SciPy.
The engine itself never calls an external solver.
"""
import itertools

import numpy as np
import pytest

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.model.variable import VariableType
from sovereign_opt.parsers.mps_parser import MPSParser
from sovereign_opt.solvers.base import SolverStatus
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver
from sovereign_opt.solvers.lp.interior_point import InteriorPointSolver
from sovereign_opt.solvers.milp.branch_bound import BranchAndBoundSolver
from sovereign_opt.solvers.qp.active_set import ActiveSetQPSolver
from sovereign_opt.solvers.qp.interior_point_qp import QPInteriorPointSolver
from sovereign_opt.solvers.dispatch import solve_model
from sovereign_opt.validation.validator import IndependentValidator

V = IndependentValidator()


def _random_lp(rng, boxed=False, infeasible=False):
    n, m = int(rng.integers(3, 25)), int(rng.integers(2, 20))
    A = rng.normal(size=(m, n)) * (rng.random((m, n)) < 0.5)
    x0 = rng.uniform(-5, 5, n)
    lb = x0 - rng.uniform(0, 5, n)
    ub = x0 + rng.uniform(0, 5, n)
    if not boxed:
        lb = np.where(rng.random(n) < 0.2, -np.inf, lb)
        ub = np.where(rng.random(n) < 0.3, np.inf, ub)
    act = A @ x0
    rl = np.where(rng.random(m) < 0.4, -np.inf, act - rng.uniform(0, 3, m))
    ru = np.where(rng.random(m) < 0.4, np.inf, act + rng.uniform(0, 3, m))
    eq = rng.random(m) < 0.15
    rl[eq], ru[eq] = act[eq], act[eq]
    if infeasible:
        k = int(rng.integers(m))
        rl[k], ru[k] = act[k] + 50.0 + np.abs(A[k]).sum() * 20, np.inf
    c = rng.normal(size=n)
    sense = ObjectiveSense.MAXIMIZE if rng.random() < 0.5 else ObjectiveSense.MINIMIZE
    model = OptimizationModel("RandomLP")
    for j in range(n):
        model.add_variable(f"x{j}", lb[j], ub[j])
    for i in range(m):
        model.add_constraint(f"r{i}", {f"x{j}": A[i, j] for j in range(n) if A[i, j] != 0},
                             ConstraintSense.RANGE, lower_bound=rl[i], upper_bound=ru[i])
    model.set_objective({f"x{j}": c[j] for j in range(n)}, sense=sense)
    return model


def test_simplex_and_ipm_agree_on_random_boxed_lps():
    rng = np.random.default_rng(101)
    for _ in range(25):
        m = _random_lp(rng, boxed=True)
        a = RevisedSimplexSolver().solve(m)
        b = InteriorPointSolver().solve(m)
        assert a.status == b.status
        if a.is_optimal:
            assert abs(a.objective_value - b.objective_value) <= 1e-6 * max(1.0, abs(a.objective_value))
            assert V.verify(m, a).status == "OPTIMAL_CERTIFIED"
            assert V.verify(m, b).status == "OPTIMAL_CERTIFIED"


def test_random_lps_against_highs_reference():
    linprog = pytest.importorskip("scipy.optimize").linprog
    rng = np.random.default_rng(202)
    for trial in range(40):
        m = _random_lp(rng, infeasible=(trial % 4 == 0))
        A, rl, ru, cl, cu = m.to_matrix_form()
        A = A.toarray()
        c = m.get_objective_vector()
        A_ub = np.vstack([A[np.isfinite(ru)], -A[np.isfinite(rl)]])
        b_ub = np.concatenate([ru[np.isfinite(ru)], -rl[np.isfinite(rl)]])
        bounds = [(None if np.isinf(l) else l, None if np.isinf(u) else u) for l, u in zip(cl, cu)]
        ref = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        expected = {0: SolverStatus.OPTIMAL, 2: SolverStatus.INFEASIBLE, 3: SolverStatus.UNBOUNDED}[ref.status]
        for solver in (RevisedSimplexSolver(), InteriorPointSolver()):
            r = solver.solve(m)
            assert r.status == expected
            if expected == SolverStatus.OPTIMAL:
                ref_obj = ref.fun if m.objective.sense == ObjectiveSense.MINIMIZE else -ref.fun
                assert abs(r.objective_value - ref_obj) <= 1e-6 * max(1.0, abs(ref_obj))


def _brute_force(model):
    names = model.variable_names
    A, rl, ru, cl, cu = model.to_matrix_form()
    A = A.toarray()
    c = np.array([model.objective.linear_coefficients.get(v, 0.0) for v in names])
    maximize = model.objective.sense == ObjectiveSense.MAXIMIZE
    best = None
    for point in itertools.product(*[range(int(cl[j]), int(cu[j]) + 1) for j in range(len(names))]):
        x = np.array(point, dtype=float)
        act = A @ x
        if np.all(act >= rl - 1e-9) and np.all(act <= ru + 1e-9):
            val = float(c @ x)
            if best is None or (val > best if maximize else val < best):
                best = val
    return best


def test_branch_and_cut_matches_brute_force_enumeration():
    rng = np.random.default_rng(303)
    for trial in range(30):
        n, m = int(rng.integers(2, 6)), int(rng.integers(1, 5))
        A = rng.integers(-4, 5, size=(m, n)).astype(float)
        x0 = rng.integers(-2, 4, size=n)
        act = A @ x0
        rl = np.where(rng.random(m) < 0.5, -np.inf, act - rng.integers(0, 4, m))
        ru = np.where(rng.random(m) < 0.5, np.inf, act + rng.integers(0, 4, m))
        if trial % 6 == 0:
            rl[0], ru[0] = np.abs(A[0]).sum() * 4 + 1, np.inf
        model = OptimizationModel("IntBrute")
        for j in range(n):
            model.add_variable(f"x{j}", -2, 4, VariableType.INTEGER)
        for i in range(m):
            model.add_constraint(f"r{i}", {f"x{j}": A[i, j] for j in range(n) if A[i, j] != 0},
                                 ConstraintSense.RANGE, lower_bound=rl[i], upper_bound=ru[i])
        model.set_objective({f"x{j}": float(rng.integers(-5, 6)) for j in range(n)},
                            sense=ObjectiveSense.MAXIMIZE if trial % 2 else ObjectiveSense.MINIMIZE)
        best = _brute_force(model)
        r = BranchAndBoundSolver().solve(model)
        if best is None:
            assert r.status == SolverStatus.INFEASIBLE
        else:
            assert r.is_optimal
            assert abs(r.objective_value - best) < 1e-9
            assert V.verify(model, r).status == "OPTIMAL_CERTIFIED"


def test_equality_constrained_qp_matches_closed_form_kkt():
    rng = np.random.default_rng(404)
    for _ in range(10):
        n, k = int(rng.integers(3, 12)), int(rng.integers(1, 3))
        M = rng.normal(size=(n, n))
        Q = M @ M.T + np.eye(n)
        c = rng.normal(size=n)
        A = rng.normal(size=(k, n))
        b = rng.normal(size=k)
        KKT = np.block([[Q, -A.T], [A, np.zeros((k, k))]])
        sol = np.linalg.solve(KKT, np.concatenate([-c, b]))
        x_star, y_star = sol[:n], sol[n:]
        f_star = 0.5 * x_star @ Q @ x_star + c @ x_star

        model = OptimizationModel("EqQP")
        for j in range(n):
            model.add_variable(f"x{j}", float("-inf"), float("inf"))
        for i in range(k):
            model.add_constraint(f"e{i}", {f"x{j}": A[i, j] for j in range(n)}, ConstraintSense.EQ, rhs=b[i])
        quad = {(f"x{i}", f"x{j}"): (Q[i, j] if i == j else 2 * Q[i, j]) for i in range(n) for j in range(i, n)}
        model.set_objective({f"x{j}": c[j] for j in range(n)}, quadratic_coefficients=quad)
        for solver in (QPInteriorPointSolver(), ActiveSetQPSolver()):
            r = solver.solve(model)
            assert r.is_optimal
            x = np.array([r.primal_solution[f"x{j}"] for j in range(n)])
            y = np.array([r.dual_solution[f"e{i}"] for i in range(k)])
            assert abs(r.objective_value - f_star) <= 1e-7 * max(1.0, abs(f_star))
            assert np.allclose(x, x_star, atol=1e-6)
            assert np.allclose(y, y_star, atol=1e-5)
            assert V.verify(model, r).status == "OPTIMAL_CERTIFIED"


def test_presolve_round_trip_matches_direct_solve():
    rng = np.random.default_rng(505)
    for trial in range(20):
        model = _random_lp(rng, boxed=True)
        if trial % 2:
            for v in model.variable_names[:3]:
                var = model.variables[v]
                var.var_type = VariableType.INTEGER
                var.lower_bound, var.upper_bound = np.floor(var.lower_bound), np.ceil(var.upper_bound)
        direct = solve_model(model, enable_presolve=False)
        reduced = solve_model(model, enable_presolve=True)
        assert direct.result.status == reduced.result.status
        if direct.result.is_optimal:
            assert abs(direct.result.objective_value - reduced.result.objective_value) <= 1e-6 * max(1.0, abs(direct.result.objective_value))
            assert reduced.certificate.status == "OPTIMAL_CERTIFIED"


def test_mps_parser_v2_features():
    text = """NAME          TESTQP
OBJSENSE
    MAX
ROWS
 N  obj
 L  c1
 G  c2
COLUMNS
    x         obj       1.0        c1        1.0
    x         c2        1.0
    y         obj       2.0        c1        1.0
RHS
    c1        4.0       obj       -3.0
    c2        1.0
RANGES
    rng       c2        2.0
BOUNDS
 UP bnd       y         -1.0
QUADOBJ
    x         x         -2.0
    x         y         -1.0
ENDATA
"""
    m = MPSParser.parse_string(text)
    assert m.objective.sense == ObjectiveSense.MAXIMIZE
    assert m.objective.offset == 3.0
    assert m.constraints["c1"].upper_bound == 4.0
    assert (m.constraints["c2"].lower_bound, m.constraints["c2"].upper_bound) == (1.0, 3.0)
    assert m.variables["y"].lower_bound == float("-inf") and m.variables["y"].upper_bound == -1.0
    assert m.objective.quadratic_coefficients[("x", "x")] == -2.0
    assert m.objective.quadratic_coefficients[("x", "y")] == -2.0
