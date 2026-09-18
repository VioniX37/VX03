"""
Sparse vertex polish: an approximate PDLP point in, an exact certified vertex out.

The polish is what lets the demo report the same objective as Gurobi / HiGHS to machine
precision, so every test compares against an exact reference -- and checks that a point the
polish cannot certify is never passed off as optimal.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from benchmarks.industrial.supply_chain import build_supply_chain_arrays, sizes_for
from benchmarks.netlib.afiro import build_netlib_afiro
from sovereign_opt.solvers.lp import pdlp as P
from sovereign_opt.solvers.lp.polish import pdlp_exact, polish_vertex

pytest.importorskip("numba")

AFIRO_OPT = -464.7531428571


def _afiro():
    m = build_netlib_afiro()
    A, rl, ru, cl, cu = m.to_matrix_form()
    return sp.csr_matrix(A), m.get_objective_vector(), rl, ru, cl, cu


def _highs_objective(A, c, rl, ru, cl, cu):
    highspy = pytest.importorskip("highspy")
    A = sp.csc_matrix(A)
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    L = highspy.HighsLp()
    L.num_col_, L.num_row_ = A.shape[1], A.shape[0]
    L.col_cost_ = np.asarray(c, dtype=np.float64)
    fix = lambda v: np.where(np.isposinf(v), inf, np.where(np.isneginf(v), -inf, v)).astype(np.float64)  # noqa: E731
    L.col_lower_, L.col_upper_, L.row_lower_, L.row_upper_ = fix(cl), fix(cu), fix(rl), fix(ru)
    L.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    L.a_matrix_.start_, L.a_matrix_.index_, L.a_matrix_.value_ = A.indptr, A.indices, A.data
    L.a_matrix_.num_col_, L.a_matrix_.num_row_ = A.shape[1], A.shape[0]
    h.passModel(L)
    h.run()
    assert h.modelStatusToString(h.getModelStatus()) == "Optimal"
    return h.getInfo().objective_function_value


def _rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def test_polish_turns_loose_pdlp_point_into_exact_afiro_vertex():
    lp = _afiro()
    r = P.pdlp(*lp, tol=1e-5, device="cpu")
    assert _rel(r.primal_objective, AFIRO_OPT) > 1e-9       # PDLP alone is visibly off
    p = polish_vertex(*lp, r.x, r.y)
    assert p.ok, p.reason
    assert _rel(p.primal_objective, AFIRO_OPT) < 1e-10
    assert abs(p.primal_objective - p.dual_objective) < 1e-9 * max(1.0, abs(AFIRO_OPT))


def test_pdlp_exact_matches_highs_on_supply_chain():
    lp = build_supply_chain_arrays(*sizes_for(5_000))
    args = (lp.A, lp.c, lp.row_lb, lp.row_ub, lp.col_lb, lp.col_ub)
    ref = _highs_objective(*args)
    r = pdlp_exact(*args)
    assert r.status == "optimal" and r.vertex
    assert _rel(r.objective, ref) < 1e-12
    x = r.x
    Ax = lp.A @ x
    assert np.all(Ax >= lp.row_lb - 1e-9 * (1 + np.abs(lp.row_lb)))
    assert np.all(Ax <= lp.row_ub + 1e-9 * (1 + np.abs(lp.row_ub)))
    assert np.all(x >= lp.col_lb - 1e-12)


def test_polish_never_certifies_a_point_far_from_optimal():
    lp = build_supply_chain_arrays(*sizes_for(5_000))
    args = (lp.A, lp.c, lp.row_lb, lp.row_ub, lp.col_lb, lp.col_ub)
    m, n = lp.A.shape
    p = polish_vertex(*args, np.zeros(n), np.zeros(m), max_steps=50, time_limit=10)
    if p.ok:  # allowed only if it genuinely found the optimum
        assert _rel(p.primal_objective, _highs_objective(*args)) < 1e-9
    else:
        assert p.reason != "certified"


def test_warm_start_resumes_instead_of_starting_over():
    lp = build_supply_chain_arrays(*sizes_for(5_000))
    args = (lp.A, lp.c, lp.row_lb, lp.row_ub, lp.col_lb, lp.col_ub)
    loose = P.pdlp(*args, tol=1e-4, device="cpu")
    cold = P.pdlp(*args, tol=1e-6, device="cpu")
    warm = P.pdlp(*args, tol=1e-6, device="cpu", x_init=loose.x, y_init=loose.y)
    assert cold.status == warm.status == "optimal"
    assert warm.iterations < cold.iterations
