"""
Halpern PDLP and its fused CPU kernels.

The fused kernels are a pure performance change, so each is checked against the plain
formulation it replaces: same iterate, same preconditioning, same answer on every backend.
"""
import copy

import numpy as np
import pytest
import scipy.sparse as sp

from benchmarks.netlib.afiro import build_netlib_afiro
from sovereign_opt.solvers.lp import pdlp as P

pytest.importorskip("numba")
from sovereign_opt.solvers.lp import pdhg_kernels as K  # noqa: E402

AFIRO_OPT = -464.7531428571


def _afiro():
    m = build_netlib_afiro()
    A, rl, ru, cl, cu = m.to_matrix_form()
    return sp.csr_matrix(A), m.get_objective_vector(), rl, ru, cl, cu


def _random_lp(m=120, n=300, seed=3):
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=0.05, random_state=seed, format="csr") * 5
    A[0, :] = 1.0                                   # one dense row, as in the supply-chain model
    A = sp.csr_matrix(A)
    x_feas = rng.uniform(0, 2, n)
    Ax = A @ x_feas
    return A, rng.uniform(-1, 1, n), Ax - 1.0, Ax + 1.0, np.zeros(n), np.full(n, 5.0)


@pytest.mark.parametrize("method", ["halpern", "adaptive"])
@pytest.mark.parametrize("device", ["cpu", "numpy"])
def test_reaches_netlib_optimum(method, device):
    r = P.pdlp(*_afiro(), tol=1e-8, device=device, method=method)
    assert r.status == "optimal"
    assert abs(r.primal_objective - AFIRO_OPT) / abs(AFIRO_OPT) < 1e-6


def test_fused_backend_matches_array_backend():
    A, c, rl, ru, cl, cu = _random_lp()
    a = P.pdlp(A, c, rl, ru, cl, cu, tol=1e-8, device="cpu")
    b = P.pdlp(A, c, rl, ru, cl, cu, tol=1e-8, device="numpy")
    assert a.status == b.status == "optimal"
    assert a.device.startswith("cpu-fused") and b.device == "cpu-numpy"
    assert abs(a.primal_objective - b.primal_objective) <= 1e-6 * max(1.0, abs(a.primal_objective))


def _engine(kern_set):
    A, c, rl, ru, cl, cu = _random_lp()
    Ks, Dr, Dc = P._precondition_fused(A, K)
    eng = P._FusedCPUEngine(Ks, Ks.T.tocsr(), c * Dc, rl * Dr, ru * Dr, cl / Dc, cu / Dc,
                            np.clip(np.zeros(A.shape[1]), cl / Dc, cu / Dc), np.zeros(A.shape[0]), K)
    eng.k = kern_set
    return eng


def _advance(eng, steps, tau, sigma):
    """A few unfused iterations so the state is not trivially zero."""
    for i in range(steps):
        eng.step(tau, sigma)
        eng.halpern((i + 1.0) / (i + 2.0), 1.0 / (i + 2.0), 1.0)


def test_fused_iteration_equals_step_plus_halpern():
    tau, sigma, a, b, g = 0.05, 0.08, 0.75, 0.25, 1.0
    base = _engine(K.PARALLEL)
    _advance(base, 5, tau, sigma)
    plain, fused = copy.deepcopy(base), copy.deepcopy(base)

    dx2, dy2, inter = plain.step(tau, sigma)
    plain.halpern(a, b, g)
    dx2_next = plain.primal_only(tau)

    f_dx2 = fused.primal_only(tau)
    f_dy2, f_inter, f_dx2_next = fused.fused_step(tau, sigma, a, b, g)

    for name in ("x", "y", "Kx", "KTy", "xp", "yp", "Kxp"):
        np.testing.assert_allclose(getattr(fused, name), getattr(plain, name), rtol=1e-12, atol=1e-12,
                                   err_msg=name)
    np.testing.assert_allclose([f_dx2, f_dy2, f_inter, f_dx2_next], [dx2, dy2, inter, dx2_next],
                               rtol=1e-10, atol=1e-14)


def test_serial_and_parallel_kernels_agree():
    tau, sigma = 0.05, 0.08
    par, ser = _engine(K.PARALLEL), _engine(K.SERIAL)
    _advance(par, 7, tau, sigma)
    _advance(ser, 7, tau, sigma)
    for name in ("x", "y", "Kx", "KTy"):
        np.testing.assert_allclose(getattr(par, name), getattr(ser, name), rtol=1e-12, atol=1e-12)


def test_fused_preconditioning_matches_reference():
    A = _random_lp()[0]
    a, b = P._precondition(A), P._precondition_fused(A, K)
    assert abs(a[0] - b[0]).max() < 1e-12
    np.testing.assert_allclose(a[1], b[1], rtol=1e-12)
    np.testing.assert_allclose(a[2], b[2], rtol=1e-12)


def test_nnz_blocks_cover_every_row_and_balance_dense_rows():
    # one row with 90% of the nonzeros: row-count splitting would give it all to one thread
    rows = [np.arange(9000)] + [np.array([i]) for i in range(1, 1001)]
    indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    blocks = K.nnz_blocks(indptr, threads=4)
    assert blocks[0] == 0 and blocks[-1] == len(rows)
    assert np.all(np.diff(blocks) > 0)
    heavy = max(indptr[blocks[i + 1]] - indptr[blocks[i]] for i in range(len(blocks) - 1))
    assert heavy <= 9000  # the dense row stands alone; nothing is piled on top of it
