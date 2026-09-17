"""
Basis factorization for the revised simplex method.

- Small bases (m <= dense_threshold): explicit inverse maintained by O(m^2) eta pivots.
- Large bases: sparse LU (SuperLU via scipy.sparse.linalg.splu) plus a product-form eta file.
Both refactorize periodically to bound error growth. Singular bases are repaired by
swapping dependent columns for slack columns of uncovered rows.

The eta file is stored sparsely (only the nonzeros of each update column) and applied by
Numba-compiled kernels when Numba is installed; otherwise by the equivalent NumPy code.
"""
from typing import List, Tuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:  # optional JIT compilation of the eta-file kernels
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap(args[0]) if args and callable(args[0]) else wrap


@njit(cache=True, nogil=True)
def _eta_ftran(w, rows, piv, ptr, idx, val, count):
    for k in range(count):
        r = rows[k]
        wr = w[r] / piv[k]
        if wr != 0.0:
            for t in range(ptr[k], ptr[k + 1]):
                w[idx[t]] -= val[t] * wr
        w[r] = wr


@njit(cache=True, nogil=True)
def _eta_btran(w, rows, piv, ptr, idx, val, count):
    for k in range(count - 1, -1, -1):
        r = rows[k]
        acc = 0.0
        for t in range(ptr[k], ptr[k + 1]):
            acc += val[t] * w[idx[t]]
        w[r] = (w[r] - acc) / piv[k]


@njit(cache=True, nogil=True)
def _dense_update(Binv, r, alpha):
    """In-place Binv <- E Binv for a column replacement at position r (no m x m temporaries)."""
    m = Binv.shape[0]
    piv = alpha[r]
    for j in range(m):
        Binv[r, j] /= piv
    for i in range(m):
        a = alpha[i]
        if i != r and a != 0.0:
            for j in range(m):
                Binv[i, j] -= a * Binv[r, j]


class _EtaFile:
    """Sparse product-form update file: eta k = (row r_k, pivot a_k[r_k], off-pivot nonzeros)."""

    def __init__(self, m: int, capacity: int):
        self.rows = np.zeros(capacity, dtype=np.int64)
        self.piv = np.zeros(capacity)
        self.ptr = np.zeros(capacity + 1, dtype=np.int64)
        self.idx = np.zeros(max(16, 4 * m), dtype=np.int64)
        self.val = np.zeros(max(16, 4 * m))
        self.count = 0

    def append(self, r: int, alpha: np.ndarray):
        nz = np.flatnonzero(alpha)
        nz = nz[nz != r]
        k = self.count
        if k + 1 >= len(self.rows):
            grow = len(self.rows)
            self.rows = np.concatenate([self.rows, np.zeros(grow, dtype=np.int64)])
            self.piv = np.concatenate([self.piv, np.zeros(grow)])
            self.ptr = np.concatenate([self.ptr, np.zeros(grow, dtype=np.int64)])
        start = self.ptr[k]
        end = start + len(nz)
        if end > len(self.idx):
            size = max(end, 2 * len(self.idx))
            self.idx = np.concatenate([self.idx, np.zeros(size - len(self.idx), dtype=np.int64)])
            self.val = np.concatenate([self.val, np.zeros(size - len(self.val))])
        self.idx[start:end] = nz
        self.val[start:end] = alpha[nz]
        self.rows[k] = r
        self.piv[k] = alpha[r]
        self.ptr[k + 1] = end
        self.count += 1

    def ftran(self, w: np.ndarray):
        if HAVE_NUMBA:
            _eta_ftran(w, self.rows, self.piv, self.ptr, self.idx, self.val, self.count)
            return
        for k in range(self.count):
            r = self.rows[k]
            wr = w[r] / self.piv[k]
            if wr != 0.0:
                s, e = self.ptr[k], self.ptr[k + 1]
                w[self.idx[s:e]] -= self.val[s:e] * wr
            w[r] = wr

    def btran(self, w: np.ndarray):
        if HAVE_NUMBA:
            _eta_btran(w, self.rows, self.piv, self.ptr, self.idx, self.val, self.count)
            return
        for k in range(self.count - 1, -1, -1):
            r = self.rows[k]
            s, e = self.ptr[k], self.ptr[k + 1]
            w[r] = (w[r] - self.val[s:e] @ w[self.idx[s:e]]) / self.piv[k]


class SingularBasisError(Exception):
    pass


class BasisFactor:
    def __init__(self, dense_threshold: int = 600, refactor_frequency: int = 100):
        self.dense_threshold = dense_threshold
        self.refactor_frequency = refactor_frequency
        self.m = 0
        self.dense = True
        self.Binv: np.ndarray = np.zeros((0, 0))
        self.lu = None
        self.etas = _EtaFile(0, refactor_frequency + 1)
        self.num_updates = 0
        self.num_factorizations = 0

    # ----------------------------------------------------------------- factor
    def factorize(self, B: sp.csc_matrix):
        """Factorize basis matrix B. Raises SingularBasisError on (near) singular bases."""
        self.m = B.shape[0]
        self.num_updates = 0
        self.etas = _EtaFile(self.m, self.refactor_frequency + 1)
        self.num_factorizations += 1
        if self.m == 0:
            self.dense = True
            self.Binv = np.zeros((0, 0))
            return
        self.dense = self.m <= self.dense_threshold
        if self.dense:
            Bd = B.toarray()
            try:
                lu, piv = la.lu_factor(Bd, check_finite=False)
            except (ValueError, la.LinAlgError) as exc:
                raise SingularBasisError(str(exc))
            diag = np.abs(np.diag(lu))
            if diag.min() <= 1e-11 * max(1.0, diag.max()):
                raise SingularBasisError("near-singular basis")
            self.Binv = np.ascontiguousarray(la.lu_solve((lu, piv), np.eye(self.m), check_finite=False))
        else:
            try:
                self.lu = spla.splu(sp.csc_matrix(B), permc_spec="COLAMD", diag_pivot_thresh=0.1)
            except RuntimeError as exc:
                raise SingularBasisError(str(exc))
            udiag = np.abs(self.lu.U.diagonal())
            if udiag.min() <= 1e-11 * max(1.0, udiag.max()):
                raise SingularBasisError("near-singular basis")

    @property
    def needs_refactor(self) -> bool:
        return self.num_updates >= self.refactor_frequency

    # ------------------------------------------------------------ solves
    def ftran(self, v: np.ndarray) -> np.ndarray:
        """Solve B w = v."""
        if self.m == 0:
            return np.zeros(0)
        if self.dense:
            return self.Binv @ v
        w = np.ascontiguousarray(self.lu.solve(np.asarray(v, dtype=np.float64)))
        self.etas.ftran(w)
        return w

    def btran(self, v: np.ndarray) -> np.ndarray:
        """Solve B^T w = v."""
        if self.m == 0:
            return np.zeros(0)
        if self.dense:
            return v @ self.Binv
        w = np.array(v, dtype=np.float64)
        self.etas.btran(w)
        return self.lu.solve(w, trans="T")

    def row_of_inverse(self, r: int) -> np.ndarray:
        """Row r of B^{-1} (= btran(e_r))."""
        if self.dense:
            return self.Binv[r].copy()
        e = np.zeros(self.m)
        e[r] = 1.0
        return self.btran(e)

    # ----------------------------------------------------------- update
    def update(self, r: int, alpha: np.ndarray):
        """Basis change: column in position r replaced; alpha = B_old^{-1} a_entering."""
        if self.dense:
            if HAVE_NUMBA:
                _dense_update(self.Binv, int(r), np.ascontiguousarray(alpha, dtype=np.float64))
            else:
                row_r = self.Binv[r] / alpha[r]
                self.Binv -= np.outer(alpha, row_r)
                self.Binv[r] = row_r
        else:
            self.etas.append(r, np.asarray(alpha, dtype=np.float64))
        self.num_updates += 1


def repair_basis(A: sp.csc_matrix, head: np.ndarray, slack_col_of_row: np.ndarray, tol: float = 1e-9):
    """
    Replace linearly dependent basic columns by slack columns of rows they fail to cover.
    Returns (new_head, list of (position, removed_column, inserted_column)).
    """
    m = A.shape[0]
    B = A[:, head].toarray()
    _, Rq, piv = la.qr(B, pivoting=True, mode="economic", check_finite=False)
    diag = np.abs(np.diag(Rq))
    if diag.size == 0:
        return head.copy(), []
    rank = int(np.sum(diag > tol * max(1.0, diag[0])))
    indep_pos = piv[:rank]
    dep_pos = piv[rank:]
    if len(dep_pos) == 0:
        return head.copy(), []
    Bind = B[:, indep_pos]
    P, _, _ = la.lu(Bind, check_finite=False)
    covered = set(int(np.argmax(P[:, k])) for k in range(rank))
    uncovered = [i for i in range(m) if i not in covered]
    new_head = head.copy()
    swaps = []
    for pos, row in zip(dep_pos, uncovered):
        swaps.append((int(pos), int(head[pos]), int(slack_col_of_row[row])))
        new_head[pos] = slack_col_of_row[row]
    return new_head, swaps
