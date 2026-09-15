"""
Basis factorization for the revised simplex method.

- Small bases (m <= dense_threshold): explicit inverse maintained by O(m^2) eta pivots.
- Large bases: sparse LU (SuperLU via scipy.sparse.linalg.splu) plus a product-form eta file.
Both refactorize periodically to bound error growth. Singular bases are repaired by
swapping dependent columns for slack columns of uncovered rows.
"""
from typing import List, Tuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
        self.etas: List[Tuple[int, np.ndarray]] = []
        self.num_updates = 0
        self.num_factorizations = 0

    # ----------------------------------------------------------------- factor
    def factorize(self, B: sp.csc_matrix):
        """Factorize basis matrix B. Raises SingularBasisError on (near) singular bases."""
        self.m = B.shape[0]
        self.num_updates = 0
        self.etas = []
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
            self.Binv = la.lu_solve((lu, piv), np.eye(self.m), check_finite=False)
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
        w = self.lu.solve(np.asarray(v, dtype=np.float64))
        for r, a in self.etas:
            wr = w[r] / a[r]
            if wr != 0.0:
                w -= a * wr
            w[r] = wr
        return w

    def btran(self, v: np.ndarray) -> np.ndarray:
        """Solve B^T w = v."""
        if self.m == 0:
            return np.zeros(0)
        if self.dense:
            return v @ self.Binv
        w = np.array(v, dtype=np.float64)
        for r, a in reversed(self.etas):
            w[r] = (w[r] - (a @ w - a[r] * w[r])) / a[r]
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
            row_r = self.Binv[r] / alpha[r]
            self.Binv -= np.outer(alpha, row_r)
            self.Binv[r] = row_r
        else:
            self.etas.append((r, np.array(alpha, dtype=np.float64)))
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
