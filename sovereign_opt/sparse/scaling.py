"""
Matrix scaling techniques for mathematical optimization.
- Ruiz Equilibration scaling (Ruiz 2001) for sparse matrices.
- Geometric-mean + equilibration scaling with power-of-two rounding (used by the v2 solvers).
"""
from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp
from sovereign_opt.sparse.matrix import SparseMatrix


def _row_max_min(B: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """Row-wise max and min of |entries| over the nonzero pattern (0 for empty rows)."""
    m = B.shape[0]
    rmax = np.zeros(m)
    rmin = np.zeros(m)
    counts = np.diff(B.indptr)
    nonempty = counts > 0
    if B.nnz == 0:
        return rmax, rmin
    starts = B.indptr[:-1][nonempty]
    rmax[nonempty] = np.maximum.reduceat(B.data, starts)
    rmin[nonempty] = np.minimum.reduceat(B.data, starts)
    return rmax, rmin


def geometric_scaling(
    A: sp.spmatrix,
    passes: int = 6,
    fixed_cols: Optional[np.ndarray] = None,
    round_power_of_two: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute row scale R and column scale C such that R * A * C has entries clustered around 1.

    Geometric-mean passes (1 / sqrt(max * min)) are followed by one infinity-norm
    equilibration pass. Scale factors are rounded to powers of two so scaling introduces
    no floating-point rounding error. Columns flagged in `fixed_cols` keep scale 1
    (used to preserve integrality of integer variables).
    """
    m, n = A.shape
    R = np.ones(m)
    C = np.ones(n)
    B = abs(sp.csr_matrix(A, dtype=np.float64))
    B.eliminate_zeros()
    if B.nnz == 0:
        return R, C
    fixed = np.zeros(n, dtype=bool) if fixed_cols is None else np.asarray(fixed_cols, dtype=bool)

    for _ in range(passes):
        rmax, rmin = _row_max_min(B)
        r = np.divide(1.0, np.sqrt(np.maximum(rmax * rmin, 1e-300)), out=np.ones_like(rmax), where=rmax > 0)
        B = sp.csr_matrix(sp.diags(r) @ B)
        R *= r

        Bc = sp.csr_matrix(B.T)
        cmax, cmin = _row_max_min(Bc)
        cc = np.divide(1.0, np.sqrt(np.maximum(cmax * cmin, 1e-300)), out=np.ones_like(cmax), where=cmax > 0)
        cc[fixed] = 1.0
        B = sp.csr_matrix(B @ sp.diags(cc))
        C *= cc

    # Final equilibration: row infinity norms to 1, then column infinity norms to 1
    rmax, _ = _row_max_min(B)
    r = np.divide(1.0, rmax, out=np.ones_like(rmax), where=rmax > 0)
    B = sp.csr_matrix(sp.diags(r) @ B)
    R *= r
    cmax, _ = _row_max_min(sp.csr_matrix(B.T))
    cc = np.divide(1.0, cmax, out=np.ones_like(cmax), where=cmax > 0)
    cc[fixed] = 1.0
    C *= cc

    if round_power_of_two:
        R = np.power(2.0, np.round(np.log2(R)))
        C = np.power(2.0, np.round(np.log2(C)))
        C[fixed] = 1.0
    return R, C


class RuizScaling:
    """
    Ruiz Equilibration:
    Computes diagonal matrices D1 and D2 such that:
        A_scaled = D1 * A * D2
    has row and column infinity norms approximately equal to 1.0.

    Transformation properties:
    - If Ax = b, then (D1 * A * D2) * (D2^-1 * x) = D1 * b
    - If min c^T x, then min (D2 * c)^T (D2^-1 * x)
    - Primal variable recovery: x = D2 * x_scaled
    - Dual variable recovery: y = D1 * y_scaled
    """

    def __init__(self, max_iterations: int = 10, tolerance: float = 1e-3):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.iterations_run = 0
        self.d1: np.ndarray = np.array([])
        self.d2: np.ndarray = np.array([])

    def fit_transform(
        self,
        A: SparseMatrix,
        c: np.ndarray,
        row_lb: np.ndarray,
        row_ub: np.ndarray,
        col_lb: np.ndarray,
        col_ub: np.ndarray,
    ) -> Tuple[SparseMatrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale the constraint matrix, objective, and bounds.
        """
        m, n = A.shape
        self.d1 = np.ones(m, dtype=np.float64)
        self.d2 = np.ones(n, dtype=np.float64)
        self.iterations_run = 0

        current_A = A

        for _ in range(self.max_iterations):
            r_norms = current_A.row_inf_norms()
            c_norms = current_A.col_inf_norms()

            # Check convergence: max deviation from 1 over non-empty rows/cols
            max_r_dev = np.max(np.abs(r_norms[r_norms > 0] - 1.0), initial=0.0)
            max_c_dev = np.max(np.abs(c_norms[c_norms > 0] - 1.0), initial=0.0)
            if max_r_dev < self.tolerance and max_c_dev < self.tolerance:
                break

            dr = 1.0 / np.where(r_norms > 1e-12, np.sqrt(r_norms), 1.0)
            dc = 1.0 / np.where(c_norms > 1e-12, np.sqrt(c_norms), 1.0)

            self.d1 *= dr
            self.d2 *= dc
            current_A = current_A.scale_rows_cols(dr, dc)
            self.iterations_run += 1

        # Scale vectors
        scaled_c = c * self.d2
        scaled_row_lb = np.where(np.isneginf(row_lb), float("-inf"), row_lb * self.d1)
        scaled_row_ub = np.where(np.isposinf(row_ub), float("inf"), row_ub * self.d1)
        scaled_col_lb = np.where(np.isneginf(col_lb), float("-inf"), col_lb / self.d2)
        scaled_col_ub = np.where(np.isposinf(col_ub), float("inf"), col_ub / self.d2)

        return current_A, scaled_c, scaled_row_lb, scaled_row_ub, scaled_col_lb, scaled_col_ub

    def unscale_primal(self, x_scaled: np.ndarray) -> np.ndarray:
        """Recover original primal variables: x = D2 * x_scaled."""
        return self.d2 * x_scaled

    def unscale_dual(self, y_scaled: np.ndarray) -> np.ndarray:
        """Recover original dual variables: y = D1 * y_scaled."""
        return self.d1 * y_scaled
