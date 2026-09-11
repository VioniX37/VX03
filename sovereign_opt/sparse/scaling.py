"""
Matrix scaling techniques for mathematical optimization.
Implements Ruiz Equilibration scaling (Ruiz 2001) for sparse matrices.
"""
from typing import Tuple
import numpy as np
from sovereign_opt.sparse.matrix import SparseMatrix


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

        current_A = A

        for _ in range(self.max_iterations):
            # Row infinity norms
            r_norms = current_A.row_inf_norms()
            r_norms = np.where(r_norms > 1e-12, np.sqrt(r_norms), 1.0)
            dr = 1.0 / r_norms

            # Col infinity norms
            c_norms = current_A.col_inf_norms()
            c_norms = np.where(c_norms > 1e-12, np.sqrt(c_norms), 1.0)
            dc = 1.0 / c_norms

            # Check convergence: max deviation from 1
            max_r_dev = np.max(np.abs(r_norms**2 - 1.0))
            max_c_dev = np.max(np.abs(c_norms**2 - 1.0))
            if max_r_dev < self.tolerance and max_c_dev < self.tolerance:
                break

            self.d1 *= dr
            self.d2 *= dc
            current_A = current_A.scale_rows_cols(dr, dc)

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
