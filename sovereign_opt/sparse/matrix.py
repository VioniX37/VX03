"""
Sparse Matrix representations and sparse linear algebra operations.
Implements Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) representations.
"""
from typing import Tuple, Optional
import numpy as np
import scipy.sparse as sp


class SparseMatrix:
    """
    Dual CSR/CSC sparse matrix wrapper optimized for linear optimization operations.
    - CSR enables fast row access and SpMV (A * x).
    - CSC enables fast column access and transpose SpMV (A^T * y).
    """

    def __init__(self, mat: sp.spmatrix):
        if sp.issparse(mat) and mat.format == "csr":
            self._csr: Optional[sp.csr_matrix] = mat
            self._csc: Optional[sp.csc_matrix] = None
        elif sp.issparse(mat) and mat.format == "csc":
            self._csc = mat
            self._csr = None
        else:
            self._csr = sp.csr_matrix(mat)
            self._csc = None

        self._shape: Tuple[int, int] = mat.shape

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def num_rows(self) -> int:
        return self._shape[0]

    @property
    def num_cols(self) -> int:
        return self._shape[1]

    @property
    def nnz(self) -> int:
        return self.csr.nnz

    @property
    def density(self) -> float:
        total = self.num_rows * self.num_cols
        return (self.nnz / total) if total > 0 else 0.0

    @property
    def csr(self) -> sp.csr_matrix:
        if self._csr is None:
            self._csr = self._csc.tocsr()
        return self._csr

    @property
    def csc(self) -> sp.csc_matrix:
        if self._csc is None:
            self._csc = self._csr.tocsc()
        return self._csc

    def spmv(self, x: np.ndarray) -> np.ndarray:
        """Compute y = A * x using CSR format."""
        return self.csr.dot(x)

    def spmv_transpose(self, y: np.ndarray) -> np.ndarray:
        """Compute x = A^T * y using CSC format."""
        return self.csc.T.dot(y)

    def get_row(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (col_indices, values) for row i."""
        csr = self.csr
        start, end = csr.indptr[i], csr.indptr[i + 1]
        return csr.indices[start:end], csr.data[start:end]

    def get_col(self, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (row_indices, values) for col j."""
        csc = self.csc
        start, end = csc.indptr[j], csc.indptr[j + 1]
        return csc.indices[start:end], csc.data[start:end]

    def row_inf_norms(self) -> np.ndarray:
        """Compute ||row_i||_infinity for all rows (vectorized)."""
        if self.nnz == 0:
            return np.zeros(self.num_rows, dtype=np.float64)
        return np.asarray(abs(self.csr).max(axis=1).todense(), dtype=np.float64).ravel()

    def col_inf_norms(self) -> np.ndarray:
        """Compute ||col_j||_infinity for all columns (vectorized)."""
        if self.nnz == 0:
            return np.zeros(self.num_cols, dtype=np.float64)
        return np.asarray(abs(self.csc).max(axis=0).todense(), dtype=np.float64).ravel()

    def scale_rows_cols(self, d_row: np.ndarray, d_col: np.ndarray) -> "SparseMatrix":
        """
        Return new SparseMatrix: A_scaled = diag(d_row) * A * diag(d_col)
        """
        scaled = sp.diags(d_row) @ self.csr @ sp.diags(d_col)
        return SparseMatrix(sp.csr_matrix(scaled))
