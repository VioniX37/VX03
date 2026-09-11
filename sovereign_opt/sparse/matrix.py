"""
Sparse Matrix representations and sparse linear algebra operations.
Implements Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) representations.
"""
from typing import Tuple, List, Optional
import numpy as np
import scipy.sparse as sp


class SparseMatrix:
    """
    Dual CSR/CSC sparse matrix wrapper optimized for linear optimization operations.
    - CSR enables fast row access and SpMV (A * x).
    - CSC enables fast column access and transpose SpMV (A^T * y).
    """

    def __init__(self, mat: sp.spmatrix):
        if sp.isspmatrix_csr(mat):
            self._csr: sp.csr_matrix = mat
            self._csc: Optional[sp.csc_matrix] = None
        elif sp.isspmatrix_csc(mat):
            self._csc: sp.csc_matrix = mat
            self._csr: Optional[sp.csr_matrix] = None
        else:
            self._csr = mat.tocsr()
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
        start = csr.indptr[i]
        end = csr.indptr[i + 1]
        return csr.indices[start:end], csr.data[start:end]

    def get_col(self, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (row_indices, values) for col j."""
        csc = self.csc
        start = csc.indptr[j]
        end = csc.indptr[j + 1]
        return csc.indices[start:end], csc.data[start:end]

    def row_inf_norms(self) -> np.ndarray:
        """Compute ||row_i||_infinity for all rows."""
        csr = self.csr
        norms = np.zeros(self.num_rows, dtype=np.float64)
        for i in range(self.num_rows):
            start = csr.indptr[i]
            end = csr.indptr[i + 1]
            if start < end:
                norms[i] = np.max(np.abs(csr.data[start:end]))
        return norms

    def col_inf_norms(self) -> np.ndarray:
        """Compute ||col_j||_infinity for all columns."""
        csc = self.csc
        norms = np.zeros(self.num_cols, dtype=np.float64)
        for j in range(self.num_cols):
            start = csc.indptr[j]
            end = csc.indptr[j + 1]
            if start < end:
                norms[j] = np.max(np.abs(csc.data[start:end]))
        return norms

    def scale_rows_cols(self, d_row: np.ndarray, d_col: np.ndarray) -> "SparseMatrix":
        """
        Return new SparseMatrix: A_scaled = diag(d_row) * A * diag(d_col)
        """
        D_r = sp.diags(d_row, shape=(self.num_rows, self.num_rows), format="csr")
        D_c = sp.diags(d_col, shape=(self.num_cols, self.num_cols), format="csr")
        scaled_csr = D_r.dot(self.csr).dot(D_c).tocsr()
        return SparseMatrix(scaled_csr)
