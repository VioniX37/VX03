"""
Unit tests for SparseMatrix and RuizScaling.
"""
import numpy as np
import scipy.sparse as sp
from sovereign_opt.sparse.matrix import SparseMatrix
from sovereign_opt.sparse.scaling import RuizScaling


def test_sparse_matrix_spmv():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    A_csr = sp.csr_matrix((data, (rows, cols)), shape=(2, 2))

    smat = SparseMatrix(A_csr)
    assert smat.num_rows == 2
    assert smat.num_cols == 2
    assert smat.nnz == 4

    x = np.array([2.0, 3.0])
    y = smat.spmv(x)
    # y[0] = 1*2 + 2*3 = 8
    # y[1] = 3*2 + 4*3 = 18
    assert np.allclose(y, [8.0, 18.0])

    yt = smat.spmv_transpose(x)
    # yt[0] = 1*2 + 3*3 = 11
    # yt[1] = 2*2 + 4*3 = 16
    assert np.allclose(yt, [11.0, 16.0])


def test_ruiz_scaling_equilibration():
    # Matrix with wide dynamic range
    A_raw = sp.csr_matrix(np.array([
        [1000.0, 0.01],
        [2000.0, 0.05],
    ]))
    c = np.array([10.0, 0.1])
    row_lb = np.array([0.0, 0.0])
    row_ub = np.array([1000.0, 2000.0])
    col_lb = np.array([0.0, 0.0])
    col_ub = np.array([1.0, 1.0])

    scaler = RuizScaling(max_iterations=10)
    A_scaled, c_s, r_lb_s, r_ub_s, c_lb_s, c_ub_s = scaler.fit_transform(
        SparseMatrix(A_raw), c, row_lb, row_ub, col_lb, col_ub
    )

    # After scaling, row infinity norms should be close to 1.0
    row_norms = A_scaled.row_inf_norms()
    col_norms = A_scaled.col_inf_norms()
    assert np.all(np.abs(row_norms - 1.0) < 0.1)
    assert np.all(np.abs(col_norms - 1.0) < 0.1)
