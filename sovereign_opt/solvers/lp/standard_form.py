"""
Shared computational form for all sovereign LP / QP solvers.

Every OptimizationModel is converted to the *bounded slack form*

    min   c^T x + 1/2 x^T Q x
    s.t.  [A  -I] [x; s] = 0
          l_x <= x <= u_x,   l_row <= s <= u_row

Each row gets one slack column s_i = a_i^T x carrying the row bounds, so
<=, >=, equality, and ranged rows are handled uniformly, free and negative variables
need no splitting, and the all-slack basis (-I) is always a valid starting basis.

Optional power-of-two geometric scaling is applied: A_s = R A C with slack columns
scaled by 1/R so they stay -I. Unscaling rules:
    x = C x_s,   y = R y_s,   d = d_s / C.
"""
from typing import List, Optional
import numpy as np
import scipy.sparse as sp

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.sparse.scaling import geometric_scaling


class BoundedForm:
    def __init__(
        self,
        model: OptimizationModel,
        scale: bool = True,
        scale_integer_columns: bool = False,
        include_quadratic: bool = True,
    ):
        A, row_lb, row_ub, col_lb, col_ub = model.to_matrix_form()
        self.var_names: List[str] = model.variable_names
        self.con_names: List[str] = model.constraint_names
        n = len(self.var_names)
        m = len(self.con_names)
        self.n = n
        self.m = m

        self.obj_sign = 1.0 if model.objective.sense == ObjectiveSense.MINIMIZE else -1.0
        self.offset = float(model.objective.offset)
        self.integer_mask = np.array([model.variables[v].is_integer for v in self.var_names], dtype=bool)

        c = model.get_objective_vector()
        Q = model.get_quadratic_matrix().tocsc() if (include_quadratic and model.objective.is_quadratic) else None
        self.c_orig = c.copy()
        self.Q_orig = Q
        self.A_orig = sp.csr_matrix(A)

        if scale and A.nnz > 0:
            fixed = None if scale_integer_columns else self.integer_mask
            R, C = geometric_scaling(A, fixed_cols=fixed)
        else:
            R, C = np.ones(m), np.ones(n)

        As = sp.diags(R) @ A @ sp.diags(C)
        self.A: sp.csc_matrix = sp.hstack([As, -sp.identity(m, format="csc")], format="csc")
        self.row_scale = R
        self.col_scale = np.concatenate([C, 1.0 / R])
        self.lb = np.concatenate([col_lb / C, row_lb * R])
        self.ub = np.concatenate([col_ub / C, row_ub * R])
        self.c = np.concatenate([c * C, np.zeros(m)])
        self.Q: Optional[sp.csc_matrix] = (sp.diags(C) @ Q @ sp.diags(C)).tocsc() if Q is not None else None

    # ------------------------------------------------------------------ sizes
    @property
    def N(self) -> int:
        return self.n + self.m

    def slack_col(self, row: int) -> int:
        return self.n + row

    # --------------------------------------------------------------- unscale
    def unscale_x(self, xs: np.ndarray) -> np.ndarray:
        return xs * self.col_scale

    def unscale_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.row_scale

    def unscale_d(self, ds: np.ndarray) -> np.ndarray:
        return ds / self.col_scale

    def user_objective(self, x_struct: np.ndarray) -> float:
        """Objective in the user's sense from unscaled structural values."""
        f = float(self.c_orig @ x_struct)
        if self.Q_orig is not None:
            f += 0.5 * float(x_struct @ (self.Q_orig @ x_struct))
        return self.obj_sign * f + self.offset

    def internal_objective(self, x_struct: np.ndarray) -> float:
        """Minimize-normalized objective (no offset) from unscaled structural values."""
        f = float(self.c_orig @ x_struct)
        if self.Q_orig is not None:
            f += 0.5 * float(x_struct @ (self.Q_orig @ x_struct))
        return f

    def column_label(self, j: int) -> str:
        if j < self.n:
            return self.var_names[j]
        return f"slack[{self.con_names[j - self.n]}]"

    # ------------------------------------------------------------ cut rows
    def add_rows(self, G: sp.spmatrix, g_lb: np.ndarray, g_ub: np.ndarray, names: Optional[List[str]] = None) -> int:
        """
        Append rows g_lb <= G x <= g_ub (G in unscaled structural coordinates).
        New slack columns are appended after all existing columns. Returns the number of rows added.
        """
        G = sp.csr_matrix(G, dtype=np.float64)
        k = G.shape[0]
        if k == 0:
            return 0
        Gs = sp.csr_matrix(G @ sp.diags(self.col_scale[: self.n]))
        rmax = np.asarray(abs(Gs).max(axis=1).todense()).ravel()
        rs = np.where(rmax > 0, np.power(2.0, np.round(np.log2(1.0 / np.where(rmax > 0, rmax, 1.0)))), 1.0)
        Gs = sp.csr_matrix(sp.diags(rs) @ Gs)

        N_old = self.N
        m_old = self.m
        top = sp.hstack([self.A, sp.csc_matrix((m_old, k))], format="csc")
        bottom = sp.hstack(
            [Gs, sp.csc_matrix((k, N_old - self.n)), -sp.identity(k, format="csc")], format="csc"
        )
        self.A = sp.vstack([top, bottom], format="csc")
        self.lb = np.concatenate([self.lb, np.asarray(g_lb, dtype=np.float64) * rs])
        self.ub = np.concatenate([self.ub, np.asarray(g_ub, dtype=np.float64) * rs])
        self.c = np.concatenate([self.c, np.zeros(k)])
        self.col_scale = np.concatenate([self.col_scale, 1.0 / rs])
        self.row_scale = np.concatenate([self.row_scale, rs])
        self.A_orig = sp.vstack([self.A_orig, G], format="csr")
        self.con_names = self.con_names + (names or [f"cut_{m_old + i}" for i in range(k)])
        self.m += k
        return k
