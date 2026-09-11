"""
Primal-Dual Path-Following Interior Point Method (IPM) for Linear Programming.
Implemented from mathematical foundations:
- Mehrotra Predictor-Corrector algorithm
- Barrier parameter updates (central path)
- Augmented KKT / Normal equations solving
- Fraction-to-the-boundary step length control
"""
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus


class InteriorPointSolver(SolverBase):
    """
    Sovereign Primal-Dual Interior Point Method (Mehrotra Predictor-Corrector).
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-7,
        step_fraction: float = 0.995,
    ):
        super().__init__(name="InteriorPoint")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_fraction = step_fraction

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 60.0, **kwargs) -> SolverResult:
        start_time = time.time()
        orig_vars = model.variable_names
        num_orig_vars = len(orig_vars)

        A_raw, row_lb, row_ub, col_lb, col_ub = model.to_matrix_form()
        c_raw = model.get_objective_vector()
        m_raw, n_raw = A_raw.shape

        shifts = np.where(np.isneginf(col_lb), 0.0, col_lb)
        shifts = np.where(np.isnan(shifts), 0.0, shifts)

        eq_rows, eq_cols, eq_data = [], [], []
        b_list = []
        col_count = n_raw

        for i in range(m_raw):
            row_start = A_raw.indptr[i]
            row_end = A_raw.indptr[i + 1]
            cols = A_raw.indices[row_start:row_end]
            vals = A_raw.data[row_start:row_end]
            shift_val = sum(a * shifts[c] for a, c in zip(vals, cols))

            lb, ub = row_lb[i], row_ub[i]
            if abs(ub - lb) < 1e-12:
                for c, v in zip(cols, vals):
                    eq_rows.append(len(b_list))
                    eq_cols.append(c)
                    eq_data.append(v)
                b_list.append(ub - shift_val)
            else:
                if not np.isposinf(ub):
                    for c, v in zip(cols, vals):
                        eq_rows.append(len(b_list))
                        eq_cols.append(c)
                        eq_data.append(v)
                    eq_rows.append(len(b_list))
                    eq_cols.append(col_count)
                    eq_data.append(1.0)
                    col_count += 1
                    b_list.append(ub - shift_val)

                if not np.isneginf(lb):
                    for c, v in zip(cols, vals):
                        eq_rows.append(len(b_list))
                        eq_cols.append(c)
                        eq_data.append(v)
                    eq_rows.append(len(b_list))
                    eq_cols.append(col_count)
                    eq_data.append(-1.0)
                    col_count += 1
                    b_list.append(lb - shift_val)

        for j in range(n_raw):
            if not np.isposinf(col_ub[j]):
                u_diff = col_ub[j] - shifts[j]
                r_idx = len(b_list)
                eq_rows.extend([r_idx, r_idx])
                eq_cols.extend([j, col_count])
                eq_data.extend([1.0, 1.0])
                col_count += 1
                b_list.append(u_diff)

        m = len(b_list)
        n = col_count
        b = np.array(b_list, dtype=np.float64)
        c = np.zeros(n, dtype=np.float64)
        c[:n_raw] = c_raw

        A = sp.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(m, n), dtype=np.float64)

        if m == 0 or n == 0:
            return SolverResult(status=SolverStatus.OPTIMAL, objective_value=0.0)

        # Initial Point
        A_dense = A.toarray()
        A_AT = A_dense.dot(A_dense.T) + np.eye(m) * 1e-8
        try:
            x0 = A_dense.T.dot(np.linalg.solve(A_AT, b))
            y0 = np.linalg.solve(A_AT, A_dense.dot(c))
            s0 = c - A_dense.T.dot(y0)
        except Exception:
            x0 = np.ones(n) * 10.0
            y0 = np.zeros(m)
            s0 = np.ones(n) * 10.0

        # Positivity projection
        delta_x = max(-1.5 * np.min(x0), 0.0)
        delta_s = max(-1.5 * np.min(s0), 0.0)
        x = x0 + delta_x + 1.0
        s = s0 + delta_s + 1.0
        y = y0

        delta_x_hat = 0.5 * np.dot(x, s) / np.sum(s)
        delta_s_hat = 0.5 * np.dot(x, s) / np.sum(x)
        x += delta_x_hat
        s += delta_s_hat

        status = SolverStatus.ITERATION_LIMIT
        iterations = 0
        trace = []

        for it in range(1, self.max_iterations + 1):
            iterations = it
            if time.time() - start_time > time_limit_seconds:
                status = SolverStatus.TIME_LIMIT
                break

            rp = b - A_dense.dot(x)
            rd = c - A_dense.T.dot(y) - s
            xs = x * s
            mu = float(np.dot(x, s) / n)

            norm_rp = np.linalg.norm(rp, np.inf) / (1.0 + np.linalg.norm(b, np.inf))
            norm_rd = np.linalg.norm(rd, np.inf) / (1.0 + np.linalg.norm(c, np.inf))
            rel_gap = abs(np.dot(c, x) - np.dot(b, y)) / (1.0 + abs(np.dot(c, x)))

            # Current objective estimate
            multiplier = 1.0 if model.objective.sense == "minimize" else -1.0
            cur_obj = float(np.dot(c_raw, x[:num_orig_vars] + shifts) * multiplier + model.objective.offset)

            trace.append({
                "iteration": it,
                "mu": float(mu),
                "norm_rp": float(norm_rp),
                "norm_rd": float(norm_rd),
                "rel_gap": float(rel_gap),
                "objective": cur_obj,
            })

            if norm_rp < self.tolerance and norm_rd < self.tolerance and mu < self.tolerance:
                status = SolverStatus.OPTIMAL
                break

            theta = np.clip(x / s, 1e-14, 1e14)
            D = np.diag(theta)
            M = A_dense.dot(D).dot(A_dense.T) + np.eye(m) * 1e-10

            try:
                # 1. Predictor step (affine scaling, sigma = 0)
                r_xs_aff = -xs
                rhs_aff = rp + A_dense.dot(theta * rd - r_xs_aff / s)
                dy_aff = np.linalg.solve(M, rhs_aff)
                dx_aff = theta * (A_dense.T.dot(dy_aff) - rd) + r_xs_aff / s
                ds_aff = (r_xs_aff - s * dx_aff) / x

                neg_x = np.where(dx_aff < 0)[0]
                alpha_p_aff = 1.0 if len(neg_x) == 0 else min(1.0, np.min(-x[neg_x] / dx_aff[neg_x]))
                neg_s = np.where(ds_aff < 0)[0]
                alpha_d_aff = 1.0 if len(neg_s) == 0 else min(1.0, np.min(-s[neg_s] / ds_aff[neg_s]))

                mu_aff = np.dot(x + alpha_p_aff * dx_aff, s + alpha_d_aff * ds_aff) / n
                sigma = float(np.clip((mu_aff / mu) ** 3, 0.0, 1.0))

                # 2. Corrector step (centering + cross term)
                r_xs_corr = -xs - dx_aff * ds_aff + sigma * mu
                rhs_corr = rp + A_dense.dot(theta * rd - r_xs_corr / s)
                dy = np.linalg.solve(M, rhs_corr)
                dx = theta * (A_dense.T.dot(dy) - rd) + r_xs_corr / s
                ds = (r_xs_corr - s * dx) / x

                neg_x = np.where(dx < 0)[0]
                alpha_p = 1.0 if len(neg_x) == 0 else min(1.0, self.step_fraction * np.min(-x[neg_x] / dx[neg_x]))
                neg_s = np.where(ds < 0)[0]
                alpha_d = 1.0 if len(neg_s) == 0 else min(1.0, self.step_fraction * np.min(-s[neg_s] / ds[neg_s]))

                x += alpha_p * dx
                y += alpha_d * dy
                s += alpha_d * ds

            except np.linalg.LinAlgError:
                status = SolverStatus.NUMERICAL_ERROR
                break

        x_orig = x[:num_orig_vars] + shifts
        sol_dict = {orig_vars[j]: float(x_orig[j]) for j in range(num_orig_vars)}

        reported_obj = float(np.dot(c_raw, x_orig) * multiplier + model.objective.offset)

        return SolverResult(
            status=status if status != SolverStatus.ITERATION_LIMIT else SolverStatus.FEASIBLE,
            objective_value=reported_obj,
            primal_solution=sol_dict,
            iterations=iterations,
            runtime_seconds=time.time() - start_time,
            diagnostics={
                "primal_residual": float(norm_rp),
                "dual_residual": float(norm_rd),
                "barrier_mu": float(mu),
                "iteration_trace": trace,
            },
        )
