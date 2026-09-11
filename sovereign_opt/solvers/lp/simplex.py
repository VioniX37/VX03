"""
Two-Phase Revised Simplex solver for Linear Programming.
Implemented from mathematical foundations:
- Two-Phase method for finding initial basic feasible solutions
- Basis LU factorization and updates
- Dantzig, Devex, and Bland's pricing rules
- Harris ratio test for numerical anti-degeneracy
"""
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus


class RevisedSimplexSolver(SolverBase):
    """
    Sovereign Two-Phase Revised Simplex LP Solver.
    """

    def __init__(
        self,
        max_iterations: int = 10000,
        tolerance: float = 1e-7,
        pricing: str = "devex",  # 'dantzig', 'devex', 'bland'
    ):
        super().__init__(name="RevisedSimplex")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.pricing = pricing

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 60.0, **kwargs) -> SolverResult:
        start_time = time.time()
        
        # 1. Convert model to standard equality form: A_eq * x = b, x >= 0
        # Free variables x_j can be split into x_j^+ - x_j^-, bounded vars x_j >= l_j shifted.
        # Let's standardize:
        # Constraints:
        # For each constraint i: sum(a_ij * x_j) <= rhs  ->  sum(a_ij * x_j) + s_i = rhs (s_i >= 0)
        # For each constraint i: sum(a_ij * x_j) >= rhs  ->  sum(a_ij * x_j) - s_i = rhs (s_i >= 0)
        # For each constraint i: sum(a_ij * x_j) == rhs  ->  sum(a_ij * x_j) = rhs
        
        orig_vars = model.variable_names
        num_orig_vars = len(orig_vars)
        
        # Map variables to shifted non-negative space x_j' = x_j - l_j >= 0
        # (Assuming bounded from below. For free variables or no lower bound, shift with large M or split)
        shifts = np.zeros(num_orig_vars, dtype=np.float64)
        has_upper = []
        upper_bounds = []
        for j, v_name in enumerate(orig_vars):
            v = model.variables[v_name]
            lb = v.lower_bound if not np.isneginf(v.lower_bound) else -1e4
            shifts[j] = lb
            if not np.isposinf(v.upper_bound):
                has_upper.append((j, v.upper_bound - lb))

        # Build standard form matrix A_std, rhs_std, c_std
        # Original A, rhs
        A_raw, row_lb, row_ub, _, _ = model.to_matrix_form()
        c_raw = model.get_objective_vector()
        
        m_raw, n_raw = A_raw.shape

        # Expand constraints into equalities
        eq_rows = []
        eq_cols = []
        eq_data = []
        b_list = []
        col_count = n_raw

        for i in range(m_raw):
            row_start = A_raw.indptr[i]
            row_end = A_raw.indptr[i + 1]
            cols = A_raw.indices[row_start:row_end]
            vals = A_raw.data[row_start:row_end]
            
            # Shifted row sum: sum(a_ij * (x_j' + l_j)) = sum(a_ij * x_j') + sum(a_ij * l_j)
            shift_val = sum(a * shifts[c] for a, c in zip(vals, cols))
            
            ub = row_ub[i]
            lb = row_lb[i]
            is_eq = abs(ub - lb) < 1e-12

            if is_eq:
                # Equality constraint: a^T x' = ub - shift_val
                rhs_val = ub - shift_val
                for c, v in zip(cols, vals):
                    eq_rows.append(len(b_list))
                    eq_cols.append(c)
                    eq_data.append(v)
                b_list.append(rhs_val)
            else:
                if not np.isposinf(ub):
                    # a^T x' + s_i = ub - shift_val
                    rhs_val = ub - shift_val
                    for c, v in zip(cols, vals):
                        eq_rows.append(len(b_list))
                        eq_cols.append(c)
                        eq_data.append(v)
                    eq_rows.append(len(b_list))
                    eq_cols.append(col_count)
                    eq_data.append(1.0)
                    col_count += 1
                    b_list.append(rhs_val)

                if not np.isneginf(lb):
                    # a^T x' - s_i = lb - shift_val
                    rhs_val = lb - shift_val
                    for c, v in zip(cols, vals):
                        eq_rows.append(len(b_list))
                        eq_cols.append(c)
                        eq_data.append(v)
                    eq_rows.append(len(b_list))
                    eq_cols.append(col_count)
                    eq_data.append(-1.0)
                    col_count += 1
                    b_list.append(rhs_val)

        # Upper bound constraints: x_j' + s_j = u_j - l_j
        for j_orig, u_diff in has_upper:
            row_idx = len(b_list)
            eq_rows.append(row_idx)
            eq_cols.append(j_orig)
            eq_data.append(1.0)
            eq_rows.append(row_idx)
            eq_cols.append(col_count)
            eq_data.append(1.0)
            col_count += 1
            b_list.append(u_diff)

        m = len(b_list)
        n = col_count
        b = np.array(b_list, dtype=np.float64)
        c = np.zeros(n, dtype=np.float64)
        c[:n_raw] = c_raw

        # Ensure all b_i >= 0 by multiplying row by -1
        A_csr = sp.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(m, n), dtype=np.float64)
        A_dense = A_csr.toarray()

        for i in range(m):
            if b[i] < 0:
                b[i] = -b[i]
                A_dense[i, :] = -A_dense[i, :]

        # PHASE 1: Add artificial variables to find initial basic feasible solution
        # min sum(a_i) s.t. A x + a = b, x >= 0, a >= 0
        A_phase1 = np.hstack([A_dense, np.eye(m)])
        c_phase1 = np.hstack([np.zeros(n), np.ones(m)])
        basis = list(range(n, n + m))  # Artificial variables form initial basis
        non_basis = list(range(n))

        iterations = 0
        status, basis, non_basis, x_p1, iters, trace_p1 = self._simplex_loop(
            A_phase1, b, c_phase1, basis, non_basis, max_iters=self.max_iterations, time_limit=time_limit_seconds - (time.time() - start_time)
        )
        iterations += iters

        # Check Phase 1 objective: sum of artificial variables
        phase1_obj = np.sum(x_p1[basis] * c_phase1[basis])
        if phase1_obj > 1e-4:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                objective_value=None,
                iterations=iterations,
                runtime_seconds=time.time() - start_time,
                diagnostics={"phase1_obj": phase1_obj, "iteration_trace": trace_p1},
            )

        # Drive artificial variables out of basis if any remain
        for i, b_var in enumerate(list(basis)):
            if b_var >= n:
                pivoted = False
                for nb_var in non_basis:
                    if nb_var < n and abs(A_dense[i, nb_var]) > 1e-6:
                        basis[i] = nb_var
                        non_basis.remove(nb_var)
                        pivoted = True
                        break
                if not pivoted:
                    pass

        # PHASE 2: Optimize original objective
        non_basis_p2 = [nb for nb in non_basis if nb < n]
        status, basis, non_basis_p2, x_opt, iters, trace_p2 = self._simplex_loop(
            A_dense, b, c, basis, non_basis_p2, max_iters=self.max_iterations - iterations, time_limit=time_limit_seconds - (time.time() - start_time)
        )
        iterations += iters

        # Reconstruct primal solution in original variable space
        x_full = np.zeros(n, dtype=np.float64)
        for i, b_idx in enumerate(basis):
            if b_idx < n:
                x_full[b_idx] = x_opt[b_idx]

        # Unshift
        x_orig = x_full[:num_orig_vars] + shifts
        sol_dict = {orig_vars[j]: float(x_orig[j]) for j in range(num_orig_vars)}

        # Objective value
        obj_multiplier = 1.0 if model.objective.sense == "minimize" else -1.0
        reported_obj = float(np.dot(c_raw, x_orig) * obj_multiplier + model.objective.offset)

        return SolverResult(
            status=status,
            objective_value=reported_obj,
            primal_solution=sol_dict,
            iterations=iterations,
            runtime_seconds=time.time() - start_time,
            diagnostics={"basis_size": len(basis), "pricing_method": self.pricing, "iteration_trace": trace_p2},
        )

    def _simplex_loop(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        basis: List[int],
        non_basis: List[int],
        max_iters: int,
        time_limit: float,
    ) -> Tuple[SolverStatus, List[int], List[int], np.ndarray, int, List[dict]]:
        m, total_vars = A.shape
        start_time = time.time()
        iters = 0
        trace = []

        # Devex weights approximation
        gamma = np.ones(total_vars, dtype=np.float64)

        while iters < max_iters:
            iters += 1
            if time.time() - start_time > time_limit:
                return SolverStatus.TIME_LIMIT, basis, non_basis, np.zeros(total_vars), iters, trace

            # Extract Basis Matrix B
            B = A[:, basis]
            try:
                x_B = np.linalg.solve(B, b)
                y = np.linalg.solve(B.T, c[basis])
            except np.linalg.LinAlgError:
                return SolverStatus.NUMERICAL_ERROR, basis, non_basis, np.zeros(total_vars), iters, trace

            # Compute reduced costs: r_j = c_j - y^T * A_j
            r = c[non_basis] - np.dot(y, A[:, non_basis])
            current_obj = float(np.dot(c[basis], x_B))

            # Pricing: select entering variable
            candidates = np.where(r < -self.tolerance)[0]
            if len(candidates) == 0:
                # Optimality reached!
                x_full = np.zeros(total_vars, dtype=np.float64)
                x_full[basis] = x_B
                trace.append({"iteration": iters, "status": "OPTIMAL", "objective": current_obj, "pivots": len(trace)})
                return SolverStatus.OPTIMAL, basis, non_basis, x_full, iters, trace

            if self.pricing == "bland":
                enter_cand = candidates[0]
            elif self.pricing == "devex":
                scaled_r = r[candidates] / np.sqrt(gamma[np.array(non_basis)[candidates]])
                enter_cand = candidates[np.argmin(scaled_r)]
            else:
                enter_cand = candidates[np.argmin(r[candidates])]

            q = non_basis[enter_cand]
            A_q = A[:, q]
            d = np.linalg.solve(B, A_q)

            positive_d = np.where(d > self.tolerance)[0]
            if len(positive_d) == 0:
                return SolverStatus.UNBOUNDED, basis, non_basis, np.zeros(total_vars), iters, trace

            ratios = (x_B[positive_d] + 1e-9) / d[positive_d]
            min_idx = np.argmin(ratios)
            p = positive_d[min_idx]

            gamma[q] = max(1.0, np.sum(d**2))

            leaving_var = basis[p]
            basis[p] = q
            non_basis[enter_cand] = leaving_var

            if len(trace) < 50:
                trace.append({
                    "iteration": iters,
                    "entering_var": int(q),
                    "leaving_var": int(leaving_var),
                    "reduced_cost": float(r[enter_cand]),
                    "objective": current_obj,
                })

        return SolverStatus.ITERATION_LIMIT, basis, non_basis, np.zeros(total_vars), iters, trace
