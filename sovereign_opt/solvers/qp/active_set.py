"""
Active Set solver for Convex Quadratic Programming (QP).
Solves:
    min  0.5 * x^T Q x + c^T x
    s.t. A x <= b,  l <= x <= u
Uses sovereign Phase-1 LP to establish initial feasible basis if needed.
"""
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus
from sovereign_opt.solvers.qp.kkt import KKTSolver
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver


class ActiveSetQPSolver(SolverBase):
    """
    Sovereign Active Set Solver for Convex Quadratic Programs.
    """

    def __init__(self, max_iterations: int = 500, tolerance: float = 1e-7):
        super().__init__(name="ActiveSetQP")
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 60.0, **kwargs) -> SolverResult:
        start_time = time.time()
        orig_vars = model.variable_names
        n = len(orig_vars)

        # Extract matrices
        Q_csr = model.get_quadratic_matrix()
        Q = Q_csr.toarray()
        c = model.get_objective_vector()

        # Check positive semi-definiteness
        min_eig = np.min(np.linalg.eigvalsh(Q))
        if min_eig < -1e-7:
            return SolverResult(
                status=SolverStatus.NUMERICAL_ERROR,
                objective_value=None,
                runtime_seconds=time.time() - start_time,
                diagnostics={"error": "Non-convex QP: Q matrix has negative eigenvalues.", "min_eigenvalue": float(min_eig)},
            )

        # Regularize Q slightly if singular or strictly positive semi-definite
        if min_eig < 1e-8:
            Q += np.eye(n) * 1e-6

        # Standardize constraints to A_ineq * x <= b_ineq
        A_raw, row_lb, row_ub, col_lb, col_ub = model.to_matrix_form()
        A_dense = A_raw.toarray()
        m_raw = A_dense.shape[0]

        ineq_rows = []
        b_ineq_list = []

        for i in range(m_raw):
            if not np.isposinf(row_ub[i]):
                ineq_rows.append(A_dense[i, :])
                b_ineq_list.append(row_ub[i])
            if not np.isneginf(row_lb[i]):
                ineq_rows.append(-A_dense[i, :])
                b_ineq_list.append(-row_lb[i])

        for j in range(n):
            if not np.isposinf(col_ub[j]):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                ineq_rows.append(e_j)
                b_ineq_list.append(col_ub[j])
            if not np.isneginf(col_lb[j]):
                e_j = np.zeros(n)
                e_j[j] = -1.0
                ineq_rows.append(e_j)
                b_ineq_list.append(-col_lb[j])

        m_ineq = len(b_ineq_list)
        if m_ineq > 0:
            A_ineq = np.vstack(ineq_rows)
            b_ineq = np.array(b_ineq_list, dtype=np.float64)
        else:
            A_ineq = np.empty((0, n))
            b_ineq = np.array([], dtype=np.float64)

        # 1. Check unconstrained minimum x_unc = -Q^-1 c
        x_unc = np.linalg.solve(Q, -c)
        working_set: List[int] = []

        if m_ineq > 0:
            viol = A_ineq.dot(x_unc) - b_ineq
            if np.all(viol <= self.tolerance):
                # Unconstrained optimum is fully feasible!
                x = x_unc
                status = SolverStatus.OPTIMAL
            else:
                # Find initial feasible point using Phase 1 Simplex
                feas_model = OptimizationModel("QP_Phase1")
                for v in model.variables.values():
                    feas_model.add_variable(v.name, lower_bound=v.lower_bound, upper_bound=v.upper_bound)
                for con in model.constraints.values():
                    feas_model.add_constraint(
                        con.name,
                        coefficients=con.coefficients,
                        sense=con.sense,
                        rhs=con.rhs,
                        lower_bound=con.lower_bound,
                        upper_bound=con.upper_bound,
                    )
                feas_model.set_objective(linear_coefficients=model.objective.linear_coefficients, sense=model.objective.sense)
                
                simplex = RevisedSimplexSolver()
                init_res = simplex.solve(feas_model, time_limit_seconds=10.0)

                if init_res.is_feasible:
                    x = np.array([init_res.primal_solution.get(v_name, 0.0) for v_name in orig_vars], dtype=np.float64)
                else:
                    x = np.zeros(n, dtype=np.float64)

                status = SolverStatus.ITERATION_LIMIT

                # Form initial working set of active constraints
                for i in range(m_ineq):
                    if abs(np.dot(A_ineq[i, :], x) - b_ineq[i]) <= 1e-5:
                        working_set.append(i)
        else:
            x = x_unc
            status = SolverStatus.OPTIMAL

        # 2. Active Set Main Iteration Loop
        iterations = 0
        trace = []

        if status != SolverStatus.OPTIMAL:
            for it in range(1, self.max_iterations + 1):
                iterations = it
                if time.time() - start_time > time_limit_seconds:
                    status = SolverStatus.TIME_LIMIT
                    break

                g = Q.dot(x) + c

                if working_set:
                    A_w = A_ineq[working_set, :]
                    b_w = np.zeros(len(working_set))
                else:
                    A_w = np.empty((0, n))
                    b_w = np.array([])

                p, lam = KKTSolver.solve(Q, g, A_w, b_w)
                p_norm = float(np.linalg.norm(p))
                current_obj = float(0.5 * np.dot(x, Q.dot(x)) + np.dot(c, x) + model.objective.offset)

                trace.append({
                    "iteration": it,
                    "objective": current_obj,
                    "p_norm": p_norm,
                    "active_constraints": len(working_set),
                })

                if p_norm < self.tolerance:
                    # Stationary point on current working surface
                    if not working_set:
                        status = SolverStatus.OPTIMAL
                        break

                    min_lam_idx = np.argmin(lam) if len(lam) > 0 else 0
                    min_lam = lam[min_lam_idx] if len(lam) > 0 else 0.0

                    if min_lam >= -self.tolerance:
                        status = SolverStatus.OPTIMAL
                        break
                    else:
                        working_set.pop(min_lam_idx)
                else:
                    # Take step p with line search
                    alpha = 1.0
                    blocking_idx = -1

                    for i in range(m_ineq):
                        if i not in working_set:
                            a_i_p = np.dot(A_ineq[i, :], p)
                            if a_i_p > self.tolerance:
                                dist = (b_ineq[i] - np.dot(A_ineq[i, :], x)) / a_i_p
                                if dist < alpha:
                                    alpha = dist
                                    blocking_idx = i

                    alpha = max(0.0, min(1.0, alpha))
                    x += alpha * p

                    if blocking_idx != -1 and alpha < 1.0:
                        working_set.append(blocking_idx)

        sol_dict = {orig_vars[j]: float(x[j]) for j in range(n)}
        obj_val = float(0.5 * np.dot(x, Q.dot(x)) + np.dot(c, x) + model.objective.offset)

        return SolverResult(
            status=status if status != SolverStatus.ITERATION_LIMIT else SolverStatus.FEASIBLE,
            objective_value=obj_val,
            primal_solution=sol_dict,
            iterations=iterations,
            runtime_seconds=time.time() - start_time,
            diagnostics={"active_constraints_count": len(working_set), "iteration_trace": trace},
        )
