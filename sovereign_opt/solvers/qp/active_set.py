"""
Primal Active Set solver for Convex Quadratic Programming (v2).
Solves:
    min  0.5 * x^T Q x + c^T x
    s.t. row_lb <= A x <= row_ub,  l <= x <= u

v2 improvements:
- Feasible start from the sovereign bounded simplex (Phase 1) instead of a zero fallback.
- Null-space method for each equality-constrained subproblem: handles singular / semidefinite Q
  exactly, with no eigenvalue perturbation of the model.
- Detection of unbounded zero-curvature descent directions.
- Degeneracy handling: unit-norm constraint rows (consistent rank decisions), tiny random
  relaxation of inequality right-hand sides, Bland-ordered working-set swaps when a linearly
  dependent constraint blocks, and a stall guard.
- Returns row duals and reduced costs.
Best suited to small dense problems; the QP interior point solver is the default for larger ones.
"""
import time
from typing import List
import numpy as np
import scipy.linalg as la

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus
from sovereign_opt.solvers.lp.standard_form import BoundedForm
from sovereign_opt.solvers.lp.simplex_engine import SimplexEngine, LPStatus
from sovereign_opt.solvers.lp.simplex import downsample_trace
from sovereign_opt.solvers.qp.interior_point_qp import check_convexity
from sovereign_opt.solvers.qp.polish import polish_solution


class ActiveSetQPSolver(SolverBase):
    """
    Sovereign Active Set Solver for Convex Quadratic Programs.
    """

    def __init__(self, max_iterations: int = 5000, tolerance: float = 1e-9):
        super().__init__(name="ActiveSetQP")
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 60.0, **kwargs) -> SolverResult:
        start = time.time()
        names = model.variable_names
        n = len(names)
        A_sp, rl, ru, cl, cu = model.to_matrix_form()
        A = A_sp.toarray()
        c = model.get_objective_vector()
        Q = model.get_quadratic_matrix().toarray()
        sign = 1.0 if model.objective.sense == "minimize" else -1.0

        convex, min_eig = check_convexity(model.get_quadratic_matrix())
        if not convex:
            return SolverResult(status=SolverStatus.NUMERICAL_ERROR, runtime_seconds=time.time() - start,
                                diagnostics={"error": "Non-convex QP: Q matrix has negative eigenvalues.", "min_eigenvalue": min_eig})

        # Constraint system: equalities E x = e, inequalities G x <= h, each with an owner for dual recovery.
        E_rows, e_vals, e_own = [], [], []
        G_rows, h_vals, g_own = [], [], []
        for i in range(A.shape[0]):
            if not np.any(A[i]):
                continue
            if np.isfinite(rl[i]) and np.isfinite(ru[i]) and ru[i] - rl[i] <= 1e-12 * max(1.0, abs(rl[i])):
                E_rows.append(A[i]); e_vals.append(rl[i]); e_own.append(("row", i, 1.0))
                continue
            if np.isfinite(ru[i]):
                G_rows.append(A[i]); h_vals.append(ru[i]); g_own.append(("row", i, 1.0))
            if np.isfinite(rl[i]):
                G_rows.append(-A[i]); h_vals.append(-rl[i]); g_own.append(("row", i, -1.0))
        I = np.eye(n)
        for j in range(n):
            if np.isfinite(cl[j]) and np.isfinite(cu[j]) and cu[j] - cl[j] <= 1e-12 * max(1.0, abs(cl[j])):
                E_rows.append(I[j]); e_vals.append(cl[j]); e_own.append(("var", j, 1.0))
                continue
            if np.isfinite(cu[j]):
                G_rows.append(I[j]); h_vals.append(cu[j]); g_own.append(("var", j, 1.0))
            if np.isfinite(cl[j]):
                G_rows.append(-I[j]); h_vals.append(-cl[j]); g_own.append(("var", j, -1.0))
        E = np.array(E_rows, dtype=float).reshape(-1, n)
        e = np.array(e_vals, dtype=float)
        G = np.array(G_rows, dtype=float).reshape(-1, n)
        h = np.array(h_vals, dtype=float)

        # Unit-norm rows make every rank / step decision scale invariant.
        e_norm = np.linalg.norm(E, axis=1) if len(e) else np.zeros(0)
        g_norm = np.linalg.norm(G, axis=1) if len(h) else np.zeros(0)
        if len(e):
            E, e = E / e_norm[:, None], e / e_norm
        if len(h):
            G, h = G / g_norm[:, None], h / g_norm
        # Tiny outward relaxation breaks exact degeneracy (ties among active inequalities).
        rng = np.random.default_rng(17)
        h = h + rng.uniform(0.5, 1.0, len(h)) * 1e-9 * (1.0 + np.abs(h))

        # Phase 1: feasible point from the sovereign simplex (zero objective).
        form = BoundedForm(model, scale=True, include_quadratic=False)
        engine = SimplexEngine(form.A, np.zeros(form.N), form.lb, form.ub, form.n)
        feas = engine.solve(time_limit=max(1.0, time_limit_seconds))
        if feas == LPStatus.INFEASIBLE:
            return SolverResult(status=SolverStatus.INFEASIBLE, runtime_seconds=time.time() - start,
                                diagnostics={"infeasibility_certificate": "simplex phase 1 proved the feasible region empty"})
        if feas != LPStatus.OPTIMAL:
            return SolverResult(status=SolverStatus.NUMERICAL_ERROR, runtime_seconds=time.time() - start,
                                diagnostics={"error": f"phase 1 simplex returned {feas}"})
        x = form.unscale_x(engine.x)[:n]

        W: List[int] = []
        for i in range(len(h)):
            if abs(G[i] @ x - h[i]) <= 1e-8 * max(1.0, abs(h[i])) and self._independent(E, G, W, i):
                W.append(i)

        trace = []
        status = SolverStatus.ITERATION_LIMIT
        degenerate = swaps = stalls = 0
        lam_eq = np.zeros(len(e))
        lam_in = np.zeros(0)
        it = 0
        gscale = 1.0 + float(np.max(np.abs(Q), initial=0.0))
        for it in range(1, self.max_iterations + 1):
            if time.time() - start > time_limit_seconds:
                status = SolverStatus.TIME_LIMIT
                break
            g = Q @ x + c
            Aw = np.vstack([E, G[W]]) if (len(e) or W) else np.zeros((0, n))
            Z = la.null_space(Aw) if Aw.shape[0] else np.eye(n)
            unbounded_dir = None
            if Z.shape[1] == 0:
                p = np.zeros(n)
            else:
                H = Z.T @ Q @ Z
                gz = Z.T @ g
                evals, evecs = np.linalg.eigh(H)
                flat = evals <= 1e-10 * gscale
                if flat.any():
                    proj = evecs[:, flat].T @ gz
                    if np.linalg.norm(proj) > 1e-9 * (1.0 + np.linalg.norm(g)):
                        unbounded_dir = -Z @ (evecs[:, flat] @ proj)
                if unbounded_dir is not None:
                    p = unbounded_dir
                else:
                    inv = np.where(flat, 0.0, 1.0 / np.where(flat, 1.0, evals))
                    p = -Z @ (evecs @ (inv * (evecs.T @ gz)))

            obj = float(0.5 * x @ Q @ x + c @ x)
            pnorm = float(np.linalg.norm(p))
            trace.append({"iteration": it, "objective": sign * obj + model.objective.offset,
                          "p_norm": pnorm, "active_constraints": len(W)})

            if pnorm <= 1e-10 * (1.0 + np.linalg.norm(x)):
                lam = np.linalg.lstsq(Aw.T, -g, rcond=None)[0] if Aw.shape[0] else np.zeros(0)
                lam_eq, lam_in = lam[: len(e)], lam[len(e):]
                neg = np.flatnonzero(lam_in < -1e-9 * (1.0 + np.linalg.norm(g)))
                if neg.size == 0:
                    status = SolverStatus.OPTIMAL
                    break
                drop = int(min(neg, key=lambda k: W[k])) if degenerate > 20 else int(neg[np.argmin(lam_in[neg])])
                W.pop(drop)
                continue

            alpha, block = (np.inf if unbounded_dir is not None else 1.0), -1
            Wset = set(W)
            Gp = G @ p if len(h) else np.zeros(0)
            slack = h - G @ x if len(h) else np.zeros(0)
            for i in np.flatnonzero(Gp > 1e-12 * pnorm):
                if i in Wset:
                    continue
                step = max(slack[i] / Gp[i], 0.0)
                if step < alpha or (step == alpha and block >= 0 and i < block):
                    alpha, block = step, int(i)
            if not np.isfinite(alpha):
                status = SolverStatus.UNBOUNDED
                break
            x = x + alpha * p
            degenerate = degenerate + 1 if alpha <= 1e-12 else 0

            added = False
            if block >= 0:
                if self._independent(E, G, W, block):
                    W.append(block)
                    added = True
                else:
                    rows = np.vstack([E, G[W]])
                    coef = np.linalg.lstsq(rows.T, G[block], rcond=None)[0][len(e):]
                    cand = [k for k in range(len(W)) if abs(coef[k]) > 1e-9]
                    if cand:
                        k = min(cand, key=lambda q: W[q])
                        W.pop(k)
                        W.append(block)
                        added = True
                        swaps += 1
            stalls = stalls + 1 if (alpha <= 1e-12 and not added) else 0
            trace[-1].update({"step": float(alpha), "blocking": int(block), "added": bool(added)})
            if stalls > 50:
                status = SolverStatus.NUMERICAL_ERROR
                break

        diagnostics = {"active_constraints_count": len(W), "iteration_trace": downsample_trace(trace),
                       "degenerate_swaps": swaps,
                       "dual_sign_convention": "minimize-normalized: y_i > 0 row at lower bound, y_i < 0 at upper bound"}
        result = SolverResult(status=status, iterations=it, runtime_seconds=time.time() - start, diagnostics=diagnostics)
        x = np.minimum(np.maximum(x, np.where(np.isfinite(cl), cl, -np.inf)), np.where(np.isfinite(cu), cu, np.inf))
        if status == SolverStatus.OPTIMAL:
            y = np.zeros(A.shape[0])
            for (kind, idx, sgn), val, nrm in zip(e_own, lam_eq, e_norm):
                if kind == "row":
                    y[idx] += -val / nrm
            for pos, val in zip(W, lam_in):
                kind, idx, sgn = g_own[pos]
                if kind == "row":
                    y[idx] += -sgn * val / g_norm[pos]
            # Remove the anti-degeneracy relaxation: re-solve the final active set exactly on the original data.
            polished = polish_solution(BoundedForm(model, scale=True), x, y)
            diagnostics["polished"] = polished is not None
            if polished is not None:
                x, y = polished
            grad = Q @ x + c
            d = grad - A.T @ y
            result.primal_solution = {nm: float(v) for nm, v in zip(names, x)}
            result.objective_value = float(sign * (0.5 * x @ Q @ x + c @ x) + model.objective.offset)
            result.dual_solution = {nm: float(v) for nm, v in zip(model.constraint_names, y)}
            result.reduced_costs = {nm: float(v) for nm, v in zip(names, d)}
        elif status in (SolverStatus.ITERATION_LIMIT, SolverStatus.TIME_LIMIT):
            result.primal_solution = {nm: float(v) for nm, v in zip(names, x)}
            result.objective_value = float(sign * (0.5 * x @ Q @ x + c @ x) + model.objective.offset)
            diagnostics["has_feasible_point"] = True
        return result

    @staticmethod
    def _independent(E: np.ndarray, G: np.ndarray, W: List[int], i: int) -> bool:
        rows = np.vstack([E, G[W]]) if (E.shape[0] or W) else np.zeros((0, G.shape[1]))
        if rows.shape[0] == 0:
            return True
        coef = np.linalg.lstsq(rows.T, G[i], rcond=None)[0]
        return float(np.linalg.norm(rows.T @ coef - G[i])) > 1e-8
