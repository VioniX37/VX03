"""
Primal-Dual Interior Point Method for convex Quadratic Programming.

Solves   min 1/2 x^T Q x + c^T x   s.t.  row_lb <= A x <= row_ub,  l <= x <= u
through the shared bounded form and the nonnegative variable transform:

- Mehrotra predictor-corrector on the full KKT conditions.
- Regularized quasi-definite augmented system
      [ -(Q + X^-1 S + W^-1 V + rho I)   A^T    ] [dx]   [r_d']
      [  A                               delta I ] [dy] = [r_p ]
  solved with dense LU or sparse SuperLU, so singular / semidefinite Q needs no model perturbation.
- Convexity verified by a Cholesky / LDL inertia test instead of a full eigen-decomposition.
- Divergence triggers a sovereign simplex feasibility check to classify infeasible vs unbounded.
"""
import time
from typing import Tuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus
from sovereign_opt.solvers.lp.standard_form import BoundedForm
from sovereign_opt.solvers.lp.interior_point import NonnegativeTransform, _max_step
from sovereign_opt.solvers.lp.simplex import downsample_trace
from sovereign_opt.solvers.lp.simplex_engine import SimplexEngine, LPStatus
from sovereign_opt.solvers.qp.polish import polish_solution


def check_convexity(Q: sp.spmatrix, tol: float = 1e-8) -> Tuple[bool, float]:
    """
    Returns (is_convex, min_pivot_estimate). Uses the variables that actually appear in Q.
    Small blocks: dense eigenvalues. Large blocks: symmetric-mode sparse LU inertia.
    """
    Q = sp.csc_matrix(Q)
    if Q.nnz == 0:
        return True, 0.0
    active = np.unique(np.concatenate([Q.tocoo().row, Q.tocoo().col]))
    Qa = Q[active][:, active]
    scale = max(1.0, float(abs(Qa).max()))
    if len(active) <= 1500:
        ev = np.linalg.eigvalsh(Qa.toarray())
        return bool(ev.min() >= -tol * scale), float(ev.min())
    shift = 1e-9 * scale
    try:
        lu = spla.splu(sp.csc_matrix(Qa + shift * sp.identity(len(active))), diag_pivot_thresh=0.0,
                       permc_spec="MMD_AT_PLUS_A", options=dict(SymmetricMode=True))
    except RuntimeError:
        return True, 0.0  # singular PSD matrices may fail to factor; treated as semidefinite
    piv = lu.U.diagonal()
    return bool(piv.min() >= -tol * scale), float(piv.min())


def _mehrotra_start(A, AT, b, c, Q, U, u):
    """
    Mehrotra-style centered starting point:
    least-norm primal / least-squares dual estimates, projected into the positive orthant,
    then shifted so all complementarity products are balanced.
    Upper-bounded variables start at the interval midpoint so z_U + w = u holds exactly.
    """
    m, K = A.shape
    nU = len(U)
    if m:
        M = (A @ AT).toarray() if sp.issparse(A) else A @ AT
        M = M + 1e-8 * max(1.0, float(np.max(np.abs(np.diag(M))))) * np.eye(m)
        try:
            cho = la.cho_factor(M, check_finite=False)
            z = AT @ la.cho_solve(cho, b, check_finite=False)
            y = la.cho_solve(cho, A @ c, check_finite=False)
        except la.LinAlgError:
            z = np.zeros(K)
            y = np.zeros(m)
    else:
        z = np.zeros(K)
        y = np.zeros(0)
    s = c - AT @ y if m else c.copy()

    dz = max(-1.5 * float(np.min(z, initial=0.0)), 0.0)
    ds = max(-1.5 * float(np.min(s, initial=0.0)), 0.0)
    z = z + dz
    s = s + ds
    xs = float(z @ s)
    z = z + 0.5 * xs / max(float(np.sum(s)), 1e-12)
    s = s + 0.5 * xs / max(float(np.sum(z)), 1e-12)
    z = np.maximum(z, 1.0)
    s = np.maximum(s, 1.0)
    if nU:
        z[U] = 0.5 * u
        w = 0.5 * u
        v = np.maximum(s[U], 1.0)
    else:
        w = np.zeros(0)
        v = np.zeros(0)
    # balance complementarity: scale duals so the mean products match
    mu_x = float(np.mean(z * s))
    if nU:
        mu_w = float(np.mean(w * v))
        v = v * (mu_x / max(mu_w, 1e-300))
        v = np.maximum(v, 1e-8)
    return z, y, s, w, v


def qp_ipm(tr: NonnegativeTransform, tol: float = 1e-8, max_iterations: int = 200, deadline: float = np.inf):
    A, b, c, U, u = tr.A, tr.b, tr.c, tr.U, tr.u
    Q = tr.Q if tr.Q is not None else sp.csc_matrix((tr.K, tr.K))
    m, K = A.shape
    nU = len(U)
    AT = sp.csr_matrix(A.T)
    dense = (m + K) <= 1500
    Qd = Q.toarray() if dense else None
    Ad = A.toarray() if dense else None

    z, y, s, w, v = _mehrotra_start(A, AT, b, c, Q, U, u)
    z_prev, y_prev = z, y
    mu0 = 1.0
    nvars = max(K + nU, 1)
    bnorm = 1.0 + max(np.linalg.norm(b, np.inf) if m else 0.0, np.linalg.norm(u, np.inf) if nU else 0.0)
    cnorm = 1.0 + (np.linalg.norm(c, np.inf) if K else 0.0)
    rho, delta = 1e-10, 1e-10
    trace = []
    alpha_hist = []
    status = LPStatus.ITERATION_LIMIT

    def Ev(vec):
        out = np.zeros(K)
        out[U] = vec
        return out

    for it in range(1, max_iterations + 1):
        Qz = Q @ z
        rp = b - A @ z
        ru = u - z[U] - w
        rd = c + Qz - AT @ y - s + Ev(v)
        mu = (z @ s + w @ v) / nvars
        pobj = c @ z + 0.5 * z @ Qz
        dobj = b @ y - u @ v - 0.5 * z @ Qz
        pinf = max(np.linalg.norm(rp, np.inf) if m else 0.0, np.linalg.norm(ru, np.inf) if nU else 0.0) / bnorm
        dinf = (np.linalg.norm(rd, np.inf) if K else 0.0) / cnorm
        gap = abs(pobj - dobj) / (1.0 + abs(pobj))
        trace.append({"iteration": it, "mu": float(mu), "norm_rp": float(pinf), "norm_rd": float(dinf),
                      "rel_gap": float(gap), "objective_internal": float(pobj),
                      "max_x": float(np.max(z, initial=0.0)), "max_y": float(np.max(np.abs(y), initial=0.0)),
                      "step": alpha_hist[-1] if alpha_hist else 0.0})
        if not (np.isfinite(pinf) and np.isfinite(dinf) and np.isfinite(mu)):
            # Iterates overflowed (typical for infeasible / unbounded problems): keep the last finite iterate.
            z, y = z_prev, y_prev
            trace.pop()
            status = "infeasible_or_unbounded"
            break
        if it == 1:
            mu0 = max(mu, 1e-300)
        if pinf <= tol and dinf <= tol and gap <= tol:
            status = LPStatus.OPTIMAL
            break
        # Divergence or complementarity collapse without feasibility: the problem is infeasible or unbounded.
        if (K and np.max(z) > 1e10 * bnorm) or (m and np.max(np.abs(y)) > 1e10 * cnorm) or (it > 5 and mu <= 1e-15 * mu0):
            status = "infeasible_or_unbounded"
            break
        if time.time() > deadline:
            status = LPStatus.TIME_LIMIT
            break

        g = v / w if nU else np.zeros(0)
        Dinv = s / z
        Dinv[U] += g
        try:
            if dense:
                H = -(Qd + np.diag(Dinv + rho))
                KKT = np.block([[H, Ad.T], [Ad, delta * np.eye(m)]])
                lu_piv = la.lu_factor(KKT, check_finite=False)
                solve = lambda rhs: la.lu_solve(lu_piv, rhs, check_finite=False)
            else:
                H = -(Q + sp.diags(Dinv + rho))
                KKT = sp.block_array([[H, A.T], [A, delta * sp.identity(m)]], format="csc")
                lu = spla.splu(KKT, permc_spec="MMD_AT_PLUS_A")
                solve = lu.solve
        except (RuntimeError, la.LinAlgError, ValueError):
            status = LPStatus.NUMERICAL_ERROR
            break

        def direction(rzs, rwv):
            r3 = rd - rzs / z
            if nU:
                r3[U] += (rwv - v * ru) / w
            sol = solve(np.concatenate([r3, rp]))
            dz, dy = sol[:K], sol[K:]
            ds = (rzs - s * dz) / z
            dw = ru - dz[U]
            dv = (rwv - v * dw) / w if nU else np.zeros(0)
            return dz, dy, ds, dw, dv

        def step(dz, ds, dw, dv):
            return min(_max_step(z, dz), _max_step(s, ds),
                       _max_step(w, dw) if nU else np.inf, _max_step(v, dv) if nU else np.inf)

        dz_a, dy_a, ds_a, dw_a, dv_a = direction(-z * s, -w * v)
        a_aff = min(1.0, step(dz_a, ds_a, dw_a, dv_a))
        mu_aff = ((z + a_aff * dz_a) @ (s + a_aff * ds_a) + (w + a_aff * dw_a) @ (v + a_aff * dv_a)) / nvars
        sigma = float(np.clip((mu_aff / mu) ** 3, 1e-8, 1.0))
        dz, dy, ds, dw, dv = direction(sigma * mu - z * s - dz_a * ds_a, sigma * mu - w * v - dw_a * dv_a)
        alpha = min(1.0, 0.995 * step(dz, ds, dw, dv))
        alpha_hist.append(float(alpha))
        z_prev, y_prev = z, y
        z = z + alpha * dz
        y = y + alpha * dy
        s = s + alpha * ds
        if nU:
            w = w + alpha * dw
            v = v + alpha * dv

    return status, z, y, trace


def _is_recession_direction(tr: NonnegativeTransform, z: np.ndarray) -> bool:
    """
    Verify that the (diverged) iterate z points along a direction of unbounded descent:
    d = z / ||z||_inf with d >= 0, A d ~ 0, d_U ~ 0, Q d ~ 0 and c^T d < 0.
    Tolerances shrink with ||z|| because A z, z_U and Q z stay bounded along a true recession ray.
    """
    if z is None or z.size == 0 or not np.all(np.isfinite(z)):
        return False
    scale = float(np.max(np.abs(z)))
    if scale < 1e5:
        return False
    d = z / scale
    bnorm = 1.0 + float(np.max(np.abs(tr.b), initial=0.0)) + float(np.max(np.abs(tr.u), initial=0.0))
    cnorm = 1.0 + float(np.max(np.abs(tr.c), initial=0.0))
    if tr.A.shape[0] and float(np.max(np.abs(tr.A @ d))) > 1e-9 + 10.0 * bnorm / scale:
        return False
    if len(tr.U) and float(np.max(np.abs(d[tr.U]))) > 1e-9 + 10.0 * bnorm / scale:
        return False
    if tr.Q is not None and tr.Q.nnz and float(np.max(np.abs(tr.Q @ d))) > 1e-9 + 10.0 * cnorm / scale:
        return False
    return float(tr.c @ d) < -1e-9 * cnorm


class QPInteriorPointSolver(SolverBase):
    """Sovereign primal-dual interior point solver for convex QP (and LP)."""

    def __init__(self, max_iterations: int = 200, tolerance: float = 1e-8, scaling: bool = True):
        super().__init__(name="InteriorPointQP")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.scaling = scaling

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 60.0, **kwargs) -> SolverResult:
        start = time.time()
        form = BoundedForm(model, scale=self.scaling)
        if form.Q_orig is not None:
            convex, min_eig = check_convexity(form.Q_orig)
            if not convex:
                return SolverResult(
                    status=SolverStatus.NUMERICAL_ERROR, runtime_seconds=time.time() - start,
                    diagnostics={"error": "Non-convex QP: Q matrix is not positive semidefinite.",
                                 "min_eigenvalue": min_eig},
                )
        if np.any(form.lb > form.ub + 1e-9):
            return SolverResult(status=SolverStatus.INFEASIBLE, runtime_seconds=time.time() - start,
                                diagnostics={"infeasibility_certificate": "a variable or row has lower bound > upper bound"})

        tr = NonnegativeTransform(form, Q_struct=form.Q)
        status, zsol, ysol, trace = qp_ipm(tr, tol=self.tolerance, max_iterations=self.max_iterations,
                                           deadline=start + time_limit_seconds)
        out_trace = []
        for t in trace:
            e = dict(t)
            e["objective"] = form.obj_sign * (e.pop("objective_internal") + tr.obj_const) + form.offset
            out_trace.append(e)
        diagnostics = {
            "iteration_trace": downsample_trace(out_trace),
            "primal_residual": trace[-1]["norm_rp"] if trace else 0.0,
            "dual_residual": trace[-1]["norm_rd"] if trace else 0.0,
            "barrier_mu": trace[-1]["mu"] if trace else 0.0,
            "algorithm": "primal-dual Mehrotra predictor-corrector on regularized KKT system",
            "dual_sign_convention": "minimize-normalized: y_i > 0 row at lower bound, y_i < 0 at upper bound",
        }

        if status == LPStatus.OPTIMAL:
            x = form.unscale_x(tr.to_x(zsol))
            xs = x[: form.n]
            y = form.unscale_y(ysol)
            polished = polish_solution(form, xs, y)
            diagnostics["polished"] = polished is not None
            if polished is not None:
                xs, y = polished
            grad = form.c_orig + (form.Q_orig @ xs if form.Q_orig is not None else 0.0)
            d = grad - form.A_orig.T @ y
            return SolverResult(
                status=SolverStatus.OPTIMAL,
                objective_value=form.user_objective(xs),
                primal_solution={nm: float(val) for nm, val in zip(form.var_names, xs)},
                dual_solution={nm: float(val) for nm, val in zip(form.con_names, y)},
                reduced_costs={nm: float(val) for nm, val in zip(form.var_names, d)},
                iterations=len(trace),
                runtime_seconds=time.time() - start,
                diagnostics=diagnostics,
            )

        final = SolverStatus.TIME_LIMIT if status == LPStatus.TIME_LIMIT else None
        if final is None:
            # Not converged: decide feasibility with the sovereign simplex (zero objective), then
            # declare unboundedness only with a verified recession direction.
            engine = SimplexEngine(form.A, np.zeros(form.N), form.lb, form.ub, form.n)
            feas = engine.solve(time_limit=max(1.0, time_limit_seconds - (time.time() - start)))
            if feas == LPStatus.INFEASIBLE:
                final = SolverStatus.INFEASIBLE
                diagnostics["infeasibility_certificate"] = "simplex phase 1 proved the feasible region empty"
            elif feas == LPStatus.OPTIMAL and _is_recession_direction(tr, zsol):
                final = SolverStatus.UNBOUNDED
                diagnostics["unboundedness_certificate"] = (
                    "feasible point exists and a verified recession direction d satisfies A d = 0, Q d = 0, c^T d < 0"
                )
            elif status == LPStatus.ITERATION_LIMIT:
                final = SolverStatus.ITERATION_LIMIT
            else:
                final = SolverStatus.NUMERICAL_ERROR
                diagnostics["note"] = f"interior point stopped with '{status}'; simplex feasibility check returned '{feas}'"
        return SolverResult(status=final, iterations=len(trace), runtime_seconds=time.time() - start,
                            diagnostics=diagnostics)
