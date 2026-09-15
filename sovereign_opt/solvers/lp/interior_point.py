"""
Homogeneous Self-Dual Interior Point Method (IPM) for Linear Programming (v2).

Implemented from mathematical foundations:
- Homogeneous self-dual embedding with native upper bounds (Xu-Hung-Ye / Andersen-Andersen),
  so primal or dual infeasibility is *certified* instead of the iterates diverging.
- Mehrotra predictor-corrector with a common primal-dual step and fraction-to-boundary 0.995.
- Normal equations A D A^T with primal/dual regularization, solved by dense Cholesky
  or sparse LU depending on size.
- Power-of-two geometric scaling (shared bounded form).
- Crossover: an IPM solution is converted into an optimal basis with a few primal simplex
  pivots, giving vertex solutions and exact duals.
"""
import time
from typing import Optional, Tuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus
from sovereign_opt.solvers.lp.standard_form import BoundedForm
from sovereign_opt.solvers.lp.simplex_engine import SimplexEngine, LPStatus, AT_LOWER, AT_UPPER, FREE_ZERO, BASIC
from sovereign_opt.solvers.lp.simplex import lp_result_from_engine, downsample_trace


class NonnegativeTransform:
    """
    Maps the bounded form (columns with arbitrary bounds) to a problem in variables z >= 0:
        x = shift + T z,  with z_U <= u for the subset U of columns that keep an upper bound.
    Fixed columns are eliminated, upper-only columns are negated, free columns are split.
    """

    def __init__(self, form: BoundedForm, Q_struct: Optional[sp.spmatrix] = None):
        A, lb, ub, c = form.A, form.lb, form.ub, form.c
        N = form.N
        fl, fu = np.isfinite(lb), np.isfinite(ub)
        fixed = fl & fu & (ub - lb <= 1e-13 * np.maximum(1.0, np.abs(lb)))
        lower = fl & ~fixed
        upper_only = ~fl & fu
        free = ~fl & ~fu
        idx_l, idx_uo, idx_f = np.flatnonzero(lower), np.flatnonzero(upper_only), np.flatnonzero(free)
        src = np.concatenate([idx_l, idx_uo, idx_f, idx_f])
        sign = np.concatenate([np.ones(len(idx_l)), -np.ones(len(idx_uo)), np.ones(len(idx_f)), -np.ones(len(idx_f))])
        K = len(src)
        self.K = K
        self.T = sp.csc_matrix((sign, (src, np.arange(K))), shape=(N, K))
        shift = np.zeros(N)
        shift[lower] = lb[lower]
        shift[upper_only] = ub[upper_only]
        shift[fixed] = 0.5 * (lb[fixed] + ub[fixed])
        self.shift = shift
        self.A = sp.csc_matrix(A @ self.T)
        self.b = -(A @ shift)

        has_u = np.isfinite(ub[idx_l])
        self.U = np.flatnonzero(has_u)  # positions within z (the first len(idx_l) columns)
        self.u = (ub[idx_l] - lb[idx_l])[has_u]

        c_eff = c.copy()
        self.Q = None
        self.obj_const = float(c @ shift)
        if Q_struct is not None:
            n = form.n
            Qf = sp.block_array([[Q_struct, None], [None, sp.csc_matrix((N - n, N - n))]], format="csc")
            Qs = Qf @ shift
            c_eff = c_eff + Qs
            self.obj_const += 0.5 * float(shift @ Qs)
            self.Q = sp.csc_matrix(self.T.T @ Qf @ self.T)
        self.c = self.T.T @ c_eff

    def to_x(self, z: np.ndarray) -> np.ndarray:
        return self.shift + self.T @ z


class NormalEquations:
    """Factorization of M = A diag(D) A^T + reg*I (dense Cholesky or sparse LU)."""

    def __init__(self, A: sp.csc_matrix, dense_threshold: int = 1200):
        self.A = A
        self.m = A.shape[0]
        density = A.nnz / max(1, A.shape[0] * A.shape[1])
        self.dense = self.m <= dense_threshold or density > 0.2
        self.Ad = A.toarray() if self.dense else None
        self.reg = 1e-10

    def factor(self, D: np.ndarray):
        if self.m == 0:
            return
        if self.dense:
            M = (self.Ad * D) @ self.Ad.T
            scale = max(1.0, float(np.max(np.abs(np.diag(M)))))
            reg = self.reg * scale
            for _ in range(8):
                try:
                    self.cho = la.cho_factor(M + reg * np.eye(self.m), lower=False, check_finite=False)
                    return
                except la.LinAlgError:
                    reg *= 100.0
            self.cho = None
            self.lstsq_M = M + reg * np.eye(self.m)
        else:
            M = sp.csc_matrix(self.A @ sp.diags(D) @ self.A.T)
            scale = max(1.0, float(np.max(np.abs(M.diagonal()))))
            reg = self.reg * scale
            for _ in range(8):
                try:
                    self.lu = spla.splu(M + reg * sp.identity(self.m, format="csc"), permc_spec="MMD_AT_PLUS_A")
                    return
                except RuntimeError:
                    reg *= 100.0
            raise np.linalg.LinAlgError("normal equations factorization failed")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self.m == 0:
            return np.zeros(0)
        if self.dense:
            if self.cho is None:
                return np.linalg.lstsq(self.lstsq_M, rhs, rcond=None)[0]
            return la.cho_solve(self.cho, rhs, check_finite=False)
        return self.lu.solve(rhs)


def _max_step(v: np.ndarray, dv: np.ndarray) -> float:
    neg = dv < 0
    if not neg.any():
        return np.inf
    return float(np.min(-v[neg] / dv[neg]))


def homogeneous_ipm(
    tr: NonnegativeTransform,
    tol: float = 1e-8,
    max_iterations: int = 200,
    deadline: float = np.inf,
):
    """
    Solve min c^T z s.t. A z = b, 0 <= z, z_U <= u by the homogeneous self-dual algorithm.
    Returns (status, z, y, trace) with status in LPStatus or 'infeasible_or_unbounded'.
    """
    A, b, c, U, u = tr.A, tr.b, tr.c, tr.U, tr.u
    m, K = A.shape
    nU = len(U)
    AT = sp.csr_matrix(A.T)
    ne = NormalEquations(A)

    # Bound-consistent start: x_U + w = u tau holds exactly at tau = 1.
    x, z = np.ones(K), np.ones(K)
    if nU:
        x[U] = np.maximum(np.minimum(1.0, 0.5 * u), 1e-4)
    w = np.maximum(u - x[U], 1e-4) if nU else np.zeros(0)
    v = np.ones(nU)
    y = np.zeros(m)
    tau, kappa = 1.0, 1.0
    nvars = K + nU + 1
    bnorm = 1.0 + max(np.linalg.norm(b, np.inf) if m else 0.0, np.linalg.norm(u, np.inf) if nU else 0.0)
    cnorm = 1.0 + (np.linalg.norm(c, np.inf) if K else 0.0)
    mu0 = (x @ z + w @ v + tau * kappa) / nvars
    trace = []
    status = LPStatus.ITERATION_LIMIT

    def Ev(vec):
        out = np.zeros(K)
        out[U] = vec
        return out

    for it in range(1, max_iterations + 1):
        rp = b * tau - A @ x
        ru = u * tau - x[U] - w
        rd = c * tau - AT @ y - z + Ev(v)
        rg = kappa + c @ x - b @ y + u @ v
        mu = (x @ z + w @ v + tau * kappa) / nvars

        pinf = max(np.linalg.norm(rp, np.inf) if m else 0.0, np.linalg.norm(ru, np.inf) if nU else 0.0) / tau / bnorm
        dinf = (np.linalg.norm(rd, np.inf) if K else 0.0) / tau / cnorm
        pobj = c @ x / tau
        dobj = (b @ y - u @ v) / tau
        gap = abs(pobj - dobj) / (1.0 + abs(pobj))
        trace.append({"iteration": it, "mu": float(mu), "norm_rp": float(pinf), "norm_rd": float(dinf),
                      "rel_gap": float(gap), "tau": float(tau), "kappa": float(kappa), "objective_internal": float(pobj)})

        if pinf <= tol and dinf <= tol and gap <= tol:
            status = LPStatus.OPTIMAL
            break
        if not (np.isfinite(pinf) and np.isfinite(dinf) and np.isfinite(mu) and np.isfinite(tau)):
            status = LPStatus.NUMERICAL_ERROR
            break

        # --- infeasibility certificates (tau -> 0 while kappa stays positive)
        if tau <= 1e-8 * max(1.0, kappa) and mu <= 1e-8 * mu0:
            dual_ray_obj = b @ y - u @ v
            primal_ray_obj = -(c @ x)
            dres = (np.linalg.norm(AT @ y + z - Ev(v), np.inf) if K else 0.0)
            pres = max(np.linalg.norm(A @ x, np.inf) if m else 0.0, np.linalg.norm(x[U] + w, np.inf) if nU else 0.0)
            if dual_ray_obj > 0 and dres <= 1e-7 * dual_ray_obj * cnorm:
                status = LPStatus.INFEASIBLE
                break
            if primal_ray_obj > 0 and pres <= 1e-7 * primal_ray_obj * bnorm:
                status = LPStatus.UNBOUNDED
                break
            if tau <= 1e-13 * max(1.0, kappa):
                status = "infeasible_or_unbounded"
                break
        if time.time() > deadline:
            status = LPStatus.TIME_LIMIT
            break

        g = v / w if nU else np.zeros(0)
        Dinv = z / x
        Dinv[U] += g
        D = 1.0 / (Dinv + 1e-12)
        h2 = c.copy()
        h2[U] += g * u
        h = c.copy()
        h[U] -= g * u
        try:
            ne.factor(D)
        except np.linalg.LinAlgError:
            status = LPStatus.NUMERICAL_ERROR
            break
        q = ne.solve(A @ (D * h) + b)
        dx_q = D * (AT @ q - h)
        denom_const = -h2 @ dx_q + b @ q + u @ (g * u) + kappa / tau

        def direction(eta, rxz, rwv, rtk):
            rhs3 = eta * rd - rxz / x
            if nU:
                rhs3[U] += (rwv - eta * v * ru) / w
            p = ne.solve(eta * rp + A @ (D * rhs3))
            dx_p = D * (AT @ p - rhs3)
            numer = eta * rg + (u @ ((rwv - eta * v * ru) / w) if nU else 0.0) + rtk / tau + h2 @ dx_p - b @ p
            dtau = numer / denom_const
            dx = dx_p + dx_q * dtau
            dy = p + q * dtau
            dz = (rxz - z * dx) / x
            dw = eta * ru - dx[U] + u * dtau
            dv = (rwv - v * dw) / w if nU else np.zeros(0)
            dkappa = (rtk - kappa * dtau) / tau
            return dx, dy, dz, dw, dv, dtau, dkappa

        def step_len(dx, dz, dw, dv, dtau, dkappa):
            a = min(_max_step(x, dx), _max_step(z, dz), _max_step(w, dw) if nU else np.inf,
                    _max_step(v, dv) if nU else np.inf,
                    -tau / dtau if dtau < 0 else np.inf, -kappa / dkappa if dkappa < 0 else np.inf)
            return a

        # predictor
        aff = direction(1.0, -x * z, -w * v, -tau * kappa)
        dx_a, dy_a, dz_a, dw_a, dv_a, dt_a, dk_a = aff
        a_aff = min(1.0, step_len(dx_a, dz_a, dw_a, dv_a, dt_a, dk_a))
        mu_aff = ((x + a_aff * dx_a) @ (z + a_aff * dz_a) + (w + a_aff * dw_a) @ (v + a_aff * dv_a)
                  + (tau + a_aff * dt_a) * (kappa + a_aff * dk_a)) / nvars
        sigma = float(np.clip((mu_aff / mu) ** 3, 1e-6, 1.0))
        eta = 1.0 - sigma

        rxz = sigma * mu - x * z - dx_a * dz_a
        rwv = sigma * mu - w * v - dw_a * dv_a
        rtk = sigma * mu - tau * kappa - dt_a * dk_a
        dirn = direction(eta, rxz, rwv, rtk)
        alpha = min(1.0, 0.995 * step_len(dirn[0], dirn[2], dirn[3], dirn[4], dirn[5], dirn[6]))

        dx, dy, dz, dw, dv, dt, dk = dirn
        x = x + alpha * dx
        y = y + alpha * dy
        z = z + alpha * dz
        if nU:
            w = w + alpha * dw
            v = v + alpha * dv
        tau = tau + alpha * dt
        kappa = kappa + alpha * dk
        # keep strictly positive
        x = np.maximum(x, 1e-300)
        z = np.maximum(z, 1e-300)
        tau = max(tau, 1e-300)
        kappa = max(kappa, 1e-300)

        # normalize the homogeneous iterate to avoid overflow
        s = max(tau, 1.0) if tau > 1e6 else 1.0
        if s != 1.0:
            x, y, z, w, v, tau, kappa = x / s, y / s, z / s, w / s, v / s, tau / s, kappa / s

    return status, x / tau, y / tau, trace


def crossover(form: BoundedForm, x_full: np.ndarray, time_limit: float = 30.0) -> Tuple[Optional[SimplexEngine], Optional[str]]:
    """Recover an optimal basis from an interior solution via weighted QR column selection + primal simplex."""
    m, N = form.A.shape
    if m * N > 3e7:
        return None, None
    lb, ub = form.lb, form.ub
    dist = np.minimum(np.where(np.isfinite(lb), x_full - lb, np.inf), np.where(np.isfinite(ub), ub - x_full, np.inf))
    interior = np.where(np.isinf(dist), 1.0 + np.abs(x_full), dist / (1.0 + np.abs(x_full)))
    weights = np.clip(interior, 0.0, 1e3) + 1e-9
    Ad = form.A.toarray() * weights
    _, _, piv = la.qr(Ad, pivoting=True, mode="economic", check_finite=False)
    head = np.array(piv[:m], dtype=np.int64)

    engine = SimplexEngine.from_form(form)
    engine.x = x_full.copy()
    status = np.empty(N, dtype=np.int8)
    for j in range(N):
        l, h, val = lb[j], ub[j], x_full[j]
        if np.isfinite(l) and np.isfinite(h):
            status[j] = AT_LOWER if (val - l) <= (h - val) else AT_UPPER
        elif np.isfinite(l):
            status[j] = AT_LOWER
        elif np.isfinite(h):
            status[j] = AT_UPPER
        else:
            status[j] = FREE_ZERO
    status[head] = BASIC
    engine.set_basis(head, status)
    st = engine.solve(time_limit=time_limit, method="primal")
    return engine, st


class InteriorPointSolver(SolverBase):
    """
    Sovereign Homogeneous Self-Dual Interior Point Method (Mehrotra predictor-corrector), with crossover.
    """

    def __init__(
        self,
        max_iterations: int = 200,
        tolerance: float = 1e-8,
        step_fraction: float = 0.995,
        crossover: bool = True,
        scaling: bool = True,
    ):
        super().__init__(name="InteriorPoint")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_fraction = step_fraction
        self.do_crossover = crossover
        self.scaling = scaling

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 60.0, **kwargs) -> SolverResult:
        start = time.time()
        if model.objective.is_quadratic:
            return SolverResult(status=SolverStatus.NUMERICAL_ERROR,
                                diagnostics={"error": "Quadratic objective: use the QP interior point solver."})
        form = BoundedForm(model, scale=self.scaling)
        if np.any(form.lb > form.ub + 1e-9):
            return SolverResult(status=SolverStatus.INFEASIBLE, runtime_seconds=time.time() - start,
                                diagnostics={"infeasibility_certificate": "a variable or row has lower bound > upper bound"})

        tr = NonnegativeTransform(form)
        status, zsol, ysol, trace = homogeneous_ipm(tr, tol=self.tolerance, max_iterations=self.max_iterations,
                                                    deadline=start + time_limit_seconds)
        ipm_trace = []
        for t in trace:
            e = dict(t)
            e["objective"] = form.obj_sign * (e.pop("objective_internal") + tr.obj_const) + form.offset
            ipm_trace.append(e)
        diagnostics = {
            "ipm_iterations": len(trace),
            "iteration_trace": downsample_trace(ipm_trace),
            "primal_residual": trace[-1]["norm_rp"] if trace else 0.0,
            "dual_residual": trace[-1]["norm_rd"] if trace else 0.0,
            "barrier_mu": trace[-1]["mu"] if trace else 0.0,
            "algorithm": "homogeneous self-dual embedding, Mehrotra predictor-corrector",
            "dual_sign_convention": "minimize-normalized: y_i > 0 row at lower bound, y_i < 0 at upper bound",
        }

        if status == LPStatus.OPTIMAL:
            x_full = tr.to_x(zsol)
            if self.do_crossover:
                remaining = max(1.0, time_limit_seconds - (time.time() - start))
                engine, cst = crossover(form, x_full, time_limit=remaining)
                if engine is not None and cst == LPStatus.OPTIMAL:
                    res = lp_result_from_engine(form, engine, cst, start)
                    res.diagnostics.update({k: v for k, v in diagnostics.items() if k != "iteration_trace"})
                    res.diagnostics["iteration_trace"] = diagnostics["iteration_trace"]
                    res.diagnostics["crossover_pivots"] = int(engine.total_iterations)
                    res.iterations = len(trace)
                    return res
                diagnostics["crossover"] = "skipped" if engine is None else f"failed ({cst}); returning interior solution"
            x = form.unscale_x(x_full)
            xs = x[: form.n]
            y = form.unscale_y(ysol)
            from sovereign_opt.solvers.qp.polish import polish_solution  # local import: qp package imports this module

            polished = polish_solution(form, xs, y)
            diagnostics["polished"] = polished is not None
            if polished is not None:
                xs, y = polished
            grad = form.c_orig.copy()
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

        if status in (LPStatus.UNBOUNDED, "infeasible_or_unbounded"):
            # A primal ray only proves DUAL infeasibility; unboundedness also needs a feasible point.
            engine = SimplexEngine(form.A, np.zeros(form.N), form.lb, form.ub, form.n)
            feas = engine.solve(time_limit=max(1.0, time_limit_seconds - (time.time() - start)))
            if feas == LPStatus.INFEASIBLE:
                status = LPStatus.INFEASIBLE
                diagnostics["infeasibility_certificate"] = "simplex phase 1 proved the feasible region empty (problem is also dual infeasible)"
            elif feas == LPStatus.OPTIMAL:
                status = LPStatus.UNBOUNDED
                diagnostics["unboundedness_certificate"] = "homogeneous IPM recession ray + simplex-verified feasible point"

        mapping = {
            LPStatus.INFEASIBLE: SolverStatus.INFEASIBLE,
            LPStatus.UNBOUNDED: SolverStatus.UNBOUNDED,
            "infeasible_or_unbounded": SolverStatus.INFEASIBLE_OR_UNBOUNDED,
            LPStatus.TIME_LIMIT: SolverStatus.TIME_LIMIT,
            LPStatus.ITERATION_LIMIT: SolverStatus.ITERATION_LIMIT,
            LPStatus.NUMERICAL_ERROR: SolverStatus.NUMERICAL_ERROR,
        }
        return SolverResult(
            status=mapping.get(status, SolverStatus.NUMERICAL_ERROR),
            iterations=len(trace),
            runtime_seconds=time.time() - start,
            diagnostics=diagnostics,
        )
