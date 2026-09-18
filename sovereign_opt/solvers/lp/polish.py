"""
Sparse vertex polishing: turn an approximate PDLP point into an exact optimal vertex.

    min  c^T x   s.t.  l_r <= A x <= u_r,   l_x <= x <= u_x

PDLP stops at a point whose primal residual, dual residual and gap are all below `tol`. That is
close to optimal but it is not a vertex, so its objective differs from a simplex / barrier solver
in the 7th or 8th digit. Pushing PDLP itself further is slow (first-order methods converge
linearly at best), and the dense crossover in interior_point.py cannot hold a 100k-column LP.

This module finishes the job with sparse linear algebra only, the way an active-set method does:

1. Identify.  From complementarity, guess which columns sit at a bound (N) and which rows are
   active (R). The rest of the columns are free (F).
2. Project.   Move x_F the least distance onto { A_RF x_F = b_R - A_RN x_N }: one solve with
   M = A_RF A_RF^T (sparse, symmetric, factored with a minimum-degree ordering).
3. Push.      The least-squares dual y_R = argmin ||c_F - A_RF^T y_R|| leaves reduced costs d_F
   that lie in the null space of A_RF, so -d_F is a feasible descent direction on the face.
   Step along it until the first column or inactive row blocks, fix that one, repeat. Each step
   strictly lowers the objective and shrinks the face; when d_F = 0 the point is a vertex
   (or an optimal face) of the current active set.
4. Price.     A column at a bound whose reduced cost has the wrong sign, or an inequality row
   whose dual has the wrong sign, is released and step 3 continues.
5. Certify.   Primal feasibility, dual feasibility and the duality gap are measured in the
   ORIGINAL space. The polished point is returned only if all three pass `tol`; otherwise the
   caller keeps its PDLP point. Nothing is ever reported as optimal on the strength of a guess.

Dual convention matches pdlp.py: y_i > 0 row at lower bound, y_i < 0 at upper; d = c - A^T y.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class PolishResult:
    ok: bool                      # certified optimal vertex within tol
    x: np.ndarray
    y: np.ndarray
    primal_objective: float
    dual_objective: float
    max_primal_violation: float   # absolute, original space
    max_dual_violation: float
    rel_gap: float
    push_steps: int
    factorizations: int
    runtime: float
    reason: str = ""
    info: Dict[str, Any] = field(default_factory=dict)


class _Face:
    """A with its entries as COO triplets, so any (rows R, cols F) block is built by masking."""

    def __init__(self, A: sp.spmatrix):
        coo = sp.coo_matrix(A)
        self.m, self.n = coo.shape
        self.r, self.c, self.v = coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data
        self.csc = sp.csc_matrix(A)

    def columns_on(self, R: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Dense |R| x len(cols) block: the given columns restricted to rows R."""
        rmap = np.cumsum(R) - 1
        out = np.zeros((int(R.sum()), len(cols)))
        ip, ix, dv = self.csc.indptr, self.csc.indices, self.csc.data
        for k, j in enumerate(cols):
            rows = ix[ip[j]:ip[j + 1]]
            keep = R[rows]
            out[rmap[rows[keep]], k] = dv[ip[j]:ip[j + 1]][keep]
        return out

    def block(self, R: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
        keep = R[self.r] & F[self.c]
        rmap = np.cumsum(R) - 1
        cmap = np.cumsum(F) - 1
        return sp.csr_matrix((self.v[keep], (rmap[self.r[keep]], cmap[self.c[keep]])),
                             shape=(int(R.sum()), int(F.sum())))


class _Normal:
    """Solves with M = B B^T (B = A_RF): regularised LU plus iterative refinement."""

    def __init__(self, B: sp.csr_matrix, reg: float = 1e-12):
        self.B = B
        self.M = (B @ B.T).tocsc()
        k = self.M.shape[0]
        diag = self.M.diagonal()
        # empty rows of B (all their free columns fixed) give zero diagonal: pin them
        shift = reg * max(float(diag.max()) if k else 1.0, 1.0)
        self.lu = spla.splu((self.M + sp.diags(np.where(diag > 0, shift, 1.0))).tocsc(),
                            permc_spec="MMD_AT_PLUS_A")
        self.empty_rows = np.diff(B.indptr) == 0

    def solve(self, rhs: np.ndarray, refine: int = 3) -> np.ndarray:
        z = self.lu.solve(rhs)
        for _ in range(refine):
            z += self.lu.solve(rhs - self.M @ z)
        return z


class _UpdatedNormal:
    """
    M = B B^T after a few columns of the factored B0 were dropped or added:
        M = M0 + U S U^T,   S = diag(+1 added, -1 dropped)
    solved by Woodbury on the factorization of M0, then refined against the true M. A push step
    or a pricing round changes a handful of columns, so this replaces a refactorization by one
    extra triangular solve per changed column.
    """

    def __init__(self, base: _Normal, B: sp.csr_matrix, U: np.ndarray, W: np.ndarray, sgn: np.ndarray):
        self.base, self.B = base, B
        self.U, self.W = U, W
        self.cap = np.diag(1.0 / sgn) + U.T @ W       # capacitance S^-1 + U^T M0^-1 U

    def _apply(self, r):
        z = self.base.lu.solve(r)
        return z - self.W @ np.linalg.solve(self.cap, self.U.T @ z)

    def solve(self, rhs: np.ndarray, refine: int = 3) -> np.ndarray:
        B = self.B
        z = self._apply(rhs)
        for _ in range(refine):
            z += self._apply(rhs - B @ (B.T @ z))
        return z

    def residual(self, rhs, z) -> float:
        B = self.B
        return float(np.linalg.norm(rhs - B @ (B.T @ z))) / (1.0 + float(np.linalg.norm(rhs)))


class _NormalCache:
    """Hands out a solver for M = A_RF A_RF^T, refactoring only when the rows change or too
    many columns have changed since the last factorization."""

    MAX_UPDATES = 48

    def __init__(self, face: _Face):
        self.face = face
        self.base: Optional[_Normal] = None
        self.R0 = self.F0 = None
        self.W: Dict[int, np.ndarray] = {}
        self.factorizations = 0

    def refactor(self, R, F, B):
        self.base, self.R0, self.F0, self.W = _Normal(B), R.copy(), F.copy(), {}
        self.factorizations += 1
        return self.base

    def get(self, R: np.ndarray, F: np.ndarray, B: sp.csr_matrix):
        if self.base is None or not np.array_equal(R, self.R0):
            return self.refactor(R, F, B)
        added = np.flatnonzero(F & ~self.F0)
        dropped = np.flatnonzero(self.F0 & ~F)
        k = added.size + dropped.size
        if k == 0:
            self.base.B = B
            return self.base
        # a row losing its last free column makes M singular, which Woodbury cannot express
        if k > self.MAX_UPDATES or np.any((np.diff(B.indptr) == 0) & ~self.base.empty_rows):
            return self.refactor(R, F, B)
        cols = np.concatenate([added, dropped])
        new = [int(j) for j in cols if int(j) not in self.W]
        if new:
            Wn = self.base.lu.solve(self.face.columns_on(R, np.array(new)))
            for t, j in enumerate(new):
                self.W[j] = Wn[:, t]
        U = self.face.columns_on(R, cols)
        W = np.column_stack([self.W[int(j)] for j in cols])
        sgn = np.concatenate([np.ones(added.size), -np.ones(dropped.size)])
        return _UpdatedNormal(self.base, B, U, W, sgn)


def _kkt(A, AT, c, rl, ru, cl, cu, x, y):
    """Absolute violations and objectives in the original space."""
    Ax = A @ x
    pv = max(float(np.max(np.maximum(rl - Ax, 0.0) + np.maximum(Ax - ru, 0.0), initial=0.0)),
             float(np.max(np.maximum(cl - x, 0.0) + np.maximum(x - cu, 0.0), initial=0.0)))
    d = c - AT @ y
    lf_c, uf_c, lf_r, uf_r = np.isfinite(cl), np.isfinite(cu), np.isfinite(rl), np.isfinite(ru)
    dv = max(float(np.max(np.where(lf_c, 0.0, np.maximum(d, 0.0)) + np.where(uf_c, 0.0, np.maximum(-d, 0.0)),
                          initial=0.0)),
             float(np.max(np.where(lf_r, 0.0, np.maximum(y, 0.0)) + np.where(uf_r, 0.0, np.maximum(-y, 0.0)),
                          initial=0.0)))
    dobj = (float(np.sum(np.where(y > 0, y * np.where(lf_r, rl, 0.0), 0.0)))
            + float(np.sum(np.where(y < 0, y * np.where(uf_r, ru, 0.0), 0.0)))
            + float(np.sum(np.where(d > 0, d * np.where(lf_c, cl, 0.0), 0.0)))
            + float(np.sum(np.where(d < 0, d * np.where(uf_c, cu, 0.0), 0.0))))
    pobj = float(c @ x)
    return pobj, dobj, pv, dv, d, Ax


def polish_vertex(A: sp.spmatrix, c, row_lb, row_ub, col_lb, col_ub, x0, y0,
                  tol: float = 1e-9, max_steps: int = 400, max_pricing: int = 12,
                  time_limit: float = 30.0) -> PolishResult:
    """
    Polish (x0, y0) -- a PDLP point, ideally at tolerance 1e-5 or tighter -- into an optimal
    vertex. `tol` is relative: violations are measured against 1 + |bound| / 1 + ||c||_inf,
    the gap against 1 + |pobj| + |dobj|.

    Gives up (ok=False, cheaply) after `max_pricing` release rounds: from a point too far from
    optimal the active-set guess is wrong in many places, and more PDLP iterations fix that far
    faster than pushing does.
    """
    t0 = time.perf_counter()
    A = sp.csr_matrix(A, dtype=np.float64)
    AT = A.T.tocsr()
    c = np.asarray(c, dtype=np.float64)
    rl, ru = np.asarray(row_lb, dtype=np.float64), np.asarray(row_ub, dtype=np.float64)
    cl, cu = np.asarray(col_lb, dtype=np.float64), np.asarray(col_ub, dtype=np.float64)
    x0, y0 = np.asarray(x0, dtype=np.float64), np.asarray(y0, dtype=np.float64)
    m, n = A.shape
    face = _Face(A)
    lf_c, uf_c, lf_r, uf_r = np.isfinite(cl), np.isfinite(cu), np.isfinite(rl), np.isfinite(ru)
    eq = lf_r & uf_r & (rl == ru)
    c_inf = 1.0 + float(np.abs(c).max(initial=0.0))
    tol_d = tol * c_inf                                   # reduced-cost zero
    tol_xc = tol * (1.0 + np.abs(np.where(lf_c, cl, np.where(uf_c, cu, 0.0))))
    tol_rl = tol * (1.0 + np.abs(np.where(lf_r, rl, 0.0)))
    tol_ru = tol * (1.0 + np.abs(np.where(uf_r, ru, 0.0)))

    # ---- 1. identify: which of (x - l) and d is "more zero", each on its own scale
    d0 = c - AT @ y0
    Ax0 = A @ x0
    xs = 1e-12 + float(np.abs(x0).max(initial=0.0))
    ys = 1e-12 + float(np.abs(y0).max(initial=0.0))
    rs = 1e-12 + float(np.abs(Ax0).max(initial=0.0))
    dn = d0 / c_inf
    at_l = lf_c & (dn > 0) & ((x0 - cl) / xs < dn)
    at_u = uf_c & (dn < 0) & ((cu - x0) / xs < -dn) & ~at_l
    yn = y0 / ys
    act_l = lf_r & (eq | ((yn > 0) & ((Ax0 - rl) / rs < yn)))
    act_u = uf_r & ~act_l & (yn < 0) & ((ru - Ax0) / rs < -yn)

    def fixed_values(x_ref):
        return np.where(at_l, cl, np.where(at_u, cu, x_ref))

    def project(x_ref):
        """Least-distance move of the free columns onto the active rows. Returns (x, B, normal)."""
        F = ~(at_l | at_u)
        R = act_l | act_u
        x = fixed_values(x_ref)
        B = face.block(R, F)
        if B.shape[0] == 0:
            return x, B, None, F, R
        N = ~F
        b = np.where(act_l, rl, ru)[R] - (face.block(R, N) @ x[N] if N.any() else 0.0)
        nrm = _Normal(B)
        xF = x[F]
        x[F] = xF + B.T @ nrm.solve(b - B @ xF)
        return x, B, nrm, F, R

    # ---- 2. project, repairing bounds / inactive rows the projection runs through
    factorizations = 0
    x = x0
    for _ in range(10):
        x, B, nrm, F, R = project(x0)
        factorizations += 1
        lo = F & lf_c & (x < cl - tol_xc)
        hi = F & uf_c & (x > cu + tol_xc)
        Ax = A @ x
        rlo = ~R & lf_r & (Ax < rl - tol_rl)
        rhi = ~R & uf_r & (Ax > ru + tol_ru)
        if not (lo.any() or hi.any() or rlo.any() or rhi.any()):
            break
        at_l |= lo
        at_u |= hi & ~at_l
        act_l |= rlo
        act_u |= rhi & ~act_l
    x = np.clip(x, cl, cu)

    # ---- 3/4. push along -d_F, price, repeat
    steps = pricing = 0
    reason = "step limit"
    yR = np.zeros(0)
    cache = _NormalCache(face)
    while steps < max_steps:
        if time.perf_counter() - t0 > time_limit:
            reason = "time limit"
            break
        F = ~(at_l | at_u)
        R = act_l | act_u
        B = face.block(R, F)
        cF = c[F]
        if B.shape[0]:
            nrm = cache.get(R, F, B)
            rhs = B @ cF
            yR = nrm.solve(rhs)
            if isinstance(nrm, _UpdatedNormal) and nrm.residual(rhs, yR) > 1e-10:
                yR = cache.refactor(R, F, B).solve(rhs)
        else:
            yR = np.zeros(0)
        dF = cF - B.T @ yR
        if float(np.abs(dF).max(initial=0.0)) <= tol_d:
            # vertex (or optimal face) of this active set: price the fixed columns and rows
            y = np.zeros(m)
            y[R] = yR
            d = c - AT @ y
            enter_l = at_l & (d < -tol_d)
            enter_u = at_u & (d > tol_d)
            leave_l = act_l & ~eq & (y < -tol_d)
            leave_u = act_u & ~eq & (y > tol_d)
            if not (enter_l.any() or enter_u.any() or leave_l.any() or leave_u.any()):
                reason = "priced out"
                break
            pricing += 1
            if pricing > max_pricing:
                reason = "pricing limit (start point too far from optimal)"
                break
            # release the most attractive few, like partial pricing in simplex
            for mask, score, arr in ((enter_l, -d, at_l), (enter_u, d, at_u),
                                     (leave_l, -y, act_l), (leave_u, y, act_u)):
                idx = np.flatnonzero(mask)
                if idx.size:
                    top = idx[np.argsort(-score[idx])[:64]]
                    arr[top] = False
            steps += 1
            continue
        p = np.zeros(n)
        p[F] = -dF
        Ap = A @ p
        Ax = A @ x
        with np.errstate(divide="ignore", invalid="ignore"):
            ac = np.where(p < 0, (x - cl) / -p, np.where(p > 0, (cu - x) / p, np.inf))
            ar = np.where(~R & (Ap < 0), (Ax - rl) / -Ap, np.where(~R & (Ap > 0), (ru - Ax) / Ap, np.inf))
        ac = np.where(np.isnan(ac), np.inf, np.maximum(ac, 0.0))
        ar = np.where(np.isnan(ar), np.inf, np.maximum(ar, 0.0))
        alpha = min(float(ac.min(initial=np.inf)), float(ar.min(initial=np.inf)))
        if not math.isfinite(alpha):
            reason = "unbounded face direction"
            break
        x = x + alpha * p
        lim = alpha * (1.0 + 1e-9) + 1e-300
        hit_c = ac <= lim
        at_l |= hit_c & (p < 0)
        at_u |= hit_c & (p > 0) & ~at_l
        x = fixed_values(x)
        hit_r = ar <= lim
        act_l |= hit_r & (Ap < 0)
        act_u |= hit_r & (Ap > 0) & ~act_l
        steps += 1

    # ---- 5. final projection onto the active set (removes step drift), final dual, certify
    factorizations += cache.factorizations
    x, B, nrm, F, R = project(x)
    factorizations += 1
    x = np.clip(x, cl, cu)
    y = np.zeros(m)
    if nrm is not None:
        y[R] = nrm.solve(B @ c[F])
    pobj, dobj, pv, dv, d, _ = _kkt(A, AT, c, rl, ru, cl, cu, x, y)
    rel_p = pv / (1.0 + float(np.abs(np.where(np.isfinite(np.r_[rl, ru, cl, cu]), np.r_[rl, ru, cl, cu], 0.0)).max(initial=0.0)))
    rel_d = dv / c_inf
    gap = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
    ok = rel_p <= tol and rel_d <= tol and gap <= tol
    if not ok and reason == "priced out":
        reason = f"certificate failed (rel primal {rel_p:.1e}, rel dual {rel_d:.1e}, gap {gap:.1e})"
    return PolishResult(ok=ok, x=x, y=y, primal_objective=pobj, dual_objective=dobj,
                        max_primal_violation=pv, max_dual_violation=dv, rel_gap=gap,
                        push_steps=steps, factorizations=factorizations,
                        runtime=time.perf_counter() - t0, reason="certified" if ok else reason,
                        info={"pricing_rounds": pricing, "free_columns": int((~(at_l | at_u)).sum()),
                              "active_rows": int((act_l | act_u).sum()),
                              "rel_primal": rel_p, "rel_dual": rel_d})


# ------------------------------------------------------------------------------ PDLP + polish
@dataclass
class ExactLPResult:
    status: str                  # optimal | time_limit | iteration_limit | numerical_error
    x: np.ndarray
    y: np.ndarray
    objective: float
    vertex: bool                 # True: certified optimal vertex; False: the PDLP point
    pdlp_iterations: int
    pdlp_seconds: float
    polish_seconds: float
    stages: list                 # one entry per (PDLP tolerance, polish attempt)
    rel_primal_residual: float
    rel_dual_residual: float
    rel_gap: float
    device: str
    restarts: int


def pdlp_exact(A, c, row_lb, row_ub, col_lb, col_ub, tols=(1e-5, 1e-7), polish_tol: float = 1e-9,
               time_limit: float = 600.0, device: str = "cpu") -> ExactLPResult:
    """
    PDLP to a loose tolerance, then polish to an exact vertex. If the polish cannot certify,
    PDLP resumes from where it stopped (warm start) at the next tolerance and polishes again.
    If every stage fails, the last PDLP point is returned as-is with vertex=False -- its
    residuals are then the honest measure of its accuracy.
    """
    from sovereign_opt.solvers.lp.pdlp import pdlp
    t0 = time.perf_counter()
    stages, x, y, iters, restarts = [], None, None, 0, 0
    pdlp_s = polish_s = 0.0
    r = None
    for tol in tols:
        left = time_limit - (time.perf_counter() - t0)
        if left <= 0:
            break
        t1 = time.perf_counter()
        r = pdlp(A, c, row_lb, row_ub, col_lb, col_ub, tol=tol, time_limit=left, device=device,
                 x_init=x, y_init=y)
        dt = time.perf_counter() - t1
        pdlp_s += dt
        iters += r.iterations
        restarts += r.restarts
        x, y = r.x, r.y
        stage = {"pdlp_tol": tol, "pdlp_status": r.status, "pdlp_iterations": r.iterations,
                 "pdlp_seconds": dt}
        if r.status != "optimal":
            stages.append(stage)
            break
        left = time_limit - (time.perf_counter() - t0)
        # a polish that works finishes well inside the time of the PDLP stage before it; one that
        # is still going after that is stuck on a bad active-set guess -- more PDLP is cheaper
        p = polish_vertex(A, c, row_lb, row_ub, col_lb, col_ub, x, y, tol=polish_tol,
                          time_limit=min(left, max(1.0, dt)))
        polish_s += p.runtime
        stage.update({"polish_ok": p.ok, "polish_reason": p.reason, "polish_seconds": p.runtime,
                      "push_steps": p.push_steps, "factorizations": p.factorizations})
        stages.append(stage)
        if p.ok:
            return ExactLPResult(status="optimal", x=p.x, y=p.y, objective=p.primal_objective, vertex=True,
                                 pdlp_iterations=iters, pdlp_seconds=pdlp_s, polish_seconds=polish_s,
                                 stages=stages, rel_primal_residual=p.info["rel_primal"],
                                 rel_dual_residual=p.info["rel_dual"], rel_gap=p.rel_gap,
                                 device=r.device, restarts=restarts)
    return ExactLPResult(status=r.status if r is not None else "time_limit",
                         x=x, y=y, objective=(r.primal_objective if r is not None else float("nan")),
                         vertex=False, pdlp_iterations=iters, pdlp_seconds=pdlp_s, polish_seconds=polish_s,
                         stages=stages, rel_primal_residual=(r.rel_primal_residual if r else float("inf")),
                         rel_dual_residual=(r.rel_dual_residual if r else float("inf")),
                         rel_gap=(r.rel_gap if r else float("inf")), device=(r.device if r else device),
                         restarts=restarts)
