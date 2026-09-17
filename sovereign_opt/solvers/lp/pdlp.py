"""
Sovereign PDLP: restarted primal-dual hybrid gradient for LP, on CPU or GPU.

    min  c^T x   s.t.  l_r <= K x <= u_r,   l_x <= x <= u_x

Why a first-order method: simplex and interior point need matrix factorizations, which are
sequential, memory-hungry and hard to run on a GPU. PDHG needs only sparse matrix-vector
products and vector arithmetic, which a GPU does thousands of in parallel. That makes it the
method of choice for very large LPs (the approach behind PDLP / cuPDLP in the literature).

Implemented here from the published method description:
- Preconditioning: Ruiz equilibration (10 passes) + Pock-Chambolle (alpha = 1) diagonal scaling,
  then objective / bound norm rescaling.
- PDHG with adaptive step size (step accepted when eta <= ||dz||_w^2 / (2 |dy^T K dx|)).
- Primal weight omega balancing primal and dual progress, updated at every restart.
- Adaptive restarts on the KKT error (sufficient / necessary / artificial criteria), restarting
  from the better of the current and the step-weighted average iterate.
- Termination on relative primal residual, dual residual and duality gap measured in the
  ORIGINAL (unscaled) space. GPU <-> CPU synchronisation only happens every `eval_every` steps.

Dual convention matches the rest of the engine (minimize-normalized): y_i > 0 means the row is at
its lower bound, y_i < 0 at its upper bound; reduced costs d = c - K^T y.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

INF = float("inf")


# ------------------------------------------------------------------------------ backends
class _NumpyOps:
    name = "cpu-numpy"

    def __init__(self, K: sp.csr_matrix):
        self.K = K.tocsr()
        self.KT = K.T.tocsr()

    def vec(self, a):
        return np.asarray(a, dtype=np.float64).copy()

    def mv(self, x):
        return self.K @ x

    def rmv(self, y):
        return self.KT @ y

    @staticmethod
    def dot(a, b):
        return float(a @ b)

    @staticmethod
    def norm(a):
        return float(np.linalg.norm(a))

    @staticmethod
    def clamp(a, lo, hi):
        return np.minimum(np.maximum(a, lo), hi)

    @staticmethod
    def where(c, a, b):
        return np.where(c, a, b)

    @staticmethod
    def zeros_like(a):
        return np.zeros_like(a)

    @staticmethod
    def to_numpy(a):
        return np.asarray(a)

    @staticmethod
    def sync():
        pass


class _TorchOps:
    def __init__(self, K: sp.csr_matrix, device: str):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.name = f"torch-{device}" + (f" ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else "")
        self.K = self._csr(K.tocsr())
        self.KT = self._csr(K.T.tocsr())

    def _csr(self, M: sp.csr_matrix):
        t = self.torch
        return t.sparse_csr_tensor(t.from_numpy(M.indptr.astype(np.int64)), t.from_numpy(M.indices.astype(np.int64)),
                                   t.from_numpy(M.data.astype(np.float64)), size=M.shape, dtype=t.float64,
                                   device=self.device)

    def vec(self, a):
        return self.torch.as_tensor(np.asarray(a, dtype=np.float64), device=self.device).clone()

    def mv(self, x):
        return self.K @ x

    def rmv(self, y):
        return self.KT @ y

    def dot(self, a, b):
        return float(self.torch.dot(a, b))

    def norm(self, a):
        return float(self.torch.linalg.vector_norm(a))

    def clamp(self, a, lo, hi):
        t = self.torch
        a = t.clamp(a, min=lo) if isinstance(lo, float) else t.maximum(a, lo)
        return t.clamp(a, max=hi) if isinstance(hi, float) else t.minimum(a, hi)

    def where(self, c, a, b):
        return self.torch.where(c, a, b)

    def zeros_like(self, a):
        return self.torch.zeros_like(a)

    @staticmethod
    def to_numpy(a):
        return a.detach().cpu().numpy()

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()


def make_ops(K: sp.csr_matrix, device: str = "auto"):
    """device: 'auto' (CUDA if present, else NumPy), 'cpu' (NumPy), 'torch-cpu', 'cuda'."""
    if device in ("auto", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return _TorchOps(K, "cuda")
        except ImportError:
            pass
        if device == "cuda":
            raise RuntimeError("CUDA requested but no CUDA device / PyTorch CUDA build is available")
    if device == "torch-cpu":
        return _TorchOps(K, "cpu")
    return _NumpyOps(K)


# ------------------------------------------------------------------------------ scaling
def _precondition(K: sp.csr_matrix, ruiz_passes: int = 10):
    """Returns (Ks, D_row, D_col) with Ks = diag(D_row) K diag(D_col)."""
    m, n = K.shape
    Dr, Dc = np.ones(m), np.ones(n)
    Ks = sp.csr_matrix(K, dtype=np.float64, copy=True)
    for _ in range(ruiz_passes):
        A = abs(Ks)
        rmax = np.asarray(A.max(axis=1).todense()).ravel()
        cmax = np.asarray(A.max(axis=0).todense()).ravel()
        r = np.where(rmax > 0, 1.0 / np.sqrt(np.where(rmax > 0, rmax, 1.0)), 1.0)
        cc = np.where(cmax > 0, 1.0 / np.sqrt(np.where(cmax > 0, cmax, 1.0)), 1.0)
        Ks = sp.csr_matrix(sp.diags(r) @ Ks @ sp.diags(cc))
        Dr *= r
        Dc *= cc
    # Pock-Chambolle, alpha = 1
    A = abs(Ks)
    rs = np.asarray(A.sum(axis=1)).ravel()
    cs = np.asarray(A.sum(axis=0)).ravel()
    r = np.where(rs > 0, 1.0 / np.sqrt(np.where(rs > 0, rs, 1.0)), 1.0)
    cc = np.where(cs > 0, 1.0 / np.sqrt(np.where(cs > 0, cs, 1.0)), 1.0)
    Ks = sp.csr_matrix(sp.diags(r) @ Ks @ sp.diags(cc))
    return Ks, Dr * r, Dc * cc


def _finite_norm(*arrays) -> float:
    tot = 0.0
    for a in arrays:
        f = a[np.isfinite(a)]
        tot += float(f @ f)
    return math.sqrt(tot)


# ------------------------------------------------------------------------------ result
@dataclass
class PDLPResult:
    status: str  # optimal | iteration_limit | time_limit | numerical_error
    x: np.ndarray
    y: np.ndarray
    primal_objective: float
    dual_objective: float
    rel_primal_residual: float
    rel_dual_residual: float
    rel_gap: float
    iterations: int
    restarts: int
    device: str
    runtime: float
    setup_time: float
    trace: List[Dict[str, Any]] = field(default_factory=list)


def pdlp(
    K: sp.spmatrix,
    c: np.ndarray,
    row_lb: np.ndarray,
    row_ub: np.ndarray,
    col_lb: np.ndarray,
    col_ub: np.ndarray,
    tol: float = 1e-6,
    max_iterations: int = 200000,
    time_limit: float = 600.0,
    device: str = "auto",
    eval_every: int = 64,
) -> PDLPResult:
    t_start = time.time()
    K = sp.csr_matrix(K, dtype=np.float64)
    m, n = K.shape
    c = np.asarray(c, dtype=np.float64)

    # ---- preconditioning (CPU, once)
    Ks, Dr, Dc = _precondition(K)
    cs = c * Dc
    rl, ru = row_lb * Dr, row_ub * Dr
    xl, xu = col_lb / Dc, col_ub / Dc
    b_scale = 1.0 + _finite_norm(rl, ru)
    c_scale = 1.0 + float(np.linalg.norm(cs))
    cs = cs / c_scale
    rl, ru, xl, xu = rl / b_scale, ru / b_scale, xl / b_scale, xu / b_scale

    # quantities for termination in the ORIGINAL space
    rhs_norm = _finite_norm(row_lb, row_ub)
    c_norm = float(np.linalg.norm(c))

    ops = make_ops(Ks, device)
    V = ops.vec
    c_d, rl_d, ru_d, xl_d, xu_d = V(cs), V(rl), V(ru), V(xl), V(xu)
    Dr_d, Dc_d = V(Dr), V(Dc)
    row_lb_o, row_ub_o, col_lb_o, col_ub_o = V(row_lb), V(row_ub), V(col_lb), V(col_ub)
    lfin_x, ufin_x = V(np.isfinite(col_lb).astype(float)) > 0, V(np.isfinite(col_ub).astype(float)) > 0
    lfin_r, ufin_r = V(np.isfinite(row_lb).astype(float)) > 0, V(np.isfinite(row_ub).astype(float)) > 0
    zero_n = ops.zeros_like(c_d)

    x = ops.clamp(zero_n, xl_d, xu_d)
    y = ops.zeros_like(V(np.zeros(m)))
    Kx, KTy = ops.mv(x), ops.rmv(y)

    max_abs = float(abs(Ks).max()) if Ks.nnz else 1.0
    eta = 1.0 / max(max_abs, 1e-12)
    bn = _finite_norm(rl, ru)
    cn = float(np.linalg.norm(cs))
    omega = cn / bn if (bn > 1e-10 and cn > 1e-10) else 1.0
    setup = time.time() - t_start

    def kkt(xv, yv, Kxv, KTyv):
        """Relative residuals and objectives in the original space."""
        # original: x_o = x * Dc * b_scale ; K_o x_o = Kx / Dr * b_scale ; y_o = y * Dr * c_scale ; d_o = d / Dc * c_scale
        Kx_o = Kxv / Dr_d * b_scale
        pres = ops.norm(Kx_o - ops.clamp(Kx_o, row_lb_o, row_ub_o))
        d_o = (c_d - KTyv) / Dc_d * c_scale
        dres_vec = ops.where(lfin_x, zero_n, ops.clamp(d_o, 0.0, INF)) + ops.where(ufin_x, zero_n, ops.clamp(d_o, -INF, 0.0))
        dres = ops.norm(dres_vec)
        x_o = xv * Dc_d * b_scale
        y_o = yv * Dr_d * c_scale
        pobj = ops.dot(c_d / Dc_d * c_scale, x_o)
        ypos, yneg = ops.clamp(y_o, 0.0, INF), ops.clamp(y_o, -INF, 0.0)
        dpos, dneg = ops.clamp(d_o, 0.0, INF), ops.clamp(d_o, -INF, 0.0)
        z_r = ops.zeros_like(y_o)
        dobj = (ops.dot(ypos, ops.where(lfin_r, row_lb_o, z_r)) + ops.dot(yneg, ops.where(ufin_r, row_ub_o, z_r))
                + ops.dot(dpos, ops.where(lfin_x, col_lb_o, zero_n)) + ops.dot(dneg, ops.where(ufin_x, col_ub_o, zero_n)))
        rp = pres / (1.0 + rhs_norm)
        rd = dres / (1.0 + c_norm)
        rg = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
        return rp, rd, rg, pobj, dobj

    def kkt_scaled_error(rp, rd, rg, w):
        return math.sqrt(w * w * rp * rp + rd * rd / (w * w) + rg * rg)

    # restart bookkeeping
    x_sum, y_sum, w_sum = ops.zeros_like(x), ops.zeros_like(y), 0.0
    x_last, y_last = x.clone() if hasattr(x, "clone") else x.copy(), y.clone() if hasattr(y, "clone") else y.copy()
    kkt_last_restart = None
    kkt_prev_candidate = INF
    iters_since_restart = 0
    restarts = 0
    trace: List[Dict[str, Any]] = []
    status = "iteration_limit"
    k = 0
    attempts = 0
    result_point = (x, y, Kx, KTy)
    rp = rd = rg = pobj = dobj = INF

    def copy(v):
        return v.clone() if hasattr(v, "clone") else v.copy()

    while k < max_iterations:
        # ---------------- one adaptive PDHG step
        for _ in range(60):
            tau, sigma = eta / omega, eta * omega
            x_new = ops.clamp(x - tau * (c_d - KTy), xl_d, xu_d)
            Kx_new = ops.mv(x_new)
            v = y - sigma * (2.0 * Kx_new - Kx)
            y_new = v + sigma * ops.clamp(-v / sigma, rl_d, ru_d)
            KTy_new = ops.rmv(y_new)
            dx, dy = x_new - x, y_new - y
            inter = abs(ops.dot(dy, Kx_new - Kx))
            movement = 0.5 * omega * ops.dot(dx, dx) + 0.5 * ops.dot(dy, dy) / omega
            eta_bar = movement / inter if inter > 0 else INF
            attempts += 1
            kk = attempts + 1
            eta_next = min((1.0 - kk ** -0.3) * eta_bar if math.isfinite(eta_bar) else INF, (1.0 + kk ** -0.6) * eta)
            if eta <= eta_bar:
                break
            eta = eta_next
        step = eta
        eta = eta_next if math.isfinite(eta_next) else eta
        x, y, Kx, KTy = x_new, y_new, Kx_new, KTy_new
        x_sum = x_sum + step * x
        y_sum = y_sum + step * y
        w_sum += step
        k += 1
        iters_since_restart += 1

        if k % eval_every and k < max_iterations:
            continue

        # ---------------- evaluate current and average, maybe restart / stop
        x_avg, y_avg = x_sum / w_sum, y_sum / w_sum
        Kx_avg, KTy_avg = ops.mv(x_avg), ops.rmv(y_avg)
        cur = kkt(x, y, Kx, KTy)
        avg = kkt(x_avg, y_avg, Kx_avg, KTy_avg)
        e_cur = kkt_scaled_error(cur[0], cur[1], cur[2], omega)
        e_avg = kkt_scaled_error(avg[0], avg[1], avg[2], omega)
        use_avg = e_avg < e_cur
        cand = (x_avg, y_avg, Kx_avg, KTy_avg) if use_avg else (x, y, Kx, KTy)
        rp, rd, rg, pobj, dobj = avg if use_avg else cur
        e_cand = min(e_avg, e_cur)
        result_point = cand
        trace.append({"iteration": k, "objective": pobj, "dual_objective": dobj, "norm_rp": rp, "norm_rd": rd,
                      "rel_gap": rg, "step": step, "primal_weight": omega, "restarts": restarts})

        if not all(math.isfinite(t) for t in (rp, rd, rg)):
            status = "numerical_error"
            break
        if rp <= tol and rd <= tol and rg <= tol:
            status = "optimal"
            break
        if time.time() - t_start > time_limit:
            status = "time_limit"
            break

        if kkt_last_restart is None:
            kkt_last_restart = e_cand
        do_restart = (e_cand <= 0.2 * kkt_last_restart
                      or (e_cand <= 0.8 * kkt_last_restart and e_cand > kkt_prev_candidate)
                      or iters_since_restart >= 0.36 * k)
        kkt_prev_candidate = e_cand
        if do_restart:
            xc, yc, Kxc, KTyc = cand
            dxn = ops.norm(xc - x_last)
            dyn = ops.norm(yc - y_last)
            if dxn > 1e-10 and dyn > 1e-10:
                omega = math.exp(0.5 * math.log(dyn / dxn) + 0.5 * math.log(omega))
            x, y, Kx, KTy = copy(xc), copy(yc), copy(Kxc), copy(KTyc)
            x_last, y_last = copy(x), copy(y)
            x_sum, y_sum, w_sum = ops.zeros_like(x), ops.zeros_like(y), 0.0
            kkt_last_restart = e_cand
            kkt_prev_candidate = INF
            iters_since_restart = 0
            restarts += 1

    ops.sync()
    xc, yc, _, _ = result_point
    x_o = ops.to_numpy(xc) * Dc * b_scale
    y_o = ops.to_numpy(yc) * Dr * c_scale
    return PDLPResult(status=status, x=x_o, y=y_o, primal_objective=pobj, dual_objective=dobj,
                      rel_primal_residual=rp, rel_dual_residual=rd, rel_gap=rg, iterations=k, restarts=restarts,
                      device=ops.name, runtime=time.time() - t_start, setup_time=setup, trace=trace)


# ------------------------------------------------------------------------------ solver wrapper
from sovereign_opt.model.model import OptimizationModel  # noqa: E402
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus  # noqa: E402

_STATUS = {"optimal": SolverStatus.OPTIMAL, "iteration_limit": SolverStatus.ITERATION_LIMIT,
           "time_limit": SolverStatus.TIME_LIMIT, "numerical_error": SolverStatus.NUMERICAL_ERROR}


class PDLPSolver(SolverBase):
    """
    First-order LP solver (CPU or GPU).

    crossover=False: pure PDLP, returns the approximate primal-dual point (tolerance `tol`).
    crossover=True ("hybrid"): PDLP finds a near-optimal point fast, then a simplex crossover turns it
    into an exact optimal vertex with an exact basis, which the independent validator then certifies.
    """

    def __init__(self, tol: float = 1e-7, crossover: bool = False, device: str = "auto",
                 max_iterations: int = 1_000_000, crossover_max_entries: float = 3e7):
        super().__init__(name="HybridPDLP" if crossover else "PDLP")
        self.tol = tol
        self.do_crossover = crossover
        self.device = device
        self.max_iterations = max_iterations
        self.crossover_max_entries = crossover_max_entries

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 600.0, **kwargs) -> SolverResult:
        start = time.time()
        if model.objective.is_quadratic:
            return SolverResult(status=SolverStatus.NUMERICAL_ERROR,
                                diagnostics={"error": "PDLP here is for LP; use qp_interior_point for QP."})
        A, rl, ru, cl, cu = model.to_matrix_form()
        if np.any(rl > ru + 1e-9) or np.any(cl > cu + 1e-9):
            return SolverResult(status=SolverStatus.INFEASIBLE, runtime_seconds=time.time() - start,
                                diagnostics={"infeasibility_certificate": "a variable or row has lower bound > upper bound"})
        sign = 1.0 if model.objective.sense == "minimize" else -1.0
        c = model.get_objective_vector()
        device = kwargs.get("device", self.device)
        r = pdlp(A, sign * c, rl, ru, cl, cu, tol=self.tol, max_iterations=self.max_iterations,
                 time_limit=time_limit_seconds, device=device)
        offset = model.objective.offset
        trace = []
        for t in r.trace:
            e = dict(t)
            e["objective"] = sign * e["objective"] + offset
            e.pop("dual_objective", None)
            trace.append(e)
        diagnostics = {
            "algorithm": "restarted primal-dual hybrid gradient (PDLP) with adaptive steps, restarts and primal weight",
            "device": r.device, "pdlp_iterations": r.iterations, "restarts": r.restarts,
            "rel_primal_residual": r.rel_primal_residual, "rel_dual_residual": r.rel_dual_residual,
            "rel_duality_gap": r.rel_gap, "pdlp_seconds": r.runtime, "setup_seconds": r.setup_time,
            "iteration_trace": trace[-60:],
            "dual_sign_convention": "minimize-normalized: y_i > 0 row at lower bound, y_i < 0 at upper bound",
        }

        if r.status == "optimal" and self.do_crossover:
            from sovereign_opt.solvers.lp.interior_point import crossover
            from sovereign_opt.solvers.lp.simplex import lp_result_from_engine
            from sovereign_opt.solvers.lp.simplex_engine import LPStatus
            from sovereign_opt.solvers.lp.standard_form import BoundedForm
            form = BoundedForm(model, scale=True)
            if form.m * form.N <= self.crossover_max_entries:
                t0 = time.time()
                x_full = np.concatenate([r.x, form.A_orig @ r.x]) / form.col_scale
                engine, st = crossover(form, x_full, time_limit=max(1.0, time_limit_seconds - (time.time() - start)))
                if engine is not None and st == LPStatus.OPTIMAL:
                    res = lp_result_from_engine(form, engine, st, start)
                    res.diagnostics.update({k: v for k, v in diagnostics.items()})
                    res.diagnostics["crossover_pivots"] = int(engine.total_iterations)
                    res.diagnostics["crossover_seconds"] = time.time() - t0
                    res.iterations = r.iterations
                    return res
                diagnostics["crossover"] = f"failed ({st}); returning the PDLP point"
            else:
                diagnostics["crossover"] = "skipped (too large for dense crossover); returning the PDLP point"

        status = _STATUS.get(r.status, SolverStatus.NUMERICAL_ERROR)
        res = SolverResult(status=status, iterations=r.iterations, runtime_seconds=time.time() - start,
                           diagnostics=diagnostics)
        if status == SolverStatus.OPTIMAL or r.rel_primal_residual <= 1e-4:
            x, y = r.x, r.y
            d = sign * c - A.T @ y
            names = model.variable_names
            res.primal_solution = {nm: float(v) for nm, v in zip(names, x)}
            res.dual_solution = {nm: float(v) for nm, v in zip(model.constraint_names, y)}
            res.reduced_costs = {nm: float(v) for nm, v in zip(names, d)}
            res.objective_value = float(c @ x) + offset
        return res
