"""
Sovereign PDLP: restarted primal-dual hybrid gradient for LP, on CPU or GPU.

    min  c^T x   s.t.  l_r <= K x <= u_r,   l_x <= x <= u_x

Why a first-order method: simplex and interior point need matrix factorizations, which are
sequential, memory-hungry and hard to run on a GPU. PDHG needs only sparse matrix-vector
products and vector arithmetic, which a GPU does thousands of in parallel. That makes it the
method of choice for very large LPs (the approach behind PDLP / cuPDLP in the literature).

Implemented here from the published method descriptions:
- Preconditioning: Ruiz equilibration (10 passes) + Pock-Chambolle (alpha = 1) diagonal scaling,
  then objective / bound norm rescaling.
- method="halpern" (default): restarted, reflected Halpern PDHG with a constant step
  1/||K||_2, restarts on the fixed-point residual ||z - T(z)||, primal-weight rebalancing at each
  restart. On CPU it runs on fused multi-threaded numba kernels (pdhg_kernels.py).
- method="adaptive": the original PDLP loop -- adaptive step size, averaged iterates, restarts
  on the KKT error -- kept so earlier recorded results stay reproducible.
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


def _kernels():
    """The fused numba kernels, or None when numba is not installed (NumPy fallback)."""
    try:
        from sovereign_opt.solvers.lp import pdhg_kernels
        return pdhg_kernels
    except ImportError:
        return None


# ------------------------------------------------------------------------------ Halpern engines
# The Halpern loop keeps its whole state inside an engine: current z = (x, y), the PDHG image
# z+ = T(z), the restart anchor z0, and the matching K x / K^T y for each. Engines differ only
# in how they compute; the algorithm in _pdlp_halpern() is written once.

class _FusedCPUEngine:
    """Multi-threaded numba kernels, preallocated buffers, zero allocation per iteration."""

    # below this many nonzeros one thread beats a thread team woken six times per iteration
    PARALLEL_MIN_NNZ = 50_000

    def __init__(self, Ks: sp.csr_matrix, KsT: sp.csr_matrix, cs, rl, ru, xl, xu, x_init, y_init, kern):
        threads = _numba_threads() if Ks.nnz >= self.PARALLEL_MIN_NNZ else 1
        self.k = kern.PARALLEL if threads > 1 else kern.SERIAL
        self.name = f"cpu-fused ({threads} thread{'s' if threads > 1 else ''})"
        self.Kp, self.Ki, self.Kd = Ks.indptr, Ks.indices, Ks.data
        self.Tp, self.Ti, self.Td = KsT.indptr, KsT.indices, KsT.data
        self.Kb, self.Tb = kern.nnz_blocks(self.Kp, threads), kern.nnz_blocks(self.Tp, threads)
        m, n = Ks.shape
        self.c, self.rl, self.ru, self.xl, self.xu = cs, rl, ru, xl, xu
        f = lambda k: np.zeros(k)  # noqa: E731
        self.x, self.xp, self.x0, self.KTy, self.KTyp, self.KTy0 = (f(n) for _ in range(6))
        self.y, self.yp, self.y0, self.Kx, self.Kxp, self.Kx0 = (f(m) for _ in range(6))
        self.x[:] = x_init
        self.y[:] = y_init
        self.mv(self.x, self.Kx)
        self.rmv(self.y, self.KTy)
        self.anchor()

    def mv(self, x, out):
        self.k.csr_matvec(self.Kp, self.Ki, self.Kd, x, out, self.Kb)

    def rmv(self, y, out):
        self.k.csr_matvec(self.Tp, self.Ti, self.Td, y, out, self.Tb)

    def norm_estimate(self, iters: int) -> float:
        v = np.random.default_rng(0).standard_normal(self.x.shape[0])
        v /= np.linalg.norm(v)
        Kv, w = np.zeros(self.y.shape[0]), np.zeros_like(v)
        lam = 0.0
        for _ in range(iters):
            self.mv(v, Kv)
            self.rmv(Kv, w)
            lam = float(np.linalg.norm(w))
            if lam == 0.0:
                return 0.0
            v = w / lam
        return math.sqrt(lam)

    def step(self, tau, sigma):
        dx2 = self.k.primal_step(self.x, self.KTy, self.c, self.xl, self.xu, tau, self.xp)
        self.mv(self.xp, self.Kxp)
        dy2, inter = self.k.dual_step(self.y, self.Kx, self.Kxp, self.rl, self.ru, sigma, self.yp)
        self.rmv(self.yp, self.KTyp)
        return dx2, dy2, inter

    def primal_only(self, tau):
        return self.k.primal_step(self.x, self.KTy, self.c, self.xl, self.xu, tau, self.xp)

    def fused_step(self, tau, sigma, a, b, g):
        """
        One whole iteration -- step + Halpern -- in two passes, when x+ is already in place.
        Returns (||dy||^2, dy^T K dx) of this iteration and ||dx||^2 of the NEXT one.
        """
        dy2, inter = self.k.fused_rows(self.Kp, self.Ki, self.Kd, self.Kb, self.xp, self.y, self.Kx,
                                       self.y0, self.Kx0, self.rl, self.ru, sigma, a, b, g, self.yp, self.Kxp)
        dx2_next = self.k.fused_cols(self.Tp, self.Ti, self.Td, self.Tb, self.yp, self.x, self.xp, self.x0,
                                     self.KTy, self.KTy0, self.KTyp, self.c, self.xl, self.xu, tau, a, b, g)
        return dy2, inter, dx2_next

    def halpern(self, a, b, g):
        self.k.halpern(self.x, self.xp, self.x0, self.KTy, self.KTyp, self.KTy0, a, b, g)
        self.k.halpern(self.y, self.yp, self.y0, self.Kx, self.Kxp, self.Kx0, a, b, g)

    def anchor_distance(self):
        """||x+ - x0||, ||y+ - y0||: how far z+ has moved from the last restart point."""
        return math.sqrt(self.k.sq_dist(self.xp, self.x0)), math.sqrt(self.k.sq_dist(self.yp, self.y0))

    def restart(self):
        """z <- z0 <- z+ (restart at the latest PDHG image)."""
        for a, b in ((self.x, self.xp), (self.y, self.yp), (self.Kx, self.Kxp), (self.KTy, self.KTyp)):
            np.copyto(a, b)
        self.anchor()

    def anchor(self):
        for a, b in ((self.x0, self.x), (self.y0, self.y), (self.Kx0, self.Kx), (self.KTy0, self.KTy)):
            np.copyto(a, b)

    def kkt_plus(self, t):
        """KKT measures at z+ in the ORIGINAL space; `t` carries the unscaling data."""
        pres, dobj_r = self.k.kkt_rows(self.Kxp, self.yp, t.Dr, t.b_scale, t.c_scale, t.row_lb, t.row_ub)
        dres, pobj, dobj_c = self.k.kkt_cols(self.xp, self.KTyp, self.c, t.Dc, t.b_scale, t.c_scale,
                                             t.col_lb, t.col_ub)
        return _kkt_finish(math.sqrt(pres), math.sqrt(dres), pobj, dobj_r + dobj_c, t)

    def result(self):
        return self.xp.copy(), self.yp.copy()


class _ArrayEngine:
    """Same state machine over NumPy or torch arrays (GPU, torch-cpu, or no numba)."""

    def __init__(self, ops, cs, rl, ru, xl, xu, x_init, y_init, t):
        self.ops, self.name = ops, ops.name
        V = ops.vec
        self.c, self.rl, self.ru, self.xl, self.xu = V(cs), V(rl), V(ru), V(xl), V(xu)
        self.x = V(x_init)
        self.y = V(y_init)
        self.Kx, self.KTy = ops.mv(self.x), ops.rmv(self.y)
        self.xp, self.yp, self.Kxp, self.KTyp = self.x, self.y, self.Kx, self.KTy
        self.anchor()
        # unscaling data on the device, for the KKT check
        self.Dr, self.Dc = V(t.Dr), V(t.Dc)
        self.row_lb, self.row_ub, self.col_lb, self.col_ub = V(t.row_lb), V(t.row_ub), V(t.col_lb), V(t.col_ub)
        self.lfin_r, self.ufin_r = V(np.isfinite(t.row_lb).astype(float)) > 0, V(np.isfinite(t.row_ub).astype(float)) > 0
        self.lfin_x, self.ufin_x = V(np.isfinite(t.col_lb).astype(float)) > 0, V(np.isfinite(t.col_ub).astype(float)) > 0

    @staticmethod
    def _copy(v):
        return v.clone() if hasattr(v, "clone") else v.copy()

    def norm_estimate(self, iters: int) -> float:
        ops = self.ops
        v = ops.vec(np.random.default_rng(0).standard_normal(len(self.xl)))
        v = v / ops.norm(v)
        lam = 0.0
        for _ in range(iters):
            w = ops.rmv(ops.mv(v))
            lam = ops.norm(w)
            if lam == 0.0:
                return 0.0
            v = w / lam
        return math.sqrt(lam)

    def step(self, tau, sigma):
        ops = self.ops
        self.xp = ops.clamp(self.x - tau * (self.c - self.KTy), self.xl, self.xu)
        self.Kxp = ops.mv(self.xp)
        v = self.y - sigma * (2.0 * self.Kxp - self.Kx)
        self.yp = v + sigma * ops.clamp(-v / sigma, self.rl, self.ru)
        self.KTyp = ops.rmv(self.yp)
        dx, dy = self.xp - self.x, self.yp - self.y
        return ops.dot(dx, dx), ops.dot(dy, dy), ops.dot(dy, self.Kxp - self.Kx)

    def halpern(self, a, b, g):
        h = lambda u, up, u0: a * ((1.0 + g) * up - g * u) + b * u0  # noqa: E731
        self.x, self.KTy = h(self.x, self.xp, self.x0), h(self.KTy, self.KTyp, self.KTy0)
        self.y, self.Kx = h(self.y, self.yp, self.y0), h(self.Kx, self.Kxp, self.Kx0)

    def anchor_distance(self):
        return self.ops.norm(self.xp - self.x0), self.ops.norm(self.yp - self.y0)

    def restart(self):
        self.x, self.y, self.Kx, self.KTy = self.xp, self.yp, self.Kxp, self.KTyp
        self.anchor()

    def anchor(self):
        c = self._copy
        self.x0, self.y0, self.Kx0, self.KTy0 = c(self.x), c(self.y), c(self.Kx), c(self.KTy)

    def kkt_plus(self, t):
        ops = self.ops
        Kx_o = self.Kxp / self.Dr * t.b_scale
        pres = ops.norm(Kx_o - ops.clamp(Kx_o, self.row_lb, self.row_ub))
        d_o = (self.c - self.KTyp) / self.Dc * t.c_scale
        zn = ops.zeros_like(d_o)
        dres = ops.norm(ops.where(self.lfin_x, zn, ops.clamp(d_o, 0.0, INF))
                        + ops.where(self.ufin_x, zn, ops.clamp(d_o, -INF, 0.0)))
        pobj = ops.dot(self.c / self.Dc * t.c_scale, self.xp * self.Dc * t.b_scale)
        y_o = self.yp * self.Dr * t.c_scale
        zm = ops.zeros_like(y_o)
        dobj = (ops.dot(ops.clamp(y_o, 0.0, INF), ops.where(self.lfin_r, self.row_lb, zm))
                + ops.dot(ops.clamp(y_o, -INF, 0.0), ops.where(self.ufin_r, self.row_ub, zm))
                + ops.dot(ops.clamp(d_o, 0.0, INF), ops.where(self.lfin_x, self.col_lb, zn))
                + ops.dot(ops.clamp(d_o, -INF, 0.0), ops.where(self.ufin_x, self.col_ub, zn)))
        return _kkt_finish(pres, dres, pobj, dobj, t)

    def result(self):
        self.ops.sync()
        return self.ops.to_numpy(self.xp).copy(), self.ops.to_numpy(self.yp).copy()


def _kkt_finish(pres, dres, pobj, dobj, t):
    rp = pres / (1.0 + t.rhs_norm)
    rd = dres / (1.0 + t.c_norm)
    rg = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
    return rp, rd, rg, pobj, dobj


def warmup() -> None:
    """
    Load the compiled numba kernels (serial and parallel) with a throwaway 2-variable LP.
    The first call of each kernel in a fresh process reads it from numba's on-disk cache,
    which costs about as much as importing a solver library; call this before timing a solve.
    """
    if _kernels() is None:
        return
    A = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, -1.0]]))
    args = (A, np.array([1.0, 2.0]), np.array([1.0, -1.0]), np.array([np.inf, 1.0]), np.zeros(2), np.full(2, 10.0))
    old = _FusedCPUEngine.PARALLEL_MIN_NNZ
    try:
        for threshold in (0, old):          # parallel kernel set, then serial
            _FusedCPUEngine.PARALLEL_MIN_NNZ = threshold
            pdlp(*args, tol=1e-6, max_iterations=256, device="cpu")
    finally:
        _FusedCPUEngine.PARALLEL_MIN_NNZ = old


def _numba_threads() -> int:
    try:
        import numba
        return int(numba.get_num_threads())
    except ImportError:
        return 1


# ------------------------------------------------------------------------------ scaling
def _precondition_fused(K: sp.csr_matrix, kern, ruiz_passes: int = 10):
    """_precondition() with in-place numba kernels: no matrix is rebuilt between passes."""
    m, n = K.shape
    Ks = sp.csr_matrix(K, dtype=np.float64, copy=True)
    Ks.sort_indices()
    ip, ix, d = Ks.indptr, Ks.indices, Ks.data
    Dr, Dc = np.ones(m), np.ones(n)
    rv, cv = np.empty(m), np.empty(n)
    blocks = kern.nnz_blocks(ip, _numba_threads())

    def inv_sqrt(v):
        return np.where(v > 0, 1.0 / np.sqrt(np.where(v > 0, v, 1.0)), 1.0)

    for _ in range(ruiz_passes):
        kern.csr_row_absmax(ip, d, rv, blocks)
        kern.csr_col_absmax(ix, d, cv)
        r, cc = inv_sqrt(rv), inv_sqrt(cv)
        kern.csr_scale(ip, ix, d, r, cc, blocks)
        Dr *= r
        Dc *= cc
    kern.csr_row_abssum(ip, d, rv, blocks)
    kern.csr_col_abssum(ix, d, cv)
    r, cc = inv_sqrt(rv), inv_sqrt(cv)
    kern.csr_scale(ip, ix, d, r, cc, blocks)
    return Ks, Dr * r, Dc * cc


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
    method: str = "halpern",
    reflection: float = 1.0,
    x_init: Optional[np.ndarray] = None,
    y_init: Optional[np.ndarray] = None,
) -> PDLPResult:
    """
    method "halpern" (default): restarted, reflected Halpern PDHG with a constant step size.
    method "adaptive": the original PDLP loop (adaptive steps, averaged iterates), kept for
    comparison with earlier recorded results.
    x_init / y_init (halpern only): warm start from a previous point, in the original space --
    e.g. to tighten the tolerance of a finished run without starting over.
    """
    t_start = time.time()
    K = sp.csr_matrix(K, dtype=np.float64)
    m, n = K.shape
    c = np.asarray(c, dtype=np.float64)
    row_lb, row_ub = np.asarray(row_lb, dtype=np.float64), np.asarray(row_ub, dtype=np.float64)
    col_lb, col_ub = np.asarray(col_lb, dtype=np.float64), np.asarray(col_ub, dtype=np.float64)
    kern = _kernels()

    # ---- preconditioning (CPU, once)
    Ks, Dr, Dc = _precondition_fused(K, kern) if kern is not None else _precondition(K)
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

    if method == "halpern":
        bn, cn = _finite_norm(rl, ru), float(np.linalg.norm(cs))
        t = _Unscale(Dr=Dr, Dc=Dc, b_scale=b_scale, c_scale=c_scale, row_lb=row_lb, row_ub=row_ub,
                     col_lb=col_lb, col_ub=col_ub, rhs_norm=rhs_norm, c_norm=c_norm)
        xs0 = np.clip(np.zeros(n) if x_init is None else np.asarray(x_init, dtype=np.float64) / (Dc * b_scale), xl, xu)
        ys0 = np.zeros(m) if y_init is None else np.asarray(y_init, dtype=np.float64) / (Dr * c_scale)
        return _pdlp_halpern(Ks, cs, rl, ru, xl, xu, t, omega=(cn / bn if (bn > 1e-10 and cn > 1e-10) else 1.0),
                             tol=tol, max_iterations=max_iterations, time_limit=time_limit, device=device,
                             eval_every=eval_every, reflection=reflection, kern=kern, t_start=t_start,
                             x_init=xs0, y_init=ys0)
    if method != "adaptive":
        raise ValueError(f"unknown PDLP method '{method}' (halpern | adaptive)")

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


@dataclass
class _Unscale:
    """What the KKT check needs to measure residuals in the original, unscaled space."""
    Dr: np.ndarray
    Dc: np.ndarray
    b_scale: float
    c_scale: float
    row_lb: np.ndarray
    row_ub: np.ndarray
    col_lb: np.ndarray
    col_ub: np.ndarray
    rhs_norm: float
    c_norm: float


def _make_engine(Ks, cs, rl, ru, xl, xu, t, device, kern, x_init, y_init):
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device == "cpu" and kern is not None:
        return _FusedCPUEngine(Ks, Ks.T.tocsr(), cs, rl, ru, xl, xu, x_init, y_init, kern)
    ops = make_ops(Ks, "cpu" if device == "numpy" else device)
    return _ArrayEngine(ops, cs, rl, ru, xl, xu, x_init, y_init, t)


# Restart thresholds on the fixed-point residual ||z - T(z)||, as in restarted Halpern PDHG.
_SUFFICIENT, _NECESSARY, _ARTIFICIAL = 0.2, 0.8, 0.36
_STEP_SAFETY = 0.998      # constant step eta = safety / ||K||_2
# Primal weight omega balances primal and dual progress. At each restart a PID controller in log
# space drives omega toward dy/dx (the dual over the primal distance moved since the last restart):
#   e = log(omega) - log(dy/dx),   log(omega) -= KP e + KI sum(e) + KD (e - e_prev)
# KI = KD = 0 is the classic PDLP smoothing omega <- (dy/dx)^KP omega^(1-KP).
_WEIGHT_KP, _WEIGHT_KI, _WEIGHT_KD = 0.8, 0.0, 0.0


def _pdlp_halpern(Ks, cs, rl, ru, xl, xu, t, omega, tol, max_iterations, time_limit, device,
                  eval_every, reflection, kern, t_start, x_init, y_init) -> PDLPResult:
    """
    Restarted reflected Halpern PDHG.

    PDHG is a fixed-point iteration z <- T(z). Halpern anchors every step to the last restart
    point z0:   z_{k+1} = (k+1)/(k+2) [(1+g) T(z_k) - g z_k] + 1/(k+2) z0
    which gives the optimal O(1/k) rate on the fixed-point residual ||z - T(z)|| (plain PDHG
    only guarantees that for the average). Reflection g in (0, 1] takes a longer step along the
    same direction; the reflected operator is still nonexpansive, so the guarantee holds.

    Because the rate is on the *last* iterate, there is no averaged sequence to maintain and the
    step size is a constant 1/||K||_2 -- no rejected steps, so every iteration is exactly two
    mat-vecs. Restarts are decided on ||z - T(z)|| itself, which the fused kernels produce for
    free, and each restart rebalances the primal weight omega.
    """
    eng = _make_engine(Ks, cs, rl, ru, xl, xu, t, device, kern, x_init, y_init)
    norm_k = eng.norm_estimate(60)
    eta = _STEP_SAFETY / max(norm_k * 1.01, 1e-12)   # power iteration approaches ||K|| from below
    setup = time.time() - t_start

    status = "iteration_limit"
    rp = rd = rg = pobj = dobj = INF
    trace: List[Dict[str, Any]] = []
    k = inner = restarts = 0
    r_anchor = r_prev = INF
    w_int, w_prev = 0.0, None
    fused = hasattr(eng, "fused_step")
    pending_dx2 = None   # x+ already computed (by the previous fused pass) for the current z
    while k < max_iterations:
        tau, sigma = eta / omega, eta * omega
        if fused and (k + 1) % eval_every and k + 1 < max_iterations:
            # no check this iteration, so no restart: the Halpern step is certain -> fuse it
            dx2 = pending_dx2 if pending_dx2 is not None else eng.primal_only(tau)
            dy2, inter, pending_dx2 = eng.fused_step(tau, sigma, (inner + 1.0) / (inner + 2.0),
                                                     1.0 / (inner + 2.0), reflection)
            k += 1
            r = math.sqrt(max(omega * dx2 + dy2 / omega - 2.0 * eta * inter, 0.0))
            if inner == 0:
                r_anchor, r_prev = r, INF
            inner += 1
            continue
        pending_dx2 = None
        dx2, dy2, inter = eng.step(tau, sigma)
        k += 1
        # fixed-point residual ||z - T(z)|| in the PDHG norm (scaled by eta)
        r = math.sqrt(max(omega * dx2 + dy2 / omega - 2.0 * eta * inter, 0.0))
        if inner == 0:
            r_anchor, r_prev = r, INF   # residual of the restart point itself

        if k % eval_every == 0 or k >= max_iterations:
            rp, rd, rg, pobj, dobj = eng.kkt_plus(t)
            trace.append({"iteration": k, "objective": pobj, "dual_objective": dobj, "norm_rp": rp,
                          "norm_rd": rd, "rel_gap": rg, "step": eta, "primal_weight": omega,
                          "restarts": restarts, "fixed_point_residual": r})
            if not all(math.isfinite(v) for v in (rp, rd, rg, r)):
                status = "numerical_error"
                break
            if rp <= tol and rd <= tol and rg <= tol:
                status = "optimal"
                break
            if time.time() - t_start > time_limit:
                status = "time_limit"
                break
            if (r <= _SUFFICIENT * r_anchor
                    or (r <= _NECESSARY * r_anchor and r > r_prev)
                    or inner >= _ARTIFICIAL * k):
                dxn, dyn = eng.anchor_distance()
                if dxn > 1e-10 and dyn > 1e-10:
                    e = math.log(omega) - math.log(dyn / dxn)
                    w_int += e
                    de = 0.0 if w_prev is None else e - w_prev
                    w_prev = e
                    omega = math.exp(math.log(omega) - (_WEIGHT_KP * e + _WEIGHT_KI * w_int + _WEIGHT_KD * de))
                eng.restart()
                restarts += 1
                inner = 0
                continue
            r_prev = r

        eng.halpern((inner + 1.0) / (inner + 2.0), 1.0 / (inner + 2.0), reflection)
        inner += 1

    xs, ys = eng.result()
    x_o = xs * t.Dc * t.b_scale
    y_o = ys * t.Dr * t.c_scale
    return PDLPResult(status=status, x=x_o, y=y_o, primal_objective=pobj, dual_objective=dobj,
                      rel_primal_residual=rp, rel_dual_residual=rd, rel_gap=rg, iterations=k,
                      restarts=restarts, device=eng.name, runtime=time.time() - t_start,
                      setup_time=setup, trace=trace)


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
