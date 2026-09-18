"""
Fused, multi-threaded CPU kernels for the PDHG hot loop (numba, parallel over rows / entries).

Why fused: one PDHG iteration is two sparse mat-vecs plus a dozen vector updates. Written as
separate NumPy / torch expressions, every update allocates a fresh vector and makes its own pass
over memory, which on a 1M-variable LP costs more than both mat-vecs together. Here each
iteration is exactly four passes -- primal step, K x, dual step, K^T y -- plus one Halpern
combination, with the norms the restart logic needs accumulated inside the same loops.

Every kernel writes into caller-owned buffers; nothing allocates inside the iteration.
"""
import math
import types
from types import SimpleNamespace

import numpy as np
from numba import njit, prange

_FLAGS = dict(parallel=True, fastmath=False, cache=True, nogil=True)


# ------------------------------------------------------------------------------ sparse algebra
@njit(**_FLAGS)
def csr_matvec(indptr, indices, data, x, out, blocks):
    """
    out = K x, parallel over row blocks holding equal numbers of nonzeros (see nnz_blocks).
    Splitting by row count instead leaves one thread with most of the work whenever a few
    dense rows -- capacity or balance constraints -- carry most of the nonzeros.
    """
    for b in prange(blocks.shape[0] - 1):
        for i in range(blocks[b], blocks[b + 1]):
            s = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                s += data[j] * x[indices[j]]
            out[i] = s


def nnz_blocks(indptr: np.ndarray, threads: int, per_thread: int = 8) -> np.ndarray:
    """Row boundaries cutting the matrix into ~threads*per_thread blocks of equal nonzeros."""
    m = indptr.shape[0] - 1
    nb = max(1, min(m, threads * per_thread))
    targets = np.linspace(0, indptr[-1], nb + 1)
    bounds = np.searchsorted(indptr, targets, side="left").astype(np.int64)
    bounds[0], bounds[-1] = 0, m
    return np.unique(np.clip(bounds, 0, m))


@njit(**_FLAGS)
def csr_row_absmax(indptr, data, out, blocks):
    for b in prange(blocks.shape[0] - 1):
        for i in range(blocks[b], blocks[b + 1]):
            m = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                a = abs(data[j])
                if a > m:
                    m = a
            out[i] = m


@njit(**_FLAGS)
def csr_row_abssum(indptr, data, out, blocks):
    for b in prange(blocks.shape[0] - 1):
        for i in range(blocks[b], blocks[b + 1]):
            s = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                s += abs(data[j])
            out[i] = s


@njit(cache=True, nogil=True)
def csr_col_absmax(indices, data, out):
    out[:] = 0.0
    for j in range(indices.shape[0]):
        a = abs(data[j])
        if a > out[indices[j]]:
            out[indices[j]] = a


@njit(cache=True, nogil=True)
def csr_col_abssum(indices, data, out):
    out[:] = 0.0
    for j in range(indices.shape[0]):
        out[indices[j]] += abs(data[j])


@njit(**_FLAGS)
def csr_scale(indptr, indices, data, r, c, blocks):
    """data <- diag(r) K diag(c), in place."""
    for b in prange(blocks.shape[0] - 1):
        for i in range(blocks[b], blocks[b + 1]):
            ri = r[i]
            for j in range(indptr[i], indptr[i + 1]):
                data[j] *= ri * c[indices[j]]


# ------------------------------------------------------------------------------ PDHG steps
@njit(**_FLAGS)
def primal_step(x, KTy, c, xl, xu, tau, xp):
    """xp = proj_[xl,xu](x - tau (c - K^T y)); returns ||xp - x||^2."""
    s = 0.0
    for i in prange(x.shape[0]):
        v = x[i] - tau * (c[i] - KTy[i])
        if v < xl[i]:
            v = xl[i]
        elif v > xu[i]:
            v = xu[i]
        xp[i] = v
        d = v - x[i]
        s += d * d
    return s


@njit(**_FLAGS)
def dual_step(y, Kx, Kxp, rl, ru, sigma, yp):
    """
    yp = v + sigma proj_[rl,ru](-v / sigma),  v = y - sigma (2 K xp - K x).
    Returns (||yp - y||^2, (yp - y)^T (K xp - K x)) for the fixed-point residual.
    """
    s_dy = 0.0
    s_int = 0.0
    for i in prange(y.shape[0]):
        v = y[i] - sigma * (2.0 * Kxp[i] - Kx[i])
        w = -v / sigma
        if w < rl[i]:
            w = rl[i]
        elif w > ru[i]:
            w = ru[i]
        nv = v + sigma * w
        yp[i] = nv
        d = nv - y[i]
        s_dy += d * d
        s_int += d * (Kxp[i] - Kx[i])
    return s_dy, s_int


@njit(**_FLAGS)
def halpern(u, up, u0, v, vp, v0, a, b, g):
    """
    Reflected Halpern step on a pair of same-length vectors, in place:
        u <- a ((1 + g) up - g u) + b u0      (and the same for v)
    a = (k+1)/(k+2), b = 1/(k+2). Carrying K x and K^T y through the same linear map keeps
    them exact without an extra mat-vec.
    """
    for i in prange(u.shape[0]):
        u[i] = a * ((1.0 + g) * up[i] - g * u[i]) + b * u0[i]
        v[i] = a * ((1.0 + g) * vp[i] - g * v[i]) + b * v0[i]


@njit(**_FLAGS)
def sq_dist(a, b):
    s = 0.0
    for i in prange(a.shape[0]):
        d = a[i] - b[i]
        s += d * d
    return s


# ------------------------------------------------------------------------------ fused iteration
# Between restart checks, a whole Halpern-PDHG iteration is two passes:
#   rows:    K x+  ->  dual step  ->  Halpern on (y, K x)          (one pass over K)
#   columns: K^T y+ -> Halpern on (x, K^T y) -> next primal step   (one pass over K^T)
# Every update is local to its row / column once that row's product is known, so nothing is
# lost by doing them in the same loop -- and four full-vector passes disappear.

@njit(**_FLAGS)
def fused_rows(indptr, indices, data, blocks, xp, y, Kx, y0, Kx0, rl, ru, sigma, a, b, g, yp, Kxp):
    s_dy = 0.0
    s_int = 0.0
    for blk in prange(blocks.shape[0] - 1):
        for i in range(blocks[blk], blocks[blk + 1]):
            s = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                s += data[j] * xp[indices[j]]
            Kxp[i] = s
            kx = Kx[i]
            yi = y[i]
            v = yi - sigma * (2.0 * s - kx)
            w = -v / sigma
            if w < rl[i]:
                w = rl[i]
            elif w > ru[i]:
                w = ru[i]
            nv = v + sigma * w
            yp[i] = nv
            d = nv - yi
            s_dy += d * d
            s_int += d * (s - kx)
            y[i] = a * ((1.0 + g) * nv - g * yi) + b * y0[i]
            Kx[i] = a * ((1.0 + g) * s - g * kx) + b * Kx0[i]
    return s_dy, s_int


@njit(**_FLAGS)
def fused_cols(indptr, indices, data, blocks, yp, x, xp, x0, KTy, KTy0, KTyp, c, xl, xu, tau, a, b, g):
    """Returns ||x+_next - x_next||^2 for the NEXT iteration's fixed-point residual."""
    s_dx = 0.0
    for blk in prange(blocks.shape[0] - 1):
        for i in range(blocks[blk], blocks[blk + 1]):
            s = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                s += data[j] * yp[indices[j]]
            KTyp[i] = s
            xi = a * ((1.0 + g) * xp[i] - g * x[i]) + b * x0[i]
            kt = a * ((1.0 + g) * s - g * KTy[i]) + b * KTy0[i]
            x[i] = xi
            KTy[i] = kt
            v = xi - tau * (c[i] - kt)
            if v < xl[i]:
                v = xl[i]
            elif v > xu[i]:
                v = xu[i]
            xp[i] = v
            d = v - xi
            s_dx += d * d
    return s_dx


# ------------------------------------------------------------------------------ termination
@njit(**_FLAGS)
def kkt_rows(Kx, y, Dr, b_scale, c_scale, row_lb, row_ub):
    """Row part of the KKT check in the ORIGINAL space: (primal residual^2, dual objective terms)."""
    pres = 0.0
    dobj = 0.0
    for i in prange(Kx.shape[0]):
        kx = Kx[i] / Dr[i] * b_scale
        lo, hi = row_lb[i], row_ub[i]
        viol = 0.0
        if kx < lo:
            viol = kx - lo
        elif kx > hi:
            viol = kx - hi
        pres += viol * viol
        yo = y[i] * Dr[i] * c_scale
        if yo > 0.0 and math.isfinite(lo):
            dobj += yo * lo
        elif yo < 0.0 and math.isfinite(hi):
            dobj += yo * hi
    return pres, dobj


@njit(**_FLAGS)
def kkt_cols(x, KTy, c, Dc, b_scale, c_scale, col_lb, col_ub):
    """Column part: (dual residual^2, primal objective, dual objective terms from bounds)."""
    dres = 0.0
    pobj = 0.0
    dobj = 0.0
    for i in prange(x.shape[0]):
        d = (c[i] - KTy[i]) / Dc[i] * c_scale
        lo, hi = col_lb[i], col_ub[i]
        lfin, ufin = math.isfinite(lo), math.isfinite(hi)
        r = 0.0
        if d > 0.0 and not lfin:
            r = d
        elif d < 0.0 and not ufin:
            r = d
        dres += r * r
        pobj += (c[i] / Dc[i] * c_scale) * (x[i] * Dc[i] * b_scale)
        if d > 0.0 and lfin:
            dobj += d * lo
        elif d < 0.0 and ufin:
            dobj += d * hi
    return dres, pobj, dobj


# ------------------------------------------------------------------------------ kernel sets
# On a small LP, waking a thread team six times per iteration costs more than the arithmetic,
# so the hot-loop kernels also exist single-threaded. Same source; prange runs as range.
_HOT = ("csr_matvec", "primal_step", "dual_step", "halpern", "sq_dist", "kkt_rows", "kkt_cols",
        "fused_rows", "fused_cols")


def _serial(dispatcher):
    py = dispatcher.py_func
    fn = types.FunctionType(py.__code__, py.__globals__, py.__name__ + "_serial",
                            py.__defaults__, py.__closure__)
    fn.__qualname__ = py.__qualname__ + "_serial"   # distinct on-disk cache entry
    return njit(cache=True, nogil=True)(fn)


PARALLEL = SimpleNamespace(nnz_blocks=nnz_blocks, **{n: globals()[n] for n in _HOT})
SERIAL = SimpleNamespace(nnz_blocks=nnz_blocks, **{n: _serial(globals()[n]) for n in _HOT})
