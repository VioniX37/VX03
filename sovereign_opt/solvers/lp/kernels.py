"""
Numba-compiled single-pass kernels for the dual simplex iteration.

Each kernel replaces a chain of vectorized NumPy expressions over all N columns (each allocating a
temporary array) with one compiled loop. Semantics are identical to the NumPy code in simplex_engine.py,
which remains the fallback when Numba is not installed.
"""
import numpy as np

from sovereign_opt.solvers.lp.basis import HAVE_NUMBA, njit

BASIC, AT_LOWER, AT_UPPER, FREE_ZERO = 0, 1, 2, 3


@njit(cache=True, nogil=True)
def dual_feasibility_flips(st, d, lb, ub, dtol):
    """-1: dual infeasible and not fixable by bound flips; 1: flips applied in place; 0: already dual feasible."""
    n = st.shape[0]
    need = False
    for j in range(n):
        s = st[j]
        if s == AT_LOWER and d[j] < -dtol:
            if not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
                return -1
            need = True
        elif s == AT_UPPER and d[j] > dtol:
            if not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
                return -1
            need = True
        elif s == FREE_ZERO and abs(d[j]) > dtol:
            return -1
    if not need:
        return 0
    for j in range(n):
        s = st[j]
        if s == AT_LOWER and d[j] < -dtol:
            st[j] = AT_UPPER
        elif s == AT_UPPER and d[j] > dtol:
            st[j] = AT_LOWER
    return 1


@njit(cache=True, nogil=True)
def choose_leaving_row(xB, lbB, ubB, weights, ptol):
    """Row with the largest squared primal infeasibility / pricing weight; -1 when primal feasible."""
    best = -1
    best_score = 0.0
    for i in range(xB.shape[0]):
        v = 0.0
        if xB[i] < lbB[i]:
            v = lbB[i] - xB[i]
        elif xB[i] > ubB[i]:
            v = xB[i] - ubB[i]
        if v > ptol:
            score = v * v / weights[i]
            if score > best_score:
                best_score = score
                best = i
    return best


@njit(cache=True, nogil=True)
def dual_ratio_test(st, at, d, lb, ub, pivtol, dtol, bland):
    """Two-pass Harris dual ratio test. Returns the entering column, or -1 if the row proves infeasibility."""
    n = st.shape[0]
    t_max = np.inf
    found = False
    for j in range(n):
        s = st[j]
        a = at[j]
        if s == BASIC or not (ub[j] > lb[j]):
            continue
        if s == AT_LOWER and a > pivtol:
            h = (d[j] + dtol) / a
        elif s == AT_UPPER and a < -pivtol:
            h = (d[j] - dtol) / a
        elif s == FREE_ZERO and abs(a) > pivtol:
            h = dtol / abs(a)
        else:
            continue
        found = True
        if h < t_max:
            t_max = h
    if not found:
        return -1
    q = -1
    best = -1.0
    for j in range(n):
        s = st[j]
        a = at[j]
        if s == BASIC or not (ub[j] > lb[j]):
            continue
        if s == AT_LOWER and a > pivtol:
            ex = d[j] / a
        elif s == AT_UPPER and a < -pivtol:
            ex = d[j] / a
        elif s == FREE_ZERO and abs(a) > pivtol:
            ex = 0.0
        else:
            continue
        if ex < 0.0:
            ex = 0.0
        if ex <= t_max:
            if bland:
                return j
            if abs(a) > best:
                best = abs(a)
                q = j
    return q
