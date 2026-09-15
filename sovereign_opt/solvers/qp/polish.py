"""
Active-set solution polishing and dual recovery for LP / QP solutions.

An interior point method stops at a point that is only approximately complementary
(x_j - l_j ~ 1e-8, z_j ~ 1e-8). Polishing:
  1. guesses the optimal active set from the relative size of slacks versus multipliers,
  2. solves the equality-constrained KKT system for the primal point,
  3. recovers sign-constrained multipliers with a sovereign Lawson-Hanson NNLS solve
     (needed on degenerate problems, where multipliers are not unique and a plain
     least-squares solve returns wrong signs),
and accepts the result only if it is primal feasible, stationary with correctly signed
multipliers, and no worse in objective.

`recover_duals` performs only step 3 for a given primal point (used after presolve/postsolve,
where the reduced problem's multipliers do not map back to the original rows).
"""
from typing import Optional, Tuple
import numpy as np

from sovereign_opt.solvers.lp.standard_form import BoundedForm


def nnls(G: np.ndarray, g: np.ndarray, max_iter: Optional[int] = None, tol: float = 1e-12) -> Tuple[np.ndarray, float]:
    """Lawson-Hanson active-set method for min ||G lam - g||_2 subject to lam >= 0."""
    n, p = G.shape
    lam = np.zeros(p)
    if p == 0:
        return lam, float(np.linalg.norm(g, np.inf)) if g.size else 0.0
    P = np.zeros(p, dtype=bool)
    max_iter = max_iter or 3 * p + 10
    scale = 1.0 + float(np.max(np.abs(G.T @ g), initial=0.0))
    w = G.T @ (g - G @ lam)
    it = 0
    while (~P).any() and np.max(np.where(~P, w, -np.inf)) > tol * scale and it < max_iter:
        j = int(np.argmax(np.where(~P, w, -np.inf)))
        P[j] = True
        while True:
            it += 1
            s = np.zeros(p)
            s[P] = np.linalg.lstsq(G[:, P], g, rcond=None)[0]
            if np.all(s[P] > tol) or it >= max_iter:
                break
            neg = P & (s <= tol)
            alpha = float(np.min(lam[neg] / np.maximum(lam[neg] - s[neg], 1e-300)))
            lam = lam + alpha * (s - lam)
            P &= lam > tol
            lam[~P] = 0.0
        lam = np.where(P, np.maximum(s, 0.0), 0.0)
        w = G.T @ (g - G @ lam)
    return lam, float(np.linalg.norm(G @ lam - g, np.inf))


def _dense_data(form: BoundedForm):
    n = form.n
    A = form.A_orig.toarray()
    c = form.c_orig
    Q = form.Q_orig.toarray() if form.Q_orig is not None else np.zeros((n, n))
    cs, rs = form.col_scale[:n], form.row_scale
    cl, cu = form.lb[:n] * cs, form.ub[:n] * cs
    rl, ru = form.lb[n:] / rs, form.ub[n:] / rs
    return A, c, Q, cl, cu, rl, ru


def _signed_multipliers(A, grad, r_low, r_up, r_eq, v_low, v_up, v_fix) -> Optional[np.ndarray]:
    """Solve grad = A^T y + z with sign-constrained y (rows) and z (bounds) by NNLS. Returns y or None."""
    n = A.shape[1]
    m = A.shape[0]
    cols, owners = [], []
    for i in np.flatnonzero(r_low | r_up):
        sides = (1.0, -1.0) if r_eq[i] else ((1.0,) if r_low[i] else (-1.0,))
        for sgn in sides:
            cols.append(sgn * A[i])
            owners.append((int(i), sgn))
    for j in np.flatnonzero(v_low | v_up):
        sides = (1.0, -1.0) if v_fix[j] else ((1.0,) if v_low[j] else (-1.0,))
        for sgn in sides:
            e = np.zeros(n)
            e[j] = sgn
            cols.append(e)
            owners.append((-1, sgn))
    G = np.column_stack(cols) if cols else np.zeros((n, 0))
    lam, resid = nnls(G, grad)
    if resid > 1e-8 * (1.0 + float(np.max(np.abs(grad), initial=0.0))):
        return None
    y = np.zeros(m)
    for (row, sgn), val in zip(owners, lam):
        if row >= 0:
            y[row] += sgn * val
    return y


def recover_duals(form: BoundedForm, x: np.ndarray, active_tol: float = 1e-7, max_dim: int = 3000) -> Optional[np.ndarray]:
    """Row multipliers certifying optimality of a given primal point, or None if none are found."""
    n, m = form.n, form.m
    if n + m > max_dim:
        return None
    A, c, Q, cl, cu, rl, ru = _dense_data(form)
    act = A @ x
    grad = Q @ x + c
    tv = active_tol * np.maximum(1.0, np.abs(x))
    tr = active_tol * np.maximum(1.0, np.abs(act))
    v_fix = np.isfinite(cl) & np.isfinite(cu) & (cu - cl <= 1e-12 * np.maximum(1.0, np.abs(cl)))
    v_low = np.isfinite(cl) & (x - cl <= tv)
    v_up = np.isfinite(cu) & (cu - x <= tv) & ~(v_low & ~v_fix)
    r_eq = np.isfinite(rl) & np.isfinite(ru) & (ru - rl <= 1e-12 * np.maximum(1.0, np.abs(rl)))
    r_low = np.isfinite(rl) & (act - rl <= tr)
    r_up = np.isfinite(ru) & (ru - act <= tr) & ~(r_low & ~r_eq)
    return _signed_multipliers(A, grad, r_low, r_up, r_eq, v_low, v_up, v_fix)


def polish_solution(
    form: BoundedForm,
    x: np.ndarray,
    y: np.ndarray,
    feas_tol: float = 1e-9,
    max_dim: int = 3000,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    form: bounded form (unscaled data is read from c_orig / Q_orig / A_orig).
    x: structural primal values (unscaled); y: row duals (unscaled, minimize-normalized convention).
    Returns (x_polished, y_polished) or None if polishing is not accepted.
    """
    n, m = form.n, form.m
    if n + m > max_dim:
        return None
    A, c, Q, cl, cu, rl, ru = _dense_data(form)
    act = A @ x
    grad = Q @ x + c
    d = grad - A.T @ y

    # 1. Active-set guess: a bound is active when its slack is smaller than its multiplier.
    v_fix = np.isfinite(cl) & np.isfinite(cu) & (cu - cl <= 1e-12 * np.maximum(1.0, np.abs(cl)))
    v_low = (np.isfinite(cl) & ((x - cl) < np.maximum(d, 0.0))) | v_fix
    v_up = np.isfinite(cu) & ((cu - x) < np.maximum(-d, 0.0)) & ~v_low
    r_eq = np.isfinite(rl) & np.isfinite(ru) & (ru - rl <= 1e-12 * np.maximum(1.0, np.abs(rl)))
    r_low = (np.isfinite(rl) & ((act - rl) < np.maximum(y, 0.0))) | r_eq
    r_up = np.isfinite(ru) & ((ru - act) < np.maximum(-y, 0.0)) & ~r_low

    # 2. Primal point from the equality-constrained KKT system.
    B = v_low | v_up
    F = ~B
    xB = np.where(v_low, cl, cu)[B]
    act_rows = np.flatnonzero(r_low | r_up)
    rhs_rows = np.where(r_low, rl, ru)[act_rows]
    nF, k = int(F.sum()), len(act_rows)
    AF = A[np.ix_(act_rows, np.flatnonzero(F))]
    AB = A[np.ix_(act_rows, np.flatnonzero(B))]
    KKT = np.block([[Q[np.ix_(F, F)], -AF.T], [AF, np.zeros((k, k))]])
    rhs = np.concatenate([-c[F] - Q[np.ix_(F, B)] @ xB, rhs_rows - AB @ xB])
    try:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(sol)) or np.linalg.norm(KKT @ sol - rhs, np.inf) > 1e-8 * (1.0 + np.linalg.norm(rhs, np.inf)):
        return None
    xp = x.copy()
    xp[B] = xB
    xp[F] = sol[:nF]

    # 3. Primal feasibility of the polished point.
    actp = A @ xp
    sv = feas_tol * np.maximum(1.0, np.abs(xp))
    sr = feas_tol * np.maximum(1.0, np.abs(actp))
    if np.any(np.isfinite(cl) & (xp < cl - sv)) or np.any(np.isfinite(cu) & (xp > cu + sv)):
        return None
    if np.any(np.isfinite(rl) & (actp < rl - sr)) or np.any(np.isfinite(ru) & (actp > ru + sr)):
        return None

    # 4. Sign-constrained multipliers at the polished point.
    yp = _signed_multipliers(A, Q @ xp + c, r_low, r_up, r_eq, v_low, v_up, v_fix)
    if yp is None:
        return None

    # 5. Objective must not get worse.
    f_old = c @ x + 0.5 * x @ Q @ x
    f_new = c @ xp + 0.5 * xp @ Q @ xp
    if f_new > f_old + 1e-7 * max(1.0, abs(f_old)):
        return None
    lo = F & np.isfinite(cl)
    xp[lo] = np.maximum(xp[lo], cl[lo])
    hi = F & np.isfinite(cu)
    xp[hi] = np.minimum(xp[hi], cu[hi])
    return xp, yp
