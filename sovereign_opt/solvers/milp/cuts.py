"""
Cutting planes for mixed-integer linear programming.

- Gomory mixed-integer (GMI) cuts from optimal simplex tableau rows of fractional basic integer variables.
- Knapsack cover cuts from original rows after bounding non-binary terms and complementing binaries.
- Cut cleaning (tiny coefficient removal with safe right-hand-side relaxation, dynamic range limits),
  efficacy filtering, and parallelism filtering.

All cuts are returned in unscaled structural coordinates as (kind, pi, lower, upper).
"""
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp

from sovereign_opt.solvers.lp.simplex_engine import BASIC, AT_UPPER, FREE_ZERO

Cut = Tuple[str, np.ndarray, float, float]


def separate_gmi_cuts(engine, form, int_col_mask: np.ndarray, max_cuts: int = 50, away: float = 0.01) -> List[Cut]:
    """
    Tableau row of basic integer x_p:  x_p + sum_{j nonbasic} alpha_j x_j = 0  (bounded slack form, b = 0).
    Substituting x_j = l_j + t_j (at lower) or x_j = u_j - t_j (at upper) gives x_p + sum a'_j t_j = beta.
    With f0 = frac(beta) the GMI inequality is
        sum_{j int} min(f_j/f0, (1-f_j)/(1-f0)) t_j + sum_{j cont} max(a'_j/f0, -a'_j/(1-f0)) t_j >= 1.
    """
    e = engine
    n = form.n
    N = e.N
    head, st, x = e.head, e.status, e.x
    lb, ub = e.lb, e.ub
    nb = st != BASIC
    at_u = st == AT_UPPER
    fixed = (ub - lb) <= 0.0
    free_nb = nb & (st == FREE_ZERO)
    int_mask = np.zeros(N, dtype=bool)
    int_mask[:n] = int_col_mask[:n]

    rows = []
    for r, p in enumerate(head):
        if p < n and int_col_mask[p]:
            f0 = x[p] - np.floor(x[p])
            if away < f0 < 1.0 - away:
                rows.append((abs(f0 - 0.5), r))
    rows.sort()
    A_struct = sp.csr_matrix(form.A[:, :n])
    cuts: List[Cut] = []
    for _, r in rows[: 2 * max_cuts]:
        rho = e.factor.row_of_inverse(r)
        alpha = e.AT @ rho
        alpha[head] = 0.0
        nz = nb & (np.abs(alpha) > 1e-11) & ~fixed
        if np.any(nz & free_nb):
            continue
        beta = x[head[r]]
        f0 = beta - np.floor(beta)
        if not (away < f0 < 1.0 - away):
            continue
        ap = np.where(at_u, -alpha, alpha)
        fj = ap - np.floor(ap)
        coef_int = np.where(fj <= f0, fj / f0, (1.0 - fj) / (1.0 - f0))
        coef_cont = np.where(ap >= 0.0, ap / f0, -ap / (1.0 - f0))
        coef = np.where(nz & int_mask, coef_int, np.where(nz, coef_cont, 0.0))
        sign = np.where(at_u, -1.0, 1.0)
        pi = coef * sign
        bnd = np.where(at_u, ub, lb)
        used = coef != 0.0
        if not np.all(np.isfinite(bnd[used])):
            continue
        pi0 = 1.0 + float(np.sum(pi[used] * bnd[used]))
        pi_struct = pi[:n].copy()
        slack_part = pi[n:]
        if np.any(slack_part != 0.0):
            pi_struct += A_struct.T @ slack_part
        pi_unscaled = pi_struct / form.col_scale[:n]
        if np.all(np.isfinite(pi_unscaled)) and np.isfinite(pi0):
            cuts.append(("gomory", pi_unscaled, pi0, np.inf))
        if len(cuts) >= max_cuts:
            break
    return cuts


def separate_cover_cuts(
    A: sp.csr_matrix,
    rl: np.ndarray,
    ru: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    binary_mask: np.ndarray,
    x: np.ndarray,
    max_cuts: int = 50,
) -> List[Cut]:
    """Minimal cover inequalities  sum_{C+} x_j - sum_{C-} x_j <= |C| - 1 - |C-|  violated by x."""
    A = sp.csr_matrix(A)
    n = A.shape[1]
    found = []
    for i in range(A.shape[0]):
        s, t = A.indptr[i], A.indptr[i + 1]
        cols, vals = A.indices[s:t], A.data[s:t]
        binm = binary_mask[cols]
        if binm.sum() < 2:
            continue
        for side in (0, 1):
            if side == 0:
                if not np.isfinite(ru[i]):
                    continue
                a, b = vals, ru[i]
            else:
                if not np.isfinite(rl[i]):
                    continue
                a, b = -vals, -rl[i]
            other = ~binm
            ao, co = a[other], cols[other]
            low = np.where(ao > 0, cl[co], cu[co])
            if np.any(~np.isfinite(low) & (ao != 0)):
                continue
            b_rel = b - float(np.sum(ao * np.where(np.isfinite(low), low, 0.0)))
            ab, cb = a[binm], cols[binm]
            neg = ab < 0
            w = np.abs(ab)
            b_rel -= float(ab[neg].sum())
            if b_rel < 0 or w.sum() <= b_rel + 1e-9:
                continue
            zstar = np.where(neg, 1.0 - x[cb], x[cb])
            order = np.argsort((1.0 - zstar) / np.maximum(w, 1e-12))
            cover, total = [], 0.0
            for k in order:
                cover.append(k)
                total += w[k]
                if total > b_rel + 1e-9:
                    break
            if total <= b_rel + 1e-9:
                continue
            for k in sorted(cover, key=lambda q: w[q]):
                if total - w[k] > b_rel + 1e-9 and len(cover) > 2:
                    cover.remove(k)
                    total -= w[k]
            cover = np.array(cover)
            violation = float(zstar[cover].sum()) - (len(cover) - 1)
            if violation <= 1e-6:
                continue
            pi = np.zeros(n)
            pos, ng = cover[~neg[cover]], cover[neg[cover]]
            pi[cb[pos]] = 1.0
            pi[cb[ng]] = -1.0
            found.append((violation, ("cover", pi, -np.inf, float(len(cover) - 1 - len(ng)))))
    found.sort(key=lambda item: -item[0])
    return [c for _, c in found[:max_cuts]]


def _clean(pi: np.ndarray, pi0: float, cl: np.ndarray, cu: np.ndarray):
    """Normalize a >= cut, drop tiny coefficients with a valid rhs relaxation, reject bad dynamic range."""
    maxc = float(np.max(np.abs(pi), initial=0.0))
    if maxc <= 0.0 or not np.isfinite(maxc) or not np.isfinite(pi0):
        return None
    pi = pi / maxc
    pi0 = pi0 / maxc
    tiny = (np.abs(pi) < 1e-9) & (pi != 0.0)
    if tiny.any():
        t = np.flatnonzero(tiny)
        worst = np.maximum(pi[t] * cl[t], pi[t] * cu[t])
        worst = np.where(np.isnan(worst), np.inf, worst)
        if not np.all(np.isfinite(worst)):
            return None
        pi0 -= float(np.sum(worst))
        pi[t] = 0.0
    nzv = np.abs(pi[pi != 0.0])
    if nzv.size == 0 or nzv.max() / nzv.min() > 1e8:
        return None
    return pi, pi0


def select_cuts(
    cuts: List[Cut],
    x: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    max_cuts: int = 50,
    min_efficacy: float = 1e-4,
    max_parallelism: float = 0.98,
) -> List[Cut]:
    scored = []
    for kind, pi, lo, hi in cuts:
        if np.isfinite(lo):
            cleaned = _clean(pi, lo, cl, cu)
        else:
            cleaned = _clean(-pi, -hi, cl, cu)
        if cleaned is None:
            continue
        g, g0 = cleaned
        violation = g0 - float(g @ x)
        norm = float(np.linalg.norm(g))
        if norm <= 0.0 or violation <= 1e-6 * max(1.0, abs(g0)):
            continue
        efficacy = violation / norm
        if efficacy >= min_efficacy:
            scored.append((efficacy, kind, g / norm, g0 / norm))
    scored.sort(key=lambda item: -item[0])
    chosen: List[Tuple[str, np.ndarray, float]] = []
    for eff, kind, g, g0 in scored:
        if any(abs(float(g @ h)) > max_parallelism for _, h, _ in chosen):
            continue
        chosen.append((kind, g, g0))
        if len(chosen) >= max_cuts:
            break
    return [(kind, g, g0, np.inf) for kind, g, g0 in chosen]
