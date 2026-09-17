"""
Bounded revised simplex engine (primal and dual) on the bounded slack form

    min c^T x   s.t.  A x = 0,   lb <= x <= ub.

Implemented from first principles:
- Nonbasic variables rest at a bound (or at zero if free); no upper-bound rows, no artificials.
- Composite primal Phase 1 (minimize sum of infeasibilities, re-priced every iteration).
- Devex pricing (primal), dual steepest-edge pricing (dual).
- Two-pass Harris ratio tests in both algorithms; bound flips for boxed entering variables.
- Random bound perturbation against primal degeneracy, Bland fallback against cycling.
- Terminal decisions (optimal / infeasible / unbounded) are only taken on a fresh
  factorization, so accumulated eta error can never produce a wrong status.
- Warm start from any (head, status) basis, which branch-and-bound uses at every node.
"""
import time
from typing import Callable, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp

from sovereign_opt.solvers.lp.basis import BasisFactor, SingularBasisError, repair_basis

BASIC, AT_LOWER, AT_UPPER, FREE_ZERO = 0, 1, 2, 3


class LPStatus:
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    NUMERICAL_ERROR = "numerical_error"
    DUAL_INFEASIBLE_START = "dual_infeasible_start"


class SimplexEngine:
    def __init__(
        self,
        A: sp.spmatrix,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_struct: int,
        primal_tol: float = 1e-9,
        dual_tol: float = 1e-9,
        pivot_tol: float = 1e-7,
        dense_threshold: int = 600,
        refactor_frequency: int = 100,
        label: Optional[Callable[[int], str]] = None,
        seed: int = 7,
    ):
        self.primal_tol = primal_tol
        self.dual_tol = dual_tol
        self.pivot_tol = pivot_tol
        self.factor = BasisFactor(dense_threshold=dense_threshold, refactor_frequency=refactor_frequency)
        self.label = label or (lambda j: str(j))
        self.rng = np.random.default_rng(seed)
        self.head: Optional[np.ndarray] = None
        self.status: Optional[np.ndarray] = None
        self.total_iterations = 0
        self.phase1_iterations = 0
        self.trace: List[dict] = []
        self.trace_enabled = True
        self.ray: Optional[Tuple[int, float, np.ndarray]] = None
        self.farkas_row: Optional[np.ndarray] = None
        self.n_struct = n_struct
        self._stale = True
        self._xb_dirty = False
        # robustness counters (reported by the benchmark robustness suite)
        self.stats = {"degenerate_pivots": 0, "bland_pivots": 0, "bound_perturbations": 0, "basis_repairs": 0,
                      "bound_flips": 0, "numerical_recoveries": 0}
        self.load(A, c, lb, ub)

    # ================================================================ setup
    @classmethod
    def from_form(cls, form, **kwargs) -> "SimplexEngine":
        return cls(form.A, form.c, form.lb, form.ub, form.n, label=form.column_label, **kwargs)

    def load(self, A, c, lb, ub, keep_basis: bool = True):
        """(Re)load problem data. New trailing rows get their slack column basic."""
        self.A = sp.csc_matrix(A, dtype=np.float64)
        self.AT = sp.csr_matrix(self.A.T)
        self.m, self.N = self.A.shape
        self.c = np.array(c, dtype=np.float64)
        self.lb0 = np.array(lb, dtype=np.float64)
        self.ub0 = np.array(ub, dtype=np.float64)
        self.lb = self.lb0.copy()
        self.ub = self.ub0.copy()
        self.slack_col_of_row = self.n_struct + np.arange(self.m)
        if keep_basis and self.head is not None and len(self.head) <= self.m:
            old_m = len(self.head)
            old_N = len(self.status)
            x_old = self.x
            self.head = np.concatenate([self.head, self.slack_col_of_row[old_m:]])
            status = np.full(self.N, AT_LOWER, dtype=np.int8)
            status[:old_N] = self.status
            status[old_N:] = BASIC
            self.status = status
            self.x = np.zeros(self.N)
            self.x[:old_N] = x_old
            self._sanitize_status()
            self._set_nonbasic_values()
            self._stale = True
        else:
            self.head = None
            self.status = None
            self.x = np.zeros(self.N)
            self._stale = True

    def set_bounds(self, lb: np.ndarray, ub: np.ndarray):
        self.lb0 = np.array(lb, dtype=np.float64)
        self.ub0 = np.array(ub, dtype=np.float64)
        self.lb = self.lb0.copy()
        self.ub = self.ub0.copy()
        if self.status is not None:
            self._sanitize_status()
            self._set_nonbasic_values()
        self._invalidate_values()

    def _invalidate_values(self):
        """Values changed but the basis matrix did not: keep the factorization, recompute x_B."""
        if not self._stale and self.head is not None and self.factor.m == len(self.head) == self.m:
            self._xb_dirty = True
        else:
            self._stale = True

    def set_cost(self, c: np.ndarray):
        self.c = np.array(c, dtype=np.float64)

    def get_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.head.copy(), self.status.copy()

    def set_basis(self, head: np.ndarray, status: np.ndarray):
        new_head = np.array(head, dtype=np.int64)
        same = self.head is not None and len(self.head) == len(new_head) and np.array_equal(self.head, new_head)
        self.head = new_head
        self.status = np.array(status, dtype=np.int8)
        self.status[self.head] = BASIC
        self._sanitize_status()
        self._set_nonbasic_values()
        if same:
            self._invalidate_values()  # same basis matrix (typical for B&B children): reuse the factorization
        else:
            self._stale = True

    def _default_status(self, idx: np.ndarray) -> np.ndarray:
        lb, ub = self.lb[idx], self.ub[idx]
        fl, fu = np.isfinite(lb), np.isfinite(ub)
        st = np.full(len(idx), FREE_ZERO, dtype=np.int8)
        st[fl] = AT_LOWER
        st[~fl & fu] = AT_UPPER
        st[fl & fu & (np.abs(ub) < np.abs(lb))] = AT_UPPER
        return st

    def _nearest_status(self, j: int) -> int:
        lb, ub, v = self.lb[j], self.ub[j], self.x[j]
        fl, fu = np.isfinite(lb), np.isfinite(ub)
        if fl and fu:
            return AT_LOWER if (v - lb) <= (ub - v) else AT_UPPER
        if fl:
            return AT_LOWER
        if fu:
            return AT_UPPER
        return FREE_ZERO

    def _init_slack_basis(self):
        self.head = self.slack_col_of_row.copy()
        self.status = self._default_status(np.arange(self.N))
        self.status[self.head] = BASIC
        self.x = np.zeros(self.N)
        self._set_nonbasic_values()
        self._stale = True

    def _sanitize_status(self):
        st = self.status
        nb = st != BASIC
        fl, fu = np.isfinite(self.lb), np.isfinite(self.ub)
        bad = nb & (((st == AT_LOWER) & ~fl) | ((st == AT_UPPER) & ~fu) | ((st == FREE_ZERO) & (fl | fu)))
        if bad.any():
            idx = np.flatnonzero(bad)
            st[idx] = self._default_status(idx)

    def _set_nonbasic_values(self):
        st = self.status
        m = st == AT_LOWER
        self.x[m] = self.lb[m]
        m = st == AT_UPPER
        self.x[m] = self.ub[m]
        self.x[st == FREE_ZERO] = 0.0

    def _column(self, j: int) -> np.ndarray:
        col = np.zeros(self.m)
        s, e = self.A.indptr[j], self.A.indptr[j + 1]
        col[self.A.indices[s:e]] = self.A.data[s:e]
        return col

    # ============================================================ factor
    def _refactor(self):
        for attempt in range(4):
            try:
                self.factor.factorize(self.A[:, self.head])
                break
            except SingularBasisError:
                tol = 1e-9 if attempt == 0 else 1e-7
                new_head, swaps = repair_basis(self.A, self.head, self.slack_col_of_row, tol=tol)
                if not swaps:
                    if attempt == 3:
                        raise
                    continue
                self.stats["basis_repairs"] += len(swaps)
                for _, old, new in swaps:
                    self.status[old] = self._nearest_status(old)
                    self.status[new] = BASIC
                self.head = new_head
                self._set_nonbasic_values()
        else:
            raise SingularBasisError("basis repair failed")
        self._compute_xB()
        self._stale = False
        self._xb_dirty = False

    def _compute_xB(self):
        xn = self.x.copy()
        xn[self.head] = 0.0
        self.x[self.head] = self.factor.ftran(-(self.A @ xn))

    def _record(self, phase: str, q: int, p: int, dq: float, cost: np.ndarray):
        if not self.trace_enabled:
            return
        it = self.total_iterations
        if it <= 40 or it % 25 == 0:
            self.trace.append({
                "iteration": int(it),
                "phase": phase,
                "entering_var": self.label(int(q)),
                "leaving_var": self.label(int(p)) if p >= 0 else "bound_flip",
                "reduced_cost": float(dq),
                "objective_internal": float(cost @ self.x),
            })

    # ============================================================ primal
    def _primal(self, max_iter: int, deadline: float) -> str:
        m, N = self.m, self.N
        ptol, dtol, pivtol = self.primal_tol, self.dual_tol, self.pivot_tol
        weights = np.ones(N)
        degenerate = 0
        perturbed = False
        restores = 0
        stalls = 0
        iters = 0

        while True:
            if iters >= max_iter:
                return LPStatus.ITERATION_LIMIT
            if iters % 25 == 0 and time.time() > deadline:
                return LPStatus.TIME_LIMIT
            if self._stale or self.factor.needs_refactor:
                self._refactor()
            elif self._xb_dirty:
                self._compute_xB()
                self._xb_dirty = False

            lb, ub = self.lb, self.ub
            head, st = self.head, self.status
            xB = self.x[head]
            lbB, ubB = lb[head], ub[head]
            below = xB < lbB - ptol
            above = xB > ubB + ptol
            phase1 = bool(below.any() or above.any())

            if phase1:
                y = self.factor.btran(above.astype(np.float64) - below.astype(np.float64))
                d = -(self.AT @ y)
            else:
                y = self.factor.btran(self.c[head])
                d = self.c - self.AT @ y
            d[head] = 0.0

            improve = np.full(N, -1.0)
            mk = st == AT_LOWER
            improve[mk] = -d[mk]
            mk = st == AT_UPPER
            improve[mk] = d[mk]
            mk = st == FREE_ZERO
            improve[mk] = np.abs(d[mk])
            improve[ub - lb <= 0.0] = -1.0
            improve[head] = -1.0
            cand = np.flatnonzero(improve > dtol)

            if cand.size == 0:
                if self.factor.num_updates > 0:
                    self._stale = True
                    continue
                if phase1:
                    if perturbed:
                        self._restore_bounds()
                        perturbed = False
                        continue
                    return LPStatus.INFEASIBLE
                if perturbed and restores < 5:
                    self._restore_bounds()
                    perturbed = False
                    restores += 1
                    continue
                return LPStatus.OPTIMAL

            if degenerate > 300:
                self.stats["bland_pivots"] += 1
                q = int(cand[0])  # Bland's rule
            else:
                q = int(cand[np.argmax(improve[cand] ** 2 / weights[cand])])
            direction = 1.0 if (st[q] == AT_LOWER or (st[q] == FREE_ZERO and d[q] < 0)) else -1.0
            alpha = self.factor.ftran(self._column(q))
            delta = -direction * alpha

            # ---------------- two-pass Harris ratio test
            exact = np.full(m, np.inf)
            harris = np.full(m, np.inf)
            to_upper = np.zeros(m, dtype=bool)
            down = delta < -pivtol
            up = delta > pivtol
            feas = ~below & ~above
            k = down & feas & np.isfinite(lbB)
            exact[k] = (xB[k] - lbB[k]) / -delta[k]
            harris[k] = (xB[k] - lbB[k] + ptol) / -delta[k]
            k = up & feas & np.isfinite(ubB)
            exact[k] = (ubB[k] - xB[k]) / delta[k]
            harris[k] = (ubB[k] - xB[k] + ptol) / delta[k]
            to_upper[k] = True
            if phase1:
                k = down & above
                exact[k] = (xB[k] - ubB[k]) / -delta[k]
                harris[k] = exact[k]
                to_upper[k] = True
                k = up & below
                exact[k] = (lbB[k] - xB[k]) / delta[k]
                harris[k] = exact[k]

            t_flip = ub[q] - lb[q] if (np.isfinite(ub[q]) and np.isfinite(lb[q])) else np.inf
            t_max = harris.min() if m > 0 else np.inf

            if np.isinf(t_max) and np.isinf(t_flip):
                if self.factor.num_updates > 0:
                    self._stale = True
                    continue
                if phase1:
                    stalls += 1
                    if stalls > 5:
                        return LPStatus.NUMERICAL_ERROR
                    self._stale = True
                    continue
                self.ray = (q, direction, delta.copy())
                return LPStatus.UNBOUNDED

            if t_flip <= t_max:
                self.stats["bound_flips"] += 1
                self.x[head] += delta * t_flip
                if direction > 0:
                    self.x[q] = ub[q]
                    st[q] = AT_UPPER
                else:
                    self.x[q] = lb[q]
                    st[q] = AT_LOWER
                iters += 1
                self.total_iterations += 1
                if phase1:
                    self.phase1_iterations += 1
                degenerate = 0
                self._record("phase1" if phase1 else "phase2", q, -1, d[q], self.c)
                continue

            elig = np.flatnonzero(exact <= t_max)
            r = int(elig[np.argmax(np.abs(delta[elig]))])
            t = max(float(exact[r]), 0.0)
            p = int(head[r])
            arq = alpha[r]

            rho = self.factor.row_of_inverse(r)
            alpha_row = self.AT @ rho

            self.x[head] += delta * t
            self.x[q] += direction * t
            if to_upper[r]:
                self.x[p] = ub[p]
                st[p] = AT_UPPER if ub[p] > lb[p] else AT_LOWER
            else:
                self.x[p] = lb[p]
                st[p] = AT_LOWER
            st[q] = BASIC
            head[r] = q
            self.factor.update(r, alpha)

            ratio = alpha_row / arq
            wq = weights[q]
            nb = st != BASIC
            weights[nb] = np.maximum(weights[nb], ratio[nb] ** 2 * wq)
            weights[p] = max(wq / (arq * arq), 1.0)
            if weights.max() > 1e10:
                weights[:] = 1.0

            iters += 1
            self.total_iterations += 1
            if phase1:
                self.phase1_iterations += 1
            self._record("phase1" if phase1 else "phase2", q, p, d[q], self.c)

            degenerate = degenerate + 1 if t <= 1e-12 else 0
            self.stats["degenerate_pivots"] += t <= 1e-12
            if degenerate >= 50 and not perturbed:
                self.stats["bound_perturbations"] += 1
                self._perturb_bounds()
                perturbed = True

    def _perturb_bounds(self):
        scale = 1e-7 * (1.0 + np.maximum(np.abs(np.nan_to_num(self.lb0, posinf=0, neginf=0)),
                                         np.abs(np.nan_to_num(self.ub0, posinf=0, neginf=0))))
        xi = self.rng.uniform(0.5, 1.0, self.N) * scale
        boxed_or_half = self.ub0 > self.lb0
        basic = self.status == BASIC
        k = basic & boxed_or_half
        self.lb = self.lb0.copy()
        self.ub = self.ub0.copy()
        self.lb[k] -= xi[k]
        self.ub[k] += xi[k]

    def _restore_bounds(self):
        self.lb = self.lb0.copy()
        self.ub = self.ub0.copy()
        self._sanitize_status()
        self._set_nonbasic_values()
        self._stale = True

    # ============================================================== dual
    def _make_dual_feasible(self, d: np.ndarray) -> bool:
        st = self.status
        dtol = self.dual_tol
        boxed = np.isfinite(self.lb) & np.isfinite(self.ub)
        bad_l = (st == AT_LOWER) & (d < -dtol)
        bad_u = (st == AT_UPPER) & (d > dtol)
        bad_f = (st == FREE_ZERO) & (np.abs(d) > dtol)
        if (bad_l & ~boxed).any() or (bad_u & ~boxed).any() or bad_f.any():
            return False
        if bad_l.any() or bad_u.any():
            st[bad_l] = AT_UPPER
            st[bad_u] = AT_LOWER
            self._set_nonbasic_values()
            self._compute_xB()
        return True

    def _dual(self, max_iter: int, deadline: float) -> str:
        m, N = self.m, self.N
        ptol, dtol, pivtol = self.primal_tol, self.dual_tol, self.pivot_tol
        dse = np.ones(m)
        iters = 0
        stalls = 0
        degenerate = 0

        while True:
            if iters >= max_iter:
                return LPStatus.ITERATION_LIMIT
            if iters % 25 == 0 and time.time() > deadline:
                return LPStatus.TIME_LIMIT
            if self._stale or self.factor.needs_refactor:
                self._refactor()
            elif self._xb_dirty:
                self._compute_xB()
                self._xb_dirty = False

            lb, ub = self.lb, self.ub
            head, st = self.head, self.status
            y = self.factor.btran(self.c[head])
            d = self.c - self.AT @ y
            d[head] = 0.0
            if not self._make_dual_feasible(d):
                return LPStatus.DUAL_INFEASIBLE_START

            xB = self.x[head]
            lbB, ubB = lb[head], ub[head]
            infeas = np.maximum(lbB - xB, 0.0) + np.maximum(xB - ubB, 0.0)
            infeas[infeas <= ptol] = 0.0
            if not infeas.any():
                if self.factor.num_updates > 0:
                    self._stale = True
                    continue
                return LPStatus.OPTIMAL

            r = int(np.argmax(infeas * infeas / dse))
            p = int(head[r])
            s = 1.0 if xB[r] > ubB[r] else -1.0
            bound = ubB[r] if s > 0 else lbB[r]
            delta0 = xB[r] - bound

            rho = self.factor.row_of_inverse(r)
            alpha_row = self.AT @ rho
            at = s * alpha_row
            cand = (
                ((st == AT_LOWER) & (at > pivtol))
                | ((st == AT_UPPER) & (at < -pivtol))
                | ((st == FREE_ZERO) & (np.abs(at) > pivtol))
            ) & (ub > lb)
            idx = np.flatnonzero(cand)
            if idx.size == 0:
                if self.factor.num_updates > 0:
                    self._stale = True
                    continue
                self.farkas_row = s * rho
                return LPStatus.INFEASIBLE

            dj, aj, sj = d[idx], at[idx], st[idx]
            harris = np.where(sj == AT_LOWER, (dj + dtol) / aj,
                              np.where(sj == AT_UPPER, (dj - dtol) / aj, dtol / np.abs(aj)))
            exact = np.maximum(np.where(sj == FREE_ZERO, 0.0, dj / aj), 0.0)
            t_max = harris.min()
            elig = np.flatnonzero(exact <= t_max)
            if degenerate > 300:
                self.stats["bland_pivots"] += 1
                q = int(idx[elig].min())
            else:
                q = int(idx[elig[np.argmax(np.abs(aj[elig]))]])

            alpha = self.factor.ftran(self._column(q))
            arq = alpha[r]
            if abs(arq) < 1e-11 or abs(arq - alpha_row[q]) > 1e-7 * (1.0 + abs(arq)):
                stalls += 1
                self.stats["numerical_recoveries"] += 1
                if stalls > 10:
                    return LPStatus.NUMERICAL_ERROR
                self._stale = True
                continue

            theta = delta0 / arq
            tau = self.factor.ftran(rho)
            wr = dse[r]
            ratio = alpha / arq
            dse = np.maximum(dse - 2.0 * ratio * tau + ratio * ratio * wr, 1e-6)
            dse[r] = max(wr / (arq * arq), 1e-6)

            self.x[head] -= theta * alpha
            self.x[q] += theta
            self.x[p] = bound
            st[p] = AT_UPPER if (s > 0 and ub[p] > lb[p]) else AT_LOWER
            st[q] = BASIC
            head[r] = q
            self.factor.update(r, alpha)

            degenerate = degenerate + 1 if abs(d[q]) <= dtol else 0
            self.stats["degenerate_pivots"] += abs(d[q]) <= dtol
            iters += 1
            self.total_iterations += 1
            self._record("dual", q, p, d[q], self.c)

    # ============================================================= driver
    def solve(self, max_iterations: int = 200000, time_limit: float = 60.0, method: str = "auto") -> str:
        """
        method: 'primal', 'dual', or 'auto' (dual simplex when the start basis is dual
        feasible after bound flips, primal otherwise). Returns an LPStatus string.
        """
        deadline = time.time() + max(time_limit, 1e-3)
        start_iters = self.total_iterations
        self.ray = None
        self.farkas_row = None
        if self.head is None:
            self._init_slack_basis()
        try:
            if self._stale or self.factor.needs_refactor or self.factor.m != self.m:
                self._refactor()
            else:
                self._compute_xB()
                self._xb_dirty = False
            if method in ("dual", "auto"):
                st = self._dual(max_iterations, deadline)
                if st != LPStatus.DUAL_INFEASIBLE_START and st != LPStatus.NUMERICAL_ERROR:
                    if st != LPStatus.OPTIMAL:
                        return st
                    # Confirm with the primal method (normally zero iterations).
            remaining = max_iterations - (self.total_iterations - start_iters)
            return self._primal(max(remaining, 1), deadline)
        except SingularBasisError:
            return LPStatus.NUMERICAL_ERROR

    # ============================================================ output
    def duals(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._stale:
            self._refactor()
        y = self.factor.btran(self.c[self.head])
        d = self.c - self.AT @ y
        d[self.head] = 0.0
        return y, d

    def primal_infeasibility(self) -> float:
        v = np.maximum(self.lb0 - self.x, 0.0) + np.maximum(self.x - self.ub0, 0.0)
        return float(v.max()) if v.size else 0.0

    def objective(self) -> float:
        return float(self.c @ self.x)
