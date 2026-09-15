# 07 — Sovereign Optimizer v2 Engine Upgrades

This chapter documents the algorithms introduced in v2.0 and the correctness arguments behind them.

## 1. Shared computational form

Every model is converted to the bounded slack form

$$\min\; c^T x + \tfrac12 x^T Q x \quad \text{s.t.}\quad [A\;\; -I]\begin{bmatrix}x\\ s\end{bmatrix} = 0,\qquad l_x \le x \le u_x,\;\; l_r \le s \le u_r .$$

One slack per row carries the row bounds, so $\le$, $\ge$, equality, and ranged rows are handled identically, free variables need no splitting, and $-I$ is always a valid starting basis. Power-of-two geometric scaling $A_s = RAC$ keeps slack columns equal to $-I$ and introduces no rounding error. Unscaling: $x = Cx_s$, $y = Ry_s$, $d = d_s / C$.

**Dual convention (minimize-normalized):** $\nabla f(x) = A^T y + z$, where $y_i > 0$ means row $i$ is active at its lower bound, and $z_j > 0$ means variable $j$ is at its lower bound.

## 2. Linear programming

### 2.1 Bounded revised simplex (`simplex_engine.py`)
- Nonbasic variables rest at a bound (or at 0 when free); boxed entering variables may *bound-flip* without a basis change.
- **Primal:** composite Phase 1 minimizes the sum of infeasibilities, re-priced every iteration; Devex pricing; two-pass Harris ratio test; random bound perturbation after 50 degenerate pivots, removed before termination; Bland's rule as a last resort.
- **Dual:** dual steepest-edge pricing, Harris dual ratio test, automatic bound flips to restore dual feasibility. Used automatically when the start basis is dual feasible (always true after a branching bound change).
- **Factorization:** explicit inverse with $O(m^2)$ eta pivots for $m \le 600$, SuperLU plus a product-form eta file otherwise, refactorized every 100 updates. Singular bases are repaired by QR column selection and slack substitution.
- **Safety rule:** optimal, infeasible, and unbounded are only declared on a fresh factorization.

### 2.2 Homogeneous self-dual IPM (`interior_point.py`)
Variables are mapped to $z \ge 0$ (shift, negate, or split), keeping native upper bounds $z_U + w = u$. The embedding

$$A z - b\tau = 0,\quad z_U + w - u\tau = 0,\quad A^T y + s - E v - c\tau = 0,\quad -c^Tz + b^Ty - u^Tv - \kappa = 0$$

is solved by Mehrotra predictor-corrector steps. Each step reduces to two normal-equation solves with $M = A D A^T$ (dense Cholesky or sparse LU, regularized). As $\tau \to 0$ with $\kappa > 0$, the iterate is a Farkas certificate: a dual ray proves infeasibility, a primal ray proves dual infeasibility. Unboundedness is reported only after the simplex verifies a feasible point.

Gondzio multiple-centrality correctors were prototyped and removed: in randomized testing they caused loss of centrality ($\mu \to 10^{-24}$ with stalled residuals), while plain Mehrotra converged in 15–19 iterations on the same instances.

**Crossover** selects a basis by weighted pivoted QR (weights = distance from bounds) and finishes with primal simplex, giving vertex solutions and exact duals.

## 3. Convex quadratic programming

### 3.1 QP interior point (`interior_point_qp.py`)
Mehrotra predictor-corrector on the regularized quasi-definite system

$$\begin{bmatrix}-(Q + X^{-1}S + W^{-1}V + \rho I) & A^T\\ A & \delta I\end{bmatrix}\begin{bmatrix}\Delta x\\ \Delta y\end{bmatrix} = \begin{bmatrix} r_d' \\ r_p\end{bmatrix}.$$

- A Mehrotra least-squares starting point (upper-bounded variables at interval midpoints) is essential: with unit slacks on bounds of size $6\cdot 10^4$ the step length collapses to $10^{-8}$.
- Convexity is checked on the active block of $Q$ (eigenvalues when small, symmetric-mode LU inertia when large).
- Divergence triggers a simplex feasibility test. Unboundedness additionally requires a verified recession direction $d \ge 0$ with $Ad \approx 0$, $d_U \approx 0$, $Qd \approx 0$, $c^T d < 0$.

### 3.2 Solution polishing (`polish.py`)
The active set is guessed by comparing slacks with multipliers. The equality-constrained KKT system is solved, and multipliers are recovered by **Lawson–Hanson NNLS** with sign constraints. This matters on degenerate problems, where multipliers are not unique and least squares returns wrongly signed values. The polished point is accepted only if it is feasible, stationary, and no worse in objective.

### 3.3 Active set (`active_set.py`)
Null-space method with an eigen-decomposed reduced Hessian (handles semidefinite $Q$ and detects zero-curvature descent), simplex Phase 1 start, unit-norm rows, a $10^{-9}$ right-hand-side relaxation against degeneracy, Bland-ordered swaps for dependent blocking constraints, and final polishing on the unrelaxed data.

## 4. Mixed-integer programming (`branch_bound.py`)

- **Warm starts:** one persistent simplex engine; each node re-solves with dual simplex from its parent's basis.
- **Gomory mixed-integer cuts:** from tableau row $x_p + \sum_j \bar a_j t_j = \beta$ with $f_0 = \mathrm{frac}(\beta)$,
  $\sum_{j\in I}\min\!\big(\tfrac{f_j}{f_0},\tfrac{1-f_j}{1-f_0}\big)t_j + \sum_{j\notin I}\max\!\big(\tfrac{\bar a_j}{f_0},\tfrac{-\bar a_j}{1-f_0}\big)t_j \ge 1$,
  with slack columns substituted back into structural space. Integer columns are left unscaled to preserve integrality.
- **Knapsack covers** from any row after bounding non-binary terms and complementing negative binaries; covers are made minimal.
- **Cut management:** tiny-coefficient removal with a valid right-hand-side relaxation, dynamic range $\le 10^8$, efficacy $\ge 10^{-4}$, parallelism $\le 0.98$, rollback if a round makes the LP non-optimal.
- **Reliability branching:** pseudocosts initialized by strong branching (8 candidates, 100 dual simplex iterations). Product score, with the ML structural score as tie-breaker. Strong branching also tightens bounds when a child is infeasible.
- **Reduced-cost fixing:** $x_j \le l_j + \lfloor(z_{inc} - z)/d_j\rfloor$ at every node (global at the root).
- **Heuristics:** rounding, fractional diving, feasibility pump (binary programs), RINS sub-MIP.
- **Bounds and gap:** best bound = minimum over open nodes; objective integrality detection gives the cutoff $z_{inc} - 1$.
- **Unbounded relaxations:** by Meyer's theorem (rational data) the MILP is unbounded if and only if it is integer feasible, decided by a zero-objective probe.
- **MIQP:** best-bound branch-and-bound over polished QP relaxations.

## 5. Presolve (`presolver.py`)

All reductions use the minimize-normalized objective. Postsolve unwinds a step stack in reverse order.

| Reduction | Rule | Postsolve |
|---|---|---|
| Fixed / empty column / dominated column | substitute value (including $Q$ terms) | value |
| Singleton row | bound on the variable (rounded for integers) | — |
| Activity analysis | infeasible, redundant, and forcing rows; implied integer bounds | — |
| Doubleton equation | $x_k = s + t\,x_j$ substituted everywhere; bounds transferred | $x_k = s + t\,x_j$ |
| Parallel rows | merge bound intervals | — |
| Duplicate columns | $y = x_j + \nu x_k$ with combined bounds | split $y$ within bounds |
| Coefficient tightening | $a_k \leftarrow a_k - d$, $b \leftarrow b - d$ for binaries | — |

Row duals of the original model are rebuilt after postsolve by crossover (LP) or signed NNLS (QP).

## 6. Independent certificate (`validator.py`)

Status levels: `OPTIMAL_CERTIFIED`, `FEASIBLE_CERTIFIED`, `FAILED`, `NO_SOLUTION_TO_VERIFY`.

- **Feasibility:** $\|(l - Ax)^+\|$, $\|(Ax - u)^+\|$, bounds, integrality (scaled tolerance $10^{-6}$), and objective recomputation.
- **LP/QP optimality:** from the reported $y$ only, recompute $z = \nabla f(x) - A^Ty$, then check sign conditions against the active set and the strong-duality gap between $f(x)$ and the Lagrangian dual objective.
- **MILP/MIQP optimality:** bound consistency and relative gap $|z_{inc} - z_{bound}| / \max(1, |z_{inc}|) \le 10^{-4}$.
