# Mathematical Foundations of the Sovereign Optimization Engine

This document provides the theoretical formulations, algorithmic mechanics, and numerical derivations underlying the Sovereign Mathematical Optimization Engine. All algorithms are implemented from first principles without wrapping third-party solver blackboxes.

---

## 1. Linear Programming (LP) Standard Form

The canonical formulation for linear programming supported by the engine is:

$$\begin{aligned}
\min_{x} \quad & c^T x + \text{offset} \\
\text{subject to} \quad & l_{\text{row}} \le A x \le u_{\text{row}} \\
& l_{\text{col}} \le x \le u_{\text{col}}
\end{aligned}$$

where:
- $x \in \mathbb{R}^n$ is the decision variable vector.
- $A \in \mathbb{R}^{m \times n}$ is the constraint matrix in sparse format.
- $c \in \mathbb{R}^n$ is the linear objective coefficient vector.
- $l_{\text{row}}, u_{\text{row}} \in (\mathbb{R} \cup \{\pm \infty\})^m$ represent two-sided constraint bounds.
- $l_{\text{col}}, u_{\text{col}} \in (\mathbb{R} \cup \{\pm \infty\})^n$ represent variable lower and upper bounds.

To apply tableau or factorized basis operations, the system transforms general inequalities into standard equality form:

$$A_{\text{std}} x_{\text{std}} = b_{\text{std}}, \quad x_{\text{std}} \ge 0$$

by shifting non-zero lower bounds $x_j' = x_j - l_j$ and introducing non-negative slack variables $s_i \ge 0$ ($+s_i$ for $\le$ constraints, $-s_i$ for $\ge$ constraints).

---

## 2. Two-Phase Revised Simplex Algorithm

The Revised Simplex method tracks an invertible basis matrix $B \in \mathbb{R}^{m \times m}$ formed by the columns of $A$ corresponding to basic variables $x_B$:

$$A = \begin{bmatrix} B & N \end{bmatrix}, \quad x = \begin{bmatrix} x_B \\ x_N \end{bmatrix}, \quad c = \begin{bmatrix} c_B \\ c_N \end{bmatrix}$$

The basic solution satisfies:
$$B x_B + N x_N = b \implies x_B = B^{-1} b - B^{-1} N x_N$$

Setting non-basic variables $x_N = 0$, the basic solution is $x_B = B^{-1} b$.

### 2.1 Two-Phase Feasibility Initiation
1. **Phase 1**: If the initial slack basis is infeasible (i.e. some $b_i < 0$ or equality/$\ge$ constraints lack an obvious identity column), artificial variables $a_i \ge 0$ are introduced:
   $$\min \sum_{i=1}^m a_i \quad \text{s.t.} \quad A x + I a = b, \quad x \ge 0, \quad a \ge 0$$
   An artificial basis $B = I$ is immediately feasible. Simplex iterations drive the artificial sum to zero. If the Phase 1 optimum exceeds numerical tolerance $\epsilon$, the model is certified mathematically **INFEASIBLE**.
2. **Phase 2**: The original objective $c^T x$ is restored, non-basic artificial variables are removed, and optimization proceeds toward the optimal extreme point.

### 2.2 Dual Multipliers and Reduced Costs
At each iteration, simplex computes simplex multipliers $y \in \mathbb{R}^m$ by solving the left linear system:
$$B^T y = c_B$$
The reduced costs $\bar{c}_N \in \mathbb{R}^{n - m}$ for non-basic variables are:
$$\bar{c}_j = c_j - y^T A_j \quad \forall j \in N$$

### 2.3 Pricing Policies
The entering variable $q \in N$ is selected using one of three strategies:
1. **Dantzig Rule**: $\min_{j \in N, \bar{c}_j < 0} \bar{c}_j$ (maximum local descent per unit step).
2. **Devex Pricing**: Approximates the steepest-edge norm in projected space:
   $$q = \arg\min_{j \in N, \bar{c}_j < 0} \frac{\bar{c}_j}{\sqrt{\gamma_j}}$$
   where column weights $\gamma_j$ are recursively updated after each pivot:
   $$\gamma_j \leftarrow \max(1.0, \|B^{-1} A_q\|_2^2)$$
3. **Bland's Anti-Cycling Rule**: Smallest index $j$ with $\bar{c}_j < 0$. Prevents infinite cycling in degenerate tableaus.

### 2.4 Harris Toleranced Ratio Test
Once entering column $A_q$ is chosen, the step direction $d = B^{-1} A_q$ is solved. The leaving variable $p \in B$ is determined by:
$$\theta = \min_{i: d_i > \epsilon_{\text{piv}}} \frac{x_{B, i} + \epsilon_{\text{Harris}}}{d_i}$$
This toleranced ratio test avoids numerical stalling when basic variables are at or near zero (degeneracy).

---

## 3. Primal-Dual Path-Following Interior Point Method (Mehrotra Predictor-Corrector)

For large-scale, dense, or quadratic instances, the engine utilizes a Primal-Dual Interior Point Method with Mehrotra's predictor-corrector acceleration.

### 3.1 KKT Optimality & Central Path
For the primal-dual pair:
$$\min c^T x \quad \text{s.t.} \quad Ax = b, \quad x \ge 0$$
$$\max b^T y \quad \text{s.t.} \quad A^T y + s = c, \quad s \ge 0$$

The Karush-Kuhn-Tucker (KKT) conditions parameterized by barrier parameter $\mu > 0$ define the central path:
$$\begin{aligned}
r_p &= b - A x = 0 && \text{(Primal Feasibility)} \\
r_d &= c - A^T y - s = 0 && \text{(Dual Feasibility)} \\
X S e &= \mu e && \text{(Complementarity Slackness)}
\end{aligned}$$
where $X = \text{diag}(x)$, $S = \text{diag}(s)$, and $e = [1, 1, \dots, 1]^T$.

The average complementarity gap is:
$$\mu = \frac{x^T s}{n}$$

### 3.2 Newton Direction via Normal Equations
Linearizing the perturbed KKT system yields:
$$\begin{bmatrix} 0 & A & 0 \\ A^T & 0 & I \\ 0 & S & X \end{bmatrix} \begin{bmatrix} \Delta y \\ \Delta x \\ \Delta s \end{bmatrix} = \begin{bmatrix} r_p \\ r_d \\ r_{xs} \end{bmatrix}$$

Eliminating $\Delta s = X^{-1} (r_{xs} - S \Delta x)$ and $\Delta x = \Theta (A^T \Delta y - r_d) + S^{-1} r_{xs}$ where $\Theta = X S^{-1}$, we arrive at the symmetric positive definite **Normal Equations**:
$$(A \Theta A^T) \Delta y = r_p + A \Theta r_d - A S^{-1} r_{xs}$$

### 3.3 Mehrotra Predictor-Corrector Two-Stage Solve
1. **Predictor Step ($\mu = 0$, Affine Direction)**:
   Set $r_{xs}^{\text{aff}} = -X S e = -x \cdot s$. Solve for $(\Delta x^{\text{aff}}, \Delta y^{\text{aff}}, \Delta s^{\text{aff}})$.
   Compute maximum affine step sizes to boundary:
   $$\alpha_p^{\text{aff}} = \min\left(1, \min_{\Delta x_j^{\text{aff}} < 0} \frac{-x_j}{\Delta x_j^{\text{aff}}}\right), \quad \alpha_d^{\text{aff}} = \min\left(1, \min_{\Delta s_j^{\text{aff}} < 0} \frac{-s_j}{\Delta s_j^{\text{aff}}}\right)$$
   Calculate projected gap:
   $$\mu_{\text{aff}} = \frac{(x + \alpha_p^{\text{aff}} \Delta x^{\text{aff}})^T (s + \alpha_d^{\text{aff}} \Delta s^{\text{aff}})}{n}$$
   Determine adaptive centering parameter $\sigma$:
   $$\sigma = \left(\frac{\mu_{\text{aff}}}{\mu}\right)^3 \in [0, 1]$$

2. **Corrector Step (Centering + Second-Order Cross Term)**:
   Add centering $\sigma \mu e$ and non-linear Hessian cross-term $-\Delta X^{\text{aff}} \Delta S^{\text{aff}} e$:
   $$r_{xs}^{\text{corr}} = -x \cdot s - \Delta x^{\text{aff}} \cdot \Delta s^{\text{aff}} + \sigma \mu e$$
   Re-solve normal equations with the combined RHS using the already-factorized $A \Theta A^T$ matrix (zero additional factorization cost!).

3. **Fraction-to-Boundary Step**:
   $$\alpha_p = \min\left(1, \eta \min_{\Delta x_j < 0} \frac{-x_j}{\Delta x_j}\right), \quad \alpha_d = \min\left(1, \eta \min_{\Delta s_j < 0} \frac{-s_j}{\Delta s_j}\right)$$
   with damping factor $\eta = 0.995$.

---

## 4. Branch-and-Bound Algorithm for Mixed-Integer Programming (MILP)

Mixed-Integer Linear Programs contain integrality restrictions:
$$\min c^T x \quad \text{s.t.} \quad A x \le b, \quad x_j \in \mathbb{Z} \quad \forall j \in I$$

### 4.1 Search Tree Architecture
The solver explores a binary search tree where each node represents an LP relaxation over bounded sub-domains:
- **Root Node**: Solves relaxation with continuous $x \in \mathbb{R}^n$.
- **Node Pruning Conditions**:
  1. *Infeasibility*: The LP relaxation is proven mathematically infeasible.
  2. *Bound Fathoming*: The relaxation lower bound $\ge z_{\text{incumbent}} - \epsilon$.
  3. *Integrality*: All $x_j$ for $j \in I$ satisfy $|x_j - \text{round}(x_j)| \le \epsilon_{\text{int}}$. If objective beats incumbent, update incumbent solution.

### 4.2 Branching Strategies
When fractional variable values exist, a variable $x_k$ is chosen and two child nodes are created:
$$\text{Child Left: } x_k \le \lfloor x_k^* \rfloor, \qquad \text{Child Right: } x_k \ge \lceil x_k^* \rceil$$

1. **Most-Fractional**: Chooses $k = \arg\min_j |x_j^* - \lfloor x_j^* \rfloor - 0.5|$.
2. **ML-Guided Branching**: Evaluates an $O(1)$ learned scoring function balancing fractionality distance, objective coefficient magnitude $|c_k|$, constraint connectivity degree, and tree depth.

### 4.3 Primal Rounding Heuristic
At the root relaxation node, the engine applies a fast rounding heuristic:
$$x_j^{\text{cand}} = \text{round}(x_j^*)$$
If $x^{\text{cand}}$ satisfies all original bounds and constraints $A x^{\text{cand}} \le b$, an initial feasible integer incumbent is locked in immediately, cutting off broad subtrees early.

---

## 5. Active Set Convex Quadratic Programming (QP)

For problems with quadratic objectives:
$$\min \frac{1}{2} x^T Q x + c^T x \quad \text{s.t.} \quad A x \le b$$
where $Q \in \mathbb{R}^{n \times n}$ is positive semi-definite ($x^T Q x \ge 0$).

### 5.1 Working Set Subproblem
At iterate $x_k$, a subset of constraints are active: $\mathcal{W}_k = \{i : a_i^T x_k = b_i\}$.
The solver computes step direction $p$ by solving the equality-constrained KKT system:
$$\begin{bmatrix} Q & A_{\mathcal{W}}^T \\ A_{\mathcal{W}} & 0 \end{bmatrix} \begin{bmatrix} p \\ \lambda \end{bmatrix} = \begin{bmatrix} -(Q x_k + c) \\ 0 \end{bmatrix}$$

### 5.2 Step and Working Set Updates
1. If $p \ne 0$: A line search determines $\alpha_k = \min(1, \min_{i \notin \mathcal{W}, a_i^T p > 0} \frac{b_i - a_i^T x_k}{a_i^T p})$. If $\alpha_k < 1$, the blocking constraint is added to $\mathcal{W}$.
2. If $p = 0$: Check multipliers $\lambda$. If all $\lambda_i \ge 0$, KKT optimality is proven. If any $\lambda_i < 0$, the constraint with the most negative multiplier is dropped from $\mathcal{W}$.
