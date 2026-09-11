# Presolve Reductions & Postsolve Reconstruction

Presolve transforms the optimization model into a smaller, tighter, and numerically superior formulation before solver invocation. Postsolve reverses all reductions to deliver the solution in the user's original variable and constraint coordinates.

---

## 1. Presolve Reduction Pipeline

Presolve executes multiple passes over the sparse model graph until no further reductions occur.

### 1.1 Fixed Variable Elimination
When a variable's bounds satisfy $|l_j - u_j| \le \epsilon$:
$$x_j^* = \frac{l_j + u_j}{2}$$
1. For every active constraint $i$ containing variable $j$:
   $$b_i \leftarrow b_i - a_{ij} x_j^*$$
   Row bound intervals are adjusted: $[l_i - a_{ij} x_j^*, u_i - a_{ij} x_j^*]$.
2. The objective constant offset accumulates:
   $$\text{offset} \leftarrow \text{offset} + c_j x_j^*$$
3. Variable $j$ and column $A_{:, j}$ are eliminated from the active problem.

### 1.2 Empty Row Detection
If row $i$ has zero non-zero coefficients ($a_{ij} = 0 \quad \forall j$):
- If $0 \in [l_i, u_i]$, the constraint is trivially satisfied and safely deleted.
- If $0 \notin [l_i, u_i]$, the problem is proven mathematically **INFEASIBLE**. Presolve immediately halts and emits a certificate of infeasibility.

### 1.3 Singleton Rows & Bound Tightening
A singleton row contains exactly one non-zero coefficient $a_{ik} \ne 0$:
$$l_i \le a_{ik} x_k \le u_i$$
- If $a_{ik} > 0$: $\frac{l_i}{a_{ik}} \le x_k \le \frac{u_i}{a_{ik}}$
- If $a_{ik} < 0$: $\frac{u_i}{a_{ik}} \le x_k \le \frac{l_i}{a_{ik}}$

The implied bounds are intersected with existing variable bounds:
$$l_k^{\text{new}} = \max\left(l_k, l_k^{\text{implied}}\right), \quad u_k^{\text{new}} = \min\left(u_k, u_k^{\text{implied}}\right)$$

If $l_k^{\text{new}} > u_k^{\text{new}} + \epsilon$, the model is certified **INFEASIBLE**.
Otherwise, the tightened bounds are stored, and row $i$ is pruned.

### 1.4 Empty Column Optimization
If variable $j$ has no non-zero entries across all active constraints:
- To minimize $c_j x_j$:
  - If $c_j > 0$: $x_j^* = l_j$. If $l_j = -\infty$, the model is **UNBOUNDED**.
  - If $c_j < 0$: $x_j^* = u_j$. If $u_j = +\infty$, the model is **UNBOUNDED**.
  - If $c_j = 0$: $x_j^* = l_j$ (or $0.0$ if $0 \in [l_j, u_j]$).
- The variable is fixed to $x_j^*$, objective offset is updated, and column $j$ is removed.

---

## 2. Postsolve Solution Reconstruction

Presolve actions are recorded sequentially on a transformation stack:
$$\mathcal{S} = \left[ \text{Step}_1, \text{Step}_2, \dots, \text{Step}_K \right]$$

When the reduced model is solved producing solution vector $x_{\text{presolved}}^* \in \mathbb{R}^{n'}$, the `PostsolveMapper` reconstructs the original vector $x^* \in \mathbb{R}^n$:
1. All variables present in $x_{\text{presolved}}^*$ are mapped to their original column indices.
2. Fixed variables are populated with their recorded substitution values $x_j^*$.
3. The reduction stack $\mathcal{S}$ is unrolled in reverse order ($K \to 1$) to calculate dependent variable values for aggregated or substituted rows.
4. The full reconstructed vector $x^*$ is passed directly to the Independent Trust Validator.
