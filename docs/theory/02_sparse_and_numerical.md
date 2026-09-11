# Sparse Computing & Numerical Scaling Architecture

Industrial mathematical optimization problems often comprise tens of thousands to millions of variables, yet typical constraint matrices exhibit densities below $0.1\%$. Dense representations would exhaust memory and cause $O(m n)$ or $O(n^3)$ operations on zeros. This document outlines the sparse linear algebra engine and numerical conditioning algorithms.

---

## 1. Dual Compressed Sparse Storage (CSR & CSC)

The engine avoids dense matrix storage through a unified sparse wrapper combining:
1. **Compressed Sparse Row (CSR)**
   - Arrays: `data`, `indices`, `indptr`
   - Memory complexity: $O(2 \cdot \text{nnz} + m + 1)$
   - Fast row slicing: Retrieving constraint $i$ requires examining only slice `indptr[i]:indptr[i+1]`.
   - Matrix-Vector Product (SpMV): Computes $y = A x$ in $O(\text{nnz})$ floating-point operations.

2. **Compressed Sparse Column (CSC)**
   - Arrays: `data`, `indices`, `indptr`
   - Memory complexity: $O(2 \cdot \text{nnz} + n + 1)$
   - Column access: Essential for Revised Simplex pricing, pivot calculations, and column generation.
   - Transpose SpMV: Computes $x = A^T y$ in $O(\text{nnz})$ operations.

---

## 2. Matrix Conditioning & Ruiz Equilibration Scaling

### 2.1 The Conditioning Challenge
In industrial models (e.g. refineries, power systems), variables and coefficients span widely differing physical units:
- Electrical power: Megawatts ($10^2$ to $10^3$)
- Chemical concentrations: Parts-per-million or fractions ($10^{-6}$ to $10^{-3}$)
- Cash flow / revenues: Hundreds of thousands to millions ($10^5$ to $10^7$)

Such dynamic ranges produce ill-conditioned systems with condition numbers $\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} > 10^8$. In floating-point arithmetic, this amplifies roundoff errors and causes pivot degeneration.

### 2.2 Ruiz Equilibration (Ruiz 2001)
The engine applies iterative diagonal scaling:
$$A_{\text{scaled}} = D_1 A D_2$$
where $D_1 = \text{diag}(d_1) \in \mathbb{R}^{m \times m}$ and $D_2 = \text{diag}(d_2) \in \mathbb{R}^{n \times n}$ are positive diagonal matrices chosen such that:
$$\|A_{\text{scaled}}(i, :)\|_\infty \approx 1 \quad \forall i \in \{1, \dots, m\}$$
$$\|A_{\text{scaled}}(:, j)\|_\infty \approx 1 \quad \forall j \in \{1, \dots, n\}$$

### 2.3 Algorithm Iterations
Initialize $D_1^{(0)} = I$, $D_2^{(0)} = I$, $A^{(0)} = A$.
For iteration $k = 0, 1, \dots, K$:
1. Compute row infinity norms: $r_i = \sqrt{\|A^{(k)}(i, :)\|_\infty}$.
2. Compute column infinity norms: $c_j = \sqrt{\|A^{(k)}(:, j)\|_\infty}$.
3. Form diagonal scaling matrices: $R = \text{diag}(r)^{-1}$, $C = \text{diag}(c)^{-1}$.
4. Update scaled matrix: $A^{(k+1)} = R A^{(k)} C$.
5. Accumulate scale factors: $D_1 \leftarrow D_1 R$, $D_2 \leftarrow D_2 C$.
6. Stop when $\max_i |r_i^2 - 1| < \epsilon_{\text{tol}}$ and $\max_j |c_j^2 - 1| < \epsilon_{\text{tol}}$.

### 2.4 Vector Transformation & Unscaling
To maintain exact mathematical equivalence:
- **Constraints**: $l_{\text{row}} \le Ax \le u_{\text{row}} \iff D_1 l_{\text{row}} \le (D_1 A D_2) (D_2^{-1} x) \le D_1 u_{\text{row}}$
- **Bounds**: $l_{\text{col}} \le x \le u_{\text{col}} \iff D_2^{-1} l_{\text{col}} \le D_2^{-1} x \le D_2^{-1} u_{\text{col}}$
- **Objective**: $\min c^T x = \min (D_2 c)^T (D_2^{-1} x)$
- **Primal Recovery**: $x = D_2 x_{\text{scaled}}$
- **Dual Recovery**: $y = D_1 y_{\text{scaled}}$
