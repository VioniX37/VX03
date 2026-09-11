# Independent Mathematical Trust Verification

The final stage of the Sovereign Optimization pipeline is the **Independent Trust Validator**. It operates independently of all solver internals, providing a rigorous mathematical certificate that the delivered solution satisfies all model specifications.

---

## 1. Mathematical Verification Criteria

Given the original model $(A, b, c, Q, l_{\text{col}}, u_{\text{col}}, l_{\text{row}}, u_{\text{row}})$ and proposed solution vector $x^* \in \mathbb{R}^n$:

### 1.1 Variable Bound Satisfaction
For every variable $j \in \{1, \dots, n\}$:
$$\text{viol}_{\text{lb}}(j) = \max\left(0, l_{\text{col}, j} - x_j^*\right)$$
$$\text{viol}_{\text{ub}}(j) = \max\left(0, x_j^* - u_{\text{col}, j}\right)$$
Toleranced test:
$$\frac{\text{viol}(j)}{\max(1, |x_j^*|)} \le \epsilon_{\text{feas}}$$

### 1.2 Constraint Feasibility
For every constraint $i \in \{1, \dots, m\}$, recompute the inner product directly from the original row:
$$\text{row\_val}_i = \sum_{j=1}^n a_{ij} x_j^*$$
Verify lower and upper satisfaction:
$$\text{viol}_{\text{row\_lb}}(i) = \max\left(0, l_{\text{row}, i} - \text{row\_val}_i\right)$$
$$\text{viol}_{\text{row\_ub}}(i) = \max\left(0, \text{row\_val}_i - u_{\text{row}, i}\right)$$
Toleranced test:
$$\frac{\text{viol}_{\text{row}}(i)}{\max(1, |l_{\text{row}, i}|, |u_{\text{row}, i}|)} \le \epsilon_{\text{feas}}$$

### 1.3 Integrality Certification
For every integer or binary variable $j \in I_{\text{int}}$:
$$\text{viol}_{\text{int}}(j) = |x_j^* - \text{round}(x_j^*)|$$
Toleranced test:
$$\text{viol}_{\text{int}}(j) \le \epsilon_{\text{int}}$$

### 1.4 Independent Objective Recomputation
The objective value is recomputed from scratch without referencing solver logs:
$$z_{\text{recomputed}} = c^T x^* + \frac{1}{2} (x^*)^T Q x^* + \text{offset}$$
Relative consistency check against reported objective $z_{\text{reported}}$:
$$\frac{|z_{\text{recomputed}} - z_{\text{reported}}|}{\max(1, |z_{\text{recomputed}}|)} \le \epsilon_{\text{obj}}$$

---

## 2. Validation Certificate Specification

When all checks pass, the engine issues a structured `ValidationCertificate`:
```json
{
  "is_valid": true,
  "status": "PASSED",
  "max_primal_violation": 0.0,
  "max_bound_violation": 0.0,
  "max_integrality_violation": 0.0,
  "reported_objective": -6094333.3306,
  "recomputed_objective": -6094333.3306,
  "objective_difference": 0.0,
  "checks": {
    "bounds_satisfied": true,
    "constraints_satisfied": true,
    "integrality_satisfied": true,
    "objective_matches": true
  }
}
```
If any check fails, the status becomes `FAILED`, and the exact violated constraints, bounds, or fractional variables are enumerated in `violation_details`.
