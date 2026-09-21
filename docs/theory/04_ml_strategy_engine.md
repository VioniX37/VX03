# AI / ML Strategy Engine Architecture

The Sovereign Optimizer couples rigorous mathematical algorithms with an adaptive machine learning strategy engine. The fundamental design tenet is:

> **"ML decides HOW the problem should be attacked. The mathematical solver determines and verifies WHAT the solution is. ML must never become the authority for mathematical correctness."**

---

## 1. Feature Representation (23 Structural Dimensions)

The ML Strategy Engine operates entirely on mathematical and graph-structural features extracted from both the raw input model and the post-presolve model. It does not inspect variable names, semantics, or problem text.

| Feature Category | Description | Mathematical Formulation |
| :--- | :--- | :--- |
| **Scale & Density** | Size and sparsity profile | $\log_{10}(n), \log_{10}(m), \log_{10}(\text{nnz}), \text{Density} = \frac{\text{nnz}}{m \cdot n}$ |
| **Aspect Ratio** | Shape of constraint matrix | $\text{Aspect} = \frac{m}{n}$ (distinguishes rectangular vs square systems) |
| **Dynamic Range** | Matrix coefficient variation | $\log_{10}\left(\frac{\max |a_{ij}|}{\min |a_{ij}|}\right)$ |
| **Variable Topology** | Variable type distribution | $\frac{n_{\text{cont}}}{n}, \frac{n_{\text{int}}}{n}, \frac{n_{\text{bin}}}{n}$ |
| **Bound Enclosure** | Polyhedral tightness | $\frac{n_{\text{bounded\_below}}}{n}, \frac{n_{\text{bounded\_above}}}{n}, \frac{n_{\text{boxed}}}{n}$ |
| **Constraint Types** | Equality vs inequality balance | $\frac{m_{\le}}{m}, \frac{m_{\ge}}{m}, \frac{m_{=}}{m}$ |
| **Presolve Reductions** | Compressibility metrics | $\Delta_{\text{vars}}\%, \Delta_{\text{cons}}\%, \Delta_{\text{nnz}}\%, \frac{n_{\text{fixed}}}{n}, \frac{m_{\text{singleton}}}{m}$ |
| **Condition Estimate** | Numerical sensitivity indicator | $\kappa_{\text{est}} \approx \min(\text{Dynamic Range}, 10^8)$ |

---

## 2. ML Strategy Models

### 2.1 Model 1: Continuous LP Algorithm Selection (Simplex vs IPM)
- **Simplex Strengths**: Highly sparse matrices ($\text{density} < 1\%$), rectangular shapes ($n \gg m$ or $m \gg n$), combinatorial bases, warm-starting capabilities.
- **Interior Point Strengths**: Moderate to high density ($\text{density} > 5\%$), square aspects ($m \approx n$), large numbers of active bounds, smooth central paths.
- **Predictive Scoring**: Computes calibrated probabilities $P(\text{Simplex})$ and $P(\text{IPM})$. Selects the algorithm with highest likelihood of lowest CPU wall-clock time.

### 2.2 Model 2: MILP Search & Branching Selection
- Evaluates candidate fractional variables in Branch-and-Bound.
- Serves as an ultra-fast $O(1)$ surrogate for strong branching (which would otherwise require solving two LP relaxations per fractional variable).
- Computes heuristic impact score:
  $$\text{Score}(x_j) = 2 \min(\text{frac}_j, 1 - \text{frac}_j) \cdot \ln(1 + |c_j|) \cdot \frac{1 + 0.1 \cdot \text{deg}(j)}{1 + 0.05 \cdot \text{depth}}$$
  ranking variables by expected dual bound improvement.

### 2.3 Model 3: Hardware Recommendation (CPU vs GPU)
- Assesses whether SpMV acceleration on GPU offsets host-to-device memory transfer latency:
  $$T_{\text{transfer}} + T_{\text{GPU}} < T_{\text{CPU}}$$
- Recommends GPU when $\text{nnz} \ge 25,000$ and density exceeds 2%. This recommendation is advisory and
  reported to the user; it does not route the solve. The GPU path that runs today is PDLP, which uses
  PyTorch CUDA automatically whenever a CUDA device is present and otherwise runs on fused multi-threaded
  CPU kernels. The simplex, interior point and branch-and-bound solvers run on CPU.

---

## 3. Deterministic Safety Harness

To guarantee absolute mathematical reliability, the engine wraps ML recommendations in a strict safety harness:
```
                Problem Features
                       │
                       ▼
               ML Strategy Engine
                       │
                       ▼
          [Confidence >= Threshold?]
                 ├── YES ──► Deploy Recommended Algorithm
                 │
                 └── NO  ──► Deterministic Fallback (Simplex / Most-Fractional / CPU)
                                 │
                                 ▼
                     Mathematical Solver Core
                                 │
                                 ▼
                    Independent Trust Validator
                                 │
                                 ▼
                     Mathematically Verified
```
If an ML recommendation encounters numerical stalling or time limits, the system triggers the deterministic fallback seamlessly.
