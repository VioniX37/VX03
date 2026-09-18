# Sovereign Mathematical Optimization Engine

An **AI-guided, sparse-first, heterogeneous CPU-GPU mathematical optimization engine** engineered from foundational mathematics. The engine delivers production-grade linear, mixed-integer, and quadratic optimization capabilities designed to run and demo transparently on standard developer laptops.

---

## 1. Core Architectural Tenet

> **"ML decides HOW the problem should be attacked. The mathematical solver determines and verifies WHAT the solution is. ML must never become the authority for mathematical correctness."**

### Sovereignty Guarantee
The solver is built **from mathematical first principles**. It does **not** wrap, call, or depend on existing commercial or open-source solver blackboxes (such as Gurobi, CPLEX, SCIP, HiGHS, or GLPK). Every algorithm—from basis LU pivots and barrier normal equations to branch-and-bound node queues and Ruiz equilibration—is natively implemented and verified.

---

## 1.1 What's New in v2.0

v2 is an accuracy and capability upgrade of the solver core. Every v1 defect below was reproduced, fixed, and locked in by a regression test.

| Area | v1 behaviour (measured) | v2 behaviour (measured) |
|---|---|---|
| Free variables (`min x, x ≥ -50000`) | Simplex −10000, IPM 0, both labelled optimal | −50000 from every LP/QP solver, KKT-certified |
| Presolve on maximization | Empty columns fixed at the wrong bound | Sense-normalized reductions; 150/150 randomized round trips correct |
| Netlib AFIRO | Unbounded transcription; IPM diverged to −6e53 | Original MPS data; −464.7531428571 (error ≤ 1e-13) on all LP solvers |
| MIQP | Sent to simplex (quadratic terms and integrality dropped) | Branch-and-bound over convex QP relaxations |
| Refinery QP | Active set stopped at iteration cap (−5,889,292) | −5,903,161.066 certified optimal (IPM: 11 iters, active set: 26) |
| Unit commitment | Gap reported wrong; crash beyond 4 periods | 4h / 8h / 24h proven optimal, gap 0 (24h: 0.18 s, 1 node) |
| Solver statuses | Iteration limits relabelled "feasible" | Honest statuses; infeasible / unbounded certified |
| Trust certificate | Feasibility only | Feasibility + optimality (KKT duals, strong duality, MIP bound) |

Randomized verification during development (SciPy's HiGHS used only as a test oracle, never by the engine): 300/300 random LPs (simplex), 80/80 random MILPs (branch-and-cut), 120/120 random convex QPs (certified or independently confirmed infeasible/unbounded).

**New algorithms:** bounded primal & dual simplex with LU/eta updates, Devex and dual steepest-edge pricing, Harris ratio tests; homogeneous self-dual IPM with crossover; convex QP interior point with active-set polishing; branch-and-cut with Gomory mixed-integer and knapsack cover cuts, reliability branching, diving / feasibility pump / RINS heuristics; presolve with activity analysis, forcing and parallel rows, doubleton aggregation, dominated and duplicate columns, coefficient tightening. Details: [`07_v2_engine_upgrades.md`](docs/theory/07_v2_engine_upgrades.md).

---

## 2. End-to-End System Pipeline

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 INPUT OPTIMIZATION MODEL                    │
  │     Universal representation (LP / MILP / QP / MIQP)        │
  │     Parsers: Standard MPS (Fixed & Free) and CPLEX LP       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 VALIDATION & CLASSIFICATION                 │
  │     Sanity checks, bound conflicts, problem class detection │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                     SPARSE MATRIX ENGINE                    │
  │      Dual CSR/CSC representations, SpMV, SpMV-transpose     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                      PRESOLVE & SCALING                     │
  │    Fixed vars, empty rows/cols, singletons, Ruiz scaling    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               POST-PRESOLVE FEATURE EXTRACTION              │
  │         28-dimensional structural problem descriptor        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                      ML STRATEGY ENGINE                     │
  │    Algorithm selection (Simplex vs IPM), branching score,   │
  │    hardware selection, backed by deterministic fallbacks    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    SOVEREIGN SOLVER CORE                    │
  │  • LP:   Bounded Primal/Dual Simplex & Homogeneous IPM      │
  │  • MILP: Branch-and-Cut (GMI + cover cuts, reliability)     │
  │  • QP:   Interior Point + Active Set; MIQP via B&B          │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    HETEROGENEOUS RUNTIME                    │
  │     CPU Multithreading ↔ PyTorch/GPU Tensor Acceleration    │
  │           Automatic transparent fallback to CPU             │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                          POSTSOLVE                          │
  │       Reconstructs solution in original variable space      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 INDEPENDENT TRUST VALIDATOR                 │
  │    Strict verification of Ax <= b, bounds, integrality,     │
  │    and independent objective calculation                    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │            NEXT.JS DASHBOARD & RICH TERMINAL CLI            │
  │    Interactive web UI, matrix heatmap, trust certificates   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Concepts & Mathematical Implementations

All mathematical formulations and proofs are documented in depth under [`docs/theory/`](file:///c:/Users/HP/Desktop/VX/VX03/docs/theory):

### 3.1 Linear Programming (LP)
- **Two-Phase Revised Simplex** ([`simplex.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/solvers/lp/simplex.py)):
  - Phase 1 introduces artificial identity variables to find an initial basic feasible solution (BFS) or certify infeasibility.
  - Phase 2 optimizes the true objective using basis LU factorization ($B x_B = b$, $B^T y = c_B$).
  - Pricing rules: **Devex** (steepest-edge approximation), **Dantzig** (most negative reduced cost), and **Bland's anti-cycling rule**.
  - **Harris ratio test**: Avoids numerical stalling in degenerate tableaus.
- **Primal-Dual Interior Point Method** ([`interior_point.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/solvers/lp/interior_point.py)):
  - **Mehrotra Predictor-Corrector**: Solves normal equations $(A \Theta A^T) \Delta y = r$ with diagonal weight $\Theta = X S^{-1}$.
  - Predictor step calculates affine scaling direction ($\sigma = 0$).
  - Corrector step computes adaptive centering parameter $\sigma = (\mu_{\text{aff}} / \mu)^3$ and non-linear cross terms.
  - Fraction-to-the-boundary step control with $\eta = 0.995$.
  - Rapid polynomial-time convergence (typically 10–25 iterations).

### 3.2 Mixed-Integer Linear Programming (MILP)
- **Branch-and-Bound Tree Engine** ([`branch_bound.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/solvers/milp/branch_bound.py)):
  - Best-Bound priority queue min-heap search.
  - LP relaxation solves at each tree node.
  - Pruning: Infeasibility fathoming, bound fathoming ($LB \ge UB - \epsilon$), and integrality fathoming.
  - Primal rounding heuristic at root node to establish early incumbents.
  - MIP Gap calculation: $\text{Gap} = \frac{|\text{Incumbent} - \text{Best Bound}|}{|\text{Incumbent}| + 10^{-10}}$.
- **Branching Selection Strategies** ([`branching.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/solvers/milp/branching.py)):
  - Most-Fractional branching: selects variable with fractionality closest to 0.5.
  - **ML-Guided Branching**: $O(1)$ learned scoring function approximating strong branching without dual-solve overhead.

### 3.3 Convex Quadratic Programming (QP)
- **Active Set QP** ([`active_set.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/solvers/qp/active_set.py)):
  - Solves $\min \frac{1}{2} x^T Q x + c^T x$ subject to $A x \le b$.
  - Checks positive semi-definiteness ($x^T Q x \ge 0$).
  - Assembles and factorizes Karush-Kuhn-Tucker (KKT) augmented systems ([`kkt.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/solvers/qp/kkt.py)).
  - Phase-1 LP initialization to find feasible starting point.

### 3.4 Sparse Linear Algebra & Scaling
- **Dual CSR/CSC Representation** ([`matrix.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/sparse/matrix.py)):
  - Compressed Sparse Row for constraint evaluation and SpMV ($A x$).
  - Compressed Sparse Column for column pricing and transpose SpMV ($A^T y$).
- **Ruiz Equilibration Scaling** ([`scaling.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/sparse/scaling.py)):
  - Computes diagonal scaling matrices $D_1 A D_2$ such that row and column infinity norms converge to $1.0$.
  - Tames coefficient dynamic ranges spanning $10^{-6}$ to $10^6$.

### 3.5 Presolve Reductions & Postsolve Mapper
- **Presolve Pipeline** ([`presolver.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/presolve/presolver.py)):
  - Fixed variable substitution into constraint bounds and objective offset.
  - Empty row removal and infeasibility detection ($0 \notin [l_i, u_i]$).
  - Singleton row detection and bound tightening ($a_{ij} x_j \in [l_i, u_i]$).
  - Empty column pruning.
- **Postsolve Reconstruction** ([`postsolve.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/presolve/postsolve.py)):
  - Inverts the reduction stack to restore original variable coordinates $x^* \in \mathbb{R}^n$.

### 3.6 AI / ML Strategy Engine
- **28-Dimensional Structural Feature Extractor** ([`features.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/ml/features.py)):
  - Size, sparsity density, aspect ratio ($m/n$), coefficient dynamic range, variable type distribution, bound enclosures, and presolve reduction ratios.
- **Adaptive Recommendations** ([`strategy.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/ml/strategy.py)):
  - Continuous LP: Simplex vs. Interior Point recommendation.
  - Hardware: CPU vs. GPU acceleration dispatch.
  - Deterministic safety fallback: Reverts to proven heuristics if ML confidence $< \tau$.

### 3.7 Independent Mathematical Trust Validator
- **Independent Verification** ([`validator.py`](file:///c:/Users/HP/Desktop/VX/VX03/sovereign_opt/validation/validator.py)):
  - Recomputes $A x \le b$ and $l \le x \le u$ directly from raw problem equations.
  - Checks integrality margin $|x_j - \text{round}(x_j)| \le \epsilon_{\text{int}}$.
  - Recomputes exact objective $c^T x + \frac{1}{2} x^T Q x + \text{offset}$.
  - Issues cryptographic/structured `ValidationCertificate(status="PASSED")`.

---

## 4. Built-in Industrial Case Studies

1. **Refinery Crude & Blendstock Blending (LP / QP)** ([`refinery_blending.py`](file:///c:/Users/HP/Desktop/VX/VX03/benchmarks/industrial/refinery_blending.py)):
   - Blends Brent, WTI, Dubai, and Maya crudes with high-octane Alkylate streams.
   - Satisfies minimum octane (Regular: 87, Premium: 93) and maximum sulfur constraints to maximize refinery profit.
2. **Power Grid Unit Commitment (MILP)** ([`power_dispatch.py`](file:///c:/Users/HP/Desktop/VX/VX03/benchmarks/industrial/power_dispatch.py)):
   - Multi-period thermal generator scheduling with binary commitment states ($u_{g, t} \in \{0, 1\}$), startup costs, capacity envelopes, and hourly demand balance.
3. **Netlib LP Benchmark (AFIRO)** ([`afiro.py`](file:///c:/Users/HP/Desktop/VX/VX03/benchmarks/netlib/afiro.py)):
   - Standard benchmark problem (27 vars, 32 cons, known optimal: $-464.75314$).

---

## 4.5 Demo Race: head-to-head vs Gurobi and HiGHS

The dashboard's **demo** tab races the engine against **Gurobi** and **HiGHS** on four curated
instances and shows, side by side, that every solver reaches the same optimum and how long each
one took. Measured on a 12-core laptop (Windows 11, no GPU). Every engine gets the same
environment: its own fresh process, the same 12 threads, the same input arrays and the same clock.
Objectives are compared against Gurobi as the reference; our LP solver runs to a 1e-7 tolerance.

| | instance | size | ours | Gurobi 13 | HiGHS | ours vs Gurobi objective |
|---|---|---|---:|---:|---:|---:|
| **A** | production-distribution LP | 19,586 x 99,728, 296k nnz | 5.11 s | **1.18 s** | 4.37 s | 1.7e-7 |
| **B** | same model at 1M variables | 193,442 x 999,188, 2.96M nnz | 125.0 s | **42.7 s** | 766.2 s | 1.6e-7 |
| **C** | `STCQP1`, Maros-Meszaros QP | 2,052 x 4,097 | 1.07 s | **0.02 s** | 134.8 s | 2e-10 |
| **D** | unit commitment, 480 binaries | 1,228 x 720 | 1.02 s | **0.06 s** | 0.08 s | exact |

Read honestly: with every engine on 12 threads, Gurobi is fastest on all four. At a million
variables the sovereign LP solver (restarted reflected Halpern PDHG on fused multi-threaded
kernels) is 6x faster than HiGHS and agrees with Gurobi's optimum to seven significant digits;
Gurobi's parallel barrier is still about 3x faster on this CPU. On the small QP and the MILP the
objectives are identical and ours carries an independent optimality certificate, but the QP
interior point and the branch-and-cut tree are not yet tuned to commercial standards. That gap
is the roadmap, and the demo states it rather than hiding it.

On a Tesla T4 (recorded on Kaggle, not live) instance B solved in **9.6 s** with the earlier PDLP
at tolerance 1e-4 through the Torch CUDA backend -- faster than Gurobi's 42.7 s on this laptop.
That run predates the Halpern solver and has not been repeated at 1e-7.

### Running it

```bash
pip install -r requirements-bench.txt      # highspy + gurobipy, comparison only
python run_demo.py                         # API on :8000, dashboard on :3000 -> "demo" tab
```

Gurobi needs a license. Drop a `gurobi.lic` into `Gurobi files/` (it is gitignored -- a WLS
file contains a private secret) or set `GRB_LICENSE_FILE`. Without one, gurobipy still works
but caps models at 2,000 variables and 2,000 constraints; the engine probes the license at
startup and greys out the bars it cannot run, with the reason shown.

From the command line, one instance and one solver per process:

```bash
python -m benchmarks.demo_race --list
python -m benchmarks.demo_race --probe
python -m benchmarks.demo_race --instance milp_uc_720 --all
python -m benchmarks.demo_race --instance supply_chain_1m --solver ours --save
```

`--save` writes `benchmarks/results/demo_<instance>_<stamp>.json`, which the dashboard then
serves as a cached result. The demo page also carries every number above as a built-in
fallback, so a failed live run still renders; the **use recorded results** toggle pins it.

Neither HiGHS nor Gurobi is ever imported by `sovereign_opt`: the comparison runners live in
`benchmarks/demo_race.py` and the server reaches them through a subprocess. The guard test
`tests/test_engine_never_imports_comparison_solvers` enforces that.

---

## 4. Workstation Dashboard & 6-Stage Workflow Visualization

The Next.js dashboard is styled as a **terminal session**, because the product itself is a CLI:
- **Status bars:** a status bar at the top and a vim-style mode line at the bottom.
- **Prompt:** shows the exact `python -m sovereign_opt.cli.app ...` command for the current selection, with a copy button.
- **Panels:** boxes with the title set into the border, the same look as the CLI's rich output.
- **Run log:** prints the six pipeline stages line by line.
- **Typography and color:** one monospace font (IBM Plex Mono), a warm dark ground, amber for prompts and selection, and green/red only for pass/fail.
- **Text-mode visuals:** small matrices print as a `+ − ·` character grid, the branch-and-bound search prints like the `tree` command, and meters are ASCII bars.
- **Keyboard:** `1`–`6` switch views, `ctrl+enter` runs, `j`/`k` move through problems.
- **Motion:** respects `prefers-reduced-motion`.

### Complete 6-Stage End-to-End Workflow Visualizer

Every phase of the backend optimization run is transparently inspectable across 6 sequential stages:

```
[01 TOPOLOGY & RUIZ] ──► [02 PRESOLVE DIFF] ──► [03 AI META-STRATEGY]
                                                        │
[06 TRUST AUDIT]     ◄── [05 POSTSOLVE MAP] ◄── [04 SOLVER DYNAMICS]
```

1. **Stage 01: Constraint Matrix Topology & Ruiz Equilibration**
   - **Interactive 2D Matrix Spy Plot**: Visual coordinate scatter of non-zero entries $a_{ij}$ of constraint matrix $A \in \mathbb{R}^{m \times n}$. Differentiates positive and negative coefficients with interactive crosshairs displaying exact $(row, col, value)$.
   - **Ruiz Scaling Telemetry**: Shows pre-scaling norm $\|A\|_\infty$ vs post-scaling norm $\|D_1 A D_2\|_\infty \approx 1.0$ alongside diagonal scaling multipliers $[D_1, D_2]$ to demonstrate numerical conditioning improvement.

2. **Stage 02: Presolve Reduction Pipeline**
   - **Side-by-Side Dimension Diff**: Quantifies reductions in variables ($n \to n'$), constraints ($m \to m'$), and non-zeros ($nnz \to nnz'$).
   - **Fixed Variable Audit Log**: Displays variables eliminated prior to solver dispatch ($[x_j = l_j = u_j]$) and singleton rows used for bound tightening.

3. **Stage 03: AI Meta-Strategy & Safety Harness**
   - **Algorithm Probability Distribution**: Visualizes confidence score across candidate solvers (Revised Simplex, IPM, Branch & Bound, Active Set QP).
   - **28-Dimensional Feature Attribution**: Table of structural descriptors (density, aspect ratio, degree of sparsity, integer fraction, coefficient dynamic range).
   - **Deterministic Safety Harness**: Verifies that ML recommendations are advisory-only with active fallback guarantees.

4. **Stage 04: Live Solver Iteration & Search Dynamics**
   - **Simplex / IPM / Active Set Convergence**: Real-time SVG polyline tracking objective trajectory across iterations alongside basis pivots, barrier parameter $\mu$, and step residuals.
   - **MILP Branch-and-Bound Tree Explorer**: Interactive hierarchical graph of search nodes. Displays node states (Root, Active Branch, Bound Pruned, Infeasible, Integer Incumbent). Clicking any node reveals its relaxation lower bound $z_{LP}$, branch variable condition, and depth.

5. **Stage 05: Canonical Postsolve Recovery**
   - **Variable Unwinding Matrix**: Searchable audit table mapping internal reduced variables back to user-space canonical variables.
   - Categorizes resolution origins: `OPTIMIZED_IN_CORE`, `FIXED_IN_PRESOLVE`, or `CANONICAL_RESTORED`.

6. **Stage 06: Independent Mathematical Trust Certification**
   - **Independent Numerical Audit**: Evaluates primal residuals $\|(Ax - b)^+\|_\infty$, bound violations $\|(l - x)^+\|_\infty + \|(x - u)^+\|_\infty$, integrality margins $\max_j |x_j - [x_j]|$, and objective recomputations.
   - Issues a cryptographically styled PASS/FAIL certification badge against strict scaled relative tolerances ($\epsilon = 10^{-4}$).

---

## 5. Quickstart & How to Run

### Installation
Ensure Python 3.10+ and Node.js v18+ are installed.
```bash
# Clone or navigate to the repository
cd VX03

# Install Python dependencies
pip install -r requirements.txt

# Install Next.js frontend dependencies
cd frontend && npm install && cd ..
```

### Option A: Launch Full-Stack Demo (Next.js Web UI + FastAPI)
Run the master launcher:
```bash
python run_demo.py
```
This boots:
- **Interactive Next.js Dashboard**: [http://localhost:3000](http://localhost:3000)
- **FastAPI REST API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option B: Run the Rich Terminal CLI
```bash
# Run Refinery Blending Demo
python -m sovereign_opt.cli.app demo --preset refinery_blending_lp

# Run Power Grid Unit Commitment (MILP) Demo
python -m sovereign_opt.cli.app demo --preset power_unit_commitment

# Solve any custom MPS or LP file
python -m sovereign_opt.cli.app solve path/to/model.mps --algorithm auto
```

### Option C: Run Automated Test Suite
```bash
python -m pytest -v tests/
```
The suite covers every v1 regression, randomized accuracy checks against brute force, closed-form KKT solutions and a HiGHS reference (test-only), presolve round trips, the MPS parser, and the REST API.

### Option D: Run the Benchmark Suite
```bash
python -m benchmarks.run_benchmarks
```
Runs every instance with each applicable algorithm through presolve, solve, postsolve and the certificate, and exits non-zero if any run is not certified optimal.

---

## 6. Project Directory Layout

```
VX03/
├── sovereign_opt/               # Sovereign Core Optimization Engine
│   ├── model/                   # Universal OptimizationModel & entities
│   ├── parsers/                 # MPS and LP format readers
│   ├── sparse/                  # Dual CSR/CSC matrix & Ruiz scaling
│   ├── presolve/                # Presolve reduction pipeline & postsolve
│   ├── ml/                      # AI Strategy Engine & 28-dim features
│   ├── solvers/                 # Sovereign mathematical solvers
│   │   ├── lp/                  # Revised Simplex & Interior Point Method
│   │   ├── milp/                # Branch & Bound, Branching, Heuristics
│   │   └── qp/                  # Active Set QP & KKT solvers
│   ├── runtime/                 # CPU/GPU heterogeneous device runtime
│   ├── validation/              # Independent Mathematical Trust Validator
│   ├── server.py                # FastAPI REST API backend
│   └── cli/                     # Rich terminal CLI
├── frontend/                    # Next.js Modern Demo Web Application
│   ├── app/                     # Next.js App Router (page.tsx, layout.tsx)
│   └── package.json
├── benchmarks/                  # Industrial case studies & Netlib models
│   ├── industrial/              # Refinery Blending & Power Grid Unit Commitment
│   └── netlib/                  # Netlib AFIRO benchmark
├── docs/                        # Complete theoretical & mathematical documentation
│   └── theory/
│       ├── 01_mathematical_foundations.md
│       ├── 02_sparse_and_numerical.md
│       ├── 03_presolve_and_postsolve.md
│       ├── 04_ml_strategy_engine.md
│       ├── 05_verification_and_trust.md
│       └── 06_industrial_case_studies.md
├── tests/                       # Comprehensive pytest suite
├── run_demo.py                  # Master launcher for Next.js + FastAPI
├── requirements.txt             # Minimal Python dependencies
└── README.md                    # This master documentation file
```

---

## 7. Mathematical Theory Documentation Directory

For complete mathematical derivations, algorithm steps, and proofs, explore:
- [`01_mathematical_foundations.md`](file:///c:/Users/HP/Desktop/VX/VX03/docs/theory/01_mathematical_foundations.md): Formulations for LP standard form, Two-Phase Revised Simplex, Mehrotra Primal-Dual IPM, Branch-and-Bound, and Active Set QP.
- [`02_sparse_and_numerical.md`](file:///c:/Users/HP/Desktop/VX/VX03/docs/theory/02_sparse_and_numerical.md): Sparse CSR/CSC storage, SpMV complexity, and Ruiz equilibration scaling.
- [`03_presolve_and_postsolve.md`](file:///c:/Users/HP/Desktop/VX/VX03/docs/theory/03_presolve_and_postsolve.md): Presolve reduction theorems and exact postsolve reconstruction proofs.
- [`04_ml_strategy_engine.md`](file:///c:/Users/HP/Desktop/VX/VX03/docs/theory/04_ml_strategy_engine.md): 28-dimensional structural feature representation, algorithm prediction, and deterministic safety fallbacks.
- [`05_verification_and_trust.md`](file:///c:/Users/HP/Desktop/VX/VX03/docs/theory/05_verification_and_trust.md): Mathematical trust certificate formulas, primal/dual residuals, and integrality margin tests.
- [`06_industrial_case_studies.md`](file:///c:/Users/HP/Desktop/VX/VX03/docs/theory/06_industrial_case_studies.md): Engineering problem formulations for Refinery Blending and Power Generation Unit Commitment.

---

## 8. License & Copyright Notice

Distributed under the MIT License.

```
Copyright (c) 2026 VioniX and its contributors. All rights reserved.
```

See [`LICENSE`](file:///c:/Users/HP/Desktop/VX/VX03/LICENSE) for complete legal terms. For community guidelines, see [`CONTRIBUTING.md`](file:///c:/Users/HP/Desktop/VX/VX03/CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](file:///c:/Users/HP/Desktop/VX/VX03/CODE_OF_CONDUCT.md). For security vulnerability disclosure, see [`SECURITY.md`](file:///c:/Users/HP/Desktop/VX/VX03/SECURITY.md).