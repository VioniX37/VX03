# VioniX Sovereign Optimization Engine

**An indigenous, GPU-ready mathematical optimization solver for LP, MILP, QP and MIQP, built from first principles, with an independent certificate on every answer.**

[![License: PolyForm Strict 1.0.0](https://img.shields.io/badge/license-PolyForm%20Strict%201.0.0-blue)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-95%20passing-brightgreen)

Built by **Team VioniX** for **Smart India Hackathon 2026**, problem statement **SIH26119**:
*Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to Xpress / CPLEX)*.

> Source-available under the [PolyForm Strict License 1.0.0](LICENSE): free for personal, research,
> educational and government use; no commercial use, redistribution or modified versions. See [License](#license).

---

## Contents

1. [Why this exists](#1-why-this-exists)
2. [What makes it different](#2-what-makes-it-different)
3. [How it works](#3-how-it-works)
4. [Algorithms](#4-algorithms)
5. [Results](#5-results)
6. [Quickstart](#6-quickstart)
7. [Dashboard](#7-dashboard)
8. [Project layout](#8-project-layout)
9. [Status, limitations and roadmap](#9-status-limitations-and-roadmap)
10. [Documentation](#10-documentation)
11. [License](#license)

---

## 1. Why this exists

Power grids, refineries, logistics networks and defence planning in India run on foreign, licensed
optimization solvers. That means recurring licence costs, dependence on external licence servers,
internals that cannot be audited, and friction in air-gapped environments.

This engine is our answer: every algorithm, from basis factorization to branch-and-cut, is written
in this repository. It does **not** wrap or call Gurobi, CPLEX, Xpress, SCIP, HiGHS or GLPK. A test
(`tests/test_benchmarks.py::test_engine_never_imports_comparison_solvers`) enforces that the engine
never imports them; Gurobi and HiGHS appear only in the benchmark scripts, as external references.

**Core tenet:** *the strategy engine decides **how** to attack a problem; the mathematics decides
**what** the answer is, and an independent validator proves it.*

## 2. What makes it different

| | |
|---|---|
| **Proof, not trust** | Every solve is re-checked from the raw model by an independent validator: constraints, bounds, integrality, objective, KKT conditions, duality gap and MIP bound. Each result is labelled `OPTIMAL_CERTIFIED`, `FEASIBLE_CERTIFIED` or `FAILED`. |
| **Glass box** | Every stage (presolve diff, strategy choice, iteration trace, branch-and-bound tree, certificate) is visible in the dashboard and CLI. |
| **One engine, four problem classes** | LP, MILP, QP and MIQP behind one model object, with the method chosen automatically from the model's structure. |
| **GPU-ready at scale** | A restarted Halpern PDHG (PDLP) solver runs on PyTorch CUDA when a GPU is present, and on fused multi-threaded CPU kernels otherwise. |
| **Sovereign deployment** | Pure Python + NumPy/SciPy/Numba (PyTorch optional for GPU). No licence server, no size caps, runs fully offline. |
| **Honest benchmarks** | We publish where we are slower than commercial solvers, not only where we are faster. |

## 3. How it works

```
 MPS / LP file or REST API
          │
 [1] Ingestion & model        universal model object (LP / MILP / QP / MIQP)
 [2] Validation & class       NaN / bound conflicts / empty structures, problem classification
 [3] Presolve & scaling       up to 10 passes of reductions, power-of-two + Ruiz scaling, postsolve map
 [4] Strategy & hardware      23 structural features pick the method; low confidence -> proven default
 [5] Core solver              simplex · IPM · PDLP · branch-and-cut · QP IPM · active set
 [6] Postsolve & recovery     original variables, shadow prices, reduced costs, vertex polish
 [7] Trust validation         independent audit -> certificate
 [8] Delivery                 CLI · Next.js dashboard · REST API
```

If an LP method stalls (iteration limit or numerical trouble) the dispatcher hands the problem to dual
simplex automatically. A `concurrent` mode races several methods on separate CPU cores and takes the
first proven answer.

## 4. Algorithms

All implemented natively in [`sovereign_opt/`](sovereign_opt/).

**Linear programming**
- Bounded primal and dual revised simplex: Devex (primal) and dual steepest-edge pricing, two-pass Harris
  ratio tests, bound flipping, composite Phase 1, perturbation and Bland fallback against cycling
  ([`simplex_engine.py`](sovereign_opt/solvers/lp/simplex_engine.py)).
- Basis factorization: sparse LU (SuperLU) with a sparse product-form eta file, Numba-compiled update
  kernels and repair of singular bases ([`basis.py`](sovereign_opt/solvers/lp/basis.py)).
- Homogeneous self-dual interior point method with Mehrotra predictor-corrector, certified
  infeasibility detection and crossover to a vertex ([`interior_point.py`](sovereign_opt/solvers/lp/interior_point.py)).
- PDLP: restarted, reflected Halpern PDHG with primal-weight rebalancing on fused, nonzero-balanced
  multi-threaded kernels, or PyTorch CUDA ([`pdlp.py`](sovereign_opt/solvers/lp/pdlp.py),
  [`pdhg_kernels.py`](sovereign_opt/solvers/lp/pdhg_kernels.py)), plus sparse vertex polishing that turns
  an approximate PDLP point into an exact optimal vertex when it can certify one ([`polish.py`](sovereign_opt/solvers/lp/polish.py)).

**Mixed-integer programming**
- Branch-and-cut with a warm-started dual simplex at every node, Gomory mixed-integer and knapsack cover
  cuts, reliability branching (pseudocosts initialised by budgeted strong branching), reduced-cost fixing
  ([`branch_bound.py`](sovereign_opt/solvers/milp/branch_bound.py), [`cuts.py`](sovereign_opt/solvers/milp/cuts.py)).
- Primal heuristics: rounding, fractional diving, feasibility pump, RINS ([`heuristics.py`](sovereign_opt/solvers/milp/heuristics.py)).
- MIQP: branch-and-bound over convex QP relaxations.

**Quadratic programming**
- Primal-dual interior point on a regularized quasi-definite KKT system; convexity checked with a
  Cholesky / LDL inertia test ([`interior_point_qp.py`](sovereign_opt/solvers/qp/interior_point_qp.py)).
- Null-space active-set method for small and medium problems ([`active_set.py`](sovereign_opt/solvers/qp/active_set.py)).
- Lawson–Hanson NNLS for sign-correct dual recovery ([`qp/polish.py`](sovereign_opt/solvers/qp/polish.py)).

**Presolve, scaling and validation**
- Presolve: fixed and empty rows/columns, singleton rows, activity-based forcing and redundant rows,
  doubleton equations, dominated and duplicate columns, parallel rows, coefficient tightening; every
  reduction is recorded and undone by postsolve ([`presolver.py`](sovereign_opt/presolve/presolver.py)).
- Scaling: geometric-mean scaling rounded to powers of two, and Ruiz equilibration ([`scaling.py`](sovereign_opt/sparse/scaling.py)).
- Strategy engine: 23 structural features and transparent scoring rules for method selection, with a
  confidence threshold and deterministic fallback ([`strategy.py`](sovereign_opt/ml/strategy.py)). A
  selector trainer exists in [`benchmarks/train_selector.py`](benchmarks/train_selector.py); no trained
  model ships yet.
- Validator: recomputes everything from the raw model ([`validator.py`](sovereign_opt/validation/validator.py)).

## 5. Results

All results below are saved as JSON / Markdown in [`benchmarks/results/`](benchmarks/results/) and can be
reproduced with the commands in [Quickstart](#6-quickstart). Unless noted, the machine is a 12-thread
Windows 11 laptop with no GPU. Wall-clock times vary by roughly ±30% between runs.

### 5.1 Head-to-head demo (every solver gets its own process, the same 12 threads and the same input)

| Instance | Size (rows × cols) | Ours | HiGHS | Gurobi 13 | Our objective vs Gurobi |
|---|---|---:|---:|---:|---:|
| Supply-chain LP, 1M variables | 193,442 × 999,188 (2.96M nnz) | **125.8 s** | 575.4 s | 29.4 s | 1.6e-7 |
| Supply-chain LP, 100k variables | 19,586 × 99,728 (296k nnz) | **6.6 s** | 12.8 s | 2.8 s | 5e-15 (exact vertex) |
| STCQP1 (Maros–Mészáros QP) | 2,052 × 4,097 | **0.32 s** | >300 s (time limit) | 0.02 s | 2e-10 |
| Unit commitment MILP, 480 binaries | 1,228 × 720 | **0.71 s** | 0.06 s | 0.06 s | exact, gap 0 |

**Read honestly:** at one million variables we are **4.6× faster than HiGHS** and reach the same optimum
as Gurobi to 1.6×10⁻⁷, but **Gurobi is 4.3× faster than us**. On small QPs and MILPs the answers are
identical and certified, but our interior point QP and branch-and-cut are not yet tuned to commercial
speed. Sources: `demo_supply_chain_1m_20260921-164036.json`, `demo_supply_chain_100k_20260918-155859.json`,
`demo_qp_stcqp1_20260918-141244.json`, `demo_milp_uc_720_20260918-141348.json`.

### 5.2 GPU (Tesla T4 on Kaggle, earlier PDLP version, tolerance 1e-4)

| LP size | PDLP on T4 | PDLP on CPU (4 vCPU) | HiGHS (1 thread, same VM) |
|---|---:|---:|---:|
| 1M variables (2.96M nnz) | 9.6 s | 101.6 s | 305.7 s |
| 3M variables (8.88M nnz) | 36.5 s | 443.2 s | did not finish in 900 s |

This is a first-order method at a loose tolerance (objective within ~1e-4 of HiGHS), so it is **not** a
like-for-like comparison with an exact solver. It shows where the GPU path is heading. Source:
`scale_kaggle_gpu_20260917-081355.json`, notebook [`notebooks/kaggle_gpu_benchmark.ipynb`](notebooks/kaggle_gpu_benchmark.ipynb).

### 5.3 Correctness on public test sets (vs HiGHS as a reference)

| Suite | Result |
|---|---|
| Netlib LP (23 instances) | 23 / 23 correct and certified |
| Netlib LP, larger instances (13) | 11 / 13 (`fit2p` error, `cre-a` too slow) |
| Infeasible LPs (6) | 6 / 6 infeasibility proven |
| Maros–Mészáros QP, small (17) / large (5) | 16 / 17 and 3 / 5 |
| Numerical robustness (degenerate, badly scaled, infeasible) | 19 / 22 |
| MIPLIB subset (11), 60 s limit | 2 / 11 solved to optimality; the rest return certified feasible incumbents |

On small and medium Netlib LPs HiGHS is 13–350× faster than us; our strength today is large sparse LP
and correctness, not small-model speed.

## 6. Quickstart

Requirements: Python 3.10+, Node.js 18+ (for the dashboard). A CUDA GPU is optional.

```bash
git clone https://github.com/VioniX37/VX03.git
cd VX03
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

**Full demo (REST API + dashboard):**
```bash
python run_demo.py
```
Dashboard: <http://localhost:3000> · API docs (OpenAPI): <http://localhost:8000/docs>

**Command line:**
```bash
python -m sovereign_opt.cli.app info                                   # hardware / CUDA detection
python -m sovereign_opt.cli.app demo --preset refinery_blending_lp     # built-in case study
python -m sovereign_opt.cli.app demo --preset power_unit_commitment
python -m sovereign_opt.cli.app solve path/to/model.mps --algorithm auto
```
Presets: `netlib_afiro`, `refinery_blending_lp`, `refinery_blending_qp`, `power_unit_commitment`,
`supply_chain_lp`, `portfolio_miqp`. Algorithms: `auto`, `concurrent`, `simplex`, `dual_simplex`,
`interior_point`, `pdlp`, `hybrid_pdlp`, `branch_and_bound`, `qp_interior_point`, `active_set`.

**Tests:**
```bash
python -m pytest -q tests/
```

**Benchmarks** (HiGHS and Gurobi are used only for comparison and are never imported by the engine):
```bash
pip install -r requirements-bench.txt
python -m benchmarks.fetch --only netlib infeas miplib     # download public instances
python -m benchmarks.compare --suite netlib
python -m benchmarks.demo_race --instance supply_chain_1m --all --save
python -m benchmarks.scale                                  # supply-chain LP at growing sizes
```
Gurobi needs a licence: place `gurobi.lic` in `Gurobi files/` (gitignored) or set `GRB_LICENSE_FILE`.
Without one, the demo greys out the Gurobi bars and says why.

## 7. Dashboard

A Next.js dashboard styled as a terminal session, because the product itself is a CLI.

- **Views:** model → presolve → strategy → solve → solution → certificate, plus **benchmarks** and a
  live **demo race** against Gurobi and HiGHS.
- **Model view:** sparsity plot of the constraint matrix and Ruiz scaling before/after.
- **Solve view:** convergence trace, and the branch-and-bound tree for MILPs.
- **Certificate view:** every check the validator ran, with its residual and tolerance.
- **Keyboard:** `1`–`6` switch views, `ctrl+enter` runs, `j`/`k` move through problems.

## 8. Project layout

```
VX03/
├── sovereign_opt/            the engine (never imports a third-party solver)
│   ├── model/                universal model: variables, constraints, objective
│   ├── parsers/              MPS (fixed/free, QP extensions), CPLEX-LP, MPS writer
│   ├── sparse/               CSR/CSC matrix, geometric and Ruiz scaling
│   ├── presolve/             presolve reductions and postsolve
│   ├── ml/                   23 structural features and strategy engine
│   ├── solvers/
│   │   ├── lp/               simplex engine, basis LU, IPM, PDLP, PDHG kernels, vertex polish
│   │   ├── milp/             branch-and-cut, branching, cuts, heuristics
│   │   ├── qp/               QP interior point, active set, KKT, NNLS polish
│   │   ├── dispatch.py       end-to-end pipeline
│   │   └── concurrent.py     multi-core method racing
│   ├── runtime/              CPU / CUDA device detection
│   ├── validation/           independent validator and certificates
│   ├── cli/                  Rich terminal CLI
│   └── server.py             FastAPI REST API
├── frontend/                 Next.js dashboard
├── benchmarks/               suites, demo race, scale tests, industrial models, saved results
├── Gurobi files/             Gurobi versions of the benchmark models (comparison only)
├── notebooks/                Kaggle GPU benchmark notebook
├── tools/                    data export and Kaggle bundling helpers
├── docs/theory/              mathematical documentation
├── tests/                    pytest suite
└── run_demo.py               launches API + dashboard
```

## 9. Status, limitations and roadmap

**Working today:** all four problem classes end to end; certified results; CLI, REST API and dashboard;
1M-variable LP on a laptop CPU; 95 automated tests on CI (Python 3.10–3.12).

**Known limitations**
- **MILP speed** is the main gap: most MIPLIB instances do not close within 60 s, and a 5,760-binary
  unit-commitment model ends at a 3.1% gap after 180 s where HiGHS needs 2 s.
- **Small and medium LPs:** HiGHS and Gurobi are much faster.
- **GPU coverage:** only PDLP runs on GPU today; simplex, IPM and branch-and-bound are CPU-only.
- **Strategy engine:** transparent scoring rules; no trained selector ships yet.

**Roadmap**
- Parallel branch-and-bound, more cut families (MIR, flow cover) and GPU-solved LP relaxations for MILP.
- Multi-GPU PDLP for 10M+ variable models.
- GPU kernels for the interior point method.
- Packaged on-premises / air-gapped install with Python and C APIs.
- Pilots on real models from power and logistics operators.

## 10. Documentation

- [01 Mathematical foundations](docs/theory/01_mathematical_foundations.md): LP standard form, simplex, IPM, branch-and-bound, active-set QP
- [02 Sparse and numerical methods](docs/theory/02_sparse_and_numerical.md): CSR/CSC storage, SpMV, Ruiz scaling
- [03 Presolve and postsolve](docs/theory/03_presolve_and_postsolve.md)
- [04 Strategy engine](docs/theory/04_ml_strategy_engine.md): features, method selection, safety fallback
- [05 Verification and trust](docs/theory/05_verification_and_trust.md): certificate formulas
- [06 Industrial case studies](docs/theory/06_industrial_case_studies.md): refinery blending, unit commitment
- [07 v2 engine upgrades](docs/theory/07_v2_engine_upgrades.md): what changed in v2 and how each fix was verified

## License

Copyright © 2026 VioniX. Released under the [PolyForm Strict License 1.0.0](LICENSE).

**You may:** clone, read, run and use the engine to solve your own problems for any non-commercial
purpose: personal study, research, teaching, hobby projects, and use by educational, charitable,
public-research and government institutions.

**You may not:** use it commercially, redistribute it, or publish modified versions or derivative works.

This is a *source-available* licence, not an OSI-approved open-source licence. For commercial licensing
or other permissions, contact [VioniX on GitHub](https://github.com/VioniX37).

Contributions are welcome under the terms in [CONTRIBUTING.md](CONTRIBUTING.md). Please also read the
[Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

**Acknowledgements:** benchmark instances come from [Netlib](https://www.netlib.org/lp/),
[MIPLIB](https://miplib.zib.de/) and the Maros–Mészáros QP test set. [HiGHS](https://highs.dev/) and
[Gurobi](https://www.gurobi.com/) were used as external reference solvers.
