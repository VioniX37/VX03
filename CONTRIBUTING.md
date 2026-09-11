# Contributing to Sovereign Optimizer

Thank you for your interest in contributing to the **Sovereign Mathematical Optimization Engine**! This project is maintained by **VioniX and its contributors**.

---

## 1. Code of Conduct & Copyright Assignment

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). 

### Copyright & Rights Notice
By submitting a Pull Request, patch, or documentation to this repository, you agree that your contributions will be licensed under the project's [MIT License](LICENSE), and that copyright and licensing rights remain held by **VioniX and its contributors**.

---

## 2. Architectural Principles

Before writing code, review the core philosophical tenets of the engine:

1. **Mathematical Sovereignty**: Never introduce dependencies on commercial or external open-source solver blackboxes (e.g., Gurobi, CPLEX, SCIP, HiGHS, GLPK). All optimization algorithms (Simplex, Interior Point, Branch & Bound, Active Set QP) must be implemented natively from foundational numerical linear algebra.
2. **First-Principles Correctness**: ML components are strictly **advisory** (heuristic selection, branching variable scores). ML must never have the authority to alter constraint tolerances or certify optimality.
3. **Independent Trust**: Every solution must be verifiable through an independent zero-trust mathematical audit without internal solver basis reuse.
4. **Clean Code & Type Annotations**: All Python modules must use strict type hints (`typing`), docstrings explaining mathematical formulations, and pass automated testing.

---

## 3. Development Setup

### Python Environment
```bash
# Clone the repository
git clone https://github.com/vionix/sovereign-opt.git
cd sovereign-opt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy
```

### Next.js Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 4. Testing & Verification Requirements

Every pull request must satisfy the following criteria:

1. **Automated Test Suite**:
   ```bash
   python -m pytest -v tests/
   ```
   All tests must pass 100% with execution time under 2.0 seconds on standard hardware.

2. **Frontend Build Verification**:
   ```bash
   cd frontend
   npm run build
   ```
   Must compile cleanly under TypeScript and Turbopack with zero lint or build errors.

3. **Mathematical Documentation**:
   Any new algorithm or modification to numerical linear algebra (e.g., matrix scaling, presolve reduction) must be documented in `docs/theory/`.

---

## 5. Submitting a Pull Request

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "feat(simplex): implement Harris ratio test for anti-degeneracy"
   ```
3. Push to your branch and open a Pull Request against `main`.
4. Ensure all CI checks pass.

Thank you for helping build a truly sovereign, open mathematical computing foundation!
