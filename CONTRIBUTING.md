# Contributing

Thank you for your interest in the VioniX Sovereign Optimization Engine. Bug reports, benchmark
results, documentation fixes and pull requests are all welcome.

## Licensing of contributions

This project is released under the [PolyForm Strict License 1.0.0](LICENSE), which does not permit
redistribution or modified versions outside this repository. Contributions are therefore made
**to VioniX**:

By opening a pull request or submitting a patch, you confirm that you wrote the contribution (or have
the right to submit it), and you grant VioniX a perpetual, worldwide, royalty-free, irrevocable licence
to use, modify, relicense and distribute your contribution as part of this project. You keep the
copyright in your own work.

Forking on GitHub to prepare a pull request is fine. Publishing or distributing a modified version
elsewhere is not permitted by the licence.

## Design principles

1. **No solver black boxes.** The engine (`sovereign_opt/`) must never import Gurobi, CPLEX, Xpress,
   SCIP, HiGHS, GLPK or any other solver. They may appear only in `benchmarks/` and `Gurobi files/` as
   external references. `tests/test_benchmarks.py::test_engine_never_imports_comparison_solvers`
   enforces this.
2. **The strategy engine is advisory.** It may choose a method or a setting; it must never change
   tolerances or certify a result.
3. **Every answer is verifiable.** Results must pass the independent validator, which recomputes
   everything from the raw model and never trusts solver-internal state.
4. **Honest statuses.** An iteration limit is not "optimal" and a heuristic point is not "proven".

## Development setup

```bash
git clone https://github.com/VioniX37/VX03.git
cd VX03
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

cd frontend
npm install
npm run dev
```

## Before opening a pull request

- `python -m pytest -q tests/` passes.
- `cd frontend && npm run build` succeeds if you touched the dashboard.
- New or changed numerical methods are documented in `docs/theory/`.
- Performance claims are backed by a saved run in `benchmarks/results/` (for example from
  `python -m benchmarks.compare --suite <suite>`), including cases where the change is slower.

## Pull request flow

1. Branch from `main`: `git checkout -b feature/short-description`
2. Write clear commit messages, for example `fix(simplex): refactorize before declaring optimality`.
3. Open a pull request against `main` and fill in the template.
4. Make sure CI is green.

## Reporting bugs

Open an issue with the model file (`.mps` / `.lp`) or a script that builds the model, the algorithm you
selected, the output you got and the output you expected. For security issues, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.
