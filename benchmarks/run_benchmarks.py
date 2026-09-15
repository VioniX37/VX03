"""
Sovereign Optimizer benchmark runner.

    python -m benchmarks.run_benchmarks

Runs every built-in instance through the full pipeline (presolve, core solve, postsolve,
certificate) with each applicable algorithm and prints status, objective error against the
reference optimum, certificate, runtime, iterations and nodes. Exits with code 1 if any run is
not certified optimal or misses its reference by more than 1e-6 (relative).
"""
import sys
import time

from sovereign_opt.solvers.dispatch import solve_model
from benchmarks.netlib.afiro import build_netlib_afiro, AFIRO_OPTIMAL_OBJECTIVE
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.power_dispatch import build_unit_commitment_model
from benchmarks.industrial.portfolio_selection import build_portfolio_selection_model

# Reference optima: AFIRO from Netlib; the others were cross-checked with HiGHS (LP / MILP)
# or certified by KKT recomputation (QP) during development.
CASES = [
    ("AFIRO (Netlib LP)", build_netlib_afiro, ("simplex", "dual_simplex", "interior_point"), AFIRO_OPTIMAL_OBJECTIVE),
    ("Refinery blending LP", lambda: build_refinery_blending_model(False), ("simplex", "interior_point"), -6094333.333333333),
    ("Refinery blending QP", lambda: build_refinery_blending_model(True), ("qp_interior_point", "active_set"), -5903161.066392752),
    ("Unit commitment 4h (MILP)", lambda: build_unit_commitment_model(4), ("branch_and_bound",), 58210.0),
    ("Unit commitment 8h (MILP)", lambda: build_unit_commitment_model(8), ("branch_and_bound",), 111562.5),
    ("Unit commitment 24h (MILP)", lambda: build_unit_commitment_model(24), ("branch_and_bound",), 363372.9),
    ("Portfolio selection (MIQP)", build_portfolio_selection_model, ("branch_and_bound",), None),
]


def main() -> int:
    header = f"{'instance':28s} {'algorithm':18s} {'status':10s} {'objective':>18s} {'rel. error':>10s} {'certificate':19s} {'time':>8s} {'iters':>6s} {'nodes':>6s}"
    print(header)
    print("-" * len(header))
    failures = 0
    for name, build, algorithms, reference in CASES:
        for algorithm in algorithms:
            t = time.time()
            out = solve_model(build(), algorithm=algorithm, enable_presolve=True)
            elapsed = time.time() - t
            r, cert = out.result, out.certificate
            obj = r.objective_value
            err = abs(obj - reference) / max(1.0, abs(reference)) if (obj is not None and reference is not None) else None
            ok = cert.status == "OPTIMAL_CERTIFIED" and (err is None or err <= 1e-6)
            failures += not ok
            print(f"{name:28s} {algorithm:18s} {r.status.value:10s} {obj if obj is not None else float('nan'):18.6f} "
                  f"{(f'{err:.1e}' if err is not None else '-'):>10s} {cert.status:19s} {elapsed:7.3f}s "
                  f"{r.iterations:6d} {r.nodes_explored:6d}{'' if ok else '  <-- FAIL'}")
    print("-" * len(header))
    print("all runs certified optimal" if not failures else f"{failures} run(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
