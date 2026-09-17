"""
Concurrent (multi-core) optimization: race several solver strategies in parallel processes.

No single algorithm is best on every model: dual simplex wins on degenerate LPs, interior point on
large sparse ones, PDLP on huge ones; for MILP, cut-heavy, heuristic-heavy and different branching
settings can differ by orders of magnitude on the same instance. Running a portfolio on separate
CPU cores and taking the first proven answer gives the best of all of them for the wall-clock time
of the fastest. (Processes, not threads: each strategy gets its own core and its own interpreter.)

LP / QP: the first worker that proves optimality, infeasibility or unboundedness wins; the others
are terminated.
MILP / MIQP: the first worker that proves optimality wins; at the time limit the result with the
best incumbent is returned.
"""
import multiprocessing as mp
import os
import queue
import time
from typing import Any, Dict, List, Optional, Tuple

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus

DECISIVE = (SolverStatus.OPTIMAL, SolverStatus.INFEASIBLE, SolverStatus.UNBOUNDED)

LP_PORTFOLIO = ["dual_simplex", "interior_point", "simplex", "hybrid_pdlp"]
QP_PORTFOLIO = ["qp_interior_point", "active_set"]
MIP_PORTFOLIO = [
    ("default", {}),
    ("aggressive_cuts", {"max_cut_rounds": 30}),
    ("heuristics_first", {"enable_cuts": False}),
    ("most_fractional", {"branching": "most_fractional"}),
]


def _worker(strategy: str, kind: str, options: Dict[str, Any], model: OptimizationModel, time_limit: float, q):
    t0 = time.time()
    try:
        if kind == "mip":
            from sovereign_opt.solvers.milp.branch_bound import BranchAndBoundSolver
            from sovereign_opt.solvers.milp.branching import MostFractionalBranching
            opts = dict(options)
            if opts.pop("branching", None) == "most_fractional":
                opts["branching_strategy"] = MostFractionalBranching()
            solver = BranchAndBoundSolver(time_limit_seconds=time_limit, **opts)
        else:
            from sovereign_opt.solvers.dispatch import make_solver
            solver = make_solver(strategy, time_limit)
        res = solver.solve(model, time_limit_seconds=time_limit)
        res.diagnostics.pop("tree_trace", None) if len(res.diagnostics.get("tree_trace", [])) > 200 else None
        q.put((strategy, res, time.time() - t0, None))
    except Exception as exc:
        q.put((strategy, None, time.time() - t0, f"{type(exc).__name__}: {exc}"))


class ConcurrentSolver(SolverBase):
    def __init__(self, workers: Optional[int] = None):
        super().__init__(name="Concurrent")
        self.workers = workers

    def portfolio(self, problem_class: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        if problem_class in ("MILP", "MIQP"):
            items = [(name, "mip", opts) for name, opts in MIP_PORTFOLIO]
            if problem_class == "MIQP":
                items = items[:1]  # MIQP branch-and-bound has a single configuration
        elif problem_class == "QP":
            items = [(a, "cont", {}) for a in QP_PORTFOLIO]
        else:
            items = [(a, "cont", {}) for a in LP_PORTFOLIO]
        cap = self.workers or max(1, min(len(items), (os.cpu_count() or 2) - 1))
        return items[:cap]

    def solve(self, model: OptimizationModel, time_limit_seconds: float = 60.0, **kwargs) -> SolverResult:
        start = time.time()
        problem_class = model.classify()
        items = self.portfolio(problem_class)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = {}
        for name, kind, opts in items:
            p = ctx.Process(target=_worker, args=(name, kind, opts, model, time_limit_seconds, q), daemon=True)
            p.start()
            procs[name] = p

        finished: Dict[str, Dict[str, Any]] = {}
        winner: Optional[Tuple[str, SolverResult]] = None
        deadline = start + time_limit_seconds + 30.0
        while len(finished) < len(items) and time.time() < deadline:
            try:
                name, res, secs, err = q.get(timeout=0.2)
            except queue.Empty:
                if all(not p.is_alive() for p in procs.values()) and q.empty():
                    break
                continue
            finished[name] = {"status": res.status.value if res else "error", "seconds": round(secs, 3),
                              "objective": res.objective_value if res else None, "error": err}
            if res is not None and res.status in DECISIVE:
                winner = (name, res)
                break
            if res is not None and res.primal_solution:
                if winner is None or _better(res, winner[1], model):
                    winner = (name, res)

        for name, p in procs.items():
            if p.is_alive():
                p.terminate()
                finished.setdefault(name, {"status": "cancelled (another strategy won)"})
        for p in procs.values():
            p.join(2)

        if winner is None:
            return SolverResult(status=SolverStatus.NUMERICAL_ERROR, runtime_seconds=time.time() - start,
                                diagnostics={"concurrent_race": finished, "error": "no strategy produced a result"})
        name, res = winner
        res.runtime_seconds = time.time() - start
        res.diagnostics["concurrent_winner"] = name
        res.diagnostics["concurrent_race"] = finished
        res.diagnostics["concurrent_workers"] = len(items)
        return res


def _better(a: SolverResult, b: SolverResult, model: OptimizationModel) -> bool:
    if a.objective_value is None:
        return False
    if b.objective_value is None:
        return True
    sign = 1.0 if model.objective.sense == "minimize" else -1.0
    return sign * a.objective_value < sign * b.objective_value
