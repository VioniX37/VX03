"""
Sovereign Branch-and-Cut solver for Mixed-Integer Linear Programming (v2), with MIQP support.

Implemented from mathematical foundations:
- One persistent bounded simplex engine: every node is re-optimized with a warm-started
  dual simplex from its parent's optimal basis (no model rebuilding, no cold starts).
- Root cutting-plane loop: Gomory mixed-integer cuts + knapsack cover cuts with efficacy
  and parallelism filtering; rounds stop when the bound stalls.
- Reliability branching: pseudocosts initialized by strong branching on unreliable candidates,
  product scoring, ML structural score as tie-breaker. Strong branching also detects
  infeasible children and tightens bounds.
- Node selection: best-bound search with depth-first plunging.
- Reduced-cost bound tightening at every node (global at the root).
- Primal heuristics: rounding, fractional diving, feasibility pump, RINS.
- Objective integrality detection for stronger pruning.
- Correct global best bound (minimum over open nodes) and sense-independent pruning / gap.
- MIQP: best-bound branch-and-bound over convex QP interior point relaxations.
"""
import heapq
import itertools
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver
from sovereign_opt.solvers.lp.standard_form import BoundedForm
from sovereign_opt.solvers.lp.simplex_engine import SimplexEngine, LPStatus, BASIC, AT_LOWER, AT_UPPER
from sovereign_opt.solvers.milp.node import TreeNode
from sovereign_opt.solvers.milp.branching import (
    BranchingStrategy,
    PseudocostTracker,
    ReliabilityBranching,
    product_score,
    static_branching_scores,
)
from sovereign_opt.solvers.milp.cuts import separate_gmi_cuts, separate_cover_cuts, select_cuts
from sovereign_opt.solvers.milp.heuristics import (
    PrimalHeuristic,
    copy_model_with_bounds,
    fractional_dive,
    feasibility_pump,
    rins,
)

TRACE_LIMIT = 60


class BranchAndBoundSolver(SolverBase):
    """
    Sovereign Branch-and-Cut MILP / MIQP Solver.
    """

    def __init__(
        self,
        max_nodes: int = 100000,
        time_limit_seconds: float = 60.0,
        mip_gap_tolerance: float = 1e-4,
        integrality_tolerance: float = 1e-6,
        branching_strategy: Optional[BranchingStrategy] = None,
        enable_cuts: bool = True,
        enable_heuristics: bool = True,
        max_cut_rounds: int = 10,
        absolute_gap_tolerance: float = 1e-6,
        feasibility_tolerance: float = 1e-6,
    ):
        super().__init__(name="BranchAndCut")
        self.max_nodes = max_nodes
        self.time_limit = time_limit_seconds
        self.mip_gap_tolerance = mip_gap_tolerance
        self.int_tol = integrality_tolerance
        self.abs_gap = absolute_gap_tolerance
        self.feas_tol = feasibility_tolerance
        self.enable_cuts = enable_cuts
        self.enable_heuristics = enable_heuristics
        self.max_cut_rounds = max_cut_rounds
        if isinstance(branching_strategy, ReliabilityBranching) or branching_strategy is None:
            self.reliability = branching_strategy or ReliabilityBranching()
            self.branching_strategy = None
        else:
            self.reliability = ReliabilityBranching()
            self.branching_strategy = branching_strategy
        self.lp_solver = RevisedSimplexSolver()

    def solve(self, model: OptimizationModel, **kwargs) -> SolverResult:
        start = time.time()
        time_limit = float(kwargs.get("time_limit_seconds", self.time_limit))
        has_int = any(v.is_integer for v in model.variables.values())
        if not has_int:
            if model.objective.is_quadratic:
                from sovereign_opt.solvers.qp.interior_point_qp import QPInteriorPointSolver
                return QPInteriorPointSolver().solve(model, time_limit_seconds=time_limit)
            return self.lp_solver.solve(model, time_limit_seconds=time_limit)
        if model.objective.is_quadratic:
            return _MIQPBranchAndBound(self, model, start, start + time_limit).run()
        return _BranchAndCut(self, model, start, start + time_limit).run()


# ============================================================================ MILP
class _BranchAndCut:
    def __init__(self, cfg: BranchAndBoundSolver, model: OptimizationModel, start: float, deadline: float):
        self.cfg = cfg
        self.model = model
        self.start = start
        self.deadline = deadline
        self.time_limit = deadline - start
        self.int_tol = cfg.int_tol

        f = BoundedForm(model, scale=True, scale_integer_columns=False)
        self.form = f
        self.n = f.n
        self.int_idx = np.flatnonzero(f.integer_mask)
        ii = self.int_idx
        f.lb[ii] = np.ceil(f.lb[ii] - cfg.int_tol)
        f.ub[ii] = np.floor(f.ub[ii] + cfg.int_tol)
        self.int_col_mask = f.integer_mask.copy()
        self.binary_mask = f.integer_mask & (f.lb[: f.n] >= 0) & (f.ub[: f.n] <= 1)

        A0, rl0, ru0, cl0, cu0 = model.to_matrix_form()
        self.A0 = A0.tocsr()
        self.rl0, self.ru0, self.cl0, self.cu0 = rl0, ru0, cl0, cu0
        self.c0 = f.c_orig
        cont = ~f.integer_mask
        self.objective_integral = bool(
            np.all(self.c0[cont] == 0.0) and np.all(np.abs(self.c0[ii] - np.round(self.c0[ii])) < 1e-12)
        )

        self.engine = SimplexEngine.from_form(f)
        self.engine.trace_enabled = False
        self.inc_obj = np.inf
        self.inc_x: Optional[np.ndarray] = None
        self.incumbent_history: List[dict] = []
        self.heuristic_hits = {"rounding": 0, "diving": 0, "pump": 0, "rins": 0, "lp_integral": 0}
        self.nodes = 0
        self.lp_iterations = 0
        self.pc = PseudocostTracker(self.n)
        self.static_scores = static_branching_scores(f)
        self.trace: List[dict] = []
        self.cut_counts = {"gomory": 0, "cover": 0}
        self.cut_rounds = 0
        self.proof_complete = True
        self.root_bound = -np.inf
        self.counter = itertools.count(1)

    # --------------------------------------------------------------- helpers
    def user(self, z: float) -> float:
        return float(self.form.obj_sign * z + self.form.offset)

    def cutoff(self) -> float:
        if not np.isfinite(self.inc_obj):
            return np.inf
        if self.objective_integral:
            return self.inc_obj - 1.0 + 1e-6
        return self.inc_obj - max(self.cfg.abs_gap, self.cfg.mip_gap_tolerance * abs(self.inc_obj))

    def fractional(self, xs: np.ndarray) -> np.ndarray:
        v = xs[self.int_idx]
        return self.int_idx[np.abs(v - np.round(v)) > self.int_tol]

    def is_feasible(self, x: np.ndarray) -> bool:
        tol = self.cfg.feas_tol
        act = self.A0 @ x
        rt = tol * np.maximum(1.0, np.abs(act))
        if np.any(act < self.rl0 - rt) or np.any(act > self.ru0 + rt):
            return False
        vt = tol * np.maximum(1.0, np.abs(x))
        return not (np.any(x < self.cl0 - vt) or np.any(x > self.cu0 + vt))

    def try_incumbent(self, x_struct: np.ndarray, source: str) -> bool:
        x = np.array(x_struct, dtype=np.float64)
        x[self.int_idx] = np.round(x[self.int_idx])
        if not self.is_feasible(x):
            return False
        obj = float(self.c0 @ x)
        if obj < self.inc_obj - 1e-9 * max(1.0, abs(obj)):
            self.inc_obj = obj
            self.inc_x = x
            self.heuristic_hits[source] = self.heuristic_hits.get(source, 0) + 1
            self.incumbent_history.append({
                "time": round(time.time() - self.start, 4), "objective": self.user(obj),
                "source": source, "node": self.nodes,
            })
            return True
        return False

    def solve_lp(self, lb, ub, basis, max_iter: int = 200000, method: str = "dual") -> str:
        e = self.engine
        e.set_bounds(lb, ub)
        if basis is not None:
            e.set_basis(*basis)
        before = e.total_iterations
        remaining = max(0.01, self.deadline - time.time())
        st = e.solve(max_iterations=max_iter, time_limit=remaining, method=method)
        if st == LPStatus.NUMERICAL_ERROR:
            e.head = None
            st = e.solve(max_iterations=max_iter, time_limit=max(0.01, self.deadline - time.time()), method="auto")
        self.lp_iterations += e.total_iterations - before
        return st

    def fix_and_complete(self, x_struct: np.ndarray, source: str) -> bool:
        """Fix integers at rounded values and optimize the continuous part."""
        f, e = self.form, self.engine
        vals = np.round(x_struct[self.int_idx])
        lb, ub = f.lb.copy(), f.ub.copy()
        if np.any(vals < lb[self.int_idx] - 1e-9) or np.any(vals > ub[self.int_idx] + 1e-9):
            return False
        lb[self.int_idx] = vals
        ub[self.int_idx] = vals
        saved = e.get_basis() if e.head is not None else None
        st = self.solve_lp(lb, ub, None, method="auto", max_iter=50000)
        ok = st == LPStatus.OPTIMAL and self.try_incumbent(f.unscale_x(e.x)[: self.n], source)
        e.set_bounds(f.lb, f.ub)
        if saved is not None:
            e.set_basis(*saved)
        return ok

    def node_bounds(self, node: TreeNode) -> Tuple[np.ndarray, np.ndarray]:
        lb, ub = self.form.lb.copy(), self.form.ub.copy()
        cur = node
        while cur is not None:
            for j, lo, hi in cur.changes:
                if lo > lb[j]:
                    lb[j] = lo
                if hi < ub[j]:
                    ub[j] = hi
            cur = cur.parent
        return lb, ub

    def add_trace(self, node: TreeNode, status: str, z: float):
        if len(self.trace) < TRACE_LIMIT:
            self.trace.append({
                "node_id": node.node_id,
                "parent_id": node.parent.node_id if node.parent is not None else None,
                "depth": node.depth,
                "lower_bound": self.user(z) if np.isfinite(z) else float("inf"),
                "status": status,
                "branch_var": node.branch_name,
                "branch_condition": node.label,
            })

    # ------------------------------------------------------------------ root
    def run(self) -> SolverResult:
        f, e = self.form, self.engine
        if np.any(f.lb > f.ub + 1e-9):
            return self.finish(SolverStatus.INFEASIBLE, "empty integer bounds", -np.inf, exhausted=True)

        st = self.solve_lp(f.lb, f.ub, None, method="auto")
        root = TreeNode(node_id=0, parent=None, depth=0)
        if st == LPStatus.INFEASIBLE:
            self.nodes = 1
            self.add_trace(root, "INFEASIBLE", np.inf)
            return self.finish(SolverStatus.INFEASIBLE, "root LP relaxation infeasible", np.inf, exhausted=True)
        if st == LPStatus.UNBOUNDED:
            return self.classify_unbounded_relaxation()
        if st != LPStatus.OPTIMAL:
            return self.finish(SolverStatus.TIME_LIMIT if st == LPStatus.TIME_LIMIT else SolverStatus.NUMERICAL_ERROR,
                               f"root LP: {st}", -np.inf)
        self.nodes = 1
        self.root_lp_bound = e.objective()
        self.add_trace(root, "ROOT", self.root_lp_bound)

        xs = f.unscale_x(e.x)[: self.n]
        if self.fractional(e.x[: self.n]).size == 0 and self.try_incumbent(xs, "lp_integral"):
            return self.finish(SolverStatus.OPTIMAL, "root LP relaxation integral", self.root_lp_bound, exhausted=True)

        if self.cfg.enable_heuristics:
            names = f.var_names
            rounded = PrimalHeuristic.simple_rounding(self.model, {nm: float(v) for nm, v in zip(names, xs)})
            if rounded is not None:
                self.try_incumbent(np.array([rounded[nm] for nm in names]), "rounding")
            saved = e.get_basis()
            self.fix_and_complete(xs, "rounding")
            e.set_basis(*saved)
            self.solve_lp(f.lb, f.ub, None)

        if self.cfg.enable_cuts:
            self.cut_loop()
        self.root_bound = e.objective()
        root_basis = e.get_basis()
        root_x = e.x[: self.n].copy()

        if self.cfg.enable_heuristics and time.time() < self.deadline:
            fractional_dive(self, f.lb, f.ub, root_basis)
            if self.inc_x is None:
                feasibility_pump(self)
            self.solve_lp(f.lb, f.ub, root_basis)

        self.reduced_cost_fixing(self.root_bound, f.lb, f.ub, global_fix=True)
        root.bound = self.root_bound
        root.basis = root_basis
        return self.tree(root, root_x)

    def classify_unbounded_relaxation(self) -> SolverResult:
        """
        LP relaxation unbounded. For rational data (Meyer 1974) the MILP is then unbounded if and only if
        it has an integer feasible point, so solve a zero-objective feasibility probe.
        """
        probe = copy_model_with_bounds(self.model, {}, name="IntegerFeasibilityProbe")
        probe.set_objective({}, sense=self.model.objective.sense)
        remaining = max(0.1, self.deadline - time.time())
        sub = BranchAndBoundSolver(max_nodes=self.cfg.max_nodes, time_limit_seconds=remaining,
                                   enable_cuts=self.cfg.enable_cuts, enable_heuristics=self.cfg.enable_heuristics)
        res = sub.solve(probe, time_limit_seconds=remaining)
        if res.status == SolverStatus.INFEASIBLE:
            return self.finish(SolverStatus.INFEASIBLE, "LP relaxation unbounded; integer feasibility probe proved infeasibility",
                               np.inf, exhausted=True)
        if res.primal_solution:
            result = self.finish(SolverStatus.UNBOUNDED, "LP relaxation unbounded and an integer feasible point exists", -np.inf)
            result.diagnostics["unboundedness_certificate"] = (
                "integer feasible point + unbounded LP relaxation with rational data (Meyer's theorem)"
            )
            result.diagnostics["feasible_point"] = res.primal_solution
            return result
        return self.finish(SolverStatus.INFEASIBLE_OR_UNBOUNDED, "LP relaxation unbounded; feasibility probe inconclusive", -np.inf)

    def cut_loop(self):
        f, e = self.form, self.engine
        stall = 0
        z_prev = e.objective()
        for _ in range(self.cfg.max_cut_rounds):
            if time.time() > self.start + 0.3 * self.time_limit:
                break
            if self.fractional(e.x[: self.n]).size == 0:
                break
            x_unscaled = f.unscale_x(e.x)[: self.n]
            candidates = separate_gmi_cuts(e, f, self.int_col_mask, max_cuts=50)
            candidates += separate_cover_cuts(self.A0, self.rl0, self.ru0, self.cl0, self.cu0,
                                              self.binary_mask, x_unscaled, max_cuts=50)
            chosen = select_cuts(candidates, x_unscaled, self.cl0, self.cu0, max_cuts=50)
            if not chosen:
                break
            backup = (f.A, f.lb.copy(), f.ub.copy(), f.c.copy(), f.col_scale.copy(), f.row_scale.copy(),
                      f.A_orig, list(f.con_names), f.m, e.get_basis())
            G = np.vstack([pi for _, pi, _, _ in chosen])
            lo = np.array([l for _, _, l, _ in chosen])
            hi = np.array([h for _, _, _, h in chosen])
            names = [f"{kind}_cut_{f.m + k}" for k, (kind, _, _, _) in enumerate(chosen)]
            f.add_rows(G, lo, hi, names)
            e.load(f.A, f.c, f.lb, f.ub, keep_basis=True)
            st = self.solve_lp(f.lb, f.ub, None)
            if st != LPStatus.OPTIMAL:
                (f.A, f.lb, f.ub, f.c, f.col_scale, f.row_scale, f.A_orig, f.con_names, f.m, basis) = backup
                e.load(f.A, f.c, f.lb, f.ub, keep_basis=False)
                self.solve_lp(f.lb, f.ub, basis)
                break
            self.cut_rounds += 1
            for kind, _, _, _ in chosen:
                self.cut_counts[kind] += 1
            z = e.objective()
            stall = stall + 1 if z - z_prev <= 1e-4 * max(1.0, abs(z)) else 0
            z_prev = z
            if stall >= 2:
                break

    def reduced_cost_fixing(self, z: float, lb: np.ndarray, ub: np.ndarray, global_fix: bool = False) -> List[Tuple[int, float, float]]:
        """Tighten integer bounds using reduced costs: z + d_j * (x_j - l_j) must stay below the cutoff."""
        cutoff = self.cutoff()
        if not np.isfinite(cutoff):
            return []
        e = self.engine
        _, d = e.duals()
        room = cutoff - z
        changes = []
        for j in self.int_idx:
            st = e.status[j]
            if st == AT_LOWER and d[j] > 1e-9:
                new_ub = lb[j] + np.floor(room / d[j] + 1e-9)
                if new_ub < ub[j] - 0.5:
                    changes.append((int(j), lb[j], float(new_ub)))
            elif st == AT_UPPER and d[j] < -1e-9:
                new_lb = ub[j] - np.floor(room / (-d[j]) + 1e-9)
                if new_lb > lb[j] + 0.5:
                    changes.append((int(j), float(new_lb), ub[j]))
        if global_fix:
            for j, lo, hi in changes:
                self.form.lb[j] = max(self.form.lb[j], lo)
                self.form.ub[j] = min(self.form.ub[j], hi)
            return []
        return changes

    # ------------------------------------------------------------- branching
    def select_branch(self, node: TreeNode, z: float, xs: np.ndarray, frac: np.ndarray, lb, ub, basis):
        vals = xs[frac]
        fr = vals - np.floor(vals)
        if self.cfg.branching_strategy is not None:
            names = [self.form.var_names[j] for j in frac]
            chosen = self.cfg.branching_strategy.select_variable(
                self.model, names, {nm: float(v) for nm, v in zip(names, vals)}, node.depth)
            return "branch", int(frac[names.index(chosen)])

        down, up = self.pc.estimates(frac, fr)
        score = product_score(down, up)
        score = score + 1e-9 * max(float(score.max()), 1.0) * self.static_scores[frac]
        order = np.argsort(-score)
        rel = self.cfg.reliability
        unreliable = [k for k in order if not self.pc.reliable(int(frac[k]), rel.reliability)]
        if not unreliable or node.depth > 12 or time.time() > self.deadline - 0.2 * self.time_limit:
            return "branch", int(frac[order[0]])

        best = None
        for k in unreliable[: rel.strong_candidates]:
            j = int(frac[k])
            gains = []
            for direction in (-1, 1):
                clb, cub = lb.copy(), ub.copy()
                if direction < 0:
                    cub[j] = np.floor(vals[k])
                else:
                    clb[j] = np.ceil(vals[k])
                st = self.solve_lp(clb, cub, basis, max_iter=rel.strong_iterations)
                if st == LPStatus.INFEASIBLE:
                    gain = np.inf
                elif st in (LPStatus.OPTIMAL, LPStatus.ITERATION_LIMIT):
                    gain = max(self.engine.objective() - z, 0.0)
                    if st == LPStatus.OPTIMAL and self.engine.objective() >= self.cutoff():
                        gain = max(gain, self.cutoff() - z)
                    self.pc.update(j, direction, fr[k], gain)
                else:
                    gain = 0.0
                gains.append(gain)
            if np.isinf(gains[0]) and np.isinf(gains[1]):
                return "infeasible", None
            if np.isinf(gains[0]):
                return "tighten", (j, float(np.ceil(vals[k])), float(ub[j]))
            if np.isinf(gains[1]):
                return "tighten", (j, float(lb[j]), float(np.floor(vals[k])))
            s = float(product_score(np.array([gains[0]]), np.array([gains[1]]))[0])
            if best is None or s > best[0]:
                best = (s, j)
        return "branch", best[1]

    # ------------------------------------------------------------------ tree
    def tree(self, root: TreeNode, root_x: np.ndarray) -> SolverResult:
        f, e = self.form, self.engine
        heap: List[TreeNode] = []
        heapq.heappush(heap, root)
        plunge: Optional[TreeNode] = None
        plunge_depth = 0
        termination = "tree exhausted"
        exhausted = False
        pending: Optional[TreeNode] = None
        first = True

        while True:
            if plunge is not None:
                node, plunge = plunge, None
            elif heap:
                node = heapq.heappop(heap)
                plunge_depth = 0
            else:
                exhausted = True
                break
            if node.bound >= self.cutoff():
                continue
            if time.time() > self.deadline:
                termination, pending = "time_limit", node
                break
            if self.nodes >= self.cfg.max_nodes:
                termination, pending = "node_limit", node
                break

            lb, ub = self.node_bounds(node)
            if np.any(lb > ub + 1e-9):
                self.add_trace(node, "INFEASIBLE", np.inf)
                continue
            if first:
                st = self.solve_lp(lb, ub, node.basis)
                first = False
            else:
                st = self.solve_lp(lb, ub, node.basis)
                self.nodes += 1
            node.basis = None

            if st == LPStatus.INFEASIBLE:
                self.add_trace(node, "INFEASIBLE", np.inf)
                continue
            if st != LPStatus.OPTIMAL:
                if st == LPStatus.TIME_LIMIT:
                    termination, pending = "time_limit", node
                    break
                self.proof_complete = False
                continue

            z = e.objective()
            if node.branch_column >= 0 and np.isfinite(node.parent_objective):
                self.pc.update(node.branch_column, node.direction, node.parent_fraction, max(z - node.parent_objective, 0.0))
            if z >= self.cutoff():
                self.add_trace(node, "PRUNED_BOUND", z)
                continue

            xs = e.x[: self.n].copy()
            frac = self.fractional(xs)
            if frac.size == 0:
                x_unscaled = f.unscale_x(e.x)[: self.n]
                if not self.try_incumbent(x_unscaled, "lp_integral"):
                    self.fix_and_complete(x_unscaled, "lp_integral")
                self.add_trace(node, "INTEGER_INCUMBENT", z)
                continue

            basis = e.get_basis()
            rc_changes = self.reduced_cost_fixing(z, lb, ub)
            for j, lo, hi in rc_changes:
                lb[j], ub[j] = lo, hi

            if self.cfg.enable_heuristics and self.nodes > 1 and time.time() < self.deadline:
                if self.nodes % 50 == 0:
                    fractional_dive(self, lb, ub, basis)
                if self.nodes % 400 == 0 and self.inc_x is not None:
                    rins(self, f.unscale_x(e.x)[: self.n], min(0.1 * self.time_limit, self.deadline - time.time()))

            action, info = self.select_branch(node, z, xs, frac, lb, ub, basis)
            if action == "infeasible":
                self.add_trace(node, "INFEASIBLE", np.inf)
                continue
            if action == "tighten":
                j, lo, hi = info
                redo = TreeNode(node_id=node.node_id, parent=node.parent, depth=node.depth,
                                changes=node.changes + tuple(rc_changes) + ((j, lo, hi),), bound=z, basis=basis,
                                parent_objective=node.parent_objective, parent_fraction=node.parent_fraction,
                                branch_column=node.branch_column, direction=0, label=node.label,
                                branch_name=node.branch_name)
                plunge = redo
                continue

            j = info
            v = xs[j]
            frac_part = float(v - np.floor(v))
            name = f.var_names[j]
            self.add_trace(node, "ROOT" if node.parent is None else "BRANCH", z)
            children = []
            for direction in (-1, 1):
                if direction < 0:
                    change = (j, float(lb[j]), float(np.floor(v)))
                    label = f"{name} <= {int(np.floor(v))}"
                else:
                    change = (j, float(np.ceil(v)), float(ub[j]))
                    label = f"{name} >= {int(np.ceil(v))}"
                children.append(TreeNode(
                    node_id=next(self.counter), parent=node, depth=node.depth + 1,
                    changes=tuple(rc_changes) + (change,), bound=z, basis=basis,
                    parent_objective=z, parent_fraction=frac_part, branch_column=j, direction=direction,
                    label=label, branch_name=name,
                ))
            preferred = children[1] if frac_part >= 0.5 else children[0]
            other = children[0] if preferred is children[1] else children[1]
            heapq.heappush(heap, other)
            best_open = heap[0].bound if heap else np.inf
            gap_room = max(1e-6, 0.05 * max(1.0, abs(z)))
            if plunge_depth < 25 and z <= best_open + gap_room:
                plunge = preferred
                plunge_depth += 1
            else:
                heapq.heappush(heap, preferred)

        open_bounds = [nd.bound for nd in heap if nd.bound < self.cutoff()]
        if plunge is not None and plunge.bound < self.cutoff():
            open_bounds.append(plunge.bound)
        if pending is not None and pending.bound < self.cutoff():
            open_bounds.append(pending.bound)
        if not open_bounds:
            exhausted = True
        if exhausted:
            best_bound = self.inc_obj if self.inc_x is not None else np.inf
        else:
            best_bound = min(min(open_bounds), self.inc_obj)
        if not self.proof_complete:
            termination += " (some nodes failed numerically; optimality not proven)"

        if self.inc_x is None:
            if exhausted and self.proof_complete:
                return self.finish(SolverStatus.INFEASIBLE, termination, best_bound, exhausted=True)
            status = SolverStatus.NODE_LIMIT if termination.startswith("node_limit") else SolverStatus.TIME_LIMIT
            if termination.startswith("tree"):
                status = SolverStatus.NUMERICAL_ERROR
            return self.finish(status, termination, best_bound)
        gap = abs(self.inc_obj - best_bound) / max(1.0, abs(self.inc_obj))
        proven = self.proof_complete and (exhausted or gap <= self.cfg.mip_gap_tolerance)
        return self.finish(SolverStatus.OPTIMAL if proven else SolverStatus.FEASIBLE, termination, best_bound, exhausted)

    # ---------------------------------------------------------------- result
    def finish(self, status: SolverStatus, termination: str, best_bound: float, exhausted: bool = False) -> SolverResult:
        has_inc = self.inc_x is not None
        diagnostics = {
            "tree_trace": self.trace,
            "termination": termination,
            "root_lp_bound": self.user(self.root_lp_bound) if hasattr(self, "root_lp_bound") else None,
            "root_bound_after_cuts": self.user(self.root_bound) if np.isfinite(self.root_bound) else None,
            "cuts_applied": int(sum(self.cut_counts.values())),
            "cuts_by_type": dict(self.cut_counts),
            "cut_rounds": self.cut_rounds,
            "heuristic_solutions": dict(self.heuristic_hits),
            "incumbent_history": self.incumbent_history[-20:],
            "lp_iterations": int(self.lp_iterations),
            "branching": "reliability pseudocost (strong-branching initialized)" if self.cfg.branching_strategy is None
            else type(self.cfg.branching_strategy).__name__,
            "has_feasible_point": has_inc,
        }
        result = SolverResult(
            status=status,
            iterations=int(self.lp_iterations),
            runtime_seconds=time.time() - self.start,
            nodes_explored=int(self.nodes),
            diagnostics=diagnostics,
        )
        if has_inc:
            result.primal_solution = {nm: float(v) for nm, v in zip(self.form.var_names, self.inc_x)}
            result.objective_value = self.user(self.inc_obj)
            if np.isfinite(best_bound):
                result.best_bound = self.user(best_bound)
                result.mip_gap = float(abs(self.inc_obj - best_bound) / max(1.0, abs(self.inc_obj)))
                diagnostics["best_bound"] = result.best_bound
        return result


# ============================================================================ MIQP
class _MIQPBranchAndBound:
    """Best-bound branch-and-bound over convex QP relaxations (interior point + polishing)."""

    def __init__(self, cfg: BranchAndBoundSolver, model: OptimizationModel, start: float, deadline: float):
        from sovereign_opt.solvers.qp.interior_point_qp import QPInteriorPointSolver
        self.cfg, self.model, self.start, self.deadline = cfg, model, start, deadline
        self.qp = QPInteriorPointSolver()
        self.names = model.variable_names
        self.int_names = [v for v in self.names if model.variables[v].is_integer]
        self.sign = 1.0 if model.objective.sense == "minimize" else -1.0
        self.trace: List[dict] = []
        self.nodes = 0
        self.iterations = 0

    def internal(self, user_obj: float) -> float:
        return self.sign * (user_obj - self.model.objective.offset)

    def relax(self, bounds: Dict[str, Tuple[float, float]]):
        sub = copy_model_with_bounds(self.model, bounds, name="MIQP_node")
        for v in self.int_names:
            sub.variables[v].var_type = type(sub.variables[v].var_type).CONTINUOUS
        res = self.qp.solve(sub, time_limit_seconds=max(0.05, self.deadline - time.time()))
        self.iterations += res.iterations
        self.nodes += 1
        return res

    def run(self) -> SolverResult:
        cfg = self.cfg
        base = {v: (float(np.ceil(self.model.variables[v].lower_bound - cfg.int_tol)),
                    float(np.floor(self.model.variables[v].upper_bound + cfg.int_tol))) for v in self.int_names}
        inc_obj, inc_sol = np.inf, None
        counter = itertools.count(1)
        heap = [(-np.inf, 0, base, None, "Root", "Root", 0)]
        termination, exhausted = "tree exhausted", True
        proof = True
        while heap:
            bound, nid, bounds, parent, name, label, depth = heapq.heappop(heap)
            if bound >= inc_obj - max(cfg.abs_gap, cfg.mip_gap_tolerance * abs(inc_obj)):
                continue
            if time.time() > self.deadline or self.nodes >= cfg.max_nodes:
                termination = "time_limit" if time.time() > self.deadline else "node_limit"
                heapq.heappush(heap, (bound, nid, bounds, parent, name, label, depth))
                exhausted = False
                break
            if any(lo > hi for lo, hi in bounds.values()):
                continue
            res = self.relax(bounds)
            if res.status == SolverStatus.INFEASIBLE:
                self._trace(nid, parent, depth, np.inf, "INFEASIBLE", name, label)
                continue
            if res.status != SolverStatus.OPTIMAL:
                proof = False
                continue
            z = self.internal(res.objective_value)
            if z >= inc_obj - max(cfg.abs_gap, cfg.mip_gap_tolerance * abs(inc_obj)):
                self._trace(nid, parent, depth, res.objective_value, "PRUNED_BOUND", name, label)
                continue
            sol = res.primal_solution
            frac = [v for v in self.int_names if abs(sol[v] - round(sol[v])) > cfg.int_tol]
            if not frac:
                cand = dict(sol)
                for v in self.int_names:
                    cand[v] = float(round(cand[v]))
                obj = self.internal(self._objective(cand))
                if obj < inc_obj:
                    inc_obj, inc_sol = obj, cand
                self._trace(nid, parent, depth, res.objective_value, "INTEGER_INCUMBENT", name, label)
                continue
            if cfg.enable_heuristics and inc_sol is None:
                fixed = {v: (float(round(sol[v])),) * 2 for v in self.int_names}
                if all(bounds.get(v, (-np.inf, np.inf))[0] <= fixed[v][0] <= bounds.get(v, (-np.inf, np.inf))[1] for v in fixed):
                    hres = self.relax({**bounds, **fixed})
                    if hres.status == SolverStatus.OPTIMAL:
                        hobj = self.internal(hres.objective_value)
                        if hobj < inc_obj:
                            inc_obj, inc_sol = hobj, dict(hres.primal_solution)
            v = max(frac, key=lambda q: 0.5 - abs(sol[q] - np.floor(sol[q]) - 0.5))
            self._trace(nid, parent, depth, res.objective_value, "ROOT" if parent is None else "BRANCH", name, label)
            lo, hi = bounds.get(v, (self.model.variables[v].lower_bound, self.model.variables[v].upper_bound))
            down = dict(bounds)
            down[v] = (lo, float(np.floor(sol[v])))
            up = dict(bounds)
            up[v] = (float(np.ceil(sol[v])), hi)
            heapq.heappush(heap, (z, next(counter), down, nid, v, f"{v} <= {int(np.floor(sol[v]))}", depth + 1))
            heapq.heappush(heap, (z, next(counter), up, nid, v, f"{v} >= {int(np.ceil(sol[v]))}", depth + 1))

        open_bounds = [b for b, *_ in heap if b < inc_obj]
        best_bound = inc_obj if (exhausted or not open_bounds) and inc_sol is not None else (min(open_bounds + [inc_obj]) if open_bounds else inc_obj)
        diagnostics = {"tree_trace": self.trace, "termination": termination, "relaxation": "convex QP interior point",
                       "has_feasible_point": inc_sol is not None}
        result = SolverResult(status=SolverStatus.INFEASIBLE, iterations=self.iterations,
                              runtime_seconds=time.time() - self.start, nodes_explored=self.nodes, diagnostics=diagnostics)
        if inc_sol is None:
            result.status = SolverStatus.INFEASIBLE if (exhausted and proof) else SolverStatus.TIME_LIMIT
            return result
        gap = abs(inc_obj - best_bound) / max(1.0, abs(inc_obj)) if np.isfinite(best_bound) else None
        proven = proof and (exhausted or (gap is not None and gap <= cfg.mip_gap_tolerance))
        result.status = SolverStatus.OPTIMAL if proven else SolverStatus.FEASIBLE
        result.primal_solution = inc_sol
        result.objective_value = self.sign * inc_obj + self.model.objective.offset
        if gap is not None:
            result.best_bound = self.sign * best_bound + self.model.objective.offset
            result.mip_gap = float(gap)
            diagnostics["best_bound"] = result.best_bound
        return result

    def _objective(self, sol: Dict[str, float]) -> float:
        obj = self.model.objective
        val = obj.offset + sum(c * sol.get(v, 0.0) for v, c in obj.linear_coefficients.items())
        val += sum(0.5 * q * sol.get(a, 0.0) * sol.get(b, 0.0) for (a, b), q in obj.quadratic_coefficients.items())
        return val

    def _trace(self, nid, parent, depth, obj, status, name, label):
        if len(self.trace) < TRACE_LIMIT:
            self.trace.append({"node_id": nid, "parent_id": parent, "depth": depth,
                               "lower_bound": float(obj) if np.isfinite(obj) else float("inf"),
                               "status": status, "branch_var": name, "branch_condition": label})
