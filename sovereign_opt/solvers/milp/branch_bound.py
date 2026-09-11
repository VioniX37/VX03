"""
Sovereign Branch-and-Bound solver for Mixed-Integer Linear Programming (MILP).
Implemented from mathematical foundations:
- LP relaxation solves at each node
- Priority queue best-bound search
- Infeasibility, bound, and integrality fathoming
- Primal rounding heuristic
- MIP gap tracking
"""
import heapq
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import Variable, VariableType
from sovereign_opt.solvers.base import SolverBase, SolverResult, SolverStatus
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver
from sovereign_opt.solvers.milp.node import Node
from sovereign_opt.solvers.milp.branching import BranchingStrategy, MLGuidedBranching, MostFractionalBranching
from sovereign_opt.solvers.milp.heuristics import PrimalHeuristic


class BranchAndBoundSolver(SolverBase):
    """
    Sovereign Branch-and-Bound MILP Solver.
    """

    def __init__(
        self,
        max_nodes: int = 1000,
        time_limit_seconds: float = 60.0,
        mip_gap_tolerance: float = 1e-4,
        integrality_tolerance: float = 1e-5,
        branching_strategy: Optional[BranchingStrategy] = None,
    ):
        super().__init__(name="BranchAndBound")
        self.max_nodes = max_nodes
        self.time_limit = time_limit_seconds
        self.mip_gap_tolerance = mip_gap_tolerance
        self.int_tol = integrality_tolerance
        self.branching_strategy = branching_strategy or MLGuidedBranching()
        self.lp_solver = RevisedSimplexSolver(max_iterations=2000, tolerance=1e-7)

    def solve(self, model: OptimizationModel, **kwargs) -> SolverResult:
        start_time = time.time()
        time_limit = kwargs.get("time_limit_seconds", self.time_limit)

        # Check integer variables
        integer_vars = [name for name, var in model.variables.items() if var.is_integer]
        if not integer_vars:
            # Pure LP, dispatch directly to LP solver
            return self.lp_solver.solve(model, time_limit_seconds=time_limit)

        is_minimize = model.objective.sense == "minimize"
        incumbent_obj = float("inf") if is_minimize else float("-inf")
        incumbent_solution: Dict[str, float] = {}

        # Build root node
        initial_bounds = {name: (var.lower_bound, var.upper_bound) for name, var in model.variables.items()}
        root_node = Node(node_id=0, parent_id=None, depth=0, bounds=initial_bounds)

        # Solve root LP relaxation
        root_res = self._solve_relaxation(model, root_node.bounds, time_limit)
        if not root_res.is_feasible:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                objective_value=None,
                iterations=root_res.iterations,
                runtime_seconds=time.time() - start_time,
                nodes_explored=1,
            )

        root_node.lower_bound = root_res.objective_value
        root_node.solution = root_res.primal_solution

        # Check if root solution is already integer feasible
        fractional = self._get_fractional_vars(model, root_node.solution, integer_vars)
        if not fractional:
            return SolverResult(
                status=SolverStatus.OPTIMAL,
                objective_value=root_node.lower_bound,
                primal_solution=root_node.solution,
                iterations=root_res.iterations,
                runtime_seconds=time.time() - start_time,
                nodes_explored=1,
                mip_gap=0.0,
            )

        # Root heuristic: try fast rounding
        rounded_sol = PrimalHeuristic.simple_rounding(model, root_node.solution)
        if rounded_sol is not None:
            r_obj = self._compute_objective(model, rounded_sol)
            if (is_minimize and r_obj < incumbent_obj) or (not is_minimize and r_obj > incumbent_obj):
                incumbent_obj = r_obj
                incumbent_solution = rounded_sol

        # Priority queue for Best-Bound Search (min-heap)
        heap: List[Tuple[float, int, Node]] = []
        node_counter = 1
        heapq.heappush(heap, (root_node.lower_bound if is_minimize else -root_node.lower_bound, 0, root_node))

        tree_trace = [
            {
                "node_id": 0,
                "parent_id": None,
                "depth": 0,
                "lower_bound": float(root_node.lower_bound),
                "status": "ROOT",
                "branch_var": "Root",
                "branch_condition": "Relaxation",
            }
        ]

        total_iterations = root_res.iterations
        nodes_explored = 1
        best_bound = root_node.lower_bound

        while heap and nodes_explored < self.max_nodes:
            if time.time() - start_time > time_limit:
                break

            bound_val, _, current_node = heapq.heappop(heap)
            current_lb = bound_val if is_minimize else -bound_val

            # Pruning by bound
            if is_minimize and current_lb >= incumbent_obj - self.mip_gap_tolerance:
                continue
            if not is_minimize and current_lb <= incumbent_obj + self.mip_gap_tolerance:
                continue

            # Update global best bound
            best_bound = current_lb

            # Check MIP gap
            if incumbent_solution:
                gap = abs(incumbent_obj - best_bound) / (abs(incumbent_obj) + 1e-10)
                if gap <= self.mip_gap_tolerance:
                    break

            # Find fractional variables
            fractional = self._get_fractional_vars(model, current_node.solution, integer_vars)
            if not fractional:
                if (is_minimize and current_lb < incumbent_obj) or (not is_minimize and current_lb > incumbent_obj):
                    incumbent_obj = current_lb
                    incumbent_solution = current_node.solution
                continue

            # Select branching variable
            branch_var = self.branching_strategy.select_variable(
                model, fractional, current_node.solution, current_node.depth
            )
            branch_val = current_node.solution[branch_var]

            # Left child: x_var <= floor(val)
            left_bounds = dict(current_node.bounds)
            left_bounds[branch_var] = (left_bounds[branch_var][0], float(np.floor(branch_val)))

            # Right child: x_var >= ceil(val)
            right_bounds = dict(current_node.bounds)
            right_bounds[branch_var] = (float(np.ceil(branch_val)), right_bounds[branch_var][1])

            # Process children
            for child_bounds in (left_bounds, right_bounds):
                lb, ub = child_bounds[branch_var]
                if lb > ub + 1e-9:
                    continue

                node_counter += 1
                child_node = Node(
                    node_id=node_counter,
                    parent_id=current_node.node_id,
                    depth=current_node.depth + 1,
                    bounds=child_bounds,
                )

                cond_str = f"{branch_var} <= {int(np.floor(branch_val))}" if child_bounds == left_bounds else f"{branch_var} >= {int(np.ceil(branch_val))}"

                child_res = self._solve_relaxation(model, child_bounds, time_limit - (time.time() - start_time))
                total_iterations += child_res.iterations
                nodes_explored += 1

                if not child_res.is_feasible:
                    if len(tree_trace) < 50:
                        tree_trace.append({
                            "node_id": node_counter,
                            "parent_id": current_node.node_id,
                            "depth": child_node.depth,
                            "lower_bound": float("inf"),
                            "status": "INFEASIBLE",
                            "branch_var": branch_var,
                            "branch_condition": cond_str,
                        })
                    continue

                child_lb = child_res.objective_value
                child_node.lower_bound = child_lb
                child_node.solution = child_res.primal_solution

                # Bound pruning
                if is_minimize and child_lb >= incumbent_obj - self.mip_gap_tolerance:
                    if len(tree_trace) < 50:
                        tree_trace.append({
                            "node_id": node_counter,
                            "parent_id": current_node.node_id,
                            "depth": child_node.depth,
                            "lower_bound": float(child_lb),
                            "status": "PRUNED_BOUND",
                            "branch_var": branch_var,
                            "branch_condition": cond_str,
                        })
                    continue

                # Check integrality
                child_frac = self._get_fractional_vars(model, child_node.solution, integer_vars)
                if not child_frac:
                    if (is_minimize and child_lb < incumbent_obj) or (not is_minimize and child_lb > incumbent_obj):
                        incumbent_obj = child_lb
                        incumbent_solution = child_node.solution
                    if len(tree_trace) < 50:
                        tree_trace.append({
                            "node_id": node_counter,
                            "parent_id": current_node.node_id,
                            "depth": child_node.depth,
                            "lower_bound": float(child_lb),
                            "status": "INTEGER_INCUMBENT",
                            "branch_var": branch_var,
                            "branch_condition": cond_str,
                        })
                else:
                    sort_key = child_lb if is_minimize else -child_lb
                    heapq.heappush(heap, (sort_key, child_node.node_id, child_node))
                    if len(tree_trace) < 50:
                        tree_trace.append({
                            "node_id": node_counter,
                            "parent_id": current_node.node_id,
                            "depth": child_node.depth,
                            "lower_bound": float(child_lb),
                            "status": "BRANCH",
                            "branch_var": branch_var,
                            "branch_condition": cond_str,
                        })

        # Final MIP Gap
        if incumbent_solution:
            final_gap = abs(incumbent_obj - best_bound) / (abs(incumbent_obj) + 1e-10)
            status = SolverStatus.OPTIMAL if final_gap <= self.mip_gap_tolerance else SolverStatus.FEASIBLE
            return SolverResult(
                status=status,
                objective_value=float(incumbent_obj),
                primal_solution=incumbent_solution,
                iterations=total_iterations,
                runtime_seconds=time.time() - start_time,
                nodes_explored=nodes_explored,
                mip_gap=float(final_gap),
                diagnostics={"best_bound": float(best_bound), "tree_trace": tree_trace},
            )
        else:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                objective_value=None,
                iterations=total_iterations,
                runtime_seconds=time.time() - start_time,
                nodes_explored=nodes_explored,
            )

    def _solve_relaxation(
        self, model: OptimizationModel, bounds: Dict[str, Tuple[float, float]], time_limit: float
    ) -> SolverResult:
        """Create a temporary relaxation model with updated bounds and solve."""
        rel_model = OptimizationModel(name="Relaxation")
        for name, var in model.variables.items():
            lb, ub = bounds.get(name, (var.lower_bound, var.upper_bound))
            # Relax integer to continuous
            rel_model.add_variable(name=name, lower_bound=lb, upper_bound=ub, var_type=VariableType.CONTINUOUS)
        for name, con in model.constraints.items():
            rel_model.add_constraint(
                name=name,
                coefficients=con.coefficients,
                sense=con.sense,
                rhs=con.rhs,
                lower_bound=con.lower_bound,
                upper_bound=con.upper_bound,
            )
        rel_model.set_objective(
            linear_coefficients=model.objective.linear_coefficients,
            sense=model.objective.sense,
            offset=model.objective.offset,
        )
        return self.lp_solver.solve(rel_model, time_limit_seconds=max(0.1, time_limit))

    def _get_fractional_vars(
        self, model: OptimizationModel, solution: Dict[str, float], integer_vars: List[str]
    ) -> List[str]:
        fractional = []
        for v in integer_vars:
            val = solution.get(v, 0.0)
            if abs(val - round(val)) > self.int_tol:
                fractional.append(v)
        return fractional

    def _compute_objective(self, model: OptimizationModel, solution: Dict[str, float]) -> float:
        obj = model.objective.offset
        multiplier = 1.0 if model.objective.sense == "minimize" else -1.0
        for v, coeff in model.objective.linear_coefficients.items():
            obj += coeff * solution.get(v, 0.0) * multiplier
        return obj
