"""
Primal heuristics for finding integer feasible solutions early.
Implements:
- Simple rounding
- Fractional diving (warm-started dual simplex, one backtrack per level)
- Feasibility pump for binary programs (L1 distance objective, random flips against cycling)
- RINS (Relaxation Induced Neighborhood Search) via a small sub-MIP
"""
from typing import Dict, Optional
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.solvers.lp.simplex_engine import LPStatus


class PrimalHeuristic:
    @classmethod
    def simple_rounding(
        cls, model: OptimizationModel, lp_solution: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """
        Attempts to round fractional variables to integers and tests if constraints are satisfied.
        """
        rounded = dict(lp_solution)
        for v_name, var in model.variables.items():
            if var.is_integer:
                rounded[v_name] = float(round(lp_solution.get(v_name, 0.0)))

        for v_name, var in model.variables.items():
            val = rounded.get(v_name, 0.0)
            if val < var.lower_bound - 1e-6 or val > var.upper_bound + 1e-6:
                return None

        for con in model.constraints.values():
            val = sum(c * rounded.get(v, 0.0) for v, c in con.coefficients.items())
            if val < con.lower_bound - 1e-6 or val > con.upper_bound + 1e-6:
                return None

        return rounded


def copy_model_with_bounds(model: OptimizationModel, overrides: Dict[str, tuple], name: str = "SubMIP") -> OptimizationModel:
    """Deep copy of a model with selected variable bounds replaced."""
    sub = OptimizationModel(name=name)
    for v_name in model.variable_names:
        var = model.variables[v_name]
        lb, ub = overrides.get(v_name, (var.lower_bound, var.upper_bound))
        sub.add_variable(v_name, lower_bound=lb, upper_bound=ub, var_type=var.var_type)
    for c_name in model.constraint_names:
        con = model.constraints[c_name]
        sub.add_constraint(c_name, con.coefficients, sense=con.sense, rhs=con.rhs,
                           lower_bound=con.lower_bound, upper_bound=con.upper_bound)
    sub.set_objective(dict(model.objective.linear_coefficients), sense=model.objective.sense,
                      quadratic_coefficients=dict(model.objective.quadratic_coefficients), offset=model.objective.offset)
    return sub


def fractional_dive(bc, lb: np.ndarray, ub: np.ndarray, basis, max_depth: int = 60, max_lp_iterations: int = 5000) -> bool:
    """Repeatedly fix the least fractional integer variable to its nearest value; backtrack once per level."""
    e = bc.engine
    lb, ub = lb.copy(), ub.copy()
    st = bc.solve_lp(lb, ub, basis, max_iter=max_lp_iterations)
    for _ in range(max_depth):
        if st != LPStatus.OPTIMAL or e.objective() >= bc.cutoff():
            return False
        xs = e.x[: bc.n]
        frac = bc.fractional(xs)
        if frac.size == 0:
            return bc.try_incumbent(bc.form.unscale_x(e.x)[: bc.n], "diving")
        vals = xs[frac]
        fr = vals - np.floor(vals)
        k = int(np.argmin(np.minimum(fr, 1.0 - fr)))
        j = int(frac[k])
        go_up = fr[k] >= 0.5
        saved = e.get_basis()
        moved = False
        for attempt in range(2):
            nlb, nub = lb.copy(), ub.copy()
            if go_up:
                nlb[j] = np.ceil(vals[k])
            else:
                nub[j] = np.floor(vals[k])
            st = bc.solve_lp(nlb, nub, saved, max_iter=max_lp_iterations)
            if st == LPStatus.OPTIMAL and e.objective() < bc.cutoff():
                lb, ub = nlb, nub
                moved = True
                break
            go_up = not go_up
        if not moved:
            return False
    return False


def feasibility_pump(bc, max_rounds: int = 30) -> bool:
    """Feasibility pump for problems whose integer variables are all binary."""
    f, e = bc.form, bc.engine
    ints = bc.int_idx
    if ints.size == 0:
        return False
    bins = ints[(f.lb[ints] >= 0.0) & (f.ub[ints] <= 1.0)]
    if bins.size != ints.size:
        return False
    lb, ub = f.lb.copy(), f.ub.copy()
    original_cost = e.c.copy()
    saved = e.get_basis()
    rng = np.random.default_rng(11)
    target = np.round(e.x[bins])
    seen = set()
    found = False
    try:
        for _ in range(max_rounds):
            cost = np.zeros(e.N)
            cost[bins] = np.where(target > 0.5, -1.0, 1.0)
            e.set_cost(cost)
            st = bc.solve_lp(lb, ub, None, method="auto", max_iter=20000)
            if st != LPStatus.OPTIMAL:
                break
            xb = e.x[bins]
            if np.all(np.abs(xb - np.round(xb)) <= bc.int_tol):
                candidate = bc.form.unscale_x(e.x)[: bc.n]
                e.set_cost(original_cost)
                found = bc.fix_and_complete(candidate, "pump")
                break
            new = np.round(xb)
            if np.array_equal(new, target) or new.tobytes() in seen:
                distance = np.abs(xb - target)
                flips = int(rng.integers(max(1, bins.size // 20), max(2, bins.size // 5) + 1))
                idx = np.argsort(-distance)[:flips]
                new[idx] = 1.0 - new[idx]
            seen.add(new.tobytes())
            target = new
    finally:
        e.set_cost(original_cost)
        e.set_bounds(f.lb, f.ub)
        e.set_basis(*saved)
    return found


def rins(bc, node_solution: np.ndarray, time_budget: float, node_limit: int = 500) -> bool:
    """Fix integer variables on which the LP solution and incumbent agree, then solve the sub-MIP."""
    if bc.inc_x is None or time_budget < 0.05:
        return False
    ints = bc.int_idx
    agree = ints[np.abs(np.round(node_solution[ints]) - bc.inc_x[ints]) < 0.5]
    agree = agree[np.abs(node_solution[agree] - np.round(node_solution[agree])) < 0.1]
    if agree.size < max(1, int(0.3 * ints.size)) or agree.size == ints.size:
        return False
    names = bc.form.var_names
    overrides = {names[j]: (float(bc.inc_x[j]), float(bc.inc_x[j])) for j in agree}
    sub_model = copy_model_with_bounds(bc.model, overrides, name="RINS")
    from sovereign_opt.solvers.milp.branch_bound import BranchAndBoundSolver

    sub = BranchAndBoundSolver(max_nodes=node_limit, time_limit_seconds=time_budget, mip_gap_tolerance=1e-6,
                               enable_cuts=False, enable_heuristics=False)
    res = sub.solve(sub_model, time_limit_seconds=time_budget)
    if not res.primal_solution:
        return False
    x = np.array([res.primal_solution.get(v, 0.0) for v in names])
    return bc.try_incumbent(x, "rins")
