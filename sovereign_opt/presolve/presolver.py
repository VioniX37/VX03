"""
Presolve reduction engine for simplifying optimization problems prior to solving.
Implements Stage 4 Presolve operations.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import Variable, VariableType
from sovereign_opt.model.constraint import Constraint, ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.presolve.postsolve import PostsolveMapper, PresolveStep, ReductionType


class PresolveInfeasibleError(Exception):
    """Raised when presolve detects that the model is mathematically infeasible."""
    pass


class PresolveUnboundedError(Exception):
    """Raised when presolve detects that the model is unbounded."""
    pass


@dataclass
class PresolveStats:
    original_vars: int
    original_cons: int
    original_nnz: int
    presolved_vars: int
    presolved_cons: int
    presolved_nnz: int
    fixed_vars_count: int
    empty_rows_count: int
    empty_cols_count: int
    singleton_rows_count: int
    redundant_rows_count: int

    @property
    def var_reduction_pct(self) -> float:
        return ((self.original_vars - self.presolved_vars) / self.original_vars * 100.0) if self.original_vars > 0 else 0.0

    @property
    def con_reduction_pct(self) -> float:
        return ((self.original_cons - self.presolved_cons) / self.original_cons * 100.0) if self.original_cons > 0 else 0.0

    @property
    def nnz_reduction_pct(self) -> float:
        return ((self.original_nnz - self.presolved_nnz) / self.original_nnz * 100.0) if self.original_nnz > 0 else 0.0


class Presolver:
    """
    Executes structural and bound reductions on an OptimizationModel.
    Returns the reduced model and a PostsolveMapper.
    """

    def __init__(self, max_passes: int = 5, tolerance: float = 1e-9):
        self.max_passes = max_passes
        self.tolerance = tolerance

    def presolve(self, model: OptimizationModel) -> Tuple[OptimizationModel, PostsolveMapper, PresolveStats]:
        orig_meta = model.get_metadata()
        mapper = PostsolveMapper(model.variable_names, model.constraint_names)

        # Working state
        var_lbs = {name: var.lower_bound for name, var in model.variables.items()}
        var_ubs = {name: var.upper_bound for name, var in model.variables.items()}
        var_types = {name: var.var_type for name, var in model.variables.items()}

        active_cons = {
            c_name: {
                "coeffs": dict(con.coefficients),
                "lb": con.lower_bound,
                "ub": con.upper_bound,
                "sense": con.sense,
            }
            for c_name, con in model.constraints.items()
        }

        obj_coeffs = dict(model.objective.linear_coefficients)
        obj_offset = model.objective.offset

        fixed_vars_count = 0
        empty_rows_count = 0
        empty_cols_count = 0
        singleton_rows_count = 0
        redundant_rows_count = 0

        for _ in range(self.max_passes):
            progress = False

            # 1. Fixed variables: lb == ub
            for v_name in list(var_lbs.keys()):
                lb = var_lbs[v_name]
                ub = var_ubs[v_name]
                if abs(lb - ub) <= self.tolerance:
                    fixed_val = (lb + ub) / 2.0
                    mapper.record_step(
                        PresolveStep(ReductionType.FIXED_VARIABLE, {"var_name": v_name, "value": fixed_val})
                    )
                    fixed_vars_count += 1
                    progress = True

                    # Substitute into constraints
                    for c_name, c_data in active_cons.items():
                        if v_name in c_data["coeffs"]:
                            a_ij = c_data["coeffs"].pop(v_name)
                            c_data["lb"] -= a_ij * fixed_val
                            c_data["ub"] -= a_ij * fixed_val

                    # Substitute into objective
                    if v_name in obj_coeffs:
                        c_j = obj_coeffs.pop(v_name)
                        obj_offset += c_j * fixed_val

                    del var_lbs[v_name]
                    del var_ubs[v_name]
                    del var_types[v_name]

            # 2. Empty rows: coeffs is empty
            for c_name in list(active_cons.keys()):
                c_data = active_cons[c_name]
                if not c_data["coeffs"]:
                    # Check feasibility: 0 in [lb, ub]
                    if c_data["lb"] > self.tolerance or c_data["ub"] < -self.tolerance:
                        raise PresolveInfeasibleError(f"Empty row '{c_name}' is infeasible: 0 not in [{c_data['lb']}, {c_data['ub']}].")
                    mapper.record_step(PresolveStep(ReductionType.EMPTY_ROW, {"con_name": c_name}))
                    empty_rows_count += 1
                    del active_cons[c_name]
                    progress = True

            # 3. Singleton rows: exactly 1 variable in constraint: a_ij * x_j in [lb, ub]
            for c_name in list(active_cons.keys()):
                c_data = active_cons[c_name]
                if len(c_data["coeffs"]) == 1:
                    v_name, a_ij = next(iter(c_data["coeffs"].items()))
                    if v_name not in var_lbs:
                        continue

                    # a_ij * x_j in [lb, ub]
                    if a_ij > 0:
                        new_lb = c_data["lb"] / a_ij if not np.isneginf(c_data["lb"]) else float("-inf")
                        new_ub = c_data["ub"] / a_ij if not np.isposinf(c_data["ub"]) else float("inf")
                    else:
                        new_lb = c_data["ub"] / a_ij if not np.isposinf(c_data["ub"]) else float("-inf")
                        new_ub = c_data["lb"] / a_ij if not np.isneginf(c_data["lb"]) else float("inf")

                    # Tighten bounds
                    old_lb = var_lbs[v_name]
                    old_ub = var_ubs[v_name]
                    tight_lb = max(old_lb, new_lb)
                    tight_ub = min(old_ub, new_ub)

                    if tight_lb > tight_ub + self.tolerance:
                        raise PresolveInfeasibleError(f"Singleton row '{c_name}' implies conflicting bounds on '{v_name}'.")

                    if tight_lb > old_lb + self.tolerance or tight_ub < old_ub - self.tolerance:
                        var_lbs[v_name] = tight_lb
                        var_ubs[v_name] = tight_ub
                        progress = True

                    mapper.record_step(PresolveStep(ReductionType.SINGLETON_ROW, {"con_name": c_name, "var_name": v_name}))
                    singleton_rows_count += 1
                    del active_cons[c_name]
                    progress = True

            # 4. Empty columns: variable does not appear in any active constraint
            active_vars_in_cons = {v for c_data in active_cons.values() for v in c_data["coeffs"].keys()}
            for v_name in list(var_lbs.keys()):
                if v_name not in active_vars_in_cons:
                    c_j = obj_coeffs.get(v_name, 0.0)
                    lb = var_lbs[v_name]
                    ub = var_ubs[v_name]

                    # Minimize c_j * x_j
                    if c_j > self.tolerance:
                        if np.isneginf(lb):
                            raise PresolveUnboundedError(f"Variable '{v_name}' has negative objective and no lower bound.")
                        fixed_val = lb
                    elif c_j < -self.tolerance:
                        if np.isposinf(ub):
                            raise PresolveUnboundedError(f"Variable '{v_name}' has positive objective and no upper bound.")
                        fixed_val = ub
                    else:
                        # c_j == 0, fix to lower bound or 0
                        fixed_val = lb if not np.isneginf(lb) else (0.0 if ub >= 0 else ub)

                    mapper.record_step(
                        PresolveStep(ReductionType.EMPTY_COL, {"var_name": v_name, "fixed_value": fixed_val})
                    )
                    empty_cols_count += 1
                    obj_offset += c_j * fixed_val
                    obj_coeffs.pop(v_name, None)
                    del var_lbs[v_name]
                    del var_ubs[v_name]
                    del var_types[v_name]
                    progress = True

            if not progress:
                break

        # Reconstruct Presolved OptimizationModel
        p_model = OptimizationModel(name=f"{model.name}_presolved")

        for v_name, lb in var_lbs.items():
            p_model.add_variable(name=v_name, lower_bound=lb, upper_bound=var_ubs[v_name], var_type=var_types[v_name])

        for c_name, c_data in active_cons.items():
            p_model.add_constraint(
                name=c_name,
                coefficients=c_data["coeffs"],
                sense=c_data["sense"],
                lower_bound=c_data["lb"],
                upper_bound=c_data["ub"],
            )

        p_model.set_objective(
            linear_coefficients=obj_coeffs,
            sense=model.objective.sense,
            quadratic_coefficients=model.objective.quadratic_coefficients,
            offset=obj_offset,
        )

        p_meta = p_model.get_metadata()
        stats = PresolveStats(
            original_vars=orig_meta.num_variables,
            original_cons=orig_meta.num_constraints,
            original_nnz=orig_meta.num_nonzeros,
            presolved_vars=p_meta.num_variables,
            presolved_cons=p_meta.num_constraints,
            presolved_nnz=p_meta.num_nonzeros,
            fixed_vars_count=fixed_vars_count,
            empty_rows_count=empty_rows_count,
            empty_cols_count=empty_cols_count,
            singleton_rows_count=singleton_rows_count,
            redundant_rows_count=redundant_rows_count,
        )

        return p_model, mapper, stats
