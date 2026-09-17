"""
Free-format MPS writer (LP / MILP / QP via QUADOBJ), the inverse of MPSParser.
Used to export models so any other solver can read exactly the same problem.
"""
from typing import List
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.variable import VariableType


def _num(v: float) -> str:
    return repr(float(v))


def write_mps(model: OptimizationModel) -> str:
    out: List[str] = [f"NAME {model.name.replace(' ', '_')}"]
    if model.objective.sense != "minimize":
        out += ["OBJSENSE", "    MAX"]
    out.append("ROWS")
    out.append(" N  OBJ")
    senses = {}
    for cn in model.constraint_names:
        con = model.constraints[cn]
        s = con.sense
        if s == ConstraintSense.RANGE:
            t = "G"
        else:
            t = {ConstraintSense.LE: "L", ConstraintSense.GE: "G", ConstraintSense.EQ: "E"}[s]
        senses[cn] = t
        out.append(f" {t}  {cn}")

    cols = {v: [] for v in model.variable_names}
    for cn in model.constraint_names:
        for v, a in model.constraints[cn].coefficients.items():
            if a != 0.0:
                cols[v].append((cn, a))
    obj = model.objective.linear_coefficients
    out.append("COLUMNS")
    in_int = False
    for v in model.variable_names:
        is_int = model.variables[v].var_type != VariableType.CONTINUOUS
        if is_int and not in_int:
            out.append("    MARKER  'MARKER'  'INTORG'")
            in_int = True
        elif not is_int and in_int:
            out.append("    MARKER  'MARKER'  'INTEND'")
            in_int = False
        entries = ([("OBJ", obj[v])] if obj.get(v, 0.0) != 0.0 else []) + cols[v]
        if not entries:
            entries = [("OBJ", 0.0)]
        for r, a in entries:
            out.append(f"    {v}  {r}  {_num(a)}")
    if in_int:
        out.append("    MARKER  'MARKER'  'INTEND'")

    out.append("RHS")
    if model.objective.offset:
        out.append(f"    RHS  OBJ  {_num(-model.objective.offset)}")
    ranges = []
    for cn in model.constraint_names:
        con = model.constraints[cn]
        if con.sense == ConstraintSense.RANGE:
            lo, hi = con.lower_bound, con.upper_bound
            if lo != 0.0:
                out.append(f"    RHS  {cn}  {_num(lo)}")
            ranges.append((cn, hi - lo))
        elif con.rhs != 0.0:
            out.append(f"    RHS  {cn}  {_num(con.rhs)}")
    if ranges:
        out.append("RANGES")
        out += [f"    RNG  {cn}  {_num(r)}" for cn, r in ranges]

    out.append("BOUNDS")
    for v in model.variable_names:
        var = model.variables[v]
        lb, ub = var.lower_bound, var.upper_bound
        if np.isneginf(lb) and np.isposinf(ub):
            out.append(f" FR BND  {v}")
            continue
        if lb == ub and np.isfinite(lb):
            out.append(f" FX BND  {v}  {_num(lb)}")
            continue
        if np.isneginf(lb):
            out.append(f" MI BND  {v}")
        elif lb != 0.0 or var.var_type != VariableType.CONTINUOUS:
            out.append(f" LO BND  {v}  {_num(lb)}")
        if np.isfinite(ub):
            out.append(f" UP BND  {v}  {_num(ub)}")
        elif var.var_type != VariableType.CONTINUOUS:
            out.append(f" PL BND  {v}")

    quad = model.objective.quadratic_coefficients
    if quad:
        out.append("QUADOBJ")
        for (a, b), q in quad.items():
            # model convention: q on (i,j), i != j contributes 1/2 q x_i x_j -> QUADOBJ lists q/2 once
            out.append(f"    {a}  {b}  {_num(q if a == b else q / 2.0)}")
    out.append("ENDATA")
    return "\n".join(out) + "\n"
