"""
Presolve reduction engine (v2) for simplifying optimization problems prior to solving.

All reductions work on the minimize-normalized objective (v1 used the raw user objective,
which fixed variables at the wrong bound for maximization problems).

Reductions (repeated until no progress):
- Fixed variables (including quadratic-term substitution)
- Empty rows (with infeasibility detection)
- Singleton rows -> variable bounds (integer-aware rounding)
- Empty columns (sense-aware)
- Activity-based analysis: infeasible rows, redundant rows, forcing rows,
  implied integer bound tightening
- Dominated columns / dual fixing
- Doubleton equation aggregation
- Parallel rows
- Duplicate continuous columns
- Coefficient tightening on binary variables (MILP)
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense
from sovereign_opt.presolve.postsolve import PostsolveMapper, PresolveStep, ReductionType

INF = float("inf")


class PresolveInfeasibleError(Exception):
    """Raised when presolve detects that the model is mathematically infeasible."""
    pass


class PresolveUnboundedError(Exception):
    """Raised when presolve detects an unbounded direction (the model is unbounded or infeasible)."""
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
    forcing_rows_count: int = 0
    doubleton_eliminations: int = 0
    dominated_cols_count: int = 0
    parallel_rows_count: int = 0
    duplicate_cols_count: int = 0
    tightened_bounds_count: int = 0
    coefficient_tightenings: int = 0
    passes: int = 0

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
    Returns the reduced model, a PostsolveMapper, and statistics.
    """

    def __init__(
        self,
        max_passes: int = 10,
        tolerance: float = 1e-9,
        enable_dual_fixing: bool = True,
        enable_doubleton: bool = True,
        enable_parallel_rows: bool = True,
        enable_duplicate_columns: bool = True,
        enable_activity_analysis: bool = True,
        enable_coefficient_tightening: bool = True,
    ):
        self.max_passes = max_passes
        self.tolerance = tolerance
        self.enable_dual_fixing = enable_dual_fixing
        self.enable_doubleton = enable_doubleton
        self.enable_parallel_rows = enable_parallel_rows
        self.enable_duplicate_columns = enable_duplicate_columns
        self.enable_activity_analysis = enable_activity_analysis
        self.enable_coefficient_tightening = enable_coefficient_tightening

    def presolve(self, model: OptimizationModel) -> Tuple[OptimizationModel, PostsolveMapper, PresolveStats]:
        state = _PresolveState(model, self)
        passes = 0
        for _ in range(self.max_passes):
            passes += 1
            progress = False
            progress |= state.fixed_variables()
            progress |= state.empty_rows()
            progress |= state.singleton_rows()
            progress |= state.empty_columns()
            if self.enable_activity_analysis:
                progress |= state.activity_analysis()
            if self.enable_dual_fixing:
                progress |= state.dominated_columns()
            if self.enable_doubleton:
                progress |= state.doubleton_equations()
            if self.enable_parallel_rows:
                progress |= state.parallel_rows()
            if self.enable_duplicate_columns:
                progress |= state.duplicate_columns()
            if not progress:
                break
        if self.enable_coefficient_tightening:
            state.coefficient_tightening()
        state.fixed_variables()
        state.empty_rows()
        p_model = state.build_model()
        state.stats["passes"] = passes

        orig_meta = model.get_metadata()
        p_meta = p_model.get_metadata()
        s = state.stats
        stats = PresolveStats(
            original_vars=orig_meta.num_variables,
            original_cons=orig_meta.num_constraints,
            original_nnz=orig_meta.num_nonzeros,
            presolved_vars=p_meta.num_variables,
            presolved_cons=p_meta.num_constraints,
            presolved_nnz=p_meta.num_nonzeros,
            fixed_vars_count=s["fixed"],
            empty_rows_count=s["empty_rows"],
            empty_cols_count=s["empty_cols"],
            singleton_rows_count=s["singleton_rows"],
            redundant_rows_count=s["redundant_rows"],
            forcing_rows_count=s["forcing_rows"],
            doubleton_eliminations=s["doubleton"],
            dominated_cols_count=s["dominated"],
            parallel_rows_count=s["parallel_rows"],
            duplicate_cols_count=s["duplicate_cols"],
            tightened_bounds_count=s["tightened"],
            coefficient_tightenings=s["coef_tightening"],
            passes=passes,
        )
        return p_model, state.mapper, stats


class _PresolveState:
    def __init__(self, model: OptimizationModel, cfg: Presolver):
        self.model = model
        self.tol = cfg.tolerance
        self.mapper = PostsolveMapper(model.variable_names, model.constraint_names)
        self.sign = 1.0 if model.objective.sense == ObjectiveSense.MINIMIZE else -1.0
        self.var_order = model.variable_names
        self.con_order = model.constraint_names
        self.lb: Dict[str, float] = {v: model.variables[v].lower_bound for v in self.var_order}
        self.ub: Dict[str, float] = {v: model.variables[v].upper_bound for v in self.var_order}
        self.vtype: Dict[str, VariableType] = {v: model.variables[v].var_type for v in self.var_order}
        self.rows: Dict[str, dict] = {}
        self.col_rows: Dict[str, Set[str]] = {v: set() for v in self.var_order}
        for c in self.con_order:
            con = model.constraints[c]
            self.rows[c] = {"coeffs": dict(con.coefficients), "lb": con.lower_bound, "ub": con.upper_bound}
            for v in con.coefficients:
                self.col_rows[v].add(c)
        self.cost: Dict[str, float] = dict(model.objective.linear_coefficients)
        self.offset = float(model.objective.offset)
        self.quad: Dict[Tuple[str, str], float] = dict(model.objective.quadratic_coefficients)
        # variable -> the quadratic terms it appears in, so removing a variable touches only those
        self.quad_of: Dict[str, Set[Tuple[str, str]]] = {}
        for key in self.quad:
            for v in key:
                self.quad_of.setdefault(v, set()).add(key)
        self.stats = {k: 0 for k in ("fixed", "empty_rows", "empty_cols", "singleton_rows", "redundant_rows", "forcing_rows",
                                     "doubleton", "dominated", "parallel_rows", "duplicate_cols", "tightened", "coef_tightening")}
        for v in self.var_order:
            if self.is_int(v):
                self.lb[v] = float(np.ceil(self.lb[v] - 1e-9)) if math.isfinite(self.lb[v]) else self.lb[v]
                self.ub[v] = float(np.floor(self.ub[v] + 1e-9)) if math.isfinite(self.ub[v]) else self.ub[v]

    # ------------------------------------------------------------ helpers
    def is_int(self, v: str) -> bool:
        return self.vtype[v] in (VariableType.INTEGER, VariableType.BINARY)

    def quad_vars(self) -> Set[str]:
        return {v for v, keys in self.quad_of.items() if keys}

    def ctil(self, v: str) -> float:
        return self.sign * self.cost.get(v, 0.0)

    def rtol(self, value: float) -> float:
        return self.tol * max(1.0, abs(value) if math.isfinite(value) else 1.0)

    def check_bounds(self, v: str):
        if self.lb[v] > self.ub[v] + self.rtol(self.lb[v]):
            raise PresolveInfeasibleError(f"Variable '{v}' has empty domain [{self.lb[v]}, {self.ub[v]}] after presolve.")

    def remove_var(self, v: str, value: float, kind: ReductionType):
        for r in list(self.col_rows[v]):
            row = self.rows[r]
            a = row["coeffs"].pop(v)
            row["lb"] -= a * value
            row["ub"] -= a * value
        del self.col_rows[v]
        c = self.cost.pop(v, 0.0)
        self.offset += c * value
        for key in self.quad_of.pop(v, ()):
            q = self.quad.pop(key)
            a, b = key
            for w in key:
                if w != v:
                    self.quad_of[w].discard(key)
            if a == b:
                self.offset += 0.5 * q * value * value
            else:
                other = b if a == v else a
                self.cost[other] = self.cost.get(other, 0.0) + 0.5 * q * value
        del self.lb[v], self.ub[v], self.vtype[v]
        self.mapper.record_step(PresolveStep(kind, {"var_name": v, "value": float(value)}))

    def remove_row(self, r: str, kind: ReductionType, extra: dict = None):
        for v in self.rows[r]["coeffs"]:
            self.col_rows[v].discard(r)
        del self.rows[r]
        details = {"con_name": r}
        if extra:
            details.update(extra)
        self.mapper.record_step(PresolveStep(kind, details))

    def activity(self, row: dict):
        mn = mx = 0.0
        inf_mn = inf_mx = 0
        for v, a in row["coeffs"].items():
            lo, hi = (self.lb[v], self.ub[v]) if a > 0 else (self.ub[v], self.lb[v])
            if math.isfinite(lo):
                mn += a * lo
            else:
                inf_mn += 1
            if math.isfinite(hi):
                mx += a * hi
            else:
                inf_mx += 1
        return mn, mx, inf_mn, inf_mx

    # --------------------------------------------------------- reductions
    def fixed_variables(self) -> bool:
        progress = False
        for v in list(self.lb.keys()):
            self.check_bounds(v)
            lo, hi = self.lb[v], self.ub[v]
            if math.isfinite(lo) and hi - lo <= self.rtol(lo):
                value = float(np.round(lo)) if self.is_int(v) else lo
                self.remove_var(v, value, ReductionType.FIXED_VARIABLE)
                self.stats["fixed"] += 1
                progress = True
        return progress

    def empty_rows(self) -> bool:
        progress = False
        for r in list(self.rows.keys()):
            row = self.rows[r]
            if not row["coeffs"]:
                if row["lb"] > self.rtol(row["lb"]) or row["ub"] < -self.rtol(row["ub"]):
                    raise PresolveInfeasibleError(f"Empty row '{r}' is infeasible: 0 not in [{row['lb']}, {row['ub']}].")
                self.remove_row(r, ReductionType.EMPTY_ROW)
                self.stats["empty_rows"] += 1
                progress = True
        return progress

    def singleton_rows(self) -> bool:
        progress = False
        for r in list(self.rows.keys()):
            row = self.rows[r]
            if len(row["coeffs"]) != 1:
                continue
            v, a = next(iter(row["coeffs"].items()))
            if a > 0:
                new_lb, new_ub = row["lb"] / a, row["ub"] / a
            else:
                new_lb, new_ub = row["ub"] / a, row["lb"] / a
            if self.is_int(v):
                new_lb = float(np.ceil(new_lb - 1e-9)) if math.isfinite(new_lb) else new_lb
                new_ub = float(np.floor(new_ub + 1e-9)) if math.isfinite(new_ub) else new_ub
            if new_lb > self.lb[v]:
                self.lb[v] = new_lb
            if new_ub < self.ub[v]:
                self.ub[v] = new_ub
            self.check_bounds(v)
            self.remove_row(r, ReductionType.SINGLETON_ROW, {"var_name": v})
            self.stats["singleton_rows"] += 1
            progress = True
        return progress

    def empty_columns(self) -> bool:
        progress = False
        qv = self.quad_vars()
        for v in list(self.lb.keys()):
            if self.col_rows[v] or v in qv:
                continue
            ct = self.ctil(v)
            lo, hi = self.lb[v], self.ub[v]
            if ct > self.tol:
                if not math.isfinite(lo):
                    raise PresolveUnboundedError(f"Variable '{v}' improves the objective without bound (no lower bound).")
                value = lo
            elif ct < -self.tol:
                if not math.isfinite(hi):
                    raise PresolveUnboundedError(f"Variable '{v}' improves the objective without bound (no upper bound).")
                value = hi
            else:
                value = min(max(0.0, lo), hi)
            self.remove_var(v, value, ReductionType.EMPTY_COL)
            self.stats["empty_cols"] += 1
            progress = True
        return progress

    def activity_analysis(self) -> bool:
        progress = False
        for r in list(self.rows.keys()):
            if r not in self.rows:
                continue
            row = self.rows[r]
            if not row["coeffs"]:
                continue
            mn, mx, inf_mn, inf_mx = self.activity(row)
            rlb, rub = row["lb"], row["ub"]
            if inf_mn == 0 and mn > rub + self.rtol(rub):
                raise PresolveInfeasibleError(f"Row '{r}': minimum activity {mn:.6g} exceeds upper bound {rub:.6g}.")
            if inf_mx == 0 and mx < rlb - self.rtol(rlb):
                raise PresolveInfeasibleError(f"Row '{r}': maximum activity {mx:.6g} below lower bound {rlb:.6g}.")
            lower_ok = (not math.isfinite(rlb)) or (inf_mn == 0 and mn >= rlb - self.rtol(rlb))
            upper_ok = (not math.isfinite(rub)) or (inf_mx == 0 and mx <= rub + self.rtol(rub))
            if lower_ok and upper_ok:
                self.remove_row(r, ReductionType.REDUNDANT_ROW)
                self.stats["redundant_rows"] += 1
                progress = True
                continue
            # forcing rows: the only feasible activity is an extreme one
            if math.isfinite(rlb) and inf_mx == 0 and mx <= rlb + self.rtol(rlb):
                for v, a in row["coeffs"].items():
                    val = self.ub[v] if a > 0 else self.lb[v]
                    self.lb[v] = self.ub[v] = val
                self.remove_row(r, ReductionType.FORCING_ROW)
                self.stats["forcing_rows"] += 1
                progress = True
                continue
            if math.isfinite(rub) and inf_mn == 0 and mn >= rub - self.rtol(rub):
                for v, a in row["coeffs"].items():
                    val = self.lb[v] if a > 0 else self.ub[v]
                    self.lb[v] = self.ub[v] = val
                self.remove_row(r, ReductionType.FORCING_ROW)
                self.stats["forcing_rows"] += 1
                progress = True
                continue
            # implied bounds for integer variables
            for v, a in row["coeffs"].items():
                if not self.is_int(v) or abs(a) < 1e-9:
                    continue
                lo_c = a * self.lb[v] if a > 0 else a * self.ub[v]
                hi_c = a * self.ub[v] if a > 0 else a * self.lb[v]
                if math.isfinite(rub) and inf_mn == 0 and math.isfinite(lo_c):
                    bound = (rub - (mn - lo_c)) / a
                    if abs(bound) < 1e12:
                        if a > 0 and np.floor(bound + 1e-9) < self.ub[v] - 0.5:
                            self.ub[v] = float(np.floor(bound + 1e-9))
                            self.stats["tightened"] += 1
                            progress = True
                        elif a < 0 and np.ceil(bound - 1e-9) > self.lb[v] + 0.5:
                            self.lb[v] = float(np.ceil(bound - 1e-9))
                            self.stats["tightened"] += 1
                            progress = True
                if math.isfinite(rlb) and inf_mx == 0 and math.isfinite(hi_c):
                    bound = (rlb - (mx - hi_c)) / a
                    if abs(bound) < 1e12:
                        if a > 0 and np.ceil(bound - 1e-9) > self.lb[v] + 0.5:
                            self.lb[v] = float(np.ceil(bound - 1e-9))
                            self.stats["tightened"] += 1
                            progress = True
                        elif a < 0 and np.floor(bound + 1e-9) < self.ub[v] - 0.5:
                            self.ub[v] = float(np.floor(bound + 1e-9))
                            self.stats["tightened"] += 1
                            progress = True
                self.check_bounds(v)
                if progress:
                    break  # activities changed; revisit this row next pass
        return progress

    def dominated_columns(self) -> bool:
        progress = False
        qv = self.quad_vars()
        for v in list(self.lb.keys()):
            if v in qv or not self.col_rows[v]:
                continue
            ct = self.ctil(v)
            down_ok = up_ok = True
            for r in self.col_rows[v]:
                row = self.rows[r]
                a = row["coeffs"][v]
                if a > 0:
                    down_ok &= not math.isfinite(row["lb"])
                    up_ok &= not math.isfinite(row["ub"])
                else:
                    down_ok &= not math.isfinite(row["ub"])
                    up_ok &= not math.isfinite(row["lb"])
            if ct >= 0 and down_ok and math.isfinite(self.lb[v]):
                self.remove_var(v, self.lb[v], ReductionType.DOMINATED_COLUMN)
            elif ct <= 0 and up_ok and math.isfinite(self.ub[v]):
                self.remove_var(v, self.ub[v], ReductionType.DOMINATED_COLUMN)
            else:
                continue
            self.stats["dominated"] += 1
            progress = True
        return progress

    def doubleton_equations(self) -> bool:
        progress = False
        qv = self.quad_vars()
        for r in list(self.rows.keys()):
            if r not in self.rows:
                continue
            row = self.rows[r]
            if len(row["coeffs"]) != 2 or not math.isfinite(row["lb"]) or abs(row["ub"] - row["lb"]) > self.rtol(row["lb"]):
                continue
            (v1, a1), (v2, a2) = row["coeffs"].items()
            rhs = row["lb"]
            options = []
            for k, ak, j, aj in ((v1, a1, v2, a2), (v2, a2, v1, a1)):
                if k in qv:
                    continue
                if self.is_int(k):
                    if not (self.is_int(j) and abs(abs(ak) - 1.0) < 1e-12 and abs(aj - round(aj)) < 1e-12
                            and abs(rhs - round(rhs)) < 1e-9):
                        continue
                options.append((0 if not self.is_int(k) else 1, -abs(ak), k, ak, j, aj))
            if not options:
                continue
            _, _, k, ak, j, aj = min(options)
            slope, intercept = -aj / ak, rhs / ak
            if not (1e-6 <= abs(slope) <= 1e6):
                continue
            lk, uk = self.lb[k], self.ub[k]
            if slope > 0:
                jl, ju = (lk - intercept) / slope, (uk - intercept) / slope
            else:
                jl, ju = (uk - intercept) / slope, (lk - intercept) / slope
            if self.is_int(j):
                jl = float(np.ceil(jl - 1e-9)) if math.isfinite(jl) else jl
                ju = float(np.floor(ju + 1e-9)) if math.isfinite(ju) else ju
            self.lb[j] = max(self.lb[j], jl)
            self.ub[j] = min(self.ub[j], ju)
            self.check_bounds(j)
            self.remove_row(r, ReductionType.EMPTY_ROW, {"note": f"doubleton row used to eliminate '{k}'"})
            for r2 in list(self.col_rows[k]):
                row2 = self.rows[r2]
                ak2 = row2["coeffs"].pop(k)
                row2["lb"] -= ak2 * intercept
                row2["ub"] -= ak2 * intercept
                newc = row2["coeffs"].get(j, 0.0) + ak2 * slope
                if abs(newc) <= 1e-12:
                    row2["coeffs"].pop(j, None)
                    self.col_rows[j].discard(r2)
                else:
                    row2["coeffs"][j] = newc
                    self.col_rows[j].add(r2)
            del self.col_rows[k]
            ck = self.cost.pop(k, 0.0)
            self.offset += ck * intercept
            if ck:
                self.cost[j] = self.cost.get(j, 0.0) + ck * slope
            del self.lb[k], self.ub[k], self.vtype[k]
            self.mapper.record_step(PresolveStep(ReductionType.DOUBLETON_EQUATION,
                                                 {"var_name": k, "base_var": j, "slope": slope, "intercept": intercept, "con_name": r}))
            self.stats["doubleton"] += 1
            progress = True
        return progress

    def parallel_rows(self) -> bool:
        progress = False
        groups: Dict[tuple, List[str]] = {}
        for r, row in self.rows.items():
            if len(row["coeffs"]) < 2:
                continue
            items = sorted(row["coeffs"].items())
            a0 = items[0][1]
            key = tuple((v, round(a / a0, 10)) for v, a in items)
            groups.setdefault(key, []).append(r)
        for rows in groups.values():
            if len(rows) < 2:
                continue
            base = rows[0]
            if base not in self.rows:
                continue
            bitems = sorted(self.rows[base]["coeffs"].items())
            bvec = np.array([a for _, a in bitems])
            for other in rows[1:]:
                if other not in self.rows:
                    continue
                oitems = sorted(self.rows[other]["coeffs"].items())
                ovec = np.array([a for _, a in oitems])
                lam = ovec[0] / bvec[0]
                if [v for v, _ in bitems] != [v for v, _ in oitems] or not np.allclose(ovec, lam * bvec, rtol=1e-12, atol=0.0):
                    continue
                olo, ohi = self.rows[other]["lb"], self.rows[other]["ub"]
                lo, hi = (olo / lam, ohi / lam) if lam > 0 else (ohi / lam, olo / lam)
                brow = self.rows[base]
                brow["lb"] = max(brow["lb"], lo)
                brow["ub"] = min(brow["ub"], hi)
                if brow["lb"] > brow["ub"] + self.rtol(brow["lb"]):
                    raise PresolveInfeasibleError(f"Parallel rows '{base}' and '{other}' have disjoint ranges.")
                self.remove_row(other, ReductionType.PARALLEL_ROW, {"merged_into": base, "ratio": float(lam)})
                self.stats["parallel_rows"] += 1
                progress = True
        return progress

    def duplicate_columns(self) -> bool:
        progress = False
        qv = self.quad_vars()
        groups: Dict[tuple, List[str]] = {}
        for v in self.lb:
            if self.is_int(v) or v in qv or not self.col_rows[v]:
                continue
            items = sorted((r, self.rows[r]["coeffs"][v]) for r in self.col_rows[v])
            a0 = items[0][1]
            key = tuple((r, round(a / a0, 10)) for r, a in items) + (round(self.ctil(v) / a0, 10),)
            groups.setdefault(key, []).append(v)
        for cols in groups.values():
            if len(cols) < 2:
                continue
            j = cols[0]
            for k in cols[1:]:
                if j not in self.lb or k not in self.lb:
                    continue
                jitems = sorted((r, self.rows[r]["coeffs"][j]) for r in self.col_rows[j])
                kitems = sorted((r, self.rows[r]["coeffs"][k]) for r in self.col_rows[k])
                if [r for r, _ in jitems] != [r for r, _ in kitems]:
                    continue
                jv = np.array([a for _, a in jitems])
                kv = np.array([a for _, a in kitems])
                nu = kv[0] / jv[0]
                if not np.allclose(kv, nu * jv, rtol=1e-12, atol=0.0) or abs(self.ctil(k) - nu * self.ctil(j)) > 1e-12 * max(1.0, abs(self.ctil(k))):
                    continue
                lj, uj, lk, uk = self.lb[j], self.ub[j], self.lb[k], self.ub[k]
                if nu > 0:
                    new_lb, new_ub = lj + nu * lk, uj + nu * uk
                else:
                    new_lb, new_ub = lj + nu * uk, uj + nu * lk
                for r in list(self.col_rows[k]):
                    self.rows[r]["coeffs"].pop(k)
                del self.col_rows[k]
                self.cost.pop(k, None)
                del self.lb[k], self.ub[k], self.vtype[k]
                self.lb[j], self.ub[j] = new_lb, new_ub
                self.mapper.record_step(PresolveStep(ReductionType.DUPLICATE_COLUMN, {
                    "var_name": k, "base_var": j, "nu": float(nu),
                    "lb_base": lj, "ub_base": uj, "lb_var": lk, "ub_var": uk,
                }))
                self.stats["duplicate_cols"] += 1
                progress = True
        return progress

    def coefficient_tightening(self):
        """For rows with one finite side: shrink binary coefficients that exceed the row's slack."""
        for r, row in self.rows.items():
            has_lb, has_ub = math.isfinite(row["lb"]), math.isfinite(row["ub"])
            if has_lb == has_ub:
                continue
            flip = -1.0 if has_lb else 1.0  # work on  sum (flip * a) x <= flip * bound
            bound = flip * (row["ub"] if has_ub else row["lb"])
            for v in list(row["coeffs"].keys()):
                if self.vtype[v] != VariableType.BINARY and not (self.is_int(v) and self.lb[v] == 0 and self.ub[v] == 1):
                    continue
                a = flip * row["coeffs"][v]
                if a <= 0:
                    continue
                mx = 0.0
                finite = True
                for w, aw in row["coeffs"].items():
                    aw = flip * aw
                    hi = self.ub[w] if aw > 0 else self.lb[w]
                    if not math.isfinite(hi):
                        finite = False
                        break
                    mx += aw * hi
                if not finite:
                    break
                rest_max = mx - a
                if rest_max < bound - 1e-9 and a - (bound - rest_max) > 1e-9:
                    d = bound - rest_max
                    row["coeffs"][v] = flip * (a - d)
                    bound -= d
                    if has_ub:
                        row["ub"] = flip * bound
                    else:
                        row["lb"] = flip * bound
                    self.mapper.record_step(PresolveStep(ReductionType.COEFFICIENT_TIGHTENING,
                                                         {"con_name": r, "var_name": v, "delta": float(d)}))
                    self.stats["coef_tightening"] += 1

    # -------------------------------------------------------------- output
    def build_model(self) -> OptimizationModel:
        p = OptimizationModel(name=f"{self.model.name}_presolved")
        for v in self.var_order:
            if v in self.lb:
                p.add_variable(name=v, lower_bound=self.lb[v], upper_bound=self.ub[v], var_type=self.vtype[v])
        for c in self.con_order:
            if c not in self.rows:
                continue
            row = self.rows[c]
            lo, hi = row["lb"], row["ub"]
            coeffs = {v: a for v, a in row["coeffs"].items() if v in self.lb}
            if math.isfinite(lo) and math.isfinite(hi) and abs(hi - lo) <= self.rtol(lo):
                p.add_constraint(c, coeffs, ConstraintSense.EQ, rhs=lo)
            elif math.isfinite(hi) and not math.isfinite(lo):
                p.add_constraint(c, coeffs, ConstraintSense.LE, rhs=hi)
            elif math.isfinite(lo) and not math.isfinite(hi):
                p.add_constraint(c, coeffs, ConstraintSense.GE, rhs=lo)
            else:
                p.add_constraint(c, coeffs, ConstraintSense.RANGE, lower_bound=lo, upper_bound=hi)
        quad = {k: q for k, q in self.quad.items() if k[0] in self.lb and k[1] in self.lb}
        p.set_objective(
            linear_coefficients={v: c for v, c in self.cost.items() if v in self.lb},
            sense=self.model.objective.sense,
            quadratic_coefficients=quad,
            offset=self.offset,
        )
        return p
