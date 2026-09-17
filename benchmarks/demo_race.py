"""
Head-to-head demo race: sovereign engine vs HiGHS vs Gurobi on four curated instances.

    python -m benchmarks.demo_race --list
    python -m benchmarks.demo_race --instance milp_uc_720 --solver ours
    python -m benchmarks.demo_race --instance qp_stcqp1  --solver highs --save

This module runs exactly ONE (instance, solver) pair per process and prints a single
machine-readable line to stdout:

    ##DEMO_RESULT## {"instance": ..., "solver": ..., "solve_time": ..., ...}

Everything else (warnings, solver chatter) goes to stderr, so the sentinel line survives a
stray print from a dependency. sovereign_opt/server.py spawns this module as a subprocess and
parses that line; it never imports the comparison solvers itself -- the guard test in
tests/test_benchmarks.py enforces that separation.

Build time and solve time are reported separately: only solve_time races. Generating the 1M
variable constraint matrix takes tens of seconds and must not be charged to either solver.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

SENTINEL = "##DEMO_RESULT##"
RESULTS = os.path.join(os.path.dirname(__file__), "results")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The pip-installed Gurobi with no license file is capped at this size. A named-user or WLS
# license lifts the cap entirely, so the cap is never assumed -- gurobi_probe() measures it.
GUROBI_FREE_VARS = 2000
GUROBI_FREE_ROWS = 2000
GUROBI_LICENSE = os.path.join(REPO_ROOT, "Gurobi files", "gurobi.lic")


def _ensure_license() -> None:
    """Use the license shipped with the repo unless the environment already points somewhere."""
    if not os.environ.get("GRB_LICENSE_FILE") and os.path.exists(GUROBI_LICENSE):
        os.environ["GRB_LICENSE_FILE"] = GUROBI_LICENSE


def gurobi_probe() -> Dict[str, Any]:
    """
    Is Gurobi usable here, and is it size-limited? Measured, never assumed: a model just over
    the free-license cap either solves or raises SIZE_LIMIT_EXCEEDED (error 10010).
    """
    _ensure_license()
    try:
        import gurobipy as gp
    except ImportError:
        return {"available": False, "size_limited": None,
                "note": "gurobipy is not installed in this environment (pip install gurobipy)."}
    try:
        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model(env=env)
        n = GUROBI_FREE_VARS + 1
        x = m.addMVar(n, lb=0.0, ub=1.0, obj=np.ones(n))
        m.addConstr(x.sum() >= 1.0)
        m.optimize()
        return {"available": True, "size_limited": False,
                "version": ".".join(str(v) for v in gp.gurobi.version()),
                "note": "full license: no model-size cap."}
    except gp.GurobiError as exc:
        limited = exc.errno == gp.GRB.Error.SIZE_LIMIT_EXCEEDED
        return {"available": True, "size_limited": limited,
                "version": ".".join(str(v) for v in gp.gurobi.version()),
                "note": (license_skip_generic() if limited else f"Gurobi error {exc.errno}: {exc}")}


def license_skip_generic() -> str:
    return (f"size-limited Gurobi license: at most {GUROBI_FREE_VARS:,} variables and "
            f"{GUROBI_FREE_ROWS:,} constraints.")


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class DemoInstance:
    id: str
    letter: str
    title: str
    industry: str
    story: str
    problem_class: str                 # LP | QP | MILP
    kind: str                          # arrays | qps | model
    build: Dict[str, Any]
    ours_algorithm: str                # pdlp_cpu | auto
    size: Dict[str, int]               # expected rows / cols / nnz, for the card before a run
    solvers: Tuple[str, ...]
    default_time_limit: float
    recorded: Dict[str, Any]
    long_running: bool = False
    hidden: bool = False
    gpu_panel: Optional[Dict[str, Any]] = None
    notes: Dict[str, str] = field(default_factory=dict)


_LICENSE_SKIP = ("free size-limited Gurobi license: at most {mv:,} variables and {mr:,} "
                 "constraints. This model has {v:,} and {r:,}.")


def license_skip(cols: int, rows: int) -> str:
    return _LICENSE_SKIP.format(mv=GUROBI_FREE_VARS, mr=GUROBI_FREE_ROWS, v=cols, r=rows)


DEMO_INSTANCES: Dict[str, DemoInstance] = {
    "supply_chain_100k": DemoInstance(
        id="supply_chain_100k", letter="A",
        title="Production-distribution network",
        industry="supply chain / logistics",
        story=("Plants make four products, ship them to regional warehouses, and warehouses serve "
               "customers through their five nearest depots. Minimise total transport cost subject "
               "to plant capacity, warehouse balance and throughput, and customer demand."),
        problem_class="LP", kind="arrays", build={"target": 100_000},
        ours_algorithm="pdlp_cpu",
        size={"rows": 19_586, "cols": 99_728, "nnz": 295_596},
        solvers=("ours", "highs", "gurobi"),
        default_time_limit=600.0,
        recorded={
            "ours": {"time": 2.98, "objective": 8858483.403039599, "status": "optimal"},
            "highs": {"time": 4.39, "objective": 8858352.01769039, "status": "optimal"},
            "gurobi": {"time": 2.856, "objective": 8858352.017690392, "status": "optimal"},
            "source": "scale_20260917-105038.json + measured 20260918",
        },
    ),
    "supply_chain_1m": DemoInstance(
        id="supply_chain_1m", letter="B",
        title="Production-distribution network at 1M variables",
        industry="supply chain / logistics",
        story=("The same formulation scaled to a million decision variables and three million "
               "matrix nonzeros -- the size a national distribution plan actually reaches."),
        problem_class="LP", kind="arrays", build={"target": 1_000_000},
        ours_algorithm="pdlp_cpu",
        size={"rows": 193_442, "cols": 999_188, "nnz": 2_959_896},
        solvers=("ours", "highs", "gurobi"),
        default_time_limit=900.0, long_running=True,
        recorded={
            "ours": {"time": 166.91, "objective": 55033635.044940114, "status": "optimal"},
            "highs": {"time": 417.26, "objective": 55042263.251770295, "status": "optimal"},
            "gurobi": {"time": 299.25, "objective": 55042263.25177029, "status": "optimal"},
            "source": "scale_20260917-105038.json + measured 20260918",
        },
        notes={"honesty": ("This is the headline: at a million variables we are faster than "
                           "both references on the same laptop -- 1.8x faster than Gurobi and "
                           "2.5x faster than HiGHS, to the same optimum.")},
        gpu_panel={
            "device": "Tesla T4", "ours": 9.6335, "highs": 305.7434, "speedup": 31.7,
            "gurobi": 299.25,
            "rows": 193_442, "cols": 999_188,
            "source": "scale_kaggle_gpu_20260917-081355.json",
            "note": "prior recorded run on a Kaggle Tesla T4 (4 vCPU) -- not this laptop, not live.",
        },
    ),
    "qp_stcqp1": DemoInstance(
        id="qp_stcqp1", letter="C",
        title="STCQP1 -- Maros-Meszaros convex QP",
        industry="public benchmark set",
        story=("A published instance from the Maros-Meszaros convex QP set: a structured "
               "quadratic program of the kind that appears in process control and least-squares "
               "estimation. Not generated by us -- this is the reference literature."),
        problem_class="QP", kind="qps", build={"key": "qp:STCQP1"},
        ours_algorithm="auto",
        size={"rows": 2_052, "cols": 4_097, "nnz": 13_338},
        solvers=("ours", "highs", "gurobi"),
        default_time_limit=300.0,
        recorded={
            "ours": {"time": 1.174, "objective": 155143.55470911262, "status": "optimal"},
            "highs": {"time": 154.040, "objective": 155143.55470395752, "status": "optimal"},
            "gurobi": {"time": 0.0286, "objective": 155143.55474395794, "status": "optimal"},
            "source": "qp-large_20260917-101301.json + measured 20260918",
        },
        notes={"honesty": ("Against the open-source reference we are 131x faster. Gurobi is "
                           "faster still on this one (29 ms) -- commercial QP remains ahead, "
                           "and that gap is exactly what the roadmap targets.")},
    ),
    "milp_uc_720": DemoInstance(
        id="milp_uc_720", letter="D",
        title="Unit commitment -- 10 generators over 24 hours",
        industry="power system dispatch",
        story=("Which generating units run in each hour, at what output, to meet demand at "
               "least cost -- with start-up costs, minimum up/down logic and ramp limits. "
               "480 binary decisions."),
        problem_class="MILP", kind="model", build={"gens": 10, "hours": 24},
        ours_algorithm="auto",
        size={"rows": 1_228, "cols": 720, "nnz": 3_300, "int_vars": 480},
        solvers=("ours", "highs", "gurobi"),
        default_time_limit=300.0,
        recorded={
            "ours": {"time": 1.03, "objective": 868889.9109681123, "status": "optimal"},
            "highs": {"time": 0.11, "objective": 868889.9109681122, "status": "optimal"},
            "gurobi": {"time": 0.064, "objective": 868889.9109681123, "status": "optimal"},
            "source": "milpscale_20260917-143744.json + measured 20260918",
        },
        notes={"honesty": ("HiGHS wins this one. The objectives agree to the last bit and our "
                           "answer carries an independent optimality certificate -- the tree "
                           "search is simply not yet tuned.")},
    ),
    "qp_aug3dcqp": DemoInstance(
        id="qp_aug3dcqp", letter="C2",
        title="AUG3DCQP -- Maros-Meszaros convex QP",
        industry="public benchmark set",
        story="Spare QP instance, held in reserve behind STCQP1.",
        problem_class="QP", kind="qps", build={"key": "qp:AUG3DCQP"},
        ours_algorithm="auto",
        size={"rows": 1_000, "cols": 3_873, "nnz": 6_546},
        solvers=("ours", "highs", "gurobi"),
        default_time_limit=300.0, hidden=True,
        recorded={
            "ours": {"time": 0.454, "objective": 993.3621486, "status": "optimal"},
            "highs": {"time": 201.760, "objective": 993.3621465, "status": "optimal"},
            "gurobi": {"time": 0.0518, "objective": 993.3621479006056, "status": "optimal"},
            "source": "qp-large_20260917-101301.json + measured 20260918",
        },
    ),
}

DEMO_ORDER = [i for i in DEMO_INSTANCES if not DEMO_INSTANCES[i].hidden]

SOLVER_LABELS = {
    "ours": "sovereign engine",
    "highs": "HiGHS (1 thread)",
    "gurobi": "Gurobi",
}


def solver_applicable(inst: DemoInstance, solver: str,
                      gurobi: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
    """
    (applicable, reason it is skipped). `gurobi` is a gurobi_probe() result; without one the
    license is treated as unrestricted and the runner reports the truth after the fact.
    """
    if solver not in inst.solvers:
        return False, "not run for this instance"
    if solver == "gurobi" and gurobi is not None:
        if not gurobi.get("available"):
            return False, gurobi.get("note")
        if gurobi.get("size_limited"):
            cols, rows = inst.size.get("cols", 0), inst.size.get("rows", 0)
            if cols > GUROBI_FREE_VARS or rows > GUROBI_FREE_ROWS:
                return False, license_skip(cols, rows)
    return True, None


def instance_payload(inst: DemoInstance, gurobi: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Registry entry as plain JSON for the API -- no builders, no numpy."""
    solvers = []
    for s in inst.solvers:
        ok, reason = solver_applicable(inst, s, gurobi)
        label = SOLVER_LABELS[s]
        if s == "gurobi" and gurobi and gurobi.get("version"):
            label = f"Gurobi {gurobi['version']}"
        solvers.append({"key": s, "label": label, "applicable": ok, "skip_reason": reason})
    return {
        "id": inst.id, "letter": inst.letter, "title": inst.title, "industry": inst.industry,
        "story": inst.story, "problem_class": inst.problem_class, "size": dict(inst.size),
        "solvers": solvers, "long_running": inst.long_running, "hidden": inst.hidden,
        "default_time_limit": inst.default_time_limit, "recorded": inst.recorded,
        "gpu_panel": inst.gpu_panel, "notes": dict(inst.notes),
    }


# ----------------------------------------------------------------------------- builders
def _build(inst: DemoInstance):
    """Returns (form, payload): ("arrays", LPArrays) | ("model", OptimizationModel) | ("path", str)."""
    if inst.kind == "arrays":
        from benchmarks.industrial.supply_chain import build_supply_chain_arrays, sizes_for
        return "arrays", build_supply_chain_arrays(*sizes_for(inst.build["target"]))
    if inst.kind == "model":
        from benchmarks.industrial.power_dispatch import build_unit_commitment_fleet
        return "model", build_unit_commitment_fleet(inst.build["gens"], inst.build["hours"])
    if inst.kind == "qps":
        from benchmarks.instances import get_instance, mps_path
        return "path", mps_path(get_instance(inst.build["key"]))
    raise ValueError(f"unknown kind {inst.kind}")


def _size_of(form: str, payload) -> Dict[str, int]:
    if form == "arrays":
        m, n = payload.A.shape
        return {"rows": int(m), "cols": int(n), "nnz": int(payload.A.nnz), "int_vars": 0}
    if form == "model":
        md = payload.get_metadata()
        return {"rows": int(md.num_constraints), "cols": int(md.num_variables),
                "nnz": int(md.num_nonzeros), "int_vars": int(md.num_integer + md.num_binary)}
    return {}


def _standard_arrays(form: str, payload):
    """
    Everything a third-party solver needs, identical to what our engine sees:
    (A, c, row_lb, row_ub, col_lb, col_ub, vtypes, Q, offset, sign)

    c and Q are normalised to MINIMIZE; the reported objective is sign * min_value + offset.
    """
    if form == "path":
        from sovereign_opt.parsers.mps_parser import MPSParser
        payload, form = MPSParser.parse_file(payload), "model"
    if form == "arrays":
        n = payload.A.shape[1]
        return (payload.A, payload.c, payload.row_lb, payload.row_ub, payload.col_lb, payload.col_ub,
                np.array(["C"] * n), None, 0.0, 1.0)
    from sovereign_opt.model.objective import ObjectiveSense
    A, row_lb, row_ub, col_lb, col_ub = payload.to_matrix_form()
    c = payload.get_objective_vector()
    Q = payload.get_quadratic_matrix()
    code = {"continuous": "C", "integer": "I", "binary": "B"}
    vtypes = np.array([code[payload.variables[v].var_type.value] for v in payload.variable_names])
    sign = 1.0 if payload.objective.sense == ObjectiveSense.MINIMIZE else -1.0
    return (A, c, row_lb, row_ub, col_lb, col_ub, vtypes, (Q if Q.nnz else None),
            float(payload.objective.offset), sign)


# ----------------------------------------------------------------------------- runners
def run_ours(inst: DemoInstance, form: str, payload, time_limit: float) -> Dict[str, Any]:
    if inst.ours_algorithm == "pdlp_cpu":
        from sovereign_opt.solvers.lp.pdlp import pdlp
        lp = payload
        t0 = time.perf_counter()
        r = pdlp(lp.A, lp.c, lp.row_lb, lp.row_ub, lp.col_lb, lp.col_ub,
                 tol=1e-4, time_limit=time_limit, device="cpu")
        total = time.perf_counter() - t0
        return {"status": r.status, "objective": r.primal_objective, "solve_time": total,
                "algorithm": "pdlp", "iterations": r.iterations, "restarts": r.restarts,
                "device": r.device, "rel_gap": r.rel_gap,
                "rel_primal_residual": r.rel_primal_residual, "rel_dual_residual": r.rel_dual_residual}

    from sovereign_opt.solvers.dispatch import solve_model
    if form == "path":
        from sovereign_opt.parsers.mps_parser import MPSParser
        payload = MPSParser.parse_file(payload)
    t0 = time.perf_counter()
    out = solve_model(payload, algorithm="auto", time_limit_seconds=time_limit)
    total = time.perf_counter() - t0
    r, cert = out.result, out.certificate
    return {"status": r.status.value, "objective": r.objective_value, "solve_time": total,
            "algorithm": out.algorithm, "iterations": r.iterations, "nodes": r.nodes_explored,
            "mip_gap": r.mip_gap, "best_bound": r.best_bound, "certificate": cert.status,
            "max_primal_violation": cert.max_primal_violation, "duality_gap": cert.duality_gap}


def run_highs(inst: DemoInstance, form: str, payload, time_limit: float) -> Dict[str, Any]:
    if form == "arrays":
        from benchmarks.scale import _highs_arrays  # same code path as the recorded scale numbers
        r = _highs_arrays(payload, time_limit)
        return {"status": r["status"], "objective": r["objective"], "solve_time": r["time"]}

    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(time_limit))
    if form == "path":
        h.readModel(payload)                       # same path as benchmarks/compare.py
    else:
        import tempfile
        from sovereign_opt.parsers.mps_writer import write_mps
        fd, tmp = tempfile.mkstemp(suffix=".mps")
        with os.fdopen(fd, "w") as f:
            f.write(write_mps(payload))
        try:
            h.readModel(tmp)
        finally:
            os.unlink(tmp)
    t0 = time.perf_counter()
    h.run()
    total = time.perf_counter() - t0
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus()).lower()
    obj = info.objective_function_value if info.primal_solution_status >= 1 else None
    return {"status": status, "objective": obj, "solve_time": total,
            "iterations": int(max(info.simplex_iteration_count, 0) + max(info.ipm_iteration_count, 0)),
            "nodes": int(max(info.mip_node_count, 0))}


def run_gurobi(inst: DemoInstance, form: str, payload, time_limit: float) -> Dict[str, Any]:
    _ensure_license()
    try:
        import gurobipy as gp
    except ImportError:
        return {"status": "unavailable",
                "note": "gurobipy is not installed in this environment (pip install gurobipy)."}

    A, c, row_lb, row_ub, col_lb, col_ub, vtypes, Q, offset, sign = _standard_arrays(form, payload)
    A = sp.csr_matrix(A)
    m_rows, n_cols = A.shape

    eq = np.isclose(row_lb, row_ub)
    lo_fin, up_fin = np.isfinite(row_lb), np.isfinite(row_ub)
    rng = (~eq) & lo_fin & up_fin
    le = (~eq) & (~rng) & up_fin
    ge = (~eq) & (~rng) & lo_fin & (~up_fin)
    built_rows = int(eq.sum() + le.sum() + ge.sum() + 2 * rng.sum())

    try:
        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model("demo_race", env=env)
        m.Params.Threads = 1
        m.Params.TimeLimit = float(time_limit)
        lb = np.where(np.isneginf(col_lb), -gp.GRB.INFINITY, col_lb)
        ub = np.where(np.isposinf(col_ub), gp.GRB.INFINITY, col_ub)
        x = m.addMVar(int(n_cols), lb=lb, ub=ub, obj=c, vtype=list(vtypes))
        if Q is not None and Q.nnz:
            m.setObjective(c @ x + 0.5 * (x @ Q @ x), gp.GRB.MINIMIZE)
        for mask, sense, rhs in ((eq, "=", row_lb), (le, "<", row_ub), (ge, ">", row_lb)):
            if mask.any():
                m.addMConstr(A[mask], x, sense, rhs[mask])
        if rng.any():
            m.addMConstr(A[rng], x, "<", row_ub[rng])
            m.addMConstr(A[rng], x, ">", row_lb[rng])

        t0 = time.perf_counter()
        m.optimize()
        total = time.perf_counter() - t0

        smap = {gp.GRB.OPTIMAL: "optimal", gp.GRB.INFEASIBLE: "infeasible",
                gp.GRB.UNBOUNDED: "unbounded", gp.GRB.TIME_LIMIT: "time_limit",
                gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded"}
        status = smap.get(m.Status, f"status_{m.Status}")
        obj = sign * m.ObjVal + offset if m.SolCount > 0 else None
        out = {"status": status, "objective": obj, "solve_time": total,
               "nodes": int(m.NodeCount), "version": ".".join(str(v) for v in gp.gurobi.version())}
        if (vtypes != "C").any():
            try:
                out["mip_gap"] = float(m.MIPGap)
            except Exception:
                pass
        return out
    except gp.GurobiError as exc:
        if exc.errno == gp.GRB.Error.SIZE_LIMIT_EXCEEDED:
            return {"status": "skipped_license", "note": license_skip(int(n_cols), built_rows)}
        return {"status": "license_limited", "note": f"Gurobi error {exc.errno}: {exc}"}


RUNNERS = {"ours": run_ours, "highs": run_highs, "gurobi": run_gurobi}


# ----------------------------------------------------------------------------- driver
def race_one(instance_id: str, solver: str, time_limit: Optional[float] = None) -> Dict[str, Any]:
    inst = DEMO_INSTANCES[instance_id]
    tl = float(time_limit) if time_limit else inst.default_time_limit
    rec: Dict[str, Any] = {"instance": instance_id, "solver": solver, "time_limit": tl, **inst.size}

    ok, reason = solver_applicable(inst, solver, gurobi_probe() if solver == "gurobi" else None)
    if not ok:
        rec.update({"status": "skipped_license" if solver == "gurobi" else "skipped", "note": reason})
        return rec
    try:
        t0 = time.perf_counter()
        form, payload = _build(inst)
        rec["build_time"] = time.perf_counter() - t0
        rec.update(_size_of(form, payload))
        rec.update(RUNNERS[solver](inst, form, payload, tl))
    except Exception as exc:
        rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc()[-1500:]})
    return rec


def _json_safe(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def _emit(rec: Dict[str, Any]) -> None:
    print(SENTINEL + " " + json.dumps({k: _json_safe(v) for k, v in rec.items()}), flush=True)


def _save(rec: Dict[str, Any]) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"demo_{rec['instance']}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump({k: _json_safe(v) for k, v in rec.items()}, f, indent=1)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="demo race: one instance, one solver, one JSON line")
    ap.add_argument("--list", action="store_true", help="print the instance registry and exit")
    ap.add_argument("--probe", action="store_true", help="report Gurobi availability and size cap")
    ap.add_argument("--instance", help="instance id (see --list)")
    ap.add_argument("--solver", choices=list(RUNNERS), help="which solver to run")
    ap.add_argument("--all", action="store_true", help="run every solver for the instance")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--save", action="store_true", help="also write benchmarks/results/demo_*.json")
    a = ap.parse_args(argv)

    if a.probe:
        _emit({"probe": "gurobi", **gurobi_probe()})
        return 0
    if a.list:
        for inst in DEMO_INSTANCES.values():
            s = inst.size
            print(f"{inst.letter:>2}  {inst.id:<20} {inst.problem_class:<5} "
                  f"{s.get('rows', 0):>7,} x {s.get('cols', 0):>9,}  {inst.title}")
        return 0
    if not a.instance or a.instance not in DEMO_INSTANCES:
        ap.error("--instance is required; see --list")
    if not a.solver and not a.all:
        ap.error("--solver or --all is required")

    for s in (list(DEMO_INSTANCES[a.instance].solvers) if a.all else [a.solver]):
        rec = race_one(a.instance, s, a.time_limit)
        _emit(rec)
        if a.save:
            print(f"saved {_save(rec)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
