"""
FastAPI REST backend service for the Sovereign Mathematical Optimization Engine (v2).
Connects the unified solve pipeline (sovereign_opt.solvers.dispatch) to the Next.js dashboard.

All v1 response fields are preserved. v2 adds row duals, reduced costs, best bound,
optimality-certificate details, the full presolve reduction breakdown, strategy notes,
and per-stage timings. Responses are sanitized so non-finite floats never break JSON.
"""
import copy
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sovereign_opt import __version__
from sovereign_opt.model.model import OptimizationModel, ModelValidationError
from sovereign_opt.parsers.mps_parser import MPSParser
from sovereign_opt.parsers.lp_parser import LPParser
from sovereign_opt.presolve.presolver import Presolver, PresolveStats, PresolveInfeasibleError, PresolveUnboundedError
from sovereign_opt.ml.strategy import MLStrategyEngine
from sovereign_opt.ml.features import FeatureExtractor
from sovereign_opt.solvers.dispatch import solve_model, ALL_ALGORITHMS
from sovereign_opt.runtime.device import DeviceDetector
from sovereign_opt.sparse.scaling import RuizScaling
from sovereign_opt.sparse.matrix import SparseMatrix
from benchmarks.netlib.afiro import build_netlib_afiro
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.power_dispatch import build_unit_commitment_model
from benchmarks.industrial.portfolio_selection import build_portfolio_selection_model
from benchmarks.industrial.supply_chain import build_supply_chain_model
from benchmarks.instances import NOTES
# Registry metadata only. The comparison solvers live behind a subprocess (see the demo race
# endpoints at the bottom of this file); nothing here imports HiGHS or Gurobi.
from benchmarks.demo_race import DEMO_INSTANCES, SENTINEL as DEMO_SENTINEL, instance_payload

app = FastAPI(
    title="Sovereign Optimizer API",
    description="Backend API serving the AI-guided sovereign optimization engine.",
    version=__version__,
)

# Enable CORS for Next.js development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for current loaded model
CURRENT_MODEL: Optional[OptimizationModel] = None
CURRENT_PRESOLVE: Optional[Tuple[OptimizationModel, Any, PresolveStats]] = None

PRESETS: Dict[str, dict] = {
    "netlib_afiro": {
        "builder": build_netlib_afiro,
        "name": "Netlib Benchmark: AFIRO",
        "category": "Linear Programming (Netlib)",
        "description": "Standard Netlib LP benchmark built from the original MPS data. 27 constraints, 32 variables. Known optimal: -464.7531.",
        "problem_class": "LP",
    },
    "refinery_blending_lp": {
        "builder": lambda: build_refinery_blending_model(as_qp=False),
        "name": "Refinery Crude Blending (LP)",
        "category": "Industrial Production",
        "description": "Refinery feedstock blending with octane & sulfur constraints to maximize product revenues.",
        "problem_class": "LP",
    },
    "refinery_blending_qp": {
        "builder": lambda: build_refinery_blending_model(as_qp=True),
        "name": "Refinery Crude Blending (QP)",
        "category": "Industrial Production (Quadratic)",
        "description": "Refinery blending with quadratic penalties on heavy crude feedstock usage.",
        "problem_class": "QP",
    },
    "power_unit_commitment": {
        "builder": lambda: build_unit_commitment_model(time_periods=4),
        "name": "Power Grid Unit Commitment (MILP)",
        "category": "Energy & Utilities",
        "description": "Hourly generator scheduling with on/off binary commitment decisions and demand balance.",
        "problem_class": "MILP",
    },
    "power_unit_commitment_24h": {
        "builder": lambda: build_unit_commitment_model(time_periods=24),
        "name": "Power Grid Unit Commitment, 24h (MILP)",
        "category": "Energy & Utilities",
        "description": "Full-day commitment schedule: 96 binaries, 96 dispatch variables, 216 constraints.",
        "problem_class": "MILP",
    },
    "supply_chain_lp": {
        "builder": lambda: build_supply_chain_model(target_vars=3000),
        "name": "Production-Distribution Network (LP)",
        "category": "Supply Chain & Logistics",
        "description": "Plants -> warehouses -> customers, 4 products. Same generator scales to millions of variables "
                       "(python -m benchmarks.scale).",
        "problem_class": "LP",
    },
    "portfolio_miqp": {
        "builder": build_portfolio_selection_model,
        "name": "Cardinality-Constrained Portfolio (MIQP)",
        "category": "Finance (Mixed-Integer Quadratic)",
        "description": "Mean-variance portfolio with binary asset selection, minimum position sizes and a cardinality limit.",
        "problem_class": "MIQP",
    },
}


from benchmarks.industrial.power_dispatch import build_unit_commitment_fleet  # noqa: E402
from benchmarks.industrial.supply_chain import build_facility_location_model  # noqa: E402

PRESETS["unit_commitment_fleet_720"] = {
    "builder": lambda: build_unit_commitment_fleet(10, 24),
    "name": "Unit Commitment, 10 generators x 24 h (MILP, 720 variables)",
    "category": "Energy & Utilities",
    "description": "On/off, start-up and output per generator-hour with ramp limits and spinning reserve. "
                   "480 binaries. Optimum cross-checked with HiGHS: 868,889.91.",
    "problem_class": "MILP",
}
PRESETS["unit_commitment_fleet_2880"] = {
    "builder": lambda: build_unit_commitment_fleet(20, 48),
    "name": "Unit Commitment, 20 generators x 48 h (MILP, 2,880 variables)",
    "category": "Energy & Utilities",
    "description": "1,920 binaries, 4,856 constraints. Optimum cross-checked with HiGHS: 3,007,444.64. Use a 5 min limit.",
    "problem_class": "MILP",
}
PRESETS["facility_location_1220"] = {
    "builder": lambda: build_facility_location_model(20, 100),
    "name": "Warehouse Location, 20 sites x 100 customers (MILP, 1,220 variables)",
    "category": "Supply Chain & Logistics",
    "description": "Which warehouses to open and how to route demand. Big-M linking makes the LP relaxation weak. "
                   "Optimum cross-checked with HiGHS: 9,945.23. Use a 5 min limit.",
    "problem_class": "MILP",
}
PRESETS["supply_chain_lp_50k"] = {
    "builder": lambda: build_supply_chain_model(target_vars=50_000),
    "name": "Production-Distribution Network, 50k variables (LP)",
    "category": "Supply Chain & Logistics",
    "description": "Same generator at 50,000 variables. Try methods pdlp, hybrid_pdlp or concurrent. "
                   "Millions of variables: python -m benchmarks.scale.",
    "problem_class": "LP",
}


def _register_benchmark_library():
    """Expose every downloaded public benchmark instance (benchmarks/data) as a loadable problem."""
    import glob
    import json
    import os
    from benchmarks.instances import SUITES, get_instance, mps_path

    sizes: Dict[str, tuple] = {}
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "results")
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        if os.path.basename(path).startswith("demo_"):
            continue  # single-record demo-race results; "rows" there is a row count, not a list
        try:
            with open(path) as f:
                rows = json.load(f).get("rows", [])
            if not isinstance(rows, list):
                continue
            for r in rows:
                if isinstance(r, dict) and r.get("key") and r.get("cols"):
                    sizes[r["key"]] = (r.get("rows"), r.get("cols"), r.get("published_optimum"))
        except (OSError, ValueError, AttributeError, TypeError):
            continue

    groups = [("netlib", "LP", "Netlib LP"), ("netlib-large", "LP", "Netlib LP (large) / Kennington"),
              ("infeas", "LP", "Netlib infeasible LP"), ("qp", "QP", "Maros-Meszaros QP"),
              ("qp-large", "QP", "Maros-Meszaros QP (large)"), ("miplib", "MILP", "MIPLIB 2017")]
    for suite, pclass, label in groups:
        for key in SUITES[suite]:
            inst = get_instance(key)
            if not os.path.exists(inst.path):
                continue
            rows, cols, opt = sizes.get(key, (None, None, inst.published_optimum))
            parts = [f"{label} instance from the official collection file, parsed by the sovereign MPS parser."]
            if rows:
                parts.append(f"{rows} constraints, {cols} variables.")
            if inst.expected_status == "infeasible":
                parts.append("Known to be infeasible: the engine must prove it.")
            elif opt is not None:
                parts.append(f"Published optimum: {opt:.10g}.")
            if inst.tags:
                parts.append("Hard because: " + ", ".join(inst.tags) + ".")
            PRESETS[f"{inst.collection}/{inst.name}"] = {
                "builder": (lambda i=inst: MPSParser.parse_file(mps_path(i))),
                "name": f"{label}: {inst.name}",
                "category": label,
                "description": " ".join(parts),
                "problem_class": pclass,
            }


_register_benchmark_library()


class PresetRequest(BaseModel):
    preset_id: str


class SolveRequest(BaseModel):
    algorithm: Optional[str] = "auto"  # see sovereign_opt.solvers.dispatch.ALL_ALGORITHMS
    enable_presolve: bool = True
    time_limit_seconds: float = 60.0


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars / arrays and enums to JSON types; non-finite floats become null."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _ui_tree_trace(trace: List[dict]) -> List[dict]:
    """The dashboard renders bounds with toFixed(): show the parent's bound for infeasible subproblems."""
    bounds = {t["node_id"]: t.get("lower_bound") for t in trace}
    out = []
    for t in trace:
        entry = dict(t)
        value = entry.get("lower_bound")
        if value is None or not math.isfinite(value):
            parent = bounds.get(entry.get("parent_id"))
            entry["lower_bound"] = parent if (parent is not None and math.isfinite(parent)) else 0.0
            entry["bound_note"] = "infeasible subproblem (parent bound shown)"
        out.append(entry)
    return out


def _require_model() -> OptimizationModel:
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    return CURRENT_MODEL


@app.get("/")
def read_root():
    return {"status": "online", "engine": "Sovereign Mathematical Optimizer", "version": __version__}


@app.get("/api/system_info")
def get_system_info():
    info = DeviceDetector.get_info()
    return {
        "cpu_cores": info.cpu_cores,
        "has_cuda": info.has_cuda,
        "cuda_device_name": info.cuda_device_name,
        "preferred_device": info.preferred_device,
        "engine_version": __version__,
        "algorithms": list(ALL_ALGORITHMS),
    }


@app.get("/api/presets")
def list_presets():
    return [{"id": pid, **{k: v for k, v in meta.items() if k != "builder"}} for pid, meta in PRESETS.items()]


def extract_model_response(model: OptimizationModel) -> dict:
    meta = model.get_metadata()
    A_csr, r_lb, r_ub, c_lb, c_ub = model.to_matrix_form()
    c = np.array([model.objective.linear_coefficients.get(v, 0.0) for v in model.variable_names], dtype=np.float64)
    coo = A_csr.tocoo()

    ruiz = RuizScaling(max_iterations=10)
    try:
        A_scaled, _, _, _, _, _ = ruiz.fit_transform(SparseMatrix(A_csr), c, r_lb, r_ub, c_lb, c_ub)
        ruiz_stats = {
            "d1_min": round(float(np.min(ruiz.d1)), 4) if ruiz.d1.size else 1.0,
            "d1_max": round(float(np.max(ruiz.d1)), 4) if ruiz.d1.size else 1.0,
            "d2_min": round(float(np.min(ruiz.d2)), 4) if ruiz.d2.size else 1.0,
            "d2_max": round(float(np.max(ruiz.d2)), 4) if ruiz.d2.size else 1.0,
            "norm_before": round(float(np.max(np.abs(A_csr.data))), 4) if A_csr.nnz else 1.0,
            "norm_after": round(float(np.max(np.abs(A_scaled.csr.data))), 4) if A_scaled.nnz else 1.0,
            "iterations": int(ruiz.iterations_run),
        }
    except Exception:
        ruiz_stats = {"d1_min": 1.0, "d1_max": 1.0, "d2_min": 1.0, "d2_max": 1.0,
                      "norm_before": 1.0, "norm_after": 1.0, "iterations": 0}

    sparsity_sample = [
        {"row": int(r), "col": int(cc), "val": round(float(v), 4)}
        for r, cc, v in zip(coo.row[:600], coo.col[:600], coo.data[:600])
    ]

    return _json_safe({
        "name": meta.name,
        "problem_class": meta.problem_class,
        "num_variables": meta.num_variables,
        "num_constraints": meta.num_constraints,
        "num_nonzeros": meta.num_nonzeros,
        "density": meta.density,
        "num_continuous": meta.num_continuous,
        "num_integer": meta.num_integer,
        "num_binary": meta.num_binary,
        "num_quadratic_terms": meta.num_quadratic_terms,
        "ruiz_stats": ruiz_stats,
        "sparsity_sample": sparsity_sample,
        "variables": [
            {
                "name": v.name,
                "type": v.var_type.value,
                "lb": v.lower_bound if not np.isneginf(v.lower_bound) else "-inf",
                "ub": v.upper_bound if not np.isposinf(v.upper_bound) else "inf",
            }
            for v in list(model.variables.values())[:30]
        ],
        "constraints": [
            {"name": con.name, "sense": con.sense.value, "rhs": con.rhs, "num_terms": len(con.coefficients)}
            for con in list(model.constraints.values())[:30]
        ],
    })


@app.post("/api/load_preset")
def load_preset(req: PresetRequest):
    global CURRENT_MODEL, CURRENT_PRESOLVE
    if req.preset_id not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset ID: {req.preset_id}")
    CURRENT_PRESOLVE = None
    CURRENT_MODEL = PRESETS[req.preset_id]["builder"]()
    return extract_model_response(CURRENT_MODEL)


@app.post("/api/upload_model")
async def upload_model(file: UploadFile = File(...)):
    global CURRENT_MODEL, CURRENT_PRESOLVE
    contents = (await file.read()).decode("utf-8", errors="replace")
    filename = (file.filename or "").lower()
    try:
        if filename.endswith((".mps", ".qps")) or "NAME" in contents[:50]:
            model = MPSParser.parse_string(contents)
        else:
            model = LPParser.parse_string(contents)
        model.validate()
    except (ModelValidationError, ValueError, IndexError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    CURRENT_PRESOLVE = None
    CURRENT_MODEL = model
    return extract_model_response(model)


@app.post("/api/presolve")
def run_presolve():
    global CURRENT_PRESOLVE
    model = _require_model()
    try:
        p_model, mapper, stats = Presolver().presolve(model)
    except PresolveInfeasibleError as e:
        raise HTTPException(status_code=422, detail=f"Presolve proved the model infeasible: {e}")
    except PresolveUnboundedError as e:
        raise HTTPException(status_code=422, detail=f"Presolve found an unbounded direction: {e}")
    CURRENT_PRESOLVE = (p_model, mapper, stats)
    fixed_list = [{"name": k, "value": round(float(v), 6)} for k, v in list(mapper.fixed_vars.items())[:20]]
    return _json_safe({
        "original_vars": stats.original_vars,
        "original_cons": stats.original_cons,
        "original_nnz": stats.original_nnz,
        "presolved_vars": stats.presolved_vars,
        "presolved_cons": stats.presolved_cons,
        "presolved_nnz": stats.presolved_nnz,
        "var_reduction_pct": round(stats.var_reduction_pct, 2),
        "con_reduction_pct": round(stats.con_reduction_pct, 2),
        "nnz_reduction_pct": round(stats.nnz_reduction_pct, 2),
        "fixed_vars_count": stats.fixed_vars_count,
        "singleton_rows_count": stats.singleton_rows_count,
        "empty_rows_count": stats.empty_rows_count,
        "empty_cols_count": stats.empty_cols_count,
        "redundant_rows_count": stats.redundant_rows_count,
        "forcing_rows_count": stats.forcing_rows_count,
        "doubleton_eliminations": stats.doubleton_eliminations,
        "dominated_cols_count": stats.dominated_cols_count,
        "parallel_rows_count": stats.parallel_rows_count,
        "duplicate_cols_count": stats.duplicate_cols_count,
        "tightened_bounds_count": stats.tightened_bounds_count,
        "coefficient_tightenings": stats.coefficient_tightenings,
        "presolve_passes": stats.passes,
        "fixed_vars_list": fixed_list,
    })


@app.post("/api/ml_recommend")
def get_ml_recommendation():
    model = _require_model()
    stats = CURRENT_PRESOLVE[2] if CURRENT_PRESOLVE else None
    rec = MLStrategyEngine().recommend(model, stats)
    raw_features = FeatureExtractor.extract(model, stats)
    return _json_safe({
        "recommended_algorithm": rec.recommended_algorithm,
        "algorithm_probabilities": rec.algorithm_probabilities,
        "confidence_score": round(rec.confidence_score * 100, 1),
        "recommended_hardware": rec.recommended_hardware,
        "branching_strategy": rec.branching_strategy,
        "feature_attributions": rec.feature_attributions,
        "deterministic_fallback": rec.deterministic_fallback,
        "cut_strategy": rec.cut_strategy,
        "heuristic_intensity": rec.heuristic_intensity,
        "relaxation_solver": rec.relaxation_solver,
        "features": {k: round(float(v), 4) for k, v in raw_features.items()},
    })


BENCH_KINDS = ("netlib", "netlib-large", "infeas", "miplib", "qp", "qp-large", "robustness", "scale")


def _latest_results() -> Dict[str, Any]:
    """Most recent saved result file per benchmark kind (benchmarks/results/<kind>_<stamp>.json)."""
    import glob
    import json
    import os
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "results")
    out: Dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        name = os.path.basename(path)[:-5]
        kind = name.rsplit("_", 1)[0]
        if kind.split("_")[0] not in BENCH_KINDS:  # also tagged runs, e.g. netlib_kaggle_gpu_hybrid
            continue
        out[kind] = path  # sorted by timestamp, so the last one wins
    data = {}
    for kind, path in out.items():
        with open(path) as f:
            data[kind] = json.load(f)
        data[kind]["file"] = os.path.basename(path)
    return data


@app.get("/api/benchmarks")
def benchmarks():
    """Saved public-benchmark results (ours vs HiGHS), robustness report and scaling runs."""
    data = _latest_results()
    slim = {}
    for kind, d in data.items():
        rows = []
        for r in d.get("rows", []):
            if kind.startswith("scale"):
                rows.append(r)
                continue
            o, h = r.get("ours") or {}, r.get("highs") or {}
            dg = o.get("diagnostics") or {}
            rows.append({
                "name": r.get("name"), "tags": r.get("tags", []), "note": r.get("note") or NOTES.get(r.get("name") or ""), "rows": r.get("rows"), "cols": r.get("cols"),
                "nnz": r.get("nnz"), "int_vars": r.get("int_vars"), "reference": r.get("reference"),
                "reference_source": r.get("reference_source"), "rel_error": r.get("rel_error"),
                "rel_error_vs_highs": r.get("rel_error_vs_highs"), "verdict": r.get("verdict"), "ok": r.get("ok"),
                "range_before": r.get("range_before"), "range_after": r.get("range_after"),
                "ours": {k: o.get(k) for k in ("status", "objective", "time", "algorithm", "iterations", "nodes",
                                               "mip_gap", "certificate")},
                "highs": {k: h.get(k) for k in ("status", "objective", "time", "iterations", "nodes", "mip_gap")},
                "counters": {k: dg.get(k) for k in ("degenerate_pivots", "bound_perturbations", "bland_pivots",
                                                    "basis_repairs", "numerical_recoveries") if k in dg},
            })
        slim[kind] = {"meta": d.get("meta", {}), "file": d.get("file"), "rows": rows}
    return _json_safe(slim)


@app.post("/api/solve")
def solve(req: SolveRequest):
    global CURRENT_PRESOLVE
    model = _require_model()
    algorithm = req.algorithm or "auto"
    if algorithm not in ALL_ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm '{algorithm}'. Choose from {list(ALL_ALGORITHMS)}.")
    try:
        outcome = solve_model(
            model,
            algorithm=algorithm,
            enable_presolve=req.enable_presolve,
            time_limit_seconds=req.time_limit_seconds,
            presolve_result=CURRENT_PRESOLVE if req.enable_presolve else None,
        )
    except ModelValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if req.enable_presolve and CURRENT_PRESOLVE is None and outcome.presolved_model is not None:
        CURRENT_PRESOLVE = (outcome.presolved_model, outcome.mapper, outcome.presolve_stats)

    result, cert, stats, mapper = outcome.result, outcome.certificate, outcome.presolve_stats, outcome.mapper
    core = outcome.core_result

    mapping_sample = []
    for v_name in model.variable_names[:40]:
        if mapper is None:
            state = "DIRECT_SOLVE"
        elif v_name in mapper.fixed_vars:
            state = "FIXED_IN_PRESOLVE"
        elif v_name in mapper.eliminated_vars:
            state = "ELIMINATED_" + mapper.eliminated_vars[v_name].upper()
        elif v_name in core.primal_solution:
            state = "OPTIMIZED_IN_CORE"
        else:
            state = "CANONICAL_RESTORED"
        mapping_sample.append({
            "variable": v_name,
            "value": round(float(result.primal_solution.get(v_name, 0.0)), 6),
            "resolution_state": state,
        })

    diagnostics = dict(result.diagnostics)
    if "tree_trace" in diagnostics:
        diagnostics["tree_trace"] = _ui_tree_trace(diagnostics["tree_trace"])

    meta = model.get_metadata()
    rec = outcome.recommendation
    solve_ms = outcome.timings.get("solve", 0.0) * 1000.0
    core_summary = f"{result.status.value} in {result.iterations} iters ({solve_ms:.1f}ms)"
    if result.nodes_explored:
        core_summary += f", {result.nodes_explored} nodes, {diagnostics.get('cuts_applied', 0)} cuts"
    if cert.status == "NO_SOLUTION_TO_VERIFY":
        audit_status = "NO_SOLUTION"
    else:
        audit_status = "CERTIFIED" if cert.is_valid else "FAILED"
    audit_summary = f"{cert.status}: primal viol {cert.max_primal_violation:.1e}, bounds viol {cert.max_bound_violation:.1e}"
    if cert.max_dual_violation is not None:
        audit_summary += f", dual viol {cert.max_dual_violation:.1e}"
    if cert.duality_gap is not None:
        audit_summary += f", gap {cert.duality_gap:.1e}"

    workflow_stages = [
        {"stage": 1, "name": "Formulation & Topology", "status": "COMPLETED",
         "summary": f"{model.num_variables} vars, {model.num_constraints} cons, {meta.num_nonzeros} nnz ({outcome.problem_class})"},
        {"stage": 2, "name": "Presolve Reductions", "status": "COMPLETED" if stats else "BYPASS",
         "summary": (f"-{stats.var_reduction_pct:.1f}% vars, -{stats.con_reduction_pct:.1f}% cons, "
                     f"{stats.doubleton_eliminations} aggregations, {stats.forcing_rows_count} forcing rows") if stats else "Presolve bypassed"},
        {"stage": 3, "name": "AI Meta-Strategy", "status": "COMPLETED",
         "summary": f"Selected {outcome.solver_name} ({round(rec.confidence_score * 100, 1)}% confidence)"
                    + (f"; {len(outcome.notes)} note(s)" if outcome.notes else "")},
        {"stage": 4, "name": "Sovereign Core Solve", "status": "COMPLETED", "summary": core_summary},
        {"stage": 5, "name": "Postsolve Reconstruction", "status": "COMPLETED" if mapper else "BYPASS",
         "summary": f"Restored {len(result.primal_solution)} variables; duals: {diagnostics.get('dual_recovery', 'from core solver' if result.dual_solution else 'n/a')}"},
        {"stage": 6, "name": "Independent Trust Audit", "status": audit_status, "summary": audit_summary},
    ]

    response = {
        "status": result.status.value,
        "objective_value": result.objective_value,
        "iterations": result.iterations,
        "runtime_seconds": round(outcome.timings.get("total", result.runtime_seconds), 4),
        "nodes_explored": result.nodes_explored,
        "mip_gap": result.mip_gap,
        "best_bound": result.best_bound,
        "algorithm_used": outcome.solver_name,
        "algorithm_key": outcome.algorithm,
        "problem_class": outcome.problem_class,
        "primal_solution": {k: round(v, 6) for k, v in list(result.primal_solution.items())[:60]},
        "dual_solution": {k: round(v, 6) for k, v in list(result.dual_solution.items())[:60]},
        "reduced_costs": {k: round(v, 6) for k, v in list(result.reduced_costs.items())[:60]},
        "diagnostics": diagnostics,
        "postsolve_mapping": mapping_sample,
        "workflow_stages": workflow_stages,
        "notes": outcome.notes,
        "timings": {k: round(v, 5) for k, v in outcome.timings.items()},
        "trust_certificate": {
            "is_valid": cert.is_valid,
            "status": cert.status,
            "max_primal_violation": cert.max_primal_violation,
            "max_bound_violation": cert.max_bound_violation,
            "max_integrality_violation": cert.max_integrality_violation,
            "recomputed_objective": cert.recomputed_objective,
            "objective_difference": cert.objective_difference,
            "checks": cert.checks,
            "violation_details": cert.violation_details,
            "optimality_certified": cert.optimality_certified,
            "optimality_method": cert.optimality_method,
            "max_dual_violation": cert.max_dual_violation,
            "duality_gap": cert.duality_gap,
        },
    }
    return _json_safe(response)


# ============================================================================== demo race
# A curated head-to-head: our engine vs HiGHS vs Gurobi on four large instances.
#
# Each solver runs in its own subprocess (python -m benchmarks.demo_race), one at a time, so
# HiGHS keeps its single thread while our PDLP gets the whole machine -- matching the recorded
# benchmark conditions. A job registry plus polling (rather than SSE) keeps a ten-minute run
# alive across a browser reload, which is what the 1M-variable instance needs on stage.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results")
DEMO_JOBS: Dict[str, dict] = {}
DEMO_CACHE: Dict[str, dict] = {}
DEMO_GUROBI: Optional[Dict[str, Any]] = None
_DEMO_LOCK = threading.Lock()


def _demo_gurobi_probe() -> Dict[str, Any]:
    """
    Whether Gurobi is usable and whether its license caps model size. Measured once, in a
    subprocess, so the UI can grey out a bar before the race rather than during it.
    """
    global DEMO_GUROBI
    if DEMO_GUROBI is not None:
        return DEMO_GUROBI
    probe: Dict[str, Any] = {"available": False, "size_limited": None, "note": "probe failed"}
    try:
        out = subprocess.run([sys.executable, "-m", "benchmarks.demo_race", "--probe"],
                             cwd=REPO_ROOT, capture_output=True, text=True, timeout=90)
        parsed = _demo_parse(out.stdout, out.stderr)
        if "available" in parsed:
            probe = parsed
    except (subprocess.SubprocessError, OSError) as exc:
        probe = {"available": False, "size_limited": None, "note": f"probe failed: {exc}"}
    DEMO_GUROBI = probe
    return probe


def _demo_blank(instance_id: str, source: str) -> dict:
    inst = DEMO_INSTANCES[instance_id]
    order = [s["key"] for s in instance_payload(inst, _demo_gurobi_probe())["solvers"]]
    return {
        "job_id": uuid.uuid4().hex[:12], "instance_id": instance_id, "source": source,
        "state": "running", "order": order, "started_at": time.time(),
        "solvers": {k: {"state": "queued"} for k in order},
        "objective_match": {"reference": None, "values": {}, "rel_error": {}},
        "speedup": {"vs_highs": None, "vs_gurobi": None},
        "recorded": inst.recorded, "gpu_panel": inst.gpu_panel,
    }


def _demo_finalize(job: dict) -> None:
    """Recompute objective agreement and speedups from whatever has finished so far."""
    done = {k: v for k, v in job["solvers"].items()
            if v.get("state") == "done" and v.get("objective") is not None}
    ref = "highs" if "highs" in done else ("gurobi" if "gurobi" in done else ("ours" if "ours" in done else None))
    values = {k: v["objective"] for k, v in done.items()}
    errs = {}
    if ref is not None:
        base = values[ref]
        # same relative-error formula as benchmarks/compare.py, so both pages agree
        errs = {k: abs(v - base) / max(1.0, abs(base)) for k, v in values.items()}
    job["objective_match"] = {"reference": ref, "values": values, "rel_error": errs}

    def t(key):
        s = job["solvers"].get(key, {})
        return s.get("solve_time") if s.get("state") == "done" else None

    ours = t("ours")
    job["speedup"] = {
        "vs_highs": (t("highs") / ours) if ours and t("highs") else None,
        "vs_gurobi": (t("gurobi") / ours) if ours and t("gurobi") else None,
    }


def _demo_spawn(instance_id: str, solver: str, time_limit: Optional[float]):
    cmd = [sys.executable, "-m", "benchmarks.demo_race", "--instance", instance_id, "--solver", solver]
    if time_limit:
        cmd += ["--time-limit", str(time_limit)]
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"}, **kwargs)


def _demo_parse(stdout: str, stderr: str) -> dict:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(DEMO_SENTINEL):
            try:
                return json.loads(line[len(DEMO_SENTINEL):].strip())
            except json.JSONDecodeError:
                break
    return {"status": "error", "error": (stderr or "no result line on stdout")[-800:]}


def _demo_write(job: dict) -> None:
    try:
        os.makedirs(DEMO_RESULTS, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(DEMO_RESULTS, f"demo_{job['instance_id']}_{stamp}.json")
        with open(path, "w") as f:
            json.dump(_json_safe(job), f, indent=1)
    except OSError:
        pass  # a demo must never fail because a result file could not be written


def _demo_runner(job_id: str, time_limit: Optional[float]) -> None:
    job = DEMO_JOBS[job_id]
    inst = DEMO_INSTANCES[job["instance_id"]]
    applicable = {s["key"]: s for s in instance_payload(inst, _demo_gurobi_probe())["solvers"]}
    limit = time_limit or inst.default_time_limit
    for solver in job["order"]:
        if job["state"] == "cancelled":
            break
        spec = applicable[solver]
        if not spec["applicable"]:
            job["solvers"][solver] = {"state": "skipped", "status": "skipped",
                                      "note": spec["skip_reason"]}
            continue
        job["solvers"][solver] = {"state": "running", "started_at": time.time()}
        proc = None
        try:
            proc = _demo_spawn(job["instance_id"], solver, time_limit)
            job["_proc"] = proc
            out, err = proc.communicate(timeout=limit * 1.5 + 300)
            res = _demo_parse(out, err)
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
            res = {"status": "killed", "error": f"no result within {limit * 1.5 + 300:.0f}s"}
        except Exception as exc:  # reported, never hidden
            res = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            job.pop("_proc", None)
        if job["state"] == "cancelled":
            res.setdefault("status", "cancelled")
        status = str(res.get("status", ""))
        if status in ("optimal", "feasible", "time_limit"):
            res["state"] = "done"
        elif status.startswith("skipped") or status in ("unavailable", "license_limited"):
            res["state"] = "skipped"
        else:
            res["state"] = "error"
        job["solvers"][solver] = res
        _demo_finalize(job)

    for s in job["order"]:
        if job["solvers"][s].get("state") in ("queued", "running"):
            job["solvers"][s] = {"state": "cancelled", "status": "cancelled"}
    job["state"] = "cancelled" if job["state"] == "cancelled" else "done"
    job["elapsed"] = time.time() - job["started_at"]
    with _DEMO_LOCK:
        DEMO_CACHE[job["instance_id"]] = copy.deepcopy({k: v for k, v in job.items() if k != "_proc"})
    _demo_write(job)


def _demo_from_recorded(instance_id: str) -> dict:
    """A completed job synthesized from the registry's recorded numbers -- the presenter's parachute."""
    inst = DEMO_INSTANCES[instance_id]
    job = _demo_blank(instance_id, "recorded")
    specs = {s["key"]: s for s in instance_payload(inst, _demo_gurobi_probe())["solvers"]}
    for key in job["order"]:
        rec = inst.recorded.get(key)
        if rec:
            job["solvers"][key] = {"state": "done", "status": rec.get("status", "optimal"),
                                   "objective": rec.get("objective"), "solve_time": rec.get("time"),
                                   **dict(inst.size)}
        elif not specs[key]["applicable"]:
            job["solvers"][key] = {"state": "skipped", "status": "skipped",
                                   "note": specs[key]["skip_reason"]}
        else:
            job["solvers"][key] = {"state": "skipped", "status": "skipped",
                                   "note": "no recorded result for this solver"}
    job["state"] = "done"
    job["source_file"] = inst.recorded.get("source")
    _demo_finalize(job)
    return job


def _demo_disk_cached() -> set:
    try:
        return {os.path.basename(p)[5:].rsplit("_", 1)[0]
                for p in glob.glob(os.path.join(DEMO_RESULTS, "demo_*.json"))}
    except OSError:
        return set()


def _demo_view(job: dict) -> dict:
    """Job document for the client: no Popen handle, live elapsed for the running solver."""
    out = {k: v for k, v in job.items() if k != "_proc"}
    out["solvers"] = {k: dict(v) for k, v in job["solvers"].items()}
    for v in out["solvers"].values():
        if v.get("state") == "running" and v.get("started_at"):
            v["elapsed"] = time.time() - v["started_at"]
    return _json_safe(out)


class DemoRaceRequest(BaseModel):
    instance_id: str
    use_recorded: bool = False
    time_limit: Optional[float] = None
    force: bool = False


@app.get("/api/demo/instances")
def demo_instances():
    """The curated demo line-up, with sizes, recorded numbers and which solvers apply."""
    on_disk = _demo_disk_cached()
    gurobi = _demo_gurobi_probe()
    out = []
    for inst in DEMO_INSTANCES.values():
        p = instance_payload(inst, gurobi)
        p["cached"] = inst.id in DEMO_CACHE or inst.id in on_disk
        p["gurobi"] = gurobi
        out.append(p)
    return _json_safe(out)


@app.post("/api/demo/race")
def demo_race(req: DemoRaceRequest):
    """Start (or replay) a head-to-head race. Returns immediately; poll /api/demo/race/{job_id}."""
    if req.instance_id not in DEMO_INSTANCES:
        raise HTTPException(status_code=404, detail=f"unknown demo instance '{req.instance_id}'")
    if req.use_recorded:
        job = _demo_from_recorded(req.instance_id)
        DEMO_JOBS[job["job_id"]] = job
        return _demo_view(job)
    with _DEMO_LOCK:
        for job in DEMO_JOBS.values():
            if job["instance_id"] == req.instance_id and job["state"] == "running":
                return _demo_view(job)  # already racing; do not start a second one
        if not req.force and req.instance_id in DEMO_CACHE:
            cached = copy.deepcopy(DEMO_CACHE[req.instance_id])
            cached["source"] = "cache"
            DEMO_JOBS[cached["job_id"]] = cached
            return _demo_view(cached)
        job = _demo_blank(req.instance_id, "live")
        DEMO_JOBS[job["job_id"]] = job
    threading.Thread(target=_demo_runner, args=(job["job_id"], req.time_limit), daemon=True).start()
    return _demo_view(job)


@app.get("/api/demo/race/{job_id}")
def demo_race_status(job_id: str):
    job = DEMO_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return _demo_view(job)


@app.post("/api/demo/race/{job_id}/cancel")
def demo_race_cancel(job_id: str):
    job = DEMO_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    job["state"] = "cancelled"
    proc = job.get("_proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(3)
        except subprocess.TimeoutExpired:
            proc.kill()
    return _demo_view(job)
