"""
FastAPI REST backend service for the Sovereign Mathematical Optimization Engine (v2).
Connects the unified solve pipeline (sovereign_opt.solvers.dispatch) to the Next.js dashboard.

All v1 response fields are preserved. v2 adds row duals, reduced costs, best bound,
optimality-certificate details, the full presolve reduction breakdown, strategy notes,
and per-stage timings. Responses are sanitized so non-finite floats never break JSON.
"""
import math
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
        try:
            with open(path) as f:
                for r in json.load(f).get("rows", []):
                    if r.get("key") and r.get("cols"):
                        sizes[r["key"]] = (r.get("rows"), r.get("cols"), r.get("published_optimum"))
        except (OSError, ValueError, AttributeError):
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
