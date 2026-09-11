"""
FastAPI REST backend service for the Sovereign Mathematical Optimization Engine.
Connects the Sovereign Solver Core to the Next.js interactive demo frontend.
"""
from typing import Dict, List, Optional, Tuple, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import numpy as np

from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.parsers.mps_parser import MPSParser
from sovereign_opt.parsers.lp_parser import LPParser
from sovereign_opt.presolve.presolver import Presolver, PresolveStats
from sovereign_opt.ml.strategy import MLStrategyEngine
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver
from sovereign_opt.solvers.lp.interior_point import InteriorPointSolver
from sovereign_opt.solvers.milp.branch_bound import BranchAndBoundSolver
from sovereign_opt.solvers.qp.active_set import ActiveSetQPSolver
from sovereign_opt.solvers.base import SolverResult, SolverStatus
from sovereign_opt.validation.validator import IndependentValidator
from sovereign_opt.runtime.device import DeviceDetector
from benchmarks.netlib.afiro import build_netlib_afiro
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.power_dispatch import build_unit_commitment_model

app = FastAPI(
    title="Sovereign Optimizer API",
    description="Backend API serving the AI-guided sovereign optimization engine.",
    version="0.1.0",
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


class PresetRequest(BaseModel):
    preset_id: str


class SolveRequest(BaseModel):
    algorithm: Optional[str] = "auto"  # 'auto', 'simplex', 'interior_point', 'branch_and_bound', 'active_set'
    enable_presolve: bool = True
    time_limit_seconds: float = 60.0


@app.get("/")
def read_root():
    return {"status": "online", "engine": "Sovereign Mathematical Optimizer", "version": "0.1.0"}


@app.get("/api/system_info")
def get_system_info():
    info = DeviceDetector.get_info()
    return {
        "cpu_cores": info.cpu_cores,
        "has_cuda": info.has_cuda,
        "cuda_device_name": info.cuda_device_name,
        "preferred_device": info.preferred_device,
    }


@app.get("/api/presets")
def list_presets():
    return [
        {
            "id": "netlib_afiro",
            "name": "Netlib Benchmark: AFIRO",
            "category": "Linear Programming (Netlib)",
            "description": "Standard Netlib LP benchmark instance. 27 variables, 32 constraints. Known optimal: -464.753.",
            "problem_class": "LP",
        },
        {
            "id": "refinery_blending_lp",
            "name": "Refinery Crude Blending (LP)",
            "category": "Industrial Production",
            "description": "Refinery feedstock blending with octane & sulfur constraints to maximize product revenues.",
            "problem_class": "LP",
        },
        {
            "id": "refinery_blending_qp",
            "name": "Refinery Crude Blending (QP)",
            "category": "Industrial Production (Quadratic)",
            "description": "Refinery blending with quadratic penalties on heavy crude feedstock usage.",
            "problem_class": "QP",
        },
        {
            "id": "power_unit_commitment",
            "name": "Power Grid Unit Commitment (MILP)",
            "category": "Energy & Utilities",
            "description": "Hourly generator scheduling with on/off binary commitment decisions and demand balance.",
            "problem_class": "MILP",
        },
    ]


from sovereign_opt.sparse.scaling import RuizScaling
from sovereign_opt.sparse.matrix import SparseMatrix
from sovereign_opt.ml.features import FeatureExtractor

def extract_model_response(model: OptimizationModel) -> dict:
    meta = model.get_metadata()
    A_csr, r_lb, r_ub, c_lb, c_ub = model.to_matrix_form()
    c = np.array([model.objective.linear_coefficients.get(v, 0.0) for v in model.variable_names], dtype=np.float64)
    coo = A_csr.tocoo()

    # Calculate Ruiz equilibration scaling metrics
    ruiz = RuizScaling(max_iterations=10)
    try:
        sp_mat = SparseMatrix(A_csr.data.copy(), A_csr.indices.copy(), A_csr.indptr.copy(), A_csr.shape)
        A_scaled, scaled_c, _, _, _, _ = ruiz.fit_transform(sp_mat, c, r_lb, r_ub, c_lb, c_ub)
        d1_min, d1_max = float(np.min(ruiz.d1)), float(np.max(ruiz.d1))
        d2_min, d2_max = float(np.min(ruiz.d2)), float(np.max(ruiz.d2))
        ruiz_stats = {
            "d1_min": round(d1_min, 4),
            "d1_max": round(d1_max, 4),
            "d2_min": round(d2_min, 4),
            "d2_max": round(d2_max, 4),
            "norm_before": round(float(np.max(np.abs(A_csr.data))), 4) if len(A_csr.data) > 0 else 1.0,
            "norm_after": round(float(np.max(np.abs(A_scaled.data))), 4) if len(A_scaled.data) > 0 else 1.0,
            "iterations": 10,
        }
    except Exception as e:
        ruiz_stats = {
            "d1_min": 1.0, "d1_max": 1.0, "d2_min": 1.0, "d2_max": 1.0,
            "norm_before": 1.0, "norm_after": 1.0, "iterations": 0
        }

    sparsity_sample = [
        {"row": int(r), "col": int(c), "val": round(float(v), 4)}
        for r, c, v in zip(coo.row[:600], coo.col[:600], coo.data[:600])
    ]

    return {
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
            {
                "name": c.name,
                "sense": c.sense.value,
                "rhs": c.rhs,
                "num_terms": len(c.coefficients),
            }
            for c in list(model.constraints.values())[:30]
        ],
    }


@app.post("/api/load_preset")
def load_preset(req: PresetRequest):
    global CURRENT_MODEL, CURRENT_PRESOLVE
    CURRENT_PRESOLVE = None

    if req.preset_id == "netlib_afiro":
        model = build_netlib_afiro()
    elif req.preset_id == "refinery_blending_lp":
        model = build_refinery_blending_model(as_qp=False)
    elif req.preset_id == "refinery_blending_qp":
        model = build_refinery_blending_model(as_qp=True)
    elif req.preset_id == "power_unit_commitment":
        model = build_unit_commitment_model(time_periods=4)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown preset ID: {req.preset_id}")

    CURRENT_MODEL = model
    return extract_model_response(model)


@app.post("/api/upload_model")
async def upload_model(file: UploadFile = File(...)):
    global CURRENT_MODEL, CURRENT_PRESOLVE
    CURRENT_PRESOLVE = None

    contents = (await file.read()).decode("utf-8", errors="replace")
    filename = file.filename.lower()

    try:
        if filename.endswith(".mps") or "NAME" in contents[:50]:
            model = MPSParser.parse_string(contents)
        else:
            model = LPParser.parse_string(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    CURRENT_MODEL = model
    return extract_model_response(model)


@app.post("/api/presolve")
def run_presolve():
    global CURRENT_MODEL, CURRENT_PRESOLVE
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    presolver = Presolver()
    p_model, mapper, stats = presolver.presolve(CURRENT_MODEL)
    CURRENT_PRESOLVE = (p_model, mapper, stats)

    fixed_list = [{"name": k, "value": round(float(v), 6)} for k, v in list(mapper.fixed_vars.items())[:20]]

    return {
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
        "fixed_vars_list": fixed_list,
    }


@app.post("/api/ml_recommend")
def get_ml_recommendation():
    global CURRENT_MODEL, CURRENT_PRESOLVE
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    stats = CURRENT_PRESOLVE[2] if CURRENT_PRESOLVE else None
    ml_engine = MLStrategyEngine()
    rec = ml_engine.recommend(CURRENT_MODEL, stats)
    raw_features = FeatureExtractor.extract(CURRENT_MODEL, stats)

    return {
        "recommended_algorithm": rec.recommended_algorithm,
        "algorithm_probabilities": rec.algorithm_probabilities,
        "confidence_score": round(rec.confidence_score * 100, 1),
        "recommended_hardware": rec.recommended_hardware,
        "branching_strategy": rec.branching_strategy,
        "feature_attributions": rec.feature_attributions,
        "deterministic_fallback": rec.deterministic_fallback,
        "features": {k: round(float(v), 4) for k, v in raw_features.items()},
    }


@app.post("/api/solve")
def solve_model(req: SolveRequest):
    global CURRENT_MODEL, CURRENT_PRESOLVE
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    model_to_solve = CURRENT_MODEL
    mapper = None
    presolve_stats = None

    if req.enable_presolve:
        if CURRENT_PRESOLVE is None:
            presolver = Presolver()
            p_model, p_mapper, p_stats = presolver.presolve(CURRENT_MODEL)
            CURRENT_PRESOLVE = (p_model, p_mapper, p_stats)
        model_to_solve, mapper, presolve_stats = CURRENT_PRESOLVE

    # Determine algorithm
    algo = req.algorithm
    ml_engine = MLStrategyEngine()
    ml_rec = ml_engine.recommend(CURRENT_MODEL, presolve_stats)
    if algo == "auto":
        algo = ml_rec.recommended_algorithm

    # Instantiate selected solver
    p_class = model_to_solve.classify()
    if p_class == "MILP" or algo == "branch_and_bound":
        solver = BranchAndBoundSolver(time_limit_seconds=req.time_limit_seconds)
    elif p_class == "QP" or algo == "active_set":
        solver = ActiveSetQPSolver()
    elif algo == "interior_point":
        solver = InteriorPointSolver()
    else:
        # Default to Revised Simplex
        solver = RevisedSimplexSolver()

    # Execute solve
    start_time = time.time()
    result: SolverResult = solver.solve(model_to_solve, time_limit_seconds=req.time_limit_seconds)
    solve_time = time.time() - start_time

    # Postsolve reconstruction if presolve was used
    final_primal = dict(result.primal_solution)
    mapping_sample = []
    if mapper is not None and result.is_feasible:
        x_full = mapper.restore_primal(result.primal_solution)
        final_primal = {name: float(x_full[i]) for i, name in enumerate(mapper.original_var_names)}
        for v_name in mapper.original_var_names[:40]:
            if v_name in mapper.fixed_vars:
                st = "FIXED_IN_PRESOLVE"
            elif v_name in result.primal_solution:
                st = "OPTIMIZED_IN_CORE"
            else:
                st = "CANONICAL_RESTORED"
            mapping_sample.append({
                "variable": v_name,
                "value": round(float(final_primal.get(v_name, 0.0)), 6),
                "resolution_state": st,
            })
    else:
        for v_name, val in list(final_primal.items())[:40]:
            mapping_sample.append({
                "variable": v_name,
                "value": round(float(val), 6),
                "resolution_state": "DIRECT_SOLVE",
            })

    final_result = SolverResult(
        status=result.status,
        objective_value=result.objective_value,
        primal_solution=final_primal,
        iterations=result.iterations,
        runtime_seconds=solve_time,
        nodes_explored=result.nodes_explored,
        mip_gap=result.mip_gap,
        diagnostics=result.diagnostics,
    )

    # Independent mathematical verification
    validator = IndependentValidator()
    certificate = validator.verify(CURRENT_MODEL, final_result)

    # Workflow stage summaries
    model_meta = CURRENT_MODEL.get_metadata()
    workflow_stages = [
        {
            "stage": 1,
            "name": "Formulation & Topology",
            "status": "COMPLETED",
            "summary": f"{CURRENT_MODEL.num_variables} vars, {CURRENT_MODEL.num_constraints} cons, {model_meta.num_nonzeros} nnz",
        },
        {
            "stage": 2,
            "name": "Presolve Reductions",
            "status": "COMPLETED" if req.enable_presolve and presolve_stats else "BYPASS",
            "summary": f"-{presolve_stats.var_reduction_pct:.1f}% vars, -{presolve_stats.con_reduction_pct:.1f}% cons" if presolve_stats else "Presolve bypassed",
        },
        {
            "stage": 3,
            "name": "AI Meta-Strategy",
            "status": "COMPLETED",
            "summary": f"Selected {solver.name} ({round(ml_rec.confidence_score * 100, 1)}% confidence)",
        },
        {
            "stage": 4,
            "name": "Sovereign Core Solve",
            "status": "COMPLETED",
            "summary": f"{final_result.status.value} in {final_result.iterations} iters ({solve_time*1000:.1f}ms)",
        },
        {
            "stage": 5,
            "name": "Postsolve Reconstruction",
            "status": "COMPLETED" if mapper else "BYPASS",
            "summary": f"Restored {len(final_primal)} variables into canonical space",
        },
        {
            "stage": 6,
            "name": "Independent Trust Audit",
            "status": "CERTIFIED" if certificate.is_valid else "FAILED",
            "summary": f"Primal viol {certificate.max_primal_violation:.1e}, Bounds viol {certificate.max_bound_violation:.1e}",
        },
    ]

    return {
        "status": final_result.status.value,
        "objective_value": final_result.objective_value,
        "iterations": final_result.iterations,
        "runtime_seconds": round(final_result.runtime_seconds, 4),
        "nodes_explored": final_result.nodes_explored,
        "mip_gap": final_result.mip_gap,
        "algorithm_used": solver.name,
        "primal_solution": {k: round(v, 6) for k, v in list(final_result.primal_solution.items())[:60]},
        "diagnostics": final_result.diagnostics,
        "postsolve_mapping": mapping_sample,
        "workflow_stages": workflow_stages,
        "trust_certificate": {
            "is_valid": certificate.is_valid,
            "status": certificate.status,
            "max_primal_violation": certificate.max_primal_violation,
            "max_bound_violation": certificate.max_bound_violation,
            "max_integrality_violation": certificate.max_integrality_violation,
            "recomputed_objective": certificate.recomputed_objective,
            "objective_difference": certificate.objective_difference,
            "checks": certificate.checks,
            "violation_details": certificate.violation_details,
        },
    }
